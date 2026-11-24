"""Running a graph forward, and reading its log-density.

Evaluation is one linear scan in topological order: every node's parents are
already in the environment by the time it is reached, which is what the
ordering rule in :class:`~bayesmith.graph.graph.Graph` buys.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import copy
from functools import cache, partial
from typing import Any
from weakref import ReferenceType

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith.errors import GraphError, StructureError
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic
from bayesmith.graph.reduction import ReducedGraph, as_graph

Env = dict[str, Any]


def _plate_arguments(
    graph: Graph, node: Node, env: Env, law_for: Callable[[str], dist.Distribution]
) -> tuple[list[Any], tuple[int | None, ...]]:
    """Map array nodes on their leading axis, laws on their last batch axis."""
    args = []
    axes = []
    for name in node.parents:
        value = env[name]
        parent = graph.node(name)
        axis = None
        if node.plate and parent.plate == node.plate:
            axis = 0
            if isinstance(parent, Probabilistic):
                law = law_for(name)
                parent_shape = distribution_batch_shape(graph, parent, law) + law.event_shape
                value = jnp.broadcast_to(value, jnp.broadcast_shapes(parent_shape, jnp.shape(value)))
                axis = -1 - law.event_dim
        args.append(value)
        axes.append(axis)
    return args, tuple(axes)


class _MemberwiseLaw(dist.Distribution):
    """Evaluate parameterized transforms within the plate member that owns them."""

    pytree_data_fields = ("members",)
    pytree_aux_fields = ("trailing", "member_batch")

    def __init__(self, members, size, trailing):
        self.members = members
        self.trailing = trailing
        padding = max(trailing - len(members.batch_shape), 0)
        self.member_batch = (1,) * padding + members.batch_shape
        axis = len(self.member_batch) - trailing
        batch = self.member_batch[:axis] + (size,) + self.member_batch[axis:]
        super().__init__(batch, members.event_shape)

    @property
    def support(self):
        support = self.members.support
        if any(jnp.ndim(leaf) for leaf in jax.tree.leaves(support)):
            raise StructureError(
                "could not certify the support of a plated transformed law "
                "with array-valued bounds"
            )
        return support

    def sample(self, key, sample_shape=()):
        axis = -1 - self.trailing - self.event_dim
        keys = jax.random.split(key, self.batch_shape[-1 - self.trailing])
        return jax.vmap(
            lambda member, k: member.expand(self.member_batch).sample(k, sample_shape),
            out_axes=axis,
        )(self.members, keys)

    def log_prob(self, value):
        value = jnp.broadcast_to(value, jnp.broadcast_shapes(jnp.shape(value), self.shape()))
        return jax.vmap(
            lambda member, v: member.expand(self.member_batch).log_prob(v),
            in_axes=(0, -1 - self.trailing - self.event_dim),
            out_axes=-1 - self.trailing,
        )(self.members, value)


_CERTIFIED_TRANSFORMED = tuple(
    family for family in (dist.Pareto, getattr(dist, "CirculantNormal", None))
    if family is not None
)


def _check_plate_transform(transform):
    """Check dynamic transform state, including the inner transform of an inverse."""
    children, (fields, static) = transform.tree_flatten()
    lost = vars(transform).keys() - set(fields) - set(static)
    if "_inv" in lost:
        cached = vars(transform)["_inv"]
        if isinstance(cached, ReferenceType):
            inverse = cached()
            if inverse is None or (
                type(inverse) is dist.transforms._InverseTransform
                and vars(inverse).keys() == {"_inv"} and inverse._inv is transform
            ):
                lost.remove("_inv")
    if lost:
        raise StructureError(
            f"A plated transform has instance state absent from its pytree: {sorted(lost)}."
        )
    nested = jax.tree.leaves(
        children, is_leaf=lambda item: isinstance(item, dist.transforms.Transform)
    )
    for child in nested:
        if isinstance(child, dist.transforms.Transform):
            _check_plate_transform(child)


def _checked_plate_member(law):
    """Refuse instance overrides a JAX pytree round trip would silently discard."""
    chain = []
    current = law
    while isinstance(current, dist.Distribution):
        chain.append(current)
        current = getattr(current, "base_dist", None)
    if any(type(part) in _CERTIFIED_TRANSFORMED for part in chain):
        for part in chain:
            if {"log_prob", "sample", "sample_with_intermediates", "rsample"} & vars(part).keys():
                raise StructureError("A plated distribution has an instance-level density or sampler override.")
            for transform in getattr(part, "transforms", ()):
                _check_plate_transform(transform)
    return law


def _stack_member_laws(
    law: dist.Distribution, size: int, trailing: int = 0
) -> dist.Distribution:
    """Insert a plate before the specified trailing member batch dimensions.

    JAX vmaps array leaves, while NumPyro keeps batch/event shapes in static
    pytree metadata. A member's categorical logits also carry a category axis
    which must remain after the complete batch, just like a Gaussian's events.
    """
    batch = law.batch_shape
    axis = max(len(batch) - trailing, 0)
    padding = max(trailing - len(batch), 0)
    shape = (*batch[:axis], size, *(1,) * padding, *batch[axis:])
    # Transform parameters have their own event dimensions. NumPyro does
    # not expose an argument-constraint layout for arbitrary transforms.
    # Apply the original member operations, which already own that layout.
    preserve_affine = type(law) in _CERTIFIED_TRANSFORMED and all(
        type(t) is dist.transforms.AffineTransform or not jax.tree.leaves(t)
        for t in law.transforms
    )
    if (isinstance(law, dist.TransformedDistribution)
            and jax.tree.leaves(law.transforms) and not preserve_affine):
        return _MemberwiseLaw(law, size, trailing)
    if type(law) is dist.ExpandedDistribution:
        return _stack_member_laws(law.base_dist, size, trailing).expand(shape)
    if type(law) is dist.Independent:
        count = law.reinterpreted_batch_ndims
        return _stack_member_laws(law.base_dist, size, trailing + count).to_event(count)

    def stack_field(field, event_rank):
        if isinstance(field, dist.Distribution):
            extra = max(len(field.batch_shape) - len(batch), 0)
            return _stack_member_laws(field, size, trailing + extra)
        # Some scalar parameters broadcast over batch dimensions of other
        # fields. Make those singleton axes explicit before moving the plate.
        member_shape = jnp.shape(field)[1:]
        rank = max(len(member_shape) - event_rank, 0)
        missing = max(len(batch) - rank, 0)
        field = jnp.reshape(field, (size, *(1,) * missing, *member_shape))
        field = jnp.moveaxis(field, 0, axis)
        for _ in range(padding):
            field = jnp.expand_dims(field, axis + 1)
        return field

    stacked = copy(law)
    stacked._batch_shape = shape
    fields, _ = law.tree_flatten()
    for name, field in zip(law.gather_pytree_data_fields(), fields, strict=True):
        if field is None:
            continue
        if name == "transforms" and preserve_affine:
            # An affine parameter broadcasts against the member's complete value,
            # including any event axes. Keep the original transforms and the
            # actual base density that the canonical family check will inspect.
            def stack_affine_parameter(value):
                broadcast = jax.vmap(lambda v: jnp.broadcast_to(v, law.shape()))(value)
                return stack_field(broadcast, law.event_dim)

            stacked.transforms = jax.tree.map(stack_affine_parameter, field)
            continue
        constraint = law.arg_constraints.get(name)
        try:
            event_rank = constraint.event_dim if constraint is not None else 0
        except NotImplementedError:
            event_rank = 0
        setattr(stacked, name, jax.tree.map(
            partial(stack_field, event_rank=event_rank), field,
            is_leaf=lambda item: isinstance(item, dist.Distribution),
        ))
    # NumPyro stores some parameter arrays as static metadata, such as the
    # count in Multinomial. These are shared by every member, so inserting
    # a singleton plate broadcasts them without altering their values.
    for name in law.arg_constraints.keys() & set(law.gather_pytree_aux_fields()):
        field = law.__dict__.get(name)
        if field is not None and jnp.ndim(field):
            setattr(stacked, name, jnp.expand_dims(field, axis))
    return stacked


def apply_deterministic(graph: Graph, node: Deterministic, env: Env) -> Any:
    """Call ``node.fn``, vmapped over the plate the node lives in.

    Public because the NumPyro bridge re-runs the same scan and must apply
    plates identically; a private name used across modules is not private.
    """
    return _apply_deterministic(graph, node, env, _law_builder(graph, env))


def _apply_deterministic(graph, node, env, law_for):
    args, in_axes = _plate_arguments(graph, node, env, law_for)
    if not node.plate:
        return node.fn(*args)

    if all(axis is None for axis in in_axes):
        raise GraphError(
            f"deterministic node {node.name!r} is in plate {node.plate[0]!r} but "
            "none of its parents are, so there is nothing to map over. Either put "
            "a parent in the plate, or drop the plate and let broadcasting do it."
        )
    return jax.vmap(node.fn, in_axes=in_axes)(*args)


def apply_probabilistic(graph: Graph, node: Probabilistic, env: Env) -> Any:
    """Build ``node``'s distribution, vmapping ``dist_fn`` over its plate.

    The plate occupies the last batch axis, immediately before the event.
    Parameterized transforms sample and score within each plate member.
    Inspecting their support raises :class:`~bayesmith.errors.StructureError`
    when the transform has array-valued bounds. Mapped Pareto and CirculantNormal
    laws also raise StructureError when an instance overrides a density, sampler
    or transform method that JAX cannot preserve through batching.

    Public for the same reason as :func:`apply_deterministic`: ``log_joint``
    and the NumPyro bridge must build the *same* distribution for the *same*
    node, so both call this one function instead of each independently
    calling ``node.dist_fn`` unmapped. Before this function existed, both
    call sites made that same independent (and wrong) choice, which is
    exactly why the bridge/log_joint cross-check could not catch it -- two
    scans with an identical blind spot agree with each other by
    construction, whether or not they agree with the truth.

    Deliberately does **not** raise when a plated node has no plated parent,
    unlike :func:`apply_deterministic`. That refusal exists there because
    ``fn``'s return value is an arbitrary array that must already be the
    right shape -- there is no fallback that is still correct, so refusing
    is the only safe option. A distribution has a real fallback: calling
    ``dist_fn`` once, unmapped, yields one shared distribution whose
    ``log_prob``/``sample`` already know how to broadcast themselves against
    a plate-shaped value -- plain array broadcasting in ``log_joint``, and
    ``numpyro.plate``'s own ``.expand()`` in ``to_numpyro``. That is not a
    workaround, it is the ordinary "N iid draws from one shared prior"
    pattern -- already relied on today by a plated node with *no* parents at
    all (see ``test_a_plated_latent_site_carries_the_plate_axis``). Refusing
    it here would break that already-correct, already-tested pattern for no
    safety benefit: there is no shape for dist_fn's output to get wrong,
    only a distribution object whose own broadcasting contract handles it.
    """
    return _apply_probabilistic(graph, node, env, _law_builder(graph, env))


def _law_builder(graph: Graph, env: Env) -> Callable[[str], dist.Distribution]:
    """Cache parent laws within this evaluation of the plate layout."""
    @cache
    def law_for(name: str) -> dist.Distribution:
        return _apply_probabilistic(graph, graph.node(name), env, law_for)

    return law_for


def _apply_probabilistic(graph, node, env, law_for):
    args, in_axes = _plate_arguments(graph, node, env, law_for)
    if any(axis is not None for axis in in_axes):
        members = jax.vmap(
            lambda *values: _checked_plate_member(node.dist_fn(*values)), in_axes=in_axes
        )(*args)
        distribution = _stack_member_laws(members, graph.plate_size(node.plate[0]))
    else:
        distribution = node.dist_fn(*args)
    if node.observed_mask is not None and distribution.event_shape:
        raise StructureError(
            f"node {node.name!r} applies an observation mask to an event-valued "
            "distribution with no per-sample sigma or scalar-event density. "
            "Masks select scalar-event observations; selecting "
            "components of a joint event requires its marginal distribution. "
            "Declare that marginal explicitly instead of masking event components."
        )
    return distribution


def distribution_batch_shape(
    graph: Graph, node: Probabilistic, distribution: Any
) -> tuple[int, ...]:
    """Independent draw axes, including the plate and recorded repetitions.

    NumPyro plates expand the rightmost batch axis. Event axes belong to
    one draw from the law and must stay outside that expansion.
    """
    batch = distribution.batch_shape
    event = distribution.event_shape
    plate = (graph.plate_size(node.plate[0]),) if node.plate else ()
    data = jnp.shape(node.observed) if node.observed is not None else ()
    try:
        batch = jnp.broadcast_shapes(batch, plate)
        shape = jnp.broadcast_shapes(batch + event, data)
        if event and shape[-len(event):] != event:
            raise ValueError("data would expand the distribution's event axes")
    except ValueError as exc:
        raise StructureError(
            f"node {node.name!r} has shapes that cannot be reconciled: its "
            f"distribution has batch {distribution.batch_shape} and event {event}, "
            f"its plate is {plate}, and its data shape is {data}."
        ) from exc
    return shape[:-len(event)] if event else shape


def evaluate(
    graph: Graph | ReducedGraph,
    values: Mapping[str, Any] | None = None,
    *,
    observed_value: Callable[[Probabilistic, Env], Any] | None = None,
) -> Env:
    """Compute every node's value.

    Args:
        graph: the graph to run.
        values: values for the latent nodes. Deterministic and constant nodes
            are computed.
        observed_value: an optional forward generator called at each observed
            node with its parents already evaluated. By default each node
            takes its recorded data.

    Returns:
        A mapping from node name to value, covering every node in the graph.

    Raises:
        GraphError: if a latent node has no value, or ``values`` names
            something that is not a latent node.
    """
    graph = as_graph(graph)
    values = dict(values or {})
    unknown = set(values) - set(graph.latents)
    if unknown:
        name = min(unknown)
        if name not in graph.names:
            raise GraphError(
                f"values names {name!r}, which does not name any node in "
                f"this graph. Latents are {list(graph.latents)}."
            )
        if isinstance(graph.node(name), Probabilistic):
            # In graph.names but excluded from graph.latents above, and
            # Probabilistic: the only way to be both is to be observed.
            raise GraphError(
                f"values names {name!r}, which is an observed node: its "
                "value comes from observe(..., obs=...), not from `values`. "
                f"Latents are {list(graph.latents)}."
            )
        raise GraphError(
            f"values names {name!r}, which is a deterministic or constant "
            "node: its value is computed, not supplied via `values`. "
            f"Latents are {list(graph.latents)}."
        )

    env: Env = {}
    law_for = _law_builder(graph, env)
    for node in graph.nodes:
        if isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = _apply_deterministic(graph, node, env, law_for)
        elif isinstance(node, Probabilistic):
            if not node.is_latent:
                env[node.name] = (
                    node.observed if observed_value is None else observed_value(node, env)
                )
            elif node.name in values:
                env[node.name] = values[node.name]
            else:
                raise GraphError(
                    f"latent node {node.name!r} has no value. Supply one in "
                    "`values`, or condition the node with observe(...)."
                )
        else:
            # Live defensive code, not dead code: Node is a plain eqx.Module
            # (not an ABC) and Graph.__check_init__ does not check node
            # subtype, so a hand-built Graph containing a bare Node (or any
            # subclass other than Const/Deterministic/Probabilistic) reaches
            # this branch and raises correctly here.
            raise GraphError(
                f"unknown node type {type(node).__name__}"
            )  # pragma: no cover
    return env


def graph_density(
    graph: Graph,
    term: Any,
    env: Env,
    *,
    label: str,
    names: tuple[str, ...],
) -> jax.Array:
    """Evaluate one graph-level term and require one scalar contribution.

    Shared by both density scans. NumPyro otherwise sums a vector-valued
    ``factor`` while :func:`log_joint` broadcasts it, so accepting a vector
    would make the two scans disagree without either one raising.
    """
    density = jnp.asarray(term.log_density(graph, {name: env[name] for name in names}))
    if density.shape != ():
        raise GraphError(
            f"{label}.log_density must return one scalar graph-level density; "
            f"got shape {density.shape}. Sum independent contributions inside "
            "the term so log_joint and to_numpyro cannot assign different "
            "semantics to the same declaration."
        )
    return density


def log_joint(
    graph: Graph | ReducedGraph, values: Mapping[str, Any] | None = None
) -> jax.Array:
    """The joint log-density of the graph at ``values``.

    Every probabilistic node contributes ``log_prob`` of its value under the
    distribution its parents parameterise; deterministic nodes contribute
    nothing but the dependence they carry.

    A node's :attr:`~bayesmith.graph.nodes.Probabilistic.observed_mask` drops
    the samples it says were never taken. That is the whole of masking on THIS
    side -- there is no infinite scale to take a limit of, because a sample
    that was not observed contributes no term rather than a term whose width
    went to infinity, and the two differ by exactly the ``log sigma`` that
    would have sent the joint to ``-inf``.
    """
    graph = as_graph(graph)
    env = evaluate(graph, values)
    total = jnp.zeros(())
    law_for = _law_builder(graph, env)
    for node in graph.nodes:
        if isinstance(node, Probabilistic):
            distribution = law_for(node.name)
            if node.plate:
                distribution = distribution.expand(distribution_batch_shape(graph, node, distribution))
            if node.observed_mask is not None:
                # NumPyro substitutes feasible values at masked coordinates
                # before scoring. Masking a log_prob afterwards can retain NaN
                # gradients from unobserved out-of-support placeholders.
                distribution = distribution.mask(node.observed_mask)
            term = distribution.log_prob(env[node.name])
            total = total + jnp.sum(term)
    # Densities over SEVERAL latents are terms of the joint rather than any
    # node's own density. Read the prior and the reduced-likelihood terms in
    # ONE loop: adding another scan here would let one of the graph-level
    # slots drift out of a future evaluator while every node-level test stayed
    # green. `to_numpyro` reads the same two declarations after every plate.
    graph_terms = tuple(
        (f"evidence_terms[{index}]", term, term.over)
        for index, term in enumerate(graph.evidence_terms)
    )
    if graph.joint_prior is not None:
        # A JeffreysPrior may depend on retained nuisance latents outside its
        # ``over`` block through the forward model, so it receives the full
        # latent environment. Evidence terms receive exactly their declared
        # block, making ``over`` an enforceable dependency boundary.
        graph_terms = (("joint_prior", graph.joint_prior, graph.latents), *graph_terms)
    for label, term, names in graph_terms:
        total = total + graph_density(graph, term, env, label=label, names=names)
    return total
