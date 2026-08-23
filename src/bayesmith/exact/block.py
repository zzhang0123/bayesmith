"""The affine action of a group of latents on every prediction.

No matrix is ever formed: ``A`` comes from ``jax.linearize`` and ``A^T`` from
``jax.vjp``, so a block with 10^6 degrees of freedom costs one forward
evaluation per application. That is what makes the conjugate-Gaussian solves
in :mod:`bayesmith.exact.solve` tractable at all.

**Where everything comes from.** rheplicant's ``unchecked_operator`` takes a
``ParameterSpace`` for the latents and their priors and a ``Pipeline`` for the
prediction. Here both come from the graph: the prediction is every observed
node's ``loc``, the prior is each member's own distribution, and sigma is
every observed node's ``scale``. There is consequently **no keyword to
override any of them** -- and so none of rheplicant's reconciliation
machinery (``_reconcile``, ``_agrees``, ``_resolve_prior``,
``_require_prior_std``) has anything to reconcile. One statement of the prior
means the exact solve and NUTS cannot target different posteriors.

**One spelling.** The domain is ``{latent: array}`` and the codomain is
``{observed: array}``, always, for one member or several. rheplicant carries
a ``name=``/``names=`` dual spelling plus ``as_dict`` to paper over it; the
graph makes that unnecessary because everything already has a name.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError, NotGaussian
from bayesmith.exact.gaussian import (
    check_gaussian,
    gaussian_parts,
    node_shape,
    observation_parts,
)
from bayesmith.graph.evaluate import apply_deterministic, evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic


@dataclasses.dataclass(frozen=True)
class LinearBlock:
    """``A x + offset``: what a group of latents does to every prediction.

    Deliberately a plain frozen dataclass rather than an ``eqx.Module``: this
    is a derived linear-algebra handle, not a differentiable model.
    ``forward`` and ``adjoint`` close over a traced computation, so a block is
    something you build where you need it, not a pytree to carry around.

    Attributes:
        names: the block's members, in the caller's order.
        shape, dtype: ``{name: ...}`` describing the domain.
        offset: ``{observed: prediction at x = 0}`` -- everything OUTSIDE the
            block contributes.
        forward: ``{name: x} -> {observed: A x}``, from ``jax.linearize``.
        adjoint: ``{observed: y} -> {name: A^T y}``, from ``jax.vjp``.
        data: ``{observed: value}``, broadcast to the prediction's shape.
        prior_mean, prior_std: ``{name: ...}``, read off each member's own
            distribution at ``at``.

    Adjoint convention: ``adjoint`` is exactly ``jax.vjp``, so the identity
    that holds is over the **real** inner product::

        sum(x * adjoint(y))  ==  sum(forward(x) * y)

    which is the pairing a Gaussian likelihood forms.
    ``test_adjoint_is_the_transpose_under_the_real_inner_product`` pins both
    halves so the distinction cannot rot into a silent factor.
    """

    names: tuple[str, ...]
    shape: dict[str, tuple[int, ...]]
    dtype: dict[str, Any]
    offset: dict[str, jax.Array]
    forward: Callable[[Any], dict[str, jax.Array]]
    adjoint: Callable[[Any], dict[str, jax.Array]]
    data: dict[str, jax.Array]
    prior_mean: dict[str, jax.Array]
    prior_std: dict[str, jax.Array]


def domain_zero(block: LinearBlock) -> dict[str, jax.Array]:
    """A zero of the block's domain."""
    return {n: jnp.zeros(block.shape[n], dtype=block.dtype[n]) for n in block.names}


def domain_centre(block: LinearBlock) -> dict[str, jax.Array]:
    """The prior's centre, laid out over the domain.

    A zero-mean prior is wrong for most physical quantities -- a noise-wave
    temperature sits near 250 K, not near zero -- and shifting the prior is
    not the same act as shifting the model even though the two give the same
    Gaussian. The graph states which one was meant, so this reads it.
    """
    return {n: block.prior_mean[n] for n in block.names}


def variance_parts(block: LinearBlock) -> dict[str, jax.Array]:
    """``S``'s diagonal, laid out over the domain.

    Block-diagonal assembly by **placement** rather than concatenation: each
    member's variance lands on the leaf its own parameters live on, so
    ``x / variance`` inside a ``jax.tree.map`` IS ``S^-1 x`` with no indices
    to get wrong.
    """
    return {n: jnp.asarray(block.prior_std[n]) ** 2 for n in block.names}


def largest_variance(prior_variance: dict[str, jax.Array]) -> jax.Array:
    """The biggest prior variance anywhere in the block.

    ``1/largest`` floors ``lambda_min`` of the normal operator: ``A^T N^-1 A`` is
    positive semi-definite, so the LOOSEST prior in the block is what bounds
    the operator from below. Taking the tightest instead would floor the
    estimate above the true ``lambda_min`` and report a condition number
    smaller than the real one -- an over-confident guard, which is the
    direction that costs something.
    """
    leaves = jax.tree.leaves(prior_variance)
    return jnp.max(jnp.stack([jnp.max(jnp.asarray(leaf)) for leaf in leaves]))


def _ancestors(graph: Graph, name: str) -> set[str]:
    """Every node ``name`` transitively depends on."""
    seen: set[str] = set()
    stack = list(graph.node(name).parents)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph.node(current).parents)
    return seen


def _validated_names(graph: Graph, names: Iterable[str]) -> tuple[str, ...]:
    names = tuple(names)
    if not names:
        raise GraphError(
            "a linear block needs at least one latent name. An empty block is "
            "solved every sweep and changes nothing, so a plan holding one runs, "
            "converges, and reports a partition that does not cover the space it "
            "claims to."
        )
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise GraphError(
            f"block {names} lists {repeated} twice. Two copies of one latent are "
            "exactly degenerate with each other, so the normal operator is "
            "singular in a direction that says nothing about the model."
        )
    latents = set(graph.latents)
    stray = [n for n in names if n not in latents]
    if stray:
        raise GraphError(
            f"block {names} names {stray}, which are not latent nodes of this "
            f"graph. Latents are {list(graph.latents)}. A deterministic or "
            "observed node has no posterior to solve for."
        )
    return names


def _validated_at(
    graph: Graph, names: tuple[str, ...], at: dict[str, Any] | None
) -> dict[str, Any]:
    """The values for the latents OUTSIDE the block, checked by name.

    Every other misuse in this module is refused by name; ``at`` was the one
    surface where a caller's mistake went unremarked. A stale entry for a
    block member is the dangerous case: it is silently discarded, so a caller
    who believes they are pinning a member is in fact solving for it.
    """
    at = dict(at or {})
    latents = set(graph.latents)
    members = set(names)

    stray = sorted(set(at) - latents)
    if stray:
        raise GraphError(
            f"`at` names {stray}, which are not latent nodes of this graph. "
            f"Latents are {list(graph.latents)}. `at` fixes the latents OUTSIDE "
            "the block; a deterministic or observed node has no value to fix."
        )
    overlap = sorted(set(at) & members)
    if overlap:
        raise GraphError(
            f"`at` names {overlap}, which are IN this block. A block is affine "
            "given the latents outside it, and its own members are what it "
            "solves for -- so an entry here is discarded, and a caller who "
            "believes they pinned a member would be silently wrong. Drop it "
            "from `at`, or drop it from the block."
        )
    missing = sorted(latents - members - set(at))
    if missing:
        raise GraphError(
            f"latents {missing} are outside the block and have no value in "
            "`at`. A block is affine GIVEN them, so they must be somewhere -- "
            "pass the current values."
        )
    return at


def _refuse_internal_ancestry(graph: Graph, names: tuple[str, ...]) -> None:
    """Refuse a block one of whose members is an ancestor of another.

    ``x ~ N(0, |tau|)`` with ``tau`` a latent is perfectly good modelling, and
    both nodes are individually Normal -- but the PAIR is not jointly
    Gaussian, because one member's width is a function of the other. Solving
    them together would produce a finite, confident posterior for a model
    nobody declared.

    The test is **transitive**, not on direct parents: ``m1 -> some
    deterministic node -> m2``'s prior is the same situation with one more
    hop, and a direct-parent check cannot see it.

    It is also what makes :func:`_env_before` correct -- with no member
    ancestral to another, a topological scan reaches every member's prior
    before it needs any member's value.
    """
    members = set(names)
    for member in names:
        clash = sorted(members.intersection(_ancestors(graph, member)))
        if clash:
            raise NotGaussian(
                f"latent {member!r} has {clash} among its ancestors, and they are "
                "in the same block. One member's distribution is then a function "
                "of another's value, so the pair is not JOINTLY Gaussian however "
                "Normal each one is on its own -- and a conjugate solve over the "
                "pair would return a confident posterior for a model nobody "
                "declared. Put them in separate blocks and alternate."
            )


def _env_before(
    graph: Graph, names: tuple[str, ...], at: dict[str, Any]
) -> tuple[
    dict[str, Any], dict[str, tuple[tuple[int, ...], Any, jax.Array, jax.Array]]
]:
    """Every node's value, with the block members replaced by their prior means.

    Also returns ``{member: (shape, dtype, prior_mean, prior_std)}`` -- the
    block's domain, derived DURING the scan from values already in hand rather
    than from any member's own value. That is what breaks the circularity:
    building the domain needs the shapes, the shapes come from each member's
    prior, and the prior needs an environment that does not yet contain the
    member.

    Correct only because :func:`_refuse_internal_ancestry` has run: otherwise
    a member's prior could depend on a member not yet reached, and the
    placeholder inserted here would silently become part of it.

    Repeats ``evaluate``'s isinstance ladder rather than calling it, for the
    reason above. ``test_env_before_agrees_with_evaluate_on_every_node`` pins
    the two together so the duplication cannot drift.
    """
    members = set(names)
    env: dict[str, Any] = {}
    domain: dict[str, tuple[tuple[int, ...], Any, jax.Array, jax.Array]] = {}
    for node in graph.nodes:
        if node.name in members:
            check_gaussian(graph, node, env)
            loc, scale = gaussian_parts(graph, node, env)
            shape = node_shape(graph, node, env)
            mean = jnp.broadcast_to(loc, shape)
            std = jnp.broadcast_to(scale, shape)
            domain[node.name] = (shape, loc.dtype, mean, std)
            env[node.name] = mean
        elif isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = apply_deterministic(graph, node, env)
        elif isinstance(node, Probabilistic):
            if not node.is_latent:
                env[node.name] = node.observed
            elif node.name in at:
                env[node.name] = at[node.name]
            else:  # pragma: no cover - _validated_at already refused this
                raise GraphError(
                    f"latent node {node.name!r} is outside the block and has no "
                    f"value in `at`. A block is affine GIVEN the latents outside "
                    "it, so they must be somewhere -- pass the current values."
                )
        else:  # pragma: no cover - defensive, mirrors evaluate()
            raise GraphError(f"unknown node type {type(node).__name__}")
    return env, domain


def isolate(
    graph: Graph, names: tuple[str, ...], at: dict[str, Any]
) -> Callable[[dict[str, Any]], dict[str, jax.Array]]:
    """``g: {name: x} -> {observed: loc}`` -- the block's action on the prediction.

    Built on ``evaluate``, so there is exactly one forward scan in this package
    and the block cannot diverge from what ``log_joint`` and the NumPyro bridge
    read.
    """

    def g(x: dict[str, Any]) -> dict[str, jax.Array]:
        _, loc, _ = observation_parts(graph, evaluate(graph, {**at, **x}))
        return loc

    return g


def unchecked_operator(
    graph: Graph, names: Iterable[str], at: dict[str, Any] | None = None
) -> LinearBlock:
    """Export ``A``, ``A^T``, the offset, the data and the prior -- **unchecked**.

    Args:
        graph: the model.
        names: the latents in the block. Solving a group JOINTLY is not the
            same as alternating over its members: two latents the data barely
            tells apart are resolved in one CG here, where alternation
            converges at the rate of their correlation while reporting a
            converged residual and a condition number of ~1 at every step.
        at: values for the latents OUTSIDE the block. A block is affine
            *given* them, so this fixes where it is built -- which is what
            makes a Gibbs sweep possible: rebuild here every sweep at the
            current values.

    Raises:
        GraphError: if ``names`` is empty, repeats a latent, or names
            something that is not a latent; or if ``at`` names something that
            is not a latent, names a member of this block (silently discarded
            otherwise -- see :func:`_validated_at`), or omits a latent that
            is outside the block.
        NotGaussian: if a member or an observed node is not a diagonal
            Gaussian, or if a member is an ancestor of another member.
        StructureError: if a node's own ``log_prob`` disagrees with the
            ``loc``/``scale`` read off it.

    Note:
        The name is the warning. This does **not** check the ``linear_in``
        declaration, so it will happily export an operator for a block that is
        not affine at all and hand back a confident, wrong posterior.
        :func:`bayesmith.exact.linearity.linear_operator` is the entry point
        that checks first, and is what callers should reach for. This one is
        for inside a Gibbs sweep, where the check has been hoisted out of the
        loop deliberately.
    """
    names = _validated_names(graph, names)
    at = _validated_at(graph, names, at)
    _refuse_internal_ancestry(graph, names)

    env, domain = _env_before(graph, names, at)
    for observed in graph.observed:
        check_gaussian(graph, graph.node(observed), env)

    g = isolate(graph, names, at)
    zero = {n: jnp.zeros(domain[n][0], dtype=domain[n][1]) for n in names}
    offset, tangent = jax.linearize(g, zero)
    _, pullback = jax.vjp(g, zero)
    data, _, _ = observation_parts(graph, evaluate(graph, {**at, **zero}))

    return LinearBlock(
        names=names,
        shape={n: domain[n][0] for n in names},
        dtype={n: domain[n][1] for n in names},
        offset=offset,
        forward=tangent,
        adjoint=lambda y: pullback(y)[0],
        data=data,
        prior_mean={n: domain[n][2] for n in names},
        prior_std={n: domain[n][3] for n in names},
    )
