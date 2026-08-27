"""The local view: a graph linearized at a point, in flat coordinates.

Every diagnostic in this package asks a question about the model *here* -- the
rank of the Jacobian at these values, the curvature at this mode, the Fisher
matrix of this block at this point. None of that is an affine CLAIM about the
model, which is why nothing in this module goes near ``linear_in`` or
:func:`~bayesmith.exact.linearity.check_linearity`: a local linearization of a
nonlinear model is exactly what a diagnostic wants and exactly what a
conjugate solve must refuse.

What it shares with the exact layer it shares by import, not by respelling.
The forward map is :func:`~bayesmith.exact.block.isolate` -- the one forward
scan in this package -- and the handle built here is a real
:class:`~bayesmith.exact.block.LinearBlock`, so
:func:`~bayesmith.exact.fisher.dense_operator` and the weighted-design
machinery consume it unchanged. The one honest difference from
``unchecked_operator`` is the linearization point: there the tangent is taken
at the domain's zero, because an affine map has one tangent everywhere; here
it is taken at the caller's values, because a nonlinear model has a different
Jacobian at every point and the diagnostic is about *this* one.

**The local block carries no prior BY DEFAULT.** Its
``prior_mean``/``prior_std`` are empty dicts rather than numbers read off the
latents, because the diagnostics built on it either refuse non-Gaussian priors
themselves (sensitivity) or must not see a prior at all (a Jeffreys prior
built from a matrix containing prior curvature would sit inside its own
definition). Passing one to ``fisher_information(include_prior=True)``
therefore fails loudly on the empty dict instead of silently folding in a
curvature nobody declared. Every one of those callers reads unchanged.

``local_block(..., priors=True)`` is the migration plan's **G15**, added
2026-08-27, and it is a THIRD constructor rather than a change of mind about
this one. A caller who wants a nonlinear model's posterior precision at a
point needs the Jacobian from here and the prior from
``unchecked_operator`` -- and neither had both, because the latter
linearizes at the domain's zero, which is the same tangent everywhere only
when the map is affine. The default keeps every sentence above true; the
keyword says at the call site that the caller has asked.

**Precision discipline.** Nothing here touches ``jax.config`` -- the package
rule. Run the whole thing, graph construction included, inside
``with jax.enable_x64(True):``; the diagnostics check the dtype of what they
computed and refuse float32 by name, because their verdicts live at 1e-17 of
the largest singular value and single precision reports a degenerate model as
identified (measured: the motivating model's null direction surfaces at
3.1e-8 in float32, above the 1e-8 rank tolerance, against 7.5e-17 in double).
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError
from bayesmith.exact.block import LinearBlock, isolate
from bayesmith.exact.gaussian import observed_data_and_loc, unwrap
from bayesmith.graph.evaluate import apply_probabilistic, evaluate
from bayesmith.graph.graph import Graph


def resolve_names(graph: Graph, names: Sequence[str] | str | None) -> tuple[str, ...]:
    """The latents to differentiate with respect to, in the caller's order.

    ``None`` means all of them, in declaration order -- the joint question.
    A bare string is one name, not an iterable of characters: without the
    normalisation ``names="gain"`` iterates into ``('g', 'a', 'i', 'n')`` and
    comes back as four undeclared latents.
    """
    if names is None:
        return graph.latents
    selected = (names,) if isinstance(names, str) else tuple(names)
    if not selected:
        raise GraphError(
            "the diagnostic needs at least one latent name; names=() would "
            "report on a block that contains nothing, which reads as a clean "
            "bill of health. Pass names=None for every latent."
        )
    unknown = [name for name in selected if name not in graph.latents]
    if unknown:
        raise GraphError(
            f"`names` contains {unknown}, which is not a latent of this graph; "
            f"its latents are {list(graph.latents)}."
        )
    repeated = sorted({name for name in selected if selected.count(name) > 1})
    if repeated:
        raise GraphError(
            f"`names` lists {repeated} more than once. Two copies of one latent "
            "are exactly degenerate with each other, so the repeat would "
            "manufacture a null direction that says nothing about the model."
        )
    return selected


def latent_values(graph: Graph, at: dict[str, jax.Array] | None) -> dict[str, jax.Array]:
    """Every latent's value: the prior's centre, overridden by ``at``.

    The graph-side spelling of "the declared initial values": each latent sits
    at the centre of its own prior, read the same way the dispatch layer reads
    it (:func:`~bayesmith.dispatch.classify.prior_environment`), so a
    diagnostic and the plan that motivated it are anchored at the same point.

    Overrides are broadcast to the latent's own shape rather than trusted: a
    scalar handed to a plated latent would otherwise shrink the flat parameter
    vector, and every count in the report would be about a smaller model than
    the graph declares.

    Raises:
        GraphError: if ``at`` names something that is not a latent, or a value
            does not broadcast to its latent's shape.
    """
    # Imported here rather than at module scope: `dispatch` imports nothing
    # from `diagnose`, so there is no cycle today, but the dependency points
    # from the diagnostics INTO the layer that owns the anchoring rule and a
    # module-scope import would drag the classifier in for every diagnose
    # import. One function is what is borrowed, so one function is imported.
    from bayesmith.dispatch.classify import prior_environment

    environment = prior_environment(graph)
    values = {name: environment[name] for name in graph.latents}
    for name, value in (at or {}).items():
        if name not in values:
            raise GraphError(
                f"`at` names {name!r}, which is not a latent of this graph; its "
                f"latents are {list(graph.latents)}. A stray key would be "
                "silently ignored by evaluation, and the report would describe "
                "the model at a point the caller did not ask for."
            )
        try:
            values[name] = jnp.broadcast_to(
                jnp.asarray(value), jnp.shape(values[name])
            )
        except (TypeError, ValueError) as error:
            raise GraphError(
                f"`at[{name!r}]` has shape {jnp.shape(jnp.asarray(value))}, which "
                f"does not broadcast to the latent's shape "
                f"{jnp.shape(values[name])}."
            ) from error
    return values


def check_differentiable(
    graph: Graph, names: Sequence[str], values: dict[str, jax.Array]
) -> None:
    """Every SELECTED latent must carry a real-valued derivative.

    Only the selected ones: a complex or integer latent held fixed outside
    the selection is no obstacle to asking about the rest.
    """
    complex_names = [
        name
        for name in names
        if jnp.issubdtype(jnp.asarray(values[name]).dtype, jnp.complexfloating)
    ]
    if complex_names:
        raise GraphError(
            f"latent(s) {complex_names} are complex. The prediction is real, so "
            "the map from complex coefficients to data is R-linear but not "
            "C-linear, and its rank over C is not the number you want -- a "
            "block with n complex coefficients has 2n real degrees of freedom, "
            "and they can be identified separately. Declare the real and "
            "imaginary parts as separate latents, or exclude these with names=."
        )
    not_floating = [
        name
        for name in names
        if not jnp.issubdtype(jnp.asarray(values[name]).dtype, jnp.floating)
    ]
    if not_floating:
        kinds = {name: str(jnp.asarray(values[name]).dtype) for name in not_floating}
        raise GraphError(
            f"latent(s) {kinds} are not floating-point, so the prediction has "
            "no derivative with respect to them and a Jacobian-based verdict "
            "is not defined. An integer latent is not a continuous parameter; "
            "exclude it with names=."
        )


def check_observed_have_locs(graph: Graph, values: dict[str, jax.Array]) -> None:
    """Every observed node must carry a ``loc`` for the prediction to be read.

    The diagnostics read "the prediction" as the observed nodes'
    distributional location, the same seam :func:`~bayesmith.exact.block.
    isolate` reads. A Poisson observation has a rate and no ``loc``; asking
    for its Jacobian through this seam would die inside a trace with a bare
    ``AttributeError`` naming neither the node nor the remedy.
    """
    if not graph.observed:
        raise GraphError(
            "this graph has no observed node, so there is no prediction to "
            "differentiate and nothing for a diagnostic to report. Condition "
            "at least one node with observe(...)."
        )
    env = evaluate(graph, values)
    for name in graph.observed:
        distribution = unwrap(apply_probabilistic(graph, graph.node(name), env))
        if not hasattr(distribution, "loc"):
            raise GraphError(
                f"observed node {name!r} returns "
                f"{type(distribution).__name__}, which carries no `loc`. The "
                "diagnostics read the prediction as the observed node's "
                "location parameter, so they apply to location-family "
                "observations only. For a non-location likelihood, sample the "
                "posterior with NUTS instead."
            )


def flat_view(
    values: dict[str, jax.Array], names: Sequence[str]
) -> tuple[jax.Array, tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    """``(x0, shapes, spans)`` for the selected latents, in the GIVEN order.

    Built by hand rather than with ``ravel_pytree``, which flattens a dict in
    SORTED key order. Sorting is fine as long as nothing else disagrees with
    it, and catastrophic as soon as something does: a report that flattened
    one way and named the other would attribute a degeneracy to the wrong
    latent with every shape still checking out.
    """
    pieces = [jnp.ravel(jnp.asarray(values[name])) for name in names]
    shapes = tuple(jnp.shape(values[name]) for name in names)
    spans: list[tuple[int, int]] = []
    offset = 0
    for piece in pieces:
        spans.append((offset, offset + piece.size))
        offset += piece.size
    return jnp.concatenate(pieces), shapes, tuple(spans)


def local_block(
    graph: Graph,
    names: Sequence[str],
    values: dict[str, jax.Array],
    *,
    priors: bool = False,
) -> LinearBlock:
    """A :class:`LinearBlock` whose ``forward`` is the tangent AT ``values``.

    See the module docstring for how this differs from
    :func:`~bayesmith.exact.block.unchecked_operator`. ``data`` and the
    codomain layout are the exact layer's own, so
    :func:`~bayesmith.exact.fisher.dense_operator` reads this handle
    unchanged.

    Args:
        graph: the model.
        names: the latents to linearize with respect to.
        values: where -- every latent, the block's members included.
        priors: whether to carry each member's DECLARED Gaussian prior in
            ``prior_mean``/``prior_std``. Default ``False``, which is the
            behaviour this function has always had and which the module
            docstring argues for: the diagnostics built on it either refuse
            non-Gaussian priors themselves or must not see a prior at all,
            and an empty dict makes
            ``fisher_information(include_prior=True)`` fail loudly instead of
            folding in a curvature nobody asked for.

            ``True`` is the migration plan's **G15**, and it is the third
            constructor that did not exist. The other two each had half of it:
            this one takes the tangent at the CALLER's point, which is the
            right Jacobian for a nonlinear model and the wrong one to attach a
            prior to by default; ``unchecked_operator`` carries the prior but
            linearizes at the domain's ZERO, because an affine map has one
            tangent everywhere -- and for a power law that is simply a
            different matrix. Neither is wrong; there was no third.

            The priors are read through
            ``bayesmith.exact.block._env_before``, which is the one place in
            this package that turns a latent's declaration into
            ``(shape, dtype, prior_mean, prior_std)``. That is deliberate
            twice over: there is no second spelling to drift, and its
            ``check_gaussian`` comes along, so a member whose prior has no
            quadratic form is refused HERE by name rather than contributing a
            silent zero to a posterior precision.

    Raises:
        NotGaussian: with ``priors=True``, if a member's prior is not a
            diagonal Gaussian. There is no such refusal with ``priors=False``,
            because there is then nothing being read off it.
    """
    names = tuple(names)
    at = {key: value for key, value in values.items() if key not in names}
    forward = isolate(graph, names, at)
    point = {name: jnp.asarray(values[name]) for name in names}
    offset, tangent = jax.linearize(forward, point)
    _, pullback = jax.vjp(forward, point)
    data, _ = observed_data_and_loc(graph, evaluate(graph, values))
    prior_mean: dict[str, jax.Array] = {}
    prior_std: dict[str, jax.Array] = {}
    if priors:
        from bayesmith.exact.block import _env_before

        _, domain = _env_before(graph, names, at)
        prior_mean = {name: domain[name][2] for name in names}
        prior_std = {name: domain[name][3] for name in names}
    return LinearBlock(
        names=names,
        shape={name: jnp.shape(point[name]) for name in names},
        dtype={name: point[name].dtype for name in names},
        offset=offset,
        forward=tangent,
        adjoint=lambda y: pullback(y)[0],
        data=data,
        prior_mean=prior_mean,
        prior_std=prior_std,
    )


def unflatten(
    x: jax.Array,
    names: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
    spans: Sequence[tuple[int, int]],
) -> dict[str, jax.Array]:
    """The flat vector back as ``{name: array}``, in the selection's layout."""
    return {
        name: jnp.reshape(x[start:stop], shape)
        for name, shape, (start, stop) in zip(names, shapes, spans, strict=True)
    }


def refuse_ambient_float32(*, doing: str) -> None:
    """Refuse before any tracing when the ambient precision is float32.

    The package rule is that ``src/`` never opens an x64 context; it refuses
    and tells the caller, the same way
    :func:`~bayesmith.exact.fisher.parameter_covariance` does. Judged by
    OUTCOME rather than by reading the config flag -- ``jnp.result_type
    (float)`` is what the tracing below will actually build with, whether the
    caller used the context manager, the process-global switch or neither.

    This must run BEFORE the linearization, not after: a graph whose
    constants were built inside x64, called from outside it, hands
    ``jax.linearize`` a float64 primal whose tangents the ambient dtype
    truncates, and the failure then arrives as a bare shape/dtype
    inconsistency from inside JAX naming neither the cause nor the remedy.

    **Why this is not replaced by a float32 tolerance, measured over a family
    rather than argued.** The obvious alternative is to keep the diagnostic and
    move the rank cut with the dtype. ``docs/probes/probe_13_d9_precision_
    policy.py`` sweeps a two-component power law whose conditioning is dialled
    across ten decades and asks, of every candidate cut from ``1e-8`` up to
    ``sqrt(eps)`` in float32, whether it reproduces float64's verdicts. None
    does. The reason is not that the cut is hard to place: float64's smallest
    singular value tracks the model (5.1e-6 down to 5.2e-16) while float32's
    sits on its own roundoff floor near 1e-7 from the second decade onward,
    non-monotonic, because it is noise. Two models float64 separates by two
    decades come back indistinguishable, so no cut can follow -- derived,
    tuned, or chosen per model. ``tests/diagnose/test_precision_policy.py``
    keeps that runnable.
    """
    if jnp.result_type(float) == jnp.float64:
        return
    raise GraphError(
        f"{doing} was asked for with float32 as the ambient precision, and "
        "this diagnostic's verdict lives many decades below float32's own "
        "roundoff (~1e-7 relative): measured on the motivating model, the "
        "null direction sits at 7.5e-17 of the largest singular value in "
        "double precision and surfaces at 3.1e-8 in single, ABOVE the 1e-8 "
        "rank tolerance, so float32 reports the degenerate model as "
        "identified. Moving the tolerance instead does not work and was "
        "measured over a family, not argued: float32's smallest singular "
        "value sits on its own roundoff floor rather than on anything about "
        "the model, so two models float64 separates by two decades come back "
        "indistinguishable and no cut can follow them. Run the call inside "
        "`with jax.enable_x64(True):`, building the graph inside the block so "
        "its constants and data are traced at the wider dtype -- wrapping only "
        "the call leaves them at float32 and is refused separately."
    )


def refuse_single_precision(array: jax.Array, *, doing: str) -> None:
    """Refuse a float32 result from a graph that pins its own arithmetic.

    :func:`refuse_ambient_float32` has already run by the time this is
    reached, so a non-float64 array here means the GRAPH truncates -- an
    ``astype`` in a ``det`` fn, a float32 ``const``, float32 data on an
    observed node -- and the cast is what to remove.

    What neither guard can catch is stated rather than papered over: a model
    that rounds an INTERMEDIATE to float32 and promotes back carries an
    honest float64 dtype and roundoff at 1e-7 anyway. Nothing cheap
    distinguishes that; if a model does it, its weakest identified direction
    sits near 1e-7 and should not be believed.
    """
    if jnp.asarray(array).dtype == jnp.float64:
        return
    raise GraphError(
        f"{doing} came back {jnp.asarray(array).dtype} even though the "
        "ambient precision is float64, so some node in the graph pins its "
        "own arithmetic to single precision -- an astype in a det fn, a "
        "float32 const, or float32 observed data. The diagnostic's verdict "
        "lives many decades below float32's roundoff (measured: a null "
        "direction at 7.5e-17 in double surfaces at 3.1e-8 in single, above "
        "the 1e-8 rank tolerance), so the cast is what to remove."
    )


__all__ = [
    "refuse_ambient_float32",
    "resolve_names",
    "latent_values",
    "check_differentiable",
    "check_observed_have_locs",
    "flat_view",
    "local_block",
    "unflatten",
    "refuse_single_precision",
]
