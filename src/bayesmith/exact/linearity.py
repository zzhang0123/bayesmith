"""Checking the linear_in claim before anything exploits it.

``Deterministic(linear_in=("w",))`` promises that, holding every other latent
fixed, every prediction is an **affine** function of ``w``::

    prediction(w) = A w + b

The promise is checkable, and this module checks it: :func:`check_linearity`
compares the model against its own linearization at zero, at several probe
magnitudes and at several values of the latents outside the block. A false
declaration would otherwise produce a confident, wrong posterior instead of
an error.

**Two entry points, not a flag.** :func:`linear_operator` checks and then
builds, and is what callers should reach for.
:func:`~bayesmith.exact.block.unchecked_operator` skips the check and says so
in its name -- for inside a Gibbs sweep, where the check is hoisted out of the
loop deliberately. rheplicant spells this as ``linear_operator(check=True)``,
which makes the most natural call name the one that is one keyword away from
unsafe.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.errors import StructureError
from bayesmith.exact.block import (
    LinearBlock,
    _env_before,
    _refuse_internal_ancestry,
    _refuse_missing_observed,
    _validated_at,
    _validated_names,
    isolate,
    unchecked_operator,
)
from bayesmith.graph.graph import Graph

#: Probe magnitudes, as multiples of each latent's declared prior standard
#: deviation. Spans six orders of magnitude on purpose: curvature that is
#: invisible near the prior's centre is what a sampler wanders into.
DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1.0, 1e3)

#: How many values of the OUTSIDE latents the claim is checked at, the
#: caller's own ``at`` included. Extras are drawn from the graph's own prior.
DEFAULT_AT_POINTS: int = 3


def _biggest(tree: Any) -> float:
    leaves = jax.tree.leaves(tree)
    return max(float(jnp.max(jnp.abs(leaf))) for leaf in leaves)


def affinity_errors(
    g: Callable[[Any], Any],
    zero: Any,
    probe_at: Callable[[int, float], Any],
    scales: Sequence[float],
    rtol: float | None,
) -> tuple[dict[float, float], list[float], float]:
    """Compare a map against its own linearization at zero, probe by probe.

    Every number below comes from ``g``, ``zero`` and the probe alone, so a
    single-latent and a grouped check cannot drift into measuring different
    things. Ported from rheplicant, generalised to a pytree codomain.
    """
    baseline, tangent = jax.linearize(g, zero)
    dtype = jnp.result_type(*jax.tree.leaves(baseline))
    if rtol is None:
        rtol = 1e4 * float(jnp.finfo(dtype).eps)
    epsilon = float(jnp.finfo(dtype).eps)

    errors: dict[float, float] = {}
    verdicts: dict[float, bool] = {}
    for index, scale in enumerate(scales):
        probe = probe_at(index, scale)
        actual = g(probe)
        predicted = jax.tree.map(lambda b, t: b + t, baseline, tangent(probe))
        # Measure against the VARIATION, not the total: a large constant
        # offset would otherwise hide a completely nonlinear response.
        variation = _biggest(jax.tree.map(jnp.subtract, actual, baseline))
        departure = _biggest(jax.tree.map(jnp.subtract, actual, predicted))
        errors[scale] = departure / max(variation, 1e-300)

        # A departure smaller than the arithmetic's OWN noise floor is not
        # evidence of curvature; without this the relative measure explodes at
        # small probes, where the variation is vanishing but roundoff is not,
        # and rejects perfectly linear blocks. The floor is set by the
        # magnitudes actually being differenced AT THIS PROBE -- not by a
        # constant, which would exempt every model whose prediction is small
        # in its own units, and not by the baseline alone, which would let an
        # unrelated bright component disable the check.
        floor = 1e4 * epsilon * max(_biggest(actual), _biggest(baseline))
        # NaN must count as a FAILURE: `nan > rtol` is False, so a naive
        # comparison reads an unusable probe as evidence of linearity.
        finite = np.isfinite(errors[scale]) and np.isfinite(departure)
        verdicts[scale] = (not finite) or (errors[scale] > rtol and departure > floor)

    failed = sorted(scale for scale, bad in verdicts.items() if bad)
    return errors, failed, rtol


def prior_at_points(
    graph: Graph, names: tuple[str, ...], count: int, key: jax.Array
) -> list[dict[str, Any]]:
    """``count`` alternative values for the latents OUTSIDE the block.

    Drawn from the graph's own prior, through the NumPyro bridge -- so they
    cover exactly the range the model itself considers plausible, and they
    work for a non-Gaussian outside latent (which is the usual case: a latent
    is outside the block precisely because it is not conjugate).
    """
    from numpyro import handlers

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    model = to_numpyro(graph)
    outside = [name for name in graph.latents if name not in set(names)]
    points: list[dict[str, Any]] = []
    for index in range(count):
        traced = handlers.trace(
            handlers.seed(model, jax.random.fold_in(key, index))
        ).get_trace()
        points.append({name: traced[name]["value"] for name in outside})
    return points


def check_linearity(
    graph: Graph,
    names: Iterable[str],
    at: dict[str, Any] | None = None,
    *,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
    at_points: Sequence[dict[str, Any]] | None = None,
    key: jax.Array | None = None,
) -> dict[int, dict[float, float]]:
    """Verify every prediction really is affine in a block -- or in a group.

    Costs one linearization plus one forward evaluation per scale per
    at-point: with the defaults, three of each.

    Args:
        graph: the model under test.
        names: the latents in the block. Checked **jointly**, which is
            strictly stronger than checking each in turn. A gain and an
            antenna temperature are each affine given the other and their
            product is not affine in the pair, so a group holding both is
            refused here rather than solved as if it were linear.
        at: values for the latents OUTSIDE the block.
        scales: probe magnitudes, as multiples of each latent's own declared
            prior standard deviation -- per element, so a latent whose prior
            width varies across its entries is probed accordingly.
        rtol: tolerance on the relative departure from affinity. Default
            ``1e4 * eps`` of the prediction's dtype.
        at_points: values of the outside latents to check at. Defaults to
            ``at`` plus ``DEFAULT_AT_POINTS - 1`` draws from the graph's own
            prior. **Passing a single point is how a check becomes a
            moderate-parameter probe**, which is the failure mode
            ``boundary-validation.md`` exists to prevent; do it only when the
            model is used at exactly one outside value.
        key: PRNG key for probes and prior draws. Fixed by default, so the
            check is reproducible. Per-latent sub-keys are folded in by
            position in the SORTED names, so permuting ``names`` probes the
            same points and returns the same verdict.

    Returns:
        ``{at_point_index: {scale: relative error}}`` -- useful for reporting
        how linear a block is, not only whether it passes.

    Raises:
        GraphError: propagated from :func:`~bayesmith.exact.block._validated_names`
            and :func:`~bayesmith.exact.block._validated_at` -- the same
            misuses :func:`~bayesmith.exact.block.unchecked_operator` refuses,
            checked here BEFORE any linearization runs so a malformed ``at``
            fails with that message rather than a confusing one from three
            layers down; or if the graph has no observed node, for the same
            reason and by the same shared guard
            (:func:`~bayesmith.exact.block._refuse_missing_observed`) that
            :func:`~bayesmith.exact.block.unchecked_operator` uses -- checked
            here too because :func:`linear_operator` calls this function
            BEFORE ``unchecked_operator``, so a guard living only there would
            never be reached.
        StructureError: if any scale at any at-point departs from affinity by
            more than ``rtol`` while also exceeding the per-probe roundoff
            floor.
        NotGaussian: propagated from the block machinery.
    """
    names = _validated_names(graph, names)
    at = _validated_at(graph, names, at)
    _refuse_internal_ancestry(graph, names)
    _refuse_missing_observed(graph)
    key = jax.random.key(0) if key is None else key

    if at_points is None:
        at_points = [
            at,
            *prior_at_points(
                graph, names, DEFAULT_AT_POINTS - 1, jax.random.fold_in(key, 7919)
            ),
        ]

    ordered = sorted(names)
    collected: dict[int, dict[float, float]] = {}
    for point_index, point in enumerate(at_points):
        _, domain = _env_before(graph, names, point)
        g = isolate(graph, names, point)
        zero = {n: jnp.zeros(domain[n][0], dtype=domain[n][1]) for n in names}
        point_key = jax.random.fold_in(key, point_index)

        def probe_at(index: int, scale: float, _domain=domain, _k=point_key):
            root = jax.random.fold_in(_k, index)
            return {
                member: _domain[member][3]
                * scale
                * jax.random.normal(
                    jax.random.fold_in(root, position),
                    _domain[member][0],
                    dtype=_domain[member][1],
                )
                for position, member in enumerate(ordered)
            }

        errors, failed, used_rtol = affinity_errors(g, zero, probe_at, scales, rtol)
        collected[point_index] = errors
        if failed:
            subject = (
                f"latent {names[0]!r} is declared linear, but the prediction is "
                "not affine in it"
                if len(names) == 1
                else f"latents {list(names)} are not JOINTLY affine -- each "
                "conditional may well be, which is exactly why this is not "
                "caught one latent at a time"
            )
            detail = ", ".join(f"{s:g}x -> {e:.2e}" for s, e in errors.items())
            where = (
                "the caller's own `at`"
                if point_index == 0
                else f"prior draw {point_index} of the outside latents"
            )
            raise StructureError(
                f"{subject}: departure from its own linearization exceeds "
                f"rtol={used_rtol:.2e} (above the per-probe roundoff floor) at "
                f"{failed} times each latent's declared prior width, evaluated at "
                f"{where} ({detail}). Either drop the linear_in declaration, or "
                "re-parameterize so the model really is affine there. For a group "
                "that is only pairwise affine, split it into separate blocks and "
                "alternate."
            )
    return collected


def linear_operator(
    graph: Graph,
    names: Iterable[str],
    at: dict[str, Any] | None = None,
    *,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
    at_points: Sequence[dict[str, Any]] | None = None,
    key: jax.Array | None = None,
) -> LinearBlock:
    """Check the linearity claim, then export the block. **The entry point.**

    Costs three forward evaluations per at-point more than
    :func:`~bayesmith.exact.block.unchecked_operator`, and buys the class of
    silent, confident errors that a false ``linear_in`` produces.

    In a Gibbs sweep, call this once outside the loop and
    ``unchecked_operator`` inside it: the claim is a property of the model,
    not of the sweep, so re-checking every sweep pays for the same answer
    repeatedly. The at-points this checks at are what make that safe.
    """
    check_linearity(
        graph, names, at, scales=scales, rtol=rtol, at_points=at_points, key=key
    )
    return unchecked_operator(graph, names, at)
