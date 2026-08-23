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

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

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
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.graph.graph import Graph

#: Probe magnitudes, as multiples of each latent's declared prior standard
#: deviation. Spans six orders of magnitude on purpose: curvature that is
#: invisible near the prior's centre is what a sampler wanders into.
DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1.0, 1e3)

#: How many values of the OUTSIDE latents the claim is checked at, the
#: caller's own ``at`` included. Extras are drawn from the graph's own prior.
DEFAULT_AT_POINTS: int = 3

#: Departure from affinity **in units of the noise sigma**, above which a
#: ``linear_in`` claim is refused. Not a relative error, so ``1e4 * eps``
#: would be meaningless for it: the likelihood divides every residual by
#: sigma, so sigma is the unit in which "this departure cannot change the
#: posterior" is a statement with content. At 1e-3, a claim that slips
#: through moves no residual by as much as a thousandth of a noise width.
#:
#: Pinned by measurement over all 48 fixture rows at both dtypes
#: (``docs/superpowers/plans/2026-08-23-p3b-task1-verdicts.md``): every named
#: false claim sits at least 4.9e+07x above it, and the worst honest fixture
#: 2.05e+04x below it in float64 -- in float32 every honest fixture's
#: above-floor departure is exactly 0.0, so nothing bounds it from below
#: there at all.
#:
#: **The number only means anything with the per-element roundoff floor in
#: place.** Ungated, this measure grows with the offset-to-noise ratio rather
#: than with curvature: an exactly affine model breaches 1e-3 at a ratio of
#: 1e2 in float32, and the window of usable thresholds is empty. The floor,
#: not this constant, is what keeps honest wide-dynamic-range models --
#: ``test_a_true_claim_with_real_roundoff_passes_at_any_offset_ratio`` is
#: where that is pinned.
WEIGHTED_RTOL: float = 1e-3


def _worse(current: float, value: float) -> float:
    """``max`` that PROPAGATES NaN.

    Python's builtin returns its first argument whenever the comparison is
    False, so ``max(0.0, nan)`` is ``0.0`` -- which would report a column of
    clean zeros for a probe whose verdict is REFUSE, and would quietly break
    ``test_affinity_errors_treats_nan_as_a_failure_in_isolation``'s reading of
    the returned number. Measured during Task 1, not assumed.
    """
    if math.isnan(current) or math.isnan(value):
        return math.nan
    return max(current, value)


def _reported(values: jax.Array, kept: jax.Array) -> float:
    """The worst of ``values`` among the elements the roundoff floor kept.

    That is the number the criterion actually judged, so it is the number the
    refusal message quotes -- reporting the raw maximum instead would print
    values above the threshold beside a verdict of "pass" and read as a
    broken guard.

    NaN survives the mask: ``nan > floor`` is False, so an unusable element
    would otherwise be masked out and the column would report ``0.0`` for a
    probe that is a failure precisely because it is unusable.
    """
    if not bool(jnp.all(jnp.isfinite(values))):
        return math.nan
    return float(jnp.max(jnp.where(kept, values, 0.0)))


def _leaf_departures(
    actual: jax.Array,
    baseline: jax.Array,
    predicted: jax.Array,
    sigma: jax.Array,
    *,
    rtol: float,
    epsilon: float,
    tiny: float,
) -> tuple[float, float, bool]:
    """One codomain leaf's two departure columns, and its verdict.

    Everything here is **per element**. A maximum over the leaf -- or over
    the whole codomain, which is what this used to take -- lets a bright
    entry supply both the yardstick and the roundoff floor for a faint one,
    and each of those dilutions is enough on its own to hide a false claim.

    Returns:
        ``(relative, sigma_weighted, refused)``. The first two are the worst
        values among the elements clearing the roundoff floor -- the numbers
        the criteria actually judged, and so the numbers a refusal quotes --
        or NaN if the column was unusable anywhere. ``refused`` is True if
        either criterion fails at any element.
    """
    # Measure against the VARIATION, not the total: a large constant offset
    # would otherwise hide a completely nonlinear response. The divisor is
    # floored at `finfo(dtype).tiny`, NOT at the 1e-300 literal this used to
    # carry -- measured, 1e-300 underflows to 0.0 in float32, so an element
    # the block cannot move at all gives 0/0 = NaN and the finiteness branch
    # below reads that as a FAILURE. `two_observations`'s covariate grid
    # contains an exact zero, so that alone refused an entirely honest model.
    variation = jnp.abs(actual - baseline)
    departure = jnp.abs(actual - predicted)
    relative = departure / jnp.maximum(variation, tiny)
    # A departure smaller than the arithmetic's OWN noise floor is not
    # evidence of curvature; without this the relative measure explodes at
    # small probes, where the variation is vanishing but roundoff is not, and
    # rejects perfectly linear blocks. The floor is set by the magnitudes
    # actually being differenced AT THIS PROBE -- not by a constant, which
    # would exempt every model whose prediction is small in its own units.
    above_floor = departure > 1e4 * epsilon * jnp.maximum(
        jnp.abs(actual), jnp.abs(baseline)
    )
    # In the units the likelihood divides by -- and gated by the SAME floor.
    # Ungated it measures DYNAMIC RANGE rather than curvature: measured on the
    # exactly affine `mu = (w + big) X`, it reaches 2.44e-02 at an
    # offset-to-noise ratio of 1e2 in float32 and 2.50e+01 at 1e17 in float64,
    # with no curvature anywhere. Gated, every one of those is exactly 0.0.
    weighted = departure / jnp.abs(sigma)
    # NaN must count as a FAILURE: `nan > rtol` is False, so a naive
    # comparison reads an unusable probe as evidence of linearity. Each
    # criterion is judged on its OWN finiteness -- sharing one check would let
    # a 0/0 in the relative column condemn a perfectly readable weighted one.
    refused = (
        bool(jnp.any((relative > rtol) & above_floor))
        or bool(jnp.any((weighted > WEIGHTED_RTOL) & above_floor))
        or not bool(jnp.all(jnp.isfinite(relative)))
        or not bool(jnp.all(jnp.isfinite(weighted)))
    )
    return (
        _reported(relative, above_floor),
        _reported(weighted, above_floor),
        refused,
    )


def affinity_errors(
    g: Callable[[Any], Any],
    zero: Any,
    probe_at: Callable[[int, float], Any],
    scales: Sequence[float],
    rtol: float | None,
    *,
    sigma: dict[str, jax.Array],
    at_description: str = "the linearisation point",
) -> tuple[dict[float, float], list[float], float, dict[float, tuple[float, float]]]:
    """Compare a map against its own linearization at zero, **per element**.

    Every number below comes from ``g``, ``zero``, ``sigma`` and the probe
    alone, so a single-latent and a grouped check cannot drift into measuring
    different things. Ported from rheplicant, generalised to a pytree
    codomain.

    Args:
        sigma: ``{observed: scale}`` read at the same point ``zero`` is
            anchored at. The unit of the second criterion below.
        at_description: names the point ``zero`` is anchored at -- used only
            by the non-finite-baseline error below, to say where the graph
            broke rather than blaming whichever ``linear_in`` happens to be
            under test.

    Returns:
        ``(worst, failed, rtol, columns)`` -- ``worst[scale]`` is the larger
        of the two criteria at that scale (NaN if either was unusable),
        ``failed`` lists the refused scales, ``rtol`` is the one actually
        used, and ``columns[scale]`` is ``(relative, sigma-weighted)`` so a
        refusal can say WHICH criterion fired.

    Two changes from the original, both forced by measurement:

    * **Per element, not a max over the whole codomain.** ``variation`` and
      ``floor`` were each a maximum over EVERY leaf, so a bright leaf
      supplied both the yardstick and the roundoff floor for a faint one.
      Measured: an honest 1e17 component beside a false ``linear_in`` claim
      on a 1e-2 one reported 2.57e-14 in float32 and PASSED, while the faint
      node alone was correctly refused at 4.93e+00 -- and the resulting
      "exact" posterior was 202 true posterior standard deviations wrong. The
      same dilution happens between ELEMENTS of one leaf, which is the
      realistic case: a spectrum with one bright foreground channel and five
      faint signal ones.
    * **A second criterion in units of sigma.** The relative measure is 0/0
      on an element the block does not move at all; the weighted one is not,
      and it is what the likelihood actually cares about. Either one failing
      is a refusal, and each has a case the other cannot see -- pinned by
      ``test_the_relative_criterion_is_load_bearing_where_sigma_hides_the_curvature``
      and ``test_the_dilution_is_caught_within_a_single_array_too``.

    The original already suspected half of this. Its own floor comment said
    the floor must not be set "by the baseline alone, which would let an
    unrelated bright component disable the check" -- and then took a global
    maximum anyway. This finishes that thought.

    :func:`~bayesmith.exact.gaussian.check_gaussian` made the elementwise
    choice first, and argues it in its docstring: a summed comparison dilutes
    a localised defect by the magnitudes of the correct entries.

    Note:
        ``sigma`` is divided by, not validated here -- :func:`check_linearity`
        clears it first with :func:`_refuse_unusable_scale`, which names the
        offending node. Called directly with a zero, negative or non-finite
        scale, this function still refuses, but reports it as curvature.
    """
    baseline, tangent = jax.linearize(g, zero)
    if not all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(baseline)):
        raise StructureError(
            f"the prediction is already non-finite at {at_description}, before "
            "any probe is taken -- so nothing here is a statement about any "
            "linear_in declaration. Some part of the graph overflows at the "
            "values the latents outside the block are sitting at. If those "
            "values came from the default prior draws, a heavy-tailed or very "
            "wide prior on an outside latent is the usual cause; pass "
            "`at_points=` explicitly to check where the model is actually "
            "used. Measured example: an outside latent with a Cauchy(0, 1e6) "
            "prior feeding exp(|z|/50) overflows float32 on about 99.7% of "
            "draws, and the resulting inf-minus-inf used to be reported as a "
            "false linearity failure of an unrelated, genuinely affine block "
            "member."
        )
    dtype = jnp.result_type(*jax.tree.leaves(baseline))
    if rtol is None:
        rtol = 1e4 * float(jnp.finfo(dtype).eps)
    epsilon = float(jnp.finfo(dtype).eps)
    tiny = float(jnp.finfo(dtype).tiny)

    errors: dict[float, float] = {}
    columns: dict[float, tuple[float, float]] = {}
    verdicts: dict[float, bool] = {}
    for index, scale in enumerate(scales):
        probe = probe_at(index, scale)
        actual = g(probe)
        predicted = jax.tree.map(lambda b, t: b + t, baseline, tangent(probe))
        worst_relative = 0.0
        worst_weighted = 0.0
        bad = False
        # `baseline` comes out of `jax.linearize`, and JAX's dict pytree sorts
        # keys unconditionally on flatten -- so it is ALREADY sorted and a
        # `sorted()` here would be a provable no-op (P3a Task 9). Iterating it
        # directly says that, instead of implying a guarantee this loop makes.
        for key in baseline:
            relative, weighted, refused = _leaf_departures(
                actual[key],
                baseline[key],
                predicted[key],
                sigma[key],
                rtol=rtol,
                epsilon=epsilon,
                tiny=tiny,
            )
            worst_relative = _worse(worst_relative, relative)
            worst_weighted = _worse(worst_weighted, weighted)
            bad = bad or refused
        errors[scale] = _worse(worst_relative, worst_weighted)
        columns[scale] = (worst_relative, worst_weighted)
        verdicts[scale] = bad

    failed = sorted(scale for scale, is_bad in verdicts.items() if is_bad)
    return errors, failed, rtol, columns


def prior_at_points(
    graph: Graph, names: tuple[str, ...], count: int, key: jax.Array
) -> list[dict[str, Any]]:
    """``count`` alternative values for the latents OUTSIDE the block.

    Drawn from the graph's own prior, through the NumPyro bridge -- so they
    cover exactly the range the model itself considers plausible, and they
    work for a non-Gaussian outside latent (which is the usual case: a latent
    is outside the block precisely because it is not conjugate).

    Raises:
        StructureError: if an outside latent's own prior has no sampler --
            an improper or reference prior, typically -- propagated from a
            NumPyro ``NotImplementedError`` with a pointer to ``at_points=``.
    """
    outside = [name for name in graph.latents if name not in set(names)]
    if not outside:
        # Nothing outside the block, so every at-point is the same empty
        # dict -- drawing it from the prior would cost a full NumPyro
        # forward trace to learn that, on exactly the small-model case where
        # the block spans everything.
        return [{} for _ in range(count)]

    from numpyro import handlers

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    model = to_numpyro(graph)
    points: list[dict[str, Any]] = []
    for index in range(count):
        try:
            traced = handlers.trace(
                handlers.seed(model, jax.random.fold_in(key, index))
            ).get_trace()
        except NotImplementedError as exc:
            raise StructureError(
                "a latent outside the block cannot be sampled from its own "
                "prior, so the default at-points cannot be drawn. An improper "
                "or reference prior has no sampler by construction. Pass "
                "`at_points=` explicitly -- the values the model is actually "
                "used at are a better choice than prior draws anyway when the "
                "prior is improper."
            ) from exc
        points.append({name: traced[name]["value"] for name in outside})
    return points


def _refuse_affinity(
    names: tuple[str, ...],
    where: str,
    columns: dict[float, tuple[float, float]],
    failed: list[float],
    used_rtol: float,
) -> None:
    """Build and raise ``check_linearity``'s affine-departure failure.

    Pure message construction with no control flow of its own -- the caller
    still decides whether to call it. Extracted so ``check_linearity``'s own
    body stays under this project's <50-line function guideline.

    Reports BOTH criteria against BOTH thresholds, per probe. The guard is a
    disjunction, so one number against one tolerance would be unreadable half
    the time: a reader would see a value under the tolerance printed beside
    it and conclude the guard was broken, when the other criterion is what
    fired.
    """
    subject = (
        f"latent {names[0]!r} is declared linear, but the prediction is "
        "not affine in it"
        if len(names) == 1
        else f"latents {list(names)} are not JOINTLY affine -- each "
        "conditional may well be, which is exactly why this is not "
        "caught one latent at a time"
    )
    detail = ", ".join(
        f"{scale:g}x -> {relative:.2e} | {weighted:.2e}"
        for scale, (relative, weighted) in columns.items()
    )
    raise StructureError(
        f"{subject}: departure from its own linearization is too large at "
        f"{failed} times each latent's declared prior width, evaluated at "
        f"{where}. Two criteria, per element, over the elements clearing the "
        f"per-element roundoff floor -- either alone is a refusal: the "
        f"relative departure against rtol={used_rtol:.2e}, and the departure "
        f"in units of the noise sigma against weighted_rtol="
        f"{WEIGHTED_RTOL:.2e}. Worst per probe (relative | sigma-weighted): "
        f"{detail}. Either drop the linear_in declaration, or re-parameterize "
        "so the model really is affine there. For a group that is only "
        "pairwise affine, split it into separate blocks and alternate."
    )


def _refuse_unusable_scale(sigma: dict[str, jax.Array]) -> None:
    """Refuse a noise scale the sigma-weighted criterion cannot be stated in.

    :func:`affinity_errors` measures departure from affinity in units of
    sigma, so a scale that is zero, negative or non-finite makes that column
    unreadable. Without this guard the refusal still happens -- the
    finiteness branch catches inf and NaN -- but it is reported as "latent
    'w' is declared linear, but the prediction is not affine in it", which is
    wrong in every word when the model IS affine and the node's scale
    expression is what is broken. It sends the modeller to rewrite the one
    part of the model that is correct.

    Measured before this guard, on an exactly affine ``mu = w * X``: a scale
    of zero, a single zero entry among honest ones, and a scale of NaN all
    produced exactly that message.

    A NEGATIVE scale was worse than mis-attributed -- it **passed silently**.
    The weighted column divides by ``abs(sigma)`` and so cannot tell -0.5
    from +0.5, while
    :func:`~bayesmith.exact.gaussian.check_gaussian` refuses a non-positive
    scale by name. A guard written only against non-finite values would leave
    that hole open, so the positivity half is load-bearing on its own.

    This is the mis-attribution class :func:`affinity_errors`'s non-finite
    BASELINE branch and its ``at_description`` argument already exist to
    prevent; the pattern is followed rather than reinvented.

    Deliberately mirrors ``check_gaussian``'s wording instead of calling it.
    That function answers a different question -- whether a node's own
    ``log_prob`` matches the loc and scale read off it -- and runs when the
    block is built, which is after this. What the two share is the sentence a
    user needs, not the check.
    """
    # `sigma` comes from `noise_std_at`, a plain dict comprehension over
    # `graph.observed`, so it carries DECLARATION order rather than JAX's
    # sorted-pytree order. This `sorted` is therefore load-bearing: without
    # it, which node gets named when two are broken depends on declaration
    # order (P3a Task 9's criterion for telling a documenting `sorted` from a
    # guarding one).
    for name in sorted(sigma):
        scale = jnp.asarray(sigma[name])
        if bool(jnp.all(jnp.isfinite(scale) & (scale > 0))):
            continue
        raise StructureError(
            f"observed node {name!r} has a scale that is not strictly "
            f"positive and finite (min {float(jnp.min(scale)):g}), so the "
            "departure from affinity cannot be measured in units of it. "
            "**This is a fault in that node's scale expression, not in any "
            "linear_in declaration** -- the linearity check stops here rather "
            "than reporting an unreadable column as curvature and blaming a "
            "latent whose model may well be affine. A conjugate solve weights "
            "by 1/scale**2, so a zero or negative sigma is an infinite or "
            "negative weight rather than a tight constraint. Add a floor to "
            "the expression that produces it."
        )


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
        rtol: tolerance on the **per-element** relative departure from
            affinity. Default ``1e4 * eps`` of the prediction's dtype --
            1.19e-03 in float32, which is what this package runs in unless a
            caller opens ``jax.enable_x64``. It is one of two criteria: the
            other, :data:`WEIGHTED_RTOL`, measures the same departure in
            units of the noise sigma and is not a relative error, so this
            keyword does not reach it.
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
        ``{at_point_index: {scale: departure}}``, where each departure is the
        worse of the two criteria at that scale, over the elements clearing
        the roundoff floor -- useful for reporting how linear a block is, not
        only whether it passes. The two are on different scales (one relative,
        one in units of sigma), so the number says how far from affine the
        block is, not which criterion it came from; the refusal message
        separates them.

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
        StructureError: if any element of any prediction, at any scale at any
            at-point, departs from affinity by more than ``rtol`` relatively
            or by more than :data:`WEIGHTED_RTOL` in units of its own noise
            sigma, while also clearing that element's roundoff floor; if the
            prediction is already non-finite at an at-point
            BEFORE any probe is taken, attributed to that at-point rather
            than misdiagnosed as a failure of this block's own
            ``linear_in`` (see :func:`affinity_errors`); or if an outside
            latent's prior has no sampler (propagated from
            :func:`prior_at_points`).
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
        # Names this at-point for BOTH errors affinity_errors can raise.
        where = (
            "the caller's own `at`"
            if point_index == 0
            else f"prior draw {point_index} of the outside latents"
        )

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

        sigma = noise_std_at(graph, {**point, **zero})
        _refuse_unusable_scale(sigma)
        errors, failed, used_rtol, columns = affinity_errors(
            g,
            zero,
            probe_at,
            scales,
            rtol,
            sigma=sigma,
            at_description=where,
        )
        collected[point_index] = errors
        if failed:
            _refuse_affinity(names, where, columns, failed, used_rtol)
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
