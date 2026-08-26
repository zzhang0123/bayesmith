"""Factor partition: as many exact blocks as the model has factors.

:func:`~bayesmith.dispatch.classify.partition` finds THE exact block -- one,
by declaration, with everything else falling to NUTS. On a multilinear model
that is the wrong shape: in ``d = g * (B t)`` with both latents declared,
``g`` and ``t`` are each affine given the other and not affine together, so
the single block fails its joint check and the WHOLE model lands in NUTS --
measured, that is exactly what ``partition`` returns -- when the correct
treatment is an alternating Gibbs sweep of two exact blocks, one per factor.

This module derives that partition, and the rule it uses is a fact of
calculus rather than a heuristic: for latents each already verified affine
on its own, every diagonal block of the group's Hessian vanishes, so JOINT
affinity is exactly the vanishing of the off-diagonal blocks -- a property
of PAIRS. Probing the ``C(n, 2)`` pairs therefore settles all ``2^n - n - 1``
subsets, and the pairwise verdicts colour a graph whose groups are the
blocks. :func:`first_fit` is that colouring: deterministic, in declaration
order, not guaranteed minimal -- a needless extra block costs mixing, never
correctness -- and a caller who knows the true factors writes them by hand.

**Log-linear factors are discovered by the same probe, not declared.** A
latent the prediction is ``exp``-affine in has no ``linear_in`` that could
be true of it, so after the linear factors are grouped, the remainder is
probed again on the :func:`~bayesmith.exact.loglinear.log_space` graph --
where such a latent IS affine, the noise is additive, and its scale is
constant. Latents that pass group into ``"log-gcr"`` blocks by the same
pairwise rule; what remains after both probes is the NUTS block.

**Execution reuses the machinery it finds blocks for.** A sweep updates each
closed-form block in turn -- ``gcr_sample`` on the original graph for a
linear block, on the transformed graph for a log one, conditioning every
block on the latest values of all the others -- and hands any NUTS remainder
to ``HMCGibbs`` exactly as the single-block assembly does. The one method
this module refuses to sweep is ``"gcr+mh"``: a Metropolis accept per block
per sweep is a different correctness argument from
:func:`~bayesmith.exact.gibbs._mh_step`'s single-block one, and until it is
written down this module says so rather than running an unargued sampler.

The pairwise rule, the log-space scenarios and their thresholds are ported
from ``rheplicant.inference.partition`` / ``loglinear``; the probes they run
on are this package's own -- per element, sigma-weighted, swept over the
outside latents' priors -- which the originals were not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.dispatch.classify import (
    SIGMA_RTOL,
    _declares_linear_in,
    _is_gaussian,
    _sigma_needs_rebuild,
    prior_environment,
)
from bayesmith.dispatch.plan import (
    CONVERGENCE_TARGET,
    Block,
    _kappa_at,
    kappa_interval,
    kappa_upper,
    working_epsilon,
)
from bayesmith.errors import GraphError, NotGaussian, NotLogLinear, StructureError
from bayesmith.exact.block import _ancestors, unchecked_operator
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.linearity import DEFAULT_SCALES, check_linearity
from bayesmith.exact.loglinear import LOG_DEFAULT_SCALES, LogSpace, log_space
from bayesmith.exact.solve import gcr_sample
from bayesmith.graph.graph import Graph

#: The methods a factor block can carry. ``"gcr"`` and ``"log-gcr"`` are the
#: two :func:`sample_factors` sweeps; ``"nuts"`` is the remainder block.
#: There is deliberately no ``"gcr+mh"`` row -- see the module docstring.
FACTOR_METHODS: tuple[str, ...] = ("gcr", "log-gcr", "nuts")


def first_fit(
    names: Sequence[str], compatible: Callable[[str, str], bool]
) -> list[list[str]]:
    """Group ``names`` so every pair within a group is compatible.

    First-fit in the caller's order: each name joins the first existing group
    whose every member it is compatible with, and otherwise opens a new one.
    One function for however many probes call it -- the linear pass, the
    log-linear pass, and any other package that partitions by a pairwise
    predicate -- so the grouping RULE cannot drift between them.

    No pair is asked twice, and no cache is needed to say so: the groups are
    disjoint, so a name meets any other name in at most one group, and
    ``all()`` short-circuits the moment a group is ruled out.
    """
    groups: list[list[str]] = []
    for name in names:
        for group in groups:
            if all(compatible(name, member) for member in group):
                group.append(name)
                break
        else:
            groups.append([name])
    return groups


class FactorPlan(eqx.Module):
    """The factor partition of one graph, ready to sweep.

    Attributes:
        blocks: closed-form blocks first (``"gcr"`` then ``"log-gcr"``, each
            in discovery order), then at most one ``"nuts"`` block holding
            everything neither probe accepted -- whose ``reason`` names, per
            latent, why not.
        log_space: the transformed graph the ``"log-gcr"`` blocks solve
            against, or ``None`` when there are none. Carried on the plan
            because every rebuild in a sweep needs it, and reconstructing it
            per sweep would re-run the scenario probes hundreds of times for
            an answer that cannot change.
    """

    blocks: tuple[Block, ...]
    log_space: LogSpace | None

    @property
    def exact(self) -> tuple[Block, ...]:
        """The closed-form blocks, in sweep order."""
        return tuple(b for b in self.blocks if b.method != "nuts")

    @property
    def nuts(self) -> tuple[str, ...]:
        """The latents left to NUTS."""
        for block in self.blocks:
            if block.method == "nuts":
                return block.latents
        return ()

    def __str__(self) -> str:
        lines = []
        for index, block in enumerate(self.blocks):
            names = "{" + ", ".join(block.latents) + "}"
            note = f"  ({block.reason})" if block.method == "nuts" and block.reason else ""
            lines.append(f"block {index}  {names}  {block.method}{note}")
        return "\n".join(lines)


def _measured(
    graph: Graph,
    group: tuple[str, ...],
    env: dict[str, Any],
    key: jax.Array,
    *,
    sweep_outside: bool = True,
) -> tuple[tuple[float, float], float, float]:
    """``(kappa, tol, epsilon)`` for one closed-form block, plan-style.

    ``sweep_outside=False`` measures kappa at the prior centre alone, and it
    is what the LOG blocks use -- not to save the probes, but because the
    sweep leaves the transform's domain: ``kappa_interval`` displaces each
    outside latent by up to its largest probe sigma, and three prior widths
    of a sky coefficient can put the prediction below zero, where the
    log-space graph has no value at all. Measured -- the sweep's own
    ``check_gaussian`` then reads a NaN ``log_prob`` and refuses. A kappa
    read only at the centre is a weaker statement, and the honest one: the
    interval collapses to a point rather than pretending the sweep ran."""
    at = {
        name: env[name]
        for name in graph.latents
        if name not in set(group) and name in env
    }
    if sweep_outside:
        low, high, _ = kappa_interval(graph, group, env=env, key=key)
    else:
        low = high = _kappa_at(graph, group, at, key)
    tol = CONVERGENCE_TARGET / kappa_upper((low, high))
    return (low, high), tol, working_epsilon(graph, group, at)


def factor_partition(
    graph: Graph,
    *,
    key: jax.Array | None = None,
    scales: Sequence[float] = DEFAULT_SCALES,
    log_scales: Sequence[float] = LOG_DEFAULT_SCALES,
    rtol: float | None = None,
) -> FactorPlan:
    """Derive the factor partition: exact blocks by pairwise probe, NUTS rest.

    The linear pass takes the latents that qualify exactly as
    :func:`~bayesmith.dispatch.classify.partition` counts qualification --
    Gaussian prior, ``linear_in`` declared along every path, not an ancestor
    of another latent -- verifies each ALONE, then groups by the pairwise
    joint check. The log pass takes what remains, transforms the graph once,
    and repeats the same two steps on it; nothing is declared for it, which
    is the point -- there is no ``log_linear_in`` field, deliberately, because
    the probe answers the question the declaration would merely assert.

    Args:
        graph: the model.
        key: PRNG key for every probe and kappa sweep. Fixed by default, so a
            partition is reproducible.
        scales, log_scales: probe magnitudes for the two passes, in units of
            each latent's own prior width. Separate arguments because the
            linear default's ``1e3`` entry, pushed through an exponential,
            measures the dtype rather than the model -- see
            :data:`~bayesmith.exact.loglinear.LOG_DEFAULT_SCALES`.
        rtol: forwarded to every affinity check.

    Returns:
        A :class:`FactorPlan`. Its ``str`` is the readable form, in the same
        shape the package front page sketches.
    """
    key = jax.random.key(0) if key is None else key
    env = prior_environment(graph)
    latents = list(graph.latents)
    why_not: dict[str, str] = {}

    ejected = {
        name
        for name in latents
        if any(name in _ancestors(graph, other) for other in latents if other != name)
    }
    for name in ejected:
        why_not[name] = f"{name!r} is an ancestor of another latent's distribution"

    linear_candidates = []
    for name in latents:
        if name in ejected:
            continue
        ok, why = _is_gaussian(graph, name, env)
        if not ok:
            why_not[name] = why
            continue
        ok, why = _declares_linear_in(graph, name)
        if not ok:
            why_not[name] = why
            continue
        linear_candidates.append(name)

    linearity: dict[str, dict] = {}

    def outside(source: Graph, group: tuple[str, ...]) -> dict[str, Any]:
        """`at` for a probe: every latent outside ``group``, at its prior centre."""
        return {
            name: env[name]
            for name in source.latents
            if name not in set(group) and name in env
        }

    def linear_alone(name: str) -> bool:
        try:
            linearity[name] = check_linearity(
                graph, (name,), outside(graph, (name,)),
                scales=scales, rtol=rtol, key=key,
            )
        except (StructureError, NotGaussian) as refused:
            why_not[name] = f"not affine alone: {refused}"
            return False
        return True

    def linear_pair(one: str, two: str) -> bool:
        try:
            check_linearity(
                graph, (one, two), outside(graph, (one, two)),
                scales=scales, rtol=rtol, key=key,
            )
        except (StructureError, NotGaussian):
            return False
        return True

    verified = [name for name in linear_candidates if linear_alone(name)]
    linear_groups = [tuple(g) for g in first_fit(verified, linear_pair)]

    remaining = [
        name
        for name in latents
        if name not in ejected and name not in {m for g in linear_groups for m in g}
    ]

    transformed: LogSpace | None = None
    log_groups: list[tuple[str, ...]] = []
    log_linearity: dict[str, dict] = {}
    if remaining:
        try:
            transformed = log_space(graph)
        except NotLogLinear as refused:
            for name in remaining:
                why_not.setdefault(name, f"no log-space route: {refused}")
            transformed = None
        if transformed is not None:
            source = transformed.graph
            translated = set(transformed.kind)

            def reaches_only_transformed(name: str) -> bool:
                """A log block may not condition on an untransformed likelihood.

                A skipped node the latent does not reach contributes a zero
                design column and a constant to chi-squared -- harmless. One
                it DOES reach would mix log data with a raw likelihood in a
                single conditional, so the latent is disqualified, by name.
                """
                reached = {
                    obs
                    for obs in graph.observed
                    if name in _ancestors(graph, obs)
                }
                untranslated = sorted(reached - translated)
                if untranslated:
                    why_not[name] = (
                        f"reaches observed node(s) {untranslated}, which "
                        "log_space left untransformed ("
                        + "; ".join(
                            f"{obs}: {transformed.skipped[obs]}"
                            for obs in untranslated
                        )
                        + ")"
                    )
                    return False
                return True

            def log_alone(name: str) -> bool:
                if not reaches_only_transformed(name):
                    return False
                ok, why = _is_gaussian(source, name, env)
                if not ok:
                    why_not[name] = why
                    return False
                try:
                    log_linearity[name] = check_linearity(
                        source, (name,), outside(source, (name,)),
                        scales=log_scales, rtol=rtol, key=key,
                    )
                except (StructureError, NotGaussian) as refused:
                    why_not[name] = f"log(prediction) not affine alone: {refused}"
                    return False
                return True

            def log_pair(one: str, two: str) -> bool:
                try:
                    check_linearity(
                        source, (one, two), outside(source, (one, two)),
                        scales=log_scales, rtol=rtol, key=key,
                    )
                except (StructureError, NotGaussian):
                    return False
                return True

            log_verified = [name for name in remaining if log_alone(name)]
            log_groups = [tuple(g) for g in first_fit(log_verified, log_pair)]
            if not log_groups:
                transformed = None

    blocks: list[Block] = []
    for group in linear_groups:
        operator = unchecked_operator(graph, group, at=outside(graph, group))
        movement = _movement_of(graph, operator, env, group, key)
        if movement > SIGMA_RTOL:
            for name in group:
                why_not[name] = (
                    f"sigma moves with block {group} (relative movement "
                    f"{movement:.3g} > {SIGMA_RTOL:g}); the frozen-sigma draw "
                    "is only a proposal there, and this module does not sweep "
                    "'gcr+mh' -- see its docstring. Use "
                    "bayesmith.exact.gibbs.assemble for that block alone, or "
                    "accept NUTS."
                )
            continue
        kappa, tol, epsilon = _measured(graph, group, env, key)
        blocks.append(
            Block(
                latents=tuple(sorted(group)),
                method="gcr",
                reason="factor block: jointly affine, sigma frozen exactly",
                linearity=linearity.get(group[0]),
                kappa=kappa,
                tol=tol,
                epsilon=epsilon,
            )
        )
    if transformed is not None:
        for group in log_groups:
            kappa, tol, epsilon = _measured(
                transformed.graph, group, env, key, sweep_outside=False
            )
            blocks.append(
                Block(
                    latents=tuple(sorted(group)),
                    method="log-gcr",
                    reason=(
                        "factor block in log space: "
                        + ", ".join(
                            f"{obs}: {transformed.kind[obs]}"
                            for obs in sorted(transformed.kind)
                        )
                    ),
                    linearity=log_linearity.get(group[0]),
                    kappa=kappa,
                    tol=tol,
                    epsilon=epsilon,
                )
            )

    solved = {name for block in blocks for name in block.latents}
    leftovers = tuple(name for name in latents if name not in solved)
    if leftovers:
        blocks.append(
            Block(
                latents=leftovers,
                method="nuts",
                reason="; ".join(
                    f"{name!r}: {why_not.get(name, 'no exact structure found')}"
                    for name in leftovers
                ),
            )
        )
    return FactorPlan(
        blocks=tuple(blocks),
        log_space=transformed if any(b.method == "log-gcr" for b in blocks) else None,
    )


def _movement_of(
    graph: Graph,
    operator: Any,
    env: dict[str, Any],
    group: tuple[str, ...],
    key: jax.Array,
) -> float:
    """Sigma's relative movement with this block, via the classify probe."""
    from bayesmith.dispatch.classify import _sigma_movement

    at = {
        name: env[name]
        for name in graph.latents
        if name not in set(group) and name in env
    }
    return _sigma_movement(graph, operator, at, key)


def _source_of(graph: Graph, plan: FactorPlan, block: Block) -> Graph:
    """The graph a block's solves run against: transformed for ``log-gcr``."""
    if block.method == "log-gcr":
        assert plan.log_space is not None
        return plan.log_space.graph
    return graph


def sample_factors(
    graph: Graph,
    plan: FactorPlan,
    key: jax.Array,
    *,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    maxiter: int | None = None,
    nuts_options: dict[str, Any] | None = None,
) -> dict[str, jax.Array]:
    """Draw from the posterior by sweeping the plan's blocks.

    Every closed-form block is updated by an exact conditional draw --
    :func:`~bayesmith.exact.solve.gcr_sample` on its own source graph --
    conditioning on the LATEST values of every other latent, in plan order.
    With no NUTS remainder the sweep is an exact Gibbs sampler and runs as a
    plain loop here; with one, the sweep becomes ``HMCGibbs``'s ``gibbs_fn``
    and NUTS advances the remainder between sweeps, exactly as the
    single-block :func:`~bayesmith.exact.gibbs.assemble` arranges.

    Args:
        graph: the model the plan was derived from.
        plan: from :func:`factor_partition`. Its per-block ``tol`` is used
            for every solve -- the compile-time choice standing in for the
            in-sweep guard, for the reason
            :func:`~bayesmith.exact.gibbs.gibbs_factory` gives.
        key: PRNG key for the whole run.
        num_warmup, num_samples: sweeps discarded and kept.
        maxiter: CG iteration cap per solve.
        nuts_options: forwarded to the inner NUTS kernel, when there is one.

    The inner NUTS kernel, when there is one, is initialised at the latents'
    prior centres unless ``nuts_options`` names its own ``init_strategy`` --
    numpyro's default draws the initial point from the priors in
    unconstrained space, and on a model whose prediction can go negative out
    there (a radiometer likelihood under a wide sky prior) that manufactures
    an invalid scale before the first sweep runs.

    Returns:
        ``{latent: draws}`` with ``num_samples`` rows each, closed-form and
        NUTS latents alike.

    Raises:
        GraphError: if the plan has no closed-form block at all -- run
            :func:`~bayesmith.bridge.numpyro_bridge.nuts` instead, and this
            refusal names it -- or if a block carries a method this module
            does not sweep.
    """
    exact = plan.exact
    if not exact:
        raise GraphError(
            "this plan has no closed-form block -- every latent fell to NUTS "
            "-- so there is no sweep for sample_factors to run. Call "
            "bayesmith.nuts(graph, ...) directly; wrapping it here would "
            "just relabel it."
        )
    for block in exact:
        if block.method not in ("gcr", "log-gcr"):
            raise GraphError(
                f"block {block.latents} carries method {block.method!r}, "
                f"which sample_factors does not sweep; it runs {FACTOR_METHODS[:2]}."
            )

    sources = {block.latents: _source_of(graph, plan, block) for block in exact}
    env = prior_environment(graph)
    centres = {name: env[name] for name in graph.latents if name in env}
    # Hoisted per block where the covariance cannot move with ANY latent;
    # rebuilt inside the sweep otherwise. For a log-gcr block the transformed
    # scale is constant by construction, so it always hoists.
    hoisted: dict[tuple[str, ...], Any] = {}
    rebuild: set[tuple[str, ...]] = set()
    for block in exact:
        source = sources[block.latents]
        if block.method == "log-gcr" or not _sigma_needs_rebuild(
            source, block.latents
        ):
            hoisted[block.latents] = precision_at(source, centres)
        else:
            rebuild.add(block.latents)

    def sweep(values: dict[str, Any], sweep_key: jax.Array) -> dict[str, Any]:
        current = dict(values)
        for index, block in enumerate(exact):
            source = sources[block.latents]
            at = {
                name: current[name]
                for name in graph.latents
                if name not in set(block.latents)
            }
            operator = unchecked_operator(
                source, block.latents, at=at, probe_gaussian=False
            )
            if block.latents in rebuild:
                noise = precision_at(source, current)
            else:
                noise = hoisted[block.latents]
            drawn, _ = gcr_sample(
                operator,
                precision=noise,
                key=jax.random.fold_in(sweep_key, index),
                tol=block.tol if block.tol is not None else 1e-6,
                maxiter=maxiter,
            )
            current.update(drawn)
        return current

    nuts_latents = plan.nuts
    if not nuts_latents:
        values = dict(centres)
        kept: dict[str, list] = {name: [] for name in graph.latents}
        for index in range(num_warmup + num_samples):
            values = sweep(values, jax.random.fold_in(key, index))
            if index >= num_warmup:
                for name in graph.latents:
                    kept[name].append(values[name])
        return {name: jnp.stack(rows) for name, rows in kept.items()}

    from numpyro.infer import MCMC, NUTS, HMCGibbs
    from numpyro.infer.initialization import init_to_value

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    gibbs_names = tuple(name for block in exact for name in block.latents)
    options = dict(nuts_options or {})
    options.setdefault("init_strategy", init_to_value(values=dict(centres)))

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        values = {k: v for k, v in hmc_sites.items() if k in set(graph.latents)}
        values.update({k: gibbs_sites[k] for k in gibbs_names})
        updated = sweep(values, rng_key)
        return {name: updated[name] for name in gibbs_names}

    inner = NUTS(to_numpyro(graph), **options)
    kernel = HMCGibbs(inner, gibbs_fn=gibbs_fn, gibbs_sites=list(gibbs_names))
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    mcmc.run(key)
    draws = mcmc.get_samples()
    return {name: draws[name] for name in graph.latents}


__all__ = [
    "FACTOR_METHODS",
    "FactorPlan",
    "factor_partition",
    "first_fit",
    "sample_factors",
]
