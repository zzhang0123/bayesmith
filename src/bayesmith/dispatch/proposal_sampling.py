"""NumPyro chains for ordered proposal sweeps, with an optional NUTS remainder.

The small HMCGibbs adapter isolates NumPyro's state contract here. Proposal
counters are state fields, so diagnostics and both RNG streams survive every
checkpoint along with the inner potential and adaptation state.
"""

from __future__ import annotations

import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer import MCMC, NUTS, HMCGibbs
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import unconstrain_fn
from numpyro.util import is_prng_key

from bayesmith.artifacts import InitializationPolicy, NamedArray
from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.dispatch.execute import Posterior, _diagnostics_or_none, chain_ess
from bayesmith.dispatch.initialization import complete_initial_values
from bayesmith.dispatch.sampling import (
    SamplingControl,
    checkpoint_diagnostics,
    prepare_initial_params,
)
from bayesmith.exact.proposals import ProposalDiagnostics


def _empty_diagnostics(compiled):
    zeros = jnp.zeros(len(compiled.blocks), dtype=jnp.int32)
    return ProposalDiagnostics(zeros, zeros, zeros, zeros)


class ProposalState(NamedTuple):
    z: dict[str, Any]
    rng_key: jax.Array
    proposal_diagnostics: ProposalDiagnostics
    diverging: jax.Array


class ProposalGibbsState(NamedTuple):
    z: dict[str, Any]
    hmc_state: Any
    rng_key: jax.Array
    proposal_diagnostics: ProposalDiagnostics


class ProposalKernel(MCMCKernel):
    """A whole-model proposal chain; correction follows each block's policy."""

    sample_field = "z"
    default_fields = ("z", "proposal_diagnostics", "diverging")

    def __init__(self, compiled):
        self.compiled = compiled
        self._vectorized = False
        self._sample_fn = None

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        if init_params is None:
            raise ValueError("whole-proposal chains require validated initial values")
        self._vectorized = not is_prng_key(rng_key)
        self._sample_fn = self.sample
        if self._vectorized:
            return jax.vmap(self._init_single)(rng_key, init_params)
        return self._init_single(rng_key, init_params)

    def _init_single(self, key, params):
        return ProposalState(
            dict(params), key, _empty_diagnostics(self.compiled), jnp.asarray(False)
        )

    def _sample_single(self, state):
        key, draw_key = jax.random.split(state.rng_key)
        values, diagnostics = self.compiled.sweep(draw_key, state.z)
        return ProposalState(values, key, diagnostics, jnp.asarray(False))

    def sample(self, state, model_args, model_kwargs):
        return (
            jax.vmap(self._sample_single)(state)
            if self._vectorized
            else self._sample_single(state)
        )


class ProposalGibbs(HMCGibbs):
    """HMCGibbs with actual MH counters and support-aware coordinate refresh."""

    default_fields = ("z", "proposal_diagnostics", "hmc_state.diverging")

    def __init__(self, inner_kernel, compiled):
        super().__init__(
            inner_kernel, gibbs_fn=lambda **_: {}, gibbs_sites=list(compiled.names)
        )
        self.compiled = compiled
        self._vectorized = False
        self._sample_fn = None

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        self._vectorized = not is_prng_key(rng_key)
        self._sample_fn = self.sample

        def initialize(key, params):
            # HMCGibbs.init pops its sites, so it gets its own shallow mapping.
            state = super(ProposalGibbs, self).init(
                key,
                num_warmup,
                None if params is None else dict(params),
                model_args,
                model_kwargs,
            )
            return ProposalGibbsState(
                state.z,
                state.hmc_state,
                state.rng_key,
                _empty_diagnostics(self.compiled),
            )

        if self._vectorized:
            return jax.vmap(initialize)(rng_key, init_params)
        return initialize(rng_key, init_params)

    def _sample_single(self, state, model_args, model_kwargs):
        kwargs = dict(model_kwargs or {})
        key, draw_key = jax.random.split(state.rng_key)
        gibbs = {name: state.z[name] for name in self.compiled.names}
        old_kwargs = {**kwargs, "_gibbs_sites": gibbs}
        constrained = self.inner_kernel.postprocess_fn(model_args, old_kwargs)(
            state.hmc_state.z
        )
        full_state = {
            name: (gibbs[name] if name in gibbs else constrained[name])
            for name in (*self.compiled.names, *self.compiled.remainder)
        }
        updated, diagnostics = self.compiled.sweep(draw_key, full_state)
        gibbs = {name: updated[name] for name in self.compiled.names}
        new_kwargs = {**kwargs, "_gibbs_sites": gibbs}
        # The support of an HMC site can depend on a Gibbs parent. Re-encode
        # the SAME constrained complement at the new support before HMC moves.
        hmc_values = {name: constrained[name] for name in state.hmc_state.z}
        encoded = unconstrain_fn(
            self.inner_kernel.model, model_args, new_kwargs, hmc_values
        )
        changed = jnp.any(diagnostics.accepted > 0)
        encoded = jax.tree.map(
            lambda new, old: jnp.where(changed, new, old), encoded, state.hmc_state.z
        )
        potential = self.inner_kernel._potential_fn_gen(*model_args, **new_kwargs)
        if self.inner_kernel._forward_mode_differentiation:
            value, gradient = potential(encoded), jax.jacfwd(potential)(encoded)
        else:
            value, gradient = jax.value_and_grad(potential)(encoded)
        hmc_state = state.hmc_state._replace(
            z=encoded, potential_energy=value, z_grad=gradient
        )
        hmc_state = self.inner_kernel.sample(hmc_state, model_args, new_kwargs)
        return ProposalGibbsState({**gibbs, **hmc_state.z}, hmc_state, key, diagnostics)

    def sample(self, state, model_args, model_kwargs):
        step = lambda s: self._sample_single(s, model_args, model_kwargs)
        return jax.vmap(step)(state) if self._vectorized else step(state)


def _whole_initial(mcmc, graph, key, control):
    points, attempts = complete_initial_values(
        graph,
        jax.random.fold_in(key, 8723),
        control.initialization or InitializationPolicy(),
        mcmc.num_chains,
    )
    control.initial_values = tuple(
        NamedArray(
            name,
            np.stack([np.asarray(point[name]) for point in points]),
            ("chain",) + tuple(f"axis_{i}" for i in range(np.ndim(points[0][name]))),
        )
        for name in graph.latents
    )
    control.details.update(
        initialization="validated_prior_or_support_candidate",
        initialization_attempts=attempts,
    )
    return (
        points[0]
        if mcmc.num_chains == 1
        else jax.tree.map(lambda *v: jnp.stack(v), *points)
    )


def _collect(mcmc, graph, compiled, key, control):
    """Retain the full warmed state at each existing stopping-policy boundary."""
    started = time.perf_counter()
    policy, cap = control.stopping, mcmc.num_samples
    if control.initialization is None:
        control.initialization = InitializationPolicy()
    initial = (
        prepare_initial_params(mcmc, graph, key, control, compiled.names)
        if compiled.remainder
        else _whole_initial(mcmc, graph, key, control)
    )
    for chain in range(mcmc.num_chains):
        point = {
            item.name: jnp.asarray(item.value[chain]) for item in control.initial_values
        }
        for block in compiled.blocks:
            if not bool(block.build(point).is_valid()):
                raise ValueError(
                    f"chain {chain} has a nonfinite initial proposal for {block.names}; supply different initial values or increase damping"
                )
    count, consecutive, divergences = 0, 0, 0
    batches = []
    totals = np.zeros((4, len(compiled.blocks)), dtype=np.int64)
    field_name = "hmc_state.diverging" if compiled.remainder else "diverging"
    reason = "fixed_budget"
    while count < cap:
        remaining = cap - count
        size = (
            remaining
            if policy.mode == "fixed" and control.max_seconds is None
            else min(
                remaining,
                policy.min_draws
                if count == 0 and policy.mode == "checkpoints"
                else policy.batch_size,
            )
        )
        mcmc.num_samples = size
        mcmc._set_collection_params()
        if count:
            state = mcmc.last_state
            mcmc.post_warmup_state = state
            mcmc.run(state.rng_key)
        else:
            mcmc.run(key, init_params=initial)
        raw = mcmc.get_samples(group_by_chain=True)
        batches.append({name: raw[name] for name in graph.latents})
        grouped = {
            name: jnp.concatenate([batch[name] for batch in batches], axis=1)
            for name in graph.latents
        }
        extra = mcmc.get_extra_fields(group_by_chain=True)
        divergences += int(np.sum(np.asarray(extra[field_name])))
        for index, field in enumerate(ProposalDiagnostics._fields):
            totals[index] += np.sum(
                np.asarray(getattr(extra["proposal_diagnostics"], field)), axis=(0, 1)
            )
        count += size
        passed, metrics = checkpoint_diagnostics(grouped, divergences, policy)
        control.checkpoints.append(
            (("draws_per_chain", count), ("passed", passed), *metrics)
        )
        control.details["diagnostics_passed"] = passed
        if policy.mode == "fixed" and control.max_seconds is None:
            break
        consecutive = consecutive + 1 if passed and count >= policy.min_draws else 0
        if policy.mode == "checkpoints" and consecutive >= policy.consecutive:
            reason = "diagnostics_converged"
            break
        reason = "draw_cap" if policy.mode == "checkpoints" else "fixed_budget"
        if (
            control.max_seconds is not None
            and time.perf_counter() - started >= control.max_seconds
        ):
            reason = "time_cap"
            break
    records = []
    for index, block in enumerate(compiled.blocks):
        corrected = block.policy.mh_correction
        attempted, accepted = int(totals[0, index]), int(totals[1, index])
        rate = accepted / attempted
        record = {
            "names": block.names,
            "method": block.policy.method,
            **{
                name: int(totals[i, index])
                for i, name in enumerate(ProposalDiagnostics._fields)
            },
            "mh_correction": corrected,
            "correction": "original_target_mh" if corrected else "none_approximate",
            "applied": accepted,
            "application_rate": rate,
            "mh_attempted": attempted if corrected else 0,
            "mh_accepted": accepted if corrected else 0,
            "mh_acceptance_rate": rate if corrected else None,
            "acceptance_rate": rate if corrected else None,
        }
        records.append(tuple(record.items()))
    control.details.update(
        stop_reason=reason,
        draws_per_chain=count,
        chains=mcmc.num_chains,
        warmup=mcmc.num_warmup,
        consecutive_passes=consecutive,
        divergences=divergences,
        proposal_diagnostics=tuple(records),
        proposal_order=tuple(block.names for block in compiled.blocks),
        proposal_coordinates="model",
        proposal_applicability="finite_primal_probes_not_global_certificate",
        nuts_remainder=compiled.remainder,
        target_fidelity="exact" if compiled.mh_corrected else "approximate",
        uncorrected_blocks=tuple(
            block.names for block in compiled.blocks if not block.policy.mh_correction
        ),
    )
    return {
        name: value.reshape((-1,) + value.shape[2:]) for name, value in grouped.items()
    }


def run_proposal_mcmc(
    graph,
    compiled,
    key,
    *,
    num_warmup,
    num_samples,
    num_chains,
    chain_method,
    nuts_options,
    progress_bar,
    control=None,
):
    """Execute explicit proposal blocks and return a real chain Posterior."""
    if num_samples < 1 or num_chains < 1 or num_warmup < 0:
        raise ValueError(
            "proposal sampling requires positive draws/chains and nonnegative warmup"
        )
    if chain_method not in ("sequential", "parallel", "vectorized"):
        raise ValueError("unknown proposal chain_method")
    if not compiled.remainder and nuts_options:
        raise ValueError("nuts_options require a NUTS remainder")
    kernel = (
        ProposalGibbs(NUTS(to_numpyro(graph), **dict(nuts_options or {})), compiled)
        if compiled.remainder
        else ProposalKernel(compiled)
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    control = control or SamplingControl()
    samples = _collect(mcmc, graph, compiled, key, control)
    methods = ", ".join(block.policy.method for block in compiled.blocks)
    if compiled.mh_corrected:
        method = "proposal_mh"
        reason = (
            f"ordered proposal/MH chain ({methods}) against the original full target"
        )
    else:
        method = "proposal_approximate"
        uncorrected = [
            block.names for block in compiled.blocks if not block.policy.mh_correction
        ]
        reason = (
            f"approximate ordered proposal chain ({methods}); MH disabled for {uncorrected}. "
            "Finite, supported proposals are applied without correction; "
            "the unadjusted updates need not have a known joint target."
        )
    if compiled.remainder:
        reason += f"; NUTS remainder {list(compiled.remainder)}"
    return Posterior(
        samples,
        None,
        chain_ess(samples, num_chains=num_chains),
        None,
        False,
        method,
        reason,
        _diagnostics_or_none(samples, num_chains),
    )
