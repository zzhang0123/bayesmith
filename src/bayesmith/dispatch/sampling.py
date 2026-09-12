"""One MCMC state across fixed-budget or diagnostic-checkpoint collection."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import init_to_value
from numpyro.infer.util import initialize_model

from bayesmith.artifacts import InitializationPolicy, NamedArray, StoppingPolicy
from bayesmith.dispatch.initialization import complete_initial_values


@dataclass
class SamplingControl:
    """Runtime attachment: inputs are artifact policies; outputs are run facts."""

    initialization: InitializationPolicy | None = None
    stopping: StoppingPolicy = field(default_factory=StoppingPolicy)
    max_seconds: float | None = None
    initial_values: tuple[NamedArray, ...] = ()
    details: dict = field(default_factory=dict)
    checkpoints: list = field(default_factory=list)

    def record(self):
        return tuple(
            sorted(
                {
                    **self.details,
                    "checkpoints": tuple(self.checkpoints),
                    "stopping_policy": tuple(
                        (name, getattr(self.stopping, name))
                        for name in self.stopping.__dataclass_fields__
                    ),
                    "max_seconds": self.max_seconds,
                }.items()
            )
        )


def prepare_initial_params(mcmc, graph, key, control, gibbs_names=()):
    """Encode validated model coordinates for NumPyro's mixed state contract."""
    if control.initialization is None:
        control.details["initialization"] = "numpyro_default"
        return None
    from bayesmith.bridge.numpyro_bridge import to_numpyro

    points, attempts = complete_initial_values(
        graph, jax.random.fold_in(key, 8723), control.initialization, mcmc.num_chains
    )
    model = to_numpyro(graph)
    encoded, params = [], []
    for chain, point in enumerate(points):
        # The bridge represents a complex latent as two real sampling sites.
        values = {}
        for name, value in point.items():
            if jnp.iscomplexobj(value):
                values[f"{name}__re"], values[f"{name}__im"] = (
                    jnp.real(value),
                    jnp.imag(value),
                )
            else:
                values[name] = value
        encoded.append(values)
        info = initialize_model(
            jax.random.fold_in(key, chain),
            model,
            init_strategy=init_to_value(values=values),
        )
        state = dict(info.param_info.z)
        # HMCGibbs pops its sites before passing the rest to HMC.init.
        for name in gibbs_names:
            if name in point:
                state[name] = point[name]
            elif name.endswith(("__re", "__im")):
                base = name[:-4]
                value = point[base]
                state[name] = jnp.real(value) if name.endswith("__re") else jnp.imag(value)
            else:
                raise KeyError(f"Gibbs site {name!r} is absent from initial values")
        params.append(state)
    # NumPyro initializes/traces its model even when init_params is present.
    # Its potential setup must use a validated point too (not init_to_uniform).
    kernel = mcmc.sampler.inner_kernel if gibbs_names else mcmc.sampler
    kernel._init_strategy = init_to_value(values=encoded[0])
    if gibbs_names:
        # NumPyro 0.21 traces a random prototype even with explicit init_params.
        # A prior sample can be invalid (or an improper prior cannot sample).
        # Install a prototype traced at the already validated first-chain point.
        mcmc.sampler._prototype_trace = handlers.trace(
            handlers.substitute(handlers.seed(model, key), data=encoded[0])
        ).get_trace()
    control.initial_values = tuple(
        NamedArray(
            name,
            np.stack([np.asarray(p[name]) for p in points]),
            ("chain",) + tuple(f"axis_{i}" for i in range(np.ndim(points[0][name]))),
        )
        for name in graph.latents
    )
    control.details.update(
        initialization="validated_prior_or_support_candidate",
        initialization_attempts=attempts,
    )
    return (
        params[0]
        if mcmc.num_chains == 1
        else jax.tree.map(lambda *v: jnp.stack(v), *params)
    )


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def checkpoint_diagnostics(grouped, divergences, policy):
    """Every coordinate is monitored; missing/nonfinite evidence cannot pass."""
    from bayesmith.dispatch.execute import (
        _real_diagnostic_values,
        chain_diagnostics,
    )

    chains, draws = next(iter(grouped.values())).shape[:2]
    if draws < 4:
        return False, (("diagnostics", "insufficient_draws"),)
    flat = {n: v.reshape((-1,) + v.shape[2:]) for n, v in grouped.items()}
    owned = chain_diagnostics(flat, num_chains=chains)
    passed = all(d.converged for d in owned.values())
    min_ess, max_rhat, max_mcse = math.inf, -math.inf, -math.inf
    for value in grouped.values():
        value = _real_diagnostic_values(value)
        ess = np.asarray(effective_sample_size(value))
        rhat = np.asarray(split_gelman_rubin(value))
        variance = np.var(
            np.asarray(value).reshape((-1,) + value.shape[2:]), axis=0, ddof=1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mcse = np.sqrt(variance / ess)
        if not (
            np.all(np.isfinite(ess))
            and np.all(ess > 0)
            and np.all(np.isfinite(rhat))
            and np.all(np.isfinite(mcse))
        ):
            passed = False
        min_ess = min(min_ess, float(np.min(ess)))
        max_rhat = max(max_rhat, float(np.max(rhat)))
        max_mcse = max(max_mcse, float(np.max(mcse)))
        if policy.ess_min is not None:
            passed &= bool(np.all(ess >= policy.ess_min))
        if policy.rhat_max is not None:
            passed &= bool(np.all(rhat <= policy.rhat_max))
        if policy.mcse_mean is not None:
            passed &= bool(np.all(mcse <= policy.mcse_mean))
    passed &= divergences is not None and divergences <= policy.max_divergences
    return bool(passed), (
        ("ess_min", _finite(min_ess)),
        ("rhat_max", _finite(max_rhat)),
        ("mcse_mean_max", _finite(max_mcse)),
        ("divergences", divergences),
    )


def run_mcmc(mcmc, key, graph, control, gibbs_names=()):
    """Collect complete batches and resume the full warmed state, once per batch.

    The draw cap is per chain. Time limits are checked at batch boundaries,
    including the first warmup/production batch; they cannot interrupt XLA.
    """
    started = time.perf_counter()
    policy, cap = control.stopping, mcmc.num_samples
    if cap < 1:
        raise ValueError("sampling requires a positive draw cap")
    initial = prepare_initial_params(mcmc, graph, key, control, gibbs_names)
    count, consecutive, divergence_count = 0, 0, 0
    batches = []
    field_name = "hmc_state.diverging" if gibbs_names else "diverging"
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
            # Both the outer Gibbs RNG and the inner HMC RNG survive this.
            mcmc.run(state.rng_key, extra_fields=(field_name,))
        else:
            mcmc.run(key, init_params=initial, extra_fields=(field_name,))
        raw = mcmc.get_samples(group_by_chain=True)
        batches.append({name: raw[name] for name in graph.latents})
        grouped = {
            name: jnp.concatenate([b[name] for b in batches], axis=1)
            for name in graph.latents
        }
        extra = mcmc.get_extra_fields(group_by_chain=True).get(field_name)
        divergence_count = (
            None
            if extra is None or divergence_count is None
            else divergence_count + int(np.sum(np.asarray(extra)))
        )
        count += size
        passed, metrics = checkpoint_diagnostics(grouped, divergence_count, policy)
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
    control.details.update(
        stop_reason=reason,
        draws_per_chain=count,
        chains=mcmc.num_chains,
        warmup=mcmc.num_warmup,
        consecutive_passes=consecutive,
        divergences=divergence_count,
    )
    return {
        name: value.reshape((-1,) + value.shape[2:]) for name, value in grouped.items()
    }
