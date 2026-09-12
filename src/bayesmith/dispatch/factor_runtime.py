"""Public execution adapter for multiple certified Gaussian conditional blocks."""

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, HMCGibbs
from numpyro.infer.initialization import init_to_value
from numpyro.infer.mcmc import MCMCKernel
from numpyro.util import is_prng_key

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.dispatch.execute import Posterior, _diagnostics_or_none, chain_ess
from bayesmith.dispatch.factor import FactorPlan, factor_sweep
from bayesmith.dispatch.sampling import SamplingControl, run_mcmc
from bayesmith.graph.reduction import check_evidence_nuts_boundary


class _State(NamedTuple):
    z: dict
    rng_key: jax.Array
    diverging: jax.Array


class _FactorKernel(MCMCKernel):
    """Pure Gaussian Gibbs sweeps in model coordinates, without HMC adaptation."""

    sample_field = "z"
    default_fields = ("z", "diverging")

    def __init__(self, sweep, centres):
        self.sweep = sweep
        self.centres = centres
        self._vectorized = False

    def _initial(self, key, params):
        values = dict(self.centres) if params is None else {
            name: (params[name] if name in params else
                   params[name + "__re"] + 1j * params[name + "__im"])
            for name in self.centres
        }
        return _State(values, key, jnp.asarray(False))

    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        self._vectorized = not is_prng_key(rng_key)
        self._sample_fn = self.sample
        if self._vectorized:
            return jax.vmap(self._initial, in_axes=(0, None if init_params is None else 0))(
                rng_key, init_params)
        return self._initial(rng_key, init_params)

    def _step(self, state):
        key, draw_key = jax.random.split(state.rng_key)
        values, _ = self.sweep(state.z, draw_key)
        return _State(values, key, jnp.asarray(False))

    def sample(self, state, model_args, model_kwargs):
        return jax.vmap(self._step)(state) if self._vectorized else self._step(state)


def _encoded_site_values(values):
    """Encode graph coordinates using the names emitted by ``to_numpyro``."""
    encoded = {}
    for name, value in values.items():
        if jnp.iscomplexobj(value):
            encoded[f"{name}__re"] = jnp.real(value)
            encoded[f"{name}__im"] = jnp.imag(value)
        else:
            encoded[name] = value
    return encoded


def _gibbs_site_names(names, centres):
    """Map exact graph latents to NumPyro's real sampling-site names."""
    return tuple(
        site
        for name in names
        for site in (
            (f"{name}__re", f"{name}__im")
            if jnp.iscomplexobj(centres[name])
            else (name,)
        )
    )


def _decode_gibbs_sites(names, centres, sites):
    """Reassemble NumPyro Gibbs sites into graph-coordinate values."""
    values = {}
    for name in names:
        if jnp.iscomplexobj(centres[name]):
            values[name] = sites[f"{name}__re"] + 1j * sites[f"{name}__im"]
        else:
            values[name] = sites[name]
    return values


@dataclass(frozen=True)
class FactorRuntimePlan:
    """A conditional chain, deliberately not a whole-graph Gaussian solve."""

    graph: object
    factor_plan: FactorPlan
    collapse_plan: object
    sigma_needs_rebuild: bool = True
    streaming: object = None
    ladder: object = None
    method: str = "factor_gibbs"

    @property
    def blocks(self):
        return self.factor_plan.blocks

    @property
    def exact(self):
        # There is no single block that can be marginalized or whose mean is
        # the joint posterior mean. Consumers must not select those routes.
        return None

    @property
    def sampled(self):
        return next((b for b in self.blocks if b.method == "nuts"), None)

    @property
    def guard_hoisted(self):
        return True

    def _execution(self):
        return "ordered Gibbs conditional sweep" + (" with NUTS remainder" if self.sampled else "")

    def __str__(self):
        from bayesmith.dispatch.plan import _evidence

        return "\n".join([
            *(f"block {i} {{{', '.join(b.latents)}}} {b.method}\n  {_evidence(b)}\n  {b.reason}"
              for i, b in enumerate(self.blocks)),
            "execution: " + self._execution(),
        ])

    def estimate(self, **kwargs):
        raise NotImplementedError(
            "multiple conditional Gaussian blocks do not supply a joint posterior mean; "
            "sample the Gibbs chain and estimate the requested quantity from its draws")

    def sample(self, key, *, num_samples=2000, num_warmup=1000, num_chains=1,
               chain_method="sequential", progress_bar=False, nuts_options=None,
               tol=None, maxiter=None, require_convergence=None, ess_floor=None,
               nuts_on_collapse=False, collapse=False, _control=None):
        if collapse:
            settings = {
                "num_samples": num_samples,
                "num_warmup": num_warmup,
                "num_chains": num_chains,
                "chain_method": chain_method,
                "progress_bar": progress_bar,
                "nuts_options": nuts_options,
                "tol": tol,
                "maxiter": maxiter,
                "require_convergence": require_convergence,
                "nuts_on_collapse": nuts_on_collapse,
                "collapse": True,
                "_control": _control,
            }
            if ess_floor is not None:
                settings["ess_floor"] = ess_floor
            return self.collapse_plan.sample(key, **settings)
        if require_convergence is not None:
            raise ValueError("in-sweep convergence guards cannot run under JIT; use the compiled tolerance")
        if ess_floor is not None:
            raise ValueError("SNIS ess_floor does not apply to a Gibbs chain")
        if num_samples < 1 or num_chains < 1 or num_warmup < 0:
            raise ValueError("sampling requires positive draws/chains and nonnegative warmup")
        if chain_method not in ("sequential", "parallel", "vectorized"):
            raise ValueError("unknown chain_method")
        if self.factor_plan.nuts and chain_method == "vectorized":
            raise NotImplementedError(
                "chain_method='vectorized' is not supported for an "
                "HMC-within-Gibbs sweep; use 'sequential' or 'parallel'.")
        if not self.factor_plan.nuts and nuts_options:
            raise ValueError("nuts_options require a NUTS remainder")
        check_evidence_nuts_boundary(self.graph, self.factor_plan.nuts)
        factors = self.factor_plan
        if tol is not None:
            factors = FactorPlan(tuple(replace(b, tol=tol) for b in self.blocks), factors.log_space)
        sweep, centres = factor_sweep(self.graph, factors, maxiter=maxiter)
        names = tuple(n for b in factors.exact for n in b.latents)
        if factors.nuts:
            gibbs_names = _gibbs_site_names(names, centres)

            def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
                values = {n: hmc_sites[n] for n in factors.nuts}
                values.update(_decode_gibbs_sites(names, centres, gibbs_sites))
                updated, _ = sweep(values, rng_key)
                return _encoded_site_values({n: updated[n] for n in names})

            options = dict(nuts_options or {})
            options.setdefault("init_strategy", init_to_value(values=_encoded_site_values(centres)))
            kernel = HMCGibbs(NUTS(to_numpyro(self.graph), **options),
                              gibbs_fn=gibbs_fn, gibbs_sites=list(gibbs_names))
        else:
            kernel = _FactorKernel(sweep, centres)
        mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup,
                    num_chains=num_chains, chain_method=chain_method, progress_bar=progress_bar)
        control = _control or SamplingControl()
        draws = run_mcmc(
            mcmc, key, self.graph, control, gibbs_names if factors.nuts else ()
        )
        return Posterior(draws, None, chain_ess(draws, num_chains=num_chains), None,
                         False, self.method, self._execution(),
                         _diagnostics_or_none(draws, num_chains))
