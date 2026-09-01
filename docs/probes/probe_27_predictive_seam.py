"""probe_27 -- the predictive seam, as numbers instead of a description.

Freezes the §0.10 measurements from
docs/superpowers/plans/2026-08-31-r2-predictive-seam.md into a re-runnable
oracle.  Run from the repository root with the test package importable:

    PYTHONPATH=. .venv/bin/python docs/probes/probe_27_predictive_seam.py

What it pins, and why each number is the number it is:

1. observation_parts is the seam.  (data, loc, scale) come out of one
   call, broadcast to the observed node's shape, so replay and replicate share
   the SAME loc/scale -- §0.1's ruling that the two differ only by log_prob vs
   sample.
2. replay is the conditioning half.  Normal(loc, scale).log_prob(data),
   summed over the node's elements and averaged over the posterior draws, is the
   pointwise log-likelihood -- NOT a posterior predictive.  For radiometer at
   key 1 that sum-of-observations mean is **-3.8373** (2000 gcr+snis draws).
3. replicate is the new-random half.  Normal(loc, scale).sample(key)
   gives a new draw whose elements are elementwise different from the observed
   data, while its grand mean lands near the observed grand mean.

The assertion is the oracle, not a tolerance: -3.8373 was measured in this
checkout and is reproduced by key 1, so the probe fails if the seam ever moves.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import compile as compile_graph
from bayesmith.exact.gaussian import observation_parts
from bayesmith.graph.evaluate import evaluate
from tests.exact.models import radiometer, straight_line

#: §0.10's oracle for radiometer: the mean over 2000 posterior draws of the
#: summed (10-observation) pointwise log-likelihood.  Key 1 reproduces it.
RADIOMETER_REPLAY_MEAN = -3.8373

DRAWS = 2000
REPLAY_TOL = 1e-3


def _replay_and_replicate(graph, draws, key):
    """(replay_mean, replicated, observed) for one posterior sample."""
    observed = {name: jnp.asarray(graph.node(name).observed) for name in graph.observed}

    def per_draw(draw_values, draw_key):
        env = evaluate(graph, draw_values)
        data, loc, scale = observation_parts(graph, env)
        return {
            name: (
                jnp.sum(dist.Normal(loc[name], scale[name]).log_prob(data[name])),
                dist.Normal(loc[name], scale[name]).sample(draw_key),
            )
            for name in graph.observed
        }

    keys = jax.random.split(key, DRAWS)
    out = jax.vmap(per_draw)(draws, keys)

    replay_total = 0.0
    replicated = {name: out[name][1] for name in graph.observed}
    for name in graph.observed:
        replay_total = replay_total + float(jnp.mean(out[name][0]))
    return replay_total, replicated, observed


def _check(graph, label, key, replay_anchor=None):
    plan = compile_graph(graph)
    posterior = plan.sample(
        key, num_samples=DRAWS, num_warmup=1000, num_chains=1, nuts_on_collapse=False
    )
    latent_draws = dict(posterior.samples)
    replay_mean, replicated, observed = _replay_and_replicate(
        graph, latent_draws, jax.random.fold_in(key, 1)
    )

    print(f"--- {label} ({posterior.method}) ---")
    mean_w = {name: float(np.mean(np.asarray(v))) for name, v in latent_draws.items()}
    print("posterior mean:", {k: round(v, 6) for k, v in mean_w.items()})

    env = evaluate(graph, {name: jnp.asarray(mean_w[name]) for name in latent_draws})
    _data, loc, scale = observation_parts(graph, env)
    for name in graph.observed:
        print(
            f"observation_parts[{name}]: loc shape {np.shape(loc[name])}, "
            f"scale mean {float(np.mean(np.asarray(scale[name]))):.6f}"
        )
    print(f"replay mean log-lik (sum over obs, mean over draws): {replay_mean:.6f}")
    for name in graph.observed:
        print(
            f"replicated grand mean {name}: {float(np.mean(np.asarray(replicated[name]))):.6f} "
            f"(observed grand mean {float(np.mean(observed[name])):.6f})"
        )
    for name in graph.observed:
        assert not bool(
            np.any(np.asarray(replicated[name]) == np.asarray(observed[name]))
        ), f"{label}: a replicated element equals the observed datum it replaced"
    if replay_anchor is not None:
        assert math.isclose(
            replay_mean, replay_anchor, rel_tol=REPLAY_TOL, abs_tol=1e-2
        ), f"{label}: replay mean {replay_mean} drifted from the oracle {replay_anchor}"
        print(f"replay mean matches the §0.10 oracle {replay_anchor}")
    return replay_mean


def main() -> None:
    _check(radiometer(), "radiometer", jax.random.key(1), replay_anchor=RADIOMETER_REPLAY_MEAN)
    _check(straight_line(), "straight_line", jax.random.key(1))


if __name__ == "__main__":
    main()

