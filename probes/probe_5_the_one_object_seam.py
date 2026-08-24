"""Probe 5 -- where the one-object discipline actually lives today.

The migration spec's load-bearing clause: ONE object must feed `log_joint`,
`wiener_solve` and `fisher_information`, so the covariance and the likelihood
cannot disagree.

Today `log_joint` reads the node's own distribution while `wiener_solve` and
`fisher_information` read a `{observed: sigma}` DICT. This measures how much
of the discipline is structural and how much is call convention.

Run:
    cd <worktree> && PYTHONPATH=$PWD/src \
        /Users/zzhang/projects/bayesmith/.venv/bin/python probes/probe_5_the_one_object_seam.py
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.exact import fisher, gls, solve
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.precision import CirculantPrecision, DiagonalPrecision
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.trace import const, det, observe, sample, trace

jax.config.update("jax_enable_x64", True)

N = 64
X = np.linspace(-1.0, 1.0, N)
rng = np.random.default_rng(3)
TRUE_W = 2.0
DATA = TRUE_W * X + rng.normal(size=N) * 0.5


def model():
    x = const("x", jnp.asarray(X))
    w = sample("w", lambda: dist.Normal(0.0, 5.0))
    mu = det("mu", lambda w_, x_: w_ * x_, w, x, linear_in=("w",))
    observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.asarray(DATA))
    return None


graph = trace(model)
block = linear_operator(graph, ("w",))

print("=" * 78)
print("(a) the signature every exact consumer actually takes")
print("=" * 78)
for name, fn in (
    ("wiener_solve", solve.wiener_solve),
    ("gcr_sample", solve.gcr_sample),
    ("condition_bound", solve.condition_bound),
    ("fisher_information", fisher.fisher_information),
    ("iterative_gls", gls.iterative_gls),
):
    params = list(inspect.signature(fn).parameters)
    noise = [p for p in params if "noise" in p or "sigma" in p]
    print(f"  {name:<20} noise-related parameters: {noise}")

print()
print("  and what the type annotation says the value is:")
src = inspect.signature(solve.wiener_solve).parameters["noise_std"]
print(f"    wiener_solve.noise_std : {src.annotation}")
print(f"    noise_std_at returns   : {type(noise_std_at(graph, {'w': jnp.array(1.0)})['d'])}")
print("  -> there is no parameter anywhere that accepts a Precision.")

print()
print("=" * 78)
print("(b) is the ONE-object discipline structural, or call convention?")
print("=" * 78)
honest = noise_std_at(graph, {"w": jnp.array(0.0)})
print(f"  the graph's own sigma          : {float(honest['d'][0]):.6f}")

for factor in (1.0, 2.0, 10.0):
    lied = {"d": honest["d"] * factor}
    mean, _ = solve.wiener_solve(block, noise_std=lied, require_convergence=None)
    fish = fisher.fisher_information(
        block, noise_std=lied, depends_on_prediction=False
    )
    cov = fisher.parameter_covariance(fish)
    lj = float(log_joint(graph, {"w": mean["w"]}))
    print(
        f"  noise_std x{factor:<5.1f} -> wiener_solve w = {float(mean['w']):+.6f}, "
        f"posterior sd = {float(cov.std()['w'][0]):.6f}, "
        f"log_joint at that w = {lj:+.4f}"
    )
print("  Nothing refused. The sigma the solver was weighted by and the sigma")
print("  the node declares are compared NOWHERE -- the only thing keeping them")
print("  equal is that the caller passed noise_std_at(graph, ...).")

print()
print("  the one place a cross-check DOES exist (fisher's centre/sigma_of):")
sigma_of = gls.sigma_from_graph(graph, {})
try:
    fisher.fisher_information(
        block,
        noise_std={"d": honest["d"] * 2.0},
        sigma_of=sigma_of,
        centre={"w": jnp.array(0.0)},
    )
    print("    accepted a contradicting noise_std -- NOT refused")
except ValueError as exc:
    print(f"    REFUSED: {str(exc).strip().splitlines()[0][:100]}")
print("  -- but only when sigma_of= and centre= are supplied, which")
print("     depends_on_prediction=False lets a caller omit entirely.")

print()
print("=" * 78)
print("(c) could a Precision be passed through today's parameter?")
print("=" * 78)
for label, obj in (
    ("DiagonalPrecision", DiagonalPrecision(sigma=honest["d"])),
    ("CirculantPrecision", CirculantPrecision(first_column=jnp.asarray(
        np.exp(-np.minimum(np.arange(N), N - np.arange(N)) / 8.0) + 1e-3))),
):
    try:
        mean, _ = solve.wiener_solve(
            block, noise_std={"d": obj}, require_convergence=None
        )
        print(f"  {label:<20} accepted -> w = {float(mean['w']):+.6f}  (!!)")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<20} REFUSED -> {type(exc).__name__}: "
              f"{str(exc).strip().splitlines()[0][:80]}")
