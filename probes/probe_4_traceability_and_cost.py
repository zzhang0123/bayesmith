"""Probe 4 -- can a Precision be BUILT under trace, and what does the
correlated path cost at realistic sizes?

`gaussian_parts` is documented as the traceable fast path: "fully traceable,
and it is what runs inside `jax.linearize` on every solve". Its correlated
analogue would have to return a Precision. This measures whether that object
can be constructed inside a trace at all, and what the three routes cost.

Run:
    cd <worktree> && PYTHONPATH=$PWD/src \
        /Users/zzhang/projects/bayesmith/.venv/bin/python probes/probe_4_traceability_and_cost.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.exact.precision import (
    CirculantPrecision,
    DiagonalPrecision,
    quadratic,
)

jax.config.update("jax_enable_x64", True)


def falling_kernel(n, corr=8.0):
    lag = np.minimum(np.arange(n), n - np.arange(n))
    row = np.exp(-lag / corr) + 1e-3
    assert np.real(np.fft.fft(row)).min() > 0
    return row


print("=" * 78)
print("(a) can a Precision be CONSTRUCTED inside a trace?")
print("=" * 78)
kernel = jnp.asarray(falling_kernel(16))
sigma = jnp.ones(16)

for label, fn in (
    ("DiagonalPrecision under jit", lambda s: quadratic(DiagonalPrecision(sigma=s), jnp.ones(16))),
    ("CirculantPrecision under jit", lambda k: quadratic(CirculantPrecision(first_column=k), jnp.ones(16))),
):
    arg = sigma if "Diagonal" in label else kernel
    try:
        out = jax.jit(fn)(arg)
        print(f"  {label:<32} OK      -> {float(out):.6f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<32} REFUSED -> {type(exc).__name__}")
        print(f"      {str(exc).strip().splitlines()[0][:110]}")

print()
print("  and inside jax.linearize, which is what block.py actually opens:")
for label, fn, arg in (
    ("DiagonalPrecision", lambda s: quadratic(DiagonalPrecision(sigma=s), jnp.ones(16)), sigma),
    ("CirculantPrecision", lambda k: quadratic(CirculantPrecision(first_column=k), jnp.ones(16)), kernel),
):
    try:
        jax.linearize(fn, arg)
        print(f"    {label:<20} OK")
    except Exception as exc:  # noqa: BLE001
        print(f"    {label:<20} REFUSED -> {type(exc).__name__}")

print()
print("  is the ARITHMETIC traceable if the check is skipped?")
def circulant_quadratic_unchecked(k, r):
    eig = jnp.real(jnp.fft.fft(k))
    return jnp.sum(r * jnp.real(jnp.fft.ifft(jnp.fft.fft(r) / eig)))

out = jax.jit(circulant_quadratic_unchecked)(kernel, jnp.ones(16))
grad = jax.grad(circulant_quadratic_unchecked, argnums=1)(kernel, jnp.ones(16))
print(f"    same quadratic without __check_init__: OK -> {float(out):.6f}, "
      f"grad finite: {bool(jnp.all(jnp.isfinite(grad)))}")

print()
print("=" * 78)
print("(b) cost: FFT route vs dense MultivariateNormal, log_prob and gradient")
print("=" * 78)


def timed(fn, *args, repeats=3):
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best * 1e3


print(f"{'n':>8} {'circ log_prob':>15} {'circ grad':>12} {'dense build':>13} "
      f"{'dense log_prob':>15} {'dense RAM':>11}")
for n in (256, 1024, 4096, 16384, 65536, 262144, 1048576):
    row = falling_kernel(n)
    loc = jnp.zeros(n)
    x = jnp.asarray(np.random.default_rng(0).normal(size=n))
    cn = dist.CirculantNormal(loc, covariance_row=jnp.asarray(row))
    f_lp = jax.jit(lambda v, d=cn: d.log_prob(v))
    f_gr = jax.jit(jax.grad(lambda v, d=cn: d.log_prob(v)))
    t_lp, t_gr = timed(f_lp, x), timed(f_gr, x)

    ram_gb = n * n * 8 / 1024**3
    if n <= 4096:
        cov = np.array([np.roll(row, i) for i in range(n)])
        t0 = time.perf_counter()
        mvn = dist.MultivariateNormal(loc, covariance_matrix=jnp.asarray(cov))
        jax.block_until_ready(mvn.log_prob(x))
        t_build = (time.perf_counter() - t0) * 1e3
        t_dlp = timed(jax.jit(lambda v, d=mvn: d.log_prob(v)), x)
        agree = abs(float(mvn.log_prob(x)) - float(cn.log_prob(x)))
        build_s, dlp_s = f"{t_build:11.1f}ms", f"{t_dlp:13.3f}ms"
        extra = f"  (|MVN-circ| = {agree:.2e})"
    else:
        build_s, dlp_s, extra = "  not built", "  not built", ""
    print(f"{n:>8} {t_lp:13.3f}ms {t_gr:10.3f}ms {build_s:>13} {dlp_s:>15} "
          f"{ram_gb:9.3f}GB{extra}")

print()
print("  dense RAM is one n x n float64 covariance; MultivariateNormal also")
print("  factorises it (Cholesky, O(n^3)) at construction.")

print()
print("=" * 78)
print("(c) the dense route's real wall: forming and factorising N")
print("=" * 78)
print("  (the 'dense build' column above is dominated by one-time JIT compilation,")
print("   so it does NOT show the O(n^3) scaling. numpy, no compile step, does:)")
print(f"{'n':>8} {'form N':>12} {'cholesky':>12} {'RAM':>11}")
for n in (256, 512, 1024, 2048, 4096):
    row = falling_kernel(n)
    t0 = time.perf_counter()
    cov = np.array([np.roll(row, i) for i in range(n)])
    t_form = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    np.linalg.cholesky(cov)
    t_chol = (time.perf_counter() - t0) * 1e3
    print(f"{n:>8} {t_form:10.2f}ms {t_chol:10.2f}ms {n*n*8/1024**3:9.4f}GB")

print()
print("=" * 78)
print("(d) does the noise really get built under a trace on the solve path?")
print("=" * 78)
import numpyro.distributions as _dist  # noqa: E402

from bayesmith.exact.block import isolate  # noqa: E402
from bayesmith.graph.trace import const, det, observe, sample, trace  # noqa: E402

seen = []


def model():
    x = const("x", jnp.linspace(-1.0, 1.0, 8))
    w = sample("w", lambda: _dist.Normal(0.0, 3.0))
    mu = det("mu", lambda w_, x_: w_ * x_, w, x, linear_in=("w",))

    def noise(m):
        seen.append(type(m).__name__)
        return _dist.Normal(m, 1.0)

    observe("d", noise, mu, obs=jnp.zeros(8))
    return None


graph = trace(model)
seen.clear()
g = isolate(graph, ("w",), {})
jax.linearize(g, {"w": jnp.zeros(())})
print(f"  types the observed node's dist_fn saw inside jax.linearize: {sorted(set(seen))}")
print("  -> `observation_parts` (and so any `precision_at` built the same way)")
print("     runs on TRACERS. A Precision whose constructor concretises cannot")
print("     be built there; see (a).")
