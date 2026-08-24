"""Probe 2 -- can a Precision be RECOVERED from a numpyro distribution, and
would `check_gaussian`'s probe verify it?

Three questions, measured:

  (a) Does `dist.CirculantNormal(loc, covariance_row=k).log_prob` equal
      `precision.log_density(CirculantPrecision(k), r)`?  If so, extraction is
      one attribute read -- exactly parallel to `Normal.loc/.scale`.
  (b) If `gaussian_parts`'s isinstance gate were relaxed, what does
      `check_gaussian`'s ELEMENTWISE comparison compute for a correlated node?
  (c) Can the existing PROBE_OFFSETS family distinguish two different circulant
      kernels at all?

Run:
    cd <worktree> && PYTHONPATH=$PWD/src \
        /Users/zzhang/projects/bayesmith/.venv/bin/python probes/probe_2_extraction_and_the_guard.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.exact.gaussian import PROBE_OFFSETS
from bayesmith.exact.precision import (
    CirculantPrecision,
    DiagonalPrecision,
    dense,
    log_density,
    quadratic,
)

jax.config.update("jax_enable_x64", True)

N = 8
KERNEL = np.array([2.0, 0.7, 0.3, 0.1, 0.05, 0.1, 0.3, 0.7])
LOC = np.array([0.3, -0.2, 0.9, 0.1, -0.4, 0.6, 0.0, 0.25])
rng = np.random.default_rng(0)

print("=" * 78)
print("(a) CirculantNormal.log_prob  vs  precision.log_density(CirculantPrecision)")
print("=" * 78)
cn = dist.CirculantNormal(jnp.asarray(LOC), covariance_row=jnp.asarray(KERNEL))
cp = CirculantPrecision(first_column=jnp.asarray(KERNEL))
worst = 0.0
for trial in range(5):
    x = jnp.asarray(rng.normal(size=N) * 1.5 + LOC)
    a = float(cn.log_prob(x))
    b = float(log_density(cp, x - jnp.asarray(LOC)))
    rel = abs(a - b) / max(abs(a), abs(b), 1.0)
    worst = max(worst, rel)
    print(f"  trial {trial}: numpyro {a:+.12f}   precision {b:+.12f}   rel {rel:.3e}")
print(f"  WORST relative disagreement over 5 draws: {worst:.3e}")

print()
print("  extraction surface on the distribution object:")
print(f"    .covariance_row  -> {np.asarray(cn.covariance_row).round(6).tolist()}")
print(f"    .covariance_rfft -> {np.asarray(cn.covariance_rfft).round(6).tolist()}")
print(f"    .loc             -> shape {np.shape(cn.loc)}")
print(f"    CirculantPrecision.eigenvalues -> {np.asarray(cp.eigenvalues).round(6).tolist()}")
print(f"    rfft(kernel).real              -> {np.fft.rfft(KERNEL).real.round(6).tolist()}")
print(f"    event_shape {cn.event_shape}, batch_shape {cn.batch_shape}, "
      f"log_prob shape {np.shape(cn.log_prob(jnp.asarray(LOC)))}")

print()
print("=" * 78)
print("(b) what check_gaussian's ELEMENTWISE comparison computes for this node")
print("=" * 78)
# The natural generalisation a relaxed gate would produce: loc from .loc,
# `scale` from sqrt(diag N) -- which for a circulant is the constant sqrt(k[0]).
scale = np.sqrt(np.full(N, KERNEL[0]))
LOG_2PI = float(np.log(2.0 * np.pi))
print(f"  sqrt(diag N) = {scale.round(6).tolist()}")
for offset in PROBE_OFFSETS:
    probe = jnp.asarray(LOC + offset * scale)
    actual = np.broadcast_to(np.asarray(cn.log_prob(probe)), (N,))
    predicted = (
        -0.5 * ((np.asarray(probe) - LOC) / scale) ** 2 - np.log(scale) - 0.5 * LOG_2PI
    )
    departure = np.abs(actual - predicted) / np.maximum(np.abs(predicted), 1.0)
    print(
        f"  offset {offset:+.1f}: joint log_prob {float(actual[0]):+10.5f}  "
        f"per-element predicted {predicted[0]:+9.5f}  worst departure {departure.max():.4e}"
    )
print("  (default rtol at float64 = 1e3*eps = {:.3e})".format(1e3 * float(np.finfo(np.float64).eps)))
