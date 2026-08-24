"""Probe 3 -- what verification a correlated node needs, and what the current
probe family can and cannot see.

`check_gaussian` probes at `loc + offset * scale` for five scalar offsets. For
a STATIONARY covariance `sqrt(diag N)` is constant, so every probe is
`loc + c * 1` -- one single direction of an n-dimensional space. This probe
builds two circulant covariances that are IDENTICAL along that direction by
construction (same DC eigenvalue, same determinant) and wildly different
elsewhere, then measures:

  (c) what the scalar-offset family reports  -> the blindness
  (d) what a random displacement reports     -> partial fix, probabilistic
  (e) what ONE gradient of log_prob reports  -> n equations from one AD pass

`grad log_prob(x) = -N^-1 (x - loc)` exactly, for any Gaussian. So a single
gradient evaluation is an ELEMENTWISE check of `Precision.apply`, and
`log_prob(loc) = -0.5 * log_normalizer()` is a scalar check of the other half.

Run:
    cd <worktree> && PYTHONPATH=$PWD/src \
        /Users/zzhang/projects/bayesmith/.venv/bin/python probes/probe_3_what_a_correlated_guard_must_probe.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.exact.gaussian import PROBE_OFFSETS
from bayesmith.exact.precision import CirculantPrecision, dense, log_density

jax.config.update("jax_enable_x64", True)

N = 8
LOC = np.array([0.3, -0.2, 0.9, 0.1, -0.4, 0.6, 0.0, 0.25])
rng = np.random.default_rng(0)


def kernel_from_eigenvalues(eig):
    """Real first column of the circulant with this (symmetric) spectrum."""
    row = np.real(np.fft.ifft(eig))
    assert np.allclose(np.real(np.fft.fft(row)), eig), "spectrum did not round-trip"
    return row


# A: an ordinary falling autocovariance.
eigA = np.real(np.fft.fft(np.array([2.0, 0.7, 0.3, 0.1, 0.05, 0.1, 0.3, 0.7])))
# B: SAME lambda_0 (so the constant-vector quadratic form matches) and SAME
# product of eigenvalues (so log det, hence log_normalizer, matches) -- push
# lambda_1 up by t and lambda_3 down by t, both of which appear twice.
t = 1.6
eigB = eigA.copy()
for idx, factor in ((1, t), (7, t), (3, 1.0 / t), (5, 1.0 / t)):
    eigB[idx] *= factor

kerA, kerB = kernel_from_eigenvalues(eigA), kernel_from_eigenvalues(eigB)
print(f"eigenvalues A = {eigA.round(6).tolist()}")
print(f"eigenvalues B = {eigB.round(6).tolist()}")
print(f"both positive definite:  A {eigA.min() > 0}  B {eigB.min() > 0}")
print(f"lambda_0     A {eigA[0]:.12f}   B {eigB[0]:.12f}")
print(f"sum log eig  A {np.log(eigA).sum():.12f}   B {np.log(eigB).sum():.12f}")
print(f"kernel A = {kerA.round(6).tolist()}")
print(f"kernel B = {kerB.round(6).tolist()}")

cnA = dist.CirculantNormal(jnp.asarray(LOC), covariance_row=jnp.asarray(kerA))
cnB = dist.CirculantNormal(jnp.asarray(LOC), covariance_row=jnp.asarray(kerB))
cpA = CirculantPrecision(first_column=jnp.asarray(kerA))
cpB = CirculantPrecision(first_column=jnp.asarray(kerB))

invA = np.linalg.inv(np.array([np.roll(kerA, i) for i in range(N)]))
invB = np.linalg.inv(np.array([np.roll(kerB, i) for i in range(N)]))
print(
    f"\nhow different the two PRECISIONS actually are: "
    f"||N_A^-1 - N_B^-1||_F / ||N_A^-1||_F = "
    f"{np.linalg.norm(invA - invB) / np.linalg.norm(invA):.4f}"
)

scale = np.sqrt(np.full(N, kerA[0]))
print(f"sqrt(diag N_A) = {scale.round(6).tolist()}  (constant: the kernel is stationary)")

print()
print("=" * 78)
print("(c) the PROBE_OFFSETS family: probe = loc + offset * sqrt(diag N)")
print("=" * 78)
worst_uniform = 0.0
for offset in PROBE_OFFSETS:
    probe = jnp.asarray(LOC + offset * scale)
    a, b = float(cnA.log_prob(probe)), float(cnB.log_prob(probe))
    rel = abs(a - b) / max(abs(a), abs(b), 1.0)
    worst_uniform = max(worst_uniform, rel)
    print(f"  offset {offset:+.1f}: A {a:+14.10f}   B {b:+14.10f}   rel {rel:.3e}")
print(f"  WORST over the whole probe family: {worst_uniform:.3e}")
print(f"  default rtol at float64 (1e3*eps): {1e3 * float(np.finfo(np.float64).eps):.3e}")
print(f"  -> the family {'CANNOT' if worst_uniform < 1e3*np.finfo(np.float64).eps else 'can'} "
      "tell these two covariances apart.")

print()
print("=" * 78)
print("(d) a RANDOM displacement instead of a constant one")
print("=" * 78)
rels = []
for trial in range(8):
    probe = jnp.asarray(LOC + rng.normal(size=N))
    a, b = float(cnA.log_prob(probe)), float(cnB.log_prob(probe))
    rel = abs(a - b) / max(abs(a), abs(b), 1.0)
    rels.append(rel)
    print(f"  trial {trial}: A {a:+14.10f}   B {b:+14.10f}   rel {rel:.3e}")
print(f"  worst {max(rels):.3e}, best {min(rels):.3e} over 8 random displacements")

print()
print("=" * 78)
print("(e) ONE gradient of log_prob -- n equations, elementwise")
print("=" * 78)
grad_logp = jax.grad(lambda x, d: d.log_prob(x))
r = jnp.asarray(rng.normal(size=N))
probe = jnp.asarray(LOC) + r

gA = np.asarray(grad_logp(probe, cnA))
applyA = np.asarray(cpA.apply(r))
print(f"  -grad log_prob_A(loc+r) = {(-gA).round(8).tolist()}")
print(f"  CirculantPrecision_A.apply(r) = {applyA.round(8).tolist()}")
err = np.abs(-gA - applyA) / np.maximum(np.abs(applyA), 1.0)
print(f"  worst elementwise relative error, A vs its own Precision: {err.max():.3e}")

gB = np.asarray(grad_logp(probe, cnB))
errAB = np.abs(-gB - applyA) / np.maximum(np.abs(applyA), 1.0)
print(f"  worst elementwise relative error, B's gradient vs A's Precision: {errAB.max():.3e}")
print(f"  -> a single gradient separates them by {errAB.max() / max(err.max(), 1e-300):.2e}x")

print()
print("  the other half -- the normaliser -- from log_prob AT the mode:")
for name, cn, cp in (("A", cnA, cpA), ("B", cnB, cpB)):
    at_mode = float(cn.log_prob(jnp.asarray(LOC)))
    half = -0.5 * float(cp.log_normalizer())
    print(
        f"    {name}: log_prob(loc) {at_mode:+.12f}   "
        f"-0.5*log_normalizer() {half:+.12f}   rel "
        f"{abs(at_mode - half) / max(abs(at_mode), 1.0):.3e}"
    )

print()
print("  cost of each verification style, for one observed node of size n:")
print("    scalar-offset family : 5 x log_prob                    (5 scalars)")
print("    random displacements : k x log_prob                    (k scalars)")
print("    gradient probe       : 1 x log_prob + 1 x grad log_prob (n+1 numbers)")
print("    dense oracle         : dense(precision, n) = n applies  (n^2 numbers)")
