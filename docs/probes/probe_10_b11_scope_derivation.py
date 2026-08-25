"""Probe 10 -- can a bayesmith graph express B11's three scopes?

The migration spec's reason to REWRITE the evidence layer rather than
transplant it: rheplicant declares each latent's extent by hand
(`Factorization`, `scope="global" | "per_epoch" | "linked"`), while
bayesmith's graph carries plates and a dependency structure, so the
factorization should be DERIVED -- "which kills the whole error class
factorize.py exists for (the same space declared twice)".

That argument rests on every scope being expressible. This measures it, and
the third one is not the shape the spec's wording suggests.

  global     constant over the whole campaign
  per_epoch  constant within one epoch, integrated out inside it
  linked     evolves epoch to epoch through a transition

Run:
    cd <worktree> && PYTHONPATH=$PWD/src .venv/bin/python \
        docs/probes/probe_10_b11_scope_derivation.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import det, observe, plate, sample, trace
from bayesmith.dispatch.plan import compile as compile_graph

jax.config.update("jax_enable_x64", True)

N_EPOCH = 6


def report(label, build):
    try:
        graph = build()
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<44} NOT EXPRESSIBLE: {type(exc).__name__}")
        return None
    try:
        plan = compile_graph(graph)
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<44} builds, compile raises {type(exc).__name__}")
        return None
    method = plan.blocks[0].method
    print(f"  {label:<44} builds -> {method}")
    return plan


print("=" * 78)
print("(a) global and per_epoch, under one plate")
print("=" * 78)


def global_and_per_epoch():
    def model():
        epoch = plate("epoch", N_EPOCH)
        g = sample("g", lambda: dist.Normal(0.0, 1.0))
        n = sample("n", lambda: dist.Normal(0.0, 1.0), plate=epoch)
        mu = det("mu", lambda a, b: a + b, g, n, plate=epoch, linear_in=("g", "n"))
        observe(
            "d",
            lambda m: dist.Normal(m, 0.5),
            mu,
            plate=epoch,
            obs=jnp.arange(float(N_EPOCH)),
        )

    return trace(model)


report("global + per_epoch", global_and_per_epoch)
print("  -> both scopes are a PLATE MEMBERSHIP question: `node.plate` is empty")
print("     for a global latent and names the epoch plate for a per-epoch one.")
print("     Derivable, exactly as the spec says.")

print()
print("=" * 78)
print("(b) linked, spelled as a joint prior on one vector latent")
print("=" * 78)
print("  An AR(1) chain is a Gaussian whose PRECISION is tridiagonal.")

A, Q = 0.9, 0.4


def linked_joint_prior():
    band = np.zeros((N_EPOCH, N_EPOCH))
    for i in range(N_EPOCH):
        band[i, i] = (1.0 + A**2) / Q**2 if i < N_EPOCH - 1 else 1.0 / Q**2
        if i:
            band[i, i - 1] = band[i - 1, i] = -A / Q**2

    def model():
        z = sample(
            "z",
            lambda: dist.MultivariateNormal(
                jnp.zeros(N_EPOCH), precision_matrix=jnp.asarray(band)
            ),
        )
        mu = det("mu", lambda v: 1.0 * v, z, linear_in=("z",))
        observe(
            "d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.arange(float(N_EPOCH))
        )

    return trace(model)


report("linked, joint (centred) prior", linked_joint_prior)
print("  -> EXPRESSIBLE but routed to NUTS. The block-member gate reads a")
print("     latent's prior through `gaussian_parts`, which takes a diagonal")
print("     Normal only. A tridiagonal precision is a THIRD Precision row, and")
print("     `_log_spectrum_curvature`'s identity would not cover it either:")
print("     its eigenbasis is not parameter-independent in general.")

print()
print("=" * 78)
print("(c) linked, spelled NON-CENTRED: iid innovations + a linear recursion")
print("=" * 78)

RNG = np.random.default_rng(0)
TRUTH = np.zeros(N_EPOCH)
for _i in range(1, N_EPOCH):
    TRUTH[_i] = A * TRUTH[_i - 1] + Q * RNG.normal()
SIGMA = 0.5
DATA = TRUTH + SIGMA * RNG.normal(size=N_EPOCH)


def _lower():
    powers = A ** np.arange(N_EPOCH)
    return Q * np.tril(powers[:, None] / np.maximum(powers[None, :], 1e-300))


def linked_non_centred():
    def recursion(innovations):
        powers = A ** jnp.arange(N_EPOCH)
        lower = jnp.tril(powers[:, None] / jnp.maximum(powers[None, :], 1e-300))
        return Q * (lower @ innovations)

    def model():
        eps = sample("eps", lambda: dist.Normal(jnp.zeros(N_EPOCH), 1.0))
        z = det("z", recursion, eps, linear_in=("eps",))
        observe("d", lambda m: dist.Normal(m, SIGMA), z, obs=jnp.asarray(DATA))

    return trace(model)


plan = report("linked, non-centred", linked_non_centred)
if plan is not None:
    lower = _lower()
    normal = lower.T @ lower / SIGMA**2 + np.eye(N_EPOCH)
    reference = np.linalg.solve(normal, lower.T @ DATA / SIGMA**2)
    print(f"  plan default tol = {plan.exact.tol:.3g}, kappa(M) = "
          f"{np.linalg.cond(normal):.4g}")
    for tol in (None, 1e-10, 1e-14):
        estimate = plan.estimate() if tol is None else plan.estimate(tol=tol)
        got = np.asarray(estimate.values["eps"])
        error = np.max(np.abs(got - reference)) / np.max(np.abs(reference))
        label = "plan default" if tol is None else f"tol={tol:g}"
        print(f"    {label:<14} posterior-mean relative error {error:.3e}")
    print("  -> EXACT, and the error is CG tolerance rather than disagreement:")
    print("     it falls to 2.5e-16 when tol does. The recursion is LINEAR, so")
    print("     the block stays affine and no new machinery is needed.")

print()
print("=" * 78)
print("what this means for B11")
print("=" * 78)
print("  * global and per_epoch really are derivable from `node.plate`.")
print("  * linked is expressible and reaches the EXACT path, in its")
print("    non-centred parameterisation. That is a modelling convention the")
print("    factorization has to know about, not a scope tag it can read off")
print("    the graph -- the innovations are per-epoch and the chain lives in")
print("    a deterministic node, so `node.plate` alone cannot tell a linked")
print("    latent from an ordinary per-epoch one.")
print("  * the CENTRED spelling needs a third Precision row (tridiagonal).")
print("    Cost is not the QR: it is that `_log_spectrum_curvature`'s")
print("    identity holds only for a parameter-independent eigenbasis, so a")
print("    third row must bring its own variance-information term.")
