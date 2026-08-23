"""Conjugate-Gaussian solves over a matrix-free linear block.

The posterior mean (:func:`wiener_solve`) and an exact posterior draw
(:func:`gcr_sample`) are the same conjugate-gradient solve of

    (A^T N^-1 A + S^-1) x = b

differing only in ``b``. They share one private routine that takes ``key=None``
for the mean and a key for a draw, which is why the two can never drift apart.

**The normal operator and the right-hand side are obtained as gradients of the
objective itself**, never assembled from ``A`` and ``A^T`` by hand. That is
not a shortcut: it makes the operator symmetric positive definite *by
construction*, with no adjoint convention left to get wrong -- and, for a
group, no cross-block bookkeeping either, since ``jax.grad`` of the group's
own chi-squared produces the full operator, off-diagonal blocks included,
which is exactly the coupling an alternating solve throws away.

Ported from ``rheplicant.inference.linear``, with the codomain generalised
from one array to a dict of observed nodes, and with every prior/noise
keyword removed -- the graph is the only statement of either.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.exact.block import (
    LinearBlock,
    domain_zero,
    largest_variance,
    variance_parts,
)
from bayesmith.exact.conditioning import largest_eigenvalue

#: Power-iteration steps for the top of the spectrum. The estimate typically
#: settles within three; this leaves margin at a fixed cost of
#: ``POWER_ITERATIONS`` operator applications per guarded solve. Only the top
#: is measured -- see :func:`_condition_bound` for where the bottom comes from
#: and why it is not measured.
POWER_ITERATIONS: int = 12

#: Multiple of the working precision's epsilon below which a relative residual
#: counts as "CG has done all it can here". Measured: a converged float32 solve
#: on this package's own toy models lands at 1.4-1.5 times eps, so 10 leaves
#: room without admitting a genuinely unconverged solve, whose residual is
#: orders of magnitude larger.
PRECISION_FLOOR: float = 10.0


def _weights(noise_std: dict[str, Any]) -> dict[str, jax.Array]:
    return {name: 1.0 / jnp.asarray(std) ** 2 for name, std in noise_std.items()}


def normal_operator(
    block: LinearBlock, weight: dict[str, Any], prior_variance: dict[str, Any]
):
    """``x -> (A^T N^-1 A + S^-1) x`` over the block's domain."""

    def half_chi2(parts):
        pushed = block.forward(parts)
        return 0.5 * sum(jnp.sum(weight[name] * pushed[name] ** 2) for name in pushed)

    def normal(parts):
        curvature = jax.grad(half_chi2)(parts)
        return jax.tree.map(lambda c, p, v: c + p / v, curvature, parts, prior_variance)

    return normal


def _condition_bound(
    block: LinearBlock,
    weight: dict[str, Any],
    prior_variance: dict[str, Any],
    key: jax.Array,
    iterations: int,
) -> jax.Array:
    """Upper bound on ``kappa`` of ``A^T N^-1 A + S^-1``.

    ``lambda_max`` is measured. ``lambda_min`` is **not**: it is bounded from
    below by the prior's own curvature, because ``A^T N^-1 A`` is positive
    semi-definite and therefore

        lambda_min(A^T N^-1 A + S^-1)  >=  lambda_min(S^-1)  =  1 / max(S)

    so the quotient is an upper bound rather than an estimate. That is the
    direction a safety guard needs -- an overestimate of kappa can only make
    the guard refuse a solve that was fine, while an underestimate makes it
    accept one that was not.

    Measuring ``lambda_min`` instead, by a second power iteration on
    ``lambda_max * I - M``, is what rheplicant does and what this plan
    originally specified. It was measured to fail *in principle* on a graded
    spectrum -- the shifted operator's leading eigenvalues all crowd against
    ``lambda_max`` with vanishing gaps -- and to fail one-sidedly in the
    dangerous direction. See :mod:`bayesmith.exact.conditioning`'s module
    docstring for the numbers.

    **The bound is tight exactly where it matters, and the looseness has a
    closed form.** ``bound / kappa`` works out to almost exactly
    ``lambda_min * max(prior_variance)`` -- how much better the data
    constrains the block's weakest direction than the prior alone does,
    since the measured ``lambda_max`` differs from the true one only by the
    power iteration's own small, one-sided error. When the prior alone holds
    that direction, ``lambda_min`` equals ``1 / max(prior_variance)``, the
    factor is 1, and the bound is exact. It grows in the opposite regime --
    data far tighter than the prior in every direction -- where the guard
    may refuse a solve that was in fact accurate; measured on
    ``two_linear_latents`` at its declared (unwidened) prior widths: 3676x.
    That regime is also where CG converges in a handful of iterations and
    the residual is small enough to absorb the slack, which is why the
    trade is worth taking.

    For a group this is the JOINT bound, and it is the number a per-block
    guard cannot produce: two latents the data barely distinguishes give a
    well-conditioned operator each and a badly conditioned one together.
    """
    largest = largest_eigenvalue(
        normal_operator(block, weight, prior_variance),
        domain_zero(block),
        key,
        iterations,
    )
    return largest * largest_variance(prior_variance)


def condition_bound(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    iterations: int = POWER_ITERATIONS,
    key: jax.Array | None = None,
) -> jax.Array:
    """An upper bound on the conditioning of the system this block is solved with.

    Use it to pick ``tol``: for a target relative accuracy ``a``, ask for
    roughly ``tol = a / condition_bound(...)``.

    A large bound is not a defect, it is the design: for a block the data does
    not fully identify, ``lambda_min`` is exactly ``1/prior_std**2`` while
    ``lambda_max`` is set by the data, so it grows with how much better the
    data constrains one direction than the prior constrains another.

    Costs ``iterations`` applications of the normal operator -- each the same
    JVP-plus-VJP a CG iteration costs -- and forms no matrix.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        noise_std: ``{observed: sigma}``, as
            :func:`bayesmith.exact.gaussian.noise_std_at` returns. A decided
            sigma, not a rule for producing one: a conditioning number belongs
            to one particular normal operator.
        iterations: power-iteration steps for ``lambda_max``.
        key: PRNG key for the starting vector. Fixed by default, so the bound
            is reproducible.

    Returns:
        ``lambda_max * max(prior_variance)``, an upper bound on ``kappa`` --
        up to the accuracy of the ``lambda_max`` estimate, which converges
        geometrically and always from BELOW, so it can only make the bound
        smaller. ``test_largest_eigenvalue_approaches_the_truth_from_below``
        in ``tests/exact/test_conditioning.py`` pins that direction.
    """
    return _condition_bound(
        block,
        _weights(noise_std),
        variance_parts(block),
        jax.random.key(0) if key is None else key,
        iterations,
    )
