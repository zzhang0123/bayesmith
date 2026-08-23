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

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.exact.block import (
    LinearBlock,
    domain_centre,
    domain_zero,
    largest_variance,
    variance_parts,
)
from bayesmith.exact.conditioning import largest_eigenvalue, tree_norm

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
) -> Callable[[dict[str, Any]], dict[str, jax.Array]]:
    """``x -> (A^T N^-1 A + S^-1) x`` over the block's domain."""

    def half_chi2(parts: dict[str, Any]) -> jax.Array:
        pushed = block.forward(parts)
        return 0.5 * sum(jnp.sum(weight[name] * pushed[name] ** 2) for name in pushed)

    def normal(parts: dict[str, Any]) -> dict[str, jax.Array]:
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
    factor is 1, and the bound is exact.
    ``test_the_bound_is_tight_when_the_prior_alone_holds_a_direction`` (in
    ``tests/exact/test_solve.py``) is what pins that half of the claim --
    every other test of this bound sits on the loose side (3676x-1e11x over
    the true kappa, because their fixtures let the data constrain every
    direction), so until that test existed the design's own justification
    was untested. It grows in the opposite regime -- data far tighter than
    the prior in every direction -- where the guard may refuse a solve that
    was in fact accurate; measured on ``two_linear_latents`` at its
    declared (unwidened) prior widths: 3676x.
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


def _split_like(key: jax.Array, template: Any) -> Any:
    """One independent key per leaf of ``template``, same structure."""
    leaves, treedef = jax.tree.flatten(template)
    return jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))


def _conjugate_solve(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    tol: float,
    maxiter: int | None,
    key: jax.Array | None,
    require_convergence: float | None,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Shared machinery for the posterior mean and for a posterior draw.

    Both solve ``(A^T N^-1 A + S^-1) x = b`` by CG. They differ only in ``b``:
    the mean uses ``A^T N^-1 (d - offset) + S^-1 m``, a draw adds the two
    fluctuation terms. ``key=None`` selects the mean.
    """
    weight = _weights(noise_std)
    prior_variance = variance_parts(block)
    residual_data = jax.tree.map(jnp.subtract, block.data, block.offset)
    zero = domain_zero(block)
    centre = domain_centre(block)

    def pair_with(vector):
        """``A^T vector``, as the gradient of a real pairing.

        Measured, not assumed: swapping this body for ``block.adjoint``
        (which is exactly ``jax.vjp``'s pullback) leaves every test in this
        module green, because in a real (never complex) domain the gradient
        of a real bilinear pairing and the VJP pullback are the same map --
        there is no conjugate-transpose convention for them to disagree
        about. rheplicant's docstring justified this choice by exactly that
        convention, which does not apply here and would be a claim this
        package cannot back up. What the two conventions being identical
        does NOT remove is the reason to keep this written as a gradient
        rather than delegate to ``block.adjoint``: it is the same
        gradient-of-a-scalar mechanism :func:`normal_operator` uses for its
        own curvature, so the two cannot independently pick different sign
        or scaling conventions from each other -- a property a hand-picked
        ``block.adjoint`` call would not carry for free if either one ever
        changed.
        """

        def pairing(parts):
            pushed = block.forward(parts)
            return sum(jnp.sum(pushed[name] * vector[name]) for name in pushed)

        return jax.grad(pairing)(zero)

    normal = normal_operator(block, weight, prior_variance)

    # S^-1 m: a zero-mean prior is wrong for most physical quantities, and
    # shifting the prior is not the same act as shifting the model even though
    # the two give the same Gaussian.
    rhs = jax.tree.map(
        lambda base, mean, variance: base + mean / variance,
        pair_with(jax.tree.map(jnp.multiply, weight, residual_data)),
        centre,
        prior_variance,
    )

    if key is not None:
        # Constrained realization: two fluctuation terms whose covariances sum
        # to the normal operator itself, which is exactly why the solve comes
        # out distributed as the posterior rather than merely centred on its
        # mean.  b = A^T N^-1 (d-offset) + S^-1 m + A^T N^-1/2 w1 + S^-1/2 w2
        data_key, prior_key = jax.random.split(key)
        omega_data = jax.tree.map(
            lambda leaf, k: jax.random.normal(k, jnp.shape(leaf), dtype=leaf.dtype),
            residual_data,
            _split_like(data_key, residual_data),
        )
        omega_prior = jax.tree.map(
            lambda leaf, k: jax.random.normal(k, leaf.shape, dtype=leaf.dtype),
            zero,
            _split_like(prior_key, zero),
        )
        rhs = jax.tree.map(
            lambda base, from_data, from_prior, variance: (
                base + from_data + from_prior / jnp.sqrt(variance)
            ),
            rhs,
            pair_with(jax.tree.map(lambda w, o: jnp.sqrt(w) * o, weight, omega_data)),
            omega_prior,
            prior_variance,
        )

    solution, _ = jax.scipy.sparse.linalg.cg(normal, rhs, tol=tol, maxiter=maxiter)
    misfit = jax.tree.map(jnp.subtract, normal(solution), rhs)
    residual = tree_norm(misfit) / jnp.maximum(tree_norm(rhs), 1e-30)

    if require_convergence is not None:
        # jax's cg reports no convergence status of its own, so an unconverged
        # solve otherwise comes back looking like any other answer.
        # eqx.error_if fires under jit, where a Python `if` on a traced value
        # cannot.
        #
        # The residual ALONE cannot decide this. Error and residual differ by
        # the condition number, and for a block the data does not fully
        # identify kappa is enormous by construction -- lambda_min is exactly
        # the prior's 1/prior_std**2 -- so CG stops on a tiny residual with the
        # prior-dominated directions still at their starting value, and hands
        # back a draw whose posterior scatter there is orders of magnitude too
        # small. Guarding on the residual certifies precisely nothing in the
        # one regime these solvers exist to serve.
        bound = _condition_bound(
            block, weight, prior_variance, jax.random.key(0), POWER_ITERATIONS
        )
        # Named so the two branches below can each exclude it: a NaN or Inf
        # residual gets its OWN message further down, and neither of the
        # "unreachable" / "did not converge" messages below is advice a
        # non-finite residual can act on -- tightening tol or raising maxiter
        # does nothing to an operator or right-hand side that was already
        # non-finite before CG started.
        non_finite = ~jnp.isfinite(residual)
        error_bound = residual * bound
        bad = jnp.logical_or(non_finite, error_bound > require_convergence)

        # Below kappa*eps no tolerance can help: the arithmetic itself cannot
        # represent the answer that accurately. Worth its own message, because
        # the remedy is precision, and the natural response to the other
        # message -- tighten tol, raise maxiter -- burns a great many
        # iterations here to arrive at an equally wrong answer.
        epsilon = float(
            jnp.finfo(jnp.result_type(*jax.tree.leaves(block.offset))).eps
        )
        # Two independent reasons no tol or maxiter can help, and rheplicant
        # carries only the first. The second was measured: at float32,
        # `two_linear_latents` -- a TWO-parameter toy -- lands at
        # residual=1.73e-7 against eps=1.19e-7, i.e. CG has already reached the
        # floor, while bound*eps = 6.9e-4 stays under a 1e-3 target. Without
        # the second test the caller is told to tighten tol, which cannot move
        # a residual that is already at the precision floor. A guard whose
        # remedy is impossible is the same defect as one that misattributes.
        at_precision_floor = residual <= PRECISION_FLOOR * epsilon
        unreachable = jnp.logical_or(
            bound * epsilon > require_convergence, at_precision_floor
        )

        solution = eqx.error_if(
            solution,
            non_finite,
            "wiener_solve/gcr_sample produced a non-finite residual. That is not "
            "a convergence problem and no tol or maxiter affects it: the normal "
            "operator or the right-hand side is already non-finite before CG "
            "starts. The usual causes are a sigma of zero somewhere in "
            "noise_std, a prior_std of zero, or a prediction that already "
            "overflowed at the point the block was built. check_gaussian "
            "refuses a non-positive sigma at block-build time, so a non-finite "
            "residual here points at the prediction or the arithmetic, not at "
            "the declaration.",
        )
        solution = eqx.error_if(
            solution,
            jnp.logical_and(jnp.logical_and(bad, unreachable), ~non_finite),
            "wiener_solve/gcr_sample cannot reach require_convergence at this "
            "precision, and no tol or maxiter will help -- either the condition "
            "bound times the machine epsilon already exceeds the target, or the "
            "relative residual is already at the precision floor, meaning CG has "
            "done everything this dtype allows. Run the solve inside "
            "`with jax.enable_x64(True):`, or strengthen the prior. Note that "
            "condition_bound() reports an UPPER bound: it exceeds the true "
            "conditioning by lambda_min * max(prior_variance), which is how much "
            "better the data constrains the weakest direction than the prior "
            "does -- measured at 3676x on this package's own two-parameter toy. "
            "So a refusal here may be the bound being conservative rather than "
            "the answer being inaccurate; x64 settles it either way.",
        )
        solution = eqx.error_if(
            solution,
            jnp.logical_and(jnp.logical_and(bad, ~unreachable), ~non_finite),
            "wiener_solve/gcr_sample did not converge: the relative residual "
            "times the condition bound -- the bound on the RELATIVE ERROR, which "
            "is what require_convergence limits -- exceeds it, and the residual "
            "is not yet at the precision floor, so there is room to improve it. "
            "The residual alone looks converged; it is not, along the directions "
            "the prior dominates. Pass tol ~ require_convergence/bound with a "
            "maxiter to match, or strengthen the prior. condition_bound() "
            "reports the bound.",
        )
    return solution, residual


def wiener_solve(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = 1e-3,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Posterior mean of a linear-Gaussian block -- the Wiener filter, by CG.

    This is the posterior **mean**, not a sample. For a draw see
    :func:`gcr_sample`, which adds a fluctuation term to this same right-hand
    side and costs exactly the same solve.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        noise_std: ``{observed: sigma}``, from
            :func:`bayesmith.exact.gaussian.noise_std_at`. A sigma that has
            already been decided -- a conjugate solve has no prediction to
            evaluate a rule at, the prediction being what it solves for. For a
            prediction-dependent noise model see
            :func:`bayesmith.exact.gls.iterative_gls`, which finds the fixed
            point and hands the result back here.
        tol: CG tolerance -- a bound on the relative RESIDUAL, which is not
            the same as accuracy. See the note below.
        maxiter: CG iteration cap. ``None`` lets JAX choose.
        require_convergence: raise unless the relative ERROR can be bounded by
            this. ``None`` disables the guard and returns whatever CG
            produced. On by default because jax's ``cg`` reports no
            convergence status, so an unconverged solve otherwise comes back
            looking exactly like a converged one.

            The bound is ``kappa * relative_residual``, with ``kappa`` from
            :func:`condition_bound`. That costs ``POWER_ITERATIONS``
            extra operator applications, which on a well-conditioned block
            roughly DOUBLES the solve. In a Gibbs sweep, call
            :func:`condition_bound` once outside the loop, choose ``tol``
            from it, and pass ``require_convergence=None`` inside. What you
            must NOT do is leave ``tol`` at its default and the guard off --
            that is the combination that returns a silently over-confident
            posterior.

    Returns:
        ``(x_hat, relative_residual)``, the residual being
        ``||M x_hat - b|| / ||b||``. Note this is the residual, not the error;
        multiply by :func:`condition_bound` for the error bound.

    Note:
        **Where S comes from.** Each latent's own ``dist_fn`` is this
        package's one statement of what it is a priori, and it is the
        statement ``to_numpyro`` reads. So it is the statement this solve
        reads too -- there is no keyword to override it, and therefore no way
        for the exact exit and NUTS to target different posteriors.
    """
    return _conjugate_solve(
        block,
        noise_std=noise_std,
        tol=tol,
        maxiter=maxiter,
        key=None,
        require_convergence=require_convergence,
    )
