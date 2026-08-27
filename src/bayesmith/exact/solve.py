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
    real_parts,
    variance_parts,
)
from bayesmith.exact.conditioning import (
    extreme_eigenvalues,
    largest_eigenvalue,
    tree_norm,
)
from bayesmith.exact.precision import Precision, quadratic

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


def normal_operator(
    block: LinearBlock, weight: dict[str, Precision], prior_variance: dict[str, Any]
) -> Callable[[dict[str, Any]], dict[str, jax.Array]]:
    """``x -> (A^T N^-1 A + S^-1) x`` over the block's REAL degrees of freedom.

    Never forms ``N``, which is why a non-diagonal covariance costs nothing
    structural here: CG only ever needs the quadratic form, and
    :func:`~bayesmith.exact.precision.quadratic` is it.

    The argument and the result live in PARTS space (see
    :func:`~bayesmith.exact.block.real_parts`), which is the block's domain
    exactly when every member is real -- so every caller predating complex
    support is unaffected. For a complex member it is ``(re, im)``, and the
    ``jax.grad`` below therefore always differentiates a real function of real
    leaves. That is what keeps the gradient an honest transpose: JAX hands
    back the CONJUGATE gradient for a complex input, so taking this derivative
    in the domain would silently introduce a conjugation the rest of the solve
    does not account for.
    """
    _, join = real_parts(block)

    def half_chi2(parts: dict[str, Any]) -> jax.Array:
        pushed = block.forward(join(parts))
        return 0.5 * sum(quadratic(weight[name], pushed[name]) for name in pushed)

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
    precision: dict[str, Precision],
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
        precision: ``{observed: N^-1}``, as
            :func:`bayesmith.exact.gaussian.precision_at` returns, or
            :func:`~bayesmith.exact.precision.diagonal_from` applied to a
            sigma dict. A decided operator, not a rule for producing one: a
            conditioning number belongs to one particular normal operator.
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
        precision,
        variance_parts(block),
        jax.random.key(0) if key is None else key,
        iterations,
    )


def condition_estimate(
    block: LinearBlock,
    *,
    precision: dict[str, Precision],
    iterations: int = POWER_ITERATIONS,
    key: jax.Array | None = None,
) -> jax.Array:
    """The MEASURED conditioning of the system this block is solved with.

    ``kappa(A^T N^-1 A + S^-1)`` says how much a solver's residual understates
    its error: for a solution with relative residual ``r``,
    ``||x - x*|| / ||x*|| <= kappa * r``, so a residual of 1e-6 against
    kappa=1e7 certifies nothing at all.

    **This is a diagnostic and not a bound. Never divide an accuracy target
    by it, and never guard on it** -- :func:`condition_bound` is the one to
    divide by, and it is what ``require_convergence`` itself reads. This one
    measures ``lambda_min`` by a second power iteration whose leading
    eigenvalues crowd against ``lambda_max`` with vanishing gaps on a graded
    spectrum, so the ``lambda_min`` it returns is too LARGE and this kappa
    **too small** -- measured on a 20-point geometric spectrum at a true
    kappa of 1e4, ``lambda_min`` came back 33.9x high and kappa 33.9x low; at
    1e7 over 50 points the factor was ~700 and 2000 iterations did not close
    it. A ``tol`` computed from it is too LOOSE by that factor, which is the
    direction that certifies an answer it should have refused.

    **What it is good for is the thing a bound cannot do: it can SEE a
    degeneracy.** A near-degenerate partition lives entirely in
    ``lambda_min``, which :func:`condition_bound` replaces with the prior's
    floor and therefore cannot report however tight the spectrum gets. Read
    it as "how badly conditioned is this partition?" and never as a
    certificate. For a group it is the JOINT condition number, and that is
    the number a per-block guard cannot produce: two latents the data barely
    distinguishes give a well-conditioned operator each and a badly
    conditioned one together.

    Large kappa is not a defect here, it is the design: for a block the data
    does not fully identify, ``lambda_min`` is exactly ``1/prior_std**2``
    while ``lambda_max`` is set by the data.

    Costs ``2 * iterations`` applications of the normal operator -- each the
    same JVP-plus-VJP a CG iteration costs -- and forms no matrix.
    :func:`condition_bound` costs half that, measuring only the top.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        precision: ``{observed: N^-1}``, as :func:`condition_bound` takes it.
            A decided operator, not a rule: a kappa belongs to one particular
            normal operator, and one computed under a different reading of
            the same array describes an operator nobody builds.
        iterations: power-iteration steps per END of the spectrum. The
            default is comfortable; the top settles within a few, and the
            bottom does not settle at all on a graded spectrum, which is the
            whole of the caveat above.
        key: PRNG key for the starting vectors. Fixed by default, so a
            diagnostic printed twice is one number.

    Returns:
        The measured condition number, floored so that ``lambda_min`` can
        never fall below the prior's own curvature -- ``A^T N^-1 A`` is
        positive semi-definite, so that floor is rigorous even where the
        measurement is not.
    """
    prior_variance = variance_parts(block)
    largest, smallest = extreme_eigenvalues(
        normal_operator(block, precision, prior_variance),
        domain_zero(block),
        jax.random.key(0) if key is None else key,
        iterations,
    )
    floor = 1.0 / largest_variance(prior_variance)
    return largest / jnp.maximum(smallest, floor)


def _split_like(key: jax.Array, template: Any) -> Any:
    """One independent key per leaf of ``template``, same structure."""
    leaves, treedef = jax.tree.flatten(template)
    return jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))


def _conjugate_solve(
    block: LinearBlock,
    *,
    precision: dict[str, Precision],
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
    weight = precision
    split, join = real_parts(block)
    prior_variance = variance_parts(block)
    residual_data = jax.tree.map(jnp.subtract, block.data, block.offset)
    zero = domain_zero(block)
    # `domain_centre` speaks the DOMAIN -- it is the prior's mean, a latent
    # value -- while everything below this line is in parts space. Split once,
    # here, rather than letting the two representations meet further down.
    centre = split(domain_centre(block))

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
            pushed = block.forward(join(parts))
            return sum(jnp.sum(pushed[name] * vector[name]) for name in pushed)

        return jax.grad(pairing)(zero)

    normal = normal_operator(block, weight, prior_variance)

    # S^-1 m: a zero-mean prior is wrong for most physical quantities, and
    # shifting the prior is not the same act as shifting the model even though
    # the two give the same Gaussian.
    rhs = jax.tree.map(
        lambda base, mean, variance: base + mean / variance,
        pair_with(
            {name: weight[name].apply(value) for name, value in residual_data.items()}
        ),
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
        # Getting the prior term's division backwards (`from_prior *
        # jnp.sqrt(variance)` instead of `/`) is not a sign error: it scales
        # the WHOLE term's amplitude by `variance` rather than `1/variance`,
        # since `omega*sqrt(v) == v * (omega/sqrt(v))`. Measured (Task 7
        # mutation testing, tests/exact/test_solve.py): on `straight_line`'s
        # `w` (variance=4.0) that mutation moved the drawn std from the true
        # 2.0 to 8.02 -- exactly the predicted 4x -- and separately widened
        # `two_linear_latents`' drawn covariance past its oracle comparison.
        rhs = jax.tree.map(
            lambda base, from_data, from_prior, variance: (
                base + from_data + from_prior / jnp.sqrt(variance)
            ),
            rhs,
            pair_with(
                {name: weight[name].whiten(value) for name, value in omega_data.items()}
            ),
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
        epsilon = float(jnp.finfo(jnp.result_type(*jax.tree.leaves(block.offset))).eps)
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
            "starts. The usual causes are a zero somewhere in the precision's "
            "own covariance, a prior_std of zero, or a prediction that already "
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
    # Back to the DOMAIN. Callers -- and the graph -- speak latent values, so
    # parts space stops at this line and a complex member comes out complex.
    return join(solution), residual


def wiener_solve(
    block: LinearBlock,
    *,
    precision: dict[str, Precision],
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = None,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Posterior mean of a linear-Gaussian block -- the Wiener filter, by CG.

    This is the posterior **mean**, not a sample. For a draw see
    :func:`gcr_sample`, which adds a fluctuation term to this same right-hand
    side and costs exactly the same solve.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        precision: ``{observed: N^-1}``, from
            :func:`bayesmith.exact.gaussian.precision_at`, or
            :func:`~bayesmith.exact.precision.diagonal_from` applied to what
            :func:`bayesmith.exact.gaussian.noise_std_at` returns. A decided
            operator -- a conjugate solve has no prediction to evaluate a rule
            at, the prediction being what it solves for. For a prediction-
            dependent noise model see
            :func:`bayesmith.exact.gls.iterative_gls`, which finds the fixed
            point and hands the result back here as
            :attr:`~bayesmith.exact.gls.GLSResult.precision`.

            **An operator, not a sigma.** CG only ever needs ``N^-1 r`` and
            ``r^T N^-1 r``, so a correlated covariance costs nothing
            structural here -- which is the whole point of taking the noise
            as a :class:`~bayesmith.exact.precision.Precision` rather than as
            per-sample values that presuppose independent samples.
        tol: CG tolerance -- a bound on the relative RESIDUAL, which is not
            the same as accuracy. See the note below.
        maxiter: CG iteration cap. ``None`` lets JAX choose.
        require_convergence: raise unless the relative ERROR can be bounded by
            this. ``None`` -- the default -- returns whatever CG produced.

            **Off by default, measured**, for the reason
            ``test_the_convergence_guard_is_off_by_default_but_reachable`` in
            ``tests/exact/test_solve.py`` records: at the defaults this
            function actually ships (float32, ``tol=1e-6``), guarding at
            ``1e-3`` refuses 6 of the 22 linear-Gaussian fixtures in
            ``tests/exact/models.py``, and all six refusals are FALSE -- the
            oracle-checked relative error of every one is between 0 and
            3.7e-07 against the 1e-3 the guard promises. The cause is
            :func:`condition_bound`'s own conservatism, which its docstring
            documents; it is an UPPER bound, and on a prediction-dependent
            sigma it runs to 1e10 while the solve is accurate to 1e-07.

            This mirrors the dispatch layer, whose defaults were settled the
            same way -- see
            ``test_the_convergence_guard_is_off_by_default_but_reachable`` in
            ``tests/dispatch/test_dispatch_entry.py``. The two layers agree
            rather than differ, and they agree because the same measurement
            was run on each.

            The bound is ``kappa * relative_residual``, with ``kappa`` from
            :func:`condition_bound`. That costs ``POWER_ITERATIONS``
            extra operator applications, which on a well-conditioned block
            roughly DOUBLES the solve. In a Gibbs sweep, call
            :func:`condition_bound` once outside the loop, choose ``tol``
            from it, and pass ``require_convergence=None`` inside.

            **What replaces the guard now that it is off.** Leaving ``tol``
            at its default and the guard off is the combination that returns
            a silently over-confident posterior, and it is now the DEFAULT
            combination -- so the obligation moved to the caller rather than
            disappearing. Call :func:`condition_bound` and multiply it by the
            returned residual to get the error bound yourself; that is the
            same number the guard computed, minus the decision to refuse on
            it. The measurement above is why the decision is yours: on a
            prediction-dependent sigma the bound is right about being an
            upper bound and wrong about being actionable.

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
        precision=precision,
        tol=tol,
        maxiter=maxiter,
        key=None,
        require_convergence=require_convergence,
    )


def gcr_sample(
    block: LinearBlock,
    *,
    precision: dict[str, Precision],
    key: jax.Array,
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = None,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Draw an EXACT posterior sample of a linear-Gaussian block.

    The constrained-realization identity: solve the same system
    :func:`wiener_solve` does, with two white-noise terms added to the
    right-hand side, so that ``b`` has the posterior-mean numerator as its
    mean and covariance ``A^T N^-1 A + S^-1`` -- the operator itself. Then
    ``x = M^-1 b`` has the posterior mean and covariance ``M^-1 M M^-1 =
    M^-1`` exactly. Not an approximation and not a Markov chain: every call is
    an independent draw, with no burn-in and nothing to diagnose.

    It costs one CG solve -- the same as the mean -- because the fluctuation
    enters the right-hand side, never the operator. That is what makes a
    10^6-dimensional block samplable at all.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        precision: ``{observed: N^-1}``, exactly as for
            :func:`wiener_solve`. This is the consumer that needs
            :meth:`~bayesmith.exact.precision.Precision.whiten`: the
            fluctuation term is ``N^-1/2 omega``, which
            :meth:`~bayesmith.exact.precision.Precision.apply` and
            :meth:`~bayesmith.exact.precision.Precision.log_normalizer`
            cannot build between them.
        key: PRNG key. ``vmap`` over split keys for many independent draws.
        tol: CG tolerance -- a bound on the residual, not on the accuracy.
        maxiter: CG iteration cap.
        require_convergence: as for :func:`wiener_solve`, which a draw is MORE
            exposed to than the mean. The fluctuation term ``S^-1/2 w2`` puts
            weight on every direction of the latent by construction, including
            the ones the data is blind to -- so a draw always has something to
            resolve where the operator is worst conditioned, whereas the mean
            does only when the prior mean is nonzero.

    Returns:
        ``(x, relative_residual)``. An unconverged CG returns a draw from the
        WRONG distribution -- and one that is too NARROW, since the directions
        left unresolved are the prior-dominated ones that should have carried
        the most scatter.
    """
    return _conjugate_solve(
        block,
        precision=precision,
        tol=tol,
        maxiter=maxiter,
        key=key,
        require_convergence=require_convergence,
    )
