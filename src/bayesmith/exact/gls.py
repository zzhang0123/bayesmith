"""Iteratively reweighted least squares: finding the covariance to solve at.

:func:`~bayesmith.exact.solve.gcr_sample` is a linear sampler *given* a
covariance and :func:`~bayesmith.exact.solve.wiener_solve` is the
corresponding mean. Both take ``noise_std`` and neither cares where it came
from. For a constant sigma there is nothing more to say.

For a sigma that tracks the prediction -- a radiometer's
``sigma_i = kappa |prediction_i|`` -- there is: the weights depend on the
solution and the solution depends on the weights, and neither is available
first. This module supplies the missing half and changes nothing about the
two solvers.

The algorithm is a fixed-point iteration: solve with the current weights,
recompute the weights at the new prediction, repeat. It is the same
iteratively-reweighted GLS as hydra-tod's ``iterative_gls``, but
**matrix-free** -- hydra-tod forms a dense design matrix and a dense
``N_inv``, while here the same algorithm runs on the block's JVP and VJP.

**What this estimator is, and is not.** Freezing sigma inside each solve is
what makes each step a linear-Gaussian problem, and it is also what makes the
converged answer *generalized least squares* rather than the maximum of the
full Gaussian likelihood: the log-determinant's dependence on the solution is
held fixed rather than differentiated. That difference is exactly what P3b's
importance weight puts back, which is why this function is both a point
estimate here and the proposal centre there.

**The gap, measured rather than asserted.** On ``radiometer()`` (true weight
3.0), minimising the FULL Gaussian NLL densely -- differentiating
``sum(log(sigma_i(w)))`` through ``w`` rather than freezing it, against the
GLS fixed point at the same kappa: at kappa=0.05 (radiometer's own default)
the two agree to 0.08% relative (``w_gls=3.0255`` vs ``w_mle=3.0232``); at
kappa=1 they are 19.5% apart (``3.4681`` vs ``2.9016``); by kappa=3.5-4 the
gap is ~50% and the two estimates sit on OPPOSITE sides of the true value
(``w_gls~4.00`` above it, ``w_mle~2.66-2.67`` below). ``NLL(w_gls) >=
NLL(w_mle)`` held at every kappa tried, as it must, and the gap shrinks to
zero as kappa does: freezing sigma's dependence on the solution is exact at
kappa=0 (sigma does not depend on the prediction at all there) and
increasingly costly as that dependence strengthens. This is the quantity
P3b's importance weight exists to correct for.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.block import LinearBlock, domain_centre
from bayesmith.exact.conditioning import tree_norm
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.solve import wiener_solve
from bayesmith.graph.graph import Graph

#: Reweighting steps taken before the convergence test is consulted. Matches
#: hydra-tod's default: the first steps of a fixed-point iteration can be
#: nearly stationary without being near the fixed point.
MIN_REWEIGHTS: int = 5

#: Cap on reweighting steps, so a non-contracting problem terminates and says
#: so through ``converged`` rather than spinning.
MAX_REWEIGHTS: int = 100

#: Multiple of the working precision's epsilon used as the default
#: ``reweight_tol``. See :func:`iterative_gls` for why it cannot be a constant.
REWEIGHT_TOL_EPS: float = 8.0

#: Multiples of the prior standard deviation the dependence probe moves the
#: block by. Movement is measured against the value at the CENTRE, not between
#: probes, so one probe would usually do -- except where sigma is clipped or
#: floored on one side (``kappa * max(mu, 0) + floor`` reads exactly constant
#: for every negative probe). Two-sided, and at unequal magnitudes so a sigma
#: that happens to be symmetric about the centre still moves the two probes by
#: different amounts.
DEPENDENCE_PROBES: tuple[float, ...] = (1.0, -0.5)


class GLSResult(NamedTuple):
    """What a reweighting run produced. A pytree, so it survives ``jit``.

    Attributes:
        noise_std: the converged sigma -- **the covariance**, and the whole
            point of the exercise. Feed it to ``gcr_sample`` or
            ``wiener_solve`` as ``noise_std=``.
        solution: the GLS point estimate at that covariance.
        residual: relative CG residual of the final solve. Not an accuracy.
        iterations: reweighting steps taken, the first solve included.
        delta: relative change of the last step.
        converged: whether ``delta`` fell below ``reweight_tol`` within
            ``max_reweights``. **False means the returned covariance is not a
            fixed point**, and everything conditioned on it inherits that.
    """

    noise_std: dict[str, jax.Array]
    solution: dict[str, jax.Array]
    residual: jax.Array
    iterations: jax.Array
    delta: jax.Array
    converged: jax.Array


def sigma_from_graph(
    graph: Graph, at: dict[str, Any]
) -> Callable[[dict[str, Any]], dict[str, jax.Array]]:
    """The ``{name: x} -> {observed: sigma}`` seam :func:`iterative_gls` iterates.

    Taking the seam as a callable rather than the graph keeps this module
    independent of the graph layer for testing, and lets P3b hand in a sigma
    frozen somewhere else without this module needing to know.
    """

    def sigma_of(x: dict[str, Any]) -> dict[str, jax.Array]:
        return noise_std_at(graph, {**at, **x})

    return sigma_of


def check_prediction_dependence(
    block: LinearBlock,
    sigma_of: Callable[[dict[str, Any]], dict[str, jax.Array]],
    *,
    declared: bool,
    rtol: float = 1e-8,
) -> float:
    """Measure how much sigma moves with the block, and check the declaration.

    ``depends_on_prediction`` is a **claim about the model**, like
    ``linear_in``: declared ``False``, a dispatcher skips the reweighting loop
    and solves at whatever sigma the prior mean implies -- a confident answer
    at the wrong covariance, with nothing to notice.

    Runs on concrete values, outside any trace, for the same reason
    :func:`~bayesmith.exact.gaussian.check_gaussian` does.

    Args:
        block: the block sigma might depend on.
        sigma_of: the seam, from :func:`sigma_from_graph`.
        declared: what the node claims.
        rtol: relative movement below which sigma counts as constant. Untested
            AT the boundary -- every fixture in this module's test suite sits
            far above it (a genuinely prediction-dependent sigma) or far
            below (a genuinely constant one), never within a decade of
            ``1e-8`` itself. Judged acceptable: this is a coarse yes/no
            movement detector guarding a declaration, not a numeric
            dispatcher choosing between two methods that must agree at a
            threshold, so there is no boundary-validation-style requirement
            that both sides produce the same answer there.

    Returns:
        The largest relative movement observed.

    Raises:
        StructureError: if ``declared`` is ``False`` and sigma does move.
            The opposite mismatch -- declared ``True``, sigma constant -- is
            merely conservative and is returned rather than raised, so a
            caller can report "the declaration is conservative; the
            reweighting loop could be skipped".
    """
    centre = domain_centre(block)
    baseline = sigma_of(centre)
    movement = 0.0
    for factor in DEPENDENCE_PROBES:
        probe = {
            name: centre[name] + factor * block.prior_std[name] for name in block.names
        }
        moved = sigma_of(probe)
        for observed, value in moved.items():
            scale = max(float(jnp.max(jnp.abs(baseline[observed]))), 1e-300)
            movement = max(
                movement,
                float(jnp.max(jnp.abs(value - baseline[observed]))) / scale,
            )
    if not declared and movement > rtol:
        raise StructureError(
            "a node declares depends_on_prediction=False, but moving the block by "
            f"one prior standard deviation moves sigma by {movement:.3e} relative "
            f"(rtol={rtol:.1e}). Declared False, the reweighting loop is skipped "
            "and the solve runs at whatever sigma the prior mean implies -- a "
            "confident answer at the wrong covariance. Drop the declaration, or "
            "make sigma genuinely independent of the prediction."
        )
    return movement


def iterative_gls(
    block: LinearBlock,
    sigma_of: Callable[[dict[str, Any]], dict[str, jax.Array]],
    *,
    depends_on_prediction: bool = True,
    tol: float = 1e-6,
    maxiter: int | None = None,
    reweight_tol: float | None = None,
    min_reweights: int = MIN_REWEIGHTS,
    max_reweights: int = MAX_REWEIGHTS,
    require_convergence: float | None = 1e-3,
) -> GLSResult:
    """Find the covariance a prediction-dependent noise model implies.

    Repeats: solve at the current sigma, recompute sigma at the new solution.
    With ``depends_on_prediction=False`` there is nothing to repeat and this is
    a single :func:`~bayesmith.exact.solve.wiener_solve`.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        sigma_of: the seam, from :func:`sigma_from_graph`.
        depends_on_prediction: the node's own claim. **Check it first** with
            :func:`check_prediction_dependence` -- this function cannot, being
            jittable.
        tol, maxiter: CG settings for each inner solve.
        reweight_tol: stop when the block's relative change falls below this.
            **The default cannot be a fixed number**, because two independent
            floors bound how small a step is measurable at all, so it defaults
            to ``max(8 * eps, tol)``:

            * the arithmetic's own epsilon -- a relative step below it is
              rounding, not a measurement. float32's is ``1.2e-7``, so a
              plausible-looking ``1e-8`` is exactly this trap;
            * **the inner solver's tolerance** ``tol`` -- consecutive solves
              differ by roughly their own CG residual no matter what the outer
              iteration is doing, so a step smaller than ``tol`` measures CG,
              not the fixed point. This is the binding floor in float64.

            Ask for less than either and the run does not fail loudly: it
            spends ``max_reweights`` steps and reports ``converged=False`` for
            a fixed point it had in fact reached.
        min_reweights: steps taken before the test is consulted.
        max_reweights: cap on steps.
        require_convergence: bound on the relative error of the **final**
            solve. Deliberately applied once, at the converged covariance, and
            not inside the loop: it bounds the error of what is returned and
            says nothing about the intermediate steps, which do not need it.

    Returns:
        A :class:`GLSResult`. **Check ``converged``** -- a covariance that is
        not a fixed point is still a number, and a draw conditioned on it is
        still a draw.

    Note:
        The iteration starts from sigma at the block's **prior mean** rather
        than from hydra-tod's unweighted least squares or rheplicant's sigma
        at the data. A graph's sigma is a function of the latents, not of a
        prediction array, so the data is not a point this seam can be
        evaluated at; the prior mean is the natural "before seeing anything"
        one. A fixed point does not depend on where the iteration started, so
        all three agree wherever any of them converges.

        Built on ``lax.while_loop``, so it is jittable but **not** reverse-mode
        differentiable. That is not the limitation it looks like: the result is
        a fixed point, so implicit differentiation -- not unrolling -- is the
        right way to take a gradient through it.
    """
    if not 1 <= min_reweights <= max_reweights:
        # GraphError is a misfit here -- this is a bad KEYWORD ARGUMENT to
        # iterative_gls, not a graph declared or evaluated inconsistently.
        # It behaves correctly (ValueError, which is what a caller doing
        # `except ValueError` around argument validation expects), so left
        # as is rather than churned this late; a dedicated argument-error
        # class is deferred rather than added for this one call site.
        raise GraphError(
            f"iterative_gls needs 1 <= min_reweights <= max_reweights, got "
            f"{min_reweights} and {max_reweights}. The loop caps at "
            "max_reweights either way, so this configuration would silently get "
            "fewer steps than it asked for."
        )
    if reweight_tol is None:
        epsilon = float(jnp.finfo(jnp.result_type(*jax.tree.leaves(block.offset))).eps)
        reweight_tol = max(REWEIGHT_TOL_EPS * epsilon, tol)

    def solve_at(sigma, guard):
        return wiener_solve(
            block, noise_std=sigma, tol=tol, maxiter=maxiter, require_convergence=guard
        )

    centre = domain_centre(block)

    if not depends_on_prediction:
        sigma = sigma_of(centre)
        solution, residual = solve_at(sigma, require_convergence)
        return GLSResult(
            noise_std=sigma,
            solution=solution,
            residual=residual,
            iterations=jnp.asarray(1),
            delta=jnp.asarray(0.0),
            converged=jnp.asarray(True),
        )

    def step(carry):
        count, latent, _ = carry
        updated, _ = solve_at(sigma_of(latent), None)
        change = jax.tree.map(jnp.subtract, updated, latent)
        # Relative to the NEW iterate: relative to the old one, a step that
        # starts near zero reports a huge change forever. The rationale is
        # sound but, at the default MIN_REWEIGHTS=5, unguarded by any fixture
        # this module ships: measured by swapping the denominator to
        # tree_norm(latent) and sweeping radiometer() over kappa
        # 0.001-60 (its full convergent range, up to the onset of
        # divergence) and weight down to 0.001 -- the two normalisations
        # agree to 10+ significant digits by the time delta is first
        # consulted at count=5, because this model's IRLS map stabilises the
        # ITERATE's magnitude within 1-2 steps even while the CHANGE keeps
        # shrinking for longer. At min_reweights=1, where delta is consulted
        # after a single step, the two DO disagree at radiometer()'s exact
        # defaults (kappa=3.5, seed=6, prior_mean=0.0) -- but this is a
        # POINT separation, not a region, and depends on all three: swept
        # seed 0-19 and only 6 preserve the split at reweight_tol=0.75
        # (several invert it outright, e.g. seed=8: correct delta=1.14 vs
        # mutated=8.00); swept prior_mean and found ONLY the exact default
        # 0.0 works (0.01 already breaks it), because a zero-mean prior
        # makes the prediction AT the prior mean exactly zero, collapsing
        # the first sigma estimate to a uniform floor -- a degenerate warm
        # start unique to that one value. See
        # test_iterative_gls_delta_denominator_uses_the_new_iterate for the
        # full sweep and the numbers this specific pin relies on.
        delta = tree_norm(change) / jnp.maximum(tree_norm(updated), 1e-30)
        return count + 1, updated, delta

    def unfinished(carry):
        count, _, delta = carry
        # max_reweights is the OUTER conjunct, so it caps the loop whatever
        # min_reweights says. Written the other way round -- keep going while
        # below the minimum OR not yet converged -- a min above the max never
        # terminates, and an infinite lax.while_loop under jit cannot be
        # interrupted.
        return jnp.logical_and(
            count < max_reweights,
            jnp.logical_or(count < min_reweights, delta > reweight_tol),
        )

    first, _ = solve_at(sigma_of(centre), None)
    count, latent, delta = lax.while_loop(
        unfinished, step, (jnp.asarray(1), first, jnp.asarray(jnp.inf))
    )

    # One final solve at the converged covariance, and the only place the
    # conditioning guard runs -- so what it certifies is what is returned.
    sigma = sigma_of(latent)
    solution, residual = solve_at(sigma, require_convergence)
    return GLSResult(
        noise_std=sigma,
        solution=solution,
        residual=residual,
        iterations=count,
        delta=delta,
        converged=delta <= reweight_tol,
    )
