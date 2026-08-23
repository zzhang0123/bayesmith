"""Threshold behaviour, checked by bypassing the dispatch and testing both sides.

`boundary-validation.md`: for a dispatcher that routes on a threshold T, the
useful check is not "method A gives the right answer at point P" but "A and B
agree at T". Every threshold P3a introduces is swept here, at values on both
sides of it, and each sweep asserts what the OTHER side does too -- a test
that only confirms the failing side would pass against a check that always
fails.

**Every test below states whether its accept/refuse separation is a POINT or
a REGION** -- whether the specific values chosen are the only ones that work,
or whether the property holds across a swept range of the "should not
matter" fixture dimensions (seed, n, magnitude of an unrelated blow-up). See
`boundary-validation.md`'s own recorded lesson: a separation that holds at
exactly one point and is documented as if it were general silently stops
discriminating the day someone changes an unrelated default.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import evaluate, observe, sample, trace
from bayesmith.errors import StructureError
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.gaussian import check_gaussian, noise_std_at
from bayesmith.exact.gls import iterative_gls, sigma_from_graph
from bayesmith.exact.linearity import check_linearity
from bayesmith.exact.solve import condition_bound, wiener_solve
from tests.exact.models import radiometer_group, tunable_curvature, two_linear_latents

# --------------------------------------------------------------------------
# check_linearity's rtol threshold (affine vs. not)
# --------------------------------------------------------------------------

# **Measured correction to the plan's own draft values.** The plan text this
# file implements proposed accepting departure in {0.0, 1e-8, 1e-6}. Run
# directly: departure=1e-6 is REFUSED, not accepted -- at DEFAULT_SCALES's
# top probe (1e3 * prior_std), the quadratic term's relative contribution is
# ~departure * 1e3, which at 1e-6 already sits ~1.4x past
# rtol=1e4*eps(float32)~1.19e-3. Bisecting: accepted up to departure=5e-7,
# refused from departure=7e-7. 1e-7 replaces 1e-6 below, comfortably (~7x
# margin) inside the accept side rather than just past the threshold; this
# is exactly the "diagnose by computing what the assertion actually reports,
# not by intuition" move `boundary-validation.md`'s own five disciplines
# call for.


@pytest.mark.parametrize("departure", [0.0, 1e-8, 1e-7])
def test_check_linearity_accepts_below_its_rtol(departure):
    """The quiet side of the threshold: a departure this small is roundoff.

    REGION, not point: check_linearity's affinity probe never reads sigma,
    n or the data -- only the deterministic node `mu` and each member's own
    prior_std, which `tunable_curvature`'s `w**2 / prior_std` term is built
    to cancel out of `departure`'s effect by design (see its docstring). So
    this separation is independent of `tunable_curvature`'s other keyword
    defaults; it depends only on `DEFAULT_SCALES` (the probe magnitudes) and
    the working dtype's `rtol`, both fixed here.
    """
    graph = tunable_curvature(departure=departure)
    check_linearity(graph, ("w",), at={}, at_points=[{}])


@pytest.mark.parametrize("departure", [1e-6, 1e-2, 1.0, 10.0])
def test_check_linearity_refuses_above_its_rtol(departure):
    """REGION: same independence from sigma/n/seed as the accept side above."""
    graph = tunable_curvature(departure=departure)
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("w",), at={}, at_points=[{}])


def test_the_roundoff_floor_does_not_reject_a_perfectly_linear_block():
    """The small-probe end, where the relative measure would explode.

    scales down to 1e-9 of the prior width: the variation there is vanishing
    but roundoff is not, so without the per-probe floor the relative departure
    blows up and a genuinely linear block is refused.

    Not a two-method boundary in the `boundary-validation.md` sense -- there
    is one method (the affinity probe), not two competing ones, and the
    floor is a roundoff guard rather than a dispatch threshold. Kept
    one-sided on purpose: it exists to confirm the floor does not produce a
    FALSE refusal, which only has an "accept" side to check.
    """
    graph = tunable_curvature(departure=0.0)
    check_linearity(
        graph, ("w",), at={}, at_points=[{}], scales=(1e-9, 1e-6, 1e-3, 1.0)
    )


# --------------------------------------------------------------------------
# check_gaussian's log_prob-agreement threshold
# --------------------------------------------------------------------------


def _probe_graph(distribution_fn):
    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", distribution_fn, w, obs=jnp.zeros(3))

    return trace(model)


# **Strengthened from the plan's draft values.** The plan proposed accepting
# scaling in {1.0, 1.0 + 1e-12, 1.0 - 1e-12}. Measured: at float32, a Python
# float multiplying a float32 array is weakly-typed, and 1e-12 is three
# orders of magnitude below float32 epsilon (1.19e-7) -- so
# `(1.0 + 1e-12) * log_prob` rounds to bit-identical to `1.0 * log_prob`
# before the probe ever runs. Those two parametrizations therefore exercise
# nothing that `scaling=1.0` does not already cover. Replaced with
# `1.0 +/- 1e-4`: measured, the worst-entry departure there is ~1.0e-4,
# against rtol=1e3*eps(float32)~1.19e-4 -- a genuine ~1.19x margin on the
# accept side, found by bisection (1.0e-4 accepted, 2.0e-4 refused).
@pytest.mark.parametrize("scaling", [1.0, 1.0 + 1e-4, 1.0 - 1e-4])
def test_the_gaussian_probe_accepts_a_log_prob_that_agrees(scaling):
    """The ACCEPT side of the probe's rtol, which is the half easy to forget.

    A guard that always raised would pass every refusal test in this suite;
    only this one distinguishes it from a working guard.

    REGION for the exact-agreement case (`scaling=1.0` passes for any probe
    offset or width); POINT-ish for the two off-1.0 cases in that they sit
    close enough to rtol to matter only because rtol itself is a function of
    the working dtype -- run this at float64 (inside `jax.enable_x64`) and a
    1e-4 relative departure would be refused instead, since rtol would then
    be ~1e3x tighter. That dtype-dependence is the point being tested, not
    an accident of this fixture.
    """

    class NearlyExact(dist.Normal):
        def log_prob(self, value):
            return scaling * super().log_prob(value)

    graph = _probe_graph(lambda w_: NearlyExact(w_, 0.7))
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    check_gaussian(graph, graph.node("d"), env)


@pytest.mark.parametrize("scaling", [1.0002, 1.01, 1.5, 0.5])
def test_the_gaussian_probe_refuses_a_log_prob_that_does_not(scaling):
    """REFUSE side, from just past the measured crossover (1.0002) to far past it."""

    class Off(dist.Normal):
        def log_prob(self, value):
            return scaling * super().log_prob(value)

    graph = _probe_graph(lambda w_: Off(w_, 0.7))
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    with pytest.raises(StructureError, match="log_prob"):
        check_gaussian(graph, graph.node("d"), env)


# --------------------------------------------------------------------------
# wiener_solve's require_convergence guard: "bad" (converged vs not)
# --------------------------------------------------------------------------


def test_the_convergence_guard_flips_at_require_convergence_over_kappa():
    """tol just above and just below the value the guard's own algebra implies.

    Bypasses the guard to compute kappa first, then asks for a tol on each
    side of `require_convergence / kappa`. Both sides are asserted: a guard
    that always raised would pass the strict half alone.

    REGION, by construction: `require_convergence` is derived from THIS
    run's own measured `bound` and `residual` (bound/2, bound*2) rather than
    a hardcoded literal, so the separation does not depend on the exact
    numeric value `condition_bound` happens to return -- only on `bound *
    2.0` and `bound / 2.0` landing on opposite sides of `residual * bound`,
    which they do for any positive `bound`. This mirrors the fix
    Task 6's own "an impossible target" note describes for a sibling test.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = unchecked_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        kappa = float(condition_bound(block, noise_std=sigma, iterations=80))

        # CG on a 2-parameter system reaches machine precision in 2 steps, so
        # maxiter -- not tol -- is what sets the residual here. Measure the
        # bound the guard will compute, with the guard itself switched off.
        _, residual = wiener_solve(
            block, noise_std=sigma, tol=1e-14, maxiter=1, require_convergence=None
        )
        bound = float(residual) * kappa
        assert bound > 0.0, "fixture no longer leaves a measurable residual"

        # Strict side: ask for half the bound the guard will find.
        with pytest.raises(Exception, match="did not converge"):
            wiener_solve(
                block,
                noise_std=sigma,
                tol=1e-14,
                maxiter=1,
                require_convergence=bound / 2.0,
            )
        # Permissive side: ask for twice it, and the SAME solve is accepted.
        # Asserting only the strict half would pass against a guard that
        # always raised.
        wiener_solve(
            block,
            noise_std=sigma,
            tol=1e-14,
            maxiter=1,
            require_convergence=bound * 2.0,
        )


# --------------------------------------------------------------------------
# wiener_solve's `unreachable` guard: WHICH message fires, not just whether
# --------------------------------------------------------------------------


def test_the_unreachable_branch_flips_at_bound_times_eps_over_require_convergence():
    """Two-sided, same fixture: which error message fires either side of
    ``bound * eps`` vs. ``require_convergence``.

    The plan text this file implements originally proposed a ONE-SIDED
    version of this test: build a starved (float32, `b`'s prior widened to
    1e4) block, confirm `kappa * eps > 1e-3`, and check only that the
    "enable_x64" message fires at the DEFAULT `require_convergence=1e-3`.
    That is almost a duplicate of `test_the_guard_points_at_enable_x64_in_
    float32` already in `tests/exact/test_solve.py` (same fixture, same
    mutation, same default target) and, per `boundary-validation.md`,
    checking only one side of a dispatch is exactly the gap that check
    exists to close: a guard that always raised "enable_x64" -- never "did
    not converge" -- would pass a one-sided version of this test.

    Rewritten to bracket the SAME threshold the `unreachable` clause tests
    (`bound * epsilon` vs. `require_convergence`) from both sides, on the
    SAME solve, with only `require_convergence` differing -- exactly the
    pattern `test_the_convergence_guard_flips_at_require_convergence_over_
    kappa` uses for the sibling `bad` threshold, applied here to the
    threshold that picks the ERROR MESSAGE once `bad` is already True.

    REGION, measured: `bound`, `epsilon` and `residual` are all measured
    from this run (not hardcoded), and the window between `bound * epsilon`
    (~1.4e3) and `residual * bound` (~2.7e9) spans ~6 orders of magnitude at
    `maxiter=1` -- robust across the prior blow-up magnitude (checked at
    1e3, 1e4, 1e5: same ~1.9e6x window) and across `two_linear_latents`'s
    seed (checked at the default and at seed=99: same window to 3
    significant figures). The one genuine POINT-like dependency is
    `maxiter=1` itself: at `maxiter=2` this 2-parameter block's CG already
    reaches machine precision (a documented property of CG on an
    n-parameter SPD system, elsewhere in this package's own tests), which
    collapses the window to a ratio of ~2 -- not a numeric coincidence, but
    a structural fact about this fixture's dimension that a reader changing
    `maxiter` should know breaks the separation.
    """
    graph = two_linear_latents()
    block = unchecked_operator(graph, ("a", "b"), at={})
    sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
    starved = dataclasses.replace(
        block, prior_std={**block.prior_std, "b": jnp.asarray(1e4, dtype=jnp.float32)}
    )
    kappa = float(condition_bound(starved, noise_std=sigma, iterations=80))
    epsilon = float(jnp.finfo(jnp.float32).eps)
    beta = kappa * epsilon

    _, residual = wiener_solve(
        starved, noise_std=sigma, tol=1e-14, maxiter=1, require_convergence=None
    )
    error_bound = float(residual) * kappa
    assert beta < error_bound, "fixture no longer leaves a bound*eps < error_bound gap"

    # Unreachable side: require_convergence below bound*eps. bad=True (far
    # below error_bound too), and the precision message names enable_x64.
    with pytest.raises(Exception, match="enable_x64"):
        wiener_solve(
            starved,
            noise_std=sigma,
            tol=1e-14,
            maxiter=1,
            require_convergence=beta / 2.0,
        )
    # Reachable side: require_convergence above bound*eps but still below
    # error_bound. bad=True (unconverged), unreachable=False (precision is
    # not the obstacle) -- the OTHER message, naming tol/maxiter, fires.
    with pytest.raises(Exception, match="did not converge"):
        wiener_solve(
            starved,
            noise_std=sigma,
            tol=1e-14,
            maxiter=1,
            require_convergence=beta * 2.0,
        )


def test_asking_for_a_reweight_tol_below_the_floor_reports_not_converged():
    """The trap the default exists to avoid, demonstrated on both sides.

    A reweight_tol below the reachable precision measures the iterate's own
    float64 noise floor rather than the fixed point: the run spends
    max_reweights steps and reports converged=False for a fixed point it did
    reach. The same model with the default (computed) tolerance converges.

    **Measured correction to the plan's own draft fixture.** The plan text
    this file implements used `radiometer()` -- whose block is a SCALAR
    latent (`w`), so CG solves its 1x1 normal equations exactly in one step
    regardless of `tol`. Run directly: with `reweight_tol=1e-16`, delta hits
    EXACTLY `0.0` (bit-identical consecutive iterates) after only 7 steps,
    so `converged` reads True -- the claimed trap never fires, for any
    `reweight_tol >= 0`, on that fixture. Swapped for `radiometer_group()`,
    whose block is 2-dimensional (`a`, `b`): there, consecutive iterates
    still agree to ~10 significant figures but the last couple of bits keep
    moving at the ~1e-15-1e-14 scale run over run (measured: honest delta
    3.06e-11 at convergence: `reweight_tol=1e-16` sits below where THAT
    noise floor bottoms out over 40 steps -- measured minimum delta reached,
    2.17e-15 -- so it is never satisfied and `converged=False`, while the
    solution itself agrees with the honest run to 10+ significant figures).

    REGION for the qualitative claim (any multi-dimensional block's
    reweighting map has a float64 noise floor no `reweight_tol` below
    ~epsilon scale can detect -- this is exactly what
    `REWEIGHT_TOL_EPS * eps` in `iterative_gls`'s own default guards
    against) demonstrated with a POINT-specific pair of numbers
    (`reweight_tol=1e-16`, `max_reweights=40`) that is not re-swept here
    against other kappa/seed choices the way `tests/exact/test_solve.py`
    sweeps `condition_bound`'s own tests. The one thing that IS load-bearing
    and worth naming: the block must be multi-dimensional. A scalar block
    (like `radiometer()`) converges bit-exactly in too few steps for this
    trap to be observable at all.
    """
    with jax.enable_x64(True):
        graph = radiometer_group()
        block = unchecked_operator(graph, ("a", "b"), at={})
        seam = sigma_from_graph(graph, {})
        honest = iterative_gls(block, seam, tol=1e-10, max_reweights=40)
        starved = iterative_gls(
            block, seam, tol=1e-10, reweight_tol=1e-16, max_reweights=40
        )
    assert bool(honest.converged)
    assert not bool(starved.converged)
    # And the answers agree: the fixture reached the fixed point either way.
    for name in block.names:
        assert np.allclose(
            np.asarray(honest.solution[name]),
            np.asarray(starved.solution[name]),
            rtol=1e-6,
        )
