"""G2 -- ``fit``: the gradient MAP exit an exact solve does not have.

Every oracle here is INDEPENDENT of the thing it checks, which for an
optimiser is unusually easy to arrange and unusually easy to skip: the answer
a descent arrives at has closed forms for the models the exact route also
covers, and those closed forms are computed by a different algorithm
(``wiener_solve``) or by no algorithm at all (numpy arithmetic on the
conjugate formula). Where the model is outside the exact route -- which is the
entire reason this exit exists -- the oracle is a brute-force scan of the true
objective over a grid, which differentiates nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.errors import StructureError
from bayesmith.graph.evaluate import log_joint
from bayesmith.optimize import (
    MAXIMIZE,
    MINIMIZE,
    Fit,
    check_loss_sense,
    fit,
    minimize,
    sense_of,
)
from tests.exact.models import radiometer, straight_line, two_linear_latents


class TestAgainstTheClosedFormPosterior:
    """A conjugate Gaussian model's MAP is its posterior mean, and both have
    closed forms this module does not use."""

    def test_one_latent_lands_on_the_conjugate_formula(self):
        """The oracle is arithmetic: ``(P0 mu0 + sum x d / s^2) / (P0 + sum
        x^2 / s^2)``, in numpy, differentiating nothing."""
        n, weight, sigma, prior_std = 8, 2.5, 0.5, 2.0
        graph = straight_line(
            n=n, weight=weight, sigma=sigma, prior_std=prior_std, prior_mean=0.0
        )
        x = np.asarray(graph.node("X").value)
        d = np.asarray(graph.node("d").observed)
        precision = 1.0 / prior_std**2 + float((x @ x) / sigma**2)
        expected = float((x @ d) / sigma**2) / precision

        found = fit(graph, steps=4000, learning_rate=0.02)
        assert float(found.values["w"]) == pytest.approx(expected, rel=1e-4)

    def test_two_latents_land_where_wiener_solve_puts_them(self):
        """The oracle is a DIFFERENT ALGORITHM: a direct linear solve of the
        normal equations against an iterative descent. They share the graph
        and nothing else.
        """
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator
        from bayesmith.exact.solve import wiener_solve

        graph = two_linear_latents()
        with jax.enable_x64(True):
            block = linear_operator(graph, ("a", "b"), at={})
            centres = prior_environment(graph)
            exact, _ = wiener_solve(
                block,
                # Latents only: `precision_at` refuses a computed node's name,
                # and `prior_environment` returns every node's value.
                precision=precision_at(
                    graph, {name: centres[name] for name in graph.latents}
                ),
            )
            found = fit(graph, steps=8000, learning_rate=0.01)
        for name in ("a", "b"):
            assert float(found.values[name]) == pytest.approx(
                float(exact[name]), rel=1e-3
            )

    def test_it_is_not_just_returning_the_prior_centre(self):
        """Anti-vacuity for both tests above.

        A ``fit`` that took no steps at all would return the prior centre, and
        on a model where the data agrees with the prior that is nearly the
        right answer. Here the prior is centred at 0.0 and the truth is 2.5,
        so "did not move" and "converged" are four sigma apart.
        """
        graph = straight_line(weight=2.5, prior_mean=0.0)
        found = fit(graph, steps=4000, learning_rate=0.02)
        assert abs(float(found.values["w"])) > 1.0


class TestTheFullDensityIsTheTarget:
    """D7: both gradient exits target the FULL density, ``sum log sigma``
    included. Under a prediction-dependent sigma that is a different point
    from the GLS-flavoured optimum, and the difference is what this class
    measures."""

    @staticmethod
    def _grid_argmin(graph, name, lo, hi, n=20001):
        """The objective's minimiser by BRUTE FORCE -- no gradient anywhere.

        This is the oracle for a model the exact route cannot solve: evaluate
        the true objective on a dense grid and take the smallest. It cannot
        share a bug with a descent because it does not descend.
        """
        grid = jnp.linspace(lo, hi, n)
        values = [
            float(-log_joint(graph, {name: jnp.asarray(g)})) for g in np.asarray(grid)
        ]
        return float(grid[int(np.argmin(np.asarray(values)))])

    def test_the_fit_lands_on_the_grid_minimum(self):
        graph = radiometer(kappa=0.3, floor=1e-3)
        best = self._grid_argmin(graph, "w", 1.0, 5.0)
        found = fit(graph, {"w": jnp.asarray(1.0)}, steps=4000, learning_rate=0.02)
        assert float(found.values["w"]) == pytest.approx(best, abs=2e-3)

    def test_the_prior_centre_is_a_cliff_on_this_model_and_at_is_the_remedy(self):
        """The starting point is load-bearing, so it is measured rather than
        left for someone to rediscover.

        ``sigma = 0.3|mu| + 1e-3``, so at ``w = 0`` -- the prior centre, and
        `fit`'s default -- sigma is the floor while the residuals are of order
        9, and the objective is **4.3e8** against 24.2 at the optimum. There
        is no line search here: 6000 Adam steps starting there travel 0.08 and
        arrive nowhere, with a loss history that decreases monotonically the
        whole way. Starting at ``w = 1`` arrives in 4000.

        This is not a defect to fix by making the optimiser cleverer; it is
        what `at=` is for, and a caller who has a solve or a previous sweep to
        start from should use it.
        """
        graph = radiometer(kappa=0.3, floor=1e-3)
        best = self._grid_argmin(graph, "w", 1.0, 5.0)
        from_centre = fit(graph, steps=6000, learning_rate=0.01)
        assert abs(float(from_centre.values["w"]) - best) > 1.0
        # ... and it looks like a healthy fit the whole way down.
        history = np.asarray(from_centre.history)
        assert np.all(np.diff(history) < 0.0)

    def test_the_full_density_optimum_is_not_the_gls_one(self):
        """The anti-vacuity clause, and the whole of D7.

        Without it both targets could be the same point and the test above
        would be green on the version that drops ``sum log sigma``. Measured:
        under a fractional sigma the log-determinant term pulls the optimum
        toward smaller predictions, and the two differ well outside the
        tolerance the test above uses.
        """
        from bayesmith.exact.gls import sigma_from_graph

        graph = radiometer(kappa=0.3, floor=1e-3)
        sigma_of = sigma_from_graph(graph, {})

        def gls_only(w):
            """The same objective with the log-determinant term REMOVED.

            Derived by subtraction rather than re-spelled: for a diagonal
            Gaussian, ``-log_joint`` carries ``sum log sigma``, so taking it
            back off leaves the GLS-flavoured potential. Writing the chi-square
            out again here would let a typo in this file masquerade as the very
            difference the test is measuring.
            """
            values = {"w": jnp.asarray(w)}
            full = float(-log_joint(graph, values))
            return full - float(jnp.sum(jnp.log(sigma_of(values)["d"])))

        grid = np.linspace(1.0, 5.0, 4001)
        gls_best = float(grid[int(np.argmin([gls_only(w) for w in grid]))])
        full_best = self._grid_argmin(graph, "w", 1.0, 5.0)
        assert abs(gls_best - full_best) > 1e-2, (gls_best, full_best)


class TestBlockCoordinate:
    """``names=`` moves a subset and HOLDS the rest -- what a gradient block
    inside a Gibbs sweep needs, and what D11 calls block coordinate."""

    def test_the_held_latents_come_back_untouched(self):
        graph = two_linear_latents()
        start = {"a": jnp.asarray(0.3), "b": jnp.asarray(-0.7)}
        found = fit(graph, start, names=("a",), steps=2000, learning_rate=0.01)
        assert float(found.values["b"]) == float(start["b"])
        assert float(found.values["a"]) != float(start["a"])

    def test_the_moved_latent_lands_on_its_conditional_optimum(self):
        """Oracle: a grid scan of the same conditional, no gradient."""
        graph = two_linear_latents()
        held = jnp.asarray(-0.7)
        grid = np.linspace(-3.0, 6.0, 40001)
        objective = [
            float(-log_joint(graph, {"a": jnp.asarray(a), "b": held})) for a in grid
        ]
        best = float(grid[int(np.argmin(objective))])
        found = fit(
            graph,
            {"a": jnp.asarray(0.3), "b": held},
            names=("a",),
            steps=4000,
            learning_rate=0.02,
        )
        assert float(found.values["a"]) == pytest.approx(best, abs=2e-3)

    def test_holding_one_is_a_different_answer_from_moving_both(self):
        """Anti-vacuity, and it needs a CORRELATED model to be able to fail.

        Measured on `two_linear_latents` first: its design is
        ``linspace(-2, 2)``, so ``sum x = 0``, the two latents are exactly
        orthogonal, and holding one changes the other's optimum by 1.2e-07.
        The clause was vacuous there and said so. `collinear_pair` is the
        opposite extreme -- the data fixes ``a + b`` and nothing else -- so
        what `b` is held at moves `a` one for one.
        """
        from tests.exact.models import collinear_pair

        graph = collinear_pair()
        start = {"a": jnp.asarray(0.3), "b": jnp.asarray(-0.7)}
        one = fit(graph, start, names=("a",), steps=4000, learning_rate=0.02)
        both = fit(graph, start, steps=4000, learning_rate=0.02)
        assert abs(float(one.values["a"]) - float(both.values["a"])) > 1e-2


class TestPerLatentStepSizes:
    """One global step size cannot serve two latents whose units differ by
    orders of magnitude -- the rate that moves one diverges on the other."""

    @staticmethod
    def _lopsided():
        """``d ~ N(big * 1e-4 + small, 0.1)`` -- two latents, 1e4 apart."""
        x = jnp.ones(6)
        data = jnp.full((6,), 2.0)

        def model():
            xs = const("X", x)
            big = sample("big", lambda: dist.Normal(0.0, 1e4))
            small = sample("small", lambda: dist.Normal(0.0, 1e-2))
            mu = det(
                "mu",
                lambda b, s, x_: (1e-4 * b + s) * x_,
                big, small, xs,
                linear_in=("big", "small"),
            )
            observe("d", lambda m: dist.Normal(m, 0.1), mu, obs=data)

        return trace(model)

    def test_per_latent_rates_reach_what_a_single_rate_does_not(self):
        graph = self._lopsided()
        start = {"big": jnp.asarray(0.0), "small": jnp.asarray(0.0)}
        flat = fit(graph, start, steps=3000, learning_rate=1e-2)
        scaled = fit(
            graph, start, steps=3000,
            step_sizes={"big": 1e2, "small": 1e-4},
        )
        assert float(scaled.objective) < float(flat.objective)

    def test_a_step_size_for_a_latent_that_is_not_moving_is_refused(self):
        graph = two_linear_latents()
        with pytest.raises(StructureError, match="step_sizes"):
            fit(graph, names=("a",), step_sizes={"b": 0.1}, steps=10)


class TestTheHistoryContract:
    def test_the_history_starts_where_the_caller_did_and_ends_lower(self):
        graph = straight_line()
        found = fit(graph, steps=500, learning_rate=0.02)
        start = float(-log_joint(graph, {"w": jnp.asarray(0.0)}))
        assert found.history.shape == (500,)
        assert float(found.history[0]) == pytest.approx(start, rel=1e-6)
        assert float(found.objective) < float(found.history[0])

    def test_the_reported_objective_is_at_the_reported_point(self):
        """The pair a caller compares, so they must be the same point.

        A history's last entry is the value BEFORE the last step, and
        returning it as `objective` alongside the point AFTER that step is the
        kind of off-by-one that reads as a converged fit.
        """
        graph = straight_line()
        # Few enough steps that the fit is still moving: at convergence the
        # last two values agree to float32 and the off-by-one is invisible.
        found = fit(graph, steps=50, learning_rate=0.02)
        assert float(found.objective) == pytest.approx(
            float(-log_joint(graph, found.values)), rel=1e-6
        )
        assert float(found.objective) != float(found.history[-1])


class TestBothMethods:
    def test_adam_and_plain_gradient_reach_the_same_optimum(self):
        """Each at a rate it is stable at -- see the test below for why those
        are not the same number."""
        graph = straight_line()
        a = fit(graph, method="adam", steps=4000, learning_rate=0.02)
        g = fit(graph, method="gradient", steps=4000, learning_rate=0.006)
        assert float(a.values["w"]) == pytest.approx(float(g.values["w"]), rel=1e-4)

    def test_plain_gradient_above_its_stability_limit_is_refused_not_returned(self):
        """The methods do not accept the same rates, and the difference is
        arithmetic rather than taste.

        Plain gradient descent diverges for any rate above ``2/L``, where
        ``L`` is the objective's curvature; Adam's step is bounded by its rate
        regardless. Measured on this model: ``L = 231.1``, limit **0.00865**,
        so 0.006 converges and 0.02 gives NaN -- while Adam converges at
        either. A `method=` switched without revisiting `learning_rate=` lands
        exactly here, so the NaN is refused rather than handed back as a fit.
        """
        graph = straight_line()
        x = np.asarray(graph.node("X").value)
        limit = 2.0 / (1.0 / 2.0**2 + float(x @ x) / 0.5**2)
        assert 0.006 < limit < 0.02, limit
        with pytest.raises(Exception, match="not finite"):
            fit(graph, method="gradient", steps=4000, learning_rate=0.02)
        # Adam at the same rate is fine, which is what makes the switch a trap.
        assert jnp.isfinite(
            fit(graph, method="adam", steps=4000, learning_rate=0.02).objective
        )

    def test_an_unknown_method_is_refused_by_name(self):
        with pytest.raises(StructureError, match="method"):
            fit(straight_line(), method="lbfgs", steps=10)


class TestMinimizeOnItsOwn:
    """The optimiser is public separately from the graph entry, because Wave
    C's calibrator scores a PREDICTION against DATA rather than a joint."""

    def test_a_quadratic_reaches_its_analytic_minimum(self):
        found = minimize(
            lambda v: jnp.sum((v["x"] - 3.0) ** 2) + (v["y"] + 1.0) ** 2,
            {"x": jnp.asarray(0.0), "y": jnp.asarray(0.0)},
            steps=3000,
            learning_rate=0.01,
        )
        assert float(found.values["x"]) == pytest.approx(3.0, abs=1e-4)
        assert float(found.values["y"]) == pytest.approx(-1.0, abs=1e-4)

    def test_it_returns_a_Fit(self):
        found = minimize(
            lambda v: jnp.sum(v["x"] ** 2), {"x": jnp.zeros(3)}, steps=5
        )
        assert isinstance(found, Fit)


class TestTheLossSenseGuard:
    """A log-density has an error's signature and the opposite optimum. A
    minimiser handed one descends a function unbounded below while the loss
    history looks like textbook convergence."""

    def test_a_declared_maximiser_is_refused(self):
        class Likelihood:
            sense = MAXIMIZE

            def __call__(self, predicted, observed):
                return -0.5 * jnp.sum((predicted - observed) ** 2)

        with pytest.raises(StructureError, match="maximize"):
            check_loss_sense(Likelihood(), jnp.zeros(3), jnp.ones(3))

    def test_an_undeclared_maximiser_is_caught_by_measurement(self):
        """The half that matters: a bare lambda declares nothing, and a
        whitelist is wrong about exactly the code it has not met."""
        with pytest.raises(StructureError, match="PERFECT"):
            check_loss_sense(
                lambda p, o: -jnp.sum((p - o) ** 2), jnp.zeros(3), jnp.ones(3)
            )

    def test_an_ordinary_error_passes_both_halves(self):
        check_loss_sense(
            lambda p, o: jnp.sum((p - o) ** 2), jnp.zeros(3), jnp.ones(3)
        )

    def test_a_non_finite_score_is_refused_rather_than_waved_through(self):
        """NaN compares False against everything, so "cannot tell, proceed"
        would let the case this guard exists for through whenever it arrives
        with a NaN attached."""
        with pytest.raises(StructureError, match="finite"):
            check_loss_sense(
                lambda p, o: jnp.asarray(jnp.nan), jnp.zeros(3), jnp.ones(3)
            )

    def test_sense_of_reads_a_declaration_and_defaults_to_minimize(self):
        class Declared:
            sense = MAXIMIZE

        assert sense_of(Declared()) == MAXIMIZE
        assert sense_of(lambda p, o: p) == MINIMIZE


class TestRefusals:
    def test_a_name_the_graph_does_not_declare_is_refused(self):
        with pytest.raises(StructureError, match="nope"):
            fit(straight_line(), names=("nope",), steps=10)

    def test_a_non_positive_step_count_is_refused(self):
        with pytest.raises(StructureError, match="steps"):
            fit(straight_line(), steps=0)

    def test_a_nan_learning_rate_is_refused(self):
        """`not > 0` rather than `<= 0`, so a NaN rate is refused too -- a NaN
        compares False against everything and would sail through the obvious
        spelling."""
        with pytest.raises(StructureError, match="learning_rate"):
            fit(straight_line(), steps=10, learning_rate=float("nan"))

    def test_an_empty_name_tuple_is_refused(self):
        with pytest.raises(StructureError, match="at least one"):
            fit(straight_line(), names=(), steps=10)

    def test_a_complex_latent_is_refused_by_name(self):
        """`jax.grad` of a real objective at a complex point returns the
        CONJUGATE gradient, so a descent using it walks the wrong way without
        erroring. Refused rather than stepped."""
        with pytest.raises(StructureError, match="complex"):
            minimize(
                lambda v: jnp.real(jnp.sum(v["z"] * jnp.conj(v["z"]))),
                {"z": jnp.asarray(1.0 + 1.0j)},
                steps=10,
            )


class TestAdamsBetasAreRefusedOutsideTheirRange:
    """`[0, 1)` is what an exponential decay rate MEANS, and the loss is silent.

    This guard exists because of what the failure looks like, not because the
    range is written down somewhere. Measured on ``(x - 3)**2`` from ``x = 0``,
    200 steps at rate 0.1: ``beta1=1.5`` returns **15.384941** -- finite, with
    no warning, five times the true minimum. Nothing downstream can tell that
    from a fit.

    The other out-of-range values on that fixture happen to trip
    :func:`minimize`'s divergence guard instead, and ``beta2=1.5`` happens to
    return 2.99994 and look perfect. That spread is the fixture's luck rather
    than a second guard, which is exactly why the range is checked at entry.

    Discovered while opening rheplicant's ``calibrate`` row: its
    ``AdamCalibrator`` refuses this at construction and this side did not, so
    the refusal had no home to be delegated to. Migration ledger D57.
    """

    def _quadratic(self):
        return (lambda p: (p["x"] - 3.0) ** 2), {"x": jnp.array(0.0)}

    @pytest.mark.parametrize("name", ["beta1", "beta2"])
    @pytest.mark.parametrize("bad", [1.0, 1.5, -0.5, float("nan")])
    def test_out_of_range_is_refused(self, name, bad):
        obj, at = self._quadratic()
        with pytest.raises(StructureError, match=f"{name} must be in "):
            minimize(obj, at, method="adam", steps=10, learning_rate=0.1, **{name: bad})

    @pytest.mark.parametrize("name", ["beta1", "beta2"])
    @pytest.mark.parametrize("ok", [0.0, 0.5, 0.9999])
    def test_the_range_itself_is_accepted(self, name, ok):
        """The anti-vacuity twin: a guard that refused everything would pass above."""
        obj, at = self._quadratic()
        out = minimize(obj, at, method="adam", steps=50, learning_rate=0.1, **{name: ok})
        assert np.isfinite(float(np.asarray(out.values["x"])))

    def test_it_refuses_before_the_wrong_answer_can_be_computed(self):
        """Entry, not exit -- the point is that the wrong answer is never formed.

        Without the guard this same call returns 15.384941. Asserting the
        refusal alone would not distinguish "refused at entry" from "refused
        after diverging", and only the first is what a caller can act on.
        """
        obj, at = self._quadratic()
        with pytest.raises(StructureError) as caught:
            minimize(obj, at, method="adam", steps=200, learning_rate=0.1, beta1=1.5)
        assert "15.38" in str(caught.value), (
            "the message no longer carries the measurement that justifies the guard"
        )

    @pytest.mark.parametrize("bad", [1.5, -0.5, float("nan")])
    def test_the_gradient_method_is_unaffected(self, bad):
        """``beta1, beta2, eps: Adam's, ignored by "gradient"`` is the contract.

        So an out-of-range beta must NOT be refused there: the value provably
        does not enter the answer, and refusing would reject a call whose
        result is correct. The rule the two halves share is *refuse where it
        changes the answer, honour "ignored" where the contract says ignored*.

        This case was written the other way round first -- asserting the
        refusal while its own docstring said the opposite -- and it passed,
        because the guard had been put in the shared settings check without
        anyone asking which methods it should reach. The contradiction between
        a test's name and its assertion is the only thing that caught it.
        """
        obj, at = self._quadratic()
        out = minimize(obj, at, method="gradient", steps=50, learning_rate=0.1,
                       beta1=bad, beta2=bad)
        assert float(np.asarray(out.values["x"])) == pytest.approx(
            float(np.asarray(minimize(obj, at, method="gradient", steps=50,
                                      learning_rate=0.1).values["x"]))
        ), "the betas changed a gradient-descent answer, so they are not ignored"
