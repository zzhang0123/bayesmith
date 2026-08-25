"""``prior_sensitivity`` -- the question a chain cannot answer about itself.

Ported from rheplicant's ``tests/inference/test_prior_sensitivity.py``; the
cross-check against rheplicant's own implementation on the shared tour
fixture lives in ``tests/crosscheck/``, and this file carries the
package-native properties: the two routes verify each other, the closed
form's matrix is the LIKELIHOOD's and not the posterior's, the
counterfactual ladder collapses onto the reported shift at the declared
width, and -- new here, the B3 repair -- an indefinite likelihood Hessian
gets a descent direction instead of a lunge at a saddle.

Every ground truth in this file is one of: an independent Newton refit
written HERE (plain, undamped, sharing no code with the module), a numpy
algebraic identity, or a fixture whose answer is closed-form. Two
implementations agreeing is not evidence; the module's own refit column is
never used as the truth for the module's closed form without the in-file
solver agreeing too.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.diagnose import sensitivity
from bayesmith.diagnose.sensitivity import (
    CRITERION_SHIFT,
    VERIFY_ATOL,
    VERIFY_RTOL,
    PriorSensitivityReport,
    _descent_direction,
    prior_sensitivity,
)
from bayesmith.errors import ConvergenceError, GraphError, NotGaussian
from bayesmith.graph.evaluate import evaluate
from tests.diagnose.models import (
    NU0,
    affine_graph,
    hierarchical_graph,
    noisy_power_law_graph,
    power_law_graph,
    saddle_graph,
)

# ------------------------------------------------- an independent refit --


def _newton(objective, x0, steps=80):
    """Plain undamped Newton. Shares nothing with the module.

    Deliberately the simplest thing that can be written: no backtracking, no
    eigvalsh, no jitter. If this and the module's damped, guarded solver
    land on the same point, the point is the mode and not an artefact of
    either search.
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    for _ in range(steps):
        step = jnp.linalg.solve(jax.hessian(objective)(x), jax.grad(objective)(x))
        x = x - step
        if float(jnp.max(jnp.abs(step))) < 1e-14:
            break
    return x


def _objectives(graph, data):
    """(neg log posterior, neg log likelihood) for the two-latent power law,
    spelled from the graph's own numbers but through none of its machinery."""
    freq = jnp.linspace(60e6, 85e6, 8)

    def neg_log_lik(x):
        prediction = jnp.exp(x[0]) * (freq / NU0) ** (-x[1])
        return 0.5 * jnp.sum(((prediction - data) / 5.0) ** 2)

    loc = jnp.array([7.8, 2.3])
    scale = jnp.array([0.5, 0.3])

    def neg_log_post(x):
        return neg_log_lik(x) + 0.5 * jnp.sum(((x - loc) / scale) ** 2)

    return neg_log_post, neg_log_lik


@pytest.fixture(scope="module")
def fitted():
    with jax.enable_x64(True):
        graph, data = noisy_power_law_graph()
        report = prior_sensitivity(graph)
    return graph, data, report


# ----------------------------------------------------------- the answer --


class TestTheShift:
    def test_an_independent_refit_lands_on_the_same_two_modes(self, fitted):
        graph, data, report = fitted
        with jax.enable_x64(True):
            neg_log_post, neg_log_lik = _objectives(graph, data)
            start = jnp.array([7.8, 2.3])
            map_mine = _newton(neg_log_post, start)
            lik_mine = _newton(neg_log_lik, start)
        assert np.asarray(map_mine) == pytest.approx(report.mode, rel=1e-9)
        refit = np.asarray(map_mine - lik_mine) / report.sigma_post
        assert refit == pytest.approx(report.shift_sigma_refit, rel=1e-7)
        # ... and the closed form agrees with the refit, which the report
        # itself also says.
        assert report.shift_sigma == pytest.approx(refit, rel=1e-3, abs=1e-6)
        assert report.refit_converged
        assert bool(np.all(report.verified))

    def test_the_sign_is_toward_the_prior_mean(self, fitted):
        """The measured fixture: beta's prior (2.3) sits BELOW its mode
        (2.5511), so beta is pulled down and the shift is negative;
        log-amp's prior (7.8) sits below its mode by less than the cross
        term pushes, and its shift is positive. ``mean_offset`` is a
        magnitude by construction, so the sign lives only in
        ``shift_sigma`` -- a diagnostic reporting |shift| would say a prior
        pulling the wrong way and one pulling the right way are the same
        situation."""
        _, _, report = fitted
        assert float(report.for_latent("fg_beta")["shift_sigma"]) < 0.0
        assert float(report.for_latent("fg_beta")["mean_offset"]) > 0.0
        assert float(report.for_latent("fg_beta")["mode"]) > 2.3

    def test_the_worst_offender_is_named_not_indexed(self, fitted):
        _, _, report = fitted
        name, index, value = report.worst
        assert (name, index) == ("fg_beta", 0)
        assert value == pytest.approx(float(report.for_latent("fg_beta")["shift_sigma"]))
        assert abs(value) == float(np.max(np.abs(report.shift_sigma)))

    def test_the_declared_row_of_the_ladder_is_the_reported_shift(self, fitted):
        """To the last bit, and that is a statement about the derivation.

        ``shift_sigma`` is ``H^-1 P (m - theta_hat)`` anchored at the mode
        it has; ``shift_at`` is ``(H + P_s)^-1 P_s (I + H^-1 P_d)(m -
        theta_hat)`` anchored at the likelihood mode it reconstructs. Put
        ``P_s = P_d`` and the second collapses onto the first
        algebraically, so agreement here is not a coincidence to be checked
        loosely. A counterfactual that dropped the ``(I + H^-1 P_d)``
        anchor would still look reasonable -- a small relative error, well
        inside anything a reader calls agreement -- and this is the
        assertion that would not have it.
        """
        _, _, report = fitted
        for name, declared in (("fg_beta", 0.3), ("fg_log_amp", 0.5)):
            assert float(report.shift_at(name, declared)) == pytest.approx(
                float(report.for_latent(name)["shift_sigma"]), rel=1e-12
            )

    def test_the_counterfactual_matches_an_actual_rerun_at_that_width(self, fitted):
        """Compare against re-declaring beta's prior and solving again.

        Both sides in the DECLARED posterior sigma: the re-run's own sigma
        is narrower, and dividing by it would compare two different
        quantities and hide exactly the anchoring error this test is for.
        """
        _, _, report = fitted
        sigma_declared = float(report.for_latent("fg_beta")["sigma_post"])
        likelihood_beta = float(report.for_latent("fg_beta")["mode"]) - (
            float(report.for_latent("fg_beta")["shift_sigma_refit"]) * sigma_declared
        )
        for hypothetical, tolerance in ((0.1, 1e-3), (0.03, 1e-2)):
            with jax.enable_x64(True):
                regraph, _ = noisy_power_law_graph(
                    prior_widths=(0.5, hypothetical)
                )
                rerun = prior_sensitivity(regraph)
            truth = (
                float(rerun.for_latent("fg_beta")["mode"]) - likelihood_beta
            ) / sigma_declared
            assert float(report.shift_at("fg_beta", hypothetical)) == pytest.approx(
                truth, rel=tolerance
            )

    def test_the_criterion_inverts_the_diagonal_law_exactly(self, fitted):
        """``criterion_std = sqrt(sigma_post * mean_offset / 0.1)`` is an
        algebraic inversion, pinned as one: putting the criterion back into
        the diagonal law returns CRITERION_SHIFT to roundoff. The FULL
        shift at that width then differs from 0.1 by the cross term from
        the other latent's prior -- small for beta (measured 0.0998), large
        for log-amp on this deliberately correlated fixture (measured
        0.0504, the cross term is comparable to the pull) -- which is why
        the criterion is a criterion and not an identity.
        """
        _, _, report = fitted
        for name in report.names:
            entry = report.for_latent(name)
            criterion = float(entry["criterion_std"])
            law = float(entry["sigma_post"]) * float(entry["mean_offset"]) / criterion**2
            assert law == pytest.approx(CRITERION_SHIFT, rel=1e-12)
        assert abs(float(report.shift_at("fg_beta", float(
            report.for_latent("fg_beta")["criterion_std"]
        )))) == pytest.approx(CRITERION_SHIFT, rel=0.05)


class TestAnAffineModel:
    """Exactly quadratic log-posterior: the closed form is exact, the refit
    is Newton-exact in one step, and the fixture is deliberately starved so
    the wrong-matrix error would be enormous."""

    def test_one_exact_newton_step_plus_one_to_confirm(self):
        with jax.enable_x64(True):
            report = prior_sensitivity(affine_graph())
        assert report.newton_steps == 2
        assert report.refit_steps == 2

    def test_the_closed_form_is_exact_and_the_wrong_matrix_would_not_be(self):
        """On this fixture ``diag((H + P)^-1 P)`` -- the error made by
        putting the posterior's matrix where the likelihood's belongs -- is
        0.57: FIFTY-SEVEN percent, against a measured closed-vs-refit
        disagreement at the refit's cancellation floor (~4e-16). A test on
        a data-rich fixture cannot make this call: H swamps P and no prior
        anyone would write is visible against it, which is why this fixture
        is small and poor on purpose."""
        with jax.enable_x64(True):
            report = prior_sensitivity(affine_graph())
        closed = float(report.shift_sigma[0])
        refit = float(report.shift_sigma_refit[0])
        assert closed == pytest.approx(refit, rel=VERIFY_RTOL, abs=VERIFY_ATOL)
        assert bool(np.all(report.verified))
        # The discrimination is real on this fixture:
        prior_precision = 1.0 / 0.2**2
        likelihood_precision = float(report.precision[0, 0]) - prior_precision
        share = prior_precision / (likelihood_precision + prior_precision)
        assert share > 0.5, share
        # ... so the wrong matrix would miss by ~share, far above the
        # verification tolerance that just passed.
        assert share > 100 * VERIFY_RTOL


class TestB3TheIndefiniteHessian:
    """The repair the port carries: ``sensitivity.py``'s ancestor solved
    ``H x = g`` raw, and an indefinite ``H`` -- routine in the
    likelihood-only refit, where the priors were the term keeping it
    positive -- made the step a lunge at a saddle."""

    def test_the_direction_through_an_indefinite_matrix_still_descends(self):
        """The sharp pin, on a matrix small enough to read: with
        ``H = diag(-2, 5)`` the exact Newton direction has NEGATIVE inner
        product with the gradient -- a step ``x - direction`` INCREASES the
        objective -- while the repaired direction's is positive, and of
        sane magnitude (the reflected shift keeps the modified matrix
        conditioned like H itself; a shift parked just above zero returns a
        2e7-length lunge the line search must then tame)."""
        with jax.enable_x64(True):
            matrix = jnp.array([[-2.0, 0.0], [0.0, 5.0]])
            gradient = jnp.array([1.0, 1.0])
            plain = jnp.linalg.solve(matrix, gradient)
            repaired = _descent_direction(matrix, gradient)
        assert float(gradient @ plain) < 0.0
        assert float(gradient @ repaired) > 0.0
        assert float(jnp.linalg.norm(repaired)) < 10.0

    def test_a_positive_definite_matrix_keeps_the_exact_newton_step(self):
        """The fallback must not touch the ordinary case: on a PD matrix
        the repaired direction IS the Newton step, bitwise."""
        with jax.enable_x64(True):
            matrix = jnp.array([[2.0, 0.3], [0.3, 5.0]])
            gradient = jnp.array([1.0, -1.0])
            assert np.array_equal(
                np.asarray(_descent_direction(matrix, gradient)),
                np.asarray(jnp.linalg.solve(matrix, gradient)),
            )

    def test_a_matrix_of_NaN_yields_NaN_for_the_callers_finiteness_check(self):
        with jax.enable_x64(True):
            matrix = jnp.full((2, 2), jnp.nan)
            direction = _descent_direction(matrix, jnp.ones(2))
        assert bool(np.all(np.isnan(np.asarray(direction))))

    def test_the_saddle_model_walks_out_to_the_mode(self):
        """End to end: ``pred = theta^2`` has likelihood curvature
        ``2 (3 theta^2 - 1)``, NEGATIVE at the 0.1 starting point, and the
        likelihood-only refit crosses the same region. Both solves must
        traverse it and land near theta = 1 -- with the old raw solve the
        first Newton direction points toward the saddle at 0. ``verified``
        is deliberately not asserted True: this toy's nonlinearity over the
        displacement sits right at VERIFY_RTOL (measured 3.4e-3), and the
        flag saying so is the flag working."""
        with jax.enable_x64(True):
            report = prior_sensitivity(saddle_graph())
        assert float(report.mode[0]) == pytest.approx(1.0, abs=5e-3)
        assert report.refit_converged
        assert float(report.shift_sigma[0]) == pytest.approx(
            float(report.shift_sigma_refit[0]), rel=5e-3
        )


# ------------------------------------------------------------- refusals --


class TestRefusals:
    def test_a_prior_with_no_quadratic_form_is_refused_by_name(self):
        """The Jeffreys-configured graph declares its latents
        ImproperUniform: exactly the density prior_sensitivity has nothing
        to measure against, and the refusal must name the latent, the
        distribution family and the way out."""
        with jax.enable_x64(True):
            graph = power_law_graph(flat_latents=True)
            with pytest.raises(NotGaussian) as excinfo:
                prior_sensitivity(graph)
        message = str(excinfo.value)
        assert "fg_log_amp" in message
        assert "NUTS" in message and "names=" in message

    def test_a_non_gaussian_prior_is_fine_as_long_as_it_is_not_asked_about(self):
        """``names=`` is the escape hatch, and it has to actually work: an
        ImproperUniform on the amplitude is no obstacle to asking whether
        beta's Gaussian prior is biasing it."""
        import numpyro.distributions as dist

        from bayesmith import det, observe, sample, trace

        with jax.enable_x64(True):
            freq = jnp.linspace(60e6, 85e6, 8)
            truth = jnp.exp(7.8) * (freq / NU0) ** (-2.55)
            data = truth + 5.0 * jax.random.normal(jax.random.key(3), (8,))

            def model():
                la = sample(
                    "fg_log_amp",
                    lambda: dist.ImproperUniform(dist.constraints.real, (), ()),
                )
                be = sample("fg_beta", lambda: dist.Normal(2.3, 0.3))
                pred = det(
                    "pred", lambda a, b: jnp.exp(a) * (freq / NU0) ** (-b), la, be
                )
                observe("d", lambda mu: dist.Normal(mu, 5.0), pred, obs=data)

            graph = trace(model)
            report = prior_sensitivity(
                graph, names=("fg_beta",), at={"fg_log_amp": jnp.array(7.8)}
            )
        assert report.names == ("fg_beta",)
        assert bool(np.all(report.verified))

    def test_a_selection_whose_priors_move_with_it_is_refused(self):
        """``child ~ Normal(parent, 0.5)`` with BOTH selected: child's (m,
        s) then move with the very parameters being analysed, and the
        closed form would differentiate a matrix it treats as constant."""
        with jax.enable_x64(True):
            graph = hierarchical_graph()
            with pytest.raises(GraphError, match="child.*parent|parameterised"):
                prior_sensitivity(graph)
            # Either alone is a well-posed conditional question.
            child_only = prior_sensitivity(graph, names=("child",))
            parent_only = prior_sensitivity(graph, names=("parent",))
        assert child_only.names == ("child",)
        assert parent_only.names == ("parent",)

    def test_a_rank_deficient_selection_is_refused_and_identifiability_named(self):
        """Two latents whose sum is the only thing the data sees: the
        posterior is proper (the priors make it so) and every number that
        would come back is finite -- the prior reporting on itself. The
        refusal names the tool that measured it and the shares."""
        import numpyro.distributions as dist

        from bayesmith import det, observe, sample, trace

        with jax.enable_x64(True):
            x = jnp.linspace(1.0, 2.0, 8)
            data = jnp.exp(7.6) * x

            def model():
                a = sample("fg_a", lambda: dist.Normal(3.8, 0.5))
                b = sample("fg_b", lambda: dist.Normal(3.8, 0.5))
                pred = det("pred", lambda a_, b_: jnp.exp(a_ + b_) * x, a, b)
                observe("d", lambda mu: dist.Normal(mu, 10.0), pred, obs=data)

            graph = trace(model)
            with pytest.raises(GraphError) as excinfo:
                prior_sensitivity(graph)
        message = str(excinfo.value)
        assert "identifiability" in message
        assert "rank 1 of 2" in message
        assert "fg_a" in message and "fg_b" in message

    def test_an_undeclared_name_is_refused(self, fitted):
        graph, _, _ = fitted
        with jax.enable_x64(True):
            with pytest.raises(GraphError, match="fg_gamma"):
                prior_sensitivity(graph, names=("fg_gamma",))
            with pytest.raises(GraphError, match="fg_gamma"):
                prior_sensitivity(graph, at={"fg_gamma": jnp.array(1.0)})

    def test_an_ambient_float32_call_is_refused_by_name(self, fitted):
        graph, _, _ = fitted
        with pytest.raises(GraphError, match="enable_x64"):
            prior_sensitivity(graph)


# ------------------------------------------------------------ the report --


class TestTheReport:
    def test_it_is_frozen_and_holds_numpy_float64(self, fitted):
        import dataclasses

        _, _, report = fitted
        assert dataclasses.is_dataclass(PriorSensitivityReport)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.shift_sigma = np.zeros(2)
        for field in ("shift_sigma", "sigma_post", "mean_offset", "criterion_std"):
            assert isinstance(getattr(report, field), np.ndarray)
            assert getattr(report, field).dtype == np.float64

    def test_the_rows_are_in_declaration_order_not_sorted_order(self, fitted):
        """``('fg_log_amp', 'fg_beta')`` sorts the other way round, so a
        report that borrowed any sorted flattening while naming rows in
        declaration order would hand back beta's numbers under log-amp's
        name with every shape agreeing. The widths differ by an order of
        magnitude on this fixture, so the swap is visible."""
        _, _, report = fitted
        assert report.names == ("fg_log_amp", "fg_beta")
        assert sorted(report.names) != list(report.names)
        assert report.spans == ((0, 1), (1, 2))
        assert report.sigma_post[0] < report.sigma_post[1] / 5.0

    def test_a_latent_that_is_not_there_is_named_in_the_refusal(self, fitted):
        _, _, report = fitted
        with pytest.raises(GraphError, match="fg_gamma"):
            report.for_latent("fg_gamma")
        with pytest.raises(GraphError, match="fg_gamma"):
            report.shift_at("fg_gamma", 0.1)

    def test_a_nonpositive_prior_width_is_refused_rather_than_dividing(self, fitted):
        _, _, report = fitted
        with pytest.raises(GraphError, match="positive"):
            report.shift_at("fg_beta", 0.0)
        with pytest.raises(GraphError, match="positive"):
            report.shift_at("fg_beta", -0.3)

    def test_a_width_that_does_not_broadcast_is_refused(self, fitted):
        _, _, report = fitted
        with pytest.raises(GraphError, match="broadcast"):
            report.shift_at("fg_beta", [0.1, 0.2, 0.3])

    def test_it_counts_the_newton_steps_it_took(self, fitted):
        """5 from the prior centres, 3 for the refit from the mode --
        measured on this fixture. Pinned because MAX_NEWTON_STEPS is 100
        against this need, so a solve that hits the ceiling has diverged
        rather than run out of budget."""
        _, _, report = fitted
        assert report.newton_steps == 5
        assert report.refit_steps == 3


class TestAVectorLatent:
    """A latent of EIGHT elements: a scalar latent hides ``for_latent``'s
    reshape, ``worst``'s index-within arithmetic and ``shift_at``'s slice,
    all of which are the identity when a span is one element wide."""

    @staticmethod
    def _vector_graph():
        import numpyro.distributions as dist

        from bayesmith import det, observe, sample, trace

        freq = jnp.linspace(60e6, 85e6, 8)
        base = jnp.exp(7.8) * (freq / NU0) ** (-2.55)
        offsets = jnp.array([260.0, 255.0, 270.0, 250.0, 265.0, 240.0, 275.0, 258.0])
        data = base + offsets

        def model():
            t_rx = sample(
                "t_rx", lambda: dist.Normal(jnp.full(8, 280.0), 40.0).to_event(1)
            )
            pred = det("pred", lambda t: base + t, t_rx)
            observe("d", lambda mu: dist.Normal(mu, 2.0).to_event(1), pred, obs=data)

        return trace(model)

    def test_every_per_element_column_is_shaped_like_the_latent(self):
        with jax.enable_x64(True):
            report = prior_sensitivity(self._vector_graph())
        assert report.names == ("t_rx",)
        assert report.spans == ((0, 8),)
        assert report.n_par == 8
        entry = report.for_latent("t_rx")
        for field, array in entry.items():
            assert array.shape == (8,), field
        # The prior was declared with a vector loc and a scalar scale; both
        # are broadcast to the latent, so neither covers one element.
        assert entry["prior_loc"] == pytest.approx(np.full(8, 280.0))
        assert entry["prior_std"] == pytest.approx(np.full(8, 40.0))

    def test_worst_names_the_element_within_the_latent(self):
        with jax.enable_x64(True):
            report = prior_sensitivity(self._vector_graph())
        name, index, value = report.worst
        assert name == "t_rx"
        column = report.for_latent("t_rx")["shift_sigma"]
        assert index == int(np.argmax(np.abs(column)))
        # Channel 5 carries the largest offset from the prior mean (240
        # against 280), so the index is NON-zero: an index returned as the
        # span's start rather than the offset within it would be 0 here and
        # correct for every scalar latent in this file.
        assert index == 5
        assert value == pytest.approx(column[index])

    def test_a_scalar_width_and_a_vector_of_that_width_are_the_same_question(self):
        with jax.enable_x64(True):
            report = prior_sensitivity(self._vector_graph())
        scalar = report.shift_at("t_rx", 4.0)
        vector = report.shift_at("t_rx", np.full(8, 4.0))
        assert scalar.shape == (8,)
        assert scalar == pytest.approx(vector, rel=1e-14)

    def test_a_per_element_width_moves_mostly_that_element(self):
        with jax.enable_x64(True):
            report = prior_sensitivity(self._vector_graph())
        widths = np.full(8, 40.0)
        widths[3] = 0.5
        shifted = report.shift_at("t_rx", widths)
        declared = report.for_latent("t_rx")["shift_sigma"]
        assert abs(shifted[3]) > 100.0 * abs(declared[3])
        others = [i for i in range(8) if i != 3]
        assert np.all(np.abs(shifted[others]) < 100.0 * np.abs(declared[others]))

    def test_both_routes_agree_and_the_affine_model_is_exact(self):
        """The prediction is affine in a receiver temperature, so the
        closed form is exact by construction and the refit's disagreement
        is its own cancellation floor, not a derivation error."""
        with jax.enable_x64(True):
            report = prior_sensitivity(self._vector_graph())
        assert report.refit_converged
        assert bool(np.all(report.verified))
        assert report.shift_sigma == pytest.approx(report.shift_sigma_refit, rel=1e-8)


class TestWhenNewtonDoesNotGetThere:
    """Both solves can fail, and they are not the same kind of failure."""

    def test_a_mode_that_is_not_found_is_fatal_and_says_what_it_prevents(
        self, monkeypatch
    ):
        monkeypatch.setattr(sensitivity, "MAX_NEWTON_STEPS", 2)
        with jax.enable_x64(True):
            graph, _ = noisy_power_law_graph()
            with pytest.raises(ConvergenceError) as excinfo:
                prior_sensitivity(graph)
        message = str(excinfo.value)
        assert "did not converge" in message
        # Naming the alternatives is the point: a bad basin, float32, or a
        # posterior this method has no business approximating.
        assert "at=" in message and "float32" in message and "NUTS" in message

    def test_a_verification_that_could_not_run_is_reported_not_raised(
        self, fitted, monkeypatch
    ):
        """Starting AT the mode, the MAP re-converges within budget; the
        refit does not. What comes back is the closed form with
        ``verified`` false and the refit column NaN -- never a NaN
        silently averaged into a verdict."""
        graph, _, report = fitted
        monkeypatch.setattr(sensitivity, "MAX_NEWTON_STEPS", 2)
        with jax.enable_x64(True):
            starved = prior_sensitivity(
                graph,
                at={
                    "fg_log_amp": jnp.asarray(report.mode_of("fg_log_amp")),
                    "fg_beta": jnp.asarray(report.mode_of("fg_beta")),
                },
            )
        assert not starved.refit_converged
        assert not bool(np.any(starved.verified))
        assert bool(np.all(np.isnan(starved.shift_sigma_refit)))
        # The closed form is unaffected -- it never needed the second solve.
        assert starved.for_latent("fg_beta")["shift_sigma"] == pytest.approx(
            report.for_latent("fg_beta")["shift_sigma"], rel=1e-6
        )


class TestHierarchyBelowTheSelection:
    """A latent whose density the SELECTION parameterises is part of "the
    likelihood": selecting only ``parent`` puts ``child``'s density into
    the rest-term, where it belongs -- the graph-native generalisation of
    "everything except the selected priors"."""

    def test_the_rest_term_includes_the_downstream_density(self):
        with jax.enable_x64(True):
            graph = hierarchical_graph()
            report = prior_sensitivity(graph, names=("parent",))
            # An independent spelling of the same objective: the parent's
            # likelihood is the child's density at the held value PLUS the
            # observed node's, and its curvature is what sigma_post reads.
            env = evaluate(graph, {"parent": jnp.asarray(report.mode[0]),
                                   "child": jnp.asarray(2.0)})

            def neg_log_rest(p):
                # child ~ Normal(parent, 0.5) at child=2.0; d fixed.
                child_term = 0.5 * ((2.0 - p) / 0.5) ** 2
                x = jnp.linspace(1.0, 2.0, 8)
                data = 2.0 * x
                prediction = 2.0 * x  # child held at 2.0 drives the data node
                observed_term = 0.5 * jnp.sum(((prediction - data) / 0.3) ** 2)
                return child_term + observed_term

            curvature = float(jax.hessian(neg_log_rest)(jnp.asarray(report.mode[0])))
            # Total posterior curvature = rest + parent's own prior 1/1.0^2.
            assert float(report.precision[0, 0]) == pytest.approx(
                curvature + 1.0, rel=1e-9
            )
        assert env["child"] == 2.0
