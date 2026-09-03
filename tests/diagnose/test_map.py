"""Graph-native MAP, with refusals belonging to MAP rather than sensitivity."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, joint_prior, observe, sample, trace
from bayesmith.diagnose.local import local_block
from bayesmith.diagnose.map import MapEstimate, NotApplicable, Refused, map_estimate
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import wiener_solve


def _linear_gaussian_graph():
    design = jnp.array([[1.0, -0.3], [0.4, 1.2], [-0.7, 0.8], [1.5, 0.2], [0.1, -1.1]])
    truth = jnp.array([1.1, -0.6])
    data = design @ truth + jnp.array([0.1, -0.2, 0.05, 0.08, -0.03])

    def model():
        matrix = const("design", design)
        weights = sample(
            "weights",
            lambda: dist.Normal(jnp.array([0.3, -0.4]), jnp.array([2.0, 3.0])).to_event(
                1
            ),
        )
        prediction = det(
            "prediction",
            lambda a, w: a @ w,
            matrix,
            weights,
            linear_in=("weights",),
        )
        observe(
            "data",
            lambda mu: dist.Normal(mu, 0.5).to_event(1),
            prediction,
            obs=data,
        )

    return trace(model)


def _funnel_graph():
    """One-dimensional Neal funnel; its exact joint mode is (-4.5, 0)."""

    def model():
        neck = sample("neck", lambda: dist.Normal(0.0, 3.0))
        sample("x", lambda y: dist.Normal(0.0, jnp.exp(y / 2.0)), neck)

    return trace(model)


def _hierarchical_graph():
    locations = jnp.linspace(1.0, 2.0, 8)
    data = 2.0 * locations

    def model():
        parent = sample("parent", lambda: dist.Normal(1.5, 1.7))
        child = sample("child", lambda p: dist.Normal(p, 0.6), parent)
        prediction = det("prediction", lambda c: c * locations, child)
        observe(
            "data",
            lambda mu: dist.Normal(mu, 0.3).to_event(1),
            prediction,
            obs=data,
        )

    return trace(model)


def _vandermonde_graph(
    *, degree: int, t_max: float, coefficient_scale: float, residual_scale: float
):
    """An ill-conditioned quadratic with a separate dense NumPy oracle."""
    points = np.linspace(0.0, t_max, 120)
    design = np.vander(points, N=degree + 1, increasing=True)
    base = np.sin(np.arange(degree + 1) + 1.0) / (np.arange(degree + 1) + 1.0)
    truth = coefficient_scale * base
    data = design @ truth + residual_scale * np.sin(1.7 * np.arange(120))

    def model():
        matrix = const("vandermonde", jnp.asarray(design))
        coefficients = sample(
            "coefficients",
            lambda: dist.Normal(
                jnp.zeros(degree + 1), jnp.full(degree + 1, 1.5)
            ).to_event(1),
        )
        prediction = det(
            "prediction",
            lambda a, w: a @ w,
            matrix,
            coefficients,
            linear_in=("coefficients",),
        )
        observe(
            "data",
            lambda mu: dist.Normal(mu, 0.05).to_event(1),
            prediction,
            obs=jnp.asarray(data),
        )

    precision = design.T @ design / 0.05**2 + np.eye(degree + 1) / 1.5**2
    information = design.T @ data / 0.05**2
    oracle = np.linalg.solve(precision, information)
    return trace(model), design, data, oracle


def _ridge_graph(
    *, observation_std: float, coordinate_scale: float = 1.0, pin_mode: bool = False
):
    """A 40x3 quadratic whose scale moves without moving its conditioning.

    The dense normal-equation oracle is deliberately NumPy-only.  A residual
    in the orthogonal complement of the design makes the floating-point
    stationarity error visible without changing that oracle.
    """
    design = np.random.default_rng(7).normal(size=(40, 3))
    prior_mean = np.array([0.3, -0.4, 0.2])
    prior_std = np.array([2.0, 3.0, 1.7])
    target = np.array([1.200038709, -0.549966357, 0.280401865])
    residual = np.sin(1.7 * np.arange(40))
    residual -= design @ np.linalg.solve(design.T @ design, design.T @ residual)
    data = design @ target + 0.01 * residual
    prior_precision = np.diag(1.0 / prior_std**2)
    if pin_mode:
        data += (
            observation_std**2
            * design
            @ np.linalg.solve(
                design.T @ design, prior_precision @ (target - prior_mean)
            )
        )

    scaled_design = coordinate_scale * design
    scaled_mean = prior_mean / coordinate_scale
    scaled_std = prior_std / coordinate_scale

    def model():
        matrix = const("ridge_design", jnp.asarray(scaled_design))
        coefficients = sample(
            "coefficients",
            lambda: dist.Normal(
                jnp.asarray(scaled_mean), jnp.asarray(scaled_std)
            ).to_event(1),
        )
        prediction = det(
            "ridge_prediction",
            lambda a, w: a @ w,
            matrix,
            coefficients,
            linear_in=("coefficients",),
        )
        observe(
            "ridge_data",
            lambda mu: dist.Normal(mu, observation_std).to_event(1),
            prediction,
            obs=jnp.asarray(data),
        )

    physical_precision = (
        design.T @ design / observation_std**2 + prior_precision
    )
    physical_information = (
        design.T @ data / observation_std**2 + prior_precision @ prior_mean
    )
    physical_oracle = np.linalg.solve(physical_precision, physical_information)
    return trace(model), physical_oracle, physical_precision


def test_linear_gaussian_map_matches_wiener_and_is_a_local_block_point():
    with jax.enable_x64(True):
        graph = _linear_gaussian_graph()
        found = map_estimate(graph)
        assert isinstance(found, MapEstimate)

        block = linear_operator(graph, ("weights",), at={})
        exact, _ = wiener_solve(
            block,
            precision=precision_at(graph, {"weights": jnp.zeros(2)}),
            tol=1e-14,
        )
        assert np.asarray(found["weights"]) == pytest.approx(
            np.asarray(exact["weights"]), abs=2e-12
        )

        local = local_block(graph, ("weights",), found, priors=True)
        assert local.names == ("weights",)
        assert np.asarray(local.prior_std["weights"]) == pytest.approx([2.0, 3.0])


def test_funnel_is_accepted_instead_of_inheriting_the_entangled_prior_refusal():
    with jax.enable_x64(True):
        found = map_estimate(_funnel_graph())
    assert isinstance(found, MapEstimate)
    assert float(found["neck"]) == pytest.approx(-4.5, abs=2e-12)
    assert float(found["x"]) == pytest.approx(0.0, abs=2e-12)


def test_a_hierarchical_model_is_accepted():
    with jax.enable_x64(True):
        found = map_estimate(_hierarchical_graph())
    assert isinstance(found, MapEstimate)
    assert np.isfinite(found.objective)
    assert found.gradient_norm < 1e-9


def test_vandermonde_mode_is_judged_by_stationarity_not_newton_step_jitter():
    with jax.enable_x64(True):
        graph, design, data, oracle = _vandermonde_graph(
            degree=12,
            t_max=2.0,
            coefficient_scale=0.2,
            residual_scale=0.03,
        )
        found = map_estimate(graph)

    assert isinstance(found, MapEstimate)
    mode = np.asarray(found["coefficients"])

    # `mode` and `oracle` are two float64 solutions of the SAME 13x13 normal
    # equations, so the gap between them is set by that system's conditioning
    # and by nothing this seam controls. The chain, measured except where it
    # says derived: cond_2(design) = 2491804794.6754827, and forming A.T @ A
    # squares it -- 6.209091e+18 derived, past 1/eps = 4.503600e+15, i.e.
    # numerically singular on its own. The prior's 1/1.5**2 ridge caps
    # lambda_min at 0.4444444326032544, which is what brings
    # cond_2(precision) back to 109089884440.88826. So a backward-stable
    # float64 solve pins either point only to cond * eps * ||x||_inf =
    # 4.1425125494005045e-06, with ||oracle||_inf = 0.1710169377250264.
    #
    # The 5e-8 that stood here was one laptop's rounding sold as a property,
    # and it was not even this seam's rounding: re-solved in exact rational
    # arithmetic (`fractions.Fraction`) from the same float64 `precision` and
    # `information`, `oracle` is ITSELF 2.2617970316107794e-08 from the true
    # solution on macOS/Accelerate -- the reference had already spent more
    # than the whole budget it was being used to police.
    #
    # Measured max|mode - oracle|, against the 4.14e-06 the derivation
    # allows: 2.20828851836552e-08 on macOS/Accelerate (188x margin),
    # 3.226749584195332e-08 from the linux/amd64 container's default OpenBLAS
    # kernel (128x), and 5.808913667096349e-08 on the GitHub ubuntu runner
    # (71x) -- the value that tripped the pin, and reproducible here to the
    # last digit by running that container with OPENBLAS_CORETYPE=ZEN. The
    # runner is an AMD EPYC 7763 with `avx avx2 fma sse4_2` and no avx-512,
    # so the spread is OpenBLAS choosing a different dgemm microkernel, not
    # an ISA gap and not an accuracy gap.
    #
    # The band is derived from the fixture and then pinned WHOLE. Pinning
    # `cond` alone would not do: cond(precision) is data-independent, so
    # raising `coefficient_scale` inflates ||oracle||_inf and the band with
    # it while a cond-only pin keeps passing -- measured, scale 0.2/2/20
    # gives 4.142513e-06/4.081815e-05/4.075745e-04 with the cond pin True on
    # all three. Pinning the product closes that. Both factors come from
    # `design` and the NumPy oracle, never from `mode`, so a broken seam
    # cannot widen its own allowance.
    precision = design.T @ design / 0.05**2 + np.eye(oracle.size) / 1.5**2
    solve_tolerance = (
        np.linalg.cond(precision)
        * np.finfo(np.float64).eps
        * np.max(np.abs(oracle))
    )
    assert solve_tolerance == pytest.approx(4.14251e-06, rel=1e-3)
    assert np.max(np.abs(mode - oracle)) < solve_tolerance

    residual = data - design @ mode
    expected_objective = 0.5 * np.sum(
        (residual / 0.05) ** 2 + np.log(2.0 * np.pi * 0.05**2)
    ) + 0.5 * np.sum((mode / 1.5) ** 2 + np.log(2.0 * np.pi * 1.5**2))
    expected_gradient = np.max(
        np.abs(design.T @ (design @ mode - data) / 0.05**2 + mode / 1.5**2)
    )
    assert found.objective == pytest.approx(expected_objective, abs=2e-10)

    # The stationarity claim this test is named for, split into the three
    # separate things `rel=0.5` was conflating.
    #
    # A gradient at this scale is not a number, it is a noise floor:
    # A.T(Aw - d)/sigma^2 + w/s^2 loses about eps * max_i sum_k |A_ik w_k| in
    # the prediction, then multiplies that by max_j sum_i |A_ij| / sigma^2
    # over dot products of length 13. The measurement that settles it: at the
    # EXACT rational minimiser of this objective the computed gradient is
    # still 6.452316421694942e-08 on macOS. Zero is unreachable, so the RATIO
    # of two values under that floor is noise over noise -- measured 0.3991
    # on macOS, 0.4950 under the runner's own kernel (one percent of the old
    # budget left) and 0.8432, a plain FAIL, under the container's default
    # kernel -- where the DISTANCE assert passes at 3.226749584195332e-08, so
    # nothing shielded it and HEAD simply fails at the ratio with no defect
    # present. CI never saw it because CI does not run that kernel: on the
    # runner's own the ratio is 0.4950 and it was the distance assert, at
    # 5.808913667096349e-08, that reported. So the ordering of these two lines
    # never protected either of them.
    #
    # (1) and (2) are stationarity: the seam's own report, and an independent
    # NumPy gradient at the point it returned, both under the derived floor
    # eps * (13 + 1) * S * colsum / sigma^2. Measured across the three
    # kernels 9.32790310972548e-06 (macOS), 9.327907e-06 (ZEN) and
    # 9.327897e-06 (the container default, the one that differs most), a
    # relative spread of 1.1e-06. Pinned for the same reason the distance
    # band is: it is built from `design` and `oracle`, so a fixture edit
    # would otherwise inflate it silently -- measured, coefficient_scale
    # 0.2/2/20 gives 9.327903e-06/3.865204e-05/3.731173e-04.
    #
    # (3) is a different property, and the only one the old line could
    # actually test: that the reported field IS the inf-norm of that
    # gradient. Measured, a seam reporting 0.0, half, or ten times the true
    # norm passes both floors above and is caught only here. It needs no
    # tolerance, because it is not a cross-implementation comparison the way
    # `expected_gradient` is -- it is the same `jax.grad(-log_joint)` at the
    # same returned point, so it is BITWISE equal, and measured so on
    # macOS/Accelerate and in the container under both OpenBLAS kernels. A
    # band would have to be honest about costing the SUM of two evaluation
    # errors (2 * 13 * eps * S * colsum / sigma^2 = 1.7e-05), which is looser
    # than the floor and so could never fire; that is why the old assertion
    # is replaced here rather than merely re-tolerated.
    gradient_floor = (
        np.finfo(np.float64).eps
        * (oracle.size + 1)
        * np.max(np.abs(design) @ np.abs(oracle))
        * np.max(np.abs(design).sum(axis=0))
        / 0.05**2
    )
    assert gradient_floor == pytest.approx(9.3279e-06, rel=1e-3)
    assert found.gradient_norm < gradient_floor
    assert expected_gradient < gradient_floor

    # The reduction stays INSIDE the x64 block with the gradient: measured,
    # hoisting the `jnp.max` out of it returns a float32 truncation
    # (1.0315038601049764e-08 against the true 1.0315038350382222e-08) and
    # turns the bitwise assert into a spurious failure.
    from bayesmith import log_joint

    with jax.enable_x64(True):
        recomputed = float(
            jnp.max(
                jnp.abs(
                    jax.grad(lambda w: -log_joint(graph, {"coefficients": w}))(
                        jnp.asarray(found["coefficients"])
                    )
                )
            )
        )
    assert found.gradient_norm == recomputed

    # Not decoration, and not a work-budget report. Measured by instrumenting
    # the real `_newton` on this fixture: it returns (steps, flag) = (100,
    # False), so it exhausts its budget without the relative-step cut ever
    # firing, which is the premise the test is named for. It is a proxy and
    # not a proof -- `_newton` also returns 100 with the flag True when
    # sensitivity.py's in-loop `if moved < NEWTON_TOL` fires on the last
    # iteration, and `map_estimate` discards the flag -- but it is the only
    # witness the seam exposes. The proof is a monkeypatch: make `_newton`
    # return NaN unless its step flag is set and this graph comes back
    # Refused, so a seam that judged convergence by Newton-step size fails
    # the isinstance above whatever the bands are set to.
    assert found.steps == 100


@pytest.mark.parametrize("observation_std", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
def test_ridge_verdict_does_not_flip_when_only_hessian_scale_changes(
    observation_std,
):
    with jax.enable_x64(True):
        graph, oracle, precision = _ridge_graph(
            observation_std=observation_std
        )
        found = map_estimate(graph)

    assert np.linalg.cond(precision) == pytest.approx(1.59644, rel=2e-6)
    assert isinstance(found, MapEstimate)
    relative_error = np.max(np.abs(np.asarray(found["coefficients"]) - oracle))
    relative_error /= np.max(np.abs(oracle))
    assert relative_error < 2e-15


def test_physical_mode_is_invariant_under_a_pure_unit_conversion():
    physical_modes = []
    with jax.enable_x64(True):
        for coordinate_scale in (1e-2, 1.0, 1e2, 1e4):
            graph, oracle, _ = _ridge_graph(
                observation_std=1e-3,
                coordinate_scale=coordinate_scale,
                pin_mode=True,
            )
            found = map_estimate(graph)
            assert isinstance(found, MapEstimate), coordinate_scale
            physical = coordinate_scale * np.asarray(found["coefficients"])
            assert physical == pytest.approx(oracle, rel=0.0, abs=2e-15)
            physical_modes.append(physical)

    expected_physical = np.array([1.200038709, -0.549966357, 0.280401865])
    for physical in physical_modes:
        np.testing.assert_array_equal(np.round(physical, 9), expected_physical)


def test_subunit_curvature_floor_is_invariant_under_a_unit_conversion():
    """Deleting max(|lambda_max|, 1) keeps the curvature verdict unit-invariant.

    With the clamp the relative curvature floor anchors at eps*n*1.0 whenever
    the Hessian norm drops below one, so a pure width rescaling flips a
    sub-unit-Hessian posterior between Refused and MapEstimate.  The smallest
    eigenvalue 4e-16 sits above the unclamped floor eps*0.5*3 but below the
    clamped floor eps*1.0*3, so the same physical point is Refused in one unit
    system and a MapEstimate in the other before the clamp is removed.
    """

    def model(widths):
        sample(
            "x",
            lambda: dist.Normal(jnp.zeros(3), jnp.asarray(widths)).to_event(1),
        )

    unit = np.array([5.0e7, math.sqrt(2.0), math.sqrt(2.0)])
    converted = unit / math.sqrt(2.0)

    with jax.enable_x64(True):
        for widths in (unit, converted):
            found = map_estimate(trace(lambda w=widths: model(w)))
            assert isinstance(found, MapEstimate), widths
            assert widths * np.asarray(found["x"]) == pytest.approx(
                np.zeros(3), abs=0.0
            )


def test_a_large_candidate_cannot_buy_its_own_gradient_allowance(monkeypatch):
    """Multiplying the floor by max|candidate| changes this refusal to a MAP."""
    centre = np.array([300.0, -400.0, 200.0])
    widths = np.full(3, math.sqrt(5.0e7))
    candidate = np.array([4.0e8, -400.0, 200.0])

    def model():
        sample(
            "position",
            lambda: dist.Normal(
                jnp.asarray(centre), jnp.asarray(widths)
            ).to_event(1),
        )

    def runaway_newton(objective, initial):
        del objective, initial
        return jnp.asarray(candidate), 7, True

    monkeypatch.setattr("bayesmith.diagnose.map._newton", runaway_newton)
    with jax.enable_x64(True):
        found = map_estimate(trace(model))

    assert isinstance(found, Refused)
    assert "gradient" in found.reason
    assert "roundoff allowance" in found.reason


def test_stationarity_boundary_scales_with_subunit_hessian_norm(monkeypatch):
    """Reintroducing max(||H||, 1) admits only the low-curvature unit choice."""
    centre = 0.3
    displacement = 2.0 * math.sqrt(np.finfo(np.float64).eps)

    for width in (0.7, 28.0):

        def model(width=width):
            sample("position", lambda: dist.Normal(centre, width))

        def boundary_newton(objective, initial):
            del objective, initial
            return jnp.asarray([centre + displacement]), 3, True

        monkeypatch.setattr(
            "bayesmith.diagnose.map._newton", boundary_newton
        )
        with jax.enable_x64(True):
            found = map_estimate(trace(model))

        assert isinstance(found, Refused), width
        assert "gradient" in found.reason


class _SteepEvenPowerPrior:
    over = ("x",)

    def log_density(self, graph, values):
        del graph
        return -1e50 * (values["x"] - 0.3) ** 202


def test_a_finite_budget_refuses_a_point_when_an_exact_stationary_oracle_exists():
    def model():
        joint_prior(_SteepEvenPowerPrior())
        sample("x", lambda: dist.Normal(0.3, 2.0))

    with jax.enable_x64(True):
        graph = trace(model)
        found = map_estimate(graph, at={"x": jnp.asarray(1.3)})
        oracle = map_estimate(graph, at={"x": jnp.asarray(0.3)})

    assert isinstance(found, Refused)
    assert "gradient" in found.reason
    assert "roundoff" in found.reason
    assert isinstance(oracle, MapEstimate)
    assert float(oracle["x"]) == 0.3
    assert oracle.gradient_norm == 0.0


def test_logistic_underflow_is_not_a_finite_map():
    def model():
        logit = sample(
            "logit", lambda: dist.ImproperUniform(dist.constraints.real, (), ())
        )
        observe(
            "success",
            lambda x: dist.Bernoulli(logits=jnp.broadcast_to(x, (3,))).to_event(1),
            logit,
            obs=jnp.ones(3),
        )

    with jax.enable_x64(True):
        found = map_estimate(trace(model))

    assert isinstance(found, Refused)
    assert "Hessian" in found.reason
    assert "degenerate" in found.reason


def test_a_true_flat_posterior_direction_is_refused_with_a_way_out():
    with jax.enable_x64(True):

        def model():
            a = sample("a", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
            b = sample("b", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
            prediction = det("prediction", lambda x, y: x + y, a, b)
            observe("data", lambda mu: dist.Normal(mu, 0.7), prediction, obs=0.0)

        found = map_estimate(trace(model))
    assert isinstance(found, Refused)
    assert "Hessian" in found.reason
    assert "proper prior" in found.reason or "redundant" in found.reason


class _RunawayPrior:
    over = ("x",)

    def log_density(self, graph, values):
        del graph
        return values["x"]


def test_a_nonconvergent_objective_returns_refused_not_the_last_iterate():
    with jax.enable_x64(True):

        def model():
            joint_prior(_RunawayPrior())
            sample("x", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))

        found = map_estimate(trace(model))
    assert isinstance(found, Refused)
    assert not hasattr(found, "values")
    assert "converge" in found.reason
    assert "at=" in found.reason or "NUTS" in found.reason


def test_float32_is_refused_by_name_and_tells_the_caller_how_to_widen():
    graph = _linear_gaussian_graph()
    found = map_estimate(graph)
    assert isinstance(found, Refused)
    assert "ambient precision" in found.reason
    assert "enable_x64" in found.reason


def test_an_explicit_float32_start_is_refused_as_the_starting_point():
    start = jnp.zeros(2, dtype=jnp.float32)
    with jax.enable_x64(True):
        graph = _linear_gaussian_graph()
        found = map_estimate(graph, at={"weights": start})
    assert isinstance(found, Refused)
    assert "starting point is" in found.reason
    assert "ambient precision" not in found.reason


def test_a_graph_without_latents_is_not_applicable_not_an_empty_map():
    with jax.enable_x64(True):
        found = map_estimate(trace(lambda: None))
    assert isinstance(found, NotApplicable)
    assert "no latent" in found.reason
