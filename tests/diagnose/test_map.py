"""Graph-native MAP, with refusals belonging to MAP rather than sensitivity."""

from __future__ import annotations

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
    assert np.max(np.abs(mode - oracle)) < 5e-8

    residual = data - design @ mode
    expected_objective = 0.5 * np.sum(
        (residual / 0.05) ** 2 + np.log(2.0 * np.pi * 0.05**2)
    ) + 0.5 * np.sum((mode / 1.5) ** 2 + np.log(2.0 * np.pi * 1.5**2))
    expected_gradient = np.max(
        np.abs(design.T @ (design @ mode - data) / 0.05**2 + mode / 1.5**2)
    )
    assert found.objective == pytest.approx(expected_objective, abs=2e-10)
    assert found.gradient_norm == pytest.approx(expected_gradient, rel=0.5)
    assert found.steps == 100


def test_relative_step_cannot_accept_a_vandermonde_point_with_large_gradient():
    with jax.enable_x64(True):
        graph, _, _, oracle = _vandermonde_graph(
            degree=10,
            t_max=1.0,
            coefficient_scale=1e8,
            residual_scale=0.01,
        )
        found = map_estimate(graph)

    assert isinstance(found, Refused)
    assert "gradient" in found.reason
    assert "roundoff" in found.reason
    assert np.max(np.abs(oracle)) > 1e7


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
