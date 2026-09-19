"""Independent polynomial and differentiability contracts for the beam interpolator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.inference.tris_beam_grid import (
    BeamFeatureGrid,
    fit_coefficients,
    width_nodes,
)


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64(True):
        yield


def analytic(e, h):
    # The cusp is intentional: a single polynomial across zero cannot reproduce it.
    x, y = e / 0.01, h / 0.01
    t = 20 + 2 * x + y * y + 0.3 * x**3 * y - 0.5 * abs(x) + 0.2 * abs(y)
    return np.array([[t, 1.0, 1.0], [2 * t, 1.0, 1.0]])


def grid():
    nodes = width_nodes()
    columns = np.array([[analytic(e, h) for h in nodes] for e in nodes])
    return BeamFeatureGrid(jnp.asarray(fit_coefficients(nodes, columns)), 0.01)


def test_independent_non_node_values_include_both_sides_and_boundaries():
    m = grid()
    for e, h in [
        (0.0012, -0.0071),
        (-0.0087, 0.0031),
        (0.0042, 0.0057),
        (-0.0025, -0.0095),
        (0.0, 0.0042),
        (0.01, -0.01),
        (0.0, 0.0),
    ]:
        np.testing.assert_allclose(m(e, h), analytic(e, h), atol=2e-12, rtol=1e-13)
    np.testing.assert_allclose(
        m.rows(np.array([1]))(0.003, -0.005), analytic(0.003, -0.005)[1:], atol=2e-12
    )
    assert np.isnan(np.asarray(m(0.01001, 0))).all()


def test_traced_derivatives_match_analytic_values_away_from_cut_knot():
    m = grid()
    for e, h in [(0.003, -0.004), (-0.003, 0.004)]:
        x, y = e / 0.01, h / 0.01
        expected = [
            (2 + 0.9 * x * x * y - 0.5 * np.sign(x)) / 0.01,
            (2 * y + 0.3 * x**3 + 0.2 * np.sign(y)) / 0.01,
        ]
        gradient = jax.jit(jax.grad(lambda a, b: m(a, b)[0, 0], argnums=(0, 1)))(e, h)
        np.testing.assert_allclose(gradient, expected, rtol=1e-11, atol=1e-9)


def test_nonconserving_or_malformed_grid_is_rejected():
    nodes = width_nodes()
    columns = np.array([[analytic(e, h) for h in nodes] for e in nodes])
    columns[0, 0, 0, -1] = 0.99
    with pytest.raises(ValueError, match="conserve"):
        fit_coefficients(nodes, columns)
    with pytest.raises(ValueError):
        width_nodes(True)
