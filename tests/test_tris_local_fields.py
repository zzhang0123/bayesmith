"""Fixed-node geometry and explicit variable-source epoch contract."""

import numpy as np
import pytest

from examples.inference.tris_bright_sources import cas_a_flux_jy
from examples.inference.tris_local_fields import (
    field_basis,
    interpolation,
    node_loading,
)


def test_node_prior_reconstructs_projected_angular_covariance():
    loading, covariance = node_loading(2, [(1.0, 30.0), (0.5, 7.0)])
    n = len(loading)
    project = np.eye(n) - np.ones((n, n)) / n
    np.testing.assert_allclose(
        loading @ loading.T, project @ covariance @ project, atol=2e-14, rtol=2e-13
    )
    np.testing.assert_allclose(loading.sum(0), 0, atol=2e-14)
    assert np.linalg.matrix_rank(loading) == n - 1


def test_interpolation_preserves_constants_and_never_extrapolates_node_range():
    import healpy as hp

    theta = np.r_[0.0, np.pi, np.linspace(0.1, 3.0, 80)]
    phi = np.linspace(-0.4, 6.7, len(theta))
    weights = interpolation(2, theta, phi)
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-14, rtol=0)
    assert np.all(weights >= 0)
    assert np.all(np.count_nonzero(weights, axis=1) <= 4)
    values = np.random.default_rng(51).normal(size=hp.nside2npix(2))
    answer = weights @ values
    assert np.all((answer >= values.min()) & (answer <= values.max()))
    np.testing.assert_allclose(
        answer, hp.get_interp_val(values, theta, phi), atol=1e-14
    )
    np.testing.assert_allclose(
        interpolation(2, theta, phi + 2 * np.pi), weights, atol=2e-14
    )
    theta_grid, phi_grid = hp.pix2ang(16, np.arange(hp.nside2npix(16)))
    boundary = interpolation(4, theta_grid, phi_grid)
    assert np.all(boundary >= 0)
    np.testing.assert_allclose(boundary.sum(1), 1, atol=1e-14, rtol=0)


def test_same_field_at_same_direction_does_not_depend_on_quadrature_batch():
    loading, _ = node_loading(2, [(1.0, 30.0), (0.5, 7.0)])
    t, p = np.array([0.4, 1.7]), np.array([0.7, 4.1])
    basis = field_basis(2, t, p, loading)
    other = field_basis(
        2,
        np.r_[t, np.linspace(0, np.pi, 50)],
        np.r_[p, np.linspace(0, 2 * np.pi, 50)],
        loading,
    )
    np.testing.assert_allclose(basis, other[:2], atol=1e-14, rtol=0)
    np.testing.assert_array_equal(basis[:, 0], 1)


def test_invalid_node_priors_and_coordinates_refuse():
    for scales in ([], [(0, 5)], [(1, -2)], [(1, np.nan)]):
        with pytest.raises(ValueError):
            node_loading(2, scales)
    with pytest.raises(ValueError):
        node_loading(16, [(1, 10)])
    with pytest.raises(ValueError):
        interpolation(2, [-0.1], [0])


def test_cas_a_uses_log10_rates_and_two_reference_times():
    # At1477MHz the spectral polynomial is zero; at2005.64 the shared
    # frequency-dependent temporal term is zero. Table5 gives the rest.
    expected = 10 ** (3.2530 - 0.00350 * (2005.64 - 2006.9))
    assert cas_a_flux_jy(1477, 2005.64, segment=3) == pytest.approx(expected, rel=1e-14)
    nu = np.array([408.0, 600.5, 817.8])
    ratio = cas_a_flux_jy(nu, 2001, segment=3) / cas_a_flux_jy(nu, 1998, segment=3)
    np.testing.assert_allclose(
        ratio, 10 ** (3 * (-0.00350 + 0.00124 * np.log10(nu / 1315))), rtol=1e-14
    )
    assert np.all(ratio < 1)
    assert np.all(
        cas_a_flux_jy(nu, 1970, segment=1) > cas_a_flux_jy(nu, 2000, segment=3)
    )
    for kwargs in ({"epoch": 2026, "segment": 3}, {"epoch": 2000, "segment": 0}):
        with pytest.raises(ValueError):
            cas_a_flux_jy(nu, **kwargs)
