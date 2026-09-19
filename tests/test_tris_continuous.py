import numpy as np
import pytest

from examples.inference.tris_continuous import (
    harmonic_basis,
    harmonic_scales,
    map_moments,
    spectral_rule,
)


def test_real_harmonics_are_orthonormal_and_monopole_is_one():
    _, _, weights, basis = spectral_rule(5)
    np.testing.assert_allclose(basis[:, 0], 1, atol=2e-15)
    np.testing.assert_allclose(
        basis.T @ (weights[:, None] * basis), np.eye(36), atol=2e-14
    )


def test_basis_is_periodic_and_single_valued_at_poles():
    theta = np.linspace(0, np.pi, 20)
    phi = np.linspace(-4, 8, 20)
    np.testing.assert_allclose(
        harmonic_basis(theta, phi, 3),
        harmonic_basis(theta, phi + 2 * np.pi, 3),
        atol=1e-14,
    )
    for pole in (0.0, np.pi):
        np.testing.assert_allclose(
            harmonic_basis(pole, phi, 3),
            np.broadcast_to(harmonic_basis(pole, 0, 3), (20, 16)),
            atol=1e-14,
        )


def test_transform_matches_direct_pixel_sum_including_imaginary_modes():
    hp = pytest.importorskip("healpy")
    theta, phi = hp.pix2ang(8, np.arange(hp.nside2npix(8)))
    values = np.random.default_rng(710).normal(size=len(theta))
    expected = values @ harmonic_basis(theta, phi, 5)
    np.testing.assert_allclose(map_moments(values, 5), expected, atol=2e-12, rtol=1e-13)


def test_prior_pointwise_variance_is_rotation_invariant():
    theta = np.linspace(0.1, 3, 24)
    basis = harmonic_basis(theta, theta * 3, 3)
    scales = harmonic_scales(3, 0.3, 0.2)
    variance = np.sum((basis * scales) ** 2, axis=-1)
    np.testing.assert_allclose(
        variance, 0.3**2 + 0.2**2 * sum(1 / l**2 for l in range(1, 4)), atol=1e-14
    )


def test_bad_basis_and_underresolved_rule_are_rejected():
    for lmax in (-1, 1.5, True):
        with pytest.raises(ValueError):
            harmonic_basis(0.0, 0.0, lmax)
    with pytest.raises(ValueError):
        harmonic_basis(-0.1, 0.0, 2)
    with pytest.raises(ValueError):
        spectral_rule(4, order=4)
