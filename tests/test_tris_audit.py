"""P0b audit checks: the map likelihood against an independent dense oracle."""

import numpy as np
import pytest

from examples.inference.tris_audit import (
    compress_map,
    dense_wiener,
    independent_pointed_beam,
    map_residual_norm2,
    monte_carlo_sampling_covariance,
    ring_residual_norm2,
    truncated_response_fraction,
    whitening_report,
)
from examples.inference.tris_audit_report import linearized_region_shift, region_columns


def small_system(seed=7, n=4, p=6):
    rng = np.random.default_rng(seed)
    a = 0.3 + rng.random((n, p))
    sigma = np.linspace(0.1, 0.4, n)
    prior_mean = rng.normal(size=p)
    prior_sigma = np.linspace(0.5, 2.5, p)
    data = rng.normal(size=n)
    return a, sigma, prior_mean, prior_sigma, data


def test_dense_oracle_reproduces_production_compress_map():
    a, sigma, prior_mean, prior_sigma, data = small_system()
    dense = dense_wiener(a, data, sigma, prior_mean, prior_sigma)
    likelihood = compress_map(
        dense.mean, prior_mean, a, sigma, dense.covariance
    )
    np.testing.assert_allclose(likelihood.weights, dense.weights, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(likelihood.bias, dense.bias, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        likelihood.noise_sigma**2, np.diag(dense.noise_covariance), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        dense.mean, dense.bias + dense.weights @ data, rtol=1e-10, atol=1e-12
    )


def test_sampling_covariance_is_wnwt_not_the_posterior_covariance():
    a, sigma, prior_mean, prior_sigma, _ = small_system()
    dense = dense_wiener(a, np.zeros(a.shape[0]), sigma, prior_mean, prior_sigma)
    assert np.max(np.abs(dense.covariance - dense.noise_covariance)) > 1e-3
    report = monte_carlo_sampling_covariance(
        a, sigma, prior_mean, prior_sigma, np.zeros(a.shape[1]), draws=400, seed=11
    )
    assert report["empirical_whitened_diagonal_max_deviation"] < 0.25
    assert report["empirical_whitened_offdiagonal_rms"] < 0.15
    assert report["reconstruction_covariance_relative_frobenius"] < 0.2
    assert report["reconstruction_covariance_diagonal_relative_max"] < 0.3
    assert report["measured_pixel_fraction"] == 1.0
    assert 0.5 < report["coverage_within_1sigma"] < 0.85
    assert 0.85 < report["coverage_within_2sigma"] < 1.0


def test_truncated_direction_is_flagged_and_retained_directions_are_not():
    a = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-10], [0.0, 0.0, 0.0]])
    sigma = np.ones(4)
    prior = np.zeros(3)
    prior_sigma = np.ones(3)
    dense = dense_wiener(a, np.zeros(4), sigma, prior, prior_sigma)
    likelihood = compress_map(dense.mean, prior, a, sigma, dense.covariance)
    report = truncated_response_fraction(likelihood, a, sigma, np.eye(3))
    fractions = report["discarded_fraction"]
    assert fractions[0] == pytest.approx(0.0, abs=1e-12)
    assert fractions[1] == pytest.approx(0.0, abs=1e-12)
    assert fractions[2] == pytest.approx(1.0, abs=1e-8)
    assert report["max_discarded_fraction"] == pytest.approx(1.0, abs=1e-8)


def test_ring_and_map_likelihood_agree_when_no_signal_is_truncated():
    a, sigma, prior_mean, prior_sigma, data = small_system()
    dense = dense_wiener(a, data, sigma, prior_mean, prior_sigma)
    likelihood = compress_map(dense.mean, prior_mean, a, sigma, dense.covariance)
    rng = np.random.default_rng(3)
    model0 = rng.normal(size=a.shape[1])
    model1 = rng.normal(size=a.shape[1])
    for zero0, zero1 in [(0.0, 0.0), (0.4, -0.7)]:
        ring = ring_residual_norm2(a, data, sigma, model1, zero1) - ring_residual_norm2(
            a, data, sigma, model0, zero0
        )
        compressed = map_residual_norm2(
            likelihood, model1, zero1
        ) - map_residual_norm2(likelihood, model0, zero0)
        assert compressed == pytest.approx(ring, abs=1e-8)


def test_whitening_report_counts_and_normalization():
    a, sigma, prior_mean, prior_sigma, data = small_system()
    dense = dense_wiener(a, data, sigma, prior_mean, prior_sigma)
    likelihood = compress_map(dense.mean, prior_mean, a, sigma, dense.covariance)
    report = whitening_report(likelihood)
    assert report["modes"] == a.shape[0]
    assert report["retained"] == a.shape[0]
    assert report["condition"] > 1.0
    assert report["whitened_rms"] == pytest.approx(
        float(np.sqrt(np.mean(likelihood.data**2)))
    )



def _region_fixture(frequency_mhz=600.5, beta=-2.8, seed=11):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(6, 5))
    sigma = np.linspace(0.1, 0.3, 6)
    template = np.abs(rng.normal(size=5)) + 1.0
    region = np.array([0, 0, 1, 2, 1])
    ratio = frequency_mhz / 408.0
    bases = [template * (region == r) * ratio**beta for r in range(3)]
    columns = [*bases, *[base * np.log(ratio) for base in bases]]
    return a, sigma, template, region, a @ np.array(columns).T


def _region_fixture(frequency_mhz=600.5, beta=-2.8, seed=11):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(6, 5))
    sigma = np.linspace(0.1, 0.3, 6)
    template = np.abs(rng.normal(size=5)) + 1.0
    region = np.array([0, 0, 1, 2, 1])
    columns = region_columns(template, region, frequency_mhz, beta=beta)
    return a, sigma, template, region, a @ columns


def test_region_columns_packs_amplitudes_then_betas():
    template = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    region = np.array([0, 0, 1, 2, 1])
    ratio = 600.5 / 408.0
    columns = region_columns(template, region, 600.5, beta=-2.8)
    for r in range(3):
        amplitude = template * (region == r) * ratio**-2.8
        np.testing.assert_allclose(columns[:, r], amplitude)
        np.testing.assert_allclose(columns[:, r + 3], amplitude * np.log(ratio))


def test_linearized_region_shift_amplitude_slice_holds_amplitudes_only():
    # Pin the indices with a tight prior and put the whole delta on the
    # amplitudes.  The interleaved packing this replaced returned
    # [amp0, beta0, amp1] from shift[:3], so delta_amplitude[1] would be 0
    # instead of amp1.
    a, sigma, template, region, jacobian = _region_fixture()
    truth = np.array([0.30, -0.20, 0.10, 0.0, 0.0, 0.0])
    result = linearized_region_shift(
        a, sigma, template, region, 600.5, jacobian @ truth,
        prior_width=[1e6, 1e6, 1e6, 1e-6, 1e-6, 1e-6],
    )
    np.testing.assert_allclose(result['delta_amplitude'], truth[:3], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(result['delta_beta'], np.zeros(3), atol=1e-6)
    assert result['names'][:3] == ['amplitude[0]', 'amplitude[1]', 'amplitude[2]']
    assert result['names'][3:] == ['beta[0]', 'beta[1]', 'beta[2]']


def test_linearized_region_shift_recovers_the_identifiable_direction():
    # Amplitude and index are exactly degenerate at one frequency (the index
    # columns are log(ratio) times the amplitude columns), so the required
    # property is that the fitted prediction reproduces the delta, not that the
    # split between amplitude and index is unique.
    a, sigma, template, region, jacobian = _region_fixture()
    truth = np.array([0.20, -0.10, 0.05, 0.30, -0.20, 0.10])
    delta = jacobian @ truth
    result = linearized_region_shift(
        a, sigma, template, region, 600.5, delta, prior_width=[1e6] * 6
    )
    shift = np.array(result['delta_amplitude'] + result['delta_beta'])
    np.testing.assert_allclose(jacobian @ shift, delta, rtol=1e-4, atol=1e-6)


def test_linearized_region_shift_prior_widths_are_ordered_like_the_shift():
    a, sigma, template, region, _ = _region_fixture()
    result = linearized_region_shift(
        a, sigma, template, region, 600.5, np.zeros(6)
    )
    np.testing.assert_allclose(result['prior_width'], [0.81] * 3 + [0.72] * 3)
    np.testing.assert_allclose(result['delta_over_prior_width'], np.zeros(6))


class _Geometry:
    """Minimal geometry duck for the independent integration oracle."""

    def __init__(self, lst_deg, latitude_deg, azimuth_deg, elevation_deg, selfrot_deg):
        self.lst_deg = np.asarray(lst_deg, dtype=float)
        self.latitude_deg = float(latitude_deg)
        self.azimuth_deg = np.asarray(azimuth_deg, dtype=float)
        self.elevation_deg = np.asarray(elevation_deg, dtype=float)
        self.selfrot_deg = np.asarray(selfrot_deg, dtype=float)


def _gaussian_beam(nside, sigma_deg):
    hp = pytest.importorskip("healpy")

    theta, _phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return np.exp(-0.5 * (np.rad2deg(theta) / sigma_deg) ** 2)


def test_independent_oracle_normalizes_a_constant_sky():
    hp = pytest.importorskip("healpy")

    geometry = _Geometry([0.0, 120.0], 42.44, [0.0, 0.0], [90.0, 90.0], [0.0, 0.0])
    beam = _gaussian_beam(64, 10.0)
    operator = independent_pointed_beam(beam, geometry, 8, beam_nside=64)
    np.testing.assert_allclose(operator.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_allclose(operator @ np.ones(hp.nside2npix(8)), 1.0, rtol=1e-12)


def test_independent_oracle_peak_is_the_pointed_zenith():
    hp = pytest.importorskip("healpy")

    latitude = 42.44
    geometry = _Geometry([0.0], latitude, [0.0], [90.0], [0.0])
    beam = _gaussian_beam(128, 8.0)
    operator = independent_pointed_beam(beam, geometry, 8, beam_nside=128)
    theta = np.deg2rad(90.0 - latitude)
    zenith_pixel = hp.ang2pix(8, theta, 0.0)
    assert int(np.argmax(operator[0])) == zenith_pixel


def test_independent_oracle_masks_below_the_horizon():
    hp = pytest.importorskip("healpy")

    geometry = _Geometry([37.0], 42.44, [0.0], [90.0], [0.0])
    beam = _gaussian_beam(64, 30.0)
    operator = independent_pointed_beam(
        beam, geometry, 8, beam_nside=64, min_elevation_deg=0.0
    )
    theta, phi = hp.pix2ang(8, np.arange(hp.nside2npix(8)))
    direction = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
    )
    lat = np.deg2rad(42.44)
    lst = np.deg2rad(37.0)
    z_loc = np.array([np.cos(lat) * np.cos(lst), np.cos(lat) * np.sin(lst), np.sin(lat)])
    elevation = np.rad2deg(np.arcsin(direction @ z_loc))
    assert np.all(operator[0][elevation < 0.0] == 0.0)
    assert np.any(operator[0][elevation >= 0.0] > 0.0)
