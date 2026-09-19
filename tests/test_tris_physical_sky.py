"""Physical contracts, not numerical outputs from the old conditional model."""

import numpy as np
import pytest

from examples.inference.tris_physical_sky import (
    calibrated_prediction,
    calibration_covariance,
    continuum_components,
    observe_rj,
    rj_blackbody,
    thermodynamic_from_rj,
)


def sky(**updates):
    fields = {
        'frequency_mhz': np.array([408.0, 817.8, 2427.8]),
        'synchrotron_408': np.array([10.0, 20.0]),
        'beta': np.array([-2.8, -2.6]),
        'curvature': 0.0,
        'emission_measure': np.array([1.0, 2.0]),
        'electron_temperature': 7000.0,
        'source_408': np.array([0.0, 2.0]),
        'source_beta': np.array([-2.7, -2.4]),
        'solid_angle': np.array([1.0, 3.0]) * np.pi,
        'isotropic_rj': np.array([12.0, 2.0, 0.1]),
        'synchrotron_front_fraction': 1.0,
    }
    fields.update(updates)
    return continuum_components(**fields)


def test_haslam_is_total_measurement_not_synchrotron_amplitude():
    parts = sky()
    np.testing.assert_array_equal(parts['synchrotron'][0], [10, 20])
    np.testing.assert_allclose(
        parts['total'],
        sum(parts[name] for name in ('synchrotron', 'free_free', 'cmb',
                                     'isotropic', 'source_fluctuation')),
    )
    assert np.all(parts['total'][0] > [10, 20])
    observed = observe_rj(parts['total'], np.eye(3), np.tile(np.eye(2), (3, 1, 1)))
    expected = calibrated_prediction(observed, gain=1.05, offset_k=3.0)
    np.testing.assert_allclose(expected, 1.05 * parts['total'] + 3)


def test_source_mean_is_removed_at_each_frequency_before_absorption():
    parts = sky(emission_measure=0.0)
    np.testing.assert_allclose(parts['source_fluctuation'] @ np.array([0.25, 0.75]),
                               0.0, atol=2e-16)
    original_sources = np.array([0.0, 2.0])[None] * (
        np.array([408.0, 817.8, 2427.8])[:, None] / 408
    ) ** np.array([-2.7, -2.4])
    np.testing.assert_allclose(parts['source_fluctuation'] + parts['source_mean'][:, None],
                               original_sources)
    # A uniform source with a common spectrum belongs to the same B_iso budget.
    a = sky(source_408=[1.0, 3.0], source_beta=-2.7, emission_measure=0.0)
    b = sky(source_408=[6.0, 8.0], source_beta=-2.7, emission_measure=0.0)
    np.testing.assert_allclose(a['total'], b['total'])
    absorbed = sky(emission_measure=[0, 1e20])
    # Foreground absorption makes a previously isotropic/centered field anisotropic.
    # Re-centering after the screen would remove a real observable signal.
    assert (absorbed['source_fluctuation'][0] @ np.array([0.25, 0.75])) < 0


def test_absorption_limits_include_background_and_keep_foreground_synchrotron():
    thin = sky(emission_measure=0.0, synchrotron_front_fraction=0.25)
    np.testing.assert_array_equal(thin['free_free'], np.zeros((3, 2)))
    thick = sky(emission_measure=1e20, synchrotron_front_fraction=0.25)
    assert np.all(thick['free_free'] == 7000)
    assert np.all(thick['cmb'] == 0) and np.all(thick['isotropic'] == 0)
    np.testing.assert_allclose(thick['synchrotron'], 0.25 * thin['synchrotron'])
    np.testing.assert_allclose(thick['total'], 7000 + thick['synchrotron'])


def test_absolute_temperature_roundtrip_is_not_differential_kcmb_conversion():
    frequency = np.array([408.0, 2427.8, 30000.0, 90000.0])
    brightness = rj_blackbody(frequency, 2.7255)
    np.testing.assert_allclose(thermodynamic_from_rj(frequency, brightness), 2.7255,
                               rtol=2e-15)
    assert brightness[-1] < 1.2  # Absolute blackbody RJ; not K_CMB fluctuation units.
    assert abs(brightness[0] - 2.7255) > 0.009


def test_bandpass_and_beam_are_applied_to_each_component_before_adding():
    parts = sky()
    beam = np.array([[[0.2, 0.8]], [[0.5, 0.5]], [[0.7, 0.3]]])
    bands = np.array([[0.2, 0.3, 0.5], [1, 0, 0]])
    forward = observe_rj(parts['total'], bands, beam)
    summed = sum(observe_rj(parts[name], bands, beam) for name in (
        'synchrotron', 'free_free', 'cmb', 'isotropic', 'source_fluctuation'))
    np.testing.assert_allclose(forward, summed)
    # Spectrum is evaluated per sky cell, not by exponentiating a mean beta.
    syn = sky(emission_measure=0, isotropic_rj=[0, 0, 0], source_408=[0, 0],
              synchrotron_408=[15, 15], beta=[-2.2, -3.2])
    actual = observe_rj(syn['synchrotron'], np.eye(3), np.full((3, 1, 2), 0.5))[:, 0]
    wrong = 15 * (np.array([408, 817.8, 2427.8]) / 408) ** -2.7
    # For equal amplitudes the exact mixture has a cosh factor (Jensen gap).
    np.testing.assert_allclose(actual / wrong,
                               np.cosh(0.5 * np.log(np.array([408, 817.8, 2427.8]) / 408)))
    assert actual[-1] > wrong[-1]


def test_shared_zero_uncertainty_does_not_shrink_with_number_of_rows():
    for size in (2, 6, 60):
        covariance = calibration_covariance(np.full(size, 0.025),
                                             np.ones((size, 1)), np.array([[0.284**2]]))
        mean_variance = covariance.sum() / size**2
        assert mean_variance == pytest.approx(0.284**2 + 0.025**2 / size)


def test_calibration_mode_includes_gain_and_matches_explicit_gaussian_integration():
    model = np.array([4.0, 7.0, 11.0])
    design = np.column_stack([np.ones(3), model])
    covariance = calibration_covariance([0.1, 0.2, 0.3], design,
                                        np.diag([0.5**2, 0.05**2]))
    np.testing.assert_allclose(covariance, np.diag([0.01, 0.04, 0.09])
                               + 0.25 + 0.0025 * np.outer(model, model))
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_invalid_sky_and_measurement_contracts_are_rejected():
    for updates in ({'frequency_mhz': [0, 817.8, 2427.8]},
                    {'synchrotron_408': [-1, 20]}, {'solid_angle': [0, 0]},
                    {'solid_angle': [1, 1]},
                    {'synchrotron_front_fraction': 1.1}, {'emission_measure': -1},
                    {'isotropic_rj': [0.1, 0.1, 0.1]}):
        with pytest.raises(ValueError):
            sky(**updates)
    with pytest.raises(ValueError):
        observe_rj(np.ones((2, 2)), np.array([[1, 1]]), np.ones((2, 1, 2)))
    with pytest.raises(ValueError):
        calibration_covariance([0.1, 0.1], np.ones((2, 1)), [[-1]])
    with pytest.raises(ValueError):
        calibrated_prediction([1], gain=-1)
