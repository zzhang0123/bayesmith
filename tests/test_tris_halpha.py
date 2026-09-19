import numpy as np
import pytest

from examples.inference.tris_halpha import (
    cygnus_a_flux_jy,
    flux_to_rj_integral,
    halpha_rayleigh_per_em,
    optical_error_allocation,
    optical_selection,
)


def test_halpha_conversion_has_photon_units_and_temperature_domain():
    # Near 10^4 K, the standard case-B conversion is about 2.75 cm^-6 pc/R.
    assert 2.7 < 1 / halpha_rayleigh_per_em(10000) < 2.8
    assert halpha_rayleigh_per_em(7000) > halpha_rayleigh_per_em(10000)
    with pytest.raises(ValueError):
        halpha_rayleigh_per_em(3000)


def test_source_flux_integrates_to_flux_not_pixel_or_beam_temperature():
    frequency = np.array([408, 600.5, 817.8, 2427.8])
    flux = cygnus_a_flux_jy(frequency)
    assert np.all(np.diff(flux) < 0)
    integral = flux_to_rj_integral(frequency, flux)
    recovered = 2 * 1.380649e-23 * (frequency * 1e6)**2 / 299792458.0**2 * integral / 1e-26
    np.testing.assert_allclose(recovered, flux, rtol=2e-15)
    assert cygnus_a_flux_jy(1000) == pytest.approx(10**3.3498)


def test_optical_mask_rejects_bad_data_without_setting_em_to_zero():
    flags = np.array([1, 9, 17, 33, 65, 129, 4], dtype=np.uint8)
    np.testing.assert_array_equal(optical_selection(1, .3, flags, .05, 30),
                                  [True, True, False, False, False, False, False])
    assert not optical_selection(.1000061, .3, np.array(1), .05, 30)
    assert not optical_selection(1, .3, np.array(1), .5, 30)


def test_optical_calibration_removed_once_before_covariance_allocation():
    intensity, error = np.array([1., 10.]), np.array([.4, 1.3])
    independent = optical_error_allocation(intensity, error)
    np.testing.assert_allclose(independent**2 + .03**2, (error - .1 * intensity)**2)
    with pytest.raises(ValueError):
        optical_error_allocation(10, .5)
