"""Reference generative continuum sky and calibrated total-power measurements.

All components are monochromatic RJ K. Amplitudes are latent quantities, never
computed here from an observed Haslam map. This NumPy reference supports the
408--2428 MHz component-closure experiments; it is not a new NUTS model.
The screen geometry is explicit, not a general Galactic transfer solution.
AME, dust, polarization and native map-making require additional operators.
"""

from __future__ import annotations

import numpy as np

from examples.inference.tris_sky_components import free_free_optical_depth

H_OVER_K_MHZ = 6.62607015e-34 / 1.380649e-23 * 1e6


def _finite(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive(value, name, *, zero=False):
    array = _finite(value, name)
    if np.any(array < 0 if zero else array <= 0):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}")
    return array


def rj_blackbody(frequency_mhz, temperature_k=2.7255):
    """Absolute Planck brightness in RJ K, not differential K_CMB units."""
    frequency = _positive(frequency_mhz, 'frequency')
    temperature = _positive(temperature_k, 'temperature')
    tnu = H_OVER_K_MHZ * frequency
    return tnu / np.expm1(tnu / temperature)


def thermodynamic_from_rj(frequency_mhz, temperature_rj):
    """Inverse absolute Planck law at a single frequency; not a band conversion."""
    frequency = _positive(frequency_mhz, 'frequency')
    brightness = _positive(temperature_rj, 'RJ temperature')
    tnu = H_OVER_K_MHZ * frequency
    return tnu / np.log1p(tnu / brightness)


def continuum_components(
    *, frequency_mhz, synchrotron_408, beta, curvature, emission_measure,
    electron_temperature, source_408, source_beta, solid_angle, isotropic_rj,
    synchrotron_front_fraction, cmb_temperature_k=2.7255,
):
    """Return (frequency, sky-cell) components, their sum, and source/opacity audit.

    Synchrotron = A408 exp(beta*x + curvature*x**2/2), x=ln(nu/408).
    A408 is the *unabsorbed* Galactic amplitude. ``solid_angle`` must describe
    the full-sky source selection, not a survey-footprint mean. ``isotropic_rj``
    includes the selected extragalactic sources' all-sky mean. Sources are
    recentered separately at each frequency, before the foreground screen.
    The caller must supply the assumed fraction of Galactic synchrotron in
    front of the isothermal free-free screen; CMB and extragalactic terms are
    behind it. A varying optical depth can therefore give them anisotropy.
    """
    frequency = _positive(frequency_mhz, 'frequency')
    amplitude = _positive(synchrotron_408, 'synchrotron amplitude', zero=True)
    if frequency.ndim != 1 or not frequency.size or amplitude.ndim != 1 or not amplitude.size:
        raise ValueError('nonempty frequency vector and sky-cell amplitude vector required')

    def field(value, name):
        return np.broadcast_to(_finite(value, name), amplitude.shape)

    index, curve = field(beta, 'beta'), field(curvature, 'curvature')
    em = field(emission_measure, 'EM')
    te = field(electron_temperature, 'electron temperature')
    source = field(source_408, 'source amplitude')
    source_index = field(source_beta, 'source index')
    front = field(synchrotron_front_fraction, 'front fraction')
    area = _positive(solid_angle, 'solid angle')
    background = _positive(isotropic_rj, 'isotropic brightness', zero=True)
    if area.shape != amplitude.shape or background.shape != frequency.shape:
        raise ValueError('one solid angle per cell and one isotropic value per frequency required')
    if not np.isclose(area.sum(), 4 * np.pi, rtol=0, atol=1e-10):
        raise ValueError('source budget requires full-sky solid angles summing to 4*pi sr')
    if np.any(source < 0) or np.any((front < 0) | (front > 1)):
        raise ValueError('physical source brightness must be nonnegative and front fraction in [0,1]')
    x = np.log(frequency[:, None] / 408.0)
    intrinsic_syn = amplitude * np.exp(index * x + curve * x**2 / 2)
    source_absolute = source * np.exp(source_index * x)
    source_mean = source_absolute @ (area / area.sum())
    source_fluctuation = source_absolute - source_mean[:, None]
    # B_iso contains more than the explicitly selected source population.
    if np.any(background < source_mean):
        raise ValueError('isotropic budget is smaller than the selected source mean')
    tau = free_free_optical_depth(frequency[:, None] / 1000, em, te)
    transmission = np.exp(-tau)
    parts = {
        'synchrotron': intrinsic_syn * (front + (1 - front) * transmission),
        'free_free': -te * np.expm1(-tau),
        'cmb': rj_blackbody(frequency, cmb_temperature_k)[:, None] * transmission,
        'isotropic': background[:, None] * transmission,
        'source_fluctuation': source_fluctuation * transmission,
    }
    parts['total'] = sum(parts.values())
    if not all(np.all(np.isfinite(value)) for value in parts.values()):
        raise ValueError('nonfinite spectrum; check spectral coefficients and frequency range')
    parts['source_mean'] = source_mean
    parts['optical_depth'] = tau
    return parts


def observe_rj(sky_rj, band_weights, beam_weights):
    """Apply a supplied chromatic beam then a bandpass calibrated for RJ K.

    Shapes: sky=(frequency,pixel), beam=(frequency,row,pixel),
    band=(channel,frequency); output=(channel,row). Rows are normalized
    *by the caller*, including solid angles and the actual map/beam convention.
    Missing coverage must be handled explicitly, not by silent renormalization.
    K_CMB/intensity-calibrated bands need a different unit response.
    """
    sky = _finite(sky_rj, 'sky')
    band = _positive(band_weights, 'band weights', zero=True)
    beam = _positive(beam_weights, 'beam weights', zero=True)
    if (sky.ndim != 2 or band.ndim != 2 or beam.ndim != 3
            or band.shape[1] != sky.shape[0]
            or beam.shape[0] != sky.shape[0] or beam.shape[2] != sky.shape[1]
            or not all(sky.shape) or not band.shape[0] or not beam.shape[1]):
        raise ValueError('unaligned sky, band and beam dimensions')
    if (not np.allclose(band.sum(-1), 1, rtol=0, atol=1e-12)
            or not np.allclose(beam.sum(-1), 1, rtol=0, atol=1e-12)):
        raise ValueError('RJ band and beam weights must each sum to one')
    return band @ np.einsum('frp,fp->fr', beam, sky)


def calibrated_prediction(prediction_rj, *, gain=1.0, offset_k=0.0):
    """Observed = gain * physical prediction + offset, in the product's RJ units.

    For old correction variables defined as physical-observed, offset=-zero.
    A shared scalar acts on every row. Do not also marginalize that same mode
    into a covariance matrix in an explicit-nuisance likelihood.
    """
    return (_positive(gain, 'gain') * _finite(prediction_rj, 'prediction')
            + _finite(offset_k, 'offset'))


def calibration_covariance(statistical_sigma, design, mode_covariance):
    """Marginal covariance diag(sigma**2) + X Lambda X.T for Gaussian modes.

    A constant column is a common zero; the physical prediction can be a gain
    column. Bounds are not Gaussian SDs. This routine is deliberately not used
    for the TRIS uniform-bound calibration branches. Include the covariance's
    parameter-dependent log determinant when using a predicted gain column.
    """
    sigma = _positive(statistical_sigma, 'statistical sigma')
    x = _finite(design, 'calibration design')
    modes = _finite(mode_covariance, 'mode covariance')
    if (sigma.ndim != 1 or not sigma.size or x.ndim != 2
            or x.shape[0] != sigma.size or modes.shape != (x.shape[1], x.shape[1])
            or not np.allclose(modes, modes.T, rtol=0, atol=1e-14)):
        raise ValueError('unaligned or asymmetric calibration covariance')
    if modes.size and np.linalg.eigvalsh(modes).min() < 0:
        raise ValueError('calibration covariance must be positive semidefinite')
    return np.diag(sigma**2) + x @ modes @ x.T
