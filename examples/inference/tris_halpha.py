"""Independent optical tracer and selected-source conventions for D12.

H-alpha EM means integral(n_e n_H+) dl, in cm^-6 pc. The radio Gaunt law
receives (1+y_He) EM for singly ionized helium. This is not the same numerical
EM convention as silently identifying n_e*n_H+ with n_e**2.
"""

from __future__ import annotations

import numpy as np


def halpha_rayleigh_per_em(temperature_k):
    """Dickinson+2003 equation 4 (case B), divided by one H-alpha Rayleigh.

    One Rayleigh = 1e6/(4*pi) photons cm^-2 s^-1 sr^-1, lambda=656.28 nm.
    Accurate-paper temperature range is 5000--20000 K.
    """
    te = np.asarray(temperature_k, float)
    if not np.all(np.isfinite(te) & (te >= 5000) & (te <= 20000)):
        raise ValueError('case-B temperature must lie in [5000,20000] K')
    t4 = te / 1e4
    rayleigh_erg = 1e6 / (4 * np.pi) * 6.62607015e-27 * 2.99792458e10 / 6.5628e-5
    return 9.41e-8 / rayleigh_erg * t4**-1.017 * 10**(-0.029 / t4)


def cygnus_a_flux_jy(frequency_mhz):
    """Perley & Butler arXiv:1609.05940v1 Table 5, 0.05--12 GHz.

    The archived v1 has rounded coefficients (-.225,.023,.043). This function
    does not claim to transcribe the more precise final-publication table.
    A wide source-amplitude nuisance, not an independent flux-data likelihood,
    is used in D12 because of shared radio calibration lineage.
    """
    nu = np.asarray(frequency_mhz, float)
    if not np.all(np.isfinite(nu) & (nu >= 50) & (nu <= 12000)):
        raise ValueError('Cygnus A polynomial supports 50--12000 MHz')
    return 10**np.polynomial.polynomial.polyval(
        np.log10(nu / 1000), [3.3498, -1.0022, -0.225, 0.023, 0.043]
    )


def flux_to_rj_integral(frequency_mhz, flux_jy):
    """Integral of RJ K over solid angle, K sr; not per-beam brightness."""
    nu, flux = np.broadcast_arrays(np.asarray(frequency_mhz, float), np.asarray(flux_jy, float))
    if not np.all(np.isfinite(nu) & (nu > 0) & np.isfinite(flux) & (flux >= 0)):
        raise ValueError('positive finite frequency and nonnegative flux required')
    return flux * 1e-26 * 299792458.0**2 / (2 * 1.380649e-23 * (nu * 1e6)**2)


def optical_selection(intensity, error, flags, ebv, latitude_deg):
    """Declared conservative D12 mask, independent of the radio residuals.

    Require WHAM coverage; remove SATUR/BRT_OBJ/BIG_OBJ/HI_VEL, low latitude,
    high dust and the observed archive floor. Never interpret rejected sky as
    EM=0. STAR bit alone records removal and is not itself a rejection.
    """
    i, e, m, dust, b = np.broadcast_arrays(intensity, error, flags, ebv, latitude_deg)
    if not np.issubdtype(m.dtype, np.integer):
        raise ValueError('H-alpha flags must be an integer bitmask')
    return (np.isfinite(i) & np.isfinite(e) & (e > 0) & (i > 0.11)
            & np.isfinite(dust) & (dust >= 0) & (dust < 0.2)
            & np.isfinite(b) & (np.abs(b) > 10)
            & ((m & 1) != 0) & ((m & 240) == 0))


def optical_error_allocation(intensity, error, *, calibration_fraction=0.1, zero_r=0.03):
    """Working Gaussian allocation of a *linear uncertainty envelope*.

    Remove C*I before assigning a coherent gain C. Allocate zero_r in
    quadrature out of the remainder. No sqrt(native-pixel count) reduction:
    caller passes coarse-cell averages of the envelope. This is an explicit
    covariance assumption, not the covariance published by Finkbeiner (none).
    """
    i, e = np.broadcast_arrays(np.asarray(intensity, float), np.asarray(error, float))
    if (not np.isfinite(calibration_fraction) or calibration_fraction < 0
            or not np.isfinite(zero_r) or zero_r < 0):
        raise ValueError('nonnegative finite calibration/zero scales required')
    remainder = e - calibration_fraction * i
    if not np.all(np.isfinite(remainder) & (remainder > zero_r)):
        raise ValueError('error envelope cannot support declared calibration/zero allocation')
    return np.sqrt(remainder**2 - zero_r**2)
