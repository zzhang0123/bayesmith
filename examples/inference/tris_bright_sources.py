"""Conditional Cas A flux scenarios; never subtract these from observations.

Trotter et al. 2017, arXiv:1704.00002, Table 5: separate segment reference
times and the shared frequency-dependent time term. Segment selection is
explicit: this function does not invent transition times or marginalize epochs.
The published statistical errors are not an independent flux likelihood here.
"""

import numpy as np

CAS_A_GALACTIC_DEG = (111.7376, -2.1345)
CAS_SEGMENTS = {
    1: (3.3731, -0.003884, 1963.9),
    2: (3.3173, -0.00227, 1983.8),
    3: (3.2530, -0.00350, 2006.9),
}


def cas_a_flux_jy(frequency_mhz, epoch, *, segment):
    frequency, epoch = np.broadcast_arrays(
        np.asarray(frequency_mhz, float), np.asarray(epoch, float)
    )
    if (
        segment not in CAS_SEGMENTS
        or not np.all(np.isfinite(frequency))
        or not np.all(np.isfinite(epoch))
        or np.any((frequency < 22) | (frequency > 22000))
        or np.any((epoch < 1950) | (epoch > 2017))
    ):
        raise ValueError(
            "explicit segment 1..3, epoch1950..2017, frequency22..22000MHz required"
        )
    log_f, rate, reference_epoch = CAS_SEGMENTS[segment]
    x = np.log10(frequency / 1477)
    return 10 ** (
        log_f
        - 0.732 * x
        - 0.0094 * x**2
        + 0.0053 * x**3
        + rate * (epoch - reference_epoch)
        + 0.00124 * np.log10(frequency / 1315) * (epoch - 2005.64)
    )


def rj_integral(frequency_mhz, flux_jy):
    return (
        np.asarray(flux_jy)
        * 1e-26
        * 299792458.0**2
        / (2 * 1.380649e-23 * (np.asarray(frequency_mhz) * 1e6) ** 2)
    )


def source_operators(arrays, archive):
    """Exact direction in nominal TRIS beam; native512 Haslam PSF integral."""
    import healpy as hp
    from astropy.coordinates import SkyCoord
    from limTOD.tris import read_tris_beam_cuts
    from limTOD.tris.beam import tris_cut_beam_response
    from limTOD.tris.geometry import tris_zenith_geometry

    from examples.inference.tris_audit import _pointing_frame

    l, b = CAS_A_GALACTIC_DEG
    nside = int(arrays["nside"])
    theta, phi = hp.pix2ang(512, np.arange(hp.nside2npix(512)))
    direction = hp.ang2vec(l, b, lonlat=True)
    separation = np.arccos(np.clip(hp.ang2vec(theta, phi) @ direction, -1, 1))
    width = np.deg2rad(56 / 60) / np.sqrt(8 * np.log(2))
    psf = np.exp(-0.5 * (separation / width) ** 2)
    psf /= psf.sum() * hp.nside2pixarea(512)
    haslam = hp.ud_grade(psf, nside)
    geometry = tris_zenith_geometry(arrays["all_ra_deg"])
    axes = _pointing_frame(
        geometry.lst_deg,
        geometry.latitude_deg,
        geometry.azimuth_deg,
        geometry.elevation_deg,
        geometry.selfrot_deg,
    )
    eq = SkyCoord(l, b, unit="deg", frame="galactic").icrs
    point = hp.ang2vec(eq.ra.deg, eq.dec.deg, lonlat=True)
    # The D12/D13 beam was integrated at256;512 is the survey PSF grid.
    # Check the existing CygA response below rather than trusting this label.
    theta, phi = hp.pix2ang(256, np.arange(hp.nside2npix(256)))
    eqtheta, eqphi = hp.Rotator(coord=["G", "C"])(theta, phi)
    vectors = hp.ang2vec(eqtheta, eqphi)
    cuts = read_tris_beam_cuts(archive / "TRIS_Beam_Profile.txt")
    response = []
    cyg = hp.ang2vec(299.8683, 40.7338, lonlat=True)
    cyg_response = []
    for bore, e, h, zenith in zip(*axes, strict=True):
        angle = np.rad2deg(np.arccos(np.clip(vectors @ bore, -1, 1)))
        azimuth = np.rad2deg(np.arctan2(vectors @ h, vectors @ e))
        normalization = np.sum(
            tris_cut_beam_response(cuts, angle, azimuth) * (vectors @ zenith >= 0)
        ) * hp.nside2pixarea(256)
        angle_s = np.rad2deg(np.arccos(np.clip(point @ bore, -1, 1)))
        azimuth_s = np.rad2deg(np.arctan2(point @ h, point @ e))
        response.append(
            tris_cut_beam_response(cuts, angle_s, azimuth_s)
            * (point @ zenith >= 0)
            / normalization
        )
        cyg_angle = np.rad2deg(np.arccos(np.clip(cyg @ bore, -1, 1)))
        cyg_azimuth = np.rad2deg(np.arctan2(cyg @ h, cyg @ e))
        cyg_response.append(
            tris_cut_beam_response(cuts, cyg_angle, cyg_azimuth)
            * (cyg @ zenith >= 0)
            / normalization
        )
    if not np.allclose(cyg_response, arrays["source_response"], atol=1e-14, rtol=2e-12):
        raise ValueError("Galactic source and frozen beam normalizations differ")
    return haslam, np.array(response)
