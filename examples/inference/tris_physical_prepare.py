"""Prepare explicit coarse-cell measurements for the D12 physical joint pilot.

Native maps are compressed as measurements, not reused as synchrotron priors.
The sky-cell approximation is measured against native-map beam integrals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.inference.tris_continuous import harmonic_basis, harmonic_scales
from examples.inference.tris_data_contract import (
    archive_facts,
    parse_ra_deg,
    read_table,
)
from examples.inference.tris_halpha import (
    cygnus_a_flux_jy,
    flux_to_rj_integral,
    optical_error_allocation,
    optical_selection,
)
from examples.inference.tris_physical_audit import fixed_background, read_stockert_map

FREQUENCY_MHZ = np.array([408., 600.5, 817.8, 1420., 2427.8])
CYGNUS_A_RA_DEC = (299.8683, 40.7338)  # SIMBAD ICRS, degrees; point approximation.


def read_nested(path, unit):
    import healpy as hp
    from astropy.io import fits

    with fits.open(path) as f:
        header = f[1].header
        value = np.array(f[1].data.field(0)).ravel()
    if (header.get('NSIDE') != 512 or header.get('ORDERING', '').strip() != 'NESTED'
            or header.get('TUNIT1', '').strip() != unit or value.size != hp.nside2npix(512)):
        raise ValueError(f'unexpected optical/dust product contract: {path}')
    if not np.all(np.isfinite(value)):
        raise ValueError(f'nonfinite values require a new product audit: {path}')
    # Galactic coordinates are specified by LAMBDA, absent from these FITS headers.
    return hp.reorder(value, n2r=True)


def compress_optical(intensity, error, flags, ebv, nside, dust_fraction):
    import healpy as hp

    theta, _ = hp.pix2ang(512, np.arange(intensity.size))
    valid = optical_selection(intensity, error, flags, ebv, 90 - np.rad2deg(theta))
    fraction = hp.ud_grade(valid.astype(float), nside)
    selected = np.flatnonzero(fraction >= 0.8)

    def mean(values):
        return hp.ud_grade(np.where(valid, values, 0), nside)[selected] / fraction[selected]

    measured, envelope = mean(intensity), mean(error)
    # Mean of attenuation, not attenuation of mean dust; EM is constant in a cell.
    attenuation = mean(np.exp(-0.4 * np.log(10) * 2.51 * dust_fraction * ebv))
    sigma = optical_error_allocation(measured, envelope)
    return {
        'ha_index': selected, 'ha_data': measured, 'ha_sigma': sigma,
        'ha_attenuation': attenuation, 'ha_valid_fraction': fraction,
        'ha_envelope': envelope,
    }, {
        'native_selected_fraction': float(valid.mean()),
        'coarse_observations': int(selected.size),
        'raw_floor_value_r': float(intensity.min()),
        'raw_floor_count': int((intensity == intensity.min()).sum()),
        'mask': 'WHAM; reject bits16/32/64/128; abs(b)>10; E(B-V)<0.2; I>0.11R; coverage>=0.8',
        'uncertainty': 'mean linear envelope minus 0.1*I; allocate 0.03R common zero out of '
                       'remaining variance; no sqrt(N) reduction; add declared 20% tracer-model discrepancy',
        'dust_fraction': dust_fraction,
        'dust_law': 'A_Halpha=2.51 E(B-V), Dickinson2003 eq1; condition on SFD',
        'em_convention': 'integral n_e*n_H+ dl; radio optical-depth EM multiplied by 1.08',
    }


def beam_cells(cuts, geometry, nside, quadrature_nside, native_features):
    """Integrate response per cell, exact source direction, and native comparisons."""
    import healpy as hp
    from limTOD.tris.beam import tris_cut_beam_response

    from examples.inference.tris_audit import _pointing_frame

    theta, phi = hp.pix2ang(quadrature_nside, np.arange(hp.nside2npix(quadrature_nside)))
    parent = hp.ang2pix(nside, theta, phi)
    native = np.stack([hp.ud_grade(field, quadrature_nside) for field in native_features], axis=1)
    eqtheta, eqphi = hp.Rotator(coord=['G', 'C'])(theta, phi)
    directions = hp.ang2vec(eqtheta, eqphi)
    source = hp.ang2vec(*CYGNUS_A_RA_DEC, lonlat=True)
    axes = _pointing_frame(geometry.lst_deg, geometry.latitude_deg, geometry.azimuth_deg,
                           geometry.elevation_deg, geometry.selfrot_deg)
    operator = np.empty((len(geometry.lst_deg), hp.nside2npix(nside)))
    point = np.empty(len(operator))
    native_projection = np.empty((len(operator), native.shape[1]))
    area = hp.nside2pixarea(quadrature_nside)
    for row in range(len(operator)):
        bore, e, h, zenith = (a[row] for a in axes)
        angle = np.rad2deg(np.arccos(np.clip(directions @ bore, -1, 1)))
        azimuth = np.rad2deg(np.arctan2(directions @ h, directions @ e))
        weights = tris_cut_beam_response(cuts, angle, azimuth) * (directions @ zenith >= 0)
        normalization = weights.sum()
        operator[row] = np.bincount(parent, weights=weights, minlength=operator.shape[1]) / normalization
        native_projection[row] = weights @ native / normalization
        angle_s = np.rad2deg(np.arccos(np.clip(source @ bore, -1, 1)))
        azimuth_s = np.rad2deg(np.arctan2(source @ h, source @ e))
        point[row] = (tris_cut_beam_response(cuts, angle_s, azimuth_s)
                      * (source @ zenith >= 0) / (normalization * area))
        if row % 30 == 0:
            print(f'beam nside={nside} quadrature={quadrature_nside} row={row}/{len(operator)}', flush=True)
    return operator, point, native_projection


def prepare(args):
    import healpy as hp
    from limTOD.tris import read_tris_beam_cuts
    from limTOD.tris.geometry import tris_zenith_geometry

    if not hp.isnsideok(args.nside) or not hp.isnsideok(args.quadrature_nside):
        raise ValueError('valid HEALPix resolutions required')
    args.output.mkdir(parents=True, exist_ok=False)
    h = hp.read_map(args.haslam, dtype=float)
    if h.size != hp.nside2npix(512) or not np.all(np.isfinite(h) & (h > 0)):
        raise ValueError('expected positive native nside512 Haslam DS map')
    i, e, m, dust = [read_nested(args.optical / name, unit) for name, unit in (
        ('halpha.fits', 'R'), ('halpha_error.fits', 'R'), ('halpha_mask.fits', ''), ('ebv.fits', 'magnitudes'))]
    optical, optical_meta = compress_optical(i, e, m, dust, args.nside, args.dust_fraction)
    facts = archive_facts(args.archive)
    tables = [read_table(args.archive / name)[1] for name in ('TRIS_absolute_600.txt', 'TRIS_absolute_820.txt')]
    ra = np.array([parse_ra_deg(row[0]) for row in tables[0]])
    ra2 = np.array([parse_ra_deg(row[0]) for row in tables[1]])
    if not np.array_equal(ra, ra2):
        raise ValueError('TRIS rings must be aligned for this pilot')
    all_ra = np.concatenate([ra, facts['point_set']['ra_deg']])
    data = np.array([[float(row[1]) for row in table] for table in tables]).T
    sigma = np.maximum(np.array([[float(row[2]) for row in table] for table in tables]).T, .004)
    source_theta, source_phi = hp.Rotator(coord=['C', 'G'])(
        np.deg2rad(90 - CYGNUS_A_RA_DEC[1]), np.deg2rad(CYGNUS_A_RA_DEC[0]))
    theta, phi = hp.pix2ang(512, np.arange(h.size))
    separation = np.arccos(np.clip(hp.ang2vec(theta, phi) @ hp.ang2vec(source_theta, source_phi), -1, 1))
    width = np.deg2rad(56 / 60) / np.sqrt(8 * np.log(2))
    source_h = np.exp(-.5 * (separation / width)**2)
    source_h /= source_h.sum() * hp.nside2pixarea(512)
    stockert_width = np.deg2rad(35.4 / 60) / np.sqrt(8 * np.log(2))
    source_stockert = np.exp(-.5 * (separation / stockert_width)**2)
    source_stockert /= source_stockert.sum() * hp.nside2pixarea(512)
    q = flux_to_rj_integral(FREQUENCY_MHZ, cygnus_a_flux_jy(FREQUENCY_MHZ))
    beam, source_response, native_projection = beam_cells(
        read_tris_beam_cuts(args.archive / 'TRIS_Beam_Profile.txt'), tris_zenith_geometry(all_ra),
        args.nside, args.quadrature_nside, [h, source_h],
    )
    coarse_h, coarse_source = (hp.ud_grade(field, args.nside) for field in (h, source_h))
    coarse_error = beam @ (coarse_h - q[0] * coarse_source) - (native_projection[:, 0] - q[0] * native_projection[:, 1])
    stockert, stockert_valid, stockert_meta = read_stockert_map(args.stockert)
    stockert_coverage = hp.ud_grade(stockert_valid.astype(float), args.nside)
    stockert_mean = hp.ud_grade(np.where(stockert_valid, stockert, 0), args.nside)
    stockert_index = np.flatnonzero(stockert_coverage > 1 - 1e-12)
    coarse_theta, coarse_phi = hp.pix2ang(args.nside, np.arange(coarse_h.size))
    arrays = {
        **optical, 'haslam': coarse_h, 'haslam_sigma': np.full(coarse_h.size, args.haslam_sigma),
        'source_stockert': hp.ud_grade(source_stockert, args.nside),
        'source_haslam': coarse_source, 'source_q': q, 'source_response': source_response,
        'source_pixel': hp.ang2pix(args.nside, source_theta, source_phi),
        'beam': beam, 'beta_basis': harmonic_basis(coarse_theta, coarse_phi, 2),
        'beta_sd': harmonic_scales(2, .3, .15), 'data': data, 'sigma': sigma,
        'ra_deg': ra, 'all_ra_deg': all_ra, 'frequency_mhz': FREQUENCY_MHZ,
        'background': fixed_background(FREQUENCY_MHZ), 'coarse_error_408_k': coarse_error,
        'stockert_index': stockert_index, 'stockert_data': stockert_mean[stockert_index],
        'heldout2428_data': np.array(facts['point_set']['temperature_k']),
        'nside': args.nside,
    }
    np.savez_compressed(args.output / 'data.npz', **arrays)
    paths = [args.haslam, args.stockert, *sorted(args.optical.glob('*.fits')),
             *sorted(args.archive.glob('TRIS*.txt'))]
    manifest = {
        'schema': 'tris.physical_joint_data.v1', 'nside': args.nside,
        'quadrature_nside': args.quadrature_nside,
        'haslam_role': 'one coarse-cell total-temperature measurement; never a pure synchrotron prior',
        'haslam_sigma_k': args.haslam_sigma,
        'tris_sigma_floor_k': .004,
        'tris_sigma_floor_basis': 'declared minimum for rounded archived noise values; fitted floor is additional',
        'haslam_sigma_basis': 'declared coarse-cell residual-noise approximation, not a measured error map',
        'sky_approximation': 'piecewise constant beam-effective diffuse components per cell; '
                             'native survey PSF boundary mixing neglected; refine and diagnose before science claims',
        'halpha': optical_meta, 'stockert': stockert_meta,
        'stockert_role': 'held-out coarse-map diagnostic only; native band mixing unresolved',
        'tris2428_role': 'held-out only; no assumed per-row sigma in training',
        'cygnus_a': {'ra_dec_icrs_deg': CYGNUS_A_RA_DEC,
                     'source': 'Perley & Butler arXiv:1609.05940v1 Table5; SIMBAD position',
                     'selection': 'Cygnus A only; remaining EG source anisotropy unmodelled',
                     'calibration': 'conditional spectrum, wide amplitude nuisance; no independent flux likelihood',
                     'haslam_fwhm_arcmin': 56, 'stockert_fwhm_arcmin': 35.4,
                     'rsb_bookkeeping': 'subtract source_q/(4pi) from B_iso at every frequency'},
        'coarse_diffuse_projection_error_408_k': {
            'rms': float(np.sqrt(np.mean(coarse_error**2))), 'max_abs': float(np.max(np.abs(coarse_error)))},
        'input_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), Path(__file__).with_name('tris_halpha.py'))},
    }
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'halpha': optical_meta,
                      'coarse_error': manifest['coarse_diffuse_projection_error_408_k']}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'haslam', 'stockert', 'optical', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--nside', type=int, default=8)
    parser.add_argument('--quadrature-nside', type=int, default=256)
    parser.add_argument('--dust-fraction', type=float, default=1 / 3)
    parser.add_argument('--haslam-sigma', type=float, default=1.0)
    args = parser.parse_args(argv)
    if (not np.isfinite(args.dust_fraction) or not 0 <= args.dust_fraction <= 1
            or not np.isfinite(args.haslam_sigma) or args.haslam_sigma <= 0):
        parser.error('dust fraction in [0,1] and positive map sigma required')
    prepare(args)


if __name__ == '__main__':
    main()
