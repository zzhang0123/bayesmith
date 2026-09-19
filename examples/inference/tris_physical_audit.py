"""Reproduce D11 component omission, TRIS 2428 and 1420 input diagnostics.

No downloads and no real-data parameter fit. The synthetic likelihood has
declared ideal calibration and independent 25-mK noise weights, not TRIS errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from examples.inference.tris_data_contract import archive_facts
from examples.inference.tris_physical_sky import continuum_components, rj_blackbody

FREQUENCIES = np.array([408.0, 600.5, 817.8, 1420.0, 2427.8])


def fixed_background(frequency_mhz):
    """D09/D10 condition: final-2011 total non-CMB isotropic baseline, RJ K."""
    return 24.1 * (np.asarray(frequency_mhz) / 310.0) ** -2.599


def single_cell_sky(parameters, frequency=FREQUENCIES):
    """Coordinates = (ln A408, beta, kappa, ln EM); fixed Te=7000 K."""
    loga, beta, curve, logem = parameters
    return continuum_components(
        frequency_mhz=frequency, synchrotron_408=[np.exp(loga)], beta=beta,
        curvature=curve, emission_measure=np.exp(logem), electron_temperature=7000,
        source_408=[0], source_beta=-2.7, solid_angle=[4 * np.pi],
        isotropic_rj=fixed_background(frequency), synchrotron_front_fraction=1,
    )


def omission_experiment(emission_measure):
    """One noiseless injection; compare correct components with an omitted ff law."""
    truth = np.array([np.log(25), -2.8, 0, np.log(emission_measure)])
    parts = single_cell_sky(truth)
    observed = parts['total'][:, 0]

    def evaluate(p, include_ff):
        return single_cell_sky(p if include_ff else [*p, -np.inf])['total'][:, 0]

    fits = {}
    for include_ff in (True, False):
        lower = [np.log(0.1), -4.5, -1, np.log(0.01)]
        upper = [np.log(1000), -1.0, 1, np.log(1e4)]
        if not include_ff:
            lower, upper = lower[:3], upper[:3]
        attempts = []
        for initial_em in ((3.0, 30.0, 300.0, 1000.0) if include_ff else (30.0,)):
            start = np.array([np.log(20), -2.7, 0.02, np.log(initial_em)])
            attempts.append(least_squares(
                lambda p, include_ff=include_ff: (evaluate(p, include_ff) - observed) / 0.025,
                start if include_ff else start[:3], bounds=(lower, upper), max_nfev=4000,
                ftol=1e-13, xtol=1e-13, gtol=1e-13,
            ))
        successful = [attempt for attempt in attempts if attempt.success]
        if not successful:
            raise RuntimeError('all injection optimizer starts failed')
        fit = min(successful, key=lambda attempt: attempt.cost)
        _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
        covariance = (vt.T / singular**2) @ vt
        fits['physical' if include_ff else 'omit_freefree'] = {
            'success': bool(fit.success),
            'coordinates': ['ln_A408', 'beta', 'kappa', 'ln_EM'][:len(fit.x)],
            'parameters': fit.x.tolist(),
            'optimizer_attempts': [
                {'success': bool(attempt.success), 'cost': float(attempt.cost),
                 'parameters': attempt.x.tolist(),
                 'active_bounds': attempt.active_mask.tolist()}
                for attempt in attempts
            ],
            'max_abs_residual_k': float(np.max(np.abs(evaluate(fit.x, include_ff) - observed))),
            'residual_k': (evaluate(fit.x, include_ff) - observed).tolist(),
            'jacobian_singular_values': singular.tolist(),
            'jacobian_condition_number': float(singular.max() / singular.min()),
            'local_fisher_kappa_sd': float(np.sqrt(covariance[2, 2])),
            'local_fisher_kappa_logem_correlation': (
                float(covariance[2, 3] / np.sqrt(covariance[2, 2] * covariance[3, 3]))
                if include_ff else None
            ),
        }
    return {
        'truth': {'A408_k': 25.0, 'beta': -2.8, 'kappa': 0.0,
                  'EM_cm6_pc': float(emission_measure), 'Te_k': 7000.0},
        'frequency_mhz': FREQUENCIES.tolist(),
        'budget_k': {key: value[:, 0].tolist() for key, value in parts.items()
                     if value.ndim == 2 and key != 'optical_depth'},
        'fits': fits,
        'scope': 'Noiseless one-cell synthetic injection; ideal known calibration; fixed Te/RSB; '
                 'independent 0.025 K objective weights. Fisher SD is local, not a posterior.',
    }


def pointset_feasibility(points):
    """Necessary temperature floors under two explicitly restricted hypotheses."""
    frequency = points['effective_frequency_mhz']
    temperature = np.array(points['temperature_k'])
    bound = points['common_zero_level_k']
    floors = {'cmb_only': float(rj_blackbody(frequency)),
              'cmb_plus_fixed_isotropic': float(rj_blackbody(frequency) + fixed_background(frequency))}
    return {
        'frequency_mhz': frequency, 'ra_deg': points['ra_deg'],
        'temperature_k': temperature.tolist(),
        'assumed_common_offset_interval_k': [-bound, bound],
        'assumption': 'Envelope only: treat quoted systematic magnitude as a hard bound. '
                      'Nonnegative Galactic components, Te well above radio sky, '
                      'no explicit selected-source fluctuation template. '
                      'Not an absolute rejection or measured per-row significance.',
        'floors_k': floors,
        'minimum_deficit_k': {key: np.maximum(floor - bound - temperature, 0).tolist()
                              for key, floor in floors.items()},
        'minimum_offset_magnitude_k': {key: float(max(0, floor - temperature.min()))
                                      for key, floor in floors.items()},
        'per_row_sigma_sensitivity_k': [0.010, 0.025, 0.103, 0.103 * np.sqrt(len(temperature))],
        'sigma_warning': 'All four per-row assignments are hypothetical. Paper I Table12 '
                         'has 25/10 mK summaries; Paper II Table7 has 103 mK experimental '
                         'statistical contribution to its CMB analysis. sqrt(6)*103 mK '
                         'tests a mean-of-six interpretation, not a measured error map.',
    }


def read_stockert_map(path):
    """Read this specific LAMBDA release, preserving invalid coverage as NaN.

    Units absent in file are supplied by the official product page (mK TB).
    Negative absolute continuum brightness is invalid, including finite -32768.
    """
    import healpy as hp
    from astropy.io import fits

    with fits.open(path) as hdus:
        header = dict(hdus[1].header)
        raw = np.asarray(hdus[1].data.field(0), float).ravel()
    if (header.get('NSIDE') != 256 or header.get('COORDSYS', '').strip() != 'G'
            or header.get('ORDERING', '').strip() != 'NESTED'
            or raw.size != hp.nside2npix(256)):
        raise ValueError('not the declared Galactic NESTED nside256 Stockert/Villa release')
    unit = header.get('TUNIT1', '').strip()
    if unit not in ('', 'mK'):
        raise ValueError(f'unexpected map unit {unit!r}; re-audit the product')
    valid = np.isfinite(raw) & (raw > 0)
    temperature = np.where(valid, raw / 1000, np.nan)
    metadata = {
        'input_path': str(Path(path).resolve()),
        'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        'nside': 256, 'input_ordering': 'NESTED', 'output_ordering': 'RING',
        'coordinates': 'Galactic', 'input_unit_header': unit or None,
        'input_unit_external': 'mK full-beam brightness temperature',
        'unit_source': 'https://lambda.gsfc.nasa.gov/product/foreground/fg_stockert_villa_info.html',
        'invalid_pixels': int((~valid).sum()),
        'finite_invalid_values': np.unique(raw[~valid & np.isfinite(raw)]).tolist(),
        'valid_range_k': [float(temperature[valid].min()), float(temperature[valid].max())],
        'invalid_policy': 'NaN plus explicit valid mask; never infer coverage from isfinite alone',
        'nominal_frequency_mhz': 1420.0,
        'effective_bandpass': None,
        'bandpass_warning': 'Villa Elisa includes 1435-MHz/14-MHz and 1420-MHz/13-MHz '
                            'HI-rejected observations; released mixing weights unknown.',
        'observation_ancestors': ['stockert_1982_1986', 'villa_elisa_2001'],
        'calibration_groups': ['reich_1420_scale'],
        'shared_accuracy_summary': {'gain_fraction': 0.05, 'zero_k': 0.5},
        'accuracy_warning': 'Published accuracy summaries, not independent pixel sigmas.',
        'independent_likelihood_admitted': False,
    }
    return hp.reorder(temperature, n2r=True), hp.reorder(valid, n2r=True), metadata


def map_beam_support(temperature, valid, archive, nside):
    """Partial measured integral and missing support, never a filled-sky prediction."""
    from limTOD.tris import read_tris_beam_cuts
    from limTOD.tris.geometry import tris_zenith_geometry

    from examples.inference.tris_quadrature import integrate_feature_maps

    facts = archive_facts(archive)
    ra = np.array(facts['point_set']['ra_deg'])
    # Full ring plus the actual six absolute pointings (not nearest RA bins).
    from examples.inference.tris_data_contract import parse_ra_deg, read_table
    _, rows = read_table(archive / 'TRIS_absolute_600.txt')
    ra = np.concatenate([[parse_ra_deg(row[0]) for row in rows], ra])
    feature = np.column_stack([np.where(valid, temperature, 0), (~valid).astype(float)])
    beam = read_tris_beam_cuts(archive / 'TRIS_Beam_Profile.txt')
    projected = integrate_feature_maps(
        beam, tris_zenith_geometry(ra), feature, nside=nside, coord='G', progress=True,
    )[0]
    return {
        'quadrature_nside': nside, 'ra_deg': ra.tolist(),
        'partial_valid_integral_k': projected[:, 0].tolist(),
        'missing_beam_weight': projected[:, 1].tolist(),
        'warning': 'Partial integral excludes missing brightness; do not fit as a complete sky. '
                   'Beam includes nominal archived cuts and horizon, composed with native map smoothing.',
    }


def plot_diagnostics(summary, output):
    """Standalone scientific figure; needs matplotlib only when explicitly called."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
    for row in summary['injections']:
        fit = row['fits']['omit_freefree']
        axes[0].plot(row['frequency_mhz'], np.array(fit['residual_k']) * 1000, 'o-',
                     label=f"EM={row['truth']['EM_cm6_pc']:.0f}, kappa={fit['parameters'][2]:.3f}")
    axes[0].axhline(0, color='grey', lw=0.8)
    axes[0].set(xscale='log', xlabel='Frequency [MHz]', ylabel='Fit - truth [mK]',
                title='Omitted free-free mimics curvature')
    axes[0].legend(fontsize=8)
    axes[0].text(0.02, 0.04, 'Injected synchrotron kappa = 0\nToy noise weight = 25 mK',
                 transform=axes[0].transAxes, fontsize=8)
    points = summary['tris2428']
    axes[1].plot(points['ra_deg'], points['temperature_k'], 'ko', label='TRIS 2427.8 MHz')
    floor = points['floors_k']['cmb_plus_fixed_isotropic']
    lower, upper = points['assumed_common_offset_interval_k']
    axes[1].axhline(floor, color='#225ea8', label='CMB + fixed isotropic floor')
    axes[1].axhspan(floor + lower, floor + upper, color='#225ea8', alpha=0.12,
                    label='Assumed common +/-284 mK envelope')
    axes[1].set(xlabel='Right ascension [deg]', ylabel='RJ brightness [K]',
                title='Necessary floor, not a significance test')
    axes[1].legend(fontsize=7, loc='lower right')
    for support in summary['stockert_beam_support']:
        axes[2].plot(support['ra_deg'][:120], np.array(support['missing_beam_weight'][:120]) * 1000,
                     label=f"Quadrature nside={support['quadrature_nside']}")
    axes[2].set(xlabel='Right ascension [deg]', ylabel='Missing beam weight [parts per thousand]',
                title='1420 MHz missing coverage')
    axes[2].legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--stockert', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--beam-nside', type=int, nargs='+', default=[256, 512])
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    temperature, valid, map_metadata = read_stockert_map(args.stockert)
    np.savez_compressed(args.output / 'stockert_prepared.npz', temperature_rj_k=temperature,
                        valid=valid)
    facts = archive_facts(args.archive)
    summary = {
        'schema': 'tris.physical_implementation.v1',
        'recorded_at_utc': datetime.now(UTC).isoformat(),
        'real_data_fit_performed': False,
        'injections': [omission_experiment(em) for em in (15.0, 100.0, 300.0)],
        'tris2428': pointset_feasibility(facts['point_set']),
        'stockert': map_metadata,
        'stockert_beam_support': [map_beam_support(temperature, valid, args.archive, nside)
                                 for nside in args.beam_nside],
        'input_sha256': {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [args.stockert, *sorted(args.archive.glob('TRIS*.txt'))]},
        'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ['tris_physical_audit.py', 'tris_physical_sky.py',
                                     'tris_physical_contract.py', 'tris_sky_components.py',
                                     'tris_data_contract.py', 'tris_quadrature.py',
                                     'tris_audit.py', 'tris_beam_identifiability.py']},
        'limtod_sha256': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for name, module in list(sys.modules.items())
                          if name.startswith('limTOD.tris') and getattr(module, '__file__', None)},
        'versions': {package: version(package) for package in ['numpy', 'scipy', 'healpy', 'astropy']},
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'stockert_invalid': map_metadata['invalid_pixels'],
                      'injected_kappa': [row['fits']['omit_freefree']['parameters'][2]
                                         for row in summary['injections']]}, indent=2))


if __name__ == '__main__':
    main()
