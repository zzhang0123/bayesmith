"""P0 acceptance budget: independent integration, two-frequency vectors, sensitivities.

The P0c ladder in tris_audit_report.py stored only the RMS and max of each rung
against the production control, and compared the high-resolution Haslam branch
by two separate RMS values against the fiducial.  Two small RMS values do not
show that two prediction vectors are close: the vectors can differ in shape and
cancel in the summary statistic.  This module stores the difference VECTOR in
per-point sigma units, its cumulative Gaussian log-likelihood change and its
data-conditioned log-likelihood change, and it validates the integration against
an independent pixel-space oracle that shares no code with limTOD's rotation
helper.

The pre-registered budget is *adjacent*: rung h is compared with rung h-1, and
the parameter shift is tested against the marginal POSTERIOR SD, not against the
prior width (they are different scales).  The lmax branch truncates the beam alm
with an ell mask built from healpy's own index tables, because healpy packs alm
with m varying slowest and a flat prefix keeps high-ell coefficients of low m
while dropping low-ell coefficients of high m.  The 0.000 K archive rows are
identified from the archive (the bundle replaces them with the 0.004 K floor),
at each frequency separately.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from examples.inference.tris_audit import (
    control_operator,
    independent_operator,
    pointed_operator_from_alm,
    pointed_operator_from_beam_map,
)
from examples.inference.tris_audit_report import linearized_region_shift

NL = chr(10)
RING_FILES = ("TRIS_absolute_600.txt", "TRIS_absolute_820.txt")


def vector_budget(control, trial, sky, sigma):
    """Difference vector of two operators, in kelvin and in per-point sigma."""
    delta = (np.asarray(trial, float) - np.asarray(control, float)) @ np.asarray(sky, float)
    sigma = np.asarray(sigma, float)
    normalized = delta / sigma
    return {
        "prediction_rms_k": float(np.sqrt(np.mean(delta**2))),
        "prediction_max_abs_k": float(np.max(np.abs(delta))),
        "prediction_rms_sigma": float(np.sqrt(np.mean(normalized**2))),
        "prediction_max_abs_sigma": float(np.max(np.abs(normalized))),
        "delta_chi_square": float(np.sum(normalized**2)),
        # A likelihood DIFFERENCE cancels the sigma-dependent normalization and
        # the 2*pi constant exactly, so neither may appear here.  Identical
        # predictions must return exactly 0.0; the old expression added the
        # constants back and returned 7.37 for control == trial.
        "gaussian_log_likelihood_delta": float(-0.5 * np.sum(normalized**2)),
    }


def data_weighted_likelihood(control, trial, data, sky, sigma):
    """Log-likelihood change from swapping the prediction, at fixed data."""
    sigma = np.asarray(sigma, float)
    data = np.asarray(data, float)
    reference = (data - np.asarray(control, float) @ np.asarray(sky, float)) / sigma
    trial_residual = (data - np.asarray(trial, float) @ np.asarray(sky, float)) / sigma
    return float(-0.5 * trial_residual @ trial_residual + 0.5 * reference @ reference)


def two_frequency_budget(control, trial, sky_600, sigma_600, sky_820, sigma_820):
    """Per-frequency vector budget plus the combined Gaussian likelihood."""
    first = vector_budget(control, trial, sky_600, sigma_600)
    second = vector_budget(control, trial, sky_820, sigma_820)
    return {
        "600_5": first,
        "817_8": second,
        "combined_delta_chi_square": first["delta_chi_square"] + second["delta_chi_square"],
        "combined_gaussian_log_likelihood_delta": (
            first["gaussian_log_likelihood_delta"] + second["gaussian_log_likelihood_delta"]
        ),
    }


def passes_budget(entry, *, rms_limit=0.1, max_limit=0.3):
    return bool(
        entry["prediction_rms_sigma"] < rms_limit
        and entry["prediction_max_abs_sigma"] < max_limit
    )


def adjacent_budget(
    control,
    trial,
    *,
    sky_600,
    sigma_600,
    data_600,
    sky_820,
    sigma_820,
    data_820,
    template,
    region,
    rms_limit=0.1,
    max_limit=0.3,
    posterior_sd_limit=0.1,
):
    """Vector, data-conditioned and parameter budgets for one adjacent rung pair.

    The pre-registered budget is *adjacent*: it asks whether moving one rung of
    the beam-resolution ladder changes the prediction by less than the stated
    fraction of a ring sigma.  Comparing every rung with the fixed nside-8
    control answers a different question and flattens the convergence.
    """
    entry = two_frequency_budget(control, trial, sky_600, sigma_600, sky_820, sigma_820)
    entry["data_weighted_log_likelihood_delta_600_5"] = data_weighted_likelihood(
        control, trial, data_600, sky_600, sigma_600
    )
    entry["data_weighted_log_likelihood_delta_817_8"] = data_weighted_likelihood(
        control, trial, data_820, sky_820, sigma_820
    )
    entry["data_weighted_log_likelihood_delta"] = (
        entry["data_weighted_log_likelihood_delta_600_5"]
        + entry["data_weighted_log_likelihood_delta_817_8"]
    )
    entry["shift"] = linearized_region_shift(
        control, sigma_600, template, region, 600.5, (trial - control) @ sky_600
    )
    entry["budget_600_5_pass"] = passes_budget(
        entry["600_5"], rms_limit=rms_limit, max_limit=max_limit
    )
    entry["budget_817_8_pass"] = passes_budget(
        entry["817_8"], rms_limit=rms_limit, max_limit=max_limit
    )
    entry["budget_adjacent_pass"] = bool(
        entry["budget_600_5_pass"] and entry["budget_817_8_pass"]
    )
    entry["budget_posterior_sd_pass"] = bool(
        entry["shift"]["max_delta_over_posterior_sd"] < posterior_sd_limit
    )
    entry["budget_pass"] = bool(
        entry["budget_adjacent_pass"] and entry["budget_posterior_sd_pass"]
    )
    return entry


def _branch_summary(control, trial, sky, sigma, template, region):
    budget = vector_budget(control, trial, sky, sigma)
    budget["shift"] = linearized_region_shift(
        control, sigma, template, region, 600.5, (trial - control) @ sky
    )
    return budget


def lmax_from_alm_size(size):
    """The healpy lmax of a packed alm vector of the given length."""
    size = int(size)
    lmax = round((-3.0 + np.sqrt(1.0 + 8.0 * size)) / 2.0)
    if (lmax + 1) * (lmax + 2) // 2 != size:
        raise ValueError(f"{size} is not a packed healpy alm length")
    return lmax


def alm_ell(alm):
    """The ell of every coefficient in a packed healpy alm array."""
    import healpy as hp

    size = np.asarray(alm).shape[-1]
    ell, _ = hp.Alm.getlm(lmax_from_alm_size(size))
    return ell


def truncate_alm(alm, lmax):
    """Zero the healpy alm coefficients with ell > lmax, on a copy.

    healpy packs alm with m varying slowest: m = 0 holds ell = 0..lmax_src, m =
    1 holds ell = 1..lmax_src, and so on.  A flat prefix of length
    (lmax+1)(lmax+2)/2 therefore keeps high-ell coefficients of the low m and
    drops low-ell coefficients of the high m -- measured on an all-ones
    lmax-383 array truncated to 47 it kept 1008 coefficients with ell > 47 and
    dropped 1008 with ell <= 47.  The ell of each coefficient has to come from
    healpy's own index tables.
    """
    truncated = np.array(alm, copy=True)
    keep = alm_ell(truncated) <= int(lmax)
    if truncated.ndim == 1:
        truncated[~keep] = 0.0
    else:
        truncated[..., ~keep] = 0.0
    return truncated


def sensitivity_branches(
    cuts,
    control,
    geometry,
    make_geometry,
    pixels,
    nside,
    sky,
    sigma,
    template,
    region,
    *,
    horizontal_mask,
    lmaxes,
    base_offset_deg,
    lmax_nside=128,
):
    """Beam, lmax, rotation, horizon and latitude branches, as difference vectors."""
    import healpy as hp
    from limTOD.tris.beam import tris_cut_beam_map

    branches = {}

    def operator_for(geo, *, mask, blend="db", lmax=None):
        beam = tris_cut_beam_map(cuts, nside=nside, blend=blend, normalization="peak")
        if lmax is not None:
            return pointed_operator_from_alm(
                hp.map2alm(beam, lmax=lmax), geo, pixels, nside, horizontal_mask=mask
            )
        return pointed_operator_from_beam_map(beam, geo, pixels, nside, horizontal_mask=mask)

    branches["latitude_42_0"] = _branch_summary(
        control,
        operator_for(make_geometry(latitude_deg=42.0), mask=horizontal_mask),
        sky,
        sigma,
        template,
        region,
    )
    branches["latitude_42_26"] = _branch_summary(
        control,
        operator_for(make_geometry(latitude_deg=42.43333333333333), mask=horizontal_mask),
        sky,
        sigma,
        template,
        region,
    )
    for blend in ("db", "power"):
        branches[f"blend_{blend}"] = _branch_summary(
            control,
            operator_for(geometry, mask=horizontal_mask, blend=blend),
            sky,
            sigma,
            template,
            region,
        )
    for delta_roll in (-0.5, 0.5):
        rolled = make_geometry(
            e_plane_east_of_meridian_deg=base_offset_deg - delta_roll
        )
        branches[f"selfrot_{delta_roll:+.1f}"] = _branch_summary(
            control, operator_for(rolled, mask=horizontal_mask), sky, sigma, template, region
        )
    branches["horizon_off"] = _branch_summary(
        control, operator_for(geometry, mask=None), sky, sigma, template, region
    )
    # The lmax axis must be varied at a FIXED, high beam resolution: at the
    # nside-8 control, lmax is already capped at 3*nside-1 = 23, so passing 47
    # or 191 there measures alm ringing, not a truncation choice.  Compare the
    # full beam alm at lmax_nside with its ell-masked truncations.
    beam_lmax = tris_cut_beam_map(cuts, nside=lmax_nside, normalization="peak")
    full_alm = hp.map2alm(beam_lmax)
    reference = pointed_operator_from_alm(
        full_alm, geometry, pixels, nside, horizontal_mask=horizontal_mask
    )
    for lmax in lmaxes:
        branches[f"lmax_{lmax}"] = _branch_summary(
            reference,
            pointed_operator_from_alm(
                truncate_alm(full_alm, lmax),
                geometry,
                pixels,
                nside,
                horizontal_mask=horizontal_mask,
            ),
            sky,
            sigma,
            template,
            region,
        )
    return branches


def independent_check(beam_map, geometry, pixels, nside, *, beam_nside, horizontal_mask):
    """Production operator vs the pixel-space oracle, as a difference vector."""
    production = pointed_operator_from_beam_map(
        beam_map, geometry, pixels, nside, horizontal_mask=horizontal_mask
    )
    oracle = independent_operator(beam_map, geometry, pixels, nside, beam_nside=beam_nside)
    delta = oracle - production
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta**2))),
        "relative_rms": float(np.linalg.norm(delta) / np.linalg.norm(production)),
    }


def archive_zero_rows(archive):
    """Bundle frequency index -> archive row index whose sigma is exactly 0.0."""
    from limTOD.tris import read_tris_ring

    rows = {}
    for index, name in enumerate(RING_FILES):
        ring = read_tris_ring(Path(archive) / name)
        sigma = np.asarray(ring.statistical_uncertainty_k, dtype=float)
        zero = np.flatnonzero(sigma == 0.0)
        if zero.size != 1:
            raise ValueError(f"{name}: expected exactly one 0.000 K row, found {zero.size}")
        rows[index] = int(zero[0])
    return rows


def zero_sigma_row_budget(payload, *, rows):
    """Sensitivity to each archive 0.000 K row, at its own frequency.

    The rows mapping maps the frequency index (0 = 600.5 MHz, 1 = 817.8 MHz) to
    the archive row index whose published statistical uncertainty is 0.000 K.
    The prepared bundle replaces those rows with the 0.004 K floor, so the row
    can only be identified from the archive, not by looking for sigma == 0 in
    the bundle (where every floored row also reads 0.004 K).
    """
    per_frequency = {}
    combined_keep = 0.0
    combined_drop = 0.0
    frequencies = np.asarray(payload["frequency_mhz"], dtype=float)
    for frequency, row in sorted(rows.items()):
        a = np.asarray(payload[f"operator_{frequency}"], dtype=float)
        data = np.asarray(payload[f"data_k_{frequency}"], dtype=float)
        sigma = np.asarray(payload[f"sigma_k_{frequency}"], dtype=float)
        prior = np.asarray(payload[f"prior_k_{frequency}"], dtype=float)
        row = int(row)
        if not 0 <= row < sigma.size:
            raise ValueError(f"row {row} is outside frequency {frequency}")
        residual = (data - a @ prior) / sigma
        keep_chi = float(residual @ residual)
        drop = np.ones(sigma.size, dtype=bool)
        drop[row] = False
        dropped = (data[drop] - a[drop] @ prior) / sigma[drop]
        drop_chi = float(dropped @ dropped)
        per_frequency[str(frequency)] = {
            "row_index": row,
            "frequency_mhz": float(frequencies[frequency]),
            "sigma_k": float(sigma[row]),
            "chi_square_keep": keep_chi,
            "chi_square_drop": drop_chi,
            "delta_chi_square": drop_chi - keep_chi,
        }
        combined_keep += keep_chi
        combined_drop += drop_chi
    return {
        "per_frequency": per_frequency,
        "chi_square_keep": combined_keep,
        "chi_square_drop": combined_drop,
        "delta_chi_square": combined_drop - combined_keep,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ladder", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--oracle-nside", type=int, default=256)
    parser.add_argument("--lmax", type=int, nargs="+", default=[47, 95, 191])
    args = parser.parse_args(argv)

    import healpy as hp
    from limTOD.tris import read_tris_beam_cuts, read_tris_ring
    from limTOD.tris.beam import tris_cut_beam_map, tris_horizon_mask
    from limTOD.tris.geometry import (
        TRIS_E_PLANE_EAST_OF_MERIDIAN_DEG,
        tris_zenith_geometry,
    )

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = np.load(args.maps, allow_pickle=False)
    nside = int(payload["nside"])
    ring = read_tris_ring(args.archive / RING_FILES[0])
    cuts = read_tris_beam_cuts(args.archive / "TRIS_Beam_Profile.txt")
    geometry = tris_zenith_geometry(ring.ra_deg)
    pixels = np.arange(hp.nside2npix(nside))
    mask = tris_horizon_mask(nside)
    sky_600 = np.asarray(payload["prior_k_0"], dtype=float)
    sigma_600 = np.asarray(payload["sigma_k_0"], dtype=float)
    data_600 = np.asarray(payload["data_k_0"], dtype=float)
    sky_820 = np.asarray(payload["prior_k_1"], dtype=float)
    sigma_820 = np.asarray(payload["sigma_k_1"], dtype=float)
    data_820 = np.asarray(payload["data_k_1"], dtype=float)
    template = np.asarray(payload["template_k"], dtype=float)
    region = np.asarray(payload["region"], dtype=np.int64)

    beam8 = tris_cut_beam_map(cuts, nside=nside, normalization="peak")
    control = control_operator(
        beam8, geometry, pixels, nside, nside_hires=64, horizontal_mask=mask
    )

    operators = {}
    ladder = {}
    for rung in args.ladder:
        beam = tris_cut_beam_map(cuts, nside=rung, normalization="peak")
        operator = pointed_operator_from_beam_map(
            beam, geometry, pixels, nside, horizontal_mask=mask
        )
        operators[int(rung)] = operator
        budget = two_frequency_budget(
            control, operator, sky_600, sigma_600, sky_820, sigma_820
        )
        budget["budget_600_5_pass"] = passes_budget(budget["600_5"])
        budget["budget_817_8_pass"] = passes_budget(budget["817_8"])
        budget["shift"] = linearized_region_shift(
            control, sigma_600, template, region, 600.5, (operator - control) @ sky_600
        )
        ladder[str(rung)] = budget

    adjacent = {}
    for low, high in itertools.pairwise(sorted(operators)):
        adjacent[f"{low}->{high}"] = adjacent_budget(
            operators[low],
            operators[high],
            sky_600=sky_600,
            sigma_600=sigma_600,
            data_600=data_600,
            sky_820=sky_820,
            sigma_820=sigma_820,
            data_820=data_820,
            template=template,
            region=region,
        )

    oracle = independent_check(
        tris_cut_beam_map(cuts, nside=args.oracle_nside, normalization="peak"),
        geometry,
        pixels,
        nside,
        beam_nside=args.oracle_nside,
        horizontal_mask=mask,
    )

    sensitivities = sensitivity_branches(
        cuts,
        control,
        geometry,
        functools.partial(tris_zenith_geometry, ring.ra_deg),
        pixels,
        nside,
        sky_600,
        sigma_600,
        template,
        region,
        horizontal_mask=mask,
        lmaxes=args.lmax,
        base_offset_deg=TRIS_E_PLANE_EAST_OF_MERIDIAN_DEG,
    )

    zero_rows = archive_zero_rows(args.archive)
    document = {
        "schema": "bayesmith.tris.p0-budget.v1",
        "run_id": args.output.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": subprocess.check_output(
            ["git", "-C", ".", "rev-parse", "HEAD"], text=True
        ).strip(),
        "ladder": ladder,
        "adjacent": adjacent,
        "independent_oracle": oracle,
        "sensitivities": sensitivities,
        "zero_sigma_row": zero_sigma_row_budget(payload, rows=zero_rows),
        "zero_sigma_archive_rows": zero_rows,
        "thresholds": {
            "adjacent_rms_sigma": 0.1,
            "adjacent_max_sigma": 0.3,
            "param_shift_posterior_sd": 0.1,
        },
    }
    document["elapsed_s"] = round(time.time() - started, 2)
    (args.output / "p0_budget.json").write_text(
        json.dumps(document, indent=2, default=float) + NL
    )
    print(json.dumps({"output": str(args.output), "elapsed_s": document["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
