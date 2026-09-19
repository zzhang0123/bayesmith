"""Validate the linear beam-width surrogate against the direct archive cuts.

The joint model uses the tangent operator

    A(phi_e, phi_h) = A0 + phi_e dA_e + phi_h dA_h,

whose two derivative columns are central differences of operators built from
deformed principal-plane cuts.  The second acceptance report found that the
first version hard-coded the difference step, recorded no provenance on the
operator file, and claimed beam-power non-negativity without measuring it.  This
module measures, over a grid of width points:

* the cut-based beam power: it is 10**(dB/10) at every sampled angle, so it is
  non-negative by construction -- measured here, not asserted;
* the operator's small negative entries, which are harmonic ringing of the
  band-limited alm pointing and are present identically at zero deformation;
* the surrogate's prediction error against the direct cuts, in kelvin and in
  ring sigma, at the pre-registered 0.1 / 0.3 sigma budget;
* the scalar-prediction gradient against a direct-cut central difference.

It writes the operator basis with its width-step provenance so the joint model
can verify the denominator it divides by, and a JSON + markdown record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from examples.inference.tris_audit import pointed_operator_from_beam_map
from examples.inference.tris_beam_identifiability import deformed_cuts
from examples.inference.tris_p0_budget import passes_budget, vector_budget

NL = chr(10)
# The full two-dimensional legal domain, not just the positive E axis: the
# four corners of the box, the two negative axes, interior points, the old
# positive-axis distances (shown to leave the budget), and the two neighbours
# of the gradient point.
DEFAULT_GRID = (
    (0.01, 0.0),
    (-0.01, 0.0),
    (0.0, 0.01),
    (0.0, -0.01),
    (0.01, 0.01),
    (0.01, -0.01),
    (-0.01, 0.01),
    (-0.01, -0.01),
    (0.005, 0.005),
    (0.005, -0.005),
    (0.03, 0.0),
    (0.05, 0.0),
    (0.055, 0.0),
    (0.045, 0.0),
)
TRUTH_WIDTHS = (0.008, -0.008)


def direct_operator(cuts, geometry, pixels, nside, mask, beam_nside, log_width_e, log_width_h):
    """Peak-normalized cut beam map and the pointed, row-normalized operator."""
    from limTOD.tris.beam import tris_cut_beam_map

    beam = np.asarray(
        tris_cut_beam_map(
            deformed_cuts(cuts, log_width_e, log_width_h),
            nside=beam_nside,
            normalization="peak",
        )
    )
    operator = pointed_operator_from_beam_map(
        beam, geometry, pixels, nside, horizontal_mask=mask
    )
    return beam, operator


def derivative_columns(operators, width_step):
    de = (
        np.asarray(operators["e_plus"], float) - np.asarray(operators["e_minus"], float)
    ) / (2.0 * width_step)
    dh = (
        np.asarray(operators["h_plus"], float) - np.asarray(operators["h_minus"], float)
    ) / (2.0 * width_step)
    return de, dh


def surrogate_operator(operators, width_step, log_width_e, log_width_h):
    de, dh = derivative_columns(operators, width_step)
    return np.asarray(operators["center"], float) + log_width_e * de + log_width_h * dh


def operator_summary(operator):
    row_sums = np.asarray(operator, float).sum(axis=1)
    return {
        "negative_entries": int(np.count_nonzero(operator < 0.0)),
        "minimum": float(np.min(operator)),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
    }


def grids_to_check():
    grid = [(0.0, 0.0), tuple(TRUTH_WIDTHS)]
    for point in DEFAULT_GRID:
        if point not in grid:
            grid.append(point)
    return grid


def key_of(log_width_e, log_width_h):
    return f"e{float(log_width_e):+.17g}_h{float(log_width_h):+.17g}"


def _box_passes(grid_report, bound, *, rms_limit=0.1, max_limit=0.3):
    """Whether all four corners of [-bound, bound]^2 stay inside the budget."""
    corners = (
        (bound, bound),
        (bound, -bound),
        (-bound, bound),
        (-bound, -bound),
    )
    for log_width_e, log_width_h in corners:
        key = key_of(log_width_e, log_width_h)
        if key not in grid_report:
            return False
        entry = grid_report[key]
        if not (
            passes_budget(entry["600_5"], rms_limit=rms_limit, max_limit=max_limit)
            and passes_budget(entry["817_8"], rms_limit=rms_limit, max_limit=max_limit)
        ):
            return False
    # Corners alone do not guarantee a nonlinear interpolation error is small
    # in the interior; every measured point within the box must also pass.
    for entry in grid_report.values():
        inside = max(abs(entry["log_width_e"]), abs(entry["log_width_h"])) <= bound
        passing = (passes_budget(entry["600_5"], rms_limit=rms_limit, max_limit=max_limit)
                   and passes_budget(entry["817_8"], rms_limit=rms_limit, max_limit=max_limit))
        if inside and not passing:
            return False
    return True


def legal_bound_from(grid_report, *, rms_limit=0.1, max_limit=0.3):
    """Largest box [-b, b]^2 whose four corners all pass the budget."""
    candidates = sorted(
        {
            abs(log_width_e)
            for log_width_e, _ in DEFAULT_GRID
            if abs(log_width_e) > 0.0
        }
        | {abs(log_width_h) for _, log_width_h in DEFAULT_GRID}
    )
    bound = 0.0
    for candidate in candidates:
        if _box_passes(grid_report, candidate, rms_limit=rms_limit, max_limit=max_limit):
            bound = float(candidate)
    return bound


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-nside", type=int, default=128)
    parser.add_argument("--width-step", type=float, default=1.0e-3)
    parser.add_argument("--rebuild-operators", action="store_true")
    args = parser.parse_args(argv)

    import healpy as hp
    from limTOD.tris import read_tris_beam_cuts, read_tris_ring
    from limTOD.tris.beam import tris_horizon_mask
    from limTOD.tris.geometry import tris_zenith_geometry

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = {
        name: value
        for name, value in np.load(
            args.batch_b / "audited-input" / "maps.npz", allow_pickle=False
        ).items()
    }
    nside = int(bundle["nside"])
    pixels = np.arange(hp.nside2npix(nside))
    mask = tris_horizon_mask(nside)
    station = Path("runs/tris-input-skyfields/archive")
    ring = read_tris_ring(station / "TRIS_absolute_600.txt")
    cuts = read_tris_beam_cuts(station / "TRIS_Beam_Profile.txt")
    geometry = tris_zenith_geometry(ring.ra_deg)

    from limTOD.tris.beam import tris_cut_beam_map

    half = args.width_step
    grid = grids_to_check()
    needed = list(grid)
    for point in ((half, 0.0), (-half, 0.0), (0.0, half), (0.0, -half)):
        if point not in needed:
            needed.append(point)
    # Cache every pointed operator: the cut-beam pointing is the expensive step,
    # and a late failure in the report must not cost another full rebuild.
    cache = args.output / "grid_operators.npz"
    fingerprint = hashlib.sha256()
    for value in (nside, args.beam_nside, args.width_step):
        fingerprint.update(repr(value).encode())
    for path in (args.batch_b / "audited-input" / "maps.npz",
                 station / "TRIS_absolute_600.txt", station / "TRIS_Beam_Profile.txt",
                 Path(__file__)):
        fingerprint.update(path.read_bytes())
    cache_key = fingerprint.hexdigest()
    metadata_path = args.output / "grid_operators.sha256"
    cache_matches = metadata_path.is_file() and metadata_path.read_text().strip() == cache_key
    if cache.is_file() and cache_matches and not args.rebuild_operators:
        with np.load(cache, allow_pickle=False) as data:
            operators = {name: data[name] for name in data.files}
    else:
        operators = {}
    for log_width_e, log_width_h in needed:
        key = key_of(log_width_e, log_width_h)
        if key not in operators:
            _beam, operator = direct_operator(
                cuts,
                geometry,
                pixels,
                nside,
                mask,
                args.beam_nside,
                log_width_e,
                log_width_h,
            )
            operators[key] = operator
    np.savez_compressed(cache, **operators)
    metadata_path.write_text(cache_key + "\n")
    # Rebuild the (cheap) cut beam maps for the legality report.
    beams = {
        key_of(log_width_e, log_width_h): np.asarray(
            tris_cut_beam_map(
                deformed_cuts(cuts, log_width_e, log_width_h),
                nside=args.beam_nside,
                normalization="peak",
            )
        )
        for log_width_e, log_width_h in needed
    }
    basis = {
        "center": operators[key_of(0.0, 0.0)],
        "e_plus": operators[key_of(half, 0.0)],
        "e_minus": operators[key_of(-half, 0.0)],
        "h_plus": operators[key_of(0.0, half)],
        "h_minus": operators[key_of(0.0, -half)],
    }
    np.savez_compressed(
        args.output / "operators.npz",
        **basis,
        width_step=np.asarray(args.width_step),
        beam_nside=np.asarray(args.beam_nside),
    )
    truth_key = key_of(*TRUTH_WIDTHS)
    np.savez_compressed(
        args.output / "truth_operators.npz",
        center=operators[truth_key],
        width_step=np.asarray(args.width_step),
        beam_nside=np.asarray(args.beam_nside),
        log_width_e=np.asarray(TRUTH_WIDTHS[0]),
        log_width_h=np.asarray(TRUTH_WIDTHS[1]),
    )

    sky_600 = np.asarray(bundle["prior_k_0"], float)
    sigma_600 = np.asarray(bundle["sigma_k_0"], float)
    sky_820 = np.asarray(bundle["prior_k_1"], float)
    sigma_820 = np.asarray(bundle["sigma_k_1"], float)

    grid_report = {}
    for log_width_e, log_width_h in grid:
        key = key_of(log_width_e, log_width_h)
        direct = operators[key]
        surrogate = surrogate_operator(basis, args.width_step, log_width_e, log_width_h)
        entry = {
            "log_width_e": float(log_width_e),
            "log_width_h": float(log_width_h),
            "beam_minimum": float(beams[key].min()),
            "beam_negative_entries": int(np.count_nonzero(beams[key] < 0.0)),
            "beam_sum": float(beams[key].sum()),
            "direct": operator_summary(direct),
            "surrogate": operator_summary(surrogate),
            "surrogate_minus_direct_frobenius": float(np.linalg.norm(surrogate - direct)),
            "surrogate_over_direct_relative_frobenius": float(
                np.linalg.norm(surrogate - direct) / np.linalg.norm(direct)
            ),
        }
        entry["600_5"] = vector_budget(direct, surrogate, sky_600, sigma_600)
        entry["817_8"] = vector_budget(direct, surrogate, sky_820, sigma_820)
        entry["budget_600_5_pass"] = passes_budget(entry["600_5"])
        entry["budget_817_8_pass"] = passes_budget(entry["817_8"])
        entry["budget_pass"] = bool(
            entry["budget_600_5_pass"] and entry["budget_817_8_pass"]
        )
        grid_report[key] = entry

    de, _ = derivative_columns(basis, args.width_step)
    base_operator = operators[key_of(0.05, 0.0)]
    plus_operator = operators[key_of(0.055, 0.0)]
    minus_operator = operators[key_of(0.045, 0.0)]
    base = base_operator @ sky_600
    plus = plus_operator @ sky_600
    minus = minus_operator @ sky_600
    direct_gradient = float(np.sum(plus**2) - np.sum(minus**2)) / (2.0 * 0.005)
    surrogate_gradient = float(2.0 * base @ (de @ sky_600))
    gradient = {
        "log_width_e": 0.05,
        "step": 0.005,
        "surrogate": surrogate_gradient,
        "direct": direct_gradient,
        "relative_error": abs(surrogate_gradient - direct_gradient)
        / max(abs(direct_gradient), 1e-300),
    }

    zero = grid_report[key_of(0.0, 0.0)]
    document = {
        "schema": "bayesmith.tris.beam-surrogate.v1",
        "run_id": args.output.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": subprocess.check_output(
            ["git", "-C", ".", "rev-parse", "HEAD"], text=True
        ).strip(),
        "beam_nside": args.beam_nside,
        "width_step": args.width_step,
        "grid": grid_report,
        "gradient": gradient,
        "zero_deformation_operator": zero["direct"],
        "legal_bound_recommendation": legal_bound_from(grid_report),
        "max_negative_entries_over_grid": int(
            max(entry["direct"]["negative_entries"] for entry in grid_report.values())
        ),
        "max_beam_negative_entries_over_grid": int(
            max(entry["beam_negative_entries"] for entry in grid_report.values())
        ),
        "elapsed_s": round(time.time() - started, 2),
    }
    (args.output / "beam_surrogate.json").write_text(
        json.dumps(document, indent=2, default=float) + NL
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "legal_bound_recommendation": document["legal_bound_recommendation"],
                "gradient_relative_error": gradient["relative_error"],
                "elapsed_s": document["elapsed_s"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
