"""Run the P0 audit for the TRIS ring analysis and write its report.

Usage (the limTOD-capable interpreter and the matching limTOD source tree):

    PYTHONPATH=/path/to/limTOD <python> -m examples.inference.tris_audit_report \
        --maps runs/tris-input-skyfields/maps.npz \
        --archive runs/tris-input-skyfields/archive \
        --output runs/tris-forward/<run-id>

P0a (the data contract) needs only numpy and runs anywhere.  P0b and P0c need
limTOD, healpy and the prepared maps.  The report records the exact command,
the git HEAD, the limTOD source hashes and every seed, so a run can be
reproduced or discounted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from examples.inference.tris_audit import (
    beam_resolution_ladder,
    compress_map,
    control_operator,
    dense_wiener,
    jsonable,
    map_residual_norm2,
    monte_carlo_sampling_covariance,
    operator_delta_report,
    pointed_prediction,
    ring_residual_norm2,
    truncated_response_fraction,
    whitening_report,
)
from examples.inference.tris_data_contract import ledger_document, render_markdown

MC_DRAWS = 200


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def limtod_root() -> Path:
    """Directory of the importable limTOD package, whose git HEAD is recorded."""
    spec = importlib.util.find_spec("limTOD")
    if spec is None or spec.origin is None:
        return Path("limTOD")
    return Path(spec.origin).resolve().parent


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        # pragma: no cover - diagnostic only
        return f"unavailable: {error}"


def p0b_frequency(payload, index: int, *, mc_draws: int = MC_DRAWS) -> dict:
    """Every P0b check for one prepared frequency."""
    a = np.asarray(payload[f"operator_{index}"], dtype=float)
    sigma = np.asarray(payload[f"sigma_k_{index}"], dtype=float)
    prior = np.asarray(payload[f"prior_k_{index}"], dtype=float)
    prior_sigma = np.asarray(payload[f"prior_sigma_k_{index}"], dtype=float)
    data = np.asarray(payload[f"data_k_{index}"], dtype=float)
    covariance = np.asarray(payload[f"posterior_covariance_{index}"], dtype=float)
    stored_weights = np.asarray(payload[f"weights_{index}"], dtype=float)
    stored_bias = np.asarray(payload[f"bias_{index}"], dtype=float)
    stored_noise_sigma = np.asarray(payload[f"noise_sigma_k_{index}"], dtype=float)

    dense = dense_wiener(a, data, sigma, prior, prior_sigma)
    likelihood = compress_map(
        np.asarray(payload[f"map_k_{index}"], dtype=float), prior, a, sigma, covariance
    )
    whitened = np.asarray(likelihood.data, dtype=float)

    algebra = {
        "weights_vs_stored_max_abs": float(
            np.max(np.abs(stored_weights - dense.weights))
        ),
        "bias_vs_stored_max_abs": float(np.max(np.abs(stored_bias - dense.bias))),
        "noise_variance_vs_stored_max_abs": float(
            np.max(np.abs(stored_noise_sigma**2 - np.diag(dense.noise_covariance)))
        ),
        "posterior_sigma_vs_covariance_diagonal_max_abs": float(
            np.max(
                np.abs(
                    np.asarray(payload[f"posterior_sigma_k_{index}"], float) ** 2
                    - np.diag(covariance)
                )
            )
        ),
        "compress_weights_vs_oracle_max_abs": float(
            np.max(np.abs(likelihood.weights - dense.weights))
        ),
        "compress_bias_vs_oracle_max_abs": float(
            np.max(np.abs(likelihood.bias - dense.bias))
        ),
        "posterior_covariance_is_not_sampling_covariance": float(
            np.max(np.abs(covariance - dense.noise_covariance))
        ),
        "whitening_roundtrip_max_abs": float(
            np.max(
                np.abs(
                    whitened
                    - np.asarray(likelihood.ring_projection, float) @ (data / sigma)
                )
            )
        ),
    }

    solver: dict = {}
    try:
        from limTOD.HPW_filter import wiener_filter_map

        solved_mean, _solved_sigma, solved_cov = wiener_filter_map(
            data,
            a,
            noise_variance=sigma**2,
            prior_inv_cov=prior_sigma**-2,
            guess=prior,
            return_full_cov=True,
        )
        solver = {
            "mean_vs_oracle_max_abs": float(np.max(np.abs(solved_mean - dense.mean))),
            "covariance_vs_oracle_max_abs": float(
                np.max(np.abs(solved_cov - dense.covariance))
            ),
        }
    except (ImportError, np.linalg.LinAlgError) as error:
        # Reported, not hidden: a solver that cannot run is a finding.
        solver = {"unavailable": repr(error)}

    rank = whitening_report(likelihood)
    rank.pop("singular_values", None)

    template = np.asarray(payload["template_k"], dtype=float)
    region = np.asarray(payload["region"], dtype=np.int64)
    directions = [template * (region == r) for r in range(3)] + [np.ones_like(template)]
    truncation = truncated_response_fraction(
        likelihood, a, sigma, np.array(directions).T
    )
    truncation["labels"] = [f"region_{r}_template" for r in range(3)] + ["uniform"]

    perturbed = prior + 1e-3 * template
    parity = {
        "sky_perturbation": {
            "ring": ring_residual_norm2(a, data, sigma, perturbed)
            - ring_residual_norm2(a, data, sigma, prior),
            "map": map_residual_norm2(likelihood, perturbed)
            - map_residual_norm2(likelihood, prior),
        },
        "zero_level_005": {
            "ring": ring_residual_norm2(a, data, sigma, prior, 0.05)
            - ring_residual_norm2(a, data, sigma, prior),
            "map": map_residual_norm2(likelihood, prior, 0.05)
            - map_residual_norm2(likelihood, prior),
        },
    }
    for pair in parity.values():
        pair["absolute_difference"] = abs(pair["map"] - pair["ring"])
        pair["relative_difference"] = pair["absolute_difference"] / max(
            abs(pair["ring"]), 1.0
        )

    mc = monte_carlo_sampling_covariance(
        a, sigma, prior, prior_sigma, prior, draws=mc_draws
    )
    mc.pop("empirical_whitened_covariance_diagonal", None)
    mc.pop("sampling_covariance_diagonal", None)

    return {
        "frequency_mhz": float(payload["frequency_mhz"][index]),
        "samples": int(a.shape[0]),
        "pixels": int(a.shape[1]),
        "algebra": algebra,
        "solver": solver,
        "rank": rank,
        "truncation": truncation,
        "parity": parity,
        "monte_carlo": mc,
    }


def region_columns(template, region, frequency_mhz, *, beta=-2.8):
    """The six region-parameter columns, packed amplitude[0..2] then beta[0..2].

    The packing is the contract the prior widths and the shift slices rely on.
    Interleaving the regions (amp, beta, amp, beta, ...) labels a beta shift as
    an amplitude, which is the defect this helper exists to prevent from coming
    back unnoticed.
    """
    template = np.asarray(template, dtype=float)
    region = np.asarray(region, dtype=np.int64)
    ratio = frequency_mhz / 408.0
    bases = [template * (region == r) * ratio**beta for r in range(3)]
    return np.array([*bases, *[base * np.log(ratio) for base in bases]]).T


def linearized_region_shift(
    a, sigma, template, region, frequency_mhz, delta_prediction, *, beta=-2.8,
    prior_width=None,
) -> dict:
    """First-order shift of the six region parameters that absorbs a delta.

    Flat region priors leave the three regions degenerate, so a weak diagonal
    prior is added.  The shift is reported in units of that PRIOR width and of
    the resulting marginal POSTERIOR SD, because they are different scales: a
    shift below 0.003 prior width is not automatically below 0.1 posterior SD.
    This is a linearised estimate, not a refit; Batch B/C owns the real one.
    """
    a = np.asarray(a, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    template = np.asarray(template, dtype=float)
    region = np.asarray(region, dtype=np.int64)
    jacobian = a @ region_columns(template, region, frequency_mhz, beta=beta)
    widths = np.asarray(
        prior_width if prior_width is not None else [0.81] * 3 + [0.72] * 3
    )
    precision = (jacobian.T * (1.0 / sigma**2)) @ jacobian + np.diag(1.0 / widths**2)
    rhs = (jacobian.T * (1.0 / sigma**2)) @ np.asarray(delta_prediction, dtype=float)
    shift = np.linalg.solve(precision, rhs)
    posterior_sd = np.sqrt(np.diag(np.linalg.inv(precision)))
    return {
        "names": [f"amplitude[{r}]" for r in range(3)]
        + [f"beta[{r}]" for r in range(3)],
        "prior_width": widths.tolist(),
        "posterior_sd": posterior_sd.tolist(),
        "delta_amplitude": shift[:3].tolist(),
        "delta_beta": shift[3:].tolist(),
        "delta_over_prior_width": (shift / widths).tolist(),
        "delta_over_posterior_sd": (shift / posterior_sd).tolist(),
        "max_delta_over_prior_width": float(np.max(np.abs(shift / widths))),
        "max_delta_over_posterior_sd": float(np.max(np.abs(shift / posterior_sd))),
    }


def p0c(payload, archive: Path, *, ladder_nsides, grid_nsides, haslam) -> dict:
    import healpy as hp
    from limTOD.tris import cmb_monopole_rj_k
    from limTOD.tris.archive import read_tris_beam_cuts, read_tris_ring
    from limTOD.tris.beam import tris_cut_beam_map, tris_horizon_mask
    from limTOD.tris.geometry import tris_zenith_geometry

    nside = int(payload["nside"])
    pixels = np.arange(hp.nside2npix(nside))
    ring = read_tris_ring(archive / "TRIS_absolute_600.txt")
    cuts = read_tris_beam_cuts(archive / "TRIS_Beam_Profile.txt")
    geometry = tris_zenith_geometry(ring.ra_deg)
    mask = tris_horizon_mask(nside)
    beam8 = tris_cut_beam_map(cuts, nside=nside, normalization="peak")
    prior_sky = np.asarray(payload["prior_k_0"], dtype=float)
    sigma = np.asarray(payload["sigma_k_0"], dtype=float)
    prior_sky_2 = np.asarray(payload["prior_k_1"], dtype=float)
    sigma_2 = np.asarray(payload["sigma_k_1"], dtype=float)
    stored_operator = np.asarray(payload["operator_0"], dtype=float)
    template = np.asarray(payload["template_k"], dtype=float)
    region = np.asarray(payload["region"], dtype=np.int64)

    control = control_operator(
        beam8, geometry, pixels, nside, nside_hires=64, horizontal_mask=mask
    )
    control_check = {
        "control_vs_stored_operator_max_abs": float(
            np.max(np.abs(control - stored_operator))
        ),
        "stored_coverage_min": float(np.asarray(payload["coverage_0"], float).min()),
        "control_coverage_min": float(control.sum(axis=1).min()),
        "control_coverage_max": float(control.sum(axis=1).max()),
    }

    ladder = beam_resolution_ladder(
        cuts, geometry, pixels, ladder_nsides, nside_target=nside, horizontal_mask=mask
    )
    ladder_report = {}
    for h, operator in ladder.items():
        report = operator_delta_report(control, operator, prior_sky, sigma)
        report["constant_sky_max_deviation"] = float(
            np.max(np.abs(operator @ np.ones(operator.shape[1]) - 1.0))
        )
        report["shift"] = linearized_region_shift(
            control, sigma, template, region, 600.5, (operator - control) @ prior_sky
        )
        # The beam is achromatic, so the operator is shared with the 817.8 MHz
        # ring; only the sky and its sigma differ, and reporting both keeps the
        # numbers comparable with the two residual RMS values.
        report["second_frequency"] = operator_delta_report(
            control, operator, prior_sky_2, sigma_2
        )
        ladder_report[str(h)] = report

    ordered = sorted(ladder)
    convergence = {}
    for low, high in itertools.pairwise(ordered):
        difference = (ladder[high] - ladder[low]) @ prior_sky
        convergence[f"{low}->{high}"] = {
            "prediction_rms_k": float(np.sqrt(np.mean(difference**2))),
            "prediction_max_abs_k": float(np.max(np.abs(difference))),
        }

    # The beam alm is the nside-8 one, evaluated on the finer grid: the beam
    # INFORMATION is held fixed, only the sky and integration grid move.
    beam_alm8 = hp.map2alm(beam8)
    baseline_grid = pointed_prediction(
        beam_alm8, geometry, prior_sky, nside, horizontal_mask=mask
    )
    grid_report = {}
    for s in grid_nsides:
        sky_s = hp.ud_grade(prior_sky, s)
        prediction = pointed_prediction(
            beam_alm8, geometry, sky_s, s, horizontal_mask=tris_horizon_mask(s)
        )
        difference = prediction - baseline_grid
        grid_report[str(s)] = {
            "prediction_vs_nside8_rms_k": float(np.sqrt(np.mean(difference**2))),
            "prediction_vs_nside8_max_abs_k": float(np.max(np.abs(difference))),
            "prediction_rms_k": float(np.sqrt(np.mean(prediction**2))),
        }

    high_res: dict = {}
    if haslam is not None and Path(haslam).is_file():
        raw = hp.read_map(str(haslam), dtype=np.float64)
        cmb408 = float(cmb_monopole_rj_k(408.0))
        cmb600 = float(cmb_monopole_rj_k(600.5))
        for s in (64, 128):
            galactic = hp.ud_grade(
                hp.Rotator(coord=["G", "C"]).rotate_map_pixel(raw - cmb408), s
            )
            sky_s = galactic * (600.5 / 408.0) ** -2.8 + cmb600
            beam_s = tris_cut_beam_map(cuts, nside=s, normalization="peak")
            prediction = pointed_prediction(
                hp.map2alm(beam_s),
                geometry,
                sky_s,
                s,
                horizontal_mask=tris_horizon_mask(s),
            )
            difference = prediction - control @ prior_sky
            high_res[str(s)] = {
                "haslam_sha256": sha256_file(Path(haslam)),
                "prediction_vs_fiducial_rms_k": float(np.sqrt(np.mean(difference**2))),
                "prediction_vs_fiducial_max_abs_k": float(np.max(np.abs(difference))),
            }

    return {
        "geometry": {
            "nside": nside,
            "samples": int(geometry.lst_deg.size),
            "latitude_deg": float(geometry.latitude_deg),
            "selfrot_deg": float(geometry.selfrot_deg[0]),
            "ra_spacing_deg": np.unique(
                np.round(np.diff(np.asarray(ring.ra_deg, float)), 6)
            ).tolist(),
        },
        "control": control_check,
        "beam_ladder": ladder_report,
        "ladder_convergence": convergence,
        "sky_grid": grid_report,
        "high_res_haslam": high_res,
    }


def render_report(document: dict) -> str:
    p0b = document.get("p0b", [])
    p0c = document.get("p0c")
    lines = [
        "# TRIS forward-model P0 audit",
        "",
        f"Run {document['run_id']} at {document['created_utc']}.",
        f"Command: {document['command']}",
        f"bayesmith HEAD: {document['git']['bayesmith']}",
        f"limTOD HEAD: {document['git']['limTOD']}",
        "",
        "## P0a -- data contract",
        "",
        f"Provenance counts: {json.dumps(document['p0a']['provenance_counts'], sort_keys=True)}",
        "See data_contract.md for the full ledger.",
        "",
        "## P0b -- map algebra on the prepared maps",
        "",
        "Max absolute differences; everything is against an independent dense oracle.",
        "",
        "| frequency | weights | bias | noise variance | whitening roundtrip | retained | max discarded model fraction |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in p0b:
        lines.append(
            "| {freq:.1f} MHz | {w:.3e} | {b:.3e} | {n:.3e} | {r:.3e} | {kept}/{modes} | {disc:.3e} |".format(
                freq=entry["frequency_mhz"],
                w=entry["algebra"]["weights_vs_stored_max_abs"],
                b=entry["algebra"]["bias_vs_stored_max_abs"],
                n=entry["algebra"]["noise_variance_vs_stored_max_abs"],
                r=entry["algebra"]["whitening_roundtrip_max_abs"],
                kept=entry["rank"]["retained"],
                modes=entry["rank"]["modes"],
                disc=entry["truncation"]["max_discarded_fraction"],
            )
        )
    lines += [
        "",
        "Solver cross-check and ring/map parity (parity is a max absolute",
        "difference between the two likelihood differences, not a ratio):",
        "",
        "| frequency | solver mean diff | solver covariance diff | parity sky | parity zero level |",
        "|---|---|---|---|---|",
    ]
    for entry in p0b:
        solver = entry["solver"]
        lines.append(
            "| {freq:.1f} MHz | {mean} | {cov} | {sky:.1e} ({skyr:.1e}) | "
            "{zero:.1e} ({zeror:.1e}) |".format(
                freq=entry["frequency_mhz"],
                mean=solver.get("mean_vs_oracle_max_abs", solver.get("unavailable")),
                cov=solver.get("covariance_vs_oracle_max_abs", solver.get("unavailable")),
                sky=entry["parity"]["sky_perturbation"]["absolute_difference"],
                skyr=entry["parity"]["sky_perturbation"]["relative_difference"],
                zero=entry["parity"]["zero_level_005"]["absolute_difference"],
                zeror=entry["parity"]["zero_level_005"]["relative_difference"],
            )
        )
    lines += [
        "",
        "Monte Carlo sampling covariance (the empirical whitened covariance should",
        "be the identity; the posterior covariance is not it):",
        "",
        "| frequency | whitened off-diagonal RMS | reconstruction cov. rel. Frobenius | 1 sigma coverage | 2 sigma coverage |",
        "|---|---|---|---|---|",
    ]
    for entry in p0b:
        mc = entry["monte_carlo"]
        lines.append(
            "| {freq:.1f} MHz | {off:.3e} | {rel:.3e} | {one:.3f} | {two:.3f} |".format(
                freq=entry["frequency_mhz"],
                off=mc["empirical_whitened_offdiagonal_rms"],
                rel=mc["reconstruction_covariance_relative_frobenius"],
                one=mc["coverage_within_1sigma"],
                two=mc["coverage_within_2sigma"],
            )
        )
    if p0c is None:
        lines += ["", "## P0c -- skipped in this run", ""]
        return "\n".join(lines)
    lines += [
        "",
        "## P0c -- beam construction and integration grid",
        "",
        "Control is the production path: beam built at nside 8, then ud_grade to 64",
        "before the harmonic transform, evaluated at nside 8.",
        "",
        (
            f"- control vs stored operator max abs: "
            f"{p0c['control']['control_vs_stored_operator_max_abs']:.3e}"
        ),
        (
            f"- geometry: lat {p0c['geometry']['latitude_deg']:.4f} deg, selfrot "
            f"{p0c['geometry']['selfrot_deg']:.1f} deg, RA spacing "
            f"{p0c['geometry']['ra_spacing_deg']}"
        ),
        "",
        "### Beam built directly from the cuts at nside h (output grid fixed at nside 8)",
        "",
        "| h | operator max abs | RMS K (600.5) | RMS sigma | max sigma | RMS K (817.8) | delta chi2 | constant-sky deviation | max param shift / prior width |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h, report in sorted(p0c["beam_ladder"].items(), key=lambda item: int(item[0])):
        lines.append(
            "| {h} | {op:.3e} | {rms_k:.4f} | {rms_s:.4f} | {mx:.3f} | "
            "{rms2:.4f} | {chi2:.1f} | {const:.3e} | {shift:.3e} |".format(
                h=h,
                op=report["operator_max_abs"],
                rms_k=report["prediction_rms_k"],
                rms_s=report["prediction_rms_sigma"],
                mx=report["prediction_max_abs_sigma"],
                rms2=report["second_frequency"]["prediction_rms_k"],
                chi2=report["delta_chi_square"],
                const=report["constant_sky_max_deviation"],
                shift=report["shift"]["max_delta_over_prior_width"],
            )
        )
    lines += [
        "",
        "### Rung-to-rung convergence of the direct beam ladder",
        "",
        "| rungs | prediction RMS K | max abs K |",
        "|---|---|---|",
    ]
    for rungs, report in p0c["ladder_convergence"].items():
        lines.append(
            "| {rungs} | {rms:.4f} | {mx:.4f} |".format(
                rungs=rungs,
                rms=report["prediction_rms_k"],
                mx=report["prediction_max_abs_k"],
            )
        )
    lines += [
        "",
        "### Sky and integration grid raised, beam information fixed at nside 8",
        "",
        "| nside | prediction vs nside 8 RMS K | max abs K |",
        "|---|---|---|",
    ]
    for s, report in sorted(p0c["sky_grid"].items(), key=lambda item: int(item[0])):
        lines.append(
            "| {s} | {rms:.4f} | {mx:.4f} |".format(
                s=s,
                rms=report["prediction_vs_nside8_rms_k"],
                mx=report["prediction_vs_nside8_max_abs_k"],
            )
        )
    if p0c["high_res_haslam"]:
        lines += [
            "",
            "### Reference built straight from the high-resolution Haslam map",
            "",
            "| nside | prediction vs fiducial RMS K | max abs K |",
            "|---|---|---|",
        ]
        for s, report in sorted(
            p0c["high_res_haslam"].items(), key=lambda item: int(item[0])
        ):
            lines.append(
                "| {s} | {rms:.4f} | {mx:.4f} |".format(
                    s=s,
                    rms=report["prediction_vs_fiducial_rms_k"],
                    mx=report["prediction_vs_fiducial_max_abs_k"],
                )
            )
    lines += [
        "",
        "## Limits",
        "",
        "- Sample covariance, archive epoch, beam cut measurement errors and the",
        "  measured 2D beam pattern are not published; they stay unconfirmable in",
        "  the ledger and are not invented here.",
        "- The parameter shift is a linearised, prior-regularised estimate, not a",
        "  refit.  The M0/M1 refit is Batch B/C.",
        "- The high-resolution Haslam branch uses the same single power law",
        "  (beta = -2.8) as the fiducial prior, so it tests sky and beam",
        "  discretisation, not a different foreground model.",
        "- The Monte Carlo reconstruction-covariance columns are limited by the",
        "  draw count; the whitened off-diagonal RMS and the coverage fractions",
        "  are the stable statistics.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="rewrite audit.md from an existing audit.json without recomputing",
    )
    parser.add_argument("--haslam", type=Path, default=None)
    parser.add_argument("--mc-draws", type=int, default=MC_DRAWS)
    parser.add_argument(
        "--ladder", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256]
    )
    parser.add_argument("--grid", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--skip-p0b", action="store_true")
    parser.add_argument("--skip-p0c", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        document = json.loads((args.output / "audit.json").read_text())
        (args.output / "audit.md").write_text(render_report(document))
        print(json.dumps({"rendered": str(args.output / "audit.md")}))
        return 0

    if args.maps is None or args.archive is None:
        parser.error("--maps and --archive are required unless --render-only is set")
    payload = np.load(args.maps, allow_pickle=True)
    document = {
        "schema": "bayesmith.tris.p0-audit.v1",
        "run_id": args.output.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": " ".join(
            [sys.executable, "-m", "examples.inference.tris_audit_report"] + sys.argv[1:]
        ),
        "host": platform.node(),
        "python": sys.version,
        "git": {
            "bayesmith": git_head(Path.cwd()),
            "limTOD": git_head(limtod_root()),
        },
    }

    contract = ledger_document(args.archive)
    document["p0a"] = contract
    (args.output / "data_contract.json").write_text(
        json.dumps(jsonable(contract), indent=2) + "\n"
    )
    (args.output / "data_contract.md").write_text(render_markdown(contract))

    if not args.skip_p0b:
        document["p0b"] = [
            p0b_frequency(payload, index, mc_draws=args.mc_draws) for index in (0, 1)
        ]
    if not args.skip_p0c:
        document["p0c"] = p0c(
            payload,
            args.archive,
            ladder_nsides=args.ladder,
            grid_nsides=args.grid,
            haslam=args.haslam,
        )

    document["elapsed_s"] = round(time.time() - started, 2)
    (args.output / "audit.json").write_text(
        json.dumps(jsonable(document), indent=2) + "\n"
    )
    (args.output / "audit.md").write_text(render_report(document))
    (args.output / "run.exit").write_text("0\n")
    print(json.dumps({"output": str(args.output), "elapsed_s": document["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
