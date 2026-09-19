"""P2 remainder: beam-width identifiability against the sky parameters.

The plan asks, before any beam is freed, whether a shared beam-width
deformation is identifiable against the foreground amplitude and index.  This
module answers that with the prior-scaled whitened Jacobian: it deforms the
archive principal-plane cuts by a log-width factor in each plane, rebuilds the
sky-to-sample operator from the deformed cuts, and measures the degeneracy with
the region amplitudes and indices.

The whitening is not optional.  The ring prediction is in kelvin; the archived
per-point statistical errors sigma_k_0 / sigma_k_1 are 0.004-0.041 K, so the
Fisher information is (J / sigma)^T (J / sigma), not J^T J.  The unwhitened
matrix this module first shipped measured the prior scaling alone and is kept
in the record only as the withdrawn approximation.

The deformation is an angle-axis rescale of each cut, so it reproduces the
archive cuts at zero and keeps the beam power non-negative; the operator's own
normalization is unchanged.  The prior width on the deformation is an
ASSUMPTION with no measurement behind it and is reported as one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from examples.inference.tris_audit import pointed_operator_from_beam_map
from examples.inference.tris_forward_baseline import zero_levels
from examples.inference.tris_identifiability import prior_sd

NL = chr(10)
FENCE = chr(96) * 3
ATOMS = ("amplitude", "beta", "zero_standard", "haslam_monopole_K")


def posterior_mean_values(path, chain_shape):
    """Posterior mean of the four TRIS atom blocks, without importing bayesmith."""
    chains, draws = (int(value) for value in chain_shape)
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    values = {}
    for name in ATOMS:
        values[name] = np.asarray(raw[name], dtype=float).reshape(
            chains, draws, *np.asarray(raw[name]).shape[1:]
        ).mean(axis=(0, 1))
    return values


def noise_sigma(bundle):
    """Per-ring statistical error, concatenated in the prediction's order."""
    return np.concatenate(
        [
            np.asarray(bundle["sigma_k_0"], dtype=float),
            np.asarray(bundle["sigma_k_1"], dtype=float),
        ]
    )


def resample_cuts(angle, e_plane_db, h_plane_db, log_width_e, log_width_h):
    """Resample each principal-plane cut under an angle-axis rescale."""
    angle = np.asarray(angle, dtype=float)
    e_plane_db = np.asarray(e_plane_db, dtype=float)
    h_plane_db = np.asarray(h_plane_db, dtype=float)
    e_shaped = np.interp(angle / np.exp(log_width_e), angle, e_plane_db)
    h_shaped = np.interp(angle / np.exp(log_width_h), angle, h_plane_db)
    return angle, h_shaped, e_shaped


def deformed_cuts(cuts, log_width_e, log_width_h):
    from limTOD.tris import TRISPrincipalPlaneCuts

    angle, h_plane, e_plane = resample_cuts(
        cuts.angle_deg, cuts.e_plane_db, cuts.h_plane_db, log_width_e, log_width_h
    )
    return TRISPrincipalPlaneCuts(angle, h_plane, e_plane)


def sky_field(bundle, values, frequency_mhz):
    """Per-pixel sky temperature at one frequency, the ring model's integrand."""
    template = np.asarray(bundle["template_k"], dtype=float)
    region = np.asarray(bundle["region"], dtype=int)
    amplitude = np.asarray(values["amplitude"], dtype=float)
    beta = np.asarray(values["beta"], dtype=float)
    monopole = float(np.asarray(values["haslam_monopole_K"]))
    scaling = amplitude[region] * (frequency_mhz / 408.0) ** beta[region]
    return (template + monopole) * scaling


def ring_prediction(bundle, values, operator):
    """operator @ sky minus the zero level, per ring, concatenated in order."""
    levels = zero_levels(values["zero_standard"])
    return np.concatenate(
        [
            operator @ sky_field(bundle, values, frequency) - levels[index]
            for index, frequency in enumerate(
                np.asarray(bundle["frequency_mhz"], dtype=float)
            )
        ]
    )


def parameter_vector(values):
    return np.concatenate(
        [
            np.asarray(values[name], dtype=float).ravel()
            if np.ndim(values[name]) > 0
            else np.atleast_1d(float(values[name]))
            for name in ATOMS
        ]
    )


def set_parameter(values, index, delta):
    """Return a copy of the atom dict with one packed coordinate shifted."""
    result = {name: np.array(value, dtype=float, copy=True) for name, value in values.items()}
    sizes = [np.size(np.atleast_1d(result[name])) for name in ATOMS]
    offset = 0
    for name, size in zip(ATOMS, sizes, strict=True):
        if index < offset + size:
            local = index - offset
            if size == 1:
                result[name] = np.asarray(result[name] + delta)
            else:
                flat = np.atleast_1d(result[name]).ravel()
                flat[local] += delta
                result[name] = flat
            return result
        offset += size
    raise IndexError(index)


def jacobian(bundle, values, operators, *, step=1e-5, width_step=1e-3):
    """(ring samples, parameters) Jacobian of the ring prediction, in kelvin."""
    fiducial = ring_prediction(bundle, values, operators["center"])
    columns = []
    count = parameter_vector(values).size
    for index in range(count):
        width = step * max(1.0, abs(parameter_vector(values)[index]))
        plus = ring_prediction(bundle, set_parameter(values, index, width), operators["center"])
        minus = ring_prediction(bundle, set_parameter(values, index, -width), operators["center"])
        columns.append((plus - minus) / (2.0 * width))
    for key in ("e", "h"):
        plus = ring_prediction(bundle, values, operators[key + "_plus"])
        minus = ring_prediction(bundle, values, operators[key + "_minus"])
        columns.append((plus - minus) / (2.0 * width_step))
    return np.array(columns).T, fiducial


def coordinate_names():
    return (
        [f"amplitude[{i}]" for i in range(3)]
        + [f"beta[{i}]" for i in range(3)]
        + ["zero_standard[0]", "zero_standard[1]", "haslam_monopole_K", "log_width_e", "log_width_h"]
    )


def prior_widths(width_prior_sd, *, calibration_coordinates=0):
    return np.concatenate(
        [prior_sd(False, calibration_coordinates=calibration_coordinates), [width_prior_sd, width_prior_sd]]
    )


def laplace_posterior(scaled):
    """Posterior SD and correlation for a prior-scaled, whitened Jacobian."""
    scaled = np.asarray(scaled, dtype=float)
    _left, singular, _right = np.linalg.svd(scaled, full_matrices=False)
    fisher = scaled.T @ scaled
    posterior = np.linalg.inv(fisher + np.eye(scaled.shape[1]))
    sd = np.sqrt(np.diag(posterior))
    correlation = posterior / np.outer(sd, sd)
    return {
        "singular_values": singular,
        "fisher_eigenvalues": singular**2,
        "posterior_over_prior_sd": sd,
        "posterior_correlation": correlation,
    }


def analyze(
    bundle,
    values,
    operators,
    *,
    sigma,
    width_step=1e-3,
    width_prior_sd=0.05,
    step=1e-5,
    prior_scale=None,
):
    """Prior-scaled, noise-whitened Fisher analysis at one point.

    prior_scale multiplies each coordinate's prior SD before the Fisher is
    formed.  It exists for unit-consistency checks: under a kelvin-to-millikelvin
    change the temperature-valued coordinates' priors scale with the data, and
    only then are the posterior/prior ratios invariant.  It is 1 by default and
    never used by the production run.
    """
    names = coordinate_names()
    raw, _fiducial = jacobian(bundle, values, operators, step=step, width_step=width_step)
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (raw.shape[0],):
        raise ValueError("sigma has shape " + str(sigma.shape) + " expected " + str((raw.shape[0],)))
    if np.any(sigma <= 0.0):
        raise ValueError("sigma must be strictly positive")
    widths = prior_widths(width_prior_sd)
    if prior_scale is not None:
        prior_scale = np.asarray(prior_scale, dtype=float)
        if prior_scale.shape != widths.shape:
            raise ValueError("prior_scale has the wrong length")
        widths = widths * prior_scale
    scaled = (raw / sigma[:, None]) * widths[None, :]
    summary = laplace_posterior(scaled)
    unwhitened = laplace_posterior(raw * widths[None, :])
    index = {name: position for position, name in enumerate(names)}
    correlation = summary["posterior_correlation"]
    return {
        "names": names,
        "whitened": True,
        "sigma_min_k": float(sigma.min()),
        "sigma_max_k": float(sigma.max()),
        "prior_sd": widths.tolist(),
        "singular_values": summary["singular_values"].tolist(),
        "fisher_eigenvalues": summary["fisher_eigenvalues"].tolist(),
        "directions_with_snr_above_one": int(np.count_nonzero(summary["singular_values"] > 1.0)),
        "posterior_over_prior_sd": summary["posterior_over_prior_sd"].tolist(),
        "withdrawn_unwhitened_posterior_over_prior_sd": unwhitened[
            "posterior_over_prior_sd"
        ].tolist(),
        "amplitude_width_correlations": [
            float(correlation[index[f"amplitude[{i}]"], index["log_width_e"]])
            for i in range(3)
        ],
        "beta_width_correlations": [
            float(correlation[index[f"beta[{i}]"], index["log_width_e"]])
            for i in range(3)
        ],
        "e_width_correlations": [
            float(correlation[index[f"amplitude[{i}]"], index["log_width_h"]])
            for i in range(3)
        ],
        "correlation": correlation.tolist(),
        "width_prior_sd_assumption": width_prior_sd,
    }


def prior_point():
    """Prior central values for the four atom blocks, as a linearization point."""
    return {
        "amplitude": np.full(3, 1.6),
        "beta": np.full(3, -2.75),
        "zero_standard": np.zeros(2),
        "haslam_monopole_K": np.asarray(0.0),
    }


def perturbed_point(values, *, amplitude=1.05, beta=0.05):
    """A second posterior-side point, to test the local approximation's spread."""
    result = {name: np.array(value, dtype=float, copy=True) for name, value in values.items()}
    result["amplitude"] = np.asarray(result["amplitude"]) * amplitude
    result["beta"] = np.asarray(result["beta"]) + beta
    return result


def support_diagnostics(values):
    """Distance to the Uniform supports and the zero-level kink, in prior SD."""
    amplitude = np.asarray(values["amplitude"], dtype=float)
    beta = np.asarray(values["beta"], dtype=float)
    return {
        "amplitude_lower_margin_prior_sd": ((amplitude - 0.2) / 0.808290384).tolist(),
        "amplitude_upper_margin_prior_sd": ((3.0 - amplitude) / 0.808290384).tolist(),
        "beta_lower_margin_prior_sd": ((beta - -4.0) / 0.721687836).tolist(),
        "beta_upper_margin_prior_sd": ((-1.5 - beta) / 0.721687836).tolist(),
        "zero_standard_1_kink_distance_prior_sd": float(
            abs(float(np.asarray(values["zero_standard"], dtype=float)[1]))
        ),
        "zero_level_kink_at_zero_standard_1": True,
    }


def step_stability(bundle, values, operators, sigma, *, width_prior_sd):
    """Posterior/prior SD across sky-parameter finite-difference steps."""
    table = {}
    for step in (1e-6, 1e-5, 1e-4):
        summary = analyze(
            bundle,
            values,
            operators,
            sigma=sigma,
            width_step=1e-3,
            width_prior_sd=width_prior_sd,
            step=step,
        )
        table[f"{step:.0e}"] = summary["posterior_over_prior_sd"]
    ratios = np.array(list(table.values()))
    reference = ratios[1]
    relative = np.max(np.abs(ratios - reference[None, :]) / np.maximum(np.abs(reference), 1e-300), axis=0)
    return {
        "steps": list(table),
        "posterior_over_prior_sd_by_step": {key: value for key, value in table.items()},
        "max_relative_spread": float(relative.max()),
        "per_coordinate_relative_spread": relative.tolist(),
    }


def point_stability(bundle, points, operators, sigma, *, width_step, width_prior_sd):
    """Whitened analysis at several linearization points."""
    table = {}
    for name, values in points.items():
        summary = analyze(
            bundle,
            values,
            operators,
            sigma=sigma,
            width_step=width_step,
            width_prior_sd=width_prior_sd,
        )
        table[name] = {
            "posterior_over_prior_sd": summary["posterior_over_prior_sd"],
            "log_width_e": summary["posterior_over_prior_sd"][-2],
            "log_width_h": summary["posterior_over_prior_sd"][-1],
            "amplitude": np.asarray(values["amplitude"], dtype=float).tolist(),
            "beta": np.asarray(values["beta"], dtype=float).tolist(),
        }
    widths = np.array(
        [[entry["log_width_e"], entry["log_width_h"]] for entry in table.values()]
    )
    spread = np.ptp(widths, axis=0) / np.maximum(np.abs(np.mean(widths, axis=0)), 1e-300)
    return {
        "points": list(table),
        "by_point": table,
        "width_ratio_max_relative_spread": float(np.max(spread)),
    }


def width_step_stability(bundle, values, sigma, operator_for, steps, *, width_prior_sd):
    """Width posterior/prior SD across the cut-resampling finite-difference step."""
    table = {}
    center = operator_for(0.0, 0.0)
    for step in steps:
        operators = {
            "center": center,
            "e_plus": operator_for(step, 0.0),
            "e_minus": operator_for(-step, 0.0),
            "h_plus": operator_for(0.0, step),
            "h_minus": operator_for(0.0, -step),
        }
        summary = analyze(
            bundle,
            values,
            operators,
            sigma=sigma,
            width_step=step,
            width_prior_sd=width_prior_sd,
        )
        table[f"{step:.0e}"] = {
            "posterior_over_prior_sd": summary["posterior_over_prior_sd"],
            "log_width_e": summary["posterior_over_prior_sd"][-2],
            "log_width_h": summary["posterior_over_prior_sd"][-1],
        }
    widths = np.array(
        [[entry["log_width_e"], entry["log_width_h"]] for entry in table.values()]
    )
    spread = np.ptp(widths, axis=0) / np.maximum(np.abs(np.mean(widths, axis=0)), 1e-300)
    return {
        "steps": list(table),
        "by_step": table,
        "width_ratio_max_relative_spread": float(np.max(spread)),
    }


def deformation_legality(cuts, *, beam_nside, log_widths):
    """Beam-power non-negativity and peak normalization across width points.

    The deformation rescales each principal-plane cut along its angle axis,
    so the power beam is 10**(dB/10) at every sampled angle and is
    non-negative by construction.  This measures that claim on the actual
    archive cuts instead of asserting it.  The map is peak-normalized, so
    the sum is only one after the pointing normalization; the sum is
    recorded to show the deformation changes it.
    """
    from limTOD.tris.beam import tris_cut_beam_map

    table = {}
    for log_width_e, log_width_h in log_widths:
        beam = np.asarray(
            tris_cut_beam_map(
                deformed_cuts(cuts, log_width_e, log_width_h),
                nside=beam_nside,
                normalization="peak",
            )
        )
        table[f"e{log_width_e:+.3f}_h{log_width_h:+.3f}"] = {
            "minimum": float(beam.min()),
            "negative_entries": int(np.count_nonzero(beam < 0.0)),
            "sum": float(beam.sum()),
            "peak": float(beam.max()),
        }
    return table


def render(document):
    lines = [
        "# Beam-width identifiability against the foreground",
        "",
        (
            f"Run {document['run_id']} at {document['created_utc']}; "
            f"elapsed {document['elapsed_s']} s."
        ),
        f"bayesmith HEAD {document['git']}",
        "",
        "The deformation rescales each principal-plane cut along its angle axis,",
        "so it reproduces the archive cuts at zero width change and keeps the beam",
        "power non-negative.  The width prior is an ASSUMPTION, not a measurement.",
        "",
        "The Jacobian is whitened by the archived per-point ring sigma before the",
        "Fisher matrix is formed.  The withdrawn unwhitened numbers are kept in the",
        "JSON and in the interpretation note, not used for any conclusion.",
        "",
        (
            f"Ring sigma range: {document['analysis']['sigma_min_k']:.4f} - "
            f"{document['analysis']['sigma_max_k']:.4f} K."
        ),
        f"Assumed width prior SD: {document['analysis']['width_prior_sd_assumption']}",
        "",
        "| coordinate | posterior SD / prior SD (whitened) | withdrawn (unwhitened) |",
        "|---|---|---|",
    ]
    for name, ratio, old in zip(
        document["analysis"]["names"],
        document["analysis"]["posterior_over_prior_sd"],
        document["analysis"]["withdrawn_unwhitened_posterior_over_prior_sd"],
    ):
        lines.append(f"| {name} | {ratio:.6f} | {old:.6f} |")
    lines += [
        "",
        (
            f"Directions with SNR > 1: "
            f"{document['analysis']['directions_with_snr_above_one']} "
            f"of {len(document['analysis']['names'])}."
        ),
        "",
        "| degeneracy | value |",
        "|---|---|",
    ]
    for key in (
        "amplitude_width_correlations",
        "beta_width_correlations",
        "e_width_correlations",
    ):
        for index, value in enumerate(document["analysis"][key]):
            lines.append(f"| {key}[{index}] | {value:+.4f} |")
    lines += [
        "",
        "## Support and kink diagnostics",
        "",
        FENCE + "json",
        json.dumps(document["support"], indent=2),
        FENCE,
        "",
        "## Finite-difference step stability (sky columns)",
        "",
        "Max relative spread across steps 1e-6, 1e-5, 1e-4: "
        + f"{document['step_stability']['max_relative_spread']:.3e}.",
        "",
        "## Linearization-point stability",
        "",
        "Width posterior/prior SD at the posterior mean, the prior center and a",
        "shifted posterior point; max relative spread "
        + f"{document['point_stability']['width_ratio_max_relative_spread']:.3e}.",
        "",
        "| point | log width E | log width H |",
        "|---|---|---|",
    ]
    for name, entry in document["point_stability"]["by_point"].items():
        lines.append(f"| {name} | {entry['log_width_e']:.6f} | {entry['log_width_h']:.6f} |")
    lines += [
        "",
        "## Cut-resampling step stability",
        "",
        "Width posterior/prior SD by finite-difference step; max relative spread "
        + f"{document['width_step_stability']['width_ratio_max_relative_spread']:.3e}.",
        "",
        "| step | log width E | log width H |",
        "|---|---|---|",
    ]
    for name, entry in document["width_step_stability"]["by_step"].items():
        lines.append(f"| {name} | {entry['log_width_e']:.6f} | {entry['log_width_h']:.6f} |")
    lines += [
        "",
        "## Limits",
        "",
        "- This is a Laplace analysis at one point, not a sampled posterior.",
        "- The width prior has no measurement behind it; the correlations scale with",
        "  it and the assumed value is recorded on every run.",
        "- The deformation is a smooth two-parameter family; a real 2D beam pattern",
        "  could carry structure this family cannot express.",
        "- The 817.8 MHz zero level has a 0.300 / 0.430 K kink at zero_standard[1] = 0;",
        "  the support table records how far the linearization point sits from it.",
        "",
    ]
    return NL.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beam-nside", type=int, default=128)
    parser.add_argument("--width-step", type=float, default=1e-3)
    parser.add_argument("--width-prior-sd", type=float, default=0.05)
    parser.add_argument("--rebuild-operators", action="store_true")
    parser.add_argument(
        "--width-steps",
        type=float,
        nargs="+",
        default=None,
        help="cut-resampling finite-difference steps for the width stability table",
    )
    args = parser.parse_args(argv)

    import healpy as hp
    from limTOD.tris import read_tris_beam_cuts, read_tris_ring
    from limTOD.tris.beam import tris_cut_beam_map, tris_horizon_mask
    from limTOD.tris.geometry import tris_zenith_geometry

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = {
        name: value
        for name, value in np.load(
            args.batch_b / "audited-input" / "maps.npz", allow_pickle=False
        ).items()
    }
    result_path = args.batch_b / "audited" / "tris_haslam_no_rsb" / "result.json"
    chain_shape = json.loads(result_path.read_text())["chain_shape"]
    values = posterior_mean_values(
        args.batch_b / "audited" / "tris_haslam_no_rsb" / "posterior.npz", chain_shape
    )
    sigma = noise_sigma(bundle)

    station = Path("runs/tris-input-skyfields/archive")
    ring = read_tris_ring(station / "TRIS_absolute_600.txt")
    cuts = read_tris_beam_cuts(station / "TRIS_Beam_Profile.txt")
    geometry = tris_zenith_geometry(ring.ra_deg)
    nside = int(bundle["nside"])
    pixels = np.arange(hp.nside2npix(nside))
    mask = tris_horizon_mask(nside)

    def operator_for(log_width_e, log_width_h):
        beam = tris_cut_beam_map(
            deformed_cuts(cuts, log_width_e, log_width_h),
            nside=args.beam_nside,
            normalization="peak",
        )
        return pointed_operator_from_beam_map(
            beam, geometry, pixels, nside, horizontal_mask=mask
        )

    half = args.width_step
    cache = args.output / "operators.npz"
    if cache.is_file() and not args.rebuild_operators:
        with np.load(cache, allow_pickle=False) as data:
            operators = {name: data[name] for name in data.files}
    else:
        operators = {
            "center": operator_for(0.0, 0.0),
            "e_plus": operator_for(half, 0.0),
            "e_minus": operator_for(-half, 0.0),
            "h_plus": operator_for(0.0, half),
            "h_minus": operator_for(0.0, -half),
        }
        np.savez_compressed(
            cache,
            **operators,
            width_step=np.asarray(args.width_step),
            beam_nside=np.asarray(args.beam_nside),
        )
    document = {
        "schema": "bayesmith.tris.beam-identifiability.v2",
        "run_id": args.output.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": subprocess.check_output(
            ["git", "-C", ".", "rev-parse", "HEAD"], text=True
        ).strip(),
        "beam_nside": args.beam_nside,
        "width_step": args.width_step,
        "operators_provenance": {
            "width_step": args.width_step,
            "beam_nside": args.beam_nside,
            "deformation": (
                "angle-axis rescale of the archive principal-plane cuts "
                "(resample_cuts); dA is the central difference with this step"
            ),
            "center_negative_entries": int(np.count_nonzero(operators["center"] < 0.0)),
            "center_minimum": float(operators["center"].min()),
            "center_row_sum_min": float(operators["center"].sum(axis=1).min()),
            "center_row_sum_max": float(operators["center"].sum(axis=1).max()),
        },
        "deformation_legality": deformation_legality(
            cuts,
            beam_nside=args.beam_nside,
            log_widths=[
                (0.0, 0.0),
                (0.05, 0.0),
                (-0.05, 0.0),
                (0.0, 0.05),
                (0.0, -0.05),
            ],
        ),
        "analysis": analyze(
            bundle,
            values,
            operators,
            sigma=sigma,
            width_step=args.width_step,
            width_prior_sd=args.width_prior_sd,
        ),
        "support": support_diagnostics(values),
        "step_stability": step_stability(
            bundle, values, operators, sigma, width_prior_sd=args.width_prior_sd
        ),
        "point_stability": point_stability(
            bundle,
            {
                "posterior_mean": values,
                "prior_center": prior_point(),
                "posterior_shifted": perturbed_point(values),
            },
            operators,
            sigma,
            width_step=args.width_step,
            width_prior_sd=args.width_prior_sd,
        ),
        "width_step_stability": width_step_stability(
            bundle,
            values,
            sigma,
            operator_for,
            args.width_steps
            or [args.width_step, args.width_step * 10.0, args.width_step / 10.0],
            width_prior_sd=args.width_prior_sd,
        ),
    }
    document["elapsed_s"] = round(time.time() - started, 2)
    (args.output / "beam_identifiability.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + NL
    )
    (args.output / "beam_identifiability.md").write_text(render(document))
    print(json.dumps({"output": str(args.output), "elapsed_s": document["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
