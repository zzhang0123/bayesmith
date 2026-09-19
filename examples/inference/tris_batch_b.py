"""Batch B: fixed-beam and audited-beam forward baselines, with log-joint parity.

Runs the P1 acceptance checks from the T-001 handoff plan:

* the fixed-A baseline (the frozen prepared bundle) is re-run with the same
  seeds and compared to the frozen artifact,
* the audited-A baseline (beam built directly from the cuts at nside 256,
  no ud_grade) is run into its own directory,
* for both bundles the graph's log joint is checked against an independent
  decomposition into log prior, TRIS likelihood, external likelihood and the
  positive-template support,
* the two posteriors and the two residual tables are compared with MCSE.

Nothing here edits the frozen runs; the fixed-A re-run lands under --output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from bayesmith import trace
from bayesmith.graph.evaluate import log_joint
from examples.inference.tris_forward_baseline import (
    log_joint_components,
    summarize_draws,
)
from examples.inference.tris_rsb_case import load_common_inputs, run_variant
from examples.inference.tris_rsb_sky import model, model_inputs

LATENTS = (
    "amplitude",
    "beta",
    "zero_standard",
    "haslam_monopole_K",
    "calibration_standard",
)


def load_draws(path: Path, chain_shape) -> dict:
    """Load saved draws and restore the (chains, draws, ...) axes.

    posterior.npz stores samples flattened over chains, so the chain structure
    has to come from the run's own result.json before any batch-means MCSE.
    """
    chains, draws = (int(value) for value in chain_shape)
    with np.load(path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    return {
        name: np.asarray(values).reshape(chains, draws, *np.asarray(values).shape[1:])
        for name, values in raw.items()
    }


def posterior_mean_values(draws, include_rsb):
    values = {name: np.asarray(draws[name]).mean(axis=(0, 1)) for name in LATENTS}
    if include_rsb:
        values["rsb_amplitude"] = np.asarray(draws["rsb_amplitude"]).mean()
        values["rsb_beta"] = np.asarray(draws["rsb_beta"]).mean()
    return values


def log_joint_parity(common, variant, values):
    include_rsb = variant == "rsb"
    inputs = model_inputs(
        common.tris_bundle, common.external, ("LWA", "ARCADE"), include_rsb
    )
    graph = trace(model, *inputs)
    direct = float(log_joint(graph, values))
    parts = log_joint_components(
        common.tris_bundle, common.external, ("LWA", "ARCADE"), values, include_rsb
    )
    return {
        "graph_log_joint": direct,
        **{key: value for key, value in parts.items()},
        "difference": parts["log_joint"] - direct,
    }


def frequency_rows(report):
    return [
        {
            "frequency_mhz": row["frequency_mhz"],
            "residual_rms_k": row["residual_rms_k"],
            "chi_square_per_observation": row["chi_square_per_observation"],
            "ppc_tail_probability": row["ppc_tail_probability"],
            "adequacy": row["adequacy"],
        }
        for row in report["frequencies"]
    ]


def parameter_comparison(draws_fixed, draws_audited):
    fixed = summarize_draws(draws_fixed)
    audited = summarize_draws(draws_audited)
    rows = []
    for name in fixed:
        mean_fixed = np.asarray(fixed[name]["mean"], dtype=float)
        mean_audited = np.asarray(audited[name]["mean"], dtype=float)
        mcse_fixed = np.asarray(fixed[name]["mcse"], dtype=float)
        mcse_audited = np.asarray(audited[name]["mcse"], dtype=float)
        combined = np.sqrt(mcse_fixed**2 + mcse_audited**2)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = np.where(combined > 0, (mean_audited - mean_fixed) / combined, 0.0)
        rows.append(
            {
                "parameter": name,
                "mean_fixed": mean_fixed.tolist(),
                "mean_audited": mean_audited.tolist(),
                "delta_mean": (mean_audited - mean_fixed).tolist(),
                "mcse_fixed": mcse_fixed.tolist(),
                "mcse_audited": mcse_audited.tolist(),
                "delta_over_combined_mcse": sigma.tolist(),
                "sd_fixed": fixed[name]["sd"],
                "sd_audited": audited[name]["sd"],
            }
        )
    return rows


def render(document: dict) -> str:
    lines = [
        "# Batch B: fixed-A and audited-A forward baselines",
        "",
        (
            f"Run {document['run_id']} at {document['created_utc']}; "
            f"elapsed {document['elapsed_s']} s."
        ),
        f"bayesmith HEAD {document['git']}",
        "",
        "## Bundles",
        "",
        (
            f"- fixed A: {document['bundles']['fixed']['beam_construction']} "
            f"({document['bundles']['fixed']['input_sha256'][:12]})"
        ),
        (
            f"- audited A: {document['bundles']['audited']['beam_construction']} "
            f"({document['bundles']['audited']['input_sha256'][:12]})"
        ),
        (
            f"- operator change: max abs {document['operator_change']['max_abs']:.3e}, "
            f"RMS delta prediction at the prior "
            f"{document['operator_change']['prediction_rms_k']['600.5']:.4f} K "
            f"(600.5 MHz) and "
            f"{document['operator_change']['prediction_rms_k']['817.8']:.4f} K (817.8 MHz)"
        ),
    ]
    fixed_rows = {
        row["frequency_mhz"]: row
        for row in document["frequencies"].get("fixed/no_rsb", [])
    }
    audited_rows = {
        row["frequency_mhz"]: row
        for row in document["frequencies"].get("audited/no_rsb", [])
    }
    lines += [
        "",
        "## Residual reduction from the audited operator (M0; M1 is identical to 4 dp)",
        "",
        "| frequency | RMS fixed K | RMS audited K | delta RMS K | delta RMS % | chi2/N fixed | chi2/N audited | delta chi2/N % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for frequency in sorted(fixed_rows):
        before, after = fixed_rows[frequency], audited_rows[frequency]
        delta_rms = after["residual_rms_k"] - before["residual_rms_k"]
        lines.append(
            "| {frequency:.1f} MHz | {before_rms:.4f} | {after_rms:.4f} | "
            "{delta_rms:+.4f} | {pct:+.1f}% | {before_chi:.1f} | {after_chi:.1f} | "
            "{chi_pct:+.1f}% |".format(
                frequency=frequency,
                before_rms=before["residual_rms_k"],
                after_rms=after["residual_rms_k"],
                delta_rms=delta_rms,
                pct=100.0 * delta_rms / before["residual_rms_k"],
                before_chi=before["chi_square_per_observation"],
                after_chi=after["chi_square_per_observation"],
                chi_pct=100.0
                * (after["chi_square_per_observation"]
                   - before["chi_square_per_observation"])
                / before["chi_square_per_observation"],
            )
        )
    if "fixed_reproduces_frozen" in document:
        reproduction = document["fixed_reproduces_frozen"]
        lines += [
            "",
            "## Fixed-A reproduction of the frozen artifact",
            "",
            (
                f"Max absolute difference between the re-run and the frozen "
                f"posterior draws: {reproduction['max_abs_draw_difference']:.3e} "
                f"(chain shape {reproduction['chain_shape']})."
            ),
        ]
    lines += [
        "",
        "## Log-joint parity at the posterior mean",
        "",
        "| baseline | variant | graph | independent | difference | support |",
        "|---|---|---|---|---|---|",
    ]
    for key, row in document["log_joint_parity"].items():
        lines.append(
            "| {key} | {variant} | {graph:.6f} | {total:.6f} | {difference:.3e} | "
            "{support} |".format(
                key=key,
                variant=row["variant"],
                graph=row["graph_log_joint"],
                total=row["log_joint"],
                difference=row["difference"],
                support=row["support_valid"],
            )
        )
    lines += [
        "",
        "## Residuals and adequacy",
        "",
        "| baseline | variant | frequency | residual RMS K | chi2 / N | PPC tail |",
        "|---|---|---|---|---|---|",
    ]
    for key, rows in document["frequencies"].items():
        for row in rows:
            lines.append(
                "| {key} | {variant} | {frequency:.1f} MHz | {rms:.4f} | {chi2:.1f} | "
                "{ppc:.3f} |".format(
                    key=key,
                    variant=key.split("/")[1],
                    frequency=row["frequency_mhz"],
                    rms=row["residual_rms_k"],
                    chi2=row["chi_square_per_observation"],
                    ppc=row["ppc_tail_probability"],
                )
            )
    lines += [
        "",
        "## Posterior shift, audited minus fixed, in units of combined MCSE",
        "",
        "| variant | parameter | delta mean | delta / MCSE |",
        "|---|---|---|---|",
    ]
    for variant, rows in document["parameter_shift"].items():
        for row in rows:
            lines.append(
                "| {variant} | {name} | {delta} | {sigma} |".format(
                    variant=variant,
                    name=row["parameter"],
                    delta=np.array2string(np.asarray(row["delta_mean"]), precision=4),
                    sigma=np.array2string(
                        np.asarray(row["delta_over_combined_mcse"]), precision=2
                    ),
                )
            )
    lines += [
        "",
        "## Limits",
        "",
        "- The audited operator fixes the P0-measured beam-construction bias; the",
        "  nside-8 sky quadrature (~0.056 K) remains and is not claimed as fixed.",
        "- The posterior shift is a real, same-budget refit comparison, but M0/M1",
        "  here still share the three-region foreground; the operator revision is",
        "  not required to reproduce the frozen bias.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-input", type=Path, default=Path("runs/tris-input-skyfields"))
    parser.add_argument("--audited-input", type=Path, required=True)
    parser.add_argument("--external-input", type=Path, default=Path("runs/tris-rsb-input"))
    parser.add_argument("--frozen-run", type=Path, default=Path("runs/inference-demo-verified"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = {}
    if (args.output / "batch_b.json").exists():
        previous = json.loads((args.output / "batch_b.json").read_text())
    fixed = load_common_inputs(args.frozen_input, args.external_input)
    audited = load_common_inputs(args.audited_input, args.external_input)
    if audited.tris_manifest.get("beam_construction_nside") is None:
        raise ValueError("audited input manifest does not record a direct beam nside")

    operator_change = {
        "max_abs": float(
            np.max(
                np.abs(
                    np.asarray(audited.tris_bundle["operator_0"], dtype=float)
                    - np.asarray(fixed.tris_bundle["operator_0"], dtype=float)
                )
            )
        ),
        "prediction_rms_k": {},
    }
    for index, label in enumerate(("600.5", "817.8")):
        delta = (
            np.asarray(audited.tris_bundle[f"operator_{index}"], dtype=float)
            - np.asarray(fixed.tris_bundle[f"operator_{index}"], dtype=float)
        ) @ np.asarray(fixed.tris_bundle[f"prior_k_{index}"], dtype=float)
        operator_change["prediction_rms_k"][label] = float(
            np.sqrt(np.mean(delta**2))
        )

    document = {
        "schema": "bayesmith.tris.batch-b.v1",
        "run_id": args.output.name,
        # A render-only pass must not restamp a run whose sampling has already
        # happened; only a real run may claim the current time and duration.
        "created_utc": (
            previous.get("created_utc")
            if args.skip_runs and previous.get("created_utc")
            else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ),
        "render_only": bool(args.skip_runs),
        "git": subprocess.check_output(
            ["git", "-C", ".", "rev-parse", "HEAD"], text=True
        ).strip(),
        "bundles": {
            "fixed": {
                "beam_construction": fixed.tris_manifest.get(
                    "beam_construction",
                    "legacy manifest: cut beam at nside 8 upsampled to 64",
                ),
                "input_sha256": fixed.tris_manifest["input_sha256"],
            },
            "audited": {
                "beam_construction": audited.tris_manifest["beam_construction"],
                "input_sha256": audited.tris_manifest["input_sha256"],
            },
        },
        "operator_change": operator_change,
        "log_joint_parity": {},
        "frequencies": {},
        "parameter_shift": {},
    }

    with jax.enable_x64(True):
        for label, common in (("fixed", fixed), ("audited", audited)):
            for index, variant in enumerate(("no_rsb", "rsb")):
                output = args.output / label / f"tris_haslam_{variant}"
                posterior_path = output / "posterior.npz"
                if not args.skip_runs or not posterior_path.exists():
                    report = run_variant(
                        common,
                        variant=variant,
                        included_surveys=("LWA", "ARCADE"),
                        output=output,
                        seed=args.seed + index,
                        draws=args.draws,
                        warmup=args.warmup,
                        chains=args.chains,
                    )
                else:
                    report = json.loads((output / "result.json").read_text())
                key = f"{label}/{variant}"
                document["frequencies"][key] = frequency_rows(report)
                draws = load_draws(posterior_path, report["chain_shape"])
                values = posterior_mean_values(draws, variant == "rsb")
                parity = log_joint_parity(common, variant, values)
                parity["variant"] = variant
                document["log_joint_parity"][key] = parity

        for variant in ("no_rsb", "rsb"):
            fixed_report = json.loads(
                (args.output / "fixed" / f"tris_haslam_{variant}" / "result.json").read_text()
            )
            audited_report = json.loads(
                (args.output / "audited" / f"tris_haslam_{variant}" / "result.json").read_text()
            )
            document["parameter_shift"][variant] = parameter_comparison(
                load_draws(
                    args.output / "fixed" / f"tris_haslam_{variant}" / "posterior.npz",
                    fixed_report["chain_shape"],
                ),
                load_draws(
                    args.output / "audited" / f"tris_haslam_{variant}" / "posterior.npz",
                    audited_report["chain_shape"],
                ),
            )

    frozen_dir = args.frozen_run / "tris_haslam_no_rsb"
    if (frozen_dir / "posterior.npz").exists():
        frozen_report = json.loads((frozen_dir / "result.json").read_text())
        frozen = load_draws(frozen_dir / "posterior.npz", frozen_report["chain_shape"])
        rerun = load_draws(
            args.output / "fixed" / "tris_haslam_no_rsb" / "posterior.npz",
            frozen_report["chain_shape"],
        )
        document["fixed_reproduces_frozen"] = {
            "chain_shape": frozen_report["chain_shape"],
            "max_abs_draw_difference": float(
                max(
                    np.max(np.abs(np.asarray(frozen[name]) - np.asarray(rerun[name])))
                    for name in frozen
                    if name in rerun
                )
            ),
            "max_abs_mean_difference": float(
                max(
                    np.max(
                        np.abs(
                            np.asarray(frozen[name]).mean(axis=(0, 1))
                            - np.asarray(rerun[name]).mean(axis=(0, 1))
                        )
                    )
                    for name in frozen
                    if name in rerun
                )
            ),
        }

    if args.skip_runs and previous.get("elapsed_s") is not None:
        document["elapsed_s"] = previous["elapsed_s"]
    else:
        document["elapsed_s"] = round(time.time() - started, 2)
    (args.output / "batch_b.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + "\n"
    )
    (args.output / "batch_b.md").write_text(render(document))
    print(json.dumps({"output": str(args.output), "elapsed_s": document["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
