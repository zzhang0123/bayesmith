"""Batch C: P2 identifiability and synthetic injection recovery.

Runs the plan's P2 gate before any physical reading of the fit:

* the whitened Jacobian, prior-scaled, at the prior mean and at the audited
  posterior mean, with its SVD and posterior correlation (which directions the
  data constrain, which the prior supplies, and which parameter pairs are
  degenerate);
* same-budget synthetic recovery for a sky-only truth, a truth with a 0.5 K
  RSB, and a truth generated with a different region split, to test false
  detection and misattribution.

Everything runs in bayesmith against the audited operator from Batch B.  The
beam-parameter block is not in the model yet and is reported as not measured.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

from examples.inference.tris_batch_b import load_draws, posterior_mean_values
from examples.inference.tris_identifiability import analyze, degeneracy_rows
from examples.inference.tris_injection import (
    inject,
    recovery_rows,
    shifted_regions,
    truth_values,
    write_inputs,
)
from examples.inference.tris_rsb_case import load_common_inputs, run_variant

SURVEYS = ("LWA", "ARCADE")


def _load_bundle(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def complete_truth(values, include_rsb):
    """Make a truth block match the model's latent set.

    A model with RSB needs those two keys even when the data have none, and a
    model without RSB must not be handed them.
    """
    completed = dict(values)
    if include_rsb:
        completed.setdefault("rsb_amplitude", np.asarray(0.0))
        completed.setdefault("rsb_beta", np.asarray(-2.6))
    else:
        completed.pop("rsb_amplitude", None)
        completed.pop("rsb_beta", None)
    return completed


def log_joint_at(bundle, external, values, include_rsb):
    """The graph's log joint at a point, for prior-versus-likelihood attribution."""
    from bayesmith import trace
    from bayesmith.graph.evaluate import log_joint
    from examples.inference.tris_rsb_sky import model, model_inputs

    inputs = model_inputs(bundle, external, SURVEYS, include_rsb)
    graph = trace(model, *inputs)
    return float(log_joint(graph, values))


def identifiability(common, values, include_rsb):
    analysis = analyze(
        common.tris_bundle, common.external, SURVEYS, values, include_rsb
    )
    analysis["degeneracies"] = degeneracy_rows(analysis, top=6)
    return analysis


def scenario_truths(draws_m0, bundle):
    m0 = posterior_mean_values(draws_m0, False)
    plus_rsb = {**m0, "rsb_amplitude": np.asarray(0.5), "rsb_beta": np.asarray(-2.6)}
    generator = {
        **bundle,
        "region": shifted_regions(bundle["galactic_latitude_deg"], (20.0, 40.0)),
    }
    return {
        "prior_mean": (truth_values(amplitude=1.6, beta=-2.75), False, None, ("M0", "M1")),
        "prior_mean_plus_rsb_0p5": (
            truth_values(
                amplitude=1.6, beta=-2.75, rsb_amplitude=0.5, rsb_beta=-2.6
            ),
            True,
            None,
            ("M1", "M0"),
        ),
        "audited_m0": (m0, False, None, ("M0", "M1")),
        "audited_m0_plus_rsb_0p5": (plus_rsb, True, None, ("M1", "M0")),
        "shifted_regions": (m0, False, generator, ("M0",)),
    }


def frequency_rows(report):
    return [
        {
            "frequency_mhz": row["frequency_mhz"],
            "residual_rms_k": row["residual_rms_k"],
            "chi_square_per_observation": row["chi_square_per_observation"],
            "ppc_tail_probability": row["ppc_tail_probability"],
        }
        for row in report["frequencies"]
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited-run", type=Path, required=True)
    parser.add_argument("--frozen-input", type=Path, default=Path("runs/tris-input-skyfields"))
    parser.add_argument("--external-input", type=Path, default=Path("runs/tris-rsb-input"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--skip-injection", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    audited_input = args.audited_run / "audited-input"
    common = load_common_inputs(audited_input, args.external_input)
    bundle = common.tris_bundle

    document = {
        "schema": "bayesmith.tris.batch-c.v1",
        "run_id": args.output.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": subprocess.check_output(
            ["git", "-C", ".", "rev-parse", "HEAD"], text=True
        ).strip(),
        "identifiability": {},
        "injection": {},
    }

    with jax.enable_x64(True):
        # C1: identifiability at the prior mean and at the audited posterior mean.
        document["identifiability"]["prior_mean"] = {
            "M0": identifiability(
                common, truth_values(amplitude=1.6, beta=-2.75), False
            )
        }
        draws_m0 = load_draws(
            args.audited_run / "audited" / "tris_haslam_no_rsb" / "posterior.npz",
            json.loads(
                (
                    args.audited_run / "audited" / "tris_haslam_no_rsb" / "result.json"
                ).read_text()
            )["chain_shape"],
        )
        posterior_identifiability = {}
        for variant, include_rsb in (("M0", False), ("M1", True)):
            draws = load_draws(
                args.audited_run / "audited" / f"tris_haslam_{'rsb' if include_rsb else 'no_rsb'}" / "posterior.npz",
                json.loads(
                    (
                        args.audited_run
                        / "audited"
                        / f"tris_haslam_{'rsb' if include_rsb else 'no_rsb'}"
                        / "result.json"
                    ).read_text()
                )["chain_shape"],
            )
            values = posterior_mean_values(draws, include_rsb)
            posterior_identifiability[variant] = identifiability(
                common, values, include_rsb
            )
        document["identifiability"]["posterior_mean"] = posterior_identifiability

        # C2: injection recovery.
        scenarios = scenario_truths(draws_m0, bundle)
        rng = np.random.default_rng(args.seed)
        for name, (truth, include_rsb, generator, variants) in scenarios.items():
            if args.skip_injection:
                continue
            bundle_syn, external_syn = inject(
                bundle,
                common.external,
                SURVEYS,
                truth,
                include_rsb,
                rng,
                generator_bundle=generator,
            )
            directory = write_inputs(
                args.output / "injection" / name / "input",
                bundle_syn,
                external_syn,
                common.tris_manifest,
                common.external_manifest,
            )
            synthetic = load_common_inputs(directory, directory)
            entry = {"variants": {}}
            for index, label in enumerate(variants):
                include = label == "M1"
                fit_dir = args.output / "injection" / name / f"fit_{label}"
                report = run_variant(
                    synthetic,
                    variant="rsb" if include else "no_rsb",
                    included_surveys=SURVEYS,
                    output=fit_dir,
                    seed=args.seed + index,
                    draws=args.draws,
                    warmup=args.warmup,
                    chains=args.chains,
                )
                draws = load_draws(fit_dir / "posterior.npz", report["chain_shape"])
                entry["variants"][label] = {
                    "frequencies": frequency_rows(report),
                    "passed": report["passed"],
                    "recovery": recovery_rows(truth, draws, include),
                    "log_joint_truth": log_joint_at(
                        synthetic.tris_bundle,
                        synthetic.external,
                        complete_truth(truth, include),
                        include,
                    ),
                    "log_joint_recovered": log_joint_at(
                        synthetic.tris_bundle,
                        synthetic.external,
                        posterior_mean_values(draws, include),
                        include,
                    ),
                }
            entry["truth"] = {
                key: np.asarray(value).tolist() for key, value in truth.items()
            }
            entry["include_rsb"] = include_rsb
            document["injection"][name] = entry

    document["elapsed_s"] = round(time.time() - started, 2)
    (args.output / "batch_c.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + "\n"
    )
    (args.output / "batch_c.md").write_text(render(document))
    print(json.dumps({"output": str(args.output), "elapsed_s": document["elapsed_s"]}))
    return 0


def render(document: dict) -> str:
    lines = [
        "# Batch C: identifiability and injection recovery (P2)",
        "",
        (
            f"Run {document['run_id']} at {document['created_utc']}; "
            f"elapsed {document['elapsed_s']} s."
        ),
        f"bayesmith HEAD {document['git']}",
        "",
        "## Identifiability: prior-scaled whitened Jacobian",
        "",
        "Singular values of J times the prior width; a direction with SNR > 1 is",
        "data-constrained, the rest are supplied by the prior.",
        "",
        "| point | model | n obs | n par | SNR > 1 | singular values (top 6) |",
        "|---|---|---|---|---|---|",
    ]
    for point, models in document["identifiability"].items():
        for model, analysis in models.items():
            singular = np.round(
                np.sort(np.asarray(analysis["singular_values"]))[::-1][:6], 1
            )
            lines.append(
                "| {point} | {model} | {obs} | {par} | {snr} | {sv} |".format(
                    point=point,
                    model=model,
                    obs=analysis["observation_count"],
                    par=len(analysis["names"]),
                    snr=analysis["directions_with_snr_above_one"],
                    sv=np.array2string(singular),
                )
            )
    lines += [
        "",
        "Posterior-to-prior standard deviation per coordinate (1 means the data",
        "say nothing):",
        "",
        "| point | model | parameter | posterior SD / prior SD |",
        "|---|---|---|---|",
    ]
    for point, models in document["identifiability"].items():
        for model, analysis in models.items():
            names = analysis["names"]
            ratios = np.asarray(analysis["posterior_over_prior_sd"])
            for index, ratio in enumerate(ratios):
                lines.append(
                    f"| {point} | {model} | {names[index]} | {ratio:.4f} |"
                )
    lines += [
        "",
        "Largest posterior correlations:",
        "",
        "| point | model | a | b | correlation |",
        "|---|---|---|---|---|",
    ]
    for point, models in document["identifiability"].items():
        for model, analysis in models.items():
            for row in analysis["degeneracies"]:
                lines.append(
                    "| {point} | {model} | {a} | {b} | {c:.4f} |".format(
                        point=point, model=model, c=row["correlation"], **row
                    )
                )
    if document["injection"]:
        lines += [
            "",
            "## Injection recovery",
            "",
            "| scenario | fit | passed | freq | residual RMS K | chi2/N |",
            "|---|---|---|---|---|---|",
        ]
        for name, entry in document["injection"].items():
            for label, fit in entry["variants"].items():
                for row in fit["frequencies"]:
                    lines.append(
                        "| {name} | {label} | {passed} | {freq:.1f} MHz | {rms:.4f} | "
                        "{chi2:.1f} |".format(
                            name=name,
                            label=label,
                            passed=fit["passed"],
                            freq=row["frequency_mhz"],
                            rms=row["residual_rms_k"],
                            chi2=row["chi_square_per_observation"],
                        )
                    )
        lines += [
            "",
            "Log joint at the injected truth versus at the recovered posterior mean:",
            "a recovered point with the higher joint is where the sampler should be,",
            "and a truth with the higher joint is a sampling failure.",
            "",
            "| scenario | fit | log joint truth | log joint recovered | truth better |",
            "|---|---|---|---|---|",
        ]
        for name, entry in document["injection"].items():
            for label, fit in entry["variants"].items():
                lines.append(
                    "| {name} | {label} | {t:.1f} | {r:.1f} | {better} |".format(
                        name=name,
                        label=label,
                        t=fit["log_joint_truth"],
                        r=fit["log_joint_recovered"],
                        better=fit["log_joint_truth"] > fit["log_joint_recovered"],
                    )
                )
        lines += [
            "",
            "Recovery pull, (posterior mean - truth) / posterior SD:",
            "",
            "| scenario | fit | parameter | truth | mean | SD | pull | inside 95% |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, entry in document["injection"].items():
            for label, fit in entry["variants"].items():
                for row in fit["recovery"]:
                    lines.append(
                        "| {name} | {label} | {p} | {t:.4g} | {m:.4g} | {s:.4g} | "
                        "{pull:+.2f} | {cov} |".format(
                            name=name,
                            label=label,
                            p=row["parameter"],
                            t=row["truth"],
                            m=row["mean"],
                            s=row["sd"],
                            pull=row["pull"],
                            cov=row["covered_95"],
                        )
                    )
    lines += [
        "",
        "## Limits",
        "",
        "- The identifiability numbers are a Laplace (linearised) analysis at one",
        "  point, not the sampled posterior; the sampled posterior is Batch B.",
        "- The beam-width and beam-chromaticity blocks are not in this model yet,",
        "  so the a-width and beta-chromaticity degeneracies are not measured here.",
        "- The injection generator is an independent numpy path, but it shares the",
        "  prepared operator; the shifted-regions scenario is the only genuine",
        "  model-misspecification case.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
