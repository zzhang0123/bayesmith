"""Batch D: same-layer M0/M1 held-out predictive comparison.

The plan's P3/P4 requirement is a like-for-like comparison: identical data,
identical calibration model, identical compute budget, and the no-RSB versus
RSB pair evaluated at the SAME operator layer.  This driver adds the four
directional held-out refits (train LWA, score ARCADE and the reverse) to each
operator's Batch B output, assembles the existing comparison artifact, and puts
the two operators side by side.

It never mixes layers: the fixed-A pair and the audited-A pair are compared
separately, and the operator difference is reported as its own effect.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import jax

from examples.inference.tris_rsb_case import load_common_inputs, run_variant
from examples.inference.tris_rsb_compare import assemble_comparison

SURVEYS = ("LWA", "ARCADE")


def ensure_heldout(common, parent, *, seed, draws, warmup, chains):
    """Run any missing directional refit under parent/heldout."""
    for index, variant in enumerate(("no_rsb", "rsb")):
        for fold, survey in enumerate(SURVEYS):
            path = parent / "heldout" / f"{variant}_train_{survey}" / "result.json"
            if path.is_file():
                continue
            run_variant(
                common,
                variant=variant,
                included_surveys=(survey,),
                output=path.parent,
                seed=seed + 10 + 2 * index + fold,
                draws=draws,
                warmup=warmup,
                chains=chains,
            )


def heldout_rows(comparison):
    rows = []
    for entry in comparison.get("heldout_scores", []):
        for variant in ("no_rsb", "rsb"):
            score = entry["models"][variant]
            rows.append(
                {
                    "train": entry["train"],
                    "heldout": entry["heldout"],
                    "variant": variant,
                    "log_predictive_density": score["log_predictive_density"],
                    "mcse": score["mcse"],
                }
            )
    return rows


def delta_rows(comparison):
    return [
        {
            "train": entry["train"],
            "heldout": entry["heldout"],
            "delta_M1_minus_M0": entry["delta_M1_minus_M0"],
            "delta_mcse": entry["delta_mcse"],
        }
        for entry in comparison.get("heldout_scores", [])
    ]


def render(document: dict) -> str:
    lines = [
        "# Batch D: same-layer held-out comparison",
        "",
        (
            f"Run {document['run_id']} at {document['created_utc']}; "
            f"elapsed {document['elapsed_s']} s."
        ),
        f"bayesmith HEAD {document['git']}",
        "",
        "Both operators use the same external data, the same calibration model and",
        "the same budget; within each row M0 and M1 differ only by the RSB term.",
        "",
        "## Held-out log predictive density",
        "",
        "| operator | train | held out | model | log predictive density | MCSE |",
        "|---|---|---|---|---|---|",
    ]
    for operator, block in document["comparisons"].items():
        for row in block["rows"]:
            lines.append(
                "| {op} | {train} | {held} | {var} | {score:.4f} | {mcse:.4f} |".format(
                    op=operator,
                    train=row["train"],
                    held=row["heldout"],
                    var=row["variant"],
                    score=row["log_predictive_density"],
                    mcse=row["mcse"],
                )
            )
    lines += [
        "",
        "## M1 minus M0, same layer",
        "",
        "| operator | train | held out | delta | combined MCSE | delta / MCSE |",
        "|---|---|---|---|---|---|",
    ]
    for operator, block in document["comparisons"].items():
        for row in block["deltas"]:
            ratio = (
                row["delta_M1_minus_M0"] / row["delta_mcse"]
                if row["delta_mcse"] > 0
                else 0.0
            )
            lines.append(
                "| {op} | {train} | {held} | {delta:+.4f} | {mcse:.4f} | {ratio:+.2f} |".format(
                    op=operator,
                    train=row["train"],
                    held=row["heldout"],
                    delta=row["delta_M1_minus_M0"],
                    mcse=row["delta_mcse"],
                    ratio=ratio,
                )
            )
    lines += [
        "",
        "## Operator effect (audited minus fixed), same model and fold",
        "",
        "| train | held out | model | fixed | audited | delta |",
        "|---|---|---|---|---|---|",
    ]
    if "operator_delta" in document:
        for row in document["operator_delta"]:
            lines.append(
                "| {train} | {heldout} | {variant} | {fixed:.4f} | {audited:.4f} | "
                "{delta:+.4f} |".format(**row)
            )
    lines += [
        "",
        "## Limits",
        "",
        "- LWA and ARCADE are derived, foreground-subtracted summaries that share",
        "  literature inputs; a directional score is not independent instrument",
        "  validation, and its calibration nuisance is marginalised only within the",
        "  stated model.",
        "- These are conditional predictive scores, not a Bayes factor.  No prior",
        "  normalisation or evidence computation is claimed.",
        "- The comparison is run twice, once per operator, because the operator",
        "  revision changes the data vector; the two are not pooled.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-b", type=Path, required=True)
    parser.add_argument("--frozen-input", type=Path, default=Path("runs/tris-input-skyfields"))
    parser.add_argument("--audited-input", type=Path, required=True)
    parser.add_argument("--external-input", type=Path, default=Path("runs/tris-rsb-input"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "bayesmith.tris.batch-d.v1",
        "run_id": args.output.name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": subprocess.check_output(
            ["git", "-C", ".", "rev-parse", "HEAD"], text=True
        ).strip(),
        "comparisons": {},
    }

    with jax.enable_x64(True):
        for operator, tris_input in (
            ("fixed", args.frozen_input),
            ("audited", args.audited_input),
        ):
            common = load_common_inputs(tris_input, args.external_input)
            parent = args.batch_b / operator
            if not args.skip_runs:
                ensure_heldout(
                    common,
                    parent,
                    seed=args.seed,
                    draws=args.draws,
                    warmup=args.warmup,
                    chains=args.chains,
                )
            comparison = assemble_comparison(parent)
            document["comparisons"][operator] = {
                "status": comparison["comparison_status"],
                "rows": heldout_rows(comparison),
                "deltas": delta_rows(comparison),
            }

    fixed_rows = {
        (row["train"], row["heldout"], row["variant"]): row
        for row in document["comparisons"]["fixed"]["rows"]
    }
    audited_rows = {
        (row["train"], row["heldout"], row["variant"]): row
        for row in document["comparisons"]["audited"]["rows"]
    }
    document["operator_delta"] = [
        {
            "train": key[0],
            "heldout": key[1],
            "variant": key[2],
            "fixed": fixed_rows[key]["log_predictive_density"],
            "audited": audited_rows[key]["log_predictive_density"],
            "delta": audited_rows[key]["log_predictive_density"]
            - fixed_rows[key]["log_predictive_density"],
        }
        for key in sorted(fixed_rows)
        if key in audited_rows
    ]

    document["elapsed_s"] = round(time.time() - started, 2)
    (args.output / "batch_d.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, default=float) + "\n"
    )
    (args.output / "batch_d.md").write_text(render(document))
    print(json.dumps({"output": str(args.output), "elapsed_s": document["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
