"""RA-block held-out prediction for the smooth TRIS case.

The external-survey held-out scores train on one survey and predict the other;
they never hold out a block of the original scan.  This module does the scan
holdout the P0/P3 plan asks for: the 120 RA samples are split into contiguous,
pre-registered blocks with a fixed buffer removed on each side, each block is
held out, the smooth M0/M1 model is REFIT on the rest (the training likelihood
is recompressed from the retained ring rows, not merely masked in mode space),
and the held-out block is scored in per-point sigma units.

Deltas carry the same convergence gate as the survey heldout: a nonconverged
refit contributes no comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

from examples.inference.tris_audit import dense_wiener
from examples.inference.tris_maps import compress_map
from examples.inference.tris_rsb_compare import _logmeanexp
from examples.inference.tris_smooth_case import (
    FUNCTIONS,
    SURVEYS,
    basis_matrix,
    run_smooth,
)

NL = chr(10)


def ra_blocks(samples, *, folds, buffer):
    """Pre-registered contiguous RA blocks, with a buffer excluded from training."""
    if samples < 2 or not 2 <= folds <= samples or buffer < 0:
        raise ValueError("require samples >= folds >= 2 and nonnegative buffer")
    edges = np.linspace(0, samples, folds + 1).astype(int)
    blocks = []
    for k in range(folds):
        held = np.arange(edges[k], edges[k + 1])
        blocked = np.unique(np.arange(edges[k] - buffer, edges[k + 1] + buffer) % samples)
        train = np.setdiff1d(np.arange(samples), blocked)
        if train.size == 0:
            raise ValueError("buffer leaves no training rows")
        blocks.append(
            {
                "fold": k,
                "heldout_start": int(edges[k]),
                "heldout_stop": int(edges[k + 1]),
                "buffer": int(buffer),
                "heldout": held.tolist(),
                "train": train.tolist(),
            }
        )
    return blocks


def compress_for_rows(bundle, keep, index):
    """Recompress the training ring rows into a map likelihood."""
    a = np.asarray(bundle[f"operator_{index}"], dtype=float)[keep]
    sigma = np.asarray(bundle[f"sigma_k_{index}"], dtype=float)[keep]
    data = np.asarray(bundle[f"data_k_{index}"], dtype=float)[keep]
    prior = np.asarray(bundle[f"prior_k_{index}"], dtype=float)
    prior_sigma = np.asarray(bundle[f"prior_sigma_k_{index}"], dtype=float)
    dense = dense_wiener(a, data, sigma, prior, prior_sigma)
    likelihood = compress_map(dense.mean, prior, a, sigma, dense.covariance)
    return likelihood, a, sigma, data


def masked_bundle(bundle, keep):
    """Bundle whose compressed and raw TRIS rows are the retained RA samples."""
    out = dict(bundle)
    for index in (0, 1):
        likelihood, a, sigma, data = compress_for_rows(bundle, keep, index)
        out[f"response_{index}"] = likelihood.response
        out[f"offset_response_{index}"] = likelihood.offset_response
        out[f"whitened_data_{index}"] = likelihood.data
        out[f"operator_{index}"] = a
        out[f"sigma_k_{index}"] = sigma
        out[f"data_k_{index}"] = data
        out[f"ra_deg_{index}"] = np.asarray(bundle[f"ra_deg_{index}"], dtype=float)[keep]
    return out


def ring_log_scores(bundle, draws, basis, include_rsb, rows):
    """Per-draw Gaussian log density of the held-out ring rows."""
    log_amplitude = np.asarray(draws["log_amplitude_coefficients"], dtype=float)
    beta = np.asarray(draws["beta_coefficients"], dtype=float)
    zero = np.asarray(draws["zero_standard"], dtype=float)
    monopole = np.asarray(draws["haslam_monopole_K"], dtype=float)
    count = log_amplitude.shape[0]
    if include_rsb:
        rsb_amplitude = np.asarray(draws["rsb_amplitude"], dtype=float)
        rsb_beta = np.asarray(draws["rsb_beta"], dtype=float)
        background_408 = rsb_amplitude * 0.408 ** rsb_beta
    else:
        rsb_amplitude = np.zeros(count)
        rsb_beta = np.zeros(count)
        background_408 = np.zeros(count)
    amplitude = np.exp(log_amplitude @ basis.T)
    slope = beta @ basis.T
    levels = zero * np.where(
        np.arange(2)[None, :] == 0, 0.066, np.where(zero[:, 1:2] < 0, 0.300, 0.430)
    )
    template = np.asarray(bundle["template_k"], dtype=float)
    scores = np.zeros(count)
    rows = np.asarray(rows, dtype=int)
    for index, frequency in enumerate(np.asarray(bundle["frequency_mhz"], dtype=float)):
        operator = np.asarray(bundle[f"operator_{index}"], dtype=float)[rows]
        sigma = np.asarray(bundle[f"sigma_k_{index}"], dtype=float)[rows]
        observed = np.asarray(bundle[f"data_k_{index}"], dtype=float)[rows]
        background = (
            rsb_amplitude * (frequency / 1000.0) ** rsb_beta if include_rsb else np.zeros(count)
        )
        sky = (
            (template[None, :] + monopole[:, None] - background_408[:, None])
            * amplitude
            * (frequency / 408.0) ** slope
            + float(bundle["cmb_k"][index])
            + background[:, None]
        )
        mean = sky @ operator.T - levels[:, index : index + 1]
        residual = (observed[None, :] - mean) / sigma[None, :]
        scores += -0.5 * np.sum(residual**2, axis=1) - np.sum(np.log(sigma))
    return scores - 0.5 * len(bundle["frequency_mhz"]) * rows.size * np.log(2.0 * np.pi)


def run_fold(bundle, external, basis, output, block, variant, *, seed, draws, warmup, chains):
    include_rsb = variant == "rsb"
    directory = output / "blocks" / f"fold{block['fold']}_{variant}"
    train_bundle = masked_bundle(bundle, np.asarray(block["train"], dtype=int))
    report = run_smooth(
        train_bundle,
        external,
        SURVEYS,
        include_rsb,
        directory,
        basis=basis,
        seed=seed,
        draws=draws,
        warmup=warmup,
        chains=chains,
    )
    with np.load(directory / "posterior.npz", allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    scores = ring_log_scores(
        bundle, saved, basis, include_rsb, np.asarray(block["heldout"], dtype=int)
    )
    diagnostics = report.get("checks", {}).get("chain_diagnostics", {})
    sites = diagnostics.get("sites", {})
    return {
        "fold": block["fold"],
        "heldout_rows": len(block["heldout"]),
        "train_rows": len(block["train"]),
        "variant": variant,
        "log_predictive_density": _logmeanexp(scores),
        "passed": bool(report["passed"]),
        "divergences": diagnostics.get("divergences"),
        "sampling_policy_passed": diagnostics.get("sampling_policy_passed"),
        "r_hat_max": max((value["r_hat"] for value in sites.values()), default=None),
        "ess_min": min((value["ess"] for value in sites.values()), default=None),
    }


def block_deltas(rows):
    """Gate each fold on BOTH refits passing; no delta from a failed chain."""
    deltas = []
    for fold in sorted({row["fold"] for row in rows}):
        selected = {row["variant"]: row for row in rows if row["fold"] == fold}
        missing = [v for v in ("no_rsb", "rsb") if v not in selected]
        invalid = [v for v, row in selected.items() if not row["passed"]]
        valid = not missing and not invalid
        entry = {
            "fold": fold,
            "valid": bool(valid),
            "invalid_refits": invalid,
            "missing_refits": missing,
        }
        if valid:
            entry["delta_M1_minus_M0"] = (
                selected["rsb"]["log_predictive_density"]
                - selected["no_rsb"]["log_predictive_density"]
            )
            entry["status"] = "available"
        else:
            entry["delta_M1_minus_M0"] = None
            entry["status"] = "invalid_nonconverged_refit" if invalid else "missing_refit"
        deltas.append(entry)
    return deltas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-b", type=Path, required=True)
    parser.add_argument("--external-input", type=Path, default=Path("runs/tris-rsb-input"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--buffer", type=int, default=2)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=800)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--variants", nargs="+", default=["no_rsb", "rsb"])
    parser.add_argument("--only-fold", type=int, nargs="+", default=None)
    parser.add_argument("--functions", type=int, default=FUNCTIONS)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    with np.load(args.batch_b / "audited-input" / "maps.npz", allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    with np.load(args.external_input / "external.npz", allow_pickle=False) as archive:
        external = {name: archive[name] for name in archive.files}
    basis = basis_matrix(bundle, args.functions)
    samples = len(np.asarray(bundle["ra_deg_0"]))
    blocks = ra_blocks(samples, folds=args.folds, buffer=args.buffer)
    if args.only_fold is not None:
        blocks = [block for block in blocks if block["fold"] in set(args.only_fold)]
        if not blocks:
            raise SystemExit("no block matches --only-fold")
    rows = []
    with jax.enable_x64(True):
        for block in blocks:
            for index, variant in enumerate(args.variants):
                rows.append(
                    run_fold(
                        bundle,
                        external,
                        basis,
                        args.output,
                        block,
                        variant,
                        seed=args.seed + 3 * block["fold"] + index,
                        draws=args.draws,
                        warmup=args.warmup,
                        chains=args.chains,
                    )
                )
    summary = {
        "case": "tris_haslam_smooth_ra_blocks",
        "folds": args.folds,
        "buffer": args.buffer,
        "block_edges": [block["heldout_start"] for block in blocks]
        + [blocks[-1]["heldout_stop"]],
        "rows": rows,
        "deltas": block_deltas(rows),
        "note": "Held-out RA blocks of the original scan, refit on the rest; ",
        "elapsed_s": round(time.time() - started, 2),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ra_block_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + NL
    )
    print(json.dumps({"output": str(args.output), "elapsed_s": summary["elapsed_s"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
