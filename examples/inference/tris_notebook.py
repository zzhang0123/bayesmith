"""Build, execute and render the runnable TRIS forward-inference notebook.

The plan's Batch E asks for a notebook that actually runs against the recorded
batch artifacts and produces the executed HTML, not a static six-figure page.
The notebook source lives here so it is version-controlled and testable; this
module writes tris_forward_inference.ipynb, executes its code cells in the
current interpreter, saves the executed notebook and renders a self-contained
HTML page with the markdown prose, the cell source, its stdout and every figure
the cell wrote.

The second acceptance found that the first renderer dropped every markdown cell,
so the titles, the equations and the conclusions never reached the page, and
that the conclusions were hard-coded.  This version renders markdown (headings,
lists, code blocks and equations), builds a table of contents, derives the
conclusions from the loaded diagnostics, and exposes a rerun entry
(--configs/--rerun) that executes a recorded inference command.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import io
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

NL = chr(10)

SETUP = '''import json
import os
from pathlib import Path

NL = chr(10)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(os.environ["TRIS_NOTEBOOK_OUTPUT"])
ASSETS = OUTPUT / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)
RUNS = json.loads(os.environ["TRIS_NOTEBOOK_RUNS"])
RUNS = {key: Path(value) for key, value in RUNS.items()}
CONFIGS = json.loads(os.environ.get("TRIS_NOTEBOOK_CONFIGS", "{}"))


def load(path, filename):
    path = Path(path) / filename
    return json.loads(path.read_text()) if path.is_file() else {}


def save(fig, name):
    target = ASSETS / name
    fig.savefig(target, format="svg")
    plt.close(fig)
    print("wrote", target)


style = {"font.size": 10, "axes.grid": True, "grid.alpha": 0.3}
plt.rcParams.update(style)
print("batches:", {key: str(value) for key, value in RUNS.items()})
print("rerun configs:", sorted(CONFIGS))
'''

DATA_CELL = '''audit = load(RUNS["a"], "audit.json")
batch_b = load(RUNS["b"], "batch_b.json")
batch_c = load(RUNS["c"], "batch_c.json")
batch_d = load(RUNS["d"], "batch_d.json")
batch_d2 = load(RUNS["d2"], "batch_d2.json")
beam = load(RUNS["beam"], "beam_identifiability.json")
smooth = load(RUNS["smooth"], "smooth_summary_gated.json") or load(
    RUNS["smooth"], "smooth_summary.json"
)
budget = load(RUNS["budget"], "p0_budget.json")
joint = load(RUNS["joint"], "joint_sky_beam.json") if "joint" in RUNS else {}
ra = load(RUNS["ra"], "ra_block_summary.json") if "ra" in RUNS else {}
surrogate = load(RUNS["surrogate"], "beam_surrogate.json") if "surrogate" in RUNS else {}
profile = load(RUNS["profile"], "joint_profile.json") if "profile" in RUNS else {}
sbc = load(RUNS["sbc"], "joint_sbc.json") if "sbc" in RUNS else {}
sensitivity = load(RUNS["sensitivity"], "sensitivity.json") if "sensitivity" in RUNS else {}
independent = load(RUNS["independent"], "joint_sky_beam.json") if "independent" in RUNS else {}
alternative_blocks = load(RUNS["block_sensitivity"], "block-sensitivity.json") if "block_sensitivity" in RUNS else {}

counts = audit.get("p0a", {}).get("provenance_counts", {})
print("P0a provenance counts:")
for key, value in sorted(counts.items()):
    print("  ", key, value)

change = batch_b.get("operator_change", {})
print()
print("operator change max abs:", change.get("max_abs"))
print("prediction rms K:", change.get("prediction_rms_k"))
print()
print("rerun entry: python -m examples.inference.tris_notebook --output <dir> --runs <json> --configs <json> --rerun <name>")
for name, command in sorted(CONFIGS.items()):
    print("  ", name, "->", " ".join(command))
'''

OPERATOR_CELL = '''ladder = audit.get("p0c", {}).get("beam_ladder", {})
keys = sorted(ladder, key=int)
budget_ladder = budget.get("ladder", {})
adjacent = budget.get("adjacent", {})
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
if keys:
    axes[0].plot([int(k) for k in keys],
                 [ladder[k]["prediction_rms_k"] for k in keys], "o-", label="vs control [K]")
if adjacent:
    akeys = list(adjacent)
    axes[0].plot([int(k.split("->")[1]) for k in akeys],
                 [adjacent[k]["600_5"]["prediction_rms_sigma"] for k in akeys],
                 "s--", label="adjacent [sigma]")
axes[0].set_ylabel("difference")
axes[0].set_xscale("log", base=2)
axes[0].legend()
axes[0].set_xlabel("beam built directly at nside h")
axes[0].axhline(0.1, color="#bbbbbb", linestyle=":")
if budget_ladder:
    bkeys = sorted(budget_ladder, key=int)
    axes[1].bar([str(k) for k in bkeys],
                [budget_ladder[k]["600_5"]["prediction_max_abs_sigma"] for k in bkeys],
                color="#4477aa")
axes[1].set_xlabel("nside h")
axes[1].set_ylabel("max difference [sigma]")
axes[1].axhline(0.3, color="#bbbbbb", linestyle=":")
save(fig, "operator_convergence.svg")

oracle = budget.get("independent_oracle", {})
print("independent pixel-space oracle vs production: max abs",
      oracle.get("max_abs"), "relative rms", oracle.get("relative_rms"))
print("zero-sigma archive rows:", budget.get("zero_sigma_archive_rows"))
print("zero-sigma delta chi-square:", budget.get("zero_sigma_row", {}).get("delta_chi_square"))
for name, entry in sorted(budget.get("sensitivities", {}).items()):
    print(f"{name:16s} rms sigma {entry['prediction_rms_sigma']:.3f} "
          f"max sigma {entry['prediction_max_abs_sigma']:.3f} "
          f"shift/posterior {entry['shift']['max_delta_over_posterior_sd']:.3f} "
          f"loglik-delta {entry['gaussian_log_likelihood_delta']:.3f}")
for name, entry in sorted(adjacent.items()):
    print(f"adjacent {name:10s} rms600 {entry['600_5']['prediction_rms_sigma']:.3f} "
          f"max600 {entry['600_5']['prediction_max_abs_sigma']:.3f} "
          f"data-loglik {entry['data_weighted_log_likelihood_delta']:.1f} "
          f"pass {entry['budget_pass']}")
'''

BASELINE_CELL = '''fixed = batch_b.get("frequencies", {}).get("fixed/no_rsb", [])
audited = batch_b.get("frequencies", {}).get("audited/no_rsb", [])
if fixed and audited:
    labels = [f"{row['frequency_mhz']:.1f}" for row in fixed]
    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    axes[0].bar(x - width / 2, [r["residual_rms_k"] for r in fixed], width, label="fixed A")
    axes[0].bar(x + width / 2, [r["residual_rms_k"] for r in audited], width, label="audited A")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("residual rms [K]")
    axes[0].legend()
    axes[1].bar(x - width / 2, [r["chi_square_per_observation"] for r in fixed], width, label="fixed A")
    axes[1].bar(x + width / 2, [r["chi_square_per_observation"] for r in audited], width, label="audited A")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("chi2 / N")
    axes[1].legend()
    save(fig, "baseline_residuals.svg")
    for row, trial in zip(fixed, audited):
        print(f"{row['frequency_mhz']:.1f} MHz: rms {row['residual_rms_k']:.4f} -> "
              f"{trial['residual_rms_k']:.4f} K, chi2/N {row['chi_square_per_observation']:.1f} -> "
              f"{trial['chi_square_per_observation']:.1f}")
    for _row, _trial in zip(fixed, audited):
        print("  published chi2/N only; not a reduced chi-square with fitted dof")
'''

BEAM_CELL = '''maps = np.load(RUNS["b"] / "audited-input" / "maps.npz", allow_pickle=False)
angle = np.asarray(maps["beam_angle_deg"], dtype=float)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
axes[0].plot(angle, np.asarray(maps["beam_e_db"], dtype=float), label="E plane")
axes[0].plot(angle, np.asarray(maps["beam_h_db"], dtype=float), label="H plane")
axes[0].set_xlabel("angle on the cut [deg]")
axes[0].set_ylabel("gain [dB]")
axes[0].set_title("archive principal-plane cuts")
axes[0].legend()
analysis = beam.get("analysis", {})
if analysis:
    names = analysis["names"]
    ratios = np.asarray(analysis["posterior_over_prior_sd"])
    order = np.argsort(ratios)
    axes[1].barh([names[i] for i in order], ratios[order], color="#228833")
    axes[1].axvline(1.0, color="#bbbbbb", linestyle="--")
    axes[1].set_xlabel("posterior SD / prior SD (whitened)")
    axes[1].set_title("beam + foreground identifiability")
save(fig, "beam_cuts_identifiability.svg")
support = beam.get("support", {})
print("width ratios:", dict(zip(analysis.get("names", [])[-2:],
                               analysis.get("posterior_over_prior_sd", [])[-2:])))
print("kink distance [prior SD]:", support.get("zero_standard_1_kink_distance_prior_sd"))
print("point spread:", beam.get("point_stability", {}).get("width_ratio_max_relative_spread"))
print("width-step spread:", beam.get("width_step_stability", {}).get("width_ratio_max_relative_spread"))
legality = beam.get("deformation_legality", {})
if legality:
    for key, entry in sorted(legality.items()):
        print(f"deformation legality {key}: min {entry['minimum']:.3e} "
              f"negative {entry['negative_entries']} sum {entry['sum']:.1f}")
provenance = beam.get("operators_provenance", {})
print("operator provenance:", provenance)
if surrogate:
    print("surrogate width step", surrogate.get("width_step"),
          "beam nside", surrogate.get("beam_nside"),
          "legal bound", surrogate.get("legal_bound_recommendation"),
          "gradient relative error", surrogate.get("gradient", {}).get("relative_error"))
    print("surrogate zero-deformation operator", surrogate.get("zero_deformation_operator"))
    print("max beam negative entries over grid",
          surrogate.get("max_beam_negative_entries_over_grid"),
          "max operator negative entries", surrogate.get("max_negative_entries_over_grid"))
    for key, entry in sorted(surrogate.get("grid", {}).items()):
        print(f"surrogate {key}: rms600 {entry['600_5']['prediction_rms_sigma']:.4f} "
              f"max600 {entry['600_5']['prediction_max_abs_sigma']:.4f} "
              f"pass {entry['budget_pass']}")

joint_draws = RUNS["joint"] / "joint_rsb_posterior.npz" if "joint" in RUNS else None
if (joint_draws is not None and joint_draws.is_file()
        and joint.get("injections", {}).get("joint_rsb", {}).get("passed", False)):
    with np.load(joint_draws, allow_pickle=False) as archive:
        joint_names = list(archive.files)
        width_e = (np.asarray(archive["log_width_e"], dtype=float).ravel()
                   if "log_width_e" in joint_names else None)
        width_h = (np.asarray(archive["log_width_h"], dtype=float).ravel()
                   if "log_width_h" in joint_names else None)
    if width_e is not None and width_h is not None:
        figure, axis = plt.subplots(figsize=(7, 3.4))
        axis.hist(width_e, bins=40, alpha=0.6, label="log width E")
        axis.hist(width_h, bins=40, alpha=0.6, label="log width H")
        axis.set_xlabel("log width deformation")
        axis.set_ylabel("posterior draws")
        axis.set_title("joint beam-width posterior (achromatic)")
        axis.legend()
        save(figure, "joint_beam_widths.svg")
        from scipy.stats import truncnorm
        prior_sd_width = float(joint.get("width_prior_sd", 0.05))
        width_bound = float(joint.get("width_legal_bound", 0.01))
        actual_prior_sd = truncnorm.std(-width_bound/prior_sd_width, width_bound/prior_sd_width, scale=prior_sd_width)
        print("Actual truncated width prior SD:", actual_prior_sd, "posterior/prior SD E,H:",
              float(width_e.std(ddof=1)/actual_prior_sd), float(width_h.std(ddof=1)/actual_prior_sd))
        print("joint log width E 16/50/84:", np.percentile(width_e, [16, 50, 84]).tolist())
        print("joint log width H 16/50/84:", np.percentile(width_h, [16, 50, 84]).tolist())
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        for axis, widths, key, label in zip(axes, (width_e, width_h), ("beam_e_db", "beam_h_db"), ("E", "H")):
            original = np.asarray(maps[key])
            curves = np.array([np.interp(angle / np.exp(width), angle, original)
                               for width in widths[::max(1,len(widths)//300)]])
            axis.plot(angle, original, label="measured cut")
            axis.plot(angle, np.median(curves,0), label="posterior median")
            axis.fill_between(angle, *np.quantile(curves,[.025,.975],axis=0), alpha=.3, label="95% conditional band")
            axis.set_xlim(0, 45)
            axis.set_ylim(-40, 0)
            axis.set_title(label + " cut: angular width posterior")
            axis.set_xlabel("angle [deg]")
            axis.set_ylabel("gain [dB]")
            axis.legend()
        save(figure, "posterior_beam_cuts.svg")
        offsets = np.linspace(-40, 40, 101)
        xx, yy = np.meshgrid(offsets, offsets)
        theta = np.hypot(xx, yy)
        phi = np.arctan2(yy, xx)
        weight_e = np.cos(phi)**2
        beam_images = []
        stride = max(1, len(width_e)//150)
        for we, wh in zip(width_e[::stride], width_h[::stride]):
            e = np.interp(theta / np.exp(we), angle, np.asarray(maps["beam_e_db"]))
            h = np.interp(theta / np.exp(wh), angle, np.asarray(maps["beam_h_db"]))
            beam_images.append(10**((weight_e*e + (1-weight_e)*h)/10))
        beam_images = np.asarray(beam_images)
        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        for axis, values, label in zip(axes, (beam_images.mean(0),beam_images.std(0,ddof=1)),
                                       ("mean peak-normalized power", "conditional posterior SD")):
            artist=axis.imshow(values, origin="lower", extent=[-40,40,-40,40])
            axis.set_xlabel("E offset [deg]")
            axis.set_ylabel("H offset [deg]")
            axis.set_title(label)
            figure.colorbar(artist, ax=axis)
        save(figure, "posterior_beam_2d.svg")
        print("2D beam uses the assumed dB principal-plane interpolation; it is not an independent 2D measurement.")
    else:
        print("joint beam width is fixed in this run; recorded columns", joint_names)
'''

PREDICTIVE_CELL = '''path = RUNS["joint"] / "predictive_products.npz"
if path.is_file():
    with np.load(path, allow_pickle=False) as archive:
        products = dict(archive)
    data, sigma, ra_axis = products["data"], products["sigma"], products["ra_deg"]
    for label in ("joint_m0", "joint_rsb"):
        mean, sd = products[label + "_mean"], products[label + "_sd"]
        figure, axes = plt.subplots(3, 2, figsize=(12, 9))
        for frequency in range(2):
            axis = axes[0, frequency]
            axis.errorbar(ra_axis, data[:, frequency], yerr=sigma[:, frequency], fmt=".", label="data")
            axis.plot(ra_axis, mean[:, frequency], label="posterior mean")
            predictive_sd = np.sqrt(sd[:, frequency]**2 + sigma[:, frequency]**2)
            axis.fill_between(ra_axis, mean[:, frequency]-2*predictive_sd,
                              mean[:, frequency]+2*predictive_sd, alpha=.3, label="predictive +/-2 SD")
            axis.set_title(label + " " + str([600.5,817.8][frequency]) + " MHz")
            axis.legend()
            residual = data[:, frequency]-mean[:, frequency]
            axes[1, frequency].plot(ra_axis, residual / sigma[:, frequency])
            axes[1, frequency].set_ylabel("residual / statistical sigma")
            axes[1, frequency].set_xlabel("RA [deg]")
            modes = np.arange(13)
            projected = np.abs(np.exp(-1j*np.deg2rad(ra_axis[:,None])*modes) .T @ residual)/len(ra_axis)
            axes[2, frequency].stem(modes, projected)
            axes[2, frequency].set_xlabel("m; direct projection at actual nonuniform RA labels")
            axes[2, frequency].set_ylabel("residual mode amplitude [K]")
        figure.tight_layout()
        save(figure, label + "_predictive.svg")
    print(load(RUNS["joint"], "predictive_products.json"))
'''


IDENT_CELL = '''analysis = batch_c.get("identifiability", {}).get("posterior_mean", {}).get("M1", {})
if analysis:
    names = analysis["names"]
    ratios = np.asarray(analysis["posterior_over_prior_sd"])
    order = np.argsort(ratios)
    fig, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.barh([names[i] for i in order], ratios[order], color="#228833")
    axis.axvline(1.0, color="#bbbbbb", linestyle="--")
    axis.set_xlabel("posterior SD / prior SD (1 = prior supplies the value)")
    save(fig, "identifiability_m1.svg")
    for name, ratio in zip(names, ratios):
        print(f"{name:28s} {ratio:.4f}")

scenarios = batch_c.get("injection", {})
rows = []
for name, entry in scenarios.items():
    for label, fit in entry.get("variants", {}).items():
        pulls = [abs(row["pull"]) for row in fit.get("recovery", [])]
        if pulls:
            rows.append((f"{name}/{label}", max(pulls)))
if rows:
    labels, values = zip(*rows)
    fig, axis = plt.subplots(figsize=(8, 4.2))
    axis.barh(list(labels), list(values), color="#aa3377")
    axis.axvline(2.0, color="#bbbbbb", linestyle="--")
    axis.set_xscale("log")
    axis.set_xlabel("max absolute recovery pull [posterior SD]")
    save(fig, "injection_recovery.svg")
'''

EXTENSIONS_CELL = '''rows = batch_d2.get("extensions", [])
if rows:
    fig, axis = plt.subplots(figsize=(6.5, 3.4))
    axis.bar([row["label"] for row in rows],
             [row["chi_square_per_observation"] for row in rows], color="#ee7733")
    axis.set_ylabel("chi2 / N at the MAP")
    axis.set_title("foreground basis flexibility")
    save(fig, "extensions.svg")
    for row in rows:
        print(f"{row['label']:12s} parameters {row['parameters']:2d} chi2/N "
              f"{row['chi_square_per_observation']:.1f}")
print()
for entry in batch_d2.get("sensitivity", []):
    print(f"{entry['label']:34s} {entry['variant']:8s} ESS {entry['ess']:.0f} "
          f"trustworthy {entry['trustworthy']}")
'''

SBC_CELL = '''if sbc and sbc.get("schema") != "rheplicant.tris.joint-sbc.v2":
    print("Legacy SBC has no valid complete gate; summaries withdrawn pending reclassification.")
    sbc = {**sbc, "effective_simulations": 0, "summary": {}, "records": []}
print("finite SBC generator:", sbc.get("truth_generator", "not configured"))
print("effective simulations:", sbc.get("effective_simulations"), "of",
      sbc.get("simulations"), " (complete multichain diagnostic gate)")
failed = sbc.get("failed_simulations", [])
if failed:
    print("gated out:", [(row["simulation"], row["divergences"]) for row in failed])
for name, entry in sorted(sbc.get("summary", {}).items()):
    print(f"{name:18s} trials {entry['n_trials']:3d}  mean rank {entry['mean_rank']:.3f} "
          f"+/- {entry['mean_rank_sd_uniform']:.3f}  coverage 95 {entry['coverage_95']:.3f} "
          f"interval {entry.get('coverage_95_interval', 'legacy: unavailable')}")
passing = sbc.get("records", [])
if passing:
    labels = []
    values = []
    for record in passing:
        diagnostics = record.get("diagnostics", {})
        rhat = max(
            (value["r_hat_max"] for value in diagnostics.values()),
            default=float("nan"),
        )
        labels.append(str(record["simulation"]))
        values.append(rhat)
    figure, axis = plt.subplots(figsize=(8, 3.2))
    axis.bar(labels, values, color="#117733")
    axis.axhline(1.01, color="#bbbbbb", linestyle="--")
    axis.set_ylabel("max rank split R-hat")
    axis.set_xlabel("passing simulation index")
    axis.set_title("SBC fits that passed complete multichain diagnostics")
    save(figure, "sbc_diagnostics.svg")
'''

COMPARISON_CELL = '''entries = []
for operator, block in batch_d.get("comparisons", {}).items():
    for row in block.get("deltas", []):
        if row.get("valid", True) and row.get("delta_M1_minus_M0") is not None:
            entries.append((operator, row))
        else:
            print("Unavailable historical comparison:", operator, row.get("status", "invalid"))
if entries:
    labels = [f"{op}/{row['train']}->{row['heldout']}" for op, row in entries]
    values = [row["delta_M1_minus_M0"] for _op, row in entries]
    valid = [bool(row.get("valid", True)) for _op, row in entries]
    fig, axis = plt.subplots(figsize=(8, 3.4))
    axis.bar(labels, values,
             color=["#332288" if ok else "#cccccc" for ok in valid])
    axis.set_ylabel("M1 - M0 held-out log density [nats]")
    axis.tick_params(axis="x", rotation=20)
    save(fig, "same_layer_comparison.svg")

smooth_deltas = smooth.get("heldout", {}).get("deltas", [])
for entry in smooth_deltas:
    status = "valid" if entry.get("valid") else entry.get("status")
    print(f"smooth {entry['train']}->{entry['heldout']}: "
          f"delta {entry.get('delta_M1_minus_M0')} [{status}]")

for row in ra.get("rows", []):
    print(f"RA fold {row['fold']} {row['variant']}: passed {row['passed']} "
          f"divergences {row.get('divergences')} rhat_max {row.get('r_hat_max')} "
          f"ess_min {row.get('ess_min')}")
for entry in ra.get("deltas", []):
    status = "valid" if entry.get("valid") else entry.get("status")
    print(f"RA block fold {entry['fold']}: delta {entry.get('delta_M1_minus_M0')} "
          f"MCSE {entry.get('mcse_delta')} [{status}]")
available = [row for row in ra.get("deltas", []) if row.get("valid")]
if available:
    fig, axis = plt.subplots(figsize=(7, 3.5))
    axis.errorbar([str(row["fold"]) for row in available],
                  [row["delta_M1_minus_M0"] for row in available],
                  yerr=[2 * row.get("mcse_delta", 0.) for row in available], fmt="o")
    axis.axhline(0, color="#888888", linestyle="--")
    axis.set_xlabel("held-out RA block")
    axis.set_ylabel("M1 - M0 predictive log density [nats]")
    axis.set_title("Matched model layers; error bars = 2 estimated MCSE")
    save(fig, "ra_bridge_comparison.svg")
    print("Predictive estimator:", ra.get("method"))
if alternative_blocks:
    print("Alternative block geometry:", alternative_blocks["geometry"])
    for row in alternative_blocks["deltas"]:
        print("  fold", row["fold"], "valid", row["valid"],
              "M1-M0", row["delta_M1_minus_M0"], "MCSE", row["mcse"])
    print(alternative_blocks["cross_fold_covariance"])
'''

SENSITIVITY_CELL = '''for row in sensitivity.get("rows", []):
    print(row["scenario"], row["variant"], "passed", row["passed"],
          "divergences", row["divergences"], "residual RMS K", row["residual_rms_k"])
    print("  energy/tree diagnostics:", row.get("sampler", {}))
    if row.get("diagnostic_failures"):
        print("  failed:", row["diagnostic_failures"])
    if row["variant"] == "rsb" and row["passed"]:
        print("  A_RSB q05/q50/q95 K:", row["posterior"]["rsb_amplitude"]["q05_q50_q95"])
valid_sensitivity = [row for row in sensitivity.get("rows", [])
                     if row["passed"] and row["variant"] == "rsb"]
if valid_sensitivity:
    fig, axis = plt.subplots(figsize=(8, 4))
    for index, row in enumerate(valid_sensitivity):
        low, median, high = row["posterior"]["rsb_amplitude"]["q05_q50_q95"]
        axis.errorbar(median, index, xerr=[[median-low], [high-median]], fmt="o")
    axis.set_yticks(range(len(valid_sensitivity)),
                    [row["scenario"] for row in valid_sensitivity])
    axis.set_xlabel("A_RSB at 1 GHz [K]; conditional median and 5--95 percent interval")
    axis.set_xscale("log")
    fig.tight_layout()
    save(fig, "joint_prior_covariance_sensitivity.svg")
print("These scenarios are stated analysis assumptions, not new measured instrument errors.")
print("All external covariance scenarios preserve the quoted marginal variances.")
print("The conditional RSB intervals do not imply physical exclusion under a failing mean model.")

for name, record in independent.get("injections", {}).items():
    print("Independent direct-beam truth recovery:", name, "passed", record.get("passed"),
          "independent", record.get("independent_truth"), "divergences", record.get("divergences"))
    if record.get("passed"):
        for parameter, recovery in record.get("recovery", {}).items():
            print("  ", parameter, "max absolute pull", recovery.get("max_abs_pull"))
'''

SAMPLER_CELL = '''from scipy.stats import rankdata
for variant in ("no_rsb", "rsb"):
    row = next((row for row in sensitivity.get("rows", [])
                if row["scenario"] == "baseline" and row["variant"] == variant), None)
    if row is None or not row["passed"]:
        continue
    with np.load(RUNS["sensitivity"] / f"baseline_{variant}.npz", allow_pickle=False) as archive:
        samples = dict(archive)
    parameters = ["monopole_k", "log_width_e", "log_width_h"]
    if variant == "rsb":
        parameters += ["rsb_amplitude", "rsb_beta"]
    fig, axes = plt.subplots(len(parameters), 2, figsize=(10, 2.0*len(parameters)))
    for index, parameter in enumerate(parameters):
        values = np.asarray(samples[parameter])
        ranks = rankdata(values).reshape(values.shape)
        edges = np.linspace(0, values.size, 21)
        for chain in range(len(values)):
            axes[index, 0].plot(values[chain], linewidth=.4, alpha=.65, label=str(chain+1))
            axes[index, 1].hist(ranks[chain], bins=edges, histtype="step")
        axes[index, 0].set_ylabel(parameter)
        axes[index, 1].axhline(values.shape[1]/20, color="#888888", linestyle=":")
    axes[0, 0].legend(title="chain", ncol=4, fontsize=7)
    axes[-1, 0].set_xlabel("retained iteration")
    axes[-1, 1].set_xlabel("pooled rank")
    fig.suptitle(f"Full-data baseline {variant}: chain traces and ranks")
    fig.tight_layout()
    save(fig, f"chain_traces_ranks_{variant}.svg")
    background = (samples["rsb_amplitude"] * 0.408**samples["rsb_beta"]
                  if variant == "rsb" else 0.)
    with np.load(RUNS["b"] / "audited-input" / "maps.npz", allow_pickle=False) as archive:
        minimum = float(np.min(archive["template_k"]))
    gap = minimum + samples["monopole_k"] - background
    print(variant, "G408 minimum-pixel gap K, 0/5/50 percent:", np.percentile(gap, [0, 5, 50]))
    print(variant, "energy diagnostics:", row.get("sampler", {}))
'''

def posterior_sky_layers(draws, template, region, cmb408):
    """Physical 408 MHz layers, keeping RSB subtraction and addition explicit."""
    amplitude = np.asarray(draws["amplitude"]).reshape(-1, 3)[:, region]
    beta = np.asarray(draws["beta"]).reshape(-1, 3)[:, region]
    monopole = np.asarray(draws.get("monopole_k", draws.get("haslam_monopole_K"))).reshape(-1, 1)
    background = (np.asarray(draws["rsb_amplitude"]).ravel()
                  * 0.408**np.asarray(draws["rsb_beta"]).ravel())[:, None] if "rsb_amplitude" in draws else 0.
    corrected = template[None, :] + monopole
    galactic = corrected - background
    return {"amplitude": amplitude, "beta": beta, "H0_plus_zH": corrected,
            "G408": galactic, "T408": amplitude * galactic + cmb408 + background}


LAYERS_CELL = '''from examples.inference.tris_notebook import posterior_sky_layers
from examples.inference.tris_forward_baseline import cmb_rj_temperature
region = np.asarray(maps["region"], dtype=int)
template = np.asarray(maps["template_k"], dtype=float)
cmb408 = float(cmb_rj_temperature(408.0))
print("Template reference CMB:", float(maps["reference_cmb_k"]), "joint model CMB408:", cmb408)
ra_pixel = np.asarray(maps["ra_pixel_deg"])
dec_pixel = np.asarray(maps["dec_pixel_deg"])
for label in ("joint_m0", "joint_rsb"):
    path = RUNS["joint"] / (label + "_posterior.npz")
    entry = joint.get("injections", {}).get(label, {})
    if not path.is_file() or not entry.get("passed", False):
        print(label, "no diagnostically valid posterior; sky maps withheld")
        continue
    with np.load(path, allow_pickle=False) as archive:
        draws = dict(archive)
    if "rsb_amplitude" in draws and "rsb_beta" not in draws:
        draws["rsb_beta"] = np.full_like(draws["rsb_amplitude"], joint.get("rsb_beta_value", -2.66))
    layers = posterior_sky_layers(draws, template, region, cmb408)
    figure, axes = plt.subplots(len(layers), 2, figsize=(11, 13))
    saved = {}
    for row, (name, values) in enumerate(layers.items()):
        mean, sd = values.mean(0), values.std(0, ddof=1)
        saved[name + "_mean"], saved[name + "_sd"] = mean, sd
        for column, (statistic, field) in enumerate((("mean", mean), ("SD", sd))):
            artist = axes[row, column].scatter(ra_pixel, dec_pixel, c=field, s=12)
            axes[row, column].set_title(label + ": " + name + " " + statistic)
            axes[row, column].set_xlabel("RA [deg]")
            axes[row, column].set_ylabel("Dec [deg]")
            figure.colorbar(artist, ax=axes[row, column])
    np.savez_compressed(OUTPUT / (label + "_sky_layers.npz"), **saved)
    figure.tight_layout()
    save(figure, label + "_sky_layers.svg")
prior_ratio = np.asarray(maps["posterior_sigma_k_0"]) / np.asarray(maps["prior_sigma_k_0"])
figure, axis = plt.subplots(figsize=(8, 3.5))
artist = axis.scatter(ra_pixel, dec_pixel, c=prior_ratio, s=15, vmin=0, vmax=1)
figure.colorbar(artist, ax=axis, label="map-making posterior/prior SD")
axis.set_title("Conditional reconstruction: near 1 marks prior-dominated pixels")
axis.set_xlabel("RA [deg]")
axis.set_ylabel("Dec [deg]")
save(figure, "prior_dominated_map.svg")
print("All maps extrapolate the conditional regional model; they are not independent all-sky measurements.")
'''

RSB_CELL = '''from examples.inference.tris_forward_baseline import cmb_rj_temperature
draws_path = RUNS["joint"] / "joint_rsb_posterior.npz"
if not draws_path.is_file() or not joint.get("injections", {}).get("joint_rsb", {}).get("passed", False):
    draws_path = None
external_path = Path("runs/tris-rsb-input") / "external.npz"
if draws_path is not None:
    with np.load(draws_path, allow_pickle=False) as archive:
        rsb = {name: archive[name] for name in archive.files}
    if "rsb_amplitude" in rsb:
        amplitude = np.asarray(rsb["rsb_amplitude"], dtype=float).ravel()
        beta = np.asarray(rsb["rsb_beta"], dtype=float).ravel()
        frequency = np.geomspace(50.0, 3000.0, 60)
        band = amplitude[:, None] * (frequency[None, :] / 1000.0) ** beta[:, None]
        fig, axis = plt.subplots(figsize=(7, 3.6))
        axis.fill_between(frequency, np.percentile(band, 16, axis=0),
                          np.percentile(band, 84, axis=0), alpha=0.35, label="68 percent band")
        axis.plot(frequency, np.median(band, axis=0), color="#332288", label="median")
        if external_path.is_file():
            with np.load(external_path, allow_pickle=False) as data:
                external = {name: data[name] for name in data.files}
            axis.errorbar(np.asarray(external["frequency_mhz"], dtype=float),
                          np.asarray(external["temperature_rj_k"], dtype=float)
                          - cmb_rj_temperature(np.asarray(external["frequency_mhz"], dtype=float)),
                          yerr=np.asarray(external["sigma_rj_k"], dtype=float),
                          fmt="o", color="#cc6677", label="LWA / ARCADE excess above CMB; marginal errors")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("frequency [MHz]")
        axis.set_ylabel("RSB brightness temperature [K, RJ]")
        axis.legend()
        save(fig, "rsb_spectrum.svg")
        print("RSB amplitude draws: median", float(np.median(amplitude)),
              "68 percent", np.percentile(amplitude, [16, 84]).tolist())
    else:
        print("This model has no sampled RSB parameters.")
'''

CONCLUSIONS_CELL = '''print("The four judgements the plan asks to keep separate:")
print()
for label, entry in sorted(joint.get("injections", {}).items()):
    diagnostics = entry.get("diagnostics", {})
    r_hat = max((value["r_hat_max"] for value in diagnostics.values()), default=float("nan"))
    bulk = min((value["bulk_ess_min"] for value in diagnostics.values()), default=float("nan"))
    tail = min((value["tail_ess_min"] for value in diagnostics.values()), default=float("nan"))
    print(f"  joint {label}: divergences {entry.get('divergences')} "
          f"max rank split R-hat {r_hat:.4f} min bulk ESS {bulk:.0f} "
          f"min tail ESS {tail:.0f} independent-truth {entry.get('independent_truth')} "
          f"max tree depth {entry.get('max_tree_depth')}")
print()
print("1. Convergence is the diagnostics above; a refit that fails them is gated")
print("   and contributes no predictive delta (see the smooth/RA rows).")
print("2. Predictive adequacy is the audited-vs-fixed-A refit in section 3.")
print("3. Identifiability is the whitened, prior-scaled ratios in section 4; a")
print("   ratio near one means the prior, not the data, supplies that direction.")
print("4. RSB summaries are conditional on the chosen foreground, beam, covariance and priors.")
print("   The profile is uncalibrated; external-only MAP/Laplace is a sensitivity calculation.")
print("   The full-data mean model remains severely inconsistent with the measured profiles.")
print("   A near-zero conditional RSB posterior is not a physical exclusion or detection.")
print()
print("Limits: the sample covariance, the coordinate epoch, the beam-cut")
print("measurement errors and the measured 2D beam pattern are unpublished; the")
print("nside-8 sky quadrature and the finite beam ladder remain.")
'''

LIMITS_MARKDOWN = """## 10. What this document does not establish

- The ring sample covariance, the coordinate epoch and the beam-cut measurement
  errors are unpublished; nothing here invents them.
- The 0.004 K floor is a modelling convention with a measured chi-square effect.
- The nside-8 sky quadrature and the finite beam-resolution ladder are retained
  numerical uncertainty, not a physical attribution.
- The external LWA and ARCADE summaries are derived from foreground separation,
  not independent repeat measurements.
"""

EQUATION_CELL = r'''equation = (
    r"$C_{post} = (A_0^{T} N^{-1} A_0 + S^{-1})^{-1}$"
    + NL
    + r"$W_0 = C_{post} A_0^{T} N^{-1}, \quad C_m = W_0 N W_0^{T}$"
)
figure = plt.figure(figsize=(7.5, 1.5))
figure.text(0.5, 0.5, equation, ha="center", va="center", fontsize=13)
figure.savefig(ASSETS / "equation_mapmaking.svg", format="svg", bbox_inches="tight")
plt.close(figure)
print("rendered equation_mapmaking.svg (matplotlib mathtext, not LaTeX source text)")
'''

PROFILE_CELL = '''if profile:
    print("TRIS RSB profile, beta_rsb fixed at", profile.get("beta_rsb_fixed"))
    print("  Uncalibrated objective-drop crossing (not a statistical bound):",
          profile.get("objective_drop_crossing"),
          "(largest passing grid amplitude:", profile.get("objective_drop_grid"), ")")
    for row in profile.get("profile", []):
        print(f"  A {row['amplitude_rsb']:8.5f}  delta log p {row.get('delta_log_joint'):.3f}  "
              f"2 delta {row.get('two_delta_log_joint'):.1f}")
    for key, label in (("external_only", "free calibration"),
                       ("external_only_fixed_calibration",
                        "same fixed calibration as the profile")):
        external = profile.get(key, {})
        if external:
            print(f"external-only LWA/ARCADE MAP ({label}): A",
                  round(external["amplitude_rsb"], 4),
                  "+/-", round(external["amplitude_rsb_sd"], 4),
                  "beta", round(external["beta_rsb"], 4),
                  "+/-", round(external["beta_rsb_sd"], 4))
    if profile.get("external_only") or profile.get("external_only_fixed_calibration"):
        print("  External fits are conditional MAP/Laplace summaries; the profile")
        print("  has no calibrated coverage and cannot establish a statistical bound.")
else:
    print("no joint_profile.json in the run configuration")

if joint:
    for label, entry in sorted(joint.get("injections", {}).items()):
        diag = entry.get("diagnostics", {})
        rhat = max((value["r_hat_max"] for value in diag.values()), default=float("nan"))
        bulk = min((value["bulk_ess_min"] for value in diag.values()), default=float("nan"))
        tail = min((value["tail_ess_min"] for value in diag.values()), default=float("nan"))
        print(f"joint {label} (reduced: zero={joint.get('zero_mode')}, width={joint.get('width_mode')}, "
              f"rsb={joint.get('rsb_mode')}): divergences {entry.get('divergences')} "
              f"rhat_max {rhat:.4f} bulk_ess_min {bulk:.1f} tail_ess_min {tail}")
    print("external data loaded:", joint.get("external_loaded"), "source:", joint.get("external_input"))
'''

CELLS = [
    ("markdown", "# TRIS forward-model audit and sky-beam inference" + NL + NL
     + "This notebook runs against the recorded batch artifacts and reproduces "
       "the audit's figures and tables. It is executed by "
       "examples/inference/tris_notebook.py; every figure is written to the "
       "assets directory and embedded in the executed HTML." + NL + NL
     + "The page renders the markdown prose and equations as well as the code; "
       "the section numbers below match the headings." + NL + NL
     + "**Data provenance.** Sections 3, 4, 4b, 7 and 8 use the real audited "
       "TRIS maps and the separate LWA/ARCADE summary. The injection-recovery "
       "table in section 5 and the SBC run are synthetic. The P0 budget "
       "posterior SD quoted in section 2 is a local linearized estimate, not a "
       "joint-posterior marginal SD."),
    ("code", SETUP),
    ("markdown", "## 1. Data contract and operator change" + NL + NL
     + "The P0a ledger classifies every correction; nothing unpublished is "
       "invented. The audited operator rebuilds the beam directly from the "
       "archive cuts instead of upsampling an nside-8 map. The map-making "
       "identity is" + NL + NL
     + r"$$C_{post} = (A_0^T N^{-1} A_0 + S^{-1})^{-1}, \quad"
       " W_0 = C_{post} A_0^T N^{-1}$$" + NL + NL
     + "and the sampling covariance of the compressed map is "
       "$W_0 N W_0^T$, not the posterior covariance."),
    ("code", DATA_CELL),
    ("markdown", "## 2. Beam construction and numerical budget" + NL + NL
     + "The independent pixel-space oracle shares no code with limTOD's "
       "rotation helper. The budget is adjacent: rung h is compared with rung "
       "h-1, in per-point sigma, and the parameter shift is tested against the "
       "posterior SD." + NL + NL
     + "The quoted posterior SD is the local linearized inverse-Fisher estimate "
       "from the single 600.5 MHz block (tris_audit_report.linearized_region_"
       "shift), so it bounds a local sensitivity budget and is not the joint "
       "model's marginal posterior SD."),
    ("code", OPERATOR_CELL),
    ("markdown", "## 3. Forward baseline and residual adequacy"),
    ("code", BASELINE_CELL),
    ("markdown", "## 4. Beam cuts and identifiability" + NL + NL
     + "The width posterior/prior SD is computed from the noise-whitened "
       "Jacobian; the unwhitened numbers are recorded only as withdrawn. The "
       "deformation rescales each cut along its angle axis, so the power beam "
       "stays non-negative; the operator's small negative entries are harmonic "
       "ringing, measured at zero deformation as well."),
    ("code", BEAM_CELL),
    ("code", EQUATION_CELL),
    ("markdown", "## 4b. Same-layer real joint M0/M1 and conditional RSB posterior" + NL + NL
     + "Each posterior is published only after complete multichain diagnostics. "
       "The deterministic joint-density profile is only "
       "an objective diagnostic; no confidence level is assigned to its crossing. "
       "The external-only calculation is a conditional MAP/Laplace approximation."),
    ("code", PROFILE_CELL),
    ("code", PREDICTIVE_CELL),
    ("markdown", "## 5. Foreground identifiability and injection recovery" + NL + NL
     + "The whitened identifiability ratios are computed on the real audited "
       "data. The recovery table below is a synthetic injection: truth "
       "amplitude [1.2, 1.15, 1.25], beta [-2.6, -2.5, -2.4], no RSB in M0 and "
       "A_RSB(1 GHz) = 0.5 in M0 + RSB, on the same operator the fit uses."),
    ("code", IDENT_CELL),
    ("markdown", "## 6. Foreground basis flexibility and prior sensitivity"),
    ("code", EXTENSIONS_CELL),
    ("markdown", "## 6a. Full joint refits of prior and covariance scenarios"),
    ("code", SENSITIVITY_CELL),
    ("markdown", "## 6b. Full-data sampler energy, traces, ranks and boundaries"),
    ("code", SAMPLER_CELL),
    ("markdown", "## 6c. Finite SBC calibration (synthetic, gated)" + NL + NL
     + "Synthetic truths are drawn from the fitting prior by rejection sampling "
       "(the normalizer that Predictive ignores is enforced); each fit is gated "
       "on zero divergences, and the mean-rank and 95 percent coverage carry "
       "their uniform and binomial uncertainties."),
    ("code", SBC_CELL),
    ("markdown", "## 7. Same-layer M0 / M1 comparison, with the gate" + NL + NL
     + "A refit that fails its chain diagnostics contributes no delta."),
    ("code", COMPARISON_CELL),
    ("markdown", "## 8. Foreground layers and the RSB spectrum"),
    ("code", LAYERS_CELL),
    ("code", RSB_CELL),
    ("markdown", "## 9. Conclusions and limits"),
    ("code", CONCLUSIONS_CELL),
    ("markdown", LIMITS_MARKDOWN),
]


def build_notebook():
    cells = []
    for kind, source in CELLS:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "python3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def execute(notebook, *, environment):
    os.environ.update(environment)
    namespace = {"__name__": "__main__"}
    assets = Path(environment["TRIS_NOTEBOOK_OUTPUT"]) / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    sections = []
    code_index = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code":
            sections.append({"kind": "markdown", "source": source, "text": "", "images": []})
            continue
        # Track modification times, not just names: a second run into the same
        # output directory rewrites the figures, and a name-only set difference
        # would report no images and drop every one from the HTML.
        before = {path.name: path.stat().st_mtime_ns for path in assets.glob("*")}
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, "<notebook>", "exec"), namespace)  # noqa: S102
        text = buffer.getvalue()
        after = {path.name: path.stat().st_mtime_ns for path in assets.glob("*")}
        produced = sorted(
            name for name, mtime in after.items() if before.get(name) != mtime
        )
        code_index += 1
        # Capture any figure a cell left open instead of dropping it: the
        # executed notebook must not lose a display-only figure.  A cell that
        # already saved a file records its figure through that file.
        import matplotlib.pyplot as plt

        for number in (plt.get_fignums() if not produced else []):
            figure = plt.figure(number)
            name = "cell" + str(code_index) + "_figure" + str(number) + ".svg"
            figure.savefig(assets / name, format="svg")
            produced.append(name)
        plt.close("all")
        produced = sorted(set(produced))
        cell["outputs"] = [{"output_type": "stream", "name": "stdout",
                            "text": text.splitlines(keepends=True)}] if text else []
        cell["execution_count"] = code_index
        sections.append({"kind": "code", "source": source, "text": text, "images": produced})
    return sections


def _anchor(text):
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in text.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "section"


def _inline(text):
    escaped = html.escape(text)
    pieces = escaped.split("**")
    return "".join(
        piece if index % 2 == 0 else "<strong>" + piece + "</strong>"
        for index, piece in enumerate(pieces)
    )


def render_equation_svg(latex, target):
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(7.5, 0.9))
    figure.text(0.5, 0.5, "$" + latex + "$", ha="center", va="center", fontsize=12)
    figure.savefig(target, format="svg", bbox_inches="tight")
    plt.close(figure)


def markdown_to_html(source, equation_renderer=None):
    fence = chr(96) * 3
    parts = []
    paragraph = []
    code_lines = []
    in_code = False
    in_list = False

    def flush_paragraph():
        if paragraph:
            parts.append("<p>" + _inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            parts.append("</ul>")
            in_list = False

    for raw in source.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith(fence):
            flush_paragraph()
            close_list()
            if in_code:
                parts.append("<pre><code>" + html.escape(NL.join(code_lines)) + "</code></pre>")
                code_lines.clear()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            flush_paragraph()
            close_list()
            continue
        if stripped.startswith("$$") and stripped.endswith("$$") and len(stripped) > 4:
            flush_paragraph()
            close_list()
            latex = stripped[2:-2].strip()
            if equation_renderer is not None:
                parts.append("<div class='equation'>" + equation_renderer(latex) + "</div>")
            else:
                parts.append("<div class='equation'>" + html.escape(latex) + "</div>")
            continue
        if stripped[0] == "#":
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            flush_paragraph()
            close_list()
            tag = "h" + str(min(level + 1, 5))
            parts.append("<" + tag + " id='" + _anchor(title) + "'>" + _inline(title) + "</" + tag + ">")
            continue
        if stripped.startswith("- "):
            flush_paragraph()
            if not in_list:
                parts.append("<ul>")
                in_list = True
            parts.append("<li>" + _inline(stripped[2:]) + "</li>")
            continue
        paragraph.append(stripped)
    flush_paragraph()
    close_list()
    if in_code and code_lines:
        parts.append("<pre><code>" + html.escape(NL.join(code_lines)) + "</code></pre>")
    return NL.join(parts)


def render_html(sections, path):
    css = (
        "body{font-family:system-ui,-apple-system,'PingFang SC',sans-serif;margin:0;"
        "background:#fafafa;color:#222}header{padding:1.2rem 2rem;background:#22303f;"
        "color:#fff}main{padding:1rem 2rem 3rem;max-width:1000px}"
        "nav{background:#eef2f5;border-radius:8px;padding:.8rem 1rem;margin-bottom:1rem;"
        "display:flex;flex-wrap:wrap;gap:.6rem 1.2rem}nav a{color:#224466;"
        "text-decoration:none;font-size:.92rem}nav a:hover{text-decoration:underline}"
        "section{background:#fff;border:1px solid #e2e6ea;border-radius:10px;"
        "padding:1rem 1.4rem;margin-bottom:1rem}section.markdown h2{border-bottom:"
        "1px solid #e2e6ea;padding-bottom:.3rem}pre{background:#f1f3f5;padding:.6rem;"
        "overflow:auto;border-radius:6px}pre.stdout{background:#f7fafc;border-left:"
        "3px solid #4477aa}.equation{background:#f7f7ff;border-left:3px solid #332288;"
        "padding:.5rem .8rem;font-family:ui-monospace,Menlo,monospace;overflow:auto}"
        "img{max-width:100%}"
    )
    sections_html = []
    nav_items = []
    code_index = 0
    assets = path.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    equation_index = [0]

    def equation_renderer(latex):
        equation_index[0] += 1
        name = "equation_markdown_" + str(equation_index[0]) + ".svg"
        render_equation_svg(latex, assets / name)
        return "<img src='assets/" + name + "' alt='equation " + str(equation_index[0]) + "'>"

    for section in sections:
        if section.get("kind") == "markdown":
            rendered = markdown_to_html(section["source"], equation_renderer)
            sections_html.append("<section class='markdown'>" + rendered + "</section>")
            for match in re.finditer(r"<h[2-5] id='([^']+)'>(.*?)</h[2-5]>", rendered):
                label = re.sub("<[^>]+>", "", match.group(2))
                nav_items.append("<a href='#" + match.group(1) + "'>" + label + "</a>")
            continue
        code_index += 1
        pieces = ["<section class='code'><h4>code cell " + str(code_index) + "</h4>"]
        pieces.append("<pre><code>" + html.escape(section["source"]) + "</code></pre>")
        if section.get("text"):
            pieces.append("<pre class='stdout'>" + html.escape(section["text"]) + "</pre>")
        for image in section.get("images", []):
            pieces.append("<img src='assets/" + html.escape(image)
                          + "' alt='" + html.escape(image) + "'>")
        pieces.append("</section>")
        sections_html.append("".join(pieces))
    parts = [
        "<!doctype html><html lang='zh'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>TRIS forward-inference notebook</title>",
        "<style>" + css + "</style></head><body>",
        "<header><h1>TRIS forward-model audit / executed notebook</h1></header><main>",
    ]
    if nav_items:
        parts.append("<nav>" + "".join(nav_items) + "</nav>")
    parts.extend(sections_html)
    parts.append("</main></body></html>")
    path.write_text("".join(parts))


def validate(path, assets):
    problems = []
    text = path.read_text()
    if "http://" in text or "https://" in text:
        problems.append("page references an external URL")
    for chunk in text.split("src='assets/")[1:]:
        name = chunk.split("'")[0]
        if not (assets / name).is_file():
            problems.append("missing image: " + name)
    if "<section class='markdown'>" not in text:
        problems.append("no markdown section was rendered")
    if "## " in text:
        problems.append("markdown heading was not rendered")
    return problems


def default_configs(runs):
    """Recorded inference commands the rerun entry can execute."""
    python = sys.executable
    configs = {}
    if "budget" in runs:
        configs["p0_budget"] = [
            python, "-m", "examples.inference.tris_p0_budget",
            "--maps", "runs/tris-input-skyfields/maps.npz",
            "--archive", "runs/tris-input-skyfields/archive",
            "--output", str(runs["budget"]),
        ]
    if "b" in runs:
        configs["beam_surrogate"] = [
            python, "-m", "examples.inference.tris_beam_surrogate",
            "--batch-b", str(runs["b"]),
            "--output", str(runs["b"]).rstrip("/") + "-beam-surrogate",
        ]
        configs["beam_identifiability"] = [
            python, "-m", "examples.inference.tris_beam_identifiability",
            "--batch-b", str(runs["b"]),
            "--output", str(runs["beam"]),
            "--rebuild-operators",
        ]
    return configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", required=True, help="JSON mapping batch keys to run directories")
    parser.add_argument("--configs", type=Path, default=None,
                        help="JSON mapping config names to argv lists for --rerun")
    parser.add_argument("--rerun", default=None, help="execute a recorded config and exit")
    args = parser.parse_args(argv)
    runs = json.loads(Path(args.runs).read_text())
    configs = json.loads(args.configs.read_text()) if args.configs is not None else default_configs(runs)
    if args.rerun is not None:
        if args.rerun not in configs:
            raise SystemExit("unknown rerun config: " + args.rerun)
        return subprocess.run(configs[args.rerun], check=False).returncode
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_configs.json").write_text(json.dumps(configs, indent=2) + NL)
    notebook = build_notebook()
    (args.output / "tris_forward_inference.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + NL
    )
    environment = {
        "TRIS_NOTEBOOK_OUTPUT": str(args.output.resolve()),
        "TRIS_NOTEBOOK_RUNS": json.dumps(runs),
        "TRIS_NOTEBOOK_CONFIGS": json.dumps(configs),
    }
    sections = execute(notebook, environment=environment)
    (args.output / "tris_forward_inference.executed.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + NL
    )
    render_html(sections, args.output / "index.html")
    problems = validate(args.output / "index.html", args.output / "assets")
    print(json.dumps({"output": str(args.output), "cells": len(sections),
                      "problems": problems, "elapsed_s": round(time.time() - started, 2)}))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
