"""The four talk figures of Demo C, in the talk's palette (16:9 at 1280 px).

Runs in the twin's venv after ``recover.py`` and ``identify.py``:

* ``signal_recovery_moments.svg`` (main) and ``signal_recovery_moments_stress.svg``:
  columns oracle, N = 0, 1, 2 at the chosen cell count; rows: the curve with
  its 68 and 95 % bands and the truth, and curve minus truth;
* ``trough_bias_vs_order.svg``: trough-depth bias with its 16-84 % range
  against N for both scenarios, the 10 mK noise marked; the sampler-free
  amplitude bias in units of its sigma; SER relative to the prior with
  Demos A and B as references;
* ``identifiable_combinations.svg``: the noise standard deviation of every
  whitened coefficient combination with the retained counts, the composition
  of the retained combinations, and the two reductions' column counts;
* ``basis_columns.svg``: the physical basis columns and the cell pivots.

Every figure is drawn from the saved products only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from global21cm import posterior_io as io

P = common.PALETTE
FIG = (12.8, 7.2)
SUBSAMPLE = 20000
plt.rcParams.update({"font.family": "Arial", "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
                     "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.edgecolor": P["ink"], "axes.labelcolor": P["ink"], "xtick.color": P["ink"],
                     "ytick.color": P["ink"], "text.color": P["ink"], "svg.fonttype": "none"})  # fmt: skip


def save(fig, path: Path) -> None:
    """SVG for the deck and a 1280 px PNG beside it."""
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=100)
    plt.close(fig)


def load(name: str, results: Path, out: Path) -> dict:
    recovery = json.loads((results / f"recovery_{name}.json").read_text())
    ident = json.loads((results / f"identifiability_{name}.json").read_text())
    with np.load(results / f"posteriors_{name}.npz") as z:
        particles = {k: z[k] for k in z.files}
    with np.load(out / f"basis_{name}.npz") as z:
        basis = {k: z[k] for k in z.files}
    return {"recovery": recovery, "ident": ident, "particles": particles, "basis": basis,
            "truth": np.load(common.SCENARIOS[name].sim_dir / "t21_truth.npy")}  # fmt: skip


def curves_mk(u: np.ndarray, freqs, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(u), size=min(SUBSAMPLE, len(u)), replace=False)
    return 1e3 * io.curves(io.theta_from_unit(u[pick]), freqs)


def bands(curves: np.ndarray) -> dict:
    q = np.percentile(curves, [2.5, 16, 50, 84, 97.5], axis=0)
    return {"lo95": q[0], "lo68": q[1], "med": q[2], "hi68": q[3], "hi95": q[4]}


def sweep_rows(recovery: dict) -> list[dict]:
    return [r for r in recovery["sweeps"]["chosen"] if "skipped" not in r]


def signal_recovery(data: dict, name: str, path: Path) -> None:
    freqs = data["particles"]["freqs"]
    truth = 1e3 * data["truth"]
    chosen = data["recovery"]["grid"]["chosen"]
    columns = [("oracle", "oracle__u", P["muted"])]
    for order in (0, 1, 2):
        key = f"cells{chosen['cells']}_beam{chosen['beam_terms']}_N{order}__u"
        if key in data["particles"]:
            columns.append((f"N = {order}", key, common.ORDER_COLOURS[order]))
    fig, axes = plt.subplots(2, len(columns), figsize=FIG, sharex=True)
    rows = {r["N"]: r for r in sweep_rows(data["recovery"])}
    for j, (label, key, colour) in enumerate(columns):
        b = bands(curves_mk(data["particles"][key], freqs))
        top, bottom = axes[0, j], axes[1, j]
        top.fill_between(freqs, b["lo95"], b["hi95"], color=colour, alpha=0.22, lw=0)
        top.fill_between(freqs, b["lo68"], b["hi68"], color=colour, alpha=0.45, lw=0)
        top.plot(freqs, b["med"], color=colour, lw=2)
        top.plot(freqs, truth, color=P["ink"], lw=1.6, ls="--")
        bottom.fill_between(freqs, b["lo95"] - truth, b["hi95"] - truth, color=colour, alpha=0.22, lw=0)
        bottom.fill_between(freqs, b["lo68"] - truth, b["hi68"] - truth, color=colour, alpha=0.45, lw=0)
        bottom.plot(freqs, b["med"] - truth, color=colour, lw=2)
        bottom.axhline(0, color=P["ink"], lw=1, ls="--")
        title = label
        if label != "oracle":
            r = rows[int(label.split("= ")[1])]
            title += f"  ({r['n_basis']} columns, " + ("calibrated" if r["calibrated"] else "not calibrated") + ")"
        top.set_title(title, fontsize=13)
        limit = float(np.max(np.abs([b["lo95"] - truth, b["hi95"] - truth]))) * 1.1
        bottom.set_ylim(-limit, limit)
        top.set_ylim(min(-260, float(b["lo95"].min()) * 1.1), max(40, float(b["hi95"].max()) * 1.1))
        bottom.set_xlabel("frequency [MHz]")
    axes[0, 0].set_ylabel("T21 [mK]")
    axes[1, 0].set_ylabel("curve - truth [mK]")
    cells, beam = chosen["cells"], chosen["beam_terms"]
    fig.suptitle(f"{name} scenario: physical moment basis, {cells} energy cell(s), {beam} beam spectra; "
                 "bands 68 / 95 %, dashed = injected truth", fontsize=14, color=P["ink"])  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def trough_bias(datasets: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIG)
    colours = {"main": P["blue"], "stress": P["orange"]}
    for name, data in datasets.items():
        rows = sweep_rows(data["recovery"])
        n = np.array([r["N"] for r in rows])
        depth = np.array([r["depth_mk"] for r in rows])  # 16 / 50 / 84 %
        bias = depth[:, 1] - common.TROUGH_TRUE_MK
        err = np.vstack([depth[:, 1] - depth[:, 0], depth[:, 2] - depth[:, 1]])
        filled = [r["calibrated"] for r in rows]
        for k in range(len(rows)):
            axes[0].errorbar(n[k], bias[k], yerr=err[:, k : k + 1], fmt="o", ms=9, color=colours[name],
                             mfc=colours[name] if filled[k] else "white", capsize=4, lw=2)  # fmt: skip
        axes[0].plot(n, bias, color=colours[name], lw=1.2, alpha=0.6, label=name)
        amp = np.array([abs(r["forecast"]["bias_sigma"]) for r in rows])
        axes[1].plot(n, amp, "o-", ms=9, color=colours[name], lw=2, label=name)
        ser = np.array([r["ser_over_prior"] for r in rows])
        axes[2].plot(n, ser, "o-", ms=9, color=colours[name], lw=2, label=f"Demo C, {name}")
        twin = data["recovery"]["twin"]
        for model, style, text in (("beamconv", ":", "A"), ("physical", "--", "B")):
            if model in twin:
                value = twin[model]["ser_over_prior"]
                axes[2].axhline(value, color=colours[name], ls=style, lw=1.5)
                axes[2].annotate(f"Demo {text}, {name}", (common.N_MAX + 0.08, value), fontsize=10, color=colours[name],
                                 va="center", ha="left", annotation_clip=False)  # fmt: skip
    axes[0].axhspan(-10, 10, color=P["muted"], alpha=0.2, lw=0, label="+/- 10 mK, the noise per sample")
    axes[0].text(0.03, 0.97, "filled marker: calibrated posterior", transform=axes[0].transAxes, va="top", fontsize=11)
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].set_ylabel("trough-depth bias [mK]  (median - truth, 16-84 %)")
    axes[0].set_title("21-cm bias against the truncation order")
    axes[0].axhline(0, color=P["ink"], lw=1, ls="--")
    axes[1].set_yscale("log")
    axes[1].axhline(1, color=P["ink"], lw=1, ls="--")
    axes[1].set_ylabel("|amplitude bias| / sigma(amp), sampler-free")
    axes[1].set_title("foreground misfit along the signal")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("SER / SER(prior)")
    axes[2].set_title("recovery against the prior")
    for ax in axes:
        ax.set_xlabel("Taylor order N")
        ax.set_xticks(range(common.N_MAX + 1))
    axes[0].legend(loc="center right", fontsize=11)
    axes[1].legend(loc="upper right", fontsize=11)
    axes[2].legend(loc="center right", fontsize=11)
    axes[2].set_xlim(-0.3, common.N_MAX + 1.1)
    fig.tight_layout()
    save(fig, path)


def identifiable(datasets: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=FIG)
    colours = {"main": P["blue"], "stress": P["orange"]}
    width, offsets = 0.36, {"main": -0.2, "stress": 0.2}
    for name, data in datasets.items():
        case = data["ident"]["cases"]["chosen"]
        sd = np.array(case["combination_sd"])
        axes[0].plot(np.arange(1, sd.size + 1), sd, "o-", ms=8, color=colours[name], lw=2,
                     label=f"{name}: {case['cells']} cells, N = {case['N']}, {case['beam_terms']} beam")  # fmt: skip
        for t, ls in ((0.1, "--"), (0.01, ":")):
            kept = case["retained"][str(t)]
            axes[0].axhline(t, color=P["muted"], lw=1, ls=ls)
            axes[0].annotate(f"{name}: {kept} at sd {t}", (kept, t), fontsize=10, color=colours[name], ha="left",
                             xytext=(8, 10 if name == "main" else -16), textcoords="offset points")  # fmt: skip
        counts = [case["columns_full_rs"], case["columns_after_identities"], case["retained"]["0.1"]]
        axes[2].bar(np.arange(3) + offsets[name], counts, width, color=colours[name], label=name)
        for i, c in enumerate(counts):
            axes[2].text(i + offsets[name], c + 0.4, str(c), ha="center", fontsize=12, color=colours[name])
    axes[0].set_yscale("log")
    axes[0].set_xlabel("combination i (whitened SVD order)")
    axes[0].set_ylabel("noise sd of combination i (zeroth moment = 1)")
    axes[0].set_title("what one LST's channels constrain")
    axes[0].legend(loc="upper left", fontsize=10)
    main = datasets["main"]["ident"]["cases"]["chosen"]
    kept = main["retained"]["0.1"]
    vectors = np.abs(np.array(main["vectors"])[:kept])
    im = axes[1].imshow(vectors, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(main["labels"])))
    axes[1].set_xticklabels(main["labels"], rotation=60, ha="right", fontsize=10)
    axes[1].set_yticks(range(kept))
    axes[1].set_yticklabels([f"{i + 1}: sd {s:.1e}" for i, s in enumerate(main["combination_sd"][:kept])], fontsize=10)
    axes[1].set_title("main: |V| of the retained combinations")
    fig.colorbar(im, ax=axes[1], fraction=0.05, pad=0.02)
    axes[2].set_xticks(range(3))
    axes[2].set_xticklabels(["all (r, s)\ncolumns", "after the\nscaling identity", "constrained\n(sd <= 0.1)"])
    axes[2].set_ylabel("coefficients per LST")
    axes[2].set_title("the two reductions")
    axes[2].legend()
    fig.tight_layout()
    save(fig, path)


def basis_columns(data: dict, name: str, path: Path) -> None:
    basis, recovery = data["basis"], data["recovery"]
    freqs = basis["freqs_mhz"]
    chosen = recovery["grid"]["chosen"]
    kind = recovery["channel_kind"]
    fig, axes = plt.subplots(1, 3, figsize=FIG)
    single = basis[f"cells1_{kind}_grouped"][:, 0, :]
    for r in range(common.N_MAX + 1):
        col = single[:, r] / np.linalg.norm(single[:, r])
        axes[0].plot(freqs, col, lw=2, color=common.ORDER_COLOURS[r], label=f"r = {r}")
    gamma1 = basis["cells1_gamma"][0]
    axes[0].set_title(f"one reference, gamma = {gamma1:.0f}: orders r = 0..{common.N_MAX}", fontsize=13)
    axes[0].set_ylabel("unit-norm column, brightness temperature")
    k = chosen["cells"]
    grouped = basis[f"cells{k}_{kind}_grouped"]
    cell_colours = [P["blue"], P["orange"], P["green"], P["purple"], P["muted"], P["ink"]]
    for c in range(k):
        for r in range(chosen["N"] + 1):
            col = grouped[:, c, r] / np.linalg.norm(grouped[:, c, r])
            axes[1].plot(freqs, col, lw=2, color=cell_colours[c % len(cell_colours)], alpha=1 - 0.25 * r,
                         label=f"cell {c}, r = {r}" if r == 0 or c == 0 else None)  # fmt: skip
    axes[1].set_title(f"chosen: {k} cells, N = {chosen['N']}: {chosen['n_basis']} columns", fontsize=13)
    for c in range(k):
        col = grouped[:, c, 0]
        axes[2].semilogy(freqs, col / col.max(), lw=2, color=cell_colours[c % len(cell_colours)],
                         label=f"gamma = {basis[f'cells{k}_gamma'][c]:.0f}, nu_c = {basis[f'cells{k}_nu_c_hz'][c] / 1e6:.0f} MHz")  # fmt: skip
    axes[2].set_title("cell kernels at order 0, peak-normalised", fontsize=13)
    axes[2].set_ylim(1e-2, 2)
    for ax in axes:
        ax.set_xlabel("frequency [MHz]")
        ax.legend(loc="best", fontsize=10)
    fig.suptitle(f"{name}: channel-integrated SyncMoments columns, isotropic pitch and field directions, Stokes I",
                 fontsize=14)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=common.RUNS)
    args = parser.parse_args()
    results = common.results_dir(args.out, args.quick)
    datasets = {name: load(name, results, args.out) for name in common.SCENARIOS}
    signal_recovery(datasets["main"], "main", results / "signal_recovery_moments.svg")
    signal_recovery(datasets["stress"], "stress", results / "signal_recovery_moments_stress.svg")
    trough_bias(datasets, results / "trough_bias_vs_order.svg")
    identifiable(datasets, results / "identifiable_combinations.svg")
    basis_columns(datasets["main"], "main", results / "basis_columns.svg")
    for name in ("signal_recovery_moments", "signal_recovery_moments_stress", "trough_bias_vs_order",
                 "identifiable_combinations", "basis_columns"):  # fmt: skip
        print("wrote", results / f"{name}.svg")


if __name__ == "__main__":
    main()
