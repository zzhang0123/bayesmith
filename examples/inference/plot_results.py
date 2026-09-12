"""Render saved inference runs as an offline, five-step walkthrough.

    python examples/inference/plot_results.py --input runs/inference-demo

Requires optional matplotlib. Reads result.json and posterior.npz, without
importing bayesmith or running inference. Signal bands exclude observation noise.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

if __package__:
    from .comparison_policy import comparison_visible
    from .presentation import CASES, LEGACY_CASES, parameter_value, write_gallery
else:
    from comparison_policy import comparison_visible
    from presentation import CASES, LEGACY_CASES, parameter_value, write_gallery

POSTERIOR = "#275cad"
TRUTH = "#b5541c"
OBSERVED = "#64717b"


def save_figure(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def draw_signal(ax, report, *, inferred, mask=None):
    signal = report["signal"]
    x = np.asarray(signal["x"])
    data = np.asarray(signal["data"])
    truth = np.asarray(signal["truth"])
    median = np.asarray(signal["median"])
    lower = np.asarray(signal["lower"])
    upper = np.asarray(signal["upper"])
    if mask is not None:
        x, data, truth, median, lower, upper = [
            a[mask] for a in (x, data, truth, median, lower, upper)
        ]
    if signal["view"].get("kind") == "power_law":
        ordered = np.sort(data)
        empirical = np.arange(data.size, 0, -1) / data.size
        # P(X >= x) drops just after each observed value, not at the next one.
        ax.step(ordered, empirical, where="pre", color=OBSERVED, alpha=0.7)
        if inferred:
            ax.fill_between(x, lower, upper, color=POSTERIOR, alpha=0.17)
            ax.plot(x, median, color=POSTERIOR, lw=2.3)
        else:
            ax.plot(x, truth, color=TRUTH, lw=2.3, ls="--")
        ax.set(xscale="log", yscale="log", xlabel="x / x_min",
               ylabel="Survival probability P(X ≥ x)", xlim=(1, max(16, data.max())),
               ylim=(0.001, 1.1))
        ax.grid(alpha=0.2)
        return
    grouped = report["case"] == "hierarchical"
    scatter_x = x.astype(float).copy()
    if grouped:
        # Jitter within each group only; no interpolated line between categories.
        for group in np.unique(x):
            indices = np.flatnonzero(x == group)
            scatter_x[indices] += np.linspace(-0.16, 0.16, len(indices))
    ax.scatter(scatter_x, data, s=16, alpha=0.42, color=OBSERVED, linewidths=0)
    if grouped:
        groups, indices = np.unique(x, return_index=True)
        if inferred:
            ax.vlines(groups, lower[indices], upper[indices], color=POSTERIOR, lw=5)
            ax.scatter(groups, median[indices], marker="o", s=42, color=POSTERIOR)
        else:
            ax.scatter(groups, truth[indices], marker="D", s=48, color=TRUTH)
        ax.set_xticks(groups, [f"Group {int(g)}" for g in groups])
        ax.set_xlabel("Independent groups (horizontal jitter separates observations)")
    else:
        order = np.argsort(x)
        if inferred:
            ax.fill_between(
                x[order], lower[order], upper[order], color=POSTERIOR, alpha=0.17
            )
            ax.plot(x[order], median[order], color=POSTERIOR, lw=2.3)
        else:
            ax.plot(x[order], truth[order], color=TRUTH, lw=2.3, ls="--")
        ax.set_xlabel("Time" if report["case"] == "exponential_decay" else "Input x")
    if report["case"] == "bernoulli":
        ax.set(ylabel="Probability / binary outcome", ylim=(-0.08, 1.08))
    else:
        ax.set_ylabel("Observation / mean signal")
    ax.grid(axis="y", color="#e7ebef", lw=0.8)
    ax.set_axisbelow(True)
    # Keep both stages on the same axes to make comparing them meaningful.
    low = min(data.min(), truth.min(), lower.min())
    high = max(data.max(), truth.max(), upper.max())
    if report["case"] != "bernoulli":
        padding = max((high - low) * 0.07, 0.01)
        ax.set_ylim(low - padding, high + padding)
    handles = [
        Line2D(
            [], [], ls="", marker="o", color=OBSERVED, label="Simulated observations"
        )
    ]
    if inferred:
        handles.append(
            Line2D(
                [],
                [],
                color=POSTERIOR,
                lw=2.3,
                label="Posterior median + 99% mean interval",
            )
        )
    else:
        handles.append(
            Line2D(
                [],
                [],
                color=TRUTH,
                ls="--",
                marker="D" if grouped else None,
                label="Generating mean / probability (known truth)",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=2,
        frameon=False,
        fontsize=10,
    )


def parameter_samples(archive):
    """Match the coordinate spelling used by common.recovery_checks."""
    values = {}
    for name in archive.files:
        samples = archive[name]
        for index in np.ndindex(samples.shape[1:]):
            suffix = str(list(index)) if index else ""
            values[name + suffix] = samples[(slice(None),) + index]
    return values


def draw_parameter(ax, row, values):
    ax.hist(
        values,
        bins=35,
        density=True,
        color=POSTERIOR,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.axvspan(row["lower"], row["upper"], color=POSTERIOR, alpha=0.08)
    ax.axvline(row["mean"], color=POSTERIOR, lw=1.8)
    ax.axvline(row["truth"], color=TRUTH, ls="--", lw=2.0)
    ax.set_title(row["name"], loc="left", fontsize=13, pad=10)
    ax.set_xlabel("Parameter value", fontsize=10)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)
    ax.text(
        0.98,
        0.96,
        f"Truth {parameter_value(row['truth'], row)}\nMean {parameter_value(row['mean'], row)}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86},
    )
    # Include a missed truth too. A failed run must not hide it off the axes.
    minimum = min(float(np.min(values)), row["truth"])
    maximum = max(float(np.max(values)), row["truth"])
    padding = max((maximum - minimum) * 0.08, 1e-8)
    ax.set_xlim(minimum - padding, maximum + padding)


def draw_surface(report, *, inferred):
    signal, view = report["signal"], report["signal"]["view"]
    rows, columns = view["shape"]
    x_min, x_max = min(signal["x"]), max(signal["x"])
    y_min, y_max = min(view["x2"]), max(view["x2"])
    # imshow takes pixel edges; the simulated regular-grid inputs are centers.
    dx = (x_max - x_min) / (columns - 1)
    dy = (y_max - y_min) / (rows - 1)
    extent = [x_min - dx / 2, x_max + dx / 2, y_min - dy / 2, y_max + dy / 2]
    if inferred:
        fields = [
            ("Posterior median probability", signal["median"], "Blues", 1),
            (
                "99% probability interval width",
                np.asarray(signal["upper"]) - signal["lower"],
                "YlGnBu",
                None,
            ),
        ]
    else:
        fields = [
            ("Binary observations", signal["data"], "Greys", 1),
            ("Generating probability", signal["truth"], "Blues", 1),
        ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), layout="constrained")
    for ax, (title, values, cmap, upper) in zip(axes, fields, strict=True):
        mesh = ax.imshow(
            np.asarray(values).reshape(view["shape"]),
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=0,
            vmax=upper,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set(title=title, xlabel=view["x_label"], ylabel=view["y_label"])
        fig.colorbar(mesh, ax=ax, shrink=0.82)
    return fig


def draw_recovery(report, rows, samples, directory):
    """Compare unlike coordinates using their own SD; preserve failed truths."""
    fig, (forest, correlation) = plt.subplots(
        1,
        2,
        figsize=(11, max(4.1, 0.4 * len(rows) + 1)),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.15, 1]},
    )
    for i, row in enumerate(rows):
        sd = row["posterior_sd"]
        lo, mean, hi = [
            (row[k] - row["truth"]) / sd for k in ("lower", "mean", "upper")
        ]
        color = POSTERIOR if row["covered"] else TRUTH
        forest.plot([lo, hi], [i, i], color=color, lw=2.5)
        forest.scatter([mean], [i], color=color, s=28, zorder=3)
    forest.axvline(0, color=TRUTH, ls="--", lw=1.5)
    forest.set(
        yticks=np.arange(len(rows)),
        yticklabels=[r["name"] for r in rows],
        xlabel="(Parameter − truth) / posterior SD",
        title="Recovery of every coordinate",
    )
    forest.invert_yaxis()
    forest.grid(axis="x", color="#e7ebef")
    values = np.column_stack([samples[r["name"]] for r in rows])
    matrix = np.corrcoef(values, rowvar=False)
    heat = correlation.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    names = [r["name"] for r in rows]
    correlation.set(
        xticks=np.arange(len(rows)),
        yticks=np.arange(len(rows)),
        xticklabels=names,
        yticklabels=names,
        title="Posterior correlation",
    )
    correlation.tick_params(axis="both", labelsize=8)
    plt.setp(
        correlation.get_xticklabels(), rotation=55, ha="right", rotation_mode="anchor"
    )
    fig.colorbar(heat, ax=correlation, shrink=0.72, ticks=[-1, 0, 1])
    save_figure(fig, directory, "recovery")


def draw_process_stages(report, directory):
    signal = report["signal"]
    view = signal["view"]
    x = np.asarray(signal["x"])
    components = view["components"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), layout="constrained")
    for ax, key, title in zip(
        axes.flat,
        ["instance", "response", "nonlinear", "linear", "gain"],
        [
            "1 · Gaussian process instance",
            "2 · Known linear response",
            "3 · Add nonlinear shape",
            "4 · Add linear background",
            "5 · Multiply log-linear gain",
        ],
        strict=False,
    ):
        ax.plot(x, components[key], color=POSTERIOR, lw=1.5)
        ax.set(
            title=title,
            xlabel="Input x",
            ylabel="Gain" if key == "gain" else "Component",
        )
        ax.grid(alpha=0.2)
    ax = axes.flat[-1]
    ax.plot(x, signal["truth"], color=TRUTH, lw=1.5, label="Mean before white noise")
    ax.scatter(
        x, signal["data"], color=OBSERVED, s=6, alpha=0.45, label="Noisy observation"
    )
    ax.set(
        title="6 · Multiply (1 + sigma_w × white noise)",
        xlabel="Input x",
        ylabel="Final observation",
    )
    ax.legend(fontsize=8, frameon=False)
    save_figure(fig, directory, "components")
    spectrum = view["spectrum"]
    k, indices = np.unique(spectrum["frequency"], return_index=True)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.fill_between(
        k,
        np.asarray(spectrum["lower"])[indices],
        np.asarray(spectrum["upper"])[indices],
        alpha=0.18,
        color=POSTERIOR,
        label="99% posterior spectrum interval",
    )
    ax.plot(
        k,
        np.asarray(spectrum["median"])[indices],
        color=POSTERIOR,
        label="Posterior median spectrum",
    )
    ax.plot(
        k,
        np.asarray(spectrum["truth"])[indices],
        color=TRUTH,
        ls="--",
        label="Generating ensemble spectrum",
    )
    ax.scatter(
        k,
        np.asarray(spectrum["realized_pair_power"])[indices],
        color=OBSERVED,
        marker="x",
        s=35,
        label="One instance: sine/cosine pair power",
    )
    ax.set(
        xlabel="Fourier frequency k",
        ylabel="Coefficient variance / pair power",
        yscale="log",
        title="One instance informs the power amplitude; its power fluctuates",
    )
    ax.legend(fontsize=8, frameon=False)
    save_figure(fig, directory, "spectrum")


def render(report, directory):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b7c0c8",
            "axes.labelcolor": "#46535f",
            "text.color": "#23313e",
            "xtick.color": "#46535f",
            "ytick.color": "#46535f",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
        }
    )
    view = report["signal"].get("view", {})
    if view.get("kind") == "process":
        draw_process_stages(report, directory)
    for stage, inferred in (("simulation", False), ("inference", True)):
        if view.get("kind") == "surface":
            fig = draw_surface(report, inferred=inferred)
        elif view.get("kind") in ("channels", "power_law"):
            channels = np.asarray(view["channel"])
            fig, axes = plt.subplots(
                1, len(view["labels"]), figsize=(11, 4.3), layout="constrained"
            )
            for i, ax in enumerate(np.atleast_1d(axes)):
                draw_signal(ax, report, inferred=inferred, mask=channels == i)
                ax.set_title(view["labels"][i], loc="left", pad=44)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
            fig.supxlabel(
                ("Blue: posterior median + 99% survival-probability interval"
                 if view.get("kind") == "power_law" else "Blue: posterior median + 99% mean interval")
                if inferred
                else ("Orange: generating survival probability; grey: empirical survival"
                      if view.get("kind") == "power_law" else "Orange: generating mean; grey: observations"),
                fontsize=10,
            )
        else:
            fig, ax = plt.subplots(figsize=(10, 4.1), layout="constrained")
            draw_signal(ax, report, inferred=inferred)
        save_figure(fig, directory, stage)
    with np.load(directory / "posterior.npz", allow_pickle=False) as archive:
        samples = parameter_samples(archive)
    rows = report["parameters"]
    for index, row in enumerate(rows):
        fig, ax = plt.subplots(figsize=(4.2, 2.7), layout="constrained")
        draw_parameter(ax, row, samples[row["name"]])
        save_figure(fig, directory, f"parameter-{index}")
    draw_recovery(report, rows, samples, directory)
    if report["chain_shape"]:
        chains, draws = report["chain_shape"]
        for i, row in enumerate(rows):
            fig, ax = plt.subplots(figsize=(9, 2.5), layout="constrained")
            grouped = samples[row["name"]].reshape(chains, draws)
            for chain, values in enumerate(grouped):
                ax.plot(
                    np.arange(1, draws + 1),
                    values,
                    lw=0.6,
                    alpha=0.7,
                    label=f"Chain {chain + 1}",
                )
            ax.set(xlabel="Retained sweep", ylabel=row["name"])
            ax.legend(loc="upper right", frameon=False, ncol=chains, fontsize=9)
            save_figure(fig, directory, f"trace-{i}")
    # A printable overview remains available, with separate axes for parameters.
    columns = min(3, len(rows))
    row_count = (len(rows) + columns - 1) // columns
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(4.1 * columns, 2.9 * row_count),
        squeeze=False,
        layout="constrained",
    )
    for ax, row in zip(axes.flat, rows):
        draw_parameter(ax, row, samples[row["name"]])
    for ax in list(axes.flat)[len(rows) :]:
        ax.set_visible(False)
    verdict = "PASS" if report["passed"] else "FAIL"
    diagnostics = report["checks"]["chain_diagnostics"]
    diagnostic_label = (
        "independent GCR draws"
        if diagnostics["kind"] == "iid_exact_draws"
        else "chain diagnostics " + ("PASS" if diagnostics["passed"] else "FAIL")
    )
    fig.suptitle(
        f"{report['title']} | {verdict}\n{report['method']} | {diagnostic_label}",
        fontsize=14,
    )
    fig.supxlabel(
        "Dashed orange: generating truth   /   Solid blue: posterior mean   /   Shading: 99% interval",
        fontsize=10,
    )
    save_figure(fig, directory, "demo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/inference-demo"))
    args = parser.parse_args()
    paths = sorted(
        args.input.glob("*/result.json"),
        key=lambda p: (
            list(CASES).index(p.parent.name) if p.parent.name in CASES else len(CASES)
        ),
    )
    if not paths:
        parser.error(
            f"No case/result.json files under {args.input}; run the demos first."
        )
    reports = []
    for path in paths:
        report = json.loads(path.read_text())
        if report.get("kind") == "real_observations" and report["case"] == "tris_haslam":
            if __package__:
                from .tris_presentation import render_figures
            else:
                from tris_presentation import render_figures
            render_figures(report, path.parent)
            reports.append((report, path.parent.name))
            continue
        if report.get("kind") == "real_observations" and report["case"] in {"tris_haslam_no_rsb", "tris_haslam_rsb"}:
            # Intermediate full-data fits support the comparison artifact only.
            continue
        if report.get("kind") == "real_observations" and report["case"] == "tris_haslam_rsb_comparison":
            reports.append((report, path.parent.name))
            continue
        if report["case"] not in CASES and report["case"] not in LEGACY_CASES:
            parser.error(f"Unknown demo case {report['case']!r} in {path}")
        if not comparison_visible(args.input.name, report["case"]):
            continue
        render(report, path.parent)
        reports.append((report, path.parent.name))
    write_gallery(reports, args.input)
    print(f"Rendered {len(reports)} walkthroughs: {args.input / 'index.html'}")


if __name__ == "__main__":
    main()
