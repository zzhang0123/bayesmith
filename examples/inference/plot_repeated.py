"""Plot saved repeated-simulation summaries; never run or alter inference."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def render(summary, directory):
    directory.mkdir(parents=True, exist_ok=True)
    seeds = summary["registered"]["seeds"]
    truth = summary["registered"]["fixed_roots"]["sigma_w"]
    fig, axes = plt.subplots(1, 2, figsize=(12, max(6, len(seeds)*0.22)), sharex=True, sharey=True, layout="constrained")
    runs = {(r["seed"], r["prior"]): r for r in summary["runs"]}
    for ax, prior, title, color in zip(axes, ("uniform", "mild"),
                                      ("Bounded Uniform", "Mild noise prior"),
                                      ("#275cad", "#198579"), strict=True):
        ax.axvline(truth, color="#b5541c", ls="--", lw=1.5, label="Generating value")
        for i, seed in enumerate(seeds):
            run = runs.get((seed, prior))
            if run is None:
                ax.text(truth, i, "No completed run", va="center", color="#9d2338")
                continue
            row = run["rows"]["sigma_w"]
            low, high = row["intervals"]["0.95"]
            ax.plot([low, high], [i, i], color=color, lw=1.6, alpha=0.7)
            marker = "o" if run["diagnostics_passed"] else "x"
            ax.scatter(row["mean"], i, color=color if run["diagnostics_passed"] else "#9d2338", marker=marker, s=24, zorder=3)
        ax.set(title=title, xlabel=r"Noise scale $\sigma_w$", yticks=np.arange(len(seeds)), yticklabels=seeds)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Independent simulation seed")
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    handles.append(Line2D([], [], color="#9d2338", marker="x", ls="none"))
    labels.append("Diagnostics failed; retained")
    axes[1].legend(handles, labels, loc="best", frameon=False, fontsize=8)
    fig.suptitle("Repeated simulation: posterior means and 95% intervals", fontsize=15)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"sigma-repeats.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(json.loads(args.summary.read_text()), args.output)


if __name__ == "__main__":
    main()
