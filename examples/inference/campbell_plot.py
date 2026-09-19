"""Render saved Campbell results without JAX or inference dependencies."""

import numpy as np


def render_figures(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(directory / "maps.npz") as maps:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        lim = np.quantile(np.abs(maps["sky"]), 0.995)
        for ax, key, title in zip(
            axes,
            ("sky", "observed_sky", "analytic_mean"),
            ("Injected Poisson sky", "Observed sky", "Analytic posterior mean"),
            strict=True,
        ):
            handle = ax.imshow(
                maps[key], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim
            )
            ax.set(title=title, xlabel="pixel x", ylabel="pixel y")
        fig.colorbar(
            handle,
            ax=axes.tolist(),
            label="Sky amplitude (colour clips at 99.5%)",
            shrink=0.75,
        )
        fig.savefig(directory / "sky.png", dpi=160)
        plt.close(fig)
    if __package__:
        from .campbell_distributions import render_distribution_figures
    else:
        from campbell_distributions import render_distribution_figures
    render_distribution_figures(report, directory)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    grid = np.asarray(report["grid"])
    oracle = np.asarray(report["oracle_profile"])
    peak = oracle.max()
    interval = np.asarray(report["oracle_interval95"])
    margin = (interval[1] - interval[0]) / 2
    view = (max(grid[0], interval[0] - margin), min(grid[-1], interval[1] + margin))
    for ax, field, title in zip(
        axes[:2],
        ("factorized_profile", "joint_profile"),
        ("Exact independence: factorized IS", "One weight per whole map"),
        strict=True,
    ):
        ax.plot(grid, oracle - peak, "k", lw=2, label="Analytic evidence")
        for bank in report["banks"]:
            ax.plot(
                grid, np.asarray(bank[field]) - peak, "--", label=f"Bank {bank['seed']}"
            )
        ax.axvline(
            report["config"]["rate"], color="#bc6b2b", ls=":", label="Injected rate"
        )
        ax.set(
            title=title,
            xlabel="Poisson rate λ",
            ylabel="log evidence − analytic grid maximum",
            ylim=(-15, 2),
            xlim=view,
        )
        ax.legend(fontsize=8)
    rows = report["scaling"]
    axes[2].loglog(
        [r["cells"] for r in rows],
        [r["cold_seconds"] for r in rows],
        "o--",
        label="First call (compile + draw)",
    )
    axes[2].loglog(
        [r["cells"] for r in rows],
        [r["warm_seconds_median"] for r in rows],
        "o-",
        label="Warm draw, median of 3",
    )
    axes[2].set(
        title="1,024 independent Gaussian maps",
        xlabel="Latent coefficients per map",
        ylabel="Synchronized wall time (s)",
    )
    axes[2].legend(fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"scaling.{extension}", dpi=170)
    plt.close(fig)
