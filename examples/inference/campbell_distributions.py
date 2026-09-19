"""Independent NumPy/SciPy density illustrations from a saved Campbell sky."""

import json

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, poisson


def mixture_logpdf(x, rate, shot=0.8, diffuse=0.2, *, sky=False, noise=0.0):
    """Poisson mixture with a recorded negligible tail; no Edgeworth involved.

    Four equal neighbouring beam weights allow their source counts to be
    summed into one independent Poisson(4*rate) variable.
    """
    k = np.arange(int(poisson.ppf(1 - 1e-13, rate)) + 1)
    logw = poisson.logpmf(k, rate)
    amplitude = np.sqrt(shot / rate)
    if sky:
        neighbours = np.arange(int(poisson.ppf(1 - 1e-13, 4 * rate)) + 1)
        means = amplitude * (0.6 * k[:, None] + 0.1 * neighbours[None, :] - rate)
        logw = (logw[:, None] + poisson.logpmf(neighbours, 4 * rate)[None, :]).ravel()
        means = means.ravel()
        variance = 0.4 * diffuse + noise**2
    else:
        means = amplitude * (k - rate)
        variance = diffuse + noise**2
    logw -= logsumexp(logw)
    return logsumexp(
        logw + norm.logpdf(np.asarray(x)[..., None], means, np.sqrt(variance)), axis=-1
    )


def edgeworth_pdf(x, rate, shot=0.8, diffuse=0.2, *, sky=False):
    variance = (shot + diffuse) * (0.4 if sky else 1.0)
    k3 = shot**1.5 / np.sqrt(rate) * (0.6**3 + 4 * 0.1**3 if sky else 1.0)
    k4 = shot**2 / rate * (0.6**4 + 4 * 0.1**4 if sky else 1.0)
    z = np.asarray(x) / np.sqrt(variance)
    correction = (
        1
        + k3 / variance**1.5 * (z**3 - 3 * z) / 6
        + k4 / variance**2 * (z**4 - 6 * z**2 + 3) / 24
    )
    correction += k3**2 / variance**3 * (z**6 - 15 * z**4 + 45 * z**2 - 15) / 72
    return norm.pdf(x, scale=np.sqrt(variance)) * correction


def render_distribution_figures(report, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = report["config"]
    if config.get("beam_neighbor", 0.1) != 0.1:
        raise ValueError("the analytic sky PDF illustration requires beam_neighbor=0.1")
    rate = config["rate"]
    shot = config.get("shot_variance", 0.8)
    diffuse = config.get("diffuse_variance", 0.2)
    noise = config.get("noise_std", 0.1)
    with np.load(directory / "maps.npz") as maps:
        coefficients = maps["coefficients"].ravel()
        sky = maps["sky"].ravel()
        observed = maps["observed_coefficients"].ravel()
    x = np.linspace(-5, 8, 1800)
    records = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for ax, values, is_sky, name in zip(
        axes,
        (coefficients, sky),
        (False, True),
        ("Source coefficients x", "Beam-smoothed sky s"),
        strict=True,
    ):
        variance = (shot + diffuse) * (0.4 if is_sky else 1)
        edges = np.linspace(-4 * np.sqrt(variance), 8 * np.sqrt(variance), 85)
        counts, _ = np.histogram(values, bins=edges)
        # Normalize by ALL cells, not only the cells inside the display range.
        density = counts / (values.size * np.diff(edges))
        ax.stairs(
            density,
            edges,
            fill=True,
            alpha=0.25,
            color="#446778",
            label=f"One saved map ({values.size:,} cells)",
        )
        ax.plot(
            x,
            np.exp(mixture_logpdf(x, rate, shot, diffuse, sky=is_sky)),
            color="#176c5c",
            lw=2,
            label="Poisson–Gaussian mixture",
        )
        ax.plot(
            x,
            norm.pdf(x, scale=np.sqrt(variance)),
            color="#66717c",
            ls="--",
            label="Matched Gaussian",
        )
        ax.set(
            title=name,
            xlabel="Mean-subtracted amplitude",
            ylabel="Density",
            xlim=(-3.8 * np.sqrt(variance), 5 * np.sqrt(variance)),
        )
        ax.legend(fontsize=8)
        records[name] = {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
            "total_cells": values.size,
            "outside_display_count": int(values.size - counts.sum()),
        }
    save(fig, directory, "distributions")
    # Show the useful regime first, at unchanged mean and variance. These are
    # analytic illustrations, not new maps or fits to the saved lambda=2 sky.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    sigma = np.sqrt(shot + diffuse)
    accuracy_grid = np.linspace(-10 * sigma, 16 * sigma, 26001)
    gaussian = norm.pdf(accuracy_grid, scale=sigma)
    records["edgeworth_accuracy"] = {
        "integration_interval": [float(accuracy_grid[0]), float(accuracy_grid[-1])],
        "grid_points": accuracy_grid.size,
        "examples": [],
    }
    for ax, lam in zip(axes, (8.0, 32.0), strict=True):
        exact = np.exp(mixture_logpdf(accuracy_grid, lam, shot, diffuse))
        approximation = edgeworth_pdf(accuracy_grid, lam, shot, diffuse)
        gaussian_error = float(np.trapezoid(abs(gaussian - exact), accuracy_grid))
        edgeworth_error = float(np.trapezoid(abs(approximation - exact), accuracy_grid))
        skew = shot**1.5 / (np.sqrt(lam) * sigma**3)
        ax.plot(accuracy_grid, exact, color="#176c5c", lw=2.5, label="Exact mixture")
        ax.plot(
            accuracy_grid, gaussian, color="#66717c", ls="--", label="Matched Gaussian"
        )
        ax.plot(
            accuracy_grid,
            approximation,
            color="#ba642c",
            ls=":",
            lw=2.5,
            label="Edgeworth (through order 1/λ)",
        )
        ax.set(
            xlim=(-3.5 * sigma, 5 * sigma),
            ylim=(0, 0.46 / sigma),
            title=f"λ = {lam:g} · skewness = {skew:.3f}",
            xlabel="Source coefficient x",
            ylabel="Density (linear scale)",
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.text(
            0.98,
            0.68,
            f"Integrated absolute error\nGaussian: {gaussian_error:.4g}\n"
            f"Edgeworth: {edgeworth_error:.4g}\n"
            f"{gaussian_error / edgeworth_error:.0f}× smaller error",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
        )
        records["edgeworth_accuracy"]["examples"].append(
            {
                "rate": lam,
                "skewness": float(skew),
                "gaussian_integrated_absolute_error": gaussian_error,
                "edgeworth_integrated_absolute_error": edgeworth_error,
                "negative_mass_on_interval": float(
                    np.trapezoid(np.maximum(-approximation, 0), accuracy_grid)
                ),
            }
        )
    save(fig, directory, "edgeworth_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    pdf = edgeworth_pdf(x, rate, shot, diffuse)
    axes[0].plot(
        x,
        np.exp(mixture_logpdf(x, rate, shot, diffuse)),
        color="#176c5c",
        label="Proper mixture",
    )
    axes[0].plot(
        x,
        norm.pdf(x, scale=np.sqrt(shot + diffuse)),
        color="#66717c",
        ls="--",
        label="Matched Gaussian",
    )
    axes[0].plot(x, pdf, color="#ba642c", label="Signed Edgeworth")
    axes[0].fill_between(
        x, pdf, 0, where=pdf < 0, color="#bd3030", alpha=0.3, label="Negative density"
    )
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set(
        ylim=(min(-0.001, float(pdf.min()) * 1.25), 1),
        xlim=(-4.5, 6),
        title=f"Tail stress case · λ = {rate:g}",
        xlabel="Source coefficient x",
        ylabel="Signed density (symlog, linear near zero)",
    )
    axes[0].set_yscale("symlog", linthresh=1e-5)
    axes[0].legend(fontsize=8)
    for lam in (0.5, 2.0, 8.0):
        axes[1].plot(
            x, np.exp(mixture_logpdf(x, lam, shot, diffuse)), label=f"λ = {lam:g}"
        )
    axes[1].plot(
        x,
        norm.pdf(x, scale=np.sqrt(shot + diffuse)),
        "k--",
        label="Same Gaussian for all λ",
    )
    axes[1].set(
        xlim=(-3.5, 5),
        xlabel="Source coefficient x",
        ylabel="Density",
        title="Different non-Gaussianity; identical covariance",
    )
    axes[1].legend(fontsize=8)
    save(fig, directory, "edgeworth")
    records["edgeworth"] = {
        "minimum_pdf_on_grid": float(pdf.min()),
        "minimum_at_x": float(x[pdf.argmin()]),
        "joint_zero_field_correction": float(
            1
            + coefficients.size
            * (
                shot**2 / rate / (8 * (shot + diffuse) ** 2)
                - 5 * shot**3 / rate / (24 * (shot + diffuse) ** 3)
            )
        ),
    }
    # A separate, reproducible two-coordinate illustration of conditional IS.
    # These are NOT the saved full-field inference banks.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    rng = np.random.default_rng(9017)
    indices = [int(np.argmin(abs(observed))), int(np.argmax(observed))]
    records["posterior"] = {
        "seed": 9017,
        "draws_per_coordinate": 40000,
        "rate": rate,
        "coordinates": [],
    }
    for ax, idx, title in zip(
        axes,
        indices,
        (
            "A cell near the map centre in amplitude",
            "The brightest observed source cell",
        ),
        strict=True,
    ):
        y = observed[idx]
        prior_variance = shot + diffuse
        variance = 1 / (1 / prior_variance + 1 / noise**2)
        mean = variance * y / noise**2
        values = mean + np.sqrt(variance) * rng.normal(size=40000)
        logw = mixture_logpdf(values, rate, shot, diffuse) - norm.logpdf(
            values, scale=np.sqrt(prior_variance)
        )
        weights = np.exp(logw - logsumexp(logw))
        grid = np.linspace(
            mean - 4 * np.sqrt(variance), mean + 4 * np.sqrt(variance), 600
        )
        posterior = np.exp(
            mixture_logpdf(grid, rate, shot, diffuse)
            + norm.logpdf(y, grid, noise)
            - mixture_logpdf(y, rate, shot, diffuse, noise=noise)
        )
        bins = np.linspace(grid[0], grid[-1], 55)
        counts, _ = np.histogram(values, bins=bins, weights=weights)
        ax.stairs(
            counts / np.diff(bins),
            bins,
            color="#775a99",
            fill=True,
            alpha=0.25,
            label="Reweighted Gaussian draws",
        )
        ax.plot(
            grid, posterior, color="#176c5c", lw=2, label="Exact non-Gaussian posterior"
        )
        ax.plot(
            grid,
            norm.pdf(grid, mean, np.sqrt(variance)),
            color="#66717c",
            ls="--",
            label="Gaussian reference posterior q",
        )
        ax.axvline(
            coefficients[idx], color="#ba642c", ls=":", label="Injected coefficient"
        )
        ax.set(
            title=title,
            xlabel=f"x at cell {idx} (observed y = {y:.2f})",
            ylabel="Conditional density",
        )
        ax.legend(fontsize=7.5)
        records["posterior"]["coordinates"].append(
            {"index": idx, "observed": float(y), "ess": float(1 / np.sum(weights**2))}
        )
    save(fig, directory, "posterior_density")
    (directory / "distribution-comparison.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )


def save(fig, directory, name):
    import matplotlib.pyplot as plt

    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=170)
    # Keep labels readable in the narrow notebook panel without resampling data.
    if len(fig.axes) == 2:
        grid = fig.add_gridspec(2, 1)
        for axis, cell in zip(fig.axes, grid, strict=True):
            axis.set_subplotspec(cell)
        fig.set_size_inches(7.5, 8.5)
        fig.savefig(directory / f"{name}-stacked.png", dpi=170)
    plt.close(fig)
