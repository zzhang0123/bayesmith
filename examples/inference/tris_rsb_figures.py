"""Offline review figures from saved paired posterior draws and PPC summaries."""

from pathlib import Path

import numpy as np


def render_review_figures(report, directory):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    directory = Path(directory)
    models, draws = report["models"], {}
    colors = {"no_rsb": "#176b78", "rsb": "#bd6430"}
    labels = {"no_rsb": "M0 · no RSB", "rsb": "M1 · RSB"}
    for variant in models:
        path = directory.parent / f"tris_haslam_{variant}" / "posterior.npz"
        if not path.exists():
            return
        with np.load(path, allow_pickle=False) as data:
            draws[variant] = dict(data)

    def save(fig, name):
        for extension in ("png", "svg"):
            fig.savefig(directory / f"{name}.{extension}", dpi=155, bbox_inches="tight")
        plt.close(fig)

    # Traces remain available when a convergence gate fails.
    fig, axes = plt.subplots(3, 2, figsize=(12, 7), layout="constrained")
    for column, variant in enumerate(("no_rsb", "rsb")):
        shape = models[variant].get("chain_shape")
        if not shape:
            plt.close(fig)
            return
        values = draws[variant]
        coordinates = [("Haslam monopole [K]", values["haslam_monopole_K"]),
                       ("600-MHz correction / prior SD", values["zero_standard"][:, 0]),
                       ("RSB amplitude at 1 GHz [K]", values["rsb_amplitude"]) if variant == "rsb"
                       else ("high-latitude spectral index", values["beta"][:, 2])]
        for row, (name, samples) in enumerate(coordinates):
            for chain, trace in enumerate(samples.reshape(shape)):
                axes[row, column].plot(trace, lw=.5, alpha=.65, label=f"chain {chain+1}")
            axes[row, column].set(ylabel=name, xlabel="retained draw", title=labels[variant] if row == 0 else "")
            axes[row, column].ticklabel_format(useOffset=False, axis="y")
        axes[0, column].legend(fontsize=8, ncol=2)
    fig.suptitle("Retained chains · distinct starts · warmup excluded", fontsize=14)
    save(fig, "tris-rsb-traces")
    if not all(model.get("passed") and model.get("frequencies") for model in models.values()):
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col", layout="constrained")
    for i in range(2):
        first = models["no_rsb"]["frequencies"][i]
        ra = first["ra_deg"]
        axes[0, i].errorbar(ra, first["data_k"], yerr=first["sigma_k"], fmt=".", color="#253a4b", ms=3, lw=.5, label="TRIS ± statistical SD")
        for variant in models:
            row = models[variant]["frequencies"][i]
            color = colors[variant]
            axes[0, i].plot(ra, row["mean_function"]["mean"], color=color, lw=1.4, ls="--" if variant == "rsb" else "-", label=labels[variant])
            axes[0, i].fill_between(ra, row["predictive"]["lower"], row["predictive"]["upper"], color=color, alpha=.18)
            axes[1, i].plot(ra, row["residual_k"], color=color, lw=1, ls="--" if variant == "rsb" else "-")
            axes[2, i].plot(ra, row["standardized_residual"], color=color, lw=1, ls="--" if variant == "rsb" else "-")
        axes[0, i].set(title=f"{first['frequency_mhz']:g} MHz", ylabel="RJ temperature [K]")
        axes[0, i].legend(fontsize=8)
        axes[1, i].set(ylabel="observed − model [K]")
        axes[2, i].axhspan(-3, 3, color="grey", alpha=.2)
        axes[2, i].set(ylabel="residual / statistical SD", xlabel="right ascension [deg]")
        for ax in axes[:, i]:
            ax.grid(alpha=.2)
            ax.set_xlim(0, 360)
    fig.suptitle("Converged chains still leave structured TRIS residuals · bands: 95% predictive", fontsize=13)
    save(fig, "tris-rsb-residuals")

    with np.load(directory.parent / "tris_haslam_rsb" / "external.npz", allow_pickle=False) as data:
        external = dict(data)
    frequency = np.geomspace(35, 12000, 250)
    a, b = draws["rsb"]["rsb_amplitude"], draws["rsb"]["rsb_beta"]
    spectrum = a[:, None] * (frequency / 1000.) ** b[:, None]
    lo, med, hi = np.quantile(spectrum, [.025, .5, .975], axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    ax = axes[0]
    nu = external["frequency_mhz"]
    x = 4.799243073e-5 * nu / 2.725
    excess = external["temperature_rj_k"] - 2.725 * x / np.expm1(x)
    for survey, color in (("LWA", "#176b78"), ("ARCADE", "#bd6430")):
        mask = external["survey"] == survey
        ax.errorbar(nu[mask], excess[mask], yerr=external["sigma_rj_k"][mask], fmt="o", ms=4, color=color, capsize=3, label=survey + " published excess")
    ax.fill_between(frequency, lo, hi, color="#bd6430", alpha=.25, label="M1 95% conditional spectrum")
    ax.plot(frequency, med, color="#bd6430", lw=1.5)
    mean_with_offsets = (
        a[:, None] * (nu[None, :] / 1000.) ** b[:, None]
        + draws["rsb"]["calibration_standard"][:, external["survey_code"]] * external["tau_rj_k"]
    ).mean(axis=0)
    ax.scatter(nu, mean_with_offsets, marker="x", s=30, color="#7e3d22", label="M1 mean + survey offset")
    ax.set(xscale="log", yscale="log", xlabel="frequency [MHz]", ylabel="RJ temperature − CMB [K]", title="Joint fit suppresses the excess")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2, which="both")
    ax = axes[1]
    mono = draws["rsb"]["haslam_monopole_K"]
    zero = draws["rsb"]["zero_standard"][:, 0]
    mesh = ax.hexbin(mono, zero, gridsize=35, mincnt=1, cmap="YlOrBr")
    ax.set(xlabel="Haslam monopole correction [K]", ylabel="600-MHz correction / prior SD", title=f"Strong calibration trade-off · r = {np.corrcoef(mono, zero)[0,1]:.3f}")
    fig.colorbar(mesh, ax=ax, label="posterior draws")
    save(fig, "tris-rsb-conflict")

    import healpy as hp
    with np.load(directory.parent / "tris_haslam_rsb" / "maps.npz", allow_pickle=False) as data:
        nside = int(data["nside"])
    lon, lat = np.meshgrid(np.linspace(-np.pi, np.pi, 361), np.linspace(-np.pi / 2, np.pi / 2, 181))
    pixels = hp.ang2pix(nside, np.pi / 2 - lat, (-lon) % (2 * np.pi))
    all_fields = {}
    for variant in models:
        with np.load(directory.parent / f"tris_haslam_{variant}" / "predictions.npz", allow_pickle=False) as data:
            all_fields[variant] = dict(data)
    for variant, fields in all_fields.items():
        fig = plt.figure(figsize=(12.5, 6.2))
        for row, stat in enumerate(("mean", "sd")):
            for col, (name, title, unit) in enumerate((("amplitude", "a", "dimensionless"), ("beta", "β", "dimensionless"), ("haslam", "Recalibrated Haslam", "RJ K"))):
                values = fields[f"{name}_{stat}"]
                ax = fig.add_subplot(2, 3, row * 3 + col + 1, projection="mollweide")
                minimum = min(field[f"{name}_{stat}"].min() for field in all_fields.values())
                maximum = max(field[f"{name}_{stat}"].max() for field in all_fields.values())
                norm = LogNorm(minimum, maximum) if name == "haslam" else Normalize(0 if stat == "sd" else minimum, maximum)
                mesh = ax.pcolormesh(lon, lat, values[pixels], norm=norm, cmap="magma" if name == "haslam" else "viridis", shading="auto", rasterized=True)
                ax.set_title(f"{title} · {'posterior SD' if stat == 'sd' else 'mean'}", fontsize=11)
                ax.grid(alpha=.25)
                ax.set_xticks(np.deg2rad([-120, 0, 120]), ["120°", "RA 0°", "240°"], fontsize=8)
                ax.tick_params(axis="y", labelsize=7)
                fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=.05, pad=.12, shrink=.85, label=unit)
        fig.suptitle(labels[variant] + " · conditional sky fields, not model-error maps", fontsize=14)
        fig.subplots_adjust(hspace=.5, wspace=.2, top=.88, bottom=.08)
        save(fig, f"tris-rsb-fields-{variant}")
