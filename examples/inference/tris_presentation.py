"""Offline figures and bilingual notebook chapter for the TRIS observations."""

from __future__ import annotations

import html
from urllib.parse import quote

import numpy as np

if __package__:
    from .methodology_guide import bi
    from .methodology_guide import math as equation
else:
    from methodology_guide import bi
    from methodology_guide import math as equation

STAGES = (
    ("model", ("Data & maps", "数据与重建图")),
    ("methods", ("Methods & parameter blocks", "方法与参数块")),
    ("diagnostics", ("Diagnostics & priors", "诊断与先验")),
    ("sampling", ("Posterior sampling", "后验采样")),
    ("recovery", ("Findings", "发现与解释")),
)


def render_figures(report, directory):
    import healpy as hp
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm

    with np.load(directory / "maps.npz", allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    with np.load(directory / "predictions.npz", allow_pickle=False) as z:
        prediction = {k: z[k] for k in z.files}
    with np.load(directory / "posterior.npz", allow_pickle=False) as z:
        draws = {k: z[k] for k in z.files}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    colors = ("#176b78", "#d16c3e")

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=155, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    longitude = np.linspace(-np.pi, np.pi, 361)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 181)
    lon, lat = np.meshgrid(longitude, latitude)
    pixels = hp.ang2pix(int(data["nside"]), np.pi / 2 - lat, (-lon) % (2 * np.pi))

    def sky_panel(fig, subplot, values, title, unit, *, cmap="viridis", norm=None, mask=None):
        ax = fig.add_subplot(*subplot, projection="mollweide")
        raster = np.asarray(values)[pixels]
        if mask is not None:
            raster = np.ma.masked_where(np.asarray(mask)[pixels], raster)
        palette = plt.get_cmap(cmap).copy()
        palette.set_bad("#e8ecef")
        mesh = ax.pcolormesh(lon, lat, raster, shading="auto", cmap=palette, norm=norm, rasterized=True)
        ax.set_title(title, fontsize=11, pad=12)
        ax.grid(alpha=.25)
        ax.set_xticks(np.deg2rad([-120, -60, 0, 60, 120]))
        ax.set_xticklabels(["120°", "60°", "0°", "300°", "240°"], fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        bar = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=.05, pad=.12, shrink=.85)
        bar.set_label(unit, fontsize=9)
        return ax

    fig = plt.figure(figsize=(12, 4.7))
    ax = sky_panel(fig, (1, 2, 1), data["template_k"], "Haslam 408 MHz foreground template", "RJ temperature [K]", norm=LogNorm())
    ring_ra = np.linspace(0, 360, 500)
    ring_lon = (-np.deg2rad(ring_ra) + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(ring_lon)
    ax.plot(ring_lon[order], np.full(500, np.deg2rad(42.433333)), color="#f3a947", lw=1.8, label="TRIS boresight")
    ax.legend(loc="lower center", fontsize=8)
    sky_panel(fig, (1, 2, 2), data["region"], "Three fixed Galactic latitude regions", "region: 0 plane · 1 transition · 2 high latitude", cmap="cividis", norm=Normalize(0, 2))
    fig.suptitle("Where the data look, and how the sky is parameterized", fontsize=15, y=1.02)
    save(fig, "tris-context")

    if report.get("sky_products"):
        # Posterior statistics are saved by the inference run, not re-fitted
        # here. Keep the same pixel grid, without smoothing across boundaries.
        fig = plt.figure(figsize=(13, 6.5))
        for j, (name, title, unit, cmap) in enumerate([
            ("amplitude", "a · Haslam amplitude", "dimensionless", "viridis"),
            ("beta", "β · spectral index", "dimensionless", "cividis"),
            ("haslam", "Recalibrated Haslam · 408 MHz", "RJ temperature [K]", "magma"),
        ], 1):
            mean, sd = prediction[f"{name}_mean"], prediction[f"{name}_sd"]
            ax = sky_panel(fig, (2, 3, j), mean, title + " · mean", unit,
                           cmap=cmap, norm=LogNorm() if name == "haslam" else None)
            ax.plot(ring_lon[order], np.full(500, np.deg2rad(42.433333)),
                    color="#f3a947", lw=1.3)
            sd_norm = (LogNorm() if name == "haslam" and np.all(sd > 0) and np.ptp(sd) > 0
                       else Normalize(0, float(sd.max()) if np.any(sd) else 1.0))
            sd_title = title + " · posterior SD" + (" (all zero)" if not np.any(sd) else "")
            sky_panel(fig, (2, 3, j + 3), sd, sd_title, unit,
                      cmap="viridis", norm=sd_norm)
        fig.suptitle("Posterior sky fields · three-region model", fontsize=16, y=1.01)
        fig.text(.5, -.035, "Same equatorial HEALPix grid · orange: TRIS boresight\nFull sky is a regional-model extrapolation; small conditional SD does not measure model error.",
                 ha="center", fontsize=10, color="#51616d")
        fig.subplots_adjust(top=.88, bottom=.10, hspace=.4, wspace=.2)
        save(fig, "tris-sky-fields")

        reference, calibrated = prediction["haslam_reference_k"], prediction["haslam_mean"]
        common = LogNorm(min(reference.min(), calibrated.min()), max(reference.max(), calibrated.max()))
        change = prediction["haslam_change_mean"]
        limit = max(np.max(np.abs(change)), 1e-9)
        fig = plt.figure(figsize=(13, 4.3))
        sky_panel(fig, (1, 3, 1), reference, "Input Haslam · same grid", "RJ temperature [K]", cmap="magma", norm=common)
        sky_panel(fig, (1, 3, 2), calibrated, "Recalibrated Haslam · posterior mean", "RJ temperature [K]", cmap="magma", norm=common)
        sky_panel(fig, (1, 3, 3), change, "Recalibrated − input", "temperature change [K]", cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit))
        fig.suptitle("408 MHz · common temperature scale · CMB is retained unchanged", fontsize=14, y=1.04)
        save(fig, "tris-haslam-change")

    for i in range(2):
        reduction = 1 - (data[f"posterior_sigma_k_{i}"] / data[f"prior_sigma_k_{i}"])**2
        mask = reduction < .001
        mapped = data[f"map_k_{i}"]
        fitted = prediction[f"map_mean_{i}"]
        residual = mapped - fitted
        common = Normalize(min(mapped[~mask].min(), fitted[~mask].min()),
                           max(mapped[~mask].max(), fitted[~mask].max()))
        maximum = max(np.max(np.abs(residual[~mask])), 1e-9)
        fig = plt.figure(figsize=(13, 8.2))
        panels = [
            (mapped, "limTOD reconstructed map", "RJ temperature [K]", "magma", common),
            (fitted, "Posterior mean through map response", "RJ temperature [K]", "magma", common),
            (residual, "Map − predicted map", "residual [K]", "RdBu_r", TwoSlopeNorm(0, -maximum, maximum)),
            (data[f"posterior_sigma_k_{i}"], "Map posterior uncertainty", "prior-conditional SD [K]", "viridis", None),
            (data[f"noise_sigma_k_{i}"], "Propagated observational noise", "SD of repeated map estimates [K]", "viridis", None),
            (reduction, "Where observations constrain the map", "fractional prior variance reduction", "cividis", Normalize(0, 1)),
        ]
        for j, (values, title, unit, cmap, norm) in enumerate(panels, 1):
            sky_panel(fig, (2, 3, j), values, title, unit, cmap=cmap, norm=norm, mask=mask if j < 6 else None)
        fig.suptitle(f"{data['frequency_mhz'][i]:.1f} MHz · response-aware map comparison", fontsize=15, y=1.01)
        fig.text(.5, -.025, "Equatorial coordinates · RA increases leftward · grey: <0.1% prior variance reduction\nPosterior SD and observational-noise SD describe different distributions.", ha="center", fontsize=10, color="#51616d")
        fig.subplots_adjust(hspace=.4, wspace=.2)
        save(fig, f"tris-maps-{i}")

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 9), sharex="col", gridspec_kw={"height_ratios": [2, 1, 1]}, layout="constrained")
    for i, summary in enumerate(report["frequencies"]):
        x, y, sigma = [np.asarray(summary[k]) for k in ("ra_deg", "data_k", "sigma_k")]
        mean, predictive = summary["mean_function"], summary["predictive"]
        axes[0, i].errorbar(x, y, yerr=sigma, fmt=".", ms=3, color="#263e4b", lw=.7, label="TRIS observed ± statistical SD")
        axes[0, i].fill_between(x, predictive["lower"], predictive["upper"], color=colors[i], alpha=.2, label="95% replicated observations")
        axes[0, i].plot(x, mean["mean"], color=colors[i], label="posterior mean")
        axes[0, i].fill_between(x, mean["lower"], mean["upper"], color=colors[i], alpha=.6, label="95% mean function")
        axes[0, i].set_title(f"{summary['frequency_mhz']:.1f} MHz", fontsize=13)
        axes[0, i].set_ylabel("RJ temperature [K]")
        axes[0, i].legend(fontsize=8)
        axes[1, i].errorbar(x, summary["residual_k"], yerr=sigma, fmt=".-", ms=3, lw=.8, color=colors[i])
        axes[1, i].set_ylabel("observed − model [K]")
        axes[2, i].plot(x, summary["standardized_residual"], color=colors[i], lw=1)
        axes[2, i].axhspan(-3, 3, color="#566977", alpha=.16, label="±3 statistical SD")
        axes[2, i].set_ylabel("residual / statistical SD")
        axes[2, i].set_xlabel("right ascension [deg]")
        axes[2, i].legend(fontsize=8)
        for ax in axes[:, i]:
            ax.set_xlim(0, 360)
            ax.set_xticks(np.arange(0, 361, 60))
            ax.grid(alpha=.15)
        for ax in axes[1:, i]:
            ax.axhline(0, color="#51616d", lw=.7)
    fig.suptitle("Structured residuals remain far above the supplied noise", fontsize=15)
    save(fig, "tris-residuals")

    fig, ax = plt.subplots(figsize=(10, 3.7), layout="constrained")
    ax.scatter(data["points_ra_deg"], data["points_temperature_k"], color="#94653c", s=42)
    ax.set(xlabel="right ascension [deg]", ylabel="RJ temperature [K]", xlim=(0, 360),
           title=f"{float(data['points_frequency_mhz']):.1f} MHz · six absolute measurements")
    ax.grid(alpha=.15)
    ax.text(.03, .93, f"Shared zero-level uncertainty: {float(data['points_zero_level_k']):.2f} K\nStatistical errors are unpublished; these points do not enter the fit.", transform=ax.transAxes, va="top", fontsize=10)
    save(fig, "tris-points")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), layout="constrained")
    for ax, name, title in zip(axes, ("amplitude", "beta", "zero_level_K"),
                                ("Haslam amplitude", "Spectral index", "Sky-temperature correction [K]"), strict=True):
        rows = [r for r in report["parameters"] if r["parameter"] == name]
        for j, row in enumerate(rows):
            ax.errorbar(row["median"], j, xerr=[[row["median"] - row["lower"]], [row["upper"] - row["median"]]], fmt="o", color=colors[0], capsize=4)
        ax.set_yticks(range(len(rows)), ["plane", "transition", "high latitude"] if name != "zero_level_K" else ["600.5 MHz", "817.8 MHz"])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=12)
        ax.grid(axis="x", alpha=.2)
        ax.ticklabel_format(style="plain", axis="x", useOffset=False)
        ax.margins(x=.2, y=.3)
    fig.suptitle("95% marginal posterior intervals · conditional on the stated model", fontsize=14)
    save(fig, "tris-parameters")

    flat = np.column_stack([draws["amplitude"], draws["beta"], draws["zero_level_K"]])
    labels = ["A plane", "A trans", "A high", "β plane", "β trans", "β high", "δ600", "δ820"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5), layout="constrained")
    corr = np.corrcoef(flat.T)
    mesh = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(8), labels, rotation=45, ha="right")
    ax.set_yticks(range(8), labels)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr[i, j]) > .65 else "#20333e")
    fig.colorbar(mesh, ax=ax, label="posterior Pearson correlation", shrink=.8)
    ax.set_title("Amplitude, spectral slope and zero level trade off")
    save(fig, "tris-correlations")
    chains, count = report["chain_shape"]
    chained = flat.reshape(chains, count, 8)
    fig, axes = plt.subplots(4, 2, figsize=(12, 8), layout="constrained")
    for i, ax in enumerate(axes.flat):
        for c in range(chains):
            ax.plot(chained[c, :, i], lw=.45, alpha=.6, color=colors[c % 2], label=f"chain {c + 1}")
        ax.set_title(labels[i], fontsize=11)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax.set_xlabel("retained draw")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Retained posterior chains · warmup excluded", fontsize=14)
    save(fig, "tris-traces")


def figure(folder, name, en, zh):
    stem = f"{quote(folder, safe='')}/{name}"
    return (f'<figure class="tris-figure"><img loading="lazy" src="{stem}.png" alt="{html.escape(en, quote=True)}">'
            f'<figcaption>{bi(en, zh)} <a href="{stem}.svg">SVG</a></figcaption></figure>')


def paragraph(en, zh):
    return '<p>' + bi(en, zh) + '</p>'


def render_case(report, folder, index=0):
    from json import dumps

    esc = html.escape
    diagnostics = report["checks"]["chain_diagnostics"]
    sites = diagnostics["sites"]
    def diagnostic_text(key, aggregate, precision):
        values = [s.get(key) for s in sites.values()]
        if not values or any(v is None or not np.isfinite(v) for v in values):
            return "unavailable"
        return f"{aggregate(values):.{precision}f}"

    rhat = diagnostic_text("r_hat", max, 4)
    ess = diagnostic_text("ess", min, 0)
    converged = diagnostics["passed"]
    content = {}
    content["model"] = paragraph(
        "Two absolute drift scans become map observations. The telescope follows declination +42° with the same broad measured beam at both frequencies. The reconstruction fills unmeasured directions with a declared prior; the likelihood retains its transfer response and correlated observational uncertainty.",
        "两条绝对温度扫描环被重建为 map 数据。望远镜沿约 +42° 赤纬扫描，两个频率使用相同的实测宽波束。重建先验补足未测方向；参数似然同时保留 mapmaking 的响应与观测噪声相关性。")
    content["model"] += '<div class="tris-metrics">' + ''.join(
        f'<div><strong>{value}</strong><span>{bi(en, zh)}</span></div>' for value, en, zh in
        [("600.5 / 817.8", "effective frequencies · MHz", "有效频率 · MHz"),
         ("120 + 120", "archive observations", "档案观测点"),
         (str(report["data_manifest"]["maps"][0]["map_pixels"]), "pixels per reconstructed map", "每幅重建图像素数"),
         ("8", "sampled coordinates", "采样参数维数")]) + '</div>'
    content["model"] += figure(folder, "tris-context", "Equatorial sky; the orange ring marks the TRIS boresight. Regions are defined by Galactic latitude.", "赤道坐标天空；橙线表示 TRIS 指向。参数区域按银纬划分。")
    content["model"] += '<div class="tris-frequency-controls"><label for="tris-frequency">' + bi("Inspect a reconstructed map", "查看重建图") + '</label> <select id="tris-frequency" data-tris-frequency><option value="0">600.5 MHz</option><option value="1">817.8 MHz</option></select></div>'
    for i in range(2):
        content["model"] += f'<div data-tris-map="{i}">' + figure(folder, f"tris-maps-{i}",
            "Data, transferred model and residual share a footprint; grey regions have <0.1% variance reduction from observations. The bottom row separates posterior uncertainty, observational noise and information from data.",
            "数据、经过同一 mapmaking 响应的模型及残差使用相同范围；灰区的观测方差缩减不足 0.1%。下排分别显示后验不确定度、传播后的观测噪声与数据带来的信息。") + '</div>'
    content["model"] += figure(folder, "tris-points", "The third frequency remains visible as contextual observations. Its common zero-level uncertainty is not six independent errors.", "第三个频率保留为背景观测。共同零点误差不能当作六个独立误差。")
    source = report["data_manifest"]["upstream_model"]
    content["methods"] = paragraph(
        'The forward equation follows bayesian_skymap: apply_beam / calc_model_spectral. This case restricts the reference amplitudes and spectral indices to three Galactic latitude regions: |b| < 10°, 10°–30°, and ≥30°. Haslam fixes the spatial template and calibration reference; no upstream sampling code is used.',
        '前向方程来自 bayesian_skymap 的 apply_beam / calc_model_spectral。本案例将参考振幅和谱指数限制在三个银纬区域：|b| < 10°、10°–30° 与 ≥30°。Haslam 固定空间模板及校准参考；采样由 bayesmith 执行。')
    content["methods"] += equation(r'T_\nu(p)=a_{r(p)}[H_{408}(p)-T_\mathrm{CMB,RJ}(408)]\left(\frac{\nu}{408\,\mathrm{MHz}}\right)^{\beta_{r(p)}}+T_\mathrm{CMB,RJ}(\nu)')
    content["methods"] += paragraph("Amplitudes are relative to Haslam at 408 MHz. The same measured TRIS beam is applied to both frequencies. CMB is converted to Rayleigh–Jeans temperature at each effective frequency. All remaining Haslam emission is treated as one phenomenological foreground.", "振幅相对于 408 MHz 的 Haslam 模板。两个频率使用同一实测 TRIS 波束。CMB 在各有效频率转为 Rayleigh–Jeans 温度；扣除 CMB 后的 Haslam 辐射作为一个经验前景分量。")
    content["methods"] += '<h3>' + bi("The map is an observation with a response", "重建图具有自己的观测响应") + '</h3>'
    content["methods"] += equation(r'\hat m=b+Wd,\quad W=C_\mathrm{post}A^TN^{-1},\quad b=m_0-WAm_0')
    content["methods"] += equation(r'\mu_m(\theta)=b+W[AT_\nu(\theta)-\delta_\nu\mathbf1],\quad C_\mathrm{noise}=WNW^T')
    content["methods"] += paragraph("SVD whitening uses the supported modes of W√N. All 120 modes per frequency survive in this recorded run. The mapmaking prior cancels from likelihood differences; a regression test changes that prior and checks the original Gaussian likelihood independently. Cpost is retained for displaying reconstruction uncertainty, while Cnoise enters the likelihood.", "通过 W√N 的 SVD 白化其支持的模式。本次每个频率保留全部 120 个模式。mapmaking 先验在似然差中抵消；独立回归测试更换重建先验后核对原始高斯似然。Cpost 用于展示重建后验不确定度，Cnoise 用于参数似然。")
    if __package__:
        from .presentation import dag_svg
    else:
        from presentation import dag_svg
    content["methods"] += '<div class="tris-dag">' + dag_svg(report["dag"], "tris", labels={
        "Haslam_region_response": ("Haslam region response", "Haslam 区域响应"),
        "RJ_CMB_response": ("CMB response", "CMB 响应"),
        "common_offset_response": ("Common correction response", "共同修正响应"),
        "frequency_index": ("Frequency channel", "频率通道"),
        "frequency_mhz": ("Effective frequency", "有效频率"),
        "beta": ("Regional spectral indices", "区域谱指数"),
        "zero_standard": ("Standardized corrections", "标准化零点修正"),
        "frequency_scaling": ("Spectral scaling", "频率缩放"),
        "zero_level_K": ("Sky correction [K]", "天空修正 [K]"),
        "whitened_map_mean": ("Whitened model map", "白化的模型图"),
    }, symbols={
        "Haslam_region_response": "B_H", "RJ_CMB_response": "c_CMB",
        "common_offset_response": "q", "frequency_index": "j", "frequency_mhz": "ν [MHz]",
        "zero_standard": "z_ν", "frequency_scaling": "a_r (ν/408)^β_r",
        "zero_level_K": "δ_ν", "whitened_map_mean": "μ_w",
    }) + '</div>'
    content["methods"] += '<p><a href="' + source["url"] + '/blob/' + source["revision"] + '/bayesian_func.py">' + bi("Original model source (pinned revision)", "原始模型源码（固定版本）") + '</a></p>'
    content["methods"] += '<details><summary>' + bi("Recorded JAX forward operators", "本次运行的 JAX 前向算子") + '</summary><pre>' + esc(report.get("model_source", "Source snapshot was not saved in this earlier run.")) + '</pre></details>'

    content["diagnostics"] = '<div class="tris-warning">' + paragraph(
        "The likelihood computation is verified, and the residual check flags model mismatch. The published statistical uncertainties are much smaller than the remaining structure along the scan.",
        "似然计算已验证，残差检查发现模型失配。扫描环中剩余的结构显著大于公布的统计不确定度。") + '</div>'
    content["diagnostics"] += figure(folder, "tris-residuals", "Top: observed profiles and separate 95% mean-function / replicated-observation intervals. Middle: residual K. Bottom: residual divided by statistical SD. The common zero-level correction is already included in the prediction.", "上：实际扫描及分别计算的 95% 均值函数区间与复制观测区间；中：K 单位残差；下：残差除以统计标准差。模型预测已包含共同零点修正。")
    content["diagnostics"] += '<div class="table-wrap"><table><thead><tr><th>MHz</th><th>RMS [K]</th><th>χ² / N</th><th>' + bi("PPC exceedances", "后验预测检验超越次数") + '</th></tr></thead><tbody>'
    for s in report["frequencies"]:
        content["diagnostics"] += f'<tr><td>{s["frequency_mhz"]}</td><td>{s["residual_rms_k"]:.3f}</td><td>{s["chi_square_per_observation"]:.1f}</td><td>{round(s["ppc_tail_probability"] * s["ppc_replicates"])}/{s["ppc_replicates"]}</td></tr>'
    content["diagnostics"] += '</tbody></table></div>' + paragraph("χ²/N is per observation, with no fitted-degrees-of-freedom correction. PPC compares the observed and replicated standardized residual sums at each posterior draw. Zero exceedances is a finite Monte Carlo result, not an exactly zero probability.", "χ²/N 按观测点计，没有扣除拟合自由度。PPC 在每个后验样本下比较真实与复制数据的标准化残差平方和。零次超越是有限 Monte Carlo 结果，不代表概率严格为零。")
    content["diagnostics"] += '<h3>' + bi("Assumptions that limit the inference", "限制推断解释的假设") + '</h3>'
    assumptions = [
        ("Statistical errors are treated as independent because the archive supplies no sample covariance. Each ring has one zero-error row; the fixed 0.004 K minimum positive error supplies its floor.", "档案没有提供样本协方差，因此统计误差按独立处理。每条环有一个零误差点，使用最小正误差 0.004 K 作为固定下限。"),
        ("One zero-level correction is shared by each frequency. At 820 MHz, +0.430/−0.300 K describes the correction to the measured sky; this published range already uses astrophysical constraints. We explicitly choose equal-probability half-normal sides because the archive specifies no probability density.", "每个频率共享一个零点修正。820 MHz 的 +0.430/−0.300 K 是对测得天空温度的修正，公布范围已包含天体物理约束。档案未定义概率密度，这里明确采用两侧等概率的半正态分布。"),
        ("Haslam calibration, its residual zero level, sample correlations, beam interpolation and pixelization uncertainty are not marginalized. Three regional power laws cannot represent arbitrary spatial or frequency structure; these data alone do not identify the cause of the mismatch.", "本次没有边缘化 Haslam 校准、残余零点、样本相关性、波束插值与像素化误差。三区域幂律不能表示任意空间或频率结构；这组数据本身无法确定失配来自哪一项。"),
    ]
    content["diagnostics"] += ''.join(paragraph(*a) for a in assumptions)
    content["sampling"] = '<div class="tris-metrics">' + ''.join(f'<div><strong>{value}</strong><span>{bi(en, zh)}</span></div>' for value, en, zh in [
        (f'{report["chain_shape"][0]} × {report["chain_shape"][1]}', "retained draws", "保留样本"),
        (rhat, "maximum site R̂", "最大站点 R̂"), (ess, "minimum site ESS", "最小站点 ESS"),
        (str(diagnostics["divergences"]), "divergences", "发散次数")]) + '</div>'
    content["sampling"] += paragraph(f'Bayesmith selected {esc(report["method"])}. The recorded run uses seed {report["seed"]}, {report["warmup"]} warmup steps per chain, float64, and a fixed budget. Initialization candidates come from the declared prior/support and are validated before warmup. No generating parameters exist for these observations.', f'Bayesmith 选择 {esc(report["method"])}。本次 seed={report["seed"]}，每链 warmup={report["warmup"]}，使用 float64 和固定预算。初始化由声明的先验/支持域产生，并在 warmup 前验证。真实观测没有用于参数恢复检查的生成真值。')
    content["sampling"] += paragraph('Amplitude priors: Uniform(0.2, 3.0). Spectral indices: Uniform(−4.0, −1.5). Both apply independently to the three regions. The two standardized common corrections have Normal(0, 1) priors and are transformed into K using the published zero-level scales.', '三区域振幅分别使用 Uniform(0.2, 3.0)，谱指数分别使用 Uniform(−4.0, −1.5)。两个标准化共同零点修正使用 Normal(0, 1)，再按公布的误差尺度变为 K。')
    content["sampling"] += figure(folder, "tris-traces", "All sampled sky coordinates and derived zero-level corrections, separated by chain.", "按链分别显示全部天空参数与转换后的零点修正。")
    content["sampling"] += '<details><summary>' + bi("Recorded preflight, initialization and stopping evidence", "预检、初始化与停止的原始记录") + '</summary><pre>' + esc(dumps({"preflight": report["preflight"], "execution": report["execution"]}, indent=2, ensure_ascii=False)) + '</pre></details>'
    content["recovery"] = '<div class="tris-warning">' + paragraph("The narrow posterior describes the best compromise inside this restricted sky model. The residual check prevents interpreting these intervals as precise physical measurements of the true spectral sky.", "窄后验描述的是这个受限天空模型内部的最佳折中。残差检查表明，这些区间不能被解释为对真实天空谱指数的精确物理测量。") + '</div>'
    if report.get("sky_products"):
        content["recovery"] += '<h3>' + bi("Maps of a, β and recalibrated Haslam", "a、β 与重校准 Haslam 的地图") + '</h3>'
        content["recovery"] += figure(folder, "tris-sky-fields",
            "Top: posterior means. Bottom: sample standard deviations of the saved joint draws. a and β are constant within each of three Galactic latitude regions; all maps retain the prepared equatorial grid. Outside the TRIS scan, these are model extrapolations, not independent pixel measurements.",
            "上排为后验均值，下排为已保存联合样本的标准差。a 与 β 在三个银纬区域内各为常数；所有图沿用准备数据时的赤道坐标网格。TRIS 扫描范围外是模型外推，不是独立像素测量。")
        content["recovery"] += equation(r'H_{408}^{\rm recal}(p)=a_{r(p)}[H_{408}(p)-T_{\rm CMB,RJ}(408)]+T_{\rm CMB,RJ}(408)')
        content["recovery"] += paragraph(
            "Recalibrated Haslam is the model sky at 408 MHz, before the beam and mapmaking response. The CMB is restored unchanged; TRIS zero-level corrections are instrument terms and are not added to this sky. At the reference frequency β drops out. These SDs propagate parameter uncertainty only, with the Haslam template fixed.",
            "重校准 Haslam 是模型在 408 MHz、经过波束与 mapmaking 响应之前的天空。CMB 原样加回；TRIS 零点修正属于仪器项，不加到这幅天空图。在参考频率处 β 不参与缩放。这些标准差只传播参数不确定度，Haslam 模板保持固定。")
        content["recovery"] += figure(folder, "tris-haslam-change",
            "Input and recalibrated Haslam use the same logarithmic temperature scale; their difference uses a linear scale centered on zero. The input is the same rotated, degraded map used in this fit.",
            "输入与重校准 Haslam 共用对数温度色标；差值使用以零为中心的线性色标。输入图是本次拟合使用的同一幅旋转、降分辨率后的地图。")
        content["recovery"] += '<p><a download href="' + quote(folder, safe='') + '/predictions.npz">' + bi("Download numerical maps, SDs and 95% intervals (NPZ)", "下载地图、标准差与 95% 区间（NPZ）") + '</a></p>'
    content["recovery"] += figure(folder, "tris-parameters", "95% marginal equal-tail intervals. No truth lines or recovery score are defined for real observations.", "95% 边缘等尾区间。真实观测不定义真值线或恢复分数。")
    content["recovery"] += '<div class="table-wrap"><table><thead><tr><th>' + bi("Parameter / region", "参数 / 区域") + '</th><th>' + bi("Mean", "均值") + '</th><th>95% interval</th></tr></thead><tbody>'
    for row in report["parameters"]:
        content["recovery"] += f'<tr><td>{esc(row["name"])} · {esc(row["region"])}</td><td>{row["mean"]:.5f}</td><td>[{row["lower"]:.5f}, {row["upper"]:.5f}]</td></tr>'
    content["recovery"] += '</tbody></table></div>'
    content["recovery"] += figure(folder, "tris-correlations", "The parameter trade-offs remain visible even when marginal intervals are narrow.", "即使边缘区间很窄，参数之间的补偿关系仍然清晰可见。")
    correction = [r for r in report["parameters"] if r["parameter"] == "zero_level_K"]
    content["recovery"] += paragraph(f'The inferred 600.5-MHz correction is {correction[0]["mean"]:.3f} K, about {correction[0]["mean"]/.066:.1f} times the published 0.066-K scale. It can absorb template/model mismatch. This case therefore establishes the data-to-map-to-posterior pipeline and exposes the limits of the three-region assumption. A next scientific model should examine spatial template flexibility, separate foreground components and calibration/systematic uncertainty.', f'600.5 MHz 的修正后验均值为 {correction[0]["mean"]:.3f} K，约为公布 0.066 K 尺度的 {correction[0]["mean"]/.066:.1f} 倍。它可能吸收模板或模型失配。本案例完成了数据→重建图→参数后验的链路，也揭示了三区域假设的局限。下一步科学模型应检验空间模板的灵活性、独立前景分量及校准/系统误差。')

    if report.get("blocking"):
        if __package__:
            from .case_panels import diagnostics_panel, methods_panel, sampling_panel
        else:
            from case_panels import diagnostics_panel, methods_panel, sampling_panel
        setup_copy = {"structure": (
            "The compiled plan jointly updates three amplitudes, three spectral indices and two standardized corrections. The actual block, selection reasons and a compile-only alternative are recorded below.",
            "编译方案联合更新三个振幅、三个谱指数和两个标准化修正。下方展示实际参数块、选择依据，以及仅编译检查的替代方案。",
        )}
        content["methods"] = methods_panel(report, folder, setup_copy) + '<h3>' + bi("Sky model & map likelihood", "天空模型与 map 似然") + '</h3>' + content["methods"]
        content["recovery"] += '<h3>' + bi("Does the inferred sky explain the observations?", "推断出的天空是否解释了观测？") + '</h3>' + content["diagnostics"]
        content["diagnostics"] = diagnostics_panel(report, folder, setup_copy)
        content["diagnostics"] += '<h3>' + bi("Shared zero-level priors in physical units", "物理单位下的共同零点先验") + '</h3>'
        content["diagnostics"] += equation(r'z_{600},z_{820}\sim\mathcal N(0,1),\qquad\delta_{600}=0.066z_{600}\ {\rm K},\quad\delta_{820}=\begin{cases}0.300z_{820}\ {\rm K}&z_{820}<0\\0.430z_{820}\ {\rm K}&z_{820}\geq0\end{cases}')
        content["diagnostics"] += paragraph(
            "Each correction is shared by all samples of its frequency, with equal probability on either side of zero. A positive δ corrects the observed sky upward; predicted archive temperatures subtract δ. The 820-MHz published asymmetric range already includes astrophysical constraints. No alternative-prior fit or prior-bound sensitivity study was performed for this case.",
            "每个修正由对应频率的全部观测共享，零点两侧概率各半。正 δ 将观测天空温度向上修正，因此预测档案温度时减去 δ。820 MHz 公布的非对称范围已包含天体物理约束。本案例未运行替代先验拟合，也未进行先验边界敏感性研究。")
        content["sampling"] = sampling_panel(report, folder, setup_copy, show_prediction=False) + content["sampling"]


    text = '<article class="case methodology tris-case" id="tris_haslam" data-case="tris_haslam" data-case-kind="real_observations">'
    text += '<header class="case-heading"><span class="status-tag">' + bi("Real observations · case study", "真实观测 · 科学案例") + '</span><h1>TRIS × Haslam</h1><p class="question">' + bi("What do two radio-sky maps constrain about a spectral sky?", "两幅射电天空图能约束怎样的频率结构？") + '</p>'
    text += '<div class="tris-verdicts"><span>' + bi("Sampling: converged" if converged else "Sampling: diagnostics failed", "采样：收敛" if converged else "采样：诊断未通过") + '</span><span class="tris-mismatch">' + bi("Model: residual mismatch" if report["model_adequacy"] == "mismatch" else "Model: no PPC flag", "模型：残差失配" if report["model_adequacy"] == "mismatch" else "模型：PPC 未标记异常") + '</span></div></header>'
    text += '<div class="case-controls design-controls"><nav class="steps design-tabs" aria-label="TRIS chapters">'
    for i, (key, title) in enumerate(STAGES):
        text += f'<button type="button" class="step-button" data-step="{key}" aria-controls="tris_haslam-{key}" aria-pressed="false"><b>{i+1}</b>{bi(*title)}</button>'
    text += '</nav><nav class="language-switch" aria-label="Language / 语言"><button type="button" data-language-choice="en">English</button><button type="button" data-language-choice="zh">中文</button></nav></div>'
    for key, title in STAGES:
        text += f'<section class="stage" id="tris_haslam-{key}" data-stage="{key}"><h2>{bi(*title)}</h2>{content[key]}</section>'
    text += '<div class="step-footer"><button type="button" class="previous">' + bi("Previous", "上一步") + '</button><span class="step-position" aria-live="polite"></span><button type="button" class="next">' + bi("Next", "下一步") + '</button></div>'
    text += '<footer class="artifacts"><span>' + bi("Data, code & numerical evidence", "数据、代码与数值证据") + '</span>'
    for filename, label in [("result.json", "Results"), ("manifest.json", "Data provenance"), ("maps.npz", "Maps & covariance"), ("posterior.npz", "Posterior"), ("task.artifact.json", "Task"), ("analysis.artifact.json", "Analysis"), ("plan.txt", "Execution plan"), ("dag.mmd", "DAG")]:
        text += f'<a href="{quote(folder, safe="")}/{filename}">{label}</a>'
    text += f'<a href="{report["data_manifest"]["archive_url"]}">LAMBDA TRIS</a><a href="https://arxiv.org/abs/0806.1415">TRIS I</a></footer></article>'
    return text


STYLE = """
.sidebar-observations {margin:26px 0;border-top:1px solid var(--line);padding-top:20px}
.sidebar-observations h2 {font-size:12px;text-transform:uppercase;letter-spacing:.08em;margin:0 12px 10px;color:#526777}
.tris-case .case-heading {border-top:4px solid #176b78}
.tris-verdicts {display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}
.tris-verdicts span {background:#e3f1ed;color:#145c4a;padding:7px 12px;border-radius:5px;font-size:13px;font-weight:650}
.tris-verdicts .tris-mismatch {background:#fff0df;color:#89430e}
.tris-metrics {display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}
.tris-metrics>div {padding:18px;background:#f1f6f7;border:1px solid #dce7e9;border-radius:6px}
.tris-metrics strong {display:block;font-size:24px;color:#176b78;font-variant-numeric:tabular-nums}
.tris-metrics span {display:block;font-size:12px;color:#536772}
.tris-warning {border-left:4px solid #c57432;background:#fff6e9;padding:8px 22px;margin:18px 0}
.tris-equation {background:#f4f8f9;padding:16px;overflow:auto;border-radius:6px;margin:20px 0}
.tris-figure {margin:26px 0;background:white;border:1px solid #e0e8ec;border-radius:7px;padding:14px}
.tris-figure img {width:100%;height:auto;display:block}
.tris-figure figcaption {font-size:12px;color:#5b6d78;margin-top:12px}
.tris-frequency-controls {padding:14px 18px;border:1px solid #dce7e9;background:#f4f8f9;border-radius:6px}
.tris-frequency-controls select {margin-left:12px;padding:6px 12px;border:1px solid #9bb0ba;border-radius:4px;background:white}
.tris-dag {overflow-x:auto;max-width:100%;margin:24px 0}
@media(max-width:700px){.tris-metrics{grid-template-columns:repeat(2,1fr)}.tris-metrics strong{font-size:20px}.tris-figure{padding:3px}.tris-frequency-controls select{margin-left:0}}
"""

SCRIPT = """
(() => {
  const select = document.querySelector('[data-tris-frequency]');
  if (!select) return;
  const update = () => document.querySelectorAll('[data-tris-map]').forEach(panel => {
    panel.hidden = panel.dataset.trisMap !== select.value;
  });
  select.addEventListener('change', update);
  update();
})();
"""
