"""Bilingual, artifact-only presentation for the TRIS + Haslam + RSB case."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

if __package__:
    from .methodology_guide import bi
    from .methodology_guide import math as equation
    from .tris_rsb_data import survey_covariance
    from .tris_rsb_figures import render_review_figures
else:
    from methodology_guide import bi
    from methodology_guide import math as equation
    from tris_rsb_data import survey_covariance
    from tris_rsb_figures import render_review_figures

STAGES = (
    ("model", ("Data & model", "数据与模型")),
    ("methods", ("Methods & parameter blocks", "方法与参数块")),
    ("diagnostics", ("Diagnostics & priors", "诊断与先验")),
    ("sampling", ("Posterior sampling", "后验采样")),
    ("recovery", ("Findings", "发现与解释")),
)


def _esc(value):
    return html.escape(str(value), quote=True)


def _paragraph(en, zh):
    return f"<p>{bi(en, zh)}</p>"


def _figure(folder, name, en, zh):
    stem = f"{quote(folder, safe='')}/{name}"
    return (
        f'<figure class="tris-figure rsb-figure"><img loading="lazy" src="{stem}.png" '
        f'alt="{_esc(en)}"><figcaption>{bi(en, zh)} '
        f'<a href="{stem}.svg">SVG</a></figcaption></figure>'
    )


def _diagnostic_summary(report, variant):
    diagnostics = report["models"].get(variant, {}).get("checks", {}).get("chain_diagnostics")
    if not diagnostics or not diagnostics.get("sites"):
        return {"passed": False, "rhat": float("nan"), "ess": float("nan"), "divergences": "—"}
    sites = diagnostics["sites"].values()
    return {
        "passed": bool(diagnostics["passed"]),
        "rhat": max(float(site["r_hat"]) if site.get("r_hat") is not None else float("inf") for site in sites),
        "ess": min(float(site["ess"]) if site.get("ess") is not None else 0. for site in sites),
        "divergences": diagnostics.get("divergences", "—"),
    }


def render_figures(report, directory):
    """Render data/provenance figures only; never derive posterior conclusions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    source = directory.parent / "tris_haslam_rsb" / "external.npz"
    if not source.exists():
        source = directory.parent / "tris_haslam_no_rsb" / "external.npz"
    if not source.exists():
        return
    with np.load(source, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    colors = {"LWA": "#176b78", "ARCADE": "#c57432"}
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), layout="constrained")
    for ax, survey in zip(axes, ("LWA", "ARCADE"), strict=True):
        mask = np.asarray(data["survey"]) == survey
        frequency = np.asarray(data["frequency_mhz"])[mask]
        temperature = np.asarray(data["temperature_rj_k"])[mask]
        sigma = np.sqrt(np.asarray(data["sigma_independent_rj_k"])[mask] ** 2 + np.asarray(data["tau_rj_k"])[mask] ** 2)
        ax.errorbar(frequency, temperature, yerr=sigma, fmt="o", color=colors[survey],
                    capsize=3, lw=1.2, label=f"{survey} published rows")
        ax.set(xscale="log", xlabel="frequency [MHz]", ylabel="RJ temperature [K]",
               title="LWA1 low-frequency background" if survey == "LWA" else "ARCADE 2 high-frequency background")
        ax.grid(alpha=.24, which="both")
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("External background data retained by the joint likelihood", fontsize=14)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"tris-rsb-data.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)

    render_review_figures(report, directory)

    covariance = survey_covariance(
        data["sigma_independent_rj_k"], data["survey_code"], data["tau_rj_k"]
    )
    scale = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scale, scale)
    fig, ax = plt.subplots(figsize=(6.2, 5.2), layout="constrained")
    image = ax.imshow(correlation, cmap="Blues", vmin=0, vmax=1)
    labels = [f"{name} {frequency:g}" for name, frequency in zip(data["survey"], data["frequency_mhz"], strict=True)]
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_title("Declared external-data correlation\n(shared calibration within each survey)")
    fig.colorbar(image, ax=ax, label="correlation", shrink=.82)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"tris-rsb-covariance.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def render_case(report, folder, index=0):
    """Render the comparison evidence using the common five-step notebook contract."""
    available = report.get("comparison_status") == "available"
    m0 = _diagnostic_summary(report, "no_rsb")
    m1 = _diagnostic_summary(report, "rsb")
    converged = m0["passed"] and m1["passed"]
    status_reason = report.get("comparison_status")
    if available:
        finding = bi(
            "Directional held-out prediction is available from the saved refits.",
            "已从保存的留出重拟合中得到定向后验预测。",
        )
    elif status_reason == "blocked_nonconverged_chains":
        finding = bi(
            "At least one full-data analysis failed the registered convergence checks. Its parameter intervals and cross-survey scores are withheld.",
            "至少一组全数据分析未通过注册的收敛检查；其参数区间与跨调查评分暂不展示。",
        )
    else:
        finding = bi(
            "Directional held-out refits are missing or have not passed their diagnostics. No comparative score is reported.",
            "定向留出重拟合缺失或尚未通过诊断；因此不报告比较评分。",
        )

    content = {}
    content["model"] = _paragraph(
        "The common likelihood joins the two saved TRIS map responses to eleven published radio-background rows: five LWA1 rows at 40–80 MHz and six ARCADE 2 rows at 3.20–10.49 GHz. The 408-MHz literature row is excluded so Haslam is not counted twice.",
        "共同似然把两幅已保存的 TRIS map 响应与 11 个已发表的射电背景数据点联合：LWA1 在 40–80 MHz 的 5 个点，以及 ARCADE 2 在 3.20–10.49 GHz 的 6 个点。408 MHz 文献点被排除，避免重复计入 Haslam。",
    )
    content["model"] += '<div class="tris-metrics rsb-metrics">' + "".join(
        f"<div><strong>{value}</strong><span>{bi(en, zh)}</span></div>"
        for value, en, zh in (
            ("600.5 / 817.8", "TRIS map frequencies · MHz", "TRIS map 频率 · MHz"),
            ("11", "external background rows", "外部背景数据行"),
            ("2", "survey-wide calibration terms", "调查级校准项"),
            ("M0 / M1", "shared-data comparison", "共用数据的模型比较"),
        )
    ) + "</div>"
    content["model"] += _figure(
        folder, "tris-rsb-data",
        "Each point retains its converted RJ uncertainty; the plot is a data display, not a fitted excess spectrum.",
        "每个点保留转换后的 RJ 不确定度；此图仅展示数据，并非拟合出的超额谱。",
    )

    content["methods"] = _paragraph(
        "M0 fits the TRIS–Haslam sky without an isotropic excess. M1 adds a positive radio-synchrotron-background amplitude at 1 GHz and a spectral index, while retaining the regional Haslam amplitudes and indices, two TRIS zero corrections, the Haslam monopole correction, and one calibration standard per survey.",
        "M0 在没有各向同性超额的条件下拟合 TRIS–Haslam 天空。M1 在此基础上加入 1 GHz 处为正的射电同步背景振幅及谱指数，同时保留分区 Haslam 振幅与谱指数、两个 TRIS 零点修正、Haslam 单极修正，以及每个调查的一个校准标准量。",
    )
    content["methods"] += equation(
        r"T_{\rm ext}(\nu)=T_{\rm CMB,RJ}(\nu)+A_{\rm RSB}(\nu/1\,{\rm GHz})^{\beta_{\rm RSB}}+c_{s(\nu)}\tau_{s(\nu)}"
    )
    content["methods"] += equation(r"G_{408}(p)=H_{408}(p)-T_{\rm CMB,RJ}(408)+z_H-B(408)>0")
    content["methods"] += equation(r"T_\nu(p)=a_{r(p)}G_{408}(p)(\nu/408\,{\rm MHz})^{\beta_{r(p)}}+T_{\rm CMB,RJ}(\nu)+B(\nu)")
    content["methods"] += _paragraph(
        "Three bounded data fits start inside the positive-template domain. Independent curvature-scaled perturbations seed the chains. NUTS learns a full mass matrix; budgets and target acceptance are recorded below. This changes initialization and integration geometry only; the joint density, prior bounds, data and uncertainty scales are retained.",
        "先从三个分散起点进行有界数据拟合，在正模板支持域内构造各链的独立扰动初值。NUTS 学习完整质量矩阵；下方记录预算与目标接受率。修改的是初值和积分几何；联合密度、先验边界、数据与误差尺度均保留。",
    )
    content["methods"] += _paragraph(
        "The final comparison is directional held-out posterior prediction: train with TRIS plus one external survey and score the other. It does not use a Bayes factor or two-survey PSIS-LOO.",
        "最终比较是定向留出后验预测：用 TRIS 加一个外部调查训练，再预测另一个调查。它不使用 Bayes factor 或双调查 PSIS-LOO。",
    )

    content["diagnostics"] = _paragraph(
        "RSB amplitude has Uniform(0, 5 K) at 1 GHz and its spectral index has Uniform(−4, −1.5). The Haslam monopole has Normal(0, 3 K); the existing three-region sky and TRIS zero-level priors are unchanged.",
        "RSB 振幅在 1 GHz 处服从 Uniform(0, 5 K)，谱指数服从 Uniform(−4, −1.5)。Haslam 单极项服从 Normal(0, 3 K)；已有的三区域天空先验和 TRIS 零点先验保持不变。",
    )
    content["diagnostics"] += _figure(
        folder, "tris-rsb-covariance",
        "Assumed covariance with the published diagonal retained; it is not the surveys' measured full covariance.",
        "此为保留公布对角误差的假设协方差，并非调查实测的完整协方差。",
    )
    content["diagnostics"] += _paragraph(
        "A draw has zero posterior density when its recalibrated 408-MHz Galactic template is not strictly positive. This is a physical-support rule, not a clipping step.",
        "若某个重校准后的 408 MHz 银河模板并非严格为正，该样本的后验密度为零。这是物理支持域规则，并不是截断操作。",
    )
    content["diagnostics"] += _paragraph("This support constraint couples the effective RSB and Haslam-monopole priors: the accepted prior is not a product of independently normalized conditional truncations.", "此支持域约束使有效的 RSB 与 Haslam 单极项先验耦合；接受后的先验并不是各自归一化条件截断分布的乘积。")
    content["diagnostics"] += _paragraph(
        'LWA Table 2 errors include scatter across five foreground-removal / sightline estimates. Their shared foreground information is not recovered by a 10-K common offset. ARCADE Table 3 identifies a 1-mK thermometer-calibration component; the retained 5-mK common scale is an analysis assumption, not that published component. The summary likelihood also assumes zero cross-survey covariance; it has not been validated by a covariance sensitivity study.',
        'LWA 表 2 的误差包含五种前景扣除／视线估计之间的散布；10 K 的共同偏移无法恢复它们共享的前景信息。ARCADE 表 3 列出了 1 mK 温度计校准分量；这里保留的 5 mK 共同尺度是分析假设，不是该文献分量。摘要似然还假设调查间协方差为零，尚未通过协方差敏感性研究验证。',
    )
    content["diagnostics"] += '<p><a href="https://arxiv.org/abs/1804.08581">Dowell &amp; Taylor · Table 2 / §3</a> · <a href="https://arxiv.org/abs/0901.0555">Fixsen et al. · Tables 3–4 / §5.1</a></p>'

    diagnostic_rows = "".join(
        f"<tr><th scope=\"row\">{label}</th><td>{'PASS' if summary['passed'] else 'FAIL'}</td><td>{summary['rhat']:.5g}</td><td>{summary['ess']:.3g}</td><td>{summary['divergences']}</td></tr>"
        for label, summary in (("M0 · no RSB", m0), ("M1 · with RSB", m1))
    )
    content["sampling"] = _paragraph(
        "The table reads the saved full-data diagnostics. The registered requirements are R-hat ≤ 1.01, ESS ≥ 400 and zero divergences, alongside the package's site checks. Actual budgets are listed for each run.",
        "表中读取已保存的全数据诊断。要求 R-hat ≤ 1.01、ESS ≥ 400、零发散，并通过软件包的站点检查。各次运行的实际预算列于下方。",
    )
    content["sampling"] += (
        '<div class="table-wrap"><table><thead><tr><th>Variant</th><th>Gate</th><th>worst R-hat</th><th>minimum ESS</th><th>divergences</th></tr></thead><tbody>'
        + diagnostic_rows + "</tbody></table></div>"
    )
    setup_rows = ""
    for variant, label in (("no_rsb", "M0"), ("rsb", "M1")):
        saved = report["models"].get(variant, {})
        run = saved.get("execution") or {}
        shape = saved.get("chain_shape")
        setup_rows += '<tr><td>' + label + '</td><td>' + (" × ".join(map(str, shape)) if shape else "—") + '</td><td>' + _esc(run.get("budget", {}).get("warmup", "—")) + '</td><td>' + _esc(run.get("nuts_options", {})) + '</td></tr>'
    content["sampling"] += '<div class="table-wrap"><table><thead><tr><th>Run</th><th>chains × draws</th><th>warmup / chain</th><th>NUTS</th></tr></thead><tbody>' + setup_rows + '</tbody></table></div>'
    content["sampling"] += _paragraph(
        "The saved artifacts retain the method, planned blocks, initial values, and numerical diagnostics so the sampling behavior and its repair can be audited.",
        "保存的产物保留方法、计划的参数块、初值和数值诊断，因此可以追查采样表现和修复过程。",
    )
    content["sampling"] += _figure(folder, "tris-rsb-traces", "Chain traces retain separate colors; tiny RSB amplitude is a conditional model compromise.", "各条链分别着色；极小的 RSB 振幅是给定模型下的拟合折中。")
    if report.get("review_baseline"):
        rows = ""
        for variant, label in (("no_rsb", "M0"), ("rsb", "M1")):
            before = _diagnostic_summary({"models": report["review_baseline"]}, variant)
            after = _diagnostic_summary(report, variant)
            for stage, summary in (("Before", before), ("After", after)):
                rows += f'<tr><td>{label} · {stage}</td><td>{summary["rhat"]:.5g}</td><td>{summary["ess"]:.5g}</td><td>{summary["divergences"]}</td></tr>'
        content["sampling"] += '<h3>' + bi("Before and after repair", "修复前后") + '</h3><div class="table-wrap"><table><thead><tr><th>Run</th><th>max R-hat</th><th>min ESS</th><th>divergences</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
        distances = report["review_baseline"]["rsb"].get("median_min_template_K_by_chain")
        if distances:
            content["sampling"] += _paragraph(
                f"The original M1 chains had median minimum Galactic temperatures {', '.join(f'{value:.3g}' for value in distances)} K. One chain was trapped at the physical boundary. The repaired chains remain well inside the domain. M0's slow mixing was a separate correlated-geometry problem.",
                f"原 M1 各链的最小银河模板温度中位数分别为 {', '.join(f'{value:.3g}' for value in distances)} K，其中一条链被困在物理边界附近。修复后的链位于支持域内部。M0 的慢混合则来自相关参数的采样几何。",
            )

    content["recovery"] = '<div class="tris-warning rsb-warning"><p>' + finding + "</p></div>"
    content["recovery"] += _paragraph(
        "The review identifies two different failures: prior starts can trap a chain at the positive-template boundary; the rigid three-region foreground also leaves residuals far above the supplied statistical noise. Convergence repairs the first, not the second. A near-zero fitted RSB is therefore not evidence that the observed background excess is absent.",
        "此次审查识别出两种问题：先验初值可能把链困在正模板边界；受限的三区域前景还留下远超统计噪声的残差。收敛修复解决前者，后者仍存在。因此，拟合出的 RSB 接近零不能解释成观测背景超额不存在。",
    )
    content["recovery"] += _paragraph(
        report["comparison_policy"], "比较对象是给定协方差假设下的已发表前景扣除摘要；共享文献输入使其不构成独立仪器验证。不计算 Bayes factor 或双调查 PSIS-LOO。",
    )
    content["recovery"] += _paragraph(report.get("note", ""), "采样收敛和预测评分不能验证三区域前景模型，也不能确定 RSB 的物理起源。")

    if converged and all(model.get("frequencies") for model in report["models"].values()):
        content["recovery"] += _figure(folder, "tris-rsb-conflict", "Conditional excess spectrum and correlated calibration shifts expose the tension.", "条件背景谱与相关的校准修正展示数据—模型冲突。")
        content["recovery"] += _figure(folder, "tris-rsb-residuals", "M0 and M1 use the same axes; the ±3σ strip tests supplied statistical noise.", "M0、M1 使用相同坐标轴；±3σ 阴影对应输入统计噪声。")
        rows = ""
        for variant, label in (("no_rsb", "M0"), ("rsb", "M1")):
            saved = report["models"][variant]
            for row in saved["frequencies"]:
                rows += f'<tr><td>{label}</td><td>{row["frequency_mhz"]:g}</td><td>{row["residual_rms_k"]:.3f}</td><td>{row["chi_square_per_observation"]:.1f}</td><td>{row["ppc_tail_probability"]:g} (N={row["ppc_replicates"]})</td></tr>'
        content["recovery"] += '<div class="table-wrap"><table><thead><tr><th>Model</th><th>MHz</th><th>RMS [K]</th><th>χ²/N</th><th>PPC tail fraction</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
        content["recovery"] += _paragraph("χ²/N uses the number of observations, not fitted degrees of freedom. Zero PPC exceedances is a finite Monte Carlo observation, not an exactly zero probability.", "χ²/N 以观测数为分母，未扣除拟合自由度。PPC 零次超越是有限 Monte Carlo 结果，并非概率严格为零。")
        for variant, label in (("no_rsb", "M0"), ("rsb", "M1")):
            saved = report["models"][variant]
            content["recovery"] += '<h3>' + label + ' · a, β, recalibrated Haslam</h3>'
            content["recovery"] += _figure(folder, f"tris-rsb-fields-{variant}", "Means and standard deviations on the equatorial HEALPix grid. Regional extrapolation; model discrepancy is excluded.", "赤道 HEALPix 网格上的均值与标准差。均为分区模型外推，不包含模型失配误差。")
            content["recovery"] += '<details><summary>' + bi("Conditional parameter intervals and prior departures", "条件参数区间与先验偏离") + '</summary><div class="table-wrap"><table><thead><tr><th>Parameter</th><th>Mean</th><th>95% interval</th></tr></thead><tbody>'
            for row in saved["parameters"]:
                content["recovery"] += f'<tr><td>{_esc(row["name"])}</td><td>{row["mean"]:.5g}</td><td>[{row["lower"]:.5g}, {row["upper"]:.5g}]</td></tr>'
            content["recovery"] += '</tbody></table></div><pre>' + _esc(json.dumps({key: saved[key] for key in ("support", "prior_departures")}, indent=2)) + '</pre></details>'
        content["recovery"] += equation(r"H_{408}^{\rm recal}(p)=a_{r(p)}[H_{408}(p)-T_{\rm CMB,RJ}(408)+z_H-B(408)]+T_{\rm CMB,RJ}(408)+B(408)")
    if available:
        rows = ""
        for fold in report.get("heldout_scores", []):
            rows += f'<tr><td>{fold["train"]} → {fold["heldout"]}</td><td>{fold["models"]["no_rsb"]["log_predictive_density"]:.3f}</td><td>{fold["models"]["rsb"]["log_predictive_density"]:.3f}</td><td>{fold["delta_M1_minus_M0"]:.4g} ± {fold["delta_mcse"]:.2g}</td></tr>'
        content["recovery"] += '<h3>' + bi("Held-out summary prediction", "留出摘要预测") + '</h3><div class="table-wrap"><table><thead><tr><th>Train → held-out</th><th>M0 log p</th><th>M1 log p</th><th>Δ log p ± MCSE</th></tr></thead><tbody>' + rows + '</tbody></table></div>'
        content["recovery"] += _paragraph("Positive Δ favors M1's conditional predictive density. The unseen survey offset is integrated analytically; uncertainty is a chain/batch Monte Carlo error, not uncertainty across independent surveys. Model inadequacy remains regardless of a small predictive improvement.", "正的 Δ 表示 M1 的条件预测密度更高。未见调查的偏移被解析积分；误差来自按链／批计算的 Monte Carlo 误差，不代表独立调查之间的变异。小幅预测改善也不消除模型失配。")

    for variant, label in (("no_rsb", "M0"), ("rsb", "M1")):
        saved = report["models"].get(variant, {})
        if saved.get("blocking"):
            if __package__:
                from .case_panels import diagnostics_panel, methods_panel
            else:
                from case_panels import diagnostics_panel, methods_panel
            copy = {"structure": ("Joint NUTS on the saved sky, calibration and background coordinates.", "对已保存的天空、校准及背景参数执行联合 NUTS 更新。")}
            content["methods"] += '<h3>' + label + '</h3>' + methods_panel(saved, f"tris_haslam_{variant}", copy)
            content["diagnostics"] += '<h3>' + label + '</h3>' + diagnostics_panel(saved, f"tris_haslam_{variant}", copy)
            content["sampling"] += '<details><summary>' + label + ' · ' + bi("Initialization and run record", "初始化与运行记录") + '</summary><pre>' + _esc(json.dumps(saved["execution"], indent=2)) + '</pre></details>'

    text = (
        '<article class="case methodology tris-case rsb-case" id="tris_haslam_rsb_comparison" '
        'data-case="tris_haslam_rsb_comparison" data-case-kind="real_observations">'
        '<header class="case-heading"><span class="status-tag">'
        + bi("Real observations · joint analysis", "真实观测 · 联合分析")
        + '</span><h1>TRIS + Haslam + RSB</h1><p class="question">'
        + bi("Does an isotropic spectrum improve cross-survey prediction?", "各向同性谱是否改善跨调查预测？")
        + '</p><div class="tris-verdicts"><span class="' + ('rsb-pass' if converged else 'rsb-fail') + '">'
        + (bi("Sampling: diagnostics passed", "采样：诊断通过") if converged else bi("Sampling: diagnostics failed", "采样：诊断未通过"))
        + '</span><span class="tris-mismatch">'
        + (bi("TRIS: model mismatch", "TRIS：模型失配") if converged else bi("Comparison: withheld", "比较：暂不展示"))
        + "</span></div></header>"
    )
    text += '<div class="case-controls design-controls"><nav class="steps design-tabs" aria-label="RSB chapters">'
    for number, (key, title) in enumerate(STAGES, 1):
        text += (
            f'<button type="button" class="step-button" data-step="{key}" '
            f'aria-controls="tris_haslam_rsb_comparison-{key}" aria-pressed="false">'
            f"<b>{number}</b>{bi(*title)}</button>"
        )
    text += '</nav><nav class="language-switch" aria-label="Language / 语言"><button type="button" data-language-choice="en">English</button><button type="button" data-language-choice="zh">中文</button></nav></div>'
    for key, title in STAGES:
        text += f'<section class="stage" id="tris_haslam_rsb_comparison-{key}" data-stage="{key}"><h2>{bi(*title)}</h2>{content[key]}</section>'
    text += '<div class="step-footer"><button type="button" class="previous">' + bi("Previous", "上一步") + '</button><span class="step-position" aria-live="polite"></span><button type="button" class="next">' + bi("Next", "下一步") + '</button></div>'
    text += '<footer class="artifacts"><span>' + bi("Data, code & numerical evidence", "数据、代码与数值证据") + '</span>'
    for filename, label in (("result.json", "Comparison artifact"), ("../tris_haslam_no_rsb/result.json", "M0 diagnostics"), ("../tris_haslam_rsb/result.json", "M1 diagnostics")):
        text += f'<a href="{quote(folder, safe="")}/{filename}">{label}</a>'
    text += '<span>Input fingerprint: ' + _esc(report["data_manifest"]["common_input_sha256"]) + "</span></footer></article>"
    return text


STYLE = """
.rsb-case .case-heading {border-top-color:#934e20}
.rsb-case .tris-verdicts .rsb-fail {background:#fbe8e6;color:#9b3029}
.rsb-case .tris-verdicts .rsb-pass {background:#e7f3ef;color:#20624b}
.rsb-case .tris-verdicts .tris-mismatch {background:#fff0df;color:#89430e}
.rsb-case .tris-verdicts span span {background:transparent;color:inherit;padding:0}
.rsb-case .design-tabs {border-bottom-color:#ead9cf}
.rsb-case .design-tabs button[aria-pressed=\"true\"] {border-bottom-color:#934e20;color:#7c3514}
.rsb-metrics>div {background:#fbf5f0;border-color:#edddd2}
.rsb-metrics strong {color:#934e20}
.rsb-figure {max-width:1000px;margin-left:auto;margin-right:auto}
.rsb-warning {border-left-color:#a33a2d;background:#fff0ee}
.rsb-case .table-wrap {overflow:auto;border:1px solid #ead9cf;border-radius:7px}
.rsb-case table {margin:0;min-width:620px}
.rsb-case table th {background:#fbf5f0}
"""

SCRIPT = """// The common gallery script owns RSB step navigation and language switching.\n"""
