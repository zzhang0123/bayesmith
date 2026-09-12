"""Bilingual, artifact-only presentation for the TRIS + Haslam + RSB case."""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import quote

import numpy as np

if __package__:
    from .methodology_guide import bi
    from .methodology_guide import math as equation
    from .tris_rsb_data import survey_covariance
else:
    from methodology_guide import bi
    from methodology_guide import math as equation
    from tris_rsb_data import survey_covariance

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
        return {"passed": False, "rhat": float("nan"), "ess": float("nan"), "divergences": 0}
    sites = diagnostics["sites"].values()
    return {
        "passed": bool(diagnostics["passed"]),
        "rhat": max(float(site["r_hat"]) for site in sites),
        "ess": min(float(site["ess"]) for site in sites),
        "divergences": int(diagnostics.get("divergences", 0)),
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
    status_reason = report.get("comparison_status")
    if available:
        finding = bi(
            "Directional held-out prediction is available from the saved refits.",
            "已从保存的留出重拟合中得到定向后验预测。",
        )
    elif status_reason == "blocked_nonconverged_chains":
        finding = bi(
            "Both full-data chains failed the registered convergence checks. Parameter intervals and cross-survey scores are deliberately withheld.",
            "两条全数据链均未通过注册的收敛检查；因此刻意不展示参数区间或跨调查评分。",
        )
    else:
        finding = bi(
            "Full-data chains passed, but the directional held-out refits and scores have not been produced. Results are withheld.",
            "全数据链虽已通过，但尚未生成定向留出重拟合及评分；结果暂不展示。",
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
        "The likelihood keeps the published diagonal uncertainty and adds the declared shared calibration correlation only within each survey.",
        "似然保留公布的对角不确定度，并只在同一调查内部加入已声明的共享校准相关性。",
    )
    content["diagnostics"] += _paragraph(
        "A draw has zero posterior density when its recalibrated 408-MHz Galactic template is not strictly positive. This is a physical-support rule, not a clipping step.",
        "若某个重校准后的 408 MHz 银河模板并非严格为正，该样本的后验密度为零。这是物理支持域规则，并不是截断操作。",
    )

    diagnostic_rows = "".join(
        f"<tr><th scope=\"row\">{label}</th><td>{'PASS' if summary['passed'] else 'FAIL'}</td><td>{summary['rhat']:.3g}</td><td>{summary['ess']:.3g}</td><td>{summary['divergences']}</td></tr>"
        for label, summary in (("M0 · no RSB", m0), ("M1 · with RSB", m1))
    )
    content["sampling"] = _paragraph(
        "Both variants use the same full data, two sequential NUTS chains, and the registered R-hat, ESS, and divergence gates. A completed draw budget is not treated as convergence.",
        "两个变体使用相同的全数据、两条顺序 NUTS 链，以及注册的 R-hat、ESS 和 divergence 门槛。完成预设抽样数不等于收敛。",
    )
    content["sampling"] += (
        '<div class="table-wrap"><table><thead><tr><th>Variant</th><th>Gate</th><th>worst R-hat</th><th>minimum ESS</th><th>divergences</th></tr></thead><tbody>'
        + diagnostic_rows + "</tbody></table></div>"
    )
    content["sampling"] += _paragraph(
        "The saved artifacts retain the method, planned blocks, initial values, and numerical diagnostics so the failure can be investigated rather than silently averaged away.",
        "保存的产物保留方法、计划的参数块、初值和数值诊断，因此该失败可以被追查，而不会被悄然平均掉。",
    )

    content["recovery"] = '<div class="tris-warning rsb-warning"><p>' + finding + "</p></div>"
    content["recovery"] += _paragraph(
        "The current output is a useful model-and-data integration result: it reveals that the present parameterization and sampler configuration are not yet adequate for a quantitative RSB conclusion. No physical origin is inferred from this failed diagnostic state.",
        "当前输出仍是有价值的模型—数据集成结果：它表明现有参数化和采样器配置尚不足以支持定量的 RSB 结论。不会从这一失败的诊断状态推断任何物理起源。",
    )
    content["recovery"] += _paragraph(
        report["comparison_policy"], report["comparison_policy"],
    )
    content["recovery"] += _paragraph(report.get("note", ""), report.get("note", ""))

    text = (
        '<article class="case methodology tris-case rsb-case" id="tris_haslam_rsb_comparison" '
        'data-case="tris_haslam_rsb_comparison" data-case-kind="real_observations">'
        '<header class="case-heading"><span class="status-tag">'
        + bi("Real observations · joint analysis", "真实观测 · 联合分析")
        + '</span><h1>TRIS + Haslam + RSB</h1><p class="question">'
        + bi("Does an isotropic spectrum improve cross-survey prediction?", "各向同性谱是否改善跨调查预测？")
        + '</p><div class="tris-verdicts"><span class="rsb-fail">'
        + bi("Sampling: diagnostics failed", "采样：诊断未通过")
        + '</span><span class="tris-mismatch">'
        + bi("Comparison: withheld", "比较：暂不展示")
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
.rsb-case .tris-verdicts .tris-mismatch {background:#fff0df;color:#89430e}
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
