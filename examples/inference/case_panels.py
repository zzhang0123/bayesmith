"""Bilingual walkthrough panels built from saved numerical evidence."""

from __future__ import annotations

import html
import json
import re
from urllib.parse import quote

if __package__:
    from .case_content import CASES, LEGACY_CASES
    from .methodology_guide import METHODS, bi, detail, math, paragraph
    from .symbols import inline_symbol, parameter_symbols
else:
    from case_content import CASES, LEGACY_CASES
    from methodology_guide import METHODS, bi, detail, math, paragraph
    from symbols import inline_symbol, parameter_symbols

STAGES = [
    ("model", ("Model & simulation", "模型与模拟")),
    ("methods", ("Methods and blocks", "方法与参数块")),
    ("diagnostics", ("Diagnostics & priors", "诊断与先验")),
    ("sampling", ("Sampling", "采样")),
    ("recovery", ("Recovery", "恢复结果")),
]
METHOD_NUMBERS = {
    "gcr": (1,),
    "log-gcr": (2,),
    "nuts": (3,),
    "gcr+mh": (1, 5, 8),
    "gcr+snis": (1, 5, 8),
    "iterative_gls+mh": (5, 8),
    "bias_corrected_log_linear+mh": (6, 8),
    "gauss_newton+mh": (7, 8),
    "iterative_gls+uncorrected": (5,),
    "bias_corrected_log_linear+uncorrected": (6,),
    "gauss_newton+uncorrected": (7,),
}
METHOD_LABELS = {
    "gcr+mh": ("GCR + iterative GLS + MH", "GCR + iterative GLS + MH"),
    "gcr+snis": ("GCR + iterative GLS + SNIS", "GCR + iterative GLS + SNIS"),
    "iterative_gls+mh": ("Iterative GLS + MH", "迭代 GLS + MH"),
    "bias_corrected_log_linear+mh": ("Bias-corrected log-linear + MH", "偏差修正对数线性 + MH"),
    "gauss_newton+mh": ("Gauss–Newton + MH", "Gauss–Newton + MH"),
    "iterative_gls+uncorrected": ("Iterative GLS · MH off", "迭代 GLS · MH 关闭"),
    "bias_corrected_log_linear+uncorrected": ("Bias-corrected log-linear · MH off", "偏差修正对数线性 · MH 关闭"),
    "gauss_newton+uncorrected": ("Gauss–Newton · MH off", "Gauss–Newton · MH 关闭"),
}
STATUS = {
    "passed": ("Passed", "通过"),
    "failed": ("Failed", "未通过"),
    "flat": ("Flat", "平坦"),
    "nonflat": ("Non-flat", "非平坦"),
    "unresolved": ("Unresolved", "未确定"),
    "skipped": ("Skipped", "跳过"),
    "not_applicable": ("Not applicable", "不适用"),
    "rank_deficient": ("Rank deficient", "秩亏"),
    "gcr": ("GCR", "GCR"),
    "nuts": ("NUTS", "NUTS"),
}
FINDINGS = {
    "exact_block": ("GCR structure", "GCR 结构"),
    "sampled_block": ("NUTS block", "NUTS 块"),
    "residual_block": ("Remaining block", "剩余块"),
    "proposal_block": ("Proposal block", "提议块"),
    "automatic_proposals": ("Automatic method selection", "自动方法选择"),
    "latent_treatment": ("Latent-variable workflow", "潜变量处理流程"),
    "prior_policy": ("Prior policy", "先验策略"),
    "initial_point": ("Valid diagnostic anchor", "有效诊断锚点"),
    "diagnostic_budget": ("Diagnostic budget", "诊断预算"),
    "joint_geometry": ("Joint geometry", "联合几何"),
    "sampling_order": ("Update order", "更新顺序"),
    "prior": ("Declared prior", "声明的先验"),
    "jeffreys": ("Jeffreys / flatness", "Jeffreys / 平坦性"),
    "prior_sensitivity": ("Prior sensitivity", "先验敏感度"),
}


def esc(value):
    return html.escape(str(value), quote=True)


def raw(value):
    return (
        '<pre class="evidence-json">'
        + esc(json.dumps(value, indent=2, ensure_ascii=False))
        + "</pre>"
    )


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, (int, float)):
        return f"{value:.4g}"
    return esc(value)


def status(value):
    style = "attention" if value.endswith("+uncorrected") else (
        "good" if value in {"passed", "flat", "nonflat", "gcr", "nuts", *METHOD_LABELS} else "attention"
    )
    label = METHOD_LABELS.get(value, STATUS.get(value, (value, value)))
    return f'<span class="finding-status {style}">{bi(*label)}</span>'


def method_label(value):
    return bi(*METHOD_LABELS.get(value, (value.upper(), value.upper())))


def execution_structure(report, copy):
    if report["case"] == "power_law" and report.get("variant"):
        return (
            "A remains linear in the mean at fixed α, with fixed noise. The joint Jeffreys prior retains A–α cross information and weights the amplitude conditional by A; it is no longer a truncated Gaussian. The recorded plan uses joint NUTS. Taking log y would change the additive-noise model.",
            "固定 α 时，均值仍对 A 线性且噪声固定。联合 Jeffreys 先验保留 A–α 交叉信息，并将振幅条件密度乘以 A，使其不再是截断高斯。记录的方案使用联合 NUTS。对 y 取对数会改变加性噪声模型。",
        )
    if report.get("proposal_selection") == "automatic":
        return (
            "The compiler automatically tests conditional log-linear and linear proposals for parameter-dependent Gaussian noise. Accepted candidates use MH; unsupported candidates remain in NUTS.",
            "编译器针对参数依赖的高斯噪声，自动检查条件对数线性和线性提议。通过检查的候选使用 MH，其余参数保留在 NUTS 中。",
        )
    if report.get("variant"):
        return (
            "The comparison uses its stated model settings and finite parameter domain. The saved plan below records this run’s actual blocks and methods.",
            "对照采用注明的模型设置和有限参数域。下方保存的方案记录本次实际使用的参数块与方法。",
        )
    if report.get("proposal_policies"):
        if any(not p.get("mh_correction", True) for p in report["proposal_policies"]):
            return (
                "Explicit proposal schedule with MH off in at least one block. These are unadjusted approximate updates; the original automatic demo remains separate.",
                "显式提议调度中至少一个块关闭了 MH。这是未经校正的近似更新；原自动选择 demo 单独保留。",
            )
        return (
            "This comparison uses an explicit proposal schedule. The displayed blocks and methods come from its saved task; the original automatic demos remain separate.",
            "这项比较使用显式提议调度。展示的参数块与方法来自已保存的任务，原自动选择 demo 单独保留。",
        )
    return copy["structure"]


def variant_label(report):
    if (report.get("variant") or {}).get("kind") == "observation_count":
        count = report["variant"]["observations"]
        return bi(f"{count:,} observations", f"{count:,} 个观测")
    if (report.get("variant") or {}).get("kind") == "mild_noise_prior":
        return bi("with mild noise prior", "使用温和噪声先验")
    return bi("with Jeffreys prior", "使用 Jeffreys 先验")


def prior_variant_note(report):
    variant = report.get("variant")
    if not variant:
        return ""
    text = '<div class="variant-note"><strong>' + variant_label(report) + '</strong>'
    if variant.get("kind") == "observation_count":
        count, base = variant["observations"], variant["baseline_observations"]
        return text + paragraph(
            f"Increase the input grid from {base:,} to {count:,} observations on [0, 1). The same seed preserves the latent process instance and parameter truths; the observation noise is a new realization at the larger array shape. The original Uniform root priors and sampling budget are retained.",
            f"在 [0, 1) 上将输入网格从 {base:,} 增加到 {count:,} 个观测。同一种子保留潜在过程实例和参数真值；观测噪声是在更大数组形状下生成的新实现。保留原有顶层 Uniform 先验和采样预算。",
        ) + paragraph(
            "This compares posterior precision for one dataset at each size. It does not measure repeated-simulation bias or coverage, and is not an extension that retains the original noisy observations.",
            "这里比较每个数据量下单组数据的后验精度，不衡量重复模拟意义下的偏差或覆盖率；也不是保留原带噪观测后追加数据。",
        ) + detail(("Observation-count setup", "观测数设置"), raw(variant)) + '</div>'
    if variant.get("formula_latex"):
        text += math(variant["formula_latex"])
    else:
        text += '<p>' + esc(variant.get("formula", "")) + '</p>'
    if variant.get("kind") == "mild_noise_prior":
        return text + paragraph(
            "Only σ_w changes: a Normal(0.02, 0.01²) prior truncated to the original [0.003, 0.06] bounds. Its central 95% interval is about [0.0051, 0.0398]. All other priors, simulation data, truth, sampling budget and recovery thresholds match the baseline.",
            "仅改变 σ_w：将 Normal(0.02, 0.01²) 先验截断到原有 [0.003, 0.06] 边界，其中央 95% 区间约为 [0.0051, 0.0398]。其他先验、模拟数据、真值、采样预算和恢复阈值均与原例相同。",
        ) + paragraph(
            "This broad illustrative assumption was chosen after inspecting the baseline failure and fixed before running the comparison. It is not an external calibration; no prior tuning was performed to obtain a PASS.",
            "这个宽松的示例假设在查看原例失败后选定，并在对照运行前固定。它并非外部标定结果；没有为获得通过而调整先验。",
        ) + detail(("Prior scope and construction", "先验范围与构造"), raw(variant)) + '</div>'
    text += paragraph(
        "Jeffreys density is restricted to the original finite bounds. The existing Uniform factors provide constant density and support; the joint prior factor supplies the stated shape. Simulation data and truth are unchanged.",
        "Jeffreys 密度限制在原来的有限边界内。已有 Uniform 因子提供常数密度及支持域，联合先验因子提供注明的形状。模拟数据与真值保持不变。",
    )
    if variant.get("equivalent_to_baseline"):
        text += paragraph(
            "Here the marginal Jeffreys density for the population mean is constant. It gives the same prior as the original bounded Uniform; Gaussian group conditionals are retained.",
            "这里总体均值的边缘 Jeffreys 密度为常数，与原有有界 Uniform 先验相同；组的条件高斯分布予以保留。",
        )
    text += detail(("Prior scope and construction", "先验范围与构造"), raw(variant)) + '</div>'
    return text


def image(folder, name, caption):
    return (
        f'<figure class="evidence-figure"><img src="{quote(folder, safe="")}/{name}.png" '
        f'alt="{esc(caption[0])}" loading="lazy"><figcaption>{bi(*caption)} '
        f'<a href="{quote(folder, safe="")}/{name}.svg" download>SVG</a></figcaption></figure>'
    )


def method_refs(blocks):
    return sorted(
        {n for block in blocks for n in METHOD_NUMBERS.get(block["method"], ())}
    )


def model_panel(report, folder, copy):
    if __package__:
        from .presentation import dag_svg, operator_table
    else:
        from presentation import dag_svg, operator_table
    rows = report["parameters"]
    text = math(copy["formula"])
    if not report.get("variant") or report["case"] == "power_law" or report["variant"].get("kind") == "observation_count":
        text += paragraph(*copy["model"], css="case-lede")
    if report.get("variant"):
        text += prior_variant_note(report)
    text += '<div class="model-facts">'
    for number, title in [
        (len(report["signal"]["data"]), ("Observations", "观测数")),
        (len(rows), ("Parameter coordinates", "参数坐标")),
        (len(report["blocking"]["executed"]["blocks"]), ("Sampled blocks", "采样块")),
    ]:
        text += f"<span><b>{number}</b>{bi(*title)}</span>"
    text += "</div>"
    text += '<div class="parameter-key">'
    for prior in report["blocking"]["priors"]:
        text += '<span>' + inline_symbol(report["case"], prior["latent"]) + f'<code>{esc(prior["latent"])}</code><small>' + bi(f'{prior["coordinates"]} coordinates', f'{prior["coordinates"]} 个坐标') + '</small></span>'
    text += '</div>'
    if report["case"] == "composed_process":
        text += '<div class="generative-story">' + math(r"s\ \longrightarrow\ Rs\ \xrightarrow{+h(x;\theta)+Bb}\ v\ \xrightarrow{\times e^{Ug}}\ \mu\ \xrightarrow{\times(1+f\epsilon)}\ y")
        text += paragraph(
            "The instance s is a random operator: its Gaussian law is conditional on the power amplitude a. Deterministic operators connect it to the random observation y.",
            "实例 s 是随机算子，其高斯分布以功率振幅 a 为条件。确定性算子将它连接到随机观测 y。",
        ) + math(r"p(y,s\mid a,\theta,b,g,\sigma_w)=p(y\mid s,\theta,b,g,\sigma_w)\,p(s\mid a)")
        text += paragraph(
            "A stochastic DAG composes the joint probability from conditional probabilities. In shorthand: p(data, instance) = p(data | instance) p(instance).",
            "随机算子的 DAG 通过条件概率复合整体概率。简写为：p(data, instance) = p(data | instance) p(instance)。",
        ) + '</div>'
    text += detail(
        ("Traced DAG & declared operators", "实际 DAG 与声明的算子"),
        paragraph("Blue nodes are random latent operators; green nodes are random observations. Symbols match the model and parameter blocks. Scroll horizontally to follow the full graph.", "蓝色节点为潜变量随机算子，绿色节点为随机观测。符号与模型和参数块一致；横向滚动可阅读完整图。", css="design-footnote")
        +
        '<div class="dag-scroll">'
        + dag_svg(report["dag"], report["case"])
        + "</div>"
        + operator_table(report["dag"], report["case"]),
    )
    text += "<h3>" + bi("Forward simulation", "正向模拟") + "</h3>"
    text += paragraph(*copy["simulation"])
    if report["signal"]["view"].get("kind") == "process":
        text += image(
            folder,
            "components",
            (
                "Read the generating stages in order; additive panels show contributions, while gain multiplies their sum.",
                "按顺序阅读生成阶段；加性面板展示各项贡献，增益乘在它们的和上。",
            ),
        )
    text += image(
        folder,
        "simulation",
        ("Generating model and simulated observations.", "生成模型与模拟观测。"),
    )
    truths = (
        '<div class="truth-vector">'
        + "".join(
            f"<span><code>{esc(row['name'])}</code><b>{fmt(row['truth'])}</b></span>"
            for row in rows
        )
        + "</div>"
    )
    text += detail(("Generating parameters", "生成参数"), truths)
    text += paragraph(
        "Inference receives the model, priors and observations. Simulation truth is reserved for the recovery check.",
        "推断接收模型、先验与观测。模拟真值仅用于恢复检查。",
        css="design-note",
    )
    return text


def methods_panel(report, folder, copy):
    if __package__:
        from .presentation import blocking_panel
    else:
        from presentation import blocking_panel
    blocks = report["blocking"]["executed"]["blocks"]
    selected = method_refs(blocks)
    text = paragraph(*execution_structure(report, copy), css="case-lede")
    references = (
        '<div class="method-index" aria-label="Basic method references / 基础方法编号">'
    )
    for n, method in enumerate(METHODS, 1):
        used = n in selected
        if not used:
            continue
        references += (
            f'<a href="#design/methods/{n}" class="method-reference {"used" if used else "unused"}">'
            f"<b>{n:02d}</b><span>{bi(*method[0])}<small>"
            + bi(
                "Used in this run" if used else "Not used in this run",
                "本次使用" if used else "本次未用",
            )
            + "</small></span></a>"
        )
    references += "</div>" + paragraph(
        "Method numbers link to the design cards. The block plan below records the methods actually used.",
        "方法编号链接到设计卡片。下面的分块方案记录实际使用的方法。",
        css="design-footnote",
    )
    if any(block["method"] in {"gcr+mh", "gcr+snis"} for block in blocks):
        text += paragraph(
            "In this corrected route, 05 iterative GLS builds the frozen covariance, 01 draws the Gaussian proposal, and 08 corrects against the original target. The original conditional is not being treated as Gaussian.",
            "在这条校正路径中，05 迭代 GLS 构造冻结协方差，01 抽取高斯提议，08 对原目标进行校正。原始条件分布并未被当作高斯分布。",
            css="design-note",
        )
    elif report.get("proposal_policies") and all(p.get("mh_correction", True) for p in report["proposal_policies"]):
        text += paragraph(
            "The selected dense builders define Gaussian draw laws. MH evaluates the full original target and both proposal directions; any remaining parameters use NUTS. Proposal budgets are separate from diagnostic budgets.",
            "选定的稠密构造器定义高斯抽样分布。MH 计算完整原目标及正反向提议密度，剩余参数使用 NUTS。提议预算与诊断预算分开设置。",
            css="design-note",
        )
    text += (
        "<h3>"
        + bi("Actual block plan", "实际分块方案")
        + '</h3><div class="block-plan">'
    )
    for block in blocks:
        methods = ", ".join(f"{n:02d}" for n in METHOD_NUMBERS.get(block["method"], ()))
        outside = ", ".join(block["conditional_on"])
        text += (
            '<section class="block-summary"><header><b>'
            + bi(f"Block {block['index'] + 1}", f"块 {block['index'] + 1}")
            + "</b>"
        )
        text += f'<span class="method-badge">{method_label(block["method"])} · {methods}</span></header>'
        text += '<div class="block-members">' + parameter_symbols(report["case"], block["latents"], show_names=True) + '</div>'
        text += '<p class="block-conditions">' + bi(f"{block['coordinates']} coordinates · Given", f"{block['coordinates']} 个坐标 · 给定") + ' '
        text += parameter_symbols(report["case"], block["conditional_on"]) if outside else bi("none; joint update", "无；联合更新")
        text += '</p>'
        text += detail(
            ("Plan selection reason", "方案选择依据"),
            '<p class="raw-reason">' + esc(block["reason"]) + "</p>",
        )
        text += "</section>"
    text += "</div><h3>" + bi("Methods used", "本次使用的方法") + "</h3>" + references
    if report.get("proposal_selection") == "automatic":
        text += detail(("Automatic selection trace", "自动选择过程"),
                       raw(report["blocking"]["executed"].get("selection_decisions", ())))
    if report["case"] == "multiplicative_noise" and report.get("proposal_selection") == "automatic":
        text += math(r"p_g:\ \log\mu=Up_g+\log(Ap_n),\qquad p_n:\ \mu=\operatorname{diag}(e^{Up_g})Ap_n") + paragraph(
            "The compiler selects log-linear + MH for p_g and iterative GLS + MH for p_n. The latter implements the Gaussian proposal step in GCR + iterative GLS + MH with a dense Cholesky draw for this two-coordinate block. It does not invoke the matrix-free GCR solver. The original parameter-dependent covariance and finite prior bounds enter MH.",
            "编译器为 p_g 选择 log-linear + MH，为 p_n 选择 iterative GLS + MH。后者用稠密 Cholesky 抽样实现 GCR + iterative GLS + MH 中的高斯提议步骤，适合这个二维块，未调用无矩阵 GCR 求解器。原始的参数依赖协方差及有限先验边界均进入 MH。",
            css="design-note",
        )
    if report["case"] == "multiplicative_noise" and not report.get("proposal_policies"):
        text += detail(("Why one NUTS block? What if the bounds are removed?", "为什么只有一个 NUTS 块？去掉边界会怎样？"), paragraph(
            "The default Gaussian-block implementation requires Normal priors. An unbounded flat prior is ImproperUniform, not a Normal distribution, so it does not unlock that route. The positive improper-prior probe also failed at its zero diagnostic anchor. That is a current initialization limitation, not proof of posterior impropriety.",
            "默认高斯块实现要求 Normal 先验。无界平坦先验是 ImproperUniform，并不属于 Normal，因此不会开启这条路径。正值无界先验的探测还在零诊断锚点处失败；这是当前初始化的限制，不能据此判断后验无法归一化。",
        ) + paragraph(
            "The separate explicit proposal demo already splits p_g and p_n while retaining the same finite bounds: bias-corrected log-linear + MH for p_g, iterative GLS + MH for p_n.",
            "单独的显式提议 demo 已在保留相同有限边界的情况下拆分 p_g 和 p_n：p_g 使用偏差修正 log-linear + MH，p_n 使用 iterative GLS + MH。",
        ))
    if report["case"] == "composed_process":
        if any(set(block["latents"]) >= {"instance", "background"} for block in blocks):
            text += math(r"\mu=D_g\!\left([R\;B]\begin{bmatrix}s\\b\end{bmatrix}+h\right),\qquad D_g=\operatorname{diag}(e^{Ug})") + paragraph(
                "s and b form one 14-coordinate linear proposal block. The compiler keeps the instance block intact and verifies joint linearity before merging the background. Gaussian process precision enters the proposal; Uniform background bounds enter MH. The Gaussian realization uses dense Cholesky. Same-layer position alone does not establish joint linearity, as a × b illustrates.",
                "s 与 b 组成一个 14 维线性提议块。编译器保持实例块完整，确认联合线性后合并背景参数。高斯过程精度进入提议，Uniform 背景边界进入 MH。高斯样本使用稠密 Cholesky 生成。同层位置本身不能确立联合线性，例如 a × b。",
                css="design-note",
            )
        text += detail(("Why is g in NUTS?", "为什么 g 仍在 NUTS 中？"), math(r"\log\mu=\log v+Ug\quad(v>0)") + paragraph(
            "When the other blocks are fixed, sigma_w is fixed too: gain remains conditionally log-linear. Updating sigma_w changes the log-noise bias and covariance between sweeps. The current automatic planner does not build that conditional proposal, and the explicit log builder currently requires globally constant fractional noise. These are implementation limits, not a loss of conditional log-linearity.",
            "固定其他块时，sigma_w 也固定，增益仍然条件对数线性。更新 sigma_w 会改变各轮之间的对数噪声偏差与协方差。当前自动规划器尚未构造这种条件提议，显式 log 构造器目前也要求全局恒定的相对噪声。这些是实现限制，并非条件对数线性消失。",
        ))
    text += detail(
        ("Structural measurements & alternative strategy", "结构测量与替代策略"),
        blocking_panel(report),
    )
    if report.get("structure_probe"):
        text += detail(("Prior-range / noise-dependence probes · compilation only", "先验范围 / 噪声依赖探测 · 仅编译"), paragraph(
            "These probes inspect compiler behavior. They produce no posterior samples and make no posterior-propriety claim. The original sampled run above is unchanged.",
            "这些探测只检查编译器行为，不生成后验样本，也不判断后验是否可归一化。上方原采样运行保持不变。",
        ) + raw(report["structure_probe"]))
    return text


def diagnostic_summary(finding):
    m = finding["measurements"]
    if finding["code"] == "latent_treatment":
        return bi(
            "Sample the instance and other parameters from the full joint model, retaining the instance’s conditional density. Marginalisation and marginal Fisher are not prerequisites for this route.",
            "从完整联合模型采样 instance 与其他参数，保留 instance 的条件密度。这条路径无需先做边缘化，也不要求边缘 Fisher。",
        ) if m.get("mode") == "joint_sampling" else bi(
            "Use the requested collapsed route and reconstruct the conditional instance.",
            "使用所请求的折叠路径，再重建条件实例。",
        )
    if finding["code"] == "automatic_proposals":
        return bi("Candidate decisions and NUTS fallbacks are recorded below; priors are retained.",
                  "下方记录候选方法的选择及 NUTS 后备原因；保留声明的先验。")
    if m.get("reason") == "gaussian_prior_perturbation_not_applicable":
        return bi(
            "This check measures a Gaussian prior’s quadratic pull. It does not apply to Uniform priors; sensitivity to their bounds has not been assessed. Sampling still uses those bounds.",
            "此检查衡量高斯先验的二次牵引，不适用于 Uniform；尚未评估边界敏感性。采样仍使用这些边界。",
        )
    if finding["code"] == "proposal_block":
        if finding["conclusion"].endswith("+uncorrected"):
            return bi("User-selected unadjusted proposal; MH is off and target fidelity is approximate.", "用户指定的未校正提议；MH 关闭，目标保真度为近似。")
        return bi(
            "Automatically selected proposal with full original-target MH correction." if "automatic_proposal_selection" in finding.get("grounds", ()) else "User-selected proposal construction with full original-target MH correction.",
            "自动选择的提议，使用完整原目标的 MH 校正。" if "automatic_proposal_selection" in finding.get("grounds", ()) else "用户指定的提议构造，使用完整原目标的 MH 校正。",
        )
    if m.get("reason") == "unsupported_derivative_semantics":
        return bi(
            "Unsupported derivative semantics: measured automatic derivatives cannot establish information rank or flatness.",
            "导数语义不受支持：测得的自动微分结果不能确立信息秩或平坦性。",
        )
    if finding["code"] == "sampled_block":
        return bi(
            "NUTS samples this block under the declared model.",
            "NUTS 按声明的模型采样该块。",
        )
    if "fisher_rank" in m:
        summary = bi(
            f"Observed-likelihood Fisher rank {m['fisher_rank']} / {m['parameters']}; mean rank {m['mean_rank']}.",
            f"观测似然的 Fisher 秩 {m['fisher_rank']} / {m['parameters']}；均值秩 {m['mean_rank']}。",
        )
        if m.get("hierarchical_latent_factors"):
            summary += " " + bi(
                "This excludes latent Gaussian densities and does not judge the marginal likelihood or posterior identifiability.",
                "这里未包含潜变量高斯密度，不判断边缘似然或后验可识别性。",
            )
        return summary
    if finding["conclusion"] == "rank_deficient" and m.get(
        "hierarchical_latent_factors"
    ):
        return bi(
            "The direct observation likelihood has zero directions when the instance is held fixed. Latent Gaussian densities and the marginal likelihood were not included; this does not assess posterior identifiability.",
            "固定随机实例时，直接观测似然存在零信息方向。本检查未包含潜变量的高斯密度或边缘似然，不能据此判断后验不可识别。",
        )
    if "max_shift_sigma" in m:
        return bi(
            f"Largest prior-perturbation shift: {fmt(m['max_shift_sigma'])} posterior SD.",
            f"先验扰动的最大位移：{fmt(m['max_shift_sigma'])} 个后验标准差。",
        )
    if "half_logdet_change" in m:
        explanation = bi(
            f"Change in ½ log|Fisher|: {fmt(m['half_logdet_change'])}.",
            f"½ log|Fisher| 的变化：{fmt(m['half_logdet_change'])}。",
        )
        if m.get("global_flatness_proved"):
            explanation += " " + bi(
                "Primal structure certifies an affine Gaussian mean and fixed covariance in this block’s model coordinates, conditional on the recorded complement.",
                "原始运算结构确证该块在模型坐标中具有仿射高斯均值和固定协方差，条件为记录的其余参数值。",
            )
        elif m.get("reason") == "structural_numerical_contradiction":
            explanation += " " + bi(
                "The numerical change contradicts the structural certificate; the conflict remains recorded.",
                "数值变化与结构证书矛盾，保留该冲突记录。",
            )
        if (
            m.get("reason")
            == "matching_local_probes_without_affine_fixed_covariance_guarantee"
        ):
            explanation += " " + bi(
                "Derivatives were computed, but matching local probes do not prove global flatness.",
                "已经计算导数，但局部探测一致不能证明全局平坦。",
            )
        return explanation
    if "reason" in m:
        return bi(
            "No conclusion for this model or analysis point. The recorded reason is available below.",
            "对该模型或分析点未获得结论。具体原因保留在下方记录中。",
        )
    if m.get("policy") == "user_specified_unchanged":
        return bi("User prior retained.", "保留用户先验。")
    if "linearity_worst_departure" in m:
        return bi(
            f"Largest measured departure from linearity: {fmt(m['linearity_worst_departure'])}.",
            f"测得的最大线性偏离：{fmt(m['linearity_worst_departure'])}。",
        )
    if finding["code"] == "initial_point":
        return bi(
            "A diagnostic anchor, separate from the sampler’s initial state.",
            "诊断锚点，与采样器初值分开记录。",
        )
    if finding["code"] == "sampling_order":
        return esc(" → ".join(", ".join(b) for b in m.get("order", [])))
    if finding["code"] == "diagnostic_budget":
        return bi(
            f"Dense checks: ≤ {m.get('max_parameters')} coordinates; ≤ {m.get('max_matrix_elements')} matrix entries.",
            f"稠密检查：最多 {m.get('max_parameters')} 个坐标、{m.get('max_matrix_elements')} 个矩阵元素。",
        )
    return bi("Full measurements below.", "完整测量见下方。")


def certificate_summary(finding):
    value = finding["measurements"].get("structural_certificate")
    if not value:
        return ""
    certificate = dict(value)
    rows = [dict(row) for row in certificate.get("observations", ())]
    text = paragraph(
        "Certificate checks primal operations, independently of the selected sampler or prior. Float64 numerical rank and probe agreement remain separate requirements.",
        "证书检查原始运算，独立于所选采样器与先验。float64 数值秩及探测一致性仍是单独的要求。",
    )
    text += '<dl class="run-facts">'
    for label, field in [
        (("Certified structure", "已确证结构"), "certified"),
        (("Numerical derivative semantics supported", "数值导数语义受支持"), "numerical_derivatives_trusted"),
        (("Conditioned on", "条件参数"), "conditioned_on"),
        (("Coordinates", "坐标"), "coordinates"),
        (("Reason", "原因"), "reason"),
    ]:
        text += '<dt>' + bi(*label) + '</dt><dd>' + esc(certificate.get(field)) + '</dd>'
    text += '</dl>'
    for row in rows:
        text += paragraph(
            f"{row['name']}: affine mean {row['mean_affine']}; independent covariance {row['covariance_independent']}; unsupported operations {row['unsupported_primitives']}.",
            f"{row['name']}：均值仿射 {row['mean_affine']}；协方差独立 {row['covariance_independent']}；不支持的运算 {row['unsupported_primitives']}。",
        )
    return detail(("Structural certificate", "结构证书"), text)


def diagnostics_panel(report, folder, copy):
    text = paragraph(
        "Recorded before sampling: structure, valid domain, geometry and prior checks. Each finding retains its scope, status and measurements.",
        "采样前记录：结构、有效域、几何与先验检查。每项保留作用范围、状态及测量值。",
        css="case-lede",
    )
    text += prior_variant_note(report)
    unavailable = report.get("jeffreys_unavailable")
    if unavailable:
        text += '<div class="variant-note attention"><strong>' + bi("Jeffreys counterpart: not constructed", "Jeffreys 对照：尚未构造") + '</strong>' + paragraph(
            "A Jeffreys counterpart based on the observed marginal Fisher is not available. The baseline already samples the full joint model with the declared priors; it does not need that marginal Fisher. Only this optional prior construction is unavailable.",
            "基于观测边缘 Fisher 的 Jeffreys 对照暂不可用。原例已经在声明先验下采样完整联合模型，无需该边缘 Fisher。暂缺的是这一可选先验的构造。",
        ) + paragraph(
            "Chosen scope: a prior for root parameters φ, retaining the process law p(s | φ). Average over the unobserved instance to define p(y | φ). The power amplitude a enters through p(s | a); fixing s in the direct observation likelihood removes that information path. This prior definition does not require a collapsed sampler.",
            "所选范围：为顶层参数 φ 定义先验，保留过程分布 p(s | φ)。对未观测实例取平均，得到 p(y | φ)。功率振幅 a 通过 p(s | a) 进入模型；在直接观测似然中固定 s，会去掉这条信息路径。这种先验定义不要求使用边缘化采样器。",
        ) + math(r"p(y\mid\phi)=\int p(y\mid s,\phi)\,p(s\mid\phi)\,ds") + detail(("Recorded scope and limitation", "记录的范围与限制"), raw(unavailable)) + '</div>'
    text += '<div class="prior-strip">'
    for prior in report["blocking"]["priors"]:
        text += (
            f"<span>{inline_symbol(report['case'], prior['latent'])}<code>{esc(prior['latent'])}</code><b>{esc(prior['family'])}</b>"
            + bi(
                f"{prior['coordinates']} coordinates", f"{prior['coordinates']} 个坐标"
            )
            + "</span>"
        )
    text += "</div>"
    if report.get("kind") == "real_observations":
        text += paragraph(
            "The three amplitudes and spectral indices have independent bounded Uniform priors. The two standardized common corrections have Normal(0, 1) priors. These are retained during sampling; a non-flat Jeffreys finding does not replace them.",
            "三区域振幅与谱指数采用独立有界 Uniform 先验；两个标准化共同修正采用 Normal(0, 1)。采样保留这些先验，Jeffreys 非平坦的诊断不会替换它们。",
            css="design-note",
        )
    elif not report.get("variant"):
        text += paragraph(
        "Root parameters use finite Uniform priors in the declared coordinates. Conditional Gaussian process/group laws remain part of the hierarchy. These flat priors do not imply flat Jeffreys density.",
        "顶层参数在声明的坐标中使用有限区间 Uniform。过程/组的条件高斯分布仍属于层级模型。先验平坦并不意味着 Jeffreys 密度平坦。",
        css="design-note",
        )
    bounded = [p for p in report["blocking"]["priors"] if "bounds" in p]
    if bounded:
        text += (
            '<div class="table-scroll"><table><thead><tr><th>'
            + bi("Parameter", "参数")
            + "</th><th>"
            + bi("Lower bound", "下界")
            + "</th><th>"
            + bi("Upper bound", "上界")
            + "</th></tr></thead><tbody>"
        )
        for p in bounded:
            text += f"<tr><th><code>{esc(p['latent'])}</code></th><td>{esc(p['bounds']['lower'])}</td><td>{esc(p['bounds']['upper'])}</td></tr>"
        text += "</tbody></table></div>"
    text += paragraph(
        "JAX computes local derivatives. Expected Fisher also needs the observation law; global flatness needs structural evidence. Hierarchical marginal Fisher requires integrating over random effects. These are distinct checks.",
        "JAX 能计算局部导数；期望 Fisher 还需要观测分布，全局平坦性还需要结构证据。层级模型的边缘 Fisher 则需积分掉随机效应。这是不同的检查。",
    )
    if report["case"] == "linear_gaussian":
        text += math(r"I(\beta)=X^T X/0.25^2") + paragraph(
            "For this declared model the likelihood Fisher is constant and full rank. The primal certificate establishes conditional flatness independently of the bounded Uniform prior and the selected sampler. The findings below report what was recorded for this run.",
            "这个已声明模型的似然 Fisher 恒定且满秩。原始运算证书确立条件平坦性，独立于有界 Uniform 先验及所选采样器。下方结果展示该次运行实际保存的记录。",
        )
    required = (
        report.get("execution", {}).get("diagnostic_policy", {}).get("required", [])
    )
    if not required:
        text += paragraph(
            "This task makes preflight findings informational. Sampling diagnostics and model adequacy are checked separately; unresolved findings are not counted as passes."
            if report.get("kind") == "real_observations" else
            "This task makes preflight findings informational. Sampling and recovery criteria are checked separately; unresolved preflight findings are not counted as passes.",
            "本任务将预检查结果作为诊断信息，采样诊断与模型适配性另行检查；未确定的预检查不计为通过。"
            if report.get("kind") == "real_observations" else
            "本任务将预检查结果作为诊断信息，采样与恢复另行验收；未确定的预检查不计为通过。",
            css="design-footnote",
        )
    text += '<div class="finding-list">'
    for finding in report.get("preflight", []):
        code = re.sub(r"^block_\d+_", "", finding["code"])
        title = FINDINGS.get(code, (finding["code"], finding["code"]))
        text += (
            '<section class="finding"><header><h3>'
            + bi(*title)
            + "</h3>"
            + status(finding["conclusion"])
            + "</header>"
        )
        text += (
            "<small>"
            + esc(finding.get("scope", {}).get("name", "scope not recorded"))
            + "</small><p>"
            + diagnostic_summary(finding)
            + "</p>"
        )
        text += (
            certificate_summary(finding)
            + detail(("Measurements & grounds", "测量与依据"), raw(finding))
            + "</section>"
        )
    if not report.get("preflight"):
        text += paragraph(
            "This saved run has no preflight record.", "此运行未保存预检查记录。"
        )
    text += "</div>"
    return text


def sampling_panel(report, folder, copy, *, show_prediction=True):
    execution = report.get("execution", {})
    sampling = execution.get("sampling", {})
    # The actual route and diagnostic availability are distinct facts.
    blocks = report["blocking"]["executed"]["blocks"]
    independent = report["chain_shape"] is None
    text = paragraph(
        "Initial state → warmup → full sweeps → diagnostics → saved posterior. The values below come from this run.",
        "初始状态 → 预热 → 完整轮转 → 诊断 → 保存后验。以下数值来自本次运行。",
        css="case-lede",
    )
    if any(not p.get("mh_correction", True) for p in report.get("proposal_policies", ())):
        text += paragraph(
            "MH is off for at least one block: this is an unadjusted approximate chain, which need not preserve the original posterior or a common surrogate posterior. ESS and R-hat assess mixing, not this approximation bias. Support and numerical-validity checks remain active.",
            "至少一个块关闭了 MH：这是未经校正的近似链，不保证保持原后验，也不保证对应一个共同的近似后验。ESS 与 R-hat 检查混合情况，不检查这种近似偏差。支持域及数值有效性检查仍然开启。",
            css="variant-note attention",
        )
    text += '<div class="sweep-flow">'
    for block in blocks:
        text += (
            "<span><b>"
            + method_label(block["method"])
            + "</b>"
            + parameter_symbols(report["case"], block["latents"])
            + "</span>"
        )
    text += "</div>"
    text += paragraph(
        "GCR generates independent whole-posterior draws here. Chain initialization and warmup are not needed."
        if independent
        else "The recorded proposal order uses the latest other-block values, followed by any NUTS remainder. One full state is saved after every sweep."
        if report.get("proposal_policies")
        else "A fixed compiler order uses the latest other-block values. One full state is saved after every sweep.",
        "这里 GCR 直接生成整个后验的独立样本，无需链初值及预热。"
        if independent
        else "按记录的提议顺序读取其他块的最新值，再更新可能存在的 NUTS 剩余块。每次完整轮转后保存一个全状态样本。"
        if report.get("proposal_policies")
        else "使用编译器的固定顺序，每块读取其他块的最新值。每次完整轮转后保存一个全状态样本。",
    )
    shape = report["chain_shape"]
    if report["case"] == "composed_process":
        text += '<div class="variant-note">' + math(r"p(s,\phi\mid y)\propto p(y\mid s,\phi)\,p(s\mid\phi)\,p(\phi)") + paragraph(
            "This run samples the instance s explicitly, together with compatible linear parameters in its recorded block. Each sweep applies the Gaussian proposal with MH, then updates the remaining parameters with NUTS. No marginal integral or marginal Fisher is required. Optional prior diagnostics do not block this run.",
            "本次显式采样 instance s，并按记录的参数块联合更新相容的线性参数。每轮先使用高斯提议与 MH，再用 NUTS 更新其余参数，无需边缘积分或边缘 Fisher。可选先验诊断不会阻止本次采样。",
        ) + '</div>'
    counts = (
        f"{shape[0]} × {shape[1]}"
        if shape
        else str(sampling.get("draws", report["requested_draws_per_chain"]))
    )
    text += '<dl class="run-facts">'
    for title, value in [
        (("Retained samples", "保留样本"), counts),
        (
            ("Warmup per chain", "每链预热"),
            str(0 if independent else sampling.get("warmup", report["warmup"])),
        ),
        (
            ("Stop reason", "停止原因"),
            sampling.get("stop_reason", execution.get("termination", "unrecorded")),
        ),
        (
            ("Policy", "策略"),
            execution.get("stopping_policy", {}).get("mode", "unrecorded"),
        ),
        (
            ("Divergences", "发散数"),
            "—" if independent else str(sampling.get(
                "divergences", report.get("checks", {}).get("chain_diagnostics", {}).get("divergences", "unrecorded")
            )),
        ),
    ]:
        text += "<dt>" + bi(*title) + "</dt><dd>" + esc(value) + "</dd>"
    text += "</dl>"
    proposal_records = sampling.get("proposal_diagnostics", ())
    if proposal_records:
        text += '<h3>' + bi("Recorded proposal updates", "已记录的提议更新") + '</h3>'
        for record in proposal_records:
            row = dict(record)
            text += '<h4>' + parameter_symbols(report["case"], row["names"]) + '</h4>'
            corrected = row.get("mh_correction", True)
            rate = row.get("acceptance_rate") if corrected else row.get("application_rate")
            text += paragraph(
                f"{'MH on' if corrected else 'MH off — approximate updates'}: {row['accepted']} adopted / {row['attempted']} attempts; {row['support_rejected']} support rejections; {row['numerical_failure']} numerical failures. {'MH acceptance' if corrected else 'Application'} rate {fmt(rate)}.",
                f"{'MH 开启' if corrected else 'MH 关闭——近似更新'}：{row['accepted']} 次采用 / {row['attempted']} 次尝试；{row['support_rejected']} 次支持域拒绝；{row['numerical_failure']} 次数值失败。{'MH 接受率' if corrected else '采用率'} {fmt(rate)}。",
            )
            text += detail(
                ("Proposal acceptance and rejection counts", "提议接受与拒绝计数"),
                raw(row),
            )
        text += paragraph(
            "Production counts exclude warmup and include rejected steps. Rejection retains the current block and continues the sweep; there is no retry-until-accepted rule.",
            "正式采样计数不含预热，包含被拒绝的步。拒绝时保留当前块并继续轮转，不采用重试到接受的规则。",
        )
    initial = execution.get("initial_values", {})
    if initial:
        text += "<h3>" + bi("Actual starting points", "实际初值") + "</h3>"
        init_record = report.get("signal", {}).get("view", {}).get("initialization")
        if init_record:
            text += paragraph(
                "This comparison explicitly starts from a bounded L-BFGS fit to the data, initialized at prior centers. Both chains start at that fit with independent random streams. No generating values are used; this is a declared setup, not automatic compiler fallback.",
                "这项比较显式使用从先验中心开始、在参数边界内对数据进行 L-BFGS 拟合得到的初值。两条链从该拟合点出发，使用独立随机数流。未使用生成真值；这是声明的运行设置，并非编译器的自动后备方法。",
            ) if report.get("proposal_policies") else paragraph(
                "This demo explicitly supplies a data-only L-BFGS fit, started from prior centers, with two small support-coordinate perturbations. It uses no generating values. This setup helps the existing mixed sampler initialize; it is not automatic fallback selection.",
                "本例显式提供从先验中心开始、仅用观测数据得到的 L-BFGS 拟合点，再在支持域坐标中作两次小扰动。未使用生成真值。此设置帮助已有混合采样器初始化，并非自动后备方法选择。",
            )
            text += detail(
                ("Initialization fit record", "初值拟合记录"), raw(init_record)
            )
        elif report.get("proposal_policies"):
            text += paragraph(
                "This comparison explicitly supplies generic zero/one starting values within the declared support. They are validated before warmup and do not use simulation truth.",
                "这项比较显式提供声明支持域内的通用零/一起点。预热前验证这些初值，未使用模拟真值。",
            )
        else:
            text += paragraph(
                "Automatic candidates were checked for support and finite joint density/gradients. No simulation truth was supplied.",
                "自动候选经过支持域及有限联合密度/梯度检查。未提供模拟真值。",
            )
        text += (
            '<div class="table-scroll"><table><thead><tr><th>'
            + bi("Latent", "参数")
            + "</th><th>"
            + bi("Values by chain", "逐链初值")
            + "</th></tr></thead><tbody>"
        )
        text += "".join(
            f"<tr><th><code>{esc(name)}</code></th><td><code>{esc(json.dumps(value))}</code></td></tr>"
            for name, value in initial.items()
        )
        text += "</tbody></table></div>"
    checkpoints = [dict(c) for c in sampling.get("checkpoints", [])]
    if checkpoints:
        text += (
            "<h3>"
            + bi("Recorded diagnostic checks", "已记录的诊断检查")
            + '</h3><div class="table-scroll"><table><thead><tr>'
        )
        for title in [
            ("Draws / chain", "每链样本"),
            ("ESS min", "最小 ESS"),
            ("R̂ max", "最大 R̂"),
            ("Mean MCSE max", "最大均值 MCSE"),
            ("Divergences", "发散"),
            ("Verdict", "结论"),
        ]:
            text += "<th>" + bi(*title) + "</th>"
        text += "</tr></thead><tbody>"
        for c in checkpoints:
            text += "<tr>" + "".join(
                "<td>" + fmt(c.get(k)) + "</td>"
                for k in (
                    "draws_per_chain",
                    "ess_min",
                    "rhat_max",
                    "mcse_mean_max",
                    "divergences",
                )
            )
            text += (
                "<td>" + status("passed" if c["passed"] else "failed") + "</td></tr>"
            )
        text += "</tbody></table></div>"
    text += paragraph(
        "Fixed-budget runs stop at the declared sample count; their diagnostic result is reported separately. Checkpoint mode can stop after consecutive passes or a draw/time cap.",
        "固定预算运行达到声明的样本数后停止，诊断结论单独报告。检查点模式可在连续通过或达到样本/时间上限时停止。",
        css="design-note",
    )
    text += paragraph(
        "Summaries use the saved draws with equal weights. Rao–Blackwell averaging and importance reweighting are not used in these runs.",
        "汇总对保存的样本等权平均。本组运行未使用 Rao–Blackwell 条件平均或重要性重加权。",
    )
    text += detail(
        ("Full execution record & settings", "完整执行记录与设置"), raw(execution)
    )
    if not show_prediction:
        return text
    text += image(
        folder,
        "inference",
        (
            "Recovered conditional mean/probability and its uncertainty; bands exclude new observation noise.",
            "恢复的条件均值/概率及其不确定性；区间不含新观测噪声。",
        ),
    )
    if report["signal"]["view"].get("kind") == "process":
        text += image(
            folder,
            "spectrum",
            (
                "Posterior ensemble spectrum versus generating spectrum and the power of the one sampled instance.",
                "后验总体功率谱、生成谱与本次单个随机实例功率的比较。",
            ),
        )
        noise = report["signal"]["view"].get("noise_realization", {})
        if noise:
            text += paragraph(
                f"The {noise['observations']} simulated standardized noise values have sample SD {noise['standardized_residual_sd']:.3f}, while the generating SD is 1. A finite realization can have lower variance; any missed generating parameter remains a failed recovery check.",
                f"本次 {noise['observations']} 个标准化模拟噪声的样本标准差为 {noise['standardized_residual_sd']:.3f}，生成分布的标准差为 1。有限实例可能具有较低方差；任何未覆盖的生成参数仍记为恢复检查失败。",
                css="design-note",
            )
    return text


def power_law_regression_panel(report):
    view = report["signal"]["view"]
    reference, experiment = view["posterior_reference"], view["bias_experiment"]
    text = '<section class="variant-note"><h3>' + bi(
        "This dataset: sampling and numerical reference", "本次数据：采样与数值参考"
    ) + '</h3>' + paragraph(
        "Both A and α are inferred from y = A x^α + ε, with known independent Gaussian noise. Refined two-dimensional integration computes posterior means; profiled optimization computes the joint MAP in (A, α).",
        "从 y = A x^α + ε 同时推断 A 与 α，独立高斯噪声强度已知。加密的二维积分计算后验均值；剖面优化计算 (A, α) 坐标中的联合 MAP。",
    )
    rows = {row["name"]:row for row in report["parameters"]}
    text += '<div class="table-scroll"><table><thead><tr><th>' + bi("Parameter", "参数")
    for label in (("Truth", "真值"), ("Sampled mean", "采样均值"),
                  ("Reference mean", "参考均值"), ("Joint MAP", "联合 MAP")):
        text += '</th><th>' + bi(*label)
    text += '</th></tr></thead><tbody>'
    names = ("amplitude[0]", "alpha[0]", "amplitude[1]", "alpha[1]")
    for i, name in enumerate(names):
        row = rows[name]
        text += '<tr><th>' + esc(reference["coordinates"][i]) + '</th>'
        for value in (row["truth"], row["mean"], reference["mean"][i], reference["map"][i]):
            text += f'<td>{value:.4f}</td>'
        text += '</tr>'
    text += '</tbody></table></div>' + detail(
        ("Integration accuracy and sampling error", "积分精度与采样误差"),
        raw({"reference":reference,"sampling_check":report["checks"].get("quadrature_oracle")}),
    ) + '</section>'
    text += '<section class="variant-note"><h3>' + bi(
        "Repeated simulations: bias = average estimate − truth",
        "重复模拟：偏差 = 估计的平均值 − 真值",
    ) + '</h3>' + paragraph(
        f"{experiment['repeats']:,} datasets, {experiment['observations_per_channel']} observations per curve, with the declared bounded Uniform priors. Entries are bias ± Monte Carlo standard error; posterior means use refined integration, without MCMC error.",
        f"{experiment['repeats']:,} 组数据，每条曲线 {experiment['observations_per_channel']} 个观测，采用声明的有界 Uniform 先验。表中为偏差 ± Monte Carlo 标准误；后验均值使用加密积分，不含 MCMC 误差。",
    )
    text += '<div class="table-scroll"><table><thead><tr><th>' + bi("Prior / estimate", "先验 / 估计量") + '</th>'
    text += ''.join('<th>' + esc(name) + '</th>' for name in experiment["coordinates"])
    text += '</tr></thead><tbody>'
    for row in experiment["rows"]:
        if row["prior"] != "flat":
            continue
        label = ("Uniform" if row["prior"] == "flat" else "Jeffreys") + " · "
        label += "MAP" if row["estimator"] == "map" else bi("posterior mean", "后验均值")
        text += '<tr><th>' + label + '</th>' + ''.join(
            f'<td>{bias:+.4f} ± {se:.4f}</td>' for bias,se in zip(row["bias"],row["mcse"],strict=True)
        ) + '</tr>'
    text += '</tbody></table></div>' + paragraph(
        "Bias is measured at these generating parameters. The original Gaussian likelihood is used throughout; taking log y would change the additive-noise model.",
        "偏差在这组生成参数处测量。全过程使用原始高斯似然；对 y 取对数会改变加性噪声模型。",
    ) + detail(("RMSE and Monte Carlo uncertainty", "RMSE 与 Monte Carlo 不确定性"), raw(
        [row for row in experiment["rows"] if row["prior"] == "flat"]
    ))
    return text + '</section>'


def power_law_bias_panel(report):
    experiment = report["signal"]["view"].get("bias_experiment")
    if not experiment:
        return ""
    if experiment.get("kind") == "power_law_regression":
        return power_law_regression_panel(report)
    text = '<section class="variant-note"><h3>' + bi(
        "Repeated simulations: which estimate is biased?", "重复模拟：哪一种估计有偏？"
    ) + '</h3>'
    text += math(r"T_j=\sum_i\log x_{ij},\quad T_j\sim\mathrm{Gamma}(n,\mathrm{rate}=\alpha_j-1)")
    text += math(r"\widehat\alpha_{\mathrm{MAP,flat}}=1+\frac{n}{T},\qquad\widehat\alpha_{\mathrm{MAP,Jeffreys}}=1+\frac{n-1}{T}")
    text += paragraph(
        f"{experiment['repeats']:,} paired datasets, {experiment['observations_per_channel']} observations per population. Bias is average estimate minus truth; ± is Monte Carlo standard error. Both priors use the same finite bounds. The formulas above describe interior MAP values, clipped to those bounds when needed.",
        f"{experiment['repeats']:,} 组配对数据集，每个总体 {experiment['observations_per_channel']} 个观测。偏差为估计的平均值减真值；± 为 Monte Carlo 标准误。两种先验使用相同有限边界。上式描述内部 MAP，必要时截断至边界。",
    )
    text += '<div class="table-scroll"><table><thead><tr><th>' + bi("Prior / estimate", "先验 / 估计量") + '</th><th>α₁</th><th>α₂</th></tr></thead><tbody>'
    for row in experiment["rows"]:
        label = ("Uniform" if row["prior"] == "flat" else "Jeffreys") + " · "
        label += "MAP" if row["estimator"] == "map" else bi("posterior mean", "后验均值")
        text += '<tr><th>' + label + '</th>' + ''.join(
            f'<td>{bias:+.4f} ± {se:.4f}</td>' for bias, se in zip(row["bias"], row["mcse"], strict=True)
        ) + '</tr>'
    text += '</tbody></table></div>' + paragraph(
        "Without boundaries, flat MAP has bias (α−1)/(n−1), and Jeffreys MAP is unbiased. Jeffreys posterior mean equals flat MAP and still has that bias. Finite bounds can change these identities. One dataset’s truth coverage is a separate check.",
        "无边界参考模型中，平坦先验 MAP 的偏差为 (α−1)/(n−1)，Jeffreys MAP 无偏。Jeffreys 后验均值等于平坦先验 MAP，仍有上述偏差。有限边界可能改变这些等式。单次数据的真值覆盖是另一项检查。",
    )
    text += detail(("Analytic posterior and boundary effects", "解析后验与边界效应"),
                   raw({"posterior": report["signal"]["view"]["analytic_posterior"],
                        "repeated_experiment": experiment}))
    return text + '</section>'


def noise_scale_panel(report):
    comparison = report.get("noise_scale_comparison")
    if not comparison:
        return ""
    text = '<section class="variant-note"><h3>' + bi("06 · Noise-scale sensitivity", "06 · 噪声尺度敏感性") + '</h3>'
    truth = comparison["rows"][0]["truth"]
    chains, draws = comparison["chain_shape"]
    text += paragraph(
        f"Same simulated data, truth σ_w = {truth:g}, {chains} chains × {draws:,} retained draws and unchanged 99% recovery criterion. Only the noise-scale prior changes.",
        f"使用相同模拟数据、真值 σ_w = {truth:g}、{chains} 条链各 {draws:,} 个保留样本和不变的 99% 恢复标准。仅改变噪声尺度的先验。",
    )
    text += '<div class="table-scroll"><table><thead><tr>' + ''.join('<th>' + bi(*label) + '</th>' for label in (
        ("Prior", "先验"), ("Posterior mean", "后验均值"), ("99% interval", "99% 区间"),
        ("Truth covered", "覆盖真值"), ("Chain diagnostics", "链诊断"),
    )) + '</tr></thead><tbody>'
    for label, row in zip((("Bounded Uniform", "有界 Uniform"), ("Mild truncated Normal", "温和截断正态")), comparison["rows"], strict=True):
        text += '<tr><th>' + bi(*label) + '</th>'
        text += f'<td>{row["mean"]:.6f}</td><td>[{row["lower"]:.6f}, {row["upper"]:.6f}]</td>'
        text += '<td>' + bi("Yes" if row["covered"] else "No", "是" if row["covered"] else "否") + '</td><td>'
        text += bi("Passed" if row["diagnostics_passed"] else "Failed", "通过" if row["diagnostics_passed"] else "未通过") + '</td></tr>'
    reference = comparison["known_mean"]
    lo, _, hi = reference["uniform_sigma_posterior_quantiles"]
    text += '</tbody></table></div>' + paragraph(
        f"Conditioning on the generating mean gives a residual RMS of {reference['realized_noise_rms']:.6f} and a Uniform-prior conditional 99% interval of [{lo:.6f}, {hi:.6f}]. The realized noise energy is at percentile {100*reference['energy_lower_tail_probability']:.3f} of its generating distribution. Truth is used here only to diagnose the simulation, never to initialize or fit the model.",
        f"固定生成均值后，残差 RMS 为 {reference['realized_noise_rms']:.6f}，Uniform 先验下的条件 99% 区间为 [{lo:.6f}, {hi:.6f}]。实际噪声能量位于生成分布的第 {100*reference['energy_lower_tail_probability']:.3f} 百分位。这里使用真值仅为诊断模拟，绝不用于初始化或拟合模型。",
    )
    baseline, mild = comparison["rows"]
    delta_sd = abs(mild["mean"] - baseline["mean"]) / baseline["posterior_sd"]
    text += paragraph(
        f"The prior changes the sampled mean by {delta_sd:.3f} baseline posterior SD. A single coverage miss does not establish repeated-simulation bias. More draws improve computational precision, not this dataset's noise realization; tail quantiles also have Monte Carlo uncertainty.",
        f"更改先验使采样均值移动了原后验标准差的 {delta_sd:.3f} 倍。单次未覆盖不能证明重复模拟意义上的偏差。更多样本会提高计算精度，但不会改变这次噪声实现；尾部分位数也有 Monte Carlo 不确定性。",
    )
    if min(baseline["chain_upper_995"]) < truth < max(baseline["chain_upper_995"]):
        text += paragraph("The baseline chains' separate upper endpoints lie on opposite sides of the truth: the recovery decision is close to a noisy tail boundary.", "原例各条链分别计算的区间上端落在真值两侧：恢复判定接近具有计算噪声的尾部边界。")
    return text + detail(("Paired-run and known-mean evidence", "配对运行与已知均值的诊断证据"), raw(comparison)) + '</section>'


def observation_count_panel(report):
    comparison = report.get("observation_count_comparison")
    if not comparison:
        return ""
    rows = comparison["rows"]
    text = '<section class="variant-note"><h3>'+bi("More observations · same process instance", "更多观测 · 相同过程实例")+'</h3>'
    text += paragraph(
        f"Both runs use the original bounded Uniform root priors, the same generating parameters and Fourier instance, and {comparison['chain_shape'][0]} chains × {comparison['chain_shape'][1]:,} retained draws. Both use the same random stream on different input grids; the larger dataset does not retain the original noisy observations, and these two runs are not independent replicates.",
        f"两次运行采用原有顶层有界 Uniform 先验、相同生成参数与 Fourier 实例，均保留 {comparison['chain_shape'][0]} 条链各 {comparison['chain_shape'][1]:,} 个样本。两次使用相同随机数流，但对应不同输入网格；较大数据集没有保留原来的带噪观测，两次也不是独立重复实验。",
    )
    text += '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+bi(*label)+'</th>' for label in (
        ("Observations", "观测数"), ("σ_w mean", "σ_w 均值"), ("σ_w SD", "σ_w 标准差"),
        ("σ_w 99% interval", "σ_w 99% 区间"), ("All-coordinate recovery", "全部坐标恢复"),
    ))+'</tr></thead><tbody>'
    for row in rows:
        sigma = row["parameters"]["sigma_w"]
        text += f'<tr><th>{row["observations"]:,}</th><td>{sigma["mean"]:.6f}</td><td>{sigma["posterior_sd"]:.6f}</td><td>[{sigma["lower"]:.6f}, {sigma["upper"]:.6f}]</td><td>'+bi("Passed" if row["recovery_passed"] else "Not covered / insufficient narrowing", "通过" if row["recovery_passed"] else "未覆盖或收窄不足")+'</td></tr>'
    small, large = rows
    factor = small["parameters"]["sigma_w"]["posterior_sd"] / large["parameters"]["sigma_w"]["posterior_sd"]
    a_sd, b_sd = (r["parameters"]["power_amplitude"]["posterior_sd"] for r in rows)
    text += '</tbody></table></div>'+paragraph(
        f"The σ_w posterior SD is {factor:.2f} times smaller. The power-amplitude SD changes from {a_sd:.3f} to {b_sd:.3f}: denser observations resolve the same instance but add no independent Fourier modes. One dataset at each size cannot establish a change in repeated-simulation bias or coverage.",
        f"σ_w 的后验标准差缩小了 {factor:.2f} 倍。功率幅度的标准差由 {a_sd:.3f} 变为 {b_sd:.3f}：更密的观测更清楚地测量同一实例，但没有增加独立 Fourier 模式。每个数据量只有一组数据，不能据此确定重复模拟意义下的偏差或覆盖率变化。",
    )
    return text+detail(("Saved-run comparison and scope", "已保存运行的比较与范围"), raw(comparison))+'</section>'


def recovery_panel(report, folder, copy):
    rows = report["parameters"]
    text = ""
    if report.get("repeated_study"):
        if __package__:
            from .repeated_panels import render_repeated
        else:
            from repeated_panels import render_repeated
        text += render_repeated(report["repeated_study"], report["repeated_study_folder"])
        text += '<h3>'+bi(f"Original single dataset · seed {report['seed']}", f"原始单次数据 · 种子 {report['seed']}")+'</h3>'
    if report.get("jeffreys_unavailable"):
        text += '<div class="variant-note attention"><strong>' + bi(
            "Baseline results · bounded Uniform root priors",
            "原始模型结果 · 顶层参数采用有界 Uniform 先验",
        ) + '</strong>' + paragraph(
            "These are baseline posterior samples. The marginal Jeffreys counterpart has not been implemented, so it has no recovery result. Sampling diagnostics and truth coverage below describe this Uniform run only.",
            "这些是原始模型的后验样本。边缘 Jeffreys 对照尚未实现，因此没有对应恢复结果。下方采样诊断与真值覆盖率仅描述本次 Uniform 运行。",
        ) + '</div>'
    text += (
        '<div class="verdict '
        + ("passed" if report["passed"] else "failed")
        + '"><strong>'
    )
    text += (
        bi(
            "Recovery checks passed" if report["passed"] else "Recovery checks failed",
            "恢复检查通过" if report["passed"] else "恢复检查未通过",
        )
        + "</strong><span>"
    )
    covered = sum(r["covered"] for r in rows)
    narrowed = sum(r["informative"] for r in rows)
    text += (
        bi(
            f"{covered}/{len(rows)} truths covered; {narrowed}/{len(rows)} posteriors narrowed.",
            f"{covered}/{len(rows)} 个真值被覆盖；{narrowed}/{len(rows)} 个后验充分收窄。",
        )
        + "</span></div>"
    )
    if report.get("variant") and report["variant"].get("kind") != "observation_count":
        text += paragraph(
            "For a like-for-like comparison, the SD threshold uses the original bounded-Uniform prior scale. It is a fixed reference scale, not a measured SD of the new prior.",
            "为保持同一比较标准，标准差阈值使用原有有界 Uniform 先验的尺度。这是固定参考尺度，并非新先验的实测标准差。",
            css="design-note",
        )
    text += power_law_bias_panel(report)
    text += observation_count_panel(report)
    text += noise_scale_panel(report)
    if report["signal"]["view"].get("model_kind") == "power_law_regression":
        text += '<h3>' + bi("One dataset: posterior recovery", "单次数据：后验恢复") + '</h3>' + paragraph(
            "Below, dots mark this dataset's sampled posterior means, and bars show 99% posterior intervals. The horizontal unit is posterior SD. These single-dataset errors are different from the repeated-simulation bias above.",
            "下图圆点为本次数据的采样后验均值，横线为 99% 后验区间，横轴单位为后验标准差。单次数据的误差与上方重复模拟的平均偏差不同。",
        )
    text += image(
        folder,
        "recovery",
        (
            "All coordinates on a common posterior-SD scale; 99% intervals crossing zero cover truth. The matrix shows posterior correlations.",
            "所有坐标以各自后验标准差为单位；跨过零的 99% 区间覆盖真值。矩阵展示后验相关性。",
        ),
    )
    text += '<label class="parameter-filter">' + bi(
        "Inspect one coordinate", "查看一个坐标"
    )
    text += '<select data-parameter-filter aria-label="Parameter / 参数">'
    text += (
        "".join(
            f'<option value="{i}">{esc(row["name"])}</option>'
            for i, row in enumerate(rows)
        )
        + "</select></label>"
    )
    text += '<div class="coordinate-details">'
    for i, row in enumerate(rows):
        text += f'<section data-parameter="{i}" {"hidden" if i else ""}><h3><code>{esc(row["name"])}</code></h3><div class="coordinate-pair">'
        text += image(
            folder,
            f"parameter-{i}",
            (
                "Posterior draws; dashed orange marks truth.",
                "后验样本；橙色虚线标记真值。",
            ),
        )
        text += '<dl class="run-facts">'
        for title, value in [
            (("Truth", "真值"), row["truth"]),
            (("Posterior mean", "后验均值"), row["mean"]),
            (("Posterior SD", "后验标准差"), row["posterior_sd"]),
            (("SD / reference prior SD", "标准差 / 参考先验标准差") if report.get("variant") else ("SD / prior SD", "标准差 / 先验标准差"), row["sd_ratio"]),
        ]:
            text += "<dt>" + bi(*title) + "</dt><dd>" + fmt(value) + "</dd>"
        text += "</dl></div>"
        if report["chain_shape"]:
            text += detail(
                ("Chain traces", "链轨迹"),
                image(
                    folder,
                    f"trace-{i}",
                    (
                        "Separate retained chains; warmup excluded.",
                        "分别展示保留的链，已去除预热。",
                    ),
                ),
            )
        text += "</section>"
    text += "</div>"
    text += detail(
        ("Numerical checks & acceptance criteria", "数值检查与通过标准"),
        paragraph(
            "Every generating value must be inside its marginal 99% interval, and every posterior SD must be below half the reference prior SD. Prior comparisons retain the original Uniform scale as that reference. Chain diagnostics must pass. The linear case also checks an independent analytic posterior.",
            "每个生成真值须在其边际 99% 区间内，每个后验标准差须小于参考先验标准差的一半。先验对照保留原 Uniform 尺度作为参考。链诊断须通过。线性例子还核对独立解析后验。",
        )
        + raw({"parameters": rows, "checks": report["checks"]}),
    )
    text += paragraph(
        "This checks recovery for one simulated dataset. It is not an SBC study or a repeated-experiment coverage guarantee.",
        "这里检查一份模拟数据的参数恢复，并非 SBC 研究或重复实验的覆盖率保证。",
        css="design-footnote",
    )
    return text


def render_case(report, folder, index):
    case = report["case"]
    copy = (CASES if case in CASES else LEGACY_CASES)[case]
    text = (
        f'<article class="case methodology" id="{esc(case)}" data-case="{esc(case)}">'
    )
    title = ((f"Example {list(CASES).index(case) + 1:02d} · recorded run",
              f"例子 {list(CASES).index(case) + 1:02d} · 已记录运行") if case in CASES
             else ("Earlier example · retained reference", "早期例子 · 保留参考"))
    text += '<header class="case-heading"><span class="status-tag">' + bi(*title)
    text += (
        "</span>"
        + status("passed" if report["passed"] else "failed")
        + "<h1>"
        + bi(*copy["title"])
        + ('<span class="variant-label">' + variant_label(report) + '</span>' if report.get("variant") else "")
        + '</h1><p class="question">'
        + bi(*execution_structure(report, copy))
        + "</p></header>"
    )
    links = report.get("comparison_links", ())
    if links:
        text += '<nav class="variant-links" aria-label="Model comparisons / 模型对照">' + ''.join(f'<a data-gallery-link data-preserve-step href="{esc(link["href"])}">{bi(*link["label"])}</a>' for link in links) + '</nav>'
    text += '<div class="case-controls design-controls"><nav class="steps design-tabs" aria-label="Example chapters / 例子章节">'
    for i, (key, title) in enumerate(STAGES):
        text += f'<button type="button" class="step-button" data-step="{key}" aria-controls="{case}-{key}" aria-pressed="{str(i == 0).lower()}"><b>{i + 1}</b>{bi(*title)}</button>'
    text += '</nav><nav class="language-switch" aria-label="Language / 语言"><button type="button" data-language-choice="en">English</button><button type="button" data-language-choice="zh">中文</button></nav></div>'
    renderers = [
        model_panel,
        methods_panel,
        diagnostics_panel,
        sampling_panel,
        recovery_panel,
    ]
    for (key, title), render in zip(STAGES, renderers, strict=True):
        text += (
            f'<section class="stage" id="{case}-{key}" data-stage="{key}"><h2>'
            + bi(*title)
            + "</h2>"
        )
        text += render(report, folder, copy) + "</section>"
    text += (
        '<div class="step-footer"><button type="button" class="previous">'
        + bi("Previous", "上一步")
        + '</button><span class="step-position" aria-live="polite"></span><button type="button" class="next">'
        + bi("Next", "下一步")
        + "</button></div>"
    )
    text += (
        '<footer class="artifacts"><span>'
        + bi("Saved numerical evidence", "已保存的数值证据")
        + "</span>"
    )
    for filename, title in [
        ("result.json", ("Results & diagnostics", "结果与诊断")),
        ("analysis.artifact.json", ("Analysis artifact", "分析记录")),
        ("task.artifact.json", ("Task settings", "任务配置")),
        ("posterior.npz", ("Posterior draws", "后验样本")),
    ]:
        text += f'<a href="{quote(folder, safe="")}/{filename}">' + bi(*title) + "</a>"
    return text + "</footer></article>"
