"""Paired English/Chinese reading guide. Shared structure, formulas and evidence."""

from __future__ import annotations

import html


def bi(en, zh):
    """Keep each translation beside its source; never translate numeric evidence."""
    return (
        f'<span class="translated" lang="en">{html.escape(str(en))}</span>'
        f'<span class="translated" lang="zh">{html.escape(str(zh))}</span>'
    )


def math(tex):
    return f'<div class="design-math" data-math="{html.escape(tex, quote=True)}">{html.escape(tex)}</div>'


def paragraph(en, zh, css=""):
    return f'<p class="{css}">{bi(en, zh)}</p>'


def detail(title, body, identity=""):
    anchor = f' id="{html.escape(identity, quote=True)}"' if identity else ""
    return f'<details class="guide-detail"{anchor}><summary>{bi(*title)}</summary><div class="detail-body">{body}</div></details>'


METHOD_MODELS = [
    (("Observation model", "观测模型"), r"y\mid\beta,c\sim\mathcal N(X_c\beta+r_c,\,\Sigma_c)"),
    (("Observation model", "观测模型"), r"\log y\mid\beta,c\sim\mathcal N(X_c\beta+r_c,\,V_c)"),
    (("General differentiable target", "通用可微目标"), r"\pi(\theta\mid y)\propto L(y\mid\theta)p(\theta)"),
    (("Integrable joint model", "可积分的联合模型"), r"\pi(c\mid y)=\int\pi(b,c\mid y)\,db"),
    (("Observation model", "观测模型"), r"y\mid\beta,c\sim\mathcal N(X_c\beta+r_c,\,\Sigma(\beta,c))"),
    (("Observation model", "观测模型"), r"\begin{gathered}y_i=\mu_i(1+f_i\epsilon_i),\quad \epsilon_i\sim\mathcal N(0,1)\\ \log\mu=X_c\beta+r_c\end{gathered}"),
    (("Gaussian observation example", "高斯观测示例"), r"y\mid\beta,c\sim\mathcal N(f(\beta,c),\,\Sigma_c)"),
    (("General target and proposal", "通用目标与提议"), r"\pi(\theta\mid y)\propto L(y\mid\theta)p(\theta),\quad \theta'\sim q(\cdot\mid\theta)"),
]


METHODS = [
    (
        ("GLS/GCR · fixed covariance", "GLS/GCR · 固定协方差"),
        ("Linear mean; full conditional is Gaussian.", "均值线性；完整条件分布为高斯。"),
        ("Conditional mean, mode and GCR draws.", "条件均值、众数及 GCR 抽样。"),
        ("Include every factor, positive-definite precision and compatible support.", "须覆盖所有因子，精度矩阵正定，且支持域相容。"),
        ("Implemented for supported graphs", "已实现，限支持的图"),
    ),
    (
        ("Exact log-space GCR", "对数空间精确 GCR"),
        ("Exact Gaussian log law; quadratic full conditional.", "对数观测律精确高斯；完整条件对数密度为二次型。"),
        ("Exact conditional draws in transformed coordinates.", "变换坐标下的精确条件抽样。"),
        ("Verify the transformation, every factor and support. A log-linear mean alone is insufficient.", "核对变换、全部因子及支持域。仅有对数均值线性还不够。"),
        ("Implemented for eligible log models", "已实现，限符合条件的对数模型"),
    ),
    (
        ("NUTS / residual sampler", "NUTS / 剩余块采样器"),
        ("Differentiable target with supported continuous coordinates.", "目标可微，且连续参数坐标受后端支持。"),
        ("MCMC for the declared posterior.", "针对已声明后验的 MCMC。"),
        ("Check support, coordinate Jacobians, divergences, ESS and convergence.", "检查支持域、坐标 Jacobian、发散、ESS 与收敛。"),
        ("NUTS implemented", "NUTS 已实现"),
    ),
    (
        ("Exact elimination / collapse", "精确消元 / 边缘化"),
        ("A conditional block can be integrated analytically.", "某条件块可被解析积分。"),
        ("Reduced target plus a reconstruction distribution.", "降维目标及重建分布。"),
        ("Retain normalization and verify integral/covariance assumptions. This removes an update rather than adding one.", "保留归一化并核对积分与协方差假设。此法消除一次更新，而非增加更新。"),
        ("Implemented for supported structures", "已实现，限支持的结构"),
    ),
    (
        ("Iterative GLS", "迭代 GLS"),
        ("Linear mean; covariance changes with the block.", "均值线性；协方差随该块变化。"),
        ("Reweighted GLS fixed point.", "重加权 GLS 不动点。"),
        ("Freeze covariance, solve, then refresh weights. A MAP solve also includes covariance derivatives and the declared prior.", "冻结协方差求解，再更新权重。MAP 求解还须包含协方差导数和声明的先验。"),
        ("Solver and explicit bounded MH proposals implemented", "求解器及显式有预算限制的 MH 提议已实现"),
    ),
    (
        ("Bias-corrected log-linear solve", "偏差修正的对数线性求解"),
        ("Log-linear mean; small multiplicative Normal noise.", "对数均值线性；小幅乘性正态噪声。"),
        ("Approximate point, Gaussian approximation or proposal.", "近似点估计、高斯近似或提议分布。"),
        ("Use log(y) + f²/2; retain the prior, check positivity and approximation error.", "使用 log(y) + f²/2；保留先验，检查正值域与近似误差。"),
        ("Transform and explicit bounded MH proposals implemented", "变换及显式有预算限制的 MH 提议已实现"),
    ),
    (
        ("Local linearization / Gauss–Newton", "局部线性化 / Gauss–Newton"),
        ("Smooth nonlinear dependence.", "光滑非线性依赖。"),
        ("Optimization step, local uncertainty or proposal.", "优化步、局部不确定性或提议分布。"),
        ("Record the expansion point and trust region. Local linearity is not global linearity.", "记录展开点与信赖域。局部线性不等于全局线性。"),
        ("Local tools and explicit bounded MH proposals implemented", "局部工具及显式有预算限制的 MH 提议已实现"),
    ),
    (
        ("Structured proposal + correction", "结构化提议与校正"),
        ("A linear, log-linear or local solve is inexpensive.", "线性、对数线性或局部求解成本低。"),
        ("MH-corrected block updates; SNIS as a separate weighted-estimation route.", "经 MH 校正的块更新；SNIS 为独立的加权估计路径。"),
        ("MH needs the full target and forward/reverse proposal densities. SNIS needs a known source density and retained weights.", "MH 需要完整目标及正反向提议密度。SNIS 需要已知的来源密度，并须保留权重。"),
        ("Explicit bounded proposal sweeps implemented; optional NUTS remainder", "显式且有预算限制的提议轮转已实现；可选 NUTS 剩余块"),
    ),
]


STEPS = [
    (
        ("Collect the complete conditional", "收集完整条件分布"),
        ("Fix the complement; collect every density factor involving the candidate parameters.", "固定其余参数；收集涉及候选参数的全部密度因子。"),
        ("Include observations, priors, joint priors and descendant latent distributions.", "包括观测、先验、联合先验及后代潜变量分布。"),
    ),
    (
        ("Find the structure", "寻找结构"),
        ("Label mean linearity, log-linearity, covariance dependence and support separately.", "分别标记均值线性、对数线性、协方差依赖及支持域。"),
        ("Prefer operator identities; numeric probes at valid points are finite evidence, not global proofs.", "优先使用算子恒等式；有效点上的数值探测是有限证据，并非全局证明。"),
    ),
    (
        ("Form compatible groups", "组成相容参数块"),
        ("Merge only when the union retains the required conditional structure.", "仅当合并后仍满足所需条件结构时，才组成一块。"),
        ("Check cross-dependence: a × b is linear in either variable separately, but not jointly. A vector can stay one block.", "检查交叉依赖：a × b 对每个变量分别线性，但不联合线性。向量可以整体作为一块。"),
    ),
    (
        ("Compare viable plans", "比较可行方案"),
        ("Apply prior/target requirements; assess joint identifiability, coupling and cost.", "应用先验与目标要求；检查联合可识别性、块间耦合及成本。"),
        ("Keep rejected and unavailable methods with reasons. Regrouping cannot repair a likelihood degeneracy.", "保留被拒绝和不可用的方法及其原因。重新分块不能修复似然退化。"),
    ),
    (
        ("Set the update schedule", "确定更新顺序"),
        ("Choose joint, conditional or collapsed updates, then freeze the production target and policy.", "选择联合、条件或边缘化更新，然后固定正式运行的目标与策略。"),
        ("Block count is not iteration count. Solver tolerances, fixed-point change or sampling diagnostics determine work.", "块数不等于迭代次数。工作量由求解容差、不动点变化或采样诊断决定。"),
    ),
]


DIAGNOSTICS = [
    (("D0 · Model and task", "D0 · 模型与任务"), ("Record data, factors, priors, target and required checks.", "记录数据、因子、先验、目标及必需检查。")),
    (("D1 · Valid domain", "D1 · 有效域"), ("Check shapes, support, transformations and covariance at valid anchors.", "在有效锚点检查形状、支持域、变换与协方差。")),
    (("D2 · Structure", "D2 · 结构"), ("Find conditional linearity, log-linearity, noise dependence and factor coverage.", "寻找条件线性、对数线性、噪声依赖并核对因子覆盖。")),
    (("D3 · Geometry", "D3 · 几何"), ("Check joint then conditional geometry, using supported quantities and recorded points.", "使用受支持的量和已记录的点，先检查联合几何，再检查条件几何。")),
    (("D4 · Prior", "D4 · 先验"), ("Check support, task normalization and supported sensitivity; retain the declared prior.", "检查支持域、任务归一化要求及受支持的敏感度；保留声明的先验。")),
    (("D5 · Selection", "D5 · 选择"), ("Validate target, correction, approximation and cost; freeze the retained-sample policy.", "验证目标、校正、近似与成本；固定保留样本阶段的策略。")),
    (("D6 · Execution", "D6 · 执行"), ("Monitor solve residuals, outer change, acceptance and applicable sampler diagnostics.", "监控求解残差、外层变化、接受率及适用的采样诊断。")),
    (("D7 · Evaluation", "D7 · 评估"), ("Report predictive checks, approximation error and simulated-truth recovery separately.", "分别报告预测检查、近似误差和模拟真值恢复。")),
]


try:
    from .case_content import CASES
except ImportError:  # Standalone plot_results.py
    from case_content import CASES

EXAMPLES = [(key, c["subtitle"], c["formula"], c["model"], c["structure"])
            for key, c in CASES.items()]


def method_cards():
    cards = []
    for index, (method, model) in enumerate(zip(METHODS, METHOD_MODELS, strict=True), 1):
        title, fit, result, guard, state = method
        model_label, equation = model
        uses_proposal = index in {5, 6, 7, 8}
        badge = (
            '<span class="proposal-mark">' + bi("Methods with proposals", "含提议机制的方法") + '</span>'
            if uses_proposal else ''
        )
        cards.append(
            f'<section class="method-card{" method-with-proposal" if uses_proposal else ""}" id="method-{index}"><header><span class="method-number">{index:02d}</span>'
            f'<h3>{bi(*title)}</h3>{badge}</header><div class="method-model"><span>{bi(*model_label)}</span>{math(equation)}</div><dl><dt>{bi("Fits", "适用")}</dt><dd>{bi(*fit)}</dd>'
            f'<dt>{bi("Returns", "返回")}</dt><dd>{bi(*result)}</dd></dl>'
            f'<p class="method-guard">{bi(*guard)}</p><footer>{bi(*state)}</footer></section>'
        )
    return '<div class="method-grid">' + "".join(cards) + "</div>"


def next_chapter(chapter, en, zh):
    return f'<div class="chapter-footer"><button type="button" data-design-go="{chapter}">{bi(en, zh)}</button></div>'


def methods():
    result = '<section id="design-methods" data-design-panel="methods">'
    result += '<h2>' + bi("Eight basic methods, different conditions.", "八类基础方法，各有适用条件。") + '</h2>'
    result += paragraph("Numbers identify the methods; they are not execution order. Match structure and requested result before comparing speed.", "编号用于引用方法，不代表执行顺序。先匹配结构与所需结果，再比较速度。", "chapter-lede")
    result += paragraph("β denotes the current block; c is held fixed. X_c, r_c, Σ_c and V_c are fixed with respect to β. Priors and all other factors must satisfy the stated conditional requirements.", "β 表示当前块，c 保持固定。X_c、r_c、Σ_c、V_c 相对于 β 固定。先验与其余因子须满足卡片所列的条件要求。", "model-notation")
    result += '<aside class="proposal-legend" aria-label="Proposal methods / 提议方法"><strong>' + bi("Methods with proposals", "含提议机制的方法") + '</strong>'
    result += paragraph("5 / 6 / 7 can construct a proposal; 8 supplies the correction. Alternatively, importance reweighting is also supported on eligible routes. Reweighting needs the source density, retained weights and weight diagnostics. A proposal must support random draws and density evaluation. The solves can also return standalone point estimates or approximations.", "5 / 6 / 7 可构造提议，8 提供校正。也可在支持的路径上使用重要性重加权。重加权需要来源密度、保留权重及权重诊断。提议须支持随机抽样和密度计算。这些求解也可独立返回点估计或近似结果。") + '</aside>'
    result += paragraph("Here, linear allows a fixed offset: Xβ + r. The mathematical term is affine; Xβ is the zero-offset case.", "这里的线性允许固定偏移：Xβ + r。数学术语为仿射；Xβ 是偏移为零的情形。", "design-footnote")
    result += method_cards()
    result += detail(
        ("GLS: unbiasedness, convergence and MAP", "GLS：无偏、收敛与 MAP"),
        paragraph("With fixed weights, fixed full-rank design and mean-zero errors, unpenalized GLS is unbiased. Estimated weights generally lose that finite-sample guarantee; a Gaussian prior adds shrinkage. An unbiased estimating equation is not the same as an unbiased estimator.", "权重固定、设计固定且满秩、误差均值为零时，无惩罚 GLS 无偏。估计得到的权重一般不再保证有限样本无偏；高斯先验还会引入收缩。估计方程无偏不等于估计量无偏。")
        + paragraph("Current iterative GLS freezes Σ at each solve. For Σ(β), the full Gaussian objective contains both the residual quadratic and log|Σ(β)|, and differentiates both. Convergence of the frozen-weight updates alone does not solve that objective.", "当前迭代 GLS 在每次求解时冻结 Σ。若 Σ(β) 随参数变化，完整高斯目标包含残差二次项和 log|Σ(β)|，并对两项都求导。仅冻结权重的更新收敛，并不等于求解了这个目标。")
        + math(r"y_i\sim N(\mu,f^2\mu^2):\quad \hat\mu_{GLS}=\bar y,\quad f^2\hat\mu_{MLE}^2+\bar y\hat\mu_{MLE}-\overline{y^2}=0")
        + paragraph("For independent observations, every unconstrained GLS step returns the sample mean: it is unbiased and already converged, although it can leave μ > 0. On datasets with a positive sample mean, this point generally still differs from the full-likelihood optimum. For n > 1 and nonzero data, a flat prior on μ > 0 gives a proper posterior with that optimum as MAP.", "独立观测下，每次无约束 GLS 都返回样本均值：它无偏且已经收敛，但可能落到 μ > 0 之外。即便数据的样本均值为正，该点一般仍不同于完整似然的最优点。当 n > 1 且数据不全为零时，μ > 0 上的平坦先验给出可归一化后验，该最优点也是 MAP。")
        + '<p class="reference-links"><a href="https://www.mathworks.com/help/econ/fgls.html">GLS / feasible GLS reference</a></p>',
        "gls-estimation",
    )
    result += '<h3 class="section-title">' + bi("One calculation can serve different targets.", "同一种计算，可以服务不同目标。") + '</h3>'
    result += '<nav class="goal-switch" aria-label="Statistical goal">'
    for key, en, zh in [
        ("estimate", "Fast estimate", "快速点估计"),
        ("original", "Original posterior", "原模型后验"),
        ("surrogate", "Approximate posterior", "近似后验"),
    ]:
        result += f'<button type="button" data-goal="{key}" aria-controls="goal-{key}" aria-pressed="{str(key == "estimate").lower()}">{bi(en, zh)}</button>'
    result += '</nav>'
    for key, en, zh in [
        ("estimate", "GLS/log-WLS returns an estimate when the inner solve and outer updates converge. MAP additionally requires optimality for the full likelihood and prior.", "GLS/log-WLS 在内层求解及外层更新收敛后返回估计。MAP 还要求满足完整似然与先验的最优性条件。"),
        ("original", "Use GCR directly for Gaussian conditionals. Approximate proposals need a valid original-target correction, such as MH. Every block retains the declared likelihood and prior.", "高斯条件分布直接使用 GCR。近似提议需要有效的原目标校正，例如 MH。每个块都保留声明的似然与先验。"),
        ("surrogate", "Declare one surrogate joint density and update every block against it. Record approximation error separately; unrelated approximate conditionals need not share a joint target.", "声明一个替代联合密度，并使所有块针对它更新。单独记录近似误差；各自近似的条件分布未必对应同一联合目标。"),
    ]:
        result += f'<div id="goal-{key}" data-goal-panel="{key}" class="target-explanation">{bi(en, zh)}</div>'
    result += paragraph("Explicit proposal composition is implemented. For otherwise all-NUTS models with parameter-dependent Gaussian noise, the compiler also tests bounded log-linear and GLS proposals automatically. Existing matrix-free GCR routes retain priority; cost-based candidate comparisons remain planned.", "显式提议组合已实现。对原本全部进入 NUTS、且高斯噪声随参数变化的模型，编译器也会自动检查有预算限制的 log-linear 和 GLS 提议。现有无需显式矩阵的 GCR 路径保持优先；基于成本的候选比较仍待实现。", "design-footnote")
    result += paragraph("MH defaults to ON in each explicit proposal policy. Set mh_correction=False to turn it off: the resulting schedule is marked approximate and keeps support/numerical guards. An unadjusted state-dependent chain need not have the original or a common surrogate posterior; convergence diagnostics do not measure that bias.", "显式提议策略中的 MH 默认开启。设置 mh_correction=False 可关闭；所得调度标记为近似，并保留支持域及数值检查。未经校正的状态依赖链未必对应原后验或共同的近似后验；收敛诊断不衡量这种偏差。", "design-note")
    return result + next_chapter("diagnostics", "Next: find the blocks", "下一步：寻找参数块") + '</section>'


def diagnostics():
    result = '<section id="design-diagnostics" data-design-panel="diagnostics"><h2>'
    result += bi("Given a graph, find compatible parameter groups.", "给定一张图，寻找相容的参数组。") + '</h2>'
    result += paragraph("Structure first, method eligibility second. Bounded posterior preflight now runs automatically; general method selection remains planned.", "先寻找结构，再判断方法资格。后验任务现在自动执行有预算限制的预检查；通用方法选择仍待实现。", "chapter-lede")
    result += paragraph("Trace DAG dependencies → inspect primal operations → differentiate and probe. The Gaussian flatness certificate checks affine means and complete covariance dependence, independently of priors and sampler selection. Declarations and zero derivatives alone are not proof. Unknown operations or custom derivatives remain unresolved; known nonlinear operations retain numerical checks.", "追踪 DAG 依赖 → 检查原始运算 → 微分与数值探测。高斯平坦性证书检查仿射均值及完整协方差依赖，独立于先验和采样器选择。声明或零导数本身不是证明。未知运算或自定义导数保持未确定；已知非线性运算仍使用数值检查。", "design-note")
    result += '<ol class="partition-flow">'
    for index, (title, action, reason) in enumerate(STEPS, 1):
        result += f'<li><span class="flow-number">{index}</span><div><h3>{bi(*title)}</h3>{paragraph(*action)}{paragraph(*reason, css="flow-reason")}</div></li>'
    result += '</ol><div class="grouping-demo">'
    result += '<div><span class="block-token">β = (β₁, …, β₅₀₀)</span>' + paragraph("One vector block when joint structure is compatible.", "联合结构相容时，一个向量可作为一块。") + '</div>'
    result += '<div><span class="block-token">a × b</span>' + paragraph("Two conditional linear candidates; their union is not linear.", "两个条件线性候选；合并后不再线性。") + '</div></div>'
    result += detail(
        ("Where D0–D7 fit", "D0–D7 如何嵌入流程"),
        '<dl class="diagnostic-list">' + "".join(f'<dt>{bi(*title)}</dt><dd>{bi(*body)}</dd>' for title, body in DIAGNOSTICS) + '</dl>'
        + paragraph("Checks needing a fit use a budgeted provisional fit, then repeat local analysis. Simulation truth is reserved for recovery checks.", "需要拟合点的检查先使用有预算限制的初步拟合，再重做局部分析。模拟真值只用于恢复检查。")
        + paragraph("Unsupported, skipped, failed and stale are distinct states. Missing evidence is not a pass; required checks stay required.", "不支持、跳过、失败与过期是不同状态。缺失证据不算通过；必需检查始终保持必需。"),
        "diagnostic-contract",
    )
    result += detail(
        ("What each selected block must explain", "每个被选中的块必须说明什么"),
        paragraph("Parameters and dimension → conditioning values and covered factors → structural evidence → eligible/rejected/unavailable methods → selection reason → actual execution → diagnostics and result target.", "参数与维度 → 条件值与覆盖因子 → 结构证据 → 可用、被拒绝及未实现的方法 → 选择原因 → 实际执行 → 诊断与结果目标。")
        + paragraph("A local mean-Jacobian rank test is not a full-likelihood identifiability proof. Variance-only information and joint degeneracies need their own checks.", "局部均值 Jacobian 的秩检查不是完整似然可识别性的证明。仅由方差提供的信息和联合退化需要各自的检查。"),
        "block-report-contract",
    )
    return result + next_chapter("priors", "Next: check the prior", "下一步：检查先验") + '</section>'


def priors():
    result = '<section id="design-priors" data-design-panel="priors"><h2>'
    result += bi("Per-block Jeffreys check. User-defined prior.", "逐块检查 Jeffreys，使用用户指定的先验。") + '</h2>'
    result += paragraph("For each block, assess Jeffreys versus flat in its named coordinates. Record the result alongside the supplied prior and method eligibility.", "对每个块，在指定坐标下判断 Jeffreys 是否平坦，并与用户先验、方法资格一同记录。", "chapter-lede")
    result += paragraph("The six baseline demos use bounded Uniform priors for all top-level parameters, while retaining the declared conditional Gaussian group and process laws. Demo 06 uses one dataset with 3,840 observations. Uniform density is flat inside its declared box in the named coordinates; this depends on the coordinates and does not imply a flat Jeffreys density.", "六个原始 demo 的全部顶层参数使用有界 Uniform 先验，同时保留已声明的条件高斯群组与过程分布。例子 06 使用一组包含 3,840 个观测的数据。Uniform 密度在指定坐标的声明区间内平坦；这一性质依赖坐标，也不意味着 Jeffreys 密度平坦。", "design-note")
    result += paragraph("Box boundaries truncate an otherwise Gaussian full conditional. An ordinary GCR draw then fails the support requirement, so the current compiler may select NUTS even when the mean is linear. Keep the declared bounds in every target evaluation.", "区间边界会把原本的高斯完整条件分布截断。普通 GCR 抽样此时不满足支持域要求，因此即使均值线性，当前编译器也可能选择 NUTS。每次目标求值都须保留声明的边界。", "design-footnote")
    result += '<div class="prior-checks">'
    for title, body in [
        (("Retain & validate", "保留并验证"), ("Check support, overlap and task-specific normalization. Gaussian, positive and flat priors can permit different updates.", "检查支持域、重叠及任务要求的归一化。高斯、正值和平坦先验可能允许不同更新。")),
        (("Explain influence", "解释影响"), ("Report supported sensitivity and identifiability checks. An informative prior moving the fit is not automatically wrong.", "报告受支持的敏感度与可识别性检查。信息性先验改变拟合并不自动意味着错误。")),
        (("Check every block", "逐块判断"), ("Assess applicability, then information rank and determinant dependence within budget. Report flat, non-flat, undefined or unresolved, with conditioning values.", "先判断适用性，再在预算内检查信息秩与行列式依赖；记录平坦、非平坦、未定义或未确定，以及条件值。")),
    ]:
        result += '<div><h3>' + bi(*title) + '</h3>' + paragraph(*body) + '</div>'
    result += '</div><div class="prior-correction"><h3>'
    result += bi("Criterion: a constant information determinant.", "判据：信息行列式为常数。") + '</h3>'
    result += paragraph("In a regular full-rank model, Jeffreys is flat when det I(θ) is a positive constant in the named coordinates. Compute information from the specified likelihood, separately from the user prior. A constant matrix is sufficient, not necessary.", "对正则且信息满秩的模型，若 det I(θ) 在指定坐标下为正常数，Jeffreys 即平坦。信息由指定似然计算，与用户先验分开。信息矩阵恒定是充分条件，并非必要条件。")
    result += math(r"\pi_J(\theta)\propto\sqrt{\det I(\theta)}")
    result += paragraph("Autodiff supplies local derivatives. Expected Fisher information still averages over data under the specified likelihood; the negative log-likelihood Hessian at the observed data is not generally that expectation. Equal determinants at two probe points do not prove global flatness.", "自动微分提供局部导数。期望 Fisher 信息仍须对指定似然下的数据取期望；观测数据处的负对数似然 Hessian 一般不等于该期望。两个探测点的行列式相等，不能证明全局平坦。")
    result += paragraph("When analytic marginalisation is unavailable, keep random instances as latent blocks and sample the full joint model with their declared conditional priors. This route needs no marginal Fisher. Missing marginal Fisher affects only that optional Jeffreys-prior construction. Gaussian-prior sensitivity is inapplicable to Uniform boundaries; boundary sensitivity remains a separate check.", "无法解析边缘化时，将随机实例保留为潜变量块，连同声明的条件先验采样完整联合模型。这条路径无需边缘 Fisher。缺少边缘 Fisher 只影响对应的可选 Jeffreys 先验构造。高斯先验敏感度检查不适用于 Uniform 边界；边界敏感性需要单独检查。", "design-note")
    result += paragraph("For a canonical Gaussian likelihood, a supported primal-operation certificate plus full numerical rank and consistent probes establishes flatness in the current block’s model coordinates, holding the recorded complement fixed. A bounded Uniform prior or a NUTS route does not erase that certificate. Numerical contradictions and unsupported derivative semantics remain visible.", "对标准高斯似然，受支持的原始运算证书加上数值满秩和一致探测，可确立当前块在模型坐标中的平坦性，并固定记录的其余参数值。有界 Uniform 先验或 NUTS 路径不会抹去该证书。数值矛盾及不支持的导数语义仍明确记录。")
    result += math(r"I(\theta)=E_{y\mid\theta}\!\left[s_\theta(y)s_\theta(y)^\top\right],\quad s_\theta(y)=\nabla_\theta\log L(y\mid\theta)")
    result += paragraph("For hierarchical hyperparameters, marginal-likelihood information requires integrating out the latent process. Holding one process realization fixed gives conditional information. A direct-observation check cannot substitute for the marginal calculation.", "对层级模型的超参数，边缘似然信息需要先积分掉潜在过程。固定某次过程实现时，得到的是条件信息。直接观测层的检查不能替代边缘计算。")
    result += paragraph("For a hierarchy such as demo 06, an optional Jeffreys prior for the root parameters φ can retain the declared process law s | φ. Its defining likelihood averages over the unobserved instance s. Gibbs sampling can still keep s explicit. Conditional and complete-data Fisher define different information quantities.", "对于例子 06 这样的层级模型，可选的顶层参数 φ 的 Jeffreys 先验可以保留已声明的过程分布 s | φ。定义该先验的似然对未观测实例 s 取平均。Gibbs 采样仍可显式保留 s。条件 Fisher 与完整数据 Fisher 是不同的信息量。")
    result += math(r"p(y\mid\phi)=\int p(y\mid s,\phi)\,p(s\mid\phi)\,ds")
    result += '</div><div class="jeffreys-comparison">'
    for title, equation, conclusion in [
        (("Fixed-noise linear Gaussian", "固定噪声的线性高斯"), r"I_\beta=X^\top\Sigma^{-1}X", ("Flat in β for fixed full-rank X and fixed positive-definite Σ. A non-flat Gaussian user prior is still valid.", "当 X 固定满列秩且 Σ 固定正定时，对 β 平坦。用户仍可使用非平坦高斯先验。")),
        (("Exact Gaussian law for log y", "log y 的精确高斯观测律"), r"I_\beta=X^\top V^{-1}X,\quad a=e^\beta", ("Flat in β for fixed full-rank X and fixed V; not flat in a, where the Jacobian contributes ∏ 1/aⱼ.", "当 X 固定满列秩且 V 固定时，对 β 平坦；对 a 则不平坦，因为 Jacobian 带来 ∏ 1/aⱼ。")),
        (("Linear mean, changing noise", "线性均值、变化噪声"), r"y\sim\mathcal N(\theta,f^2\theta^2),\quad I_\theta=\frac{f^{-2}+2}{\theta^2}", ("For known f > 0 and θ > 0, Jeffreys ∝ 1/θ. The mean supports iterative GLS while the Jeffreys density is non-flat.", "已知 f > 0 且 θ > 0 时，Jeffreys ∝ 1/θ。均值支持迭代 GLS，而 Jeffreys 密度非平坦。")),
    ]:
        result += '<section><h3>' + bi(*title) + '</h3>' + math(equation) + paragraph(*conclusion) + '</section>'
    result += '</div>'
    result += paragraph("GCR eligibility is conditional: hold c fixed while updating β. X_c and Σ_c may change with c while remaining fixed with respect to β. With full-column-rank X_c and positive-definite Σ_c, this Gaussian likelihood has flat conditional Jeffreys density in β; that does not establish flatness of the joint or marginal model.", "GCR 资格是条件性的：更新 β 时固定 c。X_c 与 Σ_c 可以随 c 改变，同时相对于 β 保持固定。若 X_c 满列秩且 Σ_c 正定，该高斯似然对 β 的条件 Jeffreys 密度平坦；这不能证明联合模型或边缘模型的 Jeffreys 平坦。", "design-note")
    result += paragraph("Keep the supplied prior in every update. A non-flat Jeffreys result is recorded as a model property; adopting that prior requires an explicit user declaration. Flatness alone does not certify identifiability or posterior propriety.", "每次更新都保留用户先验。Jeffreys 非平坦作为模型性质记录；采用它作为先验需要用户明确声明。仅有平坦性不能证明可识别性或后验可归一化。", "design-note")
    result += paragraph("The notebook retains Jeffreys/flatness diagnostics without separate Jeffreys demos. Demo 06 shows one 3,840-observation dataset with bounded Uniform root priors. Its recovery result and chain diagnostics are reported separately.", "Notebook 保留 Jeffreys／平坦性诊断，不再展示单独的 Jeffreys demo。例子 06 展示一组包含 3,840 个观测的数据，顶层参数采用有界 Uniform 先验。恢复结果与链诊断分别报告。", "design-note")
    result += detail(
        ("Planned · systematic bias diagnostics and reduction", "待实现 · 系统性偏差诊断与减偏"),
        paragraph("Specify the scientific target and estimator first: posterior mean, MAP and derived quantities have different biases. Check identification, boundaries and nuisance parameters; then select an applicable analytic correction, reference prior or simulation calibration. No method guarantees unbiased estimates for every model.", "先指定科学目标和估计量：后验均值、MAP 与派生量具有不同偏差。检查可识别性、边界和干扰参数，再选择适用的解析修正、参考先验或模拟校准。任何方法都不保证对所有模型无偏。")
        + paragraph("Keep one declared joint target independent of computational blocks. Prior changes require an explicit choice. Validate candidates on held-out simulations across parameters and sample sizes, reporting bias with uncertainty, RMSE, coverage and cost. For population parameters in demo 06, redraw both the process instance and observation noise.", "保持一个不依赖计算分块的已声明联合目标。改变先验需要明确选择。跨参数与样本规模，用留出的模拟验证候选方法，报告偏差及其不确定性、RMSE、覆盖率和成本。对例子 06 的总体参数，需要同时重新抽取过程实例与观测噪声。")
        + paragraph("Record a calibrated point estimate separately from posterior samples. Distinguish Monte Carlo error from repeated-simulation bias. This general workflow is a design proposal. The single dataset shown in demo 06 illustrates recovery, not repeated-experiment bias or coverage.", "将校准后的点估计与后验样本分别记录。区分 Monte Carlo 误差和重复模拟意义下的偏差。这一通用流程属于设计建议。例子 06 展示的单组数据用于演示恢复，不用于判断重复实验意义下的偏差或覆盖率。"),
        "bias-reduction-design",
    )
    result += '<div class="prior-timing"><section><h4>' + bi("Before sampling · diagnose the form", "采样前 · 判断形式") + '</h4>'
    result += paragraph("Run applicability and flatness analysis before warmup, using valid initialization points where numerical checks need them. Reuse valid evidence during the run; model, prior, data or coordinate changes invalidate affected reports.", "在预热前完成适用性与平坦性分析；数值检查需要时使用有效初始点。运行中复用有效证据；模型、先验、数据或坐标改变时，相关报告失效。")
    result += '</section><section><h4>' + bi("During sampling · evaluate the chosen density", "采样中 · 计算所选密度") + '</h4>'
    result += paragraph("An explicitly selected non-flat Jeffreys prior remains state dependent. Evaluate its log density, and gradients when needed, at every target evaluation. The preflight report does not freeze that density at the initial point.", "显式选用的非平坦 Jeffreys 先验仍随状态变化。每次目标求值都计算其对数密度，需要时计算梯度。运行前的报告不会把该密度冻结在初始点。")
    result += '</section></div>'
    result += detail(
        ("Scope, implementation and references", "范围、实现与参考"),
        paragraph("Computational regrouping must preserve the declared joint prior. Conditional blockwise Jeffreys priors are generally not the full joint Jeffreys prior. Finite probes cannot prove global flatness.", "计算分块改变时必须保留已声明的联合先验。逐块条件 Jeffreys 一般不等于完整联合 Jeffreys。有限探测不能证明全局平坦。")
        + paragraph("PosteriorTask schedules bounded geometry, Jeffreys/flatness and prior-sensitivity checks. Fisher calculations support observed Gaussian, eligible Bernoulli and fixed-cutoff Pareto factors; prior sensitivity has separate location-only support and can remain unresolved. Unsupported or over-budget checks are recorded, preserving the declared priors. Other observation families and general hierarchical marginal Fisher geometry remain planned.", "PosteriorTask 安排有预算限制的几何、Jeffreys/平坦性及先验敏感度检查。Fisher 计算支持高斯观测、符合条件的 Bernoulli 因子及固定下限的 Pareto 因子；先验敏感度有单独的支持范围，目前仅处理位置参数，也可能保持未确定。不支持或超预算的检查会被记录，并保留声明的先验。其他观测分布族与通用层级边缘 Fisher 几何仍待实现。")
        + '<p class="reference-links"><a href="https://arxiv.org/abs/1108.2120">Ghosh (2011)</a> · <a href="https://mc-stan.org/docs/stan-users-guide/reparameterization.html">'
        + bi("Stan: coordinate densities and Jacobians", "Stan：坐标密度与 Jacobian") + '</a></p>',
        "prior-scope",
    )
    return result + next_chapter("sampling", "Next: sampling & summaries", "下一步：采样与汇总") + '</section>'


def sampling():
    result = '<section id="design-sampling" data-design-panel="sampling"><h2>'
    result += bi("Initialize. Sweep. Check. Summarize.", "初始化，轮转更新，检查，汇总。") + '</h2>'
    result += paragraph("PosteriorTask implements automatic preflight, bounded per-chain initialization, checkpoint stopping and explicit bounded proposal/MH composition with an optional NUTS remainder.", "PosteriorTask 已实现自动预检查、有预算限制的逐链初始化、检查点停止，以及显式且有预算限制的提议/MH 组合，并可附带 NUTS 剩余块。", "chapter-lede")
    result += paragraph("Use ProposalBlockPolicy with iterative_gls, bias_corrected_log_linear or gauss_newton, then pass proposal_options(...) to PosteriorTask.backend_options. The default per-block dense limits are 64 parameters and 200,000 matrix elements. Omitting these options preserves automatic selection and existing matrix-free routes.", "使用 ProposalBlockPolicy 指定 iterative_gls、bias_corrected_log_linear 或 gauss_newton，再把 proposal_options(...) 传入 PosteriorTask.backend_options。每块稠密计算的默认上限为 64 个参数、200,000 个矩阵元素。省略这些选项时，保留自动选择及现有无需显式矩阵的路径。", "design-note")
    result += paragraph("Proposal members are small real continuous blocks with supported Normal or Uniform priors and Gaussian observation fits. Log proposals additionally support canonical LogNormal data and check positivity/small fractional noise for multiplicative Normal data. Other factors stay in the MH target. Recorded acceptance and rejection counts cover production draws, excluding warmup.", "提议成员为小规模实连续参数块，使用受支持的 Normal 或 Uniform 先验及高斯观测拟合。对数提议还支持标准 LogNormal 数据，并检查乘性 Normal 数据的正值与小相对噪声条件。其他因子保留在 MH 目标中。接受与拒绝计数覆盖正式样本，不含预热。", "design-footnote")
    result += '<ol class="sampling-lifecycle">'
    for title in [("Initial state", "初始状态"), ("Warmup", "预热"), ("Retained sweeps", "保留轮转样本"), ("Diagnostics", "诊断"), ("Summary", "汇总")]:
        result += '<li>' + bi(*title) + '</li>'
    result += '</ol><h3 class="section-title">' + bi("A simple update order", "简单的更新顺序") + '</h3>'
    result += '<div class="sampling-pair"><section><h4>' + bi("User-specified", "用户指定") + '</h4>'
    result += paragraph("Use the requested order exactly, once validated against the selected plan. List each active block once per sweep; collapsed blocks need no update. Report unsupported schedules instead of silently reordering.", "在选定方案上验证后，严格使用指定顺序。每轮列出每个活动块一次；已边缘化的块无需更新。不支持的顺序应报告，不能悄悄重排。")
    result += '</section><section><h4>' + bi("Default", "默认") + '</h4>'
    result += paragraph("Use the compiler's stable block order. Each update reads the latest values of every other block. Save one full state after a complete sweep.", "使用编译器给出的稳定块顺序。每次更新读取其余块的最新值。完整一轮结束后保存一个全状态样本。")
    result += '</section></div>'
    result += math(r"b_1^{(t+1)}\sim\pi(b_1\mid b_2^{(t)},y),\qquad b_2^{(t+1)}\sim\pi(b_2\mid b_1^{(t+1)},y)")
    result += paragraph("GCR gives Gaussian conditional Gibbs updates; MH or NUTS gives Metropolis/HMC within Gibbs. A fixed order preserves a common target when every kernel does; reachability and mixing still need checks. Explicit, seeded random scans remain planned.", "GCR 给出高斯条件 Gibbs 更新；MH 或 NUTS 构成 Gibbs 内的 Metropolis/HMC 更新。每个核均保持同一目标时，固定顺序也保持该目标；仍需检查可达性与混合。显式指定、带种子的随机扫描仍待实现。", "design-footnote")
    result += paragraph("A rejected MH proposal leaves this block unchanged; continue to the next block without undoing earlier updates. Rejections count toward any configured fixed number of inner steps. Retain the full state after each sweep, including repeats; do not retry until acceptance.", "MH 拒绝后保留该块原值，继续下一块，本轮先前更新不回滚。拒绝计入预设的固定内部步数。每轮保留完整状态，包括重复状态；不要重试到接受为止。", "design-note")
    result += '<h3 class="section-title">' + bi("Complete the initial state before warmup", "预热前补全初始状态") + '</h3>'
    result += paragraph("For coupled, hierarchical or unknown-scale models, we strongly recommend sensible initial values, supplied directly or obtained from an explicitly recorded fit using observed data only. Choose valid states away from hard boundaries and vary starting states across chains. Automatic prior/support candidates only verify a finite valid state; they do not guarantee successful warmup or mixing. Multiple-chain diagnostics, including R-hat, ESS and divergences, remain necessary.", "对耦合、层级或尺度未知的模型，强烈建议提供合理初值，可直接给定，或通过明确记录、仅使用观测数据的拟合获得。选择远离硬边界的有效状态，并让不同链从不同状态出发。自动先验/支持域候选只验证状态有限且有效，不保证预热成功或混合良好。仍须检查多条链的 R-hat、ESS 和发散等诊断。", "design-note")
    result += '<ol class="initialization-rules">'
    for title, body in [
        (("Keep supplied values", "保留已给定的值"), ("Accept full or partial values per chain in named model coordinates, with an explicit missing-entry mask. Check shape, support and finite density/gradients. Invalid supplied values produce a precise error. Initial values are starting points, not fixed parameters.", "允许按链提供模型坐标下的完整或部分初值，并显式标记缺失条目。检查形状、支持域及有限密度/梯度。用户给定值无效时明确报错。初值是起点，不是固定参数。")),
        (("Fill only missing entries", "只补缺失条目"), ("Proper-prior draws supply candidates; priors without a sampler use support-bijector interior values. Keep supplied entries and validate the full joint density and gradients. Exact missing-subvector conditional draws remain planned.", "正规先验抽样提供候选；无法抽样的先验使用支持域变换生成内部值。保留用户条目，并验证完整联合密度和梯度。缺失子向量的精确条件抽样仍待实现。")),
        (("Stop a stalled initialization", "初始化无法推进时停止"), ("Try at most 32 candidates per chain by default; max_attempts is configurable. Failure reports the chain and affected latents before warmup. Automatic optimization-based initialization remains planned.", "默认每条链最多尝试 32 个候选，可配置 max_attempts。失败时在预热前报告链号及相关参数。自动优化初始化仍待实现。")),
    ]:
        result += '<li><strong>' + bi(*title) + '</strong>' + paragraph(*body) + '</li>'
    result += '</ol>'
    result += paragraph("Use separate reproducible random streams per chain. Record automatic values and their source; never use simulation truth to initialize a recovery demo.", "每条链使用独立、可复现的随机流。记录自动初值及其来源；恢复 demo 的初始化不能使用模拟真值。", "design-footnote")
    result += '<h3 class="section-title">' + bi("When does sampling stop?", "采样何时停止？") + '</h3>'
    result += '<nav class="stop-switch" aria-label="Stopping policy / 停止策略">'
    for key, title in [("fixed", ("Fixed budget", "固定预算")), ("checkpoints", ("Diagnostic checkpoints", "诊断检查点"))]:
        result += f'<button type="button" data-stop="{key}" aria-controls="stop-{key}" aria-pressed="{str(key == "fixed").lower()}">{bi(*title)}</button>'
    result += '</nav><div class="stop-panel" id="stop-fixed" data-stop-panel="fixed">'
    result += '<h4>' + bi("Default · current execution pattern", "默认 · 当前执行方式") + '</h4>'
    result += paragraph("Run the specified warmup and retained draws per chain, then evaluate diagnostics. Warmup tunes the sampler and is discarded. Reaching the draw count means the run finished; the diagnostic verdict is separate.", "按每链指定的预热数和保留样本数运行，然后评估诊断。预热用于调节采样器，不予保留。达到样本数表示运行结束；诊断结论另行报告。")
    result += '</div><div class="stop-panel" id="stop-checkpoints" data-stop-panel="checkpoints">'
    result += '<h4>' + bi("Optional · implemented checkpoint scheduler", "可选 · 已实现检查点调度") + '</h4>'
    result += paragraph("After a declared minimum run, check at fixed batch boundaries. Continue the same frozen kernel until required precision/diagnostic criteria hold at two consecutive checkpoints, or the draw/time cap is reached. Record the stopping rule and reason.", "达到预先声明的最短运行量后，在固定批次边界检查。继续使用同一冻结的核，直到所需精度与诊断条件连续两个检查点满足，或达到样本数/时间上限。记录停止规则与原因。")
    result += paragraph("A budget cap can stop an unresolved run. Repeated checks are an operational rule, not proof of convergence or an automatic confidence guarantee. Changed targets or adaptation require a new run and warmup.", "预算上限可以终止尚未获得结论的运行。反复检查是操作规则，并非收敛证明或自动置信保证。目标或适配策略改变时，需要新运行与预热。")
    result += '</div>'
    result += '<div class="sampling-checks">'
    for title, body in [
        (("Chain agreement", "链间一致性"), ("R-hat compares within- and between-chain behavior. Current checkpoints monitor every parameter coordinate; checks on derived scientific summaries remain planned. Between-chain checks require multiple chains.", "R-hat 比较链内与链间行为。当前检查点监控每个参数坐标；对派生科学汇总量的检查仍待实现。链间检查需要多条链。")),
        (("Monte Carlo precision", "Monte Carlo 精度"), ("Current checkpoints assess ESS and optional mean MCSE for every parameter coordinate. Set tolerances before the run. Precision criteria for quantiles and other summaries remain planned.", "当前检查点对每个参数坐标评估 ESS，并可选检查均值 MCSE。容差在运行前设置。分位数及其他汇总量的精度判据仍待实现。")),
        (("Kernel validity", "更新核的可靠性"), ("Current checkpoints check NUTS divergences; existing GCR solver guards still apply. General residual and acceptance criteria remain planned. Missing required checks remain unresolved.", "当前检查点检查 NUTS 发散，已有的 GCR 求解器检查继续生效。通用残差与接受率判据仍待实现。缺失的必需检查保持未确定。")),
    ]:
        result += '<section><h4>' + bi(*title) + '</h4>' + paragraph(*body) + '</section>'
    result += '</div>'
    result += paragraph("Independent whole-posterior GCR draws need no MCMC warmup or R-hat check. A chain with GCR blocks is still MCMC.", "直接生成整个后验的独立 GCR 样本无需 MCMC 预热或 R-hat 检查。包含 GCR 块的链仍然是 MCMC。", "design-footnote")
    result += '<h3 class="section-title">' + bi("Optional posterior summaries", "可选的后验汇总") + '</h3>'
    result += paragraph("Ordinary posterior draws use equal weights. Two different operations can improve or correct a summary:", "普通后验样本使用等权平均。以下两种操作可改善或校正汇总：")
    result += detail(
        ("Conditional averaging · Rao–Blackwellization", "条件平均 · Rao–Blackwell 化"),
        paragraph("When an exact conditional moment is available, average that moment over the other block's posterior draws instead of averaging one conditional draw per state.", "若精确条件矩可得，可对其他块的后验样本计算条件矩再平均，替代每个状态只用一次条件抽样来求平均。")
        + math(r"E[h(b,c)\mid y]=E_{c\mid y}\!\left[E[h(b,c)\mid c,y]\right]")
        + paragraph("For a Gaussian block with conditional mean m(c) and covariance V(c), report mean = average m(c) and covariance = average V(c) + covariance of m(c). Conditional means alone omit within-block uncertainty.", "对条件均值 m(c)、协方差 V(c) 的高斯块，均值为 m(c) 的平均，协方差为 V(c) 的平均加上 m(c) 的协方差。只使用条件均值会漏掉块内不确定性。")
        + paragraph("This is a summary step, not a replacement of Gibbs draws by deterministic means. Use moments of the actual target; a Gaussian proposal mean is not an exact conditional moment. Assess MCSE because autocorrelation affects MCMC gains.", "这是汇总步骤，不能用确定性均值替代 Gibbs 抽样。使用实际目标的条件矩；高斯提议的均值并非精确条件矩。自相关会影响 MCMC 中的收益，因此仍需评估 MCSE。"),
        "summary-rao-blackwell",
    )
    result += detail(
        ("Importance reweighting · self-normalized importance sampling (SNIS)", "重要性重加权 · 自归一化重要性抽样（SNIS）"),
        paragraph("Use when draws come from a known proposal or surrogate q and the desired target is π, with compatible support. Correct by a density ratio, not by multiplying posterior draws by their own posterior density.", "当样本来自已知提议或替代分布 q、目标为 π 且支持域相容时使用。用密度比校正，不能把后验样本再乘上其自身后验密度。")
        + math(r"w_s\propto\frac{\pi(\theta_s)}{q(\theta_s)},\qquad \widehat E_\pi[h]=\frac{\sum_s w_s h(\theta_s)}{\sum_s w_s}")
        + paragraph("Retain normalized weights and report concentration, Kish ESS and available tail diagnostics. Correlated source draws also need autocorrelation-aware uncertainty; weight ESS is not chain ESS.", "保留归一化权重，报告权重集中度、Kish ESS 和可获得的尾部诊断。来源样本相关时还需考虑自相关的不确定性；权重 ESS 不等于链 ESS。")
        + paragraph("Weights may be computed as samples arrive. They correct estimates; assigning a weight to an approximate block draw does not make it a valid Gibbs update. Reweighting complete states requires their known joint source density q; unrelated approximate conditionals do not supply it.", "权重可随样本生成而计算。它们校正估计；给近似块样本赋权，并不能使其成为有效的 Gibbs 更新。对完整状态重加权，需要其已知的联合来源密度 q；彼此独立构造的近似条件分布不能提供这一保证。")
        + paragraph("This is optional only when choosing a different estimation route. If correction is needed to target π, dropping the weights changes the result. Existing demos retain their original unweighted sampling routes.", "选择不同估计路径时可选此法。若以 π 为目标必须校正，丢弃权重就会改变结果。现有 demo 保留其原始等权采样路径。"),
        "summary-snis",
    )
    result += detail(
        ("Implemented API · inspectable records", "已实现 API · 可检查的记录"),
        paragraph("Use DiagnosticPolicy, InitializationPolicy and StoppingPolicy on PosteriorTask. AnalysisReport stores preflight findings; RunRecord stores actual initial values, checkpoints and stop reason. Masks use True for supplied entries; a leading chain dimension gives per-chain values.", "在 PosteriorTask 上配置 DiagnosticPolicy、InitializationPolicy 和 StoppingPolicy。AnalysisReport 保存预检查结果；RunRecord 保存实际初值、检查点和停止原因。mask 的 True 表示已给条目；首维 chain 表示逐链给值。")
        + '<pre><code>' + html.escape('task = PosteriorTask(\n    meta=new_task_meta(),\n    budget=ComputeBudget(draws=2000, warmup=1000, chains=2),\n    initialization=InitializationPolicy(max_attempts=32),\n    stopping=StoppingPolicy(\n        mode="checkpoints", min_draws=400, batch_size=200,\n        consecutive=2, mcse_mean=0.01,\n    ),\n)') + '</code></pre>'
        + paragraph("Fisher diagnostics currently cover observed Gaussian and supported Bernoulli factors and require float64. Required checks that are skipped or unresolved refuse compilation. Collapsed-chain checkpoint rules and explicit collapsed initial values remain unsupported; independent GCR/SNIS rejects chain-only quality and checkpoint/time-stop settings and uses its own draw budget.", "Fisher 诊断当前覆盖高斯观测与受支持的 Bernoulli 因子，要求 float64。必需检查若被跳过或未解决，会拒绝编译。边缘化链的检查点规则及显式初值暂不支持；独立 GCR/SNIS 拒绝仅适用于链的质量判据及检查点/时间停止配置，使用各自的抽样预算。"),
        "implemented-sampling-api",
    )
    result += sampling_settings()
    return result + next_chapter("examples", "Next: model examples", "下一步：模型例子") + '</section>'


def sampling_settings():
    rows = [
        (("Budget & seed", "预算与种子"), ("Warmup, retained draws, chains, random seed and sequential/parallel execution.", "预热数、保留样本数、链数、随机种子及串行/并行执行。"), ("Task budgets and execution keys exist; backend restrictions apply.", "Task 预算和执行 key 已有；受后端能力限制。")),
        (("Order & initial values", "顺序与初值"), ("Explicit block order; complete or partial per-chain initial values; bounded automatic completion.", "显式块顺序；逐链完整或部分初值；有预算限制的自动补全。"), ("Default: structured block then NUTS. Explicit proposal policies run in declaration order, then any NUTS remainder; partial per-chain values are validated. Unsupported schedules refuse compilation.", "默认结构块先于 NUTS。显式提议策略按声明顺序执行，再更新可能存在的 NUTS 剩余块；验证逐链部分初值。不支持的调度会在编译时拒绝。")),
        (("Kernel & solve settings", "更新核与求解配置"), ("Target acceptance, tree depth, solver tolerance and iteration cap, where supported.", "受支持时可设目标接受率、树深、求解容差和迭代上限。"), ("NUTS options exist on low-level APIs; arbitrary nuts_options are not accepted by PosteriorTask backend_options.", "NUTS 配置存在于底层 API；PosteriorTask 的 backend_options 不接受任意 nuts_options。")),
        (("Stopping & checks", "停止与检查"), ("Fixed budget by default; optional batch size, minimum run, maximum work and declared precision criteria.", "默认固定预算；可选批次大小、最短运行量、最大工作量和预先声明的精度判据。"), ("Implemented: continue the same warmed chain, require consecutive checks, record actual count/reason. Uses existing ESS/split-R-hat plus optional mean MCSE and stricter limits; time caps are checked between batches.", "已实现：续跑同一预热后的链，要求连续检查通过，记录实际数量/原因。沿用 ESS/split-R-hat，可附加均值 MCSE 及更严格限制；时间上限在批次之间检查。")),
        (("Summary", "汇总"), ("Equal-weight mean, supported conditional moments, or a justified importance-weighted estimate.", "等权均值、受支持的条件矩，或有依据的重要性加权估计。"), ("Weighted results and limited SNIS routes exist. A unified Rao–Blackwell summary option is proposed.", "加权结果与有限 SNIS 路径已存在。统一的 Rao–Blackwell 汇总选项待实现。")),
    ]
    body = '<div class="table-scroll sampling-settings"><table><thead><tr><th>' + bi("Setting", "配置") + '</th><th>' + bi("User choice", "用户可选") + '</th><th>' + bi("Current boundary", "当前边界") + '</th></tr></thead><tbody>'
    for name, choice, state in rows:
        body += '<tr><th scope="row">' + bi(*name) + '</th><td>' + bi(*choice) + '</td><td>' + bi(*state) + '</td></tr>'
    body += '</tbody></table></div>'
    body += paragraph("Thresholds belong to the named diagnostic implementation and its version. Never substitute a new cutoff into an existing report; declare any additional quality profile separately.", "阈值属于具体诊断实现及其版本。不能给已有报告替换新截止值；额外质量标准应单独声明。")
    body += '<p class="reference-links"><a href="https://mc-stan.org/learn-stan/diagnostics-warnings.html">' + bi("MCMC diagnostics", "MCMC 诊断") + '</a> · <a href="https://mc-stan.org/docs/2_32/stan-users-guide/marginalization-mathematics.html">' + bi("Conditional expectations", "条件期望") + '</a> · <a href="https://arxiv.org/abs/2101.01011">Rao–Blackwellization in MCMC</a></p>'
    return detail(("User settings & implementation boundary", "用户配置与实现边界"), body, "sampling-settings")


def examples(reports):
    if __package__:
        from .symbols import parameter_symbols
    else:
        from symbols import parameter_symbols
    loaded = {report["case"]: report for report, _ in reports}
    result = '<section id="design-examples" data-design-panel="examples"><h2>'
    result += bi("Model examples and their structural lessons.", "模型例子与结构要点。") + '</h2>'
    result += paragraph("Model summaries first. Saved compiler choices remain separate from mathematical candidates.", "先读模型简介。已保存的编译器选择与数学上的候选结构分别展示。", "chapter-lede")
    result += '<div class="example-summaries">'
    for index, (key, title, equation, model, structure) in enumerate(EXAMPLES, 1):
        report = loaded.get(key)
        if report and report.get("proposal_selection") == "automatic":
            structure = (
                "The compiler automatically selected the recorded proposal blocks with MH. Candidate tests and fallback reasons are saved with the run.",
                "编译器自动选择记录中的 MH 提议块。候选检查与后备原因随运行保存。",
            )
        elif report and report.get("proposal_policies"):
            uncorrected = any(not p.get("mh_correction", True) for p in report["proposal_policies"])
            structure = (
                "This saved comparison disables MH in at least one explicit proposal block. The full schedule is approximate; its recorded blocks are shown below.",
                "这项已保存的比较在至少一个显式提议块中关闭了 MH。整个调度为近似，下方展示其已记录的参数块。",
            ) if uncorrected else (
                "This saved comparison explicitly selects bounded Gaussian proposals with MH. Its actual block schedule is recorded below; the automatic model demos are separate.",
                "这项已保存的比较显式选择有预算限制的高斯提议与 MH。下方记录其实际参数块调度，自动模型 demo 单独保留。",
            )
        result += f'<section class="example-summary" id="example-summary-{key}"><header><span>{index:02d}</span><h3>{bi(*title)}</h3></header>'
        result += math(equation) + paragraph(*model) + paragraph(*structure, css="example-structure")
        if report:
            route = html.escape(report["method"].replace("gcr+mh", "GCR + iterative GLS + MH").replace("GCR+MH", "GCR + iterative GLS + MH"))
            result += f'<div class="saved-route"><span>{bi("Saved execution", "已保存的执行")}</span><code>{route}</code></div>'
            for block in report.get("blocking", {}).get("executed", {}).get("blocks", ()):
                result += '<p class="saved-block-symbols">' + parameter_symbols(key, block["latents"]) + ' · <code>' + html.escape(block["method"].replace("gcr+mh", "GCR + iterative GLS + MH")) + '</code></p>'
            if report.get("variant"):
                result += paragraph("This saved counterpart uses its explicitly stated settings; the model description above describes the original baseline. Open its run for the data and prior setup.", "这份已保存的对照使用明确注明的设置；上方模型说明描述原始基准。打开该运行可查看数据和先验设置。", "variant-note")
            result += f'<a class="design-example-link" href="#case={key}&amp;step=model">{bi("Open operators, DAG and recorded results", "打开算子、DAG 和已记录结果")}</a>'
        else:
            result += paragraph("No saved run loaded for this model.", "当前未载入此模型的运行记录。", "design-footnote")
        if key == "multiplicative_noise":
            result += detail(
                ("Inspect the two conditional blocks", "查看两个条件块"),
                '<div class="conditional-pair"><section><span class="block-token">p_g</span>'
                + paragraph("Fix pₙ: the mean is log-linear in the gain parameters.", "固定 pₙ：均值对增益参数呈对数线性。")
                + math(r"\log\mu=Up_g+\log(Ap_n)")
                + paragraph("A log-space approximation is useful; exp does not assign a LogNormal prior to p_g.", "可利用对数空间近似；exp 不会自动给 p_g 指定 LogNormal 先验。")
                + '</section><section><span class="block-token">p_n</span>'
                + paragraph("Fix p_g: the signal mean is linear, but its covariance changes.", "固定 p_g：信号均值线性，但其协方差变化。")
                + math(r"\mu=Mp_n,\quad M=\operatorname{diag}(e^{Up_g})A")
                + paragraph("Iterative GLS with a flat/Gaussian penalty is a point method. The current bounded Uniform prior also requires enforcing its bounds; ordinary Gaussian draws do not preserve that support.", "带平坦或高斯惩罚的迭代 GLS 是点估计方法。当前有界 Uniform 先验还要求遵守其边界；普通高斯抽样不能保持该支持域。")
                + '</section></div>'
                + paragraph("If U contains a constant gain direction, gain shifts can cancel signal rescaling. Check joint identifiability; conditional rank alone misses this.", "若 U 包含常数增益方向，增益平移可抵消信号缩放。需检查联合可识别性；仅查条件秩会漏掉它。"),
                "multiplicative-blocks",
            )
            result += scale_probe(report)
        result += '</section>'
    result += '</div>'
    return result + next_chapter("proofs", "Details & proofs", "说明与证明") + '</section>'


def scale_probe(report):
    probes = report.get("blocking_probes", []) if report else []
    probe = next((p for p in probes if any(
        prior.get("latent") == "p_n" and prior.get("coordinates") == 500
        for prior in p.get("blocking", {}).get("priors", [])
    )), None)
    body = ''
    if probe:
        blocking = probe["blocking"]
        body += paragraph("Loaded compilation record; not a new inference run.", "载入的编译记录；并非本次新运行的推断。")
        body += '<dl class="probe-record">'
        for name, en, zh in [("priors", "Declared priors", "已声明先验"), ("route", "Compiler selection", "编译器选择"), ("factor", "Factor comparison", "因子策略比较"), ("posterior", "Posterior run", "后验运行")]:
            if name == "priors":
                text = "; ".join(f'{p["latent"]}: {p["family"]} [{p["coordinates"]}]' for p in blocking["priors"])
            elif name == "route":
                selected = blocking.get("executed", {})
                text = " + ".join(f'{b["method"]} {{{", ".join(b["latents"])}}}' for b in selected.get("blocks", [])) or selected.get("status", "unrecorded")
                text = text.replace("gcr+mh", "GCR + iterative GLS + MH")
            elif name == "factor":
                comparison = blocking.get("factor_comparison", {})
                text = comparison.get("status", "unrecorded") + ": " + comparison.get("exception", "—")
            else:
                state = probe.get("posterior_executed")
                text = bi("Recorded as run" if state is True else "Not run" if state is False else "Not recorded", "记录为已运行" if state is True else "未运行" if state is False else "未记录")
            body += '<dt>' + bi(en, zh) + '</dt><dd>' + (text if name == "posterior" else '<code>' + html.escape(text) + '</code>') + '</dd>'
        body += '</dl>'
    else:
        body += paragraph("No saved 500-coordinate probe is loaded.", "当前未载入 500 维探测记录。")
    body += paragraph("The structure can remain one 500-coordinate signal block. Dimension alone neither mandates NUTS nor 500 scalar updates.", "结构仍可保留为一个 500 维信号块。维度本身既不强制使用 NUTS，也不强制做 500 次标量更新。")
    body += paragraph("Historical conditional verification: five GLS reweightings, Gaussian signal penalty, gain fixed at generating truth; maximum difference from an independent solve ≈ 3.3 × 10⁻¹⁰. This is not joint recovery or a solve with the historical LogNormal prior.", "历史条件验证：GLS 重加权五次，使用高斯信号惩罚，增益固定为生成真值；与独立求解的最大差值约 3.3 × 10⁻¹⁰。这并非联合恢复，也不是使用历史 LogNormal 先验的求解。")
    return detail(("What happens with 500 positive parameters?", "500 个正参数时发生什么？"), body, "scale-probe")


def proofs():
    result = '<section id="design-proofs" data-design-panel="proofs"><h2>'
    result += bi("Keep the reasoning within reach.", "需要时，展开推理与证明。") + '</h2>'
    result += paragraph("The concise guide shares the same formulas and qualifications in both languages. The complete original technical document remains below.", "精简导读的中英文共享相同公式与限定条件。完整原始技术文档仍保留在下方。", "chapter-lede")
    result += detail(
        ("1 · Remove the leading log bias", "1 · 消去对数变换的首阶偏差"),
        paragraph("For small known f, ε ~ Normal(0, 1) and y = μ(1 + fε), expand log(1 + fε) on the positive-data domain.", "对已知的小 f，设 ε ~ Normal(0, 1)、y = μ(1 + fε)，在正值数据域展开 log(1 + fε)。")
        + math(r"E[\log(1+f\epsilon)]\sim-\tfrac12 f^2-\tfrac34 f^4+O(f^6)")
        + math(r"\operatorname{Var}[\log(1+f\epsilon)]\sim f^2+\tfrac52 f^4+O(f^6)")
        + math(r"z=\log y+\tfrac12 f^2\approx\mathcal N(\log\mu,f^2)")
        + paragraph("At f = 0.001, add 5 × 10⁻⁷; the next mean term is −7.5 × 10⁻¹³. These are asymptotic terms, not a global error bound.", "f = 0.001 时加上 5 × 10⁻⁷；下一均值项为 −7.5 × 10⁻¹³。这些是渐近项，并非全局误差界。")
        + paragraph("The untruncated Normal law allows nonpositive observations. Do not clip or discard them to enable logs. Parameter-dependent f, correlations or non-Gaussian noise require corresponding treatment.", "未截断的正态律允许非正观测。不能通过裁剪或丢弃观测来使用对数。f 随参数变化、相关性或非高斯噪声均需相应处理。")
        + paragraph("multiplicative_log_data already applies the leading correction. Its existing f cutoff is not a certificate for arbitrary dataset size or geometry.", "multiplicative_log_data 已应用首阶修正。它现有的 f 截止值不能为任意数据规模或几何提供保证。"),
        "proof-bias",
    )
    result += detail(
        ("2 · Iterative GLS and full-likelihood MAP", "2 · 迭代 GLS 与完整似然 MAP"),
        paragraph("For μ = Mβ, a Gaussian penalty with precision P and center m, freeze Wₖ = Σ(βₖ)⁻¹ and solve:", "对 μ = Mβ，使用精度 P、中心 m 的高斯惩罚，固定 Wₖ = Σ(βₖ)⁻¹ 后求解：")
        + math(r"(M^\top W_kM+P)\beta_{k+1}=M^\top W_ky+Pm")
        + paragraph("A flat penalty uses P = 0 if the conditional system is identifiable. Positivity may require a constrained solver; a LogNormal penalty is not quadratic.", "条件系统可识别时，平坦惩罚使用 P = 0。正值约束可能需要约束求解器；LogNormal 惩罚不是二次型。")
        + math(r"-\log L(\beta)=\tfrac12 r^\top\Sigma(\beta)^{-1}r+\tfrac12\log\det\Sigma(\beta)+C")
        + paragraph("The full gradient includes covariance derivatives. Reweighting to a fixed point alone does not enforce this optimum. Check the full objective for a MAP claim.", "完整梯度含协方差导数。重加权收敛至不动点本身并不满足这一最优条件。声称 MAP 时应检查完整目标。")
        + paragraph("The likelihood log determinant, Jeffreys prior and log-data bias correction are three different terms.", "似然的对数行列式、Jeffreys 先验、对数数据的偏差修正，是三个不同的项。"),
        "proof-gls",
    )
    result += detail(
        ("3 · Fisher information includes changing covariance", "3 · Fisher 信息包含变化协方差的贡献"),
        math(r"I_{ab}=(\partial_a\mu)^\top\Sigma^{-1}(\partial_b\mu)+\tfrac12\operatorname{tr}(\Sigma^{-1}\partial_a\Sigma\,\Sigma^{-1}\partial_b\Sigma)")
        + paragraph("For independent Normal observations with σᵢ = f μᵢ and fixed f, this reduces to:", "对独立正态观测，若 σᵢ = f μᵢ 且 f 固定，上式化为：")
        + math(r"I=(f^{-2}+2)J_{\log\mu}^{\top}J_{\log\mu}")
        + paragraph("In example 5, hold pₙ fixed: J_log μ = U, so I_g = (f⁻² + 2) UᵀU. Full column rank gives a flat conditional Jeffreys density in p_g.", "在例 5 中固定 pₙ：J_log μ = U，故 I_g = (f⁻² + 2) UᵀU。若满列秩，条件 Jeffreys 对 p_g 平坦。")
        + paragraph("Hold p_g fixed instead: J_log μ = diag(1/μ)M depends on pₙ in general. A linear signal mean therefore does not guarantee a flat Jeffreys density.", "若改为固定 p_g：J_log μ = diag(1/μ)M 一般随 pₙ 变化。因此信号均值线性不能保证 Jeffreys 平坦。")
        + paragraph("These are conditional analytic calculations; neither supplies the full joint prior nor constitutes an executed compiler diagnostic.", "这些是条件模型的解析计算；既未给出完整联合先验，也不构成已执行的编译器诊断。"),
        "proof-fisher",
    )
    result += detail(
        ("4 · A block sweep needs one statistical target", "4 · 分块轮转需要同一个统计目标"),
        paragraph("For target π(b, c), update block b while c is fixed. A state-dependent proposal requires both directions:", "对目标 π(b, c)，固定 c 更新 b。依赖状态的提议需要计算两个方向：")
        + math(r"\alpha=\min\left(1,\frac{\pi(b',c)\,q(b\mid b',c)}{\pi(b,c)\,q(b'\mid b,c)}\right)")
        + paragraph("Retain all factors, support, normalization and coordinate Jacobians. If the proposal uses the current block, rebuild the reverse construction at the proposed state.", "保留所有因子、支持域、归一化与坐标 Jacobian。若提议依赖当前块，应在提议状态重建反向构造。")
        + math(r"\pi(b,c)q(b'\mid b,c)\alpha(b,b')=\min\!\left\{\pi(b,c)q(b'\mid b,c),\;\pi(b',c)q(b\mid b',c)\right\}")
        + paragraph("The right side is unchanged when b and b′ are exchanged: accepted probability flows balance. Rejections supply the stay-put probability, so the update preserves the target. Composing such block updates preserves the joint posterior; convergence from initial values still requires an ergodic chain and sufficient exploration.", "交换 b 与 b′ 后右侧不变：被接受的双向概率流相等。拒绝补齐原地停留的概率，因此更新保持目标。组合这样的块更新仍保持联合后验；从初值收敛还需要链具有遍历性并充分探索。")
        + paragraph("Separate proposal-build cost from MH-evaluation cost. A GLS or local fit may dominate. Reuse a proposal during a block visit only when its definition permits it; a state-dependent reverse density may require another fit. Compare time per effective sample.", "区分提议构造成本与 MH 判断成本。GLS 或局部拟合可能占主要开销。仅在提议定义允许时，才可在一次块更新期间复用；状态相关的反向密度可能需要再次拟合。比较每个有效样本的耗时。")
        + paragraph("For an approximate posterior, define one surrogate joint target. Combining an uncorrected approximate log update with an original-space update does not establish a shared posterior.", "对于近似后验，须定义同一个替代联合目标。把未校正的近似对数更新与原空间更新混合，并不能建立共同后验。")
        + paragraph("Explicit bounded Gaussian proposals compose in declaration order, followed by optional NUTS. Automatic discovery also covers otherwise all-NUTS models with supported parameter-dependent Gaussian noise. MH retains forward/reverse normalizers. General matrix-free factor scheduling and cost-based comparison remain planned.", "显式且有预算限制的高斯提议按声明顺序组合，再接可选 NUTS。自动发现也覆盖原本全部进入 NUTS、且具有受支持的参数依赖高斯噪声的模型。MH 保留正反向归一化项。通用的无需显式矩阵的因子调度及基于成本的比较仍待实现。"),
        "proof-target",
    )
    result += '<div class="source-reference"><h3>' + bi("Original document & implementation record", "原始文档与实现记录") + '</h3>'
    result += paragraph("The original English reference is retained verbatim in the download, including D0–D7 contracts, source inventory, measured limitations and acceptance criteria.", "下载文件完整保留英文原文，包括 D0–D7 契约、源码清单、已测得的限制及验收标准。")
    result += '<a href="design-reference.html" data-reference-link target="_blank" rel="noopener">' + bi("Read the complete reference (English)", "阅读完整参考文档（英文）") + '</a> · <a href="design.md" download>' + bi("Download Markdown", "下载 Markdown") + '</a></div>'
    return result + '</section>'


def render_guide(reports):
    return methods() + diagnostics() + priors() + sampling() + examples(reports) + proofs()
