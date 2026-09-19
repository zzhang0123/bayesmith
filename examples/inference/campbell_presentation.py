"""Bilingual Campbell chapter in the existing inference notebook."""

from html import escape
from urllib.parse import quote

if __package__:
    from .methodology_guide import bi, math
    from .methodology_guide import paragraph as p
else:
    from methodology_guide import bi, math
    from methodology_guide import paragraph as p


def figure(folder, name, en, zh):
    url = f"{quote(folder, safe='')}/{name}"
    responsive = (
        f'<source media="(max-width: 900px)" srcset="{url}-stacked.png">'
        if name
        in {"distributions", "edgeworth", "edgeworth_accuracy", "posterior_density"}
        else ""
    )
    return (
        f'<figure><a href="{url}.png" target="_blank" rel="noopener"><picture>{responsive}'
        f'<img src="{url}.png" loading="lazy" alt="{escape(en)}" style="width:100%;height:auto">'
        f"</picture></a><figcaption>{bi(en, zh)} "
        f'<a href="{url}.png" target="_blank" rel="noopener">{bi("Full size", "查看原图")}</a>'
        f"</figcaption></figure>"
    )


def table(headers, rows):
    return (
        '<div class="table-scroll"><table><thead><tr>'
        + "".join("<th>" + bi(*h) + "</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join("<td>" + escape(str(c)) + "</td>" for c in row) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def render_case(report, folder, index):
    config = report["config"]
    side = config["shape"][0]
    draws = report["draws"]
    parts = {}
    parts["model"] = p(
        "Many independent point sources make one correlated sky.",
        "许多独立点源，经波束叠加，形成相关的天空图。",
        "chapter-lede",
    )
    parts["model"] += math(
        r"N_j\sim\mathrm{Poisson}(\lambda),\quad x_j=a(N_j-\lambda)+g_j,\quad a=\sqrt{0.8/\lambda},\quad g_j\sim\mathcal N(0,0.2),\quad s=Bx"
    )
    parts["model"] += p(
        "N is the source count in each cell; a is each source’s amplitude. B spreads a source onto its own pixel (0.6) and four neighbours (0.1 each), with periodic boundaries. The Gaussian diffuse field makes the prior a continuous, proper density.",
        "N 是每格的点源数，a 是每个源的振幅。B 将一个源分配到本像素（0.6）和四个邻居（各 0.1），边界周期延拓。高斯弥散分量使先验成为连续、正规化的概率密度。",
    )
    parts["campbell"] = (
        "<h3>"
        + bi(
            "Campbell’s theorem: cumulants add over independent sources",
            "Campbell 定理：独立点源的累积量可以直接相加",
        )
        + "</h3>"
    )
    parts["campbell"] += p(
        "For a Poisson population, each connected r-point cumulant is the source intensity times the product of r beam responses, summed over source positions. It follows by differentiating the Poisson log moment-generating function.",
        "对 Poisson 源群，连通 r 点累积量等于源强度乘以 r 个波束响应的乘积，再对源位置求和。它来自 Poisson 对数矩母函数的求导。",
    )
    parts["campbell"] += math(
        r"\log\mathbb E[e^{t^Ts}]=\lambda\sum_j\!\left[e^{a(B^Tt)_j}-1-a(B^Tt)_j\right]+\frac{0.2}{2}\|B^Tt\|^2"
    )
    parts["campbell"] += math(
        r"K_{i_1\cdots i_r}=\lambda a^r\sum_j B_{i_1j}\cdots B_{i_rj}\ (r\ge3),\qquad \mathbb E[s]=0,\quad K_2=BB^T"
    )
    parts["campbell"] += p(
        "The third cumulant describes asymmetry; the fourth is the connected fourth moment (subtract the three covariance products). Varying λ changes these while the mean and covariance stay fixed: the Gaussian power spectrum alone cannot identify λ.",
        "三阶累积量描述不对称性；四阶是连通四阶矩（须减去三种协方差乘积）。改变 λ 会改变这些量，但均值与协方差保持不变，因此单靠高斯功率谱无法识别 λ。",
    )
    parts["campbell"] += (
        '<p><a href="https://stoch.math.kit.edu/img/Last/lastpenrose2017.pdf">Last &amp; Penrose, Lectures on the Poisson Process · §§3.3, 15.4</a></p>'
    )
    if "cumulants" in report:
        c = report["cumulants"]
        parts["campbell"] += table(
            [
                ("Cumulant", "累积量"),
                ("Campbell", "Campbell 解析值"),
                ("Simulation", "模拟估计"),
                ("Difference / SE", "偏差 / 标准误"),
            ],
            [
                (name, f"{theory:.6f}", f"{estimate:.6f}", f"{z:+.2f}")
                for name, theory, estimate, z in zip(
                    c["names"], c["theory"], c["estimate"], c["z_score"], strict=True
                )
            ],
        )
        parts["campbell"] += p(
            f"{c['realizations']:,} independent local realizations; 0 and 1 denote neighbouring pixels. Standard errors come from 48 independent batches.",
            f"{c['realizations']:,} 次独立局部模拟；0、1 表示相邻像素。标准误来自 48 个独立批次。",
        )
    parts["model"] += figure(
        folder,
        "sky",
        f"{side} × {side} sky; the right panel is the independent analytic posterior mean.",
        f"{side} × {side} 天空；右图是独立计算的解析后验均值。",
    )

    parts["methods"] = p(
        "First condition on the observed data under a Gaussian reference prior. Then reuse the same independent samples for every candidate λ.",
        "先用高斯参考先验对观测数据做条件采样，再为每个候选 λ 复用同一批独立样本。",
        "chapter-lede",
    )
    parts["methods"] += math(
        r"d=B(x+\epsilon),\quad \epsilon\sim\mathcal N(0,0.01I),\quad y=B^{-1}d"
    )
    parts["methods"] += math(
        r"q(x\mid y)=\prod_j\mathcal N\!\left(x_j;\frac{y_j}{1.01},\frac{0.01}{1.01}\right),\qquad x^{(m)}=m_q+C_q^{1/2}z^{(m)}"
    )
    parts["methods"] += p(
        "This is the diagonal special case of GCR / a linear Gaussian sampler: no warmup, accept/reject step or field gradient. The large benchmark calls this closed-form sampler directly, rather than compiling a huge graph. The earlier small demo exercises bayesmith’s compiled GCR route.",
        "这是 GCR / 线性高斯 sampler 的对角特例：无需预热、接受拒绝或场梯度。大规模计时直接调用这个解析 sampler；此前的小图实验验证了 bayesmith 编译后的 GCR 路径。",
    )
    parts["methods"] += math(
        r"r_{mj}(\lambda)=\frac{p_\lambda(x_j^{(m)})}{p_0(x_j^{(m)})},\quad W_m=\prod_j r_{mj},\quad \frac{Z_\lambda}{Z_0}=\mathbb E_q[W]"
    )
    parts["methods"] += p(
        "The full target/reference joint ratio simplifies to a prior ratio because both models use exactly the same likelihood. If the likelihood changes, its ratio must remain in W. The target prior in this experiment is the full Poisson–Gaussian mixture.",
        "完整目标/参考联合密度之比，在两者 likelihood 相同时才化简为先验比；若 likelihood 改变，其比值也必须保留在 W 中。目标先验使用完整 Poisson–Gaussian 混合分布。",
    )
    parts["methods"] += math(
        r"\hat\lambda_{\mathrm{MAP}}=\arg\max_{\lambda\in[0.5,8]}\left[\log\!\left(\frac1M\sum_m W_m(\lambda)\right)+\log p(\lambda)\right]"
    )
    parts["methods"] += p(
        "Integrate out the sky first, then maximize the hyperparameter objective. This is marginal hyperparameter MAP, not joint MAP over sky and λ. Here p(λ) is uniform on the stated interval.",
        "先把天空场积分掉，再最大化超参数目标。这是边缘超参数 MAP；本例 λ 在给定区间上服从均匀超先验。",
    )

    parts["diagnostics"] = p(
        "Cheap draws do not guarantee useful whole-map weights.",
        "抽样便宜，不代表整图权重一定有效。",
        "chapter-lede",
    )
    parts["diagnostics"] += math(
        r"\mathrm{ESS}_{\mathrm{joint}}=\frac{(\sum_m W_m)^2}{\sum_m W_m^2}"
    )
    parts["diagnostics"] += p(
        "Multiplying thousands of small density ratios can concentrate mass on a handful of maps. ESS measures this concentration; it does not prove that all important tails were sampled.",
        "成千上万个小密度比相乘，可能让少数图承载大部分权重。ESS 衡量这种集中程度，但不能证明重要尾部都已被采到。",
    )
    parts["diagnostics"] += (
        "<h3>"
        + bi("An exact simplification available in this model", "本模型允许的精确化简")
        + "</h3>"
    )
    parts["diagnostics"] += math(
        r"\log\frac{Z_\lambda}{Z_0}=\sum_j\log\mathbb E_{q_j}[r_j]\ \approx\ \sum_j\log\!\left(\frac1M\sum_m r_{mj}\right)"
    )
    parts["diagnostics"] += p(
        "Because both reference and target factorize in source coordinates, each cell can be integrated separately. This is a different Monte Carlo estimator of the same evidence. It has finite-sample log bias; independent banks and analytic evidence check its error. Local ESS is not whole-map ESS.",
        "因为参考与目标都在源系数坐标中分解，各格可分别积分。这是同一 evidence 的另一种 Monte Carlo 估计，仍有有限样本的对数偏差；通过独立样本批次与解析 evidence 检查误差。局部 ESS 不能当作整图 ESS。",
    )
    parts["diagnostics"] += p(
        "The sky pixels remain correlated. We deliberately observe d=B(x+ε), so sky-noise covariance is 0.01 BBᵀ; invertible B preserves independence after deconvolution. Masks, independent detector noise after beam smoothing, or coupled source priors generally destroy this factorization. They need another proposal/block strategy.",
        "天空像素仍然相关。本例特意采用 d=B(x+ε)，天空噪声协方差为 0.01 BBᵀ；可逆 B 使反卷积后的坐标保持独立。遮罩、波束之后的独立探测器噪声或耦合源先验通常会破坏这种分解，需要其他 proposal 或分块策略。",
    )
    parts["diagnostics"] += p(
        "The reference must cover the target support. A pure discrete Poisson field cannot be reweighted from continuous Gaussian samples using ordinary density ratios; the Gaussian diffuse component makes this experiment continuous.",
        "参考分布必须覆盖目标的支撑。纯离散 Poisson 场不能直接使用普通密度比重加权连续高斯样本；本实验的高斯弥散分量使目标成为连续分布。",
    )

    parts["sampling"] = p(
        f"{draws:,} draws per inference bank; two independent banks. Scaling additionally reaches 65,536 latent coefficients per map.",
        f"推断每批 {draws:,} 张图，使用两个独立批次；抽样规模测试进一步达到每图 65,536 个潜变量。",
        "chapter-lede",
    )
    parts["sampling"] += table(
        [
            ("Map", "图大小"),
            ("Latents", "潜变量"),
            ("Maps drawn", "抽样张数"),
            ("Cold seconds", "首次秒数"),
            ("Warm seconds", "热运行秒数"),
            ("Bank MiB", "样本 MiB"),
        ],
        [
            (
                f"{r['side']} × {r['side']}",
                r["cells"],
                r["draws"],
                f"{r['cold_seconds']:.4f}",
                f"{r['warm_seconds_median']:.4f}",
                f"{r['bank_mib']:.0f}",
            )
            for r in report["scaling"]
        ],
    )
    parts["sampling"] += p(
        "All timings synchronize JAX completion. First call includes compilation; warm is the median of three fresh-key calls. These are measured draw times, not speedup factors over NUTS. No dense pixel covariance is allocated. Source draws are timed; forming beam-smoothed sample maps is excluded.",
        "所有计时均等待 JAX 运算完成。首次含编译；热运行取三个新随机种子调用的中位数。这是实测抽样时间，没有宣称相对 NUTS 的加速倍数。没有分配稠密像素协方差；计时对象是源系数抽样，不含将每张样本做波束卷积。",
    )
    parts["sampling"] += table(
        [
            ("Bank", "批次"),
            ("Draw seconds", "抽样秒数"),
            ("First score seconds", "首次权重秒数"),
            ("Profile + fit seconds", "曲线与拟合秒数"),
        ],
        [
            (
                b["seed"],
                f"{b['draw_seconds']:.4f}",
                f"{b['score_compile_seconds']:.3f}",
                f"{b['profile_and_fit_seconds']:.2f}",
            )
            for b in report["banks"]
        ],
    )
    parts["sampling"] += p(
        "The prior-ratio evaluation, repeated across λ, now dominates cost. Both estimators stream 64 cells at a time through the mixture calculation, bounding temporary memory. Sampling does not repeat when λ changes.",
        "现在主要成本是对不同 λ 重复计算非高斯先验比。两种估计都每次处理 64 格的混合密度，限制临时内存。λ 改变时无需重新采样。",
    )
    parts["sampling"] += (
        "<details><summary>"
        + bi("Recorded execution environment", "已记录的运行环境")
        + "</summary><pre>"
        + escape(report["environment"]["platform"] + "\n" + str(report["environment"]))
        + "</pre></details>"
    )

    parts["recovery"] = p(
        f"Injected λ = {config['rate']}; analytic marginal MAP = {report['oracle_map']:.5f}; analytic 95% interval = [{report['oracle_interval95'][0]:.3f}, {report['oracle_interval95'][1]:.3f}].",
        f"注入 λ = {config['rate']}；解析边缘 MAP = {report['oracle_map']:.5f}；解析 95% 区间 = [{report['oracle_interval95'][0]:.3f}, {report['oracle_interval95'][1]:.3f}]。",
        "chapter-lede",
    )
    parts["recovery"] += math(
        r"p_\lambda(y_j)=\sum_{k=0}^{\infty}\mathrm{Pois}(k;\lambda)\,\mathcal N\!\left(y_j;a(k-\lambda),0.21\right)"
    )
    parts["recovery"] += p(
        "The independent analytic answer sums over counts and integrates the Gaussian diffuse field exactly. A finite sky need not have its MAP at the injection; compare the numerical method with the analytic MAP for the same data.",
        "独立解析答案对源数求和，并精确积分高斯弥散场。有限天空的 MAP 不必等于注入值；数值方法应与同一份数据的解析 MAP 比较。",
    )
    parts["recovery"] += table(
        [
            ("Bank", "批次"),
            ("Factorized MAP", "分解积分 MAP"),
            ("Joint MAP", "整图 MAP"),
            ("Joint ESS at injection", "注入值处整图 ESS"),
            ("Minimum local ESS at fit", "拟合处最小局部 ESS"),
            ("Δ log evidence at fit", "拟合处 log evidence 误差"),
        ],
        [
            (
                b["seed"],
                f"{b['factorized_map']:.5f}",
                f"{b['joint_map']:.5f}",
                f"{b['joint_ess_at_truth']:.1f} / {draws}",
                f"{b['local_min_ess_at_fit']:.1f} / {draws}",
                f"{b['factorized_log_evidence_error_at_fit']:+.4f}",
            )
            for b in report["banks"]
        ],
    )
    parts["recovery"] += figure(
        folder,
        "scaling",
        "Evidence curves zoom to the posterior region and share one analytic baseline; the full rate grid is saved in JSON. Cold/warm Gaussian scaling at right.",
        "曲线放大后验集中区域，共用同一个解析 evidence 基准；完整 λ 网格保存在 JSON 中。右侧展示首次与热运行的高斯抽样计时。",
    )
    parts["recovery"] += p(
        f"The count sum is conditioned on k ≤ {config['count_max']}; the generator is untruncated. Omitted Poisson probability anywhere on this map ≤ {report['poisson_tail_union_bound']:.2e}. Doubling the cutoff changes log evidence at the injection by {report['cutoff_log_evidence_difference']:.3g}.",
        f"数值源数求和条件化于 k ≤ {config['count_max']}，模拟器没有截断。本图任意位置被省略的 Poisson 概率 ≤ {report['poisson_tail_union_bound']:.2e}；截断上限加倍后，注入值处 log evidence 改变 {report['cutoff_log_evidence_difference']:.3g}。",
    )
    parts["recovery"] += p(
        "Conclusion: Gaussian field draws are cheap and reusable; large-field accuracy still depends on the integration strategy. This controlled factorization is a benchmark, not a general solution for arbitrary non-Gaussian skies.",
        "结论：高斯场抽样便宜且可复用；大场推断的精度仍取决于积分策略。这里的精确分解是受控 benchmark，不能直接推广为任意非高斯天空的通用解。",
    )

    parts["distributions"] = p(
        "Start with the actual distribution, before choosing an approximation. A histogram is a finite realization; a PDF is a population prediction.",
        "先看真实分布，再选择近似。Histogram 是一份有限实现的统计；PDF 是模型对总体分布的预测。",
        "chapter-lede",
    )
    parts["distributions"] += figure(
        folder,
        "distributions",
        "Saved-map histograms versus the proper Poisson mixture and its matched Gaussian reference.",
        "已保存天空的直方图，与正规 Poisson 混合分布及同均值同方差的高斯参考比较。",
    )
    parts["distributions"] += math(
        r"p_\lambda(x)=\sum_{k=0}^{\infty}\mathrm{Pois}(k;\lambda)\,\mathcal N(x;a(k-\lambda),0.2)"
    )
    parts["distributions"] += p(
        "The left histogram pools independent source cells. The right pools correlated sky pixels, so it is not a set of independent realizations and we do not attach independent-bin error bars. Beam smoothing changes the marginal distribution; it does not make it exactly Gaussian. All histogram heights use the total cell count, including samples outside the plot range.",
        "左图汇集独立源格；右图汇集相关天空像素，因此不是一组独立实现，也没有附上假设独立的误差条。波束平滑改变了边缘分布，但不会使其精确高斯化。直方图使用全部格子的数目归一化，包括绘图区间之外的样本。",
    )
    parts["distributions"] += p(
        "For the sky PDF, the four equal-weight neighbour counts sum to Poisson(4λ), independently of the central Poisson(λ) count. This gives an independent two-count mixture with Gaussian variance 0.08. The plotted count sums omit less than 2×10⁻¹³ probability before renormalization.",
        "天空 PDF 利用四个等权邻格的总源数服从 Poisson(4λ)，且与中心的 Poisson(λ) 独立；因此可直接计算两种源数的混合密度，高斯分量方差为 0.08。绘图所用求和在重新归一化前省略的概率小于 2×10⁻¹³。",
    )

    parts["priors"] = p(
        "Choose a non-Gaussian prior family, then choose how to integrate its posterior. These are separate decisions: Gaussian posterior draws and importance weights do not require any particular prior parameterization.",
        "先选择非高斯先验族，再选择怎样积分其后验。这是两个独立的决定：高斯后验抽样与 importance weights 不要求特定的先验参数化。",
        "chapter-lede",
    )
    parts["priors"] += p(
        "The experiment below uses only the Poisson–Gaussian mixture. Flow and Edgeworth are alternatives discussed here; neither is used or trained in the recorded inference.",
        "后面的实验只使用 Poisson–Gaussian mixture。Flow 和 Edgeworth 在这里作为可选表示讨论，已记录的推断没有使用或训练它们。",
    )
    parts["priors"] += (
        "<h3>"
        + bi(
            "Mixture · the prior used in this demo", "Mixture · 本 demo 实际使用的先验"
        )
        + "</h3>"
    )
    parts["priors"] += p(
        "Combine normalized component densities with nonnegative weights summing to one. Here the components are Gaussians indexed by Poisson source counts; λ controls non-Gaussianity. This gives a proper, evaluable density, and this particular model also supplies an analytic evidence for validation.",
        "用非负、和为 1 的权重组合已归一化的分量密度。这里是按 Poisson 源数加权的高斯混合，λ 控制非高斯性。它给出可计算的正规密度，而且这个特定模型还提供解析 evidence 用于验证。",
    )
    parts["priors"] += (
        "<h3>"
        + bi(
            "Normalising flow · a flexible normalized density",
            "Normalising flow · 灵活且归一化的密度",
        )
        + "</h3>"
    )
    parts["priors"] += math(
        r"x=f_\theta(z),\quad z\sim p_{\rm base},\qquad p_\theta(x)=p_{\rm base}(f_\theta^{-1}(x))\left|\det Df_\theta^{-1}(x)\right|"
    )
    parts["priors"] += p(
        "An invertible transformation and its Jacobian define the density. A flow can parameterize the prior in the same evidence objective. Learning it from real observations does not inherently require simulated training maps, but its flexibility needs appropriate structure and regularization. Using a flow to improve the sampling proposal is a different role from using it as the prior.",
        "可逆变换与 Jacobian 一起定义密度。Flow 可以在同一个 evidence 目标里参数化先验；从真实观测学习它，本身不要求模拟训练图，但灵活模型需要合适的结构约束与正则化。用 flow 改善抽样 proposal，则是与用它表示先验不同的角色。",
    )
    parts["priors"] += (
        '<p><a href="https://www.jmlr.org/papers/v22/19-1028.html">Papamakarios et al. · Normalizing Flows for Probabilistic Modeling and Inference</a></p>'
    )
    parts["priors"] += (
        "<h3>"
        + bi(
            "Edgeworth · a controlled weakly non-Gaussian approximation",
            "Edgeworth · 受控的弱非高斯近似",
        )
        + "</h3>"
    )
    parts["priors"] += p(
        "Cumulant corrections can capture asymmetry that a matched Gaussian misses. In this family, increasing λ reduces standardized higher cumulants at fixed mean and variance, making a low-order expansion increasingly accurate. The cumulants can be inferred from observations; they need not come from simulations.",
        "累积量修正能够捕捉同均值同方差的高斯所遗漏的不对称性。在这个分布族中，增大 λ 会在保持均值与方差不变的同时减小标准化高阶累积量，低阶展开因而越来越准确。累积量可以从观测推断，不必来自模拟。",
    )
    parts["priors"] += figure(
        folder,
        "edgeworth_accuracy",
        "Successful weakly non-Gaussian examples: λ=8 and 32, on a linear density axis. The dotted Edgeworth curve nearly overlaps the exact mixture while improving on the matched Gaussian.",
        "弱非高斯下的良好展开：λ=8 与 32，纵轴为普通线性密度。点线表示的 Edgeworth 几乎与精确 mixture 重合，比同均值同方差的高斯更准确。",
    )
    parts["priors"] += p(
        "These are analytic prior illustrations, not additional inference runs; the main sky experiment still uses λ=2. The annotations measure ∫|p_approx−p_exact| dx on [−10σ, 16σ] with 26,001 grid points, beyond the displayed range. Larger λ reduces the shot amplitude to preserve variance; the same expansion, including the squared-skewness term, is used in both panels.",
        "这是解析先验的示意比较，没有新增推断实验；主天空实验仍使用 λ=2。图中误差是用 26,001 个网格点在 [−10σ, 16σ] 上计算的 ∫|p_approx−p_exact| dx，积分区间大于绘图区间。λ 增大时相应减小单源振幅以保持方差；两图使用同一个包含偏度平方项的展开。",
    )
    parts["priors"] += (
        f'<p><a href="{quote(folder, safe="")}/edgeworth_accuracy.pdf">{bi("Download the accuracy comparison", "下载展开精度对照图")}</a></p>'
    )
    parts["priors"] += p(
        "Approximation accuracy and validity as a probability density are separate checks. Finitely many cumulants do not uniquely specify a full density, and a truncated expansion can have negative tails even when its integrated error is small. The optional comparison below examines that limitation at λ=2; it does not represent the accuracy of every Edgeworth expansion.",
        "近似精度与作为概率密度的合法性需要分别检查。有限个累积量不能唯一确定完整密度，截断展开即使整体误差很小，也可能出现负尾部。下面的可选对照用 λ=2 检查这一局限，不能代表所有 Edgeworth 展开的精度。",
    )
    parts["priors"] += p(
        "All choices must supply the density ratio required by the inference. Retain every θ-dependent normalization factor when fitting θ. Simulation and Campbell's theorem provide testable truth in this demo; neither is a prerequisite for fitting a prior to real data.",
        "无论选择哪种表示，都要能计算推断所需的密度比；拟合 θ 时必须保留所有依赖 θ 的归一化因子。模拟与 Campbell 定理在本 demo 中提供可检验的真值，都不是从真实数据拟合先验的前提。",
    )
    edgeworth_detail = p(
        "The successful examples above and the tail stress case below use the same expansion. Examine overall accuracy and tail validity separately.",
        "上面的良好展开与下面的尾部压力示例使用同一个公式。整体精度与尾部有效性应分别考察。",
        "chapter-lede",
    )
    edgeworth_detail += math(
        r"p_E(x)=\frac{\phi(z)}{\sigma}\left[1+\frac{\gamma_3}{6}H_3(z)+\frac{\gamma_4}{24}H_4(z)+\frac{\gamma_3^2}{72}H_6(z)\right],\quad z=x/\sigma,\quad \gamma_r=\kappa_r/\sigma^r"
    )
    edgeworth_detail += p(
        "Here H₃=z³−3z and H₄=z⁴−6z²+3 are probabilists’ Hermite polynomials; H₆=z⁶−15z⁴+45z²−15. The squared-skewness term matters. The correction integrates to zero against the Gaussian, but that does not enforce positivity.",
        "这里 H₃=z³−3z、H₄=z⁴−6z²+3 是概率论约定的 Hermite 多项式，H₆=z⁶−15z⁴+45z²−15。偏度平方项不能漏掉。修正项对高斯的积分为零，但这并不保证密度非负。",
    )
    edgeworth_detail += figure(
        folder,
        "edgeworth",
        "Signed tail comparison (negative values retained), and different rates with identical mean and covariance.",
        "保留负值的尾部分布比较，以及同均值同协方差、不同 λ 的分布族。",
    )
    edgeworth_detail += p(
        "At λ=2 the scalar correction reaches about −0.525 near x=−3.181. The log-like signed vertical scale exposes negative density; it is not log(abs(PDF)). Such a target cannot supply probability importance weights. Better agreement around the peak does not repair its tails.",
        "在 λ=2 时，标量修正因子在 x≈−3.181 附近降到约 −0.525。图中使用保留符号、零点附近线性的纵轴，负密度没有被取绝对值。这样的目标不能提供概率意义的 importance weights；中心拟合得更好，也不能修复负尾部。",
    )
    edgeworth_detail += math(
        r"P_E(\mathbf 0)/P_G(\mathbf 0)=1+D\left(\frac{\kappa_4}{8}-\frac{5\kappa_3^2}{24}\right)=1-\frac{0.026666\ldots D}{\lambda}"
    )
    edgeworth_detail += p(
        "This is the joint order-four expansion for unit-variance independent source coordinates, including the cross-cell skewness products. For D=144 and λ=2 it is −0.92 (also checked with the actual field operator in the small demo). The expression is still more negative for the large map. It is not a product of separately truncated one-cell PDFs.",
        "这是单位方差、独立源系数的联合四阶展开，包含跨格的偏度乘积。D=144、λ=2 时等于 −0.92，小图实验也用实际 field operator 验证了这一点。大图的该值更负。这与逐格截断 PDF 的乘积是不同的展开。",
    )

    parts["priors"] += (
        "<details><summary>"
        + bi(
            "Optional: expansion formula and a separate tail stress case",
            "可选展开：展开公式与单独的尾部压力示例",
        )
        + "</summary>"
        + edgeworth_detail
    )
    parts["priors"] += (
        f'<p><a href="{quote(folder, safe="")}/edgeworth.pdf">{bi("Download the comparison figure", "下载先验近似对照图")}</a></p></details>'
    )

    parts["methods"] += figure(
        folder,
        "posterior_density",
        "Two separate one-cell illustrations at the injected λ: 40,000 Gaussian draws per cell, reweighted against the full proper prior.",
        "注入 λ 下的两个独立单格示意实验：每格 40,000 个高斯样本，用完整正规先验重加权。",
    )
    parts["methods"] += p(
        "These histograms are conditional posteriors, unlike the prior histograms above. They use fresh illustrative draws, not the saved whole-map banks. Low noise keeps the Gaussian reference close to the target in each coordinate; thousands of small corrections can still accumulate into a poor joint proposal.",
        "这里画的是条件后验，前面画的是先验。示意图使用新生成的抽样，不是已保存的整图样本批次。低噪声使单坐标的高斯参考接近目标，但成千上万个小修正累积后，联合 proposal 仍可能失效。",
    )

    sections = [
        ("model", ("A sky whose truth we control", "构造一个真值已知的天空")),
        ("distributions", ("See the non-Gaussianity", "先看非高斯性长什么样")),
        (
            "campbell",
            ("Why these cumulants are analytic", "Campbell 定理为何给出解析累积量"),
        ),
        (
            "priors",
            (
                "Choose a non-Gaussian prior: mixture, flow, Edgeworth",
                "非高斯先验的选择：mixture、flow、Edgeworth",
            ),
        ),
        (
            "methods",
            (
                "Gaussian draws → posterior weights → hyper-MAP",
                "高斯抽样 → 后验权重 → 超参数 MAP",
            ),
        ),
        ("sampling", ("Scale the cheap part", "把廉价抽样扩大到大图")),
        (
            "diagnostics",
            ("When whole-map weights collapse", "整图权重为何退化，怎样利用独立性"),
        ),
        (
            "recovery",
            ("Close the loop against the analytic answer", "用解析答案检验最终恢复"),
        ),
    ]
    text = '<article class="case methodology campbell-essay" id="campbell_sky" data-case="campbell_sky" data-reading-layout="continuous">'
    text += (
        '<header class="case-heading"><span class="status-tag">'
        + bi(
            "Example 07 · an experiment in eight questions",
            "例子 07 · 围绕八个问题展开的实验",
        )
        + "</span><h1>"
        + bi("A Poisson sky, cheap Gaussian draws", "Poisson 天空与廉价高斯抽样")
        + "</h1>"
    )
    text += (
        p(
            "Can we keep Gaussian sampling speed and recover non-Gaussian physics? Separate the choice of prior from Gaussian sampling and evidence estimation, then test the inference on a large field.",
            "能否保留高斯采样的速度，同时恢复非高斯物理参数？先区分先验的选择、高斯抽样与 evidence 估计，再用大图检验推断。",
        )
        + "</header>"
    )
    text += '<nav class="language-switch" aria-label="Language / 语言"><button type="button" data-language-choice="en">English</button><button type="button" data-language-choice="zh">中文</button></nav>'
    text += (
        '<nav class="campbell-outline" aria-label="Experiment contents / 实验目录"><ol>'
    )
    for key, title in sections:
        text += (
            f'<li><a href="#case=campbell_sky&amp;section={key}">{bi(*title)}</a></li>'
        )
    text += "</ol></nav>"
    for i, (key, title) in enumerate(sections, 1):
        if key == "priors":
            text += '<span id="campbell_sky-edgeworth" aria-hidden="true"></span>'
        text += f'<section class="stage" id="campbell_sky-{key}" data-stage="{key}"><h2>{i:02d} · {bi(*title)}</h2>{parts[key]}</section>'
    text += '<footer class="artifacts">'
    for name, label in [
        ("result.json", ("Inference results", "推断数值结果")),
        (
            "distribution-comparison.json",
            ("Histogram counts & density audit", "直方图计数与密度审计"),
        ),
        ("distributions.pdf", ("Distribution PDF figure", "分布对照 PDF 图")),
        ("posterior_density.pdf", ("Posterior figure", "后验对照图")),
        ("scaling.pdf", ("Scaling & recovery figure", "规模与恢复图")),
    ]:
        text += f'<a href="{quote(folder, safe="")}/{name}">{bi(*label)}</a>'
    return text + "</footer></article>"
