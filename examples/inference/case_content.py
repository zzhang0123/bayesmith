"""One bilingual model description shared by the guide and recorded demos."""

LEGACY_CASES = {
    "exponential_decay": {
        "title": ("Two decay channels, six unknowns", "两个衰减通道，六个未知量"),
        "subtitle": ("Earlier exponential-decay example", "早期指数衰减例子"),
        "formula": r"y_{ij}\sim\mathcal N(a_j e^{-r_jt_i}+b_j,\,0.08^2),\quad j=1,2",
        "model": ("Retained reference model: amplitudes Uniform(0,4), rates Uniform(0.1,2.5), backgrounds Uniform(−1,1).",
                  "保留的参考模型：振幅 Uniform(0,4)，衰减率 Uniform(0.1,2.5)，背景 Uniform(−1,1)。"),
        "structure": ("Given the rates, amplitude and background enter the mean linearly. Finite bounds truncate the conditional posterior.",
                      "给定衰减率，振幅与背景线性进入均值；有限边界截断条件后验。"),
        "simulation": ("Generate 120 noisy observations per channel.", "每个通道生成 120 个带噪观测。"),
    },
}

CASES = {
    "linear_gaussian": {
        "title": ("Four coefficients, one linear block", "四个系数，一个线性块"),
        "subtitle": ("Linear Gaussian", "线性高斯"),
        "formula": r"X_i=(1,x_i,\sin\pi x_i,\cos\pi x_i),\quad y\sim\mathcal N(X\beta,\,0.25^2I)",
        "model": (
            "A four-dimensional coefficient vector weights four known basis functions. Each β coefficient has a Uniform(−4, 4) prior; observation noise is fixed.",
            "四维系数向量对四个已知基函数加权。每个 β 系数的先验为 Uniform(−4, 4)，观测噪声固定。",
        ),
        "structure": (
            "The fixed-noise Gaussian likelihood has flat Jeffreys density in all four coefficients. The finite Uniform prior truncates the posterior; the automatic compiler selects joint NUTS.",
            "固定噪声的高斯似然对四个系数具有平坦 Jeffreys 密度。有限区间 Uniform 先验截断后验，自动编译器选择联合 NUTS。",
        ),
        "simulation": (
            "Evaluate Xβ at the declared truth, then add Gaussian noise. Only the observations and model enter inference.",
            "在声明的真值处计算 Xβ，再加入高斯噪声。推断仅接收观测与模型。",
        ),
    },
    "power_law": {
        "title": ("Power-law curves with white noise", "幂律曲线与白噪声"),
        "subtitle": ("Power law + white noise", "幂律 + 白噪声"),
        "formula": r"y_{ji}=A_jx_i^{\alpha_j}+\epsilon_{ji},\quad\epsilon_{ji}\overset{\mathrm{iid}}{\sim}\mathcal N(0,0.2^2),\quad j=1,2",
        "model": (
            "Two independent curves, each with 32 known inputs x ∈ [0.5, 2]. Infer A₁, α₁, A₂ and α₂; the additive white-noise SD is known, σ = 0.2. The baseline uses Uniform(0.2, 3) for amplitudes and Uniform(−2, 2) for indices.",
            "两条独立曲线，每条在 x ∈ [0.5, 2] 上有 32 个已知输入。推断 A₁、α₁、A₂、α₂；加性白噪声标准差已知，σ = 0.2。原始版本的振幅使用 Uniform(0.2, 3)，指数使用 Uniform(−2, 2)。",
        ),
        "structure": (
            "At fixed α, the mean is linear in A and the noise is fixed: iterative GLS is unnecessary. The baseline Uniform bounds truncate the Gaussian amplitude conditional; the current GCR path requires diagonal Normal priors, so the compiler uses joint NUTS. The exponent is nonlinear. Taking log y changes the additive noise and can be undefined; the multiplicative-noise log-linear approximation does not apply directly.",
            "固定 α 时，均值对 A 线性且噪声固定，无需迭代 GLS。原始 Uniform 边界截断振幅的高斯条件分布；当前 GCR 路径要求对角 Normal 先验，因此编译器采用联合 NUTS。指数是非线性参数。对 y 取对数会改变加性噪声，也可能无定义，不能直接套用乘性噪声的对数线性近似。",
        ),
        "simulation": (
            "Evaluate each curve at 32 known inputs, then add independent Gaussian white noise with SD 0.2. Compare noisy observations with the generating curves. A separate 512-dataset paired experiment measures MAP and posterior-mean bias.",
            "在 32 个已知输入处计算每条曲线，再加入标准差为 0.2 的独立高斯白噪声。将带噪观测与生成曲线比较。另用 512 组配对数据集测量 MAP 和后验均值的偏差。",
        ),
    },
    "hierarchical": {
        "title": ("Eight groups and their population", "八个组及其总体"),
        "subtitle": ("Hierarchical groups", "层级分组"),
        "formula": r"m\sim\mathrm{Uniform}(-3,3),\quad \eta_j\sim\mathcal N(m,0.6^2),\quad y_i\sim\mathcal N(\eta_{g_i},0.25^2)",
        "model": (
            "Eight group locations share one population mean: nine unknown coordinates. Only the population prior is flat, Uniform(−3, 3); group laws remain Gaussian. Indexing selects the group for each observation.",
            "八个组位置共享一个总体均值，共九个未知坐标。仅总体先验采用 Uniform(−3, 3)，组的条件分布保留高斯。索引为每个观测选择所属组。",
        ),
        "structure": (
            "The population → groups edge is a latent-density factor. The current compiler sends this whole graph to NUTS.",
            "总体 → 组的边是一项潜变量密度因子。当前编译器将整张图交给 NUTS。",
        ),
        "simulation": (
            "Draw group truths from their declared conditional prior, then generate 40 observations per group. Groups remain separate categories.",
            "从声明的条件先验抽取组真值，再为每组生成 40 个观测。各组作为独立类别展示。",
        ),
    },
    "bernoulli": {
        "title": ("A probability surface from binary data", "从二元数据恢复概率曲面"),
        "subtitle": ("Bernoulli regression", "Bernoulli 回归"),
        "formula": r"X_i=(1,x_{i1},x_{i2},x_{i1}x_{i2}),\quad y_i\sim\mathrm{Bernoulli}(\mathrm{sigmoid}(X_i\beta))",
        "model": (
            "Two inputs and their interaction determine a four-coefficient probability surface. Each β coefficient has a Uniform(−4, 4) prior.",
            "两个输入及其交互项决定一个四系数概率曲面。每个 β 系数的先验为 Uniform(−4, 4)。",
        ),
        "structure": (
            "Linear logits do not give a Gaussian conditional. NUTS samples all four coefficients jointly.",
            "线性 logits 不产生高斯条件分布。NUTS 联合采样四个系数。",
        ),
        "simulation": (
            "Use a 20 × 20 input grid. The declared Bernoulli operator generates 0/1 observations; the unified SimulationTask does not yet support this observation family.",
            "使用 20 × 20 输入网格。声明的 Bernoulli 算子生成 0/1 观测；统一 SimulationTask 尚不支持这一观测分布族。",
        ),
    },
    "multiplicative_noise": {
        "title": ("Gain modes × positive signal", "增益模式 × 正信号"),
        "subtitle": ("Multiplicative noise", "乘性噪声"),
        "formula": r"\mu=e^{Up_g}\odot(Ap_n),\quad y\sim\mathcal N(\mu,\,\mathrm{diag}[(0.001\mu)^2])",
        "model": (
            "Two gain modes and two positive signal coefficients: U and A are known 128 × 2 matrices. Each p_g coefficient has a Uniform(−0.5, 0.5) prior, and each p_n coefficient Uniform(0.1, 3). U has no constant mode, avoiding the gain/signal scale ambiguity.",
            "两个增益模式与两个正信号系数：U 和 A 为已知的 128 × 2 矩阵。每个 p_g 系数的先验为 Uniform(−0.5, 0.5)，每个 p_n 系数为 Uniform(0.1, 3)。U 不含常数模式，以避免增益与信号的尺度退化。",
        ),
        "structure": (
            "Gain is conditionally log-linear; signal is conditionally linear with changing covariance. The compiler tests these structures for bounded proposals with MH; the saved plan records which route ran.",
            "增益条件对数线性；信号条件线性，但协方差变化。编译器检查这些结构能否构造有预算限制的 MH 提议；保存的方案记录实际执行路径。",
        ),
        "simulation": (
            "Generate observations with 0.1% relative noise. During inference, sigma is recomputed from the current mean at every likelihood evaluation.",
            "以 0.1% 相对噪声生成观测。推断时每次计算似然，都会由当前均值重新计算 sigma。",
        ),
    },
    "composed_process": {
        "title": ("One random field, six linked stages", "一个随机场，六个相连阶段"),
        "subtitle": ("Composed process", "复合随机过程"),
        "formula": r"\begin{gathered}P_k(a)=(a/k)^2,\quad s_k\sim\mathcal N(0,P_k(a))\\\theta=(c,w),\quad h(x;\theta)=0.7\exp\!\left[-\frac{(x-c)^2}{2w^2}\right]\\v=Rs+h(x;\theta)+Bb,\quad \mu=e^{Ug}\odot v\\y=\mu\odot(1+f\epsilon),\quad \epsilon_i\sim\mathcal N(0,\sigma_w^2),\quad f=1\end{gathered}",
        "model": (
            "A 12-coefficient Fourier Gaussian process has one unknown power amplitude. Its random instance passes through a known linear response, a two-parameter Gaussian bump, a two-coefficient linear background, two log-linear gain modes and multiplicative white noise with an unknown scale: 20 unknown coordinates. All root parameters have finite Uniform priors; the process conditional remains Gaussian.",
            "12 个 Fourier 系数组成的高斯过程由一个未知功率振幅控制。随机实例依次经过已知线性响应、双参数高斯凸起、双系数线性背景、两个对数线性增益模式，以及尺度未知的乘性白噪声，共 20 个未知坐标。顶层参数均使用有限区间 Uniform；过程的条件分布保留高斯。",
        ),
        "structure": (
            "At fixed hyperparameters and other blocks, the instance and background enter the mean jointly linearly. The compiler tests their union for a 14-coordinate Gaussian GLS proposal with MH, leaving six coordinates in NUTS. The saved plan records the executed grouping.",
            "固定超参数及其余块后，实例与背景共同线性地进入均值。编译器检查它们能否合成一个 14 维高斯 GLS 提议块并使用 MH，其余六个坐标由 NUTS 更新。保存的方案记录实际分组。",
        ),
        "simulation": (
            "Draw one instance from the declared spectrum, then generate noisy observations on the specified input grid. White noise is integrated out by the Normal observation law during inference. Fixing f=1 separates the unknown sigma_w from an otherwise unidentifiable product.",
            "先从声明的功率谱抽取一个实例，再在指定输入网格上生成带噪观测。推断时，Normal 观测分布解析积分掉白噪声。固定 f=1 后推断 sigma_w，避免只能识别乘积的退化。",
        ),
    },
}
