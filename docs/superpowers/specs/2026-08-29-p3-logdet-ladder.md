# P3 — logdet 阶梯：从可证明的特例到有证书的近似

本文是 P3 的唯一决策家，承接计划
[`2026-08-29-coupling-collapse-ladder.md`](../plans/2026-08-29-coupling-collapse-ladder.md)
的 M8。记

\[
\Sigma=\Lambda+P=(I+X)\Lambda,\qquad X=P\Lambda^{-1}.
\]

`Lambda` 是调用方显式选择的预条件子，不默认等于噪声协方差。
`check_logdet_premises` 只判定、不计算；调度从上到下由第一个被实际验证的前提胜出。
公开 `logdet.py` 是稳定门面；直接 NumPy 方法、前提/调度、证书/计划、纯 JAX runtime 分居
`_logdet_eager.py` / `_logdet_ladder.py` / `_logdet_plan.py` /
`_logdet_runtime.py`，四个实现文件分别为 799 / 444 / 545 / 60 行，均低于 800 行；
runtime 没有 Python 收敛分支。

## D79 — 问题：按什么顺序和前提选择方法？；裁决：采用基例加 1–8 级，首个已验证前提胜出，第 8 级无条件拒绝。

| 顺序 | 方法 | 结果 | 具体、可检查的前提 |
|---:|---|---|---|
| 0 | `Lambda` 本身 | 精确 | `LogDetProblem` 构造时已验证 `Lambda` 数值对称且 SPD，且 `Sigma == Lambda` 逐元素成立。 |
| 1 | 低秩 determinant lemma / 有限 e 多项式 | 精确 | 代数因子重构 `P`；列数同时不超过两个低秩阈值；`Sigma` 数值对称且 SPD。稳定因式载荷不要求 `rho(X)<=1`，也不用 `1/eps` 条件门（精度由因子证书与 `κ(Λ)` 决定）；紧凑对角输入以精确非零支撑给出阶数。 |
| 2 | 状态空间 / block-LDL | 精确 | 块大小整除维数；远邻块逐位为零；`Sigma` 数值对称且正定；递推条件数严格低于 `1/eps(dtype)`。 |
| 3 | 结构化精确式 | 精确 | 条目逐位通过对角/circulant/Toeplitz/Kronecker 检查且 `Sigma` SPD；非对角递推/变换载荷通过 `1/eps(dtype)` 分辨率门；Kronecker 每个因子必须非空、方形且在 Cholesky 载荷的 SPD 域内。标签不是证据。 |
| 4 | 稠密 Cholesky | 精确 | `n <= dense_max_n` 且 `Sigma` 数值对称、SPD；condition 只报告诊断，不借用矩阵求逆的拒绝阈值。 |
| 5 | 有限 e 多项式微扰 | 精确 | 尺寸或代数 rank 在阈值内且 `Sigma` SPD；通用有限展开要求实测 `rho(X)<=1`，已经验证的 determinant-lemma 稀疏分支既不要求 rho 也不用 `1/eps` 条件门。 |
| 6 | 截断 trace-log | 确定性近似 | dispatcher 逐位验证确定性 `Tr(X**r)` 覆盖固定阶数，且标量 rho 证据不低报实测值并严格 `<1`。若调用方要求 tolerance 承诺，则计划工厂另验完整 `RhoCertificate` 的重数、order 与浮点精度。 |
| 7 | 冻结 Hutchinson trace-log | 对冻结探针确定的近似 | 对象的精确类型必须是 bytes-backed `FrozenProbes`；非负固定阶数；标量 rho 证据不低报实测值且严格 `<1`。直接入口只读一次被检查的 probe 缓冲区；计划工厂再验完整证书；抽样误差不继承第 6 级的解析尾界。 |
| 8 | 每调用重抽探针 | **拒绝** | 无条件拒绝；逐调用噪声会改变 HMC 目标并破坏 leapfrog 可逆性。 |

第 5 级的代数对象可由 Newton 恒等式写成

\[
\det\Sigma=\det\Lambda\sum_{j=0}^{n}e_j(X),\qquad
j e_j(X)=\sum_{q=1}^{j}(-1)^{q-1}e_{j-q}(X)\operatorname{Tr}(X^q).
\]

第 6 级使用

\[
\log\det\Sigma=\log\det\Lambda+
\sum_{r=1}^{\infty}\frac{(-1)^{r+1}}r\operatorname{Tr}(X^r),\quad \rho(X)<1.
\]

标量方向的尾界是
`rho^(m+1)/[(m+1)(1-rho)]`；whole-logdet 证书必须乘显式重数
`N_trace`。`X=rho*I_n` 会让漏乘的证书恰好少报 `n` 倍。公开计划工厂把
`certificate.multiplicity` 与问题的代数 rank 界绑定。普通 dispatcher 只选择适用方法，
不宣称一个未提供的 tolerance；带误差承诺的 JAX 入口必须经计划工厂。审计的 `n=40, rho=0.5,
tol=1e-6` 反例中，重数 1 / order 17 现在被拒绝，重数 40 / order 22 被接受且误差在界内。

一般黑盒 matvec 不能从有限次作用推出精确 power trace。因此第 6 级只有“提供并逐位验证
精确迹”这一条已实现入口；未实现的“结构可导出”析取已删除。随机迹估计只能是第 7 级。

精确结构行的稀疏/模板证据不使用容差准入：`structure_rtol/atol` 不会把近似 diagonal、
block-chain、circulant 或 Toeplitz 投影成精确结构；调度和直接载荷都要求结构条目逐位相等。
数值对称性单独使用这两个阈值，以接纳 D1 的 sub-ULP matmul 舍入；边界测试把同一容差
当作“接近结构”的诊断坐标。
测试仍在各自最后 `isclose` 浮点和首个拒绝浮点上
绕过调度器，直接对拍 dense Cholesky，并断言结构载荷在三格都拒绝。原因由两个反例钉死：
近奇异对角阵中一个 `atol` 大小的非对角元能主导 logdet；block-chain 中一个容差大小的远邻块
能决定正定性。把两者投影为“精确结构”会产生静默大误差。低秩阈值的 rank 6/7/8/9 ×
`rho={0.5,1,100,1e4,1e9}` 也直接求两侧：稳定 determinant lemma 在阈值两侧都与独立
lemma 和 slogdet 一致；`rho<=1` 只留在真的有限展开/trace-series 分支。紧凑对角直接求值覆盖 `n=rank={1,10,100,10000}`，包括成功的
10 000 次稳定因式化终止；不是只读取 verdict 的记账测试。第 1 级是优先级更高的特例，
但优先级不等于无条件成本承诺：无可用稀疏表示时，稳定载荷会退到 Cholesky。

D2 的决定性回归 `n=1000,k=4,rho=100,cond(Sigma)=101` 现在在 level 1 返回
`1851.0419458156762`，对独立 `slogdet` 的相对误差为 `9.83e-16`；不再因 trace-series 的
收敛条件而整梯拒绝。D1 的 `8.881784197001252e-16` matmul 不对称度被数值对称/SPD 前提
接受，而明显非对称输入仍拒绝。第 7 级把 `trace_order>=0` 放进前提，所以负阶数在 checker
处就是 False，不会产生“前提满足、载荷才抛错”的假裁决。

条件数策略按实际载荷区分。Fisher 的 `1/sqrt(eps)` 来自显式求逆及条件数平方，不能移植给
logdet：D4 在该阈值两侧和 `cond=1e14` 都直接验证了 Cholesky，因此没有任何一级借用它。
logdet 自己的分辨率门是更弱的 `1/eps(dtype)`，且**只属于 2D 稠密消元/递推/变换载荷**：第 0 级
（`Sigma == Lambda` 的 2D base 稠密算术）、第 2 级（block-LDL 递推）、第 3 级（非对角变换载荷）、
第 4 级（稠密 Cholesky）；determinant-lemma 载荷与对角结构化载荷**不以它拒绝**，condition 仅作
诊断。`nextafter(1/sqrt(2),0)` 构造的 3×3 近奇异矩阵曾让 block-LDL 和 Toeplitz 分别静默偏离
oracle `0.0589` 与 `0.1011`，所以递推/变换载荷保留这个门。这个门比借来的阈值宽约
`1/sqrt(eps)` 倍，仍会拒绝已经没有一个有效数字可分辨的递推。

## D80 — 问题：为什么第 1 级不是另一套 determinant-lemma 分支？；裁决：它是第 5 级在 rank `k` 的稀疏终止，两入口共享 `_newton_logdet` 的稳定因式化路径并逐位一致。

若 `rank(P)=k`，则 `rank(X)<=k`，所以 `j>k` 的所有 `e_j(X)` 精确为零；级数自动终止。
这是行列式引理与有限 e 多项式的同一代数事实。测试要求第 1、5 级公开入口逐位相等。
直接用 power-sum Newton recurrence 求值并不数值安全：`n=128` 的混合符号谱曾静默错
13.8%，`X=I_2000` 则在有限 logdet 仍可表示时溢出。实现因此在同一 `_newton_logdet`
核心内按已验证稀疏表示因式化求值：compact diagonal 直接逐项求
`fsum(log(Lambda_i+P_i))`，同缓冲自因子用 `k×k` Cholesky determinant lemma，其余稠密
输入用 Cholesky；不再把代数恒等式误当浮点算法。直接逐项相加也避免了
`logdet(Lambda)+sum(log1p(P/Lambda))` 在 `Lambda=1e15, Sigma=1` 时约 `8e-4` 的相消错误。

数值 SVD rank tolerance 不是代数证据：`Lambda^-1` 能把 `P` 中看似很小的遗漏放大到
量级一。更强的反例是在 `rho` 接近 1 的 SPD 边界，`5e-16 I` 的遗漏即可让 logdet 改变
`0.248`；所以任何非零重构残差都不能作为“精确” rung 的代数 rank 证明。实现只接受
`np.array_equal`，近似因子必须改走带显式误差传播的近似 rung。`LowRankFactors(L)` 和
逐位相等的 `LowRankFactors(L,L)` 都让左右因子共享同一不可变缓冲区，避免检查器自己沿
另一条 BLAS 缓冲路径制造残差。九个复现形状
`(20,2), (50,4), (12,9), (37,6), (101,7), (200,8), (300,5), (64,3), (260,4)` 现在全部逐位
重构为 True；`P` 中 `5e-21`、经 `Lambda^-1` 放大为 `0.5` 的遗漏仍被拒绝。

## D81 — 问题：哪些近似可作为 HMC 目标？；裁决：确定性截断和冻结探针安全，每调用重采样拒绝；eager 工厂与 JAX runtime 构造性分层。

有限截断是 theta 的确定性可微函数；冻结 common-random-number 探针也让每次求值面对同一
函数。每调用重抽会改变势能，必须拒绝。`make_trace_log_plan` 和
`make_frozen_trace_log_plan` 在 trace 外绑定 order 与探针；runtime 只接受 theta 相关数组。
`FrozenProbes` 以 bytes 为 backing store；duck-typed `.values` 对象与覆写 `values` 的子类
都不能冒充它。

theta 相关输入在 factory 点之后仍会改变数值尺度，而非正规矩阵乘法的内部相消也不能由
最终 probe estimate 的量级控制；所以精度由证明决定，而不是无条件写死 float64。
工厂比较最终 logdet 的 ULP，并用 `abs(logdet Lambda)+sum(abs(term))` 的固定操作数 gamma
界检查 warmup 的保守 `max_abs_lambda_logdet` 与 series scale；没有该 base-scale 证书时计划
工厂拒绝。计划绑定构造时 dtype，调用离开 x64 环境也会拒绝。
exact trace plan 把总 tolerance 的默认一半分给解析尾项，另一半留给算术，并最终强制
`whole_trace_tail + roundoff_bound <= tolerance`；两项不能各自花掉整份预算。Frozen plan
不声称 Hutchinson 抽样误差界，其 tolerance 只约束已固定估计器的运行期算术。
float32 在 tolerance `1, 1e-1, 1e-2, 1e-3, 1e-4` 的实测网格通过，在 `2e-5` 及更严时
由 ULP 或 gamma 界拒绝；需要 float64 的是具体误差预算，不是计划类型本身。float64 计划仍
必须在同一 `jax.enable_x64(True)` 上下文执行。冻结 runtime 对紧凑对角 `X` 使用逐元素 action；测试在
`n={2,10,100}`、`p={1,n/8,n}` 上与 eager 和独立 slogdet 对拍。把 runtime 内核 mutation
成常数会被数值测试杀死。

## D82 — 问题：theta 相关前提如何贯穿 warmup、runtime 和 retained samples？；裁决：warmup 保守定阶，runtime 不检查，事后分别复核 rho 与精确迹 provider。

`rho(X(theta))` 生命周期是：warmup 探测点集取最大值加 margin，以 whole-trace 重数界固定
最小 order；runtime 不放 traced Python guard；`audit_retained_rho` 在保留样本上报告越界
索引。**Warmup rho certificate is, like solver `tol`, the only number between the user and
silent error.** 这个数不是装饰性诊断，而是用户与静默错误之间唯一的数值屏障。

精确 power-trace provider 同样依赖 theta：构造计划时逐位验证，runtime 不检查，结束后由
`audit_retained_power_traces` 在每个 retained `LogDetProblem` 上按同一固定 order 重算，并
重新验证其代数 rank 界没有超过 warmup 的 `certificate.multiplicity`。provider 或重数任一
变化都会报告违规索引；空 retained 集合必须拒绝，不能返回空洞的 `passed=True`。任何一项
事后审计失败，都撤销对应近似目标的适用声明。

绝对 tolerance 还有第三个 theta 相关量：`logdet Lambda` 的算术尺度。warmup 用探测值加
margin 固定 `max_abs_lambda_logdet`；工厂以它做舍入上界，保留样本由
`audit_retained_lambda_logdet` 复核。省略这项证书、或事后越界，都撤销绝对误差承诺。

冻结 Hutchinson 还多一个非正规矩阵 action 的尺度：谱半径不控制浮点 matvec，而 IEEE
误差界需要 `||abs(X)||_2`，普通 `||X||_2` 会因符号相消最多少报一个维数相关因子。warmup
因此把 `max_x_operator_norm` 明确定义为 `||abs(X)||_2` 的上界，工厂对实测值交叉核验，
并以 probe energy、维数、探针数 `p` 和固定 order 构造 matmul/dot/mean 舍入界；
`audit_retained_operator_norm` 复核 retained `||abs(X)||_2`。Hadamard 审计例中二者为
`0.5` 与 `2.0`，普通 norm 证书被拒绝；`X=[[0.1,1e8],[0,0.2]]` 虽然 `rho=0.2`，仍会因
action 舍入界超 tolerance 而拒绝。`p=10000` 的 reduction 也不能再通过一个与 `p` 或
固定 order 无关的界。

对抗审计列出的 21 个无测试压力守卫中，4 个被证明不可达并已删除：低秩重构 shape 的
后置检查，以及 runtime precision 中被上游严格 rho 证明蕴含的 3 个 Sigma 检查。余下 17 个
可达守卫逐个经只读 import-hook 删除；17 次独立 pytest 调用均为 exit 1，包括 ULP 证明和
空 retained trace audit 两个会静默接受的高风险守卫。这里的“覆盖”是删除变异结果，不是
只执行到相应源码行。

## D83 — 问题：谁选择 `Lambda`，幂迭代能否给严格 rho 证书？；裁决：`Lambda` 是调用方的预条件子设计；一般非对称 `P Lambda^-1` 不接受下偏幂迭代作严格证书。

前景主导时相对噪声直接展开可能 `rho>=1`；更接近 `Sigma` 的对角、块对角或 circulant
`Lambda` 可让级数收敛。仓库已有 `exact.conditioning` 的谱迭代和 precision 的结构 action，
但没有有物理依据的通用 `Lambda` selector。SPD 幂迭代从下方逼近，且一般
`P Lambda^-1` 非对称；它不能安全证明严格上界。当前 dense/compact-diagonal 路用
`np.linalg.eigvals`/对角最大值精确测量。dense oracle 测试用已知特征值
`Q diag(rho,0.2rho) Q.T` 的字面参数 `rho`，不再与实现共享 `eigvals`；其对角最大值严格小于
`rho`，因此会杀掉低估谱半径、错误放行 `rho=1.01` 的变异。

代价必须明说：dense `spectral_radius` 会物化 `X` 并做 `O(n^3)` eigvals；它是 warmup
验证，不是无矩阵 runtime。probe 22 的解析 rho 和计时外 `eigvalsh(B.T@B)` 是 M8 特化，
不能当作发布模块的通用成本。

## D84 — 问题：认证后的 trace-log+Neumann 是否救回 M8 的 `k=512`？；裁决：没有；whole-trace 认证显著降误差但进一步拉高成本，P3 不承诺大块 collapse 加速。

原 M8 每梯度事实保留：

| n | k | collapse QR | 条件梯度 | 一次 action | `r_QR` |
|---:|---:|---:|---:|---:|---:|
| 100 | 8 | 43.2 us | 5.2 us | 6.8 us | 8.3 |
| 100 | 64 | 332.8 us | 5.7 us | 7.7 us | 58.6 |
| 400 | 256 | 5 312 us | 9.5 us | 37.5 us | 559 |
| 1000 | 512 | 28 012 us | 13.0 us | 110.6 us | 2163 |

probe 22 比较完整 collapsed 目标：固定 `B.T@B` 谱给精确 power traces，并用同阶 Neumann
级数算 quadratic；setup、JIT 和一次性谱均不计时。所有这些不对称都偏向 trace 路：QR
没有一份对应的免费分解，rho 也用解析式而没有支付发布模块 dense eigvals 的 `O(n^3)`。
模型为 CPU float64、`theta=0.7`、`prior_std=2`、`noise_std=0.5`，whole-logdet tolerance
`1e-6`。被计时的是 probe 内的 M8 特化，不是发布的通用 runtime；百分比依赖机器状态。

2026-08-30 按认证重数 `n` 的四次重跑：

| run | n | k | QR us | 条件 us | action us | `r_QR` | rho | m | 完整 trace us | `r_trace` | grad rel_err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| certified 1 | 100 | 8 | 22.6 | 10.6 | 6.3 | 2.1 | 0.847 | 94 | 399.8 | 37.6 | 1.682e-4 |
| certified 1 | 100 | 64 | 191.0 | 13.6 | 7.8 | 14.1 | 0.930 | 215 | 889.5 | 65.4 | 1.447e-5 |
| certified 1 | 400 | 256 | 3964.2 | 10.3 | 26.5 | 385.0 | 0.926 | 220 | 8487.9 | 824.3 | 4.638e-6 |
| certified 1 | 1000 | 512 | 20914.1 | 10.5 | 95.9 | 1996.8 | 0.919 | 212 | 38060.3 | 3633.9 | 2.803e-6 |
| certified 2 | 100 | 8 | 23.9 | 10.9 | 7.1 | 2.2 | 0.847 | 94 | 478.0 | 43.8 | 1.682e-4 |
| certified 2 | 100 | 64 | 191.7 | 10.5 | 5.8 | 18.2 | 0.930 | 215 | 892.1 | 84.7 | 1.447e-5 |
| certified 2 | 400 | 256 | 3876.6 | 10.6 | 26.4 | 365.0 | 0.926 | 220 | 8469.5 | 797.3 | 4.638e-6 |
| certified 2 | 1000 | 512 | 21015.1 | 10.3 | 94.7 | 2046.4 | 0.919 | 212 | 39251.4 | 3822.2 | 2.803e-6 |
| certified 3 | 100 | 8 | 32.3 | 12.6 | 7.2 | 2.6 | 0.847 | 94 | 602.0 | 47.9 | 1.682e-4 |
| certified 3 | 100 | 64 | 208.6 | 15.2 | 10.2 | 13.7 | 0.930 | 215 | 946.5 | 62.3 | 1.447e-5 |
| certified 3 | 400 | 256 | 4318.5 | 13.1 | 28.2 | 330.6 | 0.926 | 220 | 9905.6 | 758.3 | 4.638e-6 |
| certified 3 | 1000 | 512 | 24029.4 | 13.3 | 121.2 | 1801.5 | 0.919 | 212 | 49522.9 | 3712.8 | 2.803e-6 |
| certified 4 | 100 | 8 | 21.3 | 15.5 | 6.9 | 1.4 | 0.847 | 94 | 429.9 | 27.8 | 1.682e-4 |
| certified 4 | 100 | 64 | 201.1 | 13.0 | 6.9 | 15.5 | 0.930 | 215 | 818.7 | 63.2 | 1.447e-5 |
| certified 4 | 400 | 256 | 3942.8 | 12.9 | 26.5 | 305.0 | 0.926 | 220 | 8843.4 | 684.0 | 4.638e-6 |
| certified 4 | 1000 | 512 | 21017.8 | 10.3 | 108.8 | 2035.2 | 0.919 | 212 | 41168.3 | 3986.5 | 2.803e-6 |

`k=512` 的旧标量规则用 `m=135`，梯度相对误差约 `1.17e-3`；认证规则用 `m=212`，误差降到
`2.803e-6`；四次中 trace 需 38.06–49.52 ms，而 QR 为 20.91–24.03 ms，即慢约 82–106%。
机器状态使百分比波动，但最新重跑落入审计预估的 95–130%，方向和裁决不变：不能把它
包装成通用大块加速。

同次 probe 21：所有精确行 rel_err 为 0 至 `1.792e-16`；第 6 级在 `rho=0.5,m=12,n=4`
的 rel_err 为 `1.079e-6`，whole-trace 尾界 `7.512019e-5`；第 7 级改用非对角 `X` 和
`p=2<n=4` 后 rel_err 为 `2.612e-2`，且不再打印第 6 级尾界。第 8 级报告 `REFUSED`。

## 审计续（2026-08-30）：F3 / F4 裁决

- **F3**（`_power_traces_match` 逐位比较，`np.array_equal`）：**记录为何容忍，不改**。
  精确迹 provider 是证书的一部分（D82）：计划按固定 order 与逐位验证的迹构造，误差界
  假定 provider 与运行期浮点算术**完全一致**。Decimal 精确迹舍入到 float64 后差 1 ULP，
  说明它与运行期重算的迹不是同一对象，放行它会让证书误差界不再适用。调用方应提供
  **用同一 float64 算术算出的迹**，而不是更高精度再舍入的迹。
- **F4**（eta 证书对“巨大+微小”SPD 模式过保守，`Λ=I, P=diag(1e16,0,0), L=[[1e8],[0],[0]]`
  时 `eta=1` 拒绝但真实 logdet 误差仅 ~1e-16）：**记录，不改**。这是保守方向的假拒绝
  （安全侧），宁可拒绝也不放行一个无证书的近似；除非出现被误伤的真实用例，否则保持现状。
