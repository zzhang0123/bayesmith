# 交接 prompt — 把「全面移交」程序跑到完全结束

> 与 kickoff 同目录、同被跟踪。使用方式:把 `---` 以下整段粘贴给新 session。
> **日期**:2026-08-28(第二十一次改写)· 交接自 **Wave B 的 `linear` 求解面
> 与 `gls` 全模块两批切换**
> 之后的会话(前一次交接自 G5、G6 本体、D23、多数据集联合后验推导、架构叙事审计、
> `evidence/`→`marginal/` 改名、D46)。
> **0.5.0 已发布并核实上索引;历史重写已完成并强推;两仓已同步。**
> 本页 §四 是**剩余工作的完整盘点**,为 compact 之后接着做而写。

---

你要把「全面移交」程序**跑到完全结束**:把 rheplicant 的全部贝叶斯数值实现
移交给 bayesmith,`rheplicant.inference` 收缩为门面。计划已定稿,不要重新
评审、不要重新设计。

## 〇、owner 的授权(最重要,先读)

**owner 已授权:不要停下来问,一直做到程序结束。** 这条改变了裁决纪律的形态,
所以要写清楚它**具体**是什么意思:

- 计划 §二 登记簿里**仍未拍板**的 D 项,owner **委托给计划自己的建议**。对每一项:
  采纳该行 `*建议*` 所写的选项,并把裁决回填进那一行,格式为
  `**【owner 已授权委托,YYYY-MM-DD:取 (x),依 2026-08-27 的一次性委托】**`,
  后接一句你为什么读成这个选项。
- **委托不是空白支票。** 如果照建议执行会与**已实测的事实**冲突,**不要**照做:
  把冲突写进那一行,按事实选择,并在批次记录页里点名说明。**这已经发生过四次**
  ——D13 的版本号、D9 的整条建议、D21 的契约页那句、以及本页 §三 早先那条关于
  CLAUDE.md 的过期断言。所以这条不是理论。
- 遇到**计划完全没有预见**的新裁决点:自己选,选保守的一侧(能保住正确性、
  能被守卫钉住的那一侧),把它作为新条目写进登记簿并注明是本次委托下自定的。
  **D20、D21、D22、D23、D24、D25 都是这样来的。**
- **唯一仍要停下来的情况**:一个动作会**不可逆地对外**(发一个新 PyPI 版本号、
  强推覆盖远端、删除未跟踪的用户文件)**且**计划没有明写该动作。

## 一、第一步永远是读(按此顺序)

1. **计划本体**:`docs/superpowers/specs/2026-08-26-one-implementation.md`
   ——终局形态、七条铁律、登记簿 **D7–D45**、缺口 G1–G15、四波切换、附录 A/B/C。
2. 本程序至今的执行记录,同目录。**最近三份最要紧,按序读**:
   `2026-08-28-g5-amortize.md`、`2026-08-28-g6-consumption.md`、
   `2026-08-28-d23-two-criteria.md`。再往前:
   `2026-08-27-g6-enumeration.md`(G6 的逐项登记,§三 已结清)、
   `2026-08-27-g4-reduced-basis.md`、`2026-08-27-g3-chain.md`、
   `2026-08-27-wave-A-uncertainty-covariance.md`、
   `2026-08-27-p2-g2-fit.md`、`2026-08-27-p2-g10-g12.md`、
   `2026-08-27-p2-g14-condition-estimate.md`、
   `2026-08-27-ci-flat-chain.md`(**本机绿而远端红的唯一一次,读它的 §三 判别方法**)、
   `2026-08-27-wave-A-uncertainty-fisher.md`、
   `2026-08-27-wave-A-numpyro-bridge.md`、`2026-08-27-numpyro-bridge-measurements.md`、
   `2026-08-27-wave-A-priors.md`。其余:`2026-08-26-wave-P0.md`、
   `2026-08-26-wave-P2a.md`、`2026-08-27-d17-protocol.md`、`2026-08-27-d16-five-axes.md`、
   `2026-08-27-wave-P1.md`、`2026-08-27-wave-P2-G1.md`、`2026-08-27-wave-P2-G13.md`、
   `2026-08-27-wave-P2-G7.md`、`2026-08-27-wave-P2-D9.md`、
   `2026-08-27-wave-A-opening.md`、`2026-08-27-wave-A-identifiability.md`、
   `2026-08-27-wave-A-g1-wiring.md`、`2026-08-27-wave-A-sensitivity.md`、
   `2026-08-27-wave-A-s6-widened.md`、`2026-08-27-wave-A-g13-wiring.md`。
3. 前代 spec 的 §四(模块绑定契约台账)与 §六(本程序的由来):
   `2026-08-24-rheplicant-migration.md`。
4. 两仓工作笔记(操作规则,条条是学费):`/Users/zzhang/projects/e-RHINO/CLAUDE.md`
   (与 AGENTS.md 逐字节一致,有测试钉着)与 `/Users/zzhang/projects/bayesmith/CLAUDE.md`。
   **两份都在 2026-08-27 补了第 (0) 条的第二半**,见 §六。

## 二、当前状态(2026-08-28 晚,实测,先复核再信)

- **e-RHINO** 全套 **exit 0**(`-n 4 --ignore=tests/gui/e2e`,407 s),
  e2e 第二阶段 **21 passed**(69 s)。全套 **10143 passed**。**566 skipped**(其中 `tests/evidence` 是
  x64 子会话)。README 计数 **10729**(守卫核过,数字取自它自己的失败消息);
  拒绝普查 **254**;coverage **89.39 %** 实测(串行,`fail_under` **89**,
  README 写截断后的 **89.3**,不是四舍五入);bayesmith 地板 **`>=0.5`**。
- **bayesmith** 全套 **exit 0**(含 `tests/crosscheck/` **87 passed**,跨仓,
  读的是本地 e-RHINO 的 editable 安装);README 计数 **1530**;`0.5.0` 在 PyPI。
- **两仓已推送并按远端核实**(2026-08-28):e-RHINO `610d106`、
  bayesmith `38d6188` 之后 `git ls-remote` 两仓 `ahead=0`;
  本页这次改写与 `gls` 开波页在其后。
  **CI 全绿,四个 job 逐个核过**:rheplicant `Tests` run **33180929073**
  (`610d106`)的 `Suite (Python 3.12)` ✅ 与 `Coverage (serial)` ✅
  ——**Coverage 跑了约 49 分钟**(15:36 → 16:25),比 `Suite` 的约 20 分钟长得多,
  所以看结论时**按 job 等,不要按 run 等**;bayesmith 的 `Seam` ✅ 与
  `Cross-check` ✅(`d92791b`)。**两仓第二次同时全绿**,而这一次绿在
  `linear` 求解面已切之后。
  **用 `git ls-remote` 数,不要读本地 `origin/main`。**
  推送顺序:**e-RHINO 先,bayesmith 后**(bayesmith 的 Seam job checkout 的是
  `e-RHINO@main`)。
- 执行页共 **46** 份(含本页),探针 **17** 个。**登记簿 D7–D54**;
  未裁决只剩 **D39**。**D19 的延期部分已于本会话结清。**

**最近一次会话(2026-08-28)落地一批:Wave B 的 `linear` 求解面**——
四个公开求解名委托远端,**十二条拒绝逐条问过「过缝之后还到不到得了」**,
答案是十二条**全部**被缝抹掉、且原因是结构性的(那四个量在近端是参数、
在远端是 block 的字段)。其中**一条不留守就变成错答案**
(`check_noise_std_axis`),其余十一条只损失消息。见 §四.三 与
`2026-08-28-wave-B-linear.md`。

**再前一次会话落地四批**:

1. **G15 的 rheplicant 一半**(**D48**)——解除是**四处**不是计划写的一处;
   算术委托后与被删拼写**逐比特相同**;五条准入按 P1 总原则留守。
2. **CI 全量分诊**(**D49**、**D50**)——六条红里 **一条是真缺陷**
   (`_reject_a_foreign_block` 比较了一个非不变量),五条是守卫/环境。
3. **Wave B 开波**——铁律 1 普查、铁律 7 契约阅读,读出**两处过期契约**,
   其中一处关掉了 `CLAUDE.md` 记的那个「变异测试结构性盲点」的样本。
4. **D51**——计划点名要「显式重谈」的三条 config 侧测试,逐条裁决。

**已完成**:P0、P2a、D16、D17、P1、P2 余项全部、0.4.0 与 **0.5.0** 发布、
Wave A 全部五模块、G15(**两侧都完成了**)、G4/G3、D12 前置、G6 逐项登记、G5、
G6 本体(7 里的 6)、D23、多数据集联合后验推导、架构叙事审计、
`evidence/`→`marginal/`、D46、**e-RHINO 历史净化**、**CI 六条全处置**、
**Wave B 开波仪式**。

## 三、下一位会先撞到的几件事

1. **一条缺陷形状,本程序至今出现十次,当成收尾检查项**:**守卫够不到它要守的
   那个条件**。G10 的 W8、G14 的 X4、X4 的第一次修补、G3 的常数类、G5 的 Z9、
   G6 的 M6、D23 探针第一版、D23 判别 fixture 第一版、**G15 顺序守卫的
   monkeypatch(删掉它套件还是绿的,S1 第一轮 SURVIVED)**、
   **`_quadratic_form` 的第三个系数(没有任何 fixture 在 `rho != 0` 处问过它)**。
   收工前问:**「这条守卫真的制造出它要防的那个情形了吗?」**
2. **本会话新增一条,和上面是一对**:**一条守卫可能对着正确的数据开火**。
   `_reject_a_foreign_block` 在 x86-64 上抛 `StateValidationError` 指控「这条链被
   打乱了」,而块是对的。**大效应不等于真缺陷,但它值得先查**——判别法
   (`ci-flat-chain.md` §三)只告诉你哪些**不必**查。
3. **手加的总数**:能派生就不要手抄。README 计数本会话被守卫抓了**四次**。
4. **一个幸存的变异有三种成因**,不要当成一种(真缺口 / 够不到的代码 / 等价变异)。
5. **退出码**:本会话 harness 又说了两次「exited with code 0」而 `run.exit` 是
   `PYTEST_EXIT=1`。**把退出码写进自己的文件再读。**
6. **两个模块只切了一半**(`numpyro_bridge`、`uncertainty`),都不在 `SWITCHED` 里。
7. **`CLAUDE.md` 记的「变异测试结构性盲点」的那个实例已闭**(分支已并,按远端量过),
   **教训未变**,并补了方法:**问一条绿守卫依赖着什么你从未变动过的东西,
   并且用「量远端」来回答**。
8. **本会话新增,和第 1 条是第三个变种:守卫够得到,但够不够得到是随机的,
   而测试假装它不是。** 三条测试用**一个 key** 钉住「这个解必然被拒」,实测
   20 个 key 里只有 15 / 14 个被拒——`key(0)`/`key(1)` 只是恰好在被拒的那一半。
   切换把它们挪到另一半,于是三条同时变红,**而没有一条是因为代码错了**。
   收工前的第一问因此有了一个更锋利的形式:**「这条守卫的结果是确定的吗?」**
   ——如果不是,那个 pin 就是一张随机票据。处置见 D53:断言与 key 无关的东西
   (地板本身、拒绝的**种类**),外加一句「至少有一个 key 被拒」的自检。
9. **变异集的靶子集选错,长得和「守卫不存在」一模一样。** M12 记为 SURVIVED,
   而杀它的测试**一直存在**,只是不在我列的九个文件里。这是 `CLAUDE.md` 那条
   「变异被三个函数之外的守卫先杀了」的**镜像**:那条是「红的不是你写的那一条」,
   这条是「测试在,但你没跑它」。**两者在输出里都只是一行**。
   跑变异集之前先问:**这条被变异的代码,谁在测它?**——用拒绝普查或
   `grep -rl` 回答,不要凭印象列文件。
10. **`/tmp` 里的 `run.exit` 会活过会话。** 本会话 `until [ -f /tmp/full2.exit ]`
    命中了一个**两天前**的同名文件,于是一个还在 23% 的运行被读成了「已完成、
    退出码 1」。这与第 5 条是同一个家族(「是 X」对「这次查询根本没发生」),
    只是这一次骗人的不是退出码而是**文件的存在本身**。
    **把 `run.exit` 写进本次会话的 scratchpad,或者先 `rm -f` 再等**;
    等的条件用**日志里的摘要行**,不要用文件是否存在。

## 四、剩余工作(2026-08-28 全量盘点)

### 〇、本次会话结束时的状态

见 §二。要点:两仓干净、未推送(e-RHINO 领先 5,bayesmith 领先 4);
`0.5.0` 在 PyPI;登记簿 **D7–D51**,未裁决只剩 **D39**。

### 一、CI —— **两仓同时全绿,已验证**

推送 `d90028f`(e-RHINO)/ `d92ed9a`(bayesmith)之后:

```
e-RHINO   Tests 33171191850  Suite (Python 3.12) success  +  Coverage (serial) success
bayesmith Seam  33171200907  success        (checkout 的是 e-RHINO@main)
bayesmith Cross-check 33171200991  success
```

**这是本程序两仓第一次同时全绿。** `Tests` 在 2026-08-27 红两次、08-28 红一次,
两个 job 都红;六条失败已逐条查清并修好,详见 `2026-08-28-ci-triage.md`:

| 失败 | 判别 | 处置 |
|---|---|---|
| `Block 0 does not come from epoch 'collinear'` | **大效应**(2e1 对 2e-6 带) | **真缺陷**:`_quadratic_form` 比较 `z.z + offset`,而 `combine` 不保它;真不变量是 `offset - z.z/2`。已修 + 12 条新测试 |
| bias-budget 排序反转 | 小差值(两个量 ~1e-11) | 命题随随机 key 变号,不是方法的性质;换成**单位不变性**(argmax 翻转) |
| 三条壁钟预算 | 小差值(硬件) | 新增 `machine_factor()`,三个数字一个没放宽,标定表全保留 |
| A13 `336 Hz` 文案 | 小差值(半个 ulp) | 并入本模块**已有**的派生器 + 命名空间扫描守卫 |
| `tsc` 找不到 ×2 | 环境 | `Coverage (serial)` 补上 Suite 的三步 Node 配置 |
| `reweight_tol` 括号 | fixture 不存在 | **D50**:改为直接钉转发(monkeypatch + 兄弟测试) |

**六条里五条是关于 linux/x86-64 的,现在在那台机器上验过了。**
`Coverage (serial)` 绿也说明新增的测试没有把 coverage 压到 floor(89)以下。

**它后来确实又红了一次,而那一次的教训是本仓最贵的那条税。** 2026-08-28 推送
`gls` 批之后,`Suite` 红在 `test_inflight_axes.py` 的 axes 预算上:
**0.0961 ms 对 0.09**,差 6.8%。本机 10143 passed 全绿,而那个文件与本批无关
(`gls` 在里面出现 **0 次**)。

**根因不是硬件,是一个数字在两个地方拼写。** 那条断言与
`test_config_inflight.py:1780` 是**同一个调用、同一个界、同一句话**,而后者
**上一批已经乘了 `machine_factor()`**——两份拷贝,改了一份。更糟的是
`machine_factor` 自己的 docstring 开头写着「**Every absolute cost bound in this
directory is multiplied by this**」,而那句话**写下来的那天就是假的**:上一批
只改了**已经红过的三条**,留下五条没动。

**处置:修这一类,不是这一条。** `tests/config` 里八条标定预算现在全部乘
`machine_factor()`(靠 grep `best_ms(` 带比较找出来,逐条读过再分类,不是扫替换),
**一个数字都没放宽**;唯一不乘的那条(`< 10.0` 的上界控制)**就地写明了理由**
——它自己的 docstring 说它被紧的那条蕴含、是来承载契约数字而不是加灵敏度的,
给一条不可能失败的界乘系数只会让它**看起来像一次测量**。
docstring 也改成了成立的说法,并记下**是 CI 找出这个缺口的**。

**再红时的判别法不变**:按 `2026-08-27-ci-flat-chain.md` §三——**问这条断言钉的
是大效应还是小差值**;注意兄弟命题:**大效应也可能是守卫对着正确的数据开火**。
壁钟预算若再红,读失败消息里的 `machine factor`:**若它 > 5**,是参照负载与被测
pass 缩放不一致,要**重新设计参照**而不是放宽界。

~~清理:满意之后删 `backup/pre-draft-purge-2026-08-28` 与
`refs/original/refs/heads/main`~~ —— **已做,2026-08-28**。条件是「满意」,
而满意的判据取的是**历史重写本身**已在远端且 CI 绿(`Suite` 在 `610d106`
上 success,重写自 `d90028f` 起就在远端),不是当次 `Coverage` 的结论
——那个 job 与重写是否可信无关。删除前逐条核过:远端 tip 是本地 HEAD 的祖先,
两个 ref 都指向 `b2ef299`(重写前)。**这一行从此不再是待办。**

### 二、~~G15 的 rheplicant 一行~~ —— **已完成(D48)**

**不是一行,是四处**;算术与被删拼写逐比特相同;五条准入按 P1 总原则留守;
两包对「什么算高斯先验」差**恰好一个拼写**(`.expand([2])`),故过缝的是规范形。
地板已提到 `>=0.5`。计划点名要重测的那件(G9 广播缺陷)答案是「**仍然够不到**,
理由比原来的好」:`Latent.__check_init__` 在构造期就拒绝形状不符。
证据链:`2026-08-28-g15-rheplicant-discharge.md`、`probe_16_g15_discharge.py`。

### 三、Wave B —— **`linear` 与 `gls` 都已切,下一批是 `plan`+`engines`**

已做:开波仪式(`2026-08-28-wave-B-opening.md`)、**`linear` 求解面**
(`2026-08-28-wave-B-linear.md`,**D52**、**D53**、`probe_17`)、
**`gls` 全模块**(`2026-08-28-wave-B-gls-opening.md`,**D19 延期部分结清**、
**D54**)。`gls.py` **已进 `SWITCHED`**。

`linear` 那一批的四件套齐备,要点(细节在那一页,这里只留下一位需要的):

* **十二条拒绝全部留守**,因为缝把它们全抹掉了——`observed`/`noise_std`/
  `prior_std`/`prior_mean` 在近端是参数、在远端是 block 的字段。**枚举是派生的**
  (从四个公开名走调用图闭包),这很重要:其中**两条**住在兄弟模块里,
  grep `linear.py` 看不见。
* **一条不留守就变成错答案**:`check_noise_std_axis`。远端**静默给出**一个有限、
  形状正确、差 2.5e-03 的答案。**它不是本页发现的**——cross-check 早就记着,
  而且理由更深(numpyro 的 `promote_shapes`)。两处记录现在互指。
* **`linear.py` 不在 `SWITCHED` 里**,cross-check 文件**保留**:只切了求解面,
  `linear_operator`/`check_linearity` 仍在近端。与 `numpyro_bridge`、
  `uncertainty` 同一读法。14 条 cross-check 退役 8 条,每条都在
  `tests/exact/test_solve.py` 里**指认**了既有的家。
* **保持面到期提醒**:`gls.py` 借的 `linear._check_solve_arguments` 随 `gls` 到期。

**`gls` 已切完,四件套齐备**,要点(细节见那一页 §5):

* **分诊表是空的** —— 44 条 gls 测试第一次运行就全绿,一条没改;全套与切换前
  **同一个数**。原因是上一批已经把 `_check_solve_arguments` 搬进了门面。
* **两条拒绝留守**,理由不同:`NoiseModel` 那条是**缝抹掉了证据**(远端收
  `sigma_of` 可调用对象),是 `linear` 那条「拒绝 NoiseModel」的**镜像**;
  reweight 边界那条是**远端类不同**(`GraphError`,而它自己的注释说这是 misfit)。
* **D19 的延期部分结清:起点不动。** 而**第一次测量一文不值**——44 条全绿,
  但两条硬钉的步数**都是地板**,地板看不见种子。降低地板之后 `iterations`
  两侧都是 4,`delta` 从 1.6e-07 到 3.1e-07(都在舍入量级)。
  方向:对 float64 不动点,委托版 **9.61e-08**、旧循环 **1.055e-07**。
* **变异集 14/14**(第一轮 13 杀)。唯一的幸存 N13(`reweight_tol` 不转发)
  **不是「没有测试」**:远端用同一个公式算同一个默认值,所以调用方不声明时
  丢掉转发**恰好等价**。照 **D50 自己的办法**处理——直接钉接缝,而 D50 钉的是
  上一跳,本批新造的这一跳没人看着。
* **D54:`SWITCHED` 的别名表把 `test_noise_logdet.py` 记成 `gls` 的
  cross-check,而实测它与 `gls` 无关**(4 个函数,`iterative_gls` 出现 0 次)。
  这条别名在两个方向上都错:未切时用一个不测它的文件满足了「有 cross-check」,
  已切时会要求删掉 B1 的台账。已改为 `("gls",)`,`uncertainty` 那条不受影响。

**下一批:`plan` + `engines`**(计划 §五 Wave B 行:验证与词汇留守,执行经
G10)。已知三件:

1. **D51 第 3 条的源文本 pin 在这一批到期**
   (`test_the_package_guard_this_enum_mirrors_is_still_that_guard`,
   `inspect.getsource(SamplingPlan._prepare)`),**替代形态已写死在登记簿里**
   ——行为等价扫描,外加「加一个模式就要加一个候选」。
2. **B1 在这一批到期,而它已经被定位和量过了**(`probe_18_b1_logdet_gap.py`,
   本会话新增)。它**不是**门面切换,是**改生产采样行为**的一条真缺陷,
   是本程序至今第一条这样的项目 —— 所以下一位不要按前两批的节奏开工。

   **定位**(逐个 grep 出来的,不是读来的):`conditional_potential` 只喂
   **GRADIENT 块**(`plan.py:957`、`engines.py:544`);`chi2` 另外喂诊断迹线
   (`plan.py:1052/1068/1218`)。所以「同一模型两个出口不同目标」具体是
   **gradient 块 对 nuts 出口**。

   **量级**(scipy 在纯 numpy 闭包上求极值,两包的梯度都不参与,n=2e4):

   | f | 无 log-det 的极值 | 带 log-det 的极值 | 比值 | (1+f²) |
   |---|---|---|---|---|
   | 0.05 | 5.010654 | 4.998159 | 1.002500 | 1.0025 |
   | 0.2 | 5.193557 | 4.993752 | 1.040011 | 1.0400 |
   | 0.5 | 6.241192 | **4.990950** | 1.250502 | 1.2500 |

   两件事:比值在每个 f 上都是 `(1+f²)` 到四位,**所以这确实是台账说的那个量**;
   而**带 log-det 的那个极值回到真值 5.0,不带的没有**——**错的是 gradient 块**,
   在 f=0.5 上错 **25%**。这不是舍入。

   **补救没有被这次测量决定,而且很可能是两件**:给
   `conditional_potential` 加上 `Σ log σ` 修的是 **gradient 块**;
   **conjugate 块靠冻结 σ 丢掉同一项**,那一半的既定设计是
   `bayesmith.exact.correct` 的重要性权重——它要一个 `Graph`(`graph_bridge.to_graph`
   给得出),而且**会改变 `plan.sample` 的返回契约**。后者是一个 owner 会想看见的
   改动,不是一次沉默的等价替换。
3. `engines.py` 的 `_conjugate_transition` 用 `eqx.filter_jit` 而不是 `jax.jit`,
   而那是**正确性要求**(守卫的异常类),不是偏好——`test_conjugate_transition.py`
   钉着它,本会话已把那条从单 key 改成扫 key。

### 四、Wave C

`calibrate`、`npe`、`reduced_basis`。另有:
1. **`compress_reduced_basis`**(**D44**)与 G4 余下三名(`score_directions`、
   `build_reduced_basis`、`basis_fidelity`)同批。
2. **`npe` 接线按 D42**:三名委托、`simulate_pairs` 留守。**那一批要量的一件**:
   rheplicant 的 `NeuralPosterior` **类身份**有没有被测试钉住——若钉住,门面要包装
   而不是重导出(D12 在证据容器上的同一形状)。
3. **D33 的分诊**:`fit` 拒绝发散的下降,而 rheplicant 的 calibrator 今天**交回
   NaN**;有没有测试依赖它,**今天没有量过**,是那一批的第一件事。
4. **`min_scale` 的两层**:上游 config 拒绝 `min_scale: 0`,而 bayesmith 只拒绝负值。
   两者**不冲突**(上游更严),但要有人写下来。
5. R2 清单:`reduced_basis` 的测试族**必须**在 x64 会话(D41 已经把这半个答案定死)。

### 五、Wave D

`chain` + 证据族七模块。另有:
1. **D39 拍板**(归档 manifest 与二进制不绑定)。两个方向已钉住,解除条件写在行内;
   若绑定,是一次 `_FORMAT_VERSION` 提升,两份 fixture 要按新版本重写并**保留旧的**。
2. **要量的两件**(`2026-08-28-g6-consumption.md` §十):(a) `EpochResidual`/`HeldOut`
   的类身份有没有被钉住;(b) `systematic_floor` 上游读 `memory` 并**微分**其先验
   (`_prior_curvature`),而 bayesmith 的入口收现成的 `prior_fisher`——**那一层由谁算**。
3. `marginal/diagnostics.py` 现在 **775 行**,项目上限 800。**下次往里加东西之前先拆。**

### 六、P4 – P7

- **P4**:双岗已建好(`seam.yml` + `crosscheck.yml`),剩下的是 **cross-check 随模块
  退役**——每切一个模块,同批删它的 cross-check 文件,oracle 按铁律 2 改籍或指认。
- **P5**:18 个 run kind 逐 `_KINDS` 冒烟;每批四件套附 extractor 往返。
- **P6**:bayesmith 的发布列车**已走完** 0.1→0.5;**rheplicant 自己的发版还欠着**
  (计划 §七:「程序结束前 rheplicant 发版清 385 提交旧账」)。
- **P7 具名清单**:(a) e-RHINO 的 CLAUDE.md **与** AGENTS.md **成对**改(有测试逐字节
  钉着);(b) README 计数 pin;(c) coverage 截断值 + `fail_under`;(d)
  `_migration-to-bayesmith.md` 整篇;(e) `rheplicant/inference/__init__.py` 的模块
  docstring(**今日三论点在终局全假**,而它是 106 名单门户);(f)
  `tests/test_published_contracts.py`(27 条实测)与其 docstring;(g) bayesmith
  README/CHANGELOG。

### 七、两份新推导页留下的工作

**`2026-08-28-multi-dataset-joint-posterior.md`**:
- §9 是**真正要新建的东西**,最上游是 **`iota` —— 一个被声明的跨 graph 身份映射**。
  理由是本包已经做过的决定:`graph/trace.py` 的 `NodeRef._owner` 拒绝来自另一次
  `trace()` 的 handle,所以**从名字相等推断共享 latent 会反转它**。其余:列见证、
  第二个 graph 级 likelihood factor slot、**realify 层(D46 的另一半)**、
  从 rheplicant 移植的准入闸、nuisance 先验的类型分裂、Tier 1.5 的存储契约。
- §7 是**拒绝清单**(12 条),每条都写了它保护哪个数学陈述。**7.1 已落地(D46)**,
  其余 11 条未落地。
- §10 是**未决问题 + 能了断它们的测量**,9 条。
- §11 明确标了**哪些数字未验证**——引用它们之前先量。

**`2026-08-28-architecture-narrative.md`** §7 表里放行但未做的:
- **R9**:3 处 `Section 9.3` 解不到,而且**本会话让它变糟了**——新推导页有一个主题
  完全不同的 `### 9.3`,所以在 bayesmith 里 grep "9.3" 会得到一个**自信的错误答案**。
- **R10** 的兜底:一份随 sdist 发布的 `docs/plan-codes.md`(`src/` 里还有 ~80 处
  计划代号;**不能盲清**——`tests/exact/test_condition_estimate.py:243` 断言
  `"G14" in text or "D15" in text`,而那个测试在 sdist 里)。
- §8 的「不建议动的」**7 条是结论,不是待办**:不要改 `diagnose/`(e-RHINO 生产代码
  钉着它)、不要把 `optimize.py`/`amortize.py` 提升为子包、不要动 `sqrtinfo.py` 的
  位置与 `information()` 的命名、不要为「全包一致」去补齐门面(40/81 是惯例不是事故)、
  不要碰 `Graph.joint_prior: Any`。

### 八、接线时已经量好、不要重量的七件

- `fit` 收算好的 `step_sizes`;`_magnitude` 留 rheplicant。**两侧默认步长是否等价
  没有量过**。
- `condition_estimate` 两侧默认迭代数**相同**(都是 12),但这个数**改变答案**。
- **D33**:见 §四.3。
- **D19 的前提不成立**(实测):`floor=0` 的块在任何求解之前就被拒。
- **G15 的那一行**:见 §二。
- **Wave C 的 `npe`**:见 §四.2。
- **Wave D 的证据族**:见 §五.2。


> **第二个问题在 2026-08-28 被回答了一次,答案是「被安装的 bayesmith」。**
> `pyproject.toml` 声明 `bayesmith>=0.5`,而**这个 checkout 里没有任何东西能
> 检查它**:安装是 `../bayesmith` 的 editable 且 `--no-deps`(两仓互相开发,
> 这是有意的),所以那条地板从不经过解析器。实测:
> **`bayesmith.__version__` 报 `0.2.0`,而那边 `pyproject.toml` 写 `0.5.0`**
> ——metadata 是做 editable 安装那天写的,之后四次版本提升都没刷新它。
> **代码是 0.5 的代码**(editable 就是工作树),所以没坏;坏的是标签,
> 而 `src/`、`tests/` 里**没有任何东西读它**,这就是它没被发现的原因。
>
> 补了 `tests/test_bayesmith_floor.py`:**按能力断言,不按版本号**——
> `CLAUDE.md` 写明每一档地板买到什么,那四句话就是四条用例,外加一条
> 「pyproject 抬了地板而这里没有对应用例」的自检。**五条里三条在能用之前
> 就得先修**,而三条都是「问它能不能失败」问出来的、不是它自己红的:
> 0.4 那条**因为错误的理由通过**(它在 `bayesmith.exact` 里搜名字,而
> `gaussian.Probabilistic` 恰好有个**同名参数**,真正的函数在 `marginal` 里);
> 0.3 那条断言 `hasattr(cls, "__mro__")`,**对任何一个类都为真**;
> 而第一次瞄准 0.3 的变异**打在 docstring 行上**、报了一次干净的通过。
> 按 AST 读出行号重新瞄准之后,四档**各自独立地**被自己那条用例抓住。

## 五、批次纪律(铁律 4,每批四件套)

一个批次 = 一个模块集。完成 = 四件套齐备:

1. 分诊后**该批测试全绿**(分诊三列:原样重放 / 改写对适配器 / 带理由退役);
2. **接缝变异红**,按 §六 五行协议;
3. 旧实现删除,**计数守卫同批刷新**(README 计数、coverage floor);
4. **文档实测数字重测**。

每批把证据链写进本目录的 tracked 执行页(`2026-XX-XX-*.md`)。
**不要**把计划性文件放进 e-RHINO 的 `docs/superpowers`(gitignored,八份计划死在
那里过)。

**改判据 = 新裁决项**,要走登记簿并写进记录页——即使在本次委托下也是。

## 六、这台机器上的操作陷阱(全部实测于 2026-08-26/27/28)

**跑测试**

```bash
# e-RHINO(共享机器上两阶段跑,见其 CLAUDE.md)
.venv/bin/python -m pytest -n 4 --ignore=tests/gui/e2e > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
# bayesmith
.venv/bin/python -m pytest -n 4 > run.log 2>&1; echo "PYTEST_EXIT=$?" > run.exit
```

- **退出码写进自己的文件再读。** 只有 **1** 是测试失败;**5** 是一条都没收集到。
- **harness 的完成通知报的是复合命令的退出码,不是 pytest 的。** 2026-08-27 见了
  一次(通知说 0,`run.exit` 说 1,真有三条红的);**2026-08-28 又见了两次**,
  两次都真有一条红的(都是 README 计数守卫,它该红)。**这条不是偶发。**
- **不要再加 `-q`**:bayesmith 的 `addopts` 已有一个,叠成 `-qq` 会吃掉摘要行。
- e-RHINO 的**部分运行不需要 `--no-cov`**(那条规则早已反转)。

**变异集**

- **五行协议第 (0) 条有两半。** 先提交再变异;**而且每一次跑之前 HEAD 都必须已经
  是你想要回的东西**。修补写在第一轮和第二轮之间、还没提交,会被第二轮自己的开场
  `git checkout` 回退——**输出里没有任何东西说得出这件事**,一个被回退的修补和一个
  不起作用的修补长得一模一样。本会话实测,两条幸存因此被读了两遍。
- **恢复的路径不要宽于变异点本身**:一份脚本为了撤销只在 `src/` 里的变异而恢复了
  整个 `tests/`,那就是上一条的成因。
- **一个幸存的变异有三种成因,而它们要三种不同的处置**(2026-08-28 实测,一组里
  三种全见到了):**真缺口**(补 fixture)、**变异的是够不到的代码**(说出来,并钉住
  够得到的那一半——给它编 fixture 是更贵的错误)、**等价变异**(登记为必存;错的是
  期望值,不是守卫)。
- **一条变异可能必须幸存。** `priors` 批的 P3(合成数据 0 → 1e4)全绿,而这**正是**
  一条测试断言为真的性质。通常读法「没有守卫」在这里是错的;但**追到底仍然适用**
  ——追下去发现测量是拿手搭的图做的,适配器可能造出另一张,于是补了闭环。
- 变异集**可能要跑两个会话**:`priors` 批的两个靶子文件精度需求相反,合在一条命令
  里跑**基线就是红的**。

**git**

- **提交信息里出现 `-n 4` 之类会被 `block-no-verify` 钩子拦下**。把信息写进
  scratchpad 文件再 `git commit -F <文件>`,并且**不要在同一条复合命令里既跑
  `pytest -n 4` 又 `git commit`**。
- `cd` 不跨回合存活:每条 git 命令都用 `git -C <绝对路径>`。
- **推送后用 `git ls-remote` 核实**,不要读本地记录。
- **不要 `git add -A`**(它一天之内把九份草稿扫进库两次);逐路径 `git add`。

**计数与文档**

- e-RHINO 的 `tests/test_readme_counts.py` **按等号钉住** README 的计数,并且
  会**先红**。让它红,然后用**它自己报的数字**改 README——不要自己加。
- **拒绝普查的计数可以在内容变了两处时保持不变**(退役一条、新增一条)。
  `priors`/`G13` 那两批各发生过一次,而附录 B 因此**过期了一整批**没人发现。
  附录 B 的逐文件清单现在**由 `test_refusal_census._sites()` 重生成**,不要手抄。
- coverage floor 只住在 `[tool.coverage.report] fail_under`。
- **fixture 普查会无参驱动每一个 `*_document` builder**,所以一份文档能声明什么
  受限于**谁会去构建它**——`runtime.jax_enable_x64: true` 写进 builder 会打红两条
  毫不相干的普查测试(实测)。这类声明写进**运行它的 helper**。

**探针**

```bash
cd /Users/zzhang/projects/bayesmith
/Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_11_d17_dual_run.py
/Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_12_d16_five_axes.py
```

```bash
cd /Users/zzhang/projects/e-RHINO
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/seam
```

**全套里有两个 x64 会话**——`tests/evidence/` 与 `tests/seam/`,各有自己的门
conftest 与驱动。**另有两个 module 级 autouse x64 fixture**
(`test_prior_sensitivity.py`、`test_jeffreys_prior.py`),它们**各自制造过一次
「守卫不会失败了」**(S6,以及 D25 的范围只有三条而非四十九条)。**遇到一个
module 级 x64 fixture,先问它拿掉了什么条件。**

## 七、本程序反复付学费的形状(检查表)

1. **分不清「是 X」与「这次查询根本没发生」的结果。** 退出码、zsh glob、PyPI 索引、
   被回退的修补——同一个家族。
2. **比对悄悄变成「比两个不同模型」。** 每个 fixture 的数学只写一次,两侧各包一层。
3. **判据在正确的地方、时机却是错的。** `auto_blocks` 在求解期问了一个分区期的
   问题;G13 那批的守卫用了一个**没有 joint prior 的空间**去测「未覆盖的 latent
   保留先验」,于是变异前后两种读法都说「未覆盖」。**问「这个检查拿到它需要的全部
   输入了吗」。**
4. **一条从另一个库搬来的「常识」。** JAX 的 `bool * float` 是选择,NumPy 是相乘。
   跨库搬事实要重测;**一个幸存的变异要追到底**——真正的缺陷常在第二层。
   S6 是这条的第二次兑现:`astype` 做了两件事,承重的是**剥掉弱类型**那一半,
   而不是加宽那一半。
5. **一个数字在两个地方拼写,其中一份会过期。** 附录 B 过期了一整批;`identifiability`
   的契约页 §5.2 过期过;本页 §三 关于 CLAUDE.md 的那条过期过。**能推导就不要重抄。**
6. **一条链量不出的东西,不要让它去量。** `priors` 批第一版拿两条链的均值差去钉
   先验位移,跨三个种子比值 0.55/1.44/1.09 且有一个分量换了符号——那是
   `prior_sensitivity` 自己 docstring 里的算术。改用**共同随机数**之后判据里一个
   容差都没有:平先验必须让轨迹**逐比特不变**,不平的必须让它变。

## 八、上下文用尽时怎么收尾(不要硬撑)

程序全长估计 27–40 个会话,一个会话跑不完。**不要为了「跑到结束」而把批次做
半截。** 当上下文接近用尽:

1. 把当前批次收到一个**四件套齐备**的状态(或干净回退到批次开始);
2. **两仓提交,但不要推送。** owner 于 2026-08-27 新增:推送攒到全部任务做完时
   两仓一起推,因为每次推送触发两个 workflow、失败邮件太多。**这条只改推送时机,
   不改提交纪律**——每批仍必须收到四件套齐备并本地提交,否则一次上下文用尽就会丢掉;
3. 更新本文件(它是交接页):改写 §二 当前状态、§三 已处置的项、§四 剩余工作,
   让下一个会话能从这里直接开工;
4. 提交这次更新,然后停。

**没跑完不是失败;把一个半截批次留在工作树里才是。**

> **推送攒着有一个已知代价,写在这里以免它变成一次意外**:CI 只在推送时跑,而
> `2026-08-27-ci-flat-chain.md` 记的那次是「本机连绿五次而远端连红五次」。所以
> **最后那一次推送之后必须去看两个 workflow 的结论**,而且要预期它可能红——积压
> 的批次越多,一次推送要一起验证的东西就越多。那份记录页 §六 提的「把查一次 CI
> 写进收尾清单」**仍未入登记簿**;它是一条判据改动,要走登记簿,而**没有哪一批
> 自作主张改过铁律**。
