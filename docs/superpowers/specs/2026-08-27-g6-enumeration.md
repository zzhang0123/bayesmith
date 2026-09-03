# 测量页 — **G6 的逐项登记**:证据消费面,一行一个判决

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§四 **G6 证据消费面**——原文只有一句「D12 包装所需缺口,**逐项登记**」。
> 本页把那句话兑现:把证据族九个模块的公开面**数出来**,逐个判它是缺口、是留守、
> 还是已有对应物。
> **日期**:2026-08-27 · **本页不是一个切换批次,也不是一个实现批次**,形状同
> `2026-08-27-numpyro-bridge-measurements.md` 与 `2026-08-27-d17-protocol.md`:
> 把一条没人量过的清单量出来,实现留给它自己的波次。

## 〇、为什么现在做,以及它解掉了什么

G6 是 **Wave D 的先决**,而它在计划里是一句没有内容的话:「逐项登记」。一条
先决如果没人知道它有多大,Wave D 开工时第一件事就是重新发明它——而那时上下文
里装的是切换,不是清点。**清点便宜、可重跑、不受发布门约束**,所以它现在做。

## 一、先数一遍(可重跑)

探针枚举九个模块里 `__module__` 属于本模块的公开函数与类,再按**名字**去
bayesmith 的十个候选模块里找对应物:

```
40 个公开名:6 个有同名对应物,34 个没有
```

有同名对应物的六个:`SqrtInfo`、`marginalise`、`marginalise_arrays`
(`evidence/sqrtinfo.py`)、`Factorization`(`evidence/factorize.py`)、
`compress`(`evidence/compress.py`)、`coherent_mode`(`evidence/diagnostics.py`)。

> **同名只是线索,不是判决**,而这一点在本程序里已经付过学费(`identifiability`
> 两侧同名而展开点不同,D21)。下表逐行判的是**它是什么**,不是它叫什么。

## 二、逐行判决

判据是计划 §〇 的五类:声明层 / 噪声物理 / 适配器 / 探针层 / 薄包装(容器)。
凡属**贝叶斯数值**的归 bayesmith;凡属**容器、格式、声明**的留守。

### 2.1 已有对应物(6)——**不是缺口**

| 名字 | 对应物 | 注 |
|---|---|---|
| `SqrtInfo` | `evidence/sqrtinfo.SqrtInfo` | 已有 cross-check(`test_sqrtinfo_agrees.py`) |
| `marginalise` / `marginalise_arrays` | 同上 | 同 |
| `Factorization` | `evidence/factorize.Factorization` | D12 下**容器留守**,算术过缝 |
| `compress` | `evidence/compress.compress` | 分层路由;远端已有 |
| `coherent_mode` | `evidence/diagnostics.coherent_mode` | 同名同题 |

### 2.2 **确定留守**(13)——容器、格式、声明,不是数值

| 名字 | 类别 | 为什么 |
|---|---|---|
| `CompressedLikelihood` / `QuadraticLikelihood` / `RawLikelihood` / `ReducedBasisLikelihood` | 容器 | **D12 已裁**:`__check_init__` 的异常身份在构造期抛、基类先行,**子类化无法翻译** |
| `EpochResidual` / `HeldOut` / `FidelityReport` / `ReducedBasis` | 容器 | 报告与字段布局;§〇 第 5 类 |
| `BayesMemory` / `ChainMemory` | 容器 | 同上;`ChainMemory` 还带一个活的 transition |
| `save_memory` / `load_memory` | **格式** | 在盘格式与 manifest 是**簿记**不是数值。远端没有、也不该有:一个通用贝叶斯包不需要知道 RHINO 的归档布局。**D39** 就住在这里 |
| `reject_bad_term` | 准入规则 | **2026-08-28 结清**,原 OPEN;它守的两个容器都在本表里,而远端**没有增量累加器可以守**。见 §三 |

### 2.3 **G3 名下**(**6**)——`exact.chain`,不是 G6

`chain.py` 的七个公开名里,容器 `ChainMemory` 归 §2.2,其余**六个**(`HyperTransition`、
`LinearGaussianTransition`、`chain_log_likelihood`、`chain_marginal`、
`ornstein_uhlenbeck`、`smooth`)整族属于 **G3**(RTS/Kalman + `linked` 转移),
计划 §四 已经为它开了自己的条目。**不重复登记进 G6。**

> 记在这里的理由:一次不加区分的「34 个缺口」会把 G3 的活算进 G6,于是两处都
> 以为对方在做。

### 2.4 **G4 名下**(**8**)——`exact.reduced_basis`,不是 G6

`reduced_basis.py` 的十个公开名里,两个容器(`FidelityReport`、`ReducedBasis`)
归 §2.2,其余**八个**属于 **G4**,同理不重复登记。

### 2.5 **真正的 G6**(**7** 个名字,全部确定)——D12 的包装会调、而远端没有的**数值**

| 名字 | 它算什么 | 为什么是缺口 |
|---|---|---|
| `compress_linear` | 把一个 epoch 的线性高斯似然压成充分统计量 | 远端 `compress` 是**路由**;这一层的两个具体压缩器远端只有其中一个的等价物 |
| `compress_reduced_basis` | 同上,针对共享基 | 依赖 G4;**排期上跟 G4 走,登记在 G6** |
| `epoch_residuals` | 逐 epoch 残差表(§9.3) | 远端 `evidence/diagnostics` 只有 `epoch_chi_square` 与 `coherent_mode` |
| `held_out_z` | 留一 epoch 的 z(§9.1) | 远端无 |
| `shrinkage_power` / `shrinkage_report` / `systematic_floor` | §9.4/9.5 的三条统计诊断 | 远端无 |

> **展开是 7 个名字,全部确定是缺口。** 上表把 `shrinkage_power` /
> `shrinkage_report` / `systematic_floor` 并成一行,所以行数与名字数不同
> ——**数字取自探针,不是数行**。
> `reject_bad_term` 原列在这里作第 8 个待判项,已于 2026-08-28 判为**留守**,
> 移入 §2.2。

## 三、本页原来留下的那条待判项,已结清(2026-08-28)

`memory.reject_bad_term` —— docstring 说是「每个累加器共享的准入规则」。本页当时
写的判据是:检查**结构**(名字、形状、支撑相容)就是容器的构造期契约,留守;检查
**统计**性质就是数值,归 G6。**本页当时不猜,而读完之后发现这条判据本身不够**
——它六条检查里两类都有。

**读出来的六条**:

| # | 检查 | 是哪一类 |
|---|---|---|
| 1 | `REQUIRED_TERM_MEMBERS` 都在 | 结构 |
| 2 | `latents_ok(term)`(bag 与 chain 答得不同) | 结构 |
| 3 | `prior_share[0] != 0` —— 加温过的项会把先验用两次 | **算术** |
| 4 | estimator 不一致 —— GLS 与全高斯似然「其和两者都不是」 | **算术** |
| 5 | `epoch_id` 重复 —— 数据算两遍 | 簿记(后果是数值) |
| 6 | `_reject_shared_inputs`(§9.5)—— 共享输入产品即非条件独立,实测 **52.6σ** | **统计** |

**所以按原判据它是混的,而混的东西不能靠原判据判。** 改用三条**可测**的事实:

1. **它守的两个容器都留守。** 唯一的两个调用方是 `BayesMemory.remember`
   (`memory.py:459`)与 `ChainMemory.remember`(`chain.py:994`),两个类都在 §2.2。
   一条守着留守容器的准入规则,没有可以迁过去的东西。
2. **远端根本没有它要回答的那个问题。** `src/bayesmith/` 里**没有任何**
   `remember` / `admit` / `add_term` / `append`——`compress_campaign` 是**一张图
   一次调用**折完所有 epoch。「这是别人建的一个项,能不能加进我已经有的?」在那边
   从来不被问。**迁这条规则要先发明它所守的那个累加器**,那是「一份实现」的反面。
3. **它读的每一个字段远端都没有。** `epoch_id`、`prior_share`、`estimator`、
   `inputs`、`Factorization.represents` 在 `bayesmith/evidence/` 七个模块里出现
   **0 次**(实测计数);远端的 `Factorization` 只有 `epoch_plate` / `survivors` /
   `per_epoch` 三个字段,**没有 `represents`**,整个 provenance 概念不存在。

**判为留守,而且不拆。** 第 3、4、6 条确实是算术与统计的陈述,§9.5 那条更是**通用
贝叶斯**而非射电天文;但它们每一条读的都是容器字段,而这个函数自己的 docstring 给了
不拆的理由:「a rule enforced in one accumulator and not the other is a rule with a
way round it」。**跨两个包拆,是同一个缺陷高一层的版本。**

> 顺带一条对本页原判据的更正,写下来以免下一位再用它:「结构还是统计」**不是**一条
> 能判归属的判据,因为一条准入规则天然两样都查。能判的是「**它读的东西在哪一侧**」。

## 四、这份清单本身的两条限制,写明以免被读成完备

1. **它按名字数,而名字是 `__module__` 属于本模块的公开函数与类。** 一个模块内
   的私名(以 `_` 开头)不在其中,而本程序已经因为「被未切模块 import 的私名是
   保持面」付过学费(铁律 1 的私名普查)。**G6 那一批仍要做私名普查**,本页不
   替代它。
2. **同名不等于同题。** 2.1 那六行只说「远端有一个同名的东西」,**没有**说两者
   数值一致——`SqrtInfo` 有 cross-check,其余没有。哪些需要 cross-check 由那一批
   按铁律 2 定。

## 五、结论,以及一次自己抓到的算错

**G6 不是 34 个缺口。** 探针数出来的分布(40 个公开名):

| 判决 | 数 | 是什么 |
|---|---|---|
| `HAVE` | **6** | 远端已有同名对应物(线索,不是判决) |
| `STAY` | **13** | 容器、格式、声明——D12 与 §〇 |
| `G3` | **6** | `exact.chain` 自己的条目 |
| `G4` | **8** | `exact.reduced_basis` 自己的条目 |
| `G6` | **7** | 真正的缺口 |
| `OPEN` | **0** | —— |

**G6 = 7,全部确定**(2026-08-28 起;此前是「7 定 1 待判」)。

> **本页第一版这里写的是「6 个确定加 1 个待判」和「17 个属于 G3/G4」,两个都错。**
> 正确是 7+1 与 14(**那是 2026-08-27 的正确;`reject_bad_term` 于 08-28 判为留守
> 之后是 7+0 与 14**——这条历史留着,因为它记的是一次算错怎么被抓到的)。
> 抓到它的不是复核,是**把判决表搬进探针、让它自己求和**——
> 手加的三个总数里错了两个。所以上表的每个数字都由
> `probe_14_g6_enumeration.py` 打印,本页只转录;探针里还有一条:一个上游新增
> 而没被分类的名字会被**大声**印成 `UNCLASSIFIED`,因为那正是这张表会静默过期
> 的方式。

计划 §九 给 Wave D 排的 4–6 个会话,现在有了一张可以对着走的表。

## 六、复现

```bash
cd /Users/zzhang/projects/bayesmith
/Users/zzhang/projects/rheplicant/.venv/bin/python docs/probes/probe_14_g6_enumeration.py
```

退出码 0 表示**数完了**,不表示清单没变——数字在它打印的表里,与 D17 的协议同一
个理由:一个把「清单变了」变成非零退出码的探针,会诱使下一位去让它变绿。
