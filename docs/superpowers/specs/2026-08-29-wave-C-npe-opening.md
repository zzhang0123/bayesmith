# 执行页 Wave C / `npe` 开波 —— 全缝逐位相同,而形状由 7 条异常类 pin 决定

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。

> 计划:§五 Wave C / 铁律 1、7;裁决 **D10**(owner 2026-08-27 授权)+ **D42**。
> 前一批次:`2026-08-29-wave-C-calibrate-opening.md`。
> **日期**:2026-08-29 · **本页状态:开波、验收、切换都已做(§六)。**

## 〇、铁律 7 两问(新清单第一次实战)

1. **§四 哪一栏?** → **§4.3 不迁移**,和 `calibrate` 同一栏。
2. **最新点名裁决?** → **D10「NPE 迁移(`bayesmith.amortize`)」**,
   **owner 2026-08-27 授权**;由 **D42**(2026-08-28 自裁)细化范围。

**清单有效,而且顺手照出了 §4.3 的两条已作废**(`calibrate`→D11、`npe`→D10)。
**注意本页初版在这里写的是「整节作废」,那是错的**——§4.3 还有第三条「整个流式
证据层」(八个模块,含 `reduced_basis.py`)**仍然有效**,它说的是「不移植,
在这边通用化重写」,归 §五 B11。漏掉它的原因见该节标注:自动核查只找加粗的
**模块名**,而那一条加粗的是一个中文短语。
**它自己也被用出两个洞**,已回填规则:搜推翻要 `-i`(D10 标题写作大写「NPE」,
按 `npe.py` 搜会漏),以及要**整栏数**而不是逐个模块查。

## 一、铁律 1 私名普查 —— 干净

`from rheplicant.inference.npe import _*`:**全仓 0 处**。唯一私名
`_standardize` 只在本模块内用。

## 二、验收:**整条缝逐位相同**,而且必须现在量

| 量 | near vs far |
|---|---|
| 未训练 `log_prob` | **0.0** |
| **训练 200 步后** `log_prob` | **0.0** |
| `TrainingHistory.train` (200,) | `max\|Δ\| = 0.0` |
| `TrainingHistory.validation` (200,) | `max\|Δ\| = 0.0` |
| `TrainingHistory.best_step` | `0` |
| `create` / `train_posterior` 签名 | **逐字符相同** |
| `NeuralPosterior` 字段布局 | **九个字段逐名相同** |

`('train', 'validation', 'best_step')` 两侧同名同序。
**训练那一行是关键**:200 步 Adam、批采样、验证集划分,全部逐位重合,说明两侧
连 key 的拆法都一样。**这条和 `calibrate` 一样只能在删除之前量**,删完之后
近端调远端,同样的比较是循环的。

## 三、八条拒绝:六条过缝,两条留守(D42),**而形状由异常类决定**

| 位置 | 类 | 何时 | 判定 |
|---|---|---|---|
| `simulate_pairs` ×2 | `ParameterSpaceError` ×1 + `StateValidationError` ×1 | call | **留守**(D42:`simulate_pairs` 不迁) |
| `NeuralPosterior.create` ×3 | `StateValidationError` | **`create` 内,不是 `__check_init__`** | 过缝 |
| `train_posterior` ×3 | `StateValidationError` | call | 过缝 |
| *(远端多一条)* `min_scale` 非负 | `StructureError` | call | 近端无对应物 |

**六条文案 pin 全部成立**——不是看出来的,是**把每条 pin 的正则真跑到远端消息上**
得到的(`calibrate` 那一批我在这件事上错了两次,方法记在那一页)。

**但异常类不成立,而且被钉了 7 处**:
`test_inference_construction_guards.py` 里有 **7 个 npe 守卫测试**
`pytest.raises(StateValidationError, ...)`,远端一律 `StructureError`。
**所以裸重导出会一次打断 7 条**——这正是 D10(3)「**薄包装**保持三名」的理由,
而那句话现在有了可测的依据而不只是指示。

## 四、切换该长什么样(建议,未做)

**D12 的「子类化无法翻译」在这里不适用,而理由是可测的。** D12 说的是
`__check_init__`:异常在**构造期**抛、基类先行,所以子类改不了。
**而 `npe` 的三条拒绝在 `create` 里**(`@classmethod`),不在 `__check_init__`。
两侧的 `create` **都以 `return cls(...)` 收尾**(远端 `amortize.py:204`、
近端 `npe.py:244`),所以子类覆写 `create` 是干净的。

于是建议形状:

* `rheplicant...NeuralPosterior` **子类化** `bayesmith.amortize.NeuralPosterior`,
  只覆写 `create`——做三条拒绝的类翻译,其余(`log_prob`/`sample`/`_mixture`)
  **继承**,于是算术只有一份。
* **字段布局与 pytree 结构自动保持**(九个字段逐名相同,14 个叶子),
  这比「持有一个远端实例」好:后者会把 pytree 变成嵌套的,而
  `NeuralPosterior` 是 `eqx.Module`,config 层读它的 `create` 签名与
  `sample` 的位置参数契约。
* `train_posterior` 薄包装:翻译三条拒绝后转调。
* `TrainingHistory` 字段同名同序,**可直接重导出**(无身份 pin——实测 27 处
  引用 0 处身份钉)。
* `simulate_pairs` **原样留守**(D42),它是唯一碰 `ParameterSpace`/pipeline/
  `NoiseModel` 的一个。

**下一位第一件事**:确认 `eqx.Module` 子类化在这里不会改 `__init__` 的参数序
(equinox 的冻结 dataclass 语义),那是这个建议唯一没被实测的一环。

## 五、留给下一位的一句

**验收数字(§二)在删除之后就取不到了**,而它是这一行最强的证据。若切换分几次
做,**先把 §二 那张表当成回归钉住**(近端对闭式/独立 oracle),再动实现。

## 六、切换(已做)—— 以及我把自己写在 §四 里的约束当场违反了

**建议形状按 §四 落地**,那条唯一没实测的一环先补测了:`eqx.Module` 子类化
保住**九个字段、14 个叶子**,`log_prob` 继承后**逐位相同**,拒绝正确翻译成
`StateValidationError`。`npe.py` **443 → 275 行**。

### 6.1 一个我有了事实却没有用上的错

§四 里我自己写着「**config 层读它的 `create` 签名**与 `sample` 的位置参数契约」
——那是我用来主张「子类化优于持有远端实例」的理由。

**然后我把 `create` 写成了 `(*args, **kwargs)`**,理由是「重述一遍参数表就是又一个
会漂移的地方」。那个理由本身成立,**但它撞上了同一页上三行之前的那个事实**:
config 的 `npe:` 语法是**从这个签名派生的**——哪些键存在、哪些可选、各自默认什么、
`embed` 属于 `create` 而不属于 `train`。

> **`*args` 不是一次中性的转发,它是「一个什么都没有的语法」。**
> 实测:一次打掉 `TestTheGrammarMatchesTheSignatures` **七条**。

**把约束写进记录,不等于在下一步用上它。** 这一条和 D58 是同一族——那次是没去读
该读的,这次是读了、写下来了、三行之后自己违反了。

### 6.2 解法不是二选一

重述签名(让派生能工作)**并且**钉住它等于远端的(把重述带回来的漂移风险捂住):
`test_npe.py::TestTheRestatedSignaturesStillMatchTheFarSide` 逐参数、逐默认值、
逐 kind 比对两侧,外加一条反空洞——**一个不小心拿远端和远端自己比的签名检查会永远
通过**,所以另外钉了「两个签名彼此可区分」。

远端加一个参数时,若无此守卫,**近端会保持自洽地落后**:config 语法是照近端签名
校验的,于是它会安静地不再提供那个新旋钮,而两边的测试都是绿的。

### 6.3 独立预言机降级 —— 见 **D59**

`npe` 的闭式预言机取自 `wiener_solve`/`gcr_sample`,**而那两个已是远端门面**。
这条预言机从「两包互证」降为「同包两模块互证」。已量规模:8 个模块已委托、
11 个测试文件的被测与预言机双方都路由到 bayesmith。**不是缺陷,是会静默退化的
性质**,判据写在 D59 里:**问的是「两条不同的推导,还是同一条走两遍」。**

### 6.4 接缝变异:8 条,8 杀 —— 而旋钮那三条是**既有守卫**杀的

| 变异 | 杀它的 |
|---|---|
| M1 `create` 重抛远端类而不翻译 | `test_npe::test_mismatched_bank_halves_are_refused` |
| M2 `create` 吞掉拒绝 | 同上 |
| M3 `train_posterior` 重抛远端类 | `test_inference_construction_guards::test_a_non_positive_step_count_is_refused` |
| M4 `n_components` 写死 | `..._non_positive_component_count_is_refused` |
| M5 漏传 `validation_fraction` | `..._validation_fraction_outside_the_unit_interval_is_refused` |
| **M6 `width` 写死默认值** | **`config/test_config_exits_npe::TestTheEstimator::test_the_declared_knobs_land_on...`** |
| **M7 `depth` 写死默认值** | 同上 + `TestTheTrainedPosterior::test_the_draws_follow_th...` |
| **M8 `embed` 写死 `jnp.ravel`** | **`TestTheEstimator::test_the_embed_reaches_create_a...`** |

**M6–M8 是专门挑「没有拒绝守着的旋钮」问的**,因为 M4 那种「写死一个值」其实是被
**拒绝测试**杀的——拒绝不再触发,所以红;那不等于「用错的值被发现了」。
`width`/`depth`/`embed` 没有任何拒绝,所以它们才是真正的旋钮问题。

**答案是:早就有人看着,而且是 config 层自己的测试。**
这与 `calibrate` 那一批**正好相反**——在那边,「漏传 `beta1`」**没有任何测试**杀得了,
我得手写一个 spy。两相对照说明的不是「npe 更好」,而是:
**`calibrate` 的 config 面比 `npe` 的薄**,而薄在哪里只有这样问一次才知道。
