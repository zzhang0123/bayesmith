# 执行页 Wave A · 开放项 S6 结清 — `_widened` 是承重的,缺的是 fixture

> 计划:§五 Wave A / 铁律 1、4。前一批次:`2026-08-27-wave-A-sensitivity.md`(§五 S6)。
> **日期**:2026-08-27 · **本页状态:完成。S6 的两个方向已由测量分开;
> 量出来的机制比开放项本身写的更窄,也更值得钉。**

## 〇、开放项是什么

`sensitivity` 那一批把 `_widened`(把 latent 值升到 float64)换成恒等函数,
**64 条全绿**;同一条变异在模块 1 `identifiability` 里**45 条全红**。两个处置方向
相反:

1. 它在这里确实不需要 → **死代码,该删**;
2. 它需要,只是本文件没有 fixture 制造那个条件 → **该补 fixture**。

上一批明写「不猜、不删、不加断言」,把它留成开放项。**本批次量了。**

## 一、第一次读数:先看住这个文件为什么绿

`tests/inference/test_prior_sensitivity.py` 有一条 **module 级 autouse 的 `_float64`
fixture**——整个文件在 x64 打开的情况下构造 twin、state 和每一个 `Latent`。
`identifiability` 那个文件**没有**这条 fixture。

所以 64 条绿不是「`_widened` 没用」,而是**这个文件从来没有制造过它防的那个条件**。
这正是本仓最常复发的那种缺陷:一条守卫停止了失败的能力,而没人看得见。

## 二、第二次读数:把变量减到一个,结论**翻转**

第一次探针(`probe_s6`)在**普通 float32 会话**里声明整个模型——init、state、
observed 三样都是 float32——照原样跑通过,去掉 `_widened` 则被
`ParameterSpaceError` 拒绝(bayesmith 的 `refuse_single_precision`,报
「the prediction at '__data__' came back float32」)。

**但那次探针一次动了三个变量,所以它说不出是哪一个。** 第二次(`probe_s6b`)让
twin、state、observed 全部保持 float64,**只有 `Latent` 的声明**在 float32 会话里发生:

```
--- as shipped ---            declared x64: shift=+7.900074460228306e-01
                              declared f32: shift=+7.900074460228306e-01
--- _widened -> identity ---  declared x64: shift=+7.900074460228306e-01
                              declared f32: shift=+7.900074460228306e-01
```

**变异不咬了。** 「init 是 float32」不是那个条件。

## 三、第三次读数:2×2 表,以及第三行

`probe_s6c`,`{init, 模型自己的常数} × {float32, float64}`,每格跑两遍:

| 模型常数 | init | 照原样 | `_widened` → 恒等 |
|---|---|---|---|
| float64 | float64 | OK(+7.900074460e-01) | OK(+7.900074460e-01) |
| float64 | float32 | OK(+7.900074460e-01) | OK(+7.900074460e-01) |
| **float32** | **float64** | OK(+7.900073735e-01) | **ParameterSpaceError** |
| float32 | float32 | OK(+7.900073735e-01) | **ParameterSpaceError** |

**条件是模型自己的数组是 float32**,不是 init 的 dtype。而**第三行**是那个说不通的:
init 已经是 float64 了,变异**照样**拒绝。

## 四、机制:`astype` 做了两件事,只有一件是「加宽」

`probe_s6d` 直接打印:

```
declared x64: dtype=float64 weak=True
  after _widened: dtype=float64 weak=False
  forward(raw)      -> float32
  forward(_widened) -> float64
```

`jnp.array(1.0)` 是**弱类型**的 float64。**弱 float64 遇到强 float32 会采纳对方的
dtype,而不是提升它。** `astype(float64)` 除了加宽,还**剥掉弱类型**——而那一半才是
让 init 在与模型的 float32 常数相遇时赢下提升、让预测以 double 抵达
`refuse_single_precision` 的东西。

这一层是上一批的开放项没写到的:S6 只问了「它在这里承重吗」,而答案是「承重,
且承的重和你以为的那根不是同一根」。**追一个幸存变异要追到底**(交接页 §七.4)
在这里第二次兑现:补一条断言就收工的话,钉住的会是 init 的 dtype,而那一条
**第二行**就证明了它是空的。

## 五、处置:补 fixture,三条

`tests/inference/test_prior_sensitivity.py::TestAModelDeclaredInSinglePrecision`,
用的是本文件已有的仿射模型(8 通道、单位噪声,先验对似然可见),
只是**在 float32 会话里声明并调用**——也就是 config 驱动的运行的样子。

| 测试 | 钉什么 |
|---|---|
| `test_the_fixture_really_is_declared_in_single_precision` | **兄弟断言**:init/observed/`state.coords.freq` 三样都是 float32。没有它,后两条会在某人给这个 fixture 加上 x64 的那天静默变空 |
| `test_the_verdict_comes_back_in_double` | 报告的 `mode`/`precision`/`sigma_post` 是 float64,**且**闭式与独立 refit 在 `VERIFY_RTOL` 内一致、`verified` 全真。钉的是性质不是 dtype:仿射后验精确二次,refit 是 Newton 精确的 |
| `test_a_weak_float64_init_does_not_carry_a_float32_model` | §四 那一半。init 声明在 x64 下(**已经是 float64**),模型是 float32;一个「只在 dtype 是 float32 时才 cast」的 `_widened` 会通过前两条而死在这一条 |

第三条自带一条断言 `init.dtype == float64 and init.weak_type`——同样是兄弟断言:
若 `initial_values` 哪天返回强类型,这条测试的主题就没了,而它会说出来。

**两处 docstring 同批改写**:`sensitivity._widened` 与 `identifiability._widened`。
两份是**故意重复**的(上一批写明了理由:跨模块进口第二个私名会让铁律 1 的普查
长大),所以两份都要说出弱类型这一半,否则就是本程序反复付学费的那个形状——
一个事实两份拼写,其中一份悄悄过期。identifiability 那份额外写明:它的文件全部
在 float32 里声明,所以**只行使了加宽那一半**,弱类型那一半由 sensitivity 的
测试对着它的孪生体钉住。

## 六、这算不算「改判据」?

**不算,所以没上登记簿。** 判据一行未动:`_widened` 的实现、`refuse_single_precision`
的位置、两侧的拒绝措辞全部照旧。本批次做的是让一条已经存在的行为**可以失败**。
写在这里是为了让下一位能不同意——如果读成改判据,那它该走登记簿。

S6 本身也不是一个裁决点:两个方向不是取舍,是一次测量的两个可能结果,而测量做完
就只剩一个。这也是为什么它当初该被留成开放项而不是被猜掉。

## 七、变异集:4 条,全部击杀,而且**各自被该负责的那一条钉住**

`scratchpad/mutate_s6.py`,五行协议含第 (0) 条(两棵树在跑之前都已提交且干净;
恢复用 `git checkout --`,两侧各扫一次包内 `__pycache__`,日志逐行 flush)。
靶子是整个 `test_prior_sensitivity.py`,这样「我的红」与「旧 64 条有没有被牵连」
在同一张表上可读。

| # | 变异 | 仓 | 指名红 | 判决 |
|---|---|---|---|---|
| M1 | `_widened` → 恒等(**就是 S6 那条**) | e-RHINO | `test_the_verdict_comes_back_in_double`、`test_a_weak_float64_init_does_not_carry_a_float32_model` | KILLED(2 红) |
| M2 | 只 cast 真 float32,放过弱 float64 | e-RHINO | `test_a_weak_float64_init_does_not_carry_a_float32_model` | KILLED(**1** 红) |
| M3 | 给新 fixture 加上 x64(=拿掉那个条件) | e-RHINO | `test_the_fixture_really_is_declared_in_single_precision` | KILLED(1 红) |
| M4 | M1 **再加**远端 `refuse_single_precision` 一并去掉 | 两仓 | `test_the_verdict_comes_back_in_double` | KILLED(1 红) |

基线前后各一次绿(exit 0,0 红)。

**M1 是 S6 的正式答复**:同一条变异,上一批 64 条无一动静,本批 2 条红,而**旧 64 条
仍然一条不红**。所以「64 条全绿」从来不是关于 `_widened` 的读数,是关于那个文件的。

**M2 只红一条**,而且是弱类型那一条——§四 的第二层不是文风,是一个能被单独杀死的
判据。补一条断言就收工的方案会通过 M2。

**M4 是「红的是不是我的」那一问的答案**,而它给的答案比预期精确:去掉远端拒绝之后,
`test_a_weak_float64_init_does_not_carry_a_float32_model` **转绿**(mode 仍是 float64
——先验的二次项在 float64 里累加),红的只剩
`test_the_verdict_comes_back_in_double`,而它红在**闭式与 refit 不一致**那条断言上,
不是 dtype 断言上。也就是说:当远端守卫不在时,承重的是**数值一致**那一条,dtype
断言够不到。这条差别只有 M4 看得见。

**没有一条纯跨仓变异(改 bayesmith、看 e-RHINO 红),这是有理由的**:本批次没有引入
对任何新远端表面的依赖,它行使的是一条**已有**的远端拒绝,而在门面正常工作时那条
拒绝**根本不触发**——同上一批 S3 的形状。M4 是能拿到的最接近的读数,做法是把两侧
一起变异。

## 八、铁律 4 四件套

| | 项 | 结果 |
|---|---|---|
| (i) | 该批测试全绿 | e-RHINO **10066 passed / 522 skipped** exit 0(347.3 s,`-n 4 --ignore=tests/gui/e2e`;junit 报 tests=10588、failures=0、errors=0)加 **21 passed** exit 0(`tests/gui/e2e -n 2`,66.0 s);bayesmith **1280 passed / 0 skipped** exit 0(208.0 s,junit tests=1280、failures=0、errors=0)——本批只动 e-RHINO,但 `tests/crosscheck/` 从 editable 装的兄弟仓读,所以两侧都跑 |
| (ii) | 接缝变异红 | **4 条全杀**,§七 |
| (iii) | 旧实现删除、计数守卫刷新 | **本批无删除**(`_widened` 留守,它是承重的);README 计数 10605 → **10608**(+3,由守卫报数);coverage floor 未动 |
| (iv) | 文档实测数字重测 | README 计数;两处 `_widened` docstring;交接页 §三.7 的一条**过期项**顺手核实(见 §九) |

## 九、顺手核实掉的一条过期交接项

交接页 §三 第 7 条写着「五行协议现在有第 (0) 条……**e-RHINO 的 CLAUDE.md/AGENTS.md
还没写**」。**实测为假**:`2fe13a0 docs: commit the batch before you mutate it` 已经
成对写进两份,`cmp` 逐字节一致,两份各含一次该段。交接页那一条在本页写完时一并
删掉。

同一形状第三次出现(D13 的版本号、D9 的整条建议、D21 的契约页),所以照
「委托不是空白支票」那条处置:**按事实**,并在批次页里点名。

## 十、留给下一位

1. **Wave A 还剩三个模块**:`priors`、`numpyro_bridge`、`uncertainty`。
   `priors` 与 `numpyro_bridge` **实际上是一批**:`JeffreysPrior.log_density` 在
   rheplicant 里**只有一个消费者**,就是 `to_numpyro_model` 的 factor site;而远端
   G13 的读取点也在 `to_numpyro` 里。分开切会让其中一个在半途上没有消费者。
2. **那一批欠一条真链验收**(交接页 §三.4):G13 只用 `log_density` 验了势能,
   没跑过 `nuts()`。
3. **D23** 仍是已登记、未裁决、无守卫。
4. 九份草稿仍在 `860703d` 的历史里,是否重写历史需强推,是 owner 的决定。
   > **【已处置 2026-08-28】** owner 授权重写历史;九份草稿已从 e-RHINO 的历史中移除(`860703d~1..HEAD` 22 个提交重写为 21,`f8a73eb` 因变空被剪掉),**重写后 HEAD 的 tree 与重写前逐字节相同**,九份未跟踪的工作副本原样保留。本行提到的两个 SHA 自此不再存在。

