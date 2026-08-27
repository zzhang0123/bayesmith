# 测量页 — 切 `numpyro_bridge` 之前必须先量的两件事(D26、D27)

> 计划:§五 Wave A / 铁律 1、5。前一批次:`2026-08-27-wave-A-priors.md`。
> **日期**:2026-08-27 · **本页不是一个切换批次**,是一份**先决测量**,形状同
> `2026-08-27-d17-protocol.md` 与 `2026-08-27-d16-five-axes.md`:两条裁决点在
> 上一批的记录页里被标成「先量再决定」,这里把它们量掉并登记,**实现留给下一批**。

## 〇、为什么先量

`priors` 批的 §十.2 写着 `numpyro_bridge` 的第一个问题是**站点名**,并注明
「大概率是一条新登记项」。开工前先读了契约页 `docs/migration/numpyro_bridge.md`
§6,它自己就说明了这条边界:

> `to_numpyro_model` 的构造**没有**与 `to_numpyro` 逐站点对照:两者是在**不同的
> 词汇**上建模的(`ParameterSpace` 上的 pipeline 叶子 vs 图节点)。

所以这不是一个实现细节,是契约页记录在案的一条未决边界。两条:**站点名**,
以及**被抽样的 `noise_std`**。

## 一、D26 — 站点名:量出来的代价比预期**小得多**,而我第一次读错了

`to_numpyro_model` 的站点是 `"prediction"`(deterministic)与 `obs_name`
(默认 `"obs"`);`to_graph` 的内部节点是 `__mu__` 与 `__data__`。直接委托会改名。

### 第一次读:以为是「五条守卫静默失效」

`grep` 出四处 config 断言,形如 `assert "prediction" not in <samples>`,而
`test_config_exits_npe.py` 那一处的注释**逐字**写着:

> 按**名字**断言它的缺席才是那个有分辨力的一半;断言 d 和 a 在里面,在整条 TOD
> 也在里面的时候一样通过。

读起来就是「改名会让四条守卫变成恒真」。

### 实测:改名之后**整个 `tests/inference` + `tests/config` 只红一条**

把两个站点改成 `__mu__` / `__data__` 再跑两目录:

```
FAILED tests/inference/test_numpyro_bridge.py::TestModel::test_sites_are_named_after_their_latents
```

**只有这一条。** 原因是 config 层**根本读不到那个站点**:`nuts.py` 把
`get_samples()` 过滤成 `{name: drawn[name] for name in space.names}`,所以
deterministic 站点**无论叫什么**都不在 product 里。

### 那四行的真实分辨力,以及那句注释是错的

四处**每一处旁边都已经有一条集合断言**(`set(samples) == {"g"}` 之类),而承重的
是**那一条**:它对「过滤被去掉」是敏感的,并且**改名之后仍然敏感**(会看到
`__mu__`)。按名字那一行只对「过滤被去掉**且**名字没变」敏感,**改名之后它就死了**。

所以那句「按名字断言缺席才是有分辨力的一半」**是反的**:集合断言才是,名字那行是
更弱的冗余。**已在四处逐一改正措辞**(行本身保留——它对今天的过滤仍然有效)。

> 记在这里的理由:**我按那句注释先得出了一个错误的结论**,而它是仓库里写下来的
> 一句话。这正是本程序 §七 那条「一个事实两份拼写,其中一份悄悄过期」的形状,
> 只不过这次过期的是一句**关于守卫强度的断言**,而它把下一位(我)带偏了一次。

### 裁决

**站点名仍然是保持面**,尽管代价小:它出现在 `to_numpyro_model` 的 docstring
(「noiseless prediction 记在 deterministic 站点 `"prediction"`」)、
`examples/tutorial_nuts.py` 的注释里,而一个用户读 `mcmc.get_samples()["prediction"]`
是被文档邀请的用法。铁律 1 说的是公开名字保持;它不在
`tests/test_published_contracts.py` 的 27 条里,但**便宜地保住**比**便宜地改掉**
更符合铁律 1。

**【本次委托下自定,2026-08-27:取「保持」;`to_graph` 增加可选的节点命名参数,
只由 `to_numpyro_model` 使用。】** 见计划 §二 D26。

## 二、D27 — 被抽样的 `noise_std`:图**能**表达它,而且不需要新的远端表面

`to_numpyro_model(noise_std=<numpyro 分布>)` 会产出站点 `"noise_std"`,而
`to_graph` 只收一个具体的 `NoiseModel`。乍看这是一个**发布门**问题(铁律 5:
依赖未发布的远端表面)。

**实测不是。** `bayesmith.observe(name, dist_fn, *parents, ...)` 收**多个** parent,
所以 scale 可以是另一个节点。端到端探针(`probe_sigma.py`)一次跑通:

```
built: ['gain', 'noise_std', '__mu__', '__data__'] | latents: ['gain', 'noise_std']
log_joint: -1.927643458236838
numpyro sites: ['__data__', '__mu__', 'gain', 'noise_std']
obs site value shape: (8,)
```

声明、求值、`log_joint`、`to_numpyro` 四步全通。**所需能力已在 0.4.0 里**,
`git log v0.4.0..HEAD -- src/` 为空,**铁律 5 满足**,工作全在适配器一侧。

### 用量:两处,且 config 够不到

`noise_std=<分布>` 在全仓只有两个消费者:
`tests/inference/test_numpyro_bridge.py::test_sampled_noise_std` 与
`tests/inference/test_jeffreys_prior.py` 里用 `allow_sampled_noise_std=True` 的那条。
**config 层没有任何路径**能产出一个分布形的 `noise_std`(实测:
`src/rheplicant/config/sections/` 里没有 `Distribution`)。

但它是**签名的一部分**,`allow_sampled_noise_std` 也是,两者都被 docstring 与
2C/2D 两份计划记着——铁律 1 要签名保持。

### 裁决

**【本次委托下自定,2026-08-27:取「适配器把它声明成一个名为 `"noise_std"` 的
图 latent」。】** 三条理由:
1. 它**本来就是一个声明**(一个 σ 的先验),不是数值;把它写成图的一个节点是把
   同一句话换成图的词汇,不是新增语义。
2. 另一条路是让门面为这一个分支保留手写的似然,那就**留下第二份实现**——正是
   本计划要消灭的东西,而且留在最容易忘记的那个分支上。
3. 它可被守卫钉住:`"noise_std"` 会出现在 `graph.latents` 里,而空间没有声明它,
   所以**空间里一个同名 latent 就是一次碰撞**。今天那次碰撞是什么行为(numpyro
   重复站点)**没有量过**,所以下一批的第一件事是量它,并按结果补一条拒绝。

**这一条给下一批留了一个必须先量的点**,写在这里而不是留在记忆里。

## 三、下一批(实现)的开工清单

1. 量「空间里有一个名叫 `noise_std` 的 latent + 分布形 `noise_std`」今天是什么行为;
   按结果补拒绝,并进拒绝普查(计数由守卫报数,附录 B 由普查重生成)。
2. `to_graph` 加可选节点名(D26)与分布形 scale(D27),各自带守卫;
   `_refuse_internal_names` 的对象随之变成**本次调用选定的**名字。
3. `to_numpyro_model` 委托 `bayesmith.to_numpyro`;
   `init_to_declared` **留守**(契约页 §5(a):「带过去的是教训不是代码」——
   图的 latent 有先验没有 `init`,没有东西可移);
   `predict_from_samples` 按 G7 走。
4. `tests/crosscheck/test_bridge.py` 同批退役,`SWITCHED` 加 `numpyro_bridge.py`,
   独立 oracle 逐条指认或改籍。
5. 分诊:`tests/inference/test_numpyro_bridge.py` **16 条**,加 config 侧消费者。

## 四、本页动了什么

**没有源码改动。** 只有四处 config 测试的**注释**改正(§一),它们的断言一行未动。
两条裁决进登记簿(D26、D27),计划 §二 标题随之到 **D7–D27**。
探针留在 scratchpad,`probe_sigma.py` 的输出逐字记在 §二。
