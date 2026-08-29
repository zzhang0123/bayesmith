# 执行页 Wave D 开波 —— 八个模块,而「切」的其实很少

> 计划:§五 Wave D;先决 **D12 + G3 + G6**(三条**均已满足**,最后一条由本会话
> 结清 D44 补上)。新增裁决 **D61**(切换顺序)。
> **日期**:2026-08-29 · **本页状态:开波已做,验收已量,未动代码。**

## 〇、先决三条,逐条有据

* **D12** 已拍(owner 2026-08-27)。
* **G3** 已落地 2026-08-27,`marginal/chain.py`,六个名字全数。
* **G6** 7 个里的 6 个已落地;**第 7 个 `compress_reduced_basis` 经 D44 结清为
  留守**,所以那个缺口**不需要填**——它根本不是远端的项目。

## 一、名字重合严重高估兼容性,签名才算数

先按**名字**量,再按**签名**量,两个答案差很多:

| near | 名字重合 | **同签名** |
|---|---|---|
| `sqrtinfo` | 3/3 | **4/4** |
| `chain` | 6/7 | 10/11 |
| `diagnostics` | 6/8 | **6/9** |
| `factorize` | 1/1 | **1/2** |
| `compress` | 1/3 | — |
| `memory`/`compressed`/`archive` | **0** | — |

**`Factorization` 是最能说明问题的**:近端字段
`['space','linked','hyper','represents']`,远端 `['epoch_plate','survivors','per_epoch']`
——**一个都不重合**。那正是 B11 写的「factorization **由图推导**」。
**按名字看是 1/1 完美匹配,按签名看根本不是同一件东西。**

> 一条更正:本页初稿说远端 `HyperTransition` 的参数重排「会静默交换两个实参」。
> **查过了,不会**:所有构造点都用关键字,`src/` 里除定义外不构造它,且两个参数
> 类型差得够远。**是潜在不匹配,不是活的静默风险。**

## 二、判定(逐名量,不逐模块猜)

判据是 D42/D31 那把尺,`reduced_basis` 刚用过。

| 模块 | 判定 | 决定它的那一条事实 |
|---|---|---|
| `sqrtinfo` 453 | **委托** | 九个公开名**零词汇接触**,唯一一等公民 import 是 `StateValidationError` |
| `chain` 1282 | **半切** | 六个 G3 名字数组级;第七个 `ChainMemory` 是 D12 容器 |
| `diagnostics` 839 | **半切** | `shrinkage_power`/`shrinkage_report` 收 `Mapping[int, Any]`,无容器;`systematic_floor` 收 `BayesMemory` |
| `compress` 1029 | **半切,偏留守** | `compress_reduced_basis` 留守(D44);`compress` 路由**远端无对应物** |
| `compressed` 594 | **留守** | 四个公开名全是 D12 容器;11 条拒绝里 8 条在 `__check_init__` |
| `factorize` 205 | **留守** | 字段 #1 是 `space: ParameterSpace`;10 条拒绝全是 `ParameterSpaceError` 且在 `__check_init__` |
| `memory` 1045 | **留守** | 远端**没有累加器**可守;`fold_epochs` 是 `lax.scan` 折一批,近端一次折一条 |
| `archive` 395 | **留守** | 在盘格式 + 两个 D12 容器;**卡在 D39** |

**私名普查(按 import)**:整族**只有一处**生产借用——
`chain.py:838,1149` 借 `memory._stored_names`,**随 `chain` 切换到期**。

## 三、D61:顺序由**证据**决定,`sqrtinfo` 最后

依赖图说先切它(它是叶子);**证据纪律说最后切它**。理由见 D61:
这一波每条验收都是拿 `SqrtInfo` 的字段表达的,**近端 `SqrtInfo` 一旦变成远端那个,
所有比较的两侧都是远端,而且永远绿**。

**次序**:0 裁决(**D39**、D46 的复数拒绝、`smooth` 容差)→ 1 `diagnostics` 两个
干净名字 → 2 `chain` 六个 G3 名字 → 3 `compressed` 然后 `compress` →
4 `diagnostics` 消费面 → 5 `memory` → 6 `archive`(卡 D39)→ **7 `sqrtinfo`**。

## 四、验收:已量到的逐位相同(**删除之前**)

`chain_log_likelihood`、`chain_marginal`(五个字段全部)、`ornstein_uhlenbeck`
(四个)、`shrinkage_power`/`shrinkage_report`、`held_out_z`、
近端 `coherent_mode` 对远端 `template_modes`(**五个键含嵌套模板全部**——
**这条以测量结清了 D45 的「指认」**)、`compress_linear.info` 对 `compress_epoch`。

**两条不是逐位**,都已定位:
* `compress_linear.info` 对远端 `compress`(plain T2):Δlog_prob = **2.27e-13**,
  **成因是存储不是算术**——近端回 5×5 约化 `R`,远端回 40×5 白化设计阵。
* `smooth` —— **容差已定,见 D63**:实测 `rel ≈ 1.8 × eps × n^1.25`
  (n=8 时 5.6e-15,n=512 时 8.4e-13,外推 12,007 leaves 为 4.7e-11)。
  成因是 RTS 顺序递推的**累积舍入**,不是另一个算法——指数 1.25 与此一致。
  **验收写成 `rtol = 10 * eps * n**1.25` 并连 n 一起扫**,不写常数:
  只在一个 n 上比,看不出它在增长,也就看不出常数容差为什么是错的形状。

**量不到的**(远端无对应物):`compress` 路由、`memory`、`archive`、`compressed`、
`factorize`。它们的验收是 **D12 读档回归**(`test_d12_read_back.py` 对着已提交的
fixture),**不是近远对比**。

## 五、拦路的

1. **`archive` 卡 D39**(未裁决,且是在盘格式决定,不能顺手做掉)。
2. ~~**D46 的复数拒绝**会随 `sqrtinfo` 委托到达近端~~ ——**这句是错的,已结清为 D62。**
   `SqrtInfo` 是 **D12 的留守容器**(G6 自己写着远端数组级函数**返回**这个类),
   所以近端 `SqrtInfo` **永远不会变成远端那个**,拒绝**不会到达**。
   近端确实在静默给错答案(实测 `fisher()` 返回 2.0 而真值恰为 0),
   **已于 2026-08-29 在近端 `__check_init__` 加上拒绝**(e-RHINO `1a5a535`),
   含兄弟断言。**次序第 7 步不再有这一项。**
3. **远端 `marginal/diagnostics.py` 779/800 行**——往里加东西之前先拆。
4. **§4.3 与 Wave D 的表面矛盾已解**,靠的是**已执行的先例**:`reduced_basis`
   同时在 §4.3 名单里和 §五 Wave C 行里,做成了半切(D60)。所以 §4.3
   **不禁止**近端委托到重写出来的远端面,它排除的是**整体**切换。已就地收窄。

## 六、留给下一位的一句

**这一波「切」的其实很少**:八个模块里三个整体留守、三个半切、一个委托、
一个卡在裁决上。**不要按模块数估工作量**——按**逐名判定**估。

## 七、第 0 步与第 1 步已落地(2026-08-29)

### 7.1 第 0 步:三条裁决

* **D39** 前两步落地(字节检查 + 可操作的版本拒绝),第 3 步经对抗复核**从一件事
  改成两件事**:摘要(关同形状误配 **与** `save_memory` 的重归档崩溃窗口)
  **加上**二进制内的逐 term 身份(关排列,摘要对它无效)。
* **D62** 复数拒绝加在**近端**——**更正了本页原来的排期**。本页曾写「随
  `sqrtinfo` 委托到达」,而 `SqrtInfo` 是 **D12 留守容器**,永远不会变成远端那个。
  近端实测 `fisher()` 返回 2.0 而真值恰为 0。
* **D63** `smooth` 容差 `rel ≈ 1.8 × eps × n^1.25`,**扫 n 不写常数**。

### 7.2 第 1 步:两个 shrinkage 名字(e-RHINO `b2ae646`)

**验收**:`shrinkage_power` 三个 bank **逐位**(`|Δ|=0.0`);`shrinkage_report`
的 `power`/`n_values`/`detects_coherent_bias` 三键相同。

**开波查出两件读不出来的:**
1. **四条拒绝全在私有 `_shrinkage_table`**,两个公开名自己零条——`calibrate`
   教的那个形状(`ast.Raise` 扫不到经帮手抛的)。远端四条俱在。
2. **异常类看起来会断,实际不会,而只有跑才知道**:近 `ValueError` 对远
   `StructureError`,但 `ValueError` 在后者的 MRO 里,**pin 原样成立**。
   (`calibrate` 那两次错在并列读文案;这次把 pin 真跑到远端异常上。)

**`caveat` 留守**,两条理由:远端那句结尾指向 **`template_modes()`**,
而**近端没有这个名字**(与 `calibrate` 的补救句同一个坑);且近端这句携带
rheplicant 自己的测量(**52.6 sigma** 上十二位),那是证据不是措辞。

**接缝变异 4 条 4 杀,而 M2(替换悄悄失效)只有新写的守卫杀得了**——
不带 `-x` 复核过:三条红全在 `TestTheCaveatIsThisPackagesOwn` 里,文件内其余
一条不响。**这是本会话第三次,「如果这件事悄悄停了,谁会红?」这一问给出了
唯一能抓住真失败的守卫**(前两次:`calibrate` 的 beta 接线、`npe` 的签名漂移)。

**一件要说清楚的**:这次委托**让文件从 839 涨到 868 行**——解释委托的 docstring
比它替换掉的函数体长。换来的是算术只有一份。**「委托 = 变少」在这一批不成立,
不该那样宣称。**
