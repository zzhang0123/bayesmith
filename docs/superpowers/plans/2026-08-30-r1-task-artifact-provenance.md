# R1 Task、artifact 与 provenance 执行计划

> **文档状态：`record`** · 已落地批次/审计/测量的历史记录，写作当天为真，非当前权威。索引见 docs/README.md。
>
> R1 已 closed，证据见 `../specs/2026-08-30-r1-close-out.md`；本文保留为协议冻结的完整出处。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变现有推断数值的前提下，为五类 Bayesian Task 建立稳定、可序列化、可失效、可评价的共同协议，并把现有 posterior 与 point-estimate 路径适配为第一批真实 Result。

**Architecture:** 新建无 JAX/NumPyro/Graph 模块级依赖的 `bayesmith.artifacts` 叶层，持有纯数据协议、canonical codec、fingerprint、Refusal 和 gate 聚合；`bayesmith.dispatch.task` 是唯一把 runtime `Graph`/现有 `InferencePlan` 接到该协议的桥。现有 `compile()`、`InferencePlan.sample()`、`InferencePlan.estimate()`、`Posterior` 和 `Estimate` 保持兼容，R1 不重写其数值内核。

**Tech Stack:** Python 3.11 frozen/slots dataclasses、`enum.StrEnum`、NumPy、标准库 `json`/`hashlib`/`base64`/`datetime`/`uuid`，现有 JAX/Equinox/NumPyro execution stack，pytest、ruff。

**Spec:** [Bayesmith 顶层设计](../specs/2026-08-30-bayesmith-top-level-design.md) §2、§4、§8 R1；[R0 close-out](../specs/2026-08-30-r0-close-out.md)。

---

## 0. 冻结裁决与执行边界

这些是 R1 的实现输入，不在执行过程中重新发明：

1. **五进五出。** `PosteriorTask → PosteriorResult`、`EvidenceTask → EvidenceResult`、`PredictiveTask → PredictiveResult`、`PointEstimateTask → PointEstimateResult`、`SimulationTask → SimulationResult`。编译或执行不成功时返回 `Refusal`，Report 不能代替主 Result。
2. **R1 只接真实已有执行能力。** `PosteriorTask` 接当前 analytic/exact-draw 与 NumPyro MCMC 路径；`PointEstimateTask` 接当前 posterior-mean/GLS 和 MAP/optimizer 路径。Evidence、predictive 和 simulation 的 Task/Result schema 在 R1 完整存在，但 runtime bridge 返回 code 为 `capability_unavailable_r1` 的 typed `Refusal`。R2/R4 接执行，不在 R1 伪造结果。
3. **`grounds` 是冻结字段名。** `Refusal` 不得出现机器字段 `evidence`，也不得通过解析 exception message 形成结构化判断。
4. **artifact 是数据，不是 runtime object dump。** 不 pickle `Graph`、callable、JAX executable、backend state 或 estimator object。它们由 `ModelRef`、fingerprint、稳定引用和 runtime attachment 表达。
5. **单向依赖。** `bayesmith.artifacts` 在模块作用域不得导入 `bayesmith.graph`、`bayesmith.dispatch`、JAX、Equinox 或 NumPyro；`dispatch.task` 可以导入 artifacts 和现有 runtime 层。
6. **兼容优先。** 现有入口与 NamedTuple 返回保持不变；新入口是 `compile_task`/`execute_task`。适配器包住旧结果，不改变随机 key、预算、容忍度、fallback 或数值数组。
7. **状态和统计结论是两根轴。** `BLOCKED`、`INVALIDATED`、`ERROR` 没有 verdict；只有 `EVALUATED` 有 `PASS`、`FAIL` 或 `ABSTAIN`。
8. **canonical form 是安全白名单。** 不允许任意 import/type construction；不允许 object array、callable 或依赖 `repr()`/内存地址的 identity。
9. **每个 task 顺序执行。** 本计划共享 `artifacts/__init__.py`、`dispatch/task.py` 和 public API tests，不适合在同一 checkout 并行写。每一 task 只 stage 自己列出的文件；禁止 `git add -A`。
10. **不清扫格式漂移。** R1 lint gate 是 `.venv/bin/ruff check src/ tests/`，不是 `ruff format --check`。

### 0.1 包结构

```text
src/bayesmith/artifacts/
├── __init__.py       # 唯一 public artifact inventory
├── _codec.py         # canonical JSON-safe encoding/decoding
├── base.py           # envelope、RunRecord、NamedArray、通用小值对象
├── identity.py       # ModelRef、fingerprint、lineage/invalidation
├── tasks.py          # 五类 Task tagged union
├── results.py        # 五类 Result 与 posterior representations
├── refusal.py        # Finding、Remedy、FallbackOption、Refusal
├── reports.py        # AnalysisReport、InferencePlanRecord、EvaluationReport
└── gates.py          # operational status、verdict、truth table

src/bayesmith/dispatch/task.py  # Graph/runtime ↔ artifact bridge
```

内部依赖梯级固定为 `_codec ← identity ← base ← tasks ← results`，`refusal` 依赖 `base/tasks`，`reports` 依赖 `base/identity/refusal`，`gates` 依赖 `base/refusal/reports`。`NamedArray` 放在 `base`，使 `ParameterSource.FIXED` 与各 Result 复用同一数组类型而不会形成 `tasks ↔ results` 循环。

### 0.2 冻结的 artifact envelope 与运行记录

所有时间使用 UTC RFC 3339；所有 collection 使用 tuple 或排序后的 `(key, value)` tuple。
数组进入构造器时复制成 C-contiguous、read-only NumPy array，避免 artifact 建立后被调用方原地改写。

```python
@dataclass(frozen=True, slots=True)
class ProducerRef:
    package: str
    version: str

@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    revision: int
    artifact_type: ArtifactKind

class ArtifactStatus(StrEnum):
    CURRENT = "current"
    INVALIDATED = "invalidated"

@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    status: ArtifactStatus
    invalidated_at: str | None = None
    changed_inputs: tuple[FingerprintKind, ...] = ()

@dataclass(frozen=True, slots=True)
class ArtifactMeta:
    artifact_type: ArtifactKind
    schema_version: int
    artifact_id: str
    revision: int
    created_at: str
    producer: ProducerRef
    parent_refs: tuple[ArtifactRef, ...]
    fingerprints: FingerprintBundle
    lifecycle: LifecycleRecord
    warnings: tuple[RunWarning, ...]
    summary: str

@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    plan_ref: ArtifactRef
    fingerprints: FingerprintBundle
    seed: SeedRecord | None
    dtype: str
    devices: tuple[DeviceRecord, ...]
    jax_config: tuple[tuple[str, CanonicalScalar], ...]
    backend: BackendRef
    budget: ComputeBudget
    termination: TerminationRecord
    timing: TimingRecord
    approximation: ApproximationRecord
    warnings: tuple[RunWarning, ...]
```

`artifact_id`/`task_id`/`run_id` 是创建一次并在 round-trip 中保持的 UUID4 identity；它们不是 semantic hash。artifact 的唯一版本身份是 `(artifact_id, revision)`：初建 revision 为 0，`invalidate_meta(meta, *, before, after, policy)` 返回同一 artifact id、`revision + 1` 的 INVALIDATED immutable copy，并把 changed inputs 与失效时间写入 `LifecycleRecord`。`ArtifactRef` 总是带 revision 和 kind，所以旧 Plan/Result 不会被同 id 的新状态静默冒充。语义相等与 cache/invalidation 判定只看 fingerprint；旧 revision 保持 append-only，不被原地改写。

`ApproximationRecord` 必须把两个不同问题拆开：

```python
class ApproximationClass(StrEnum):
    EXACT = "exact"
    CERTIFIED_DETERMINISTIC = "certified_deterministic"
    MONTE_CARLO = "monte_carlo"
    HEURISTIC = "heuristic"

class TargetFidelity(StrEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"

@dataclass(frozen=True, slots=True)
class ApproximationRecord:
    representation_class: ApproximationClass
    target_fidelity: TargetFidelity
    details: tuple[tuple[str, CanonicalScalar], ...] = ()
```

因此 iid exact-linear draws 的 representation class 是 `MONTE_CARLO`，但 target fidelity 是 `EXACT`；NUTS 也是前者 `MONTE_CARLO`，并在无目标近似时标为 `EXACT`。amortized posterior 是 `HEURISTIC`/`APPROXIMATE`。

### 0.3 Fingerprint 粒度与失效矩阵

`Fingerprint` 为 `(kind, algorithm="sha256-v1", digest)`；digest 对 canonical payload 的 UTF-8 bytes 做 SHA-256。`FingerprintBundle` 的槽位固定为：

```python
model_source: Fingerprint
graph_structure: Fingerprint
data: Fingerprint
task: Fingerprint
compilation: Fingerprint | None
evaluation: Fingerprint | None
environment: Fingerprint | None
```

各槽位边界：

| Fingerprint | 包含 | 明确不包含 |
|---|---|---|
| model source | `ModelRef.identifier`、source digest、distribution package/version、显式 build arguments | Python 内存地址、裸 `repr(callable)` |
| graph structure | node 声明顺序、node 类型/name/parents/plate、support、`linear_in`、`depends_on_prediction`、shape/dtype metadata、joint-prior/evidence-term 类型与 `over`、plate name/size、callable module+qualname | `Const.value`、observed、observed mask 的具体值 |
| data | Const/observed/mask 的 name、dtype、shape、bytes，以及调用方显式附加的数据 payload | display options、runtime cache |
| task | Task kind、统计语义、backend policy、数值预算、solver/optimizer/NS 等改变执行或目标的选项、gate identity | progress bar、打印宽度 |
| compilation | block partition、exact elimination、residual variables、method、tol、fallback policy、compiler version | wall-clock timing |
| evaluation | report kind、threshold、grouping、重复次数、applicability policy、gate definition/version | Result 数组本身（由 parent identity 引用） |
| environment | Python/bayesmith/backend/JAX versions、x64/dtype/device platform | host path、随机临时目录 |

`ModelRef.from_callable()` 只有在 `inspect.getsource()` 得到稳定 source 时才能自动生成 source digest；否则必须要求调用方提供 digest，不能退化到 `repr()`。closure 内未显式声明的状态不承诺可发现，必须放进 `build_arguments`。

失效规则在 `InvalidationPolicy.default()` 中按 artifact 类别编码，并用表驱动测试固定：

| 改变 | Plan | Result | EvaluationReport/Gate |
|---|:---:|:---:|:---:|
| model source / graph structure | invalidate | invalidate | invalidate |
| data / task | invalidate | invalidate | invalidate |
| compilation | — | invalidate | invalidate |
| evaluation threshold/grouping | — | reusable | invalidate |
| display option | reusable | reusable | reusable |
| backend patch/environment | 已有 artifact 仍可读；新运行产生新 provenance | 已有 artifact 仍可读；新运行产生新 provenance | 按新 Result identity 重评 |

“invalidate”通过 `invalidate_meta` 产生带 `changed_inputs` 和 `invalidated_at` 的 immutable metadata copy；不原地改历史 artifact，也不删除旧 verdict。

### 0.4 五类 Task 的字段

每个 Task 都是 frozen/slots dataclass，含 `TaskMeta(task_id, schema_version, created_at, label)`；Task fingerprint 排除 `task_id`、`created_at` 和人类 label，包含下列统计字段。`backend_options` 等开放集合使用已排序的 canonical tuple，不接任意 Python object。

```python
PosteriorTask(
    meta, backend, budget, chain_method, solver_tolerance,
    solver_maxiter, require_convergence, ess_floor,
    nuts_on_collapse, backend_options, quality_gate,
)

EvidenceTask(
    meta, backend, budget, reconstruct_posterior,
    repeat_count, backend_options, quality_gate,
)

PredictiveTask(
    meta, source_posterior_ref, conditioning_data,
    prediction_design, conditioned_sites, replicated_sites,
    latent_sites, budget, backend, backend_options, quality_gate,
)

PointEstimateTask(
    meta, estimand, names, backend, budget,
    optimizer_options, quality_gate,
)

SimulationTask(
    meta, parameter_source, prediction_design,
    latent_sites, observed_sites, budget, backend, backend_options,
)
```

`Estimand = POSTERIOR_MEAN | MAP`。`ParameterSource` 是 tagged value：`PRIOR`、`FIXED` 或 `POSTERIOR_RESULT`，后两者分别携带 canonical named values 或 `ArtifactRef`。Predictive 与 Simulation 的边界由字段直接表达：Simulation 只拥有 parameter source + forward generation；Predictive 额外拥有 conditioning、source posterior、prediction design 和 observed/held-out site 语义。

字段类型在 R1 固定如下，避免 backend adapter 自行发明不兼容形状：

| 字段族 | 类型 |
|---|---|
| `backend` | 非空 `str`，`"auto"` 是通用策略值 |
| `budget` | `ComputeBudget`；不适用的计数为 `None`，计数不得为负 |
| `backend_options` / `optimizer_options` | 排序且 key 唯一的 `tuple[tuple[str, CanonicalValue], ...]` |
| `quality_gate` | versioned gate identity 的 `str | None` |
| `conditioning_data` / `prediction_design` | 对应 payload 的 `Fingerprint` |
| site/name collections | 去重的 `tuple[str, ...]`，声明顺序保留 |
| `PointEstimateTask.names` | `tuple[str, ...] | None`；`None` 表示所有 latents |
| `ParameterSource.FIXED` values | `tuple[NamedArray, ...]`，其 bytes 进入 task fingerprint |
| `ParameterSource.POSTERIOR_RESULT` | `ArtifactRef`，并由 lineage 检查其 kind 为 PosteriorResult |

### 0.5 五类 Result 的精确形状

`base.NamedArray(name, value, dims)` 是数组的唯一公共容器；`value` 为 read-only NumPy array。每个 Result 恰有 `meta` 和 `run`，Report 通过 id 引用它。

```python
PosteriorResult(
    meta, run, representation, latent_names, eliminated_latents,
    reconstruction_ref, log_density_availability,
    pointwise_log_likelihood, predictive_ready, report_refs,
)
```

`representation` 为穷尽 tagged union：

- `DrawsPosterior(draws, chain_shape, method)`；
- `WeightedDrawsPosterior(draws, log_weights, ess, khat, unreliable, method)`；
- `AnalyticPosterior(family, parameters, moments)`；
- `FittedConditionalPosterior(estimator_ref, simulation_bank_ref, training_run_id, validation_report_refs)`。

R1 的现有 GCR/NUTS/SNIS routes 分别适配前两种；协议为 analytic representation 预留真实闭式分布，但不把 draws 冒充解析参数。amortized representation 只冻结 schema，不在 R1 扩展本地 NPE 算法。

```python
EvidenceResult(
    meta, run, log_evidence, standard_error, posterior_representation,
    normalization_audit_refs, exact_components, residual_component,
    repeat_result_refs, consistency_report_ref,
)

PredictiveResult(
    meta, run, source_posterior_ref, conditioning_data,
    prediction_design, conditioned_sites, latent_draws,
    replicated_draws, pointwise_log_density, observation_unit,
    grouping, report_refs,
)

PointEstimateResult(
    meta, run, estimand, values, objective, uncertainty,
    gradient_norm, residual, iterations, local_only, report_refs,
)

SimulationResult(
    meta, run, parameter_source, parameters, latent_draws,
    observation_draws, prediction_design, report_refs,
)
```

`EvidenceComponent(name, log_value, standard_error, method, artifact_refs)` 明确 exact/residual 分解。不可用字段用 `None`，不填 `nan`；序列化仍保留字段，因此 schema shape 稳定。

Result 的复合字段类型也在 R1 固定：

| 字段 | 类型/约束 |
|---|---|
| `NamedArray` | `(name: str, value: np.ndarray, dims: tuple[str, ...])`；`len(dims) == value.ndim` |
| posterior `draws` | `tuple[NamedArray, ...]`，latent name 唯一且 leading draw count 相同 |
| `DrawsPosterior.chain_shape` | `(num_chains, draws_per_chain) | None`；iid draws 用 `None` |
| `log_weights` | 一个一维 `NamedArray`，长度等于 weighted draw count |
| analytic `parameters` / `moments` | key 唯一的 `tuple[NamedArray, ...]` |
| artifact/report refs | `tuple[ArtifactRef, ...]` 或单个 `ArtifactRef | None`，并由 lineage 校验 kind/revision |
| `log_density_availability` | `NONE | JOINT | POINTWISE` enum；pointwise 可用时相应数组不得为 `None` |
| `EvidenceResult.log_evidence` | finite `float`；`standard_error` 是非负 finite `float | None` |
| `PointEstimateResult.values` | name 唯一的 `tuple[NamedArray, ...]` |
| `PointEstimateResult.uncertainty` | `UncertaintyRecord(kind, arrays) | None`，kind 为 `COVARIANCE | PRECISION | STANDARD_ERROR` |
| scalar diagnostics | 未计算为 `None`；已计算必须 finite，iteration/count 不得为负 |

### 0.6 Refusal、Report 与 gate truth table

```python
Finding(code, message, observed, expected, artifact_refs)
ScopeRef(kind, name)
Remedy(action, message, parameters)
FallbackOption(task_kind, backend, conservatism, automatic)

Refusal(
    meta, task, failed_premise, grounds, scope, remedies, fallback,
)
```

`grounds` 与 `remedies` 均须非空。普通方法不适用转为 Refusal；malformed Graph、编程错误和运行时资源错误继续是 exception/error，不伪装成 statistical refusal。

```python
EvaluationReport(
    meta, subject_ref, report_kind, applicability, conclusion, findings,
)
```

合法 pair 只有：

- `APPLICABLE × {PASS, FAIL, ABSTAIN}`；
- `INAPPLICABLE × ABSTAIN`；
- `UNVERIFIABLE × ABSTAIN`。

Gate schema：

```python
ReportRequirement(
    name, required, optional_error_blocks,
)
ReportSlot(
    requirement, report, attempt_status, invalidated, error,
)
GateDefinition(name, version, requirements, blocked_actions, remedies)
GateResult(
    meta, definition, status, verdict, report_refs,
    findings, blocked_actions, remedies,
)

aggregate_gate(
    definition, *, meta, prerequisites_ready, inputs_current, slots
) -> GateResult
```

`ReportSlot.error` 是 `ErrorRecord(code, message, exception_type, traceback_ref) | None`，只保存可序列化的失败摘要和外部 traceback artifact reference，不保存 exception object。

聚合按以下优先级，输入顺序不得影响结果：

1. 直接前置 artifact 缺失 → `BLOCKED`, `verdict=None`；
2. gate/input/任何被消费 report 失效 → `INVALIDATED`, `verdict=None`；
3. 已尝试的 required report 出错、聚合出错，或 schema 声明会阻断的 optional error → `ERROR`, `verdict=None`；
4. 其余为 `EVALUATED`：任一 required applicable FAIL → `FAIL`；否则 required report 缺失、UNVERIFIABLE、ABSTAIN 或 required INAPPLICABLE → `ABSTAIN`；optional INAPPLICABLE 被忽略；只有所有 required applicable reports 都 PASS → `PASS`。

相同 requirement name 重复、schema 未声明的 slot、一个 slot 同时有 report/error、非法 report pair 都在构造或聚合边界拒绝，防止“最后一个覆盖前一个”的 silent bug。

---

## Task 1：canonical codec

**Files:**

- Create: `src/bayesmith/artifacts/_codec.py`
- Create: `src/bayesmith/artifacts/__init__.py`
- Create: `tests/artifacts/__init__.py`
- Create: `tests/artifacts/test_codec.py`
- Modify: `src/bayesmith/__init__.py`

- [ ] **1.1 先写 codec 的失败测试**

覆盖：dataclass/StrEnum/tuple/sorted mapping、UTC datetime、`float.hex()` 正常值与规范化的 NaN/±Inf、C/F-order array 归一后同 fingerprint、dtype/shape/bytes round-trip、object array/callable/未知 tag 拒绝、同一对象两次 dumps byte-identical。

```python
def test_canonical_dumps_is_byte_stable_and_mapping_order_independent(): ...
def test_array_round_trip_preserves_dtype_shape_and_values(): ...
def test_decoder_refuses_unregistered_type_and_object_array(): ...
```

- [ ] **1.2 运行红灯**

Run: `.venv/bin/python -m pytest tests/artifacts/test_codec.py`

Expected: FAIL，因为 `bayesmith.artifacts._codec` 尚不存在。

- [ ] **1.3 实现白名单 codec**

公开内部接口固定为：

```python
def register_artifact_type(cls: type[T]) -> type[T]: ...
def canonical_payload(value: object) -> object: ...
def canonical_dumps(value: object) -> bytes: ...
def canonical_loads(payload: bytes, *, expected: type[T] | None = None) -> T | object: ...
```

数组编码为 base64 C-order bytes + NumPy dtype string + shape；decoder 只查本包显式 registry，不做 `importlib.import_module()`。错误类型为 `ArtifactCodecError(ValueError)`。

- [ ] **1.4 用测试内注册类型钉住 registry 行为**

`artifacts/__init__.py` 在本 task 只含包 docstring，不提前公开未实现名字。在 test module 内定义一个显式 `@register_artifact_type` 的 frozen dataclass 和一个同形状但未注册类型：前者必须 round-trip，后者必须拒绝。这样 Task 1 的提交独立可绿，不预注册尚未存在的协议类型。

**同一提交内必须把 `"artifacts"` 加入 `src/bayesmith/__init__.py` 的 `_LAZY_SUBMODULES`。** 这不是 Task 8 的公共 API 工作，而是 Task 1 自身的绿灯条件：`tests/test_public_api.py::test_every_submodule_is_reachable_after_a_bare_import` 用 `pkgutil.iter_modules` 从**文件系统推导**期望集合并断言双向相等，所以 `artifacts/__init__.py` 一旦存在，该测试立即失败。把注册推迟到 Task 8 会让全套测试从 Task 1 红到 Task 8，跨越七个提交，而每个 task 的定向 gate 都看不见它——违反本仓库“HEAD 必须是你想恢复的状态、提交后才允许 mutation 测试”的铁律。从文件系统推导的守卫没有宽限期：**新子包的创建与注册必须同处一个提交。** Task 8 仍然负责 `artifacts` 的 public re-export 表面，那是另一件事。

`tests/artifacts/__init__.py` 同样属于本 task：`tests/` 自己是包，仓库现有五个测试子目录全部带 `__init__.py`，缺它会让本目录的测试以不同的模块命名方式被导入。

- [ ] **1.5 运行 Task 1 tests 和 lint**

Run: `.venv/bin/python -m pytest tests/artifacts/test_codec.py tests/test_public_api.py`

Expected: PASS。`test_public_api.py` 在此明确入列，因为它是唯一能证明新子包已被注册的守卫；它红就是 `_LAZY_SUBMODULES` 漏了 `artifacts`，不是 codec 的问题。

Run: `.venv/bin/ruff check src/bayesmith/artifacts/__init__.py src/bayesmith/artifacts/_codec.py src/bayesmith/__init__.py tests/artifacts/test_codec.py`

Expected: `All checks passed!`

- [ ] **1.6 提交独立变更**

```bash
git add src/bayesmith/artifacts/__init__.py src/bayesmith/artifacts/_codec.py src/bayesmith/__init__.py tests/artifacts/__init__.py tests/artifacts/test_codec.py
git commit -m "feat: add canonical artifact codec"
```

---

## Task 2：identity、基础 envelope 与 invalidation

**Files:**

- Create: `src/bayesmith/artifacts/identity.py`
- Create: `src/bayesmith/artifacts/base.py`
- Create: `tests/artifacts/test_identity.py`
- Create: `tests/artifacts/test_base.py`

- [ ] **2.1 写 fingerprint 边界的表驱动测试**

固定 §0.3 七槽 bundle、`ModelRef`、`fingerprint(kind, payload)` 和失效矩阵。至少包含这些 mutant：改变 array 一个 byte 必须改变 data digest；改变 progress label 不得改变 task digest；mapping insertion order 不得改变 digest；callable `repr` 含地址时必须拒绝；只改变 evaluation threshold 不得使 Result 失效。

```python
@pytest.mark.parametrize(("changed", "plan", "result", "report"), [...])
def test_default_invalidation_matrix(changed, plan, result, report): ...
```

- [ ] **2.2 运行红灯**

Run: `.venv/bin/python -m pytest tests/artifacts/test_identity.py`

Expected: FAIL，因为 identity API 尚不存在。

- [ ] **2.3 实现纯数据 identity 层**

```python
def fingerprint(kind: FingerprintKind, payload: object) -> Fingerprint: ...
def model_ref_from_callable(
    fn: Callable[..., object], *, identifier: str,
    source_digest: str | None = None,
    package: str | None = None,
    package_version: str | None = None,
    build_arguments: tuple[tuple[str, object], ...] = (),
) -> ModelRef: ...

def changed_fingerprints(
    before: FingerprintBundle, after: FingerprintBundle
) -> frozenset[FingerprintKind]: ...

class InvalidationPolicy:
    @classmethod
    def default(cls) -> InvalidationPolicy: ...
    def affected(self, artifact_type: ArtifactKind, changed: frozenset[FingerprintKind]) -> bool: ...
```

`identity.py` 可以依赖 `_codec.py`，不能依赖 `base.py`、Graph 或 JAX；它拥有 `Fingerprint`、`FingerprintBundle`、`ModelRef`、`ArtifactKind` 和 `InvalidationPolicy`，因此依赖保持 `_codec ← identity ← base`，没有循环。

- [ ] **2.4 实现 §0.2 的基础 envelope 与不变量测试**

`base.py` 依赖 `identity.py`，定义 `NamedArray`、`RunWarning(code, message, scope)`、`ErrorRecord(code, message, exception_type, traceback_ref)`、budget/seed/device/backend/termination/timing/approximation、`ArtifactRef`、`LifecycleRecord`、`ArtifactMeta`、`RunRecord`、`new_artifact_meta` 和 `invalidate_meta`。测试 NamedArray defensive copy/read-only/dims、UUID/created-at round-trip、非法 schema version、非 UTC 时间、负预算/时长、重复 parent ref、revision-aware immutable invalidation，以及 `ApproximationRecord` 的 enum/字段约束。Refusal 的 `Finding` 留到 Task 4；provenance warning 不借用尚未定义的 Refusal 类型。

- [ ] **2.5 跑定向测试、codec 回归和 lint**

Run: `.venv/bin/python -m pytest tests/artifacts/test_identity.py tests/artifacts/test_base.py tests/artifacts/test_codec.py`

Expected: PASS。

Run: `.venv/bin/ruff check src/bayesmith/artifacts tests/artifacts`

Expected: PASS。

- [ ] **2.6 提交**

```bash
git add src/bayesmith/artifacts/base.py src/bayesmith/artifacts/identity.py tests/artifacts/test_base.py tests/artifacts/test_identity.py
git commit -m "feat: define artifact identity and invalidation"
```

---

## Task 3：五类 Task 与五类 Result tagged unions

**Files:**

- Create: `src/bayesmith/artifacts/tasks.py`
- Create: `src/bayesmith/artifacts/results.py`
- Create: `tests/artifacts/test_tasks_results.py`

- [ ] **3.1 先写穷尽性与 round-trip tests**

每种 Task/Result 至少构造一个最小合法对象，round-trip 后检查类型和所有字段；TaskKind set 与主 Result 映射 key set 必须完全相等，ResultKind set 与映射 value set 完全相等且一对一。增加 PredictiveResult 的独立 round-trip，防止再次出现 5:4。

```python
assert set(PRIMARY_RESULT_BY_TASK) == set(TaskKind)
assert set(PRIMARY_RESULT_BY_TASK.values()) == set(ResultKind)
assert len(set(PRIMARY_RESULT_BY_TASK.values())) == len(ResultKind)
```

还要测：mutable input array 在构造后改变不影响 artifact；重复 latent/site name 拒绝；weighted draws 的 leading dimension 与 log_weights 不同拒绝；缺失/多余 named values 拒绝；不可用数值用 `None` 而非 NaN。

- [ ] **3.2 运行红灯**

Run: `.venv/bin/python -m pytest tests/artifacts/test_tasks_results.py`

Expected: FAIL，因为 Task/Result modules 尚不存在。

- [ ] **3.3 实现 §0.4 的 Task family**

定义 `TaskMeta`、复用 `ComputeBudget`、五个 task dataclass、`TaskKind` 和 `Task` type alias，并提供不依赖 Result 层的：

```python
def task_fingerprint(task: Task) -> Fingerprint: ...
```

fingerprint 明确排除 identity/timestamp/display fields，保留所有改变统计语义和执行预算的字段。

- [ ] **3.4 实现 §0.5 的 Result family**

复用 `base.NamedArray`，实现四个 posterior representations、五个 Result、`ResultKind`、`Result` union、`result_kind()` 以及 `PRIMARY_RESULT_BY_TASK: Mapping[TaskKind, ResultKind]`。mapping 位于 `results.py`，由较高层依赖 `tasks.py`，不能为了把常量放在 Task 模块而制造 `tasks ↔ results` 循环。`FittedConditionalPosterior` 只保存 estimator/backend artifact reference 与训练/验证 lineage，不保存 callable/model object。

- [ ] **3.5 测试 codec registry 的安全闭包**

确认 `canonical_loads` 可还原所有 Task/Result，但伪造相同模块名的未注册 class 不能解码。确认 `PosteriorTask` 绝不可能解码为 `EvidenceTask`。

- [ ] **3.6 跑 tests 和 lint**

Run: `.venv/bin/python -m pytest tests/artifacts/test_tasks_results.py tests/artifacts`

Expected: PASS。

Run: `.venv/bin/ruff check src/bayesmith/artifacts tests/artifacts`

Expected: PASS。

- [ ] **3.7 提交**

```bash
git add src/bayesmith/artifacts/tasks.py src/bayesmith/artifacts/results.py tests/artifacts/test_tasks_results.py
git commit -m "feat: freeze task and result protocols"
```

---

## Task 4：AnalysisReport、plan record 与 structured Refusal

**Files:**

- Create: `src/bayesmith/artifacts/refusal.py`
- Create: `src/bayesmith/artifacts/reports.py`
- Create: `tests/artifacts/test_refusal.py`
- Create: `tests/artifacts/test_reports.py`

- [ ] **4.1 写 Refusal schema tests**

测试 `grounds`/`remedies` 非空、没有 `evidence` field、Scope 和 fallback round-trip、message 改写不影响 code/scope、malformed shape 不可构造。

```python
assert "grounds" in {field.name for field in dataclasses.fields(Refusal)}
assert "evidence" not in {field.name for field in dataclasses.fields(Refusal)}
```

- [ ] **4.2 写 Analysis/Plan report tests**

冻结：

```python
AnalysisFinding(code, conclusion, scope, measurements, grounds)
AnalysisReport(meta, model_ref, graph_fingerprint, findings, candidate_routes)
PlanBlockRecord(names, method, reason_codes, kappa, tolerance, approximation)
InferencePlanRecord(
    meta, task_id, model_ref, analysis_report_ref, blocks,
    exact_elimination, residual_parameters, backend, premises,
    budget, quality_gate, fallback_policy,
)
```

测试 plan round-trip、parent lineage、block order 稳定、duplicate name 拒绝、runtime Graph/callable 无法进入字段。

- [ ] **4.3 运行红灯**

Run: `.venv/bin/python -m pytest tests/artifacts/test_refusal.py tests/artifacts/test_reports.py`

Expected: FAIL。

- [ ] **4.4 实现 Refusal 与 report records**

`Refusal` 的 `task` 保存完整 Task；`failed_premise` 是稳定 code；human message 只做解释。`AnalysisReport` 是 compile-time graph interpretation；`EvaluationReport` 暂只定义基础形状，Task 5 再钉合法状态组合。

- [ ] **4.5 跑 tests、全 artifact 回归和 lint**

Run: `.venv/bin/python -m pytest tests/artifacts`

Expected: PASS。

Run: `.venv/bin/ruff check src/bayesmith/artifacts tests/artifacts`

Expected: PASS。

- [ ] **4.6 提交**

```bash
git add src/bayesmith/artifacts/refusal.py src/bayesmith/artifacts/reports.py tests/artifacts/test_refusal.py tests/artifacts/test_reports.py
git commit -m "feat: add plan reports and structured refusals"
```

---

## Task 5：EvaluationReport 与 gate 聚合真值表

**Files:**

- Create: `src/bayesmith/artifacts/gates.py`
- Create: `tests/artifacts/test_gates.py`
- Modify: `src/bayesmith/artifacts/reports.py`

- [ ] **5.1 写两轴状态的 exhaustive tests**

参数化全部 applicability × conclusion 组合，非法 pair 必须在 `EvaluationReport` 构造时失败。分别验证 BLOCKED、INVALIDATED、ERROR 没有 verdict，EVALUATED 必有 verdict。

- [ ] **5.2 写 gate truth-table 和 permutation tests**

至少覆盖：

| 情景 | status | verdict |
|---|---|---|
| prerequisite missing | BLOCKED | None |
| input/report stale | INVALIDATED | None |
| required attempted error | ERROR | None |
| blocking optional error | ERROR | None |
| non-blocking optional error + required pass | EVALUATED | PASS |
| required applicable fail | EVALUATED | FAIL |
| required never produced | EVALUATED | ABSTAIN |
| required unverifiable/abstain | EVALUATED | ABSTAIN |
| required inapplicable | EVALUATED | ABSTAIN |
| optional inapplicable + all required pass | EVALUATED | PASS |
| all required applicable pass | EVALUATED | PASS |

对每个多 slot case 遍历或 property-generate slot permutations，传入同一个 `meta` 并断言完整 `GateResult` 相同。另测 FAIL 与 ABSTAIN 同时存在时 FAIL 获胜；这是已冻结 precedence，不依赖 slot 顺序。

- [ ] **5.3 运行红灯**

Run: `.venv/bin/python -m pytest tests/artifacts/test_gates.py`

Expected: FAIL。

- [ ] **5.4 实现 deterministic aggregator**

`aggregate_gate` 首先按 requirement name 建唯一映射，验证 schema/slot 一致，再依 §0.6 的固定 priority 聚合。不得用输入 list 的 first/last verdict。错误详情进入 `Finding`；blocked actions/remedies 从 `GateDefinition` 原样传递。

- [ ] **5.5 跑 gate、codec 和全 artifact tests**

Run: `.venv/bin/python -m pytest tests/artifacts/test_gates.py tests/artifacts`

Expected: PASS。

Run: `.venv/bin/ruff check src/bayesmith/artifacts tests/artifacts`

Expected: PASS。

- [ ] **5.6 提交**

```bash
git add src/bayesmith/artifacts/reports.py src/bayesmith/artifacts/gates.py tests/artifacts/test_gates.py
git commit -m "feat: add deterministic quality gate aggregation"
```

---

## Task 6：Graph manifest、task-aware compile 与 refusal adapters

**Files:**

- Create: `src/bayesmith/dispatch/task.py`
- Create: `tests/dispatch/test_task_protocol.py`
- Modify: `src/bayesmith/artifacts/refusal.py`
- Modify: `src/bayesmith/artifacts/reports.py`
- Modify: `tests/test_layering.py`

- [ ] **6.1 写 Graph/data fingerprint seam tests**

使用现有 fixtures 构造只改一项的 graph pairs：

- 改 observed value：只改变 data fingerprint；
- 改 mask bit：只改变 data fingerprint；
- 改 node parents、plate size、support、`linear_in` 或 callable module/qualname：改变 graph structure；
- 保持 callable identity 但改 model source digest：只改变 model-source fingerprint；
- 改 Const value：只改变 data；
- 相同 graph 重 trace：fingerprints 相同；
- closure source 无法稳定识别且没有显式 ModelRef digest：typed refusal，不用 `repr`。

`graph_manifest(graph, model_ref)` 和 `data_manifest(graph, extra_data=())` 在 `dispatch.task`，因为它们知道 Graph；artifacts 层只 hash manifest。

- [ ] **6.2 写 compile_task contract tests**

冻结 public seam：

```python
@dataclass(frozen=True, slots=True)
class PlannedTask:
    task: Task
    analysis: AnalysisReport
    record: InferencePlanRecord
    runtime_plan: InferencePlan = field(compare=False, repr=False)

def compile_task(
    graph: Graph,
    task: Task,
    *,
    model_ref: ModelRef,
    key: jax.Array | None = None,
    extra_data: tuple[tuple[str, object], ...] = (),
) -> PlannedTask | Refusal: ...
```

PosteriorTask 和受支持的 PointEstimateTask 返回 PlannedTask；Evidence/Predictive/Simulation 返回 `failed_premise="capability_unavailable_r1"`、非空 grounds/remedies、所请求 Task 的 Refusal。unsupported backend 同样 typed refusal。GraphError/StructureError 不捕获成方法 refusal。

- [ ] **6.3 写 exception-to-refusal adapters tests**

为 `NotGaussian`、`NotLogLinear` 和 `diagnose.map.Refused` 各做一个 test：改变其 human-readable message 后，Refusal code/scope/observed fields 不变。adapter 只读公开结构化 attributes，或在只有一个 reason 字段的既有 verdict 上按**结果类型与调用上下文**给固定 code；不能 split/regex `str(exc)`。`AffinityRefused` 是现有 `StructureError` 子类，表示用户声明的 affinity claim 被证伪，继续作为 graph-contract exception 传播，不降格成 ordinary method Refusal。

- [ ] **6.4 运行红灯**

Run: `.venv/bin/python -m pytest tests/dispatch/test_task_protocol.py`

Expected: FAIL。

- [ ] **6.5 实现 bridge 与 manifest**

复用 `dispatch.plan.compile()` 产生 runtime plan；把 plan blocks、method、reason code、kappa/tol、fallback policy 投影为 serializable records。`reason` 的人类文字可以保存作 message，但决策 code 必须来自结构化 route/classification 字段或显式 adapter mapping。

- [ ] **6.6 钉住分层和轻量 import**

Modify: `tests/test_layering.py`

Add assertions:

```python
assert _graph()["artifacts"] == set()
assert "artifacts" in _graph()["dispatch"]
```

另在 subprocess 中 `import bayesmith.artifacts`，断言 `jax`、`numpyro`、`equinox` 未进入 `sys.modules`。

- [ ] **6.7 跑定向 suite 和 lint**

Run: `.venv/bin/python -m pytest tests/dispatch/test_task_protocol.py tests/test_layering.py tests/artifacts`

Expected: PASS。

Run: `.venv/bin/ruff check src/bayesmith/artifacts src/bayesmith/dispatch/task.py tests/artifacts tests/dispatch/test_task_protocol.py tests/test_layering.py`

Expected: PASS。

- [ ] **6.8 提交**

```bash
git add src/bayesmith/dispatch/task.py src/bayesmith/artifacts/refusal.py src/bayesmith/artifacts/reports.py tests/dispatch/test_task_protocol.py tests/test_layering.py
git commit -m "feat: compile tasks into artifact-backed plans"
```

---

## Task 7：适配现有 posterior 与 point-estimate 执行结果

**Files:**

- Modify: `src/bayesmith/dispatch/task.py`
- Create: `tests/dispatch/test_task_execution.py`
- Read/verify only unless a structured field is demonstrably missing: `src/bayesmith/dispatch/execute.py`
- Read/verify only unless a structured field is demonstrably missing: `src/bayesmith/optimize.py`
- Read/verify only unless a structured field is demonstrably missing: `src/bayesmith/diagnose/map.py`

- [ ] **7.1 先写 numerical parity tests**

同一个 graph、key 和 budget 各跑旧/新入口：

```python
def execute_task(
    planned: PlannedTask, *, key: jax.Array | None = None
) -> Result | Refusal: ...
```

至少覆盖：pure exact `gcr`、weighted `gcr+snis`、pure `nuts`、`InferencePlan.estimate()` posterior mean、`map_estimate()` 或 `fit()` 的 MAP。逐 latent `assert_allclose` 旧数组和 `NamedArray.value`；method、weights、ESS、khat、unreliable、residual、iterations 和 objective 必须逐字段一致。相同 key 只能执行一次旧路、一次新路且不额外 split，保证随机流 parity。

- [ ] **7.2 写 provenance tests**

检查 Result 的 parent ref 与 RunRecord.plan_ref 都精确指向 plan 的 id/revision；RunRecord 记录实际 backend/method、seed/dtype/device/JAX x64/budget/termination/timing；requested backend 与实际 fallback 分开记录；weighted sample 不丢 weights；未运行 chain 的 diagnostics 保持“不适用/未产生”，不伪造 R-hat。

- [ ] **7.3 运行红灯**

Run: `.venv/bin/python -m pytest tests/dispatch/test_task_execution.py`

Expected: FAIL，因为 execute adapter 尚不存在。

- [ ] **7.4 实现无数值逻辑的 adapters**

新函数只调用 `planned.runtime_plan.sample()`/`.estimate()` 或现有 MAP/fit seam，再做以下机械投影：

- legacy `Posterior.log_weights is None` → `DrawsPosterior`；否则 `WeightedDrawsPosterior`；
- legacy `Estimate`/`MapEstimate`/`Fit` → `PointEstimateResult`，按 `Estimand` 填语义；
- timing 包围一次旧入口调用；
- exception 只有已知 method-inapplicability 进入 Refusal，`ConvergenceError` 等实际执行故障继续抛出，供 workflow 标成 ERROR；
- Evidence/Predictive/Simulation 的 PlannedTask 在 R1 不应存在；若手工伪造，`execute_task` 防御性返回 `capability_unavailable_r1` Refusal。

- [ ] **7.5 验证旧 API 完全不变**

Run: `.venv/bin/python -m pytest tests/dispatch tests/test_public_api.py tests/test_conjugate_oracle.py tests/test_bridge.py`

Expected: PASS；旧 `Posterior`/`Estimate` identity 和行为不变。

- [ ] **7.6 跑执行定向 tests 和 lint**

Run: `.venv/bin/python -m pytest tests/dispatch/test_task_execution.py tests/artifacts`

Expected: PASS。

Run: `.venv/bin/ruff check src/bayesmith/dispatch/task.py tests/dispatch/test_task_execution.py`

Expected: PASS。

- [ ] **7.7 提交**

```bash
git add src/bayesmith/dispatch/task.py tests/dispatch/test_task_execution.py
git commit -m "feat: adapt inference runs to typed results"
```

---

## Task 8：公共 API、持久化入口、兼容说明

**Files:**

- Modify: `src/bayesmith/artifacts/__init__.py`
- Modify: `src/bayesmith/artifacts/_codec.py`
- Modify: `src/bayesmith/__init__.py`
- Modify: `tests/test_public_api.py`
- Create: `docs/artifacts.md`
- Modify: `README.md`
- Modify: `docs/ownership.md`

- [ ] **8.1 定义最小 public surface tests**

`bayesmith.artifacts` re-export 本计划的 Task/Result/Report/Refusal/Gate/identity 类型以及 `dump_artifact`、`load_artifact`；root 只新增 lazy `compile_task`、`execute_task`，不把几十个 schema 名全铺到 root。identity tests 必须像现有 public API tests 一样检查 owning object，而不只 `hasattr`。

- [ ] **8.2 增加文件持久化 helpers**

```python
def dump_artifact(artifact: Artifact, path: str | os.PathLike[str]) -> None: ...
def load_artifact(
    path: str | os.PathLike[str], *, expected: type[T] | None = None
) -> T | Artifact: ...
```

写入采用同目录 temporary file + `os.replace`，避免半文件；读取验证 schema、digest 和 expected type。R1 只支持本 schema version 1；未来版本返回明确 `UnsupportedSchemaVersion`，不猜迁移。

磁盘格式不是裸 pickle，也不让 artifact 自证自身 digest；它是一个 canonical JSON transport envelope：

```python
@dataclass(frozen=True, slots=True)
class ArtifactFile:
    format: Literal["bayesmith-artifact"]
    codec_version: Literal[1]
    payload_sha256: str
    payload_base64: str
```

`payload_sha256` 是 decoded payload bytes 的 SHA-256 hex digest；`payload_base64` 是 `canonical_dumps(artifact)` 的 base64 文本。`load_artifact` 先校验固定 format/codec version 和 payload digest，再把 decoded payload 交给白名单 decoder。测试分别翻转 payload byte、digest byte 和 type tag，三者都必须在返回对象前失败。

- [ ] **8.3 写文档验收 tests**

README 示例展示 `PosteriorTask → compile_task → execute_task → PosteriorResult`，同时明确 legacy entry points 仍支持；`docs/artifacts.md` 包含五进五出表、fingerprint 边界、失效矩阵、Refusal.grounds 和 gate truth table。增加轻量文档 guard，钉住五类 Result 名称和 `grounds`，但不把完整 prose 做脆弱 snapshot。

- [ ] **8.4 更新 ownership**

把 `bayesmith.artifacts` 标为 first-party semantic core，把 `dispatch.task` 标为 first-party orchestration adapter；不得把 generic backend 算法 ownership 拉回本包。`bayesmith.evidence` tombstone 裁决不变。

- [ ] **8.5 跑 public/docs tests 和 lint**

Run: `.venv/bin/python -m pytest tests/test_public_api.py tests/test_readme_count.py tests/artifacts tests/dispatch/test_task_protocol.py tests/dispatch/test_task_execution.py`

Expected: PASS。

Run: `.venv/bin/ruff check src/ tests/`

Expected: `All checks passed!`

- [ ] **8.6 提交**

```bash
git add src/bayesmith/artifacts/__init__.py src/bayesmith/artifacts/_codec.py src/bayesmith/__init__.py tests/test_public_api.py docs/artifacts.md README.md docs/ownership.md
git commit -m "docs: publish the R1 artifact protocol"
```

---

## Task 9：R1 完成门槛、wheel 与 consumer gate

**Files:**

- Create: `docs/superpowers/specs/2026-08-30-r1-close-out.md`
- Modify only if a measured failure requires it: files owned by Tasks 1–8

- [ ] **9.1 建立验收矩阵，不先写绿色结论**

close-out 先列门槛和空的 measured-result cells；只有后续命令真实完成才填数字。至少逐项回答：

- analytic/exact draw、MCMC、optimization 是否都返回统一协议；
- 同一 artifact 是否 byte-stable round-trip；
- model/graph/data/task/compilation/evaluation changes 是否按矩阵失效；
- Refusal 是否只读 structured fields；
- 五类 Task→Result 是否穷尽一对一；
- gate 状态/verdict 是否两轴且 permutation-invariant；
- 新旧执行数组是否数值一致；
- sibling consumer 是否无回归。

- [ ] **9.2 跑 source full suite，单独保存 exit code 和 JUnit**

```bash
.venv/bin/python -m pytest -n 4 --junit-xml=run.r1.junit.xml > run.r1.log 2>&1
printf 'PYTEST_EXIT=%s\n' $? > run.r1.exit
cat run.r1.exit
```

Expected: exit 0；passed/skipped/failed/error 数从 JUnit 读取，不从 `-qq` 消失的终端摘要猜。

- [ ] **9.3 跑 lint**

Run: `.venv/bin/ruff check src/ tests/`

Expected: `All checks passed!`。不要运行 format sweep。

- [ ] **9.4 构建 sdist/wheel 并在 repository 外验证**

```bash
r1_build_dir="$(mktemp -d /tmp/bayesmith-r1-build.XXXXXX)"
.venv/bin/python -m build --outdir "$r1_build_dir"
python3 -m venv "$r1_build_dir/venv"
uv pip install --python "$r1_build_dir/venv/bin/python" "$r1_build_dir"/bayesmith-*.whl pytest pytest-xdist hypothesis
(cd /tmp && "$r1_build_dir/venv/bin/python" -c 'import bayesmith; print(bayesmith.__file__)')
(cd /tmp && "$r1_build_dir/venv/bin/python" -m pytest /Users/zzhang/projects/bayesmith/tests -n 4 --junit-xml="$r1_build_dir/wheel.junit.xml")
printf 'WHEEL_PYTEST_EXIT=%s\n' $? > "$r1_build_dir/wheel.exit"
cat "$r1_build_dir/wheel.exit"
```

确认打印的 `bayesmith.__file__` 位于该 venv 的 `site-packages`。记录 wheel suite 的 JUnit、独立 exit code 以及因为 sibling 未安装产生的预期 skips；不得用 source-tree import 代替 wheel identity。

- [ ] **9.5 跑 rheplicant consumer gate**

在 `/Users/zzhang/projects/e-RHINO` 先记录 revision、dirty status 和实际 import paths；不修改或清理 sibling 的既有改动。用它自己的环境运行：

```bash
r1_consumer_dir="$(mktemp -d /tmp/bayesmith-r1-consumer.XXXXXX)"
.venv/bin/python -m pytest tests/inference -n 4 --junit-xml="$r1_consumer_dir/inference.xml"
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/seam --junit-xml="$r1_consumer_dir/seam.xml"
```

Expected: 无新 failure/error；已声明 xfail 保持 xfail。若 consumer 需要迁移，先记录协调决定，不能静默破坏。

- [ ] **9.6 用新数据完成 close-out**

记录 commit SHA、dirty/concurrent changes、exact commands、JUnit counts、wheel identity、consumer revision 和每条 R1 completion criterion 的证据。若任何 gate 未过，文件状态写 `R1 open` 并列 blocker，不能把部分绿色写成关闭。

- [ ] **9.7 最终 diff 与文档自检**

Run: `git diff --check`

Expected: 无输出。

Run: `git status --short`

Expected: 只含已审计的 R1 files 和开始前已存在的 user/concurrent changes；没有 run/verify artifact 被误 stage。

- [ ] **9.8 提交 close-out**

```bash
git add docs/superpowers/specs/2026-08-30-r1-close-out.md
git commit -m "docs: close R1 artifact foundation"
```

---

## 10. R1 明确不做的事

- 不接 nested sampling，不计算 graph-level evidence；那是 R4。
- 不建立 PPC/SBC/LOO/calibration 执行 workflow；那是 R3，R1 只冻结 Result/Report seam。
- 不扩展本地 NPE/flow/score architecture；只保留 fitted-conditional representation 和 provenance 字段。
- 不替换现有 linear sampler、NumPyro backend 或 optimizer；upstream 替换按顶层设计 §1.5 的真实 workload 门槛另行裁决。
- 不收复 deprecated `bayesmith.evidence` namespace。
- 不让 LLM/agent 参与 gate truth value；R1 的 aggregator 是确定性纯函数。
- 不做 artifact store、数据库、分布式 scheduler、自动迁移器或完整 workflow engine。

## 11. 完成定义

只有同时满足以下条件才可写 “R1 closed”：

1. 五种 Task、五种 Result、四种 posterior representation 和共同 RunRecord 都能安全 round-trip；
2. mapping 在测试中穷尽且一对一，PredictiveResult 不是 SimulationResult + Report 的隐式组合；
3. `Refusal.grounds` 已冻结，method inapplicability 不解析 exception string；
4. fingerprint 粒度与失效矩阵有 boundary mutant tests；
5. gate 两轴状态、完整 truth table 和 permutation invariance 有穷尽测试；
6. 当前 exact/MCMC/optimization 新旧入口通过逐字段数值 parity；
7. legacy API、source full suite、ruff check、built-wheel suite 和 rheplicant consumer gate 全绿；
8. close-out 使用本次实际 SHA、JUnit 与 consumer revision，不复用 R0 的绿色结果。
