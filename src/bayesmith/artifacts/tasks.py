"""The five questions a caller can ask, as data (§0.4).

A Task is the QUESTION, not the machinery that answers it. §0 ruling 1 fixes
five of them and §0 ruling 4 fixes what they may hold: a reference, a
fingerprint, a budget and a bounded option table -- never a Graph, a callable,
a compiled executable or a backend handle. That is why a predictive task
carries the *fingerprint* of its conditioning data rather than the data, and
why a simulation task's fixed parameters are :class:``NamedArray``s whose bytes
are copied in.

**A task's identity is what it asks, not when or by whom.**
:func:``task_fingerprint`` digests the kind and every statistical field, and
excludes ``meta`` entirely -- the id is minted, the timestamp is a clock, the
label is for a human, and the schema version is the envelope's own business
rather than one of the §0.3 task-slot contents. Renaming a task must not
invalidate a cached result; changing its tolerance must.

**Two kinds of collection, two rules, stated once.** A list of SITE NAMES keeps
the order it was declared in, because that order is what a caller reads back;
a collection keyed by name -- a parameter source's fixed values -- is sorted,
because the same values in two orders are one value and a digest that
disagreed would depend on the order somebody typed keywords in. Duplicates are
refused in both: one of the two would be dropped and nothing would say which.

The small validators are :mod:``bayesmith.artifacts.base``'s own, imported
rather than restated. A second ``_count`` here would be a second answer to "is
a negative budget legal", and this package has already paid for one
measurement living in six places.

Layering: ``_codec ← identity ← base ← tasks``. Nothing of the Graph, the
dispatch layer, JAX, Equinox or NumPyro (§0 ruling 5).
"""

from __future__ import annotations

import dataclasses
import math
import uuid
from enum import StrEnum

from ._codec import register_artifact_type
from .base import (
    SCHEMA_VERSION,
    ArtifactRef,
    CanonicalValue,
    ComputeBudget,
    NamedArray,
    _count,
    _instance,
    _member,
    _optional_count,
    _optional_text,
    _text,
    _timestamp,
    _tuple_of,
    _uuid4,
    canonical_value_options,
    utc_timestamp,
)
from .identity import ArtifactKind, Fingerprint, FingerprintKind, fingerprint

__all__ = [
    "TaskKind",
    "TaskMeta",
    "new_task_meta",
    "Estimand",
    "ParameterSourceKind",
    "ParameterSource",
    "PosteriorTask",
    "EvidenceTask",
    "PredictiveTask",
    "PointEstimateTask",
    "SimulationTask",
    "Task",
    "TASK_CLASSES",
    "task_kind",
    "task_fingerprint",
    "site_names",
]


@register_artifact_type
class TaskKind(StrEnum):
    """The five questions of §0 ruling 1, and no sixth.

    The kind is a member of the tagged union rather than a free string because
    it is what every dispatch, every gate and every result pairing branches on.
    """

    POSTERIOR = "posterior"
    EVIDENCE = "evidence"
    PREDICTIVE = "predictive"
    POINT_ESTIMATE = "point_estimate"
    SIMULATION = "simulation"


@register_artifact_type
class Estimand(StrEnum):
    """Which point of the posterior a point estimate is of (§0.4).

    Two members, and they are not interchangeable: a posterior mean is an
    integral, a MAP is an argmax, and they coincide only for a symmetric
    unimodal posterior. A single "point estimate" with a free-text method would
    let the difference be lost in a string.
    """

    POSTERIOR_MEAN = "posterior_mean"
    MAP = "map"


@register_artifact_type
class ParameterSourceKind(StrEnum):
    """Where a simulation's parameters come from (§0.4)."""

    PRIOR = "prior"
    FIXED = "fixed"
    POSTERIOR_RESULT = "posterior_result"


# ---------------------------------------------------------------- validation


def _flag(label: str, value: object) -> bool:
    # type(), not isinstance(): every int is truthy or falsy, and a flag that
    # accepted 2 would record a decision nobody made.
    if type(value) is not bool:
        raise TypeError(f"{label} is a bool; got {value!r}")
    return value


def _finite(label: str, value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{label} is a number; got {value!r}")
    if not math.isfinite(value):
        raise ValueError(
            f"{label} must be finite; got {value!r}. An unavailable number is "
            "None -- NaN compares unequal to itself and reads as a measurement"
        )
    return float(value)


def _positive(label: str, value: object) -> float:
    number = _finite(label, value)
    if number <= 0.0:
        raise ValueError(f"{label} is a positive number; got {value!r}")
    return number


def _optional_positive(label: str, value: object) -> float | None:
    if value is None:
        return None
    return _positive(label, value)


def _optional_non_negative(label: str, value: object) -> float | None:
    if value is None:
        return None
    number = _finite(label, value)
    if number < 0.0:
        raise ValueError(f"{label} is a non-negative number; got {value!r}")
    return number


def site_names(label: str, value: object) -> tuple[str, ...]:
    """Unique, non-empty site names IN THE ORDER THEY WERE DECLARED.

    Not sorted: a caller reads these back and the order they wrote is the
    order they mean. Duplicates are refused rather than collapsed -- a repeated
    site would silently shorten a collection whose length a caller may be
    checking.
    """
    if not isinstance(value, tuple):
        raise TypeError(f"{label} is a tuple of site names; got {value!r}")
    seen: set[str] = set()
    for name in value:
        _text(f"{label} entries", name)
        if name in seen:
            raise ValueError(
                f"{label} names {name!r} twice; a repeated site would be "
                "dropped from the collection it indexes without a word"
            )
        seen.add(name)
    return value


def _optional_site_names(label: str, value: object) -> tuple[str, ...] | None:
    """``None`` means "all of them"; an empty tuple asks for nothing and is
    refused, because a task that names nothing would produce nothing."""
    if value is None:
        return None
    names = site_names(label, value)
    if not names:
        raise ValueError(
            f"{label} is None for 'all of them'; an empty tuple asks for "
            "nothing, which is not a task"
        )
    return names


def _backend(label: str, value: object) -> str:
    """A backend NAME or the policy value "auto" (§0.4).

    Unlike :class:``bayesmith.artifacts.base.BackendRef``, which records what
    actually ran, a task may legitimately ask for "auto" -- choosing the
    backend is the dispatcher's job and the task is the request.
    """
    return _text(label, value)


def _data_fingerprint(label: str, value: object) -> Fingerprint | None:
    if value is None:
        return None
    _instance(label, value, Fingerprint)
    if value.kind is not FingerprintKind.DATA:
        raise ValueError(
            f"{label} is a fingerprint of the data slot; got one of kind "
            f"{value.kind.value!r}. §0.3 gives each slot a boundary, and a "
            "payload digest filed under another slot would answer for an "
            "input it was never taken from"
        )
    return value


def _result_ref(label: str, value: object) -> ArtifactRef:
    _instance(label, value, ArtifactRef)
    if value.artifact_type is not ArtifactKind.RESULT:
        raise ValueError(
            f"{label} points at a {value.artifact_type.value}; it names a "
            "result artifact"
        )
    return value


def _named_values(label: str, value: object) -> tuple[NamedArray, ...]:
    """Sorted, name-unique named arrays: a name-keyed collection has no order."""
    _tuple_of(label, value, NamedArray)
    seen: set[str] = set()
    for array in value:
        if array.name in seen:
            raise ValueError(
                f"{label} names {array.name!r} twice; one of the two arrays "
                "would be dropped without anything saying which"
            )
        seen.add(array.name)
    return tuple(sorted(value, key=lambda array: array.name))


#: "No budget was stated", shared by every task that does not carry one. One
#: object rather than a default_factory because ComputeBudget is frozen: there
#: is nothing to mutate, so five tasks holding this value hold one value, and
#: two tasks built a second apart compare equal.
_NO_BUDGET = ComputeBudget()


# -------------------------------------------------------------- task envelope


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class TaskMeta:
    """Who asked, and when. None of it is part of the task's fingerprint."""

    task_id: str
    schema_version: int
    created_at: str
    label: str = ""

    def __post_init__(self) -> None:
        _uuid4("task_id", self.task_id)
        if type(self.schema_version) is not int:
            raise TypeError(f"schema_version is an int; got {self.schema_version!r}")
        if self.schema_version < 1:
            raise ValueError(
                f"schema_version is a positive integer; got {self.schema_version!r}"
            )
        object.__setattr__(
            self, "created_at", _timestamp("created_at", self.created_at)
        )
        if type(self.label) is not str:
            raise TypeError(f"a task label is a string; got {self.label!r}")


def new_task_meta(
    *,
    label: str = "",
    task_id: str | None = None,
    created_at: str | None = None,
    schema_version: int = SCHEMA_VERSION,
) -> TaskMeta:
    """A fresh task envelope, minting the id and the time unless given them."""
    return TaskMeta(
        task_id=task_id if task_id is not None else str(uuid.uuid4()),
        schema_version=schema_version,
        created_at=created_at if created_at is not None else utc_timestamp(),
        label=label,
    )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ParameterSource:
    """Where a simulation's parameters come from: a three-armed tagged value.

    Exactly one arm carries a payload, and the other two must be empty. A
    single class with an unconstrained pair of optional fields would admit
    "fixed values AND a posterior reference", which is two answers to one
    question -- and whichever one the executor read first would become the
    silent rule.
    """

    kind: ParameterSourceKind
    values: tuple[NamedArray, ...] = ()
    posterior_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        _member("a parameter source's kind", self.kind, ParameterSourceKind)
        object.__setattr__(
            self, "values", _named_values("a parameter source's values", self.values)
        )
        if self.kind is ParameterSourceKind.FIXED:
            if not self.values:
                raise ValueError(
                    "a fixed parameter source carries the values it fixes; "
                    "with none it is the prior source spelled differently"
                )
        elif self.values:
            raise ValueError(
                f"a {self.kind.value} parameter source carries no values; the "
                f"{len(self.values)} given would be ignored by every executor"
            )

        if self.kind is ParameterSourceKind.POSTERIOR_RESULT:
            if self.posterior_ref is None:
                raise ValueError(
                    "a posterior parameter source names the result artifact it "
                    "draws from; without the reference there is nothing to draw"
                )
            _result_ref("a parameter source's posterior_ref", self.posterior_ref)
        elif self.posterior_ref is not None:
            raise ValueError(
                f"a {self.kind.value} parameter source carries no posterior "
                "reference"
            )

    @classmethod
    def prior(cls) -> ParameterSource:
        return cls(kind=ParameterSourceKind.PRIOR)

    @classmethod
    def fixed(cls, values: tuple[NamedArray, ...]) -> ParameterSource:
        return cls(kind=ParameterSourceKind.FIXED, values=values)

    @classmethod
    def from_posterior_result(cls, ref: ArtifactRef) -> ParameterSource:
        return cls(kind=ParameterSourceKind.POSTERIOR_RESULT, posterior_ref=ref)


# ------------------------------------------------------------- the five tasks


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class PosteriorTask:
    """Draw, or represent, the posterior over the model's latents.

    ``nuts_on_collapse`` is a fallback POLICY rather than a fallback: the task
    says whether the caller consents to being moved onto NUTS when an exact
    route collapses, and the run record says whether it happened.
    """

    meta: TaskMeta
    backend: str = "auto"
    budget: ComputeBudget = _NO_BUDGET
    chain_method: str | None = None
    solver_tolerance: float | None = None
    solver_maxiter: int | None = None
    require_convergence: bool = True
    ess_floor: float | None = None
    nuts_on_collapse: bool = True
    backend_options: tuple[tuple[str, CanonicalValue], ...] = ()
    quality_gate: str | None = None

    def __post_init__(self) -> None:
        _instance("a posterior task's meta", self.meta, TaskMeta)
        _backend("a posterior task's backend", self.backend)
        _instance("budget", self.budget, ComputeBudget)
        _optional_text("chain_method", self.chain_method)
        object.__setattr__(
            self,
            "solver_tolerance",
            _optional_positive("solver_tolerance", self.solver_tolerance),
        )
        _optional_count("solver_maxiter", self.solver_maxiter)
        _flag("require_convergence", self.require_convergence)
        object.__setattr__(
            self, "ess_floor", _optional_non_negative("ess_floor", self.ess_floor)
        )
        _flag("nuts_on_collapse", self.nuts_on_collapse)
        object.__setattr__(
            self,
            "backend_options",
            canonical_value_options("backend_options", self.backend_options),
        )
        _optional_text("quality_gate", self.quality_gate)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceTask:
    """The marginal likelihood, and whether a posterior falls out of it.

    ``repeat_count`` is how many independent runs the caller wants for a
    consistency check. ``None`` is a single run; zero is refused, because
    repeating a run no times is not a second spelling of running it once.
    """

    meta: TaskMeta
    backend: str = "auto"
    budget: ComputeBudget = _NO_BUDGET
    reconstruct_posterior: bool = False
    repeat_count: int | None = None
    backend_options: tuple[tuple[str, CanonicalValue], ...] = ()
    quality_gate: str | None = None

    def __post_init__(self) -> None:
        _instance("an evidence task's meta", self.meta, TaskMeta)
        _backend("an evidence task's backend", self.backend)
        _instance("budget", self.budget, ComputeBudget)
        _flag("reconstruct_posterior", self.reconstruct_posterior)
        if self.repeat_count is not None:
            _count("repeat_count", self.repeat_count)
            if self.repeat_count < 1:
                raise ValueError(
                    "repeat_count is None for a single run; zero repeats is "
                    "not a second spelling of it"
                )
        object.__setattr__(
            self,
            "backend_options",
            canonical_value_options("backend_options", self.backend_options),
        )
        _optional_text("quality_gate", self.quality_gate)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class PredictiveTask:
    """Push an existing posterior forward onto new or replicated observations.

    What separates this from :class:``SimulationTask`` is stated in the fields
    rather than in prose (§0.4): a predictive task conditions on data, names
    the posterior it came from, and distinguishes the sites it replicates from
    the latents it carries forward. A simulation has none of those.
    """

    meta: TaskMeta
    source_posterior_ref: ArtifactRef
    conditioning_data: Fingerprint | None = None
    prediction_design: Fingerprint | None = None
    conditioned_sites: tuple[str, ...] = ()
    replicated_sites: tuple[str, ...] = ()
    latent_sites: tuple[str, ...] = ()
    budget: ComputeBudget = _NO_BUDGET
    backend: str = "auto"
    backend_options: tuple[tuple[str, CanonicalValue], ...] = ()
    quality_gate: str | None = None

    def __post_init__(self) -> None:
        _instance("a predictive task's meta", self.meta, TaskMeta)
        _result_ref("source_posterior_ref", self.source_posterior_ref)
        _data_fingerprint("conditioning_data", self.conditioning_data)
        _data_fingerprint("prediction_design", self.prediction_design)
        for name in ("conditioned_sites", "replicated_sites", "latent_sites"):
            site_names(name, getattr(self, name))
        if not self.replicated_sites and not self.latent_sites:
            raise ValueError(
                "a predictive task names at least one site to replicate or one "
                "latent to carry forward; naming neither asks for an empty "
                "result"
            )
        _instance("budget", self.budget, ComputeBudget)
        _backend("a predictive task's backend", self.backend)
        object.__setattr__(
            self,
            "backend_options",
            canonical_value_options("backend_options", self.backend_options),
        )
        _optional_text("quality_gate", self.quality_gate)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class PointEstimateTask:
    """One point of the posterior, for named latents or for all of them.

    ``names=None`` is "every latent" and an empty tuple is refused: an empty
    request and an unrestricted one are different questions, and a field that
    could mean either would make the answer depend on how the caller spelled
    "I do not care".
    """

    meta: TaskMeta
    estimand: Estimand
    names: tuple[str, ...] | None = None
    backend: str = "auto"
    budget: ComputeBudget = _NO_BUDGET
    optimizer_options: tuple[tuple[str, CanonicalValue], ...] = ()
    quality_gate: str | None = None

    def __post_init__(self) -> None:
        _instance("a point estimate task's meta", self.meta, TaskMeta)
        _member("estimand", self.estimand, Estimand)
        _optional_site_names("names", self.names)
        _backend("a point estimate task's backend", self.backend)
        _instance("budget", self.budget, ComputeBudget)
        object.__setattr__(
            self,
            "optimizer_options",
            canonical_value_options("optimizer_options", self.optimizer_options),
        )
        _optional_text("quality_gate", self.quality_gate)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class SimulationTask:
    """Generate forward from a parameter source. No conditioning, no gate.

    §0.4 gives the other four tasks a ``quality_gate`` and this one none, and
    the omission is deliberate rather than an oversight to be tidied up: a
    simulation makes no statistical claim to check -- it produced what the
    model says, and whether that is a good model is what the OTHER four ask.
    """

    meta: TaskMeta
    parameter_source: ParameterSource
    prediction_design: Fingerprint | None = None
    latent_sites: tuple[str, ...] = ()
    observed_sites: tuple[str, ...] = ()
    budget: ComputeBudget = _NO_BUDGET
    backend: str = "auto"
    backend_options: tuple[tuple[str, CanonicalValue], ...] = ()

    def __post_init__(self) -> None:
        _instance("a simulation task's meta", self.meta, TaskMeta)
        _instance("parameter_source", self.parameter_source, ParameterSource)
        _data_fingerprint("prediction_design", self.prediction_design)
        for name in ("latent_sites", "observed_sites"):
            site_names(name, getattr(self, name))
        if not self.latent_sites and not self.observed_sites:
            raise ValueError(
                "a simulation task names at least one latent or observed site "
                "to generate; naming neither asks for an empty result"
            )
        _instance("budget", self.budget, ComputeBudget)
        _backend("a simulation task's backend", self.backend)
        object.__setattr__(
            self,
            "backend_options",
            canonical_value_options("backend_options", self.backend_options),
        )


Task = PosteriorTask | EvidenceTask | PredictiveTask | PointEstimateTask | SimulationTask

#: kind -> class, and the single place the five are enumerated. Five
#: ``isinstance`` branches would be this table written so that a missing arm
#: reads as a fall-through rather than as a missing row.
TASK_CLASSES: dict[TaskKind, type] = {
    TaskKind.POSTERIOR: PosteriorTask,
    TaskKind.EVIDENCE: EvidenceTask,
    TaskKind.PREDICTIVE: PredictiveTask,
    TaskKind.POINT_ESTIMATE: PointEstimateTask,
    TaskKind.SIMULATION: SimulationTask,
}

_KIND_BY_CLASS: dict[type, TaskKind] = {cls: kind for kind, cls in TASK_CLASSES.items()}


def task_kind(task: Task) -> TaskKind:
    """Which of the five ``task`` is.

    By exact type, not ``isinstance``: a subclass of one task carrying extra
    state would be reported as the base kind and executed as it, which is the
    quiet half of §0 ruling 4.
    """
    kind = _KIND_BY_CLASS.get(type(task))
    if kind is None:
        raise TypeError(
            f"{type(task).__name__} is not one of the five tasks "
            f"({sorted(cls.__name__ for cls in TASK_CLASSES.values())})"
        )
    return kind


def task_fingerprint(task: Task) -> Fingerprint:
    """The §0.3 task-slot digest of ``task``.

    The kind is IN the payload, so two tasks that agree field for field but ask
    different questions are different. ``meta`` is not: its id is minted, its
    time is a clock and its label is for a human, and §0.3 keeps all three out
    of the task slot so that renaming a task cannot invalidate a cached result.
    """
    kind = task_kind(task)
    payload = {
        "kind": kind,
        "fields": tuple(
            (field.name, getattr(task, field.name))
            for field in sorted(dataclasses.fields(task), key=lambda f: f.name)
            if field.name != "meta"
        ),
    }
    return fingerprint(FingerprintKind.TASK, payload)
