"""The artifact envelope: what every artifact carries, and how it retires.

Plan §0.2's frozen envelope, plus the small value objects a run record is made
of. Three rulings shape nearly every line here.

**An artifact is data, and data does not change under you.** Everything is a
frozen dataclass; :class:``NamedArray`` copies its input into a C-contiguous
read-only array, so a caller who keeps a reference to the array they passed
cannot edit an artifact that has already been fingerprinted. The copy and the
read-only flag close different holes -- one from outside the artifact, one
from inside -- and neither substitutes for the other.

**Identity and content are different axes.** ``artifact_id`` and ``run_id`` are
UUID4s minted once and preserved across round trips; they say WHICH artifact.
What it was made OF is the fingerprint bundle, and that is what caching and
invalidation read. The version identity is ``(artifact_id, revision)``:
:func:``invalidate_meta`` APPENDS revision ``n + 1`` marked INVALIDATED, carrying
the changed input kinds and the time, and leaves revision ``n`` exactly as it
was. Provenance that can be rewritten is not provenance, and a verdict that
can be deleted is not a record of having judged.

**Refusals happen at construction, in the smallest type that can see the
problem.** A budget refuses a negative count, a timestamp refuses a naive or
offset time, a lifecycle refuses "invalidated, but nothing changed". The
alternative is a validator somewhere upstream that a second caller does not
know to run.

**Which exception a refusal raises is a contract, not a mood.** A ``TypeError``
means "that is the wrong kind of thing" -- a bare string where an enum member
belongs, a list where a tuple of refs belongs. A ``ValueError`` means "right kind
of thing, malformed or out of range" -- a negative count, a non-UTC timestamp,
a UUID that is not version 4. The split is worth stating because most fields
here can fail either way, and a caller sanitising input needs to know which
question it just got an answer to.

Two details that look like fussiness and are not. Times are UTC RFC 3339 and
are NORMALISED to the ``Z`` spelling: ``+00:00`` is the same instant, and two
spellings of one instant are two byte strings for one artifact. And the enums
are checked with ``isinstance`` rather than by string value, because a
:class:``enum.StrEnum`` member IS its value -- ``ApproximationClass.EXACT`` and
``TargetFidelity.EXACT`` are equal strings answering different questions, and a
string check would let them be swapped.

Layering: ``_codec ← identity ← base``. NumPy and the standard library, and
nothing of the Graph, the dispatch layer or JAX (§0 ruling 5).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import math
import uuid
from collections.abc import Iterable
from enum import StrEnum

import numpy as np

from ._codec import canonical_payload, register_artifact_type
from .identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    InvalidationPolicy,
    changed_fingerprints,
)

__all__ = [
    "SCHEMA_VERSION",
    "CanonicalScalar",
    "CanonicalValue",
    "canonical_scalar_options",
    "canonical_value_options",
    "utc_timestamp",
    "NamedArray",
    "RunWarning",
    "ErrorRecord",
    "ComputeBudget",
    "SeedRecord",
    "DeviceRecord",
    "BackendRef",
    "TerminationReason",
    "TerminationRecord",
    "TimingRecord",
    "ApproximationClass",
    "TargetFidelity",
    "ApproximationRecord",
    "ProducerRef",
    "ArtifactStatus",
    "ArtifactRef",
    "LifecycleRecord",
    "ArtifactMeta",
    "RunRecord",
    "new_artifact_meta",
    "invalidate_meta",
]

#: The envelope's version. Bumped when a field is added, removed or given a
#: new meaning -- a stored artifact has to be able to say which shape it was
#: written in, since the reader is a later version of this package by
#: construction.
SCHEMA_VERSION = 1

#: The open-set option values §0.2 and §0.4 admit. Deliberately narrow: an
#: option table is a place where "just put the object in" is tempting, and an
#: object in an option table is a runtime handle in an artifact.
CanonicalScalar = bool | int | float | str | None
CanonicalValue = CanonicalScalar | tuple["CanonicalValue", ...]

_SCALAR_TYPES = (bool, int, float, str)


# --------------------------------------------------------------- validation


def _text(label: str, value: object) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} is a string; got {value!r}")
    if not value:
        raise ValueError(f"{label} is a non-empty string; got {value!r}")
    return value


def _optional_text(label: str, value: object) -> str | None:
    if value is None:
        return None
    return _text(label, value)


def _count(label: str, value: object) -> int:
    # type(), not isinstance(): True is an int, and a budget of True draws
    # would otherwise run, quietly, as a budget of one.
    if type(value) is not int:
        raise TypeError(f"{label} is an int; got {value!r}")
    if value < 0:
        raise ValueError(f"{label} is a non-negative integer; got {value!r}")
    return value


def _optional_count(label: str, value: object) -> int | None:
    if value is None:
        return None
    return _count(label, value)


def _duration(label: str, value: object) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{label} is a number of seconds; got {value!r}")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"{label} is a finite, non-negative number of seconds; got {value!r}"
        )
    return float(value)


def _optional_duration(label: str, value: object) -> float | None:
    if value is None:
        return None
    return _duration(label, value)


def _member(label: str, value: object, enum: type[StrEnum]) -> object:
    if not isinstance(value, enum):
        raise TypeError(
            f"{label} is a {enum.__name__} member; got {value!r}. A StrEnum "
            "member equals its own value, so a bare string passes every "
            "comparison here and belongs to no enum at all"
        )
    return value


def _instance(label: str, value: object, kind: type) -> object:
    if not isinstance(value, kind):
        raise TypeError(f"{label} is a {kind.__name__}; got {value!r}")
    return value


def _uuid4(label: str, value: object) -> str:
    _text(label, value)
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError(f"{label} is a UUID4 string; got {value!r}") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(
            f"{label} is a UUID4 in canonical lowercase form; got {value!r}. "
            "Identity is minted, not derived -- a v1 UUID carries a MAC "
            "address and a clock, neither of which identifies an artifact"
        )
    return value


def _render(moment: _dt.datetime) -> str:
    return moment.astimezone(_dt.UTC).isoformat().replace("+00:00", "Z")


def utc_timestamp(moment: _dt.datetime | None = None) -> str:
    """``moment`` (default: now) as an RFC 3339 UTC string in the ``Z`` spelling."""
    return _render(moment if moment is not None else _dt.datetime.now(_dt.UTC))


def _timestamp(label: str, value: object) -> str:
    _text(label, value)
    try:
        moment = _dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} is not RFC 3339: {value!r}") from exc
    if moment.tzinfo is None or moment.utcoffset() != _dt.timedelta(0):
        raise ValueError(
            f"{label} must be UTC: {value!r} is either naive -- a different "
            "instant on every machine that reads it -- or carries the offset "
            "of whoever wrote it rather than the artifact's own time"
        )
    return _render(moment)


def _optional_timestamp(label: str, value: object) -> str | None:
    if value is None:
        return None
    return _timestamp(label, value)


def _tuple_of(label: str, value: object, kind: type) -> tuple:
    if not isinstance(value, tuple) or any(
        not isinstance(item, kind) for item in value
    ):
        raise TypeError(f"{label} is a tuple of {kind.__name__}; got {value!r}")
    return value


def _scalar(label: str, name: str, value: object) -> None:
    if value is not None and type(value) not in _SCALAR_TYPES:
        raise TypeError(
            f"{label}[{name!r}] is a canonical scalar -- bool, int, float, str "
            f"or None; got {value!r}"
        )


def _nested(label: str, name: str, value: object, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError(f"{label}[{name!r}] nests deeper than an option should")
    if type(value) is tuple:
        for item in value:
            _nested(label, name, item, depth + 1)
        return
    if value is not None and type(value) not in _SCALAR_TYPES:
        raise TypeError(
            f"{label}[{name!r}] is a canonical value -- a scalar or a tuple of "
            f"them; got {value!r}"
        )


def _options(label: str, pairs: object, check) -> tuple[tuple[str, object], ...]:
    """Sorted, key-unique ``(name, value)`` pairs, each value checked by ``check``.

    Sorting is normalisation -- the same options in two orders are one value,
    and a fingerprint taken over them must not depend on the order a caller
    typed keywords. A repeated key is refused rather than resolved: one of the
    two values would be dropped, and nothing would say which.
    """
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Iterable):
        raise TypeError(f"{label} is a tuple of (name, value) pairs; got {pairs!r}")
    collected: list[tuple[str, object]] = []
    seen: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{label} holds (name, value) pairs; got {pair!r}")
        name, value = pair
        _text(f"{label} keys", name)
        if name in seen:
            raise ValueError(
                f"{label} names {name!r} twice; one of the two values would be "
                "dropped without anything saying which"
            )
        seen.add(name)
        check(label, name, value)
        collected.append((name, value))
    return tuple(sorted(collected, key=lambda item: item[0]))


def canonical_scalar_options(
    label: str, pairs: object
) -> tuple[tuple[str, CanonicalScalar], ...]:
    """Sorted, key-unique options whose values are :data:``CanonicalScalar``."""
    return _options(label, pairs, _scalar)


def canonical_value_options(
    label: str, pairs: object
) -> tuple[tuple[str, CanonicalValue], ...]:
    """Sorted, key-unique options whose values are :data:``CanonicalValue``."""
    return _options(label, pairs, _nested)


# --------------------------------------------------------------- NamedArray


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class NamedArray:
    """The one public container for an array in an artifact (§0.5).

    The value is copied in and handed back read-only, and ``dims`` names one axis
    each. A repeated dim name is allowed on purpose: a covariance matrix is
    indexed by the same parameter axis twice, and refusing that would push
    every covariance into an unnamed array.

    Equality is over NAME, DIMS and BYTES rather than by ``==`` on the arrays.
    Element-wise comparison would make ``artifact == artifact`` raise for any
    array of more than one element, and byte equality is what a fingerprint
    already means -- which is also why two NaNs with one bit pattern compare
    equal here, while ``np.array_equal`` would call them different.
    """

    name: str
    value: np.ndarray
    dims: tuple[str, ...]

    def __post_init__(self) -> None:
        _text("a named array's name", self.name)
        raw = self.value
        if isinstance(raw, np.ndarray) and type(raw) is not np.ndarray:
            raise TypeError(
                f"{type(raw).__name__} is an ndarray subclass; the state that "
                "makes it one -- a masked array's mask, say -- would be "
                "dropped here without a word"
            )
        array = np.array(raw, order="C", copy=True)
        # The dtype whitelist is the codec's, asked rather than restated: an
        # empty array of this dtype costs nothing to encode and there is
        # exactly one list of admissible kinds in this package.
        canonical_payload(np.empty((0,), dtype=array.dtype))
        array.setflags(write=False)

        dims = self.dims
        if not isinstance(dims, tuple) or any(type(name) is not str for name in dims):
            raise TypeError(f"dims is a tuple of strings; got {dims!r}")
        if any(not name for name in dims):
            raise ValueError(f"dims are non-empty strings; got {dims!r}")
        if len(dims) != array.ndim:
            raise ValueError(
                f"dims {dims!r} names {len(dims)} axes, but the array has "
                f"{array.ndim}"
            )
        object.__setattr__(self, "value", array)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NamedArray):
            return NotImplemented
        return (
            self.name == other.name
            and self.dims == other.dims
            and self.value.dtype == other.value.dtype
            and self.value.shape == other.value.shape
            and self.value.tobytes() == other.value.tobytes()
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.dims,
                self.value.dtype.str,
                self.value.shape,
                self.value.tobytes(),
            )
        )


# ------------------------------------------------------- small value objects


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class RunWarning:
    """Something a run noticed and did not fail over.

    ``scope`` is a plain string here. Task 4 introduces the structured
    ``ScopeRef`` that the Refusal family uses, and borrowing it before it
    exists would make this module depend on a type nobody has designed yet.
    """

    code: str
    message: str
    scope: str | None = None

    def __post_init__(self) -> None:
        _text("a warning's code", self.code)
        _text("a warning's message", self.message)
        _optional_text("a warning's scope", self.scope)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ErrorRecord:
    """A serialisable summary of a failure -- never the exception object.

    ``traceback_ref`` is an opaque external reference (§0.6): a traceback is not
    one of the three :class:``ArtifactKind`` the invalidation matrix reasons
    about, so it is stored beside the artifacts rather than as one.
    """

    code: str
    message: str
    exception_type: str
    traceback_ref: str | None = None

    def __post_init__(self) -> None:
        _text("an error's code", self.code)
        _text("an error's message", self.message)
        _text("an error's exception_type", self.exception_type)
        _optional_text("an error's traceback_ref", self.traceback_ref)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ComputeBudget:
    """What a run was allowed to spend. ``None`` where a count does not apply.

    ``None`` and ``0`` are different answers -- "this task has no such knob"
    against "you may take none" -- so the field is kept and left empty rather
    than filled with a stand-in.
    """

    draws: int | None = None
    warmup: int | None = None
    chains: int | None = None
    max_iterations: int | None = None
    max_wall_clock_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("draws", "warmup", "chains", "max_iterations"):
            _optional_count(name, getattr(self, name))
        object.__setattr__(
            self,
            "max_wall_clock_seconds",
            _optional_duration("max_wall_clock_seconds", self.max_wall_clock_seconds),
        )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class SeedRecord:
    """The entropy a run started from, and the key algorithm it was fed to."""

    seed: int
    key_algorithm: str

    def __post_init__(self) -> None:
        _count("seed", self.seed)
        _text("key_algorithm", self.key_algorithm)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class DeviceRecord:
    """One device a run used, described the way a backend describes it."""

    platform: str
    device_kind: str
    device_id: int

    def __post_init__(self) -> None:
        _text("platform", self.platform)
        _text("device_kind", self.device_kind)
        _count("device_id", self.device_id)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class BackendRef:
    """The backend that RAN, which is never the policy value ``"auto"``.

    A task may ask for ``"auto"`` (§0.4); a run record answering ``"auto"`` would
    be provenance that forgot which backend produced the numbers.
    """

    name: str
    version: str | None = None

    def __post_init__(self) -> None:
        _text("a backend ref's name", self.name)
        if self.name == "auto":
            raise ValueError(
                "'auto' is a request, not a backend: a run record names the "
                "backend that actually ran"
            )
        _optional_text("a backend ref's version", self.version)


@register_artifact_type
class TerminationReason(StrEnum):
    """Why a run stopped, which is not the same question as whether it worked."""

    COMPLETED = "completed"
    CONVERGED = "converged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TOLERANCE_UNMET = "tolerance_unmet"
    DIVERGED = "diverged"
    INTERRUPTED = "interrupted"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class TerminationRecord:
    reason: TerminationReason
    iterations: int | None = None
    message: str = ""

    def __post_init__(self) -> None:
        _member("a termination reason", self.reason, TerminationReason)
        _optional_count("iterations", self.iterations)
        if type(self.message) is not str:
            raise TypeError(f"a termination message is a string; got {self.message!r}")


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class TimingRecord:
    """When a run ran and how long it took. Wall clock only -- §0.3 keeps
    timing out of every fingerprint, so nothing here can change an identity."""

    started_at: str
    finished_at: str
    wall_clock_seconds: float
    compile_seconds: float | None = None

    def __post_init__(self) -> None:
        started = _timestamp("started_at", self.started_at)
        finished = _timestamp("finished_at", self.finished_at)
        object.__setattr__(self, "started_at", started)
        object.__setattr__(self, "finished_at", finished)
        object.__setattr__(
            self,
            "wall_clock_seconds",
            _duration("wall_clock_seconds", self.wall_clock_seconds),
        )
        object.__setattr__(
            self,
            "compile_seconds",
            _optional_duration("compile_seconds", self.compile_seconds),
        )
        if _dt.datetime.fromisoformat(finished) < _dt.datetime.fromisoformat(started):
            raise ValueError(
                f"a run cannot finish before it started: {started} -> {finished}"
            )


@register_artifact_type
class ApproximationClass(StrEnum):
    """How the representation was PRODUCED (§0.2)."""

    EXACT = "exact"
    CERTIFIED_DETERMINISTIC = "certified_deterministic"
    MONTE_CARLO = "monte_carlo"
    HEURISTIC = "heuristic"


@register_artifact_type
class TargetFidelity(StrEnum):
    """Whether the TARGET is the intended one, or an approximation of it."""

    EXACT = "exact"
    APPROXIMATE = "approximate"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ApproximationRecord:
    """Two orthogonal answers, kept apart (§0.2).

    iid exact-linear draws are ``MONTE_CARLO`` and ``EXACT``: sampled, but from the
    posterior itself. NUTS is ``MONTE_CARLO`` and, with no target approximation,
    also ``EXACT``; an amortized posterior is ``HEURISTIC``/``APPROXIMATE``. One
    enum would have to call the first case something, and every available
    something is a lie about one of the two axes.
    """

    representation_class: ApproximationClass
    target_fidelity: TargetFidelity
    details: tuple[tuple[str, CanonicalScalar], ...] = ()

    def __post_init__(self) -> None:
        _member("representation_class", self.representation_class, ApproximationClass)
        _member("target_fidelity", self.target_fidelity, TargetFidelity)
        object.__setattr__(
            self, "details", canonical_scalar_options("details", self.details)
        )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ProducerRef:
    """The package and version that wrote an artifact."""

    package: str
    version: str

    def __post_init__(self) -> None:
        _text("a producer's package", self.package)
        _text("a producer's version", self.version)


@register_artifact_type
class ArtifactStatus(StrEnum):
    CURRENT = "current"
    INVALIDATED = "invalidated"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A reference to one VERSION of one artifact.

    The revision and the kind travel with the id because ``(artifact_id,
    revision)`` is the version identity: a reference carrying only an id would
    be silently satisfied by a later, invalidated revision of the same
    artifact, which is exactly the impersonation §0.2 rules out.
    """

    artifact_id: str
    revision: int
    artifact_type: ArtifactKind

    def __post_init__(self) -> None:
        _uuid4("artifact_id", self.artifact_id)
        _count("revision", self.revision)
        _member("an artifact ref's artifact_type", self.artifact_type, ArtifactKind)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class LifecycleRecord:
    """Current, or invalidated with the reason and the time recorded."""

    status: ArtifactStatus
    invalidated_at: str | None = None
    changed_inputs: tuple[FingerprintKind, ...] = ()

    def __post_init__(self) -> None:
        _member("an artifact status", self.status, ArtifactStatus)

        kinds: list[FingerprintKind] = []
        seen: set[FingerprintKind] = set()
        for kind in self.changed_inputs:
            if not isinstance(kind, FingerprintKind):
                raise TypeError(
                    f"changed_inputs holds FingerprintKind members; got {kind!r}"
                )
            if kind in seen:
                raise ValueError(f"changed_inputs names {kind.value!r} twice")
            seen.add(kind)
            kinds.append(kind)
        object.__setattr__(self, "changed_inputs", tuple(sorted(kinds)))

        if self.status is ArtifactStatus.CURRENT:
            if self.invalidated_at is not None or self.changed_inputs:
                raise ValueError(
                    "a current artifact has no invalidated_at and no changed "
                    "inputs; it is current because nothing it was made from "
                    "has moved"
                )
            return
        if self.invalidated_at is None:
            raise ValueError("an invalidated artifact records invalidated_at")
        object.__setattr__(
            self, "invalidated_at", _timestamp("invalidated_at", self.invalidated_at)
        )
        if not self.changed_inputs:
            raise ValueError(
                "an invalidated artifact records the changed_inputs that "
                "retired it; 'invalid for no stated reason' is not a record"
            )


# ------------------------------------------------------------- the envelope


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactMeta:
    """The §0.2 envelope every artifact carries."""

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

    def __post_init__(self) -> None:
        _member("an artifact_type", self.artifact_type, ArtifactKind)
        if type(self.schema_version) is not int:
            raise TypeError(f"schema_version is an int; got {self.schema_version!r}")
        if self.schema_version < 1:
            raise ValueError(
                f"schema_version is a positive integer; got {self.schema_version!r}"
            )
        _uuid4("artifact_id", self.artifact_id)
        _count("revision", self.revision)
        object.__setattr__(self, "created_at", _timestamp("created_at", self.created_at))
        _instance("producer", self.producer, ProducerRef)
        _tuple_of("parent_refs", self.parent_refs, ArtifactRef)
        seen: set[str] = set()
        for ref in self.parent_refs:
            if ref.artifact_id in seen:
                raise ValueError(
                    f"parent {ref.artifact_id} is listed twice; two revisions "
                    "of one parent leave no answer to which version this was "
                    "made from"
                )
            seen.add(ref.artifact_id)
        _instance("fingerprints", self.fingerprints, FingerprintBundle)
        _instance("lifecycle", self.lifecycle, LifecycleRecord)
        _tuple_of("warnings", self.warnings, RunWarning)
        if type(self.summary) is not str:
            raise TypeError(f"summary is a string; got {self.summary!r}")
        if self.lifecycle.status is ArtifactStatus.INVALIDATED and self.revision < 1:
            raise ValueError(
                "revision 0 cannot be invalidated: invalidation appends "
                "revision n + 1 and leaves the revision it retired alone"
            )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class RunRecord:
    """What one execution of a plan actually did (§0.2)."""

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

    def __post_init__(self) -> None:
        _uuid4("run_id", self.run_id)
        _instance("plan_ref", self.plan_ref, ArtifactRef)
        if self.plan_ref.artifact_type is not ArtifactKind.PLAN:
            raise ValueError(
                f"plan_ref points at a {self.plan_ref.artifact_type.value}; a "
                "run executes a plan"
            )
        _instance("fingerprints", self.fingerprints, FingerprintBundle)
        if self.seed is not None:
            _instance("seed", self.seed, SeedRecord)
        _text("dtype", self.dtype)
        _tuple_of("devices", self.devices, DeviceRecord)
        if len(set(self.devices)) != len(self.devices):
            raise ValueError(f"devices names one device twice; got {self.devices!r}")
        object.__setattr__(
            self, "jax_config", canonical_scalar_options("jax_config", self.jax_config)
        )
        for name, kind in (
            ("backend", BackendRef),
            ("budget", ComputeBudget),
            ("termination", TerminationRecord),
            ("timing", TimingRecord),
            ("approximation", ApproximationRecord),
        ):
            _instance(name, getattr(self, name), kind)
        _tuple_of("warnings", self.warnings, RunWarning)


def new_artifact_meta(
    *,
    artifact_type: ArtifactKind,
    fingerprints: FingerprintBundle,
    producer: ProducerRef,
    summary: str = "",
    parent_refs: tuple[ArtifactRef, ...] = (),
    warnings: tuple[RunWarning, ...] = (),
    schema_version: int = SCHEMA_VERSION,
    artifact_id: str | None = None,
    created_at: str | None = None,
) -> ArtifactMeta:
    """A CURRENT envelope at revision 0, with a freshly minted UUID4.

    ``artifact_id`` and ``created_at`` are arguments so that a caller restoring an
    artifact keeps its identity; left out, they are minted here. They are not
    derived from the content: two artifacts with identical fingerprints are
    still two artifacts, with separate lineage and separate verdicts.
    """
    return ArtifactMeta(
        artifact_type=artifact_type,
        schema_version=schema_version,
        artifact_id=artifact_id if artifact_id is not None else str(uuid.uuid4()),
        revision=0,
        created_at=created_at if created_at is not None else utc_timestamp(),
        producer=producer,
        parent_refs=parent_refs,
        fingerprints=fingerprints,
        lifecycle=LifecycleRecord(status=ArtifactStatus.CURRENT),
        warnings=warnings,
        summary=summary,
    )


def invalidate_meta(
    meta: ArtifactMeta,
    *,
    before: FingerprintBundle,
    after: FingerprintBundle,
    policy: InvalidationPolicy,
    at: str | None = None,
) -> ArtifactMeta:
    """The INVALIDATED copy of ``meta`` at ``revision + 1`` (§0.2, §0.3).

    Same ``artifact_id``, same fingerprints -- the retired revision records what
    it was MADE from, not what replaced it -- with the changed input kinds and
    the time written into its lifecycle. ``meta`` itself is untouched, because
    the point of a revision is that the earlier one survives.

    Refused, loudly, in four cases: ``before`` is not this artifact's own bundle
    (so the comparison would be about some other artifact), nothing changed,
    the policy says this artifact kind does not depend on what changed, and the
    artifact was already invalidated. Each of these is a caller error that
    would otherwise produce a plausible-looking revision recording a judgement
    nobody made.
    """
    _instance("invalidate_meta's first argument", meta, ArtifactMeta)
    _instance("policy", policy, InvalidationPolicy)
    if meta.lifecycle.status is ArtifactStatus.INVALIDATED:
        raise ValueError(
            f"artifact {meta.artifact_id} revision {meta.revision} is already "
            "invalidated; invalidation is not a counter"
        )
    if meta.fingerprints != before:
        raise ValueError(
            "before must be this artifact's own fingerprint bundle; comparing "
            "it against somebody else's would name inputs it never had"
        )
    changed = changed_fingerprints(before, after)
    if not changed:
        raise ValueError(
            "nothing changed between the two bundles, so there is nothing to "
            "invalidate"
        )
    if not policy.affected(meta.artifact_type, changed):
        raise ValueError(
            f"a change to {sorted(kind.value for kind in changed)} does not "
            f"invalidate a {meta.artifact_type.value} under this policy; that "
            "artifact is still reusable, and marking it otherwise would throw "
            "away work the matrix says is good"
        )
    return dataclasses.replace(
        meta,
        revision=meta.revision + 1,
        lifecycle=LifecycleRecord(
            status=ArtifactStatus.INVALIDATED,
            invalidated_at=at if at is not None else utc_timestamp(),
            changed_inputs=tuple(sorted(changed)),
        ),
    )
