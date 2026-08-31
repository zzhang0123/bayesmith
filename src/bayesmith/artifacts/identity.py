"""Model references, fingerprints, and what a changed input invalidates.

Identity here is deliberately of two unrelated kinds, and keeping them apart is
most of the module:

**A fingerprint is semantic.** It is a SHA-256 over the canonical bytes of a
payload (:mod:``bayesmith.artifacts._codec``), so two runs that saw the same model,
the same data and the same task carry the same digest on different machines,
in different processes, in either order of a mapping's keys. Fingerprints are
what caching, reuse and invalidation are decided by.

**An artifact id is not.** ``artifact_id`` and ``run_id`` are UUID4s minted once
(:mod:``bayesmith.artifacts.base``) -- they say *which* artifact, never *what it
was made of*. Two artifacts with one fingerprint bundle are interchangeable in
content and still separate artifacts with separate lineage.

Plan §0.3 fixes seven fingerprint slots, and the boundary of each is a ruling
rather than a detail. The two that pay for the rest: the **data** slot holds
the bytes of every Const, observation and mask, so a single flipped bit is a
different data set; the **task** slot holds statistical semantics and budget
and holds no progress bar, no print width and not the task's own id, so
renaming a label invalidates nothing. What a slot contains is the caller's
payload -- :func:``fingerprint`` hashes what it is handed -- and the slot table in
§0.3 is the contract those callers are written against.

``ModelRef`` exists because a model is a callable and a callable has no
canonical form. Its ``repr`` carries a memory address, which would make one
model two in a single process and a third after a restart, so
:func:``model_ref_from_callable`` digests ``inspect.getsource`` or refuses --
plan §0.3, and §0 ruling 4: an artifact is data, not a runtime object dump.

The invalidation matrix (§0.3) is a table, not an ``if`` chain:
:class:``InvalidationPolicy`` holds one row per :class:``ArtifactKind`` naming the
fingerprint kinds that artifact kind is sensitive to. A compilation change
spares the Plan; an evaluation threshold spares the Result and re-runs only
the report; a display option is in no slot at all, so it changes nothing and
reuses everything.

Refusals follow the same contract as :mod:``bayesmith.artifacts.base``: a
``TypeError`` is "that is the wrong kind of thing" (a bare string where an enum
member belongs), a ``ValueError`` is "right kind of thing, malformed" (a digest
that is not lowercase SHA-256 hex). ``ArtifactCodecError`` -- a ``ValueError`` --
is what an uncanonical payload raises, and it arrives from the codec unwrapped
so that the offending type is still named.

Layering: ``_codec`` only. Nothing here may import ``base``, the Graph, or JAX --
``_codec ← identity ← base`` is the ladder, and the envelope in ``base`` names
these types rather than the other way round.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from collections.abc import Callable, Iterable
from enum import StrEnum

from ._codec import canonical_dumps, canonical_payload, register_artifact_type

__all__ = [
    "FINGERPRINT_ALGORITHM",
    "FingerprintKind",
    "ArtifactKind",
    "Fingerprint",
    "FingerprintBundle",
    "ModelRef",
    "fingerprint",
    "model_ref_from_callable",
    "changed_fingerprints",
    "InvalidationPolicy",
]

#: The digest this package takes, named in every fingerprint it produces. The
#: version suffix is not decoration: the digest is over CANONICAL bytes, so a
#: change to the codec's wire format changes every digest without changing the
#: hash function, and a stored artifact has to be able to say which pairing it
#: was made under.
FINGERPRINT_ALGORITHM = "sha256-v1"

_HEX = frozenset("0123456789abcdef")
_DIGEST_LENGTH = 64


@register_artifact_type
class FingerprintKind(StrEnum):
    """The seven slots of plan §0.3, one per independently changeable input.

    Named after the slots of :class:``FingerprintBundle`` and checked against them,
    so a slot cannot come to hold a digest of some other kind -- a ``data`` slot
    holding a task digest would answer "the data changed" for a renamed
    solver, and answer it consistently enough to be believed.
    """

    MODEL_SOURCE = "model_source"
    GRAPH_STRUCTURE = "graph_structure"
    DATA = "data"
    TASK = "task"
    COMPILATION = "compilation"
    EVALUATION = "evaluation"
    ENVIRONMENT = "environment"


@register_artifact_type
class ArtifactKind(StrEnum):
    """The rows of the §0.3 invalidation matrix: what invalidation reasons about.

    Three members, because the matrix has three columns and each is a distinct
    sensitivity: a Plan is made before compilation, a Result is made by it, and
    an EvaluationReport is a judgement about a Result. This is the INVALIDATION
    taxonomy and not a catalogue of artifacts -- which of the five Results a
    reference points at is ``ResultKind``'s business (Task 3), and asking this enum
    to carry that too would put five identical rows in the matrix.
    """

    PLAN = "plan"
    RESULT = "result"
    EVALUATION_REPORT = "evaluation_report"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class Fingerprint:
    """One slot's digest, with the kind it belongs to and the algorithm used."""

    kind: FingerprintKind
    algorithm: str
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, FingerprintKind):
            raise TypeError(
                f"{self.kind!r} is not a FingerprintKind member; a fingerprint "
                "kind spelled as a bare string compares equal to the member "
                "and hashes differently, so it would pass here and match "
                "nothing later"
            )
        if self.algorithm != FINGERPRINT_ALGORITHM:
            raise ValueError(
                f"{self.algorithm!r} is not this package's fingerprint "
                f"algorithm ({FINGERPRINT_ALGORITHM!r}); a bundle mixing two "
                "algorithms could not be compared slot by slot"
            )
        if type(self.digest) is not str:
            raise TypeError(f"a digest is a string; got {self.digest!r}")
        if len(self.digest) != _DIGEST_LENGTH or not _HEX.issuperset(self.digest):
            raise ValueError(
                f"{self.digest!r} is not a digest: SHA-256 in lowercase hex is "
                f"{_DIGEST_LENGTH} characters, and one spelling per value is "
                "the entire point"
            )


def fingerprint(kind: FingerprintKind, payload: object) -> Fingerprint:
    """The fingerprint of ``payload`` in slot ``kind``.

    What goes into ``payload`` is the §0.3 boundary for that slot, and it is the
    caller's to decide: this function hashes what it is handed. A payload the
    codec cannot encode -- a callable, an object array, an unregistered class
    -- raises :class:``ArtifactCodecError`` here rather than being rendered
    best-effort, because a best-effort digest of a callable is its address.
    """
    return Fingerprint(
        kind=kind,
        algorithm=FINGERPRINT_ALGORITHM,
        digest=hashlib.sha256(canonical_dumps(payload)).hexdigest(),
    )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class FingerprintBundle:
    """The seven slots of §0.3, in the order the plan lists them.

    The first four always exist: every artifact was made from a model, a graph,
    data and a task. The last three are ``None`` when they do not apply -- a Plan
    has no compilation digest before it is compiled, and nothing but a report
    has an evaluation one. ``None`` and a digest are different values, so a slot
    appearing or vanishing is itself a change.
    """

    model_source: Fingerprint
    graph_structure: Fingerprint
    data: Fingerprint
    task: Fingerprint
    compilation: Fingerprint | None = None
    evaluation: Fingerprint | None = None
    environment: Fingerprint | None = None

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if not isinstance(value, Fingerprint):
                raise TypeError(
                    f"FingerprintBundle.{field.name} holds {value!r}, which is "
                    "not a Fingerprint"
                )
            if value.kind.value != field.name:
                raise ValueError(
                    f"FingerprintBundle.{field.name} holds a fingerprint of "
                    f"kind {value.kind.value!r}; a slot answering for another "
                    "slot's input would name the wrong thing as changed"
                )

    def slots(self) -> tuple[tuple[FingerprintKind, Fingerprint | None], ...]:
        """The bundle as (kind, fingerprint-or-None) pairs, in slot order."""
        return tuple(
            (FingerprintKind(field.name), getattr(self, field.name))
            for field in dataclasses.fields(self)
        )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ModelRef:
    """A stable reference to the model a graph was built from.

    Never the callable itself: §0 ruling 4. ``identifier`` is what a human calls
    the model, ``source_digest`` pins its text, and ``package``/``package_version``
    pin a released model whose source is not at hand. At least one of those
    two pins is required, because a reference with only a name is a reference
    to whatever that name means today.

    ``build_arguments`` carries the state that would otherwise hide in a
    closure -- §0.3 does not promise to discover it, so a builder's arguments
    are part of the reference or they are lost. They are canonical values
    (arrays included, since a knot vector is an argument), key-unique, and
    sorted, so two callers who passed the same arguments in different order
    hold the same reference.
    """

    identifier: str
    source_digest: str | None = None
    package: str | None = None
    package_version: str | None = None
    build_arguments: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if type(self.identifier) is not str:
            raise TypeError(f"a model identifier is a string; got {self.identifier!r}")
        if not self.identifier:
            raise ValueError("a model needs a non-empty identifier")
        for name in ("source_digest", "package", "package_version"):
            value = getattr(self, name)
            if value is None:
                continue
            if type(value) is not str:
                raise TypeError(f"{name} is a string or None; got {value!r}")
            if not value:
                raise ValueError(f"{name} is non-empty when given")
        if self.source_digest is None and not (self.package and self.package_version):
            raise ValueError(
                "a model reference must pin its source: give source_digest, or "
                "a package and package_version that identify a released model"
            )
        object.__setattr__(
            self,
            "build_arguments",
            _canonical_pairs("build_arguments", self.build_arguments),
        )


def _canonical_pairs(
    label: str, pairs: object
) -> tuple[tuple[str, object], ...]:
    """Sorted, key-unique ``(name, canonical value)`` pairs.

    Sorting is normalisation: the same arguments in two orders are one value,
    and a fingerprint that disagreed would depend on the order a caller typed
    keywords in. A repeated key is not normalised away -- one of the two values
    would be silently discarded, and which one is not something a caller can be
    expected to reason about.
    """
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Iterable):
        raise TypeError(f"{label} is a tuple of (name, value) pairs; got {pairs!r}")
    collected: list[tuple[str, object]] = []
    seen: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"{label} holds (name, value) pairs; got {pair!r}")
        name, value = pair
        if type(name) is not str:
            raise TypeError(f"{label} keys are strings; got {name!r}")
        if not name:
            raise ValueError(f"{label} keys are non-empty")
        if name in seen:
            raise ValueError(
                f"{label} names {name!r} twice; one of the two values would be "
                "dropped without anything saying which"
            )
        seen.add(name)
        # Raises ArtifactCodecError, and the offending type is in the message.
        canonical_payload(value)
        collected.append((name, value))
    return tuple(sorted(collected, key=lambda item: item[0]))


def model_ref_from_callable(
    fn: Callable[..., object],
    *,
    identifier: str,
    source_digest: str | None = None,
    package: str | None = None,
    package_version: str | None = None,
    build_arguments: tuple[tuple[str, object], ...] = (),
) -> ModelRef:
    """A :class:``ModelRef`` for ``fn``, digesting its source or refusing.

    §0.3: an automatic source digest exists only when ``inspect.getsource``
    returns stable text. When it does not -- a function built by ``exec``, a
    builtin, a callable defined in a REPL -- the caller must supply
    ``source_digest``. There is no fallback, and the missing fallback is the
    feature: ``repr(fn)`` carries a memory address, so a digest that degraded
    to it would report a changed model on every restart and an unchanged one
    for a genuine edit that happened to reuse an address.

    ``package`` defaults to the top-level package of ``fn.__module__``, which is
    read off the function object rather than imported.
    """
    if not callable(fn):
        raise TypeError(f"a model reference is taken from a callable; got {fn!r}")

    digest = source_digest
    if digest is None:
        try:
            source = inspect.getsource(fn)
        except (OSError, TypeError) as exc:
            raise ValueError(
                f"the source of {getattr(fn, '__qualname__', fn)!r} is not "
                "available, so no source digest can be taken; pass "
                "source_digest explicitly -- repr() is not a fallback, it "
                "carries a memory address"
            ) from exc
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()

    if package is None:
        module = getattr(fn, "__module__", None)
        if isinstance(module, str) and module:
            package = module.partition(".")[0]

    return ModelRef(
        identifier=identifier,
        source_digest=digest,
        package=package,
        package_version=package_version,
        build_arguments=build_arguments,
    )


def changed_fingerprints(
    before: FingerprintBundle, after: FingerprintBundle
) -> frozenset[FingerprintKind]:
    """The slots whose fingerprint differs between two bundles.

    Slot by slot, including the optional ones: a compilation digest that
    appears where there was ``None`` is a change, and so is one that vanishes.
    The result is a frozenset because invalidation is a question about a SET of
    changed inputs -- the order they were discovered in is not part of it.
    """
    for label, value in (("before", before), ("after", after)):
        if not isinstance(value, FingerprintBundle):
            raise TypeError(f"{label} is not a fingerprint bundle; got {value!r}")
    return frozenset(
        kind
        for (kind, left), (_, right) in zip(before.slots(), after.slots(), strict=True)
        if left != right
    )


@dataclasses.dataclass(frozen=True, slots=True)
class InvalidationPolicy:
    """The §0.3 matrix: which fingerprint kinds each artifact kind depends on.

    A table, and a table for a reason. The rule set is small enough to write as
    branches and exactly the kind of thing that grows an unwritten case when it
    is: "compilation invalidates a Result but not a Plan" and "an evaluation
    threshold invalidates the report but not the Result" are not related
    statements, and nothing but a table keeps them from being merged by
    somebody simplifying.

    Not registered with the codec: a policy is code that reads artifacts, not
    an artifact. What an invalidated artifact records is which inputs changed
    (``LifecycleRecord.changed_inputs``), which stays true whoever judged it.
    """

    sensitivities: tuple[tuple[ArtifactKind, tuple[FingerprintKind, ...]], ...]

    def __post_init__(self) -> None:
        rows: list[tuple[ArtifactKind, tuple[FingerprintKind, ...]]] = []
        seen: set[ArtifactKind] = set()
        for row in self.sensitivities:
            if not isinstance(row, tuple) or len(row) != 2:
                raise TypeError(f"a policy row is (artifact kind, kinds); got {row!r}")
            artifact_type, kinds = row
            if not isinstance(artifact_type, ArtifactKind):
                raise TypeError(
                    f"{artifact_type!r} is not an ArtifactKind member; an "
                    "artifact kind spelled as a bare string hashes differently "
                    "and would match no row"
                )
            if artifact_type in seen:
                raise ValueError(
                    f"the policy names {artifact_type.value!r} twice; the "
                    "second row would silently win"
                )
            seen.add(artifact_type)
            rows.append((artifact_type, _kind_set(kinds)))
        if seen != set(ArtifactKind):
            raise ValueError(
                "a policy needs a row for every artifact kind; missing "
                f"{sorted(kind.value for kind in set(ArtifactKind) - seen)}"
            )
        object.__setattr__(self, "sensitivities", tuple(sorted(rows)))

    @classmethod
    def default(cls) -> InvalidationPolicy:
        """The §0.3 matrix as shipped."""
        return _DEFAULT_POLICY

    def affected(
        self, artifact_type: ArtifactKind, changed: Iterable[FingerprintKind]
    ) -> bool:
        """Does an artifact of this kind stop being current, given ``changed``?

        ``changed`` is any iterable of :class:``FingerprintKind`` -- normally the
        frozenset :func:``changed_fingerprints`` returned. An empty one answers
        ``False`` for every artifact kind, which is the matrix's display-option
        row: a display option is in no slot, so nothing moved and everything
        is reusable.
        """
        if not isinstance(artifact_type, ArtifactKind):
            raise TypeError(
                f"{artifact_type!r} is not an ArtifactKind member; an artifact "
                "kind spelled as a bare string hashes differently and would "
                "match no row"
            )
        sensitive = dict(self.sensitivities)[artifact_type]
        # Materialised ONCE: `changed` may be an iterator, and reading it twice
        # would answer the second question against an exhausted one -- which
        # reads as 'nothing changed', the safest-looking wrong answer here.
        kinds = set(_kind_set(changed))
        return bool(kinds & set(sensitive))


def _kind_set(kinds: Iterable[FingerprintKind]) -> tuple[FingerprintKind, ...]:
    """Sorted, unique fingerprint kinds, refusing anything that is not one."""
    if isinstance(kinds, (str, bytes)) or not isinstance(kinds, Iterable):
        raise TypeError(f"expected fingerprint kinds; got {kinds!r}")
    collected: set[FingerprintKind] = set()
    for kind in kinds:
        if not isinstance(kind, FingerprintKind):
            raise TypeError(
                f"{kind!r} is not a FingerprintKind member; a fingerprint kind "
                "spelled as a bare string hashes differently, so a set of them "
                "would intersect nothing and quietly answer 'unaffected'"
            )
        collected.add(kind)
    return tuple(sorted(collected))


_MODEL_AND_INPUTS = (
    FingerprintKind.MODEL_SOURCE,
    FingerprintKind.GRAPH_STRUCTURE,
    FingerprintKind.DATA,
    FingerprintKind.TASK,
)

#: §0.3, read row by row. ENVIRONMENT is in no row on purpose: a backend patch
#: leaves existing artifacts readable and gives the NEXT run new provenance,
#: and a report follows the identity of the Result it judged rather than the
#: interpreter that produced it.
_DEFAULT_POLICY = InvalidationPolicy(
    sensitivities=(
        (ArtifactKind.PLAN, _MODEL_AND_INPUTS),
        (ArtifactKind.RESULT, (*_MODEL_AND_INPUTS, FingerprintKind.COMPILATION)),
        (
            ArtifactKind.EVALUATION_REPORT,
            (
                *_MODEL_AND_INPUTS,
                FingerprintKind.COMPILATION,
                FingerprintKind.EVALUATION,
            ),
        ),
    )
)
