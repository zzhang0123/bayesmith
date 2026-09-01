"""Structured refusal: why a route does not apply, and what to do instead.

§0 ruling 3 in code. A refusal is a JUDGEMENT -- the premise this route needs
is false -- and it is carried as data: a stable ``failed_premise`` code, a
non-empty tuple of :class:``Finding`` with the numbers that were measured, a
:class:``ScopeRef`` saying what it is about, and a non-empty tuple of
:class:``Remedy`` saying where to go. The human messages explain; nothing reads
them.

**The field is called ``grounds`` and never ``evidence``.** Frozen by §0 ruling
3, and not a matter of taste: this package computes marginal likelihoods, and
"evidence" is already the name of a number here. A refusal field with that
name would collide with the quantity in the next module along, in a codebase
whose sibling repository has to translate one refusal into the other.

**A refusal is not an exception wearing a schema.** A malformed Graph, a
programming error, an exhausted device: those stay exceptions. What becomes a
Refusal is the statistical case -- the likelihood is not Gaussian, the operator
is not affine, the condition number is past what the tolerance allows.

**Both lists are non-empty.** No grounds is a verdict with nothing behind it;
no remedies is a dead end. Each is refused where the caller who could still
supply them is standing.

Layering: ``base`` and ``tasks`` (§0.1). A refusal holds the whole Task it
refused, because "which question was this an answer to" is the first thing a
consumer asks and a task id would send them looking.
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum

from ._codec import register_artifact_type
from .base import (
    ArtifactMeta,
    ArtifactRef,
    _instance,
    _member,
    _nested,
    _text,
    _tuple_of,
    canonical_value_options,
)
from .identity import ArtifactKind
from .tasks import Task, TaskKind, task_kind

__all__ = [
    "ScopeKind",
    "ScopeRef",
    "Finding",
    "Remedy",
    "Conservatism",
    "FallbackOption",
    "Refusal",
    "CAPABILITY_UNAVAILABLE_R1",
    "PREMISES",
]

#: The premise a task names when this package can hold the QUESTION and not
#: yet answer it -- R1 freezes five tasks and executes two. Spelled once, here,
#: because it is written by the compiler that refuses and read by every
#: consumer that branches on the refusal, and two spellings of one code is two
#: branches that agree until somebody improves one of them.
CAPABILITY_UNAVAILABLE_R1: str = "capability_unavailable_r1"

#: This package's own premise vocabulary: the codes a
#: :class:`~bayesmith.artifacts.reports.InferencePlanRecord` lists in
#: ``premises`` and a :class:`Refusal` names in ``failed_premise``. One
#: vocabulary read in both directions, which is what ``reports``' docstring
#: claims and what ``tests/dispatch/test_task_protocol.py`` holds it to.
#:
#: **Not enforced at construction, deliberately.** A ``Refusal`` is the type two
#: repositories agree about (§0 ruling 3), and the sibling's translation names
#: premises this package has never heard of; refusing them here would make the
#: shared type unusable on the side that does not own this list. What IS
#: enforced is that a code is a code -- :func:`_code` refuses prose -- and what
#: is checked is that bayesmith's own refusals and plans draw from this set.
#:
#: Not every member appears in both directions: a capability gap is never a
#: premise a plan RELIES on, it is the reason no plan exists.
PREMISES: frozenset[str] = frozenset(
    {
        CAPABILITY_UNAVAILABLE_R1,
        # the compile-time checks a plan rests on
        "backend_supported",
        "model_source_identified",
        "task_options_recognised",
        # what an exact route needs of the graph
        "gaussian_likelihood",
        "affine_prediction",
        "log_linear_route",
        # what a point estimate needs of the plan and of the arithmetic
        "whole_graph_exact_solve",
        "local_mode_certified",
        "graph_has_latents",
    }
)


def _code(label: str, value: object) -> str:
    """A stable machine code: non-empty, and with no whitespace in it.

    The whitespace rule is the whole check, and it is worth the line. A code is
    what a consumer branches on, so the moment one reads "not a gaussian" it
    has become a sentence -- and a sentence gets reworded by whoever improves
    the prose, silently changing what the branch matches.
    """
    _text(label, value)
    if any(character.isspace() for character in value):
        raise ValueError(
            f"{label} is a machine code, not prose; {value!r} contains "
            "whitespace, and prose gets reworded by people who are not "
            "thinking about the consumer branching on it"
        )
    return value


def _code_tuple(label: str, value: object) -> tuple[str, ...]:
    """Unique codes in the order they were declared.

    Not sorted: where these are used the order is meaningful -- a route
    preference, a schedule, the sequence a reason was reached in. Duplicates
    are refused because a repeated code makes one reason look like two.
    """
    if not isinstance(value, tuple):
        raise TypeError(f"{label} is a tuple of codes; got {value!r}")
    seen: set[str] = set()
    for item in value:
        _code(f"{label} entries", item)
        if item in seen:
            raise ValueError(f"{label} names {item!r} twice")
        seen.add(item)
    return value


@register_artifact_type
class ScopeKind(StrEnum):
    """What a finding or a refusal is ABOUT.

    Kept small and structural: these are the things this package can name and
    a consumer can look up. A free-text scope would be a second message field.
    """

    MODEL = "model"
    NODE = "node"
    PARAMETER = "parameter"
    PLATE = "plate"
    BLOCK = "block"
    DATA = "data"
    TASK = "task"
    BACKEND = "backend"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ScopeRef:
    """One thing, named by kind and by name."""

    kind: ScopeKind
    name: str

    def __post_init__(self) -> None:
        _member("a scope's kind", self.kind, ScopeKind)
        _text("a scope's name", self.name)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One measured reason, with the numbers that produced it.

    ``observed`` and ``expected`` are the pair a consumer compares: what was
    measured, and what the premise required. They are canonical values -- a
    scalar or a tuple of them -- so they survive the codec and cannot become a
    runtime handle. G11's lesson is exactly this class: the failing path must
    return what the passing path measured, because a number rendered into a
    sentence is a number nobody downstream can use.
    """

    code: str
    message: str
    observed: object = None
    expected: object = None
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _code("a finding's code", self.code)
        _text("a finding's message", self.message)
        _nested("a finding", "observed", self.observed)
        _nested("a finding", "expected", self.expected)
        _tuple_of("a finding's artifact_refs", self.artifact_refs, ArtifactRef)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class Remedy:
    """Somewhere to go: an action code, an explanation, and its arguments.

    ``parameters`` is sorted and key-unique like every other option table here,
    so "the same remedy" is one value however the caller spelled it.
    """

    action: str
    message: str
    parameters: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        _code("a remedy's action", self.action)
        _text("a remedy's message", self.message)
        object.__setattr__(
            self,
            "parameters",
            canonical_value_options("a remedy's parameters", self.parameters),
        )


@register_artifact_type
class Conservatism(StrEnum):
    """How a fallback compares to the route that refused.

    Named rather than assumed. "We fell back" reads as "we were careful", and
    sometimes the offered route is the cheaper, weaker one -- a caller
    consenting to an automatic fallback needs to know which of those they are
    consenting to.
    """

    EQUIVALENT = "equivalent"
    MORE_CONSERVATIVE = "more_conservative"
    LESS_CONSERVATIVE = "less_conservative"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class FallbackOption:
    """A route the caller could take instead, and whether it would be automatic."""

    task_kind: TaskKind
    backend: str
    conservatism: Conservatism = Conservatism.EQUIVALENT
    automatic: bool = False

    def __post_init__(self) -> None:
        _member("a fallback's task_kind", self.task_kind, TaskKind)
        _text("a fallback's backend", self.backend)
        _member("a fallback's conservatism", self.conservatism, Conservatism)
        if type(self.automatic) is not bool:
            raise TypeError(f"a fallback's automatic is a bool; got {self.automatic!r}")


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class Refusal:
    """What is returned when a task cannot be answered as asked (§0 ruling 1).

    It stands in for the artifact that did not happen -- a plan that could not
    be compiled, or a result that could not be produced -- so its envelope
    carries that kind. It is never filed as an evaluation report: a report is a
    judgement ABOUT an artifact, and a refusal is the absence of one.
    """

    meta: ArtifactMeta
    task: Task
    failed_premise: str
    grounds: tuple[Finding, ...]
    scope: ScopeRef
    remedies: tuple[Remedy, ...]
    fallback: FallbackOption | None = None

    def __post_init__(self) -> None:
        _instance("a refusal's meta", self.meta, ArtifactMeta)
        if self.meta.artifact_type is ArtifactKind.EVALUATION_REPORT:
            raise ValueError(
                "a refusal is filed as the plan or result that did not happen, "
                "never as an evaluation_report; a report judges an artifact and "
                "a refusal is the artifact's absence"
            )
        # Raises TypeError for anything that is not one of the five tasks.
        task_kind(self.task)
        _code("failed_premise", self.failed_premise)
        _tuple_of("a refusal's grounds", self.grounds, Finding)
        if not self.grounds:
            raise ValueError(
                "a refusal states its grounds; a verdict with nothing behind it "
                "leaves a caller parsing the message, which is the failure §0 "
                "ruling 3 exists to prevent"
            )
        _instance("a refusal's scope", self.scope, ScopeRef)
        _tuple_of("a refusal's remedies", self.remedies, Remedy)
        if not self.remedies:
            raise ValueError(
                "a refusal's remedies name at least one thing to try; a "
                "refusal with nowhere to go is a dead end wearing a schema"
            )
        if self.fallback is not None:
            _instance("a refusal's fallback", self.fallback, FallbackOption)
