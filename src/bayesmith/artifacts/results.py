"""The five answers, and the four shapes a posterior can take (§0.5).

A Result is what an execution produced, in the form a later reader can
believe: every array is a :class:``bayesmith.artifacts.base.NamedArray`` copied
in and handed back read-only, every reference to another artifact carries the
revision and kind it points at, and nothing here holds a runtime object.

**A posterior is not one thing.** §0.5 fixes four representations and keeps
them apart rather than flattening them into "draws plus optional weights": iid
exact draws, importance-weighted draws with their diagnostics, a closed-form
family, and a fitted conditional estimator that is only a reference. A weighted
sample whose weights were dropped is a WRONG unweighted sample rather than a
lossy one, which is precisely why the union is tagged by class.

**The mapping from task to result is a bijection, and it lives here.**
``PRIMARY_RESULT_BY_TASK`` is in this module, not in ``tasks``, because the
dependency runs ``tasks ← results`` and a constant placed for tidiness in the
lower module would close a cycle. It is checked in both directions: five keys
that cover ``TaskKind`` AND five distinct values that cover ``ResultKind``, the
pair of checks that would have caught the 5:4 mapping this protocol shipped
once already.

**Unavailable is None, and the field stays.** A diagnostic that was not
computed is ``None`` rather than NaN, and the field is not dropped -- the shape
of a Result is part of the schema, so a reader can tell "not computed" from
"this kind of result has no such number". The single exception is
``log_evidence``, which must be finite: an evidence result whose evidence is
missing is not an evidence result.

Layering: ``_codec ← identity ← base ← tasks ← results`` (§0.1).
"""

from __future__ import annotations

import dataclasses
from enum import StrEnum
from types import MappingProxyType

from ._codec import register_artifact_type
from .base import (
    ArtifactMeta,
    ArtifactRef,
    NamedArray,
    RunRecord,
    _instance,
    _member,
    _optional_count,
    _optional_text,
    _text,
    _tuple_of,
    _uuid4,
)
from .identity import ArtifactKind, Fingerprint
from .tasks import (
    Estimand,
    ParameterSource,
    TaskKind,
    _data_fingerprint,
    _finite,
    _flag,
    _named_values,
    _optional_non_negative,
    _result_ref,
    site_names,
)

__all__ = [
    "ResultKind",
    "LogDensityAvailability",
    "UncertaintyKind",
    "UncertaintyRecord",
    "DrawsPosterior",
    "WeightedDrawsPosterior",
    "AnalyticPosterior",
    "FittedConditionalPosterior",
    "PosteriorRepresentation",
    "POSTERIOR_REPRESENTATIONS",
    "EvidenceComponent",
    "PosteriorResult",
    "EvidenceResult",
    "PredictiveResult",
    "PointEstimateResult",
    "SimulationResult",
    "Result",
    "RESULT_CLASSES",
    "result_kind",
    "PRIMARY_RESULT_BY_TASK",
]


@register_artifact_type
class ResultKind(StrEnum):
    """The five answers of §0 ruling 1, one per task kind."""

    POSTERIOR = "posterior"
    EVIDENCE = "evidence"
    PREDICTIVE = "predictive"
    POINT_ESTIMATE = "point_estimate"
    SIMULATION = "simulation"


@register_artifact_type
class LogDensityAvailability(StrEnum):
    """How much of the log density this result can answer for (§0.5).

    Three levels rather than a bool, because "I can score the joint" and "I can
    score each observation" are different capabilities and only the second
    supports pointwise cross-validation.
    """

    NONE = "none"
    JOINT = "joint"
    POINTWISE = "pointwise"


@register_artifact_type
class UncertaintyKind(StrEnum):
    """Which of the three an uncertainty record holds (§0.5).

    A covariance and a precision are inverses of one another and a standard
    error is neither; a single unlabelled matrix would be read as whichever the
    consumer expected, and would be wrong silently rather than loudly.
    """

    COVARIANCE = "covariance"
    PRECISION = "precision"
    STANDARD_ERROR = "standard_error"


# ---------------------------------------------------------------- validation


def _optional_finite(label: str, value: object) -> float | None:
    """``None``, or the finite number :func:``_finite`` admits. A wrapper, not
    a second rule -- the finiteness question has one answer in this package."""
    if value is None:
        return None
    return _finite(label, value)


def _named_arrays(
    label: str, value: object, *, allow_empty: bool = True
) -> tuple[NamedArray, ...]:
    arrays = _named_values(label, value)
    if not arrays and not allow_empty:
        raise ValueError(f"{label} holds at least one named array")
    return arrays


def _draw_count(label: str, arrays: tuple[NamedArray, ...]) -> int | None:
    """The leading dimension every one of ``arrays`` shares, or ``None`` if empty.

    Refused when they disagree: two latents drawn a different number of times
    are not one sample, and code that later zipped them would silently truncate
    to the shorter -- an answer that looks like a sample of the right shape.
    """
    counts: set[int] = set()
    for array in arrays:
        if array.value.ndim < 1:
            raise ValueError(
                f"{label}[{array.name!r}] has no draw axis; a draw array is "
                "indexed by draw first"
            )
        counts.add(int(array.value.shape[0]))
    if len(counts) > 1:
        raise ValueError(
            f"{label} disagrees about the draw count: "
            + ", ".join(
                f"{array.name!r} has {array.value.shape[0]}" for array in arrays
            )
        )
    return counts.pop() if counts else None


def _report_refs(label: str, value: object) -> tuple[ArtifactRef, ...]:
    _tuple_of(label, value, ArtifactRef)
    seen: set[str] = set()
    for ref in value:
        if ref.artifact_type is not ArtifactKind.EVALUATION_REPORT:
            raise ValueError(
                f"{label} points at a {ref.artifact_type.value}; it names "
                "evaluation reports"
            )
        if ref.artifact_id in seen:
            raise ValueError(f"{label} names {ref.artifact_id} twice")
        seen.add(ref.artifact_id)
    return value


def _optional_report_ref(label: str, value: object) -> ArtifactRef | None:
    if value is None:
        return None
    _report_refs(label, (value,))
    return value


def _result_refs(label: str, value: object) -> tuple[ArtifactRef, ...]:
    _tuple_of(label, value, ArtifactRef)
    seen: set[str] = set()
    for ref in value:
        _result_ref(label, ref)
        if ref.artifact_id in seen:
            raise ValueError(f"{label} names {ref.artifact_id} twice")
        seen.add(ref.artifact_id)
    return value


def _envelope(meta: object, run: object) -> None:
    """Every Result carries these two, and they must agree about what it is."""
    _instance("a result's meta", meta, ArtifactMeta)
    if meta.artifact_type is not ArtifactKind.RESULT:
        raise ValueError(
            f"a result's envelope is of artifact kind result; got "
            f"{meta.artifact_type.value!r}. The kind drives the §0.3 "
            "invalidation matrix, so an artifact filed under the wrong row "
            "would be kept when it should have been retired"
        )
    _instance("a result's run", run, RunRecord)


# -------------------------------------------------- posterior representations


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class DrawsPosterior:
    """Equally weighted draws, with the chain structure that produced them.

    ``chain_shape`` is ``None`` for iid draws and ``(chains, per_chain)`` for a
    sampler that ran chains. ``None`` is not ``(1, n)``: the first says the
    draws have no chain structure to diagnose, the second says a diagnostic
    over one chain is possible, and R-hat means different things to each.
    """

    draws: tuple[NamedArray, ...]
    chain_shape: tuple[int, int] | None = None
    method: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "draws", _named_arrays("draws", self.draws, allow_empty=False)
        )
        count = _draw_count("draws", self.draws)
        _text("a posterior representation's method", self.method)
        shape = self.chain_shape
        if shape is None:
            return
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or any(type(n) is not int for n in shape)
        ):
            raise ValueError(
                f"chain_shape is (chains, draws per chain) or None; got {shape!r}"
            )
        if any(n < 1 for n in shape):
            raise ValueError(f"chain_shape counts are positive; got {shape!r}")
        if shape[0] * shape[1] != count:
            raise ValueError(
                f"chain_shape {shape!r} accounts for {shape[0] * shape[1]} "
                f"draws, but the arrays hold {count}"
            )


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class WeightedDrawsPosterior:
    """Importance-weighted draws, with the diagnostics that say whether to
    believe them.

    ``unreliable`` is a stored verdict rather than something a reader
    recomputes from ``khat``: the threshold that produced it belongs to the run
    that took it, and a consumer applying today's threshold to yesterday's khat
    would silently re-judge a result nobody re-ran.
    """

    draws: tuple[NamedArray, ...]
    log_weights: NamedArray
    ess: float | None = None
    khat: float | None = None
    unreliable: bool = False
    method: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "draws", _named_arrays("draws", self.draws, allow_empty=False)
        )
        count = _draw_count("draws", self.draws)
        _instance("log_weights", self.log_weights, NamedArray)
        if self.log_weights.value.ndim != 1:
            raise ValueError(
                "log_weights is one array over one axis, the draw axis; got "
                f"{self.log_weights.value.ndim} axes"
            )
        if int(self.log_weights.value.shape[0]) != count:
            raise ValueError(
                f"log_weights holds {self.log_weights.value.shape[0]} weights "
                f"for {count} draws; a weight per draw is what makes the sample "
                "a weighted sample rather than two unrelated arrays"
            )
        object.__setattr__(self, "ess", _optional_non_negative("ess", self.ess))
        object.__setattr__(self, "khat", _optional_finite("khat", self.khat))
        _flag("unreliable", self.unreliable)
        _text("a posterior representation's method", self.method)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class AnalyticPosterior:
    """A closed-form posterior: the family and its parameters, not a sample.

    The protocol keeps this arm for a genuinely analytic posterior and refuses
    the temptation to fill it from draws (§0.5). Sample moments in the
    ``parameters`` of a "gaussian" would be a claim of exactness that nothing
    took.
    """

    family: str
    parameters: tuple[NamedArray, ...]
    moments: tuple[NamedArray, ...] = ()

    def __post_init__(self) -> None:
        _text("an analytic posterior's family", self.family)
        object.__setattr__(
            self,
            "parameters",
            _named_arrays("parameters", self.parameters, allow_empty=False),
        )
        object.__setattr__(self, "moments", _named_arrays("moments", self.moments))


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class FittedConditionalPosterior:
    """An amortized posterior, as REFERENCES to what was fitted (§0 ruling 4).

    No estimator object, no callable, no weights blob: the estimator and the
    simulation bank it was trained on are artifacts of their own, and this arm
    records which ones plus the run that fitted them. R1 freezes the schema and
    fits nothing.
    """

    estimator_ref: ArtifactRef
    simulation_bank_ref: ArtifactRef | None = None
    training_run_id: str | None = None
    validation_report_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _result_ref("estimator_ref", self.estimator_ref)
        if self.simulation_bank_ref is not None:
            _result_ref("simulation_bank_ref", self.simulation_bank_ref)
        if self.training_run_id is not None:
            _uuid4("training_run_id", self.training_run_id)
        _report_refs("validation_report_refs", self.validation_report_refs)


PosteriorRepresentation = (
    DrawsPosterior | WeightedDrawsPosterior | AnalyticPosterior | FittedConditionalPosterior
)

#: The four arms, enumerated once so that ``isinstance`` checks and
#: exhaustiveness tests read the same list.
POSTERIOR_REPRESENTATIONS: tuple[type, ...] = (
    DrawsPosterior,
    WeightedDrawsPosterior,
    AnalyticPosterior,
    FittedConditionalPosterior,
)


def _representation_draw_count(representation: object) -> int | None:
    """How many draws the representation holds, or ``None`` when it is not a
    sample. An analytic posterior has no draw axis, so nothing may be checked
    against one."""
    if isinstance(representation, (DrawsPosterior, WeightedDrawsPosterior)):
        return _draw_count("draws", representation.draws)
    return None


# ------------------------------------------------------------ small records


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceComponent:
    """One additive term of a log evidence, and how it was obtained (§0.5).

    The split between exact and residual components is what makes a hybrid
    evidence auditable: a number that is closed-form and a number that was
    sampled carry different warrants, and a single total would hide which part
    the error bar belongs to.
    """

    name: str
    log_value: float
    standard_error: float | None = None
    method: str = ""
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _text("an evidence component's name", self.name)
        object.__setattr__(self, "log_value", _finite("log_value", self.log_value))
        object.__setattr__(
            self,
            "standard_error",
            _optional_non_negative("standard_error", self.standard_error),
        )
        _text("an evidence component's method", self.method)
        _tuple_of("artifact_refs", self.artifact_refs, ArtifactRef)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class UncertaintyRecord:
    """A point estimate's uncertainty, labelled with what it actually is."""

    kind: UncertaintyKind
    arrays: tuple[NamedArray, ...]

    def __post_init__(self) -> None:
        _member("an uncertainty record's kind", self.kind, UncertaintyKind)
        object.__setattr__(
            self, "arrays", _named_arrays("arrays", self.arrays, allow_empty=False)
        )


# ----------------------------------------------------------- the five results


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class PosteriorResult:
    """A posterior over the model's latents, in one of four representations.

    ``latent_names`` and ``eliminated_latents`` are disjoint by construction:
    an eliminated latent is one the plan integrated out, so it is exactly the
    set that has no draws here. ``reconstruction_ref`` points at what can put
    them back, and may only be present when something was eliminated.
    """

    meta: ArtifactMeta
    run: RunRecord
    representation: PosteriorRepresentation
    latent_names: tuple[str, ...]
    eliminated_latents: tuple[str, ...] = ()
    reconstruction_ref: ArtifactRef | None = None
    log_density_availability: LogDensityAvailability = LogDensityAvailability.NONE
    pointwise_log_likelihood: NamedArray | None = None
    predictive_ready: bool = False
    report_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _envelope(self.meta, self.run)
        if not isinstance(self.representation, POSTERIOR_REPRESENTATIONS):
            raise TypeError(
                f"{self.representation!r} is not one of the four posterior "
                f"representations "
                f"({sorted(cls.__name__ for cls in POSTERIOR_REPRESENTATIONS)})"
            )
        site_names("latent_names", self.latent_names)
        if not self.latent_names:
            raise ValueError("a posterior result names at least one latent")
        site_names("eliminated_latents", self.eliminated_latents)
        both = sorted(set(self.latent_names) & set(self.eliminated_latents))
        if both:
            raise ValueError(
                f"{both} are named as both reported and eliminated latents; an "
                "eliminated latent is precisely one this result has no draws for"
            )

        drawn = _representation_draw_count(self.representation)
        if drawn is not None:
            names = {array.name for array in self.representation.draws}
            if names != set(self.latent_names):
                raise ValueError(
                    f"the representation draws {sorted(names)} but latent_names "
                    f"is {sorted(self.latent_names)}; a missing array and an "
                    "extra one are both a result that does not say what it claims"
                )

        if self.reconstruction_ref is not None:
            _instance("reconstruction_ref", self.reconstruction_ref, ArtifactRef)
            if not self.eliminated_latents:
                raise ValueError(
                    "reconstruction_ref is for putting eliminated latents back, "
                    "and this result eliminated none"
                )
        _member(
            "log_density_availability",
            self.log_density_availability,
            LogDensityAvailability,
        )
        pointwise = self.pointwise_log_likelihood
        if (self.log_density_availability is LogDensityAvailability.POINTWISE) != (
            pointwise is not None
        ):
            raise ValueError(
                "log_density_availability is pointwise exactly when "
                "pointwise_log_likelihood is present; got "
                f"{self.log_density_availability.value!r} and "
                f"{'an array' if pointwise is not None else 'None'}"
            )
        if pointwise is not None:
            _instance("pointwise_log_likelihood", pointwise, NamedArray)
            if drawn is not None and int(pointwise.value.shape[0]) != drawn:
                raise ValueError(
                    f"pointwise_log_likelihood is indexed by draw first, so its "
                    f"leading axis is {drawn}; got {pointwise.value.shape[0]}"
                )
        _flag("predictive_ready", self.predictive_ready)
        _report_refs("report_refs", self.report_refs)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceResult:
    """A log marginal likelihood, its error bar, and its decomposition."""

    meta: ArtifactMeta
    run: RunRecord
    log_evidence: float
    standard_error: float | None = None
    posterior_representation: PosteriorRepresentation | None = None
    normalization_audit_refs: tuple[ArtifactRef, ...] = ()
    exact_components: tuple[EvidenceComponent, ...] = ()
    residual_component: EvidenceComponent | None = None
    repeat_result_refs: tuple[ArtifactRef, ...] = ()
    consistency_report_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        _envelope(self.meta, self.run)
        object.__setattr__(
            self, "log_evidence", _finite("log_evidence", self.log_evidence)
        )
        object.__setattr__(
            self,
            "standard_error",
            _optional_non_negative("standard_error", self.standard_error),
        )
        if self.posterior_representation is not None and not isinstance(
            self.posterior_representation, POSTERIOR_REPRESENTATIONS
        ):
            raise TypeError(
                f"{self.posterior_representation!r} is not a posterior "
                "representation"
            )
        _report_refs("normalization_audit_refs", self.normalization_audit_refs)
        _tuple_of("exact_components", self.exact_components, EvidenceComponent)
        names = [component.name for component in self.exact_components]
        if len(set(names)) != len(names):
            raise ValueError(f"exact_components names a component twice: {names}")
        object.__setattr__(
            self,
            "exact_components",
            tuple(sorted(self.exact_components, key=lambda item: item.name)),
        )
        if self.residual_component is not None:
            _instance("residual_component", self.residual_component, EvidenceComponent)
        _result_refs("repeat_result_refs", self.repeat_result_refs)
        _optional_report_ref("consistency_report_ref", self.consistency_report_ref)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class PredictiveResult:
    """Draws pushed through the model onto observations, held-out or replicated.

    Latent and replicated draws share ONE draw axis, because they are two views
    of the same posterior sample; a result whose halves disagreed about how
    many draws there were could not be zipped back together.
    """

    meta: ArtifactMeta
    run: RunRecord
    source_posterior_ref: ArtifactRef
    conditioning_data: Fingerprint | None = None
    prediction_design: Fingerprint | None = None
    conditioned_sites: tuple[str, ...] = ()
    latent_draws: tuple[NamedArray, ...] = ()
    replicated_draws: tuple[NamedArray, ...] = ()
    pointwise_log_density: NamedArray | None = None
    observation_unit: str | None = None
    grouping: str | None = None
    report_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _envelope(self.meta, self.run)
        _result_ref("source_posterior_ref", self.source_posterior_ref)
        _data_fingerprint("conditioning_data", self.conditioning_data)
        _data_fingerprint("prediction_design", self.prediction_design)
        site_names("conditioned_sites", self.conditioned_sites)
        object.__setattr__(
            self, "latent_draws", _named_arrays("latent_draws", self.latent_draws)
        )
        object.__setattr__(
            self,
            "replicated_draws",
            _named_arrays("replicated_draws", self.replicated_draws),
        )
        if not self.latent_draws and not self.replicated_draws:
            raise ValueError(
                "a predictive result holds latent draws, replicated draws, or "
                "both; holding neither is a result with no content"
            )
        count = _draw_count(
            "the predictive draws", self.latent_draws + self.replicated_draws
        )
        if self.pointwise_log_density is not None:
            _instance(
                "pointwise_log_density", self.pointwise_log_density, NamedArray
            )
            if int(self.pointwise_log_density.value.shape[0]) != count:
                raise ValueError(
                    "pointwise_log_density is indexed by draw first, so its "
                    f"leading axis is {count}; got "
                    f"{self.pointwise_log_density.value.shape[0]}"
                )
        _optional_text("observation_unit", self.observation_unit)
        _optional_text("grouping", self.grouping)
        _report_refs("report_refs", self.report_refs)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class PointEstimateResult:
    """One point of the posterior, with what the optimiser or solver knows.

    ``local_only`` is the honest field: an optimiser reports a stationary
    point, and calling it THE MAP would be a global claim nothing verified.
    """

    meta: ArtifactMeta
    run: RunRecord
    estimand: Estimand
    values: tuple[NamedArray, ...]
    objective: float | None = None
    uncertainty: UncertaintyRecord | None = None
    gradient_norm: float | None = None
    residual: float | None = None
    iterations: int | None = None
    local_only: bool = False
    report_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _envelope(self.meta, self.run)
        _member("estimand", self.estimand, Estimand)
        object.__setattr__(
            self, "values", _named_arrays("values", self.values, allow_empty=False)
        )
        object.__setattr__(self, "objective", _optional_finite("objective", self.objective))
        if self.uncertainty is not None:
            _instance("uncertainty", self.uncertainty, UncertaintyRecord)
        object.__setattr__(
            self,
            "gradient_norm",
            _optional_non_negative("gradient_norm", self.gradient_norm),
        )
        object.__setattr__(
            self, "residual", _optional_non_negative("residual", self.residual)
        )
        _optional_count("iterations", self.iterations)
        _flag("local_only", self.local_only)
        _report_refs("report_refs", self.report_refs)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class SimulationResult:
    """Forward draws from a parameter source. No conditioning, no verdict."""

    meta: ArtifactMeta
    run: RunRecord
    parameter_source: ParameterSource
    parameters: tuple[NamedArray, ...] = ()
    latent_draws: tuple[NamedArray, ...] = ()
    observation_draws: tuple[NamedArray, ...] = ()
    prediction_design: Fingerprint | None = None
    report_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        _envelope(self.meta, self.run)
        _instance("parameter_source", self.parameter_source, ParameterSource)
        object.__setattr__(
            self, "parameters", _named_arrays("parameters", self.parameters)
        )
        object.__setattr__(
            self, "latent_draws", _named_arrays("latent_draws", self.latent_draws)
        )
        object.__setattr__(
            self,
            "observation_draws",
            _named_arrays("observation_draws", self.observation_draws),
        )
        if not self.latent_draws and not self.observation_draws:
            raise ValueError(
                "a simulation result holds latent draws, observation draws, or "
                "both; holding neither is a result with no content"
            )
        _draw_count(
            "the simulation draws", self.latent_draws + self.observation_draws
        )
        _data_fingerprint("prediction_design", self.prediction_design)
        _report_refs("report_refs", self.report_refs)


Result = (
    PosteriorResult
    | EvidenceResult
    | PredictiveResult
    | PointEstimateResult
    | SimulationResult
)

#: kind -> class, enumerated once (see ``TASK_CLASSES`` for why it is a table).
RESULT_CLASSES: dict[ResultKind, type] = {
    ResultKind.POSTERIOR: PosteriorResult,
    ResultKind.EVIDENCE: EvidenceResult,
    ResultKind.PREDICTIVE: PredictiveResult,
    ResultKind.POINT_ESTIMATE: PointEstimateResult,
    ResultKind.SIMULATION: SimulationResult,
}

_KIND_BY_CLASS: dict[type, ResultKind] = {
    cls: kind for kind, cls in RESULT_CLASSES.items()
}


def result_kind(result: Result) -> ResultKind:
    """Which of the five ``result`` is, by exact type."""
    kind = _KIND_BY_CLASS.get(type(result))
    if kind is None:
        raise TypeError(
            f"{type(result).__name__} is not one of the five results "
            f"({sorted(cls.__name__ for cls in RESULT_CLASSES.values())})"
        )
    return kind


#: §0 ruling 1's five-in-five-out, as a table a test can check from both sides.
#: A read-only mapping: this is the protocol's pairing, and a caller who could
#: rebind an entry would redirect every dispatch that reads it.
PRIMARY_RESULT_BY_TASK = MappingProxyType(
    {
        TaskKind.POSTERIOR: ResultKind.POSTERIOR,
        TaskKind.EVIDENCE: ResultKind.EVIDENCE,
        TaskKind.PREDICTIVE: ResultKind.PREDICTIVE,
        TaskKind.POINT_ESTIMATE: ResultKind.POINT_ESTIMATE,
        TaskKind.SIMULATION: ResultKind.SIMULATION,
    }
)
