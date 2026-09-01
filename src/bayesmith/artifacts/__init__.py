"""The public inventory of the artifact protocol, and nothing heavier.

:mod:`bayesmith.artifacts` is the leaf layer: nothing here imports
:mod:`bayesmith.graph`, :mod:`bayesmith.dispatch`, JAX, Equinox or NumPyro at
module scope, and nothing here holds a runtime object. An artifact is data --
a Graph, a callable, a compiled executable or a backend handle is represented
by a reference, a fingerprint or a runtime attachment, never pickled into the
artifact itself.

This module is the one public surface for that protocol. Every name below is
re-exported from the submodule that owns it, and each submodule's own __all__
is the single statement of what it defines -- so the identity test in
tests/test_public_api.py can require that an attribute IS the owning module's
object rather than settling for hasattr. The canonical codec itself
(:mod:`bayesmith.artifacts._codec`) stays private: the wire format is this
package's own business, and the two public entry points are dump_artifact
and load_artifact.

The runtime bridge is NOT here. :mod:`bayesmith.dispatch.task` is the only
module that meets a Graph and a plan, and its compile_task/execute_task are
the root-level entry points -- kept there rather than here because importing
them pulls in JAX and NumPyro, which this layer must not.
"""

from bayesmith.artifacts._codec import (
    ArtifactCodecError,
    ArtifactFile,
    UnsupportedSchemaVersion,
    dump_artifact,
    load_artifact,
)
from bayesmith.artifacts.base import (
    SCHEMA_VERSION,
    ApproximationClass,
    ApproximationRecord,
    ArtifactMeta,
    ArtifactRef,
    ArtifactStatus,
    BackendRef,
    CanonicalScalar,
    CanonicalValue,
    ComputeBudget,
    DeviceRecord,
    ErrorRecord,
    LifecycleRecord,
    NamedArray,
    ProducerRef,
    RunRecord,
    RunWarning,
    SeedRecord,
    TargetFidelity,
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    canonical_scalar_options,
    canonical_value_options,
    invalidate_meta,
    new_artifact_meta,
    utc_timestamp,
)
from bayesmith.artifacts.gates import (
    AttemptStatus,
    GateDefinition,
    GateResult,
    OperationalStatus,
    ReportRequirement,
    ReportSlot,
    aggregate_gate,
)
from bayesmith.artifacts.identity import (
    FINGERPRINT_ALGORITHM,
    ArtifactKind,
    Fingerprint,
    FingerprintBundle,
    FingerprintKind,
    InvalidationPolicy,
    ModelRef,
    changed_fingerprints,
    fingerprint,
    model_ref_from_callable,
)
from bayesmith.artifacts.refusal import (
    CAPABILITY_UNAVAILABLE_R1,
    PREMISES,
    Conservatism,
    FallbackOption,
    Finding,
    Refusal,
    Remedy,
    ScopeKind,
    ScopeRef,
)
from bayesmith.artifacts.reports import (
    AnalysisFinding,
    AnalysisReport,
    Applicability,
    Conclusion,
    EvaluationReport,
    InferencePlanRecord,
    PlanBlockRecord,
)
from bayesmith.artifacts.results import (
    POSTERIOR_REPRESENTATIONS,
    PRIMARY_RESULT_BY_TASK,
    RESULT_CLASSES,
    AnalyticPosterior,
    DrawsPosterior,
    EvidenceComponent,
    EvidenceResult,
    FittedConditionalPosterior,
    LogDensityAvailability,
    PointEstimateResult,
    PosteriorRepresentation,
    PosteriorResult,
    PredictiveResult,
    Result,
    ResultKind,
    SimulationResult,
    UncertaintyKind,
    UncertaintyRecord,
    WeightedDrawsPosterior,
    result_kind,
)
from bayesmith.artifacts.tasks import (
    TASK_CLASSES,
    Estimand,
    EvidenceTask,
    ParameterSource,
    ParameterSourceKind,
    PointEstimateTask,
    PosteriorTask,
    PredictiveTask,
    SimulationTask,
    Task,
    TaskKind,
    TaskMeta,
    new_task_meta,
    site_names,
    task_fingerprint,
    task_kind,
)

__all__ = [
    # persistence and the codec's public corner
    "ArtifactCodecError",
    "ArtifactFile",
    "UnsupportedSchemaVersion",
    "dump_artifact",
    "load_artifact",
    # identity
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
    # base envelope and small value objects
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
    # tasks
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
    # results
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
    # refusal
    "ScopeKind",
    "ScopeRef",
    "Finding",
    "Remedy",
    "Conservatism",
    "FallbackOption",
    "Refusal",
    "CAPABILITY_UNAVAILABLE_R1",
    "PREMISES",
    # reports
    "AnalysisFinding",
    "AnalysisReport",
    "PlanBlockRecord",
    "InferencePlanRecord",
    "Applicability",
    "Conclusion",
    "EvaluationReport",
    # gates
    "OperationalStatus",
    "AttemptStatus",
    "ReportRequirement",
    "ReportSlot",
    "GateDefinition",
    "GateResult",
    "aggregate_gate",
]
