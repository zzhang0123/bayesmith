"""Five in, five out: the Task and Result protocols, pinned.

Plan §0.4 and §0.5 fix the two families and §0 ruling 1 fixes their pairing.
The tests here are written against the failures that pairing has already had,
or would have, rather than against the shape of the dataclasses.

**Five is a count, and it has been wrong before.** ``PRIMARY_RESULT_BY_TASK``
is asserted to be a bijection in BOTH directions -- every ``TaskKind`` is a key,
every ``ResultKind`` is a value, and no two tasks share a result -- because the
failure this replaces was five tasks mapping onto four results, which looks
correct from either side alone. ``PredictiveResult`` additionally gets its own
round trip, since it was the one that went missing.

**An artifact is not a handle on the caller's array.** The array handed to a
Result is copied on the way in; a caller who edits theirs afterwards has not
edited history. :class:``NamedArray`` is where that is implemented and
``tests/artifacts/test_base.py`` is where it is proved, so what is checked here
is that the Result families actually route their arrays through it.

**A name-keyed collection is refused when it is ambiguous, and normalised
when it is not.** Two draw arrays called ``theta`` are refused, because one of
them would be dropped and nothing would say which; the same two arrays given
in two orders are one artifact, because the order of a name-keyed collection
is not part of what it says. Site LISTS are the other case -- they keep the
order they were declared in, since that order is what a caller reads back --
and duplicates there are refused for the same reason.

**Unavailable is ``None``; it is never NaN.** §0.5 keeps the field and leaves
it empty rather than filling it with a number that compares unequal to itself
and reads, at a glance, as a measurement. The one number that must be finite
when it exists is ``log_evidence``.

**The decoder constructs only what it was told about.** A payload naming a
class this package never registered is refused, and a ``PosteriorTask`` payload
is refused as an ``EvidenceTask`` rather than being coerced into one -- the
tagged union is the thing the whole protocol discriminates on, so a decoder
that could be talked into the wrong arm would make every downstream branch a
guess.
"""

from __future__ import annotations

import dataclasses
import typing
import uuid

import numpy as np
import pytest

from bayesmith.artifacts._codec import (
    ArtifactCodecError,
    canonical_dumps,
    canonical_loads,
)
from bayesmith.artifacts.base import (
    ApproximationClass,
    ApproximationRecord,
    ArtifactRef,
    BackendRef,
    ComputeBudget,
    DeviceRecord,
    NamedArray,
    ProducerRef,
    RunRecord,
    SeedRecord,
    TargetFidelity,
    TerminationReason,
    TerminationRecord,
    TimingRecord,
    new_artifact_meta,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    fingerprint,
)
from bayesmith.artifacts.results import (
    POSTERIOR_REPRESENTATIONS,
    PRIMARY_RESULT_BY_TASK,
    AnalyticPosterior,
    DrawsPosterior,
    EvidenceComponent,
    EvidenceResult,
    FittedConditionalPosterior,
    LogDensityAvailability,
    PointEstimateResult,
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
    task_fingerprint,
    task_kind,
)

K = FingerprintKind

START = "2026-08-30T12:00:00Z"
FINISH = "2026-08-30T12:00:09.500000Z"

#: A FIXED reference, so that two calls to the predictive factory differ in
#: nothing but what the test changed. A freshly minted id per call would make
#: every "this field moves the fingerprint" assertion below pass on the id
#: alone -- a guard that cannot fail.
SOURCE_POSTERIOR = "7c1c8a1e-4f5b-4a2e-9a2a-0f1d5b6c7d8e"


# ------------------------------------------------------------------ fixtures


def bundle(**overrides: object) -> FingerprintBundle:
    payloads: dict[str, object] = {
        "model_source": "pilot",
        "graph_structure": "theta -> y",
        "data": "y=[1,2,3]",
        "task": "posterior",
    }
    payloads.update(overrides)
    return FingerprintBundle(
        **{name: fingerprint(K(name), value) for name, value in payloads.items()}
    )


def ref(kind: ArtifactKind = ArtifactKind.RESULT, revision: int = 0) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=str(uuid.uuid4()), revision=revision, artifact_type=kind
    )


def meta(**overrides: object):
    fields: dict[str, object] = {
        "artifact_type": ArtifactKind.RESULT,
        "fingerprints": bundle(),
        "producer": ProducerRef(package="bayesmith", version="0.6.2"),
        "summary": "four draws of theta",
    }
    fields.update(overrides)
    return new_artifact_meta(**fields)


def run(**overrides: object) -> RunRecord:
    fields: dict[str, object] = {
        "run_id": str(uuid.uuid4()),
        "plan_ref": ref(ArtifactKind.PLAN),
        "fingerprints": bundle(),
        "seed": SeedRecord(seed=7, key_algorithm="threefry2x32"),
        "dtype": "float64",
        "devices": (DeviceRecord(platform="cpu", device_kind="cpu", device_id=0),),
        "jax_config": (("jax_enable_x64", True),),
        "backend": BackendRef(name="numpyro", version="0.15.0"),
        "budget": ComputeBudget(draws=4),
        "termination": TerminationRecord(reason=TerminationReason.COMPLETED),
        "timing": TimingRecord(
            started_at=START, finished_at=FINISH, wall_clock_seconds=1.5
        ),
        "approximation": ApproximationRecord(
            representation_class=ApproximationClass.MONTE_CARLO,
            target_fidelity=TargetFidelity.EXACT,
        ),
        "warnings": (),
    }
    fields.update(overrides)
    return RunRecord(**fields)


def named(name: str, values, dims: tuple[str, ...] = ("draw",)) -> NamedArray:
    return NamedArray(name=name, value=np.asarray(values, dtype=float), dims=dims)


def draws() -> DrawsPosterior:
    return DrawsPosterior(
        draws=(named("theta", [1.0, 2.0, 3.0, 4.0]),),
        chain_shape=(2, 2),
        method="nuts",
    )


def weighted() -> WeightedDrawsPosterior:
    return WeightedDrawsPosterior(
        draws=(named("theta", [1.0, 2.0, 3.0, 4.0]),),
        log_weights=named("log_weights", [-1.0, -2.0, -3.0, -4.0]),
        ess=3.25,
        khat=0.4,
        unreliable=False,
        method="snis",
    )


def analytic() -> AnalyticPosterior:
    return AnalyticPosterior(
        family="gaussian",
        parameters=(named("mean", [0.0]), named("cov", [[1.0]], ("row", "column"))),
        moments=(named("variance", [1.0]),),
    )


def fitted() -> FittedConditionalPosterior:
    return FittedConditionalPosterior(
        estimator_ref=ref(),
        simulation_bank_ref=ref(),
        training_run_id=str(uuid.uuid4()),
        validation_report_refs=(ref(ArtifactKind.EVALUATION_REPORT),),
    )


# ------------------------------------------------------------- minimal tasks


def posterior_task(**overrides: object) -> PosteriorTask:
    fields: dict[str, object] = {
        "meta": new_task_meta(label="fit theta"),
        "backend": "numpyro",
        "budget": ComputeBudget(draws=4, warmup=2, chains=2),
        "chain_method": "parallel",
        "solver_tolerance": 1e-8,
        "solver_maxiter": 100,
        "require_convergence": True,
        "ess_floor": 200.0,
        "nuts_on_collapse": True,
        "backend_options": (("target_accept_prob", 0.9),),
        "quality_gate": "posterior-gate@1",
    }
    fields.update(overrides)
    return PosteriorTask(**fields)


def evidence_task(**overrides: object) -> EvidenceTask:
    fields: dict[str, object] = {
        "meta": new_task_meta(label="log Z"),
        "backend": "auto",
        "budget": ComputeBudget(draws=64),
        "reconstruct_posterior": True,
        "repeat_count": 3,
        "backend_options": (),
        "quality_gate": None,
    }
    fields.update(overrides)
    return EvidenceTask(**fields)


def predictive_task(**overrides: object) -> PredictiveTask:
    fields: dict[str, object] = {
        "meta": new_task_meta(label="posterior predictive"),
        "source_posterior_ref": ArtifactRef(
            artifact_id=SOURCE_POSTERIOR,
            revision=0,
            artifact_type=ArtifactKind.RESULT,
        ),
        "conditioning_data": fingerprint(K.DATA, "y=[1,2,3]"),
        "prediction_design": fingerprint(K.DATA, "x_new"),
        "conditioned_sites": ("y",),
        "replicated_sites": ("y_rep",),
        "latent_sites": ("theta",),
        "budget": ComputeBudget(draws=4),
        "backend": "auto",
    }
    fields.update(overrides)
    return PredictiveTask(**fields)


def point_estimate_task(**overrides: object) -> PointEstimateTask:
    fields: dict[str, object] = {
        "meta": new_task_meta(label="MAP"),
        "estimand": Estimand.MAP,
        "names": ("theta",),
        "backend": "auto",
        "budget": ComputeBudget(max_iterations=50),
        "optimizer_options": (("method", "lbfgs"),),
    }
    fields.update(overrides)
    return PointEstimateTask(**fields)


def simulation_task(**overrides: object) -> SimulationTask:
    fields: dict[str, object] = {
        "meta": new_task_meta(label="prior draws"),
        "parameter_source": ParameterSource.prior(),
        "prediction_design": fingerprint(K.DATA, "x_new"),
        "latent_sites": ("theta",),
        "observed_sites": ("y",),
        "budget": ComputeBudget(draws=4),
        "backend": "auto",
    }
    fields.update(overrides)
    return SimulationTask(**fields)


TASK_FACTORIES = {
    "posterior": posterior_task,
    "evidence": evidence_task,
    "predictive": predictive_task,
    "point_estimate": point_estimate_task,
    "simulation": simulation_task,
}


# ----------------------------------------------------------- minimal results


def posterior_result(**overrides: object) -> PosteriorResult:
    fields: dict[str, object] = {
        "meta": meta(),
        "run": run(),
        "representation": draws(),
        "latent_names": ("theta",),
        "log_density_availability": LogDensityAvailability.POINTWISE,
        "pointwise_log_likelihood": named(
            "log_likelihood",
            [[-1.0, -1.0], [-2.0, -2.0], [-3.0, -3.0], [-4.0, -4.0]],
            ("draw", "observation"),
        ),
        "predictive_ready": True,
        "report_refs": (ref(ArtifactKind.EVALUATION_REPORT),),
    }
    fields.update(overrides)
    return PosteriorResult(**fields)


def evidence_result(**overrides: object) -> EvidenceResult:
    fields: dict[str, object] = {
        "meta": meta(),
        "run": run(),
        "log_evidence": -12.5,
        "standard_error": 0.25,
        "posterior_representation": weighted(),
        "normalization_audit_refs": (ref(ArtifactKind.EVALUATION_REPORT),),
        "exact_components": (
            EvidenceComponent(
                name="linear_block",
                log_value=-4.0,
                standard_error=None,
                method="closed_form",
                artifact_refs=(),
            ),
        ),
        "residual_component": EvidenceComponent(
            name="residual",
            log_value=-8.5,
            standard_error=0.25,
            method="snis",
            artifact_refs=(),
        ),
        "repeat_result_refs": (ref(),),
        "consistency_report_ref": ref(ArtifactKind.EVALUATION_REPORT),
    }
    fields.update(overrides)
    return EvidenceResult(**fields)


def predictive_result(**overrides: object) -> PredictiveResult:
    fields: dict[str, object] = {
        "meta": meta(),
        "run": run(),
        "source_posterior_ref": ref(),
        "conditioning_data": fingerprint(K.DATA, "y=[1,2,3]"),
        "prediction_design": fingerprint(K.DATA, "x_new"),
        "conditioned_sites": ("y",),
        "latent_draws": (named("theta", [1.0, 2.0, 3.0, 4.0]),),
        "replicated_draws": (
            named("y_rep", [[1.0], [2.0], [3.0], [4.0]], ("draw", "observation")),
        ),
        "pointwise_log_density": named(
            "log_density", [[-1.0], [-1.0], [-2.0], [-2.0]], ("draw", "observation")
        ),
        "observation_unit": "visit",
        "grouping": "patient",
        "report_refs": (),
    }
    fields.update(overrides)
    return PredictiveResult(**fields)


def point_estimate_result(**overrides: object) -> PointEstimateResult:
    fields: dict[str, object] = {
        "meta": meta(),
        "run": run(),
        "estimand": Estimand.POSTERIOR_MEAN,
        "values": (named("theta", [0.5], ("component",)),),
        "objective": -3.25,
        "uncertainty": UncertaintyRecord(
            kind=UncertaintyKind.COVARIANCE,
            arrays=(named("theta", [[0.25]], ("row", "column")),),
        ),
        "gradient_norm": 1e-9,
        "residual": 0.0,
        "iterations": 12,
        "local_only": False,
        "report_refs": (),
    }
    fields.update(overrides)
    return PointEstimateResult(**fields)


def simulation_result(**overrides: object) -> SimulationResult:
    fields: dict[str, object] = {
        "meta": meta(),
        "run": run(),
        "parameter_source": ParameterSource.fixed((named("theta", [0.5], ("site",)),)),
        "parameters": (named("theta", [0.5], ("site",)),),
        "latent_draws": (named("theta", [1.0, 2.0, 3.0, 4.0]),),
        "observation_draws": (
            named("y", [[1.0], [2.0], [3.0], [4.0]], ("draw", "observation")),
        ),
        "prediction_design": fingerprint(K.DATA, "x_new"),
        "report_refs": (),
    }
    fields.update(overrides)
    return SimulationResult(**fields)


RESULT_FACTORIES = {
    "posterior": posterior_result,
    "evidence": evidence_result,
    "predictive": predictive_result,
    "point_estimate": point_estimate_result,
    "simulation": simulation_result,
}


# ------------------------------------------------------- exhaustive coverage


def test_the_task_union_has_exactly_the_five_task_classes():
    assert len(typing.get_args(Task)) == len(TaskKind) == 5
    assert {task_kind(factory()) for factory in TASK_FACTORIES.values()} == set(TaskKind)


def test_the_result_union_has_exactly_the_five_result_classes():
    assert len(typing.get_args(Result)) == len(ResultKind) == 5
    assert {result_kind(factory()) for factory in RESULT_FACTORIES.values()} == set(
        ResultKind
    )


def test_the_primary_result_mapping_is_a_bijection_in_both_directions():
    """The 5:4 guard, stated from both sides.

    A mapping missing one result still covers every task key, and a mapping
    that sends two tasks to one result still has five entries. Only the pair of
    set equalities plus the length check refuses both.
    """
    assert set(PRIMARY_RESULT_BY_TASK) == set(TaskKind)
    assert set(PRIMARY_RESULT_BY_TASK.values()) == set(ResultKind)
    assert len(set(PRIMARY_RESULT_BY_TASK.values())) == len(ResultKind)


def test_the_primary_result_of_each_task_is_the_result_its_factory_builds():
    """The mapping is checked against the objects, not against a second copy
    of itself: a table that agrees only with its own spelling proves nothing."""
    for name, task_factory in TASK_FACTORIES.items():
        kind = task_kind(task_factory())
        assert PRIMARY_RESULT_BY_TASK[kind] == result_kind(RESULT_FACTORIES[name]())


def test_the_primary_result_mapping_cannot_be_edited():
    with pytest.raises(TypeError):
        PRIMARY_RESULT_BY_TASK[TaskKind.POSTERIOR] = ResultKind.EVIDENCE


# ------------------------------------------------------------ round tripping


@pytest.mark.parametrize("name", sorted(TASK_FACTORIES))
def test_every_task_round_trips_through_the_codec(name):
    task = TASK_FACTORIES[name]()
    restored = canonical_loads(canonical_dumps(task), expected=type(task))
    assert type(restored) is type(task)
    assert restored == task
    for field in dataclasses.fields(task):
        assert getattr(restored, field.name) == getattr(task, field.name), field.name


@pytest.mark.parametrize("name", sorted(RESULT_FACTORIES))
def test_every_result_round_trips_through_the_codec(name):
    result = RESULT_FACTORIES[name]()
    restored = canonical_loads(canonical_dumps(result), expected=type(result))
    assert type(restored) is type(result)
    assert restored == result
    for field in dataclasses.fields(result):
        assert getattr(restored, field.name) == getattr(result, field.name), field.name


def test_a_predictive_result_round_trips_on_its_own():
    """Named separately because this is the one that went missing.

    A parametrized sweep passes with four arms when the fifth was never added
    to the table it sweeps over; a test that names the class cannot.
    """
    result = predictive_result()
    restored = canonical_loads(canonical_dumps(result), expected=PredictiveResult)
    assert isinstance(restored, PredictiveResult)
    assert restored.replicated_draws == result.replicated_draws
    assert restored.conditioned_sites == ("y",)
    assert result_kind(restored) is ResultKind.PREDICTIVE


@pytest.mark.parametrize("representation", ["draws", "weighted", "analytic", "fitted"])
def test_every_posterior_representation_round_trips(representation):
    built = {
        "draws": draws,
        "weighted": weighted,
        "analytic": analytic,
        "fitted": fitted,
    }[representation]()
    restored = canonical_loads(canonical_dumps(built), expected=type(built))
    assert restored == built


def test_the_four_posterior_representations_are_declared_once():
    """The declared tuple and the classes actually built agree, both ways: a
    fifth arm nobody listed and a listed arm nobody can build are the same
    defect seen from two sides."""
    assert len(POSTERIOR_REPRESENTATIONS) == len(set(POSTERIOR_REPRESENTATIONS)) == 4
    built = (draws(), weighted(), analytic(), fitted())
    assert {type(representation) for representation in built} == set(
        POSTERIOR_REPRESENTATIONS
    )
    for representation in built:
        assert isinstance(representation, POSTERIOR_REPRESENTATIONS)


# ------------------------------------------------ arrays are copied, not held


def test_editing_the_array_a_result_was_built_from_does_not_edit_the_result():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = posterior_result(
        representation=DrawsPosterior(
            draws=(NamedArray(name="theta", value=values, dims=("draw",)),),
            chain_shape=None,
            method="exact",
        )
    )
    values[0] = 99.0
    assert result.representation.draws[0].value[0] == 1.0
    with pytest.raises(ValueError):
        result.representation.draws[0].value[0] = 99.0


def test_editing_the_array_a_task_was_built_from_does_not_edit_the_task():
    values = np.array([0.5])
    source = ParameterSource.fixed(
        (NamedArray(name="theta", value=values, dims=("site",)),)
    )
    before = task_fingerprint(simulation_task(parameter_source=source))
    values[0] = 99.0
    assert source.values[0].value[0] == 0.5
    assert task_fingerprint(simulation_task(parameter_source=source)) == before


# ------------------------------------------------------- names and duplicates


def test_two_draw_arrays_with_one_name_are_refused():
    with pytest.raises(ValueError, match="theta"):
        DrawsPosterior(
            draws=(named("theta", [1.0, 2.0]), named("theta", [3.0, 4.0])),
            chain_shape=None,
            method="exact",
        )


def test_a_repeated_latent_name_is_refused():
    with pytest.raises(ValueError, match="theta"):
        posterior_result(latent_names=("theta", "theta"))


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (predictive_task, "conditioned_sites"),
        (predictive_task, "replicated_sites"),
        (predictive_task, "latent_sites"),
        (simulation_task, "latent_sites"),
        (simulation_task, "observed_sites"),
    ],
)
def test_a_repeated_site_name_is_refused(factory, field):
    with pytest.raises(ValueError, match="y"):
        factory(**{field: ("y", "y")})


def test_a_repeated_point_estimate_name_is_refused():
    with pytest.raises(ValueError, match="theta"):
        point_estimate_task(names=("theta", "theta"))


def test_site_lists_keep_the_order_they_were_declared_in():
    """Sites are read back by a caller, so their order is theirs. This is the
    OTHER half of the name rule: values keyed by name are normalised, lists a
    caller wrote are not."""
    task = predictive_task(latent_sites=("theta", "alpha", "beta"))
    assert task.latent_sites == ("theta", "alpha", "beta")


def test_two_orders_of_the_same_fixed_parameters_are_one_task():
    """A name-keyed collection has no order, so two spellings of it must not
    be two artifacts with two fingerprints."""
    first = named("alpha", [1.0], ("site",))
    second = named("theta", [2.0], ("site",))
    forward = simulation_task(parameter_source=ParameterSource.fixed((first, second)))
    backward = simulation_task(parameter_source=ParameterSource.fixed((second, first)))
    assert forward.parameter_source == backward.parameter_source
    assert task_fingerprint(forward) == task_fingerprint(backward)


def test_a_draws_posterior_must_name_exactly_the_latents():
    with pytest.raises(ValueError, match="alpha"):
        posterior_result(latent_names=("theta", "alpha"))
    with pytest.raises(ValueError, match="alpha"):
        posterior_result(
            representation=DrawsPosterior(
                draws=(named("theta", [1.0, 2.0]), named("alpha", [3.0, 4.0])),
                chain_shape=None,
                method="exact",
            ),
            latent_names=("theta",),
        )


def test_an_eliminated_latent_cannot_also_be_a_reported_latent():
    with pytest.raises(ValueError, match="theta"):
        posterior_result(eliminated_latents=("theta",))


# ------------------------------------------------------------ shape agreement


def test_weighted_draws_refuse_log_weights_of_the_wrong_length():
    with pytest.raises(ValueError, match="log_weights"):
        WeightedDrawsPosterior(
            draws=(named("theta", [1.0, 2.0, 3.0, 4.0]),),
            log_weights=named("log_weights", [-1.0, -2.0]),
            ess=None,
            khat=None,
            unreliable=True,
            method="snis",
        )


def test_weighted_draws_refuse_log_weights_that_are_not_one_dimensional():
    with pytest.raises(ValueError, match="one"):
        WeightedDrawsPosterior(
            draws=(named("theta", [1.0, 2.0]),),
            log_weights=named("log_weights", [[-1.0], [-2.0]], ("draw", "extra")),
            ess=None,
            khat=None,
            unreliable=True,
            method="snis",
        )


def test_draws_of_two_latents_must_share_a_draw_count():
    with pytest.raises(ValueError, match="draw"):
        DrawsPosterior(
            draws=(named("theta", [1.0, 2.0, 3.0, 4.0]), named("alpha", [1.0, 2.0])),
            chain_shape=None,
            method="exact",
        )


def test_a_chain_shape_must_multiply_to_the_draw_count():
    with pytest.raises(ValueError, match="chain_shape"):
        DrawsPosterior(
            draws=(named("theta", [1.0, 2.0, 3.0, 4.0]),),
            chain_shape=(3, 2),
            method="nuts",
        )


def test_iid_draws_say_so_with_none_rather_than_a_one_chain_shape():
    """``None`` is a different answer from ``(1, n)`` -- 'these draws have no
    chain structure' against 'they came from one chain' -- and §0.5 keeps it."""
    assert draws().chain_shape == (2, 2)
    assert (
        DrawsPosterior(
            draws=(named("theta", [1.0, 2.0]),), chain_shape=None, method="exact"
        ).chain_shape
        is None
    )


def test_predictive_latent_and_replicated_draws_share_one_draw_axis():
    with pytest.raises(ValueError, match="draw"):
        predictive_result(latent_draws=(named("theta", [1.0, 2.0]),))


# ----------------------------------------------- unavailable is None, not NaN


@pytest.mark.parametrize(
    ("factory", "field"),
    [
        (point_estimate_result, "objective"),
        (point_estimate_result, "gradient_norm"),
        (point_estimate_result, "residual"),
        (evidence_result, "log_evidence"),
        (evidence_result, "standard_error"),
    ],
)
def test_a_number_that_is_not_available_is_none_and_never_nan(factory, field):
    with pytest.raises(ValueError, match="finite"):
        factory(**{field: float("nan")})
    with pytest.raises(ValueError, match="finite"):
        factory(**{field: float("inf")})


def test_the_optional_numbers_accept_none():
    result = point_estimate_result(
        objective=None, gradient_norm=None, residual=None, iterations=None
    )
    assert result.objective is None
    assert (
        canonical_loads(canonical_dumps(result), expected=PointEstimateResult) == result
    )


def test_log_evidence_is_required_and_finite():
    """The one exception to 'unavailable is None': an evidence result whose
    log evidence is missing is not an evidence result."""
    assert evidence_result().log_evidence == -12.5
    with pytest.raises(ValueError, match="finite"):
        evidence_result(log_evidence=float("-inf"))


def test_a_negative_iteration_count_is_refused():
    with pytest.raises(ValueError):
        point_estimate_result(iterations=-1)


# ---------------------------------------------------------- pointwise density


def test_pointwise_availability_and_the_pointwise_array_agree():
    with pytest.raises(ValueError, match="pointwise"):
        posterior_result(pointwise_log_likelihood=None)
    with pytest.raises(ValueError, match="pointwise"):
        posterior_result(log_density_availability=LogDensityAvailability.JOINT)


def test_a_posterior_with_no_log_density_carries_no_pointwise_array():
    result = posterior_result(
        log_density_availability=LogDensityAvailability.NONE,
        pointwise_log_likelihood=None,
    )
    assert result.log_density_availability is LogDensityAvailability.NONE
    assert result.pointwise_log_likelihood is None


# --------------------------------------------------------- no runtime objects


def test_a_fitted_conditional_posterior_holds_references_not_an_estimator():
    fields = {field.name for field in dataclasses.fields(FittedConditionalPosterior)}
    assert fields == {
        "estimator_ref",
        "simulation_bank_ref",
        "training_run_id",
        "validation_report_refs",
    }
    with pytest.raises((TypeError, ArtifactCodecError)):
        FittedConditionalPosterior(
            estimator_ref=lambda x: x,
            simulation_bank_ref=None,
            training_run_id=None,
            validation_report_refs=(),
        )


@pytest.mark.parametrize("name", sorted(TASK_FACTORIES))
def test_a_callable_cannot_be_smuggled_into_an_option_table(name):
    option = "optimizer_options" if name == "point_estimate" else "backend_options"
    with pytest.raises(TypeError):
        TASK_FACTORIES[name](**{option: (("callback", lambda: None),)})


def test_a_simulation_task_has_no_quality_gate_field():
    """§0.4 gives four tasks a gate and simulation none. Written down because
    the five task classes are near-copies of each other, and a field that
    arrives by copy-paste is a field nobody decided on."""
    assert "quality_gate" not in {
        field.name for field in dataclasses.fields(SimulationTask)
    }
    for other in (PosteriorTask, EvidenceTask, PredictiveTask, PointEstimateTask):
        assert "quality_gate" in {field.name for field in dataclasses.fields(other)}


# ------------------------------------------------------------ parameter source


def test_a_parameter_source_is_one_of_exactly_three_things():
    assert set(ParameterSourceKind) == {
        ParameterSourceKind.PRIOR,
        ParameterSourceKind.FIXED,
        ParameterSourceKind.POSTERIOR_RESULT,
    }
    assert ParameterSource.prior().kind is ParameterSourceKind.PRIOR
    assert (
        ParameterSource.fixed((named("theta", [0.5], ("site",)),)).kind
        is ParameterSourceKind.FIXED
    )
    assert (
        ParameterSource.from_posterior_result(ref()).kind
        is ParameterSourceKind.POSTERIOR_RESULT
    )


def test_a_fixed_parameter_source_with_no_values_is_refused():
    with pytest.raises(ValueError):
        ParameterSource(kind=ParameterSourceKind.FIXED, values=())


def test_a_posterior_parameter_source_with_no_reference_is_refused():
    with pytest.raises(ValueError):
        ParameterSource(kind=ParameterSourceKind.POSTERIOR_RESULT, posterior_ref=None)


def test_a_prior_parameter_source_carries_no_values_and_no_reference():
    with pytest.raises(ValueError):
        ParameterSource(
            kind=ParameterSourceKind.PRIOR,
            values=(named("theta", [0.5], ("site",)),),
        )
    with pytest.raises(ValueError):
        ParameterSource(kind=ParameterSourceKind.PRIOR, posterior_ref=ref())


def test_a_posterior_parameter_source_points_at_a_result():
    with pytest.raises(ValueError, match="result"):
        ParameterSource.from_posterior_result(ref(ArtifactKind.PLAN))


# --------------------------------------------------------------- lineage kinds


def test_a_report_reference_points_at_an_evaluation_report():
    with pytest.raises(ValueError, match="evaluation"):
        posterior_result(report_refs=(ref(ArtifactKind.PLAN),))


def test_a_result_envelope_is_a_result_envelope():
    with pytest.raises(ValueError, match="result"):
        posterior_result(meta=meta(artifact_type=ArtifactKind.PLAN))


def test_a_predictive_result_points_at_the_posterior_it_came_from():
    with pytest.raises(ValueError, match="result"):
        predictive_result(source_posterior_ref=ref(ArtifactKind.EVALUATION_REPORT))


def test_a_conditioning_payload_is_fingerprinted_in_the_data_slot():
    """A payload digest in the task slot would answer 'the data changed' for a
    renamed solver, which is the confusion §0.3's slot table exists to stop."""
    with pytest.raises(ValueError, match="data"):
        predictive_task(conditioning_data=fingerprint(K.TASK, "y=[1,2,3]"))


# ----------------------------------------------------------- task fingerprints


def test_a_task_fingerprint_is_in_the_task_slot():
    assert task_fingerprint(posterior_task()).kind is FingerprintKind.TASK


def test_a_task_built_twice_is_one_task():
    """The sibling of every assertion below it: if two identical tasks already
    had two fingerprints, 'this field moves the fingerprint' would pass for
    every field, including the ones that must not move it."""
    for name, factory in TASK_FACTORIES.items():
        assert task_fingerprint(factory()) == task_fingerprint(factory()), name


@pytest.mark.parametrize("name", sorted(TASK_FACTORIES))
def test_identity_time_and_label_are_not_part_of_a_task_fingerprint(name):
    """§0.3: renaming a task is not a change of task."""
    original = TASK_FACTORIES[name]()
    relabelled = dataclasses.replace(
        original,
        meta=TaskMeta(
            task_id=str(uuid.uuid4()),
            schema_version=original.meta.schema_version,
            created_at="2030-01-01T00:00:00Z",
            label="a different label entirely",
        ),
    )
    assert relabelled.meta != original.meta
    assert task_fingerprint(relabelled) == task_fingerprint(original)


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (posterior_task, "solver_tolerance", 1e-4),
        (posterior_task, "ess_floor", 50.0),
        (posterior_task, "require_convergence", False),
        (posterior_task, "backend", "exact"),
        (posterior_task, "quality_gate", "posterior-gate@2"),
        (posterior_task, "budget", ComputeBudget(draws=8, warmup=2, chains=2)),
        (posterior_task, "backend_options", (("target_accept_prob", 0.95),)),
        (evidence_task, "repeat_count", 5),
        (evidence_task, "reconstruct_posterior", False),
        (predictive_task, "replicated_sites", ("y_new",)),
        (point_estimate_task, "estimand", Estimand.POSTERIOR_MEAN),
        (point_estimate_task, "names", None),
        (simulation_task, "observed_sites", ("z",)),
    ],
)
def test_a_statistical_field_moves_the_task_fingerprint(factory, field, value):
    assert task_fingerprint(factory(**{field: value})) != task_fingerprint(factory())


def test_two_task_kinds_with_the_same_options_have_different_fingerprints():
    """The kind is IN the payload: a posterior task and an evidence task that
    happen to agree field for field are still two different questions."""
    common = {
        "meta": new_task_meta(label="same"),
        "backend": "auto",
        "budget": ComputeBudget(),
        "backend_options": (),
        "quality_gate": None,
    }
    assert task_fingerprint(PosteriorTask(**common)) != task_fingerprint(
        EvidenceTask(**common)
    )


# -------------------------------------------------------- the codec's closure


def test_an_unregistered_look_alike_cannot_be_encoded():
    @dataclasses.dataclass(frozen=True, slots=True)
    class PosteriorTaskLookAlike:
        meta: TaskMeta
        backend: str

    with pytest.raises(ArtifactCodecError, match="not a registered artifact type"):
        canonical_dumps(PosteriorTaskLookAlike(meta=new_task_meta(), backend="auto"))


def test_a_payload_naming_an_unregistered_class_is_refused():
    payload = canonical_dumps(posterior_task())
    forged = payload.replace(
        b"bayesmith.artifacts.tasks.PosteriorTask",
        b"bayesmith.artifacts.tasks.SneakyTask",
    )
    assert forged != payload
    with pytest.raises(ArtifactCodecError, match="not a registered artifact type"):
        canonical_loads(forged)


def test_a_posterior_task_never_decodes_as_an_evidence_task():
    payload = canonical_dumps(posterior_task())
    with pytest.raises(ArtifactCodecError):
        canonical_loads(payload, expected=EvidenceTask)
    forged = payload.replace(
        b"bayesmith.artifacts.tasks.PosteriorTask",
        b"bayesmith.artifacts.tasks.EvidenceTask",
    )
    with pytest.raises(ArtifactCodecError, match="fields"):
        canonical_loads(forged)


def test_a_posterior_result_never_decodes_as_a_predictive_result():
    payload = canonical_dumps(posterior_result())
    with pytest.raises(ArtifactCodecError):
        canonical_loads(payload, expected=PredictiveResult)


@pytest.mark.parametrize("name", sorted(RESULT_FACTORIES))
def test_every_result_is_registered_under_its_own_module_and_name(name):
    result = RESULT_FACTORIES[name]()
    payload = canonical_dumps(result)
    assert f"bayesmith.artifacts.results.{type(result).__name__}".encode() in payload
