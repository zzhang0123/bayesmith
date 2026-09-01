"""Encoding a trained NeuralPosterior as a FittedConditionalPosterior.

R2 Task 7, and the one place R2 touches R1's frozen schema: a trained
:class:`bayesmith.amortize.NeuralPosterior` is an eqx.Module plus a static
callable plus standardization arrays, none of which has a canonical artifact
form.  It is encoded as a :class:`FittedConditionalPosterior` -- a reference
to a fitted estimator, not the estimator itself (§0.7) -- whose
`estimator_ref` points at the new :class:`ArtifactKind.ESTIMATOR`.

The estimator artifact is a canonical JSON envelope (the same `ArtifactFile`
transport `dump_artifact`/`load_artifact` already write) around an opaque
equinox leaf-serialization blob and a static manifest.  The artifacts layer
never sees JAX or equinox: the blob is `bytes` to the codec, and the
serialize/deserialize lives in `bayesmith.dispatch.amortized`.

Three instruments, one theme -- the reference, not the object:

* **round-trip** -- encode a trained posterior, decode it back, and the
  sampler agrees draw-for-draw;
* **the codec** -- a callable or a module cannot be smuggled into an
  artifact, which is exactly why the estimator is a reference;
* **the invalidation matrix** -- ESTIMATOR retires on the same inputs as
  RESULT (model / graph / data / task / compilation).
"""

from __future__ import annotations

import uuid

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.amortize import NeuralPosterior, train_posterior
from bayesmith.artifacts._codec import (
    ArtifactCodecError,
    canonical_dumps,
    dump_artifact,
    load_artifact,
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
    utc_timestamp,
)
from bayesmith.artifacts.identity import (
    ArtifactKind,
    FingerprintBundle,
    FingerprintKind,
    InvalidationPolicy,
    fingerprint,
)
from bayesmith.artifacts.results import FittedConditionalPosterior, SimulationResult
from bayesmith.artifacts.tasks import ParameterSource
from bayesmith.dispatch.amortized import (
    EstimatorArtifact,
    decode_estimator,
    fitted_conditional_posterior,
)

K = FingerprintKind

# --- the amortized problem, in numpy (mirrors tests/test_amortize.py) -------
M0 = 0.5
S0 = 2.0
SIGMA = 0.4
A = np.linspace(0.5, 1.5, 8)


def _draw_bank(key, n_simulations: int):
    theta_key, noise_key = jax.random.split(key)
    theta = M0 + S0 * jax.random.normal(theta_key, (n_simulations, 1))
    data = theta * jnp.asarray(A) + SIGMA * jax.random.normal(
        noise_key, (n_simulations, len(A))
    )
    return theta, data


# --------------------------------------------------------------- fixtures


def _bundle() -> FingerprintBundle:
    return FingerprintBundle(
        model_source=fingerprint(K.MODEL_SOURCE, "amortized_npe"),
        graph_structure=fingerprint(K.GRAPH_STRUCTURE, "none"),
        data=fingerprint(K.DATA, "bank"),
        task=fingerprint(K.TASK, "simulation"),
    )


def _meta(**overrides: object):
    fields: dict[str, object] = {
        "artifact_type": ArtifactKind.RESULT,
        "fingerprints": _bundle(),
        "producer": ProducerRef(package="bayesmith", version="0.6.2"),
        "summary": "prior simulation bank",
    }
    fields.update(overrides)
    return new_artifact_meta(**fields)


def _run(**overrides: object) -> RunRecord:
    fields: dict[str, object] = {
        "run_id": str(uuid.uuid4()),
        "plan_ref": ArtifactRef(
            artifact_id=str(uuid.uuid4()), revision=0, artifact_type=ArtifactKind.PLAN
        ),
        "fingerprints": _bundle(),
        "seed": SeedRecord(seed=0, key_algorithm="threefry2x32"),
        "dtype": "float64",
        "devices": (DeviceRecord(platform="cpu", device_kind="cpu", device_id=0),),
        "jax_config": (),
        "backend": BackendRef(name="bayesmith", version="0.6.2"),
        "budget": ComputeBudget(draws=256),
        "termination": TerminationRecord(reason=TerminationReason.COMPLETED),
        "timing": TimingRecord(
            started_at=utc_timestamp(), finished_at=utc_timestamp(), wall_clock_seconds=0.0
        ),
        "approximation": ApproximationRecord(
            representation_class=ApproximationClass.MONTE_CARLO,
            target_fidelity=TargetFidelity.EXACT,
        ),
        "warnings": (),
    }
    fields.update(overrides)
    return RunRecord(**fields)


def _named(name: str, value, dims: tuple[str, ...]) -> NamedArray:
    return NamedArray(name=name, value=np.asarray(value), dims=dims)


def simulation_bank(thetas, data) -> tuple[SimulationResult, ArtifactRef]:
    """The prior-simulation bank as a SimulationResult, and its reference."""
    result = SimulationResult(
        meta=_meta(),
        run=_run(),
        parameter_source=ParameterSource.prior(),
        latent_draws=(_named("theta", np.asarray(thetas), ("draw", "component")),),
        observation_draws=(
            _named("y", np.asarray(data), ("draw", "observation")),
        ),
    )
    ref = ArtifactRef(
        artifact_id=result.meta.artifact_id,
        revision=result.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )
    return result, ref


@pytest.fixture(scope="module")
def trained():
    """One small training run, reused.  The point is the ENCODING, not the fit."""
    theta, data = _draw_bank(jax.random.key(0), 256)
    start = NeuralPosterior.create(
        theta, data, key=jax.random.key(1), n_components=1, width=16, depth=1
    )
    fitted, history = train_posterior(
        start, theta, data, key=jax.random.key(2), n_steps=10, batch_size=64
    )
    return fitted, theta, data, history


# ---------------------------------------------------------------- round trip


def test_a_trained_posterior_encodes_to_a_fitted_conditional_reference(trained):
    posterior, theta, data, _history = trained
    _bank, bank_ref = simulation_bank(theta, data)
    run_id = str(uuid.uuid4())
    report = ArtifactRef(
        artifact_id=str(uuid.uuid4()), revision=0, artifact_type=ArtifactKind.EVALUATION_REPORT
    )

    fitted, artifact = fitted_conditional_posterior(
        posterior,
        simulation_bank_ref=bank_ref,
        training_run_id=run_id,
        validation_report_refs=(report,),
    )

    assert isinstance(fitted, FittedConditionalPosterior)
    assert fitted.estimator_ref.artifact_type is ArtifactKind.ESTIMATOR
    assert fitted.simulation_bank_ref == bank_ref
    assert fitted.training_run_id == run_id
    assert fitted.validation_report_refs == (report,)
    assert isinstance(artifact, EstimatorArtifact)
    assert artifact.artifact_id == fitted.estimator_ref.artifact_id


def test_the_estimator_round_trips_and_the_sampler_agrees(trained):
    posterior, theta, data, _history = trained
    _bank, bank_ref = simulation_bank(theta, data)
    _fitted, artifact = fitted_conditional_posterior(
        posterior,
        simulation_bank_ref=bank_ref,
        training_run_id=str(uuid.uuid4()),
    )

    restored = decode_estimator(artifact)
    datum = data[3]
    key = jax.random.key(7)
    before = posterior.sample(datum, key, n_samples=50)
    after = restored.sample(datum, key, n_samples=50)
    assert jnp.array_equal(before, after)


def test_the_estimator_artifact_persists_through_the_artifact_file_envelope(
    trained, tmp_path
):
    posterior, theta, data, _history = trained
    _bank, bank_ref = simulation_bank(theta, data)
    _fitted, artifact = fitted_conditional_posterior(
        posterior,
        simulation_bank_ref=bank_ref,
        training_run_id=str(uuid.uuid4()),
    )

    path = tmp_path / "estimator.json"
    dump_artifact(artifact, path)
    loaded = load_artifact(path, expected=EstimatorArtifact)
    restored = decode_estimator(loaded)

    datum = data[3]
    key = jax.random.key(7)
    assert jnp.array_equal(
        posterior.sample(datum, key, n_samples=50),
        restored.sample(datum, key, n_samples=50),
    )


# ------------------------------------------------------- the codec's refusal


def test_a_callable_cannot_enter_an_artifact():
    with pytest.raises(ArtifactCodecError):
        canonical_dumps({"embed": lambda x: x})


def test_a_module_cannot_enter_an_artifact():
    import math

    with pytest.raises(ArtifactCodecError):
        canonical_dumps(math)


# ------------------------------------------------------ the invalidation row


def test_the_estimator_row_invalidates_on_model_graph_data_and_task():
    policy = InvalidationPolicy.default()
    sensitive = (
        K.MODEL_SOURCE,
        K.GRAPH_STRUCTURE,
        K.DATA,
        K.TASK,
        K.COMPILATION,
    )
    for kind in sensitive:
        assert policy.affected(ArtifactKind.ESTIMATOR, frozenset({kind})) is True
    for kind in (K.EVALUATION, K.ENVIRONMENT):
        assert policy.affected(ArtifactKind.ESTIMATOR, frozenset({kind})) is False
