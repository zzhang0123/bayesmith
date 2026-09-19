"""Run with JAX_ENABLE_X64=1 python five_tasks.py."""

from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

import bayesmith as bs
from bayesmith.artifacts import (
    ArtifactKind,
    ArtifactRef,
    ComputeBudget,
    Estimand,
    EvidenceResult,
    EvidenceTask,
    ParameterSource,
    ParameterSourceKind,
    PointEstimateResult,
    PointEstimateTask,
    PosteriorResult,
    PosteriorTask,
    PredictiveResult,
    PredictiveTask,
    Refusal,
    SimulationResult,
    SimulationTask,
    dump_artifact,
    load_artifact,
    model_ref_from_callable,
    new_task_meta,
)


def model(data):
    level = bs.sample("level", lambda: dist.Normal(0.0, 2.0))
    bs.observe("signal", lambda x: dist.Normal(x, 0.5), level, obs=data)


def main():
    graph = bs.trace(model, jnp.array([0.8, 1.0, 1.2]))
    model_ref = model_ref_from_callable(model, identifier="level-example")
    keys = iter(jax.random.split(jax.random.key(7), 10))

    def run(task, expected, source=None):
        planned = bs.compile_task(graph, task, model_ref=model_ref, key=next(keys))
        if isinstance(planned, Refusal):
            raise SystemExit(planned.failed_premise)
        result = bs.execute_task(planned, key=next(keys), source_posterior=source)
        if isinstance(result, Refusal):
            raise SystemExit(result.failed_premise)
        assert isinstance(result, expected)
        print(type(result).__name__)
        return result

    # 1. Posterior: retain uncertainty about the latent level.
    posterior = run(
        PosteriorTask(
            meta=new_task_meta(label="level posterior"),
            budget=ComputeBudget(draws=256),
        ),
        PosteriorResult,
    )

    # 2. Point estimate: ask explicitly for the posterior mean.
    point = run(
        PointEstimateTask(
            meta=new_task_meta(label="level mean"),
            estimand=Estimand.POSTERIOR_MEAN,
        ),
        PointEstimateResult,
    )
    print({array.name: array.value for array in point.values})

    # 3. Predictive: replicate the measurements using this posterior.
    source_ref = ArtifactRef(
        artifact_id=posterior.meta.artifact_id,
        revision=posterior.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )
    predictive = run(
        PredictiveTask(
            meta=new_task_meta(label="replicate signal"),
            source_posterior_ref=source_ref,
            conditioned_sites=("signal",),
            replicated_sites=("signal",),
            latent_sites=("level",),
        ),
        PredictiveResult,
        source=posterior,
    )
    print(predictive.replicated_draws[0].value.shape)

    # 4. Simulation: sample from the declared prior, then generate data.
    run(
        SimulationTask(
            meta=new_task_meta(label="prior simulation"),
            parameter_source=ParameterSource(kind=ParameterSourceKind.PRIOR),
            latent_sites=("level",),
            observed_sites=("signal",),
            budget=ComputeBudget(draws=32),
        ),
        SimulationResult,
    )

    # 5. Evidence: x64 and a fully normalized linear-Gaussian model.
    evidence = run(
        EvidenceTask(meta=new_task_meta(label="analytic evidence")), EvidenceResult
    )
    print("log p(data):", evidence.log_evidence)

    # Store data artifacts, never a live Graph or compiled executable.
    with TemporaryDirectory() as directory:
        path = Path(directory) / "posterior.json"
        dump_artifact(posterior, path)
        restored = load_artifact(path)
        assert isinstance(restored, PosteriorResult)
        assert restored.meta.artifact_id == posterior.meta.artifact_id
    return graph, model_ref, posterior, predictive, evidence


if __name__ == "__main__":
    main()
