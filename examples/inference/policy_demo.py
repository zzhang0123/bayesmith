"""Automatic preflight + initialization + continuing checkpoint sampling.

Run: python -m examples.inference.policy_demo
Outputs are separate from the five saved gallery runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import compile_task, det, execute_task, observe, sample, trace
from bayesmith.artifacts import (
    ComputeBudget,
    PosteriorTask,
    Refusal,
    StoppingPolicy,
    dump_artifact,
    model_ref_from_callable,
    new_task_meta,
)

from .common import simulate_fixed


def model(data):
    log_amplitude = sample("log_amplitude", lambda: dist.Normal(0.0, 0.7))
    mean = det("mean", jnp.exp, log_amplitude)
    observe("obs", lambda m: dist.Normal(m, 0.2), mean, obs=data)


def run(directory=Path("runs/inference-policy-demo")):
    directory.mkdir(parents=True, exist_ok=True)
    with jax.enable_x64():
        truth = 0.35
        # Truth is used only for forward simulation and the final recovery check.
        data, simulation = simulate_fixed(
            trace(model, jnp.zeros(32)),
            model,
            {"log_amplitude": truth},
            jax.random.key(50),
        )
        task = PosteriorTask(
            meta=new_task_meta(label="Automatic sampling policies"),
            budget=ComputeBudget(draws=1200, warmup=300, chains=2),
            stopping=StoppingPolicy(
                mode="checkpoints", min_draws=300, batch_size=300, mcse_mean=0.01
            ),
        )
        planned = compile_task(
            trace(model, data),
            task,
            model_ref=model_ref_from_callable(model, identifier="sampling-policy-demo"),
        )
        if isinstance(planned, Refusal):
            raise RuntimeError(planned.meta.summary)  # noqa: TRY004 -- capability refusal
        posterior = execute_task(planned, key=jax.random.key(51))
        draws = posterior.representation.draws[0].value
        interval = np.quantile(draws, [0.005, 0.995])
        recovered = bool(interval[0] <= truth <= interval[1])
        report = {
            "model": "y ~ Normal(exp(log_amplitude), 0.2)",
            "truth": truth,
            "posterior_mean": float(draws.mean()),
            "interval_99": interval.tolist(),
            "recovered": recovered,
            "termination": posterior.run.termination.reason.value,
            "sampling": dict(posterior.run.sampling_details),
            "initial_values": {
                a.name: a.value.tolist() for a in posterior.run.initial_values
            },
            "diagnostics": [
                {
                    "code": f.code,
                    "status": f.conclusion,
                    "measurements": dict(f.measurements),
                }
                for f in planned.analysis.findings
            ],
        }
        for name, artifact in (
            ("task", task),
            ("analysis", planned.analysis),
            ("simulation", simulation),
            ("posterior", posterior),
        ):
            dump_artifact(artifact, directory / f"{name}.artifact.json")
        (directory / "result.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    key: report[key]
                    for key in (
                        "truth",
                        "posterior_mean",
                        "interval_99",
                        "recovered",
                        "termination",
                    )
                },
                indent=2,
            )
        )
        if not recovered:
            raise RuntimeError(
                "Simulation truth was not recovered; the failed result is saved."
            )
        return report


if __name__ == "__main__":
    run()
