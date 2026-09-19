"""A small NUTS tutorial; short chains demonstrate the API, not convergence."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

import bayesmith as bs
from bayesmith.artifacts import (
    ComputeBudget,
    PosteriorTask,
    Refusal,
    model_ref_from_callable,
    new_task_meta,
)

DATA = jnp.array([1.02, 0.57, 0.30, 0.21, 0.12])


def model(data):
    rate = bs.sample("rate", lambda: dist.LogNormal(-0.5, 0.4))
    mean = bs.det("mean", lambda r: jnp.exp(-r * jnp.arange(5.0)), rate)
    bs.observe("signal", lambda m: dist.Normal(m, 0.1), mean, obs=data)


def main():
    graph = bs.trace(model, DATA)
    print(bs.compile(graph))
    task = PosteriorTask(
        meta=new_task_meta(label="decay rate"),
        budget=ComputeBudget(draws=100, warmup=100, chains=1),
        require_convergence=False,  # tutorial budget only; inspect real runs separately
    )
    planned = bs.compile_task(
        graph,
        task,
        model_ref=model_ref_from_callable(model, identifier="decay"),
        key=jax.random.key(1),
    )
    if isinstance(planned, Refusal):
        raise SystemExit(planned.failed_premise)
    result = bs.execute_task(planned, key=jax.random.key(2))
    if isinstance(result, Refusal):
        raise SystemExit(result.failed_premise)
    print(result.run.backend)
    print(result.run.termination)


if __name__ == "__main__":
    main()
