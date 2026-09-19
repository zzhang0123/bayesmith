"""GCR draws -> full non-Gaussian posterior weights -> hyperparameter MAP.

Run from the repository root:
    .venv/bin/python examples/inference/non_gaussian_reweight.py

The mixture fraction is estimated after integrating out the latent field.
The independent reference is the analytic Gaussian-mixture convolution.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import compile, observe, sample, trace
from bayesmith.reweight import PosteriorReweighting


def make_graph(data, logit=None):
    def model():
        if logit is None:
            prior = dist.Normal(jnp.zeros_like(data), 1.5)
        else:
            prior = dist.MixtureSameFamily(
                dist.Categorical(logits=jnp.stack([jnp.zeros_like(logit), logit])),
                dist.Normal(jnp.array([-1.0, 1.0]), 0.6),
            ).expand(data.shape)
        field = sample("field", lambda: prior)
        observe("data", lambda x: dist.Normal(x, 0.35), field, obs=data)

    return trace(model)


def main():
    data = jnp.array([-0.8, -0.6, 0.3, 0.9, 1.2, 0.7, 1.1, -0.9])
    reference = make_graph(data)
    plan = compile(reference)
    if (
        plan.sampled is not None
        or plan.exact is None
        or plan.exact.method != "gcr"
        or plan.sigma_needs_rebuild
    ):
        raise RuntimeError(
            "this example requires a fixed linear-Gaussian GCR reference"
        )
    posterior = plan.sample(jax.random.key(24), num_samples=12000)
    problem = PosteriorReweighting.from_graphs(
        reference,
        posterior.samples,
        lambda params: make_graph(data, params["logit"]),
    )

    # MAP is defined in the mixture fraction a, optimized via a=sigmoid(u).
    # No change-of-density Jacobian: adding it would find the mode in u.
    def hyperprior(params):
        return dist.Beta(2.0, 2.0).log_prob(jax.nn.sigmoid(params["logit"]))

    fitted = problem.fit(
        {"logit": jnp.array(0.0)},
        log_hyperprior=hyperprior,
        steps=350,
        learning_rate=0.03,
        min_ess=1000,
    )
    fraction = jax.nn.sigmoid(fitted.parameters["logit"])
    grid = jnp.linspace(0.01, 0.99, 2001)
    component = dist.Normal(jnp.array([-1.0, 1.0]), jnp.sqrt(0.6**2 + 0.35**2))
    likelihood = component.log_prob(data[:, None])
    exact = jnp.sum(
        jnp.logaddexp(
            jnp.log1p(-grid[:, None]) + likelihood[:, 0],
            jnp.log(grid[:, None]) + likelihood[:, 1],
        ),
        axis=1,
    ) + dist.Beta(2.0, 2.0).log_prob(grid)

    validation_draws = plan.sample(jax.random.key(25), num_samples=12000)
    validation = PosteriorReweighting.from_graphs(
        reference,
        validation_draws.samples,
        lambda params: make_graph(data, params["logit"]),
    )
    checked = validation.estimate(fitted.parameters)
    print("Reference sampling method:", posterior.method)
    print("Non-Gaussian mixture fraction MAP:", float(fraction))
    print("Analytic marginal MAP (grid):", float(grid[jnp.argmax(exact)]))
    print(
        "Training / fresh-bank Kish ESS:",
        float(fitted.importance.ess),
        float(checked.ess),
    )
    print(
        "Training / fresh-bank log evidence ratio:",
        float(fitted.importance.log_mean_weight),
        float(checked.log_mean_weight),
    )
    print("Final gradient norm:", float(fitted.gradient_norm))
    print(
        "Plug-in posterior field mean:",
        jnp.sum(
            fitted.importance.weights[:, None] * posterior.samples["field"],
            axis=0,
        ),
    )


if __name__ == "__main__":
    main()
