"""NUTS through the bridge, against a posterior that is known in closed form.

Normal-normal conjugacy: x ~ N(0, tau^2), d_i ~ N(x, sigma^2), i = 1..N gives

    var_post  = 1 / (1/tau^2 + N/sigma^2)
    mean_post = var_post * sum(d) / sigma^2

Everything upstream of this file is self-consistent by construction; this is
where the package first has to be *right*.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.diagnostics import effective_sample_size

from bayesmith.bridge.numpyro_bridge import nuts
from bayesmith.graph.trace import observe, sample, trace

TAU = 2.0
SIGMA = 0.5


def _graph(data):
    def model():
        x = sample("x", lambda: dist.Normal(0.0, TAU))
        observe("d", lambda v: dist.Normal(v, SIGMA), x, obs=data)

    return trace(model)


def _analytic_posterior(data):
    n = data.size
    var = 1.0 / (1.0 / TAU**2 + n / SIGMA**2)
    mean = var * jnp.sum(data) / SIGMA**2
    return float(mean), float(jnp.sqrt(var))


@pytest.mark.parametrize("n", [1, 20, 200])
def test_nuts_recovers_the_analytic_posterior(n):
    """Includes n=1, where the prior still dominates -- the awkward end."""
    data = jnp.linspace(0.5, 2.5, n)
    draws = nuts(_graph(data), jax.random.key(0), num_warmup=1000, num_samples=4000)

    mean_hat = float(jnp.mean(draws["x"]))
    sd_hat = float(jnp.std(draws["x"]))
    mean_true, sd_true = _analytic_posterior(data)

    # Compare the mean on the scale of its own Monte-Carlo error.
    ess = float(effective_sample_size(np.asarray(draws["x"])[None, :]))
    assert ess > 400, f"chain too autocorrelated to judge: ESS={ess:.0f}"
    z = abs(mean_hat - mean_true) / (sd_true / np.sqrt(ess))
    assert z < 4.0, f"posterior mean off by {z:.1f} sigma (n={n})"
    assert abs(sd_hat - sd_true) / sd_true < 0.1, (
        f"posterior sd {sd_hat:.4f} vs analytic {sd_true:.4f} (n={n})"
    )


def test_two_seeds_agree_within_monte_carlo_error():
    """A wrong graph often shows up as seed-dependent answers."""
    data = jnp.linspace(0.5, 2.5, 20)
    a = nuts(_graph(data), jax.random.key(0), num_warmup=1000, num_samples=4000)
    b = nuts(_graph(data), jax.random.key(1), num_warmup=1000, num_samples=4000)
    _, sd_true = _analytic_posterior(data)
    assert abs(float(jnp.mean(a["x"])) - float(jnp.mean(b["x"]))) < 0.2 * sd_true
