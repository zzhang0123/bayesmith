"""The plans on the Overview page. Run with: JAX_ENABLE_X64=1 python overview_plan.py"""

import jax.numpy as jnp
import numpyro.distributions as dist

import bayesmith as bs

A = jnp.array([[1.0, 0.3], [0.2, 1.0], [0.5, -0.4], [1.0, 1.0]])
DATA = jnp.array([0.4, 1.1, -0.2, 1.5])


def mixed(data):
    """Affine in x once nu is fixed; not affine in nu. Nothing is declared."""
    x = bs.sample("x", lambda: dist.Normal(jnp.zeros(2), 2.0).to_event(1))
    nu = bs.sample("nu", lambda: dist.LogNormal(0.0, 0.5))
    mu = bs.det("mu", lambda x_, nu_: (A @ x_) * jnp.exp(0.1 * nu_), x, nu)
    bs.observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=data)


def curved(data):
    """Declared affine in x, but exp(A x) is not."""
    x = bs.sample("x", lambda: dist.Normal(jnp.zeros(2), 2.0).to_event(1))
    mu = bs.det("mu", lambda x_: jnp.exp(A @ x_), x, linear_in=("x",))
    bs.observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=data)


if __name__ == "__main__":
    for model in (mixed, curved):
        print(f"# {model.__name__}")
        print(bs.compile(bs.trace(model, DATA)))
