"""Toy graphs the exact-solve tests share.

Kept in one place so a change to a toy model cannot make two test modules
quietly disagree about what they are testing.

**Numbers are chosen to be pairwise distinct.** In every model below the true
value, the noise width, the prior width and the prior centre are all
different numbers. P1's Task 4 lost its only guard on this package's most
important guarantee to a fixture where `w = 3.0` and the true gradient
`sum(X) = 3.0` happened to agree, so a broken implementation returning the
parameter instead of its gradient passed anyway.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import const, det, observe, plate, sample, trace


class LyingNormal(dist.Normal):
    """A Normal whose ``log_prob`` is not the one its own loc/scale imply.

    Exists to make :func:`~bayesmith.exact.gaussian.check_gaussian`'s probe
    testable: introspection reads ``(loc, scale)`` off the type and is
    perfectly happy, and only evaluating ``log_prob`` reveals the
    disagreement. Not a contrived shape -- any ``Distribution`` subclass that
    overrides ``log_prob`` (a censored likelihood, a tempered one, a
    hand-written approximation) lands exactly here.
    """

    def log_prob(self, value):
        return 1.5 * super().log_prob(value)


def straight_line(*, n=8, weight=2.5, sigma=0.5, prior_std=2.0, prior_mean=0.0, seed=0):
    """``d ~ N(w X, sigma)``, ``w ~ N(prior_mean, prior_std)``.

    One linear latent, one observed node, no plate, sigma constant. The
    smallest graph the exact path applies to.
    """
    x = jnp.linspace(1.0, 4.0, n)
    data = weight * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(prior_mean, prior_std))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def two_linear_latents(*, n=12, slope=1.5, intercept=-3.0, sigma=0.4, seed=1):
    """``d ~ N(a X + b, sigma)``. Two latents that must be solved JOINTLY."""
    x = jnp.linspace(-2.0, 2.0, n)
    data = slope * x + intercept + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, 5.0))
        b = sample("b", lambda: dist.Normal(0.0, 7.0))
        mu = det("mu", lambda a_, b_, x_: a_ * x_ + b_, a, b, xs, linear_in=("a", "b"))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def bilinear_pair(*, n=10, sigma=0.3, seed=2):
    """``mu = gain * t_ant * X`` -- affine in EACH, not affine in the PAIR.

    The declaration ``linear_in=("gain", "t_ant")`` is therefore false, and
    `Graph.__check_init__` cannot see it: both names really are parents. Only
    the joint affinity probe can. This is rheplicant's motivating failure --
    a hand-rolled alternating solve here lands thousands of kelvin away while
    the CG residual reads 1e-7 and every per-block condition number reads ~1.5.
    """
    x = jnp.linspace(0.5, 3.0, n)
    data = 2.0 * 1.5 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        g = sample("gain", lambda: dist.Normal(1.0, 1.0))
        t = sample("t_ant", lambda: dist.Normal(2.0, 3.0))
        mu = det(
            "mu", lambda g_, t_, x_: g_ * t_ * x_, g, t, xs, linear_in=("gain", "t_ant")
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def quadratic_claim(*, n=6, sigma=0.5, seed=3):
    """``mu = w**2 X`` declared ``linear_in=("w",)`` -- a false single claim."""
    x = jnp.linspace(1.0, 2.0, n)
    data = 4.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 2.0))
        mu = det("mu", lambda w_, x_: w_**2 * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def two_observations(*, n=7, m=5, weight=1.25, s1=0.3, s2=0.9, seed=4):
    """One latent constrained by TWO observed nodes -- the pytree codomain.

    rheplicant's codomain is a single array because a pipeline has one
    output. A graph can have several observed nodes, so every reduction in
    the solve runs over a dict of leaves rather than one array.
    """
    k1, k2 = jax.random.split(jax.random.key(seed))
    x1 = jnp.linspace(1.0, 3.0, n)
    x2 = jnp.linspace(-1.0, 1.0, m)
    d1 = weight * x1 + s1 * jax.random.normal(k1, (n,))
    d2 = 2.0 * weight * x2 + s2 * jax.random.normal(k2, (m,))

    def model():
        a = const("X1", x1)
        b = const("X2", x2)
        w = sample("w", lambda: dist.Normal(0.0, 4.0))
        m1 = det("mu1", lambda w_, x_: w_ * x_, w, a, linear_in=("w",))
        m2 = det("mu2", lambda w_, x_: 2.0 * w_ * x_, w, b, linear_in=("w",))
        observe("d1", lambda u: dist.Normal(u, s1), m1, obs=d1)
        observe("d2", lambda u: dist.Normal(u, s2), m2, obs=d2)

    return trace(model)


def plated_latent(*, n=6, sigma=0.4, tau=1.5, seed=5):
    """``z_i ~ N(0, tau)``, ``d_i ~ N(z_i, sigma)`` under one plate.

    The observed node's loc is the latent ITSELF, with no deterministic node
    on the path -- so the "every Deterministic on the path declares
    linear_in" rule holds vacuously, which the exact path must accept rather
    than trip over.
    """
    key = jax.random.key(seed)
    truth = tau * jax.random.normal(key, (n,))
    data = truth + sigma * jax.random.normal(jax.random.fold_in(key, 1), (n,))

    def model():
        obs = plate("obs", n)
        z = sample("z", lambda: dist.Normal(0.0, tau), plate=obs)
        observe("d", lambda z_: dist.Normal(z_, sigma), z, plate=obs, obs=data)

    return trace(model)


def radiometer(*, n=10, weight=3.0, kappa=0.05, floor=1e-3, seed=6):
    """``sigma_i = kappa |mu_i| + floor`` -- sigma tracks the prediction.

    The GLS / correction case. ``floor`` keeps sigma strictly positive where
    the prediction crosses zero, which the probe guard requires.
    """
    x = jnp.linspace(1.0, 5.0, n)
    truth = weight * x
    data = truth + (kappa * jnp.abs(truth) + floor) * jax.random.normal(
        jax.random.key(seed), (n,)
    )

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 10.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, kappa * jnp.abs(m) + floor),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


def indirect_ancestor(*, n=6, sigma=0.5, seed=8):
    """`tau` reaches `x`'s prior through a deterministic node, not directly.

    A direct-parent ancestry check passes this and is wrong: x's parents are
    ("width",), and `width` is a function of `tau`.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        width = det("width", lambda t: jnp.abs(t) + 0.1, tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def shared_ancestor(*, n=6, sigma=0.5, seed=7):
    """``tau`` is a latent AND an ancestor of the latent ``x``.

    Both nodes are Gaussian, so a naive classifier would put them in one
    block -- and the pair's joint distribution is not Gaussian, because x's
    own width is a function of tau. The block builder must refuse the pair.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        x = sample("x", lambda t: dist.Normal(0.0, jnp.abs(t)), tau)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)
