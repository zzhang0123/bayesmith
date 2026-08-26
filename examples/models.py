"""The two documented models, buildable at any noise realisation.

One definition, three consumers -- ``three_routes.py``, ``hierarchy.py`` and
``validate_sampling.py`` -- so the model the docs show, the model the demos
run and the model the validation experiment replicates cannot drift apart.
Everything numerical matches ``docs/factor-partition-examples.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import det, observe, sample, trace

N = 32
XI = jnp.linspace(-1.0, 1.0, N)
A = jnp.stack([XI, XI**2 - jnp.mean(XI**2)], axis=1)
B = jnp.stack([jnp.ones(N), XI, XI**2], axis=1)
C = jnp.stack([XI, XI**3], axis=1)
D = jnp.eye(3)
F = 4.05e-3  # 1 / sqrt(61 kHz x 1 s)

X_TRUE = jnp.array([0.3, -0.2])
Y_TRUE = jnp.array([5.0, 1.0, 0.8])
Z_TRUE = jnp.array([0.5, -0.3])
W1_PRIOR_STD = 0.3


def _drawn_truths(key: jax.Array, spec: dict) -> dict:
    """One truth per latent, drawn from that latent's own prior.

    ``spec`` is ``{name: (centre, width)}``. Proper calibration draws the
    truth from the prior: coverage of the posterior's intervals is then
    exactly nominal for a correct sampler, with no dependence on how
    informative the prior happens to be -- where a FIXED truth at the prior's
    centre makes the posterior shrink toward it and over-cover, which the
    validation experiment's first registered run measured (pull rms 0.41
    against the >= 0.5 band) before this function existed.
    """
    return {
        name: centre + width * jax.random.normal(
            jax.random.fold_in(key, index), jnp.shape(centre)
        )
        for index, (name, (centre, width)) in enumerate(sorted(spec.items()))
    }


def three_routes(key: jax.Array, randomize_truth: bool = False):
    """Example 1: ``d = exp(Ax) [B y + exp(C z)] (1 + F w)``.

    Returns ``(graph, truths)`` with the data realised at ``key``.
    ``randomize_truth`` draws the truths from the priors (the calibration
    regime); the default keeps the documented showcase truths, so the demo
    scripts print the numbers the docs page shows.
    """
    key_truth, key = jax.random.split(key)
    if randomize_truth:
        truths = _drawn_truths(key_truth, {
            "x": (jnp.zeros(2), 0.5),
            "y": (Y_TRUE, 0.5),
            "z": (jnp.zeros(2), 0.5),
        })
        x_true, y_true, z_true = truths["x"], truths["y"], truths["z"]
    else:
        x_true, y_true, z_true = X_TRUE, Y_TRUE, Z_TRUE
    truth = jnp.exp(A @ x_true) * (B @ y_true + jnp.exp(C @ z_true))
    data = truth * (1.0 + F * jax.random.normal(key, (N,)))

    def model(observed_data):
        x = sample("x", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
        # Width 0.5 keeps the summed sky positive over the prior's bulk,
        # which is what makes "log-linear in x" a property of the model
        # rather than of the probe's luck -- see the docs page.
        y = sample("y", lambda: dist.Normal(Y_TRUE, 0.5).to_event(1))
        z = sample("z", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
        s = det("s", lambda y_, z_: B @ y_ + jnp.exp(C @ z_), y, z, linear_in=("y",))
        mu = det("mu", lambda x_, s_: jnp.exp(A @ x_) * s_, x, s, linear_in=("s",))
        observe("d", lambda m: dist.Normal(m, F * m), mu, obs=observed_data)

    return trace(model, data), {"x": x_true, "y": y_true, "z": z_true}


def hierarchy(key: jax.Array, kind: str = "linear", randomize_truth: bool = False):
    """Example 2: ``d = exp(Ax) [B w1 + exp(C z)] (1 + F w2)``, ``w1 ~ p(. | y)``.

    ``kind`` selects how ``y`` parameterises the field's statistics --
    ``"linear"`` (the mean, ``w1 ~ N(D y, 0.3)``) or ``"nonlinear"`` (the
    scale, through an exponential). The field's truth is one realisation of
    its own prior at ``y``'s truth, drawn from ``key``, so a replication
    study exercises the hierarchy rather than a fixed pseudo-truth.

    Returns ``(graph, truths)`` -- ``truths["w1"]`` is that realisation.
    """
    key_truth, key_field, key_noise = jax.random.split(key, 3)
    if randomize_truth:
        drawn = _drawn_truths(key_truth, {
            "x": (jnp.zeros(2), 0.5),
            "y": (Y_TRUE, 0.5),
            "z": (jnp.zeros(2), 0.5),
        })
        x_true, y_true, z_true = drawn["x"], drawn["y"], drawn["z"]
    else:
        x_true, y_true, z_true = X_TRUE, Y_TRUE, Z_TRUE
    # The field's truth is ALWAYS a draw from its own prior at y's truth --
    # that level of the hierarchy is calibrated in every regime.
    w1_true = y_true + W1_PRIOR_STD * jax.random.normal(key_field, (3,))
    truth = jnp.exp(A @ x_true) * (B @ w1_true + jnp.exp(C @ z_true))
    data = truth * (1.0 + F * jax.random.normal(key_noise, (N,)))

    def model(observed_data):
        # Width 0.5, matching Example 1's positivity argument: under
        # prior-drawn truths a width-1.0 hyperprior admits skies that dip
        # negative, where the log route genuinely fails and the partition
        # would (rightly) flip between replications.
        y = sample("y", lambda: dist.Normal(Y_TRUE, 0.5).to_event(1))
        if kind == "linear":
            w1 = sample(
                "w1", lambda y_: dist.Normal(D @ y_, W1_PRIOR_STD).to_event(1), y
            )
        elif kind == "nonlinear":
            w1 = sample(
                "w1",
                lambda y_: dist.Normal(
                    Y_TRUE, W1_PRIOR_STD * jnp.exp(0.2 * (y_ - Y_TRUE))
                ).to_event(1),
                y,
            )
        else:
            raise ValueError(f"kind must be 'linear' or 'nonlinear', got {kind!r}")
        x = sample("x", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
        z = sample("z", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
        s = det(
            "s", lambda w_, z_: B @ w_ + jnp.exp(C @ z_), w1, z, linear_in=("w1",)
        )
        mu = det("mu", lambda x_, s_: jnp.exp(A @ x_) * s_, x, s, linear_in=("s",))
        observe("d", lambda m: dist.Normal(m, F * m), mu, obs=observed_data)

    return trace(model, data), {
        "x": x_true,
        "w1": w1_true,
        "y": y_true,
        "z": z_true,
    }
