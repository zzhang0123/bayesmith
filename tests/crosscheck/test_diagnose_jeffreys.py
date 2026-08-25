"""JeffreysPrior, both packages, one block: §四 4.2's priors row.

The acceptance names one number -- the radiometer flat constant
``+15.80169853`` -- and one discipline, ``eigh`` plus the rank floor. Both
are checked here against rheplicant's LIVE evaluation rather than only
against the constant: the constant catches the two packages drifting
together, the live comparison catches either drifting alone. The noise
models are matched by construction (``RadiometerNoise(channel_width=1e6,
integration_time=1)`` has fractional level 1e-3; the graph declares
``Normal(mu, 1e-3 |mu|)``), and the forward functions compute the same
power law -- rheplicant's through a values-dict callable, bayesmith's
through the graph.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.crosscheck

N_TIME, N_FREQ = 8, 8
NU0 = 70e6
F = 1e-3  # 1 / sqrt(channel_width * integration_time), channel_width = 1e6

GRID = [(la, be) for la in (6.8, 7.8, 8.8) for be in (2.05, 2.55, 3.05)]


def _rheplicant_prior():
    from rheplicant.inference.priors import JeffreysPrior

    return JeffreysPrior(over=("fg_log_amp", "fg_beta"))


def _rheplicant_forward(floor):
    def forward(values):
        nu = jnp.linspace(60e6, 85e6, N_FREQ) / NU0
        row = jnp.exp(values["fg_log_amp"]) * nu ** (-values["fg_beta"]) + floor
        return jnp.broadcast_to(row, (N_TIME, N_FREQ))

    return forward


def _radiometer():
    from rheplicant.inference.noise import RadiometerNoise

    return RadiometerNoise(channel_width=1e6, integration_time=1.0)


def _homoscedastic():
    from rheplicant.inference.noise import HomoscedasticNoise

    return HomoscedasticNoise(jnp.array(0.5))


def _bayesmith_graph(floor=0.0, noise="radiometer"):
    import numpyro.distributions as dist

    from bayesmith import det, observe, sample, trace

    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    data = jnp.broadcast_to(
        jnp.exp(7.8) * (freq / NU0) ** (-2.55) + floor, (N_TIME, N_FREQ)
    )

    def model():
        la = sample(
            "fg_log_amp", lambda: dist.ImproperUniform(dist.constraints.real, (), ())
        )
        be = sample(
            "fg_beta", lambda: dist.ImproperUniform(dist.constraints.real, (), ())
        )
        pred = det(
            "pred",
            lambda a, b: jnp.broadcast_to(
                jnp.exp(a) * (freq / NU0) ** (-b) + floor, (N_TIME, N_FREQ)
            ),
            la,
            be,
        )
        if noise == "radiometer":
            observe(
                "d",
                lambda mu: dist.Normal(mu, F * jnp.abs(mu)).to_event(2),
                pred,
                obs=data,
                depends_on_prediction=True,
            )
        else:
            observe(
                "d",
                lambda mu: dist.Normal(mu, 0.5).to_event(2),
                pred,
                obs=data,
                depends_on_prediction=False,
            )

    return trace(model)


def _bayesmith_prior():
    from bayesmith.diagnose.priors import JeffreysPrior

    return JeffreysPrior(over=("fg_log_amp", "fg_beta"))


def _values(log_amp, beta):
    return {"fg_log_amp": jnp.array(log_amp), "fg_beta": jnp.array(beta)}


@pytest.mark.parametrize(("log_amp", "beta"), GRID)
def test_the_flat_constant_agrees_live_at_every_grid_point(log_amp, beta):
    """+15.80169853 at all nine points, both packages, differenced to
    1e-8 -- and against the pinned constant, so the pair cannot drift
    together either."""
    with jax.enable_x64(True):
        theirs = _rheplicant_prior().log_density(
            _rheplicant_forward(0.0), _values(log_amp, beta), _radiometer()
        )
        ours = _bayesmith_prior().log_density(
            _bayesmith_graph(), _values(log_amp, beta)
        )
    assert float(ours) == pytest.approx(float(theirs), abs=1e-8)
    assert float(ours) == pytest.approx(15.80169853, abs=1e-8)


def test_the_floored_power_law_gradients_agree_under_both_noise_models():
    """Where the prior is NOT flat, the two packages' gradients must still
    be the same function: the floored power law's d/dbeta under the
    radiometer and the constant-sigma declarations, both signs and digits.
    """
    with jax.enable_x64(True):
        at = _values(7.8, 2.55)

        their_radiometer = jax.grad(
            lambda v: _rheplicant_prior().log_density(
                _rheplicant_forward(300.0), v, _radiometer()
            )
        )(at)["fg_beta"]
        our_radiometer = jax.grad(
            lambda v: _bayesmith_prior().log_density(
                _bayesmith_graph(floor=300.0), v
            )
        )(at)["fg_beta"]

        their_constant = jax.grad(
            lambda v: _rheplicant_prior().log_density(
                _rheplicant_forward(300.0), v, _homoscedastic()
            )
        )(at)["fg_beta"]
        our_constant = jax.grad(
            lambda v: _bayesmith_prior().log_density(
                _bayesmith_graph(floor=300.0, noise="homo"), v
            )
        )(at)["fg_beta"]

    assert float(our_radiometer) == pytest.approx(float(their_radiometer), rel=1e-9)
    assert float(our_constant) == pytest.approx(float(their_constant), rel=1e-9)
    # The signs really are opposite -- the fixture is not degenerate.
    assert float(our_radiometer) < 0.0 < float(our_constant)


def test_the_singular_block_floors_identically():
    """The eigh discipline, compared live: both packages floor the
    doubled-amplitude block to the same effectively-zero density, on
    matrices whose slogdet would have returned a plausible +6.42."""
    import numpyro.distributions as dist
    from rheplicant.inference.priors import JeffreysPrior as TheirJeffreys

    from bayesmith import det, observe, sample, trace
    from bayesmith.diagnose.priors import JeffreysPrior as OurJeffreys

    with jax.enable_x64(True):
        freq = jnp.linspace(60e6, 85e6, N_FREQ)

        def their_forward(values):
            nu = freq / NU0
            row = jnp.exp(values["a"] + values["b"]) * nu ** (-values["fg_beta"])
            return jnp.broadcast_to(row, (N_TIME, N_FREQ))

        data = jnp.broadcast_to((freq / NU0) ** (-2.55), (N_TIME, N_FREQ))

        def model():
            a = sample("a", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
            b = sample("b", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
            be = sample(
                "fg_beta", lambda: dist.ImproperUniform(dist.constraints.real, (), ())
            )
            pred = det(
                "pred",
                lambda a_, b_, be_: jnp.broadcast_to(
                    jnp.exp(a_ + b_) * (freq / NU0) ** (-be_), (N_TIME, N_FREQ)
                ),
                a,
                b,
                be,
            )
            observe(
                "d",
                lambda mu: dist.Normal(mu, F * jnp.abs(mu)).to_event(2),
                pred,
                obs=data,
                depends_on_prediction=True,
            )

        graph = trace(model)
        values = {"a": jnp.array(0.0), "b": jnp.array(0.0), "fg_beta": jnp.array(2.55)}
        theirs = TheirJeffreys(over=("a", "b", "fg_beta")).log_density(
            their_forward, values, _radiometer()
        )
        ours = OurJeffreys(over=("a", "b", "fg_beta")).log_density(graph, values)
    assert float(theirs) < -300.0
    assert float(ours) == pytest.approx(float(theirs), abs=0.5)
