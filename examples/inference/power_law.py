"""Two power-law regression curves with additive Gaussian white noise."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace

from .common import Demo, infer, samples, simulate_fixed
from .priors import uniform_prior

CHANNELS = 2
OBSERVATIONS = 32
NOISE_SD = 0.2


def design():
    return (
        jnp.tile(jnp.geomspace(0.5, 2.0, OBSERVATIONS), CHANNELS),
        jnp.repeat(jnp.arange(CHANNELS), OBSERVATIONS),
    )


def amplitude_prior():
    return uniform_prior("power_law", "amplitude")


def index_prior():
    return uniform_prior("power_law", "alpha")


def power_basis(x, alpha, channel):
    return jnp.exp(alpha[channel] * jnp.log(x))


def scaled_power(basis, amplitude, channel):
    return amplitude[channel] * basis


def mean_signal(x, amplitude, alpha, channel):
    return scaled_power(power_basis(x, alpha, channel), amplitude, channel)


def gaussian_observation(location):
    return dist.Normal(location, NOISE_SD)


def model(x_values, channels, data):
    x = const("x", x_values)
    channel = const("channel", channels)
    amplitude = sample("amplitude", amplitude_prior)
    alpha = sample("alpha", index_prior)
    basis = det("basis", power_basis, x, alpha, channel)
    location = det(
        "location", scaled_power, basis, amplitude, channel, linear_in=("amplitude",)
    )
    observe("obs", gaussian_observation, location, obs=data)


def run(seed=0, draws=2000, warmup=1000, *, graph_transform=None):
    # Numerical reference integration is host-side validation, outside the JAX model.
    from .power_law_reference import bias_experiment, posterior_summaries

    sim_key, infer_key = jax.random.split(jax.random.key(seed))
    x, channel = design()
    truths = {"amplitude": jnp.array([1.4, 0.9]), "alpha": jnp.array([-1.1, 0.8])}
    template = trace(model, x, channel, jnp.zeros_like(x))
    data, simulation = simulate_fixed(template, model, truths, sim_key)
    graph = trace(model, x, channel, data)
    if graph_transform is not None:
        graph = graph_transform(graph)
    plan, posterior = infer(graph, model, infer_key, draws, warmup)
    s = samples(posterior)
    signal = jax.vmap(lambda a, p: mean_signal(x, a, p, channel))(
        s["amplitude"], s["alpha"]
    )
    reference = posterior_summaries(
        x[:OBSERVATIONS],
        data.reshape(CHANNELS, OBSERVATIONS),
        jeffreys=graph.joint_prior is not None,
    )
    return Demo(
        "Power-law curves with white noise",
        graph,
        plan,
        posterior,
        simulation,
        truths,
        {
            "amplitude": jnp.sqrt(amplitude_prior().variance),
            "alpha": jnp.sqrt(index_prior().variance),
        },
        x,
        data,
        mean_signal(x, **truths, channel=channel),
        signal,
        note="Two curves y = A x^alpha + epsilon, with independent Normal(0, 0.2^2) noise. "
        "Both amplitudes and indices are inferred; x and the noise SD are known. "
        "The original Gaussian likelihood is fitted in observation space.",
        view={
            "kind": "channels",
            "model_kind": "power_law_regression",
            "channel": channel.tolist(),
            "labels": ["Curve 1", "Curve 2"],
            "x_label": "Input x",
            "noise_sd": NOISE_SD,
            "posterior_reference": reference,
            "bias_experiment": bias_experiment(truths),
        },
    )
