"""Two decay channels: four conditional Gaussian coefficients + two rates."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace

from .common import Demo, infer, samples, simulate_fixed
from .priors import uniform_prior

CHANNELS = 2


def amplitude_prior():
    return uniform_prior("exponential_decay", "amplitude")


def rate_prior():
    return uniform_prior("exponential_decay", "rate")


def offset_prior():
    return uniform_prior("exponential_decay", "offset")


def exponential(time, rate, channel):
    return jnp.exp(-rate[channel] * time)


def scaled_offset(basis, amplitude, offset, channel):
    return amplitude[channel] * basis + offset[channel]


def decay(time, amplitude, rate, offset, channel):
    return scaled_offset(exponential(time, rate, channel), amplitude, offset, channel)


def gaussian_observation(location):
    return dist.Normal(location, 0.08)


def model(time_values, channels, data):
    time = const("time", time_values)
    channel = const("channel", channels)
    rate = sample("rate", rate_prior)
    amplitude = sample("amplitude", amplitude_prior)
    offset = sample("offset", offset_prior)
    basis = det("basis", exponential, time, rate, channel)
    location = det(
        "location",
        scaled_offset,
        basis,
        amplitude,
        offset,
        channel,
        linear_in=("amplitude", "offset"),
    )
    observe("obs", gaussian_observation, location, obs=data)


def run(seed=0, draws=2000, warmup=1000, *, proposals=(), initialization=None,
        initialization_factory=None, graph_transform=None):
    sim_key, infer_key = jax.random.split(jax.random.key(seed))
    time = jnp.tile(jnp.linspace(0.0, 5.0, 120), CHANNELS)
    channel = jnp.repeat(jnp.arange(CHANNELS), 120)
    truths = {
        "amplitude": jnp.array([2.0, 1.4]),
        "rate": jnp.array([0.7, 1.1]),
        "offset": jnp.array([0.3, -0.1]),
    }
    template = trace(model, time, channel, jnp.zeros_like(time))
    data, simulation = simulate_fixed(template, model, truths, sim_key)
    graph = trace(model, time, channel, data)
    if graph_transform is not None:
        graph = graph_transform(graph)
    init_record = None
    if initialization_factory is not None:
        if initialization is not None:
            raise ValueError("Choose supplied initial values or an initialization factory.")
        initialization, init_record = initialization_factory(graph)
    plan, posterior = infer(graph, model, infer_key, draws, warmup,
                            proposals=proposals, initialization=initialization)
    s = samples(posterior)
    signal = jax.vmap(lambda a, r, b: decay(time, a, r, b, channel))(
        s["amplitude"], s["rate"], s["offset"]
    )
    return Demo(
        "Two-channel exponential decay",
        graph,
        plan,
        posterior,
        simulation,
        truths,
        {
            "amplitude": jnp.sqrt(amplitude_prior().variance),
            "rate": jnp.sqrt(rate_prior().variance),
            "offset": jnp.sqrt(offset_prior().variance),
        },
        time,
        data,
        decay(time, **truths, channel=channel),
        signal,
        note="The mean remains linear in amplitude/background at fixed rates. Their bounded Uniform priors truncate the conditional Gaussian; the current compiler uses joint NUTS for all six coordinates.",
        view={
            **({"initialization": init_record} if init_record is not None else {}),
            "kind": "channels",
            "channel": channel.tolist(),
            "labels": ["Channel 1", "Channel 2"],
            "x_label": "Time",
        },
    )
