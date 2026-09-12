"""A Gaussian process instance passed through additive and multiplicative stages.

The process is a finite Fourier Gaussian model, with one unknown power amplitude.
The instance is random, not fixed to its mean. The final Gaussian likelihood
integrates out white noise exactly: y = mu * (1 + f*w), w ~ N(0, sigma_w**2).
Only sigma_w is inferred; f=1 fixes the otherwise unidentifiable product.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace

from .common import Demo, infer, samples, simulate_fixed
from .priors import uniform_prior


def data_initialization(graph, *, perturbation_scale=0.02):
    """Bounded joint-density fit from prior centers, using data but never truth.

    This is an explicit demo task setup, not an automatic compiler fallback.
    Two small perturbations in support coordinates seed distinct chains.
    """
    from jax.flatten_util import ravel_pytree
    from numpyro.distributions.transforms import biject_to
    from scipy.optimize import minimize

    from bayesmith.artifacts import InitializationPolicy, NamedArray
    from bayesmith.dispatch.classify import prior_environment
    from bayesmith.graph.evaluate import apply_probabilistic, log_joint

    env = prior_environment(graph)
    transforms = {
        name: biject_to(apply_probabilistic(graph, graph.node(name), env).support)
        for name in graph.latents
    }
    flat, unravel = ravel_pytree(
        {name: transforms[name].inv(env[name]) for name in graph.latents}
    )

    def decode(vector):
        return {
            name: transforms[name](value) for name, value in unravel(vector).items()
        }

    objective = jax.jit(
        jax.value_and_grad(lambda vector: -log_joint(graph, decode(vector)))
    )
    fit = minimize(
        objective,
        np.asarray(flat),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-6},
    )
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f"Data-only initialization failed: {fit.message}")
    perturbation = perturbation_scale * jax.random.normal(jax.random.key(710), flat.shape)
    starts = [decode(jnp.asarray(fit.x) + sign * perturbation) for sign in (-1, 1)]
    policy = InitializationPolicy(
        values=tuple(
            NamedArray(
                name,
                np.stack([s[name] for s in starts]),
                ("chain",) + tuple(f"axis{i}" for i in range(np.ndim(starts[0][name]))),
            )
            for name in graph.latents
        )
    )
    record = {
        "method": "bounded_data_only_L-BFGS_initialization",
        "iterations": int(fit.nit),
        "success": bool(fit.success),
        "message": str(fit.message),
        "truth_used": False,
        "max_iterations": 500,
        "support_coordinate_perturbation_scale": perturbation_scale,
    }
    return policy, record


MODES = 12
NOISE_FACTOR = 1.0


def power_amplitude_prior():
    return uniform_prior("composed_process", "power_amplitude")


def power_spectrum(frequencies, power_amplitude):
    return (power_amplitude / frequencies) ** 2


def process_law(power):
    return dist.Normal(jnp.zeros_like(power), jnp.sqrt(power)).to_event(1)


def nonlinear_prior():
    return uniform_prior("composed_process", "nonlinear")


def background_prior():
    return uniform_prior("composed_process", "background")


def gain_prior():
    return uniform_prior("composed_process", "gain")


def white_noise_prior():
    return uniform_prior("composed_process", "sigma_w")


def mild_white_noise_prior():
    """Illustrative 2% ± 1% noise scale, with the original finite support."""
    return dist.TruncatedNormal(loc=0.02, scale=0.01, low=0.003, high=0.06)


def linear_response(response, instance):
    return response @ instance


def nonlinear_shape(x, theta):
    center, width = theta
    return 0.7 * jnp.exp(-0.5 * ((x - center) / width) ** 2)


def add_nonlinear(response, shape):
    return response + shape


def add_background(signal, design, background):
    return signal + design @ background


def multiply_gain(signal, design, gain):
    return signal * jnp.exp(design @ gain)


def relative_noise(mu, sigma_w):
    # abs changes no distribution: mu*w has variance (mu*sigma_w)**2 even
    # when mu is negative. No clipping or fixed-at-truth weights are used.
    return dist.Normal(mu, NOISE_FACTOR * sigma_w * jnp.abs(mu))


def design_matrices(x):
    frequencies = jnp.repeat(jnp.arange(4.0, 10.0), 2)
    phase = 2 * jnp.pi * x[:, None] * jnp.arange(4.0, 10.0)[None, :]
    fourier = jnp.stack([jnp.sin(phase), jnp.cos(phase)], axis=-1).reshape((-1, MODES))
    transfer = 1.0 / (1.0 + (frequencies / 12.0) ** 2)
    response = fourier * transfer
    background = jnp.column_stack([jnp.ones_like(x), x - 0.5])
    gain = jnp.column_stack([jnp.sin(2 * jnp.pi * x), jnp.cos(2 * jnp.pi * x)])
    return frequencies, fourier, response, background, gain


def model(x_values, frequencies, response, background_design, gain_design, data,
          *, noise_prior=white_noise_prior):
    x = const("x", x_values)
    k = const("frequencies", frequencies)
    R = const("response", response)
    B = const("background_design", background_design)
    U = const("gain_design", gain_design)
    amplitude = sample("power_amplitude", power_amplitude_prior)
    power = det("power", power_spectrum, k, amplitude)
    instance = sample("instance", process_law, power)
    convolved = det(
        "response_signal", linear_response, R, instance, linear_in=("instance",)
    )
    theta = sample("nonlinear", nonlinear_prior)
    shape = det("nonlinear_shape", nonlinear_shape, x, theta)
    combined = det(
        "combined",
        add_nonlinear,
        convolved,
        shape,
        linear_in=("response_signal", "nonlinear_shape"),
    )
    background = sample("background", background_prior)
    signal = det(
        "additive_signal",
        add_background,
        combined,
        B,
        background,
        linear_in=("combined", "background"),
    )
    gain = sample("gain", gain_prior)
    mu = det("mu", multiply_gain, signal, U, gain, linear_in=("additive_signal",))
    sigma_w = sample("sigma_w", noise_prior)
    observe("obs", relative_noise, mu, sigma_w, obs=data, depends_on_prediction=True)


def mild_model(x_values, frequencies, response, background_design, gain_design, data):
    model(x_values, frequencies, response, background_design, gain_design, data,
          noise_prior=mild_white_noise_prior)


def mean_signal(x, R, B, U, instance, nonlinear, background, gain):
    return multiply_gain(
        add_background(
            add_nonlinear(linear_response(R, instance), nonlinear_shape(x, nonlinear)),
            B,
            background,
        ),
        U,
        gain,
    )


def run(seed=0, draws=2000, warmup=1000, *, mild_prior=False, observations=3840):
    if type(observations) is not int or observations < 4:
        raise ValueError("observations must be an integer of at least four")
    process_key, sim_key, infer_key = jax.random.split(jax.random.key(seed), 3)
    x = jnp.linspace(0.0, 1.0, observations, endpoint=False)
    k, fourier, R, B, U = design_matrices(x)
    amplitude = jnp.array(0.5)
    instance = process_law(power_spectrum(k, amplitude)).sample(process_key)
    truths = {
        "power_amplitude": amplitude,
        "instance": instance,
        "nonlinear": jnp.array([0.45, 0.09]),
        "background": jnp.array([3.0, 0.2]),
        "gain": jnp.array([0.08, -0.05]),
        "sigma_w": jnp.array(0.015),
    }
    template = trace(model, x, k, R, B, U, jnp.zeros_like(x))
    data, simulation = simulate_fixed(template, model, truths, sim_key)
    inference_model = mild_model if mild_prior else model
    graph = trace(inference_model, x, k, R, B, U, data)
    initialization, init_record = data_initialization(graph)
    plan, posterior = infer(
        graph, inference_model, infer_key, draws, warmup, initialization=initialization
    )
    s = samples(posterior)
    signal = jax.vmap(lambda a, t, b, g: mean_signal(x, R, B, U, a, t, b, g))(
        s["instance"], s["nonlinear"], s["background"], s["gain"]
    )
    convolved = R @ instance
    shape = nonlinear_shape(x, truths["nonlinear"])
    linear = B @ truths["background"]
    gain = jnp.exp(U @ truths["gain"])
    power_draws = (s["power_amplitude"][:, None] / np.asarray(k)[None, :]) ** 2
    power_interval = np.quantile(power_draws, [0.005, 0.5, 0.995], axis=0)
    demo = Demo(
        "Composed Gaussian process with unknown white-noise scale",
        graph,
        plan,
        posterior,
        simulation,
        truths,
        {
            "power_amplitude": jnp.sqrt(power_amplitude_prior().variance),
            "instance": jnp.sqrt(
                power_amplitude_prior().variance + power_amplitude_prior().mean ** 2
            )
            / k,
            "nonlinear": jnp.sqrt(nonlinear_prior().variance),
            "background": jnp.sqrt(background_prior().variance),
            "gain": jnp.sqrt(gain_prior().variance),
            "sigma_w": jnp.sqrt(white_noise_prior().variance),
        },
        x,
        data,
        (convolved + shape + linear) * gain,
        signal,
        note="One random Fourier instance with 12 coefficients and one power amplitude; "
        "linear response, nonlinear shape, linear background, log-linear gain, then "
        "multiplicative white noise with inferred sigma_w and fixed f=1. "
        "The process law remains part of the hierarchy. The automatic plan jointly updates the instance and linear background with a dense Gaussian iterative-GLS proposal and MH, then uses NUTS for the remaining parameters, with explicit data-only initialization.",
        view={
            "kind": "process",
            "initialization": init_record,
            "noise_realization": {
                "standardized_residual_sd": float(
                    jnp.std(
                        (data / ((convolved + shape + linear) * gain) - 1)
                        / truths["sigma_w"],
                        ddof=1,
                    )
                ),
                "observations": int(data.size),
                "generating_sigma_w": float(truths["sigma_w"]),
            },
            "x_label": "Input x",
            "noise_factor": NOISE_FACTOR,
            "components": {
                "instance": np.asarray(fourier @ instance).tolist(),
                "response": np.asarray(convolved).tolist(),
                "nonlinear": np.asarray(shape).tolist(),
                "linear": np.asarray(linear).tolist(),
                "gain": np.asarray(gain).tolist(),
            },
            "spectrum": {
                "frequency": np.asarray(k).tolist(),
                "truth": np.asarray(power_spectrum(k, amplitude)).tolist(),
                "lower": power_interval[0].tolist(),
                "median": power_interval[1].tolist(),
                "upper": power_interval[2].tolist(),
                "realized_pair_power": np.repeat(
                    (np.asarray(instance).reshape(-1, 2) ** 2).mean(axis=1), 2
                ).tolist(),
            },
        },
    )
    if mild_prior:
        demo.view["prior_variant"] = {
            "kind": "mild_noise_prior",
            "label": "with mild noise prior",
            "baseline_case": "composed_process",
            "coordinates": ["sigma_w"],
            "family": "TruncatedNormal",
            "loc": 0.02,
            "scale": 0.01,
            "support": [0.003, 0.06],
            "formula_latex": r"\sigma_w\sim\mathcal N(0.02,0.01^2)\ \text{restricted to }[0.003,0.06]",
            "rationale": "Illustrative broad 2% noise-scale belief; not independent calibration. "
            "Specified after inspecting the baseline, before running this comparison. "
            "No hyperparameter retries to obtain a pass.",
            "recovery_scale": "The original Uniform prior SD is retained as a fixed comparison reference.",
            "prior_central_95_interval": np.asarray(mild_white_noise_prior().icdf(jnp.array([0.025,0.975]))).tolist(),
        }
        demo.note += " Only sigma_w changes to the declared mild truncated-Normal prior; " \
            "data, truth, seed and recovery thresholds match the baseline."
    demo.note += f" This run uses {observations} observations on [0, 1)."
    return demo
