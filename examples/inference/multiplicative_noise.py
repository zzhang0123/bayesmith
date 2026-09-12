"""Multiplicative gain and linear signal with 0.1% relative Gaussian noise.

    mu = exp(U @ p_g) * (A @ p_n)
    sigma = 0.001 * mu

U and A are fixed, known design matrices. U has no constant gain mode; such a
mode would be exactly degenerate with the overall scale of p_n. Each nonnegative
row of A has a positive entry, and p_n has a positive prior, so sigma stays
positive without abs(), clipping, or replacing it by a fixed noise level.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import compile_task, const, det, observe, sample, trace
from bayesmith.artifacts import (
    ComputeBudget,
    PosteriorTask,
    Refusal,
    model_ref_from_callable,
    new_task_meta,
)

from .block_inspection import inspect_blocks
from .common import Demo, infer, samples, simulate_fixed
from .priors import TOP_LEVEL_BOUNDS, uniform_prior


# 1. Declare operators. Noise depends on the CURRENT mean during inference too.
def gain_prior():
    return uniform_prior("multiplicative_noise", "p_g")


def signal_prior(size=2):
    if size == 2:
        return uniform_prior("multiplicative_noise", "p_n")
    low, high = TOP_LEVEL_BOUNDS["multiplicative_noise"]["p_n"]
    if len(set(low)) != 1 or len(set(high)) != 1:
        raise ValueError("The scale probe requires a shared signal-coefficient range.")
    return dist.Uniform(jnp.full(size, low[0]), jnp.full(size, high[0])).to_event(1)


def exponential_gain(U, p_g):
    return jnp.exp(U @ p_g)


def linear_signal(A, p_n):
    return A @ p_n


def multiply(gain, signal):
    return gain * signal


def mean_signal(U, A, p_g, p_n):
    return multiply(exponential_gain(U, p_g), linear_signal(A, p_n))


def relative_gaussian(mu):
    return dist.Normal(mu, 0.001 * mu)


def design_matrices(x):
    """Two gain modes and a positive linear signal on x in [0, 1]."""
    x = jnp.asarray(x)
    U = jnp.column_stack([jnp.sin(4 * jnp.pi * x), jnp.cos(4 * jnp.pi * x)])
    A = jnp.column_stack([jnp.ones_like(x), x])
    return U, A


# 2. Two branches join at mu. U/A are graph inputs, not hidden closure arrays.
def model(gain_design, signal_design, data):
    U = const("U", gain_design)
    p_g = sample("p_g", gain_prior)
    A = const("A", signal_design)
    p_n = sample("p_n", partial(signal_prior, size=signal_design.shape[1]))
    gain = det("gain", exponential_gain, U, p_g)
    signal = det("signal", linear_signal, A, p_n, linear_in=("p_n",))
    mu = det("mu", multiply, gain, signal, linear_in=("signal",))
    observe("obs", relative_gaussian, mu, obs=data, depends_on_prediction=True)


def inspect_500_parameters(seed=0):
    """Simulate a larger graph and record compilation; do not run its sampler."""
    sim_key, compile_key = jax.random.split(jax.random.key(seed))
    x = jnp.linspace(0.0, 1.0, 1500)
    U, _ = design_matrices(x)
    A = jnp.tile(jnp.eye(500), (3, 1))
    truths = {"p_g": jnp.array([0.12, -0.08]), "p_n": jnp.linspace(1.0, 2.0, 500)}
    template = trace(model, U, A, jnp.zeros_like(x))
    data, _ = simulate_fixed(template, model, truths, sim_key)
    graph = trace(model, U, A, data)
    task = PosteriorTask(
        meta=new_task_meta(
            label="Inspect automatic blocks for 500 positive parameters"
        ),
        budget=ComputeBudget(draws=2000, warmup=1000, chains=2),
        chain_method="sequential",
        nuts_on_collapse=False,
        backend_options=(("progress_bar", False),),
    )
    plan = compile_task(
        graph,
        task,
        model_ref=model_ref_from_callable(model, identifier=model.__module__),
        key=compile_key,
    )
    if isinstance(plan, Refusal):
        raise RuntimeError(f"500-parameter compilation refused: {plan}")  # noqa: TRY004
    return {
        "title": "500 positive signal parameters",
        "observations": int(data.size),
        "seed": seed,
        "design": "U has two sine/cosine gain modes; A repeats the 500 × 500 identity three times. Each positive signal coefficient contributes to three observations.",
        "simulation_method": "SimulationTask(FIXED)",
        "posterior_executed": False,
        "blocking": inspect_blocks(graph, plan.runtime_plan, executed=False),
    }


def run(seed=0, draws=2000, warmup=1000, *, proposals=(), initialization=None,
        initialization_factory=None, graph_transform=None):
    sim_key, infer_key = jax.random.split(jax.random.key(seed))
    x = jnp.linspace(0.0, 1.0, 128)
    U, A = design_matrices(x)
    truths = {"p_g": jnp.array([0.12, -0.08]), "p_n": jnp.array([1.2, 0.7])}
    # 3. SimulationTask uses the same relative-noise operator at the fixed truth.
    template = trace(model, U, A, jnp.zeros_like(x))
    data, simulation = simulate_fixed(template, model, truths, sim_key)
    graph = trace(model, U, A, data)
    if graph_transform is not None:
        graph = graph_transform(graph)
    if initialization is None and initialization_factory is None and not proposals:
        from functools import partial

        from .composed_process import data_initialization

        initialization_factory = partial(data_initialization, perturbation_scale=0.0)
    init_record = None
    if initialization_factory is not None:
        if initialization is not None:
            raise ValueError("Choose supplied initial values or an initialization factory.")
        initialization, init_record = initialization_factory(graph)
    # 4. Compile and infer with the full heteroscedastic likelihood. No truth,
    # fixed-at-truth sigma, or true-parameter initialisation is passed to infer().
    plan, posterior = infer(graph, model, infer_key, draws, warmup,
                            proposals=proposals, initialization=initialization)
    s = samples(posterior)
    signal_draws = jax.vmap(lambda g, n: mean_signal(U, A, g, n))(s["p_g"], s["p_n"])
    return Demo(
        "Multiplicative signal with relative noise",
        graph,
        plan,
        posterior,
        simulation,
        truths,
        {
            "p_g": jnp.sqrt(gain_prior().variance),
            "p_n": jnp.sqrt(signal_prior().variance),
        },
        x,
        data,
        mean_signal(U, A, **truths),
        signal_draws,
        note="mu = exp(U @ p_g) * (A @ p_n); sigma = 0.001 * mu at every likelihood evaluation. "
        "U has no constant gain mode; A and p_n keep mu positive. "
        "The saved plan records the selected updates of the full heteroscedastic posterior; "
        "automatic proposals retain the finite priors through MH.",
        blocking_probes=(inspect_500_parameters(seed),),
        view={"initialization": init_record} if init_record is not None else {},
    )
