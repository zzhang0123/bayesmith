"""A plated hierarchy must be planned, not rejected by the streaming probe.

``compile`` asks :func:`~bayesmith.dispatch.streaming.streaming_route`, on
every graph, whether a plate could be streamed one epoch at a time. That
function's contract is to never raise: a plate the evidence layer cannot
fold is a refusal it records. Until 2026-09-19 the textbook plated hierarchy
(a population mean, a plated level drawn around it, data around the level)
made ``factorize``'s linearity check raise ``NotGaussian`` for the ancestry
between the two latents; ``NotGaussian`` is a ``TypeError``, the route caught
only ``StructureError``/``GraphError``, and ``compile`` raised on a graph
whose vector-latent spelling it plans as exact levels inside NUTS.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import compile as compile_graph
from bayesmith import det, observe, plate, sample, trace
from bayesmith.dispatch.streaming import streaming_route

DATA = jnp.array([0.5, 1.7, -0.3])


def _plated_hierarchy():
    def model(data):
        population_mean = sample("population_mean", lambda: dist.Normal(0.0, 2.0))
        units = plate("units", len(data))
        level = sample(
            "level", lambda mean: dist.Normal(mean, 0.5), population_mean, plate=units
        )
        signal = det("signal", lambda value: 2.0 * value, level, plate=units)
        observe("data", lambda m: dist.Normal(m, 0.2), signal, obs=data, plate=units)

    return trace(model, DATA)


def test_the_streaming_probe_records_the_ancestry_as_a_refusal():
    with jax.enable_x64(True):
        route = streaming_route(_plated_hierarchy())
    assert not route.available
    why = dict(route.refused)["units"]
    assert "among its ancestors" in why
    assert "could not be analysed" not in why


def test_compile_plans_exact_levels_inside_nuts():
    with jax.enable_x64(True):
        plan = compile_graph(_plated_hierarchy())
    assert [(b.latents, b.method) for b in plan.blocks] == [
        (("level",), "gcr"),
        (("population_mean",), "nuts"),
    ]
