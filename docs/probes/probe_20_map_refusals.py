"""P2: rerun the graph-native MAP comparisons and refusal boundaries.

This probe keeps the reference paths separate: the linear-Gaussian answer is
``wiener_solve``; the funnel mode is the handwritten stationary point
``(-4.5, 0)``.  The last two rows have no finite isolated MAP and must return
structured refusals rather than their final iterates.

Measured 2026-08-29::

    linear max |MAP - Wiener|        1.1102230246251565e-16
    funnel mode (neck, x)           (-4.500000000000, 0.000000000000)
    hierarchical mode               (1.943963410376, 1.999266810977)
    flat ridge                      refused: Hessian not positive
    unbounded joint prior           refused: did not converge

Exit code 0 means the probe completed, never that a mode is globally unique.

Run from the repository root:

    .venv/bin/python docs/probes/probe_20_map_refusals.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import const, det, joint_prior, observe, sample, trace
from bayesmith.diagnose.map import MapEstimate, Refused, map_estimate
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import wiener_solve


def linear_graph():
    design = jnp.array([[1.0, -0.3], [0.4, 1.2], [-0.7, 0.8], [1.5, 0.2], [0.1, -1.1]])
    truth = jnp.array([1.1, -0.6])
    data = design @ truth + jnp.array([0.1, -0.2, 0.05, 0.08, -0.03])

    def model():
        matrix = const("design", design)
        weights = sample(
            "weights",
            lambda: dist.Normal(jnp.array([0.3, -0.4]), jnp.array([2.0, 3.0])).to_event(
                1
            ),
        )
        prediction = det(
            "prediction",
            lambda a, w: a @ w,
            matrix,
            weights,
            linear_in=("weights",),
        )
        observe(
            "data",
            lambda mu: dist.Normal(mu, 0.5).to_event(1),
            prediction,
            obs=data,
        )

    return trace(model)


def funnel_graph():
    def model():
        neck = sample("neck", lambda: dist.Normal(0.0, 3.0))
        sample("x", lambda y: dist.Normal(0.0, jnp.exp(y / 2.0)), neck)

    return trace(model)


def hierarchical_graph():
    locations = jnp.linspace(1.0, 2.0, 8)
    data = 2.0 * locations

    def model():
        parent = sample("parent", lambda: dist.Normal(1.5, 1.7))
        child = sample("child", lambda p: dist.Normal(p, 0.6), parent)
        prediction = det("prediction", lambda c: c * locations, child)
        observe(
            "data",
            lambda mu: dist.Normal(mu, 0.3).to_event(1),
            prediction,
            obs=data,
        )

    return trace(model)


def flat_ridge_graph():
    def model():
        a = sample("a", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
        b = sample("b", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
        prediction = det("prediction", lambda x, y: x + y, a, b)
        observe("data", lambda mu: dist.Normal(mu, 0.7), prediction, obs=0.0)

    return trace(model)


class RunawayPrior:
    over = ("x",)

    def log_density(self, graph, values):
        del graph
        return values["x"]


def runaway_graph():
    def model():
        joint_prior(RunawayPrior())
        sample("x", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))

    return trace(model)


def main():
    graph = linear_graph()
    found = map_estimate(graph)
    assert isinstance(found, MapEstimate)
    block = linear_operator(graph, ("weights",), at={})
    exact, _ = wiener_solve(
        block,
        precision=precision_at(graph, {"weights": jnp.zeros(2)}),
        tol=1e-14,
    )
    difference = np.max(
        np.abs(np.asarray(found["weights"]) - np.asarray(exact["weights"]))
    )
    print(f"linear max |MAP - Wiener|        {difference:.16e}")

    funnel = map_estimate(funnel_graph())
    assert isinstance(funnel, MapEstimate)
    print(
        "funnel mode (neck, x)           "
        f"({float(funnel['neck']):.12f}, {float(funnel['x']):.12f})"
    )

    hierarchy = map_estimate(hierarchical_graph())
    assert isinstance(hierarchy, MapEstimate)
    print(
        "hierarchical mode               "
        f"({float(hierarchy['parent']):.12f}, {float(hierarchy['child']):.12f})"
    )

    flat = map_estimate(flat_ridge_graph())
    assert isinstance(flat, Refused)
    print(f"flat ridge                      {flat.verdict}: Hessian not positive")

    runaway = map_estimate(runaway_graph())
    assert isinstance(runaway, Refused)
    print(f"unbounded joint prior           {runaway.verdict}: did not converge")


if __name__ == "__main__":
    with jax.enable_x64(True):
        main()
