"""P4 probe: a double count that shape diagnostics cannot see.

Run from the repository root::

    .venv/bin/python docs/probes/probe_23_double_count_visibility.py

The retained parameter ``theta`` is deliberately independent of the block
``x`` and its observation.  Attaching the integrated evidence to the original
graph therefore adds a non-zero constant to theta's marginal log-density.  A
normalised posterior has the same mean and width, and the score has the same
gradient, while every absolute density is wrong.  This is the smallest case
that proves why P4's regression guard must compare absolute densities rather
than only posterior shape.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic
from bayesmith.graph.reduction import reduce_with_evidence


class ConstantEvidence(eqx.Module):
    over: tuple[str, ...] = eqx.field(static=True, default=())
    value: jax.Array = eqx.field(default_factory=lambda: jnp.zeros(()))

    def log_density(self, graph: Graph, values: dict[str, Any]) -> jax.Array:
        del graph, values
        return self.value


X_LOC = 0.3
X_SCALE = 1.7
THETA_LOC = -0.4
THETA_SCALE = 2.3
NOISE = 0.6
OBSERVED = 1.1


def graph() -> Graph:
    return Graph(
        nodes=(
            Probabilistic(
                name="x",
                parents=(),
                plate=(),
                dist_fn=lambda: dist.Normal(X_LOC, X_SCALE),
                observed=None,
            ),
            Probabilistic(
                name="theta",
                parents=(),
                plate=(),
                dist_fn=lambda: dist.Normal(THETA_LOC, THETA_SCALE),
                observed=None,
            ),
            Probabilistic(
                name="d",
                parents=("x",),
                plate=(),
                dist_fn=lambda x: dist.Normal(x, NOISE),
                observed=jnp.asarray(OBSERVED),
            ),
        ),
        plates=(),
    )


def dense_marginal(graph_: Graph, theta: jax.Array) -> jax.Array:
    x = jnp.linspace(X_LOC - 12.0 * X_SCALE, X_LOC + 12.0 * X_SCALE, 80_001)
    values = jax.vmap(lambda one: log_joint(graph_, {"x": one, "theta": theta}))(x)
    peak = jnp.max(values)
    return peak + jnp.log(jnp.trapezoid(jnp.exp(values - peak), x))


def moments(grid: jax.Array, log_density: jax.Array) -> tuple[float, float]:
    weights = jnp.exp(log_density - jnp.max(log_density))
    normalisation = jnp.trapezoid(weights, grid)
    mean = jnp.trapezoid(weights * grid, grid) / normalisation
    variance = jnp.trapezoid(weights * (grid - mean) ** 2, grid) / normalisation
    return float(mean), float(jnp.sqrt(variance))


def main() -> None:
    with jax.enable_x64(True):
        original = graph()
        log_evidence = dist.Normal(X_LOC, jnp.sqrt(X_SCALE**2 + NOISE**2)).log_prob(
            OBSERVED
        )
        term = ConstantEvidence(value=log_evidence)
        reduced = reduce_with_evidence(
            original,
            remove_latents=("x",),
            absorb_observed=("d",),
            evidence_term=term,
            nuts_latents=("theta",),
        )
        doubled = Graph(
            nodes=original.nodes,
            plates=original.plates,
            evidence_terms=(term,),
        )

        check_points = jnp.asarray(
            [
                THETA_LOC - 2.0 * THETA_SCALE,
                THETA_LOC,
                THETA_LOC + 2.0 * THETA_SCALE,
            ]
        )
        correct_absolute = jax.vmap(lambda theta: log_joint(reduced, {"theta": theta}))(
            check_points
        )
        doubled_absolute = jax.vmap(lambda theta: dense_marginal(doubled, theta))(
            check_points
        )
        absolute_gap = doubled_absolute - correct_absolute

        theta_grid = jnp.linspace(
            THETA_LOC - 8.0 * THETA_SCALE,
            THETA_LOC + 8.0 * THETA_SCALE,
            20_001,
        )
        correct_curve = jax.vmap(lambda theta: log_joint(reduced, {"theta": theta}))(
            theta_grid
        )
        # The dense checks above establish that this is the mutant graph's
        # marginal curve; adding the constant on the full grid avoids an
        # unnecessary 20,001 x 80,001 allocation.
        doubled_curve = correct_curve + log_evidence
        correct_mean, correct_width = moments(theta_grid, correct_curve)
        doubled_mean, doubled_width = moments(theta_grid, doubled_curve)

        gradient_point = jnp.asarray(0.7)
        correct_gradient = jax.grad(lambda theta: log_joint(reduced, {"theta": theta}))(
            gradient_point
        )
        doubled_gradient = jax.grad(lambda theta: dense_marginal(doubled, theta))(
            gradient_point
        )

        np.testing.assert_allclose(absolute_gap, log_evidence, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(correct_mean, doubled_mean, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(correct_width, doubled_width, rtol=0.0, atol=2e-12)
        np.testing.assert_allclose(
            correct_gradient, doubled_gradient, rtol=0.0, atol=2e-12
        )
        assert abs(float(log_evidence)) > 1.0

        print(f"log evidence counted twice = {float(log_evidence): .12f} nats")
        print(
            f"normalised mean           = {correct_mean: .12f} / {doubled_mean: .12f}"
        )
        print(
            f"normalised width          = {correct_width: .12f} / {doubled_width: .12f}"
        )
        print(
            "gradient at theta=0.7     = "
            f"{float(correct_gradient): .12f} / {float(doubled_gradient): .12f}"
        )
        print(
            "absolute-density gaps     = "
            + np.array2string(np.asarray(absolute_gap), precision=12)
        )
        print("PASS: shape and gradient agree; every absolute density is wrong.")


if __name__ == "__main__":
    main()
