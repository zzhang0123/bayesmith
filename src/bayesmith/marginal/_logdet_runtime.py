"""Pure JAX kernels for warmup-certified log-determinant plans.

Nothing in this module checks a theta-dependent premise.  The eager factory
has already fixed the order, certified strict rho, checked provider/probe
provenance, and verified that the runtime dtype can represent the promised
tolerance.  These functions therefore contain no Python convergence branch.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def truncated_trace_logdet(
    lambda_logdet_value: Any,
    exact_power_traces: Any,
    *,
    order: int,
) -> jnp.ndarray:
    """Evaluate a fixed deterministic trace-log series."""
    traces = jnp.asarray(exact_power_traces)
    if traces.ndim != 1 or traces.shape[0] < order:
        raise ValueError(
            "runtime exact_power_traces must be one-dimensional and contain at "
            "least the certified order"
        )
    traces = traces[:order]
    powers = jnp.arange(1, order + 1, dtype=traces.dtype)
    signs = jnp.where(powers % 2 == 1, 1.0, -1.0)
    return jnp.asarray(lambda_logdet_value) + jnp.sum(signs * traces / powers)


def frozen_hutchinson_trace_logdet(
    lambda_logdet_value: Any,
    x_matrix: Any,
    frozen_probe_values: Any,
    *,
    order: int,
) -> jnp.ndarray:
    """Evaluate a fixed frozen-probe trace-log series."""
    x = jnp.asarray(x_matrix)
    vectors = jnp.asarray(frozen_probe_values).T
    n = vectors.shape[0]
    if not (
        (x.ndim == 1 and x.shape == (n,))
        or (x.ndim == 2 and x.shape == (n, n))
    ):
        raise ValueError(
            "runtime X must have compact (n,) or dense (n, n) shape matching "
            "the frozen probe width"
        )
    images = vectors
    correction = jnp.asarray(0.0, dtype=jnp.result_type(x, vectors))
    for power in range(1, order + 1):
        images = x[:, None] * images if x.ndim == 1 else x @ images
        trace_estimate = jnp.mean(jnp.sum(vectors * images, axis=0))
        correction = correction + ((-1.0) ** (power + 1) / power) * trace_estimate
    return jnp.asarray(lambda_logdet_value) + correction
