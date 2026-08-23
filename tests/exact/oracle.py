"""The independent dense oracle -- no autodiff, no JAX transformation.

Builds the same posterior the matrix-free CG path builds, by the most naive
route available: evaluate the block's own map on a basis of its domain,
assemble a dense ``A`` column by column, and solve the normal equations with
``numpy.linalg``.

**What it shares, and what it must not.** It shares the MODEL -- `isolate`'s
``g``, which is `evaluate` plus reading each observed node's loc -- because an
oracle has to evaluate the same model or it answers a different question. It
shares none of the LINEAR ALGEBRA: no `jax.linearize`, no `jax.vjp`, no `cg`,
no `tree_norm`, no power iteration. `A[:, j] = g(e_j) - g(0)` is the
definition of a linear map on a basis, exact for an affine ``g``, and the only
thing it trusts is that calling ``g`` twice gives the same answer twice.

That independence is the point. The P1 design record names the failure this
guards against: two readings of one graph that share an implementation share
its blind spots, and agreed on -225.65 while the truth was -364.95. "Exact vs
NUTS" is a self-consistency check for the same reason -- both go through
`apply_probabilistic`. This is not.

**Build the graph INSIDE the x64 context.** `jax.enable_x64(True)` is
thread-local and affects arrays created after it, so a graph traced at
float32 stays float32 no matter what context the solve runs in -- `const` and
`observe` call `jnp.asarray` at trace time. Every test below therefore traces
its model inside the `with` block, and the oracle would otherwise be the less
accurate of the two things being compared.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np


class Oracle(NamedTuple):
    """Everything the dense route computes, in one flat layout.

    ``order`` is ``[(latent_name, flat_index), ...]`` for the domain, so a
    caller can line a solve's ``{name: array}`` answer up against ``mean``
    without guessing offsets -- :func:`flat_domain` is that map.
    """

    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    design: np.ndarray
    offset: np.ndarray
    data: np.ndarray
    sigma: np.ndarray
    prior_mean: np.ndarray
    prior_std: np.ndarray
    order: list[tuple[str, int]]


def _flatten(tree: dict[str, Any], order: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(tree[key]).ravel() for key in order])


def flat_domain(values: dict[str, Any], names: tuple[str, ...]) -> np.ndarray:
    """A solve's ``{name: array}`` answer, flattened the way ``Oracle`` is."""
    return _flatten(values, list(names))


def dense_design(g, shapes, dtypes, names, obs_order):
    """``(A, offset)`` with ``A[:, j] = g(e_j) - g(0)``."""
    zero = {n: jnp.zeros(shapes[n], dtype=dtypes[n]) for n in names}
    offset = _flatten(g(zero), obs_order)
    columns = []
    for name in names:
        size = int(np.prod(shapes[name], dtype=int))
        for index in range(size):
            flat = np.zeros(size)
            flat[index] = 1.0
            probe = dict(zero)
            probe[name] = jnp.asarray(
                flat.reshape(shapes[name] if shapes[name] else ()),
                dtype=dtypes[name],
            )
            columns.append(_flatten(g(probe), obs_order) - offset)
    return np.stack(columns, axis=1), offset


def analytic_posterior(design, offset, data, sigma, prior_mean, prior_std):
    """``(mean, covariance, precision)`` of the linear-Gaussian posterior."""
    noise_precision = np.diag(1.0 / np.asarray(sigma) ** 2)
    prior_precision = np.diag(1.0 / np.asarray(prior_std) ** 2)
    precision = design.T @ noise_precision @ design + prior_precision
    rhs = design.T @ noise_precision @ (data - offset) + prior_precision @ prior_mean
    covariance = np.linalg.inv(precision)
    return covariance @ rhs, covariance, precision


def graph_oracle(graph, names, at=None, sigma_at=None) -> Oracle:
    """The dense posterior of a block of ``graph``.

    Args:
        graph, names, at: the same three a block is built from.
        sigma_at: latent values to freeze sigma at, for a prediction-dependent
            noise model. Defaults to the block's zero, which is where a
            constant sigma is the same everywhere anyway.
    """
    from bayesmith.exact.block import _env_before, isolate
    from bayesmith.exact.gaussian import noise_std_at, observation_parts
    from bayesmith.graph.evaluate import evaluate

    names = tuple(names)
    at = dict(at or {})
    _, domain = _env_before(graph, names, at)
    shapes = {n: domain[n][0] for n in names}
    dtypes = {n: domain[n][1] for n in names}
    zero = {n: jnp.zeros(shapes[n], dtype=dtypes[n]) for n in names}

    obs_order = sorted(graph.observed)
    design, offset = dense_design(
        isolate(graph, names, at), shapes, dtypes, names, obs_order
    )
    data_tree, _, _ = observation_parts(graph, evaluate(graph, {**at, **zero}))
    sigma = noise_std_at(graph, {**at, **(sigma_at or zero)})

    prior_mean = _flatten({n: domain[n][2] for n in names}, list(names))
    prior_std = _flatten({n: domain[n][3] for n in names}, list(names))
    mean, covariance, precision = analytic_posterior(
        design,
        offset,
        _flatten(data_tree, obs_order),
        _flatten(sigma, obs_order),
        prior_mean,
        prior_std,
    )
    order = [(n, i) for n in names for i in range(int(np.prod(shapes[n], dtype=int)))]
    return Oracle(
        mean=mean,
        covariance=covariance,
        precision=precision,
        design=design,
        offset=offset,
        data=_flatten(data_tree, obs_order),
        sigma=_flatten(sigma, obs_order),
        prior_mean=prior_mean,
        prior_std=prior_std,
        order=order,
    )
