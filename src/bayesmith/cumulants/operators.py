"""Cumulant contractions, independent of the coordinates used by a graph.

LowRankCumulants represents K_r = sum_a weights[r][a] v_a**(tensor r).
Its storage scales with the number of directions, not the tensor order.
DenseCumulants is a small-field implementation/oracle, not a large-map route.
"""

import math
from collections.abc import Mapping
from functools import cache
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp


class CumulantContractions(Protocol):
    def contract(self, orders: tuple[int, ...]) -> jax.Array:
        """Contract the product of these cumulants with H_(sum orders)."""
        ...


class CumulantOperator(Protocol):
    event_shape: tuple[int, ...]
    orders: tuple[int, ...]

    @property
    def batch_shape(self) -> tuple[int, ...]: ...

    def prepare(self, value: jax.Array, reference: Any) -> CumulantContractions:
        """Prepare contractions against this same Gaussian reference."""
        ...


def _real_array(value):
    array = jnp.asarray(value)
    if jnp.iscomplexobj(array):
        raise TypeError(
            "field coordinates and cumulants must be real; use a coordinate adapter"
        )
    return array.astype(jnp.result_type(array, float))


def _orders(cumulants):
    orders = tuple(sorted(cumulants))
    if not orders or any(
        isinstance(r, bool) or not isinstance(r, int) or r < 3 for r in orders
    ):
        raise ValueError(
            "cumulants must map integer orders >= 3 to explicit coefficients"
        )
    return orders


def _geometry(value, reference, directions, event_shape):
    """Projected score and precision via Gaussian AD, never an N-by-N Hessian.

    Differentiating the sum handles independent sample/batch dimensions. Inputs
    are broadcast BEFORE differentiation so broadcast data do not merge scores.
    """
    n = len(event_shape)
    leading = jnp.broadcast_shapes(
        reference.log_prob(value).shape, directions.shape[: -n - 1], value.shape[:-n]
    )
    dtype = jnp.result_type(value, directions, reference.log_prob(value))
    value = jnp.broadcast_to(value, leading + event_shape).astype(dtype)
    rank = directions.shape[-n - 1]
    directions = jnp.broadcast_to(directions, leading + (rank,) + event_shape).astype(
        dtype
    )
    score_fn = jax.grad(lambda x: -jnp.sum(reference.log_prob(x)))
    score = score_fn(value)
    vectors = jnp.moveaxis(directions, -n - 1, 0)
    precision_vectors = jax.vmap(lambda v: jax.jvp(score_fn, (value,), (v,))[1])(
        vectors
    )
    precision_vectors = jnp.moveaxis(precision_vectors, 0, -n - 1)
    v = directions.reshape(leading + (rank, math.prod(event_shape)))
    qv = precision_vectors.reshape(v.shape)
    h = jnp.einsum("...ri,...i->...r", v, score.reshape(leading + (-1,)))
    q = jnp.einsum("...ri,...si->...rs", v, qv)
    return h, q


def _hermite(orders, h, q):
    """Directional Hermite recurrence with multiplicities, not field tensors."""

    @cache
    def recur(counts):
        if not any(counts):
            return jnp.ones(h.shape[:-1], dtype=h.dtype)
        i = next(i for i, count in enumerate(counts) if count)
        remaining = list(counts)
        remaining[i] -= 1
        result = h[..., i] * recur(tuple(remaining))
        for j, count in enumerate(remaining):
            if count:
                lower = remaining.copy()
                lower[j] -= 1
                result = result - count * q[..., i, j] * recur(tuple(lower))
        return result

    return recur(orders)


class _LowRankContractions(eqx.Module):
    h: jax.Array
    q: jax.Array
    weights: tuple[jax.Array, ...]
    available: tuple[int, ...] = eqx.field(static=True)

    def contract(self, orders):
        weights = tuple(self.weights[self.available.index(r)] for r in orders)
        rank = self.h.shape[-1]
        if len(orders) == 1:
            h = self.h[..., :, None]
            q = jnp.diagonal(self.q, axis1=-2, axis2=-1)[..., :, None, None]
            return jnp.sum(weights[0] * _hermite(orders, h, q), axis=-1)

        def add_term(index, total):
            indices = []
            for _ in orders:
                indices.append(index % rank)
                index = index // rank
            indices = jnp.stack(indices)
            h = jnp.take(self.h, indices, axis=-1)
            q = self.q[..., indices[:, None], indices[None, :]]
            amplitude = jnp.ones(self.h.shape[:-1], dtype=self.h.dtype)
            for w, i in zip(weights, indices):
                amplitude = amplitude * w[..., i]
            return total + amplitude * _hermite(orders, h, q)

        # Static loop bounds support reverse-mode gradients. No R**k array is
        # materialized, though runtime and reverse-mode storage grow with R**k.
        return jax.lax.fori_loop(
            0,
            rank ** len(orders),
            add_term,
            jnp.zeros(self.h.shape[:-1], dtype=self.h.dtype),
        )


class LowRankCumulants(eqx.Module):
    """Symmetric CP cumulants in the array's own coordinates.

    ``directions`` has shape ``batch + (rank,) + event_shape``; by default its
    first axis is rank and all remaining axes are one field. ``cumulants[r]``
    has shape ``batch + (rank,)`` (a scalar broadcasts). Coefficients are raw
    cumulants, with no factorial or variance scaling. Missing orders are not
    inferred as zero. The same directions may carry different weights at each
    order; changing their normalization requires changing those weights.
    """

    directions: jax.Array
    weights: tuple[jax.Array, ...]
    orders: tuple[int, ...] = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, directions, cumulants: Mapping[int, Any], *, event_shape=None):
        self.directions = _real_array(directions)
        self.event_shape = tuple(
            self.directions.shape[1:] if event_shape is None else event_shape
        )
        n = len(self.event_shape)
        if (
            not n
            or not all(s > 0 for s in self.event_shape)
            or self.directions.ndim < n + 1
        ):
            raise ValueError("directions need a rank axis and a nonempty event_shape")
        if self.directions.shape[-n:] != self.event_shape:
            raise ValueError("directions do not match event_shape")
        rank = self.directions.shape[-n - 1]
        if rank < 1:
            raise ValueError("at least one direction is required")
        self.orders = _orders(cumulants)
        weights = []
        for r in self.orders:
            w = _real_array(cumulants[r])
            if w.ndim == 0:
                w = jnp.broadcast_to(w, (rank,))
            if w.shape[-1] != rank:
                raise ValueError(f"order {r} needs one coefficient per direction")
            weights.append(w)
        self.weights = tuple(weights)
        _ = self.batch_shape  # Check broadcast compatibility at construction.

    @property
    def batch_shape(self):
        return jnp.broadcast_shapes(
            self.directions.shape[: -len(self.event_shape) - 1],
            *(w.shape[:-1] for w in self.weights),
        )

    def prepare(self, value, reference):
        n = len(self.event_shape)
        rank = self.directions.shape[-n - 1]
        leading = jnp.broadcast_shapes(
            self.batch_shape, reference.log_prob(value).shape
        )
        dtype = jnp.result_type(value, self.directions, *self.weights)
        directions = jnp.broadcast_to(
            self.directions, leading + (rank,) + self.event_shape
        ).astype(dtype)
        h, q = _geometry(value, reference, directions, self.event_shape)
        weights = tuple(jnp.broadcast_to(w, h.shape) for w in self.weights)
        return _LowRankContractions(h, q, weights, self.orders)


class _DenseContractions(eqx.Module):
    h: jax.Array
    q: jax.Array
    tensors: tuple[jax.Array, ...]
    available: tuple[int, ...] = eqx.field(static=True)

    def contract(self, orders):
        degree = sum(orders)

        def generating(t):
            return jnp.exp(
                jnp.einsum("...i,i->...", self.h, t)
                - 0.5 * jnp.einsum("i,...ij,j->...", t, self.q, t)
            )

        derivative = generating
        for _ in range(degree):
            derivative = jax.jacfwd(derivative)
        hermite = derivative(jnp.zeros(self.h.shape[-1], dtype=self.h.dtype))
        args = [hermite, [Ellipsis, *range(degree)]]
        start = 0
        for r in orders:
            args.extend(
                (
                    self.tensors[self.available.index(r)],
                    [Ellipsis, *range(start, start + r)],
                )
            )
            start += r
        return jnp.einsum(*args, [Ellipsis], optimize="greedy")


class DenseCumulants(eqx.Module):
    """Small-field tensors, each shaped batch + (N,)*r in flattened C order.

    Computes dense Hermite derivatives up to degree 3*(order-2). Memory can
    grow as N**degree; use LowRankCumulants or a custom contraction operator
    for maps/cubes with many elements. Tensor symmetrization is implicit in
    the contraction with a symmetric Hermite tensor.
    """

    tensors: tuple[jax.Array, ...]
    orders: tuple[int, ...] = eqx.field(static=True)
    event_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, cumulants: Mapping[int, Any], *, event_shape):
        self.event_shape = tuple(event_shape)
        if not self.event_shape or not all(s > 0 for s in self.event_shape):
            raise ValueError("event_shape must be nonempty and positive")
        self.orders = _orders(cumulants)
        size = math.prod(self.event_shape)
        tensors = []
        for r in self.orders:
            tensor = _real_array(cumulants[r])
            if tensor.shape[-r:] != (size,) * r:
                raise ValueError(f"order {r} tensor must end in {(size,) * r}")
            tensors.append(tensor)
        self.tensors = tuple(tensors)
        _ = self.batch_shape

    @property
    def batch_shape(self):
        return jnp.broadcast_shapes(
            *(t.shape[:-r] for t, r in zip(self.tensors, self.orders))
        )

    def prepare(self, value, reference):
        size = math.prod(self.event_shape)
        leading = jnp.broadcast_shapes(
            self.batch_shape, reference.log_prob(value).shape
        )
        dtype = jnp.result_type(value, *self.tensors)
        directions = jnp.broadcast_to(
            jnp.eye(size, dtype=dtype).reshape((size,) + self.event_shape),
            leading + (size,) + self.event_shape,
        )
        h, q = _geometry(value, reference, directions, self.event_shape)
        return _DenseContractions(h, q, self.tensors, self.orders)
