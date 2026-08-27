"""Spectral diagnostics for matrix-free symmetric operators.

An iterative solver can cheaply report ``‖M x - b‖``; what a caller actually
wants is ``‖x - x*‖``. The two differ by the condition number, so an honest
convergence guard over a matrix-free operator needs the ends of its spectrum,
and needs them without ever forming a matrix.

Everything here takes the operator as a callable and works on pytrees, so it
knows nothing about :mod:`bayesmith.exact.block`'s blocks and nothing about
graphs. That keeps the numerics separable from the model machinery and the
dependency pointing one way.

**Two ways to get at the bottom of the spectrum, and they answer different
questions.** :func:`extreme_eigenvalues` MEASURES ``lambda_min`` by a second
power iteration on ``lambda_max * I - M``. It fails in principle on a graded
spectrum -- the shifted operator's leading eigenvalues all crowd against
``lambda_max`` with vanishing gaps, so the iteration cannot separate them
however long it runs (2000 steps still left a factor of 700 on a 50-point
geometric spectrum at kappa=1e7). Worse, the bias is **one-sided in the
dangerous direction**: ``lambda_min`` comes back too large, so kappa comes
back too small, so a convergence guard built on it stays silent exactly when
it should fire.

So a GUARD bounds ``lambda_min`` from below by the prior's own curvature
instead: ``A^T N^-1 A`` is positive semi-definite, so
``lambda_min(A^T N^-1 A + S^-1) >= 1 / max(prior_variance)``. See
:func:`bayesmith.exact.solve.condition_bound`, which turns that into an
UPPER bound on kappa -- the direction a safety guard needs.

A DIAGNOSTIC is a different job, and it is why the measured route is here at
all (migration ledger D15(a), gap G14). The bound floors ``lambda_min``, so
it structurally cannot report a near-degenerate partition -- the whole of
which lives in ``lambda_min``. The measured one can see it. See
:func:`bayesmith.exact.solve.condition_estimate`, which says in its own
docstring that it is not a bound; an earlier version of THIS paragraph said
``extreme_eigenvalues`` was "deliberately not ported", which was true of the
guard and became false of the package.

Ported from ``rheplicant.inference.conditioning``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError


def tree_norm(parts: Any) -> jax.Array:
    """Euclidean norm over a pytree, scaled so it survives float32.

    Squaring first overflows for entries beyond ~1.8e19, which turns the only
    convergence signal these solvers give into ``inf/inf = NaN`` exactly when
    the problem is badly scaled and the answer is most likely wrong.
    """
    leaves = [leaf for leaf in jax.tree.leaves(parts) if eqx.is_array(leaf)]
    if not leaves:  # pragma: no cover - defensive
        return jnp.array(0.0)
    biggest = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    biggest = jnp.where(biggest > 0, biggest, 1.0)
    total = sum(jnp.sum(jnp.abs(leaf / biggest) ** 2) for leaf in leaves)
    return biggest * jnp.sqrt(total)


def _scaled(parts: Any, factor: jax.Array) -> Any:
    return jax.tree.map(lambda leaf: leaf / factor, parts)


def _random_like(template: Any, key: jax.Array) -> Any:
    leaves, treedef = jax.tree.flatten(template)
    for leaf in leaves:
        if not eqx.is_array(leaf):
            raise GraphError(
                "largest_eigenvalue's template must be a pytree of arrays "
                "giving the operator's domain (shapes and dtypes); found a "
                f"{type(leaf).__name__} leaf ({leaf!r}) instead"
            )
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        treedef,
        [
            jax.random.normal(subkey, leaf.shape, dtype=leaf.dtype)
            for leaf, subkey in zip(leaves, keys, strict=True)
        ],
    )


def extreme_eigenvalues(
    operator: Callable[[Any], Any],
    template: Any,
    key: jax.Array,
    iterations: int,
) -> tuple[jax.Array, jax.Array]:
    """``(lambda_max, lambda_min)`` of a symmetric positive-definite operator.

    ``lambda_min`` comes from a second power iteration on
    ``lambda_max * I - M``, whose top eigenvalue is
    ``lambda_max - lambda_min``. Measuring it beats bounding it **for a
    diagnostic**: a caller who assumed the worst about ``lambda_min`` would
    call every well-conditioned operator ill-conditioned by the whole dynamic
    range of the problem, and would never be able to tell two partitions
    apart.

    **Read the module docstring before using this for anything that
    branches.** The returned ``lambda_min`` is biased HIGH on a graded
    spectrum, so a kappa built from it is biased LOW -- the direction that
    certifies an answer it should have refused. It is measured here so that
    :func:`bayesmith.exact.solve.condition_estimate` can report it, and that
    function is a diagnostic; the guard is
    :func:`bayesmith.exact.solve.condition_bound`.

    The difference is taken between two numbers of size ``lambda_max``, so it
    is cancellation-prone precisely when ``lambda_min`` is tiny. Callers
    holding an independent lower bound on ``lambda_min`` -- a prior's
    curvature -- should floor the result with it; that is both rigorous and
    the scale at which the cancellation bites.

    Ported from ``rheplicant.core.conditioning`` unchanged, including the
    reuse of :func:`largest_eigenvalue` for the top, so there is one
    implementation of the power iteration rather than two.

    Args:
        operator: the symmetric positive-definite map, pytree to pytree.
        template: a pytree of the operator's domain, for shapes and dtypes.
        key: PRNG key for the starting vectors.
        iterations: number of steps per end.

    Raises:
        GraphError: as :func:`largest_eigenvalue`.
    """
    largest = largest_eigenvalue(operator, template, key, iterations)
    spread = largest_eigenvalue(
        lambda parts: jax.tree.map(
            lambda leaf, image: largest * leaf - image, parts, operator(parts)
        ),
        template,
        jax.random.fold_in(key, 1),
        iterations,
    )
    return largest, largest - spread


def largest_eigenvalue(
    operator: Callable[[Any], Any],
    template: Any,
    key: jax.Array,
    iterations: int,
) -> jax.Array:
    """Top eigenvalue of a symmetric positive-definite operator, by power iteration.

    Each step costs one application of ``operator`` -- for a normal operator
    that is the same JVP-plus-VJP a CG iteration costs, and no matrix is
    formed. The estimate approaches the true value from BELOW.

    Args:
        operator: the symmetric positive-definite map, pytree to pytree.
        template: a pytree of the operator's domain, for shapes and dtypes.
        key: PRNG key for the starting vector.
        iterations: number of steps.

    Raises:
        GraphError: if ``iterations < 1``, or if ``template`` contains a
            leaf that is not an array.
    """
    if iterations < 1:
        raise GraphError(
            "largest_eigenvalue needs at least one iteration to say "
            f"anything about the operator; got iterations={iterations}"
        )
    vector = _random_like(template, key)
    largest = tree_norm(vector)
    vector = _scaled(vector, largest)
    for _ in range(iterations):
        image = operator(vector)
        largest = tree_norm(image)
        vector = _scaled(image, jnp.where(largest > 0, largest, 1.0))
    return largest
