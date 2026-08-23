# tests/exact/test_conditioning.py
"""Spectral diagnostics: known spectra, and the float32 overflow they must survive."""

import jax
import jax.numpy as jnp
import pytest

from bayesmith.exact.conditioning import (
    extreme_eigenvalues,
    largest_eigenvalue,
    tree_norm,
)


def _diagonal(diag):
    """A symmetric positive-definite operator over a one-leaf pytree."""
    return lambda parts: {"x": diag * parts["x"]}


def test_tree_norm_matches_the_flattened_euclidean_norm():
    parts = {"a": jnp.array([3.0, 4.0]), "b": jnp.array([[12.0]])}
    # sqrt(9 + 16 + 144) = 13 exactly, and 13 is not any input value.
    assert float(tree_norm(parts)) == pytest.approx(13.0)


def test_tree_norm_survives_a_leaf_whose_square_overflows_float32():
    """The naive sum-of-squares route really does overflow here.

    Asserting that first is what makes the second assertion mean something:
    without it this test would pass against an implementation that never
    needed the rescale.
    """
    big = jnp.array([3e19, 4e19], dtype=jnp.float32)
    assert not jnp.isfinite(jnp.sum(big**2))
    assert float(tree_norm({"x": big})) == pytest.approx(5e19, rel=1e-5)


def test_tree_norm_of_an_all_zero_pytree_is_zero():
    assert float(tree_norm({"x": jnp.zeros(4)})) == 0.0


def test_largest_eigenvalue_finds_the_top_of_a_known_spectrum():
    diag = jnp.array([1.0, 1.0, 1.0, 100.0])
    got = largest_eigenvalue(
        _diagonal(diag), {"x": jnp.zeros(4)}, jax.random.key(0), 20
    )
    assert float(got) == pytest.approx(100.0, rel=1e-4)


def test_extreme_eigenvalues_finds_both_ends_of_a_known_spectrum():
    diag = jnp.array([1.0, 1.0, 1.0, 100.0])
    largest, smallest = extreme_eigenvalues(
        _diagonal(diag), {"x": jnp.zeros(4)}, jax.random.key(0), 20
    )
    assert float(largest) == pytest.approx(100.0, rel=1e-4)
    assert float(smallest) == pytest.approx(1.0, rel=1e-3)


def test_extreme_eigenvalues_spans_several_pytree_leaves():
    """The spectrum must be the JOINT one, not one leaf's.

    Two leaves whose individual extremes differ: a per-leaf implementation
    would report (10, 2) or (100, 20), never (100, 2).
    """
    operator = lambda parts: {
        "a": jnp.array([2.0, 10.0]) * parts["a"],
        "b": jnp.array([20.0, 100.0]) * parts["b"],
    }
    template = {"a": jnp.zeros(2), "b": jnp.zeros(2)}
    # The second power iteration runs on {98, 90, 80, 0}, a top-two ratio of
    # 90/98 ~= 0.918 -- at 40 iterations that is not yet tight (measured
    # rel_err ~ 5e-4 for this key, but other keys measured up to 2.3 at 40
    # iterations: this key's pass at rel=5e-2 was luck, not margin). 200
    # iterations drives 0.918**200 ~= 4e-8, measured exact (rel_err == 0.0)
    # across 30 keys, so rel=1e-2 is comfortable and still well inside the
    # ~0.5 needed to separate this from the per-leaf mutation in Step 6.
    largest, smallest = extreme_eigenvalues(operator, template, jax.random.key(1), 200)
    assert float(largest) == pytest.approx(100.0, rel=1e-3)
    assert float(smallest) == pytest.approx(2.0, rel=1e-2)
