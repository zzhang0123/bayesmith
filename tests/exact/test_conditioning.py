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


@pytest.mark.parametrize("extremes_in", ["a", "b"])
def test_extreme_eigenvalues_spans_several_pytree_leaves(extremes_in):
    """The spectrum must be the JOINT one, not any single leaf's.

    Both power iterations are exercised, and the parametrisation is what makes
    that true. Whichever leaf holds an extreme reproduces it when restricted
    to that leaf -- unavoidable -- so no single spectrum can catch a
    single-leaf implementation of both iterations. Measured, on the joint
    spectrum {2, 10, 20, 100}:

        spectrum              iter1 a  iter1 b  iter2 a  iter2 b
        a=[10,20] b=[2,100]   caught   missed   caught   missed
        a=[2,100] b=[10,20]   missed   caught   missed   caught

    Their union is complete, which is why this runs as two cases and not one.
    An earlier single-case version of this test used a=[2,10] b=[20,100] and
    silently missed the iter2-first-leaf bug entirely.

    200 iterations, not 40: the shifted operator's top two eigenvalues are 98
    and 90, a ratio of 0.918, so 40 steps leave ~3% error on a target of 2.0
    and the test passes or fails on the luck of the starting vector. Measured
    across 30 keys: at 40 iterations the worst is 229% relative error; at 200
    all 30 are float32-exact.
    """
    extremes = jnp.array([2.0, 100.0])
    middle = jnp.array([10.0, 20.0])
    diagonals = (
        {"a": extremes, "b": middle}
        if extremes_in == "a"
        else {"a": middle, "b": extremes}
    )

    def operator(parts):
        return {name: diagonals[name] * parts[name] for name in diagonals}

    template = {"a": jnp.zeros(2), "b": jnp.zeros(2)}
    largest, smallest = extreme_eigenvalues(operator, template, jax.random.key(1), 200)
    assert float(largest) == pytest.approx(100.0, rel=1e-2)
    assert float(smallest) == pytest.approx(2.0, rel=1e-2)
