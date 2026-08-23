"""Spectral diagnostics: a known spectrum, and the float32 overflow it survives."""

import jax
import jax.numpy as jnp
import pytest

from bayesmith.errors import GraphError
from bayesmith.exact.conditioning import largest_eigenvalue, tree_norm


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


def test_tree_norm_survives_a_leaf_small_enough_to_underflow_when_squared():
    """The other end of the same rescale, and the naive route fails here too.

    Entries at 1e-30 square to 1e-60, which is zero in float32 -- so the naive
    implementation returns exactly 0.0 for a vector that is emphatically not
    zero. Both ends matter: a normal operator's domain spans whatever units
    the model's latents happen to be in.
    """
    small = jnp.array([3e-30, 4e-30], dtype=jnp.float32)
    assert float(jnp.sqrt(jnp.sum(small**2))) == 0.0
    # abs=0.0 is load-bearing: pytest.approx applies a DEFAULT abs=1e-12 floor
    # and takes max(rel * expected, abs), so `approx(5e-30, rel=1e-5)` accepts
    # anything within 1e-12 of it -- including the exact 0.0 the naive
    # implementation returns, which is the bug this test exists to catch.
    assert float(tree_norm({"x": small})) == pytest.approx(5e-30, rel=1e-5, abs=0.0)


def test_tree_norm_of_an_all_zero_pytree_is_zero():
    assert float(tree_norm({"x": jnp.zeros(4)})) == 0.0


def test_largest_eigenvalue_finds_the_top_of_a_known_spectrum():
    diag = jnp.array([1.0, 1.0, 1.0, 100.0])
    got = largest_eigenvalue(
        _diagonal(diag), {"x": jnp.zeros(4)}, jax.random.key(0), 20
    )
    assert float(got) == pytest.approx(100.0, rel=1e-4)


@pytest.mark.parametrize("spectrum", [[1.0, 1.0, 1.0, 100.0], [1.0, 99.9, 100.0]])
def test_largest_eigenvalue_approaches_the_truth_from_below(spectrum):
    """Power iteration underestimates, and the guard depends on knowing it does.

    `condition_bound` divides lambda_max by a prior-derived LOWER bound on
    lambda_min to get an UPPER bound on kappa. That bound is only as good as
    lambda_max, which must therefore never overshoot.

    Both a well-separated and a nearly-degenerate spectrum, because only the
    first can catch an overshoot. Measured: `[1, 99.9, 100]` plateaus at
    99.9396 and is still 0.0029 short after 2000 iterations, so a 0.01%
    overshoot hides inside its own shortfall; `[1, 1, 1, 100]` reaches exactly
    100.0 by ten iterations, where any overshoot at all is visible. An earlier
    version of this test used only the degenerate case and could not catch the
    mutation named in its own docstring.
    """
    diag = jnp.asarray(spectrum)
    truth = float(jnp.max(diag))
    template = {"x": jnp.zeros(len(spectrum))}
    for iterations in (1, 3, 10, 40):
        got = float(
            largest_eigenvalue(_diagonal(diag), template, jax.random.key(4), iterations)
        )
        assert got <= truth * (1.0 + 1e-6), (iterations, got)


@pytest.mark.parametrize("top_in", ["a", "b"])
def test_largest_eigenvalue_spans_several_pytree_leaves(top_in):
    """The spectrum must be the JOINT one, not any single leaf's.

    Parametrised because whichever leaf holds the top reproduces it when
    restricted to that leaf -- unavoidable -- so one case cannot catch a
    single-leaf implementation. With the top in "b", restricting to the first
    leaf reports 20 instead of 100; with it in "a", restricting to the last
    leaf does. Their union is complete. An earlier single-case version of this
    idea silently missed one of the two, found by mutation testing, which is
    the only thing that finds a guard that does not guard.
    """
    top = jnp.array([2.0, 100.0])
    rest = jnp.array([10.0, 20.0])
    diagonals = {"a": top, "b": rest} if top_in == "a" else {"a": rest, "b": top}

    def operator(parts):
        return {name: diagonals[name] * parts[name] for name in diagonals}

    template = {"a": jnp.zeros(2), "b": jnp.zeros(2)}
    got = largest_eigenvalue(operator, template, jax.random.key(1), 60)
    assert float(got) == pytest.approx(100.0, rel=1e-3)


def test_largest_eigenvalue_refuses_fewer_than_one_iteration():
    """iterations=0 (or negative) would otherwise return the start vector's own
    norm untouched by the operator -- a number with no relationship to it at all.

    Not in the plan's Step 2 block: added per the code-quality review's second
    finding, which asked for this guard and a test pinning it (see mutation 6).
    """
    with pytest.raises(GraphError, match="iterations"):
        largest_eigenvalue(
            _diagonal(jnp.array([1.0, 2.0])), {"x": jnp.zeros(2)}, jax.random.key(0), 0
        )
