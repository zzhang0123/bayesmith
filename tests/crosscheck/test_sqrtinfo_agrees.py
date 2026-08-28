"""`bayesmith.marginal.sqrtinfo` against the module it was ported from.

The migration spec's B11 row lists this arithmetic among the kernels to be
**preserved exactly**, having been checked line by line by two reviews. That
word is worth a measurement rather than an intention: these tests run the
same inputs through both packages and compare the outputs, so "preserved"
means "measured to agree on this tree" and a divergence shows up as a
failure rather than as prose.

**What is deliberately NOT compared: the exceptions.** bayesmith raises
`StructureError` where rheplicant raises `StateValidationError`; the two
packages have their own error families and the port is allowed to differ
there on purpose. `tests/marginal/test_sqrtinfo.py` owns the refusals.

Everything here is `float64`, because that is what the evidence layer runs at
and a `float32` comparison would agree to a tolerance that hides a real
difference in the constant.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.marginal import SqrtInfo, marginalise, marginalise_arrays

pytestmark = pytest.mark.crosscheck


def _pair(rows, width, seed, offset=0.0, names=("x",), shapes=None):
    """The same ``[R | z]``, built as both packages' own type."""
    from rheplicant.inference.sqrtinfo import SqrtInfo as TheirSqrtInfo

    rng = np.random.default_rng(seed)
    factor = jnp.asarray(rng.normal(size=(rows, width)))
    target = jnp.asarray(rng.normal(size=rows))
    shapes = shapes or ((width,),)
    kwargs = {
        "factor": factor,
        "target": target,
        "offset": jnp.asarray(offset),
        "names": names,
        "shapes": shapes,
    }
    return SqrtInfo(**kwargs), TheirSqrtInfo(**kwargs)


def _same(ours, theirs, *, at):
    assert ours.names == theirs.names
    assert ours.shapes == theirs.shapes
    assert jnp.allclose(ours.factor, theirs.factor, rtol=0, atol=0), "factor"
    assert jnp.allclose(ours.target, theirs.target, rtol=0, atol=0), "target"
    assert float(ours.offset) == float(theirs.offset), "offset"
    assert float(ours.log_prob(at)) == float(theirs.log_prob(at)), "log_prob"


def test_combine_agrees_bitwise():
    """Same QR, same corner, same offset -- to the bit, not to a tolerance.

    Bitwise is the right bar here: it is the same arithmetic in the same
    order on the same library. A tolerance would let a genuinely different
    fold pass, which is the thing this file exists to notice.
    """
    with jax.enable_x64(True):
        ours_a, theirs_a = _pair(4, 3, seed=2)
        ours_b, theirs_b = _pair(5, 3, seed=3, offset=-0.75)
        _same(
            SqrtInfo.combine(ours_a, ours_b),
            type(theirs_a).combine(theirs_a, theirs_b),
            at={"x": jnp.asarray([1.5, -0.5, 2.0])},
        )


def test_null_agrees_bitwise():
    with jax.enable_x64(True):
        from rheplicant.inference.sqrtinfo import SqrtInfo as TheirSqrtInfo

        names, shapes = ("a", "b"), ((2,), ())
        _same(
            SqrtInfo.null(names, shapes),
            TheirSqrtInfo.null(names, shapes),
            at={"a": jnp.asarray([0.5, -1.0]), "b": jnp.asarray(2.0)},
        )


@pytest.mark.parametrize("n_block", [0, 1, 2, 5])
def test_marginalise_arrays_agrees_bitwise(n_block):
    """Including ``n_block=0``, which is the identity on the density, and
    ``n_block=5``, which integrates the whole term away."""
    from rheplicant.inference.sqrtinfo import (
        marginalise_arrays as their_marginalise_arrays,
    )

    with jax.enable_x64(True):
        rng = np.random.default_rng(17)
        factor = jnp.asarray(rng.normal(size=(9, 5)))
        target = jnp.asarray(rng.normal(size=9))
        offset = jnp.asarray(0.25)
        ours = marginalise_arrays(factor, target, offset, n_block)
        theirs = their_marginalise_arrays(factor, target, offset, n_block)
        # Compared INSIDE the context: these are float64 arrays and
        # `jnp.allclose` outside it runs the comparison at float32, which is a
        # `lax.mul` dtype clash rather than a mismatch anyone would read as a
        # scoping problem.
        for mine, yours, what in zip(
            ours, theirs, ("factor", "target", "offset", "pivots"), strict=True
        ):
            assert jnp.allclose(mine, yours, rtol=0, atol=0), what


@pytest.mark.parametrize("prior_std", [0.7, 1.0, 3.0])
def test_marginalise_agrees_bitwise_including_the_constant(prior_std):
    """The checked path, at a NON-unit prior.

    The prior scale is swept because the constant rheplicant once shipped
    missing is exactly zero at ``std = 1``. A crosscheck run only at unit
    prior would agree with a port that had dropped the same term.
    """
    from rheplicant.inference.sqrtinfo import marginalise as their_marginalise

    n_block, n_keep, rows = 2, 3, 9
    with jax.enable_x64(True):
        rng = np.random.default_rng(41)
        width = n_block + n_keep
        factor = jnp.concatenate(
            [
                jnp.asarray(rng.normal(size=(rows, width))),
                jnp.concatenate(
                    [jnp.eye(n_block) / prior_std, jnp.zeros((n_block, n_keep))],
                    axis=1,
                ),
            ],
            axis=0,
        )
        kwargs = {
            "factor": factor,
            "target": jnp.concatenate(
                [jnp.asarray(rng.normal(size=rows)), jnp.zeros(n_block)]
            ),
            "offset": jnp.asarray(
                -n_block * math.log(prior_std)
                - 0.5 * n_block * math.log(2.0 * math.pi)
            ),
            "names": ("b", "k"),
            "shapes": ((n_block,), (n_keep,)),
        }
        from rheplicant.inference.sqrtinfo import SqrtInfo as TheirSqrtInfo

        _same(
            marginalise(SqrtInfo(**kwargs), ["b"]),
            their_marginalise(TheirSqrtInfo(**kwargs), ["b"]),
            at={"k": jnp.asarray([0.5, -0.25, 1.0])},
        )


def test_the_comparison_can_still_fail():
    """ANTI-VACUITY. A crosscheck that compared nothing would pass silently.

    Perturbs one entry of our factor and asserts the comparison notices --
    so a future `_same` that stopped reading a field, or a `_pair` that
    stopped building both, is caught here rather than by nobody.
    """
    with jax.enable_x64(True):
        ours, theirs = _pair(4, 3, seed=2)
        bent = SqrtInfo(
            factor=ours.factor.at[0, 0].add(1e-9),
            target=ours.target,
            offset=ours.offset,
            names=ours.names,
            shapes=ours.shapes,
        )
        with pytest.raises(AssertionError, match="factor"):
            _same(bent, theirs, at={"x": jnp.asarray([1.0, 0.0, 0.0])})
