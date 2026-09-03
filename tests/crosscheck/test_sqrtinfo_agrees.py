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

**2026-08-30: one of the subjects switched sides, and its four cases retired
with it (D90, migration spec §五 B11).** rheplicant ``b87e44f`` delegated
``marginalise_arrays``'s Schur complement to ``bayesmith.marginal.sqrtinfo``
-- neither side raises anywhere in it, which is why it could move whole. Its
bitwise comparison here (``test_marginalise_arrays_agrees_bitwise``) then
spent two days comparing this package with itself before anyone noticed;
``test_provenance.py`` now asserts every subject's side at the symbol level
so the next single-symbol switch fails a test instead of waiting for a
reader. What remains compared, and why each is still a comparison:

* ``combine`` and ``null`` -- both sides own their arithmetic. The QR fold
  and its ``rho`` corner exist twice, so bitwise disagreement is
  information.
* ``marginalise`` -- SHELL against SHELL over the one shared kernel: their
  name-to-permutation mapping, offset threading and pivot reading against
  ours. A kernel defect is invisible to this comparison now (both sides
  would carry it); the kernel's own oracles live one-sided in
  ``tests/marginal/test_sqrtinfo.py`` and
  ``tests/marginal/test_streaming_equals_batch.py``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.marginal import SqrtInfo, marginalise

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


# `test_marginalise_arrays_agrees_bitwise` retired here on 2026-08-30 (D90):
# rheplicant `b87e44f` made THEIR `marginalise_arrays` a wrapper over OUR
# kernel, so its four bitwise cases compared this package with itself and
# could not fail. The wrapper's pass-through contract belongs to rheplicant's
# own suite (the consumer-compatibility direction the top-level design's §8
# assigns there); the kernel's oracles stay one-sided in
# `tests/marginal/test_sqrtinfo.py`. `test_provenance.py` asserts the
# delegation's direction, so an un-delegation shows up as a failure here
# rather than as a silently-vacuous comparison springing back to life.


@pytest.mark.parametrize("prior_std", [0.7, 1.0, 3.0])
def test_marginalise_agrees_bitwise_including_the_constant(prior_std):
    """The checked path, at a NON-unit prior -- a SHELL comparison since D90.

    Both sides now compute the Schur complement and its Gaussian-integral
    constant in the same kernel, so this sweep's original rationale --
    catching a port that dropped the constant, which is exactly zero at
    ``std = 1`` -- is dead: a kernel that dropped it would drop it for both
    sides and this would stay green. The constant is pinned one-sided
    instead, by ``tests/marginal/test_sqrtinfo.py`` and the nat-cost table
    in the migration spec's B11.

    What still has two sides, and what a red here now means: their checked
    shell against ours -- the name-to-leading-block permutation, the offset
    threading into and out of the kernel, and the pivot reading their five
    refusals stand on. The non-unit prior stays because it drives a non-zero
    offset through both shells, so a shell that mangled the offset cannot
    agree bitwise.
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
