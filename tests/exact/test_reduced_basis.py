"""G4 -- the reduced-basis selection and orthonormalisation, on this side.

What moves is the ARRAY-LEVEL linear algebra: choose candidate directions from
a bank, and turn candidates into a basis. What stays in rheplicant is the
containers (``ReducedBasis``, ``FidelityReport``) and the declaration layer
that produces a bank from a ``ParameterSpace`` and a pipeline -- the G6
enumeration classifies both, and D12 rules the containers.

Every oracle here is numpy, and mostly numpy that computes the PROPERTY rather
than the procedure: a Gram matrix that is the identity, a projector that is
idempotent, a span that is nested. Those cannot be satisfied by a routine that
happens to do the same arithmetic wrongly, which a step-by-step replay could.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.exact.reduced_basis import (
    numerical_rank,
    orthonormal_transform,
    orthonormalise,
    select_greedy,
    select_svd,
)


@pytest.fixture(autouse=True)
def _double_precision():
    """Every test here runs in float64, because the module refuses otherwise.

    A module-level autouse x64 fixture is a shape this programme has been
    caught by twice -- it can remove the very condition a guard was written
    for -- so what it takes away is written down and asserted separately by
    ``TestItRefusesSinglePrecision``, which turns it back off.
    """
    with jax.enable_x64(True):
        yield


def _bank(rows=9, columns=14, rank=None, seed=0):
    """A bank with a known rank: ``rank`` independent directions, the rest
    exact combinations of them."""
    generator = np.random.default_rng(seed)
    full = generator.normal(size=(rows, columns))
    if rank is None:
        return jnp.asarray(full)
    basis = full[:rank]
    weights = generator.normal(size=(rows, rank))
    return jnp.asarray(weights @ basis)


class TestOrthonormalTransform:
    """``M @ candidates`` orthonormal, and ``M`` returned rather than the
    result, because a basis has to be applicable to the RAW rows as well --
    the whitened copy is infinite wherever the reference could not see."""

    def test_the_result_is_orthonormal(self):
        """The property, not the procedure: the Gram matrix is the identity."""
        candidates = _bank(rows=5, columns=12)
        transform, kept = orthonormal_transform(candidates)
        rows = np.asarray(transform @ candidates)
        assert kept == (0, 1, 2, 3, 4)
        assert np.allclose(rows @ rows.T, np.eye(5), atol=1e-12)

    def test_the_transform_carries_the_SAME_combination_to_the_raw_rows(self):
        """Why ``M`` is returned at all, pinned by the use it exists for.

        **The first version of this test was vacuous**, and a mutation found
        it: it compared ``transform @ candidates`` against
        ``orthonormalise(candidates)``, which is DEFINED as that -- the same
        expression on both sides. Its docstring claimed to catch "a transform
        that did not reproduce the rows", which is exactly what it could not
        do. Dropping the transform's accumulation left it green.

        What the transform is for: a basis built from whitened rows has to be
        applied to the RAW ones, because an epoch whose flag pattern differs
        from the reference's cannot divide the whitened copy by a weight that
        is zero where the reference could not see. So the property is that one
        combination serves both -- ``(M @ raw) * weight == M @ (raw * weight)``
        -- and it is checked here on a weight with a genuine zero in it.
        """
        raw = np.array(_bank(rows=4, columns=10))
        weight = np.ones(10)
        weight[3] = 0.0          # a sample the reference could not see
        weight[7] = 4.0
        whitened = raw * weight

        transform, kept = orthonormal_transform(jnp.asarray(whitened))
        assert transform.shape == (4, 4) and kept == (0, 1, 2, 3)

        carried = np.asarray(transform) @ raw
        assert np.allclose(carried * weight, np.asarray(transform @ jnp.asarray(whitened)))
        # ... and the whitened image really is orthonormal, so the combination
        # being carried is the basis one and not an arbitrary matrix.
        rows = np.asarray(transform @ jnp.asarray(whitened))
        assert np.allclose(rows @ rows.T, np.eye(4), atol=1e-12)
        # The zero-weight column is where the raw rows are NOT recoverable
        # from the whitened ones, which is the whole reason M is returned.
        assert np.any(np.abs(carried[:, 3]) > 1e-6)

    def test_the_span_is_NESTED_in_the_candidates_order(self):
        """What seeding depends on: a direction placed first survives whatever
        the later candidates do. Checked as a span containment, so it cannot
        be satisfied by an implementation that merely keeps index 0 somewhere.
        """
        candidates = np.asarray(_bank(rows=5, columns=12))
        rows = np.asarray(orthonormalise(jnp.asarray(candidates)))
        for depth in range(1, 6):
            # Each prefix of the OUTPUT must span the same space as the same
            # prefix of the INPUT: project the input prefix onto the output
            # prefix and nothing may be left over.
            projected = candidates[:depth] @ rows[:depth].T @ rows[:depth]
            assert np.allclose(projected, candidates[:depth], atol=1e-10), depth

    def test_a_rank_deficient_set_drops_directions_and_SAYS_WHICH(self):
        """``r < k`` is a fact about the model, so ``kept`` names the
        survivors rather than the caller having to infer them from a shape."""
        independent = np.asarray(_bank(rows=3, columns=9))
        candidates = np.vstack([independent, independent[0] * 2.0])
        transform, kept = orthonormal_transform(jnp.asarray(candidates))
        assert kept == (0, 1, 2)
        assert transform.shape == (3, 4)
        rows = np.asarray(transform @ candidates)
        assert np.allclose(rows @ rows.T, np.eye(3), atol=1e-12)

    def test_an_exact_duplicate_of_a_LATER_row_drops_the_later_one(self):
        """Order decides which of a dependent pair survives, and that is the
        nesting property again seen from the other side."""
        independent = np.asarray(_bank(rows=3, columns=9))
        candidates = np.vstack([independent[0], independent[1], independent[0]])
        _, kept = orthonormal_transform(jnp.asarray(candidates))
        assert kept == (0, 1)

    def test_it_beats_a_single_pass_gram_schmidt_on_a_nasty_set(self):
        """The reorthogonalisation pass is load-bearing, not defensive.

        On a near-dependent set, one Gram-Schmidt pass loses orthogonality;
        two recover it. Measured here against a single-pass reference computed
        in this file, so the claim is a comparison rather than an assertion.
        """
        # Ill-conditioned on purpose: five rows spanning three directions
        # with the dependent ones a part in 1e12 away, and a 1e6 dynamic
        # range across the block. Measured: a 1e-9 perturbation was not
        # enough -- both passes hit 2.2e-16 and the comparison had nothing
        # to compare.
        base = np.array(_bank(rows=3, columns=20, seed=3))  # writable copy
        base[2] *= 1e6
        candidates = np.vstack(
            [base, base[0] + 1e-12 * base[1], base[1] + 1e-12 * base[2]]
        )

        def single_pass(rows):
            out = []
            for row in rows:
                vector = row.copy()
                for kept in out:
                    vector = vector - (kept @ vector) * kept
                norm = np.linalg.norm(vector)
                if norm > 1e-12:
                    out.append(vector / norm)
            return np.asarray(out)

        once = single_pass(candidates)
        twice = np.asarray(orthonormalise(jnp.asarray(candidates)))
        error_once = np.abs(once @ once.T - np.eye(len(once))).max()
        error_twice = np.abs(twice @ twice.T - np.eye(len(twice))).max()
        assert error_twice < error_once, (error_once, error_twice)
        assert error_twice < 1e-10

    def test_an_empty_candidate_set_is_empty_rather_than_an_error(self):
        transform, kept = orthonormal_transform(jnp.zeros((0, 7)))
        assert kept == () and transform.shape == (0, 0)

    def test_a_one_dimensional_candidate_array_is_refused(self):
        """``(k, n)`` is the shape, and a single row passed as ``(n,)`` would
        orthonormalise into one direction of length one -- finite, plausible,
        and not what the caller meant."""
        with pytest.raises(StructureError, match="two-dimensional|2-D|shape"):
            orthonormal_transform(jnp.ones(7))


class TestNumericalRank:
    """``sqrt(eps)``, not ``eps``: the quadratic form squares the
    conditioning, so a set that is merely invertible is not usable."""

    def test_it_finds_a_planted_rank(self):
        assert numerical_rank(_bank(rows=9, columns=14, rank=4)) == 4

    def test_a_full_rank_bank_is_its_row_count(self):
        assert numerical_rank(_bank(rows=6, columns=14)) == 6

    def test_an_empty_bank_is_zero(self):
        assert numerical_rank(jnp.zeros((0, 5))) == 0

    def test_an_all_zero_bank_is_zero(self):
        """``s_0 == 0`` is the case a ratio cannot be taken in, so it is
        answered before the ratio rather than by it."""
        assert numerical_rank(jnp.zeros((4, 5))) == 0

    def test_the_cut_is_sqrt_eps_and_not_eps(self):
        """Constructed so the two answers differ: a direction at ``1e-10`` of
        the leading one is above ``eps`` and below ``sqrt(eps)``.

        Without this, ``eps`` would pass every other test in this class -- and
        it is the version that hands back a set whose Gram matrix no float64
        quadratic form survives.
        """
        directions = np.eye(3, 8)
        bank = np.vstack([directions[0], directions[1], 1e-10 * directions[2]])
        assert numerical_rank(jnp.asarray(bank)) == 2
        singular = np.linalg.svd(bank, compute_uv=False)
        assert singular[2] / singular[0] > np.finfo(np.float64).eps


class TestSelectSvd:
    def test_it_returns_the_leading_right_singular_directions(self):
        bank = _bank(rows=7, columns=11)
        found = np.asarray(select_svd(bank, 3))
        expected = np.linalg.svd(np.asarray(bank), full_matrices=False)[2][:3]
        # Sign is a convention of the decomposition, so compare subspaces.
        assert np.allclose(np.abs(found @ expected.T), np.eye(3), atol=1e-10)

    def test_they_are_orthonormal_even_though_that_is_incidental(self):
        found = np.asarray(select_svd(_bank(rows=7, columns=11), 4))
        assert np.allclose(found @ found.T, np.eye(4), atol=1e-12)

    def test_a_non_positive_count_gives_an_empty_set(self):
        assert select_svd(_bank(), 0).shape == (0, 14)


class TestSelectGreedy:
    """Rows OF THE BANK, in the order chosen. Not a basis -- storing the raw
    picks is what makes the Gram matrix unusable, and orthonormalising them is
    a separate call for that reason."""

    def test_every_pick_is_a_row_of_the_bank(self):
        bank = np.asarray(_bank(rows=8, columns=10))
        picked = np.asarray(select_greedy(jnp.asarray(bank), 3))
        for row in picked:
            assert any(np.allclose(row, candidate) for candidate in bank)

    def test_the_first_pick_is_the_largest_row(self):
        # `.copy()`: `np.asarray` of a JAX array is a READ-ONLY view, and
        # `bank[4] *= 50` on it raises "output array is read-only".
        bank = np.array(_bank(rows=6, columns=10))
        bank[4] *= 50.0
        picked = np.asarray(select_greedy(jnp.asarray(bank), 1))
        assert np.allclose(picked[0], bank[4])

    def test_it_keeps_picking_past_the_rank_and_ORTHONORMALISE_drops_them(self):
        """The honest chain, and it is not what I first asserted.

        A rank-2 bank in float64 leaves a residual of order 1e-16 rather than
        exactly zero, so the early stop -- which tests ``norm == 0.0``, as the
        upstream it was ported from does -- does not fire, and five picks come
        back. That is not a defect to fix here: selection returns candidates,
        and it is ``orthonormalise`` that turns candidates into a basis and
        drops the dependent ones. Measured: 5 picks, 2 kept.

        Written down rather than repaired because the first version of this
        test asserted the stronger property and nearly had me put a tolerance
        into the selector -- which would have moved a decision from the place
        that makes it to the place that does not.
        """
        bank = _bank(rows=6, columns=10, rank=2)
        picked = select_greedy(bank, 5)
        assert picked.shape[0] == 5
        _, kept = orthonormal_transform(picked)
        assert len(kept) == 2

    def test_it_picks_a_SPANNING_set_rather_than_the_largest_rows(self):
        """The property that separates greedy selection from sorting by norm.

        Two large and nearly parallel rows plus one small and orthogonal one:
        sorting takes the two parallel ones, greedy takes one of them and the
        orthogonal one, because after the first pick the second is almost
        entirely explained.
        """
        columns = 10
        first = np.zeros(columns)
        first[0] = 10.0
        # 0.99, not 1.0: `first + 1e-6 e1` is LONGER than `first`, so greedy
        # picks it first and the test measures nothing. Measured -- that is
        # what the first version did.
        second = 0.99 * first + 1e-6 * np.eye(columns)[1]
        third = np.zeros(columns)
        third[2] = 1.0
        bank = np.vstack([first, second, third])
        picked = np.asarray(select_greedy(jnp.asarray(bank), 2))
        assert np.allclose(picked[0], first)
        assert np.allclose(picked[1], third)
        # ... where sorting by norm would have taken `second`.
        by_norm = np.argsort(-np.linalg.norm(bank, axis=1))[:2]
        assert set(by_norm.tolist()) == {0, 1}

    def test_a_non_positive_count_gives_an_empty_set(self):
        assert select_greedy(_bank(), 0).shape == (0, 14)


class TestTheTwoSelectorsAreDifferentQuestions:
    def test_they_disagree_on_a_bank_where_they_should(self):
        """Anti-vacuity for the pair. If the two always agreed, one of them
        would be redundant and every test above would be testing one routine
        twice."""
        bank = _bank(rows=8, columns=12, seed=11)
        svd = np.asarray(select_svd(bank, 3))
        greedy = np.asarray(select_greedy(bank, 3))
        assert not np.allclose(np.abs(svd @ greedy.T), np.eye(3), atol=1e-6)

    def test_but_both_span_the_bank_when_asked_for_its_full_rank(self):
        """...and they agree about the SPAN at full rank, which is the sense
        in which they are two answers to one question."""
        bank = _bank(rows=5, columns=12, rank=3, seed=5)
        svd = np.asarray(orthonormalise(select_svd(bank, 3)))
        greedy = np.asarray(orthonormalise(select_greedy(bank, 3)))
        rows = np.asarray(bank)
        for projector in (svd, greedy):
            residual = rows - rows @ projector.T @ projector
            assert np.abs(residual).max() < 1e-9


class TestItRefusesSinglePrecision:
    """The guard the autouse fixture above hides, turned back on.

    ``c^T G c`` squares the conditioning and the retention cut is
    ``sqrt(eps)`` of the arithmetic in hand: 3.4e-04 in float32 against
    1.5e-08 in float64. On a foreground-dominated bank the direction three
    orders below the largest is exactly the one the basis exists to keep, and
    in float32 it is silently dropped -- so this refuses rather than returning
    a basis that has quietly discarded the science.
    """

    @pytest.fixture(autouse=True)
    def _double_precision(self):
        """Overrides the module fixture BY NAME -- which is the only thing
        that overrides it. A differently-named class fixture leaves the
        module's in place, and every test below then runs in x64 and passes
        for the wrong reason. Measured: that is what the first version did."""
        yield

    @pytest.mark.parametrize(
        "call",
        [
            lambda: orthonormal_transform(jnp.zeros((2, 5))),
            lambda: numerical_rank(jnp.zeros((2, 5))),
            lambda: select_svd(jnp.zeros((2, 5)), 1),
            lambda: select_greedy(jnp.zeros((2, 5)), 1),
        ],
        ids=["orthonormal_transform", "numerical_rank", "select_svd", "select_greedy"],
    )
    def test_every_entry_point_refuses(self, call):
        with pytest.raises(StructureError, match="float32"):
            call()

    def test_the_refusal_names_the_cut_and_where_to_widen(self):
        """A refusal that did not say WHERE is how a caller wraps this call in
        `enable_x64` and recovers nothing -- the bank's digits are already
        spent."""
        with pytest.raises(StructureError) as caught:
            numerical_rank(jnp.zeros((2, 5)))
        message = str(caught.value)
        assert "3.4e-04" in message and "1.5e-08" in message
        assert "building the BANK inside the block" in message

    def test_it_is_the_AMBIENT_precision_that_decides_not_the_input(self):
        """Judged by outcome. A float64 array handed in under a float32
        ambient is still refused, because everything downstream of here is
        traced at the ambient dtype -- which is the half a dtype check on the
        input would miss."""
        wide = np.zeros((2, 5), dtype=np.float64)
        with pytest.raises(StructureError, match="float32"):
            numerical_rank(jnp.asarray(wide))
