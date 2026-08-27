"""The spectral diagnostics, compared against the package they came from.

``bayesmith.exact.conditioning`` says in its own docstring that it was ported
from rheplicant's, and that one function was deliberately left behind. Both
halves of that sentence are asserted here: what was ported must still agree
number for number, and what was rejected must still be absent.

The second half is the one worth having. A port is compared once, by hand, on
the day it is written; after that the two drift in silence. These run.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bayesmith.exact import conditioning as ours

pytestmark = pytest.mark.crosscheck


@pytest.fixture(scope="module")
def theirs():
    """rheplicant's copy.

    It moved from ``rheplicant.inference.conditioning`` to
    ``rheplicant.core.conditioning`` upstream, because ``radio`` needed it and
    may not import ``inference``. Imported by its current path; if that moves
    again this fails at collection, which is the correct loudness for a
    harness whose whole job is to be looking at the other package.
    """
    return pytest.importorskip("rheplicant.core.conditioning")


def _dense_operator(eigenvalues, seed: int = 0):
    """A symmetric operator with exactly this spectrum, as a callable."""
    size = len(eigenvalues)
    basis, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (size, size)))
    matrix = basis @ jnp.diag(jnp.asarray(eigenvalues, dtype=jnp.float32)) @ basis.T
    return (lambda vector: matrix @ vector), matrix


class TestWhatWasPortedStillAgrees:
    @pytest.mark.parametrize(
        "parts",
        [
            {"a": jnp.array([3.0, 4.0])},
            {"a": jnp.array([1e20, 1e20]), "b": jnp.array([0.0])},
            {"a": jnp.zeros(3)},
        ],
    )
    def test_tree_norm_returns_the_same_number(self, theirs, parts):
        """Including the overflow case both docstrings argue about: squaring
        first turns 1e20 into inf, and the scaling that avoids it is the whole
        content of the function. A port that dropped the scaling agrees on the
        first row and disagrees on the second."""
        assert float(ours.tree_norm(parts)) == float(theirs.tree_norm(parts))

    @pytest.mark.parametrize(
        "spectrum",
        [
            [1.0, 2.0, 3.0],
            [1e-6, 1.0, 5.0, 5.5],
            [1.0] * 6,
        ],
    )
    def test_largest_eigenvalue_returns_the_same_number(self, theirs, spectrum):
        """Same operator, same template, same key, same iteration count -- so a
        difference here is the algorithm and not the setup."""
        operator, _ = _dense_operator(spectrum)
        template = jnp.zeros(len(spectrum))
        key = jax.random.key(7)

        assert float(ours.largest_eigenvalue(operator, template, key, 12)) == (
            float(theirs.largest_eigenvalue(operator, template, key, 12))
        )

    def test_both_approach_the_top_of_the_spectrum_from_below(self, theirs):
        """Agreeing on a wrong number is still agreement, so the shared claim
        is checked against the truth as well as against each other."""
        spectrum = [0.5, 1.0, 4.0, 9.0]
        operator, matrix = _dense_operator(spectrum)
        truth = float(jnp.max(jnp.linalg.eigvalsh(matrix)))
        template = jnp.zeros(len(spectrum))

        for module in (ours, theirs):
            found = float(
                module.largest_eigenvalue(operator, template, jax.random.key(7), 12)
            )
            assert found <= truth * (1 + 1e-5), (module.__name__, found, truth)
            assert found == pytest.approx(truth, rel=1e-3), module.__name__


class TestWhatWasRejectedIsHereNowAsADiagnostic:
    """**This class used to assert an ABSENCE, and the ruling changed.**

    ``lambda_min`` by a second power iteration on ``lambda_max * I - M`` was
    rejected here because it fails in principle on a graded spectrum and fails
    in the DANGEROUS direction: ``lambda_min`` too large, kappa too small, a
    guard built on it silent exactly when it should fire. Every word of that
    is still true and none of it is retracted.

    What changed is what the routine is FOR. Migration ledger D15(a) rules
    that ``condition_estimate`` is ported as a DIAGNOSTIC -- it can see a
    near-degenerate partition, which lives entirely in ``lambda_min`` and
    which ``condition_bound`` structurally cannot report because it floors
    ``lambda_min`` with the prior. So the absence assertion is replaced by the
    two that actually carry the rule now: the packages must AGREE, and no
    guard in this package may read it (``tests/exact/
    test_condition_estimate.py::TestNoGuardReadsTheMeasuredRoute``).

    The old assertion's own docstring said to read the argument rather than
    take a green suite as agreement. It has been read; it is unchanged; the
    conclusion it supports is about guards and not about the package.
    """

    def test_both_packages_now_carry_it_and_agree(self, theirs):
        """A real cross-package comparison, which the absence check could not
        be. Same operator, same template, same key, same iteration count."""
        spectrum = [1.0, 2.0, 3.0]
        operator, _ = _dense_operator(spectrum)
        template = jnp.zeros(len(spectrum))
        key = jax.random.key(7)

        assert hasattr(ours, "extreme_eigenvalues")
        mine = ours.extreme_eigenvalues(operator, template, key, 12)
        yours = theirs.extreme_eigenvalues(operator, template, key, 12)
        assert float(mine[0]) == float(yours[0])
        assert float(mine[1]) == float(yours[1])

    def test_they_agree_on_the_graded_spectrum_too_where_both_are_wrong(self):
        """Agreeing on a wrong number is still agreement, so the case that
        matters is the one where both are wrong in the same direction -- and
        that is the whole reason this routine is not a guard."""
        spectrum = list(jnp.geomspace(1.0, 1e4, 20))
        operator, matrix = _dense_operator(spectrum)
        template = jnp.zeros(len(spectrum))
        truth = float(jnp.min(jnp.linalg.eigvalsh(matrix)))
        _, smallest = ours.extreme_eigenvalues(
            operator, template, jax.random.key(7), 12
        )
        # Too LARGE, which is the dangerous direction, and by a lot.
        assert float(smallest) > truth * 5.0, (float(smallest), truth)

    def test_rheplicant_still_carries_it_and_still_leans_the_unsafe_way(self, theirs):
        """The finding this harness exists to keep visible, as a live number.

        On a graded spectrum the second power iteration cannot separate the
        eigenvalues crowded against ``lambda_max``, and the error is one-sided:
        the ``lambda_min`` it reports is too LARGE. Downstream that is a kappa
        too SMALL and a convergence guard that under-reports the error it is
        supposed to bound.

        Asserted as an inequality against the truth rather than pinned to a
        value, so it stays true whatever the iteration count is tuned to. If
        rheplicant ever fixes this, this test goes red and should be deleted
        along with the row it guards.
        """
        # Geometric spectrum, kappa = 1e4: the shape the rejection was measured on.
        spectrum = [10.0 ** (-4.0 * index / 19.0) for index in range(20)]
        operator, matrix = _dense_operator(spectrum)
        truth = jnp.linalg.eigvalsh(matrix)
        template = jnp.zeros(len(spectrum))

        _, smallest = theirs.extreme_eigenvalues(
            operator, template, jax.random.key(7), theirs.POWER_ITERATIONS
        )

        assert float(smallest) > float(jnp.min(truth)) * 10, (
            "rheplicant's lambda_min estimate is no longer badly over-large on "
            "a graded spectrum. If that is a real fix upstream, delete this "
            "test and the paragraph in bayesmith/exact/conditioning.py that "
            f"cites it. Measured: {float(smallest)} against a true "
            f"{float(jnp.min(truth))}."
        )
