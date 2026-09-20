"""The FULL joint covariance over ``zeta_1:N``, against a forward oracle.

:func:`~bayesmith.marginal.chain.smooth` returns the per-epoch marginal mean
and variance, so every existing test of it reads a DIAGONAL. The quantity the
smoother actually computes is bigger than that: ``_zeta_joint`` assembles the
whole block-tridiagonal joint, and ``smooth`` throws the off-diagonal away on
the way out. Nothing in this package looked at what it threw away --
``tests/marginal/test_chain.py``'s
``test_the_mean_and_variance_match_the_dense_solve`` forms the dense
``covariance`` and then compares only ``np.diagonal(covariance)``. The
cross-epoch check existed only in rheplicant, over its own copy of this
assembly, which is the copy T-004 is retiring. This file is where it lives now.

**A second oracle, not a second copy, and that is the whole design.** The dense
reference in ``test_chain.py`` assembles the same PRECISION blocks the
production rows are the square root of -- ``phi.T @ Q^-1 @ phi`` on the
diagonal, ``-phi.T @ Q^-1`` off it. An error in that algebraic form would be
written into the oracle as readily as into the code, so agreeing proves less
than it looks. The oracle here never inverts ``process_std`` at all:

* the prior joint over ``zeta_1:N`` is propagated FORWARD from the generative
  statement -- ``mean_{e+1} = phi mean_e``, ``P_{e+1} = phi P_e phi.T + Q``,
  ``Cov(zeta_f, zeta_e) = phi**(f - e) P_e`` -- which is the model as declared
  rather than as factorised;
* the data are folded in by the COVARIANCE-form Gaussian update
  (``K = Sigma A.T (A Sigma A.T + I)^-1``), so no precision matrix, no
  square root, no QR and no ``1 / process_std`` appears anywhere in the
  reference.

Measured on the fixture below: the two agree to ``4.857e-16`` absolute and
``7.963e-13`` relative over all 225 entries, and the joint mean to
``1.110e-15``.

**What this file kills that nothing here killed before.** Six mutants in
``_zeta_joint``, applied to a scratch copy of ``src/``, run against this file
(8 passed clean) beside ``test_chain.py`` + ``test_chain_conditioning.py``
(38 passed clean). Exit 1 alone counts as a kill:

===================================  ==========  ============
mutant                               this file   pre-existing
===================================  ==========  ============
``diag(1/q) @ phi`` -> ``phi @ ...``  KILLED      SURVIVED
couplings removed entirely            KILLED      KILLED
``initial_mean`` dropped              KILLED      KILLED
coupling sign flipped                 KILLED      KILLED
forward term left unscaled            KILLED      KILLED
``theta`` not subtracted from rhs     KILLED      KILLED
===================================  ==========  ============

The first row is the gap. Every pre-existing test that drives ``_zeta_joint``
numerically uses a WIDTH-1 chain, where ``diag(1/q) @ phi`` and
``phi @ diag(1/q)`` are the same number. ``test_chain.py``'s
``test_it_holds_for_a_WIDE_chain_too`` does use width 3 and does name that
exact confusion -- but it runs through ``chain_log_likelihood``, which is
assembled by ``_plan``/``_fold``, a SECOND copy of the expression. Coverage
of one copy was being read as coverage of both.

The fixture is chosen so a diagonal-only check could not stand in for this
one. ``phi`` is non-symmetric and genuinely rotates, ``process_std`` and
``initial_std`` differ per component, ``initial_mean`` is non-zero, and theta
is non-zero so the ``targets - B theta`` reduction is exercised. Measured:
the largest off-diagonal entry is ``1.904e-1`` against a largest diagonal of
``4.823e-1`` -- 39.5 %, not a rounding term -- and the corner block linking
epoch 1 to epoch 5, four transitions apart, still carries ``1.480e-2``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.marginal.chain import LinearGaussianTransition, _zeta_joint, smooth

NAMES = ("gain", "offset")
SHAPES = ((), ())
EPOCHS, WIDTH, N_THETA = 5, 3, 2
SIZE = EPOCHS * WIDTH

#: Non-symmetric and rotating: a symmetric or diagonal phi cannot tell
#: ``diag(1/q) @ phi`` from ``phi @ diag(1/q)``, and those are different models.
PHI = np.array([[0.72, 0.21, 0.0], [-0.13, 0.64, 0.28], [0.05, 0.11, 0.53]])
PROCESS_STD = np.array([0.4, 0.7, 0.55])
INITIAL_STD = np.array([1.1, 0.9, 1.4])
INITIAL_MEAN = np.array([0.2, -0.3, 0.1])
THETA = np.array([0.4, -1.1])


@pytest.fixture(autouse=True)
def _double_precision():
    """float64, for the same reason ``test_chain.py`` uses it: the agreement
    being measured is at 1e-12 relative and float32 has no such number."""
    with jax.enable_x64(True):
        yield


def _blocks(seed: int = 11):
    """Per-epoch square joint forms over ``(theta..., zeta)``, zeta LAST.

    Random rather than structured, for ``test_chain.py``'s reason: a block with
    a special shape lets a slice at ``n_theta`` take one half for the other and
    still agree.
    """
    generator = np.random.default_rng(seed)
    width = N_THETA + WIDTH
    return (
        jnp.asarray(generator.normal(size=(EPOCHS, width, width))),
        jnp.asarray(generator.normal(size=(EPOCHS, width))),
        jnp.asarray(generator.normal(size=(EPOCHS,)) * 0.1),
    )


def _transition() -> LinearGaussianTransition:
    return LinearGaussianTransition(
        phi=PHI,
        process_std=PROCESS_STD,
        initial_std=INITIAL_STD,
        initial_mean=INITIAL_MEAN,
    )


def _values() -> dict[str, jax.Array]:
    return {"gain": jnp.asarray(THETA[0]), "offset": jnp.asarray(THETA[1])}


def forward_oracle(blocks) -> tuple[np.ndarray, np.ndarray]:
    """``(mean, covariance)`` of ``p(zeta_1:N | d_1:N, theta)``, densely.

    Two steps, neither of which is the production arithmetic:

    1. The prior joint over ``zeta_1:N`` straight from the generative law.
       ``P_1 = diag(initial_std**2)``, ``P_{e+1} = phi P_e phi.T + diag(
       process_std**2)``, and ``Cov(zeta_f, zeta_e) = phi**(f-e) P_e`` for
       ``f >= e``. Note what is absent: no ``1 / process_std`` and no matrix
       inverse of anything the transition declares.
    2. The covariance-form Gaussian update against the whitened rows.
       ``factors[e][:, n_theta:]`` is the design on ``zeta_e`` and
       ``targets[e] - factors[e][:, :n_theta] @ theta`` the residual data, with
       unit noise -- that is what "square-root information rows" means. The
       update inverts the ``(rows, rows)`` innovation covariance, which is a
       different matrix, of a different size, than anything the recursion forms.

    Its own correctness is pinned by
    :func:`test_the_oracle_s_prior_is_the_transition_s_own_stationary_law`
    below, so a broken oracle fails on its own terms rather than by disagreeing.
    """
    factors, targets, _ = (np.asarray(part) for part in blocks)

    means = [INITIAL_MEAN]
    marginals = [np.diag(INITIAL_STD**2)]
    for _ in range(EPOCHS - 1):
        means.append(PHI @ means[-1])
        marginals.append(PHI @ marginals[-1] @ PHI.T + np.diag(PROCESS_STD**2))
    prior_mean = np.concatenate(means)

    prior = np.zeros((SIZE, SIZE))
    for early in range(EPOCHS):
        for late in range(early, EPOCHS):
            block = np.linalg.matrix_power(PHI, late - early) @ marginals[early]
            prior[late * WIDTH : (late + 1) * WIDTH, early * WIDTH : (early + 1) * WIDTH] = block
            prior[early * WIDTH : (early + 1) * WIDTH, late * WIDTH : (late + 1) * WIDTH] = block.T

    rows = factors.shape[1]
    design = np.zeros((EPOCHS * rows, SIZE))
    data = np.zeros(EPOCHS * rows)
    for epoch in range(EPOCHS):
        design[epoch * rows : (epoch + 1) * rows, epoch * WIDTH : (epoch + 1) * WIDTH] = (
            factors[epoch][:, N_THETA:]
        )
        data[epoch * rows : (epoch + 1) * rows] = (
            targets[epoch] - factors[epoch][:, :N_THETA] @ THETA
        )

    innovation = design @ prior @ design.T + np.eye(EPOCHS * rows)
    gain = prior @ design.T @ np.linalg.inv(innovation)
    return prior_mean + gain @ (data - design @ prior_mean), prior - gain @ design @ prior


def _assembled() -> tuple[np.ndarray, np.ndarray]:
    """``(mean, covariance)`` read off the module's own square-root assembly.

    ``_zeta_joint`` returns ``R`` with joint precision ``R.T @ R``, so the
    covariance is ``R^-1 R^-T`` and the mean ``R^-1 rhs``. Inverting ``R`` here
    rather than asking ``smooth`` is the point of the file: ``smooth`` returns
    the diagonal and this is what it had.
    """
    triangular, rhs, epochs, n_zeta = _zeta_joint(
        _blocks(), _transition(), _values(), NAMES, SHAPES
    )
    assert (epochs, n_zeta) == (EPOCHS, WIDTH)
    inverse = np.linalg.inv(np.asarray(triangular))
    return inverse @ np.asarray(rhs), inverse @ inverse.T


class TestTheJointAgreesWithTheForwardOracle:
    def test_every_entry_of_the_joint_covariance_agrees(self):
        """All 225 entries, not the 15 on the diagonal.

        Measured: 4.857e-16 absolute, 7.963e-13 relative. The tolerances are
        two orders looser than that, which is a margin for BLAS variation
        across platforms and not a place to absorb a disagreement -- see this
        repository's rule on fixtures that pin one machine's arithmetic.
        """
        _, covariance = _assembled()
        _, expected = forward_oracle(_blocks())
        np.testing.assert_allclose(covariance, expected, rtol=1e-10, atol=1e-14)

    def test_the_joint_mean_agrees(self):
        """Measured: 1.110e-15 at worst, on entries of order 1."""
        mean, _ = _assembled()
        expected, _ = forward_oracle(_blocks())
        np.testing.assert_allclose(mean, expected, rtol=1e-10, atol=1e-13)

    def test_the_cross_epoch_part_is_large_enough_to_have_been_a_test(self):
        """The anti-vacuity half: the off-diagonal is 39.5 % of the diagonal.

        Without this, a joint that happened to be near-diagonal would make the
        test above no stronger than the marginal check that already existed,
        and nobody reading a green run could tell.
        """
        _, covariance = _assembled()
        diagonal = np.abs(np.diag(covariance)).max()
        off = np.abs(covariance - np.diag(np.diag(covariance))).max()
        assert diagonal == pytest.approx(4.822666e-01, rel=1e-5)
        assert off == pytest.approx(1.904192e-01, rel=1e-5)
        assert off / diagonal > 0.3

    def test_the_first_and_last_epoch_still_covary(self):
        """Four transitions apart, and the corner block is 1.480e-2.

        A recursion that coupled only neighbours -- or a smoother that had
        quietly become a filter -- would put roughly zero here while every
        marginal stayed plausible.
        """
        _, covariance = _assembled()
        corner = covariance[:WIDTH, (EPOCHS - 1) * WIDTH :]
        assert np.abs(corner).max() == pytest.approx(1.479850e-02, rel=1e-5)

    def test_a_diagonal_only_comparison_would_not_have_caught_this(self):
        """The claim the file is named for, as a failing comparison.

        The marginals of the assembly agree with the oracle's diagonal -- so a
        check written at that granularity passes on a matrix whose cross-epoch
        structure has been deleted entirely. Here that matrix is built and the
        full comparison is shown to reject it.
        """
        _, covariance = _assembled()
        _, expected = forward_oracle(_blocks())
        stripped = np.diag(np.diag(covariance))

        np.testing.assert_allclose(
            np.diag(stripped), np.diag(expected), rtol=1e-10, atol=1e-14
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(stripped, expected, rtol=1e-10, atol=1e-14)

    def test_smooth_reports_this_joint_s_own_diagonal(self):
        """The public function is tied to the object just checked.

        Otherwise the file would verify a private assembly while ``smooth``
        returned something else. Measured: 2.100e-16 relative on the variance,
        4.441e-16 absolute on the mean.
        """
        mean, variance = smooth(_blocks(), _transition(), _values(), NAMES, SHAPES)
        joint_mean, covariance = _assembled()
        np.testing.assert_allclose(
            np.asarray(variance).ravel(), np.diag(covariance), rtol=1e-11
        )
        np.testing.assert_allclose(np.asarray(mean).ravel(), joint_mean, atol=1e-12)


class TestTheOracleItself:
    """A wrong oracle agrees with nothing and blames the code, so pin it."""

    def test_the_oracle_s_prior_is_the_transition_s_own_stationary_law(self):
        """With no data the posterior must BE the prior, and the prior must
        satisfy the Lyapunov recursion the transition declares.

        Built by passing zero design rows, so the update has nothing to fold
        in -- the one case where the oracle's answer is known in closed form
        without consulting the implementation it is used to check.
        """
        empty = (
            jnp.zeros((EPOCHS, 1, N_THETA + WIDTH)),
            jnp.zeros((EPOCHS, 1)),
            jnp.zeros((EPOCHS,)),
        )
        mean, covariance = forward_oracle(empty)

        np.testing.assert_allclose(mean[:WIDTH], INITIAL_MEAN, atol=1e-14)
        np.testing.assert_allclose(
            covariance[:WIDTH, :WIDTH], np.diag(INITIAL_STD**2), atol=1e-14
        )
        for epoch in range(EPOCHS - 1):
            here = slice(epoch * WIDTH, (epoch + 1) * WIDTH)
            nxt = slice((epoch + 1) * WIDTH, (epoch + 2) * WIDTH)
            np.testing.assert_allclose(mean[nxt], PHI @ mean[here], atol=1e-14)
            np.testing.assert_allclose(
                covariance[nxt, nxt],
                PHI @ covariance[here, here] @ PHI.T + np.diag(PROCESS_STD**2),
                atol=1e-14,
            )
            np.testing.assert_allclose(
                covariance[nxt, here], PHI @ covariance[here, here], atol=1e-14
            )

    def test_the_oracle_is_symmetric_and_positive_definite(self):
        """Measured: 5.612e-16 asymmetry, smallest eigenvalue 4.192e-02."""
        _, covariance = forward_oracle(_blocks())
        assert np.max(np.abs(covariance - covariance.T)) < 1e-14
        assert float(np.linalg.eigvalsh(covariance).min()) > 0.0
