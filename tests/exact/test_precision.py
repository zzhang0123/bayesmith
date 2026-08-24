"""B9's acceptance: the diagonal case must come back numerically."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from bayesmith.exact.precision import (
    CirculantPrecision,
    DiagonalPrecision,
    Precision,
    dense,
    log_density,
    quadratic,
)


def _kernel(size: int = 8, decay: float = 0.45, amplitude: float = 4.0):
    """A symmetric, periodic autocovariance -- correlated but positive definite.

    Symmetric under ``k -> n - k`` by construction rather than by luck, which
    is what makes the matrix symmetric and its FFT real.
    """
    lag = np.minimum(np.arange(size), size - np.arange(size))
    return jnp.asarray(amplitude * decay**lag)


class TestTheDiagonalCaseIsTheOldNumbers:
    """The cross-check B9 names: degeneracy has to be numerical, not moral.

    Compared against the literal expressions the package used before -- not
    against another implementation of the same idea -- because two spellings
    of one formula agree even when both are wrong.
    """

    @pytest.mark.parametrize("size", [1, 5, 32])
    def test_apply_is_division_by_sigma_squared(self, size):
        with jax.enable_x64(True):
            sigma = jnp.linspace(0.2, 1.7, size)
            residual = jnp.linspace(-3.0, 2.0, size)
            found = DiagonalPrecision(sigma=sigma).apply(residual)
        assert np.allclose(np.asarray(found), np.asarray(residual / sigma**2), rtol=0)

    @pytest.mark.parametrize("size", [1, 5, 32])
    def test_log_normalizer_is_the_old_sum(self, size):
        with jax.enable_x64(True):
            sigma = jnp.linspace(0.2, 1.7, size)
            found = DiagonalPrecision(sigma=sigma).log_normalizer()
            expected = jnp.sum(jnp.log(2.0 * jnp.pi * sigma**2))
        assert float(found) == float(expected)

    def test_the_density_is_the_gaussian_the_package_already_writes(self):
        """``-1/2 sum[r^2/sigma^2 + log 2 pi sigma^2]``, spelled out."""
        with jax.enable_x64(True):
            sigma = jnp.linspace(0.3, 1.1, 6)
            residual = jnp.linspace(-2.0, 1.5, 6)
            found = log_density(DiagonalPrecision(sigma=sigma), residual)
            expected = -0.5 * jnp.sum(
                residual**2 / sigma**2 + jnp.log(2.0 * jnp.pi * sigma**2)
            )
        assert float(found) == pytest.approx(float(expected), rel=1e-14)

    def test_a_diagonal_circulant_is_the_diagonal_case(self):
        """The two implementations must meet where they overlap.

        A circulant whose kernel is ``[s, 0, 0, ...]`` IS ``s I``. If the FFT
        route and the division route disagree there, one of them is wrong,
        and this is the only fixture where both are defined on the same
        matrix.
        """
        size, variance = 16, 0.75
        with jax.enable_x64(True):
            kernel = jnp.zeros(size).at[0].set(variance)
            circulant = CirculantPrecision(first_column=kernel)
            diagonal = DiagonalPrecision(sigma=jnp.full((size,), math.sqrt(variance)))
            residual = jnp.linspace(-1.0, 1.0, size)
            assert np.allclose(
                np.asarray(circulant.apply(residual)),
                np.asarray(diagonal.apply(residual)),
                rtol=1e-12,
            )
            assert float(circulant.log_normalizer()) == pytest.approx(
                float(diagonal.log_normalizer()), rel=1e-12
            )


class TestTheCirculantOracleIsDirectMatrixInversion:
    """B9's second acceptance clause, against a dense inverse rather than a
    second FFT.

    The oracle shares nothing with the implementation: scipy builds the matrix
    from the same kernel, numpy inverts it densely, numpy takes its
    log-determinant by LU. If the FFT route were wrong in a way that happened
    to be self-consistent, this is what would say so.
    """

    @pytest.mark.parametrize("size", [4, 8, 15])
    def test_apply_reproduces_the_dense_inverse(self, size):
        with jax.enable_x64(True):
            kernel = _kernel(size)
            found = np.asarray(dense(CirculantPrecision(first_column=kernel), size))
            matrix = sla.circulant(np.asarray(kernel))
        assert np.allclose(found, np.linalg.inv(matrix), rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("size", [4, 8, 15])
    def test_the_normalizer_reproduces_the_dense_log_determinant(self, size):
        with jax.enable_x64(True):
            kernel = _kernel(size)
            found = float(CirculantPrecision(first_column=kernel).log_normalizer())
            matrix = sla.circulant(np.asarray(kernel))
        _, expected = np.linalg.slogdet(2.0 * np.pi * matrix)
        assert found == pytest.approx(expected, rel=1e-10)

    def test_the_quadratic_form_matches_the_dense_one(self):
        with jax.enable_x64(True):
            kernel = _kernel(12)
            residual = jnp.asarray(np.linspace(-2.0, 3.0, 12))
            found = float(quadratic(CirculantPrecision(first_column=kernel), residual))
            matrix = sla.circulant(np.asarray(kernel))
            r = np.asarray(residual)
        expected = float(r @ np.linalg.solve(matrix, r))
        assert found == pytest.approx(expected, rel=1e-10)

    def test_correlation_is_actually_present_in_the_fixture(self):
        """ANTI-VACUITY. Every test above would also pass on a diagonal.

        Without this, a kernel that had quietly become ``[a, 0, 0, ...]`` --
        a decay of 0, a broken construction -- would leave the whole class
        green while testing nothing the diagonal case does not already cover.
        """
        with jax.enable_x64(True):
            matrix = np.asarray(dense(CirculantPrecision(first_column=_kernel(8)), 8))
        off = matrix - np.diag(np.diag(matrix))
        assert np.max(np.abs(off)) > 0.05 * np.max(np.abs(np.diag(matrix)))


class TestTheContractRefusesWhatItCannotDescribe:
    def test_a_kernel_that_is_not_positive_definite_is_refused(self):
        """The FFT answers finitely for an indefinite covariance, so the
        constructor has to be the thing that does not.

        A negative eigenvalue gives ``log`` a NaN and ``apply`` a sign-flipped
        weight in that mode -- a residual pushed the WRONG WAY, which no
        finiteness check downstream would catch.
        """
        with jax.enable_x64(True):
            # alternating kernel: its highest-frequency eigenvalue goes negative
            kernel = jnp.asarray([1.0, -0.9, 0.8, -0.9])
            with pytest.raises(ValueError, match="positive definite"):
                CirculantPrecision(first_column=kernel)

    def test_both_implementations_satisfy_the_protocol(self):
        """``Precision`` is runtime-checkable, so this is a real check."""
        with jax.enable_x64(True):
            assert isinstance(DiagonalPrecision(sigma=jnp.ones(3)), Precision)
            assert isinstance(CirculantPrecision(first_column=_kernel(4)), Precision)

    def test_the_quadratic_form_cannot_disagree_with_the_operator(self):
        """``quadratic`` is derived, so this holds for any implementation.

        Stated as a test anyway because it is the property the module is for:
        a consumer taking ``r^T N^-1 r`` and a consumer taking ``N^-1 r`` are
        reading one object.
        """
        with jax.enable_x64(True):
            precision = CirculantPrecision(first_column=_kernel(8))
            residual = jnp.linspace(-1.0, 2.0, 8)
            assert float(quadratic(precision, residual)) == pytest.approx(
                float(jnp.sum(residual * precision.apply(residual))), rel=1e-15
            )
