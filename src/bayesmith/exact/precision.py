"""``N^-1`` as two operations, so a covariance need not be per-sample diagonal.

Every consumer of noise in this package -- the normal operator's quadratic
form, the Gaussian log-density, the Fisher matrix, the evidence layer's
whitening -- reads the same ``1 / sigma**2`` weight today, and so they share
one assumption: that samples are independent. For a real radiometer that is
the most conspicuous gap in the physics, ahead of any engine choice. There is
no path for 1/f gain drift, for atmospheric correlation, or for channel-to-
channel covariance.

**The generalisation is two operations rather than a sigma**::

    apply(residual)     ->  N^-1 r
    log_normalizer()    ->  log det (2 pi N)

Diagonal degenerates to per-sample division and ``sum(log 2 pi sigma**2)``,
which is the whole of the acceptance criterion: the existing behaviour must
come back numerically, not merely in spirit.

**One object, three consumers.** A :class:`Precision` is built AT a prediction
and then handed unchanged to the density, the solve and the information. That
is the load-bearing discipline, not a convenience: the two halves of a
Gaussian -- the quadratic form and the log-determinant -- disagreeing about
which covariance they describe is exactly defect B1, where one engine dropped
``sum log sigma`` and another did not. Two objects can drift; one cannot.

**On FFT and Toeplitz.** A CIRCULANT matrix is exactly diagonalised by the
DFT; a Toeplitz matrix is not. Measured on a symmetric 8-point kernel, the
FFT of a circulant's first column reproduces its eigenvalues, its inverse
applied to a vector, and its log-determinant, all to roundoff -- while the
same construction on the Toeplitz matrix built from the same kernel does not
reproduce the eigenvalues at all. So the stationary implementation here is
:class:`CirculantPrecision`, named for what it is. A genuinely non-periodic
Toeplitz covariance is a different object, and treating it by FFT is an
approximation that would have to declare itself; it is not offered rather
than being offered under a name that implies exactness.

Circulant is also the right model for the physics that motivated this:
1/f drift and atmospheric correlation are stationary, and a periodic
boundary is the honest statement of "stationary on this stretch".
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp


@runtime_checkable
class Precision(Protocol):
    """``N^-1``, as the two things every consumer of a covariance needs.

    Deliberately not ``sigma``: a sigma is only a covariance when the samples
    are independent, and the whole point here is that they need not be.
    """

    def apply(self, residual: jax.Array) -> jax.Array:
        """``N^-1 r``, shaped like ``r``."""
        ...

    def log_normalizer(self) -> jax.Array:
        """``log det (2 pi N)`` -- the Gaussian's normalisation, a scalar."""
        ...


def quadratic(precision: Precision, residual: jax.Array) -> jax.Array:
    """``r^T N^-1 r``.

    Derived from :meth:`Precision.apply` rather than being a third protocol
    method, so an implementation cannot make the quadratic form and the
    operator disagree -- which is the same failure this module exists to
    prevent one level up.
    """
    return jnp.sum(residual * precision.apply(residual))


def log_density(precision: Precision, residual: jax.Array) -> jax.Array:
    """``-1/2 [ r^T N^-1 r + log det 2 pi N ]``.

    Both halves from ONE object. A caller assembling them from two would be
    free to take them at different covariances, which is defect B1.
    """
    return -0.5 * (quadratic(precision, residual) + precision.log_normalizer())


class DiagonalPrecision(eqx.Module):
    """Independent samples: the behaviour this package had before.

    The degenerate case, and the cross-check for everything else. Its numbers
    must be the old numbers, which is why
    ``tests/exact/test_precision.py`` compares it against the literal
    ``1 / sigma**2`` and ``sum(log 2 pi sigma**2)`` expressions rather than
    against another implementation of itself.
    """

    sigma: jax.Array

    def apply(self, residual: jax.Array) -> jax.Array:
        return residual / self.sigma**2

    def log_normalizer(self) -> jax.Array:
        return jnp.sum(jnp.log(2.0 * jnp.pi * self.sigma**2))


class CirculantPrecision(eqx.Module):
    """A stationary covariance with a periodic boundary, exactly by FFT.

    ``first_column`` is the covariance's first column, i.e. the
    autocovariance at lags ``0, 1, ..., n-1`` wrapped periodically, so it must
    be symmetric under ``k -> n - k`` for the matrix to be. The eigenvalues
    are ``fft(first_column)``, real for such a column.

    Attributes:
        first_column: the autocovariance kernel, length ``n``.

    Raises:
        ValueError: at construction, if any eigenvalue is not strictly
            positive. A covariance that is not positive-definite has no
            inverse and no log-determinant, and the FFT will happily return a
            finite answer for both -- a negative eigenvalue gives a NaN log
            and a sign-flipped weight, neither of which announces itself.
    """

    first_column: jax.Array

    def __check_init__(self):
        eigenvalues = jnp.real(jnp.fft.fft(self.first_column))
        smallest = float(jnp.min(eigenvalues))
        if not smallest > 0.0:  # `not >` so a NaN eigenvalue is refused too
            raise ValueError(
                f"circulant covariance is not positive definite: its smallest "
                f"eigenvalue is {smallest:.6g}. The eigenvalues are the FFT of "
                "the first column, so this is a statement about the "
                "autocovariance kernel, not about conditioning -- a kernel "
                "that falls off too slowly, or one that is not symmetric "
                "under k -> n - k, produces it."
            )

    @property
    def eigenvalues(self) -> jax.Array:
        """``fft(first_column)``, real. The covariance's spectrum."""
        return jnp.real(jnp.fft.fft(self.first_column))

    def apply(self, residual: jax.Array) -> jax.Array:
        spectrum = jnp.fft.fft(residual) / self.eigenvalues
        return jnp.real(jnp.fft.ifft(spectrum))

    def log_normalizer(self) -> jax.Array:
        size = self.first_column.shape[0]
        return size * math.log(2.0 * jnp.pi) + jnp.sum(jnp.log(self.eigenvalues))


def dense(precision: Precision, size: int, dtype: Any = None) -> jax.Array:
    """``N^-1`` materialised, by applying it to a basis. For tests and oracles.

    Deliberately built by APPLICATION rather than by any implementation's own
    internals, so an oracle comparing against it is comparing against what
    callers will actually get.
    """
    basis = jnp.eye(size, dtype=dtype or jnp.zeros(()).dtype)
    return jax.vmap(precision.apply)(basis).T
