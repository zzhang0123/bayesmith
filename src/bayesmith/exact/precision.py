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

    def log_spectrum(self) -> jax.Array:
        """``log lambda_k`` -- the log of ``N``'s eigenvalues, as a VECTOR.

        **The fourth operation, and it is what the Fisher matrix needs.** When
        the covariance depends on the parameters, the information carries a
        second term ``1/2 tr(N^-1 d_a N N^-1 d_b N)``, and whenever ``N``'s
        eigenBASIS does not move with the parameters that collapses onto the
        spectrum::

            1/2 sum_k d_a log lambda_k  d_b log lambda_k

        Both implementations here have a fixed basis -- ``I`` for a diagonal,
        the DFT for a circulant -- so the identity covers both, and the
        diagonal rule ``exact/fisher.py`` shipped,
        ``2 (dlog sigma/dx)^T (dlog sigma/dx)``, is that at
        ``lambda_i = sigma_i**2``. Derived in
        ``docs/derivations/variance_information_spectral.wls`` and measured
        against a dense finite-difference Fisher matrix in
        ``docs/probes/probe_9_correlated_variance_information.py``.

        It is the VECTOR and not its sum, which is why
        :meth:`log_normalizer` cannot supply it: a sum of logs cannot give
        back the sum of their products. The two are related --
        ``log_normalizer() == n log 2 pi + sum(log_spectrum())`` -- and
        ``test_log_normalizer_is_the_log_spectrums_own_sum`` pins that, since
        nothing else renders the two side by side.

        A covariance whose eigenbasis DOES move with the parameters (say
        ``D(theta) C D(theta)`` with a per-sample ``D``) is not covered by the
        identity, and is also not something either implementation here can
        represent -- measured: exact 2.7847 against spectral 2.7595 on such a
        case.
        """
        ...

    def whiten(self, omega: jax.Array) -> jax.Array:
        """``N^-1/2 omega`` -- a standard normal draw given this covariance.

        **The interface needs three operations, not the two B9 specifies.**
        ``gcr_sample`` draws ``omega ~ N(0, I)`` and forms ``sqrt(w) * omega``
        with ``w = 1/sigma**2``, which is exactly ``N^-1/2 omega``; it cannot
        be built from :meth:`apply` and :meth:`log_normalizer`. The spec
        acknowledges the operation for the evidence layer -- "the whitening
        row becomes ``L^-1 r``" -- without carrying it into the interface that
        layer would call.

        The defining property, and what the tests assert rather than any
        particular square root: applying it twice is :meth:`apply`, because
        ``N^-1/2 N^-1/2 = N^-1``. Any square root satisfying that is a legal
        implementation, which is why this does not promise a Cholesky.
        """
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

    def log_spectrum(self) -> jax.Array:
        # `2 log sigma` rather than `log(sigma**2)`: multiplying a float by 2
        # is exact in binary, so half of this is `log sigma` BITWISE, and the
        # Fisher term's diagonal case stays the number it always was.
        return 2.0 * jnp.log(self.sigma)

    def whiten(self, omega: jax.Array) -> jax.Array:
        return omega / self.sigma


class CirculantPrecision(eqx.Module):
    """A stationary covariance with a periodic boundary, exactly by FFT.

    ``first_column`` is the covariance's first column, i.e. the
    autocovariance at lags ``0, 1, ..., n-1`` wrapped periodically, so it must
    be symmetric under ``k -> n - k`` for the matrix to be. The eigenvalues
    are ``fft(first_column)``, real for such a column.

    Attributes:
        first_column: the autocovariance kernel, length ``n``.

    **Construction does not validate.** Positive-definiteness is checked by
    :func:`check_positive_definite`, a separate eager call, and the split is
    not stylistic: validating in ``__check_init__`` requires reading the
    smallest eigenvalue as a Python float, which concretises. Measured, that
    made this class raise ``ConcretizationTypeError`` under ``jit``,
    ``linearize`` AND ``grad`` -- so it could not be built anywhere its kernel
    was traced, which is the entire solve path. ``gaussian.py`` already splits
    ``gaussian_parts`` from ``check_gaussian`` for exactly this reason, and
    this follows it: validate once, eagerly, at the boundary; trace freely
    afterwards.
    """

    first_column: jax.Array

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

    def log_spectrum(self) -> jax.Array:
        return jnp.log(self.eigenvalues)

    def whiten(self, omega: jax.Array) -> jax.Array:
        spectrum = jnp.fft.fft(omega) / jnp.sqrt(self.eigenvalues)
        return jnp.real(jnp.fft.ifft(spectrum))


def check_positive_definite(precision: CirculantPrecision) -> None:
    """Refuse a circulant covariance that is not positive definite.

    Call this EAGERLY, at the point a kernel enters the system, the way
    ``block.py`` and ``classify.py`` call ``check_gaussian``. It reads the
    smallest eigenvalue as a Python float and therefore cannot run under a
    trace -- which is precisely why it is not in the constructor.

    The check is needed because the FFT does not fail on an indefinite
    covariance, it answers finitely: a negative eigenvalue gives ``log`` a NaN
    and pushes that mode's residual the WRONG WAY, and neither announces
    itself downstream.

    Raises:
        ValueError: if any eigenvalue is not strictly positive.
    """
    eigenvalues = jnp.real(jnp.fft.fft(precision.first_column))
    smallest = float(jnp.min(eigenvalues))
    if not smallest > 0.0:  # `not >` so a NaN eigenvalue is refused too
        raise ValueError(
            f"circulant covariance is not positive definite: its smallest "
            f"eigenvalue is {smallest:.6g}. The eigenvalues are the FFT of "
            "the first column, so this is a statement about the "
            "autocovariance kernel, not about conditioning -- a kernel that "
            "falls off too slowly, or one that is not symmetric under "
            "k -> n - k, produces it."
        )


class PrecisionMismatch(Exception):
    """A :class:`Precision` does not describe the distribution it was paired with."""


def check_precision(
    distribution: Any,
    precision: Precision,
    loc: jax.Array,
    *,
    rtol: float | None = None,
    key: jax.Array | None = None,
) -> dict[str, float]:
    """Verify a ``Precision`` really is the covariance of ``distribution``.

    The counterpart of :func:`~bayesmith.exact.gaussian.check_gaussian` for a
    covariance that is not per-sample diagonal, and the piece that decides
    whether extracting a ``Precision`` from a node's own distribution is sound
    at all. Runs on concrete values, **outside** any trace.

    **Why a scalar-offset family is not enough here.** ``check_gaussian``
    probes ``log_prob`` at five multiples of ``sqrt(diag N)``. For a
    STATIONARY covariance that quantity is constant, so the five offsets are
    five points along ONE direction, and the family constrains only two
    scalars of the covariance. Measured on two circulant covariances 0.4060
    apart in relative Frobenius norm: they agree at every offset to 1.851e-16,
    against an rtol of 2.220e-13. More offsets do not help -- they are more
    points on the same line. For an ``n``-point kernel that leaves ``n/2 - 1``
    independent spectral parameters unchecked, and no fixture in this package
    is correlated, so nothing would have shown it.

    **What replaces it.** For any Gaussian,
    ``grad log_prob(x) = -N^-1 (x - loc)`` exactly, so ONE reverse-mode pass
    yields the whole vector and can be compared against
    :meth:`Precision.apply` elementwise -- ``n`` equations from one AD
    evaluation where a scalar probe gives one. On the same pair: 2.220e-16 for
    the matched pairing against 1.882e-01 for the mismatched, a separation of
    8.48e+14.

    **Both halves are required and neither subsumes the other**, which is why
    this returns two numbers rather than a verdict. The gradient cannot see
    the normaliser, since it does not appear in it; and the normaliser cannot
    separate that pair either, because the two covariances share it by
    construction -- both report ``log_prob(loc) = -9.596667351679``. Only the
    two together pin the Gaussian.

    Linearity is checked for free, and is the third thing that must hold: the
    gradient of a quadratic log-density is linear, so ``grad`` at ``2r`` must
    be exactly twice ``grad`` at ``r``. A density that is not quadratic has no
    covariance to extract, and this is what notices.

    Args:
        distribution: the node's own distribution, the authority on the
            density. Must expose ``log_prob``.
        precision: the ``N^-1`` claimed to describe it.
        loc: the distribution's mean, where the normaliser is read.
        rtol: per-number tolerance. Defaults to ``1e3 * eps`` of ``loc``'s
            dtype, matching ``check_gaussian``.
        key: PRNG key for the displacement. Fixed by default, so a failure is
            reproducible; the check is deterministic given it.

    Returns:
        ``{"operator": worst elementwise relative error of N^-1 r,
        "normalizer": relative error at the mode,
        "linearity": worst relative error of grad(2r) - 2 grad(r)}``.
        Reported rather than reduced to a bool, so a caller can say HOW far
        off a node is, not only that it is.

    Raises:
        PrecisionMismatch: if any of the three exceeds ``rtol``. Which one is
            named, because "the covariance is wrong" and "the normaliser is
            wrong" and "the density is not Gaussian" need different fixes.
    """
    tolerance = (
        1e3 * float(jnp.finfo(jnp.asarray(loc).dtype).eps) if rtol is None else rtol
    )
    centre = jnp.asarray(loc)
    displacement = jax.random.normal(
        jax.random.key(0) if key is None else key, centre.shape, centre.dtype
    )

    def density(value: jax.Array) -> jax.Array:
        return jnp.sum(distribution.log_prob(value))

    gradient = jax.grad(density)(centre + displacement)
    doubled = jax.grad(density)(centre + 2.0 * displacement)
    applied = precision.apply(displacement)

    def worst(found: jax.Array, expected: jax.Array) -> float:
        scale = jnp.maximum(jnp.abs(expected), jnp.abs(found))
        return float(
            jnp.max(jnp.abs(found - expected) / jnp.where(scale > 0.0, scale, 1.0))
        )

    at_mode = float(jnp.sum(distribution.log_prob(centre)))
    expected_mode = -0.5 * float(precision.log_normalizer())
    denominator = max(abs(expected_mode), abs(at_mode), 1.0)

    errors = {
        "operator": worst(-gradient, applied),
        "normalizer": abs(at_mode - expected_mode) / denominator,
        "linearity": worst(doubled, 2.0 * gradient),
    }
    offenders = {
        name: value for name, value in errors.items() if not value <= tolerance
    }
    if offenders:
        named = ", ".join(
            f"{name}={value:.3e}" for name, value in sorted(offenders.items())
        )
        raise PrecisionMismatch(
            f"this Precision does not describe that distribution: {named}, "
            f"against rtol {tolerance:.3e}. "
            "'operator' means the covariance itself disagrees -- the gradient "
            "identity grad log_prob(x) = -N^-1 (x - loc) is exact for any "
            "Gaussian, so a mismatch there is a different covariance, not "
            "roundoff. 'normalizer' means log det N disagrees while the "
            "operator may not. 'linearity' means the log-density is not "
            "quadratic, so it has no covariance to extract and the other two "
            "numbers are meaningless."
        )
    return errors


def diagonal_from(noise_std: dict[str, Any]) -> dict[str, DiagonalPrecision]:
    """``{observed: sigma}`` -> ``{observed: Precision}``.

    The bridge from the decided-sigma dict the package passes around today.
    It exists so the generalisation can land without every caller changing at
    once, and so the diagonal path keeps going through the SAME code the
    correlated one does -- a compatibility shim that bypassed the protocol
    would leave the degeneracy untested where it matters.
    """
    return {
        name: DiagonalPrecision(sigma=jnp.asarray(sigma))
        for name, sigma in noise_std.items()
    }


def dense(precision: Precision, size: int, dtype: Any = None) -> jax.Array:
    """``N^-1`` materialised, by applying it to a basis. For tests and oracles.

    Deliberately built by APPLICATION rather than by any implementation's own
    internals, so an oracle comparing against it is comparing against what
    callers will actually get.
    """
    basis = jnp.eye(size, dtype=dtype or jnp.zeros(()).dtype)
    return jax.vmap(precision.apply)(basis).T
