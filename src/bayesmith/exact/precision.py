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

**An unobserved sample is the third thing a covariance can say, and it is a
DECLARATION rather than a value.** ``sigma = inf`` is how upstream spells a
flagged sample, and the four operations above disagree about what it means to
them -- ``apply`` and ``whiten`` give zero, ``log_normalizer`` gives ``+inf``,
correctly, because infinite variance has no density. Reading that ``+inf`` as
``0`` is the statement "this sample was not observed", which is a modelling
concept the interface does not have. So the node carries the mask
(:attr:`~bayesmith.graph.nodes.Probabilistic.observed_mask`),
:class:`MaskedPrecision` is what :func:`~bayesmith.exact.gaussian.precision_at`
builds from it, and :class:`DiagonalPrecision` keeps a normaliser that is never
silently wrong. Masking is DIAGONAL and that is measured, not assumed --
:mod:`bayesmith.evidence.compress` carries the number.

**Nothing in this module is exported, and that is the design rather than an
oversight.** :func:`~bayesmith.exact.gaussian.precision_at` is the seam a
caller touches: it builds the object from the graph at a point, and the three
consumers above take it from there. A user declares correlated noise by
declaring the DISTRIBUTION -- numpyro's ``CirculantNormal`` is the case B9 was
built for -- not by constructing a :class:`CirculantPrecision` and handing it
in. So the protocol and its implementations are machinery, and
``bayesmith.exact.__all__`` says so by leaving them out.

Written down because omission is a poor way to say anything. An absent export
is indistinguishable from a forgotten one, and this one was read as forgotten
by a reviewer who had no way to tell -- the same shape as every other entry in
this repository that exists only as a gap. If a caller ever does need to
implement :class:`Precision` themselves, a Toeplitz kernel being the obvious
candidate and the paragraph above saying why it is not this class, then the
protocol becomes public API and this is the paragraph to revisit, on purpose.
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

    **A sample whose weight is exactly zero contributes exactly zero, whatever
    its residual.** That is what zero weight means, and without saying it here
    the derivation above would be broken by the one implementation that has
    zero weights: ``apply`` returns ``0`` at a
    :class:`MaskedPrecision`'s unobserved samples, but the multiplication is
    HERE, and ``nan * 0`` is ``nan``. A flagged datum is routinely ``nan`` --
    that is what an unrecorded sample looks like in a file -- so a masked model
    whose solve was clean would still hand back a ``nan`` log-density. Measured:
    ``[1, 2, nan, 3]`` against a mask ``[T, T, F, T]`` gave ``apply`` a clean
    ``[4, 8, 0, 12]`` and ``quadratic`` a ``nan``.

    The guard is on the WEIGHT and not on the residual, which is the whole
    distinction: a non-zero weight times a ``nan`` is still ``nan``, so a
    poisoned datum that WAS observed stays loud. Mapping that one to zero would
    mean "unobserved", which is a claim only the model may make -- the same
    argument the sibling package makes for keeping two weight formulas that
    disagree on ``nan``. For every finite residual this is bitwise the
    expression it replaces, since ``r * 0`` is exactly ``0``.
    """
    weighted = precision.apply(residual)
    return jnp.sum(jnp.where(weighted == 0, jnp.zeros_like(weighted), residual * weighted))


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

    def log_normalizer_terms(self) -> jax.Array:
        """``log 2 pi sigma_i**2`` per sample -- the summands, as a VECTOR.

        Beyond the protocol, and only :class:`MaskedPrecision` reads it: a
        subset determinant is a subset SUM, and a sum of logs cannot be
        un-summed. It exists so the masked normaliser is the same expression
        as the full one with terms dropped, rather than a second spelling of
        it that can drift -- ``test_the_masked_normaliser_is_the_kept_terms``
        pins that the two agree bitwise.
        """
        return jnp.log(2.0 * jnp.pi * self.sigma**2)

    def log_normalizer(self) -> jax.Array:
        return jnp.sum(self.log_normalizer_terms())

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

    **A BATCHED kernel is supported and is how a plate arrives.** Shape
    ``(batch, n)`` is ``batch`` independent circulants, one per row -- every
    operation here works along the LAST axis, which is where the FFT belongs,
    and the whole thing describes a block-diagonal covariance over
    ``batch * n`` samples. Checked against a dense block-diagonal reference.

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
        # `.size`, not `.shape[0]`. A BATCHED kernel -- shape `(batch, n)`, one
        # independent circulant per row -- describes `batch * n` samples, and
        # `shape[0]` counted the batch. Identical for the 1-D case this class
        # was written against; measured wrong by `(batch*n - batch) log 2 pi`
        # otherwise, which is 16.54 nats at `(3, 4)`.
        #
        # `apply`, `whiten`, `quadratic` and `log_spectrum` were all already
        # right batched -- they work along the LAST axis, which is where the
        # FFT belongs. Only the count was wrong, which is why nothing that
        # exercised the operator half could see it.
        size = self.first_column.size
        return size * math.log(2.0 * jnp.pi) + jnp.sum(jnp.log(self.eigenvalues))

    def log_spectrum(self) -> jax.Array:
        return jnp.log(self.eigenvalues)

    def whiten(self, omega: jax.Array) -> jax.Array:
        spectrum = jnp.fft.fft(omega) / jnp.sqrt(self.eigenvalues)
        return jnp.real(jnp.fft.ifft(spectrum))


class MaskedPrecision(eqx.Module):
    """A per-sample covariance in which some samples were not taken at all.

    ``sigma = inf`` is how an unobserved sample arrives from upstream --
    rheplicant's ``FlaggedNoise`` spells RFI that way, and
    :func:`~bayesmith.evidence.compress.observed_mask` already reads it. What
    this class adds is the exact path: a masked sample must contribute
    **nothing** to the quadratic form, nothing to the normaliser, nothing to
    the information, and nothing to a draw.

    **Why the mask is a separate object rather than an ``inf`` inside
    :class:`DiagonalPrecision`.** ``inf`` is a value, and the four operations
    disagree about what it means to them: ``apply`` and ``whiten`` happen to
    give zero, while ``log_normalizer`` gives ``+inf`` -- correctly, because a
    sample with infinite variance has no density. Reading that ``+inf`` as
    ``0`` is a MODELLING statement ("this sample was not observed"), not an
    arithmetic one, and :mod:`bayesmith.evidence.compress` says so at length.
    Making ``DiagonalPrecision`` mask silently would delete the only guard
    that distinguishes "the sigma expression produced an infinity" from "the
    sample was flagged", and those need different fixes.

    So the mask is carried, and every operation reads it:

    ==================  =================================================
    operation           at a masked sample
    ==================  =================================================
    ``apply``           ``0`` -- the sample enters no normal equation
    ``log_normalizer``  omitted from the sum -- the subset determinant
    ``log_spectrum``    ``0`` (``lambda = 1``, a CONSTANT), so the
                        variance-information term gets no contribution
    ``whiten``          ``0`` -- the noise draw has no term for it
    ==================  =================================================

    **Masking is a SELECTION and not a multiplication, and that is what makes
    a flagged ``nan`` harmless.** A flagged datum is routinely ``nan`` -- that
    is what an unrecorded sample looks like in a file -- and ``jnp.where``
    discards the branch it does not take, so ``where(False, nan, 0)`` is
    ``0.0``. Written as ``seen * vector`` the same mask would give ``nan``, and
    one flagged channel would poison the whole solve while every weight looked
    right. ``test_a_nan_at_a_masked_sample_does_not_reach_the_solution`` pins
    the property; the mutation that spells ``_kept`` as a product is what shows
    the guard can fail.

    Stated because it was measured the other way round first: this class began
    by zeroing ``apply``'s INPUT as well as its output, on the reasoning above
    about ``0 * nan``. Mutating that inner call away changed nothing at all --
    correctly, because ``where`` selects -- so the belt was removed and the
    braces named. The one place the multiplication really happens is
    :func:`quadratic`, and it guards there.

    Attributes:
        base: the covariance in force on the samples that WERE taken.
        seen: boolean, shaped like ``sigma``; ``True`` = observed.
    """

    base: DiagonalPrecision
    seen: jax.Array

    def _kept(self, vector: jax.Array) -> jax.Array:
        return jnp.where(self.seen, vector, jnp.zeros((), vector.dtype))

    def apply(self, residual: jax.Array) -> jax.Array:
        return self._kept(self.base.apply(residual))

    def log_normalizer(self) -> jax.Array:
        return jnp.sum(jnp.where(self.seen, self.base.log_normalizer_terms(), 0.0))

    def log_spectrum(self) -> jax.Array:
        return jnp.where(self.seen, self.base.log_spectrum(), 0.0)

    def whiten(self, omega: jax.Array) -> jax.Array:
        return self._kept(self.base.whiten(omega))


def masked(precision: Any, seen: jax.Array) -> MaskedPrecision:
    """Wrap ``precision`` so the samples ``seen`` is ``False`` at inform nothing.

    Asks :func:`per_sample_sigma` whether the covariance HAS per-sample sigmas
    to mask, rather than testing its type here -- that question has one home
    in this package and this is not a second one.

    Raises:
        StructureError: if the covariance has no per-sample sigma. An
            unobserved sample inside a correlated epoch has no exact meaning:
            the observed submatrix of a stationary covariance is not itself
            stationary, and its log-determinant is not a subset sum of the
            spectrum. Measured in :mod:`bayesmith.evidence.compress`: on a
            6-point kernel with one sample dropped the observed submatrix's
            log-determinant is ``-0.7084`` and the closest subset sum of log-
            eigenvalues is 0.47 nats away.
        StructureError: if ``seen`` is not boolean, or does not match the
            sigma's shape. A float mask would multiply rather than select, and
            a broadcasting one would mask a different set of samples than the
            caller named while every shape downstream stayed right.
    """
    from bayesmith.errors import StructureError

    sigma = per_sample_sigma({"_": precision})
    if sigma is None:
        raise StructureError(
            f"a {type(precision).__name__} has no per-sample sigma, so there "
            "is nothing for a mask to select. An unobserved sample inside a "
            "CORRELATED covariance has no exact meaning: the observed "
            "submatrix of a stationary covariance is not itself stationary, "
            "and its log-determinant is not a subset sum of the spectrum. "
            "Split the node at the gap, or declare the covariance over the "
            "samples that were actually taken."
        )
    flags = jnp.asarray(seen)
    if flags.dtype != jnp.bool_:
        raise StructureError(
            f"the observation mask has dtype {flags.dtype}; it must be "
            "boolean. A float mask multiplies where a boolean one selects, so "
            "a 0.5 would halve a sample's weight and call it unobserved."
        )
    if flags.shape != jnp.shape(sigma["_"]):
        raise StructureError(
            f"the observation mask has shape {flags.shape} but the covariance "
            f"is {jnp.shape(sigma['_'])}. Broadcasting these would mask a "
            "different set of samples than the caller named, and every shape "
            "downstream would still be right."
        )
    return MaskedPrecision(
        base=DiagonalPrecision(sigma=jnp.asarray(sigma["_"])), seen=flags
    )


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


def per_sample_sigma(
    precision: dict[str, Any],
) -> dict[str, jax.Array] | None:
    """``{observed: sigma}`` when every covariance has one, else ``None``.

    The inverse of :func:`diagonal_from`, and the one place in this package
    that asks which IMPLEMENTATION a ``Precision`` is. That question is
    legitimate exactly here: "does this covariance have per-sample sigmas" is
    a real property of a model, not a leak -- a stationary covariance has an
    n-point kernel and no per-sample sigma, and reporting its per-mode
    amplitudes under the name ``noise_std`` would be a lie by naming.

    ``None`` rather than a raise: callers who want the numbers want to know
    they are unavailable, not to guard every read. ``GLSResult.noise_std``
    and ``Estimate.noise_std`` are both this, which is why neither stores a
    second copy of the covariance to answer with.

    Returns the ``sigma`` arrays THEMSELVES, so a result built by
    ``diagonal_from(sigma)`` reports back exactly the arrays it was given --
    bitwise, not to a tolerance.

    **A :class:`MaskedPrecision` reports ``inf`` where it was masked**, which
    is not a second encoding but the one this package already had: it is what
    :func:`~bayesmith.evidence.compress.observed_mask` reads, and it is how
    the mask reached bayesmith from upstream in the first place. So
    ``GLSResult.noise_std`` and ``Estimate.noise_std`` say "not observed" in
    the same word their caller used, and ``compress`` masks a masked
    covariance without knowing this class exists.
    """
    sigma: dict[str, jax.Array] = {}
    for name, value in precision.items():
        if isinstance(value, MaskedPrecision):
            sigma[name] = jnp.where(value.seen, value.base.sigma, jnp.inf)
        elif isinstance(value, DiagonalPrecision):
            sigma[name] = value.sigma
        else:
            return None
    return sigma


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
