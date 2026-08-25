"""One epoch compressed, against the exact Gaussian log-likelihood.

The oracle is `scipy`-free dense NumPy: ``log N(d | A x + c, N)`` written out
as ``-1/2 r^T N^-1 r - 1/2 log det(2 pi N)``, with ``N`` materialised and
inverted by `numpy.linalg`. Nothing here compares one of our own routines
against another.

**Absolute log-densities, always.** The compressor's whole job at this layer
is the NORMALISATION -- the quadratic half is a whitening anyone could get
right. A test comparing posteriors would pass for a compressor that dropped
``-1/2 log det(2 pi N)`` entirely.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.evidence import compress, observed_mask
from bayesmith.exact.precision import CirculantPrecision, DiagonalPrecision, dense


def _pieces(n=7, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(size=(n, k))),
        jnp.asarray(rng.normal(size=n)),
        jnp.asarray(rng.normal(size=k)),
    )


def _dense_log_likelihood(design, data, covariance, x, offset_prediction=None):
    """``log N(d | A x + c, N)`` in NumPy, from a materialised covariance."""
    prediction = np.asarray(design) @ np.asarray(x)
    if offset_prediction is not None:
        prediction = prediction + np.asarray(offset_prediction)
    residual = np.asarray(data) - prediction
    inverse = np.linalg.inv(covariance)
    _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
    return -0.5 * float(residual @ inverse @ residual) - 0.5 * float(logdet)


class TestTheCompressedTermIsTheLikelihood:
    @pytest.mark.parametrize("sigma", [0.5, 2.0])
    def test_a_diagonal_epoch_matches_the_dense_log_likelihood(self, sigma):
        with jax.enable_x64(True):
            design, data, x = _pieces()
            precision = DiagonalPrecision(sigma=jnp.full(data.shape, sigma))
            term = compress({"x": design}, data, precision, {"x": (3,)})
            got = float(term.log_prob({"x": x}))
            covariance = np.diag(np.full(data.shape[0], sigma**2))
        assert got == pytest.approx(
            _dense_log_likelihood(design, data, covariance, x), rel=1e-10
        )

    def test_a_heteroscedastic_epoch_matches_it_too(self):
        with jax.enable_x64(True):
            design, data, x = _pieces()
            sigma = jnp.asarray(np.linspace(0.3, 2.1, data.shape[0]))
            term = compress({"x": design}, data, DiagonalPrecision(sigma=sigma), {"x": (3,)})
            got = float(term.log_prob({"x": x}))
            covariance = np.diag(np.asarray(sigma) ** 2)
        assert got == pytest.approx(
            _dense_log_likelihood(design, data, covariance, x), rel=1e-10
        )

    def test_a_CORRELATED_epoch_matches_it(self):
        """The row B9 opened, reaching the evidence layer.

        The covariance is materialised by `precision.dense` -- built by
        APPLICATION, so the reference is inverted from the operator callers
        actually get -- and then inverted in NumPy. No FFT on the oracle side.
        """
        with jax.enable_x64(True):
            design, data, x = _pieces()
            size = data.shape[0]
            lag = np.minimum(np.arange(size), size - np.arange(size))
            precision = CirculantPrecision(
                first_column=jnp.asarray(1.0 * 0.4**lag + 0.5)
            )
            term = compress({"x": design}, data, precision, {"x": (3,)})
            got = float(term.log_prob({"x": x}))
            covariance = np.linalg.inv(
                np.asarray(dense(precision, size, jnp.float64))
            )
        assert got == pytest.approx(
            _dense_log_likelihood(design, data, covariance, x), rel=1e-9
        )

    def test_the_offset_prediction_is_subtracted_before_whitening(self):
        with jax.enable_x64(True):
            design, data, x = _pieces()
            constant = jnp.asarray(np.linspace(-1.0, 1.0, data.shape[0]))
            precision = DiagonalPrecision(sigma=jnp.full(data.shape, 0.8))
            term = compress(
                {"x": design}, data, precision, {"x": (3,)},
                offset_prediction=constant,
            )
            got = float(term.log_prob({"x": x}))
            covariance = np.diag(np.full(data.shape[0], 0.8**2))
        assert got == pytest.approx(
            _dense_log_likelihood(design, data, covariance, x, constant), rel=1e-10
        )

    def test_two_latent_blocks_keep_the_mappings_column_order(self):
        """`combine` needs two terms to agree on column order, and the only
        thing deciding it is the design mapping's iteration order."""
        with jax.enable_x64(True):
            rng = np.random.default_rng(4)
            n = 6
            first = jnp.asarray(rng.normal(size=(n, 2)))
            second = jnp.asarray(rng.normal(size=(n, 1)))
            data = jnp.asarray(rng.normal(size=n))
            precision = DiagonalPrecision(sigma=jnp.full(n, 0.7))
            term = compress(
                {"a": first, "b": second}, data, precision, {"a": (2,), "b": ()}
            )
            values = {"a": jnp.asarray([0.5, -1.0]), "b": jnp.asarray(2.0)}
            got = float(term.log_prob(values))
            stacked = np.concatenate([np.asarray(first), np.asarray(second)], axis=1)
            flat = np.asarray([0.5, -1.0, 2.0])
            covariance = np.diag(np.full(n, 0.7**2))
        assert term.names == ("a", "b")
        assert got == pytest.approx(
            _dense_log_likelihood(stacked, data, covariance, flat), rel=1e-10
        )


class TestTheMaskIsB11sFirstDecisionMade:
    """`sigma = inf` means UNOBSERVED, and this layer is where that is said."""

    def test_an_unobserved_sample_contributes_nothing_at_all(self):
        """Against a reference that simply DROPS that sample.

        Both halves in one assertion: the quadratic (whitening already gives
        weight zero) and the normaliser (which this module masks, because
        `log_normalizer()` is `+inf` there and right to be).
        """
        with jax.enable_x64(True):
            design, data, x = _pieces(n=7)
            sigma = np.linspace(0.4, 1.6, 7)
            sigma[2] = np.inf
            sigma[5] = np.inf
            precision = DiagonalPrecision(sigma=jnp.asarray(sigma))
            term = compress({"x": design}, data, precision, {"x": (3,)})
            got = float(term.log_prob({"x": x}))

        seen = np.isfinite(sigma)
        reference = _dense_log_likelihood(
            np.asarray(design)[seen],
            np.asarray(data)[seen],
            np.diag(sigma[seen] ** 2),
            x,
        )
        assert got == pytest.approx(reference, rel=1e-10)

    def test_dropping_the_mask_would_be_infinitely_wrong_and_this_says_so(self):
        """ANTI-VACUITY, and it names the size of the error.

        `Precision.log_normalizer()` is `+inf` on this covariance, so a
        compressor that used it unmasked would produce `-inf` -- every epoch's
        evidence, and therefore the campaign's, gone. The masked value is
        finite and correct, and the two are compared here so the mask cannot
        quietly stop being applied.
        """
        with jax.enable_x64(True):
            design, data, _ = _pieces(n=7)
            sigma = np.linspace(0.4, 1.6, 7)
            sigma[2] = np.inf
            precision = DiagonalPrecision(sigma=jnp.asarray(sigma))
            unmasked = -0.5 * float(precision.log_normalizer())
            term = compress({"x": design}, data, precision, {"x": (3,)})
            masked = float(term.offset)
            seen = np.isfinite(sigma)
            expected = -0.5 * float(
                np.sum(np.log(2.0 * math.pi * sigma[seen] ** 2))
            )
        assert unmasked == -np.inf
        assert masked == pytest.approx(expected, rel=1e-12)
        assert np.isfinite(masked)

    def test_observed_mask_says_None_when_the_question_does_not_apply(self):
        """A correlated covariance has no per-sample "was this taken"."""
        with jax.enable_x64(True):
            size = 6
            lag = np.minimum(np.arange(size), size - np.arange(size))
            correlated = CirculantPrecision(
                first_column=jnp.asarray(1.0 * 0.4**lag + 0.5)
            )
            diagonal = DiagonalPrecision(sigma=jnp.full(size, 0.5))
            assert observed_mask(correlated) is None
            assert bool(jnp.all(observed_mask(diagonal)))

    def test_a_correlated_epoch_with_a_broken_normaliser_is_refused(self):
        """Not masked, because there is no exact masked normaliser.

        Measured on a 6-point kernel with one sample dropped: the observed
        submatrix's log-determinant is `-0.7084` while the closest subset sum
        of log-eigenvalues is `0.47` nats away, and the submatrix is not
        circulant. Any "mask the spectrum" rule would be an approximation
        wearing an exact result's name.
        """
        with jax.enable_x64(True):
            design, data, _ = _pieces(n=6)
            size = 6
            lag = np.minimum(np.arange(size), size - np.arange(size))
            kernel = np.asarray(1.0 * 0.4**lag + 0.5)
            kernel[0] = np.inf  # an infinite marginal variance
            precision = CirculantPrecision(first_column=jnp.asarray(kernel))
            with pytest.raises(StructureError, match="not itself stationary"):
                compress({"x": design}, data, precision, {"x": (3,)})


class TestTheShapeChecksAreReal:
    def test_a_design_block_that_does_not_match_the_data_is_refused(self):
        with jax.enable_x64(True):
            _, data, _ = _pieces(n=7)
            precision = DiagonalPrecision(sigma=jnp.full(7, 0.5))
            with pytest.raises(StructureError, match="to match the data"):
                compress({"x": jnp.zeros((5, 3))}, data, precision, {"x": (3,)})

    def test_a_block_whose_columns_do_not_match_its_shape_is_refused(self):
        with jax.enable_x64(True):
            design, data, _ = _pieces(n=7)
            precision = DiagonalPrecision(sigma=jnp.full(7, 0.5))
            with pytest.raises(StructureError, match="ravels to"):
                compress({"x": design}, data, precision, {"x": (4,)})

    def test_an_empty_design_is_refused(self):
        with jax.enable_x64(True):
            _, data, _ = _pieces(n=7)
            precision = DiagonalPrecision(sigma=jnp.full(7, 0.5))
            with pytest.raises(StructureError, match="no design blocks"):
                compress({}, data, precision, {})
