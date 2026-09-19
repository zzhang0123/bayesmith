"""Optional Fourier constructions. No graph or inference code imports these."""

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution, constraints

from .operators import LowRankCumulants, _real_array


def _reverse_modes(array, event_ndim):
    for axis in range(array.ndim - event_ndim, array.ndim):
        array = jnp.take(
            array, (-jnp.arange(array.shape[axis])) % array.shape[axis], axis=axis
        )
    return array


class PeriodicNormal(Distribution):
    """Real Gaussian field with a periodic covariance diagonalized by FFT.

    ``power`` contains covariance eigenvalues on the FULL FFT grid (orthonormal
    convention), not an rFFT half-grid or a continuum power spectrum. It must
    be real, positive, finite and exactly even under simultaneous mode reversal.
    ``loc`` and power broadcast; the last ``event_ndim`` axes form one field.
    The default makes all axes of loc one event. No global settings change.
    """

    arg_constraints: ClassVar[dict] = {
        "loc": constraints.real,
        "power": constraints.positive,
    }
    pytree_data_fields = ("loc", "power")
    pytree_aux_fields = ("field_ndim",)

    def __init__(self, loc, power, *, event_ndim=None, validate_args=None):
        self.loc = _real_array(loc)
        self.power = _real_array(power)
        self.field_ndim = self.loc.ndim if event_ndim is None else event_ndim
        if (
            isinstance(self.field_ndim, bool)
            or not isinstance(self.field_ndim, int)
            or self.field_ndim < 1
        ):
            raise ValueError("event_ndim must be a positive static integer")
        shape = jnp.broadcast_shapes(self.loc.shape, self.power.shape)
        if len(shape) < self.field_ndim or any(s == 0 for s in shape):
            raise ValueError("loc and power need nonempty field axes")
        self.loc = jnp.broadcast_to(self.loc, shape)
        self.power = jnp.broadcast_to(self.power, shape)
        self.power = eqx.error_if(
            self.power,
            jnp.any(~jnp.isfinite(self.power) | (self.power <= 0)),
            "PeriodicNormal power must be positive and finite",
        )
        self.power = eqx.error_if(
            self.power,
            jnp.any(self.power != _reverse_modes(self.power, self.field_ndim)),
            "PeriodicNormal power must be even under mode reversal",
        )
        super().__init__(
            batch_shape=shape[: -self.field_ndim],
            event_shape=shape[-self.field_ndim :],
            validate_args=validate_args,
        )

    @property
    def support(self):
        return constraints.independent(constraints.real, self.field_ndim)

    def log_prob(self, value):
        axes = tuple(range(-self.field_ndim, 0))
        residual = _real_array(value) - self.loc
        spectrum = jnp.fft.fftn(residual, axes=axes, norm="ortho")
        # abs(z)**2 has the right value but JAX's abs JVP loses the Hessian
        # at z=0. Hermite contractions need that Hessian even at the mean.
        quadratic = jnp.sum(
            (spectrum.real**2 + spectrum.imag**2) / self.power, axis=axes
        )
        logdet = jnp.sum(jnp.log(self.power), axis=axes)
        return -0.5 * (
            quadratic + logdet + math.prod(self.event_shape) * math.log(2 * math.pi)
        )

    def sample(self, key, sample_shape=()):
        axes = tuple(range(-self.field_ndim, 0))
        noise = jax.random.normal(
            key, tuple(sample_shape) + self.loc.shape, dtype=self.loc.dtype
        )
        spectrum = jnp.fft.fftn(noise, axes=axes, norm="ortho") * jnp.sqrt(self.power)
        return self.loc + jnp.fft.ifftn(spectrum, axes=axes, norm="ortho").real

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        axes = tuple(range(-self.field_ndim, 0))
        return jnp.broadcast_to(
            jnp.mean(self.power, axis=axes, keepdims=True), self.loc.shape
        )


def low_rank_from_fourier(coefficients, cumulants, *, event_shape):
    """Build real-coordinate CP cumulants from full-grid Fourier directions.

    The trailing axes have ``event_shape``; the preceding axis is CP rank.
    Inverts an orthonormal full FFT, keeping the coefficients at each cumulant
    order unchanged. This transforms every tensor index through its direction
    factors. It is NOT a change of the observed random variable or a general
    arbitrary-polyspectrum factorization algorithm.

    Reality is checked with a 64-epsilon, amplitude-relative numerical allowance
    for FFT roundoff (a practical allowance, not a certified FFT error bound).
    Only that roundoff imaginary residue is discarded; complex fields and
    unpaired complex modes are rejected, including under JIT.
    """
    event_shape = tuple(event_shape)
    spectrum = jnp.asarray(coefficients)
    n = len(event_shape)
    if not n or spectrum.ndim < n + 1 or spectrum.shape[-n:] != event_shape:
        raise ValueError("Fourier directions must have rank and event_shape axes")
    axes = tuple(range(-n, 0))
    directions = jnp.fft.ifftn(spectrum, axes=axes, norm="ortho")
    scale = jnp.max(jnp.abs(directions), axis=axes, keepdims=True)
    tolerance = 64 * jnp.finfo(directions.real.dtype).eps * scale
    directions = eqx.error_if(
        directions,
        jnp.any(~jnp.isfinite(directions))
        | jnp.any(jnp.abs(directions.imag) > tolerance),
        "Fourier directions must describe real fields (conjugate-symmetric modes)",
    )
    return LowRankCumulants(directions.real, cumulants, event_shape=event_shape)
