"""Distributions this package declares because numpyro has none.

``ComplexNormal`` exists because a complex latent has to be
DECLARABLE before any of the machinery that solves one is reachable: a graph
states a latent's prior through its ``dist_fn`` and nothing else, and the
block's dtype is read off that prior's ``loc``. Measured before this module
existed: every numpyro distribution samples real (``Normal``, ``LogNormal``,
``Cauchy``, ``StudentT`` all return float32, and nothing in
``dir(numpyro.distributions)`` mentions complex), so a complex latent could be
solved only in a hand-built block -- the solver accepted one that the
declaration layer could not express.

**The convention, stated once here because three places depend on it agreeing.**
``ComplexNormal(loc, scale)`` means the real and imaginary parts are
independent, each ``Normal(part of loc, scale)``. So each half carries
``scale**2`` and the latent's TOTAL prior variance is ``2 * scale**2``. That is
what :func:`~bayesmith.exact.block.variance_parts` duplicates across the two
halves, and it is the convention the sibling package's ``gcr_sample``
documents for a complex latent. Halving instead -- ``scale**2`` split between
the parts -- is the other defensible reading, and choosing it silently would
report a factor of sqrt(2) as a physics result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution, Normal, constraints

from bayesmith.cumulants._series import edgeworth_powers, truncation_order
from bayesmith.cumulants.field import FieldEdgeworth

__all__ = ["ComplexNormal", "EdgeworthExpansion", "FieldEdgeworth"]


class EdgeworthExpansion(Distribution):
    """A scalar conditional likelihood with a manually truncated Edgeworth series.

    ``order=N`` consumes cumulants through kappa_N and retains products of
    standardized cumulants whose total Edgeworth weight is at most N-2;
    kappa_r has weight r-2. Thus order 4 includes kappa_3 squared times H_6.
    This presumes the usual weak-non-Gaussian hierarchy; selecting N does not
    certify the approximation's accuracy. See ``docs/cumulant-expansion.md``.

    Args:
        loc: Conditional mean, kappa_1.
        scale: Positive conditional standard deviation, sqrt(kappa_2).
        cumulants: Sequence (kappa_3, kappa_4, ...), NOT standardized cumulants
            or central moments. Each entry may be a batch array. All entries
            through ``order`` must be supplied; later entries are ignored.
        order: Static integer >= 2, selected by the user. 2 is Gaussian.
        validate_args: NumPyro's parameter validation; this does not check
            global positivity or select a truncation order.

    This object implements NumPyro's density interface for ``observe``. A
    truncated expansion can be negative: ``correction`` exposes its signed
    Gaussian multiplier, and ``log_prob`` returns NaN at negative values,
    without clipping, absolute values, renormalization or automatic fallback.
    A zero multiplier gives -inf. Global validity and automatic order
    selection are TODO. Order > 2 sampling is deliberately unimplemented:
    neither a latent prior nor posterior predictive sampling is supported.
    """

    arg_constraints: ClassVar[dict[str, Any]] = {
        "loc": constraints.real,
        "scale": constraints.positive,
    }
    support = constraints.real
    pytree_data_fields = ("loc", "scale", "cumulants")
    pytree_aux_fields = ("order",)

    def __init__(
        self,
        loc: Any,
        scale: Any,
        *,
        cumulants: Sequence[Any] = (),
        order: int = 2,
        validate_args: bool | None = None,
    ):
        order = truncation_order(order)
        if len(cumulants) < order - 2:
            raise ValueError(
                f"order={order} needs {order - 2} cumulants (kappa_3 through "
                f"kappa_{order}); got {len(cumulants)}"
            )
        self.order = order
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)
        self.cumulants = tuple(jnp.asarray(k) for k in cumulants[: order - 2])
        batch_shape = jnp.broadcast_shapes(
            self.loc.shape, self.scale.shape, *(k.shape for k in self.cumulants)
        )
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _correction_delta(self, value: Any) -> jax.Array:
        z = (jnp.asarray(value) - self.loc) / self.scale
        weight = self.order - 2
        if not weight:
            return jnp.zeros_like(z)
        # Probabilists' Hermite polynomials: H_(n+1) = z H_n - n H_(n-1).
        hermites = [jnp.ones_like(z), z]
        for n in range(1, 3 * weight):
            hermites.append(z * hermites[-1] - n * hermites[-2])
        coefficients = tuple(
            (k / self.scale**r) / math.factorial(r)
            for r, k in enumerate(self.cumulants, start=3)
        )
        delta = jnp.zeros_like(z)
        for powers in edgeworth_powers(weight):
            degree = 0
            coefficient = 1.0
            for r, (a, power) in enumerate(zip(coefficients, powers), start=3):
                if power:
                    coefficient = coefficient * a**power / math.factorial(power)
                    degree += r * power
            delta = delta + coefficient * hermites[degree]
        return delta

    def correction(self, value: Any) -> jax.Array:
        """Signed multiplier of Normal(loc, scale); may be negative."""
        return 1.0 + self._correction_delta(value)

    def log_prob(self, value: Any) -> jax.Array:
        """Approximate log likelihood; log1p preserves small corrections."""
        return Normal(self.loc, self.scale).log_prob(value) + jnp.log1p(
            self._correction_delta(value)
        )

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        if self.order != 2:
            raise NotImplementedError(
                "EdgeworthExpansion with order > 2 is a likelihood approximation; "
                "latent and predictive sampling are not implemented."
            )
        return Normal(self.loc, self.scale).sample(key, sample_shape)


class ComplexNormal(Distribution):
    """Independent Gaussian real and imaginary parts, of equal width.

    Args:
        loc: the mean, complex (a real ``loc`` is promoted, and means an
            imaginary part centred on zero).
        scale: the width of EACH part, real and strictly positive.

    Not a ``MultivariateNormal`` in disguise and not a circularly-symmetric
    complex Gaussian with a pseudo-covariance: the two parts are independent
    with the same width, which is the isotropic case and the one a sky ``alm``
    prior states. A model needing correlated parts declares two real latents
    and says so.

    ``arg_constraints`` is deliberately EMPTY, and re-adding it is wrong
    whichever entry is chosen. numpyro validates by walking this mapping and
    calling each constraint on the named attribute, so an entry that rejects a
    complex ``loc`` refuses the one argument this class exists to accept --
    measured with ``{"loc": constraints.positive}`` standing in for one:
    ``ValueError: ComplexNormal distribution got invalid loc parameter``.

    The obvious entry does not reject it and is inert. On numpyro 0.21
    ``_Real.__call__`` is ``(x == x) & (x != inf) & (x != -inf)``, which
    ``1+2j`` satisfies, so ``{"loc": constraints.real}`` checks nothing while
    reading as though ``loc`` were checked. Measured: it passes the whole fast
    layer, 3637 tests, exit 0.

    The checks that matter here -- a positive finite scale, and a ``log_prob``
    that agrees with the ``(loc, scale)`` read off the instance -- are made by
    :func:`~bayesmith.exact.gaussian.check_gaussian` against concrete values,
    which is a stronger test than a constraint on construction.
    ``TestTheDistribution::test_a_complex_loc_survives_construction_under_validation``
    in ``tests/exact/test_complex.py`` holds the capability rather than the
    spelling, so the fatal re-add fails there instead of arriving as a
    tightening.

    **NUTS reaches this through a reparameterisation, not through this class.**
    ``to_numpyro`` emits a complex latent as two real sites plus a
    deterministic that recombines them, because HMC's transforms are defined
    on real unconstrained space. That keeps the package's standing rule true
    -- every graph an exact method accepts is also runnable through NUTS,
    which is what the exact paths are checked against -- rather than carving
    an exception into it for the one node type that would need one.
    """

    arg_constraints: ClassVar[dict[str, Any]] = {}
    support = constraints.real
    reparametrized_params: ClassVar[list[str]] = ["loc", "scale"]

    def __init__(self, loc: Any, scale: Any, *, validate_args: bool | None = None):
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)
        batch_shape = jnp.broadcast_shapes(jnp.shape(self.loc), jnp.shape(self.scale))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        shape = tuple(sample_shape) + self.batch_shape
        real_key, imag_key = jax.random.split(key)
        part = jnp.result_type(jnp.finfo(self.loc.dtype).dtype, self.scale.dtype)
        real = jax.random.normal(real_key, shape, dtype=part)
        imag = jax.random.normal(imag_key, shape, dtype=part)
        return self.loc + self.scale * (real + 1j * imag)

    def log_prob(self, value: Any) -> jax.Array:
        """Two real Gaussians, summed. Elementwise, like ``Normal``'s.

        Written from the parts rather than from ``abs(value - loc)**2`` so the
        expression names the two independent halves it is a density over --
        the magnitude form is the same number and hides which convention is
        in force.
        """
        value = jnp.asarray(value)
        real = (jnp.real(value) - jnp.real(self.loc)) / self.scale
        imag = (jnp.imag(value) - jnp.imag(self.loc)) / self.scale
        return (
            -0.5 * (real**2 + imag**2)
            - 2.0 * jnp.log(self.scale)
            - math.log(2.0 * math.pi)
        )

    @property
    def mean(self) -> jax.Array:
        return jnp.broadcast_to(self.loc, self.batch_shape)

    @property
    def variance(self) -> jax.Array:
        """``2 * scale**2`` -- the TOTAL, both parts.

        Not ``scale**2``. A reader who takes this for one part's variance gets
        the sqrt(2) this module's docstring is about, so it says which.
        """
        return jnp.broadcast_to(2.0 * self.scale**2, self.batch_shape)
