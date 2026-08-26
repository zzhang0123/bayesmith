"""Distributions this package declares because numpyro has none.

One so far. ``ComplexNormal`` exists because a complex latent has to be
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
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from numpyro.distributions import Distribution, constraints

__all__ = ["ComplexNormal"]


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

    ``arg_constraints`` is deliberately EMPTY. numpyro would otherwise check
    ``loc`` against a real-valued constraint and refuse the one argument this
    class exists to accept; the checks that matter here -- a positive finite
    scale, and a ``log_prob`` that agrees with the ``(loc, scale)`` read off
    the instance -- are made by
    :func:`~bayesmith.exact.gaussian.check_gaussian` against concrete values,
    which is a stronger test than a constraint on construction.

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
