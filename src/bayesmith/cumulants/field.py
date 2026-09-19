"""One field likelihood, whatever coordinates its numerical operators use."""

from typing import ClassVar

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms

from ._series import edgeworth_terms, truncation_order
from .fourier import PeriodicNormal
from .operators import CumulantOperator, _real_array


def _gaussian_reference(base):
    # Exact types prevent a log_prob override from acquiring Gaussian algebra
    # merely by subclassing Normal. Affine/reshape wrappers preserve Gaussianity.
    if type(base) in (
        dist.Normal,
        dist.MultivariateNormal,
        dist.LowRankMultivariateNormal,
        PeriodicNormal,
    ):
        return True
    if type(base) in (dist.Independent, dist.ExpandedDistribution):
        return _gaussian_reference(base.base_dist)
    if type(base) is dist.TransformedDistribution:
        affine = (
            transforms.AffineTransform,
            transforms.ReshapeTransform,
            transforms.PermuteTransform,
            transforms.LowerCholeskyAffine,
        )
        return _gaussian_reference(base.base_dist) and all(
            type(t) in affine for t in base.transforms
        )
    return False


class FieldEdgeworth(dist.Distribution):
    """Manually truncated joint non-Gaussian likelihood for one array event.

    ``base`` is a real Gaussian NumPyro distribution whose event_shape is the
    map/cube shape. ``cumulants`` supplies contractions in those coordinates.
    ``order=N`` retains Edgeworth weight <= N-2, including cross-products.
    This class knows nothing about grids, spectra or how contractions execute.

    As with the scalar EdgeworthExpansion, negative corrections yield NaN and
    are never clipped or renormalized. This is an approximation, not a globally
    certified density. Non-Gaussian simulation/predictive sampling and automatic
    order selection are intentionally not supplied. Order 2 samples ``base``.
    """

    arg_constraints: ClassVar[dict] = {}
    pytree_data_fields = ("base", "cumulants")
    pytree_aux_fields = ("order",)

    def __init__(
        self,
        base: dist.Distribution,
        cumulants: CumulantOperator | None = None,
        *,
        order: int = 2,
        validate_args=None,
    ):
        self.order = truncation_order(order)
        if not _gaussian_reference(base):
            raise TypeError("base must be a supported Gaussian reference distribution")
        if not base.event_shape or not all(base.event_shape):
            raise ValueError("base must declare the whole field as one nonempty event")
        if self.order > 2 and cumulants is None:
            raise ValueError("cumulants are required for order > 2")
        if cumulants is not None:
            if tuple(cumulants.event_shape) != base.event_shape:
                raise ValueError(
                    "cumulants and Gaussian reference event_shape must agree"
                )
            missing = set(range(3, self.order + 1)) - set(cumulants.orders)
            if missing:
                raise ValueError(
                    f"missing cumulant orders {sorted(missing)}; supply explicit zeros"
                )
        self.base = base
        self.cumulants = cumulants
        batch_shape = jnp.broadcast_shapes(
            base.batch_shape,
            () if cumulants is None or self.order == 2 else cumulants.batch_shape,
        )
        super().__init__(
            batch_shape=batch_shape,
            event_shape=base.event_shape,
            validate_args=validate_args,
        )

    @property
    def support(self):
        return constraints.independent(constraints.real, len(self.event_shape))

    def _value(self, value):
        value = _real_array(value)
        if value.shape[-len(self.event_shape) :] != self.event_shape:
            raise ValueError(f"value must end in event_shape={self.event_shape}")
        return value

    def _delta(self, value):
        baseline = self.base.log_prob(value)
        if self.order == 2:
            return jnp.zeros_like(baseline)
        leading = jnp.broadcast_shapes(baseline.shape, self.cumulants.batch_shape)
        value = jnp.broadcast_to(value, leading + self.event_shape)
        contractions = self.cumulants.prepare(value, self.base)
        delta = jnp.zeros(leading, dtype=jnp.result_type(value, float))
        for orders, denominator in edgeworth_terms(self.order):
            term = jnp.asarray(contractions.contract(orders))
            if term.shape != leading:
                raise ValueError(
                    f"cumulant contraction must return batch shape {leading}; got {term.shape}"
                )
            delta = delta + term / denominator
        return delta

    def correction(self, value):
        """Signed Gaussian multiplier, one per field event."""
        return 1 + self._delta(self._value(value))

    def log_prob(self, value):
        value = self._value(value)
        return self.base.log_prob(value) + jnp.log1p(self._delta(value))

    def sample(self, key, sample_shape=()):
        if self.order != 2:
            raise NotImplementedError(
                "FieldEdgeworth is a likelihood approximation; non-Gaussian sampling is not implemented"
            )
        return self.base.sample(key, sample_shape)
