"""Structural dispatch: deriving what to run from the graph computation.

:mod:`bayesmith.dispatch.classify` answers one question -- which latents an
exact linear-Gaussian method applies to, how groups form, and which method the
group needs. Affinity is discovered structurally and checked numerically;
support and prediction-dependent covariance then constrain the method. The
classifier produces no samples of its own.
"""

from bayesmith.dispatch.classify import (
    SIGMA_RTOL,
    Classification,
    block_at,
    partition,
    prior_environment,
)
from bayesmith.dispatch.factor import (
    FACTOR_METHODS,
    FactorPlan,
    factor_partition,
    first_fit,
    sample_factors,
)

__all__ = [
    "partition",
    "Classification",
    "block_at",
    "prior_environment",
    "SIGMA_RTOL",
    "FACTOR_METHODS",
    "FactorPlan",
    "factor_partition",
    "first_fit",
    "sample_factors",
]
