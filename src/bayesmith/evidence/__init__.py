"""The streaming evidence layer -- B11.

Nothing here reaches the graph yet. :mod:`bayesmith.evidence.sqrtinfo` is the
numerical kernel the layer is built on, and the migration spec's B11 row says
it is to be **preserved exactly** rather than reinvented; see
``docs/evidence-layer-readiness.md`` for what the rest of the layer still
needs and what was measured about it.
"""

from bayesmith.evidence.campaign import compress_campaign, epoch_terms
from bayesmith.evidence.compress import (
    compress,
    compress_epoch,
    epoch_joint,
    nuisance_prior,
    observed_mask,
)
from bayesmith.evidence.factorize import (
    Factorization,
    epoch_leakage,
    factorize,
)
from bayesmith.evidence.sqrtinfo import (
    SqrtInfo,
    marginalise,
    marginalise_arrays,
)

__all__ = [
    "SqrtInfo",
    "marginalise",
    "marginalise_arrays",
    "compress",
    "compress_epoch",
    "epoch_joint",
    "nuisance_prior",
    "observed_mask",
    "Factorization",
    "factorize",
    "epoch_leakage",
    "compress_campaign",
    "epoch_terms",
]
