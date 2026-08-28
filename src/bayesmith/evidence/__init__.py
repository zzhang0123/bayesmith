"""The streaming evidence layer -- B11.

Nothing here reaches the graph yet. :mod:`bayesmith.evidence.sqrtinfo` is the
numerical kernel the layer is built on, and the migration spec's B11 row says
it is to be **preserved exactly** rather than reinvented; see
``docs/evidence-layer-readiness.md`` for what the rest of the layer still
needs and what was measured about it.
"""

from bayesmith.evidence.campaign import (
    compress_campaign,
    epoch_observation,
    epoch_terms,
)
from bayesmith.evidence.compress import (
    ResidualSummary,
    compress,
    compress_epoch,
    epoch_joint,
    nuisance_prior,
    observed_mask,
    residual_summary,
)
from bayesmith.evidence.diagnostics import (
    coherent_mode,
    epoch_chi_square,
    epoch_residuals,
    held_out_z,
    refuse_mixed_templates,
    refuse_undeclared_coherent_error,
    shrinkage_power,
    shrinkage_report,
    systematic_floor,
    template_modes,
    tightest_direction,
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
    "epoch_observation",
    "coherent_mode",
    "epoch_chi_square",
    "refuse_undeclared_coherent_error",
    # G6: the evidence consumption surface
    "ResidualSummary",
    "residual_summary",
    "epoch_residuals",
    "refuse_mixed_templates",
    "template_modes",
    "held_out_z",
    "shrinkage_power",
    "shrinkage_report",
    "systematic_floor",
    "tightest_direction",
]
