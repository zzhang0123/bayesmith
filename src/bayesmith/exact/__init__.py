"""Structure-dispatched exact solves for linear-Gaussian blocks.

Import order matters only in that ``linear_operator`` -- the checked entry
point -- lives in :mod:`bayesmith.exact.linearity`, while the unchecked
primitive it wraps lives in :mod:`bayesmith.exact.block`. Re-exported here so
the checked name is the one a bare ``from bayesmith.exact import ...`` finds.
"""

from bayesmith.exact.block import LinearBlock, unchecked_operator
from bayesmith.exact.fisher import (
    FlatMatrix,
    dense_operator,
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)
from bayesmith.exact.gaussian import (
    check_gaussian,
    gaussian_parts,
    noise_std_at,
    precision_at,
    precision_parts,
)
from bayesmith.exact.gls import (
    GLSResult,
    check_prediction_dependence,
    iterative_gls,
    sigma_from_graph,
)
from bayesmith.exact.linearity import (
    DEFAULT_AT_POINTS,
    DEFAULT_SCALES,
    RELATIVE_FLOOR_FACTOR,
    WEIGHTED_FLOOR_FACTOR,
    WEIGHTED_RTOL,
    Unresolved,
    check_linearity,
    linear_operator,
)
from bayesmith.exact.loglinear import (
    FIRST_ORDER_MAX_FRACTIONAL,
    LOG_DEFAULT_SCALES,
    LogSpace,
    check_log_linearity,
    log_linear_operator,
    log_space,
    multiplicative_log_data,
)
from bayesmith.exact.solve import condition_bound, gcr_sample, wiener_solve

__all__ = [
    # The affinity check's own vocabulary. Advertised because the sibling
    # package adopted these criteria (D16, ruled 2026-08-27) and a criterion
    # whose CONSTANTS have a second copy is the defect that adoption is
    # meant to remove -- one statement of each number, both sides reading it.
    "DEFAULT_SCALES",
    "DEFAULT_AT_POINTS",
    "RELATIVE_FLOOR_FACTOR",
    "WEIGHTED_FLOOR_FACTOR",
    "WEIGHTED_RTOL",
    "Unresolved",
    "LinearBlock",
    "unchecked_operator",
    "linear_operator",
    "check_linearity",
    "gaussian_parts",
    "check_gaussian",
    "noise_std_at",
    "precision_parts",
    "precision_at",
    "wiener_solve",
    "gcr_sample",
    "condition_bound",
    "iterative_gls",
    "GLSResult",
    "sigma_from_graph",
    "check_prediction_dependence",
    "FlatMatrix",
    "dense_operator",
    "fisher_information",
    "parameter_covariance",
    "propagate_covariance",
    "push_forward",
    "FIRST_ORDER_MAX_FRACTIONAL",
    "LOG_DEFAULT_SCALES",
    "LogSpace",
    "check_log_linearity",
    "log_linear_operator",
    "log_space",
    "multiplicative_log_data",
]
