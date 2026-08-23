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
)
from bayesmith.exact.gaussian import check_gaussian, gaussian_parts, noise_std_at
from bayesmith.exact.gls import (
    GLSResult,
    check_prediction_dependence,
    iterative_gls,
    sigma_from_graph,
)
from bayesmith.exact.linearity import check_linearity, linear_operator
from bayesmith.exact.solve import condition_bound, gcr_sample, wiener_solve

__all__ = [
    "LinearBlock",
    "unchecked_operator",
    "linear_operator",
    "check_linearity",
    "gaussian_parts",
    "check_gaussian",
    "noise_std_at",
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
]
