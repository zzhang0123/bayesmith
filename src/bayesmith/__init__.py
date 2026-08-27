"""bayesmith: a graph of operators is a Bayesian model.

Deterministic operators propagate dependence; probabilistic ones contribute a
conditional density. The graph's structure is what selects the inference
method -- exact where a subgraph permits one, NUTS where it does not.
"""

from __future__ import annotations

import importlib
import importlib.metadata as _metadata
from typing import Any

from bayesmith.errors import (
    NOT_GAUSSIAN_REASONS,
    NOT_LOG_LINEAR_REASONS,
    AffinityRefused,
    BayesmithError,
    ConvergenceError,
    GraphError,
    NotGaussian,
    NotLogLinear,
    StructureError,
    TraceError,
)

__all__ = [
    # tracing
    "trace",
    "const",
    "det",
    "sample",
    "observe",
    "plate",
    "joint_prior",
    "NodeRef",
    "PlateRef",
    # graph
    "Graph",
    "Plate",
    "Node",
    "Const",
    "Deterministic",
    "Probabilistic",
    # evaluation
    "evaluate",
    "log_joint",
    # inference
    "compile",
    "factor_partition",
    "sample_factors",
    "declared_partition",
    "estimate_factors",
    "SweepReport",
    "SweepEstimate",
    "Posterior",
    "Estimate",
    "to_numpyro",
    "nuts",
    "predict",
    # exact
    "linear_operator",
    "check_linearity",
    "check_log_linearity",
    "log_linear_operator",
    "log_space",
    "multiplicative_log_data",
    "wiener_solve",
    "gcr_sample",
    "condition_bound",
    "condition_estimate",
    # reduced basis
    "orthonormal_transform",
    "orthonormalise",
    "numerical_rank",
    "select_svd",
    "select_greedy",
    # chain
    "LinearGaussianTransition",
    "HyperTransition",
    "ornstein_uhlenbeck",
    "chain_marginal",
    "chain_log_likelihood",
    "smooth",
    "iterative_gls",
    "sigma_from_graph",
    "noise_std_at",
    "precision_at",
    "fisher_information",
    "parameter_covariance",
    # diagnose
    "identifiability",
    "prior_sensitivity",
    "JeffreysPrior",
    "init_to_declared",
    "propagate_covariance",
    "push_forward",
    # optimise
    "fit",
    "minimize",
    "Fit",
    "check_loss_sense",
    "MINIMIZE",
    "MAXIMIZE",
    # amortize
    "NeuralPosterior",
    "train_posterior",
    "TrainingHistory",
    "MIN_SCALE",
    # errors
    "BayesmithError",
    "GraphError",
    "TraceError",
    "StructureError",
    "AffinityRefused",
    "ConvergenceError",
    "NotGaussian",
    "NotLogLinear",
    # the refusal vocabularies, exported because a consumer that branches on
    # `reason` should compare against the package's own spelling rather than
    # keep a second copy of the strings
    "NOT_GAUSSIAN_REASONS",
    "NOT_LOG_LINEAR_REASONS",
    # declarations numpyro has none of
    "ComplexNormal",
]

# Every public name above except the six error classes is resolved lazily,
# on first attribute access, rather than imported here at module scope.
# Importing eagerly would make `import bayesmith` load numpyro (hence jax) as
# a side effect -- exactly the regression this module previously had: Python
# always runs a package's __init__.py before any of its submodules, so even
# `import bayesmith.errors` was dragging in the whole bridge, which broke the
# stdlib-only contract errors.py documents for itself and which
# test_errors_module_imports_no_heavy_dependency enforces. Only errors.py is
# cheap and stdlib-only, so it alone is still imported eagerly above.
#
# name -> (owning submodule, attribute name within it)
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "trace": ("bayesmith.graph.trace", "trace"),
    "const": ("bayesmith.graph.trace", "const"),
    "det": ("bayesmith.graph.trace", "det"),
    "sample": ("bayesmith.graph.trace", "sample"),
    "observe": ("bayesmith.graph.trace", "observe"),
    "plate": ("bayesmith.graph.trace", "plate"),
    "joint_prior": ("bayesmith.graph.trace", "joint_prior"),
    "NodeRef": ("bayesmith.graph.trace", "NodeRef"),
    "PlateRef": ("bayesmith.graph.trace", "PlateRef"),
    "Graph": ("bayesmith.graph.graph", "Graph"),
    "ComplexNormal": ("bayesmith.distributions", "ComplexNormal"),
    "Plate": ("bayesmith.graph.graph", "Plate"),
    "Node": ("bayesmith.graph.nodes", "Node"),
    "Const": ("bayesmith.graph.nodes", "Const"),
    "Deterministic": ("bayesmith.graph.nodes", "Deterministic"),
    "Probabilistic": ("bayesmith.graph.nodes", "Probabilistic"),
    "evaluate": ("bayesmith.graph.evaluate", "evaluate"),
    "log_joint": ("bayesmith.graph.evaluate", "log_joint"),
    "compile": ("bayesmith.dispatch.plan", "compile"),
    "Posterior": ("bayesmith.dispatch.execute", "Posterior"),
    "Estimate": ("bayesmith.dispatch.execute", "Estimate"),
    "to_numpyro": ("bayesmith.bridge.numpyro_bridge", "to_numpyro"),
    "nuts": ("bayesmith.bridge.numpyro_bridge", "nuts"),
    "predict": ("bayesmith.bridge.numpyro_bridge", "predict"),
    "linear_operator": ("bayesmith.exact.linearity", "linear_operator"),
    "check_linearity": ("bayesmith.exact.linearity", "check_linearity"),
    "check_log_linearity": ("bayesmith.exact.loglinear", "check_log_linearity"),
    "log_linear_operator": ("bayesmith.exact.loglinear", "log_linear_operator"),
    "log_space": ("bayesmith.exact.loglinear", "log_space"),
    "multiplicative_log_data": ("bayesmith.exact.loglinear", "multiplicative_log_data"),
    "factor_partition": ("bayesmith.dispatch.factor", "factor_partition"),
    "sample_factors": ("bayesmith.dispatch.factor", "sample_factors"),
    "declared_partition": ("bayesmith.dispatch.factor", "declared_partition"),
    "estimate_factors": ("bayesmith.dispatch.factor", "estimate_factors"),
    "SweepReport": ("bayesmith.dispatch.factor", "SweepReport"),
    "SweepEstimate": ("bayesmith.dispatch.factor", "SweepEstimate"),
    "wiener_solve": ("bayesmith.exact.solve", "wiener_solve"),
    "gcr_sample": ("bayesmith.exact.solve", "gcr_sample"),
    "condition_bound": ("bayesmith.exact.solve", "condition_bound"),
    "condition_estimate": ("bayesmith.exact.solve", "condition_estimate"),
    "orthonormal_transform": (
        "bayesmith.exact.reduced_basis", "orthonormal_transform",
    ),
    "orthonormalise": ("bayesmith.exact.reduced_basis", "orthonormalise"),
    "numerical_rank": ("bayesmith.exact.reduced_basis", "numerical_rank"),
    "select_svd": ("bayesmith.exact.reduced_basis", "select_svd"),
    "select_greedy": ("bayesmith.exact.reduced_basis", "select_greedy"),
    "LinearGaussianTransition": (
        "bayesmith.evidence.chain", "LinearGaussianTransition",
    ),
    "HyperTransition": ("bayesmith.evidence.chain", "HyperTransition"),
    "ornstein_uhlenbeck": ("bayesmith.evidence.chain", "ornstein_uhlenbeck"),
    "chain_marginal": ("bayesmith.evidence.chain", "chain_marginal"),
    "chain_log_likelihood": ("bayesmith.evidence.chain", "chain_log_likelihood"),
    "smooth": ("bayesmith.evidence.chain", "smooth"),
    "iterative_gls": ("bayesmith.exact.gls", "iterative_gls"),
    "sigma_from_graph": ("bayesmith.exact.gls", "sigma_from_graph"),
    "noise_std_at": ("bayesmith.exact.gaussian", "noise_std_at"),
    "precision_at": ("bayesmith.exact.gaussian", "precision_at"),
    "fisher_information": ("bayesmith.exact.fisher", "fisher_information"),
    "parameter_covariance": ("bayesmith.exact.fisher", "parameter_covariance"),
    "identifiability": ("bayesmith.diagnose.identifiability", "identifiability"),
    "prior_sensitivity": ("bayesmith.diagnose.sensitivity", "prior_sensitivity"),
    "JeffreysPrior": ("bayesmith.diagnose.priors", "JeffreysPrior"),
    "init_to_declared": ("bayesmith.bridge.numpyro_bridge", "init_to_declared"),
    "propagate_covariance": ("bayesmith.exact.fisher", "propagate_covariance"),
    "push_forward": ("bayesmith.exact.fisher", "push_forward"),
    "fit": ("bayesmith.optimize", "fit"),
    "minimize": ("bayesmith.optimize", "minimize"),
    "Fit": ("bayesmith.optimize", "Fit"),
    "check_loss_sense": ("bayesmith.optimize", "check_loss_sense"),
    "MINIMIZE": ("bayesmith.optimize", "MINIMIZE"),
    "MAXIMIZE": ("bayesmith.optimize", "MAXIMIZE"),
    "NeuralPosterior": ("bayesmith.amortize", "NeuralPosterior"),
    "train_posterior": ("bayesmith.amortize", "train_posterior"),
    "TrainingHistory": ("bayesmith.amortize", "TrainingHistory"),
    "MIN_SCALE": ("bayesmith.amortize", "MIN_SCALE"),
}

# Subpackages reachable as `bayesmith.<name>` after a bare `import bayesmith`,
# without eagerly importing any of them -- `bridge` in particular is what
# pulls in numpyro, and `exact` reaches numpyro through `bridge` too (see
# `gaussian.py`'s use of `numpyro.distributions`). `errors` is listed too for
# __dir__'s sake even though the eager import above already binds it as a
# real attribute, so __getattr__ is never actually consulted for it.
#
# `evidence` reaches jax through `sqrtinfo.py`'s module-scope import, so it is
# listed here and NOT imported eagerly, for the same reason `exact` is not.
# It was missing from this tuple for the whole of B11: the layer was complete,
# dense-oracled and cross-checked, and `import bayesmith; bayesmith.evidence`
# still raised AttributeError, so only an explicit `import bayesmith.evidence`
# reached it. Both halves are pinned in `tests/test_public_api.py` -- that it
# resolves, and that resolving it is what pulls jax in rather than importing
# this package.
_LAZY_SUBMODULES = (
    "graph",
    "bridge",
    "exact",
    "dispatch",
    "evidence",
    "diagnose",
    "errors",
)


#: The installed distribution's version, READ from the installed metadata
#: rather than written here. Two spellings of one version is the defect this
#: package has spent the most effort repairing, and a release number is the
#: worst candidate for a second copy: it changes on exactly the commit where
#: everyone is busy doing something else.
__version__ = _metadata.version("bayesmith")


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS) | set(_LAZY_SUBMODULES))
