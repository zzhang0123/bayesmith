"""bayesmith: a graph of operators is a Bayesian model.

Deterministic operators propagate dependence; probabilistic ones contribute a
conditional density. The graph's structure is what selects the inference
method -- exact where a subgraph permits one, NUTS where it does not.
"""

from __future__ import annotations

import importlib
from typing import Any

from bayesmith.errors import (
    BayesmithError,
    ConvergenceError,
    GraphError,
    NotGaussian,
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
    "Posterior",
    "Estimate",
    "to_numpyro",
    "nuts",
    # exact
    "linear_operator",
    "check_linearity",
    "wiener_solve",
    "gcr_sample",
    "condition_bound",
    "iterative_gls",
    "sigma_from_graph",
    "noise_std_at",
    "precision_at",
    "fisher_information",
    "parameter_covariance",
    # errors
    "BayesmithError",
    "GraphError",
    "TraceError",
    "StructureError",
    "ConvergenceError",
    "NotGaussian",
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
    "NodeRef": ("bayesmith.graph.trace", "NodeRef"),
    "PlateRef": ("bayesmith.graph.trace", "PlateRef"),
    "Graph": ("bayesmith.graph.graph", "Graph"),
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
    "linear_operator": ("bayesmith.exact.linearity", "linear_operator"),
    "check_linearity": ("bayesmith.exact.linearity", "check_linearity"),
    "wiener_solve": ("bayesmith.exact.solve", "wiener_solve"),
    "gcr_sample": ("bayesmith.exact.solve", "gcr_sample"),
    "condition_bound": ("bayesmith.exact.solve", "condition_bound"),
    "iterative_gls": ("bayesmith.exact.gls", "iterative_gls"),
    "sigma_from_graph": ("bayesmith.exact.gls", "sigma_from_graph"),
    "noise_std_at": ("bayesmith.exact.gaussian", "noise_std_at"),
    "precision_at": ("bayesmith.exact.gaussian", "precision_at"),
    "fisher_information": ("bayesmith.exact.fisher", "fisher_information"),
    "parameter_covariance": ("bayesmith.exact.fisher", "parameter_covariance"),
}

# Subpackages reachable as `bayesmith.<name>` after a bare `import bayesmith`,
# without eagerly importing any of them -- `bridge` in particular is what
# pulls in numpyro, and `exact` reaches numpyro through `bridge` too (see
# `gaussian.py`'s use of `numpyro.distributions`). `errors` is listed too for
# __dir__'s sake even though the eager import above already binds it as a
# real attribute, so __getattr__ is never actually consulted for it.
_LAZY_SUBMODULES = ("graph", "bridge", "exact", "dispatch", "errors")


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
