"""Deprecated alias for :mod:`bayesmith.marginal`. Retires at 1.0.

This subpackage was called ``evidence`` through 0.4.0. The name claimed
something the package does not do: the Bayesian evidence is
``p(d) = INT p(theta) PROD_i L_i(theta) d theta``, a number obtained by
integrating the parameters OUT, and nothing here computes or consumes one --
there is no Bayes factor and no model comparison anywhere in the package. What
the modules under here actually build is each dataset's MARGINAL LIKELIHOOD
``L_i(theta)``, with its own nuisances integrated away: a function of the
parameters, not a number. Hence ``marginal``.

The rename is not cosmetic bookkeeping. ``evidence`` is a word this package
uses correctly in fifteen other places, in its ordinary English sense of
"the grounds for a verdict" -- including in ``errors.py``, which is the only
eagerly imported module and so the first bayesmith prose a user reads. A
subpackage holding the term made every one of those ambiguous, and no
disclaimer written inside the subpackage can reach the prose outside it.

**Everything still works through this path**, including deep imports::

    from bayesmith.evidence import SqrtInfo          # fine, warns
    from bayesmith.evidence.compress import compress # fine, warns
    import bayesmith.evidence.chain                  # fine, warns

Deep imports are aliased through ``sys.modules`` rather than left to a
module-level ``__getattr__``, because a ``__getattr__`` shim does NOT support
them -- ``from pkg.old.kernel import helper`` raises ``ModuleNotFoundError``
against one. That distinction is load-bearing here: of the seventeen names
0.4.0 published from this subpackage, the intersection with bayesmith's
top-level ``__all__`` was EMPTY, so every one of them was reachable only by
the deep path this file has to keep working.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings

from bayesmith import marginal as _marginal

warnings.warn(
    "bayesmith.evidence is deprecated and will be removed in 1.0; it is now "
    "bayesmith.marginal. The old name claimed the Bayesian evidence p(d), "
    "which this package does not compute -- what these modules build is each "
    "dataset's marginal likelihood L_i(theta).",
    DeprecationWarning,
    stacklevel=2,
)

#: The submodules the old path must keep resolving. Derived from the new
#: package rather than listed, so a module added to `marginal/` is reachable
#: through the deprecated path too, and one deleted stops being aliased -- a
#: hand-written list here would go stale in exactly the direction that leaves
#: a dangling alias pointing at nothing.
_ALIASED: tuple[str, ...] = tuple(
    info.name
    for info in pkgutil.iter_modules(_marginal.__path__)
    if not info.name.startswith("_")
)

for _name in _ALIASED:
    sys.modules[f"bayesmith.evidence.{_name}"] = importlib.import_module(
        f"bayesmith.marginal.{_name}"
    )

__all__ = list(_marginal.__all__)


def __getattr__(name: str):
    """Forward every public name to :mod:`bayesmith.marginal`."""
    try:
        return getattr(_marginal, name)
    except AttributeError as error:
        raise AttributeError(
            f"module 'bayesmith.evidence' has no attribute {name!r}. This is a "
            "deprecated alias for 'bayesmith.marginal'; the attribute is not "
            "there either."
        ) from error


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_ALIASED))
