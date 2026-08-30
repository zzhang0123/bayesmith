"""Public API for the premise-checked log-determinant ladder.

Direct eager methods live in :mod:`bayesmith.marginal._logdet_eager`, premise
checking/dispatch in :mod:`bayesmith.marginal._logdet_ladder`, certificates
and plans in :mod:`bayesmith.marginal._logdet_plan`, and pure JAX kernels in
:mod:`bayesmith.marginal._logdet_runtime`.
"""

from bayesmith.marginal._logdet_eager import *
from bayesmith.marginal._logdet_eager import __all__ as _EAGER_EXPORTED
from bayesmith.marginal._logdet_ladder import *
from bayesmith.marginal._logdet_ladder import __all__ as _LADDER_EXPORTED
from bayesmith.marginal._logdet_plan import *
from bayesmith.marginal._logdet_plan import __all__ as _PLAN_EXPORTED

__all__ = _EAGER_EXPORTED + _LADDER_EXPORTED + _PLAN_EXPORTED  # noqa: PLE0605
