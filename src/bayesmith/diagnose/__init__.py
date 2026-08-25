"""Design-time diagnostics: what the model cannot tell apart, and what the
priors did to it.

Three questions, each answered about the model *at a point* rather than
claimed globally -- which is why this package linearizes locally and never
reads ``linear_in``:

* :func:`identifiability` -- the rank of the joint Jacobian: which
  combinations of latents the data is blind to, across blocks, where every
  per-block guard is structurally silent.
* :func:`prior_sensitivity` -- how far each latent's own prior moved the
  mode, in posterior sigmas, from two deterministic routes that verify each
  other.
* :class:`JeffreysPrior` -- ``sqrt(det I)`` over a named block, evaluated
  from the graph's own noise, with the determinant taken by ``eigvalsh``
  plus a rank floor because ``slogdet`` and ``cholesky`` both return
  plausible finite answers on a singular block.

Ported from ``rheplicant.inference.{identifiability,sensitivity,priors}``
(migration spec §八 step 5); the per-module cross-check records live in
``docs/migration/``. Precision discipline: run everything, graph
construction included, inside ``with jax.enable_x64(True):`` -- these
verdicts live at 1e-17 of the largest singular value, and a float32 result
is refused by name rather than silently reported.
"""

from bayesmith.diagnose.identifiability import (
    DEFAULT_RANK_RTOL,
    IdentifiabilityReport,
    identifiability,
)
from bayesmith.diagnose.priors import JeffreysPrior
from bayesmith.diagnose.sensitivity import (
    CRITERION_SHIFT,
    PriorSensitivityReport,
    prior_sensitivity,
)

__all__ = [
    # identifiability
    "identifiability",
    "IdentifiabilityReport",
    "DEFAULT_RANK_RTOL",
    # sensitivity
    "prior_sensitivity",
    "PriorSensitivityReport",
    "CRITERION_SHIFT",
    # joint priors
    "JeffreysPrior",
]
