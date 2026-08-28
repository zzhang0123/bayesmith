"""Each dataset's marginal likelihood, as a fixed-size sufficient statistic.

One epoch -- or one dataset -- of a linear-Gaussian model, compressed into a
square-root information term ``[R | z]`` plus a constant, together with
everything needed to build such terms, fold them, and say afterwards whether
the campaign that produced them can be trusted.

**This layer does not compute the Bayesian evidence, and the name is chosen to
stop implying that it does.** The evidence is
``p(d) = INT p(theta) PROD_i L_i(theta) d theta`` -- the parameters integrated
OUT, a single number, the thing model comparison needs. Nothing here computes
or consumes one. What a term stores is ``L_i(theta)``: dataset ``i``'s
likelihood with its OWN nuisances integrated away, a function of the
parameters. Every "marginal" in this subpackage is marginal over nuisances and
conditional on theta.

(The array layer can in fact produce a marginal likelihood of the whole model
if you marginalise every name -- :func:`~bayesmith.marginal.sqrtinfo.marginalise`
says so, and returns a zero-width term whose ``log_prob({})`` is that number.
The GRAPH entry points deliberately do not: :func:`compress_campaign` reads
priors only for the per-epoch latents. So the capability exists one level down
and is not offered one level up.)

The four quantities this subpackage sits between, since three of them get
called "evidence" in casual use and only one of them is:

* ``p(theta)`` -- the shared prior, applied exactly ONCE, and never stored in
  a term. A term is prior-free in the survivors on purpose: fold ``N`` of them
  and a prior baked into each would be applied ``N`` times.
* ``L_i(theta)`` -- what a term IS.
* ``p(theta | d_1..d_N) ~ p(theta) PROD_i L_i(theta)`` -- the target.
* ``p(d_1..d_N)`` -- the evidence. Out of scope; see above.

What lives here, and it is two things rather than one -- the arithmetic of
terms, and what a campaign of them can report about itself:

* :mod:`~bayesmith.marginal.sqrtinfo` -- the kernel: the ``[R | z]`` form,
  exact combination by QR, exact marginalisation of a block. It knows nothing
  about graphs, epochs or plans, which is what lets it be checked against a
  dense oracle.
* :mod:`~bayesmith.marginal.compress` -- one epoch's design and data to a term.
* :mod:`~bayesmith.marginal.factorize` -- which latents survive an epoch and
  which are integrated away inside it, derived from the graph.
* :mod:`~bayesmith.marginal.campaign` -- forming and folding a whole campaign.
* :mod:`~bayesmith.marginal.chain` -- a nuisance that DRIFTS across epochs,
  integrated out exactly by a square-root recursion.
* :mod:`~bayesmith.marginal.diagnostics` -- what a campaign can say about its
  own trustworthiness once the data is gone.

**A term is real by construction.** ``log_prob`` takes ``sum(residual**2)``, a
bilinear form, and a complex QR preserves the sesquilinear one instead; the
two disagree silently, so a complex ``factor`` or ``target`` is refused with
the route out (ledger D46). Carry a complex latent as its real degrees of
freedom -- :func:`~bayesmith.exact.block.real_parts` explains why, and
``ComplexNormal`` fixes the column convention.
"""

from bayesmith.marginal.campaign import (
    compress_campaign,
    epoch_observation,
    epoch_terms,
)
from bayesmith.marginal.compress import (
    ResidualSummary,
    compress,
    compress_epoch,
    epoch_joint,
    nuisance_prior,
    observed_mask,
    residual_summary,
)
from bayesmith.marginal.diagnostics import (
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
from bayesmith.marginal.factorize import (
    Factorization,
    epoch_leakage,
    factorize,
)
from bayesmith.marginal.sqrtinfo import (
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
