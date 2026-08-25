"""What a campaign can say about its own trustworthiness, from stored terms.

Nothing here reads a graph, a prediction or a byte of raw data -- only the
fixed-size ``[R | z]`` terms :func:`~bayesmith.evidence.compress.compress`
stored while the data still existed. A diagnostic is something you run ON a
campaign, not something a campaign is.

**The blindness is the design, not a gap.** A deterministic error shared by
every epoch -- one calibration solution, one beam model, one flag table --
contributes NO variance. Split-half agrees to roundoff, leave-one-out returns
the same scores, the posterior width is the same array element for element,
and the answer is wrong. Such an error splits into two halves, and only one is
visible from data at all:

* the **out-of-span** half leaves a residual, and :func:`coherent_mode`
  reports it -- as a z-score, because what a common mode moves is a MEAN, and
  a mean over N epochs is resolved at ``sqrt(N)``;
* the **in-span** half is absorbed into the survivors identically in every
  epoch. It leaves no residual at the displaced point, so it passes every
  statistic here and biases the answer without limit as the campaign grows.

``tests/evidence/test_diagnostics.py`` measures that second half rather than
asserting it: an in-span injection of the same size as a detected out-of-span
one leaves ``chi2_z`` at noise level while displacing the answer by exactly
the injected amount. **That is not a gap to be closed with a better
statistic** -- it is why the honest response to an undetectable error class is
a refusal based on what the analyst declares, and why this module offers one
(:func:`refuse_undeclared_coherent_error`) instead of a number.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from bayesmith.errors import StructureError
from bayesmith.evidence.sqrtinfo import SqrtInfo


def epoch_chi_square(
    terms: Sequence[SqrtInfo], values: Mapping[str, Any]
) -> np.ndarray:
    """``||R x - z||^2`` per epoch, at one point over the survivors.

    The residual each stored term still carries -- computed from the term
    alone, which is what makes it available a thousand epochs after the data
    was discarded.
    """
    terms = tuple(terms)
    if not terms:
        raise StructureError(
            "epoch_chi_square needs at least one epoch. An empty campaign has "
            "no residual, and returning an empty array would read downstream "
            "as a clean result rather than as no result."
        )
    out = []
    for term in terms:
        residual = term.factor @ term.ravel(dict(values)) - term.target
        out.append(float(jnp.sum(residual**2)))
    return np.asarray(out)


def coherent_mode(
    terms: Sequence[SqrtInfo], values: Mapping[str, Any]
) -> dict[str, float]:
    """Is there a fault common to every epoch? -- the detectable half.

    A common mode contributes no variance, so it is invisible to split-half,
    to leave-one-out and to the posterior width. What it moves is the MEAN of
    the per-epoch residual, and a mean over ``N`` epochs is resolved at
    ``sqrt(N)`` -- which is why this reports a z-score rather than a
    magnitude.

    ``scatter`` is reported beside the mean and is the part to read second: a
    mean-level fault leaves it near 1, while an under-estimated noise model
    raises both together. The two failing differently is what tells them
    apart.

    Args:
        terms: the campaign's stored per-epoch terms.
        values: the point to evaluate the residual at -- normally the
            campaign's own posterior mean.

    Returns:
        ``{"n_epochs", "chi2_mean", "chi2_dof", "chi2_z", "scatter"}``.

    Raises:
        StructureError: if the campaign is empty, or if the epochs do not
            agree about which latents they are over -- a mean over terms that
            describe different parameters is not a mean of anything.
    """
    terms = tuple(terms)
    chi_square = epoch_chi_square(terms, values)
    first = terms[0]
    for term in terms[1:]:
        if term.names != first.names or term.shapes != first.shapes:
            raise StructureError(
                "coherent_mode was given epochs over different latents: "
                f"{list(first.names)} vs {list(term.names)}. Averaging a "
                "residual across terms that describe different parameters is "
                "not a mean of anything."
            )
    dof = float(first.factor.shape[0])
    n_epochs = len(terms)
    if dof <= 0.0:
        raise StructureError(
            "these terms have no rows, so their residual is identically zero "
            "and its mean carries no information. A term with no rows is one "
            "whose epoch constrained nothing."
        )
    # The mean of `n` chi-square(dof) variables has standard error
    # sqrt(2 dof / n): that sqrt(n) is the whole reason a common mode is
    # findable at all, and the reason this reports a z rather than a size.
    standard_error = math.sqrt(2.0 * dof / n_epochs)
    return {
        "n_epochs": float(n_epochs),
        "chi2_mean": float(chi_square.mean()),
        "chi2_dof": dof,
        "chi2_z": float((chi_square.mean() - dof) / standard_error),
        "scatter": float(chi_square.std(ddof=1) / math.sqrt(2.0 * dof))
        if n_epochs > 1
        else float("nan"),
    }


def refuse_undeclared_coherent_error(
    declared: bool,
    *,
    what: str = "a coherent error shared by every epoch",
) -> None:
    """Refuse to report on the half no statistic can see.

    **There is no number to return here, and that is the point.** The in-span
    half of a coherent error is absorbed into the survivors identically in
    every epoch: it leaves no residual, so :func:`coherent_mode` reads noise,
    a held-out epoch reads noise, split-half agrees to roundoff -- and the
    answer is biased, without limit, as the campaign grows. Measured in this
    package's own tests: an in-span injection leaves ``chi2_z`` inside its
    noise band while displacing the estimate by exactly the injected amount.

    So the honest interface is a refusal keyed on what the analyst DECLARES,
    not a report keyed on what the numbers show. A campaign that has not
    declared this class is not clean; it is unexamined, and the two must not
    read the same.

    Args:
        declared: whether the analyst has stated that this error class has
            been bounded by some means OUTSIDE the campaign's own residuals --
            an end-to-end simulation, a redundant instrument, an independent
            measurement of the shared component.
        what: names the class, for the message.

    Raises:
        StructureError: unless ``declared``.
    """
    if declared:
        return
    raise StructureError(
        f"{what} has not been declared as bounded, and this campaign cannot "
        "bound it from its own residuals. The half of such an error lying "
        "inside the design's column space is absorbed into the survivors "
        "identically in every epoch: it leaves NO residual, so every "
        "statistic here reads clean while the answer is biased, and the bias "
        "does not shrink as epochs accumulate -- it is the one error class "
        "that gets worse with more data, because the error bar keeps "
        "shrinking around it.\n\n"
        "This is a refusal rather than a warning because there is no "
        "statistic to improve. Bound it outside the campaign -- an end-to-end "
        "simulation, a redundant instrument, an independent measurement of "
        "the shared component -- and pass declared=True to record that you "
        "did. `coherent_mode` reports the OTHER half, which does leave a "
        "residual; a clean result from it is not evidence about this one."
    )
