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
from bayesmith.evidence.compress import ResidualSummary
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

    **Not the same statistic as :func:`template_modes`, and the pair is worth
    reading together** (ledger D45). This one evaluates ``||R x - z||^2`` at a
    point YOU choose and can be re-asked at any value of the survivors; that
    one reads what was recorded when the raw data still existed -- the
    residual left after each epoch's own best fit, which no later call can
    recompute -- and it carries the named template projections this one has
    no way to know about. They coincide only at the per-epoch minimiser, which
    is not the point a campaign evaluates at. This one also reads its dof off
    the FIRST term's row count, so a campaign whose epochs differ in flagging
    wants ``template_modes``, which sums each epoch's own.

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


def epoch_residuals(
    summaries: Sequence[ResidualSummary],
) -> tuple[dict[str, Any], ...]:
    """Section 9.3's per-epoch table, in the order the summaries were given.

    Order is preserved rather than sorted by anything. For a chained campaign
    the archive order IS the time order, and a diagnostic that reordered it
    would make a drift look like scatter.

    **Distinct from :func:`coherent_mode`, and the difference is the input.**
    That one evaluates ``||R x - z||^2`` at a point you choose, from the
    stored term, and can be re-asked at any value of the survivors. This one
    reads what :func:`~bayesmith.evidence.compress.residual_summary` recorded
    when the raw data still existed: the residual left after the epoch's OWN
    best fit, which no later call can recompute. The two coincide only at the
    per-epoch minimiser, which is not the point a campaign evaluates at.

    Args:
        summaries: one per epoch, from
            :func:`~bayesmith.evidence.compress.residual_summary`.

    Returns:
        One dict per epoch with ``"chi2"``, ``"dof"``, ``"reduced_chi2"`` and
        ``"templates"``, the last being ``{name: projection}``.

    Raises:
        StructureError: for an empty campaign, or one whose epochs name
            different templates -- see :func:`refuse_mixed_templates`.
    """
    summaries = tuple(summaries)
    names = refuse_mixed_templates(summaries)
    return tuple(
        {
            "chi2": float(summary.chi2),
            "dof": int(summary.dof),
            "reduced_chi2": float(summary.reduced_chi2),
            "templates": {
                name: float(np.asarray(summary.projections)[index])
                for index, name in enumerate(names)
            },
        }
        for summary in summaries
    )


def refuse_mixed_templates(
    summaries: Sequence[ResidualSummary],
) -> tuple[str, ...]:
    """The one template list this campaign shares, or a refusal.

    The projections are stored as a bare array whose meaning is POSITIONAL,
    and the summary is read by name across a whole campaign. Two epochs naming
    ``("gain_ripple", "ground_pickup")`` and ``("ground_pickup",
    "gain_ripple")`` have compatible shapes and incompatible contents, so
    averaging them positionally reports each template's mean as a mixture of
    both -- a finite, plausible number about nothing.
    """
    summaries = tuple(summaries)
    if not summaries:
        raise StructureError(
            "this campaign has no epochs. An empty campaign has no residual "
            "and no mean to resolve, and an empty table would read downstream "
            "as a clean result rather than as no result."
        )
    declared = {summary.template_names for summary in summaries}
    if len(declared) > 1:
        raise StructureError(
            f"these epochs name different systematic templates: "
            f"{sorted(declared)}. The projections are stored positionally, so "
            "reading them by name across a campaign needs one list in one "
            "order. Recompress the odd epochs with the same templates, or "
            "score the groups separately."
        )
    return next(iter(declared))


def template_modes(summaries: Sequence[ResidualSummary]) -> dict[str, Any]:
    """Is there a fault shaped like one of the named templates?

    The named half of section 9.3, and the counterpart to
    :func:`coherent_mode`'s unnamed one. Neither substitutes for the other: a
    chi-square z needs no guess about what the fault looks like and therefore
    cannot say what it IS, while a named template says what it is and is
    silent when the guess was wrong.

    Under the null each projection is a standard normal in its own right --
    that is what the whitening buys -- so the campaign mean is a z-score at
    ``sqrt(N)`` with no calibration step.

    ``scatter`` is reported beside every mean and is the part to read second.
    A mean-level fault leaves it at 1.0 while an under-estimated noise model
    raises both together, and the two failing differently is what tells them
    apart. It is the population spread about the sample mean (``ddof=0``), so
    a shifted mean does not inflate it.

    **The chi-square z here uses each epoch's OWN dof**, summed, rather than
    the first epoch's. A campaign whose nights differ in flagging has a
    different null per night, and it is the SUM that is chi-square distributed:
    ``Var(sum chi2_k) = 2 sum k``. That reduces to the textbook form for a
    uniform campaign and stays exact for a ragged one, which
    :func:`coherent_mode` -- reading a row count off the first term -- cannot
    do.

    Returns:
        ``{"n_epochs", "chi2_mean", "chi2_dof", "chi2_z", "templates"}``,
        where ``chi2_dof`` is the shared dof or ``None`` for a ragged
        campaign, and ``templates`` is ``{name: {"mean", "scatter", "z"}}``.

    Raises:
        StructureError: for an empty campaign or mixed template lists.
    """
    summaries = tuple(summaries)
    names = refuse_mixed_templates(summaries)
    n_epochs = len(summaries)
    chi2 = np.array([float(summary.chi2) for summary in summaries])
    dofs = {int(summary.dof) for summary in summaries}
    total_dof = sum(int(summary.dof) for summary in summaries)
    spread = math.sqrt(2.0 * total_dof) if total_dof > 0 else float("nan")
    report: dict[str, Any] = {
        "n_epochs": n_epochs,
        "chi2_mean": float(chi2.mean()),
        "chi2_dof": next(iter(dofs)) if len(dofs) == 1 else None,
        "chi2_z": float((chi2.sum() - total_dof) / spread),
        "templates": {},
    }
    for index, name in enumerate(names):
        values = np.array(
            [float(np.asarray(summary.projections)[index]) for summary in summaries]
        )
        report["templates"][name] = {
            "mean": float(values.mean()),
            "scatter": float(values.std()),
            # sqrt(N), which is the whole point: the mean does not shrink with
            # N and its uncertainty does.
            "z": float(values.mean() * math.sqrt(n_epochs)),
        }
    return report


_PRIOR_REMEDY = (
    "pass prior_fisher over the same latents in the same column order -- a "
    "single epoch legitimately constrains only a subspace, so the "
    "leave-one-out information is singular at small N without one."
)


def _campaign_arrays(
    terms: Sequence[SqrtInfo],
) -> tuple[list[np.ndarray], list[np.ndarray], int]:
    """``[R_e]``, ``[z_e]`` and the shared width, or a refusal.

    In numpy, because everything that reads them here is an ``O(N)`` loop of
    dense solves on an offline diagnostic, and because a campaign's terms have
    ragged row counts -- which is exactly what a stacked jax array cannot be.
    """
    terms = tuple(terms)
    if not terms:
        raise StructureError(
            "this campaign has no epochs, so there is nothing to score. An "
            "empty result would read downstream as a clean run."
        )
    first = terms[0]
    for term in terms[1:]:
        if term.names != first.names or term.shapes != first.shapes:
            raise StructureError(
                "these epochs are over different latents: "
                f"{list(first.names)} vs {list(term.names)}. Scoring one "
                "against the rest means adding their information column for "
                "column, and columns that describe different parameters do "
                "not add."
            )
    factors = [np.asarray(term.factor, dtype=float) for term in terms]
    targets = [np.asarray(term.target, dtype=float) for term in terms]
    return factors, targets, int(first.width)


def held_out_z(
    terms: Sequence[SqrtInfo],
    prior_fisher: Any,
    prior_mean: Any = None,
) -> tuple[dict[str, float], ...]:
    """How surprising is each epoch to the rest of the campaign?

    For a linear-Gaussian model this is exact and needs no simulation. Write
    the leave-one-out posterior ``N(mu_-e, Sigma_-e)`` and the epoch's own
    factor ``[R_e | z_e]``. Then ``z_e = R_e theta + eps`` with unit-covariance
    ``eps`` -- that is what the square-root form MEANS -- and ``mu_-e - theta``
    is independent of ``eps``, so::

        m = R_e mu_-e - z_e  ~  N(0, I + R_e Sigma_-e R_e^T)

    and ``m^T (I + V)^-1 m`` is chi-square on ``rank(R_e)`` degrees of
    freedom. The returned ``z`` is that, standardised.

    **Computed by subtracting one summand, never by downdating the
    accumulator.** A QR accumulation cannot be un-summed stably. The campaign
    total is formed once in ``(F, b)`` form and one epoch's contribution
    subtracted per row, which is ``O(N)`` rather than ``O(N^2)`` and loses at
    most ``log10(N)`` of float64's sixteen digits -- affordable because this
    is offline, and safe because it removes one positive-semidefinite summand
    out of N rather than a triangular factor out of its own product.

    **What it can see, and what it cannot.** A single rogue epoch stands out
    by a wide margin, and so does a campaign whose epochs genuinely differ.
    But where every epoch carries the SAME design -- the realistic case -- a
    coherent error's in-span half shifts ``z_e`` and ``mu_-e`` by amounts that
    cancel in ``m``, and the clean and biased campaigns return the same
    scores. ``tests/evidence/test_held_out.py`` measures both directions
    rather than asserting them. Read this beside
    :func:`refuse_undeclared_coherent_error`, never instead of it.

    Args:
        terms: the campaign's stored terms, in any order -- this statistic is
            exchangeable even when the campaign is not, because each epoch is
            scored against all the others.
        prior_fisher: ``F_prior`` over the same latents in the same column
            order. **Required, not optional**: ``Sigma_-e`` is singular at
            small N without one.
        prior_mean: the prior's mean over the raveled columns. Zero if absent.

    Returns:
        One dict per epoch, in the order given, with ``"chi2"``, ``"dof"`` and
        ``"z"``.

    Raises:
        StructureError: for an empty campaign, epochs over different latents, a
            prior over a different number of columns, a singular leave-one-out
            information, or a score that comes out non-finite.
    """
    terms = tuple(terms)
    factors, targets, width = _campaign_arrays(terms)

    fisher = np.asarray(prior_fisher, dtype=float)
    if fisher.shape != (width, width):
        raise StructureError(
            f"prior_fisher has shape {fisher.shape}, but this campaign's "
            f"epochs are over {width} raveled values "
            f"({list(terms[0].names)}). The prior is added column for column, "
            "so a prior over a different parameter set is not a prior over "
            "these."
        )
    mean = (
        np.zeros(width) if prior_mean is None else np.asarray(prior_mean, dtype=float)
    )
    if mean.shape != (width,):
        raise StructureError(
            f"prior_mean has shape {mean.shape}, but this campaign's epochs "
            f"are over {width} raveled values ({list(terms[0].names)})."
        )

    total_fisher = fisher + sum(row.T @ row for row in factors)
    total_b = fisher @ mean + sum(
        row.T @ target for row, target in zip(factors, targets, strict=True)
    )

    rows: list[dict[str, float]] = []
    for index, (row, target) in enumerate(zip(factors, targets, strict=True)):
        left_fisher = total_fisher - row.T @ row
        left_b = total_b - row.T @ target
        try:
            covariance = np.linalg.inv(np.linalg.cholesky(left_fisher)).T
        except np.linalg.LinAlgError as error:
            raise StructureError(
                f"the campaign with epoch {index} left out carries singular "
                "information, so there is no leave-one-out posterior to score "
                f"that epoch against. Remedy: {_PRIOR_REMEDY}"
            ) from error
        covariance = covariance @ covariance.T
        residual = row @ (covariance @ left_b) - target
        spread = np.eye(row.shape[0]) + row @ covariance @ row.T
        chi2 = float(residual @ np.linalg.solve(spread, residual))
        dof = int(row.shape[0])
        # `not (x >= 0.0)` rather than `x < 0.0`: NaN is False for BOTH, and it
        # is NaN that has to be caught. A NaN z sails through every
        # `z > threshold` an audit could write and reads as the quietest epoch
        # of the run.
        if not (chi2 >= 0.0) or not np.isfinite(chi2):
            raise StructureError(
                f"epoch {index} scored {chi2}, which is not finite and "
                "non-negative. A stored factor or target carries NaN or inf, "
                "and NaN loses every comparison a campaign audit could make "
                "about it. Recompress that epoch."
            )
        rows.append(
            {
                "chi2": chi2,
                "dof": dof,
                "z": (chi2 - dof) / float(np.sqrt(2.0 * dof)) if dof > 0 else float("nan"),
            }
        )
    return tuple(rows)


def _shrinkage_table(sigmas: Mapping[int, Any]) -> tuple[np.ndarray, np.ndarray]:
    """``(log N, log sigma)`` as two flat arrays, with every trap refused first.

    Three of them, and each is here because the failure it prevents is a
    finite, plausible number rather than a crash:

    * fewer than two campaign sizes -- one point admits every slope, so a
      fitted power would be an invention;
    * a non-finite or non-positive sigma -- ``log`` maps those to ``nan`` and
      ``-inf``, and a ``nan`` power loses every comparison a caller could make
      about it, exactly as a ``nan`` z-score does;
    * a ragged table -- a pooled fit over two widths at one size and three at
      another is not one fit, and its slope weights the sizes unequally
      without saying so.
    """
    sizes = sorted(sigmas)
    if len(sizes) < 2:
        raise StructureError(
            f"shrinkage_power needs at least two campaign sizes; got {sizes}. "
            "A single point admits every slope, so any number returned here "
            "would be the caller's assumption rather than a measurement."
        )
    bad_sizes = [n for n in sizes if not (int(n) > 0)]
    if bad_sizes:
        raise StructureError(
            f"these campaign sizes are not positive: {bad_sizes}. The fit is "
            "in log N, and log of a non-positive size is -inf or nan, which "
            "would make the returned power finite-looking or nan rather than "
            "refused."
        )
    columns = [np.atleast_1d(np.asarray(sigmas[n], dtype=float)) for n in sizes]
    widths = {column.shape for column in columns}
    if len(widths) > 1:
        raise StructureError(
            "these campaign sizes report different numbers of widths: "
            f"{ {n: c.shape for n, c in zip(sizes, columns, strict=True)} }. A "
            "pooled power is one fit over the same latents at every size; a "
            "ragged table weights the sizes unequally without saying so. Fit "
            "the shared latents, or fit each size separately."
        )
    stacked = np.concatenate(columns)
    # `not (x > 0)` rather than `x <= 0`: NaN is False for both comparisons,
    # and NaN is the case that has to be caught. `isfinite` alone would let a
    # negative sigma through to `log`, which returns nan with a warning.
    if not np.all(np.isfinite(stacked)) or not np.all(stacked > 0.0):
        offenders = [
            (n, column.tolist())
            for n, column in zip(sizes, columns, strict=True)
            if not (np.all(np.isfinite(column)) and np.all(column > 0.0))
        ]
        raise StructureError(
            "every sigma must be finite and strictly positive; these are not: "
            f"{offenders}. A posterior width of zero, inf or nan is a broken "
            "covariance, not a very tight measurement -- check the campaign "
            "that produced it rather than fitting its logarithm."
        )
    log_size = np.repeat(
        np.log(np.asarray(sizes, dtype=float)), stacked.size // len(sizes)
    )
    return log_size, np.log(stacked)


def shrinkage_power(sigmas: Mapping[int, Any]) -> float:
    """The fitted exponent of ``sigma_N ~ N^p``. **A sanity check, not a test.**

    Returned as a bare float only from here; :func:`shrinkage_report` is what
    a caller should print, because it carries the caveat in the same object as
    the number.

    For a Gaussian model ``sigma_N = (sum_e F_e + F_prior)^-1/2`` does not read
    the data. So this quantity is **data-independent**, ``p = -0.5`` holds by
    construction, and a uniform rescaling ``F_e -> (1+c) F_e`` cannot move it
    at all -- the intercept is free. A diagnostic that could be fooled by
    injecting a shared systematic and watching this number is therefore no
    diagnostic, which is the whole reason
    :func:`refuse_undeclared_coherent_error` exists.

    Args:
        sigmas: ``{campaign size: posterior widths}``. A scalar width is
            accepted and treated as a one-element array. Every size must
            report the same number of widths, in the same order.

    Returns:
        The ordinary-least-squares slope of ``log sigma`` on ``log N``, pooled
        over the widths.

    Raises:
        StructureError: for fewer than two campaign sizes, a non-positive
            size, a ragged table, or a sigma that is not finite and strictly
            positive.
    """
    log_size, log_sigma = _shrinkage_table(sigmas)
    centred = log_size - log_size.mean()
    return float(centred @ (log_sigma - log_sigma.mean()) / (centred @ centred))


def shrinkage_report(sigmas: Mapping[int, Any]) -> dict[str, Any]:
    """:func:`shrinkage_power` with its limits attached to it.

    ``detects_coherent_bias`` is ``False``, always, and it is a FIELD rather
    than a sentence in a docstring because a number and its caveat travel
    together or they do not travel. A deterministic error shared across
    epochs contributes no variance at all, so the posterior width is the same
    array element for element on a clean campaign and on a badly biased one,
    and this power is the same to every digit either way.
    """
    return {
        "power": shrinkage_power(sigmas),
        "n_values": tuple(sorted(sigmas)),
        "detects_coherent_bias": False,
        "caveat": (
            "sigma_N is data-independent for a Gaussian model, so this power "
            "is -0.5 by construction. A deterministic error shared across "
            "epochs contributes no variance, so it cannot move this number at "
            "all. Use template_modes(), coherent_mode() and the systematic "
            "floor."
        ),
    }


def tightest_direction(block: Any) -> tuple[float, Any]:
    """``(width, unit direction)`` of the narrowest direction of a covariance block.

    **The tightest DIRECTION, not the tightest coordinate**, and the
    difference is a basis rotation wide. For any correlated posterior the
    smallest eigen-direction is below every diagonal entry, so a refusal
    keyed on ``min(diag(block))`` watches a quantity that is not the first to
    go under. ``tests/evidence/test_systematic_floor.py`` measures the gap on
    a near-collinear campaign, where the coordinate widths sit an order of
    magnitude ABOVE a floor that the tightest direction is well below.

    The direction is returned because a bare number is half the value: "your
    error bar is too tight" is not actionable, "your error bar on
    ``0.707 x0 + 0.707 x1`` is too tight" is. Its sign is fixed by making the
    largest-magnitude component positive, so the same posterior reports the
    same vector rather than one that flips with LAPACK's mood.

    ``(nan, None)`` for a block that is not finite: NaN cannot be
    eigendecomposed, and a poisoned campaign must report ``nan`` and be
    refused by the NaN-safe comparison in :func:`systematic_floor` rather than
    raise a linear-algebra error here.
    """
    block = np.asarray(block, dtype=float)
    if not np.all(np.isfinite(block)):
        return float("nan"), None
    values, vectors = np.linalg.eigh(block)
    smallest = float(values[0])
    direction = np.asarray(vectors[:, 0], dtype=float)
    direction = direction * np.sign(direction[int(np.argmax(np.abs(direction)))])
    # A covariance is positive definite by construction here, so a
    # non-positive eigenvalue is roundoff on a direction the campaign
    # constrains not at all. Zero is the honest width for it and it is below
    # every legal floor, which is the safe direction. `nan` was handled above,
    # never here, because `smallest > 0.0` is False for NaN and would silently
    # become zero.
    return (math.sqrt(smallest) if smallest > 0.0 else 0.0), direction


def systematic_floor(
    term: SqrtInfo,
    prior_fisher: Any,
    floors: Mapping[str, Any],
    n_epochs: int,
) -> dict[str, dict[str, Any]]:
    """Has this campaign out-run its own calibration?

    The floor is the declared width of a shared calibration product -- one
    solution, one beam model, one flag table serving every epoch --
    **projected into the latents' units by the analyst**. It is a declaration
    and not a measurement, and that is forced rather than lazy: the in-span
    half of a coherent error biases the survivors identically in every epoch
    and leaves no residual anywhere, so it passes per-epoch chi-square,
    split-half and leave-one-out, and NO statistic computed from the stored
    terms can recover it.

    What the campaign does know is its own width, and that width falls as
    ``N^-1/2`` while the shared product's does not fall at all. So the whole
    content of this function is: when does one pass under the other.

    ``crossing_epoch`` is **computed from the observed width**, not quoted:
    ``sigma_N`` is extrapolated as ``sigma_N sqrt(N / N')`` and solved for
    ``sigma_N' = floor``, giving ``N' = ceil(N (sigma_N / floor)^2)``. That
    extrapolation ignores the prior's share, which shrinks as N grows, so it
    predicts the crossing marginally early -- the conservative direction.

    Args:
        term: the campaign's folded term over the survivors.
        prior_fisher: ``F_prior`` over the same columns, added to the term's
            own information before inverting.
        floors: ``{latent: declared width in that latent's units}``. Every
            entry must name a latent the term is over and be finite and
            strictly positive.
        n_epochs: how many epochs the term was folded from. Used only for the
            extrapolation.

    Returns:
        One entry per named latent, each a dict of ``"sigma"``, ``"floor"``,
        ``"below_floor"``, ``"crossing_epoch"`` and ``"direction"``.

    Raises:
        StructureError: for a campaign of no epochs, a floor naming a latent
            the term is not over, a floor that is not a finite positive width,
            a prior of the wrong shape, or information that is not positive
            definite.
    """
    if n_epochs <= 0:
        raise StructureError(
            "systematic_floor needs at least one epoch. The crossing epoch is "
            "N (sigma_N / floor)^2, which is zero for N = 0 -- an empty "
            "campaign would report that it passed under its systematic floor "
            "before it started."
        )
    width = int(term.width)
    fisher = np.asarray(prior_fisher, dtype=float)
    if fisher.shape != (width, width):
        raise StructureError(
            f"prior_fisher has shape {fisher.shape}, but this term is over "
            f"{width} raveled values ({list(term.names)})."
        )
    unknown = [name for name in floors if name not in term.names]
    if unknown:
        raise StructureError(
            f"these floors name latents this campaign is not over: {unknown}. "
            f"The term carries {list(term.names)}. A floor on something the "
            "campaign does not estimate has no width to be compared against, "
            "and reporting it as passing would be a claim about nothing."
        )
    bad = [
        name
        for name, value in floors.items()
        if not (float(np.asarray(value)) > 0.0 and np.isfinite(float(np.asarray(value))))
    ]
    if bad:
        raise StructureError(
            f"these floors are not finite positive widths: {bad}. A floor of "
            "zero, inf or nan is not a very good calibration -- it is a "
            "missing declaration, and comparing a width against it would "
            "return a verdict about nothing."
        )

    information = np.asarray(term.information(), dtype=float) + fisher
    try:
        chol = np.linalg.cholesky(information)
    except np.linalg.LinAlgError as error:
        raise StructureError(
            "this campaign's information is not positive definite, so it has "
            f"no posterior covariance to compare against a floor. {_PRIOR_REMEDY}"
        ) from error
    root = np.linalg.inv(chol).T
    covariance = root @ root.T

    spans, position = {}, 0
    for name, shape in zip(term.names, term.shapes, strict=True):
        size = 1
        for dim in shape:
            size *= int(dim)
        spans[name] = (position, position + size)
        position += size

    report: dict[str, dict[str, Any]] = {}
    for name, declared in floors.items():
        start, stop = spans[name]
        sigma, direction = tightest_direction(covariance[start:stop, start:stop])
        floor = float(np.asarray(declared))
        # `not (sigma > floor)`, NEVER `sigma <= floor`: NaN is False for both
        # comparisons, so the second form waves a poisoned campaign through
        # while the same dict reports the nan. `inf` is handled by the same
        # expression and correctly -- a campaign that constrains nothing is
        # not tighter than anything.
        #
        # **A NaN cannot reach this line from inside this function today, and
        # that is measured rather than assumed**: the cholesky above raises on
        # a non-finite information matrix, and it raises long before the
        # inverse could overflow to inf (a separation of 1e-160 is already
        # refused while 1e-12 still gives a finite covariance of order 1e12).
        # The form is kept because `tightest_direction` is public and DOES
        # return nan, and because the rule is right whatever forms the
        # covariance -- but a mutation between the two spellings survives this
        # suite, and pretending otherwise would be the more expensive error.
        report[name] = {
            "sigma": sigma,
            "floor": floor,
            "below_floor": not (sigma > floor),
            "crossing_epoch": (
                math.ceil(n_epochs * (sigma / floor) ** 2)
                if math.isfinite(sigma) and sigma > 0.0
                else None
            ),
            "direction": direction,
        }
    return report
