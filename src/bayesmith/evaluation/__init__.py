"""Model checking: reports ABOUT results, never results.

The layer the top-level design's §7.2 puts between the execution adapters and
a workflow.  Everything here reads a finished
:class:`~bayesmith.artifacts.results.PosteriorResult`,
:class:`~bayesmith.artifacts.results.PredictiveResult` or
:class:`~bayesmith.artifacts.results.SimulationResult` and produces an
:class:`~bayesmith.artifacts.reports.EvaluationReport` whose ``subject_ref``
points back at what it read.

**Three things this layer does not do**, and they are the reason it is a layer
rather than a handful of functions inside ``dispatch``:

* It does not MODIFY a result.  A check that could reweight a posterior would
  be an inference step wearing a verdict's schema.
* It does not CHOOSE an algorithm.  Deciding that a check failed and then
  re-running with a longer chain is a workflow's decision, made from a report,
  not a decision this layer may take on the execution layer's behalf (§2.4).
* It does not RE-JUDGE a verdict that already has a home.  ``identifiability``
  and ``prior_sensitivity`` decide their own thresholds inside
  :mod:`bayesmith.diagnose`; the projections here read the report fields and
  file them, so a threshold has one owner rather than two that agree until
  they do not.

**Dependencies run one way** -- ``evaluation`` imports ``dispatch``, ``graph``,
``artifacts`` and ``bridge.arviz``, and none of them imports it.
``tests/test_layering.py`` holds that direction, because the shortcut it
forbids (``dispatch`` importing a check so a run can grade itself) is exactly
the arrangement §2.4 describes as the thing to avoid.

**ArviZ stays optional.**  It is imported inside the function that needs it,
never at module scope and never through this ``__init__``, so a clone without
it installed gets an UNVERIFIABLE report rather than an ImportError (§7.3).
``tests/test_layering.py`` checks that in a subprocess.
"""

from __future__ import annotations

from bayesmith.evaluation.diagnostics import (
    identifiability_report,
    prior_sensitivity_report,
)

__all__ = ["ALPHA", "identifiability_report", "prior_sensitivity_report"]

#: D104. The two-sided false-positive rate EVERY random check in this layer
#: declares in advance, as §9.3 requires.
#:
#: Provenance is **borrowed**: 0.05 is statistics' conventional default, not a
#: number derived from anything measured in this repository, and writing it
#: down as borrowed is the point -- a derived threshold and a conventional one
#: answer differently to "why that value?", and a reader who cannot tell which
#: kind they are looking at will treat both as negotiable.
#:
#: One constant rather than a default argument per check.  Five checks each
#: choosing their own α would be five magic numbers wearing a statistical hat,
#: and the close-out cannot multiply five different rates by the number of
#: tests to state an expected false-positive count for the suite.  Everything
#: downstream is DERIVED from it and carries no number of its own: the
#: predictive band is ``[ALPHA / 2, 1 - ALPHA / 2]``, the Bonferroni factor for
#: m held-out points is ``ALPHA / (2 * m)``, and for K latent coordinates in
#: SBC it is ``ALPHA / K``.
ALPHA = 0.05
