"""The design-time diagnostics, projected into evaluation reports (§0.10).

Two verdicts already have a home: ``identifiability`` decides against
:data:`~bayesmith.diagnose.identifiability.DEFAULT_RANK_RTOL` and
``prior_sensitivity`` against
:data:`~bayesmith.diagnose.sensitivity.CRITERION_SHIFT`, both inside
:mod:`bayesmith.diagnose`.  What this module adds is a PROJECTION -- the
report fields read off and filed on §0 ruling 7's two axes -- and the whole
content of these tests is that it projects rather than judges again.  A second
copy of ``0.1`` here would be a second answer to one question, agreeing with
the first until somebody retunes one of them.

**Everything runs inside ``jax.enable_x64(True)``, graph construction
included.**  That is not tidiness: both diagnostics refuse a float32 ambient
precision by name, and they refuse a graph whose constants were traced outside
the block SEPARATELY -- probe_28 §9's first run died on exactly that.  The
module is marked ``x64`` so a reader knows why, and the context manager is
still opened per test so the file also passes in a default-precision session.

The measured anchors below were re-taken in this worktree with
``docs/probes/probe_28_model_checking_seams.py 100 9`` before any tolerance
here was chosen; the fixtures the probe does not carry (a prior tight enough
to FAIL, a nonlinear pair whose two routes disagree, a selection whose
likelihood curvature is singular) were measured the same way and their numbers
are quoted at the assertion that uses them.
"""

from __future__ import annotations

import ast
import pathlib

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts._codec import canonical_dumps, canonical_loads
from bayesmith.artifacts.base import ComputeBudget
from bayesmith.artifacts.identity import ArtifactKind
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.tasks import PosteriorTask, new_task_meta
from bayesmith.diagnose.identifiability import DEFAULT_RANK_RTOL
from bayesmith.diagnose.sensitivity import CRITERION_SHIFT
from bayesmith.dispatch.task import compile_task, execute_task
from bayesmith.evaluation.diagnostics import (
    IDENTIFIABILITY,
    PRIOR_SENSITIVITY,
    identifiability_report,
    prior_sensitivity_report,
)
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import collinear_pair, straight_line

pytestmark = pytest.mark.x64

#: Small on purpose. The subject of these reports is a Result, and what the
#: projection reads off it is its identity -- not one draw.
BUDGET = ComputeBudget(draws=8, warmup=8, chains=1)


def posterior_for(graph, *, seed=0):
    """A real ``PosteriorResult`` over ``graph``, to be the report's subject."""
    task = PosteriorTask(
        meta=new_task_meta(label="t7-subject"),
        budget=BUDGET,
        nuts_on_collapse=False,
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    result = execute_task(planned, key=jax.random.key(seed))
    assert not isinstance(result, Refusal), result
    return result


def nonlinear_pair(
    *,
    prior_amp=(0.2, 0.4),
    prior_rate=(2.0, 0.3),
    amp=1.4,
    rate=0.8,
    sigma=0.15,
    n=12,
    seed=7,
):
    """``d ~ N(amp * exp(-rate x), sigma)`` with priors that pull on both.

    Nonlinear in ``rate``, so the closed form and the refit expand a genuinely
    curved posterior about two different points, and how far apart they land is
    a dial: the declared priors. The three cells used below were measured in
    this worktree by sweeping that dial, and they are chosen to straddle
    ``CRITERION_SHIFT`` **while ``verified`` is False on both sides** -- which
    is the only arrangement that can tell a projection reading the
    verification first from one reading the shift first:

    * ``prior_amp=(0.2, 0.4)``, ``prior_rate=(2.0, 0.3)`` -- worst shift
      ``rate[0] = +2.219541`` sigma, ``verified`` 0 of 2, refit converged. A
      shift-first projection files FAIL.
    * ``prior_amp=(1.4, 3.0)``, ``prior_rate=(1.0, 1.0)`` -- worst shift
      ``rate[0] = +0.037928`` sigma, ``verified`` 0 of 2, refit converged. A
      shift-first projection files **PASS**, which is the worse of the two
      mistakes: a gate acts on a PASS.
    * ``prior_rate=(3.0, 0.2)`` -- the likelihood's own curvature at the mode
      is singular and ``prior_sensitivity`` refuses by name.

    The fourth cell is on the other side of that transition and is the only
    PASS in this file with more than one latent:

    * ``prior_amp=(1.4, 3.0)``, ``prior_rate=(0.8, 1.0)`` -- worst shift
      ``rate[0] = +0.00938137439161523`` sigma, ``verified`` **2 of 2**, refit
      converged. APPLICABLE x PASS.

    So the transition from ``verified`` 0 of 2 to 2 of 2 sits between
    ``prior_rate`` ``(1.0, 1.0)`` and ``(0.8, 1.0)`` at ``prior_amp=(1.4,
    3.0)``, and both sides of it are asserted.
    """
    x = jnp.linspace(0.1, 3.0, n)
    data = amp * jnp.exp(-rate * x) + sigma * jax.random.normal(
        jax.random.key(seed), (n,)
    )

    def model():
        xs = const("X", x)
        a = sample("amp", lambda: dist.Normal(*prior_amp))
        r = sample("rate", lambda: dist.Normal(*prior_rate))
        mu = det("mu", lambda a_, r_, x_: a_ * jnp.exp(-r_ * x_), a, r, xs)
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def collinear_trio(*, n=8, sigma=0.4, prior_std=3.0, seed=13):
    """``mu = (a + b + c) X`` -- ``collinear_pair`` with a third summand.

    ``collinear_pair`` has nullity 1, and every fixture in this file that
    reaches ``_identifiability_findings`` has nullity 0 or 1. A loop over
    ``range(nullity)`` and a loop over ``range(min(nullity, 1))`` are the same
    function on that set, so the payload §0.2 asks a FAILING report to carry
    ("finding 带 participation") could be truncated to its first row for every
    model with more than one blind direction and nothing here would go red.
    This is the fixture that separates them.

    Measured in this worktree (x64):
    ``n_par=3 rank=1 nullity=2 n_data=8``, participation
    ``dir 0 {a 0.6666666666666666, b 0.1666666666666665, c 0.16666666666666663}``,
    ``dir 1 {a 0.0, b 0.4999999999999999, c 0.5000000000000001}``.

    **Those per-direction numbers are not asserted, and the reason is worth
    stating.** Both null directions have the same singular value -- exactly
    zero -- so the basis SVD returns for that 2-plane is arbitrary, and a
    LAPACK that rotated it would change every number above while changing
    nothing about the model. What the test asserts instead is basis-invariant:
    the participation shares SUMMED over an orthonormal basis of the null
    space are the diagonal of the projector onto it, which for the null space
    of ``(1, 1, 1)`` is ``I - vv^T/3`` and therefore ``2/3`` at every latent.
    That two-thirds is DERIVED; the tolerance it is asserted at is measured
    (worst deviation 2.2e-16 here, pinned at 1e-12).
    """
    x = jnp.linspace(1.0, 3.0, n)
    data = 2.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, prior_std))
        b = sample("b", lambda: dist.Normal(0.0, prior_std))
        c = sample("c", lambda: dist.Normal(0.0, prior_std))
        mu = det(
            "mu",
            lambda a_, b_, c_, x_: (a_ + b_ + c_) * x_,
            a,
            b,
            c,
            xs,
            linear_in=("a", "b", "c"),
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def findings(report: EvaluationReport, code: str):
    """Every finding with this code, in the order the report filed them."""
    return [item for item in report.findings if item.code == code]


def finding(report: EvaluationReport, code: str):
    """The single finding with this code, or a failure naming what there was."""
    matches = [item for item in report.findings if item.code == code]
    assert len(matches) == 1, (code, [item.code for item in report.findings])
    return matches[0]


# ------------------------------------------------------------ identifiability


class TestIdentifiability:
    """nullity 0 PASSes, nullity > 0 FAILs, and the finding names who mixed."""

    def test_a_line_with_one_slope_is_identified(self):
        with jax.enable_x64(True):
            graph = straight_line()
            report = identifiability_report(graph, posterior_for(graph))
        assert report.report_kind == IDENTIFIABILITY
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.PASS
        # probe_28 §9: n_par=1 rank=1 nullity=0
        assert finding(report, "nullity").observed == 0
        assert finding(report, "nullity").expected == 0
        spectrum = finding(report, "rank_spectrum")
        n_par, n_data, rank, rtol, _threshold, _weakest = spectrum.observed
        assert (n_par, n_data, rank) == (1, 8, 1)
        assert rtol == DEFAULT_RANK_RTOL
        assert spectrum.expected == 1

    def test_two_latents_the_data_only_sees_the_sum_of_fail(self):
        with jax.enable_x64(True):
            graph = collinear_pair()
            report = identifiability_report(graph, posterior_for(graph))
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.FAIL
        # probe_28 §9: n_par=2 rank=1 nullity=1, participation a 0.5 / b 0.5
        assert finding(report, "nullity").observed == 1
        n_par, _n_data, rank, *_ = finding(report, "rank_spectrum").observed
        assert (n_par, rank) == (2, 1)

    def test_the_failing_report_names_who_the_blind_direction_mixes(self):
        """§0.2's parenthesis: the finding carries the participation.

        The verdict alone says "something is degenerate"; a user acts on WHICH
        latents. Measured here and by probe_28 §9 to fifteen digits:
        ``{'a': 0.4999999999999999, 'b': 0.4999999999999999}`` -- a deviation
        from 0.5 of 1.11e-16 on both, measured, not reasoned. The ``1e-9`` is
        a FITTED constant, not a derived band: it sits about seven decades
        above that measured deviation and eight below the 0.5-vs-1.0
        distinction the number exists to carry, and any value in that gap
        would do.
        """
        with jax.enable_x64(True):
            graph = collinear_pair()
            report = identifiability_report(graph, posterior_for(graph))
        direction = finding(report, "null_participation")
        index, shares = direction.observed
        assert index == 0
        assert dict(shares).keys() == {"a", "b"}
        for name, share in shares:
            assert abs(share - 0.5) < 1e-9, (name, share)

    def test_every_blind_direction_is_filed_not_just_the_first(self):
        """Nullity 2, and the report has to carry TWO participation findings.

        What this catches, exactly: ``for index in range(report.nullity)``
        weakened to ``range(min(report.nullity, 1))``. That mutant leaves the
        verdict alone -- FAIL is decided by ``nullity`` and nothing else -- and
        silently truncates the payload to the first blind direction for every
        model that has more than one. With ``collinear_pair`` as the only
        failing fixture, ``min(n, 1)`` was a no-op and the whole suite stayed
        green under it.

        The count is asserted against the report's OWN ``nullity``, not
        against a literal 2: the rule is "one finding per direction", and a
        rule stated against the number it derives from cannot be satisfied by
        a fixture that happens to have as many of one as of the other.
        """
        with jax.enable_x64(True):
            graph = collinear_trio()
            report = identifiability_report(graph, posterior_for(graph))
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.FAIL
        nullity = finding(report, "nullity")
        assert (nullity.observed, nullity.expected) == (2, 0)
        n_par, _n_data, rank, *_ = finding(report, "rank_spectrum").observed
        assert (n_par, rank) == (3, 1)

        directions = findings(report, "null_participation")
        assert len(directions) == nullity.observed
        assert [item.observed[0] for item in directions] == [0, 1]

    def test_the_two_blind_directions_span_the_plane_they_should(self):
        """The payload of the pair, asserted where it is basis-independent.

        The two null directions share a singular value of exactly zero, so
        which orthonormal basis of that plane the SVD hands back is arbitrary
        and the per-direction shares are not a property of the model. Their
        SUM over the basis is: it is the diagonal of the projector onto the
        null space, ``I - vv^T/3`` for ``v = (1, 1, 1)``, hence exactly ``2/3``
        at each of ``a``, ``b`` and ``c``. Derived -- the ``1e-12`` it is
        asserted at is not: worst measured deviation here is 2.2e-16.

        Each direction's own shares still sum to 1, which says they are shares
        of something normalized rather than of whatever length the SVD chose.
        """
        with jax.enable_x64(True):
            graph = collinear_trio()
            report = identifiability_report(graph, posterior_for(graph))
        totals = {"a": 0.0, "b": 0.0, "c": 0.0}
        for item in findings(report, "null_participation"):
            shares = dict(item.observed[1])
            assert shares.keys() == totals.keys()
            assert sum(shares.values()) == pytest.approx(1.0, abs=1e-12)
            for name, share in shares.items():
                totals[name] += share
        for name, total in totals.items():
            assert total == pytest.approx(2.0 / 3.0, abs=1e-12), (name, total)

    def test_a_fully_identified_model_files_no_participation(self):
        """The finding exists only where there is a direction to describe."""
        with jax.enable_x64(True):
            graph = straight_line()
            report = identifiability_report(graph, posterior_for(graph))
        assert [item.code for item in report.findings] == ["nullity", "rank_spectrum"]

    def test_the_verdict_recomputes_from_the_findings_alone(self):
        """G8: no reader of this report has to run the diagnostic again."""
        with jax.enable_x64(True):
            pairs = [
                (straight_line(), Conclusion.PASS),
                (collinear_pair(), Conclusion.FAIL),
            ]
            for graph, expected in pairs:
                report = identifiability_report(graph, posterior_for(graph))
                nullity = finding(report, "nullity")
                recomputed = (
                    Conclusion.PASS
                    if nullity.observed == nullity.expected
                    else Conclusion.FAIL
                )
                assert recomputed is report.conclusion is expected


# ---------------------------------------------------------- prior sensitivity


class TestPriorSensitivity:
    """PASS below CRITERION_SHIFT, FAIL at or above it, ABSTAIN unverified."""

    def test_a_prior_two_sigma_wide_is_not_driving_the_line(self):
        with jax.enable_x64(True):
            graph = straight_line()
            report = prior_sensitivity_report(graph, posterior_for(graph))
        assert report.report_kind == PRIOR_SENSITIVITY
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.PASS
        # probe_28 §9: worst=w[0] shift=-0.0421 sigma
        worst = finding(report, "worst_shift")
        name, index, shift = worst.observed
        assert (name, index) == ("w", 0)
        assert shift == pytest.approx(-0.0421, abs=1e-4)
        assert worst.expected == CRITERION_SHIFT

    def test_the_report_carries_the_width_that_would_have_moved_it(self):
        """probe_28 §9: ``criterion_std=1.297`` against a declared ``2``.

        The margin, not just the verdict: a reader who is choosing the prior
        needs the width at which this latent's shift would REACH
        ``CRITERION_SHIFT``, and 1.297 < 2 is the statement that the declared
        prior is on the safe side of it.
        """
        with jax.enable_x64(True):
            graph = straight_line()
            report = prior_sensitivity_report(graph, posterior_for(graph))
        margin = finding(report, "prior_width_margin")
        name, index, prior_std = margin.observed
        assert (name, index) == ("w", 0)
        assert prior_std == pytest.approx(2.0, abs=1e-12)
        assert margin.expected == pytest.approx(1.297, abs=1e-3)
        assert margin.expected < prior_std

    def test_a_prior_tight_enough_to_move_the_mode_fails(self):
        """``straight_line(prior_std=0.5)``: measured shift ``-0.66865`` sigma.

        Six and a half times ``CRITERION_SHIFT``, so the FAIL does not turn on
        the last digit of either number -- what it turns on is the projection
        comparing against the diagnose constant rather than a copy of it.
        """
        with jax.enable_x64(True):
            graph = straight_line(prior_std=0.5)
            report = prior_sensitivity_report(graph, posterior_for(graph))
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.FAIL
        _name, _index, shift = finding(report, "worst_shift").observed
        assert shift == pytest.approx(-0.66865, abs=1e-4)
        assert abs(shift) >= CRITERION_SHIFT

    def test_two_routes_that_disagree_abstain_rather_than_fail(self):
        """The ordering §0.2 fixes, and it is the one a shortcut gets wrong.

        ``nonlinear_pair`` measures a worst shift of ``+2.21954`` sigma -- far
        past ``CRITERION_SHIFT`` -- while ``verified`` carries a False. A
        projection that compared the shift first would report a confident FAIL
        from a number nothing cross-checked. The two routes disagreeing is a
        statement about the ROUTES, not about the prior.
        """
        with jax.enable_x64(True):
            graph = nonlinear_pair()
            report = prior_sensitivity_report(graph, posterior_for(graph))
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.ABSTAIN
        verification = finding(report, "refit_verification")
        refit_converged, verified, n_par = verification.observed
        assert refit_converged is True
        assert verified < n_par
        assert verification.expected == (True, n_par, n_par)
        _name, _index, shift = finding(report, "worst_shift").observed
        assert abs(shift) >= CRITERION_SHIFT  # a FAIL, had the order been wrong

    def test_an_unverified_small_shift_abstains_rather_than_passing(self):
        """The other side of the same boundary, and the cell that matters.

        Measured: worst shift ``+0.037928`` sigma -- comfortably INSIDE
        ``CRITERION_SHIFT`` -- with ``verified`` 0 of 2. A projection that read
        the shift first would file APPLICABLE x PASS here, and a PASS is what a
        gate acts on: the report would say "the priors are not driving this
        fit" on the strength of a number whose second route disagreed with it.

        The pair of cells is the point. Both have ``verified`` False and they
        sit on opposite sides of the criterion, so no single-cell mutation of
        the branch order survives both.
        """
        with jax.enable_x64(True):
            graph = nonlinear_pair(prior_amp=(1.4, 3.0), prior_rate=(1.0, 1.0))
            report = prior_sensitivity_report(graph, posterior_for(graph))
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.ABSTAIN
        _name, _index, shift = finding(report, "worst_shift").observed
        assert shift == pytest.approx(0.037928, abs=1e-5)
        assert abs(shift) < CRITERION_SHIFT  # a PASS, had the order been wrong
        _refit, verified, n_par = finding(report, "refit_verification").observed
        assert verified < n_par

    def test_a_verified_two_latent_fit_passes_on_a_count_not_a_boolean(self):
        """The only PASS in this file with ``n_par > 1``, and it is needed.

        What this catches, exactly: ``verified = int(np.count_nonzero(
        report.verified))`` weakened to ``int(np.any(report.verified))``.
        Every other prior-sensitivity fixture here is either one-latent
        (``straight_line``, where a count and a boolean coincide) or never
        verified (``nonlinear_pair``'s three unverified cells, where both are
        0), so no cell distinguished them and the mutant kept the file green
        while deleting the PASS branch for every multi-latent model: with
        ``verified`` ``[True, True]`` it files 1, and ``1 < n_par`` abstains.

        The assertion is therefore on the COUNT -- ``(True, 2, 2)`` -- and not
        merely on the conclusion. ``2`` is what the mutant falsifies.

        Measured (x64, this worktree): ``worst = ('rate', 0,
        +0.00938137439161523)``, ``verified`` 2 of 2, ``refit_converged``
        True, ``criterion_std = 0.27451407384852783`` against a declared width
        of ``1.0``. The two digit pins below are value pins -- their FORM is
        not derived, they are the digits measured cut at the last one -- and
        they sit next to the assertions that carry the meaning: ``abs(shift) <
        CRITERION_SHIFT``, the verification triple, and ``expected <
        prior_std``.
        """
        with jax.enable_x64(True):
            graph = nonlinear_pair(prior_amp=(1.4, 3.0), prior_rate=(0.8, 1.0))
            report = prior_sensitivity_report(graph, posterior_for(graph))
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.PASS

        verification = finding(report, "refit_verification")
        refit_converged, verified, n_par = verification.observed
        assert n_par == 2
        assert (refit_converged, verified, n_par) == (True, 2, 2)
        assert verification.expected == (True, n_par, n_par)

        name, index, shift = finding(report, "worst_shift").observed
        assert (name, index) == ("rate", 0)
        assert shift == pytest.approx(0.0093814, abs=1e-5)
        assert abs(shift) < CRITERION_SHIFT

        margin = finding(report, "prior_width_margin")
        _name, _index, prior_std = margin.observed
        assert prior_std == pytest.approx(1.0, abs=1e-12)
        assert margin.expected == pytest.approx(0.27451, abs=1e-4)
        assert margin.expected < prior_std

    def test_a_refused_diagnostic_is_unverifiable_and_carries_the_refusal(self):
        """``prior_rate=(3.0, 0.2)``: the likelihood curvature at the mode is
        singular, so the mode the shift is measured FROM does not exist.

        The check applies to this model and could not be run -- §0 ruling 7's
        UNVERIFIABLE, not a FAIL and not an exception escaping into a caller
        that asked for a report.
        """
        with jax.enable_x64(True):
            graph = nonlinear_pair(prior_rate=(3.0, 0.2))
            report = prior_sensitivity_report(graph, posterior_for(graph))
        assert report.applicability is Applicability.UNVERIFIABLE
        assert report.conclusion is Conclusion.ABSTAIN
        refused = finding(report, "diagnostic_refused")
        assert refused.observed == "GraphError"
        assert "prior_sensitivity cannot report a prior shift" in refused.message


# ------------------------------------------------------------------- float32


class TestAmbientFloat32:
    """§0.10: a float32 environment is UNVERIFIABLE, and it does not raise."""

    @pytest.mark.parametrize(
        "project", [identifiability_report, prior_sensitivity_report]
    )
    def test_a_float32_environment_returns_a_report_rather_than_raising(self, project):
        """probe_28 §9's own first run died here, and a caller assembling a
        gate has no more right to an exception than probe_28 had.

        The subject is built in x64 and the projection is CALLED outside it,
        which is the second half of the refusal the diagnose family states:
        wrapping only the call leaves the graph's constants at float32.
        """
        with jax.enable_x64(True):
            graph = straight_line()
            subject = posterior_for(graph)
        with jax.enable_x64(False):
            report = project(graph, subject)
        assert report.applicability is Applicability.UNVERIFIABLE
        assert report.conclusion is Conclusion.ABSTAIN
        refused = finding(report, "diagnostic_refused")
        assert refused.observed == "GraphError"
        assert "enable_x64" in refused.message

    def test_the_refusal_text_reaches_the_finding_verbatim(self):
        """Not a paraphrase: the remedy is IN the diagnose message, and a
        projection that summarised it would drop the half that says the graph
        has to be built inside the block too."""
        with jax.enable_x64(True):
            graph = straight_line()
            subject = posterior_for(graph)
        with jax.enable_x64(False):
            report = identifiability_report(graph, subject)
        message = finding(report, "diagnostic_refused").message
        assert "building the graph inside the block" in message


# ------------------------------------------------------- the projection rules


class TestItProjectsRatherThanJudges:
    """§0.2's last sentence and 红线 3, as things that can go red."""

    def test_the_module_writes_down_no_number_of_its_own(self):
        """A float literal in this module would be a second copy of a
        threshold that already has an owner -- the defect §0.2 names when it
        says the thresholds live in ``diagnose/`` and the projection only
        reads verdict fields.

        Floats specifically: an int is an index, a count or a nullity, and
        those are structure rather than tuning.
        """
        from bayesmith.evaluation import diagnostics

        source = pathlib.Path(diagnostics.__file__).read_text(encoding="utf-8")
        floats = [
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Constant) and isinstance(node.value, float)
        ]
        assert floats == [], f"diagnostics.py hard-codes {floats}"

    def test_the_thresholds_it_compares_against_are_the_diagnose_objects(self):
        """Identity, not equality: two 0.1s compare equal until one moves."""
        from bayesmith import diagnose
        from bayesmith.evaluation import diagnostics

        assert diagnostics.CRITERION_SHIFT is diagnose.CRITERION_SHIFT
        assert diagnostics.DEFAULT_RANK_RTOL is diagnose.DEFAULT_RANK_RTOL

    @pytest.mark.parametrize(
        "project", [identifiability_report, prior_sensitivity_report]
    )
    def test_the_subject_ref_points_at_the_result_it_was_given(self, project):
        with jax.enable_x64(True):
            graph = straight_line()
            subject = posterior_for(graph)
            report = project(graph, subject)
        assert report.subject_ref.artifact_id == subject.meta.artifact_id
        assert report.subject_ref.revision == subject.meta.revision
        assert report.subject_ref.artifact_type is ArtifactKind.RESULT
        assert report.meta.parent_refs == (report.subject_ref,)
        assert report.meta.fingerprints == subject.run.fingerprints

    @pytest.mark.parametrize(
        "project", [identifiability_report, prior_sensitivity_report]
    )
    def test_the_report_survives_the_codec_with_its_findings(self, project):
        with jax.enable_x64(True):
            graph = collinear_pair()
            report = project(graph, posterior_for(graph))
        restored = canonical_loads(canonical_dumps(report), expected=EvaluationReport)
        assert restored == report
        assert restored.findings == report.findings
