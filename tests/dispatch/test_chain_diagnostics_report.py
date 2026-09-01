"""Projecting multi-chain diagnostics into one EvaluationReport (Task 5).

The unified multi-chain diagnostics are the SiteDiagnostic entries that
chain_diagnostics already computed. Task 5 does NOT re-judge convergence: it
projects the stored verdict -- r_hat, ess, ceiling, converged, worst and
reason -- into one EvaluationReport whose findings are referenced by
PosteriorResult.report_refs.

Which routes carry diagnostics is measured, not assumed. The NUTS path and
the mixed Gibbs sweep draw through a chain, so they carry per-site
diagnostics. The gcr/gcr+snis whole-graph paths draw independently and carry
None (r-hat has no referent). The collapse arm is the third None case: its
regression draws are iid conditional on the retained thetas, so its
diagnostics stay None for exactly the same reason -- see the docstring of
bayesmith.dispatch.execute._collapsed. A route with None diagnostics
produces no report, which is abstention rather than endorsement.
"""

from __future__ import annotations

import uuid

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts import dump_artifact, load_artifact
from bayesmith.artifacts.base import ArtifactKind, ArtifactRef, ComputeBudget
from bayesmith.artifacts.identity import (
    FingerprintBundle,
    FingerprintKind,
    fingerprint,
)
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.reports import Applicability, Conclusion, EvaluationReport
from bayesmith.artifacts.tasks import PosteriorTask, new_task_meta
from bayesmith.dispatch.execute import chain_diagnostics
from bayesmith.dispatch.task import (
    _chain_diagnostics_report,
    compile_task,
    execute_task,
)
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import (
    bilinear_pair,
    mixed_radiometer,
    radiometer,
    straight_line,
)

BUDGET = ComputeBudget(draws=8, warmup=8, chains=1)


def posterior_task(**overrides) -> PosteriorTask:
    fields = {
        "meta": new_task_meta(label="run"),
        "budget": BUDGET,
        "nuts_on_collapse": False,
    }
    fields.update(overrides)
    return PosteriorTask(**fields)


def planned_for(graph, task):
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    return planned


def _converged(draws, coords, seed=0, rho=0.0):
    """A chain that IS sampling the target, optionally autocorrelated."""
    g = np.random.default_rng(seed)
    x = g.standard_normal((draws, coords))
    if rho:
        for t in range(1, draws):
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho**2) * x[t]
    return x


def _bundle() -> FingerprintBundle:
    """A minimal fingerprint bundle: the four slots every artifact has."""
    return FingerprintBundle(
        model_source=fingerprint(FingerprintKind.MODEL_SOURCE, "pilot"),
        graph_structure=fingerprint(FingerprintKind.GRAPH_STRUCTURE, "theta -> y"),
        data=fingerprint(FingerprintKind.DATA, "y=[1,2,3]"),
        task=fingerprint(FingerprintKind.TASK, "posterior"),
    )


def _subject_ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=str(uuid.uuid4()), revision=0, artifact_type=ArtifactKind.RESULT
    )


def _collapse_graph(n=6, sigma=0.5, x_loc=0.35, x_scale=1.7, th_loc=-0.2, th_scale=1.1, seed=3):
    """The collapse arm's mixed graph: a constant-sigma exact block + a NUTS block.

    Mirrors the fixture in tests/dispatch/test_collapse.py, because the collapse
    arm refuses a prediction-dependent exact block and this shape is the one
    that routes there.
    """
    basis = jnp.linspace(-1.0, 1.0, n) + 0.3
    data = 1.2 * (basis * 0.9) + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = sample("x", lambda: dist.Normal(x_loc, x_scale))
        th = sample("th", lambda: dist.Normal(th_loc, th_scale))
        Bc = const("basis", basis)
        mu = det("mu", lambda t_, b_, x_: t_ * (b_ * x_), th, Bc, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma).to_event(1), mu, obs=data)

    return trace(model)


# ------------------------------------------------- the routes that ran a chain


class TestAChainRunReferencesItsDiagnosticsReport:
    def test_a_nuts_run_references_a_report(self):
        result = execute_task(
            planned_for(bilinear_pair(), posterior_task()), key=jax.random.key(3)
        )
        assert result.representation.method == "nuts"
        assert len(result.report_refs) == 1
        assert result.report_refs[0].artifact_type is ArtifactKind.EVALUATION_REPORT

    def test_a_mixed_run_references_a_report(self):
        result = execute_task(
            planned_for(mixed_radiometer(), posterior_task()), key=jax.random.key(6)
        )
        assert result.representation.method == "gcr+mh"
        assert len(result.report_refs) == 1
        assert result.report_refs[0].artifact_type is ArtifactKind.EVALUATION_REPORT


class TestAChainWithoutDiagnosticsProducesNoReport:
    def test_gcr_records_no_report(self):
        result = execute_task(
            planned_for(straight_line(), posterior_task()), key=jax.random.key(2)
        )
        assert result.representation.method == "gcr"
        assert result.report_refs == ()

    def test_gcr_snis_records_no_report(self):
        result = execute_task(
            planned_for(radiometer(), posterior_task()), key=jax.random.key(5)
        )
        assert result.representation.method == "gcr+snis"
        assert result.report_refs == ()

    def test_collapse_records_no_report(self):
        """The collapse arm's regression draws are iid, so its diagnostics are
        None for the same reason as the whole-graph iid paths -- no r-hat
        referent -- and it produces no report either."""
        graph = _collapse_graph()
        task = posterior_task(backend_options=(("collapse", True),))
        result = execute_task(planned_for(graph, task), key=jax.random.key(0))
        assert result.representation.method == "collapse"
        assert result.report_refs == ()


# ------------------------------------------- the projection, and its round-trip


class TestTheProjectionCarriesEverySiteAndRoundTrips:
    def test_each_site_becomes_one_finding_carrying_the_diagnostic(self):
        """A frozen coordinate decides the worst index, and the Finding carries
        every field the SiteDiagnostic computed -- r_hat, ess, ceiling,
        converged, worst and reason -- without re-judging convergence."""
        draws = _converged(400, 4, seed=3)
        draws[:, 2] = 0.0  # frozen, and not at either end
        diagnostics = chain_diagnostics({"alm": draws})
        assert diagnostics["alm"].worst == (2,)
        assert not diagnostics["alm"].converged

        report = _chain_diagnostics_report(
            diagnostics, subject_ref=_subject_ref(), fingerprints=_bundle()
        )

        assert report.report_kind == "chain_diagnostics"
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.FAIL
        assert len(report.findings) == 1
        finding = report.findings[0]
        assert finding.code == "chain_diagnostics"
        assert finding.expected is True
        name, r_hat, ess, ceiling, worst, converged = finding.observed
        assert name == "alm"
        assert r_hat == diagnostics["alm"].r_hat
        assert ess == diagnostics["alm"].ess
        assert ceiling == diagnostics["alm"].ceiling
        assert worst == (2,)
        assert converged is False
        assert finding.message == diagnostics["alm"].reason

    def test_a_converged_site_reports_pass(self):
        draws = _converged(2000, 5, seed=8, rho=0.9)
        diagnostics = chain_diagnostics({"x": draws})
        assert diagnostics["x"].converged

        report = _chain_diagnostics_report(
            diagnostics, subject_ref=_subject_ref(), fingerprints=_bundle()
        )
        assert report.applicability is Applicability.APPLICABLE
        assert report.conclusion is Conclusion.PASS
        assert report.findings[0].message == "converged"
        assert report.findings[0].observed[5] is True

    def test_the_report_round_trips_preserving_worst_coordinate(self, tmp_path):
        draws = _converged(400, 4, seed=3)
        draws[:, 2] = 0.0
        diagnostics = chain_diagnostics({"alm": draws})
        report = _chain_diagnostics_report(
            diagnostics, subject_ref=_subject_ref(), fingerprints=_bundle()
        )

        path = tmp_path / "report.json"
        dump_artifact(report, path)
        loaded = load_artifact(path, expected=EvaluationReport)

        by_name = {finding.observed[0]: finding for finding in loaded.findings}
        assert by_name["alm"].observed[4] == (2,)
        assert by_name["alm"].observed[5] is False
        assert by_name["alm"].observed[1] == diagnostics["alm"].r_hat
        assert by_name["alm"].observed[3] == diagnostics["alm"].ceiling

