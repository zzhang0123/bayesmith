"""R4 Tasks 6 and 8 -- what an evidence can say about itself and another one.

Three report kinds and a gate. The interesting assertions are the two the R3
layer could not make: that ``evidence_prior_sensitivity`` and R3's
``prior_sensitivity`` disagree on the perturbation ``Z`` is most sensitive to,
and that comparability is a REPORT rather than a Refusal because a Refusal
cannot have a pair of Results as its subject.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import compile_task, const, det, execute_task, observe, sample, trace
from bayesmith.artifacts.identity import ArtifactKind, FingerprintKind
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.reports import Applicability, Conclusion
from bayesmith.artifacts.tasks import EvidenceTask, new_task_meta
from bayesmith.evaluation.evidence import (
    EVIDENCE,
    EVIDENCE_COMPARABILITY,
    EVIDENCE_NORMALIZATION_AUDIT,
    EVIDENCE_PRIOR_SENSITIVITY,
    comparability_report,
    normalization_audit_report,
    prior_sensitivity_report,
)
from tests.dispatch.test_task_protocol import model_ref

SIGMA = 0.5


def _graph(prior_std=1.7, seed=3, n=6, improper=False):
    basis = jnp.linspace(-1.0, 1.0, n) + 0.3
    data = 1.2 * (basis * 0.9) + SIGMA * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        if improper:
            w = sample("w", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
        else:
            w = sample("w", lambda: dist.Normal(0.35, prior_std))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data)

    return trace(model)


def _evidence(graph):
    planned = compile_task(
        graph, EvidenceTask(meta=new_task_meta(label="Z")), model_ref=model_ref()
    )
    assert not isinstance(planned, Refusal), planned.failed_premise
    return execute_task(planned)


class TestTheNormalizationAudit:
    def test_a_proper_model_passes(self):
        with jax.enable_x64(True):
            graph = _graph()
            report = normalization_audit_report(_evidence(graph), graph)
            assert report.report_kind == EVIDENCE_NORMALIZATION_AUDIT
            assert report.applicability is Applicability.APPLICABLE
            assert report.conclusion is Conclusion.PASS
            assert report.meta.artifact_type is ArtifactKind.EVALUATION_REPORT

    def test_the_verdict_recomputes_from_the_findings(self):
        """§8 R3 gate 8, carried into R4: a reader must be able to redo the
        verdict from the report's own fields, without a log."""
        with jax.enable_x64(True):
            graph = _graph()
            report = normalization_audit_report(_evidence(graph), graph)
            verdicts = {f.observed[1] for f in report.findings}
            assert verdicts == {"proper"}
            recomputed = (
                Conclusion.PASS
                if all(f.code == "prior.proper" for f in report.findings)
                else Conclusion.FAIL
            )
            assert recomputed is report.conclusion

    def test_an_improper_prior_fails_the_report_as_well_as_the_gate(self):
        """The compile gate refuses before assembling. This says the same thing
        about a Result someone hands you later, which is the case the report
        exists for -- an artifact outlives the compile that made it."""
        with jax.enable_x64(True):
            proper = _graph()
            subject = _evidence(proper)
            report = normalization_audit_report(subject, _graph(improper=True))
            assert report.applicability is Applicability.APPLICABLE
            assert report.conclusion is Conclusion.FAIL
            assert any("improper" in f.code for f in report.findings)

    def test_a_prior_the_audit_cannot_decide_abstains(self):
        with jax.enable_x64(True):
            subject = _evidence(_graph())

            def hierarchical():
                s = sample("s", lambda: dist.HalfNormal(1.0))
                w = sample("w", lambda s_: dist.Normal(0.0, s_), s)
                b = const("basis", jnp.linspace(-1.0, 1.0, 6) + 0.3)
                mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
                observe(
                    "d",
                    lambda m: dist.Normal(m, SIGMA).to_event(1),
                    mu,
                    obs=jnp.zeros(6),
                )

            report = normalization_audit_report(subject, trace(hierarchical))
            assert report.conclusion is Conclusion.ABSTAIN
            assert report.applicability is Applicability.APPLICABLE


class TestThePriorSensitivityThatIsNotR3s:
    def test_the_slope_is_reported_in_nats_per_e_fold(self):
        with jax.enable_x64(True):
            graph = _graph()
            report = prior_sensitivity_report(_evidence(graph), graph, ("w",))
            assert report.report_kind == EVIDENCE_PRIOR_SENSITIVITY
            assert report.report_kind != "prior_sensitivity", (
                "R3's kind measures a posterior mode displacement; sharing the "
                "name would put two statistical objects under one code"
            )
            assert report.applicability is Applicability.APPLICABLE
            assert report.conclusion is Conclusion.PASS
            slopes = [
                f for f in report.findings if f.code == "d_log_evidence_d_log_width"
            ]
            assert len(slopes) == 1
            assert abs(slopes[0].observed[1]) > 0.01

    def test_the_perturbation_is_one_a_mode_metric_cannot_see(self):
        """The measured disagreement with R3's report, stated as a property.

        R3's ``prior_sensitivity`` is a posterior MODE displacement in
        posterior sigmas. This report perturbs a prior's WIDTH and leaves its
        CENTRE alone, so the displacement a mode metric measures is zero by
        construction -- and ``log Z`` moves by 0.5 nats over the same
        perturbation, measured below.

        That is why the two are different report kinds. Sharing a code would
        put a quantity that is zero here and a quantity that is not under one
        name, which §1.4 invariant 3 forbids.
        """
        with jax.enable_x64(True):
            centre = 0.35
            narrow = _evidence(_graph(prior_std=0.5))
            wide = _evidence(_graph(prior_std=0.5 * float(jnp.e)))

            # The perturbation leaves the prior's centre exactly where it was.
            for graph in (_graph(prior_std=0.5), _graph(prior_std=0.5 * float(jnp.e))):
                from bayesmith.graph.evaluate import apply_probabilistic

                declared = apply_probabilistic(graph, graph.node("w"), {})
                assert float(jnp.asarray(declared.loc)) == pytest.approx(centre)

            moved = float(wide.log_evidence) - float(narrow.log_evidence)
            assert abs(moved) > 0.1, (
                "log Z must move under a perturbation a mode metric reports as "
                "zero, or the two reports would not need separate kinds"
            )

            report = prior_sensitivity_report(narrow, _graph(prior_std=0.5), ("w",))
            slope = next(
                f.observed[1]
                for f in report.findings
                if f.code == "d_log_evidence_d_log_width"
            )
            assert slope != 0.0
            assert slope * moved > 0.0, (
                "the reported slope and the measured move must at least agree "
                "in sign; they span different intervals of a curve that is not "
                "linear in log s, so their magnitudes need not match"
            )

    def test_a_prior_with_no_declared_width_is_reported_not_guessed(self):
        with jax.enable_x64(True):
            graph = _graph()
            report = prior_sensitivity_report(_evidence(graph), graph, ())
            assert report.applicability is Applicability.INAPPLICABLE
            assert report.conclusion is Conclusion.ABSTAIN


class TestComparability:
    def test_two_evidences_on_the_same_data_are_comparable(self):
        with jax.enable_x64(True):
            first = _evidence(_graph(prior_std=1.7))
            second = _evidence(_graph(prior_std=0.5))
            report = comparability_report(first, second)
            assert report.report_kind == EVIDENCE_COMPARABILITY
            assert report.applicability is Applicability.APPLICABLE
            assert report.conclusion is Conclusion.PASS
            factor = report.findings[0].observed[2]
            assert factor == pytest.approx(
                float(first.log_evidence) - float(second.log_evidence), abs=1e-12
            )

    def test_two_evidences_on_different_data_are_not(self):
        """The R4 completion gate: comparing different data semantics is
        refused. INAPPLICABLE x ABSTAIN -- the method does not apply here, and
        that is not the same as the models being indistinguishable."""
        with jax.enable_x64(True):
            first = _evidence(_graph(seed=3))
            second = _evidence(_graph(seed=11))
            assert FingerprintKind.DATA in _changed(first, second)
            report = comparability_report(first, second)
            assert report.applicability is Applicability.INAPPLICABLE
            assert report.conclusion is Conclusion.ABSTAIN
            assert report.findings[0].code == "different_data_semantics"

    def test_a_result_that_disagrees_with_itself_is_unverifiable(self):
        """Nothing in ``_envelope`` requires ``meta.fingerprints`` to equal
        ``run.fingerprints``, so a rule reading one on each side would compare
        two different bundles while looking symmetric."""
        with jax.enable_x64(True):
            first = _evidence(_graph(seed=3))
            other = _evidence(_graph(seed=11))
            forged = dataclasses.replace(
                first,
                meta=dataclasses.replace(
                    first.meta, fingerprints=other.run.fingerprints
                ),
            )
            report = comparability_report(forged, other)
            assert report.applicability is Applicability.UNVERIFIABLE
            assert report.conclusion is Conclusion.ABSTAIN
            assert report.findings[0].code == "result_disagrees_with_itself"

    def test_comparability_cannot_be_a_refusal_and_here_is_why(self):
        """The design constraint, asserted rather than asserted-in-prose.

        ``Refusal.__post_init__`` calls ``task_kind(self.task)``, which raises
        for anything that is not one of the five Tasks. So there is no Refusal
        whose subject is a pair of Results, and the report is not a stylistic
        choice.
        """
        with jax.enable_x64(True):
            first = _evidence(_graph())
            with pytest.raises(TypeError):
                Refusal(
                    meta=first.meta,
                    task=first,  # a Result, not a Task
                    failed_premise="evidence_prior_proper",
                    grounds=(),
                    scope=None,
                    remedies=(),
                )


def _changed(first, second):
    from bayesmith.artifacts.identity import changed_fingerprints

    return changed_fingerprints(first.run.fingerprints, second.run.fingerprints)


class TestTheGate:
    def test_the_gate_declares_its_three_requirements(self):
        assert EVIDENCE.name == "evidence"
        assert EVIDENCE.version == 1
        names = [item.name for item in EVIDENCE.requirements]
        assert names == [
            EVIDENCE_NORMALIZATION_AUDIT,
            EVIDENCE_PRIOR_SENSITIVITY,
            EVIDENCE_COMPARABILITY,
        ]
        required = [item.name for item in EVIDENCE.requirements if item.required]
        assert required == [EVIDENCE_NORMALIZATION_AUDIT], (
            "only the audit is required: an evidence with no comparison and no "
            "sensitivity sweep is still an evidence, and one whose priors were "
            "never weighed is not"
        )

    def test_the_gate_is_not_model_checking(self):
        """Folding these into ``model_checking@1`` would move its ``== 8``
        requirement pin and the permutation sweep behind it, for reports about
        a different artifact kind. One gate, one subject."""
        from bayesmith.evaluation.gate import MODEL_CHECKING

        assert EVIDENCE.identity != MODEL_CHECKING.identity
        assert EVIDENCE.identity == "evidence@1"
        assert len(MODEL_CHECKING.requirements) == 8
        overlap = {item.name for item in EVIDENCE.requirements} & {
            item.name for item in MODEL_CHECKING.requirements
        }
        assert not overlap
