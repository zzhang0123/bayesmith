"""R4 Task 5 -- ``EvidenceTask`` compiles, executes, and refuses by name.

The fifth Task was the one R1 froze and could not answer. It answers now, for
exactly one structure class: a whole-graph-exact linear-Gaussian model, which
is ``classification.nuts == ()`` and ``classification.exact.method == "gcr"``.
Everything else is refused, and the refusal names the residual integral R5 will
supply rather than returning a number nothing graded.

Every refusal below is checked twice: that the evidence task is refused, AND
that the SAME graph still compiles a posterior task unchanged. A refusal that
also breaks the neighbouring capability is not a refusal, it is a regression
wearing one -- and asserting only the first half would not tell them apart.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import compile_task, const, det, execute_task, observe, sample, trace
from bayesmith.artifacts.identity import ArtifactKind
from bayesmith.artifacts.refusal import PREMISES, Refusal
from bayesmith.artifacts.results import EvidenceComponent, EvidenceResult
from bayesmith.artifacts.tasks import (
    EvidenceTask,
    PosteriorTask,
    TaskKind,
    new_task_meta,
)
from tests.dispatch.test_task_protocol import model_ref as _model_ref
from bayesmith.dispatch.evidence import EVIDENCE_COMPONENT_NAMES, assemble_exact
from bayesmith.dispatch.task import SUPPORTED_TASK_KINDS

SIGMA = 0.5
PRIOR_MEAN = 0.35


def _model_factory(prior_std=1.7, n=6):
    basis = jnp.linspace(-1.0, 1.0, n) + 0.3
    data = 1.2 * (basis * 0.9) + SIGMA * jax.random.normal(jax.random.key(3), (n,))

    def model():
        w = sample("w", lambda: dist.Normal(PRIOR_MEAN, prior_std))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data)

    return model, basis, data


def _ref():
    return _model_ref()


def _evidence_task(**kw):
    return EvidenceTask(meta=new_task_meta(label="log evidence"), **kw)


def _compile(graph, task=None):
    return compile_task(graph, task or _evidence_task(), model_ref=_ref())


class TestTheSeamIsOpen:
    def test_evidence_is_no_longer_the_unanswered_task_kind(self):
        assert TaskKind.EVIDENCE in SUPPORTED_TASK_KINDS
        assert SUPPORTED_TASK_KINDS == set(TaskKind), (
            "all five task kinds are answered now; the capability fall-through "
            "in execute_task is unreachable from compile_task and its test "
            "must say so rather than quietly losing its subject"
        )

    def test_a_whole_graph_exact_model_returns_an_evidence_result(self):
        with jax.enable_x64(True):
            model, basis, data = _model_factory()
            graph = trace(model)
            planned = _compile(graph)
            assert not isinstance(planned, Refusal), getattr(
                planned, "failed_premise", None
            )
            result = execute_task(planned)
            assert isinstance(result, EvidenceResult)
            assert result.meta.artifact_type is ArtifactKind.RESULT

    def test_the_number_is_the_dense_analytic_log_evidence(self):
        """The gate that matters. A dense oracle built from the model's own
        parameters, sharing no QR, no pivots and no offset arithmetic."""
        with jax.enable_x64(True):
            model, basis, data = _model_factory()
            result = execute_task(_compile(trace(model)))

            b = np.asarray(basis, dtype=float)
            d = np.asarray(data, dtype=float)
            covariance = 1.7**2 * np.outer(b, b) + SIGMA**2 * np.eye(b.size)
            residual = d - b * PRIOR_MEAN
            _, logdet = np.linalg.slogdet(covariance)
            oracle = -0.5 * (
                residual @ np.linalg.solve(covariance, residual)
                + logdet
                + b.size * np.log(2.0 * np.pi)
            )
            assert result.log_evidence == pytest.approx(float(oracle), abs=1e-9)

    @pytest.mark.parametrize("prior_std", [0.05, 0.25, 1.0, 4.0, 60.0])
    def test_the_seam_carries_the_five_terms_at_every_prior_scale(self, prior_std):
        with jax.enable_x64(True):
            model, _, _ = _model_factory(prior_std=prior_std)
            result = execute_task(_compile(trace(model)))
            names = {c.name for c in result.exact_components}
            assert names == set(EVIDENCE_COMPONENT_NAMES)
            assert all(
                isinstance(c, EvidenceComponent) for c in result.exact_components
            )
            direct = assemble_exact(trace(model), ("w",), {})
            assert result.log_evidence == pytest.approx(
                float(direct.log_evidence), abs=1e-12
            )

    def test_no_residual_and_no_error_bar_are_reported_as_absent(self):
        """``standard_error = None`` says none was computed; ``0.0`` would say
        one was measured and found to be zero. R4 assembles exactly, so the
        honest field is the absent one -- and a Bayes-factor consumer has to
        branch on which."""
        with jax.enable_x64(True):
            result = execute_task(_compile(trace(_model_factory()[0])))
            assert result.standard_error is None
            assert result.residual_component is None
            assert result.repeat_result_refs == ()
            assert result.consistency_report_ref is None
            assert result.posterior_representation is None

    def test_the_result_is_fingerprinted_consistently_with_itself(self):
        """Nothing in ``_envelope`` checks that a Result's ``meta.fingerprints``
        equals its ``run.fingerprints``, and every comparability rule reads one
        or the other. A rule reading ``meta`` on one artifact and ``run`` on
        another would compare two different bundles while looking symmetric."""
        with jax.enable_x64(True):
            result = execute_task(_compile(trace(_model_factory()[0])))
            assert result.meta.fingerprints == result.run.fingerprints


class TestWhatItRefuses:
    """Each case: the evidence task is refused, the posterior task is not."""

    def _refused(self, graph, premise, task=None):
        outcome = _compile(graph, task)
        assert isinstance(outcome, Refusal), (
            f"expected a refusal naming {premise}, got {type(outcome).__name__}"
        )
        assert outcome.failed_premise == premise
        assert outcome.failed_premise in PREMISES
        assert outcome.grounds
        assert outcome.remedies
        return outcome

    def _posterior_still_compiles(self, graph):
        outcome = compile_task(
            graph, PosteriorTask(meta=new_task_meta(label="p")), model_ref=_ref()
        )
        assert not isinstance(outcome, Refusal), (
            "the refusal broke the neighbouring capability, which makes it a "
            "regression rather than a coverage boundary"
        )

    def test_a_model_with_a_sampled_block_names_the_residual_integral(self):
        """§0.14: R4 stops at exact assembly. A model whose posterior needs
        NUTS needs a residual integral, and R5 supplies it; saying so by name
        is the difference between a boundary and a silence."""
        with jax.enable_x64(True):
            observed = jnp.asarray([0.4, -0.2, 1.1])

            def model():
                tau = sample("tau", lambda: dist.HalfNormal(1.0))
                nu = det("nu", lambda t_: t_ * jnp.ones(3), tau)
                observe(
                    "e", lambda m: dist.Normal(m, 0.8).to_event(1), nu, obs=observed
                )

            graph = trace(model)
            refusal = self._refused(graph, "evidence_residual_integral_required")
            assert "residual" in refusal.remedies[0].message.lower()
            self._posterior_still_compiles(graph)

    def test_an_improper_prior_is_refused_and_the_posterior_is_not(self):
        """A model whose posterior is perfectly well defined and whose evidence
        is not. That asymmetry is the whole reason compilation is task-aware:
        §2.2 names exactly this case."""
        with jax.enable_x64(True):
            basis = jnp.linspace(-1.0, 1.0, 6) + 0.3
            data = 1.2 * (basis * 0.9)

            def model():
                w = sample(
                    "w", lambda: dist.ImproperUniform(dist.constraints.real, (), ())
                )
                b = const("basis", basis)
                mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
                observe(
                    "d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data
                )

            graph = trace(model)
            self._refused(graph, "evidence_prior_proper")
            self._posterior_still_compiles(graph)

    def test_a_prior_with_finite_mass_that_is_not_one_is_refused_too(self):
        """Proper and unnormalised are different failures and both stop a Z.

        The mass is 7, so the integral converges -- and the number it converges
        to is 7x the one a Bayes factor compares. A single "is it proper" check
        would pass this.
        """
        with jax.enable_x64(True):
            basis = jnp.linspace(-1.0, 1.0, 6) + 0.3
            data = 1.2 * (basis * 0.9)

            def model():
                w = sample(
                    "w",
                    lambda: dist.ImproperUniform(
                        dist.constraints.interval(-2.0, 5.0), (), ()
                    ),
                )
                b = const("basis", basis)
                mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
                observe(
                    "d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data
                )

            graph = trace(model)
            refusal = self._refused(graph, "evidence_prior_normalised")
            found = {f.code: f for f in refusal.grounds}
            assert any("mass" in f.message for f in found.values())

    def test_repeating_a_run_is_refused_rather_than_ignored(self):
        """Both options sit inside the TASK fingerprint slot, so a caller
        naming one gets a different digest and reasonably expects different
        behaviour. Silently ignoring it is the worst of the three available
        behaviours."""
        with jax.enable_x64(True):
            graph = trace(_model_factory()[0])
            self._refused(
                graph, "task_options_recognised", task=_evidence_task(repeat_count=3)
            )
            self._refused(
                graph,
                "task_options_recognised",
                task=_evidence_task(reconstruct_posterior=True),
            )
            self._posterior_still_compiles(graph)


def test_a_float32_environment_is_refused_by_name():
    """R4's numbers are absolute constants and float32 does not carry them.

    Measured in probe_31 §5: the same assembly agrees with a dense analytic log
    evidence to 3.4e-07 at float32 and to 8.9e-16 at float64.  An evidence
    reported at the first is not wrong in a way anyone would notice, which is
    what makes it worth refusing rather than warning about.

    Deliberately NOT inside an x64 context -- that is the measurement.  Judged
    by ``jnp.result_type(float)``, the dtype the arithmetic would actually run
    at, rather than by reading a config flag, so the caller who set the
    process-global switch and the caller who used the context manager get the
    same answer.  This follows ``refuse_ambient_float32``'s shape, which R3 set
    for identifiability and prior sensitivity.
    """
    graph = trace(_model_factory()[0])
    outcome = _compile(graph)
    assert isinstance(outcome, Refusal)
    assert outcome.failed_premise == "evidence_requires_x64"
    assert outcome.grounds[0].observed == "float32"
    assert outcome.grounds[0].expected == "float64"

    # And the neighbouring capability is untouched: a posterior task compiles
    # in float32 exactly as it always did.
    posterior = compile_task(
        graph, PosteriorTask(meta=new_task_meta(label="p")), model_ref=_ref()
    )
    assert not isinstance(posterior, Refusal)
