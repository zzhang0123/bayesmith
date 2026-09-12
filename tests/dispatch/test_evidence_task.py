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
from bayesmith.artifacts.refusal import (
    CAPABILITY_UNAVAILABLE_R1,
    PREMISES,
    Refusal,
)
from bayesmith.artifacts.results import EvidenceComponent, EvidenceResult
from bayesmith.artifacts.tasks import (
    EvidenceTask,
    PosteriorTask,
    TaskKind,
    new_task_meta,
)
from bayesmith.dispatch.evidence import EVIDENCE_COMPONENT_NAMES, assemble_exact
from bayesmith.dispatch.task import SUPPORTED_TASK_KINDS
from tests.dispatch.test_task_protocol import model_ref as _model_ref

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


def _corner_divergent(*, collapsible):
    """``x``'s prior width is 1.0 above ``tau = 1.0`` and infinite at or below.

    ``tau ~ Normal(2.0, 0.5)``, and the package probes each outside latent at
    plus and minus one and three prior widths -- ``{0.5, 1.5, 2.5, 3.5}`` -- so
    the corner sits inside that range and nothing about the graph AT ITS PRIOR
    CENTRE says so: at ``tau = 2.0`` this is an ordinary straight line.

    ``collapsible`` decides which boundary the fault meets. Linear in ``x``, it
    joins the exact block and ``check_gaussian`` refuses it during compilation,
    for every task kind alike. Quadratic, it stays in the residual block, where
    no exact-block check reads it and the evidence premise is what answers.
    """

    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 6))
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        width = det("width", lambda t: jnp.where(t > 1.0, 1.0, jnp.inf), tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        if collapsible:
            mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        else:
            mu = det("mu", lambda x_, g_: x_ * x_ * g_, x, xs)
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

    return model


def _residual_corner_divergent():
    return _corner_divergent(collapsible=False)


def _exact_corner_divergent():
    return _corner_divergent(collapsible=True)


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
            model, _, _ = _model_factory()
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
        return outcome

    def test_a_model_with_a_sampled_block_reaches_the_capability_refusal(self):
        """R5 admits this class and has nothing to run it, and says which.

        R4 refused every model with a sampled block under
        ``evidence_residual_integral_required``. R5's gates admit it: the prior
        is proper, the conditional has no parent to degenerate over, and the
        whole graph is the residual problem. What stops it is the sampler that
        would run the integral, which is an optional extra and absent here --
        a statement about the RELEASE, not about the model.

        The distinction is the whole reason the two are different premises. A
        caller told "reduce your model to an exactly integrable block" would
        rewrite a model that is already fine.
        """
        with jax.enable_x64(True):
            observed = jnp.asarray([0.4, -0.2, 1.1])

            def model():
                tau = sample("tau", lambda: dist.HalfNormal(1.0))
                nu = det("nu", lambda t_: t_ * jnp.ones(3), tau)
                observe(
                    "e", lambda m: dist.Normal(m, 0.8).to_event(1), nu, obs=observed
                )

            graph = trace(model)
            refusal = self._refused(graph, CAPABILITY_UNAVAILABLE_R1)
            assert refusal.grounds[0].code == "residual_backend_unavailable"
            self._posterior_still_compiles(graph)

    def test_a_graph_with_no_latent_at_all_is_told_what_is_true_of_it(self):
        """The one graph still reaching ``evidence_residual_integral_required``.

        ``const``/``det``/``observe`` and nothing else. It traces, it plans, and
        R4 refused it saying "what is left over here needs a numerical integral
        over the residual problem" -- which is false about it: there is no
        integral, its p(d) is the likelihood's own normalising constant. After
        R5's widening it is the only graph reaching that premise, so the
        message is now about it.
        """
        with jax.enable_x64(True):

            def model():
                xs = const("X", jnp.linspace(1.0, 2.0, 5))
                mu = det("mu", lambda g_: 2.0 * g_, xs)
                observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(5))

            graph = trace(model)
            assert graph.latents == ()
            refusal = self._refused(graph, "evidence_residual_integral_required")
            message = refusal.grounds[0].message
            assert "no latent" in message
            assert "residual problem" not in message, (
                "this is the sentence R5 deleted: it named an integral this "
                "graph does not have, and after the widening no other graph "
                "reaches this premise for it to be true of"
            )
            outcome = self._posterior_still_compiles(graph)
            geometry = next(f for f in outcome.analysis.findings if f.code == "joint_geometry")
            assert geometry.conclusion == "not_applicable"
            assert dict(geometry.measurements)["reason"] == "no_latent_parameters"

    @pytest.mark.parametrize(
        "fixture, method",
        [("radiometer", "gcr+snis"), ("mixed_radiometer", "gcr+mh")],
    )
    def test_a_prediction_dependent_scale_is_refused_by_its_own_name(
        self, fixture, method
    ):
        """These blocks' residual factor is not the integral R5 runs.

        Refused rather than admitted, because that factor is an
        importance-weight normaliser: a different estimator, a different
        failure mode, and no oracle in this repository. Admitting it under
        R5's gate would put a number behind a gate that never tested it.

        **Both methods, because they arrive by different roads.**
        ``radiometer`` is ``gcr+snis`` with no sampled block at all;
        ``mixed_radiometer`` is ``gcr+mh`` WITH one, so it is an
        exact-plus-residual graph that the structure widening would otherwise
        have admitted. Testing only the first would leave the row that
        actually costs something uncovered.
        """
        from bayesmith.graph.reduction import as_graph
        from tests.exact import models

        with jax.enable_x64(True):
            graph = as_graph(getattr(models, fixture)())
            refusal = self._refused(graph, "evidence_residual_method_unsupported")
            assert method in refusal.grounds[0].message
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

    def test_repeating_an_exact_run_is_refused_rather_than_ignored(self):
        """Both options sit inside the TASK fingerprint slot, so a caller
        naming one gets a different digest and reasonably expects different
        behaviour. Silently ignoring it is the worst of the three available
        behaviours.

        Still true of an EXACT evidence, which is assembled once and
        reconstructs nothing. R5 narrowed the arm rather than deleting it.
        """
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

    def test_the_same_options_are_admitted_on_a_residual_evidence(self):
        """The narrowing, asserted by the thing it exists to permit.

        A residual evidence runs an estimator: repeated runs are how its own
        error bar is checked, and weighted draws are a required output rather
        than an extra. Refusing them there would leave later work unable to ask
        for what it needs -- and the module the refusal lives in is not one
        that later work edits.

        The task gets PAST the option arm, which is what is being tested. It
        then stops at the capability refusal, because no residual backend is
        installed; that is a different premise and the assertion says so rather
        than accepting any refusal at all.
        """
        with jax.enable_x64(True):
            from bayesmith.graph.reduction import as_graph
            from tests.exact import models

            graph = as_graph(models.diamond_ancestor())
            for task in (
                _evidence_task(repeat_count=3),
                _evidence_task(reconstruct_posterior=True),
            ):
                refusal = self._refused(graph, CAPABILITY_UNAVAILABLE_R1, task=task)
                assert refusal.failed_premise != "task_options_recognised"
            self._posterior_still_compiles(graph)


class TestThePremiseChainsOrder:
    """The order is asserted, because reading structure off the answer is not.

    Two premises can both hold and the earlier one answers; that is not a bug.
    Reading a graph's STRUCTURE off which premise it refused under is, and it
    is the specific mistake that would have passed before R5 widened anything:
    ``mixed_radiometer`` is a ``gcr+mh`` graph and R4 refused it under
    ``evidence_prior_proper``, so a test asserting "the method row is what
    stops it" would have been green for the wrong reason.
    """

    def _premise(self, graph, task=None):
        outcome = compile_task(graph, task or _evidence_task(), model_ref=_ref())
        return outcome.failed_premise if isinstance(outcome, Refusal) else None

    def test_x64_answers_before_anything_about_the_model(self):
        """First, and decided by OUTCOME rather than by reading a config flag.

        The graph below has an improper prior AND names an unread option, so
        two later premises are also false of it. Outside an x64 context the
        dtype answers.
        """
        basis = jnp.linspace(-1.0, 1.0, 6) + 0.3

        def model():
            w = sample("w", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
            b = const("basis", basis)
            mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
            observe("d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=basis)

        graph = trace(model)
        assert (
            self._premise(graph, _evidence_task(repeat_count=3))
            == "evidence_requires_x64"
        )
        with jax.enable_x64(True):
            assert self._premise(trace(model)) == "evidence_prior_proper"

    def test_the_prior_audit_answers_before_the_structure_gate(self):
        """``improper_outside_prior`` is an exact-plus-residual graph whose
        residual root has a genuinely improper prior. Both the prior premise
        and -- were the prior fixed -- the capability one are false of it, and
        the prior answers, because a model fault is named before a release
        limit."""
        with jax.enable_x64(True):
            from bayesmith.graph.reduction import as_graph
            from tests.exact import models

            graph = as_graph(models.improper_outside_prior())
            assert self._premise(graph) == "evidence_prior_proper"

    def test_the_method_row_answers_before_the_capability(self):
        """``mixed_radiometer`` is the case the plan warns about by name.

        It is ``gcr+mh`` WITH a sampled block, so before R5 it answered
        ``evidence_prior_proper`` -- a prior premise on a graph whose real
        problem is its method. The restatement makes its prior proper, and what
        answers now is the method row, ahead of the capability refusal that
        would otherwise claim it.
        """
        with jax.enable_x64(True):
            from bayesmith.dispatch.evidence import PriorVerdict, audit_graph_priors
            from bayesmith.graph.reduction import as_graph
            from tests.exact import models

            graph = as_graph(models.mixed_radiometer())
            assert all(
                audit.verdict is PriorVerdict.PROPER
                for audit in audit_graph_priors(graph)
            ), "its prior no longer stops it, which is what moves the answer"
            assert self._premise(graph) == "evidence_residual_method_unsupported"

    def test_a_model_fault_answers_before_the_missing_capability(self):
        """A conditional that degenerates inside the range is a statement about
        the MODEL, and it is named ahead of the missing sampler. A caller told
        "come back when the extra is installed" would install it and get the
        same wrong integral."""
        with jax.enable_x64(True):
            graph = trace(_residual_corner_divergent())
            assert self._premise(graph) == "evidence_conditional_prior_proper"
            # And the asymmetry, which is what makes it a boundary rather than
            # a regression: the same graph still plans a posterior.
            outcome = compile_task(
                graph, PosteriorTask(meta=new_task_meta(label="p")), model_ref=_ref()
            )
            assert not isinstance(outcome, Refusal)

    def test_an_exact_block_conditional_is_already_refused_structurally(self):
        """Where the same fault lands when the latent is collapsible, measured.

        The R5 plan's section 0.20 worries that the propriety restatement stops
        auditing the exact block's own priors, and rules that such a latent
        "still passes R4's one-dimensional rule". Measured, it did not: that
        rule short-circuits on ``node.parents``. But the hole it was worried
        about is closed anyway, and by something else -- ``check_gaussian``,
        which every exact-block member passes through, refuses a scale that is
        not strictly positive and finite at the same probe points this check
        would have used.

        So for an exact-block latent the two agree exactly and the earlier one
        answers. It answers as a raised ``StructureError`` rather than a
        Refusal, and symmetrically: **the posterior task raises too**, which is
        what makes it a graph-level structural refusal rather than an evidence
        boundary, and why Task 7 does not restate it as one.
        """
        from bayesmith.errors import StructureError

        with jax.enable_x64(True):
            graph = trace(_exact_corner_divergent())
            with pytest.raises(StructureError, match="strictly positive"):
                compile_task(graph, _evidence_task(), model_ref=_ref())
            with pytest.raises(StructureError, match="strictly positive"):
                compile_task(
                    graph,
                    PosteriorTask(meta=new_task_meta(label="p")),
                    model_ref=_ref(),
                )

    def test_every_refused_graph_still_compiles_a_posterior_task(self):
        """The asymmetry, over every premise this chain can produce at once.

        Asserting only that the evidence is refused would not tell a boundary
        apart from a regression wearing one.
        """
        from bayesmith.graph.reduction import as_graph
        from tests.exact import models

        seen: set[str] = set()
        with jax.enable_x64(True):
            for name in (
                "improper_outside_prior",
                "mixed_radiometer",
                "radiometer",
                "diamond_ancestor",
                "three_latent_chain",
            ):
                graph = as_graph(getattr(models, name)())
                premise = self._premise(graph)
                assert premise is not None, name
                seen.add(premise)
                outcome = compile_task(
                    graph,
                    PosteriorTask(meta=new_task_meta(label="p")),
                    model_ref=_ref(),
                )
                assert not isinstance(outcome, Refusal), (
                    f"{name}'s evidence refusal broke its posterior task"
                )
        assert len(seen) >= 3, (
            f"these five graphs are meant to spread across the chain and only "
            f"reached {sorted(seen)}"
        )


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


def test_every_premise_r4_added_is_reachable_through_the_public_seam():
    """A premise nothing can produce is a remedy nobody will read.

    ``evidence_base_measure_undeclared`` was added in R4 for a log-space graph
    whose change of variables is unrecorded, with a remedy and a place in the
    vocabulary -- and nothing could ever raise it. A graph reaches log space
    only through ``dispatch/factor.py``'s ``log-gcr`` route, and that method is
    already outside R4's admitted class, so the structure gate refuses it first.
    The premise was a second answer to a question that already had one.

    Both directions of the vocabulary were checked and neither noticed: it had
    a ``_REMEDIES`` row, so the symmetry test passed, and no fixture named it,
    so the produce-side test passed too. What neither asked was whether
    anything CAN name it.

    Scoped to R4's own premises. The older ones have their own coverage and
    this test is not the place to re-litigate them.
    """
    reachable: set[str] = set()
    with jax.enable_x64(True):
        basis = jnp.linspace(-1.0, 1.0, 6) + 0.3
        data = 1.2 * (basis * 0.9)

        def improper():
            w = sample("w", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
            b = const("basis", basis)
            mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
            observe("d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data)

        def unnormalised():
            w = sample(
                "w",
                lambda: dist.ImproperUniform(
                    dist.constraints.interval(-2.0, 5.0), (), ()
                ),
            )
            b = const("basis", basis)
            mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
            observe("d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data)

        def hierarchical():
            s = sample("s", lambda: dist.HalfNormal(1.0))
            w = sample("w", lambda s_: dist.Normal(0.0, s_), s)
            b = const("basis", basis)
            mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
            observe("d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data)

        def sampled_only():
            tau = sample("tau", lambda: dist.HalfNormal(1.0))
            nu = det("nu", lambda t_: t_ * jnp.ones(3), tau)
            observe(
                "e",
                lambda m: dist.Normal(m, 0.8).to_event(1),
                nu,
                obs=jnp.asarray([0.4, -0.2, 1.1]),
            )

        # R5's three. `no_latents` is the only graph still reaching
        # `evidence_residual_integral_required`; `prediction_dependent_sigma`
        # is the `gcr+snis` row; `corner_divergent` is a conditional that is a
        # density at the prior centre and not across the range the residual
        # integral walks -- the one this test would have called unreachable if
        # it had only looked at the centre, which is what the old audit did.
        def no_latents():
            b = const("basis", basis)
            mu = det("mu", lambda b_: b_ * 0.9, b)
            observe("d", lambda m: dist.Normal(m, SIGMA).to_event(1), mu, obs=data)

        def prediction_dependent_sigma():
            from tests.exact.models import radiometer

            return radiometer()

        def corner_divergent():
            # `mu` is QUADRATIC in `x` on purpose, so nothing is collapsed and
            # `x` stays in the residual block. Made linear, `x` joins the exact
            # block and `check_gaussian` raises a StructureError out of
            # `compile_plan` before any evidence premise runs -- which is a
            # graph-level refusal that breaks the posterior task too, and so is
            # a different boundary rather than this one.
            xs = const("X", jnp.linspace(1.0, 2.0, 6))
            tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
            width = det("width", lambda t: jnp.where(t > 1.0, 1.0, jnp.inf), tau)
            x = sample("x", lambda w: dist.Normal(0.0, w), width)
            mu = det("mu", lambda x_, g_: x_ * x_ * g_, x, xs)
            observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

        from bayesmith.graph.reduction import as_graph

        outcome = _compile(as_graph(prediction_dependent_sigma()))
        if isinstance(outcome, Refusal):
            reachable.add(outcome.failed_premise)

        for factory in (
            improper,
            unnormalised,
            hierarchical,
            sampled_only,
            no_latents,
            corner_divergent,
        ):
            outcome = _compile(trace(factory))
            if isinstance(outcome, Refusal):
                reachable.add(outcome.failed_premise)

        # A latent whose prior IS the graph-level reference prior. Its
        # node-level ImproperUniform is required by diagnose/priors.py, so the
        # improper arm must not claim it.
        from bayesmith.diagnose.priors import JeffreysPrior
        from bayesmith.graph.graph import Graph

        bare = trace(improper)
        covered = Graph(
            nodes=bare.nodes,
            plates=bare.plates,
            joint_prior=JeffreysPrior(over=("w",)),
        )
        outcome = _compile(covered)
        if isinstance(outcome, Refusal):
            reachable.add(outcome.failed_premise)

    # And the two that need something other than a graph.
    outcome = _compile(trace(_model_factory()[0]))  # float32, no x64 context
    if isinstance(outcome, Refusal):
        reachable.add(outcome.failed_premise)
    with jax.enable_x64(True):
        outcome = _compile(trace(_model_factory()[0]), _evidence_task(repeat_count=3))
        if isinstance(outcome, Refusal):
            reachable.add(outcome.failed_premise)

    evidence_premises = {p for p in PREMISES if p.startswith("evidence_")}
    unreachable = evidence_premises - reachable
    assert not unreachable, (
        f"evidence premises nothing in this test can produce: "
        f"{sorted(unreachable)}. Either a fixture is missing here or the "
        f"premise is a second answer to a question that already has one."
    )
