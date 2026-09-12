"""Primal structure, independent of priors, annotations and sampler choice."""

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jax import lax
from jax.extend import core

from bayesmith import compile, det, observe, plate, sample, trace
from bayesmith.artifacts import DiagnosticPolicy, PosteriorTask, new_task_meta
from bayesmith.diagnose.structure import gaussian_flatness_certificate
from bayesmith.dispatch.plan import Block
from bayesmith.dispatch.preflight import analyze_preflight, likelihood_information
from bayesmith.errors import BayesmithError


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64():
        yield


def _findings(graph):
    runtime = compile(graph)
    return runtime, _runtime_findings(runtime)


def _runtime_findings(runtime):
    task = PosteriorTask(
        meta=new_task_meta(), diagnostics=DiagnosticPolicy(max_prior_parameters=0)
    )
    return {finding.code: finding for finding in analyze_preflight(runtime, task)}


def _one_block(graph, method="nuts"):
    return SimpleNamespace(
        graph=graph,
        blocks=(Block(graph.latents, method, "structural test"),),
        sampled=None,
        exact=None,
    )


def _scalar_graph(mean=lambda x: x, scale=lambda x: 0.3):
    def model():
        x = sample("x", lambda: dist.Uniform(-2.0, 2.0))
        mu = det("mu", mean, x, linear_in=("x",))
        observe("y", lambda m, v: dist.Normal(m, scale(v)), mu, x, obs=0.1)

    return trace(model)


def _certificate(graph, values=None, names=None):
    return gaussian_flatness_certificate(
        graph,
        graph.latents if names is None else names,
        {"x": jnp.array(0.2)} if values is None else values,
    )


@pytest.mark.parametrize("prior", ["normal", "uniform"])
def test_fixed_noise_affine_flatness_does_not_depend_on_prior_or_method(prior):
    def model():
        x = sample(
            "x",
            lambda: (
                dist.Normal(0.0, 1.0) if prior == "normal" else dist.Uniform(-2.0, 2.0)
            ),
        )
        observe("y", lambda v: dist.Normal(3.0 * v + 12.0, 0.2), x, obs=12.1)

    runtime, findings = _findings(trace(model))
    assert runtime.blocks[0].method == ("gcr" if prior == "normal" else "nuts")
    finding = findings["block_0_jeffreys"]
    assert finding.conclusion == "flat"
    measurements = dict(finding.measurements)
    assert measurements["global_flatness_proved"] is True
    assert dict(measurements["structural_certificate"])["certified"] is True
    assert measurements["prior_action"] == "unchanged"


def test_nested_jit_multidimensional_affine_array_operations_and_translation():
    @jax.jit
    def affine(x):
        pieces = jnp.concatenate((x[:1], x[1:]), axis=0)
        return jnp.transpose(pieces).reshape(-1) * jnp.arange(1.0, 7.0) + 40.0

    @jax.jit
    def mean(x):
        return affine(x).reshape(2, 3) + jnp.sum(x) / 7.0

    def model():
        x = sample("x", lambda: dist.Uniform(jnp.full((3, 2), -2.0), 2.0))
        observe(
            "y",
            lambda v: dist.Normal(mean(v), 0.3).to_event(2),
            x,
            obs=jnp.full((2, 3), 40.1),
        )

    graph = trace(model)
    certificate = _certificate(graph, {"x": jnp.full((3, 2), 0.1)})
    assert certificate["certified"] is True
    json.dumps(certificate)  # Report evidence carries no JAX objects.
    finding = _runtime_findings(_one_block(graph))["block_0_jeffreys"]
    assert finding.conclusion == "flat"
    assert dict(finding.measurements)["rank"] == 6


@pytest.mark.parametrize("nonlinear", [False, True])
def test_unstack_has_trusted_derivatives_without_certifying_nonlinearity(nonlinear):
    def mean(x):
        first, second = x
        return jnp.stack((first**2 if nonlinear else first, second))

    def model():
        x = sample("x", lambda: dist.Uniform(-jnp.ones(2), jnp.ones(2)))
        observe("y", lambda v: dist.Normal(mean(v), 0.2).to_event(1),
                x, obs=jnp.zeros(2))

    graph = trace(model)
    certificate = _certificate(graph, {"x": jnp.array([0.2, 0.3])})
    assert certificate["numerical_derivatives_trusted"] is True
    assert certificate["certified"] is (not nonlinear)
    finding = _runtime_findings(_one_block(graph))["block_0_jeffreys"]
    assert finding.conclusion == ("nonflat" if nonlinear else "flat")


def test_hierarchy_records_joint_sampling_and_uniform_sensitivity_scope():
    def model():
        scale = sample("scale", lambda: dist.Uniform(0.5, 2.0))
        instance = sample("instance", lambda a: dist.Normal(0.0, a), scale)
        observe("y", lambda s: dist.Normal(s, 0.1), instance, obs=0.2)

    graph = trace(model)
    runtime = _one_block(graph)
    task = PosteriorTask(meta=new_task_meta())
    findings = {f.code: f for f in analyze_preflight(runtime, task)}
    treatment = findings["latent_treatment"]
    assert treatment.conclusion == "passed"
    assert dict(treatment.measurements)["mode"] == "joint_sampling"
    assert dict(treatment.measurements)["conditional_density_factors"] == ("instance",)
    assert dict(treatment.measurements)["marginal_fisher_required"] is False
    sensitivity = findings["block_0_prior_sensitivity"]
    assert sensitivity.conclusion == "not_applicable"
    assert dict(sensitivity.measurements)["boundary_sensitivity"] == "not_assessed"


@pytest.mark.parametrize("stopped", [False, True])
def test_covariance_dependencies_survive_stop_gradient(stopped):
    def scale(x):
        value = jnp.exp(x)
        return lax.stop_gradient(value) if stopped else value

    graph = _scalar_graph(scale=scale)
    certificate = _certificate(graph)
    assert certificate["certified"] is False
    assert certificate["observations"][0]["covariance_independent"] is False
    # Even a falsely assigned GCR method must not promote numerical evidence.
    finding = _runtime_findings(_one_block(graph, "gcr"))["block_0_jeffreys"]
    assert finding.conclusion != "flat"
    assert dict(finding.measurements)["global_flatness_proved"] is False


def test_dense_covariance_rotation_cannot_hide_behind_constant_eigenvalues():
    def covariance(x):
        rotation = jnp.array([[jnp.cos(x), -jnp.sin(x)], [jnp.sin(x), jnp.cos(x)]])
        return rotation @ jnp.diag(jnp.array([1.0, 4.0])) @ rotation.T

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.MultivariateNormal(
                jnp.array([v, 0.0]), covariance_matrix=covariance(v)
            ),
            x,
            obs=jnp.zeros(2),
        )

    np.testing.assert_allclose(
        jnp.linalg.eigvalsh(covariance(0.2)), jnp.linalg.eigvalsh(covariance(0.7))
    )
    certificate = _certificate(trace(model))
    assert certificate["certified"] is False
    assert certificate["observations"][0]["covariance_independent"] is False


@pytest.mark.parametrize("mean", [lambda x: x**2, lambda x: jnp.sin(x)])
def test_dishonest_linearity_annotation_does_not_certify_nonlinear_mean(mean):
    certificate = _certificate(_scalar_graph(mean=mean))
    assert certificate["certified"] is False
    assert certificate["reason"] == "mean_not_certified_affine"


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_custom_derivative_cannot_certify_a_nonlinear_primal(mode):
    if mode == "jvp":

        @jax.custom_jvp
        def deceptive(x):
            return x**2

        @deceptive.defjvp
        def derivative(primals, tangents):
            return deceptive(primals[0]), tangents[0]
    else:

        @jax.custom_vjp
        def deceptive(x):
            return x**2

        deceptive.defvjp(lambda x: (x**2, None), lambda _, tangent: (tangent,))

    assert float(jax.grad(deceptive)(0.2)) == 1.0
    certificate = _certificate(_scalar_graph(mean=deceptive))
    assert certificate["certified"] is False
    assert certificate["reason"] == "unsupported_primal_or_derivative_semantics"


@pytest.mark.parametrize(
    "mean",
    [
        lambda x: x - lax.stop_gradient(x),
        lambda x: x + lax.stop_gradient(x**2),
    ],
)
def test_stopped_primal_is_not_treated_as_a_constant_by_ad(mean):
    graph = _scalar_graph(mean=mean)
    certificate = _certificate(graph)
    assert certificate["certified"] is False
    assert "stop_gradient" in certificate["observations"][0]["unsupported_primitives"]
    finding = _runtime_findings(_one_block(graph, "gcr"))["block_0_jeffreys"]
    assert finding.conclusion == "unresolved"


def test_outer_parameters_are_fixed_only_in_the_declared_conditional_scope():
    def model():
        x = sample("x", lambda: dist.Uniform(-2.0, 2.0))
        rate = sample("rate", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y", lambda v, r: dist.Normal(v * jnp.exp(r), jnp.exp(-r)), x, rate, obs=0.1
        )

    graph = trace(model)
    values = {"x": jnp.array(0.2), "rate": jnp.array(0.5)}
    conditional = _certificate(graph, values, ("x",))
    assert conditional["certified"] is True
    assert conditional["conditioned_on"] == ("rate",)
    assert _certificate(graph, values)["certified"] is False


@pytest.mark.parametrize("method", ["gcr", "nuts"])
def test_rank_deficiency_is_not_promoted_by_affine_structure(method):
    def model():
        x = sample("x", lambda: dist.Normal(jnp.zeros(2), 1.0))
        observe("y", lambda v: dist.Normal(jnp.sum(v), 0.3), x, obs=0.1)

    finding = _runtime_findings(_one_block(trace(model), method))["block_0_jeffreys"]
    measurements = dict(finding.measurements)
    assert finding.conclusion == "rank_deficient"
    assert measurements["rank"] == 1
    assert dict(measurements["structural_certificate"])["certified"] is True
    assert measurements["global_flatness_proved"] is False


def test_control_flow_matching_probes_remains_unresolved():
    graph = _scalar_graph(
        mean=lambda x: lax.cond(x > 0, lambda v: v, lambda v: v + 0.0, x)
    )
    certificate = _certificate(graph)
    assert certificate["certified"] is False
    finding = _runtime_findings(_one_block(graph, "gcr"))["block_0_jeffreys"]
    assert finding.conclusion == "unresolved"
    assert dict(finding.measurements)["half_logdet_change"] == 0.0


@pytest.mark.parametrize("kind", ["subclass", "mask", "instance"])
def test_noncanonical_density_semantics_are_refused(kind):
    class TemperedNormal(dist.Normal):
        def log_prob(self, value):
            return 2 * super().log_prob(value)

    def likelihood(x):
        if kind == "subclass":
            return TemperedNormal(x, 0.3)
        if kind == "mask":
            return dist.Normal(x, 0.3).mask(False)
        distribution = dist.Normal(x, 0.3)
        distribution.log_prob = lambda value: 2 * dist.Normal(x, 0.3).log_prob(value)
        return distribution

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("y", likelihood, x, obs=0.1)

    assert _certificate(trace(model))["certified"] is False


def test_custom_primitive_reusing_add_name_cannot_claim_add_semantics():
    fake_add = core.Primitive("add")
    fake_add.def_impl(lambda x: x**2)
    fake_add.def_abstract_eval(lambda x: x)
    certificate = _certificate(_scalar_graph(mean=lambda x: fake_add.bind(x)))
    assert certificate["certified"] is False
    assert certificate["reason"] == "unsupported_primal_or_derivative_semantics"


def test_block_dependent_index_is_not_an_affine_array_operation():
    graph = _scalar_graph(mean=lambda x: x + jnp.array([1.0, 2.0])[jnp.int32(x)])
    certificate = _certificate(graph)
    assert certificate["certified"] is False
    assert certificate["reason"] == "mean_not_certified_affine"


def test_observation_dependent_gaussian_parameters_are_not_certified():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        y = observe("y", lambda v: dist.Normal(v, 1.0), x, obs=0.2)
        observe("z", lambda v, u: dist.Normal(v * u, 1.0), x, y, obs=0.3)

    graph = trace(model)
    certificate = _certificate(graph)
    assert certificate["certified"] is False
    assert "observation_dependent_parameters" in certificate["detail"]
    with pytest.raises(BayesmithError, match="observation-dependent"):
        likelihood_information(graph, ("x",), {"x": jnp.array(0.2)})


def test_fixed_design_matrix_has_the_analytic_fisher_under_uniform_prior():
    design = jnp.array([[1.0, 2.0], [3.0, -1.0], [1.0, 1.0]])

    def model():
        x = sample("x", lambda: dist.Uniform(jnp.full(2, -2.0), 2.0))
        observe(
            "y", lambda v: dist.Normal(design @ v + 7.0, 0.5), x, obs=jnp.full(3, 7.1)
        )

    _, findings = _findings(trace(model))
    finding = findings["block_0_jeffreys"]
    assert finding.conclusion == "flat"
    measurements = dict(finding.measurements)
    expected = np.linalg.slogdet(4.0 * np.asarray(design.T @ design))[1] / 2
    assert measurements["half_logdet"] == pytest.approx(expected)


def test_constant_dense_covariance_certifies_structure_but_retains_numeric_limit():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.MultivariateNormal(
                jnp.array([v, 2 * v]),
                covariance_matrix=jnp.array([[1.0, 0.2], [0.2, 2.0]]),
            ),
            x,
            obs=jnp.zeros(2),
        )

    graph = trace(model)
    assert _certificate(graph)["certified"] is True
    finding = _runtime_findings(_one_block(graph))["block_0_jeffreys"]
    assert finding.conclusion == "unresolved"
    assert dict(finding.measurements)["global_flatness_proved"] is False


def test_density_override_on_a_canonical_wrapper_is_not_discarded():
    def likelihood(x):
        distribution = dist.Normal(jnp.array([x, x]), 0.3).to_event(1)
        distribution.log_prob = lambda value: jnp.sum(value**2)
        return distribution

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("y", likelihood, x, obs=jnp.zeros(2))

    assert _certificate(trace(model))["certified"] is False


@pytest.mark.parametrize("descriptor", ["row", "spectrum"])
def test_fixed_circulant_covariance_has_a_complete_constant_descriptor(descriptor):
    def likelihood(x):
        if descriptor == "row":
            return dist.CirculantNormal(
                jnp.full(3, x), covariance_row=jnp.array([2.0, 0.2, 0.2])
            )
        return dist.CirculantNormal(
            jnp.full(3, x), covariance_rfft=jnp.array([2.4, 1.8])
        )

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("y", likelihood, x, obs=jnp.zeros(3))

    assert _certificate(trace(model))["certified"] is True


def test_a_numerical_contradiction_is_retained_even_when_structure_certifies(
    monkeypatch,
):
    calls = iter((np.array([[1.0]]), np.array([[1.0]]), np.array([[4.0]])))
    monkeypatch.setattr(
        "bayesmith.dispatch.preflight.likelihood_information",
        lambda *args, **kwargs: next(calls),
    )
    finding = _runtime_findings(_one_block(_scalar_graph()))["block_0_jeffreys"]
    assert finding.conclusion == "nonflat"
    measurements = dict(finding.measurements)
    assert dict(measurements["structural_certificate"])["certified"] is True
    assert measurements["global_flatness_proved"] is False
    assert measurements["reason"] == "structural_numerical_contradiction"


@pytest.mark.parametrize("family", ["gaussian", "bernoulli"])
@pytest.mark.parametrize("derivative", ["zero", "varying", "stopped"])
def test_untrusted_derivatives_cannot_produce_definitive_numerical_findings(
    derivative, family
):
    @jax.custom_jvp
    def identity(x):
        return x

    @identity.defjvp
    def false_derivative(primals, tangents):
        multiplier = 0.0 if derivative == "zero" else jnp.exp(primals[0])
        return primals[0], multiplier * tangents[0]

    mean = lax.stop_gradient if derivative == "stopped" else identity
    if family == "gaussian":
        graph = _scalar_graph(mean=mean)
    else:

        def model():
            x = sample("x", lambda: dist.Uniform(-2.0, 2.0))
            observe("y", lambda v: dist.Bernoulli(logits=mean(v)), x, obs=1)

        graph = trace(model)
    findings = _runtime_findings(_one_block(graph, "gcr"))
    for name in ("joint_geometry", "block_0_jeffreys"):
        finding = findings[name]
        assert finding.conclusion == "unresolved"
        assert (
            dict(finding.measurements)["reason"] == "unsupported_derivative_semantics"
        )
    block = dict(findings["block_0_jeffreys"].measurements)
    assert block["rank"] == (1 if derivative == "varying" else 0)
    if derivative == "varying":
        assert block["half_logdet_change"] != 0.0


@pytest.mark.parametrize("plated", [False, True])
def test_gaussian_broadcast_size_is_counted_before_dense_information(
    monkeypatch, plated
):
    def model():
        rows = plate("rows", 3) if plated else None
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.Normal(v if plated else jnp.full(3, v), 1.0),
            x,
            obs=jnp.array(0.1),
            plate=rows,
        )

    def must_not_assemble(*args, **kwargs):
        raise AssertionError("expanded observations exceed the dense budget")

    monkeypatch.setattr(
        "bayesmith.dispatch.preflight.likelihood_information", must_not_assemble
    )
    task = PosteriorTask(
        meta=new_task_meta(),
        diagnostics=DiagnosticPolicy(
            max_matrix_elements=2,
            max_prior_parameters=0,
        ),
    )
    findings = {
        finding.code: finding
        for finding in analyze_preflight(_one_block(trace(model)), task)
    }
    assert findings["joint_geometry"].conclusion == "skipped_budget"
    assert findings["block_0_jeffreys"].conclusion == "skipped_budget"
