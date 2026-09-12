"""Independent likelihood-information oracles and the scope of their findings."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import compile, const, det, observe, plate, sample, trace
from bayesmith.artifacts import DiagnosticPolicy, PosteriorTask, new_task_meta
from bayesmith.dispatch.preflight import analyze_preflight, likelihood_information
from bayesmith.errors import BayesmithError


@pytest.fixture(autouse=True)
def double_precision():
    with jax.enable_x64():
        yield


def findings_for(graph):
    runtime = compile(graph)
    task = PosteriorTask(
        meta=new_task_meta(), diagnostics=DiagnosticPolicy(max_prior_parameters=0)
    )
    return runtime, {f.code: f for f in analyze_preflight(runtime, task)}


@pytest.mark.parametrize("parameterization", ["logits", "probs"])
@pytest.mark.parametrize("event", [False, True])
def test_bernoulli_information_matches_analytic_coordinate_order(
    parameterization, event
):
    design = np.array([[1.0, -0.5], [0.3, 1.2], [-0.7, 2.0]])

    def likelihood(coef, bias):
        eta = design @ coef + bias**2
        distribution = (
            dist.Bernoulli(logits=eta)
            if parameterization == "logits"
            else dist.Bernoulli(probs=jax.nn.sigmoid(eta))
        )
        return distribution.to_event(1) if event else distribution

    def model():
        coef = sample("a_coef", lambda: dist.Normal(jnp.zeros(2), 1.0))
        bias = sample("z_bias", lambda: dist.Normal(0.0, 1.0))
        observe("y", likelihood, coef, bias, obs=jnp.array([0, 1, 1]))

    values = {"a_coef": jnp.array([0.2, -0.4]), "z_bias": jnp.array(0.7)}
    eta = design @ np.asarray(values["a_coef"]) + 0.7**2
    probability = 1 / (1 + np.exp(-eta))
    jacobian = np.column_stack((np.full(3, 1.4), design))
    expected = jacobian.T @ ((probability * (1 - probability))[:, None] * jacobian)
    actual = likelihood_information(trace(model), ("z_bias", "a_coef"), values)
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-15)


@pytest.mark.parametrize("expanded", [False, True])
def test_scalar_bernoulli_broadcast_and_mask_count_actual_observations(expanded):
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: (
                dist.Bernoulli(logits=v).expand((4,))
                if expanded
                else dist.Bernoulli(logits=v)
            ),
            x,
            obs=jnp.array([0, 1, 1, 0]),
            mask=jnp.array([True, False, True, True]),
        )

    actual = likelihood_information(trace(model), ("x",), {"x": jnp.array(0.0)})
    np.testing.assert_allclose(actual, [[3 / 4]], rtol=1e-14)


@pytest.mark.parametrize("shared", [False, True])
def test_bernoulli_plates_keep_mapped_and_broadcast_parameters(shared):
    def model():
        rows = plate("rows", 3)
        x = sample("x", lambda: dist.Normal(0.0, 1.0), plate=None if shared else rows)
        observe(
            "y",
            lambda v: dist.Bernoulli(logits=2 * v),
            x,
            obs=jnp.array([0, 1, 0]),
            plate=rows,
        )

    value = jnp.array(0.0) if shared else jnp.zeros(3)
    expected = np.array([[3.0]]) if shared else np.eye(3)
    np.testing.assert_allclose(
        likelihood_information(trace(model), ("x",), {"x": value}), expected, rtol=1e-14
    )


@pytest.mark.parametrize("event_size", [None, 2])
def test_scalar_observation_plate_uses_bridge_batch_and_event_shape(event_size):
    from numpyro import handlers

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    def likelihood(value):
        if event_size is None:
            return dist.Bernoulli(logits=value)
        return dist.Bernoulli(logits=jnp.full(event_size, value)).to_event(1)

    def model():
        rows = plate("rows", 3)
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("y", likelihood, x, obs=jnp.array(0), plate=rows)

    graph = trace(model)
    values = {"x": jnp.array(0.0)}
    conditioned = handlers.condition(to_numpyro(graph), data=values)
    sites = handlers.trace(handlers.seed(conditioned, rng_seed=0)).get_trace()
    distribution = sites["y"]["fn"]
    assert distribution.batch_shape == (3,)
    assert distribution.event_shape == (() if event_size is None else (event_size,))
    expected = 3 * (event_size or 1) / 4
    np.testing.assert_allclose(
        likelihood_information(graph, ("x",), values), [[expected]], rtol=1e-14
    )


def test_bernoulli_joint_geometry_does_not_require_location_family():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("y", lambda v: dist.Bernoulli(logits=v), x, obs=jnp.array([0, 1]))

    _, findings = findings_for(trace(model))
    assert findings["joint_geometry"].conclusion == "passed"
    assert dict(findings["joint_geometry"].measurements)["mean_rank"] is None
    assert findings["block_0_jeffreys"].conclusion == "nonflat"


@pytest.mark.parametrize("plated", [False, True])
def test_scalar_observation_expansion_counts_toward_diagnostic_budget(
    monkeypatch, plated
):
    from types import SimpleNamespace

    from bayesmith.dispatch.plan import Block

    def model():
        rows = plate("rows", 3) if plated else None
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.Bernoulli(logits=v if plated else jnp.full(3, v)),
            x,
            obs=jnp.array(0),
            plate=rows,
        )

    def must_not_assemble(*args, **kwargs):
        raise AssertionError("expanded plate exceeds the matrix budget")

    monkeypatch.setattr(
        "bayesmith.dispatch.preflight.likelihood_information", must_not_assemble
    )
    # Exercise the diagnostic budget independently of the streaming classifier.
    runtime = SimpleNamespace(
        graph=trace(model),
        blocks=(Block(("x",), "nuts", "budget diagnostic fixture"),),
        sampled=None,
        exact=None,
    )
    task = PosteriorTask(
        meta=new_task_meta(),
        diagnostics=DiagnosticPolicy(max_matrix_elements=2, max_prior_parameters=0),
    )
    findings = {f.code: f for f in analyze_preflight(runtime, task)}
    assert findings["joint_geometry"].conclusion == "skipped_budget"
    assert findings["block_0_jeffreys"].conclusion == "skipped_budget"


def test_gcr_outside_mean_parameter_does_not_erase_conditional_flatness():
    def model():
        time = const("time", jnp.linspace(0.0, 2.0, 8))
        rate = sample("rate", lambda: dist.LogNormal(0.0, 0.5))
        a = sample("a", lambda: dist.Normal(0.0, 1.0))
        basis = det("basis", lambda t, r: jnp.exp(-t * r), time, rate)
        mu = det("mu", lambda x, b: x * b, a, basis, linear_in=("a",))
        observe("y", lambda m: dist.Normal(m, 0.2), mu, obs=jnp.ones(8))

    runtime, findings = findings_for(trace(model))
    assert runtime.sigma_needs_rebuild
    index = next(i for i, b in enumerate(runtime.blocks) if b.method == "gcr")
    finding = findings[f"block_{index}_jeffreys"]
    assert finding.conclusion == "flat"
    measurements = dict(finding.measurements)
    assert (
        measurements["flatness_evidence"] == "primal_jaxpr_affinity_and_covariance_independence"
    )
    assert measurements["global_flatness_proved"] is True


def test_parameter_dependent_covariance_is_not_flat():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.Normal(v, jnp.exp(v)),
            x,
            obs=jnp.array([0.1, 0.2, -0.2]),
            depends_on_prediction=False,
        )

    graph = trace(model)
    point = jnp.array(0.4)
    expected = 3 * (np.exp(-0.8) + 2)
    np.testing.assert_allclose(
        likelihood_information(graph, ("x",), {"x": point}), [[expected]], rtol=1e-13
    )
    _, findings = findings_for(graph)
    assert findings["block_0_jeffreys"].conclusion == "nonflat"


def test_hierarchical_rank_deficiency_is_scoped_to_direct_observations():
    def model():
        scale = sample("scale", lambda: dist.LogNormal(0.0, 0.5))
        process = sample("process", lambda s: dist.Normal(jnp.zeros(3), s), scale)
        observe("y", lambda p: dist.Normal(p, 0.2), process, obs=jnp.ones(3))

    _, findings = findings_for(trace(model))
    joint = findings["joint_geometry"]
    assert joint.conclusion == "rank_deficient"
    measurements = dict(joint.measurements)
    assert measurements["information_scope"] == "direct_observed_likelihood"
    assert measurements["hierarchical_latent_factors"] == ("process",)
    assert set(measurements["latent_density_factors_excluded"]) == {"scale", "process"}
    assert measurements["marginal_information"] == "not_assessed"
    assert measurements["posterior_identifiability"] == "not_assessed"


@pytest.mark.parametrize(
    "family", ["poisson", "masked", "event_mask", "subclass", "mixed"]
)
def test_unsupported_likelihood_semantics_are_explicitly_refused(family):
    class ModifiedBernoulli(dist.BernoulliLogits):
        def log_prob(self, value):
            return 2 * super().log_prob(value)

    def likelihood(v):
        if family == "poisson":
            return dist.Poisson(jnp.exp(v))
        if family == "masked":
            return dist.Bernoulli(logits=v).mask(False)
        if family == "subclass":
            return ModifiedBernoulli(v)
        if family == "event_mask":
            return dist.Bernoulli(logits=jnp.full(2, v)).to_event(1)
        return dist.Bernoulli(logits=v)

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            likelihood,
            x,
            obs=jnp.array([0, 1]),
            mask=jnp.array([True, False]) if family == "event_mask" else None,
        )
        if family == "mixed":
            observe("other", lambda v: dist.Normal(v, 1.0), x, obs=jnp.array(0.0))

    with pytest.raises(BayesmithError):
        likelihood_information(trace(model), ("x",), {"x": jnp.array(0.1)})


def test_uniform_prior_and_nuts_do_not_hide_linear_gaussian_structure():
    def model():
        x = sample("x", lambda: dist.Uniform(-0.5, 0.5))
        # The primal graph proves this for all x, independently of the prior
        # and the sampler selected to respect its finite support.
        observe("y", lambda v: dist.Normal(v, 0.2), x, obs=jnp.array(0.1))

    runtime, findings = findings_for(trace(model))
    assert runtime.blocks[0].method == "nuts"
    assert findings["block_0_jeffreys"].conclusion == "flat"
    assert dict(findings["block_0_jeffreys"].measurements)["global_flatness_proved"]


@pytest.mark.parametrize("value", [-40.0, 40.0])
def test_extreme_finite_logits_retain_small_expected_information(value):
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("y", lambda v: dist.Bernoulli(logits=v), x, obs=jnp.array(0))

    expected = np.exp(-abs(value)) / (1 + np.exp(-abs(value))) ** 2
    actual = likelihood_information(trace(model), ("x",), {"x": jnp.array(value)})
    np.testing.assert_allclose(actual, [[expected]], rtol=1e-14, atol=0)


@pytest.mark.parametrize("probability", [1e-20, 1 - 4 * np.finfo(float).eps])
def test_probability_parameterization_matches_density_inside_clipping_interval(
    probability,
):
    def model():
        p = sample("p", lambda: dist.Uniform(0.0, 1.0))
        observe("y", lambda v: dist.Bernoulli(probs=v), p, obs=jnp.array(0))

    actual = likelihood_information(trace(model), ("p",), {"p": jnp.array(probability)})
    np.testing.assert_allclose(
        actual, [[1 / (probability * (1 - probability))]], rtol=1e-14, atol=0
    )


@pytest.mark.parametrize(
    "probability",
    [
        np.finfo(float).tiny / 2,
        np.finfo(float).tiny,
        1 - np.finfo(float).eps,
        np.nextafter(1.0, 0.0),
    ],
)
def test_probability_density_clipping_and_kinks_are_explicitly_refused(probability):
    def model():
        p = sample("p", lambda: dist.Uniform(0.0, 1.0))
        observe("y", lambda v: dist.Bernoulli(probs=v), p, obs=jnp.array(0))

    if probability > 1 - np.finfo(float).eps:
        # The installed density is locally constant here: the unconstrained
        # Bernoulli oracle 1/(p*(1-p)) would report false information.
        score = jax.grad(lambda p: dist.Bernoulli(probs=p).log_prob(0.0))(probability)
        assert float(score) == 0.0
    with pytest.raises(BayesmithError, match="clipping"):
        likelihood_information(trace(model), ("p",), {"p": jnp.array(probability)})


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_boundary_probabilities_are_not_regular_information(probability):
    def model():
        p = sample("p", lambda: dist.Uniform(0.0, 1.0))
        observe("y", lambda v: dist.Bernoulli(probs=v), p, obs=jnp.array(0))

    with pytest.raises(BayesmithError, match="boundary/invalid probabilities"):
        likelihood_information(trace(model), ("p",), {"p": jnp.array(probability)})


def test_multiple_observations_and_an_empty_mask_add_the_right_information():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("first", lambda v: dist.Bernoulli(logits=v), x, obs=jnp.array([0, 1]))
        observe("second", lambda v: dist.Bernoulli(logits=2 * v), x, obs=jnp.array(1))
        observe(
            "missing",
            lambda v: dist.Bernoulli(logits=4 * v),
            x,
            obs=jnp.array([1, 1]),
            mask=jnp.array([False, False]),
        )

    actual = likelihood_information(trace(model), ("x",), {"x": jnp.array(0.0)})
    np.testing.assert_allclose(actual, [[1.5]], rtol=1e-14)


def test_observation_dependent_logits_refuse_an_unperformed_expectation():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        first = observe(
            "first", lambda v: dist.Bernoulli(logits=v), x, obs=jnp.array(0)
        )
        observe(
            "second",
            lambda v, y: dist.Bernoulli(logits=v * y),
            x,
            first,
            obs=jnp.array(1),
        )

    with pytest.raises(BayesmithError, match="observation-dependent logits"):
        likelihood_information(trace(model), ("x",), {"x": jnp.array(0.3)})
