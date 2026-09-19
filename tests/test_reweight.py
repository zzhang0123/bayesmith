"""Prior recycling checked against conjugate integrals and analytic gradients."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import compile, observe, sample, trace
from bayesmith.reweight import PosteriorReweighting


def normal_logpdf(x, mean, scale):
    return -0.5 * ((x - mean) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)


def mixture_logpdf(x, location):
    # A normalized, genuinely non-Gaussian prior with a learnable location.
    return jnp.logaddexp(
        jnp.log(0.7) + normal_logpdf(x, location - 0.5, 0.7),
        jnp.log(0.3) + normal_logpdf(x, location + 1.0, 0.9),
    )


def reference_bank(n=12000):
    # Fixed stratified normal quantiles remove random test failures. The
    # independently integrated target is a Gaussian mixture convolved with
    # N(0, noise^2), not a second call to the reweighting implementation.
    observation, noise, prior_scale = 1.1, 0.6, 1.5
    variance = 1 / (1 / prior_scale**2 + 1 / noise**2)
    mean = variance * observation / noise**2
    quantiles = (jnp.arange(n) + 0.5) / n
    draws = mean + jnp.sqrt(variance) * jax.scipy.special.ndtri(quantiles)
    return draws, observation, noise, prior_scale


def problem_from_bank():
    draws, observation, noise, prior_scale = reference_bank()
    problem = PosteriorReweighting(
        draws,
        reference_log_density=lambda x: normal_logpdf(x, 0.0, prior_scale),
        target_log_density=lambda params, x: mixture_logpdf(x, params["location"]),
    )
    return problem, observation, noise, prior_scale


def analytic_log_evidence(location, observation, noise):
    return jnp.logaddexp(
        jnp.log(0.7)
        + normal_logpdf(observation, location - 0.5, jnp.sqrt(0.7**2 + noise**2)),
        jnp.log(0.3)
        + normal_logpdf(observation, location + 1.0, jnp.sqrt(0.9**2 + noise**2)),
    )


def test_identity_ratio_and_diagnostics():
    draws = jnp.array([-1.0, 0.0, 1.0, 2.0])
    problem = PosteriorReweighting(
        draws,
        reference_log_density=lambda x: normal_logpdf(x, 0.0, 1.0),
        target_log_density=lambda params, x: normal_logpdf(x, params, 1.0),
    )
    result = eqx.filter_jit(problem.estimate)(jnp.array(0.0))
    np.testing.assert_allclose(result.weights, 0.25)
    assert float(result.log_mean_weight) == pytest.approx(0.0, abs=1e-6)
    assert float(result.ess) == pytest.approx(4.0)
    assert float(result.ess_fraction) == pytest.approx(1.0)
    assert float(result.max_weight) == pytest.approx(0.25)


@pytest.mark.parametrize("location", [-0.4, 0.3, 1.0, 1.7])
def test_non_gaussian_evidence_ratio_and_gradient_against_closed_form(location):
    problem, observation, noise, prior_scale = problem_from_bank()
    point = {"location": jnp.array(location)}
    actual = problem.estimate(point).log_mean_weight
    expected = analytic_log_evidence(location, observation, noise) - normal_logpdf(
        observation, 0.0, jnp.sqrt(prior_scale**2 + noise**2)
    )
    assert float(actual) == pytest.approx(float(expected), abs=2e-3)
    derivative = jax.jit(jax.grad(problem.log_objective))(point)["location"]
    expected_derivative = jax.grad(analytic_log_evidence)(
        jnp.array(location), observation, noise
    )
    assert float(derivative) == pytest.approx(float(expected_derivative), abs=3e-3)


def test_fit_hyper_map_matches_independent_integrated_grid():
    problem, observation, noise, _ = problem_from_bank()
    hyperprior = lambda params: normal_logpdf(params["location"], 0.0, 1.2)
    result = problem.fit(
        {"location": jnp.array(-0.3)},
        log_hyperprior=hyperprior,
        steps=450,
        learning_rate=0.03,
        min_ess=1000,
    )
    grid = jnp.linspace(-1.0, 2.5, 1401)
    scores = analytic_log_evidence(grid, observation, noise) + normal_logpdf(
        grid, 0.0, 1.2
    )
    expected = float(grid[jnp.argmax(scores)])
    assert float(result.parameters["location"]) == pytest.approx(expected, abs=6e-3)
    assert float(result.log_objective) > float(result.history[0])
    assert float(result.gradient_norm) < 2e-3
    assert result.history.shape == (450,)
    assert float(result.log_objective) == pytest.approx(
        float(problem.log_objective(result.parameters, log_hyperprior=hyperprior)),
        abs=1e-6,
    )
    assert float(result.importance.ess) >= 1000


def test_prior_sampling_includes_likelihood_and_gives_absolute_evidence():
    n = 16000
    reference = dist.Normal(0.0, 1.5)
    draws = reference.icdf((jnp.arange(n) + 0.5) / n)
    problem = PosteriorReweighting(
        draws,
        reference_log_density=reference.log_prob,
        target_log_density=lambda location, x: (
            mixture_logpdf(x, location) + normal_logpdf(1.1, x, 0.6)
        ),
    )
    expected = analytic_log_evidence(0.4, 1.1, 0.6)
    assert float(problem.estimate(jnp.array(0.4)).log_mean_weight) == pytest.approx(
        float(expected), abs=1e-4
    )


def test_log_mean_is_not_mean_log_or_sum_of_normalized_weights():
    # log p_target(x)/p_reference(x) = location*x - location^2/2.
    problem = PosteriorReweighting(
        jnp.array([-1.0, 1.0]),
        reference_log_density=lambda x: normal_logpdf(x, 0.0, 1.0),
        target_log_density=lambda p, x: normal_logpdf(x, p, 1.0),
    )
    actual = float(problem.log_objective(jnp.array(1.0)))
    assert actual == pytest.approx(np.log(np.cosh(1.0)) - 0.5, abs=2e-6)
    assert abs(actual - (-0.5)) > 0.4


def test_field_event_and_dict_bank_reduce_only_the_sample_axis():
    samples = {"field": jnp.arange(24.0).reshape(4, 2, 3) / 10}
    ref = dist.Normal(jnp.zeros((2, 3)), 1.0).to_event(2)
    problem = PosteriorReweighting(
        samples,
        reference_log_density=lambda draw: ref.log_prob(draw["field"]),
        target_log_density=lambda loc, draw: (
            dist.Normal(jnp.full((2, 3), loc), 1.0).to_event(2).log_prob(draw["field"])
        ),
    )
    result = problem.estimate(jnp.array(0.0))
    assert result.weights.shape == (4,)
    assert float(result.ess) == pytest.approx(4.0)


def test_zero_weights_are_allowed_but_no_overlap_and_invalid_densities_fail():
    problem = PosteriorReweighting(
        jnp.array([-1.0, 1.0]),
        reference_log_density=lambda x: normal_logpdf(x, 0.0, 1.0),
        target_log_density=lambda p, x: jnp.where(x > 0, p, -jnp.inf),
    )
    np.testing.assert_allclose(problem.estimate(jnp.array(0.0)).weights, [0.0, 1.0])
    for invalid in [jnp.nan, jnp.inf, -jnp.inf]:
        with pytest.raises(eqx.EquinoxRuntimeError, match="invalid|overlap"):
            problem.estimate(jnp.array(invalid))


def test_stable_likelihood_offset_and_hyperprior_counted_once():
    def build(offset):
        return PosteriorReweighting(
            jnp.array([-1.0, 1.0]),
            reference_log_density=lambda x: normal_logpdf(x, 0.0, 1.0),
            target_log_density=lambda p, x: (
                normal_logpdf(x, p, 1.0) + normal_logpdf(0.3, x, 1.0) + offset
            ),
        )

    normal = build(0).estimate(jnp.array(0.2))
    shifted = build(800).estimate(jnp.array(0.2))
    assert float(shifted.log_mean_weight - normal.log_mean_weight) == pytest.approx(
        800, abs=1e-4
    )
    np.testing.assert_allclose(shifted.weights, normal.weights, atol=3e-5)
    score = build(0).log_objective(jnp.array(0.2), log_hyperprior=lambda p: -(p**2))
    assert float(score - normal.log_mean_weight) == pytest.approx(-0.04, abs=1e-6)


@pytest.mark.parametrize(
    "samples", [jnp.array([]), jnp.array(1.0), {}, {"a": jnp.ones(2), "b": jnp.ones(3)}]
)
def test_invalid_sample_banks_fail(samples):
    with pytest.raises(ValueError, match="sample|leading"):
        PosteriorReweighting(
            samples,
            reference_log_density=lambda x: 0.0,
            target_log_density=lambda p, x: 0.0,
        )


def test_non_scalar_prior_score_is_not_silently_summed():
    with pytest.raises(ValueError, match="scalar"):
        PosteriorReweighting(
            jnp.ones((3, 2)),
            reference_log_density=lambda x: -(x**2),
            target_log_density=lambda p, x: -jnp.sum(x**2),
        )


def test_explicit_ess_floor_refuses_degenerate_fit():
    problem = PosteriorReweighting(
        jnp.array([-1.0, 1.0]),
        reference_log_density=lambda x: normal_logpdf(x, 0.0, 1.0),
        target_log_density=lambda p, x: normal_logpdf(x, p, 1.0),
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="ESS"):
        problem.fit(jnp.array(4.0), steps=1, learning_rate=0.001, min_ess=1.5)


def graph_for(data=1.1, location=None, noise=0.6, name="field", mask=None):
    def model():
        if location is None:
            prior = dist.Normal(0.0, 1.5)
        else:
            prior = dist.MixtureSameFamily(
                dist.Categorical(probs=jnp.array([0.7, 0.3])),
                dist.Normal(
                    jnp.stack([location - 0.5, location + 1.0]), jnp.array([0.7, 0.9])
                ),
            )
        field = sample(name, lambda: prior)
        observe("data", lambda x: dist.Normal(x, noise), field, obs=data, mask=mask)

    return trace(model)


def test_full_graph_ratio_includes_hyperparameter_dependent_likelihood():
    draws, observation, noise, scale = reference_bank()
    reference = graph_for()
    target = lambda params: graph_for(
        location=params["location"], noise=jnp.exp(params["log_noise"])
    )
    problem = PosteriorReweighting.from_graphs(reference, {"field": draws}, target)
    point = {"location": jnp.array(0.4), "log_noise": jnp.log(0.7)}
    # Independent, expanded density expression: both changed terms matter.
    expected_weights = (
        mixture_logpdf(draws, 0.4)
        + normal_logpdf(observation, draws, 0.7)
        - normal_logpdf(draws, 0.0, scale)
        - normal_logpdf(observation, draws, noise)
    )
    np.testing.assert_allclose(problem.log_weights(point), expected_weights, atol=2e-6)
    actual = problem.estimate(point).log_mean_weight
    expected = analytic_log_evidence(0.4, observation, 0.7) - normal_logpdf(
        observation, 0.0, jnp.sqrt(scale**2 + noise**2)
    )
    assert float(actual) == pytest.approx(float(expected), abs=2e-3)


def test_gcr_bank_to_hyperparameter_map_end_to_end():
    reference = graph_for()
    plan = compile(reference)
    assert plan.exact.method == "gcr"
    assert plan.sampled is None
    assert not plan.sigma_needs_rebuild
    posterior = plan.sample(jax.random.key(24), num_samples=10000)
    assert posterior.log_weights is None
    problem = PosteriorReweighting.from_graphs(
        reference, posterior.samples, lambda p: graph_for(location=p["location"])
    )
    hyperprior = lambda p: normal_logpdf(p["location"], 0.0, 1.2)
    result = problem.fit(
        {"location": jnp.array(0.0)},
        log_hyperprior=hyperprior,
        steps=350,
        learning_rate=0.03,
    )
    grid = jnp.linspace(-1.0, 2.5, 1401)
    expected = float(
        grid[
            jnp.argmax(
                analytic_log_evidence(grid, 1.1, 0.6) + normal_logpdf(grid, 0.0, 1.2)
            )
        ]
    )
    # MC rather than arithmetic tolerance. The independent delta-method error
    # of the optimizing root is approximately sd[w * score] / sqrt(N) / |H|.
    # A fixed 0.04 band is a conservative recovery check, not a precision claim.
    assert float(result.parameters["location"]) == pytest.approx(expected, abs=0.04)
    assert float(result.importance.ess_fraction) > 0.5


@pytest.mark.parametrize("change", ["latents", "data", "mask"])
def test_graphs_cannot_silently_change_random_variables_or_observations(change):
    reference = graph_for()

    def target(p):
        return graph_for(
            location=p,
            name="other" if change == "latents" else "field",
            data=2.0 if change == "data" else 1.1,
            mask=True if change == "mask" else None,
        )

    problem = PosteriorReweighting.from_graphs(
        reference, {"field": jnp.zeros(4)}, target
    )
    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError), match="latents|observations"
    ):
        problem.estimate(jnp.array(0.0))


def test_bad_reference_and_target_scores_and_hyperprior_are_refused():
    with pytest.raises(eqx.EquinoxRuntimeError, match="reference density"):
        PosteriorReweighting(
            jnp.ones(2),
            reference_log_density=lambda x: -jnp.inf,
            target_log_density=lambda p, x: 0.0,
        )
    problem = PosteriorReweighting(
        jnp.ones(2),
        reference_log_density=lambda x: 0.0,
        target_log_density=lambda p, x: jnp.ones(2),
    )
    with pytest.raises(ValueError, match="scalar"):
        problem.estimate(jnp.array(0.0))
    valid, *_ = problem_from_bank()
    with pytest.raises(ValueError, match="scalar"):
        valid.log_objective(
            {"location": jnp.array(0.0)}, log_hyperprior=lambda p: jnp.ones(2)
        )
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite"):
        valid.fit(
            {"location": jnp.array(0.0)}, log_hyperprior=lambda p: -jnp.inf, steps=1
        )


def test_negative_edgeworth_correction_is_not_dropped_or_clipped():
    from bayesmith.distributions import EdgeworthExpansion

    problem = PosteriorReweighting(
        jnp.array([-3.0, 0.0, 3.0]),
        reference_log_density=dist.Normal(0.0, 1.0).log_prob,
        target_log_density=lambda k, x: EdgeworthExpansion(
            0.0, 1.0, cumulants=(k,), order=3
        ).log_prob(x),
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="invalid"):
        problem.estimate(jnp.array(1.0))


def test_jitted_invalid_target_cannot_bypass_runtime_guard():
    problem = PosteriorReweighting(
        jnp.ones(2),
        reference_log_density=lambda x: -(x**2),
        target_log_density=lambda p, x: jnp.log(p) - x**2,
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="invalid"):
        eqx.filter_jit(problem.estimate)(jnp.array(-1.0))


def test_min_ess_settings_and_empty_hyperparameters_are_refused():
    problem, *_ = problem_from_bank()
    for floor in [0, jnp.nan, 12001]:
        with pytest.raises(ValueError, match="min_ess"):
            problem.fit({"location": jnp.array(0.0)}, min_ess=floor)
    with pytest.raises(ValueError, match="hyperparameter"):
        problem.fit({})


def test_ess_refusal_survives_reading_only_the_fitted_score_under_jit():
    problem = PosteriorReweighting(
        jnp.array([-1.0, 1.0]),
        reference_log_density=lambda x: normal_logpdf(x, 0.0, 1.0),
        target_log_density=lambda p, x: normal_logpdf(x, p, 1.0),
    )
    score_only = eqx.filter_jit(
        lambda p: (
            problem.fit(p, steps=1, learning_rate=0.001, min_ess=1.5).log_objective
        )
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="ESS"):
        score_only(jnp.array(4.0))


def _prior_graph(distribution):
    def model():
        sample("x", lambda: distribution)

    return trace(model)


def test_graph_bank_and_target_must_preserve_latent_value_shape():
    scalar = _prior_graph(dist.Normal(0.0, 1.0))
    vector = _prior_graph(dist.Normal(jnp.zeros(2), 1.0))
    with pytest.raises(ValueError, match="shape"):
        PosteriorReweighting.from_graphs(
            scalar, {"x": jnp.zeros((3, 2))}, lambda _: scalar
        )
    problem = PosteriorReweighting.from_graphs(
        scalar, {"x": jnp.array([-1.0, 0.0, 1.0])}, lambda _: vector
    )
    with pytest.raises(ValueError, match="shape"):
        problem.estimate(0.0)


def test_graph_reweighting_accepts_same_coordinates_with_different_event_grouping():
    reference = _prior_graph(dist.Normal(jnp.zeros(2), 1.0))
    target = _prior_graph(dist.Normal(jnp.zeros(2), 1.0).to_event(1))
    problem = PosteriorReweighting.from_graphs(
        reference, {"x": jnp.array([[0.0, 0.2], [0.4, -0.1]])}, lambda _: target
    )
    np.testing.assert_allclose(problem.estimate(0.0).weights, [0.5, 0.5])


@pytest.mark.parametrize(
    "reference,target,bank",
    [
        (dist.Normal(0.0, 1.0), dist.Bernoulli(0.5), jnp.array([0.0, 1.0])),
        (dist.Normal(0.0, 1.0), dist.Delta(0.0), jnp.array([0.0, 1.0])),
        (
            dist.Normal(0.0, 1.0),
            dist.MixtureGeneral(
                dist.Categorical(jnp.array([0.5, 0.5])),
                [dist.Delta(0.0), dist.Delta(1.0)],
            ),
            jnp.array([0.0, 1.0]),
        ),
        (
            dist.Normal(jnp.zeros(3), 1.0).to_event(1),
            dist.Dirichlet(jnp.ones(3)),
            jnp.array([[0.2, 0.3, 0.5], [0.1, 0.3, 0.6]]),
        ),
    ],
)
def test_graph_reweighting_refuses_incompatible_measures(reference, target, bank):
    problem = PosteriorReweighting.from_graphs(
        _prior_graph(reference), {"x": bank}, lambda _: _prior_graph(target)
    )
    with pytest.raises(ValueError, match="measure"):
        problem.estimate(0.0)


def test_graph_reweighting_support_restriction_gives_zero_weights():
    problem = PosteriorReweighting.from_graphs(
        _prior_graph(dist.Normal(0.0, 1.0)),
        {"x": jnp.array([-1.0, 0.0, 1.0])},
        lambda _: _prior_graph(dist.Uniform(-0.5, 0.5)),
    )
    np.testing.assert_array_equal(problem.estimate(0.0).weights, [0.0, 1.0, 0.0])
    with pytest.raises(eqx.EquinoxRuntimeError, match="reference density"):
        PosteriorReweighting.from_graphs(
            _prior_graph(dist.Uniform(-0.5, 0.5)),
            {"x": jnp.array([-1.0, 0.0, 1.0])},
            lambda _: _prior_graph(dist.Normal(0.0, 1.0)),
        )


def test_graph_support_refusal_preserves_lognormal_target_gradients():
    problem = PosteriorReweighting.from_graphs(
        _prior_graph(dist.Normal(0.0, 1.0)),
        {"x": jnp.array([-1.0, 1.0, 2.0])},
        lambda location: _prior_graph(dist.LogNormal(location, 1.0)),
    )

    def oracle(location):
        values = jnp.array([1.0, 2.0])
        return jax.scipy.special.logsumexp(
            dist.LogNormal(location, 1.0).log_prob(values)
            - dist.Normal(0.0, 1.0).log_prob(values)
        ) - jnp.log(3.0)

    actual = jax.jit(jax.value_and_grad(problem.log_objective))(0.2)
    np.testing.assert_allclose(actual, jax.value_and_grad(oracle)(0.2), rtol=2e-6)


def test_graph_reweighting_checks_observed_support_and_honors_missing_data():
    def graph(bounded, mask):
        def model():
            sample("x", lambda: dist.Normal(0.0, 1.0))
            observe(
                "y",
                lambda: dist.Uniform(-1.0, 1.0) if bounded else dist.Normal(0.0, 1.0),
                obs=jnp.array([0.0, 2.0]),
                mask=mask,
            )

        return trace(model)

    for mask in (None, jnp.array([True, False])):
        problem = PosteriorReweighting.from_graphs(
            graph(False, mask),
            {"x": jnp.array([0.0, 1.0])},
            lambda _, mask=mask: graph(True, mask),
        )
        if mask is None:
            with pytest.raises(eqx.EquinoxRuntimeError, match="no overlap"):
                problem.estimate(0.0)
        else:
            expected = -jnp.log(2.0) - dist.Normal(0.0, 1.0).log_prob(0.0)
            # Four single-precision ulps cover the subtraction/logsumexp path;
            # counting the excluded datum changes the answer by order unity.
            np.testing.assert_allclose(
                problem.estimate(0.0).log_mean_weight,
                expected,
                rtol=0,
                atol=4 * jnp.finfo(expected.dtype).eps,
            )


def test_graph_reweighting_accepts_shared_plated_priors():
    from bayesmith import plate

    def model():
        members = plate("members", 2)
        sample("x", lambda: dist.Normal(0.0, 1.0), plate=members)

    graph = trace(model)
    problem = PosteriorReweighting.from_graphs(
        graph, {"x": jnp.array([[0.0, 0.2], [0.4, -0.1]])}, lambda _: graph
    )
    np.testing.assert_allclose(problem.estimate(0.0).weights, [0.5, 0.5])


def test_graph_reweighting_keeps_mixed_precision_density_scores():
    with jax.enable_x64():
        problem = PosteriorReweighting.from_graphs(
            _prior_graph(dist.Normal(0.0, 1.0)),
            {"x": jnp.array([-0.2, 0.3], dtype=jnp.float32)},
            lambda location: _prior_graph(dist.Normal(location, 1.0)),
        )
        result = problem.estimate(jnp.array(0.2, dtype=jnp.float64))
        assert result.log_mean_weight.dtype == jnp.float64
        assert jnp.isfinite(result.log_mean_weight)


def test_masked_out_of_support_placeholders_do_not_poison_density_gradients():
    from numpyro.infer.util import log_density

    from bayesmith import log_joint, to_numpyro

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda location: dist.LogNormal(location, 1.0),
            x,
            obs=jnp.array([1.0, -1.0]),
            mask=jnp.array([True, False]),
        )

    graph = trace(model)
    actual = jax.jit(jax.value_and_grad(lambda x: log_joint(graph, {"x": x})))(0.2)
    bridged = jax.value_and_grad(
        lambda x: log_density(to_numpyro(graph), (), {}, {"x": x})[0]
    )(0.2)
    oracle = jax.value_and_grad(
        lambda x: (
            dist.Normal(0.0, 1.0).log_prob(x) + dist.LogNormal(x, 1.0).log_prob(1.0)
        )
    )(0.2)
    np.testing.assert_allclose(actual, oracle, rtol=2e-6)
    np.testing.assert_allclose(bridged, oracle, rtol=2e-6)


@pytest.mark.parametrize(
    "distribution,bank",
    [
        (dist.Bernoulli(0.4), jnp.array([0, 1, 1])),
        (dist.Dirichlet(jnp.ones(3)), jnp.array([[0.2, 0.3, 0.5], [0.1, 0.3, 0.6]])),
    ],
)
def test_matching_discrete_and_simplex_measures_remain_supported(distribution, bank):
    graph = _prior_graph(distribution)
    result = PosteriorReweighting.from_graphs(
        graph, {"x": bank}, lambda _: graph
    ).estimate(0.0)
    np.testing.assert_allclose(result.log_mean_weight, 0.0, atol=1e-7)


def test_unknown_graph_support_requires_explicit_density_callbacks():
    class UnknownReal(dist.constraints.Constraint):
        def __call__(self, value):
            return jnp.isfinite(value)

        def tree_flatten(self):
            return (), ((), {})

    class CustomNormal(dist.Normal):
        support = UnknownReal()

    graph = _prior_graph(CustomNormal(0.0, 1.0))
    with pytest.raises(ValueError, match="explicit density callbacks"):
        PosteriorReweighting.from_graphs(
            graph, {"x": jnp.array([0.0, 1.0])}, lambda _: graph
        )


@pytest.mark.parametrize("observed", [jnp.asarray(0.0), jnp.zeros(2)])
def test_observation_coordinate_space_tracks_distribution_broadcasting(observed):
    def model(vector):
        sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y", lambda: dist.Normal(jnp.zeros(2) if vector else 0.0, 1.0), obs=observed
        )

    reference = trace(model, False)
    problem = PosteriorReweighting.from_graphs(
        reference, {"x": jnp.array([-0.2, 0.3])}, lambda _: trace(model, True)
    )
    if observed.shape == ():
        with pytest.raises(ValueError, match="coordinate measures"):
            problem.estimate(None)
    else:
        # Both condition on the same two measurements; only law batching changed.
        np.testing.assert_allclose(
            problem.estimate(None).log_mean_weight, 0.0, atol=0.0
        )


def test_plated_scalar_observation_requires_explicit_reweight_coordinates():
    from bayesmith import plate

    def model():
        sample("x", lambda: dist.Normal(0.0, 1.0))
        rows = plate("rows", 2)
        observe("y", lambda: dist.Normal(0.0, 1.0), obs=jnp.asarray(0.0), plate=rows)

    graph = trace(model)
    with pytest.raises(ValueError, match="explicit plate observation axis"):
        PosteriorReweighting.from_graphs(
            graph, {"x": jnp.array([-0.2, 0.3])}, lambda _: graph
        )
