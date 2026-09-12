"""Proposal laws and target correction, checked independently of the builders."""

import importlib.util
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import det, observe, sample, trace


def api():
    assert importlib.util.find_spec("bayesmith.exact.proposals") is not None, (
        "the normalized proposal/MH implementation is missing"
    )
    from bayesmith.exact import proposals

    return proposals


def policy(names=("x",), method="gauss_newton", **options):
    defaults = {
        "steps": 1,
        "iterations": 3,
        "scale": 1.0,
        "damping": 1e-6,
        "max_parameters": 64,
        "max_matrix_elements": 200000,
        "mh_correction": True,
    }
    return SimpleNamespace(names=names, method=method, **(defaults | options))


def gaussian_graph(bounded=False, moving=False):
    def model():
        x = sample(
            "x", lambda: dist.Uniform(-1.0, 1.0) if bounded else dist.Normal(0.0, 2.0)
        )
        observe(
            "y",
            lambda v: dist.Normal(v, 0.4 + 0.3 * v**2 if moving else 0.5),
            x,
            obs=jnp.asarray(0.7),
        )

    return trace(model)


def test_normalized_density_and_sample_use_the_same_precision():
    p = api()
    mean = jnp.array([0.4, -0.7])
    precision = jnp.array([[2.0, 0.7], [0.7, 1.3]])
    q = p.DenseGaussianProposal(mean, jnp.linalg.cholesky(precision))
    reference = dist.MultivariateNormal(mean, precision_matrix=precision)
    points = jnp.array([[0.0, 0.0], [1.2, -0.5], [-2.0, 3.0]])
    np.testing.assert_allclose(
        jax.vmap(q.log_prob)(points), reference.log_prob(points), rtol=2e-6
    )
    draws = jax.jit(jax.vmap(q.sample))(jax.random.split(jax.random.key(0), 12000))
    np.testing.assert_allclose(np.mean(draws, axis=0), mean, atol=0.025)
    np.testing.assert_allclose(np.cov(draws.T), np.linalg.inv(precision), atol=0.035)


def test_asymmetric_forward_reverse_density_includes_both_normalizers():
    p = api()

    def build(x):
        return p.DenseGaussianProposal(0.6 * x + 0.2, jnp.diag(jnp.exp(0.4 * x)))

    x = jnp.array([0.8])
    key = jax.random.key(2)
    _, info = p.mh_transition(key, x, build, lambda v: -0.5 * jnp.sum(v * v))
    qx = build(x)
    y = qx.sample(jax.random.split(key)[0])
    qy = build(y)
    reference = (
        -0.5 * jnp.sum(y * y)
        + 0.5 * jnp.sum(x * x)
        + dist.MultivariateNormal(
            qy.mean, precision_matrix=qy.precision_cholesky @ qy.precision_cholesky.T
        ).log_prob(x)
        - dist.MultivariateNormal(
            qx.mean, precision_matrix=qx.precision_cholesky @ qx.precision_cholesky.T
        ).log_prob(y)
    )
    np.testing.assert_allclose(info.log_accept_ratio, reference, rtol=2e-6)
    assert (
        abs(
            float(
                jnp.sum(jnp.log(jnp.diag(qy.precision_cholesky)))
                - jnp.sum(jnp.log(jnp.diag(qx.precision_cholesky)))
            )
        )
        > 0.01
    )


def test_one_step_preserves_normal_target_and_moves_with_vmap_jit():
    p = api()

    def build(x):
        return p.DenseGaussianProposal(0.8 * x + 0.1, jnp.diag(jnp.exp(0.25 * x)))

    initial = jax.random.normal(jax.random.key(7), (16000, 1))
    keys = jax.random.split(jax.random.key(8), len(initial))
    actual, info = jax.jit(
        jax.vmap(
            lambda k, x: p.mh_transition(k, x, build, lambda v: -0.5 * jnp.sum(v * v))
        )
    )(keys, initial)
    assert abs(float(jnp.mean(actual))) < 0.025
    assert abs(float(jnp.mean(actual**2)) - 1.0) < 0.035
    assert 0.2 < float(jnp.mean(info.accepted)) < 0.95
    assert float(jnp.mean(jnp.abs(actual - initial))) > 0.2


@pytest.mark.parametrize("method", ["iterative_gls", "gauss_newton"])
def test_builder_matches_gaussian_normal_equations(method):
    p = api()
    compiled = p.compile_proposals(
        gaussian_graph(), (policy(method=method, damping=0.0),), {"x": jnp.asarray(0.2)}
    )
    q = compiled.blocks[0].build({"x": jnp.asarray(-0.8)})
    np.testing.assert_allclose(q.mean, [2.8 / 4.25], rtol=2e-6)
    np.testing.assert_allclose(q.precision_cholesky**2, [[4.25]], rtol=2e-6)
    assert compiled.names == ("x",)
    assert compiled.remainder == ()


def test_gauss_newton_uses_mean_minus_jacobian_times_current_point():
    p = api()

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 2.0))
        observe("y", lambda v: dist.Normal(v * v + 0.3, 0.5), x, obs=jnp.asarray(1.1))

    q = (
        p.compile_proposals(
            trace(model), (policy(iterations=1, damping=0.0),), {"x": jnp.asarray(0.8)}
        )
        .blocks[0]
        .build({"x": jnp.asarray(0.8)})
    )
    precision = 1.6**2 / 0.5**2 + 0.25
    mean = 1.6 * (1.1 - (0.8**2 + 0.3 - 1.6 * 0.8)) / 0.5**2 / precision
    np.testing.assert_allclose(q.mean, [mean], rtol=2e-6)


def test_uniform_proposals_reject_without_clipping_and_count_attempts():
    p = api()
    compiled = p.compile_proposals(
        gaussian_graph(bounded=True),
        (policy(scale=8.0, steps=5),),
        {"x": jnp.asarray(0.2)},
    )
    states, info = jax.jit(jax.vmap(compiled.sweep, in_axes=(0, None)))(
        jax.random.split(jax.random.key(3), 200), {"x": jnp.asarray(0.2)}
    )
    assert bool(jnp.all(jnp.abs(states["x"]) <= 1.0))
    assert bool(jnp.all(info.attempted == 5))
    assert int(jnp.sum(info.support_rejected)) > 100
    assert int(jnp.sum(info.numerical_failure)) == 0
    assert bool(jnp.any(states["x"] == jnp.asarray(0.2)))
    assert not bool(jnp.any(jnp.abs(states["x"]) == 1.0))


def test_parent_update_checks_support_of_unchanged_child():
    p = api()

    def model():
        x = sample("x", lambda: dist.Uniform(0.0, 2.0))
        sample("child", lambda v: dist.Uniform(0.0, v), x)
        observe("y", lambda v: dist.Normal(v, 0.2), x, obs=jnp.asarray(0.2))

    compiled = p.compile_proposals(
        trace(model), (policy(),), {"x": jnp.asarray(1.0), "child": jnp.asarray(0.9)}
    )
    states, info = jax.jit(jax.vmap(compiled.sweep, in_axes=(0, None)))(
        jax.random.split(jax.random.key(4), 100),
        {"x": jnp.asarray(1.0), "child": jnp.asarray(0.9)},
    )
    assert bool(jnp.all(states["x"] >= 0.9))
    assert int(jnp.sum(info.support_rejected)) > 90
    assert bool(jnp.all(states["child"] == 0.9))


@pytest.mark.parametrize("failure", ["mean", "metric", "reverse", "target"])
def test_nonfinite_proposals_preserve_current_state_bitwise(failure):
    p = api()
    x = jnp.asarray([-0.0, 0.5])

    def build(v):
        mean = jnp.full(2, jnp.nan) if failure == "mean" else jnp.zeros(2)
        metric = jnp.full((2, 2), jnp.nan) if failure == "metric" else jnp.eye(2)
        if failure == "reverse":
            mean = jnp.where(jnp.all(v == x), mean, jnp.full(2, jnp.nan))
        return p.DenseGaussianProposal(mean, metric)

    target = lambda v: (
        jnp.where(jnp.all(v == x), 0.0, jnp.nan)
        if failure == "target"
        else -jnp.sum(v * v)
    )
    actual, info = jax.jit(lambda k: p.mh_transition(k, x, build, target))(
        jax.random.key(9)
    )
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), np.asarray(x).view(np.uint32)
    )
    assert int(info.numerical_failure) == 1
    assert int(info.accepted) == 0


def test_bias_corrected_log_builder_sign_matches_closed_form():
    p = api()
    data = jnp.array([1.1, 1.2, 1.3])

    def model():
        x = sample("x", lambda: dist.Uniform(-3.0, 3.0))
        m = det("m", jnp.exp, x)
        observe("y", lambda v: dist.Normal(v, 0.04 * v), m, obs=data)

    compiled = p.compile_proposals(
        trace(model),
        (policy(method="bias_corrected_log_linear", damping=0.0),),
        {"x": jnp.asarray(0.0)},
    )
    q = compiled.blocks[0].build({"x": jnp.asarray(0.1)})
    np.testing.assert_allclose(
        q.mean, [jnp.mean(jnp.log(data)) + 0.04**2 / 2], rtol=2e-6
    )
    np.testing.assert_allclose(q.precision_cholesky**2, [[3 / 0.04**2]], rtol=2e-6)


@pytest.mark.parametrize("data,fractional", [(0.0, 0.04), (1.0, 0.3)])
def test_log_builder_refuses_bad_data_and_noise(data, fractional):
    p = api()

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.Normal(jnp.exp(v), fractional * jnp.exp(v)),
            x,
            obs=jnp.asarray(data),
        )

    with pytest.raises(ValueError, match="positive|fractional"):
        p.compile_proposals(
            trace(model),
            (policy(method="bias_corrected_log_linear"),),
            {"x": jnp.asarray(0.0)},
        )


def test_dense_budget_is_checked_before_building_a_matrix():
    with pytest.raises(ValueError, match="budget"):
        api().compile_proposals(
            gaussian_graph(), (policy(max_matrix_elements=0),), {"x": jnp.asarray(0.0)}
        )


def test_moving_noise_one_step_matches_independent_quadrature_and_moves():
    from scipy.integrate import cumulative_trapezoid, trapezoid

    p = api()
    graph = gaussian_graph(bounded=True, moving=True)
    grid = np.linspace(-1.0, 1.0, 16001)
    sigma = 0.4 + 0.3 * grid**2
    density = np.exp(-0.5 * ((0.7 - grid) / sigma) ** 2) / sigma
    density /= trapezoid(density, grid)
    cdf = cumulative_trapezoid(density, grid, initial=0.0)
    initial = jnp.asarray(
        np.interp(np.random.default_rng(49).uniform(size=24000), cdf, grid)
    )
    compiled = p.compile_proposals(
        graph, (policy(iterations=1),), {"x": jnp.asarray(0.2)}
    )
    result, info = jax.jit(jax.vmap(compiled.sweep))(
        jax.random.split(jax.random.key(23), len(initial)), {"x": initial}
    )
    for power in (1, 2):
        expected = trapezoid(grid**power * density, grid)
        assert abs(float(jnp.mean(result["x"] ** power)) - expected) < 0.012
    assert float(jnp.mean(info.accepted)) > 0.35
    assert float(jnp.mean(jnp.abs(result["x"] - initial))) > 0.1


def test_ordered_sweep_conditions_each_block_on_latest_other_blocks():
    from tests.dispatch.test_proposal_sampling import mixed_graph

    p = api()
    initial = {"x": jnp.asarray(0.1), "z": jnp.asarray(-0.3)}
    compiled = p.compile_proposals(
        mixed_graph(), (policy(), policy(names=("z",))), initial
    )
    key = jax.random.key(31)
    actual, _ = compiled.sweep(key, initial)
    first_key, second_key = jax.random.split(key)
    first, _ = compiled.blocks[0].step(jax.random.split(first_key, 1)[0], initial)
    expected, _ = compiled.blocks[1].step(jax.random.split(second_key, 1)[0], first)
    stale, _ = compiled.blocks[1].step(jax.random.split(second_key, 1)[0], initial)
    assert abs(float(expected["z"] - stale["z"])) > 0.01
    for name in initial:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_iterative_gls_final_metric_has_its_matching_final_solve():
    p = api()
    graph = gaussian_graph(moving=True)
    x = 0.2
    iterations = 2
    for _ in range(iterations):
        variance = (0.4 + 0.3 * x * x) ** 2
        x = 0.7 / variance / (1 / variance + 0.25)
    variance = (0.4 + 0.3 * x * x) ** 2
    precision = 1 / variance + 0.25
    expected = 0.7 / variance / precision
    compiled = p.compile_proposals(
        graph,
        (policy(method="iterative_gls", iterations=iterations, damping=0.0),),
        {"x": jnp.asarray(0.2)},
    )
    q = compiled.blocks[0].build({"x": jnp.asarray(0.2)})
    np.testing.assert_allclose(q.mean, [expected], rtol=2e-6)
    np.testing.assert_allclose(q.precision_cholesky**2, [[precision]], rtol=2e-6)


def test_dense_budget_precedes_log_fraction_jacobian(monkeypatch):
    p = api()

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.Normal(jnp.exp(v), 0.04 * jnp.exp(v)),
            x,
            obs=jnp.ones(5),
        )

    def forbidden(*args, **kwargs):
        pytest.fail("budget refusal came after allocating a Jacobian")

    monkeypatch.setattr(jax, "jacfwd", forbidden)
    with pytest.raises(ValueError, match="budget"):
        p.compile_proposals(
            trace(model),
            (policy(method="bias_corrected_log_linear", max_matrix_elements=1),),
            {"x": jnp.asarray(0.0)},
        )


def test_mixed_float_dtypes_preserve_site_dtypes_and_rejected_bytes():
    p = api()
    with jax.enable_x64():

        def model():
            a = sample("a", lambda: dist.Normal(jnp.float32(0.0), jnp.float32(1.0)))
            b = sample("b", lambda: dist.Normal(jnp.float64(0.0), jnp.float64(1.0)))
            observe(
                "y", lambda x, z: dist.Normal(x + z, 0.5), a, b, obs=jnp.asarray(0.4)
            )

        initial = {
            "a": jnp.asarray(0.1, dtype=jnp.float32),
            "b": jnp.asarray(0.2, dtype=jnp.float64),
        }
        compiled = p.compile_proposals(
            trace(model), (policy(names=("a", "b")),), initial
        )
        result, _ = jax.jit(compiled.sweep)(jax.random.key(6), initial)
        assert result["a"].dtype == initial["a"].dtype
        assert result["b"].dtype == initial["b"].dtype


def test_broadcast_scalar_data_uses_full_gaussian_batch_shape():
    p = api()

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y", lambda v: dist.Normal(v * jnp.ones(3), 0.5), x, obs=jnp.asarray(0.4)
        )

    compiled = p.compile_proposals(
        trace(model), (policy(damping=0.0),), {"x": jnp.asarray(0.1)}
    )
    q = compiled.blocks[0].build({"x": jnp.asarray(0.1)})
    np.testing.assert_allclose(q.mean, [4.8 / 13], rtol=2e-6)
    np.testing.assert_allclose(q.precision_cholesky**2, [[13.0]], rtol=2e-6)


@pytest.mark.parametrize("method", ["iterative_gls", "bias_corrected_log_linear"])
def test_affine_methods_refuse_nonlinear_mean_or_log_mean(method):
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        if method == "iterative_gls":
            observe("y", lambda v: dist.Normal(v * v, 0.3), x, obs=jnp.asarray(0.4))
        else:
            observe(
                "y",
                lambda v: dist.Normal(jnp.exp(v * v), 0.04 * jnp.exp(v * v)),
                x,
                obs=jnp.asarray(1.4),
            )

    with pytest.raises(ValueError, match="affine|Gauss|gauss"):
        api().compile_proposals(
            trace(model), (policy(method=method),), {"x": jnp.asarray(0.3)}
        )


@pytest.mark.parametrize("factor", ["joint", "evidence"])
def test_mh_target_retains_hierarchical_and_graph_level_factors(factor):
    from bayesmith.graph.graph import Graph

    p = api()

    class Joint:
        over = ("x",)

        def log_density(self, graph, values):
            return -0.3 * values["x"] ** 4

    class Evidence:
        over = ("x",)

        def log_density(self, graph, values):
            return -0.4 * (values["x"] + 0.8) ** 2

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        sample("child", lambda v: dist.Normal(v * v, 0.5), x)
        observe("y", lambda v: dist.Normal(v, 0.4), x, obs=jnp.asarray(0.3))

    plain = trace(model)
    graph = Graph(
        nodes=plain.nodes,
        plates=plain.plates,
        joint_prior=Joint() if factor == "joint" else None,
        evidence_terms=(Evidence(),) if factor == "evidence" else (),
    )
    state = {"x": jnp.asarray(0.2), "child": jnp.asarray(0.8)}
    block = p.compile_proposals(graph, (policy(),), state).blocks[0]
    key = jax.random.key(27)
    _, info = block.step(key, state)
    qx = block.build(state)
    y = qx.sample(jax.random.split(key)[0])
    qy = block.build({**state, "x": y[0]})

    def exact(v):
        return (
            -0.5 * v * v
            - 0.5 * ((0.8 - v * v) / 0.5) ** 2
            - 0.5 * ((0.3 - v) / 0.4) ** 2
            + (-0.3 * v**4 if factor == "joint" else -0.4 * (v + 0.8) ** 2)
        )

    expected = (
        exact(y[0])
        - exact(state["x"])
        + qy.log_prob(jnp.array([state["x"]]))
        - qx.log_prob(y)
    )
    np.testing.assert_allclose(info.log_accept_ratio, expected, rtol=3e-6, atol=3e-6)


def test_scalar_plated_observation_refused_before_different_targets_can_mix():
    from bayesmith import plate

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        rows = plate("rows", 3)
        observe("y", lambda v: dist.Normal(v, 0.5), x, obs=jnp.asarray(0.4), plate=rows)

    with pytest.raises(ValueError, match="plate.*target|target.*plate"):
        api().compile_proposals(trace(model), (policy(),), {"x": jnp.asarray(0.1)})


def test_declared_latent_shape_is_required_for_fixed_flatten_metadata():
    with pytest.raises(ValueError, match="initial.*shape|shape.*initial"):
        api().compile_proposals(
            gaussian_graph(), (policy(),), {"x": jnp.array([0.1, 0.2])}
        )


@pytest.mark.parametrize("kind", ["discrete", "complex", "complex_real_start"])
def test_unsupported_nuts_remainder_is_refused_before_execution(kind):
    from bayesmith.distributions import ComplexNormal

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        sample(
            "other",
            lambda: (
                dist.Bernoulli(0.5) if kind == "discrete" else ComplexNormal(0j, 1.0)
            ),
        )
        observe("y", lambda v: dist.Normal(v, 0.5), x, obs=jnp.asarray(0.4))

    initial = {
        "x": jnp.asarray(0.1),
        "other": jnp.asarray(0.2j if kind == "complex" else 0.0),
    }
    with pytest.raises(ValueError, match="remainder|real floating|discrete"):
        api().compile_proposals(trace(model), (policy(),), initial)


def test_normal_scale_batch_broadcast_is_preserved_in_the_proposal_law():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "y",
            lambda v: dist.Normal(v, jnp.array([0.5, 1.0, 2.0])),
            x,
            obs=jnp.asarray(0.4),
        )

    block = (
        api()
        .compile_proposals(
            trace(model), (policy(damping=0.0),), {"x": jnp.asarray(0.1)}
        )
        .blocks[0]
    )
    q = block.build({"x": jnp.asarray(0.1)})
    precision = 1.0 + 4.0 + 1.0 + 0.25
    np.testing.assert_allclose(
        q.mean, [0.4 * (4.0 + 1.0 + 0.25) / precision], rtol=2e-6
    )
    np.testing.assert_allclose(q.precision_cholesky**2, [[precision]], rtol=2e-6)


def test_default_mh_is_bitwise_identical_to_explicitly_enabled_mh():
    p = api()
    current = jnp.asarray([0.3])
    build = lambda x: p.DenseGaussianProposal(0.7 * x + 0.1, jnp.diag(jnp.exp(0.2 * x)))
    target = lambda x: -0.5 * jnp.sum(x * x)
    key = jax.random.key(104)
    default = p.mh_transition(key, current, build, target)
    enabled = p.mh_transition(key, current, build, target, mh_correction=True)
    for a, b in zip(jax.tree.leaves(default), jax.tree.leaves(enabled), strict=True):
        np.testing.assert_array_equal(a, b)


def test_mh_off_applies_forward_draw_without_reverse_density_or_acceptance_test():
    p = api()
    current = jnp.asarray([0.3])

    def build(x):
        mean = jnp.where(jnp.all(x == current), jnp.array([4.0]), jnp.array([jnp.nan]))
        return p.DenseGaussianProposal(mean, jnp.eye(1))

    key = jax.random.key(108)
    result, info = jax.jit(
        lambda k: p.mh_transition(
            k, current, build, lambda x: -100 * jnp.sum(x * x), mh_correction=False
        )
    )(key)
    expected = build(current).sample(jax.random.split(key)[0])
    np.testing.assert_array_equal(result, expected)
    assert int(info.accepted) == int(info.attempted) == 1
    assert int(info.numerical_failure) == 0
    assert bool(jnp.isnan(info.log_accept_ratio))


@pytest.mark.parametrize("failure", ["mean", "metric", "target", "support"])
def test_mh_off_still_rejects_invalid_proposals_without_clipping_or_retry(failure):
    p = api()
    current = jnp.asarray([-0.0, 0.5])

    def build(x):
        mean = jnp.full(2, jnp.nan) if failure == "mean" else jnp.zeros(2)
        metric = jnp.full((2, 2), jnp.nan) if failure == "metric" else jnp.eye(2)
        return p.DenseGaussianProposal(mean, metric)

    target = lambda x: jnp.asarray(jnp.nan) if failure == "target" else -jnp.sum(x * x)
    support = lambda x: jnp.asarray(failure != "support")
    result, info = jax.jit(
        lambda k: p.mh_transition(
            k, current, build, target, support, mh_correction=False
        )
    )(jax.random.key(112))
    np.testing.assert_array_equal(
        np.asarray(result).view(np.uint32), np.asarray(current).view(np.uint32)
    )
    assert int(info.attempted) == 1
    assert int(info.accepted) == 0
    assert int(info.numerical_failure) == int(failure != "support")
    assert int(info.support_rejected) == int(failure == "support")
