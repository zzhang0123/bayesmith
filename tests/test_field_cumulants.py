"""Field densities must retain mixed cumulants and coordinate invariance."""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import ReshapeTransform
from numpyro.infer.util import log_density

from bayesmith import log_joint, observe, sample, to_numpyro, trace
from bayesmith.cumulants import (
    DenseCumulants,
    FieldEdgeworth,
    LowRankCumulants,
    PeriodicNormal,
    low_rank_from_fourier,
)
from bayesmith.distributions import EdgeworthExpansion


def _wick_hermite(indices, h, q):
    """Host-side Wick pairing oracle, independent of projected recurrence."""
    if not indices:
        return 1.0
    i, rest = indices[0], indices[1:]
    return h[i] * _wick_hermite(rest, h, q) - sum(
        q[i, j] * _wick_hermite(rest[:p] + rest[p + 1 :], h, q)
        for p, j in enumerate(rest)
    )


def _tensor(directions, weights, order):
    result = np.zeros((directions.shape[-1],) * order)
    for v, w in zip(directions, weights):
        term = v
        for _ in range(order - 1):
            term = np.multiply.outer(term, v)
        result += w * term
    return result


def _contract_oracle(tensors, h, q):
    degree = sum(t.ndim for t in tensors)
    result = 0.0
    for indices in itertools.product(range(len(h)), repeat=degree):
        amplitude, start = 1.0, 0
        for tensor in tensors:
            amplitude *= tensor[indices[start : start + tensor.ndim]]
            start += tensor.ndim
        result += amplitude * _wick_hermite(indices, h, q)
    return result


@pytest.mark.parametrize("provider", [LowRankCumulants, DenseCumulants])
@pytest.mark.parametrize("order", [2, 3, 4])
def test_correlated_joint_density_against_wick_oracle(provider, order):
    covariance = np.array([[1.4, 0.3], [0.3, 0.8]])
    mean, value = np.array([0.2, -0.1]), np.array([0.6, 0.25])
    directions = np.array([[1.0, 0.7], [-0.2, 0.8]])
    weights = {3: np.array([0.08, -0.03]), 4: np.array([0.04, 0.02])}
    tensors = {r: _tensor(directions, w, r) for r, w in weights.items()}
    cumulants = (
        provider(directions, weights)
        if provider is LowRankCumulants
        else provider(tensors, event_shape=(2,))
    )
    base = dist.MultivariateNormal(mean, covariance_matrix=covariance)
    expansion = FieldEdgeworth(base, cumulants, order=order)
    q = np.linalg.inv(covariance)
    h = q @ (value - mean)
    correction = 1.0
    if order >= 3:
        correction += _contract_oracle([tensors[3]], h, q) / 6
    if order >= 4:
        correction += _contract_oracle([tensors[4]], h, q) / 24
        correction += _contract_oracle([tensors[3], tensors[3]], h, q) / 72
    np.testing.assert_allclose(expansion.correction(value), correction, rtol=4e-6)
    np.testing.assert_allclose(
        expansion.log_prob(value), base.log_prob(value) + np.log(correction), rtol=4e-6
    )


@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 2)])
def test_map_cube_are_one_event_with_sample_dimensions(shape):
    directions = jnp.arange(math.prod(shape), dtype=float).reshape((1,) + shape) / 10
    base = dist.Normal(jnp.zeros(shape), 1.0).to_event(len(shape))
    expansion = FieldEdgeworth(
        base, LowRankCumulants(directions, {3: [0.03], 4: [0.01]}), order=4
    )
    values = (
        jnp.ones((3,) + shape) * jnp.arange(3).reshape((3,) + (1,) * len(shape)) / 10
    )
    assert expansion.event_shape == shape
    assert expansion.batch_shape == ()
    actual = jax.jit(expansion.log_prob)(values)
    assert actual.shape == (3,)
    np.testing.assert_allclose(
        actual, jnp.stack([expansion.log_prob(x) for x in values]), rtol=4e-6
    )


def test_joint_correction_is_not_a_product_of_pixel_marginals():
    base = dist.Normal(jnp.zeros(2), 1.0).to_event(1)
    expansion = FieldEdgeworth(
        base, LowRankCumulants(jnp.ones((1, 2)), {3: [0.1]}), order=3
    )
    # Same one-point marginals, but mixed K112/K122 are present in the joint.
    x = jnp.array([0.5, 0.8])
    h = np.asarray(x)
    expected = 1 + 0.1 / 6 * ((h.sum()) ** 3 - 6 * h.sum())
    marginals = np.prod(1 + 0.1 / 6 * (h**3 - 3 * h))
    np.testing.assert_allclose(expansion.correction(x), expected, rtol=3e-6)
    assert abs(float(expansion.correction(x)) - marginals) > 0.01


def test_linear_coordinate_change_transforms_all_cumulants_and_density():
    directions = jnp.array([[0.8, 0.3], [0.1, 0.7]])
    weights = {3: jnp.array([0.05, -0.02]), 4: jnp.array([0.03, 0.01])}
    mean = jnp.array([0.1, -0.2])
    covariance = jnp.array([[1.0, 0.2], [0.2, 1.5]])
    a = jnp.array([[2.0, 0.3], [-0.2, 0.8]])
    x = jnp.array([0.5, -0.1])
    original = FieldEdgeworth(
        dist.MultivariateNormal(mean, covariance),
        LowRankCumulants(directions, weights),
        order=4,
    )
    transformed = FieldEdgeworth(
        dist.MultivariateNormal(a @ mean, a @ covariance @ a.T),
        LowRankCumulants(directions @ a.T, weights),
        order=4,
    )
    np.testing.assert_allclose(
        transformed.correction(a @ x), original.correction(x), rtol=3e-6
    )
    np.testing.assert_allclose(
        transformed.log_prob(a @ x),
        original.log_prob(x) - jnp.linalg.slogdet(a)[1],
        rtol=3e-6,
    )


def test_graph_bridge_and_parent_gradients_with_field_likelihood():
    directions = jnp.array([[[0.5, 0.2], [0.1, 0.7]]])
    data = jnp.array([[0.2, 0.1], [0.4, -0.1]])

    def likelihood(theta):
        base = dist.Normal(jnp.ones((2, 2)) * theta, 1.0).to_event(2)
        return FieldEdgeworth(
            base,
            LowRankCumulants(directions, {3: jnp.reshape(theta / 10, (1,)), 4: [0.02]}),
            order=4,
        )

    def model():
        theta = sample("theta", lambda: dist.Normal(0.0, 1.0))
        observe("map", likelihood, theta, obs=data)

    graph = trace(model)
    at = {"theta": jnp.array(0.15)}
    truth = dist.Normal(0.0, 1.0).log_prob(at["theta"]) + likelihood(
        at["theta"]
    ).log_prob(data)
    bridged, _ = log_density(to_numpyro(graph), (), {}, at)
    np.testing.assert_allclose(log_joint(graph, at), truth, rtol=3e-6)
    np.testing.assert_allclose(bridged, truth, rtol=3e-6)
    fn = lambda x: log_joint(graph, {"theta": x})
    derivative = jax.jit(jax.grad(fn))(at["theta"])
    step = 0.001
    difference = (fn(at["theta"] + step) - fn(at["theta"] - step)) / (2 * step)
    np.testing.assert_allclose(derivative, difference, rtol=0.002, atol=0.0002)


def test_fourier_utility_and_periodic_reference_match_real_dense_calculation():
    shape = (2, 3)
    mean = jnp.zeros(shape)
    power = jnp.array([[1.0, 1.3, 1.3], [0.8, 1.7, 1.7]])
    directions = jnp.arange(12, dtype=float).reshape((2,) + shape) / 20
    weights = {3: [0.03, -0.02], 4: [0.01, 0.02]}
    coefficients = jnp.fft.fftn(directions, axes=(-2, -1), norm="ortho")
    restored = low_rank_from_fourier(coefficients, weights, event_shape=shape)
    base = PeriodicNormal(mean, power)
    expansion = FieldEdgeworth(base, restored, order=4)
    n = math.prod(shape)

    def apply_cov(v):
        return jnp.fft.ifftn(
            jnp.fft.fftn(v.reshape(shape), norm="ortho") * power, norm="ortho"
        ).real.ravel()

    covariance = jax.vmap(apply_cov)(jnp.eye(n)).T
    dense_base = dist.TransformedDistribution(
        dist.MultivariateNormal(jnp.zeros(n), covariance), ReshapeTransform(shape, (n,))
    )
    reference = FieldEdgeworth(
        dense_base, LowRankCumulants(directions, weights), order=4
    )
    value = jnp.linspace(-0.3, 0.4, n).reshape(shape)
    np.testing.assert_allclose(
        base.log_prob(value), dense_base.log_prob(value), rtol=3e-6
    )
    np.testing.assert_allclose(
        expansion.log_prob(value), reference.log_prob(value), rtol=3e-6
    )
    assert base.sample(jax.random.key(0), (4,)).shape == (4,) + shape


def test_no_non_gaussian_reference_or_missing_order_is_silently_accepted():
    cumulants = LowRankCumulants(jnp.ones((1, 2)), {3: [0.01]})
    with pytest.raises(TypeError, match="Gaussian"):
        FieldEdgeworth(
            dist.StudentT(4, jnp.zeros(2), 1.0).to_event(1), cumulants, order=3
        )
    with pytest.raises(ValueError, match="4"):
        FieldEdgeworth(dist.Normal(jnp.zeros(2), 1).to_event(1), cumulants, order=4)


def test_negative_field_correction_remains_visible_and_sampling_refuses():
    expansion = FieldEdgeworth(
        dist.Normal(jnp.zeros(2), 1).to_event(1),
        LowRankCumulants(jnp.ones((1, 2)), {3: [6.0]}),
        order=3,
    )
    value = jnp.array([-3.0, -3.0])
    assert expansion.correction(value) < 0
    assert jnp.isnan(jax.jit(expansion.log_prob)(value))
    with pytest.raises(NotImplementedError, match="likelihood"):
        expansion.sample(jax.random.key(0))


@pytest.mark.parametrize("order", [2, 3, 4, 5, 6])
def test_single_mode_reduces_to_scalar_at_higher_orders(order):
    scale, loc, value = 1.7, 0.2, 0.8
    ks = [0.03, 0.02, -0.001, 0.0002]
    base = dist.Normal(jnp.array([loc]), scale).to_event(1)
    field = FieldEdgeworth(
        base,
        LowRankCumulants(jnp.ones((1, 1)), {r: [k] for r, k in enumerate(ks, 3)}),
        order=order,
    )
    scalar = EdgeworthExpansion(loc, scale, cumulants=ks, order=order)
    np.testing.assert_allclose(
        field.log_prob(jnp.array([value])), scalar.log_prob(value), rtol=3e-6
    )


def test_reference_batch_and_cumulant_batch_broadcast_without_merging_scores():
    base = dist.Normal(jnp.array([[0.0, 0.1], [0.2, -0.1]]), 1.0).to_event(1)
    directions = jnp.array([[0.5, 0.8]])
    weights = jnp.array([[0.02], [0.03]])
    expansion = FieldEdgeworth(
        base, LowRankCumulants(directions, {3: weights}), order=3
    )
    assert expansion.batch_shape == (2,)
    value = jnp.array([0.3, 0.5])  # shared observation, separate conditional densities
    expected = jnp.stack(
        [
            FieldEdgeworth(
                dist.Normal(base.base_dist.loc[i], 1).to_event(1),
                LowRankCumulants(directions, {3: weights[i]}),
                order=3,
            ).log_prob(value)
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(jax.jit(expansion.log_prob)(value), expected, rtol=3e-6)
    assert expansion.log_prob(jnp.ones((4, 2, 2))).shape == (4, 2)


def test_graph_plate_vmaps_conditional_distributions_and_keeps_field_event():
    from bayesmith import const, plate

    centers = jnp.array([0.0, 0.2, -0.1])
    directions = jnp.array([[[0.3, 0.6], [0.2, 0.1]]])
    data = jnp.arange(12, dtype=float).reshape(3, 2, 2) / 20

    def likelihood(m):
        return FieldEdgeworth(
            dist.Normal(jnp.ones((2, 2)) * m, 1).to_event(2),
            LowRankCumulants(directions, {3: jnp.array([0.03 + m / 100])}),
            order=3,
        )

    def model():
        p = plate("fields", 3)
        mu = const("mu", centers, plate=p)
        observe("maps", likelihood, mu, obs=data, plate=p)

    graph = trace(model)
    expected = sum(likelihood(m).log_prob(y) for m, y in zip(centers, data))
    bridged, _ = log_density(to_numpyro(graph), (), {}, {})
    np.testing.assert_allclose(jax.jit(lambda: log_joint(graph))(), expected, rtol=3e-6)
    np.testing.assert_allclose(bridged, expected, rtol=3e-6)


def test_mixed_precision_parameters_and_pytree_gradients():
    import equinox as eqx

    with jax.enable_x64():
        directions = jnp.array([[0.3, 0.7]], dtype=jnp.float32)
        weights = jnp.array([0.02], dtype=jnp.float64)
        base = dist.Normal(jnp.zeros(2, dtype=jnp.float64), 1).to_event(1)
        field = FieldEdgeworth(
            base, LowRankCumulants(directions, {3: weights, 4: weights}), order=4
        )
        value = jnp.array([0.1, 0.2], dtype=jnp.float32)
        result = jax.jit(field.log_prob)(value)
        assert result.dtype == jnp.float64
        gradients = eqx.filter_grad(
            lambda op: FieldEdgeworth(base, op, order=4).log_prob(value)
        )(field.cumulants)
        assert jnp.all(jnp.isfinite(gradients.directions))
        assert jnp.any(gradients.directions != 0)
        assert all(jnp.any(w != 0) for w in gradients.weights)


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_periodic_sampling_and_log_density_describe_the_same_covariance(shape):
    n = math.prod(shape)
    # Even spectrum, including nontrivial Nyquist and zero modes.
    frequencies = jnp.meshgrid(*(jnp.fft.fftfreq(s) for s in shape), indexing="ij")
    power = 0.8 + sum(f * f for f in frequencies)
    field = PeriodicNormal(jnp.zeros(shape), power)

    def sqrt_apply(v):
        return jnp.fft.ifftn(
            jnp.fft.fftn(v.reshape(shape), norm="ortho") * jnp.sqrt(power), norm="ortho"
        ).real.ravel()

    root = np.asarray(jax.vmap(sqrt_apply)(jnp.eye(n))).T
    covariance = root @ root.T
    key = jax.random.key(7)
    white = jax.random.normal(key, (3,) + shape)
    samples = field.sample(key, (3,))
    np.testing.assert_allclose(
        samples.reshape(3, n),
        np.asarray(white).reshape(3, n) @ root.T,
        rtol=2e-5,
        atol=1e-6,
    )
    expected = dist.MultivariateNormal(jnp.zeros(n), covariance).log_prob(
        samples.reshape(3, n)
    )
    np.testing.assert_allclose(field.log_prob(samples), expected, rtol=3e-6)
    np.testing.assert_allclose(
        field.variance, np.diag(covariance).reshape(shape), rtol=3e-6
    )


def test_fourier_reality_and_spectrum_errors_are_not_silently_projected():
    import equinox as eqx

    with pytest.raises(eqx.EquinoxRuntimeError, match="real fields"):
        low_rank_from_fourier(jnp.array([[0, 1j, 0, 0]]), {3: [0.01]}, event_shape=(4,))
    with pytest.raises(eqx.EquinoxRuntimeError, match="mode reversal"):
        PeriodicNormal(jnp.zeros(4), jnp.array([1, 2, 3, 4]))
    with pytest.raises(eqx.EquinoxRuntimeError, match="positive"):
        PeriodicNormal(jnp.zeros(4), jnp.array([1, 0, 1, 0]))


@pytest.mark.parametrize(
    "residual", [jnp.zeros(4), jnp.ones(4), jnp.array([1.0, -1.0, 1.0, -1.0])]
)
def test_periodic_precision_remains_correct_at_zero_fourier_modes(residual):
    power = jnp.array([1.0, 2.0, 3.0, 2.0])
    base = PeriodicNormal(jnp.zeros(4), power)
    direction = jnp.array([0.5, 0.2, -0.1, 0.4])
    score = jax.grad(lambda x: -base.log_prob(x))
    actual = jax.jit(lambda x: jax.jvp(score, (x,), (direction,))[1])(residual)
    truth = jnp.fft.ifft(jnp.fft.fft(direction) / power).real
    np.testing.assert_allclose(actual, truth, rtol=3e-6, atol=1e-7)
    field = FieldEdgeworth(
        base, LowRankCumulants(direction[None, :], {3: [0.0], 4: [0.1]}), order=4
    )
    q = float(jnp.dot(direction, truth))
    h = float(jnp.dot(direction, jnp.fft.ifft(jnp.fft.fft(residual) / power).real))
    expected = 1 + 0.1 / 24 * (h**4 - 6 * q * h**2 + 3 * q**2)
    np.testing.assert_allclose(field.correction(residual), expected, rtol=3e-6)


def test_custom_contractions_receive_cross_products_and_keep_coefficients_in_field():
    import equinox as eqx

    class Prepared(eqx.Module):
        amplitude: jax.Array

        def contract(self, orders):
            return {
                (3,): self.amplitude,
                (4,): 2 * self.amplitude,
                (3, 3): 3 * self.amplitude,
            }[orders]

    class Operator(eqx.Module):
        amplitude: jax.Array
        event_shape: tuple = eqx.field(static=True, default=(2,))
        orders: tuple = eqx.field(static=True, default=(3, 4))
        batch_shape: tuple = eqx.field(static=True, default=())

        def prepare(self, value, reference):
            return Prepared(self.amplitude)

    base = dist.Normal(jnp.zeros(2), 1).to_event(1)
    field = FieldEdgeworth(base, Operator(jnp.array(0.1)), order=4)
    np.testing.assert_allclose(
        field.correction(jnp.zeros(2)), 1 + 0.1 / 6 + 0.2 / 24 + 0.3 / 72, rtol=2e-6
    )


def test_covariance_parameter_gradient_reaches_gaussian_and_cumulant_terms():
    directions = jnp.array([[0.5, 0.7], [-0.2, 0.4]])
    weights = {3: jnp.array([0.04, -0.01]), 4: jnp.array([0.03, 0.01])}
    value = jnp.array([0.3, -0.1])

    def density(scale):
        covariance = scale**2 * jnp.array([[1.0, 0.2], [0.2, 1.5]])
        return FieldEdgeworth(
            dist.MultivariateNormal(jnp.zeros(2), covariance),
            LowRankCumulants(directions, weights),
            order=4,
        ).log_prob(value)

    actual = jax.jit(jax.grad(density))(1.2)
    step = 0.002
    finite_difference = (density(1.2 + step) - density(1.2 - step)) / (2 * step)
    np.testing.assert_allclose(actual, finite_difference, rtol=0.001)


def test_compile_and_sample_use_the_conditional_field_density():
    from bayesmith import compile

    def model():
        theta = sample("theta", lambda: dist.Uniform(-0.2, 0.2))
        observe(
            "map",
            lambda t: FieldEdgeworth(
                dist.Normal(jnp.ones((2, 2)) * t, 1).to_event(2),
                LowRankCumulants(jnp.ones((1, 2, 2)) / 2, {3: [0.04], 4: [0.02]}),
                order=4,
            ),
            theta,
            obs=jnp.array([[0.1, 0.3], [-0.2, 0.1]]),
        )

    graph = trace(model)
    plan = compile(graph)
    posterior = plan.sample(jax.random.key(18), num_warmup=30, num_samples=40)
    draws = posterior.samples
    assert draws["theta"].shape == (40,)
    assert jnp.all(jnp.isfinite(draws["theta"]))
    assert jnp.all(jnp.abs(draws["theta"]) < 0.2)
    assert jnp.std(draws["theta"]) > 0


@pytest.mark.parametrize("mask", [[True, True], [True, False]])
@pytest.mark.parametrize("bridge", [False, True])
def test_event_component_masks_are_refused_before_joint_density_is_duplicated(
    mask, bridge
):
    from bayesmith.errors import StructureError

    def model():
        observe(
            "field",
            lambda: FieldEdgeworth(dist.Normal(jnp.zeros(2), 1.0).to_event(1)),
            obs=jnp.array([0.0, 1.0]),
            mask=jnp.array(mask),
        )

    graph = trace(model)
    with pytest.raises(StructureError, match="event"):
        if bridge:
            log_density(to_numpyro(graph), (), {}, {})
        else:
            log_joint(graph)
