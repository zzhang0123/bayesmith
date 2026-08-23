"""The conjugate solves, against a dense oracle that shares none of them."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.block import domain_zero, variance_parts
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import condition_bound, normal_operator
from tests.exact.models import straight_line, two_linear_latents
from tests.exact.oracle import graph_oracle


def _sigma(graph, at):
    return noise_std_at(graph, at)


def test_the_bound_is_lambda_max_times_the_loosest_prior_variance():
    """The bound's definition, checked against a dense eigendecomposition.

    lambda_max comes from the oracle's precision matrix -- which is built by
    probing g on a basis and never differentiates anything -- so this is the
    matrix-free power iteration against an independent route, not against
    itself.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        bound = float(condition_bound(block, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    largest = float(np.linalg.eigvalsh(oracle.precision)[-1])
    loosest_variance = float(np.max(oracle.prior_std**2))
    assert bound == pytest.approx(largest * loosest_variance, rel=1e-3)


@pytest.mark.parametrize("loosened", [1.0, 1e2, 1e4])
def test_the_bound_is_never_below_the_true_condition_number(loosened):
    """The whole point: it may refuse a good solve, never accept a bad one.

    Swept across four orders of magnitude of prior width, because the bound is
    tight at one end (the prior alone holds a direction, so lambda_min IS the
    prior curvature) and loose at the other. Both must stay on the safe side.
    """
    import dataclasses

    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        widened = dataclasses.replace(
            block,
            prior_std={**block.prior_std, "b": block.prior_std["b"] * loosened},
        )
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        bound = float(condition_bound(widened, noise_std=sigma, iterations=80))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    # `b`'s flat index depends on the oracle's own domain order, not on an
    # assumed position -- derive it rather than hardcoding 1, so a change to
    # that ordering cannot silently point this mutation at the wrong
    # diagonal entry of the dense precision.
    names = [name for name, _ in oracle.order]
    assert names == ["a", "b"], names
    b_index = names.index("b")
    precision = oracle.precision.copy()
    # Rebuild the dense precision with b's widened prior, so the comparison is
    # against the system the bound was actually computed for.
    precision[b_index, b_index] += (
        1.0 / (oracle.prior_std[b_index] * loosened) ** 2
        - 1.0 / oracle.prior_std[b_index] ** 2
    )
    true_kappa = float(np.linalg.cond(precision))
    assert bound >= true_kappa * (1.0 - 1e-6), (bound, true_kappa)


def test_the_bound_is_loose_when_the_data_constrains_every_direction():
    """Stated rather than hidden: this is the price of a one-sided guarantee.

    A one-parameter block has a true kappa of exactly 1 -- M is 1x1 -- and the
    bound reports lambda_max times the prior variance, which is hundreds. That
    is not a defect; it is what an upper bound derived from the prior must
    say when the data, not the prior, is what sets lambda_min. CG on such a
    block converges in one step, so the residual absorbs the slack.
    """
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        sigma = _sigma(graph, {"w": jnp.asarray(0.0)})
        bound = float(condition_bound(block, noise_std=sigma, iterations=40))
        oracle = graph_oracle(graph, ("w",), at={})
    assert float(np.linalg.cond(oracle.precision)) == pytest.approx(1.0, rel=1e-9)
    assert bound > 100.0
    assert bound == pytest.approx(
        float(oracle.precision[0, 0]) * float(oracle.prior_std[0] ** 2), rel=1e-3
    )


def test_the_bound_grows_in_proportion_to_the_loosest_prior_variance():
    """It is linear in max(prior_variance) by construction -- verify it is.

    A bound that ignored the prior would be flat across this pair, and a bound
    that took the TIGHTEST prior would move the wrong way.
    """
    import dataclasses

    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = _sigma(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        tight = float(condition_bound(block, noise_std=sigma, iterations=80))
        widened = dataclasses.replace(
            block, prior_std={**block.prior_std, "b": block.prior_std["b"] * 100.0}
        )
        loose = float(condition_bound(widened, noise_std=sigma, iterations=80))
    # b's prior variance grows by 1e4 and it was already the loosest (7 vs 5).
    assert loose == pytest.approx(1e4 * tight, rel=1e-2)


def test_the_normal_operator_is_symmetric():
    """`<u, M v> == <M u, v>` -- taking the curvature as a gradient of chi2
    makes this true by construction, and an adjoint assembled by hand from A
    and A^T is where the sign conventions would go wrong."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"a": 0.0, "b": 0.0}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        keys = jax.random.split(jax.random.key(3), 2)
        u = {
            n: jax.random.normal(k, block.shape[n])
            for n, k in zip(block.names, keys, strict=True)
        }
        v = {
            n: jax.random.normal(jax.random.fold_in(k, 1), block.shape[n])
            for n, k in zip(block.names, keys, strict=True)
        }
        left = sum(float(jnp.sum(u[n] * normal(v)[n])) for n in block.names)
        right = sum(float(jnp.sum(normal(u)[n] * v[n])) for n in block.names)
    assert left == pytest.approx(right, rel=1e-10)


def test_the_normal_operator_reproduces_the_dense_precision_matrix():
    """Applying M to each basis vector must give the dense precision's columns.

    The dense matrix comes from the oracle, which never calls linearize or
    vjp; M comes from `jax.grad` of the block's own chi-squared. They agree
    or one of them is wrong.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"a": 0.0, "b": 0.0}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        columns = []
        for name in block.names:
            probe = dict(domain_zero(block))
            probe[name] = jnp.asarray(1.0)
            image = normal(probe)
            columns.append(
                np.concatenate([np.asarray(image[n]).ravel() for n in block.names])
            )
        applied = np.stack(columns, axis=1)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert np.allclose(applied, oracle.precision, rtol=1e-8)


def test_the_normal_operator_sums_over_every_observed_node():
    """Two observed nodes both contribute curvature; dropping one changes it."""
    from tests.exact.models import two_observations

    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        weight = {
            o: 1.0 / jnp.asarray(s) ** 2
            for o, s in _sigma(graph, {"w": jnp.asarray(0.0)}).items()
        }
        normal = normal_operator(block, weight, variance_parts(block))
        applied = float(normal({"w": jnp.asarray(1.0)})["w"])
        oracle = graph_oracle(graph, ("w",), at={})
    assert applied == pytest.approx(float(oracle.precision[0, 0]), rel=1e-8)
