"""P6 -- the collapse arm: integrate the exact block out of the NUTS target.

The dangerous failure is the same shape as P4's double count: a graph that
attaches the marginal evidence term without removing the data, or a marginal
that is not the true integral.  Only an absolute-density check against a dense
integral can say so -- so the equivalence guard here compares log_joint of the
reduced graph against a dense integral over the exact block of the original
graph's log_joint, at K points that include both endpoints of every retained
parameter and use non-unit prior widths.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import compile as compile_graph
from bayesmith import const, det, observe, sample, trace
from bayesmith.dispatch.collapse import collapse_graph, marginal_log_density
from bayesmith.dispatch.execute import _depends_on_prediction
from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import log_joint
from tests.exact.models import mixed_radiometer


def _collapse_graph(
    *,
    n=6,
    sigma=0.5,
    x_loc=0.35,
    x_scale=1.7,
    th_loc=-0.2,
    th_scale=1.1,
    seed=3,
):
    basis = jnp.linspace(-1.0, 1.0, n) + 0.3
    data = 1.2 * (basis * 0.9) + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = sample("x", lambda: dist.Normal(x_loc, x_scale))
        th = sample("th", lambda: dist.Normal(th_loc, th_scale))
        Bc = const("basis", basis)
        mu = det(
            "mu", lambda t_, b_, x_: t_ * (b_ * x_), th, Bc, xs, linear_in=("x",)
        )
        observe("d", lambda m: dist.Normal(m, sigma).to_event(1), mu, obs=data)

    return trace(model)


PARAMETER_POINTS = (
    {"th": jnp.asarray(-2.0)},
    {"th": jnp.asarray(-0.5)},
    {"th": jnp.asarray(0.4)},
    {"th": jnp.asarray(2.5)},
)


def _dense_integral_over_x(graph, point):
    grid = jnp.linspace(0.35 - 12.0 * 1.7, 0.35 + 12.0 * 1.7, 60_001)
    log_values = jax.vmap(lambda x: log_joint(graph, {**point, "x": x}))(grid)
    peak = jnp.max(log_values)
    integral = jnp.trapezoid(jnp.exp(log_values - peak), grid)
    return float(peak + jnp.log(integral))


def test_collapse_marginal_matches_a_dense_integral_at_four_points():
    """The reduced graph's log_joint IS the dense integral over the block."""
    with jax.enable_x64(True):
        graph = _collapse_graph()
        reduced = collapse_graph(graph, ("x",), ("th",))
        oracle = np.asarray([_dense_integral_over_x(graph, p) for p in PARAMETER_POINTS])
        found = np.asarray([float(log_joint(reduced, p)) for p in PARAMETER_POINTS])
        np.testing.assert_allclose(found, oracle, rtol=0.0, atol=2.0e-9)


def test_collapse_marginal_matches_the_slogdet_oracle_directly():
    """The marginal term itself, before graph reduction, against slogdet."""
    with jax.enable_x64(True):
        graph = _collapse_graph()
        th = jnp.asarray(0.7)
        got = float(marginal_log_density(graph, ("x",), {"th": th}))
        # dense oracle: d ~ N(th * basis * x_loc, sigma^2 I + x_scale^2 (th basis)(th basis)^T)
        basis = jnp.linspace(-1.0, 1.0, 6) + 0.3
        direction = (basis * th).astype(jnp.float64)
        data = (1.2 * (basis * 0.9) + 0.5 * jax.random.normal(jax.random.key(3), (6,))).astype(jnp.float64)
        cov = 0.5**2 * jnp.eye(6) + 1.7**2 * jnp.outer(direction, direction)
        mean = direction * 0.35
        residual = data - mean
        sign, logdet = jnp.linalg.slogdet(cov)
        oracle = -0.5 * (6 * jnp.log(2 * jnp.pi) + jnp.where(sign > 0, logdet, jnp.nan)
                         + residual @ jnp.linalg.solve(cov, residual))
        assert got == pytest.approx(float(oracle), abs=1e-12)


def test_sample_collapse_routes_and_regresses():
    """collapse=True runs the collapse arm: method, samples, diagnostics=None."""
    plan = compile_graph(_collapse_graph(n=8))
    post = plan.sample(
        jax.random.key(0), num_warmup=300, num_samples=600, collapse=True
    )
    assert post.method == "collapse"
    assert post.log_weights is None
    assert post.khat is None
    assert post.diagnostics is None
    assert set(post.samples) == {"x", "th"}
    assert post.samples["x"].shape[0] == 600
    assert post.samples["th"].shape[0] == 600
    assert post.ess > 0.0


def test_collapse_refuses_a_prediction_dependent_block():
    """A gcr+mh exact block cannot be marginalised exactly; refuse loudly."""
    plan = compile_graph(mixed_radiometer())
    with pytest.raises(GraphError, match="constant-sigma"):
        plan.sample(jax.random.key(0), num_warmup=50, num_samples=50, collapse=True)


def _marginal_quadrature(graph, reduced, lo=-6.0, hi=6.0, points=40001):
    """The true th marginal, by quadrature of the reduced graph's log_joint."""
    grid = jnp.linspace(lo, hi, points)
    logp = jax.vmap(lambda t: log_joint(reduced, {"th": t}))(grid)
    logp = logp - jnp.max(logp)
    density = jnp.exp(logp)
    density = density / jnp.trapezoid(density, grid)
    mean = float(jnp.trapezoid(grid * density, grid))
    sd = float(jnp.sqrt(jnp.trapezoid((grid - mean) ** 2 * density, grid)))
    return mean, sd


def test_the_collapse_arm_matches_the_marginal_quadrature():
    """The collapse arm samples the true marginal, which neither split nor full
    NUTS mixes on: the joint has a ridge (th and x trade off in the product),
    so both ridge-bound samplers underestimate th's marginal spread while the
    collapsed target, with the ridge integrated away, mixes cleanly.

    The oracle is quadrature of the REDUCED graph's own log_joint, which is
    itself pinned against the dense integral by
    test_collapse_marginal_matches_a_dense_integral_at_four_points.
    """
    with jax.enable_x64(True):
        graph = _collapse_graph(n=8)
        reduced = collapse_graph(graph, ("x",), ("th",))
        mean, sd = _marginal_quadrature(graph, reduced)
        post = compile_graph(graph).sample(
            jax.random.key(2), num_warmup=500, num_samples=2000, collapse=True
        )
        draws = np.asarray(post.samples["th"])
        se = draws.std() / np.sqrt(post.ess)
        assert abs(draws.mean() - mean) < 4 * se
        assert draws.std() == pytest.approx(sd, rel=0.2)


@pytest.mark.full
@pytest.mark.parametrize("th_scale", [0.05, 0.5, 2.0, 8.0])
def test_collapse_arm_runs_across_the_coupling_range(th_scale):
    """Bypass the dispatcher: the collapse arm runs and mixes across couplings.

    The coupling between the exact block x and the NUTS block th is swept by
    widening th's prior. The arm is exercised directly (collapse=True), never
    through the cost scheduler. Correctness of the collapsed TARGET is pinned
    elsewhere -- the dense-integral guard and the quadrature match in the fast
    layer -- so this full-layer cell only checks that the arm runs, returns
    both blocks, and carries a positive ESS at every coupling. A broad th
    prior leaves a multimodal th marginal that NUTS can stick in (many
    identical draws), which is a property of the marginal geometry, not of the
    collapse routing, and is deliberately not asserted against here.
    """
    from bayesmith.diagnose.coupling import block_coupling

    with jax.enable_x64(True):
        graph = _collapse_graph(n=12, th_scale=th_scale)
        report = block_coupling(
            graph, ("x",), ("th",), at={"x": jnp.asarray(0.35), "th": jnp.asarray(-0.2)}
        )
        c = float(report.canonical_correlations[0])
        post = compile_graph(graph).sample(
            jax.random.key(2), num_warmup=500, num_samples=1000, collapse=True
        )
        assert post.method == "collapse"
        assert set(post.samples) == {"x", "th"}
        assert post.diagnostics is None
        assert post.ess > 0.0, f"c={c:.3f}"
        for draws in post.samples.values():
            assert draws.shape[0] == 1000
            assert bool(jnp.all(jnp.isfinite(draws))), f"c={c:.3f}"


def test_depends_on_prediction_is_a_capability_table_not_a_string_compare():
    """gcr and log-gcr are fixed-sigma; gcr+snis and gcr+mh are not."""
    assert not _depends_on_prediction("gcr")
    assert not _depends_on_prediction("log-gcr")
    assert _depends_on_prediction("gcr+snis")
    assert _depends_on_prediction("gcr+mh")
