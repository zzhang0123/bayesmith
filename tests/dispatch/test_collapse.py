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
from bayesmith.dispatch.collapse import (
    collapse_graph,
    marginal_log_density,
    observed_descendants,
)
from bayesmith.dispatch.execute import _depends_on_prediction
from bayesmith.errors import GraphError
from bayesmith.exact.block import unchecked_operator
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


def _two_observation_graph(*, n_d=6, n_e=3, sigma_d=0.5, sigma_e=0.8, seed=3):
    """A graph whose second observation is NOT downstream of the exact block.

    ``d`` is a descendant of the exact latent ``w``; ``e`` depends only on the
    sampled latent ``tau``.  Every fixture in this file before R4 had exactly
    one observed node, so the distinction between "the observed nodes" and
    "the observed nodes the exact block reaches" was never exercised.
    """
    basis = jnp.linspace(-1.0, 1.0, n_d) + 0.3
    d_obs = 1.2 * (basis * 0.9) + sigma_d * jax.random.normal(
        jax.random.key(seed), (n_d,)
    )
    e_obs = jnp.asarray([0.4, -0.2, 1.1])[:n_e]

    def model():
        w = sample("w", lambda: dist.Normal(0.35, 1.7))
        tau = sample("tau", lambda: dist.Normal(-0.2, 1.1))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma_d).to_event(1), mu, obs=d_obs)
        nu = det("nu", lambda t_: t_ * jnp.ones(n_e), tau)
        observe("e", lambda m: dist.Normal(m, sigma_e).to_event(1), nu, obs=e_obs)

    return trace(model), basis, d_obs, e_obs


def test_an_observation_outside_the_exact_block_is_counted_once():
    """The defect this fixture exists for, and the reason it is a defect.

    :func:`observed_descendants` already says, in its own docstring, that only
    the observed nodes the removed block reaches may have their likelihood
    moved into the marginal term, "absorbing it would count its density twice".
    It is right, and :func:`collapse_graph` uses it -- for the ABSORB half.
    The term itself was built by :func:`marginal_log_density`, which compressed
    ``for observed in sorted(block.data)``: every observed node the block
    operator carries, reachable or not.

    So an observation outside the block was counted twice -- once inside the
    marginal term and once as the explicit likelihood that survives in the
    reduced graph.  Measured on 2026-09-04 at ``fb1c21f``, the surplus was
    ``-2.9155099456713867`` against a dense oracle, which is ``log p(e|tau)``
    to better than 1e-9.  A likelihood counted twice is a posterior narrowed by
    sqrt(2), reported with no warning; the collapse arm is opt-in, and its only
    guard was ``plan.exact.method != "gcr"``.

    The oracle is dense numpy from the model's own parameters -- it shares no
    QR, no pivots and no offset arithmetic with the implementation.
    """
    with jax.enable_x64(True):
        graph, basis, d_obs, e_obs = _two_observation_graph()
        assert observed_descendants(graph, ("w",)) == ("d",)

        at = {"tau": jnp.asarray(0.7)}
        reduced = collapse_graph(graph, ("w",), ("tau",))
        found = float(log_joint(reduced, at))

        b = np.asarray(basis, dtype=float)
        d = np.asarray(d_obs, dtype=float)
        covariance = (1.7**2) * np.outer(b, b) + 0.5**2 * np.eye(b.size)
        residual = d - b * 0.35
        _, logdet = np.linalg.slogdet(covariance)
        log_p_d = -0.5 * (
            residual @ np.linalg.solve(covariance, residual)
            + logdet
            + b.size * np.log(2.0 * np.pi)
        )
        tau = 0.7
        e = np.asarray(e_obs, dtype=float)
        log_p_e = float(
            np.sum(
                -0.5 * ((e - tau) / 0.8) ** 2
                - np.log(0.8)
                - 0.5 * np.log(2.0 * np.pi)
            )
        )
        log_p_tau = float(
            -0.5 * ((tau + 0.2) / 1.1) ** 2 - np.log(1.1) - 0.5 * np.log(2.0 * np.pi)
        )

        assert found == pytest.approx(log_p_d + log_p_e + log_p_tau, abs=1e-9)


def test_the_marginal_term_carries_only_the_blocks_own_observations():
    """The same defect one level down, so a repair in the wrong place fails.

    The test above compares a total; this one asserts what the TERM is, which
    is where the double count lives.  A repair that removed ``e`` from the
    reduced graph instead of from the term would satisfy the total and still
    have a marginal log-density that is not one.
    """
    with jax.enable_x64(True):
        graph, basis, d_obs, _ = _two_observation_graph()
        at = {"tau": jnp.asarray(0.7)}
        found = float(marginal_log_density(graph, ("w",), at))

        b = np.asarray(basis, dtype=float)
        d = np.asarray(d_obs, dtype=float)
        covariance = (1.7**2) * np.outer(b, b) + 0.5**2 * np.eye(b.size)
        residual = d - b * 0.35
        _, logdet = np.linalg.slogdet(covariance)
        oracle = -0.5 * (
            residual @ np.linalg.solve(covariance, residual)
            + logdet
            + b.size * np.log(2.0 * np.pi)
        )
        assert found == pytest.approx(float(oracle), abs=1e-9)


def test_the_block_determinant_enters_with_a_minus_one_half():
    """The module docstring says the opposite sign, and the code is right.

    ``collapse.py``'s docstring stated that the ``-sum(log pivots[:n_block])``
    folded in by ``marginalise_arrays`` IS ``0.5 * logdet(F_bb)``.  Measured,
    that quantity is ``-0.5 * logdet(F_bb)``: ``-2.1914870216494005`` against
    ``-2.1914870216494``.  The prose was corrected in R4; a corrected docstring
    cannot be pinned, so the convention is asserted here, because this is the
    sentence an evidence assembler copies.

    Asserted through the determinant lemma rather than by reaching for the
    pivots, so the test uses only what a consumer can call:

        logdet(A S A^T + N) = logdet(N) + logdet(S) + logdet(A^T N^-1 A + S^-1)

    **This test kills nothing its neighbour does not**, and saying so is better
    than letting a reader count it twice. The lemma form and the ``slogdet``
    form of the same oracle are algebraically identical -- they differ by one
    ULP here -- so flipping the pivot term inside ``marginalise_arrays`` fails
    seven tests in this file at once, measured, and this is one of the seven.

    It is kept because it is the only one that names the determinant and its
    sign in a form a reader can compare against the module docstring. The
    docstring got that sign backwards for a whole release; a test that spells
    out ``logdet(N) + logdet(S) + logdet(F)`` is where the next reader checks.
    """
    with jax.enable_x64(True):
        graph, basis, d_obs, _ = _two_observation_graph()
        at = {"tau": jnp.asarray(0.7)}
        found = float(marginal_log_density(graph, ("w",), at))

        b = np.asarray(basis, dtype=float)
        d = np.asarray(d_obs, dtype=float)
        prior_variance = 1.7**2
        noise = 0.5**2
        residual = d - b * 0.35
        covariance = prior_variance * np.outer(b, b) + noise * np.eye(b.size)

        fisher = float(b @ b / noise + 1.0 / prior_variance)
        logdet_noise = b.size * np.log(noise)
        logdet_prior = np.log(prior_variance)
        logdet_fisher = np.log(fisher)

        assembled = -0.5 * (
            residual @ np.linalg.solve(covariance, residual)
            + logdet_noise
            + logdet_prior
            + logdet_fisher
            + b.size * np.log(2.0 * np.pi)
        )
        assert found == pytest.approx(float(assembled), abs=1e-9)

        wrong_sign = assembled + logdet_fisher
        assert abs(float(wrong_sign) - found) > 1.0, (
            "the wrong sign must be far from the right answer, or this test "
            "would pass under both conventions"
        )


def _sorts_before_graph(*, n_d=6, n_a=3, sigma_d=0.5, sigma_a=0.8, seed=3):
    """Like :func:`_two_observation_graph`, but the unreached node sorts FIRST.

    ``marginal_log_density`` walks ``sorted(block.data)`` and advances a row
    cursor through a design laid out in that same order, skipping the row group
    of any node the block does not reach.  When the skipped node sorts LAST the
    cursor advance in the skip branch is dead code -- nothing after it reads the
    cursor -- so deleting it is invisible.  Measured 2026-09-04: with the
    unreached node named ``e`` the mutant and the shipped code both return
    -4.284669145867; renaming it ``a`` the mutant returns -8.446550081297, wrong
    by 4.16 nats.

    Found by an adversarial review, not by reading, and the hazard is the one
    the code comment beside the loop explicitly names.
    """
    basis = jnp.linspace(-1.0, 1.0, n_d) + 0.3
    d_obs = 1.2 * (basis * 0.9) + sigma_d * jax.random.normal(
        jax.random.key(seed), (n_d,)
    )
    a_obs = jnp.asarray([0.4, -0.2, 1.1])[:n_a]

    def model():
        w = sample("w", lambda: dist.Normal(0.35, 1.7))
        tau = sample("tau", lambda: dist.Normal(-0.2, 1.1))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma_d).to_event(1), mu, obs=d_obs)
        nu = det("nu", lambda t_: t_ * jnp.ones(n_a), tau)
        # Named to sort BEFORE "d". That is the whole fixture.
        observe("a", lambda m: dist.Normal(m, sigma_a).to_event(1), nu, obs=a_obs)

    return trace(model), basis, d_obs


def test_the_skipped_row_group_still_advances_the_design_cursor():
    """The hazard the loop's own comment names, with a fixture that can see it.

    The design's rows are laid out in ``sorted(block.data)`` order, so skipping
    a node's compression without advancing past its rows hands every LATER node
    the wrong slice.  With the unreached observation sorting last there is no
    later node and the bug is silent; here there is one.
    """
    with jax.enable_x64(True):
        graph, basis, d_obs = _sorts_before_graph()
        assert observed_descendants(graph, ("w",)) == ("d",)
        block = unchecked_operator(
            graph, ("w",), at={"tau": jnp.asarray(0.7)}, probe_gaussian=False
        )
        assert sorted(block.data) == ["a", "d"], (
            "the fixture only tests the cursor if the unreached node sorts first"
        )

        found = float(marginal_log_density(graph, ("w",), {"tau": jnp.asarray(0.7)}))
        b = np.asarray(basis, dtype=float)
        d = np.asarray(d_obs, dtype=float)
        covariance = (1.7**2) * np.outer(b, b) + 0.5**2 * np.eye(b.size)
        residual = d - b * 0.35
        _, logdet = np.linalg.slogdet(covariance)
        oracle = -0.5 * (
            residual @ np.linalg.solve(covariance, residual)
            + logdet
            + b.size * np.log(2.0 * np.pi)
        )
        assert found == pytest.approx(float(oracle), abs=1e-9)


def test_a_block_the_data_does_not_reach_integrates_its_prior_to_one():
    """The empty-terms branch, which is production-reachable and was untested.

    A latent with no observed descendant is still classified exact and still
    reaches the collapse arm -- an adversarial review ran
    ``compile(...).sample(collapse=True)`` on one end to end.  With no data term
    the Gaussian integral is over the prior alone, so the marginal log-density
    is ``log INT p(z) dz = 0``.

    Measured before this test existed: a mutant folding the prior TWICE returned
    a number wrong by 0.349 nats with the whole fast layer green.
    """
    with jax.enable_x64(True):
        observed = jnp.asarray([0.4, -0.2, 1.1])

        def model():
            z = sample("z", lambda: dist.Normal(1.5, 0.4))
            tau = sample("tau", lambda: dist.Normal(-0.2, 1.1))
            nu = det("nu", lambda t_: t_ * jnp.ones(3), tau)
            observe("e", lambda m: dist.Normal(m, 0.8).to_event(1), nu, obs=observed)
            # `z` is declared and never observed through: no descendant.
            det("unused", lambda z_: z_ * 2.0, z, linear_in=("z",))

        graph = trace(model)
        assert observed_descendants(graph, ("z",)) == ()
        found = float(marginal_log_density(graph, ("z",), {"tau": jnp.asarray(0.7)}))
        assert found == pytest.approx(0.0, abs=1e-12), (
            "a normalised prior integrates to one, so its log is zero; a "
            "number away from zero means the prior was folded a second time "
            "or its normaliser was dropped"
        )
