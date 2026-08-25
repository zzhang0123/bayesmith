"""Acceptance gate two: the statistics, checked against oracles that share nothing.

Every other test module in this suite asks whether one function does what its
docstring says. This one asks the question section 7.2 asks: **does the thing
a user actually runs return the right posterior?** So each guard here runs
``compile(...).sample()``, ``compile(...).estimate()`` or the ``gibbs_fn`` the
plan wires up, and compares the answer against grid quadrature of ``log_joint``
or against ``tests/exact/oracle.py``'s dense linear algebra -- never against a
second reading of the path under test.

**Four measurements this file is built on, all taken before it was written.**

*The frozen-sigma proposal on ``steep_radiometer`` is wrong by a full posterior
sd.* Under x64, quadrature gives mean 1.3909078, sd 0.29251207; 4000 raw GCR
draws at the GLS fixed point read mean -0.921 to -0.958 sd away and 0.78-0.80
times too narrow, over keys 0/1/4. One Metropolis step brings that to -0.004 to
+0.021 sd and a sd ratio of 0.993-0.998. That gap is what makes the accept step
testable at all rather than a no-op wrapped in ceremony.

*A wrong-but-x-independent sigma-hat costs ESS and not accuracy.* On
``radiometer`` at N=2000 over keys 0/1/4/7, sigma frozen at the GLS fixed point
reads Kish ESS/N 0.998 while sigma frozen at ``w=27`` reads 0.151-0.161 -- a
6.2-6.6x collapse -- and the weighted mean stays within 0.084 posterior sd on
both, with the weighted sd within 4.7% on both, while the UNWEIGHTED draws
behind the wrong one are 9.1-9.5x too wide. The moments are blind to the freeze
point; only the ESS reads it. That one is asserted in
``tests/exact/test_correct.py``, next to the ``self_normalise`` it is a
property of.

*The plan's derived ``tol`` delivers what it advertises.* Solving each
constant-sigma fixture at ``plan.exact.tol`` and comparing against the dense
oracle, the true relative error is 0 to 1.38e-04 against a
``CONVERGENCE_TARGET`` of 1e-3 -- worst case ``collinear_pair``, 7.3x of margin
-- across kappa from 1.0 to 5.8e+11.

*NUTS is not a reference on every graph in this repository, and two of them are
recorded here rather than discovered again.* On ``mixed_radiometer``, both
``HMCGibbs`` and pure NUTS settle at ``tau = -0.69 +/- 0.18`` from numpyro's
default init, where ``log_joint`` is **26.8 nats** below its value at the true
posterior mean of 4.13 -- a local mode 4e+11 times less probable, behind the
barrier ``sigma = |tau| + 0.2`` puts at zero. On ``plated_radiometer(n=6)``,
per-coordinate quadrature says the SNIS answer is right to 0.14 posterior sd on
every coordinate while NUTS on the same graph is off by **204 sd** on
coordinate 1. Both are why the composite-vs-NUTS comparison below runs on a
jointly Gaussian mixed fixture defined in this file, and why the ESS-floor
boundary asserts the dispatch DECISION rather than agreement between the two
branches.
"""

from __future__ import annotations

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import compile as compile_graph
from bayesmith import const, det, observe, sample, trace
from bayesmith.bridge.numpyro_bridge import nuts as nuts_draws
from bayesmith.dispatch.classify import block_at
from bayesmith.dispatch.plan import CONVERGENCE_TARGET, chain_ess, kappa_upper
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.correct import log_weight, self_normalise
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.gibbs import _precision_at, gibbs_factory
from bayesmith.exact.gls import iterative_gls, sigma_from_graph
from bayesmith.exact.precision import diagonal_from
from bayesmith.exact.solve import gcr_sample, wiener_solve
from bayesmith.graph.evaluate import log_joint
from tests.exact.models import (
    collinear_pair,
    many_observations,
    plated_and_scalar_latents,
    plated_latent,
    plated_radiometer,
    prior_held_direction,
    radiometer,
    radiometer_group,
    sigma_functional_block,
    steep_radiometer,
    straight_line,
    two_linear_latents,
    two_observations,
    unconstrained_latent,
    wide_plate,
)
from tests.exact.oracle import flat_domain, graph_oracle

# --------------------------------------------------------------------------
# shared references: grid quadrature of `log_joint`, and a local mixed fixture
# --------------------------------------------------------------------------


def quadrature(graph, name, lo, hi, points=20001, place=None):
    """``(mean, sd, grid, density)`` of a ONE-dimensional posterior.

    Shares ``log_joint`` with the model and nothing else -- no operator, no
    CG, no importance weight -- so a disagreement is a disagreement about the
    posterior. Insensitive to both knobs, measured on ``radiometer()``: mean
    2.94632607 / sd 0.04665180 at ``(-2, 8, 4001)`` against 2.94632609 /
    0.04665179 at ``(-4, 12, 20001)`` and 2.94632609 / 0.04665179 at
    ``(0, 6, 40001)`` -- eight digits over a window 2.7x wider and a grid 10x
    finer.

    Args:
        place: how a scalar becomes ``log_joint``'s ``values``. Defaults to
            ``{name: value}``, which is every scalar-latent graph here; a
            plated one passes a builder that writes the scalar into one
            element and holds the rest, which is exact wherever the plate
            factorises.
    """
    place = (lambda value: {name: value}) if place is None else place
    grid = jnp.linspace(lo, hi, points)
    log_p = jax.vmap(lambda value: log_joint(graph, place(value)))(grid)
    density = np.array(jnp.exp(log_p - jnp.max(log_p)), dtype=float)
    axis = np.array(grid, dtype=float)
    density /= np.trapezoid(density, axis)
    mean = float(np.trapezoid(axis * density, axis))
    spread = math.sqrt(float(np.trapezoid((axis - mean) ** 2 * density, axis)))
    return mean, spread, axis, density


def quadrature_pair(graph, names, ranges, points=401):
    """Both marginals of a TWO-dimensional posterior, by the same route.

    Evaluated row by row under ``vmap`` rather than over a flattened product:
    ``jax.lax.map`` over 401**2 points is a 160,801-step scan and does not
    finish in a test's lifetime, while 401 vmapped rows take 0.27 s.

    Grid-insensitive, measured on ``radiometer_group()``: ``a`` reads
    1.50482928 / 0.02124955 at 301 points, 1.50474336 / 0.02126071 at 401 and
    1.50474250 / 0.02126164 at 601, against 1.50474251 / 0.02126163 from the
    801-point run this constant was taken from.
    """
    first, second = names
    grids = [jnp.linspace(*window, points) for window in ranges]
    row = jax.jit(
        jax.vmap(
            lambda one, two: log_joint(graph, {first: one, second: two}),
            in_axes=(None, 0),
        )
    )
    log_p = jnp.stack([row(value, grids[1]) for value in grids[0]])
    density = np.array(jnp.exp(log_p - jnp.max(log_p)), dtype=float)
    axes = [np.array(grid, dtype=float) for grid in grids]
    density /= np.trapezoid(np.trapezoid(density, axes[1], axis=1), axes[0])
    out = {}
    for index, name in enumerate(names):
        marginal = np.trapezoid(density, axes[1 - index], axis=1 - index)
        mean = float(np.trapezoid(axes[index] * marginal, axes[index]))
        variance = float(
            np.trapezoid((axes[index] - mean) ** 2 * marginal, axes[index])
        )
        out[name] = (mean, math.sqrt(variance))
    return out


def quantile_sample(axis, density, count):
    """``count`` midpoint quantiles of a tabulated density.

    Deterministic, so the invariance comparison below carries no Monte-Carlo
    error in its INPUT -- only in the step applied to it.
    """
    cell = (density[1:] + density[:-1]) / 2 * np.diff(axis)
    cdf = np.concatenate([[0.0], np.cumsum(cell)])
    return np.interp((np.arange(count) + 0.5) / count, cdf / cdf[-1], axis)


def weighted_moments(draws, log_weights):
    """``(mean, sd)`` per coordinate under self-normalised weights."""
    weights = np.asarray(jax.nn.softmax(log_weights), dtype=float)
    values = np.asarray(draws, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    mean = (weights[:, None] * values).sum(axis=0)
    spread = np.sqrt((weights[:, None] * (values - mean) ** 2).sum(axis=0))
    return mean, spread


#: How tightly ``tau`` sets ``w``'s prior LOCATION in :func:`located_ancestor`.
#: Measured across 0.1, 0.3 and 3.0: the block's conditional mean travels 36.9,
#: 4.5 and 0.0 conditional sds respectively as ``tau`` sweeps [-2.5, 6], while
#: ``tau``'s own ESS under the sweep reads 446-471, 532-570 and 561-697 of 1500.
#: 3.0 is unusable -- the conditional does not move, so a ``gibbs_fn`` ignoring
#: ``hmc_sites`` entirely would pass -- and 0.1 buys movement at the cost of
#: mixing. 0.3 is the interior point.
LINK_STD: float = 0.3


def located_ancestor(
    *,
    n=12,
    w_true=1.9,
    tau_loc=1.3,
    tau_std=0.8,
    sigma=0.35,
    link_std=LINK_STD,
    seed=34,
):
    """A mixed graph that is JOINTLY GAUSSIAN, so both marginals are known.

    Local to this file, and the reason is a measurement rather than taste.
    Every mixed fixture ``tests/exact/models.py`` ships ejects its outside
    latent by making it set another latent's prior WIDTH -- ``mixed_radiometer``
    (``w ~ N(0, |tau| + 0.2)``), ``three_latent_chain``, ``indirect_ancestor``,
    ``shared_ancestor``, ``diamond_ancestor``. That ``|tau|`` puts a barrier at
    zero, and on ``mixed_radiometer`` both ``HMCGibbs`` and pure NUTS fall
    behind it: measured at 1500 draws over keys 0 and 3, both report
    ``tau = -0.69 +/- 0.18`` where ``log_joint`` is 26.8 nats below its value at
    the true posterior mean of 4.13. The two samplers AGREE, which is exactly
    the self-consistency trap P3a recorded, and neither is right. So a
    marginal-moment comparison there measures nothing.

    Here ``tau`` sets ``w``'s prior **loc** instead. The ancestor rule still
    ejects it -- measured, the partition is ``{w}`` exact by ``gcr`` and
    ``{tau}`` by NUTS -- but the joint is a plain bivariate Gaussian, so it has
    no barrier, no second mode, and marginals that 2-D quadrature returns to
    eight digits. ``sigma_needs_rebuild`` still comes out True: ``tau`` is a
    transitive ancestor of ``d``, and ``_sigma_needs_rebuild``'s own docstring
    says it cannot tell a loc-ancestor from a scale-ancestor and errs towards
    rebuilding.

    Non-Gaussian is the other obvious way to force a latent to NUTS and it is
    NOT available: with a constant sigma the sweep hoists ``noise_std`` at the
    graph's prior centre, and ``gibbs._prior_centre`` refuses a latent with no
    ``loc``. Every number here is distinct from every other -- 1.9, 1.3, 0.8,
    0.35, 0.3 -- so a solve returning the wrong one could not pass.
    """
    x = jnp.linspace(1.0, 3.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        tau = sample("tau", lambda: dist.Normal(tau_loc, tau_std))
        w = sample("w", lambda t: dist.Normal(t, link_std), tau)
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


# --------------------------------------------------------------------------
# 7.2 (b): the Metropolis step, at one member and at three
# --------------------------------------------------------------------------


@pytest.mark.parametrize("draws, seed", [(2000, 0), (2000, 4)])
def test_the_frozen_sigma_proposal_is_wrong_and_the_accept_step_is_what_fixes_it(
    draws, seed
):
    """Section 5.3's guard, with the positive control that gives it content.

    **The draft's name for this test was**
    ``test_dropping_the_reverse_density_term_moves_the_moments``, **and that
    mutation does not move the moments.** Dropping ``log_weight(now)`` leaves
    ``log alpha = log w(x')``, which on this fixture is so negative that not
    one proposal in 96,000 is accepted (Task 8's measurement): the kernel
    becomes the IDENTITY, and an identity kernel passes an invariance test
    perfectly. The acceptance clause is what kills it, so the acceptance
    clause is named in the assertion below rather than left as decoration.

    Three clauses, three different mutations:

    * ``proposal`` vs ``truth`` is a POSITIVE CONTROL, not a correctness
      claim. If the frozen-sigma draw were already right the accept step would
      have nothing to do and every other clause here would be green against a
      kernel that did nothing. Measured under x64 at 4000 draws over keys
      0/1/4: the raw proposal's mean is -0.921, -0.958, -0.930 posterior sd
      away and its sd is 0.801, 0.793, 0.781 times the truth's.
    * the paired shift and the sd ratio kill sigma-hat rebuilt at the CURRENT
      state -- section 5.3's original error -- which Task 8 measured at -16.1
      to -17.0 standard errors and a 13.0-17.3% sd narrowing at 8000 draws,
      i.e. -8.0 to -8.5 SE at the 2000 used here.
    * the acceptance band kills the identity kernel above, and equally kills
      an accept that always fires (which returns the proposal, whose moments
      the first clause has just measured to be a full sd out).

    x64 is load-bearing and not for precision: ``jax.random.normal`` returns
    different VALUES at the two dtypes, so ``steep_radiometer`` is two
    datasets, and every number here was measured on the float64 one (float32
    quadrature reads mean 2.2499376 / sd 0.42078377 instead). The graph is
    therefore traced inside the context manager.

    Measured over keys 0/1/4 at 2000 draws: acceptance 0.4455/0.4555/0.4655,
    paired shift -0.290/+1.746/+1.132 SE, sd ratio 0.9979/0.9929/0.9968,
    output mean -0.0036/+0.0209/+0.0132 posterior sd from quadrature.
    """
    with jax.enable_x64(True):
        graph = steep_radiometer()
        truth, spread, axis, density = quadrature(graph, "w", -4.0, 8.0, 4001)
        block = unchecked_operator(graph, ("w",), {})
        precision = _precision_at(graph, block, {}, "gcr+mh", 1e-10, None)
        keys = jax.random.split(jax.random.key(seed), draws)

        raw, _ = jax.vmap(
            lambda one: gcr_sample(
                block, precision=precision, key=one, tol=1e-10, require_convergence=None
            )
        )(keys)
        proposal = np.asarray(raw["w"], dtype=float)
        assert abs(proposal.mean() - truth) / spread > 0.5
        assert proposal.std() / spread < 0.90

        step = gibbs_factory(graph, ("w",), tol=1e-10, method="gcr+mh")
        start = jnp.asarray(quantile_sample(axis, density, draws))
        out = jax.vmap(
            lambda one, now: step(rng_key=one, gibbs_sites={"w": now}, hmc_sites={})[
                "w"
            ]
        )(keys, start)

        # The acceptance band comes FIRST because the identity kernel it kills
        # makes every other clause degenerate rather than red: with the
        # reverse-density term dropped, `out - start` is bitwise zero, so the
        # paired shift and its own standard error are both 0.0 and the shift
        # clause fails on `0.0 < 0.0` -- true only by the accident of a strict
        # inequality. Ordered this way the failure names the mutation.
        accepted = float(np.mean(np.asarray(out) != np.asarray(start)))
        assert 0.30 < accepted < 0.65
        shift = np.asarray(out - start, dtype=float)
        standard_error = float(shift.std()) / math.sqrt(draws)
        assert abs(float(shift.mean())) < 4.0 * standard_error
        assert float(np.asarray(out).std()) == pytest.approx(
            float(np.asarray(start).std()), rel=2e-2
        )
        assert abs(float(np.asarray(out).mean()) - truth) / spread < 0.15


#: The four-member block ``sigma_functional_block`` is asked for, split three
#: in and one out, so the sweep has both a multi-member block and a non-empty
#: ``at``. ``d`` is a legal member name here because this fixture's observed
#: node is ``y``.
MULTI = ("a", "b", "c")
MULTI_OUTSIDE = {"d": 0.4}
#: ``sigma = base * exp(0.4a - 0.25b + 0.55c + 0.3d)``: no two coefficients
#: equal, none zero, and the sign pattern is not any of
#: ``DEPENDENCE_PATTERNS``' own.
MULTI_WEIGHTS = (0.4, -0.25, 0.55, 0.3)
#: ``n=20`` and not this fixture's default 200. The log weight carries
#: ``-n * (w . theta)`` from the log-determinant, so the acceptance rate falls
#: with ``n``: measured from a proposal-drawn start, ``(n, scale)`` gives 0.951
#: at (6, 0.08) down to 0.526 at (60, 1.0), and 0.560 at (20, 1.0). Near 0.5 is
#: where the all-or-nothing claim below has the most to say -- under a
#: per-member accept the fraction of MIXED steps is ``1 - p^3 - (1-p)^3``,
#: which peaks at p = 0.5.
MULTI_N = 20


def _multi_step(weights, seed, count, tol=1e-8):
    """One MH step per key, started from a draw of the step's own proposal.

    Starting from a fixed point instead makes the acceptance rate a property
    of that point rather than of the kernel: measured at
    ``{a: 0.3, b: -0.7, c: 1.1}`` with ``n=60`` the rate is 0.0000, because
    the log weight at a state that far out dominates every proposal. Started
    from the proposal, the rate is the independence sampler's own.
    """
    graph = sigma_functional_block(weights=weights, n=MULTI_N)
    outside = {name: jnp.asarray(value) for name, value in MULTI_OUTSIDE.items()}
    block = unchecked_operator(graph, MULTI, outside)
    precision = _precision_at(graph, block, outside, "gcr+mh", tol, None)
    start, _ = jax.vmap(
        lambda one: gcr_sample(
            block, precision=precision, key=one, tol=tol, require_convergence=None
        )
    )(jax.random.split(jax.random.key(seed + 1_000), count))
    step = gibbs_factory(graph, MULTI, tol=tol, method="gcr+mh")
    out = jax.vmap(
        lambda one, a, b, c: step(
            rng_key=one, gibbs_sites={"a": a, "b": b, "c": c}, hmc_sites=outside
        )
    )(
        jax.random.split(jax.random.key(seed), count),
        start["a"],
        start["b"],
        start["c"],
    )
    moved = np.stack(
        [np.asarray(out[name]) != np.asarray(start[name]) for name in MULTI]
    )
    return graph, outside, out, moved.sum(axis=0)


@pytest.mark.parametrize(
    "weights, lo, hi",
    [
        ((0.0,) * 4, 0.99, 1.0),
        (MULTI_WEIGHTS, 0.35, 0.80),
    ],
)
@pytest.mark.parametrize("seed", [0, 6])
def test_the_metropolis_accept_is_one_decision_for_the_whole_block(
    weights, lo, hi, seed
):
    """One ``take`` per STEP, not one per member -- the claim width 1 cannot make.

    ``_mh_step`` builds ``proposed``, ``now`` and the accept as three per-name
    comprehensions over ``names``. Task 8 shipped and declared it with every
    block having exactly ONE member, and at width 1 a per-member ``take``, an
    accept that only ever moves the first member, and a member mixed up
    between ``proposed`` and ``now`` are all indistinguishable from the correct
    code. Sub-criterion 3 asks for at least THREE values on any axis the code
    branches over, and this is that axis.

    The assertion is that the accept is **all-or-nothing across the block**:
    over every key, the number of members whose value changed is 0 or 3, never
    1 or 2. Under a per-member ``take`` at acceptance p the mixed fraction is
    ``1 - p^3 - (1-p)^3``, i.e. 0.74 at the 0.56 this fixture measures -- so
    the mutation is not a marginal call, it is three quarters of the sample.

    Both arms matter and neither alone is enough. With ``weights`` all zero
    sigma is genuinely constant, ``q`` IS the exact conditional, ``log alpha``
    is zero and every step accepts (measured 0.9985 over 2000 keys -- the three
    rejections are float32 noise in a log ratio constructed to be zero) -- so
    the all-or-nothing claim is satisfied trivially there and the arm's real
    job is the acceptance band, which pins that ``q == p`` really does mean
    "always accept". With ``weights`` nonzero the rate is 0.53-0.60 and BOTH
    outcomes occur, which is what gives the all-or-nothing claim something to
    rule out. Measured: under a per-member ``take`` the two nonzero-weight
    cells go red and the two constant-sigma cells stay GREEN, which is that
    division of labour showing up in the mutation table rather than only in
    this paragraph. An accept that only ever moves the first member reds all
    four.
    """
    _, _, _, moved = _multi_step(weights, seed, 1500)
    assert not ((moved > 0) & (moved < len(MULTI))).any(), (
        "a step moved some members and not others: the accept is being taken "
        "per member instead of once per block"
    )
    accepted = float((moved == len(MULTI)).mean())
    assert lo <= accepted <= hi


@pytest.mark.parametrize("seed", [0, 6])
def test_a_three_member_block_reproduces_the_dense_posterior_and_its_cross_terms(seed):
    """Width 3 against dense linear algebra, covariance included.

    Task 8 declined to extend the invariance guard past one member on the
    grounds that the only oracle sharing no linear algebra with the solve is
    1-D quadrature. True for INVARIANCE, and not true for this: a bug in the
    per-name comprehensions is not in the ``_env_before``/``isolate`` layer
    ``graph_oracle`` shares, so the dense route can see it.

    Run on the constant-sigma arm, where ``q`` is the exact conditional and
    every step accepts, so the output IS a draw of the block's posterior and
    the dense oracle is a statement about the same distribution. The
    off-diagonal is the clause that carries the weight: this fixture's mean is
    ``(a + b + c + d) X``, so the three members are EXACTLY collinear and the
    oracle covariance is ``0.6668 I - 0.3332 (J - I)`` -- every pair
    anticorrelated at -0.50. A ``proposed`` dict reading one member's draw for
    every name would leave all three columns identical, i.e. a correlation of
    +1, and a per-member key would leave them independent at 0. Marginal
    variances alone separate neither.

    Measured at 2000 keys, seed 0: means 1.2113/1.2042/1.2469 against the
    oracle's 1.2220/1.2220/1.2224, sds 0.7993/0.7982/0.8053 against 0.81655,
    off-diagonals -0.3084/-0.3271/-0.3202 against -0.33324.
    """
    graph, outside, out, moved = _multi_step((0.0,) * 4, seed, 2000)
    assert float((moved == len(MULTI)).mean()) > 0.99
    oracle = graph_oracle(graph, MULTI, at=outside)
    drawn = np.stack([np.asarray(out[name], dtype=float) for name in MULTI])
    reference = np.sqrt(np.diag(oracle.covariance))
    # 2000 draws give a 2.2% standard error on each sd and 0.0183 on each mean
    # in absolute units; the tolerances below are 4-5 of those.
    assert np.max(np.abs(drawn.mean(axis=1) - oracle.mean) / reference) < 0.10
    assert np.max(np.abs(drawn.std(axis=1) / reference - 1.0)) < 0.10
    covariance = np.cov(drawn)
    off = ~np.eye(len(MULTI), dtype=bool)
    scale = np.outer(reference, reference)
    departure = np.abs(covariance - oracle.covariance) / scale
    assert np.max(departure[off]) < 0.15


# --------------------------------------------------------------------------
# 7.2 (c): the self-normalised correction, against quadrature
# --------------------------------------------------------------------------


@pytest.mark.parametrize("draws, seed", [(2000, 0), (2000, 3)])
def test_the_weighted_posterior_matches_quadrature_on_a_scalar_block(draws, seed):
    """``gcr+snis`` end to end, against a reference with no linear algebra in it.

    ``radiometer`` is scalar, so its posterior integrates to eight digits on a
    grid: mean 2.94632609, sd 0.04665179. What this guard covers that
    ``test_the_weights_recover_the_posterior_the_gls_fixed_point_misses``
    does not is the **sd** -- that test compares means only -- and the fact
    that the comparison is absolute rather than paired against the unweighted
    draws.

    Deliberately NOT run through a long NUTS chain: P3a's record is that
    exact-vs-NUTS is a self-consistency check, both routes going through
    ``apply_probabilistic``, and this file's own header records two graphs in
    this repository where NUTS is the one that is wrong.

    Honest about its power: at ``kappa=0.05`` this fixture's Kish ESS/N is
    0.998, the log weights span under one nat, and the UNWEIGHTED draws are
    already within 0.04 sd -- so this guard is on the DRAWS and the GLS fixed
    point, not on the weights. ``steep_radiometer`` is where the weights
    matter and it is covered above and in ``test_dispatch_entry.py``.

    Measured over keys 0/3 at N=2000: weighted mean -0.017 to -0.003 posterior
    sd out, weighted sd ratio 0.998 to 1.005, Kish ESS/N 0.9984-0.9988.

    What it kills, measured: freezing the SNIS proposal at the block's prior
    centre instead of at the GLS fixed point -- which on ``radiometer`` means
    sigma at ``w = 0``, i.e. the floor 1e-3 alone and 148x too narrow -- reds
    every cell here. It reds on ``method``: the Kish ESS collapses so far that
    the floor fires and the dispatcher returns ``"nuts"``.
    """
    graph = radiometer()
    truth, spread, _, _ = quadrature(graph, "w", -4.0, 12.0)
    post = compile_graph(graph).sample(jax.random.key(seed), num_samples=draws)
    assert post.method == "gcr+snis"
    mean, drawn_sd = weighted_moments(post.samples["w"], post.log_weights)
    assert abs(float(mean[0]) - truth) / spread < 0.12
    assert float(drawn_sd[0]) / spread == pytest.approx(1.0, abs=0.08)
    assert post.ess / draws > 0.9


@pytest.mark.parametrize("draws, seed", [(2000, 0), (2000, 3)])
def test_the_weighted_posterior_matches_quadrature_on_a_two_dimensional_block(
    draws, seed
):
    """The same claim where the block has TWO members over TWO observed nodes.

    ``radiometer_group`` is the only fixture in this package with more than
    one domain leaf AND more than one codomain leaf AND a prediction-dependent
    sigma on one of them, so it is the smallest graph on which the weighted
    moments can be wrong per-member rather than uniformly. Its 2-D posterior
    still integrates: ``a`` 1.50474251 / 0.02126163, ``b`` -2.00315469 /
    0.02978981 at 801 grid points, which the 401 used here reproduces to
    8.5e-07 -- 4e-05 of the posterior sd being divided by.

    Measured over keys 0/3 at N=2000: ``a`` -0.028 to -0.007 sd out with sd
    ratio 1.006-1.011, ``b`` +0.007 to +0.024 sd out with sd ratio 1.006-1.016,
    Kish ESS/N 0.984-0.989.
    """
    graph = radiometer_group()
    truth = quadrature_pair(graph, ("a", "b"), ((0.6, 2.4), (-3.2, -0.8)))
    post = compile_graph(graph).sample(jax.random.key(seed), num_samples=draws)
    assert post.method == "gcr+snis"
    for name, (reference, spread) in truth.items():
        mean, drawn_sd = weighted_moments(post.samples[name], post.log_weights)
        assert abs(float(mean[0]) - reference) / spread < 0.12
        assert float(drawn_sd[0]) / spread == pytest.approx(1.0, abs=0.08)


# --------------------------------------------------------------------------
# 7.2 (d): the CG tolerance, at three points
# --------------------------------------------------------------------------

#: ``plated_radiometer(n=12)``: twelve dimensions and kappa 8049 at the GLS
#: fixed point. The dimension is what makes this the only shape in the suite
#: where ``tol`` is the binding stopping rule -- CG on an m-dimensional SPD
#: system is exact in m steps, so on ``radiometer_group``'s two-member block
#: the residual is 5e-15 at ``tol=1e-1`` and the sweep below would measure
#: nothing at all (measured: bitwise identical moments at 1e-1, 1e-3, 1e-6 and
#: 1e-12).
TOL_BLOCK = {"n": 12}
TOL_TIGHT = 1e-12


@pytest.mark.parametrize("tol, agrees", [(1e-1, False), (1e-6, True), (1e-12, True)])
@pytest.mark.parametrize("seed", [0, 5])
def test_weighted_moments_agree_at_tight_tolerances_and_move_at_a_loose_one(
    tol, agrees, seed
):
    """Three points, because two would prove nothing in either direction.

    The draft used ``{1e-6, 1e-12}``. Two problems, both fatal: the plan's own
    default here is ``CONVERGENCE_TARGET / 8049 = 1.24e-07``, i.e. BELOW the
    loose end of that pair, so the sweep ran entirely in the direction where
    nothing can happen; and two arms asserted to AGREE cannot detect a ``tol``
    that is ignored entirely, because then both are bit-identical and the test
    is green. ``1e-1`` is the third point and it must measurably DISAGREE.
    With the default at 1.24e-07 the three points bracket it, 1e-1 on one side
    and 1e-12 on the other.

    Sigma is frozen once at the tight-tolerance GLS fixed point and reused by
    every arm, so what is being swept is the CG tolerance of the SOLVE and the
    DRAW alone rather than the outer reweighting as well -- otherwise
    ``reweight_tol``'s ``max(8 eps, tol)`` default moves at the same time and
    the two knobs cannot be told apart.

    x64 throughout: at float32 the same block's CG plateaus near the precision
    floor and ``maxiter``, not ``tol``, becomes the stopping rule.

    Measured at 800 draws, seed 0, against the dense oracle at the same frozen
    sigma:

    ==========  =================  ========  ==================
    ``tol``     mean relative err  Kish/N    weighted sd ratio
    ==========  =================  ========  ==================
    ``1e-1``    4.164e-01          0.0026    0.046 .. 1.339
    ``1e-6``    3.175e-10          0.5986    0.935 .. 1.135
    ``1e-12``   1.637e-14          0.5986    0.935 .. 1.135
    ==========  =================  ========  ==================

    Section 5.5's failure mode in one row: an unconverged CG leaves the
    prior-dominated directions at their starting value, and the sd ratio's
    lower end falls to 0.046 -- a posterior 22x too NARROW, reported with no
    complaint.
    """
    count = 800
    with jax.enable_x64(True):
        graph = plated_radiometer(**TOL_BLOCK)
        names = ("z",)
        at = block_at(graph, names)
        block = unchecked_operator(graph, names, at)
        sigma = iterative_gls(
            block,
            sigma_from_graph(graph, at),
            depends_on_prediction=True,
            tol=TOL_TIGHT,
            require_convergence=None,
        ).noise_std
        oracle = graph_oracle(graph, names, at=at)
        # `sigma` is read BOTH ways here: `_dense_at` ravels it into the
        # analytic posterior's own sigma vector (values), and the solve needs
        # the operator. Converting the variable would break the oracle.
        reference, _, _ = _dense_at(oracle, graph, sigma)

        precision = diagonal_from(sigma)
        solution, _ = wiener_solve(
            block, precision=precision, tol=tol, require_convergence=None
        )
        error = float(
            np.max(np.abs(flat_domain(solution, names) - reference))
            / np.max(np.abs(reference))
        )
        draws, _ = jax.vmap(
            lambda one: gcr_sample(
                block, precision=precision, key=one, tol=tol, require_convergence=None
            )
        )(jax.random.split(jax.random.key(seed), count))
        log_weights = jax.vmap(
            lambda x: log_weight(
                graph, block, x, at=at, precision=precision, mu=solution
            )
        )(draws)
        ess = float(self_normalise(log_weights)[1]) / count

    if agrees:
        assert error < 1e-6
        assert ess > 0.4
    else:
        assert error > 0.05
        assert ess < 0.05


def _dense_at(oracle, graph, sigma):
    """The dense posterior of ``oracle``'s block at a sigma chosen by the caller.

    ``graph_oracle`` evaluates sigma at the block's own zero, which is the
    right default for a constant noise model and the wrong one when the point
    the solve froze at is the argument. Rebuilt here from the same design and
    data rather than by re-deriving them.
    """
    from tests.exact.oracle import analytic_posterior

    order = sorted(graph.observed)
    flat = np.concatenate([np.asarray(sigma[name]).ravel() for name in order])
    return analytic_posterior(
        oracle.design,
        oracle.offset,
        oracle.data,
        flat,
        oracle.prior_mean,
        oracle.prior_std,
    )


# --------------------------------------------------------------------------
# 7.2 (e): the composite sweep against its conditional, and against pure NUTS
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [-2.5, 0.0, 6.0])
def test_the_swept_block_reproduces_its_conditional_at_a_held_hmc_site(tau):
    """The conditional half of (e): deterministic, no chain, dense oracle.

    ``HMCGibbs`` calls ``gibbs_fn`` with whatever ``hmc_sites`` the inner
    kernel currently holds, and the contract is that the returned draw is from
    ``p(block | hmc_sites, d)``. Held at three values of ``tau`` spanning
    [-2.5, 6.0], across which the conditional mean travels 4.5 conditional sds
    -- so a sweep that ignored ``hmc_sites`` and used the prior centre 1.3
    instead would be 2.0 conditional sds out at ``tau = -2.5``, against a
    standard error of 0.018 sd on 3000 draws.

    Measured: oracle 1.861434 / 1.924345 / 2.075333 at the three points, all
    at conditional sd 0.047590; the draws land +0.0161 conditional sd out with
    a sd ratio of 1.0078 at every one of them.
    """
    graph = located_ancestor()
    at = {"tau": jnp.asarray(tau)}
    oracle = graph_oracle(graph, ("w",), at=at)
    step = gibbs_factory(graph, ("w",), tol=1e-9, method="gcr")
    drawn = np.asarray(
        jax.vmap(
            lambda one: step(
                rng_key=one, gibbs_sites={"w": jnp.asarray(0.0)}, hmc_sites=at
            )["w"]
        )(jax.random.split(jax.random.key(3), 3000)),
        dtype=float,
    )
    spread = float(np.sqrt(oracle.covariance[0, 0]))
    assert abs(drawn.mean() - float(oracle.mean[0])) / spread < 0.10
    assert drawn.std() / spread == pytest.approx(1.0, abs=0.06)


@pytest.mark.parametrize("seed", [0, 3])
def test_the_composite_and_pure_nuts_marginals_agree_where_the_outside_latent_mixes(
    seed,
):
    """The marginal half of (e), on a fixture whose outside latent was MEASURED
    to mix before the comparison was written.

    Both samplers are compared against 2-D quadrature rather than against each
    other, because two samplers agreeing is what ``mixed_radiometer`` does
    while both are wrong by 26.8 nats (see this module's header). The
    composite is not asked to beat NUTS, only to agree with the truth to the
    same tolerance.

    ``tau``'s ESS under the sweep is what licenses reading its marginal at
    all, so it is asserted rather than assumed: measured 532-570 of 1500 over
    keys 0 and 3, against pure NUTS's 1187-1401. The composite is WORSE at
    mixing the outside latent -- one Gibbs sweep per NUTS step gives ``tau`` a
    conditional that moves less -- and saying so is the point of asserting a
    floor rather than a ratio.

    Measured over keys 0/3 at 1500 draws, warmup 800, against quadrature's
    ``w`` 1.971882 / 0.048124 and ``tau`` 1.889048 / 0.284050: composite
    -0.043 to +0.000 sd on ``w`` and -0.001 to +0.015 on ``tau``; pure NUTS
    +0.011 to +0.014 and -0.002 to +0.052. Sd ratios 0.991-1.042 composite,
    0.970-1.047 NUTS.
    """
    draws = 1500
    graph = located_ancestor()
    truth = quadrature_pair(graph, ("w", "tau"), ((1.0, 3.0), (-3.0, 6.0)))
    plan = compile_graph(graph)
    assert plan.exact.latents == ("w",) and plan.sampled.latents == ("tau",)
    composite = plan.sample(jax.random.key(seed), num_samples=draws, num_warmup=800)
    pure = nuts_draws(
        graph,
        jax.random.key(seed),
        num_samples=draws,
        num_warmup=800,
        progress_bar=False,
    )
    assert chain_ess({"tau": composite.samples["tau"]}) > 0.2 * draws
    for name, (reference, spread) in truth.items():
        for label, samples in (("composite", composite.samples), ("nuts", pure)):
            values = np.asarray(samples[name], dtype=float)
            assert abs(values.mean() - reference) / spread < 0.15, label
            assert values.std() / spread == pytest.approx(1.0, abs=0.12), label


# --------------------------------------------------------------------------
# 7.2 (f): ESS per second -- a report, and the docstring says why
# --------------------------------------------------------------------------


def test_report_ess_per_second_without_asserting_a_threshold(capsys):
    """A report, not a guard, and the reason is that the metric is monotone in
    the wrong direction.

    A ``gibbs_fn`` that returns a cheap WRONG draw is faster and its ESS is
    higher, so the ratio goes UP. A performance assertion sitting beside a
    correctness question rewards breaking the correctness. It is also plainly
    unstable: measured over keys 0/11/30 at 1000 draws and 500 warmup --
    a wider sweep than the two keys and 600 draws this test itself runs --
    ``radiometer``'s in-block ratio reads 3.84x, 6.06x and 2.34x and
    ``straight_line``'s reads 11.04x, 10.96x and 9.83x, so an assertion of
    ">= 5.98x", which the spec's own benchmark C would have suggested, is
    already red at two of those six cells.

    Two directions do NOT improve and are printed for the same reason the wins
    are: on the mixed sweep the in-block ratio is 0.75-0.94x and the OUTSIDE
    parameter reads 0.25-0.44x, because one Gibbs sweep per NUTS step is
    strictly less exploration of ``tau`` than NUTS moving both together.

    ``num_samples``, ``num_warmup``, the seed and the dtype are pinned so the
    numbers mean something when they are read, and each timed call is run once
    first so what is measured is the run rather than the XLA compile.
    """
    draws, warmup = 600, 300
    rows = []
    for build, kwargs in (
        (straight_line, {}),
        (radiometer, {}),
        (located_ancestor, {}),
    ):
        graph = build(**kwargs)
        plan = compile_graph(graph)
        inside = plan.exact.latents
        outside = tuple(n for n in graph.latents if n not in set(inside))
        for seed in (0, 11):
            key = jax.random.key(seed)
            post, exact_seconds = _timed(
                plan.sample, key, num_samples=draws, num_warmup=warmup
            )
            chain, nuts_seconds = _timed(
                nuts_draws,
                graph,
                key,
                num_samples=draws,
                num_warmup=warmup,
                progress_bar=False,
            )
            rows.append(
                (
                    build.__name__,
                    seed,
                    post.method,
                    exact_seconds,
                    nuts_seconds,
                    _rate(post.samples, inside, exact_seconds),
                    _rate(chain, inside, nuts_seconds),
                    _rate(post.samples, outside, exact_seconds),
                    _rate(chain, outside, nuts_seconds),
                )
            )
    with capsys.disabled():
        print(f"\nESS/second, {draws} draws + {warmup} warmup, float32")
        for name, seed, method, te, tn, ie, inuts, oe, onuts in rows:
            print(
                f"  {name:18s} seed={seed} {method:9s} "
                f"exact {te:5.2f}s nuts {tn:5.2f}s  "
                f"in-block {ie:9.1f} vs {inuts:9.1f} ({ie / inuts:5.2f}x)"
                + (
                    ""
                    if math.isnan(oe)
                    else f"  outside {oe:8.1f} vs {onuts:8.1f} ({oe / onuts:5.2f}x)"
                )
            )
    for row in rows:
        assert row[3] > 0.0 and row[4] > 0.0
        assert math.isfinite(row[5]) and row[5] > 0.0


def _timed(call, *args, **kwargs):
    """``(result, seconds)``, discarding a first run so XLA compiles once first.

    Without the throwaway call the exact path -- whose ``vmap`` over
    ``gcr_sample`` traces a fresh program per shape -- pays its compile inside
    the measurement while numpyro's cached kernel does not, which is a
    comparison of compilers rather than of samplers.
    """
    call(*args, **kwargs)
    start = time.perf_counter()
    result = call(*args, **kwargs)
    return result, time.perf_counter() - start


def _rate(samples, names, seconds):
    """ESS per second over a subset of sites, or NaN where the subset is empty."""
    if not names:
        return float("nan")
    return chain_ess({name: samples[name] for name in names}) / seconds


# --------------------------------------------------------------------------
# 7.3: boundary validation -- the two thresholds that are real
# --------------------------------------------------------------------------

#: Every fixture whose exact block is solved at a CONSTANT sigma, so CG really
#: does stop at ``tol`` and ``residual * kappa`` is a bound on something. The
#: prediction-dependent ones are excluded deliberately and the reason is
#: measured: the plan derives kappa from sigma at the block's prior CENTRE
#: while the solve runs at the GLS fixed point, and the two differ by 232,000x
#: on ``radiometer`` (1.06e+10 against 4.58e+04), 120x on ``radiometer_group``,
#: 8.4x on ``plated_radiometer`` and 8.5x on ``steep_radiometer``. All four are
#: conservative -- kappa too LARGE, so tol too tight, which costs iterations
#: and not accuracy -- but the quantity is not bounded in that direction by
#: anything, and this file is not the place to fix it.
CONSTANT_SIGMA = [
    (straight_line, {}),
    (two_linear_latents, {}),
    (collinear_pair, {}),
    (two_observations, {}),
    (plated_latent, {}),
    (plated_and_scalar_latents, {}),
    (prior_held_direction, {}),
    (unconstrained_latent, {}),
    (many_observations, {"count": 1}),
    (many_observations, {"count": 5}),
    (wide_plate, {"size": 1}),
    (wide_plate, {"size": 1000}),
    (straight_line, {"sigma": 1e-3}),
    (straight_line, {"sigma": 1e3}),
    (straight_line, {"prior_std": 0.05, "sigma": 100.0}),
    (straight_line, {"prior_std": 1e3, "sigma": 1e-2}),
]


@pytest.mark.parametrize(
    "build, kwargs", CONSTANT_SIGMA, ids=lambda v: str(v) if isinstance(v, dict) else v
)
def test_the_derived_tol_delivers_its_target_at_every_extreme(build, kwargs):
    """Threshold one of section 7.3, and the extreme-parameter sweep, in one pass.

    ``tol = CONVERGENCE_TARGET / kappa_upper(kappa)`` is a PROMISE: solve to
    that residual and the relative ERROR is at most ``CONVERGENCE_TARGET``.
    Both the derivation and ``wiener_solve``'s own ``require_convergence``
    guard evaluate that promise through ``residual * condition_bound(...)``, so
    checking one against the other is checking a quantity against itself. The
    dense oracle shares the model and none of the linear algebra, which is what
    makes it able to falsify the promise rather than restate it.

    Measured, ``true error / CONVERGENCE_TARGET`` over the sixteen cells: 0 on
    eight of them, 9.4e-04 on ``plated_and_scalar_latents``, 1.39e-03 on
    ``unconstrained_latent`` and 0.1378 on ``collinear_pair``, the worst. So
    the promise holds with at least 7.3x of margin everywhere, across kappa
    from 1.0 (``prior_std=0.05, sigma=100``) to 5.8e+11
    (``prior_std=1e3, sigma=1e-2``).

    **Two-sided, and the other side is a different test.** A bound satisfied
    with 7.3x of margin could equally be satisfied by a ``tol`` that was
    ignored, so it needs a partner showing the promise BROKEN once ``tol`` is
    loosened past it: that is
    ``test_weighted_moments_agree_at_tight_tolerances_and_move_at_a_loose_one``'s
    ``1e-1`` arm, where the same measurement reads 4.164e-01, i.e. 416x the
    target. Spelling the loose arm here too would need a many-dimensional
    block, which every cell in this list but the wide plates is not: CG on an
    m-dimensional SPD system is exact in m steps.

    **The oracle clause carries this on its own**, which was measured rather
    than assumed: with ``tol_for`` mutated to MULTIPLY by kappa instead of
    dividing -- ``CONVERGENCE_TARGET``'s docstring names that inversion as the
    one that matters -- ten of these sixteen cells fail the oracle comparison,
    seven of them at a relative error of exactly 1.0 because CG returns its
    zero starting guess untouched. The six that survive are the low-kappa ones
    where ``TARGET * kappa`` is still a tight enough tol to be irrelevant, and
    they are why the ``tol`` algebra is asserted here as well.

    **The same cells are the extreme-parameter sweep, run through the two entry
    points a user has rather than through ``wiener_solve`` alone.**
    ``tests/exact/test_extremes.py`` already sweeps plate size, observed-node
    count, prior width, noise width and exact collinearity through
    ``wiener_solve`` and ``gcr_sample`` DIRECTLY; what it cannot see is the
    partition, the kappa probe, the derived ``tol``, ``estimate``'s reweighting
    wrapper and ``sample``'s dispatch. The extremes, and which cell is which:
    block size 1 (``wide_plate(1)``, the scalar-degenerate plate) and 1,000,
    with 10,000 in the test below; observed nodes 1 and 5
    (``many_observations``); kappa over twelve decades; sigma over six, 1e-3 to
    1e3; exactly collinear parents (``collinear_pair``, whose data cannot
    separate ``a`` from ``b`` at all); and a latent that reaches no observed
    node, so its column of the design is exactly zero and its posterior is its
    prior -- ``unconstrained_latent``'s ``u``, which must come back at
    1.37 +/- 0.75 and not at zero, and which the ``sample`` clause checks.

    Measured: every cell reports ``method="gcr"`` and 200 draws come back
    finite. ``estimate`` and the bare ``wiener_solve`` are both compared
    because they are not the same call -- ``estimate`` goes through
    ``iterative_gls``'s ``depends_on_prediction=False`` branch and promotes
    ``converged``.
    """
    graph = build(**kwargs)
    plan = compile_graph(graph)
    names = plan.exact.latents
    at = block_at(graph, names)
    block = unchecked_operator(graph, names, at)
    assert plan.exact.kappa is not None
    assert math.isfinite(kappa_upper(plan.exact.kappa))
    assert plan.exact.tol == pytest.approx(
        CONVERGENCE_TARGET / kappa_upper(plan.exact.kappa), rel=1e-6, abs=0.0
    )

    zero = {name: jnp.zeros_like(block.prior_mean[name]) for name in names}
    solution, _ = wiener_solve(
        block,
        precision=precision_at(graph, {**at, **zero}),
        tol=plan.exact.tol,
        require_convergence=None,
    )
    oracle = graph_oracle(graph, names, at=at)
    scale = float(np.max(np.abs(oracle.mean)))
    for values in (solution, plan.estimate().values):
        departure = np.max(np.abs(flat_domain(values, names) - oracle.mean))
        assert float(departure) / scale <= CONVERGENCE_TARGET

    post = plan.sample(jax.random.key(0), num_samples=200)
    assert post.method == "gcr"
    assert set(post.samples) == set(graph.latents)
    for drawn in post.samples.values():
        assert bool(np.all(np.isfinite(np.asarray(drawn))))


def test_a_ten_thousand_member_block_compiles_and_runs_against_a_closed_form():
    """The wide end of the block-size axis, with an O(n) reference.

    ``graph_oracle`` inverts a dense precision matrix, so it costs O(n**3) and
    at n = 10,000 that is the whole test's runtime (measured: 11.3 s a cell,
    against 1.3 s for everything else it was checking). ``wide_plate`` is
    ``z_i ~ N(0, tau)``, ``d_i ~ N(z_i, sigma)`` with no coupling between
    elements, so its posterior is available in closed form -- mean
    ``d_i tau**2 / (tau**2 + sigma**2)``, sd ``tau sigma / sqrt(tau**2 +
    sigma**2)`` -- which shares no linear algebra with anything and costs O(n).

    The SNIS dimension ceiling is why this cell is off the weighted path and
    stays there: :func:`~bayesmith.exact.correct.self_normalise`'s own
    measurements put Kish ESS/N at 1.00 by n = 500, so a 10,000-member block
    is only a meaningful question where no importance weight is involved.
    Here sigma is constant, the method is plain ``gcr``, and every draw is
    independent.

    Measured: compile 0.65 s, ``estimate`` 0.14 s, 200 draws 0.46 s -- so the
    widest block in this suite needs no ``slow`` marker.
    """
    size, tau, sigma = 10_000, 1.5, 0.4
    graph = wide_plate(size=size)
    plan = compile_graph(graph)
    assert plan.exact.latents == ("z",)
    shrink = tau**2 / (tau**2 + sigma**2)
    data = np.asarray(graph.node("d").observed, dtype=float)
    estimate = np.asarray(plan.estimate().values["z"], dtype=float)
    assert estimate.shape == (size,)
    assert np.max(np.abs(estimate - data * shrink)) < 1e-3 * np.max(np.abs(data))

    post = plan.sample(jax.random.key(0), num_samples=200)
    drawn = np.asarray(post.samples["z"], dtype=float)
    assert post.method == "gcr" and post.ess == 200.0
    assert bool(np.all(np.isfinite(drawn)))
    # 200 draws over 10,000 independent coordinates: the pooled sd is measured
    # to 0.05% and no per-coordinate claim is being made.
    closed_form = tau * sigma / math.sqrt(tau**2 + sigma**2)
    assert float(drawn.std(axis=0).mean()) == pytest.approx(closed_form, rel=0.05)


@pytest.mark.parametrize("seed", [0, 5])
def test_the_collapse_floor_flips_the_dispatch_at_the_measured_kish_ratio(seed):
    """Threshold two of section 7.3, bracketed at the value this run measures.

    Not at a hardcoded number: the floor is asked for at 0.7x and 1.3x of the
    Kish ESS/N this very run produced, so the separation does not depend on
    the ratio landing anywhere in particular -- only on 0.7 and 1.3 being
    either side of 1. Measured, ``plated_radiometer(n=6)`` reads 0.652, 0.696
    and 0.859 over keys 5, 0 and 2, so the bracket is a real interval and not
    a coincidence at one key.

    **The two branches do NOT agree at the threshold, and that is the finding
    rather than a defect in the test.** ``boundary-validation.md`` asks that
    both methods be evaluated at the dispatch point and compared, on the
    assumption that both are valid in a neighbourhood of it. Here they are
    not: per-coordinate quadrature of this fixture says the SNIS answer is
    within 0.14 posterior sd on every coordinate at every key, while the NUTS
    that used to replace it is off by up to 204 sd on coordinate 1 -- the
    ``sigma = 0.06|z| + 0.002`` funnel, which NUTS does not traverse. At the
    genuinely collapsed cell the ordering is the same: ``n=25, kappa=0.4``
    reads Kish ESS/N 0.012-0.029, and there SNIS is out by at most 1.40 sd
    against NUTS's 18.5.

    **That disagreement is now acted on rather than only recorded.** When this
    test was first written the floor DISCARDED the weighted sample, so these
    numbers sat in a docstring saying the dispatcher preferred the worse
    answer at both cells; :data:`~bayesmith.dispatch.plan.SNIS_ESS_FLOOR` now
    annotates instead, and substituting NUTS is ``nuts_on_collapse=True``. So
    what the floor decides is the VERDICT -- ``unreliable`` and the reason --
    and the ``nuts_on_collapse`` arm below is where the substitution is still
    checked to happen when it is asked for. Both arms are swept, because a
    floor wired to the keyword rather than to the ratio would satisfy either
    one alone.
    """
    draws = 1200
    graph = plated_radiometer(n=6)
    key = jax.random.key(seed)
    measured = compile_graph(graph).sample(
        key, num_samples=draws, num_warmup=600, ess_floor=0.0
    )
    assert measured.method == "gcr+snis"
    assert not measured.reason.startswith("exact block ['z']: the SNIS")
    ratio = measured.ess / draws

    below = compile_graph(graph).sample(
        key, num_samples=draws, num_warmup=600, ess_floor=ratio * 0.7
    )
    above = compile_graph(graph).sample(
        key, num_samples=draws, num_warmup=600, ess_floor=ratio * 1.3
    )
    assert below.method == "gcr+snis"
    assert below.unreliable is measured.unreliable
    assert "collapsed" not in below.reason
    assert above.method == "gcr+snis"
    assert above.unreliable is True
    assert above.log_weights is not None
    assert "Kish ESS/N" in above.reason and "collapsed" in above.reason
    assert "ESS/N this floor reads FALLS" in above.reason

    replaced = compile_graph(graph).sample(
        key,
        num_samples=draws,
        num_warmup=600,
        ess_floor=ratio * 1.3,
        nuts_on_collapse=True,
    )
    kept = compile_graph(graph).sample(
        key,
        num_samples=draws,
        num_warmup=600,
        ess_floor=ratio * 0.7,
        nuts_on_collapse=True,
    )
    assert replaced.method == "nuts"
    assert kept.method == "gcr+snis"

    # The kept side is the accurate one, per coordinate, against quadrature of
    # the plate's own independent factors: element i of `plated_radiometer`
    # touches only `d_i`, so sweeping `z_i` with the rest held anywhere gives
    # that element's exact marginal up to the constant normalisation removes.
    base = jnp.zeros((6,))
    mean, spread = weighted_moments(measured.samples["z"], measured.log_weights)
    for index in range(6):
        reference, width, _, _ = quadrature(
            graph,
            "z",
            -12.0,
            12.0,
            4001,
            place=lambda value, i=index: {"z": base.at[i].set(value)},
        )
        assert abs(float(mean[index]) - reference) / width < 0.35
        assert float(spread[index]) / width == pytest.approx(1.0, abs=0.25)


# --------------------------------------------------------------------------
# B10's acceptance, which nothing demonstrated until now.
# --------------------------------------------------------------------------
#
# The migration spec's §五 B10 states it in one line: "一个非高斯节点能产出
# **抽样**，而不只是一个 loss". rheplicant's posterior engines hard-code a
# Gaussian observation site, so a Poisson or Student-t likelihood can drive an
# optimisation there and cannot produce draws or enter a SamplingPlan. The
# claim here is that a `Probabilistic` node is the right seam and dispatch
# routes what it cannot solve to NUTS.
#
# Mechanically that has been true since the classifier existed. It was never
# DEMONSTRATED: no test took a non-Gaussian observed node all the way to
# samples, so the acceptance rested on reading the dispatch table rather than
# on running it. "Assumed to hold, with no guard" is the state this file
# exists to convert into a measurement.


def _poisson_graph(rate=12.0, n=60, seed=1):
    """Counts from a Poisson whose rate is the exponential of a latent."""
    counts = jax.random.poisson(jax.random.key(seed), rate, (n,))

    def model():
        log_rate = sample("log_rate", lambda: dist.Normal(0.0, 5.0))
        mu = det("mu", lambda r: jnp.exp(r) * jnp.ones(n), log_rate)
        observe(
            "d",
            lambda m: dist.Poisson(m).to_event(1),
            mu,
            depends_on_prediction=True,
            obs=counts,
        )

    return trace(model)


def _student_t_graph(n=60, seed=7):
    """A straight line observed through heavy tails."""
    x = jnp.linspace(0.0, 1.0, n)
    y = 2.0 + 3.0 * x + 0.3 * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("x", x)
        a = sample("a", lambda: dist.Normal(0.0, 10.0))
        b = sample("b", lambda: dist.Normal(0.0, 10.0))
        mu = det("mu", lambda a_, b_, xx: a_ + b_ * xx, a, b, xs)
        observe(
            "d",
            lambda m: dist.StudentT(4.0, m, 0.3).to_event(1),
            mu,
            depends_on_prediction=False,
            obs=y,
        )

    return trace(model)


def test_a_poisson_observation_produces_draws_and_they_match_quadrature():
    """B10's acceptance, on the shape that also has a prediction-dependent
    variance -- a Poisson's mean IS its variance, so this is not merely a
    non-Gaussian density but one the exact path could not fake.

    The oracle is this file's own grid quadrature of ``log_joint``, which
    shares the model and nothing else. Compared in posterior sd, which is
    the only scale on which "the sampler found the right distribution"
    means anything.
    """
    with jax.enable_x64(True):
        graph = _poisson_graph()
        plan = compile_graph(graph)
        assert plan.exact is None, str(plan)
        assert "not a diagonal Gaussian" in str(plan)
        posterior = plan.sample(
            jax.random.key(0), num_samples=2000, num_warmup=1000
        )
        drawn = np.asarray(posterior.samples["log_rate"])
        mean, spread, _, _ = quadrature(graph, "log_rate", 1.5, 3.5, points=4001)
    assert drawn.shape == (2000,)
    assert abs(float(drawn.mean()) - mean) < 0.15 * spread, (drawn.mean(), mean)
    assert float(drawn.std()) == pytest.approx(spread, rel=0.15)


def test_a_student_t_observation_produces_draws_on_both_of_its_latents():
    """The second shape, and a two-latent one, so the claim is not about a
    single scalar site.

    Heavy tails rather than a different mean-variance link: between them the
    two fixtures cover both ways a node can fail to be a diagonal Gaussian.
    """
    with jax.enable_x64(True):
        graph = _student_t_graph()
        plan = compile_graph(graph)
        assert plan.exact is None, str(plan)
        posterior = plan.sample(
            jax.random.key(0), num_samples=2000, num_warmup=1000
        )
        drawn = {k: np.asarray(v) for k, v in posterior.samples.items()}
        marginals = quadrature_pair(
            graph, ("a", "b"), ((1.0, 3.0), (2.0, 4.0)), points=401
        )
    # `quadrature_pair` returns {name: (mean, sd)}; zipping over it walks its
    # KEYS, which read as one-character strings and index like them.
    for name in ("a", "b"):
        mean, spread = marginals[name]
        assert drawn[name].shape == (2000,)
        assert abs(float(drawn[name].mean()) - mean) < 0.15 * spread, (
            name,
            drawn[name].mean(),
            mean,
        )
        assert float(drawn[name].std()) == pytest.approx(spread, rel=0.2)
