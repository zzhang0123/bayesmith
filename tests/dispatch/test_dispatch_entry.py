"""``plan.sample()`` and ``plan.estimate()`` -- the two entry points that run.

Every other module in ``dispatch/`` decides; this one is where the decision is
carried out, so the failure mode this file exists to prevent is a dispatcher
whose *printed* plan and *executed* run disagree -- a plan that says "GCR +
SNIS" over a weighted sample worth four draws out of a thousand, an ``ess``
that averages a stuck site's 3 with a healthy one's 40, or a ``converged=False``
turned into a number and handed back.

**Three measurements this file is built on, all taken before it was written.**

*The SNIS collapse is real, exponential, and invisible to k-hat.* Sweeping
only the plate of ``plated_radiometer`` at N=2000, Kish ESS/N reads 0.966
(n=1), 0.739 (n=6), 0.445 (n=12), 0.123 (n=50), 0.054 (n=100), 0.0039 (n=400).
Meanwhile ``radiometer()`` -- a ONE-dimensional block whose Kish ESS/N is
0.9988 at every N and seed tried -- reads k-hat 0.91 to 1.80 at N=1200 across
six seeds, i.e. "unreliable" on a sample that is essentially perfect. The two
diagnostics disagree by that much, which is why the fallback in
``test_the_collapse_floor_is_read_off_the_kish_ess_and_not_off_khat`` must be
driven by the Kish ESS and never by k-hat.

*The convergence guard cannot be left on by default at float32.* ``tol`` is
derived as ``CONVERGENCE_TARGET / kappa`` precisely so that a CG stopping at
``tol`` delivers ``CONVERGENCE_TARGET``, so ``require_convergence=
CONVERGENCE_TARGET`` compares two numbers that were constructed to be equal
and fires on rounding: measured, ``straight_line`` lands at ``1.078e-07 *
924.4 = 9.97e-04`` (accepted, by 0.3%) and ``two_linear_latents`` at
``2.179e-07 * 5792 = 1.262e-03`` (refused, by 26%) -- two fixtures of the same
shape on opposite sides of the guard. It is exposed as a keyword and off by
default; ``test_the_convergence_guard_is_off_by_default_but_reachable`` pins
both halves of that.

*The mixed sweep's ESS really does separate the reductions.* On
``mixed_radiometer`` at key 0, warmup 200, 400 draws: ESS(w)=350.6,
ESS(tau)=150.2. MIN is 150.2 and MEAN is 250.4 -- 67% apart, so the
min-vs-mean mutation is not a close call.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.diagnostics import effective_sample_size

from bayesmith import compile as compile_graph
from bayesmith import const, det, observe, sample, trace
from bayesmith.dispatch import execute as execute_module
from bayesmith.dispatch.plan import (
    SNIS_ESS_FLOOR,
    Estimate,
    Posterior,
    chain_ess,
)
from bayesmith.errors import ConvergenceError
from bayesmith.exact.correct import self_normalise
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.graph.evaluate import log_joint
from tests.exact.models import (
    bilinear_pair,
    mixed_radiometer,
    plated_and_scalar_latents,
    plated_latent,
    plated_radiometer,
    radiometer,
    radiometer_group,
    steep_radiometer,
    straight_line,
    two_linear_latents,
    two_observations,
)
from tests.exact.oracle import flat_domain, graph_oracle

#: Where the collapse fixture sits, both sides measured. ``kappa=0.4`` on a
#: 25-element plate reads a Kish ESS/N of at most **0.0301** over 18 cells
#: (N in {300, 600, 1200} x six keys), i.e. never closer than 3.3x BELOW
#: :data:`SNIS_ESS_FLOOR`; the plain ``n=6`` fixture reads at least **0.720**
#: over the same 18 cells, 7.2x ABOVE it. The floor sits inside a factor-24
#: gap, which is what makes this a region rather than a point.
COLLAPSED = {"n": 25, "kappa": 0.4}
HEALTHY = {"n": 6}


def detached_ancestor(*, n=8, w_true=1.4, sigma=0.5, seed=33):
    """A mixed graph whose outside latent never reaches the data. LOCAL.

    Defined here rather than in ``tests/exact/models.py`` because it exists for
    one dimension of ONE test in this file: every mixed fixture that module
    ships reports ``sigma_needs_rebuild=True`` (measured over all seven), so
    the ``False`` branch of ``sample()``'s ``sigma_rebuild=`` wiring has no
    fixture anywhere in the suite.

    ``z`` is Gaussian and would qualify on its own, but it is an ancestor of
    ``q``'s prior width, so the ejection rule sends it to NUTS -- and it sits
    on no path to ``d``, so ``noise_std`` really can be hoisted out of the
    sweep. ``q`` is Gaussian, unconstrained by data, and vacuously declares
    ``linear_in``, so it joins ``w`` in the exact block: the partition comes out
    ``{q, w}`` exact by plain ``gcr``, ``{z}`` by NUTS.

    Every latent stays GAUSSIAN on purpose. ``sigma_rebuild=False`` hoists
    sigma at the graph's prior centre, and ``gibbs._prior_centre`` refuses a
    latent with no ``loc`` -- so a non-Gaussian outside latent, the other
    obvious way to force one to NUTS, cannot reach this branch at all.
    """
    x = jnp.linspace(1.0, 3.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        z = sample("z", lambda: dist.Normal(0.6, 0.4))
        sample("q", lambda t: dist.Normal(0.0, jnp.abs(t) + 0.2), z)
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def quadrature(graph, name, lo, hi, points=4001):
    """The true 1-D posterior of ``name``, by grid quadrature of ``log_joint``.

    Independent of every solver under test: it evaluates the graph's own joint
    on a grid and normalises. The only thing it shares with ``sample()`` is
    ``log_joint``, which is also what NUTS targets, so a disagreement here is
    a disagreement about the posterior and not about a convention.
    """
    grid = jnp.linspace(lo, hi, points)
    log_p = jax.vmap(lambda value: log_joint(graph, {name: value}))(grid)
    density = jnp.exp(log_p - jnp.max(log_p))
    grid, density = np.asarray(grid, float), np.asarray(density, float)
    density /= np.trapezoid(density, grid)
    mean = np.trapezoid(grid * density, grid)
    return mean, float(np.sqrt(np.trapezoid((grid - mean) ** 2 * density, grid)))


def every_coordinate_ess(samples):
    """numpyro's ESS for every site and every coordinate, as a flat list.

    Deliberately NOT :func:`~bayesmith.dispatch.plan.chain_ess`: the reduction
    is the thing under test, so the reference is spelled out a second time and
    left unreduced, for a caller to reduce whichever way it means to.
    """
    return [
        value
        for draws in samples.values()
        for value in np.asarray(
            effective_sample_size(np.asarray(draws)[None]), dtype=float
        ).ravel()
    ]


# --------------------------------------------------------------------------
# the fully exact path: iid draws, no chain
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [straight_line, two_linear_latents, plated_latent, plated_and_scalar_latents,
     two_observations],
)
@pytest.mark.parametrize("num_samples,seed", [(200, 0), (500, 3)])
def test_a_fully_exact_graph_samples_without_a_chain(build, num_samples, seed):
    """The unambiguous win: iid draws, ESS = N, no warmup, nothing to diagnose.

    ``ess`` is N EXACTLY here rather than an estimate, and that is a claim
    about the algorithm rather than a shortcut --
    ``test_the_iid_path_really_is_iid`` is what stops it being a lie.

    Structural coverage: one scalar member (``straight_line``,
    ``two_observations``), two scalar members (``two_linear_latents``), one
    plated member of six (``plated_latent``), and a plate plus a scalar in the
    same block (``plated_and_scalar_latents``); one observed node and two.
    """
    graph = build()
    post = compile_graph(graph).sample(
        jax.random.key(seed), num_samples=num_samples
    )
    assert isinstance(post, Posterior)
    assert post.method == "gcr"
    assert post.log_weights is None
    assert post.khat is None
    assert post.ess == pytest.approx(float(num_samples), rel=1e-9)
    assert not post.unreliable
    assert set(post.samples) == set(graph.latents)
    for name, draws in post.samples.items():
        assert draws.shape[0] == num_samples
        assert bool(jnp.all(jnp.isfinite(draws)))
        assert name in post.reason or "sigma does not move" in post.reason


@pytest.mark.parametrize(
    "build", [straight_line, two_linear_latents, plated_latent,
              plated_and_scalar_latents, two_observations]
)
@pytest.mark.parametrize("num_samples,seed", [(2000, 0), (2000, 5)])
def test_the_iid_path_really_is_iid(build, num_samples, seed):
    """``ess = num_samples`` is only honest if the draws carry no correlation.

    Two statistics, because the obvious one is too noisy to assert on. Over
    ten keys and these five fixtures, the estimator's spread at N=500 is
    0.476-1.284 of N per coordinate -- the MIN over a six-element plate is the
    minimum of six noisy estimates and drifts down with the plate size, which
    is a property of the estimator and not of the draws. At N=2000 the
    per-coordinate MEAN lands in 0.886-1.023 over the same fifty cells, and
    the lag-1 autocorrelation of every coordinate stays under 0.069 against
    the white-noise band ``3/sqrt(N) = 0.067``.

    The autocorrelation is the load-bearing half: it is what "iid" actually
    says, it is distribution-free, and unlike the ESS ratio it cannot be
    satisfied by a degenerate sample -- a constant array reads ``nan`` on
    both, and a chain reads a positive ``r``.
    """
    post = compile_graph(build()).sample(jax.random.key(seed),
                                         num_samples=num_samples)
    per_coordinate = every_coordinate_ess(post.samples)
    assert np.all(np.isfinite(per_coordinate))
    assert 0.75 <= float(np.mean(per_coordinate)) / num_samples <= 1.25
    for draws in post.samples.values():
        flat = np.asarray(draws).reshape(num_samples, -1)
        flat = flat - flat.mean(axis=0)
        lag_one = (flat[:-1] * flat[1:]).sum(axis=0) / (flat * flat).sum(axis=0)
        assert np.all(np.abs(lag_one) < 4.0 / np.sqrt(num_samples))


@pytest.mark.parametrize(
    "build", [straight_line, two_linear_latents, plated_latent,
              plated_and_scalar_latents]
)
def test_the_iid_draws_reproduce_the_dense_posterior(build):
    """The draws are of the right distribution, not merely of the right shape.

    Checked against ``tests/exact/oracle.py``'s dense linear-Gaussian
    posterior, which forms the design matrix column by column and inverts it
    -- no CG, no autodiff, no shared code with the solve. For a constant-sigma
    whole-graph block that oracle IS the posterior, so both moments are pinned:
    a mean-only check passes for a sampler that returns the Wiener mean
    N times.
    """
    graph = build()
    plan = compile_graph(graph)
    post = plan.sample(jax.random.key(0), num_samples=4000)
    oracle = graph_oracle(graph, plan.exact.latents)
    got_mean = flat_domain(
        {n: v.mean(axis=0) for n, v in post.samples.items()}, plan.exact.latents
    )
    got_sd = flat_domain(
        {n: v.std(axis=0) for n, v in post.samples.items()}, plan.exact.latents
    )
    want_sd = np.sqrt(np.diag(oracle.covariance))
    assert np.allclose(got_mean, oracle.mean, atol=0.06 * want_sd.max())
    assert np.allclose(got_sd, want_sd, rtol=0.08)


# --------------------------------------------------------------------------
# the SNIS path, and the collapse it has to refuse
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,kwargs",
    [(radiometer, {}), (radiometer_group, {}), (steep_radiometer, {}),
     (plated_radiometer, HEALTHY)],
)
@pytest.mark.parametrize("num_samples,seed", [(600, 0), (1200, 4)])
def test_a_moving_sigma_returns_weights_and_the_kish_ess_of_them(
    build, kwargs, num_samples, seed
):
    """``gcr+snis``: the draws, the weights that correct them, and their ESS.

    ``ess`` must be the Kish ESS **of the weights this object carries**, not
    of some other set -- so it is recomputed here from ``post.log_weights``
    through ``self_normalise``. Structural coverage: one scalar member
    (``radiometer``, ``steep_radiometer``), two scalar members over two
    observed nodes (``radiometer_group``), one plated member of six.
    """
    graph = build(**kwargs)
    post = compile_graph(graph).sample(jax.random.key(seed),
                                       num_samples=num_samples)
    assert post.method == "gcr+snis"
    assert post.log_weights is not None
    assert post.log_weights.shape == (num_samples,)
    _, kish = self_normalise(post.log_weights)
    assert post.ess == pytest.approx(float(kish), rel=1e-6)
    assert post.ess / num_samples >= SNIS_ESS_FLOOR
    assert post.khat is not None
    assert set(post.samples) == set(graph.latents)


@pytest.mark.parametrize("num_samples,seed", [(600, 0), (600, 3), (1500, 1)])
def test_the_weights_recover_the_posterior_the_gls_fixed_point_misses(
    num_samples, seed
):
    """The correction must actually correct, checked against 1-D quadrature.

    ``steep_radiometer`` is the fixture whose frozen-sigma proposal is a
    VISIBLE approximation. Measured over twelve cells (N in {600, 1500} x six
    keys), in units of the true posterior sd: the UNWEIGHTED draws' mean is
    0.166 to 0.325 away from the quadrature truth and their sd is 1.107 to
    1.167 times too WIDE, while the weighted mean is 0.004 to 0.060 away. So
    the two bands do not touch, and the paired form below (weighted error at
    most half the unweighted one) holds with a 1.43x margin in its worst cell.

    A paired comparison rather than two absolute ones because both estimates
    are computed from the SAME draws: whatever Monte-Carlo error the sample
    happens to carry is common to them and cancels.
    """
    graph = steep_radiometer()
    post = compile_graph(graph).sample(jax.random.key(seed),
                                       num_samples=num_samples)
    draws = np.asarray(post.samples["w"])
    weights = np.asarray(jax.nn.softmax(post.log_weights))
    truth, spread = quadrature(graph, "w", -4.0, 9.0)
    unweighted = abs(float(draws.mean()) - truth) / spread
    weighted = abs(float((weights * draws).sum()) - truth) / spread
    assert weighted < 0.10 < unweighted
    assert weighted < 0.5 * unweighted


@pytest.mark.parametrize("num_samples,seed", [(400, 0), (800, 2)])
def test_a_collapsed_snis_falls_back_instead_of_relabelling_one_draw(
    num_samples, seed
):
    """§6.4's decision: below the floor the dispatcher must not hand SNIS back.

    ``plated_radiometer(n=25, kappa=0.4)`` reads a Kish ESS/N of at most
    0.0301 over eighteen (N, key) cells, so the fallback fires with 3.3x of
    margin. The fallback target is NUTS rather than the Gibbs+MH path (B) and
    that is FORCED, not chosen: ``gibbs.assemble`` refuses a block covering
    every latent, in those words, because the inner NUTS kernel would have no
    site left to sample.

    The paired ``HEALTHY`` cell is the other side of the floor, on the same
    fixture family with only the plate size and ``kappa`` moved.
    """
    key = jax.random.key(seed)
    fell = compile_graph(plated_radiometer(**COLLAPSED)).sample(
        key, num_samples=num_samples, num_warmup=num_samples
    )
    assert fell.method == "nuts"
    assert fell.log_weights is None
    assert fell.khat is None
    assert "Kish ESS/N" in fell.reason and "NUTS" in fell.reason
    assert fell.ess > 0.0

    stayed = compile_graph(plated_radiometer(**HEALTHY)).sample(
        key, num_samples=num_samples
    )
    assert stayed.method == "gcr+snis"
    assert "Kish ESS/N" in stayed.reason


@pytest.mark.parametrize("floor,expect", [(0.99, "nuts"), (0.01, "gcr+snis")])
def test_the_floor_is_what_decides_and_it_is_two_sided_on_one_fixture(
    floor, expect
):
    """Both verdicts on ONE graph, moved only by the floor.

    ``plated_radiometer(n=6)``'s Kish ESS/N sits at 0.72-0.86, i.e. strictly
    between these two floors, so the fixture is held fixed and the threshold
    is what crosses it. Neither floor equals :data:`SNIS_ESS_FLOOR`, so this
    cannot pass by the default happening to be right.
    """
    post = compile_graph(plated_radiometer(**HEALTHY)).sample(
        jax.random.key(0), num_samples=400, num_warmup=400, ess_floor=floor
    )
    assert post.method == expect


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_the_collapse_floor_is_read_off_the_kish_ess_and_not_off_khat(seed):
    """The two diagnostics disagree, and only one of them may drive dispatch.

    Measured on ``radiometer()`` -- a one-dimensional block -- at N=1200 over
    six keys: Kish ESS/N is 0.9986-0.9988 every time while k-hat reads 0.91 to
    1.80, i.e. past :data:`~bayesmith.exact.correct.FINITE_MEAN_KHAT` and so
    ``unreliable=True``, on a weighted sample that has discarded essentially
    nothing. The log weights span under one nat there, which is a tail PSIS
    has no business fitting a generalised Pareto to.

    So: ``unreliable`` may be True and the dispatcher must still return SNIS.
    A fallback wired to ``unreliable`` instead of to the ESS sends this graph
    to NUTS and reports the reason as a collapse that did not happen.
    """
    post = compile_graph(radiometer()).sample(jax.random.key(seed),
                                              num_samples=1200)
    assert post.method == "gcr+snis"
    assert post.ess / 1200 > 0.99
    assert post.unreliable is True


# --------------------------------------------------------------------------
# the mixed path, the all-NUTS path, and the ESS reduction they share
# --------------------------------------------------------------------------


def test_ess_reduces_over_every_site_and_coordinate():
    """``ess`` is one float, so the reduction has to be written down: MIN.

    Measured in the spec's benchmark C: ESS(logw)=3.0 and ESS(alm,min)=40.2
    coexist in one run. This field exists to make dividing by N a deliberate
    act, so it must report the WORST of them -- a mean would let a
    well-mixing site hide a stuck one.

    The fixture separates the three candidate reductions by a wide margin, so
    this is not a close call: at this key ESS(w)=350.6 and ESS(tau)=150.2, so
    MIN is 150.2, MEAN 250.4 and MAX 350.6.
    """
    post = compile_graph(mixed_radiometer()).sample(
        jax.random.key(0), num_warmup=200, num_samples=400
    )
    per_site = {
        name: float(jnp.min(effective_sample_size(np.asarray(draws)[None])))
        for name, draws in post.samples.items()
    }
    assert len(per_site) > 1
    average = sum(per_site.values()) / len(per_site)
    assert min(per_site.values()) < 0.8 * average < max(per_site.values())
    assert post.ess == pytest.approx(min(per_site.values()), rel=1e-6)


def test_a_mixed_graph_runs_the_sweep_the_plan_printed():
    """What ran is what the plan said would run, down to the gibbs sites.

    ``mixed_radiometer``'s ``mu`` is a Deterministic, and numpyro's
    ``get_samples`` returns it alongside the two latents -- measured, shape
    (400, 10). It is dropped here: ``Posterior.samples`` is the posterior over
    LATENTS, and leaving ``mu`` in silently changes ``ess`` (its ESS tracks
    ``w``'s 350.6, not ``tau``'s 150.2, so the min is unaffected here -- but
    only by luck).
    """
    graph = mixed_radiometer()
    plan = compile_graph(graph)
    post = plan.sample(jax.random.key(0), num_warmup=200, num_samples=400)
    assert plan.exact.method == "gcr+mh"
    assert post.method == "gcr+mh"
    assert post.log_weights is None
    assert set(post.samples) == set(graph.latents) == {"w", "tau"}
    assert "mu" not in post.samples
    assert all(v.shape[0] == 400 for v in post.samples.values())
    assert "gibbs_sites=['w']" in post.reason


@pytest.mark.parametrize(
    "build,method,rebuild",
    [(mixed_radiometer, "gcr+mh", True), (detached_ancestor, "gcr", False)],
)
def test_the_sweep_is_handed_the_numbers_the_plan_decided(
    monkeypatch, build, method, rebuild
):
    """What ``assemble`` receives is what ``str(plan)`` printed, field by field.

    A spy rather than a statistical check, deliberately: the effect of getting
    ``sigma_rebuild`` wrong is already measured where it belongs
    (``tests/exact/test_gibbs.py``, on ``contrast_sigma_pair``: hoisting where
    a rebuild was needed gives a posterior 1.64x and 7.35x too wide), so what
    is left unguarded is only the WIRING -- and a Gibbs posterior compared
    against an oracle is a far noisier instrument for that than reading the
    argument.

    Two-sided on ``sigma_rebuild``, which is why ``detached_ancestor`` exists:
    **every** mixed fixture in ``tests/exact/models.py`` reports
    ``sigma_needs_rebuild=True`` (measured over all seven of them), because
    ``_sigma_needs_rebuild`` is an over-approximation and in each of them the
    outside latent does reach the data. It is also the only cell here where
    the sweep is plain ``gcr`` rather than ``gcr+mh``.
    """
    seen: dict = {}
    real = execute_module.assemble

    def recording(graph, names, **kwargs):
        seen.update(names=tuple(names), **kwargs)
        return real(graph, names, **kwargs)

    monkeypatch.setattr(execute_module, "assemble", recording)
    plan = compile_graph(build())
    plan.sample(jax.random.key(0), num_warmup=60, num_samples=60)
    assert seen["names"] == plan.exact.latents
    assert seen["tol"] == plan.exact.tol
    assert seen["method"] == plan.exact.method == method
    assert seen["sigma_rebuild"] == plan.sigma_needs_rebuild == rebuild


@pytest.mark.parametrize("seed", [0, 4])
def test_the_sweep_returns_both_halves_of_a_mixed_graph_correctly(seed):
    """End to end on ``detached_ancestor``: the block against a dense oracle,
    and the sampled latent against its own prior.

    ``z`` is not an ancestor of any observed node, so its posterior IS its
    prior, ``N(0.6, 0.4)`` -- a value nothing in the block can produce by
    accident, and one that a sweep returning the wrong site or dropping the
    inner NUTS kernel entirely would miss. ``w``'s posterior is unaffected by
    ``z`` (sigma is constant and ``q`` decouples), so the dense oracle applies
    to it exactly. Bands are 4 Monte-Carlo standard errors, computed from each
    site's OWN measured ESS rather than from ``num_samples``, so a badly mixing
    chain widens its own band instead of failing on the mixing.
    """
    graph = detached_ancestor()
    plan = compile_graph(graph)
    post = plan.sample(jax.random.key(seed), num_warmup=500, num_samples=1500)
    assert post.method == "gcr"
    assert set(post.samples) == {"w", "q", "z"}
    per_site = {
        name: float(jnp.min(effective_sample_size(np.asarray(draws)[None])))
        for name, draws in post.samples.items()
    }
    oracle = graph_oracle(graph, plan.exact.latents, at={"z": jnp.asarray(0.6)})
    want = dict(zip([n for n, _ in oracle.order], oracle.mean, strict=True))
    want_sd = dict(
        zip([n for n, _ in oracle.order], np.sqrt(np.diag(oracle.covariance)),
            strict=True)
    )
    drawn_w = np.asarray(post.samples["w"])
    assert abs(drawn_w.mean() - want["w"]) < 4 * want_sd["w"] / np.sqrt(per_site["w"])
    assert drawn_w.std() == pytest.approx(want_sd["w"], rel=0.2)
    drawn_z = np.asarray(post.samples["z"])
    assert abs(drawn_z.mean() - 0.6) < 4 * 0.4 / np.sqrt(per_site["z"])
    assert drawn_z.std() == pytest.approx(0.4, rel=0.2)


def test_a_graph_with_no_exact_structure_says_nuts_and_says_why():
    """No exact block: NUTS, with the classifier's own refusal carried over.

    ``bilinear_pair``'s members are printed by the plan whatever the reason
    says, so "gain in reason" is checked against the refusal text and not
    against a member list.
    """
    graph = bilinear_pair()
    plan = compile_graph(graph)
    post = plan.sample(jax.random.key(0), num_warmup=200, num_samples=400)
    assert plan.exact is None
    assert post.method == "nuts"
    assert post.log_weights is None
    assert set(post.samples) == set(graph.latents)
    assert "linear_in" in post.reason
    assert "Kish ESS/N" not in post.reason
    assert 0.0 < post.ess <= 400.0


@pytest.mark.parametrize(
    "draws,expect",
    [
        ({"a": np.full((40,), 2.0)}, 1.0),
        ({"a": np.array([np.nan] * 40)}, 1.0),
    ],
)
def test_a_site_that_never_moved_reports_one_draw_rather_than_nan(draws, expect):
    """A constant chain has numpyro's ESS return ``nan``, which breaks ``min``.

    Measured: ``mixed_radiometer`` at key 2 leaves ``w`` bitwise constant over
    400 sweeps, so ``effective_sample_size`` divides 0 by 0 and the site the
    reduction most needs to report becomes the one value ``min`` silently
    steps over -- ``min(nan, 150.2)`` is ``nan`` or ``150.2`` depending on
    argument order. Mapped to 1.0 instead: a chain that never moved carries
    exactly one draw's worth of information, which is the smallest ESS a
    non-empty sample can have, so it wins the ``min`` as it should.
    """
    assert chain_ess(draws) == pytest.approx(expect)


def test_the_reduction_takes_the_worst_coordinate_within_one_site():
    """Per-COORDINATE, not per-site: one stuck entry of a plate must show.

    A site whose first coordinate is a constant and whose second is iid noise
    has ESS 1 and ~N; ``min`` over sites alone would average nothing and
    report whichever the site-level estimator produced. Built here rather than
    sampled so the two coordinates are unambiguous.
    """
    rng = np.random.default_rng(11)
    draws = np.stack([np.full(500, 0.7), rng.normal(size=500)], axis=1)
    assert chain_ess({"z": draws}) == pytest.approx(1.0)
    assert chain_ess({"z": draws[:, 1:]}) > 100.0


# --------------------------------------------------------------------------
# estimate()
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build", [straight_line, two_linear_latents, plated_latent, two_observations]
)
def test_estimate_solves_a_constant_sigma_graph_in_one_pass(build):
    """A constant sigma has no fixed point to find, so there is nothing to loop.

    ``iterations == 1`` is the live half of this: routing a constant-sigma
    graph through the reweighting loop returns the same answer after
    ``MIN_REWEIGHTS`` solves, so only the count can tell the two apart.
    """
    graph = build()
    plan = compile_graph(graph)
    got = plan.estimate()
    assert isinstance(got, Estimate)
    assert got.converged is True
    assert int(got.iterations) == 1
    oracle = graph_oracle(graph, plan.exact.latents)
    values = flat_domain(got.values, plan.exact.latents)
    assert np.allclose(values, oracle.mean,
                       atol=2e-3 * max(1.0, float(np.abs(oracle.mean).max())))


@pytest.mark.parametrize("build", [radiometer, steep_radiometer, plated_radiometer])
def test_estimate_returns_a_covariance_that_is_a_fixed_point(build):
    """The defining property of the GLS answer, checked as such.

    ``noise_std`` must be the sigma the graph's own rule gives AT the returned
    solution -- that is what "fixed point" means, and it is what a caller who
    feeds ``Estimate.noise_std`` back into ``gcr_sample`` is relying on.
    Recomputed here through ``noise_std_at``, which the estimate did not use
    directly.

    ``iterations > 1`` separates this from the constant-sigma branch above.
    """
    graph = build()
    got = compile_graph(graph).estimate()
    assert got.converged is True
    assert int(got.iterations) > 1
    recomputed = noise_std_at(graph, dict(got.values))
    for name, sigma in got.noise_std.items():
        assert np.allclose(np.asarray(sigma), np.asarray(recomputed[name]),
                           rtol=2e-3)


@pytest.mark.parametrize("build", [radiometer, steep_radiometer])
@pytest.mark.parametrize("max_reweights", [1, 2, 3])
def test_estimate_raises_convergence_error_rather_than_returning_a_number(
    build, max_reweights
):
    """P3a defined ConvergenceError and left it unraised, by design.

    ``iterative_gls`` returns ``converged`` as a field and leaves promotion to
    its caller. This is that caller -- the first one in the package.

    ``min_reweights=1`` is passed alongside because ``iterative_gls`` refuses
    ``min_reweights > max_reweights`` with a ``GraphError``, and its default
    minimum is 5; without it this call fails for the wrong reason. The
    negative control is
    ``test_estimate_returns_a_covariance_that_is_a_fixed_point``, which runs
    the same two graphs at the defaults and does NOT raise.
    """
    plan = compile_graph(build())
    with pytest.raises(ConvergenceError) as caught:
        plan.estimate(min_reweights=1, max_reweights=max_reweights,
                      reweight_tol=1e-14)
    assert "reweight_tol" in str(caught.value)
    assert str(max_reweights) in str(caught.value)


@pytest.mark.parametrize("build", [mixed_radiometer, bilinear_pair])
def test_estimate_refuses_a_mixed_graph_and_says_where_to_go(build):
    """No point estimate exists for a graph part of which is only samplable.

    Both shapes refuse: a graph with an exact block and a NUTS one
    (``mixed_radiometer``), and one with no exact block at all
    (``bilinear_pair``). The message must name ``sample`` -- a refusal that
    does not say what to do instead is where a user goes and writes their own
    alternating solve, which is rheplicant's motivating failure.
    """
    with pytest.raises(NotImplementedError, match="sample"):
        compile_graph(build()).estimate()


def test_the_convergence_guard_is_off_by_default_but_reachable():
    """Measured, and the reason it is off is measured too.

    ``tol = CONVERGENCE_TARGET / kappa`` is constructed so that a CG stopping
    exactly at ``tol`` delivers exactly ``CONVERGENCE_TARGET``, so switching
    the guard on at that same target makes it fire on rounding:
    ``two_linear_latents`` -- a two-parameter toy -- lands at residual
    2.179e-07 against a condition bound of 5792, i.e. 1.262e-03 against a
    1e-03 target, and is refused, while ``straight_line`` lands at 9.97e-04
    and is accepted. Both halves are asserted, so this is not "the guard is
    absent" but "the guard is a keyword and the keyword works".
    """
    import equinox as eqx

    plan = compile_graph(two_linear_latents())
    assert plan.estimate().converged is True
    with pytest.raises(eqx.EquinoxRuntimeError):
        plan.estimate(require_convergence=1e-3)


def test_the_printed_plan_does_not_claim_a_guard_the_run_leaves_off():
    """Print and run must agree about the guard, which is the whole file's thesis.

    ``guard_hoisted`` is False for a whole-graph plan -- there is no sweep to
    hoist anything out of -- and the plan used to read that as "convergence
    guard on". It is not on: both entry points default
    ``require_convergence=None`` for the reason
    ``test_the_convergence_guard_is_off_by_default_but_reachable`` measures.
    Checked against the SIGNATURE defaults rather than against another string,
    so re-enabling the guard without updating the line, or restoring the line
    without enabling the guard, each go red.
    """
    import inspect

    plan = compile_graph(two_linear_latents())
    assert plan.guard_hoisted is False
    printed = str(plan)
    assert "convergence guard on" not in printed
    assert "off by default" in printed
    for entry in (plan.sample, plan.estimate):
        parameter = inspect.signature(entry).parameters["require_convergence"]
        assert parameter.default is None, entry.__name__


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build,kwargs",
    [(two_linear_latents, {}), (steep_radiometer, {}),
     (plated_radiometer, HEALTHY)],
)
def test_the_same_key_gives_the_same_posterior(build, kwargs):
    """Nothing in the entry point reads a global RNG."""
    plan = compile_graph(build(**kwargs))
    first = plan.sample(jax.random.key(7), num_samples=300)
    again = plan.sample(jax.random.key(7), num_samples=300)
    assert first.ess == again.ess
    for name, draws in first.samples.items():
        assert bool(jnp.all(draws == again.samples[name]))
