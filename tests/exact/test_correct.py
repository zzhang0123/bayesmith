"""Importance weights, self-normalisation, and the two SNIS diagnostics.

Every weight test compares against `tests.exact.oracle`, which builds the same
posterior by probing `g` on a basis and solving the normal equations with
`numpy.linalg` -- no `jax.linearize`, no `jax.vjp`, no CG. The one place that
is not enough is the DROPPED constant `C`: it never appears in `correct.py`'s
body, so an oracle comparison of the returned value cannot see it. That claim
is checked against `scipy.stats.multivariate_normal` instead (guaranteed
present -- jax itself requires `scipy>=1.15`), which supplies the
`-(n/2) log 2pi - 1/2 log det Sigma` normalisation from an implementation this
package shares nothing with.
"""

import math
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.stats import multivariate_normal

from bayesmith import const, det, observe, sample, trace
from bayesmith.exact.block import domain_zero, unchecked_operator
from bayesmith.exact.correct import (
    FINITE_MEAN_KHAT,
    khat,
    log_weight,
    self_normalise,
    unreliable,
)
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.graph.evaluate import log_joint
from tests.exact.models import (
    plated_and_scalar_latents,
    radiometer,
    two_linear_latents,
    two_observations,
)
from tests.exact.oracle import flat_domain, graph_oracle


def three_scalar_latents(
    *, n=9, a_true=1.7, b_true=-0.6, c_true=2.3, sigma=0.45, seed=31
):
    """``d ~ N(a X + b Y + c, sigma)`` -- THREE members in one block.

    Local to this module rather than added to `tests/exact/models.py`, which
    another P3b task is editing concurrently.

    Every multi-member fixture in `models.py` has exactly two members
    (`two_linear_latents`, `radiometer_group`, `plated_and_scalar_latents`,
    `prior_held_direction`, `collinear_pair`). :func:`log_weight` reduces over
    members with a `sum(... for name in delta)`, and a reduction that handles
    two members can still mishandle three -- the P3a lesson that a fix correct
    on two members failed on three. Measured here: an implementation summing
    only the first leaf gets 1.743 where the truth is 130.838, against 4.741
    versus 10.869 on the two-member fixture, so this is also the sharper of
    the two probes.

    The prior means are 0.3, -1.1 and 0.7 -- all NONZERO and all different, so
    a `delta` that forgot to subtract `mu` is a different number rather than
    the same one, and `Y = (linspace)**2` keeps the two covariates
    non-collinear so the three columns of `A` are genuinely independent.
    """
    x = jnp.linspace(-2.0, 2.5, n)
    y = jnp.linspace(0.5, 3.5, n) ** 2
    truth = a_true * x + b_true * y + c_true
    data = truth + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        ys = const("Y", y)
        a = sample("a", lambda: dist.Normal(0.3, 4.0))
        b = sample("b", lambda: dist.Normal(-1.1, 2.0))
        c = sample("c", lambda: dist.Normal(0.7, 6.0))
        mu = det(
            "mu",
            lambda a_, b_, c_, x_, y_: a_ * x_ + b_ * y_ + c_,
            a,
            b,
            c,
            xs,
            ys,
            linear_in=("a", "b", "c"),
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def _mu_and_draw(block):
    """A proposal centre and one draw, with every element displaced differently.

    ``mu`` sits at the prior centre plus a per-MEMBER shift and the draw adds a
    per-ELEMENT step, so no two leaves and no two entries of one leaf move by
    the same amount. An implementation that reduces over only the first leaf,
    or that broadcasts one entry across a plate, then lands on a different
    number instead of a coincidentally equal one.
    """
    mu, draw = {}, {}
    for position, name in enumerate(block.names):
        shape = block.shape[name]
        size = int(np.prod(shape, dtype=int))
        step = np.arange(1, size + 1, dtype=float).reshape(shape)
        centre = np.asarray(block.prior_mean[name], dtype=float) + 0.37 * (position + 1)
        mu[name] = jnp.asarray(centre)
        draw[name] = jnp.asarray(centre + 0.19 * step + 0.11 * position)
    return mu, draw


# ---------------------------------------------------------------- log_weight


def test_log_weight_equals_log_p_minus_log_q_against_a_dense_gaussian():
    """The sign of the quadratic term, pinned against a dense reference.

    Both P3's spec and P3b's first draft wrote
    `C = 1/2 log det M - (n/2) log 2pi`, which has BOTH signs inverted:
    `log q(y) = -1/2 r^T M r + 1/2 log det M - (n/2) log 2pi`, so in
    `log w = log p - log q` the quadratic and the log-det must carry OPPOSITE
    signs.

    This test compares the DIFFERENCE between two draws' log weights, which is
    the only thing either consumer uses -- self-normalisation and the MH ratio
    both cancel any constant. That makes it insensitive to `C` by
    construction, so `test_the_documented_constant_makes_the_weight_the_log_evidence`
    is what pins `C` and this one pins the quadratic. Measured here: the
    differences agree BITWISE (relative error exactly 0.0), while the
    log_joint differences that would remain if the quadratic vanished are
    4.5e-4 of them -- six orders above the tolerance asserted.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = unchecked_operator(graph, ["w"])
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        oracle = graph_oracle(graph, ["w"])
        mu = {"w": jnp.asarray(oracle.mean[0])}
        draws = jnp.asarray([0.4, 1.1, 2.7])

        got = jax.vmap(
            lambda x: log_weight(
                graph, block, {"w": x}, at={}, noise_std=sigma, mu=mu
            )
        )(draws)

        precision = oracle.precision[0, 0]
        expect = []
        for x in [0.4, 1.1, 2.7]:
            lp = float(log_joint(graph, {"w": jnp.asarray(x)}))
            lq = (
                -0.5 * precision * (x - oracle.mean[0]) ** 2
                + 0.5 * np.log(precision)
                - 0.5 * np.log(2 * np.pi)
            )
            expect.append(lp - lq)
        expect = np.asarray(expect)
    got = np.asarray(got)
    assert np.allclose(got - got[0], expect - expect[0], rtol=1e-10)


def test_the_documented_constant_makes_the_weight_the_log_evidence():
    """`C` itself, which no comparison of the RETURNED value can reach.

    `C` is dropped, so it appears only in :func:`log_weight`'s docstring --
    and a docstring is exactly where the inverted sign survived two drafts.
    Made checkable by choosing a case where the whole weight has a known
    closed form: `plated_and_scalar_latents` has a CONSTANT sigma, so the
    frozen-sigma proposal `q = N(oracle.mean, oracle.precision^-1)` is the
    EXACT conditional, and `log p(x, d) - log q(x)` is therefore `log p(d)` --
    the marginal likelihood -- for every `x` alike.

    So the test makes two independent claims at once:

    * `log_weight(x) + C` does not depend on `x` (three displaced draws,
      including `x = mu`), which no wrong quadratic can satisfy;
    * its value is the evidence `N(d; A m + offset, A S A^T + N)`, which
      scipy computes with its own normalisation and which no wrong `C` can
      satisfy.

    Measured: the three draws agree with the evidence to 4.7e-15, 1.1e-14 and
    2.1e-16 relative. The draft's `C` would be off by 0.341 on an evidence of
    -8.333 -- 4.1% -- because `log det M = 9.530` and `5 log 2pi = 9.189` are
    NOT equal here, which is the coincidence this fixture had to avoid.
    """
    with jax.enable_x64(True):
        graph = plated_and_scalar_latents()
        names = ("z", "w")
        block = unchecked_operator(graph, names)
        sigma = noise_std_at(graph, domain_zero(block))
        oracle = graph_oracle(graph, names)
        dimension = oracle.mean.size
        mu = {"z": jnp.asarray(oracle.mean[:-1]), "w": jnp.asarray(oracle.mean[-1])}

        _, logdet = np.linalg.slogdet(oracle.precision)
        constant = -0.5 * logdet + 0.5 * dimension * math.log(2.0 * math.pi)

        offsets = [
            {"z": jnp.asarray([0.3, -0.7, 1.1, -0.2]), "w": jnp.asarray(0.45)},
            {"z": jnp.asarray([-1.4, 0.8, 0.05, 2.2]), "w": jnp.asarray(-1.3)},
            {"z": jnp.zeros(4), "w": jnp.asarray(0.0)},
        ]
        weights = [
            float(
                log_weight(
                    graph,
                    block,
                    {name: mu[name] + offset[name] for name in mu},
                    at={},
                    noise_std=sigma,
                    mu=mu,
                )
            )
            + constant
            for offset in offsets
        ]

    marginal_covariance = (
        oracle.design @ np.diag(oracle.prior_std**2) @ oracle.design.T
        + np.diag(oracle.sigma**2)
    )
    evidence = float(
        multivariate_normal.logpdf(
            oracle.data,
            mean=oracle.offset + oracle.design @ oracle.prior_mean,
            cov=marginal_covariance,
        )
    )
    # Guard on this test's own power: if these two happened to be equal, an
    # inverted C would be invisible here and the assertions below would be
    # decoration.
    assert abs(logdet - dimension * math.log(2.0 * math.pi)) > 0.1
    for weight in weights:
        assert weight == pytest.approx(evidence, rel=1e-9)


@pytest.mark.parametrize(
    "builder, names, at, members, sizes, observed",
    [
        (radiometer, ("w",), {}, 1, [1], 1),
        (two_observations, ("w",), {}, 1, [1], 2),
        (two_linear_latents, ("a",), {"b": -2.9}, 1, [1], 1),
        (plated_and_scalar_latents, ("z", "w"), {}, 2, [4, 1], 1),
        (three_scalar_latents, ("a", "b", "c"), {}, 3, [1, 1, 1], 1),
    ],
    ids=[
        "one-scalar-member-prediction-dependent-sigma",
        "one-member-two-observed-nodes",
        "one-member-with-a-latent-held-in-at",
        "two-heterogeneous-members-one-a-plate-of-4",
        "three-scalar-members",
    ],
)
def test_the_quadratic_is_half_delta_M_delta_on_every_block_shape(
    builder, names, at, members, sizes, observed
):
    """`log_weight - log_joint` is `1/2 delta^T M delta`, over four dimensions.

    The reduction inside :func:`log_weight` runs over block MEMBERS and, per
    member, over the ELEMENTS of that member's leaf; the operator it reduces
    runs over the OBSERVED nodes; and the graph scan runs over `at`. All four
    are swept here, and the member count reaches three, because a sum that is
    right on one member is right on none of the others for free -- measured,
    an implementation reducing only the first leaf gets 4.741 instead of
    10.869 on the two-member case and 1.743 instead of 130.838 on the
    three-member one, while being exactly right on all three single-member
    rows.

    A REGION, not a point, in one more sense: the frozen sigma spans 1e-3
    (`radiometer` at its zero, where `M` is 1.06e8) to 0.9
    (`two_observations`, where the quadratic is 6.46), so the agreement is not
    an artefact of one operator scale. Relative error measured across the five
    rows: 0.0, 4.1e-16, 6.2e-16, 0.0, 0.0.
    """
    with jax.enable_x64(True):
        graph = builder()
        at = {name: jnp.asarray(value) for name, value in at.items()}
        block = unchecked_operator(graph, names, at)
        sigma = noise_std_at(graph, {**at, **domain_zero(block)})
        oracle = graph_oracle(graph, names, at=at)
        mu, draw = _mu_and_draw(block)

        quadratic = float(
            log_weight(graph, block, draw, at=at, noise_std=sigma, mu=mu)
        ) - float(log_joint(graph, {**at, **draw}))
        delta = flat_domain(draw, block.names) - flat_domain(mu, block.names)

    # The parametrisation's own structural claims, so a fixture that quietly
    # changed shape could not leave a dimension unswept while still passing.
    assert len(block.names) == members
    assert [int(np.prod(block.shape[n], dtype=int)) for n in block.names] == sizes
    assert len(graph.observed) == observed
    assert quadratic == pytest.approx(
        float(0.5 * delta @ oracle.precision @ delta), rel=1e-8
    )


# ------------------------------------------------------------ self_normalise


def test_kish_ess_is_one_when_a_single_draw_carries_all_the_weight():
    """The degenerate end, which is where SNIS actually fails.

    Measured in the spec's dimension sweep: a mild radiometer at n=500 gives
    Kish ESS 1.00 out of 40000 draws. This is not a pathological input, it is
    what self-normalised importance sampling does as the number of mismatched
    coordinates grows, and the dispatcher has to be able to see it.

    Four draws, so `ess = len(log_weights)` is 4 rather than 1 -- the count
    and the answer must not coincide or this end pins nothing.
    """
    log_w = jnp.asarray([0.0, -400.0, -400.0, -400.0])
    weights, ess = self_normalise(log_w)
    assert float(ess) == pytest.approx(1.0, abs=1e-6)
    assert float(jnp.max(weights)) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("count", [7, 1000])
def test_kish_ess_is_n_when_every_weight_is_equal(count):
    """The other end. Without it, `ess = 1.0` unconditionally passes above.

    `rel=1e-6`, not tighter: the suite runs at float32 and `softmax` of seven
    zeros gives 6.9999990, a relative departure of 1.4e-7 that is arithmetic
    rather than a defect. Two counts so the answer tracks N rather than
    matching one fixture's length.
    """
    _, ess = self_normalise(jnp.zeros(count))
    assert float(ess) == pytest.approx(float(count), rel=1e-6)


@pytest.mark.parametrize(
    "log_w, expect",
    [
        ([0.0, math.log(3.0)], 1.6),
        ([0.0, math.log(3.0), math.log(4.0)], 32.0 / 13.0),
        ([0.0, 0.0, math.log(2.0)], 8.0 / 3.0),
    ],
)
def test_kish_ess_between_the_ends(log_w, expect):
    """Kish's formula in the interior, at hand-computed values.

    The two end tests are each satisfied by a wrong formula -- `1/sum(w)` is
    1.0 at the degenerate end, `len(log_weights)` is N at the equal end -- and
    they only kill each other's mutant because they disagree elsewhere. These
    three points are that elsewhere, written out rather than recomputed:
    weights `(1,3)/4` give `1/(1/16+9/16) = 1.6`, `(1,3,4)/8` give `64/26`,
    and `(1,1,2)/4` give `1/0.375`. None equals the number of draws.
    """
    _, ess = self_normalise(jnp.asarray(log_w))
    assert float(ess) == pytest.approx(expect, rel=1e-6)


def test_the_weight_constant_cancels_between_draws():
    """The dropped `C` is invisible to both consumers -- with real teeth.

    :func:`log_weight` returns the weight up to a constant, which is only
    sound because self-normalisation is shift-invariant. Asserting that
    against `exp`-then-divide would be arithmetic trivia, so the shift here is
    +800: measured, `exp` of these shifted weights overflows float64 (asserted
    below, so the stress is a fact about the input and not a claim about it),
    while `softmax` moves the normalised weights by 2.4e-14 and the ESS by
    1.4e-15.

    Runs under x64 with the arrays built inside the block, because at float32
    adding 800 to 0.4 costs 6e-5 of resolution and the invariance would be
    measurable at 1e-4 for reasons that have nothing to do with the estimator.
    """
    with jax.enable_x64(True):
        log_w = jnp.asarray([0.4, -1.7, 2.9, 0.0, -5.2])
        weights, ess = self_normalise(log_w)
        shifted_weights, shifted_ess = self_normalise(log_w + 800.0)
        with np.errstate(over="ignore"):
            naive = np.exp(np.asarray(log_w, dtype=np.float64) + 800.0)

    assert not np.isfinite(naive.sum())
    assert np.allclose(
        np.asarray(shifted_weights), np.asarray(weights), rtol=1e-11, atol=0.0
    )
    assert float(shifted_ess) == pytest.approx(float(ess), rel=1e-11)


# ------------------------------------------------------------------ khat


@pytest.mark.parametrize(
    "seed, n, spread, lo, hi",
    [
        (0, 2000, 0.5, 0.05, 0.26),
        (1, 2000, 0.5, 0.08, 0.29),
        (0, 200, 0.5, 0.20, 0.35),
        (0, 20000, 0.5, -0.10, 0.05),
    ],
)
def test_khat_pins_the_private_numpyro_entry_point(seed, n, spread, lo, hi):
    """`_psis_khat` is private, so its existence is pinned deliberately.

    The RANGE, not a point, and `(seed, n)` are parameters rather than
    constants: measured over 20 seeds at n=2000 the Gaussian case has mean
    0.076, sd 0.066 and spans -0.017..0.184, and it is strongly N-dependent at
    the same spread (200 -> 0.275, 2000 -> 0.156, 20000 -> -0.028). An earlier
    draft pinned 0.184 -- the MAXIMUM over those 20 seeds -- which is a test
    that fails on almost any other seed. Every band here clears its measured
    value by at least 0.07.

    Two-sided in N deliberately: 2000 sits between the 200 and 20000 rows, so
    the sweep has an interior rather than an endpoint default.
    """
    log_w = jax.random.normal(jax.random.key(seed), (n,)) * spread
    assert lo < khat(log_w) < hi


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_khat_crosses_the_finite_mean_band_on_heavy_tailed_weights(seed):
    """The other end of k-hat, which is the end the diagnostic exists for.

    Every band above sits near zero, so a `khat` that returned a constant 0.15
    would pass all of them. Widening the log-weight spread from 0.5 to 3.0 at
    the same n=2000 moves it to 1.185, 1.236 and 1.209 on these three seeds --
    past :data:`~bayesmith.exact.correct.FINITE_MEAN_KHAT`, i.e. past where
    the importance-sampling MEAN stops existing, not merely its variance.

    n is fixed at 2000 here rather than swept: measured at n=8000 the same
    three seeds give 1.001, 0.875 and 1.115, which straddles 1.0 and would
    make the moment-band assertion a coin flip rather than a claim.
    """
    log_w = jax.random.normal(jax.random.key(seed), (2000,)) * 3.0
    value = khat(log_w)
    assert 1.10 < value < 1.32
    assert value > FINITE_MEAN_KHAT
    assert unreliable(value, 2000) is True


def test_khat_is_none_rather_than_an_exception_when_the_private_entry_is_gone():
    """A private upstream name disappearing must degrade, not crash."""
    with mock.patch.dict(sys.modules, {"numpyro.infer.importance": None}):
        assert khat(jnp.zeros(100)) is None


# ------------------------------------------------------------- unreliable


@pytest.mark.parametrize(
    "khat_value, n, expect",
    [
        # Below N = 2154 the formula binds and a hard-wired 0.7 is optimistic:
        # the threshold is 0.500 at N=100 and 0.667 at N=1000.
        (0.60, 100, True),
        (0.40, 100, False),
        (0.68, 1000, True),
        (0.55, 1000, False),
        # Above it the cap binds: without `min(..., 0.7)` the threshold would
        # be 0.800 at N=1e5 and 0.833 at N=1e6, and both True rows flip.
        (0.75, 100_000, True),
        (0.75, 1_000_000, True),
        (0.65, 1_000_000, False),
        # At or below KHAT_MIN_DRAWS the formula is <= 0 and everything is
        # unreliable; just above it the threshold is 0.0397.
        (0.02, 1, True),
        (0.02, 10, True),
        (0.02, 11, False),
    ],
)
def test_unreliable_uses_a_sample_size_dependent_threshold(khat_value, n, expect):
    """Vehtari et al. (2024)'s `min(1 - 1/log10(N), 0.7)`, on both branches.

    N spans 1 to 1e6 and crosses the 2154 changeover in both directions, and
    every row has a partner of the opposite verdict at the same N, so neither
    `return True` nor `return False` survives. No `khat_value` here is 0.7:
    a mutation that hard-wires the cap must not be able to hide behind a probe
    that happens to sit on it.
    """
    assert unreliable(khat_value, n) is expect


def test_unreliable_abstains_rather_than_condemning_when_khat_is_unavailable():
    """`None` is "no diagnostic", and it must outrank the small-N branch.

    The second assertion is the one with content: with the `None` test placed
    after the `n <= KHAT_MIN_DRAWS` branch instead of before it, an
    unavailable k-hat at three draws would be reported as an established
    failure rather than as a missing measurement.
    """
    assert unreliable(None, 2000) is False
    assert unreliable(None, 3) is False
