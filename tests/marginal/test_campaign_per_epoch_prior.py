"""A per-epoch nuisance whose declared prior varies across epochs.

``epoch_terms`` read one entry of each per-epoch latent's declared prior and
broadcast it to every epoch. For a campaign declaring ``tau = [0.5, 1, 2, 4]``
that means three of the four epochs were integrated against a prior the model
does not declare, and what came back was a finite, plausible, wrong marginal
likelihood -- measured at 2.75 nats over four epochs, with no warning and no
refusal.

The comment at the reduction site acknowledged the limitation and named no
ledger row, which is the shape of thing that stays true for a release. The
information was never missing: ``gaussian_parts`` returns the whole per-epoch
vector and the code took ``[0]``; ``nuisance_prior`` already broadcasts a
per-component std; and ``one_epoch`` is already ``vmap``ped, so the prior only
had to stop being a closed-over scalar and start being a mapped argument.

**The declaration has three dimensions, and a fixture that varies one of them
convicts one third of the repair.** The first version of this file varied the
WIDTH only, and an adversarial review put half the bug back -- ``prior_locs``
rebuilt from ``loc.ravel()[0]``, the MEAN collapsed exactly as before -- and
ran the whole fast layer green, 3257 passed, while being wrong by 4.221 nats on
a fixture below. A second mutant deleted the ``[1:]`` from the broadcast target
and also passed 3257. So each of the three is now varied and asserted
separately: width, mean, and component width.

**The oracles here share nothing with the implementation.** Each epoch's
marginal is a Gaussian written out by hand and summed in Python. No QR, no
pivots, no offset arithmetic, and no ``SqrtInfo``.
"""

from __future__ import annotations

import math
import pathlib
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import bayesmith
from bayesmith import det, observe, plate, sample, trace
from bayesmith.marginal import compress_campaign

EPOCHS = 4
SIGMA = 0.55
PRIOR_MEAN = 0.4
DATA = (0.3, 1.1, 0.4, 0.8)

#: Four different declared widths. The first is the one the old code used for
#: all of them, which is why it is the one a wrong implementation reproduces.
HETEROGENEOUS = (0.5, 1.0, 2.0, 4.0)
HOMOGENEOUS = (0.5, 0.5, 0.5, 0.5)

#: Four different declared means, for the half of the repair that the width
#: fixtures above leave completely unasserted. Again the first entry is the one
#: a collapsing implementation reproduces.
HETEROGENEOUS_MEANS = (0.4, -0.7, 1.3, 0.05)


def _campaign(widths, means=None):
    """`n` is per-epoch and declares its own prior in each epoch.

    Both parameters are ``(E,)`` arrays. The spelling with a SCALAR mean and a
    vector width is refused -- pinned, with the frame that refuses it, by
    :func:`test_a_scalar_mean_beside_a_vector_width_is_refused`.
    """
    taus = jnp.asarray(widths)
    locs = jnp.full((EPOCHS,), PRIOR_MEAN) if means is None else jnp.asarray(means)
    data = jnp.asarray(DATA)

    def model():
        epoch = plate("epoch", EPOCHS)
        g = sample("g", lambda: dist.Normal(0.0, 2.0))
        n = sample("n", lambda: dist.Normal(locs, taus), plate=epoch)
        mu = det(
            "mu", lambda g_, n_: 2 * g_ + n_, g, n, plate=epoch, linear_in=("g", "n")
        )
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data, plate=epoch)

    return trace(model)


def _dense(g_value, widths, means=None):
    """``sum_e log N(d_e | 2g + m_e, tau_e^2 + sigma^2)``, by hand."""
    locs = (PRIOR_MEAN,) * len(widths) if means is None else means
    total = 0.0
    for observation, tau, mean in zip(DATA, widths, locs, strict=True):
        variance = tau**2 + SIGMA**2
        residual = observation - (2.0 * g_value + mean)
        total += -0.5 * (
            residual * residual / variance
            + np.log(variance)
            + np.log(2.0 * np.pi)
        )
    return float(total)


@pytest.mark.parametrize("g_value", [0.0, 1.5])
def test_each_epoch_is_integrated_against_the_width_it_declares(g_value):
    with jax.enable_x64(True):
        term = compress_campaign(_campaign(HETEROGENEOUS), "epoch")
        found = float(term.log_prob({"g": jnp.asarray(g_value)}))
        assert found == pytest.approx(_dense(g_value, HETEROGENEOUS), abs=1e-9)


@pytest.mark.parametrize("g_value", [0.0, 1.5, -2.0])
def test_each_epoch_is_integrated_against_the_mean_it_declares(g_value):
    """The other half of the declaration, and the half a review had to find.

    Every campaign fixture in this package declares ONE mean for the whole
    campaign, so ``prior_locs`` could go on being rebuilt from
    ``loc.ravel()[0]`` -- the original bug, in the new plumbing -- and pass
    3257 tests. An adversarial review measured that mutant at 4.221 nats from
    the dense truth; collapsing the mean in the ORACLE alone, which is what
    :func:`test_the_wrong_answer_is_far_enough_away_to_fail_on` measures, moves
    it 3.789 nats at ``g = -2`` and 1.162 at ``g = 0``. Two different pairs of
    quantities, both far outside any tolerance here.
    """
    with jax.enable_x64(True):
        term = compress_campaign(
            _campaign(HETEROGENEOUS, HETEROGENEOUS_MEANS), "epoch"
        )
        found = float(term.log_prob({"g": jnp.asarray(g_value)}))
        expected = _dense(g_value, HETEROGENEOUS, HETEROGENEOUS_MEANS)
        assert found == pytest.approx(expected, abs=1e-9)


def test_the_wrong_answer_is_far_enough_away_to_fail_on():
    """The fixtures must be able to fail, or the tests above prove nothing.

    A campaign that used the first epoch's width everywhere answers
    ``-3.086436`` where the truth is ``-5.838560``: 2.75 nats, on four epochs.
    A campaign that used the first epoch's MEAN everywhere is 1.162 nats out
    at ``g = 0`` and 3.789 at ``g = -2``. The gaps grow with the campaign,
    which is what makes them worth tests rather than a comment.
    """
    with jax.enable_x64(True):
        truth = _dense(0.0, HETEROGENEOUS)
        first_width_everywhere = _dense(0.0, HOMOGENEOUS)
        assert abs(truth - first_width_everywhere) > 2.0
        assert first_width_everywhere == pytest.approx(-3.086435510, abs=1e-8)
        assert truth == pytest.approx(-5.838560122, abs=1e-8)

        one_mean = (HETEROGENEOUS_MEANS[0],) * EPOCHS
        for g_value, floor in ((0.0, 1.0), (-2.0, 3.5)):
            varying = _dense(g_value, HETEROGENEOUS, HETEROGENEOUS_MEANS)
            collapsed = _dense(g_value, HETEROGENEOUS, one_mean)
            assert abs(varying - collapsed) > floor


@pytest.mark.parametrize("g_value", [0.0, 1.5])
def test_a_homogeneous_campaign_is_unchanged(g_value):
    """The half that must NOT move.

    Every campaign fixture in this package declares one width, so the repair
    has to leave that case exactly where it was -- against the same hand-written
    oracle, not against a recorded number.
    """
    with jax.enable_x64(True):
        term = compress_campaign(_campaign(HOMOGENEOUS), "epoch")
        found = float(term.log_prob({"g": jnp.asarray(g_value)}))
        assert found == pytest.approx(_dense(g_value, HOMOGENEOUS), abs=1e-9)


#: A per-epoch latent of COMPONENT width > 1, which is the only thing that
#: makes ``(size, *domain.shape[name][1:])`` different from ``(size,)``.
#: ``node_shape`` broadcasts the plate as a trailing axis, so a ``(E, k)``
#: declaration is expressible only at ``k == E``; hence 3 epochs, 3 components.
SQUARE_EPOCHS = 3
SQUARE_SCALES = np.array([[0.5, 1.0, 2.0], [3.0, 0.25, 1.5], [0.75, 4.0, 0.6]])
SQUARE_MEANS = np.array([[0.1, -0.2, 0.3], [0.4, 0.0, -0.5], [1.0, 0.2, -0.1]])
SQUARE_WEIGHTS = np.array([1.0, -2.0, 0.5])
SQUARE_DATA = np.array([0.3, 1.1, 0.4])


def _square_campaign():
    def model():
        epoch = plate("epoch", SQUARE_EPOCHS)
        g = sample("g", lambda: dist.Normal(0.0, 2.0))
        n = sample(
            "n",
            lambda: dist.Normal(jnp.asarray(SQUARE_MEANS), jnp.asarray(SQUARE_SCALES)),
            plate=epoch,
        )
        mu = det(
            "mu",
            lambda g_, n_: 2.0 * g_ + jnp.sum(jnp.asarray(SQUARE_WEIGHTS) * n_),
            g,
            n,
            plate=epoch,
            linear_in=("g", "n"),
        )
        observe(
            "d",
            lambda m: dist.Normal(m, SIGMA),
            mu,
            obs=jnp.asarray(SQUARE_DATA),
            plate=epoch,
        )

    return trace(model)


def _square_dense(g_value):
    """``sum_e log N(d_e | 2g + w.m_e, sigma^2 + sum_j w_j^2 s_ej^2)``.

    The scale matrix is deliberately NOT symmetric, so a route that indexed the
    plate on the wrong axis would disagree rather than coincide.
    """
    total = 0.0
    for epoch in range(SQUARE_EPOCHS):
        variance = SIGMA**2 + float(
            np.sum(SQUARE_WEIGHTS**2 * SQUARE_SCALES[epoch] ** 2)
        )
        residual = SQUARE_DATA[epoch] - (
            2.0 * g_value + float(np.sum(SQUARE_WEIGHTS * SQUARE_MEANS[epoch]))
        )
        total += -0.5 * (
            residual * residual / variance
            + math.log(variance)
            + math.log(2.0 * math.pi)
        )
    return float(total)


@pytest.mark.parametrize("g_value", [0.0, 1.5, -2.0])
def test_a_per_epoch_latent_of_component_width_uses_each_component(g_value):
    """Every component of every epoch's declaration is read.

    ``target = (size,)`` -- the broadcast target with ``[1:]`` deleted -- passes
    the entire fast layer, because every other campaign fixture in this package
    has a per-epoch latent of component width 1, where the two tuples are equal.
    """
    with jax.enable_x64(True):
        term = compress_campaign(_square_campaign(), "epoch")
        found = float(term.log_prob({"g": jnp.asarray(g_value)}))
        assert found == pytest.approx(_square_dense(g_value), abs=1e-9)


def test_a_scalar_mean_beside_a_vector_width_is_refused():
    """The spelling this repair does NOT reach, pinned where it is refused.

    ``dist.Normal(0.4, taus)`` with a vector ``taus`` is refused by
    ``gaussian_parts``' ``broadcast_to(scale, shape(loc))``, reached from
    ``epoch_terms``' own first statement through ``factorize`` ->
    ``check_linearity`` -> ``check_gaussian``. An earlier version of this
    docstring said it fails inside ``unchecked_operator`` before ``epoch_terms``
    sees it; both halves were false, and the traceback assertion below is here
    so that a repeat is a failure rather than a sentence.

    The assertion does not read the message. An earlier version of this
    docstring said JAX prints ``float32[4]`` or ``float64[4]`` "depending on
    when the trace ran". That was a guess written to explain a real
    observation, and it is false: measured twice, deterministically, the dtype
    tracks WHERE the array was built -- outside ``jax.enable_x64`` gives
    float32, inside it or a plain Python tuple gives float64.

    The guard was loose in two ways rather than one. ``match="[Bb]roadcast"``
    reads a third-party message, AND at least four of this package's own
    refusals say "broadcast" too (``graph/graph.py:147``,
    ``diagnose/local.py:137``, ``diagnose/sensitivity.py:605``,
    ``exact/precision.py:421``), so an unrelated first-party failure satisfies
    it as well. What is asserted instead: the refusal comes from OUR frame, and
    the same model written with an explicit vector mean compresses and is
    correct. That makes this a limitation of one SPELLING, not of the model.
    """
    with jax.enable_x64(True):
        taus = jnp.asarray(HETEROGENEOUS)
        data = jnp.asarray(DATA)

        def model():
            epoch = plate("epoch", EPOCHS)
            g = sample("g", lambda: dist.Normal(0.0, 2.0))
            n = sample("n", lambda: dist.Normal(PRIOR_MEAN, taus), plate=epoch)
            mu = det(
                "mu",
                lambda g_, n_: 2 * g_ + n_,
                g,
                n,
                plate=epoch,
                linear_in=("g", "n"),
            )
            observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data, plate=epoch)

        with pytest.raises(ValueError) as caught:
            compress_campaign(trace(model), "epoch")

        frames = traceback.extract_tb(caught.value.__traceback__)
        # Which frames are OURS is asked of the package's own location, not of
        # how a path is spelled. The spelled version --
        # ``"bayesmith" in frame.filename and ".venv" not in frame.filename`` --
        # was walked past by a rename, and CI is where it happened: the wheel
        # job installs into ``.testenv``, not ``.venv``, and checks out to
        # ``/home/runner/work/bayesmith/bayesmith/``, so EVERY path under it
        # contains "bayesmith" and the exclusion matched nothing. jax's own
        # ``lax.py`` was then counted as ours and became "the deepest frame",
        # failing this assertion with ``broadcast_to``. Measured in run
        # 34049524335: the wheel job red, the source job -- same commit, same
        # OS, a venv named ``.venv`` -- green.
        package_root = pathlib.Path(bayesmith.__file__).resolve().parent
        ours = [
            frame
            for frame in frames
            if package_root in pathlib.Path(frame.filename).resolve().parents
        ]
        assert ours, "the refusal must pass through this package"
        assert ours[-1].name == "gaussian_parts", (
            f"the declaration is refused where the prior is READ; the deepest "
            f"frame of ours is {ours[-1].name} at "
            f"{ours[-1].filename.rsplit('/', 1)[-1]}:{ours[-1].lineno}"
        )
        assert "epoch_terms" in {frame.name for frame in ours}, (
            "epoch_terms IS on the stack -- the claim that this fails before "
            "epoch_terms sees it was measured false"
        )
        assert "unchecked_operator" not in {frame.name for frame in ours}

        # ... and the same model, spelled with the mean written out, is fine.
        term = compress_campaign(_campaign(HETEROGENEOUS), "epoch")
        found = float(term.log_prob({"g": jnp.asarray(0.0)}))
        assert found == pytest.approx(_dense(0.0, HETEROGENEOUS), abs=1e-9)


def test_the_fold_is_still_flat_in_the_campaign_length():
    """The property the vectorisation exists for, kept -- and the VALUE with it.

    A folded term is three leaves whatever the campaign length, and the cost is
    dominated by tracing rather than by the epoch count. Carrying the prior as
    a mapped argument rather than a closed-over scalar must not change either.

    The leaf count alone would pass a fold that is right at four epochs and
    wrong at sixty-four, so each length is also checked against the oracle --
    which is free, the campaigns are already built and already heterogeneous.
    """
    with jax.enable_x64(True):
        for size in (8, 64):
            widths = tuple(np.linspace(0.4, 2.0, size))
            means = tuple(np.linspace(-0.9, 0.9, size))
            data = tuple(np.linspace(-1.0, 1.0, size))
            taus = jnp.asarray(widths)
            locs = jnp.asarray(means)
            observations = jnp.asarray(data)

            def model(taus=taus, locs=locs, observations=observations, size=size):
                epoch = plate("epoch", size)
                g = sample("g", lambda: dist.Normal(0.0, 2.0))
                n = sample("n", lambda: dist.Normal(locs, taus), plate=epoch)
                mu = det(
                    "mu",
                    lambda g_, n_: 2 * g_ + n_,
                    g,
                    n,
                    plate=epoch,
                    linear_in=("g", "n"),
                )
                observe(
                    "d",
                    lambda m: dist.Normal(m, SIGMA),
                    mu,
                    obs=observations,
                    plate=epoch,
                )

            term = compress_campaign(trace(model), "epoch")
            assert len(jax.tree.leaves(term)) == 3, (
                f"a folded campaign of {size} epochs is three leaves whatever "
                f"its length; that is the archive property the square-root "
                f"form exists for"
            )

            found = float(term.log_prob({"g": jnp.asarray(0.35)}))
            expected = 0.0
            for observation, tau, mean in zip(data, widths, means, strict=True):
                variance = tau**2 + SIGMA**2
                residual = observation - (2.0 * 0.35 + mean)
                expected += -0.5 * (
                    residual * residual / variance
                    + math.log(variance)
                    + math.log(2.0 * math.pi)
                )
            assert found == pytest.approx(expected, rel=1e-9), (
                f"the fold is flat in the length AND right at it; {size} epochs"
            )


def test_a_prior_mean_over_a_tiny_width_is_not_refused_and_not_reliable():
    """A DISCLOSURE, not a contract. The limit is recorded so it is countable.

    An earlier version of this docstring named the WIDTH as the controlling
    quantity and ``s**2`` underflowing to zero as the mechanism. Both were
    invented to explain a real observation, and both are false:

    * Nothing on this path squares ``s``. ``nuisance_prior`` writes ``1/s``
      into the factor and ``m/s`` into the target (``marginal/compress.py``);
      the only ``**2`` there is on the NOISE sigma.
    * ``s**2`` is already exactly 0.0 at ``s = 1e-200``, where the route
      returns a finite number, and ``s = 1e-307`` is finite again while
      ``1e-300`` is ``-inf``. An underflow story cannot be non-monotonic.
    * The controlling quantity is the prior MEAN. With ``m = 0`` and nothing
      else changed, every width down to ``1e-307`` is exact to about one ulp.
      The error law is ``eps * |m| / s``, so a floor on ``s`` alone cannot
      separate the good cells from the bad.

    The mechanism is catastrophic cancellation of the ``m/s`` target entry
    inside ``SqrtInfo.combine``'s QR: at ``s = 1e-300`` the retained target
    comes back as rounding noise of size ``(m/s) * eps``, and ``log_prob``
    squares it to ``inf``.

    Measured here, a four-epoch campaign whose LAST epoch declares the width:

    ========= ================== ================
    width     rel err, m = 0.4   rel err, m = 0
    ========= ================== ================
    1e-8      1.7e-09            0.0
    1e-12     6.9e-06            1.7e-16
    1e-16     5.7e-02            8.4e-16
    1e-200    6.4e-02            1.9e-15
    1e-300    ``-inf``           8.9e-15
    ========= ================== ================

    **The failure PRE-DATES the repair** -- the homogeneous all-``1e-300``
    campaign is ``-inf`` at both commits, because the old code used epoch 0's
    width everywhere. The repair only makes it reachable from any of the E
    slots rather than the first.

    **Why nothing is refused or repaired here.** Recentring the prior (folding
    ``A m`` into the offset) makes this column exact and introduces its OWN
    unbounded law at large ``|m|`` and large ``s``: against an exact Fraction
    oracle at ``m = s = 2**60`` the shipped route is 1.4e-14 out and the
    recentred one 3.7e+05. Neither origin dominates -- over a 7x7 sweep the
    shipped route wins 15 cells and the recentred one 21 -- so this is a
    dispatcher with two valid regions, not a substitution, and this repository
    has a written method for those (agree at the threshold, sweep the
    extremes). Until that grid exists, changing the arithmetic would move the
    failure rather than remove it.

    Asserted: the good side agrees, the far side is not finite, and ``m = 0``
    is exact at every width. Only the last is derived -- ``m = 0`` removes the
    ``m/s`` entry entirely. The percentages are this machine's.
    """
    with jax.enable_x64(True):
        usable = compress_campaign(_campaign((0.5, 1.0, 2.0, 1e-4)), "epoch")
        found = float(usable.log_prob({"g": jnp.asarray(0.0)}))
        expected = _dense(0.0, (0.5, 1.0, 2.0, 1e-4))
        assert found == pytest.approx(expected, rel=1e-10)

        for widths in ((0.5, 1.0, 2.0, 1e-300), (1e-300,) * 4):
            term = compress_campaign(_campaign(widths), "epoch")
            answer = float(term.log_prob({"g": jnp.asarray(0.0)}))
            assert not math.isfinite(answer), (
                f"a width of 1e-300 beside a prior mean of {PRIOR_MEAN} "
                f"returns {answer}; if this becomes finite the limit has moved"
            )
            assert math.isfinite(_dense(0.0, widths)), (
                "the dense oracle answers where the implementation does not, "
                "which is what makes this a limit rather than a singularity"
            )

        # The discriminator. If the WIDTH were the controlling quantity these
        # would degrade too; they are exact, which is what refutes a floor on s.
        centred = (0.0,) * EPOCHS
        for width in (1e-8, 1e-16, 1e-200, 1e-300, 1e-307):
            widths = (0.5, 1.0, 2.0, width)
            term = compress_campaign(_campaign(widths, centred), "epoch")
            answer = float(term.log_prob({"g": jnp.asarray(0.0)}))
            assert answer == pytest.approx(
                _dense(0.0, widths, centred), rel=1e-13
            ), (
                f"at prior mean 0, a width of {width:.0e} is exact; if this "
                f"fails the law is not eps*|m|/s and the table above is stale"
            )
