"""Residual-evidence fixtures whose ``log Z`` is known in closed form.

**Why these are not in ``models.py``.** That module is a CENSUS DENOMINATOR.
R5 Task 1 walks it to grade the prior/likelihood split over every shipped
fixture, ``tests/dispatch/test_residual_oracle.py`` asserts the structural class
of all 49 of its no-argument graphs and all 54 of its graphs, and Task 7 will
walk it again for the propriety audit. Three of those counts are pinned by
equality. Adding a fixture there moves numbers in three places that have nothing
to do with the fixture, and the R5 plan's 0.15 records what happens when a count
whose denominator is unstated moves: the next reader re-derives it differently.
So the fixtures R5's completion gates need live here, and the census keeps its
denominator.

**Every ``log Z`` below is CONSTRUCTED, not sampled.** R5's 8 gate asks for a
non-Gaussian or multimodal fixture; a fixture whose answer came from a sampler
would grade a sampler against itself. What is here instead:

===============================  ===================================  ==========
fixture                          why it exists                        closed form
===============================  ===================================  ==========
``mixture_prior_residual``       the only MULTIMODAL posterior in      a sum of
                                 this package -- every other fixture   Gaussian
                                 is unimodal, so nothing else can      evidences
                                 tell a method that finds THE
                                 posterior from one that finds A mode
``cauchy_residual_pair``         a heavy tail IN the likelihood.       Cauchy is
                                 ``overflowing_outside_latent`` puts   stable
                                 a Cauchy on a latent the likelihood   under
                                 never sees, which is a different      convolution
                                 thing and much easier
``undeclared_quartet``           four latents in ONE sampled block,    the linear-
                                 above anything shipped; affine in     Gaussian
                                 all four and not DECLARED so          evidence
===============================  ===================================  ==========

Two habits the file keeps, both because the alternative has cost this project
something:

* **The graph and its closed form read the same constants**, through a shared
  ``_parts`` helper or by reading them back off the built graph. Two copies of a
  prior width would let the model drift away from the quantity that certifies
  it, and the drift would show up as a PASSING test -- the closed form still
  returns a number, just not this graph's evidence.
* **No value below is pinned as a literal anywhere.** ``jax.random.normal`` on
  one key returns different draws under ``jax.enable_x64``, so
  ``undeclared_quartet``'s ``log Z`` is -5.9873 at x64 and -8.1724 at the
  suite's default float32 -- a different data vector, not a rounding difference.
  Red line 9, and four release tags.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample, trace

# ------------------------------------------------------------------ multimodal


def _mixture_residual_parts(
    *,
    n=7,
    mode_weights=(0.35, 0.65),
    mode_centres=(-2.3, 3.7),
    mode_widths=(0.45, 0.8),
    offset_mean=0.9,
    offset_std=1.6,
    sigma=1.1,
    slope=0.24,
    offset=1.3,
):
    """The numbers :func:`mixture_prior_residual` and its closed form BOTH read."""
    x = jnp.linspace(0.62, 1.94, n)
    return {
        "x": x,
        "data": slope * x + offset,
        "weights": jnp.asarray(mode_weights),
        "centres": jnp.asarray(mode_centres),
        "widths": jnp.asarray(mode_widths),
        "offset_mean": offset_mean,
        "offset_std": offset_std,
        "sigma": sigma,
    }


def mixture_prior_residual(**overrides):
    """``w`` carries a two-component Gaussian-mixture prior; the rest is linear.

    The residual posterior is genuinely bimodal and the evidence is still a
    closed form, because marginalising ``(w, b)`` out of

        ``d ~ N(w X + b, sigma)``, ``w ~ sum_k pi_k N(m_k, s_k)``,
        ``b ~ N(b0, sb)``

    leaves one multivariate normal density per mixture component,

        ``Z = sum_k pi_k N(d ; A [m_k, b0], A diag(s_k^2, sb^2) A^T + sigma^2 I)``

    with ``A = [X | 1]``. Nothing is sampled to get it.

    **Measured on this checkout** (macOS/Accelerate, float64) on the 801 x 401
    grid over ``w`` in (-8, 9.5) and ``b`` in (-7.5, 9) that
    ``test_residual_fixtures._mixture_marginal`` actually builds: the ``w``
    marginal peaks at -1.613 and +1.953, with a valley at -0.125 whose density is
    697.8x below the lower mode and 645.4x below the upper. The two peaks are
    3.566 apart, which is 9.20 lower-mode standard deviations. The grid puts
    0.431527 of the mass below the valley against the closed form's own component
    weight of 0.431502 -- they agree to 2.5e-05, so the two agree about the SHAPE
    and not only about the total, which is the check the total alone cannot make.

    ``slope=0.24`` is tuned, and only for BALANCE. Swept, with BOTH valley ratios
    named because "the valley ratio" is two numbers and the smaller is the one
    that bounds the multimodality:

    ======  ===============  ============  ============
    slope   split            valley/lower  valley/upper
    ======  ===============  ============  ============
    0.05    0.7066 / 0.2934        1144.6         333.7
    0.15    0.5679 / 0.4321         880.4         470.3
    0.24    0.4315 / 0.5685         697.8         645.4
    0.36    0.2663 / 0.7337         514.6         995.6
    ======  ===============  ============  ============

    The second mode survives every row -- the smaller ratio never falls below
    333.7 -- so the tuning buys balance and not the second mode.

    〔**Two of these numbers were false and a second adversarial review caught
    them.** The sweep's 0.15 row read ``0.625/0.375``; it is ``0.5679/0.4321``,
    and 0.625/0.375 is reached at ``slope = 0.110894``. The mass split read
    ``0.4316 / 0.5684``; both printed digits were wrong in the last place. They
    came from a design agent's report and I transcribed them without re-running
    those particular rows, having re-measured the closed form and the gap myself
    -- which is the whole of the lesson, since nothing in the suite read the
    sweep and so nothing could have noticed. ``test_the_slope_sweep_is_measured``
    now reads it.〕

    **No seed.** ``data`` is exactly ``slope * X + offset``. The closed form is
    exact for any data but the mass split is not, so a noise draw would make the
    table above a statement about one key rather than about the fixture.
    """
    parts = _mixture_residual_parts(**overrides)
    x, data, sigma = parts["x"], parts["data"], parts["sigma"]
    weights, centres, widths = parts["weights"], parts["centres"], parts["widths"]
    offset_mean, offset_std = parts["offset_mean"], parts["offset_std"]

    def two_mode_prior():
        return dist.MixtureSameFamily(
            dist.Categorical(probs=weights), dist.Normal(centres, widths)
        )

    def model():
        xs = const("X", x)
        w = sample("w", two_mode_prior)
        b = sample("b", lambda: dist.Normal(offset_mean, offset_std))
        mu = det("mu", lambda w_, b_, x_: w_ * x_ + b_, w, b, xs, linear_in=("w", "b"))
        observe("d", lambda m_: dist.Normal(m_, sigma), mu, obs=data)

    return trace(model)


def mixture_prior_residual_log_evidence(**overrides):
    """``(log Z, per-component log terms)`` for :func:`mixture_prior_residual`.

    The second return is not decoration: ``exp(terms - log Z)`` is each mode's
    posterior weight, which is what the grid's mass split is checked against.
    """
    parts = _mixture_residual_parts(**overrides)
    x, data, sigma = parts["x"], parts["data"], parts["sigma"]
    weights, centres, widths = parts["weights"], parts["centres"], parts["widths"]
    offset_mean, offset_std = parts["offset_mean"], parts["offset_std"]

    count = x.shape[0]
    design = jnp.stack([x, jnp.ones_like(x)], axis=1)
    noise = sigma**2 * jnp.eye(count)
    terms = []
    for index in range(centres.shape[0]):
        mean = design @ jnp.asarray([centres[index], offset_mean])
        prior_cov = jnp.diag(jnp.asarray([widths[index] ** 2, offset_std**2]))
        cov = design @ prior_cov @ design.T + noise
        residual = data - mean
        _sign, logdet = jnp.linalg.slogdet(cov)
        quadratic = residual @ jnp.linalg.solve(cov, residual)
        terms.append(
            jnp.log(weights[index])
            - 0.5 * (count * jnp.log(2.0 * jnp.pi) + logdet + quadratic)
        )
    stacked = jnp.stack(terms)
    return float(jax.scipy.special.logsumexp(stacked)), np.asarray(stacked, float)


# ----------------------------------------------------------------- heavy tails


def cauchy_residual_pair(*, gamma=1.75, sigma=0.4, datum=1.3):
    """``z ~ Cauchy(0, gamma)``, ``d ~ Cauchy(z, sigma)``, ONE observation.

    The heavy tail is in the LIKELIHOOD, which is the case quadrature finds
    hardest and the one ``overflowing_outside_latent`` does not exercise: there
    the Cauchy sits on a latent whose contribution to ``mu`` is
    ``exp(|z| / 50)``, so the likelihood cuts the tail off exponentially and the
    integrand decays like a Gaussian. Here the integrand decays like ``z**-4``.

    Cauchy is stable under convolution, so with a SINGLE observation
    ``log Z = Cauchy(datum; 0, gamma + sigma).log_prob`` exactly. Two
    observations multiply two densities rather than convolving them, and a
    product of Cauchys is not a Cauchy -- measured, the naive
    ``Cauchy(0, gamma + s1 + s2)`` is 2.17 nats out -- so the fixture has one.

    Numbers pairwise distinct, including the derived ones: ``gamma + sigma`` is
    2.15 and ``datum`` is 1.3, so a closed form that forgot to add the two
    widths cannot land on the right number anyway. No seed and no noise draw:
    ``datum`` is the datum, which also keeps the fixture identical under
    ``jax.enable_x64`` -- measured, one key gives data 1.27 apart between the
    two, three times ``sigma``.
    """

    def model():
        z = sample("z", lambda: dist.Cauchy(0.0, gamma))
        mu = det("mu", lambda z_: z_, z, linear_in=("z",))
        observe("d", lambda m_: dist.Cauchy(m_, sigma), mu, obs=jnp.asarray(datum))

    return trace(model)


def cauchy_residual_pair_log_evidence(*, gamma=1.75, sigma=0.4, datum=1.3):
    """``log Z`` for :func:`cauchy_residual_pair`, in closed form.

    Same keywords as the fixture, so the two cannot drift apart silently.
    """
    return float(dist.Cauchy(0.0, gamma + sigma).log_prob(jnp.asarray(datum)))


def cauchy_tail_mass(*, span, gamma=1.75, sigma=0.4, datum=1.3):
    """The exact mass of :func:`cauchy_residual_pair`'s integrand past ``|z| > span``.

    The integrand is ``Cauchy(z; 0, gamma) Cauchy(datum; z, sigma)``, which for
    large ``|z|`` goes as ``gamma sigma / (pi**2 z**4)``, so the mass beyond
    ``span`` on both sides is ``2 gamma sigma / (3 pi**2 span**3)``, relative to
    a total of ``exp(log Z)``.

    Exists so the oracle's own edge bound can be graded against a truth rather
    than against another quadrature: a geometric model of a ``z**-p`` tail sums
    to ``p - 1`` over ``p`` of it, which is the shortfall
    ``test_the_edge_bound_is_short_on_a_polynomial_tail`` pins.
    """
    beyond = 2.0 * gamma * sigma / (3.0 * math.pi**2 * span**3)
    return beyond / math.exp(
        cauchy_residual_pair_log_evidence(gamma=gamma, sigma=sigma, datum=datum)
    )


# ------------------------------- dimensions the rest of the family holds constant


def outside_observation_pair(
    *, n=6, sigma=0.45, tau_loc=1.6, tau_scale=0.5, outer_sigma=0.7, outer=1.15
):
    """An observation the exact block does NOT reach.

    **Written because an adversarial review found the Wave B comparison blind to
    a whole class of elimination defect.** Every other fixture here and in
    ``models.py`` has exactly one observed node, and it is always a descendant of
    the exact block -- so the filter in ``marginal_log_density`` that decides
    WHICH observations may enter the marginal term (``observed not in
    absorbed``) has only ever been asked a question with one answer. Reinstating
    R4's double count by deleting that filter left all 36 Wave B tests green.

    Here ``e`` observes ``tau`` directly, so it is in the block's data and is not
    a descendant of ``x``: counting it into the marginal term as well as leaving
    it in the reduced graph counts its density twice, which is exactly the defect
    ``fb1c21f`` carried and R4 repaired.

    (The full fast layer DOES catch that mutant -- five tests in
    ``tests/dispatch/test_collapse.py``. So this fixture closes a gap in what the
    Wave B ORACLE is sensitive to, not a gap in the repository. Both statements
    are worth having, and only the second one was true before it was measured.)
    """
    grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * grid + sigma * jnp.linspace(-0.6, 0.7, n)

    def model():
        columns = const("X", grid)
        tau = sample("tau", lambda: dist.Normal(tau_loc, tau_scale))
        x = sample("x", lambda t: dist.Normal(0.0, jnp.abs(t) + 0.2), tau)
        prediction = det("mu", lambda x_, g_: x_ * g_, x, columns, linear_in=("x",))
        observe("d", lambda m_: dist.Normal(m_, sigma), prediction, obs=data)
        observe(
            "e", lambda t_: dist.Normal(t_, outer_sigma), tau, obs=jnp.asarray(outer)
        )

    return trace(model)


def shifted_block_prior(
    *, n=6, sigma=0.4, tau_loc=2.2, tau_scale=0.45, block_width=0.3
):
    """The eliminated block's prior MEAN is the residual latent, so ``|m|/s`` moves.

    **Also written from an adversarial review.** ``block_prior_ratio`` reports the
    largest ``|prior mean| / prior width`` the eliminated block reaches anywhere
    on the span, probing each axis at its ends and its centre. Three mutants of
    it survived the whole Wave B suite -- forcing the mean to zero, taking the
    minimum over the probe points instead of the maximum, and probing only the
    centre -- for one reason with a count behind it: **the eliminated block's
    prior mean is exactly 0 in four of the five fixtures and constant in the
    fifth**, so nothing could tell those three apart from the real thing.

    Here ``x ~ N(tau, block_width)`` puts the residual latent in the block's prior
    MEAN, so the ratio is ``|tau| / block_width`` and sweeps about an order of
    magnitude across a declared span. A probe that reads only the centre, or
    takes a minimum, or ignores the mean, now answers differently from one that
    does not.
    """
    grid = jnp.linspace(1.0, 2.0, n)
    data = 2.05 * grid + sigma * jnp.linspace(-0.5, 0.65, n)

    def model():
        columns = const("X", grid)
        tau = sample("tau", lambda: dist.Normal(tau_loc, tau_scale))
        x = sample("x", lambda t: dist.Normal(t, block_width), tau)
        prediction = det("mu", lambda x_, g_: x_ * g_, x, columns, linear_in=("x",))
        observe("d", lambda m_: dist.Normal(m_, sigma), prediction, obs=data)

    return trace(model)


# ------------------------------------------------------- four residual axes

#: Six latents' worth of prior and truth, pairwise distinct. The fixtures below
#: take a prefix: ``undeclared_quartet`` is the first four, and the stop-rule
#: sweep walks one to six.
FAMILY_NAMES = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
FAMILY_MEAN = (0.6, -0.9, 0.25, 1.15, -0.4, 0.72)
FAMILY_STD = (1.3, 2.1, 0.7, 1.9, 1.05, 2.4)
FAMILY_TRUE = (1.7, -0.35, 0.85, -1.35, 0.55, -0.95)
FAMILY_SIGMA = 0.55

#: Kept as the name the fixture had before the family was generalised, because
#: it is what a caller passing ``sigma=`` is overriding.
QUARTET_SIGMA = FAMILY_SIGMA
QUARTET_NAMES = FAMILY_NAMES[:4]


def _family_design(grid, dimension):
    """The first ``dimension`` columns ``mu`` is built from, in name order.

    Chosen so no column is a multiple of another and none is orthogonal to the
    rest on the grid: a design whose columns separated would make the evidence a
    product of one-dimensional integrals, and an oracle could then get a
    ``d``-dimensional answer one axis at a time -- which is exactly the thing the
    stop-rule sweep below is trying to measure the cost of NOT being able to do.
    """
    columns = (
        jnp.ones_like(grid),
        grid,
        grid**2,
        jnp.sin(grid),
        jnp.cos(2.0 * grid),
        grid**3,
    )
    return jnp.stack(columns[:dimension], axis=1)


def undeclared_family(*, dimension=4, n=5, sigma=FAMILY_SIGMA, seed=31):
    """``dimension`` latents in ONE sampled block, on a design that does not separate.

    ``mu = sum_j theta_j f_j(t) + C`` is affine in every ``theta_j`` jointly and
    the graph does not SAY so: ``mu`` declares no ``linear_in``, so the
    classifier cannot prove the map affine and every latent goes to the sampled
    block. The mathematics is untouched by the omission, so the evidence is
    still ``Z = N(d ; A m + c, A S A^T + sigma^2 I)``, which
    :func:`gaussian_log_evidence` evaluates.

    ``dimension`` is what makes it a stop-rule instrument as well as a fixture:
    the same family at one to six latents is how R5 Task 3.2 measures where a
    product-grid quadrature stops being an oracle.
    """
    grid = jnp.linspace(0.4, 1.8, n)
    offset = 0.45 * jnp.cos(grid)
    design = _family_design(grid, dimension)
    truth = design @ jnp.asarray(FAMILY_TRUE[:dimension]) + offset
    data = truth + sigma * jax.random.normal(jax.random.key(seed), (n,))
    names = FAMILY_NAMES[:dimension]

    def model():
        grids = const("T", grid)
        offsets = const("C", offset)
        latents = [
            sample(name, (lambda m=FAMILY_MEAN[j], s=FAMILY_STD[j]: dist.Normal(m, s)))
            for j, name in enumerate(names)
        ]
        mu = det(
            "mu",
            lambda *args: (
                _family_design(args[dimension], dimension) @ jnp.stack(args[:dimension])
                + args[dimension + 1]
            ),
            *latents,
            grids,
            offsets,
        )
        observe("d", lambda m_: dist.Normal(m_, sigma), mu, obs=data)

    return trace(model)


def undeclared_quartet(*, n=5, sigma=FAMILY_SIGMA, seed=31):
    """FOUR latents in one sampled block -- :func:`undeclared_family` at four.

    Above anything shipped in ``models.py``, where the largest residual is two.
    **The four axes do not factor**, which matters because a separable fixture
    could be got right one axis at a time and would grade nothing about a
    four-dimensional integral: measured, the posterior precision has off-diagonal
    entries up to 35.4, condition number 322.6, and posterior correlations up to
    0.817 in magnitude.
    """
    return undeclared_family(dimension=4, n=n, sigma=sigma, seed=seed)


def undeclared_family_parts(graph, *, dimension=4, sigma=FAMILY_SIGMA):
    """The linear-Gaussian parts of :func:`undeclared_family`, read OFF ``graph``.

    Read rather than recomputed, so the closed form cannot describe a different
    data vector from the one the graph carries. It would: ``t`` and the noise
    draw are both evaluated at the graph's own precision, and one key gives
    different draws under ``jax.enable_x64`` -- measured, ``log Z`` is -5.9873
    at x64 and -8.1724 at the suite's default float32, which is a different data
    vector rather than a rounding difference. Red line 9.
    """
    grid = jnp.asarray(np.asarray(graph.node("T").value, float))
    return {
        "design": np.asarray(_family_design(grid, dimension), float),
        "offset": np.asarray(graph.node("C").value, float),
        "data": np.asarray(graph.node("d").observed, float),
        "prior_mean": np.asarray(FAMILY_MEAN[:dimension], float),
        "prior_std": np.asarray(FAMILY_STD[:dimension], float),
        "sigma": float(sigma),
        "latents": FAMILY_NAMES[:dimension],
    }


def undeclared_quartet_parts(graph, *, sigma=FAMILY_SIGMA):
    """:func:`undeclared_family_parts` at four."""
    return undeclared_family_parts(graph, dimension=4, sigma=sigma)


def gaussian_log_evidence(design, offset, data, prior_mean, prior_std, sigma):
    """``log Z`` for ``d ~ N(A theta + c, sigma^2 I)``, ``theta ~ N(m, diag(s)^2)``.

    Marginalising ``theta`` leaves ``Z = N(d ; A m + c, A diag(s)^2 A^T +
    sigma^2 I)``. The identity holds whether or not the graph DECLARES the map
    affine, which is what makes it the check for a fixture built from an
    undeclared affine ``det``.

    Through a Cholesky factor rather than ``slogdet`` plus a general solve: the
    covariance is a positive-semidefinite outer product plus a positive
    diagonal, so it is positive definite by construction, and one triangular
    factor gives both the quadratic form and the log-determinant.
    """
    a = np.asarray(design, float)
    c = np.asarray(offset, float)
    d = np.asarray(data, float)
    m = np.asarray(prior_mean, float)
    s = np.asarray(prior_std, float)
    count = d.size
    noise = np.broadcast_to(np.asarray(sigma, float), (count,))
    covariance = (a * s**2) @ a.T + np.diag(noise**2)
    residual = d - (a @ m + c)
    factor = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(factor, residual)
    return float(
        -0.5 * (whitened @ whitened)
        - np.log(np.diag(factor)).sum()
        - 0.5 * count * np.log(2.0 * np.pi)
    )


def gaussian_posterior(design, offset, data, prior_mean, prior_std, sigma):
    """The exact posterior precision, mean and marginal sds of the same model.

    Not part of the evidence identity. It is here so a quadrature span can be
    placed at the posterior's own scale rather than guessed, and so the
    off-diagonal of the precision can be reported as evidence that the integrand
    does not factor over the axes.
    """
    a = np.asarray(design, float)
    c = np.asarray(offset, float)
    d = np.asarray(data, float)
    m = np.asarray(prior_mean, float)
    s = np.asarray(prior_std, float)
    count = d.size
    weight = np.diag(1.0 / np.broadcast_to(np.asarray(sigma, float), (count,)) ** 2)
    prior_precision = np.diag(1.0 / s**2)
    precision = prior_precision + a.T @ weight @ a
    covariance = np.linalg.inv(precision)
    mean = covariance @ (prior_precision @ m + a.T @ weight @ (d - c))
    return mean, np.sqrt(np.diag(covariance)), precision


__all__ = [
    "FAMILY_NAMES",
    "QUARTET_NAMES",
    "cauchy_residual_pair",
    "cauchy_residual_pair_log_evidence",
    "cauchy_tail_mass",
    "gaussian_log_evidence",
    "gaussian_posterior",
    "mixture_prior_residual",
    "outside_observation_pair",
    "mixture_prior_residual_log_evidence",
    "shifted_block_prior",
    "undeclared_family",
    "undeclared_family_parts",
    "undeclared_quartet",
    "undeclared_quartet_parts",
]
