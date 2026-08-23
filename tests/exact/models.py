"""Toy graphs the exact-solve tests share.

Kept in one place so a change to a toy model cannot make two test modules
quietly disagree about what they are testing.

**Numbers are chosen to be pairwise distinct.** In every model below the true
value, the noise width, the prior width and the prior centre are all
different numbers. P1's Task 4 lost its only guard on this package's most
important guarantee to a fixture where `w = 3.0` and the true gradient
`sum(X) = 3.0` happened to agree, so a broken implementation returning the
parameter instead of its gradient passed anyway.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

from bayesmith import const, det, observe, plate, sample, trace


class LyingNormal(dist.Normal):
    """A Normal whose ``log_prob`` is not the one its own loc/scale imply.

    Exists to make :func:`~bayesmith.exact.gaussian.check_gaussian`'s probe
    testable: introspection reads ``(loc, scale)`` off the type and is
    perfectly happy, and only evaluating ``log_prob`` reveals the
    disagreement. Not a contrived shape -- any ``Distribution`` subclass that
    overrides ``log_prob`` (a censored likelihood, a tempered one, a
    hand-written approximation) lands exactly here.
    """

    def log_prob(self, value):
        return 1.5 * super().log_prob(value)


def straight_line(*, n=8, weight=2.5, sigma=0.5, prior_std=2.0, prior_mean=0.0, seed=0):
    """``d ~ N(w X, sigma)``, ``w ~ N(prior_mean, prior_std)``.

    One linear latent, one observed node, no plate, sigma constant. The
    smallest graph the exact path applies to.
    """
    x = jnp.linspace(1.0, 4.0, n)
    data = weight * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(prior_mean, prior_std))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def two_linear_latents(*, n=12, slope=1.5, intercept=-3.0, sigma=0.4, seed=1):
    """``d ~ N(a X + b, sigma)``. Two latents that must be solved JOINTLY."""
    x = jnp.linspace(-2.0, 2.0, n)
    data = slope * x + intercept + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, 5.0))
        b = sample("b", lambda: dist.Normal(0.0, 7.0))
        mu = det("mu", lambda a_, b_, x_: a_ * x_ + b_, a, b, xs, linear_in=("a", "b"))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def bilinear_pair(*, n=10, sigma=0.3, seed=2):
    """``mu = gain * t_ant * X`` -- affine in EACH, not affine in the PAIR.

    The declaration ``linear_in=("gain", "t_ant")`` is therefore false, and
    `Graph.__check_init__` cannot see it: both names really are parents. Only
    the joint affinity probe can. This is rheplicant's motivating failure --
    a hand-rolled alternating solve here lands thousands of kelvin away while
    the CG residual reads 1e-7 and every per-block condition number reads ~1.5.
    """
    x = jnp.linspace(0.5, 3.0, n)
    data = 2.0 * 1.5 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        g = sample("gain", lambda: dist.Normal(1.0, 1.0))
        t = sample("t_ant", lambda: dist.Normal(2.0, 3.0))
        mu = det(
            "mu", lambda g_, t_, x_: g_ * t_ * x_, g, t, xs, linear_in=("gain", "t_ant")
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def quadratic_claim(*, n=6, sigma=0.5, seed=3):
    """``mu = w**2 X`` declared ``linear_in=("w",)`` -- a false single claim."""
    x = jnp.linspace(1.0, 2.0, n)
    data = 4.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 2.0))
        mu = det("mu", lambda w_, x_: w_**2 * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def two_observations(*, n=7, m=5, weight=1.25, s1=0.3, s2=0.9, seed=4):
    """One latent constrained by TWO observed nodes -- the pytree codomain.

    rheplicant's codomain is a single array because a pipeline has one
    output. A graph can have several observed nodes, so every reduction in
    the solve runs over a dict of leaves rather than one array.
    """
    k1, k2 = jax.random.split(jax.random.key(seed))
    x1 = jnp.linspace(1.0, 3.0, n)
    x2 = jnp.linspace(-1.0, 1.0, m)
    d1 = weight * x1 + s1 * jax.random.normal(k1, (n,))
    d2 = 2.0 * weight * x2 + s2 * jax.random.normal(k2, (m,))

    def model():
        a = const("X1", x1)
        b = const("X2", x2)
        w = sample("w", lambda: dist.Normal(0.0, 4.0))
        m1 = det("mu1", lambda w_, x_: w_ * x_, w, a, linear_in=("w",))
        m2 = det("mu2", lambda w_, x_: 2.0 * w_ * x_, w, b, linear_in=("w",))
        observe("d1", lambda u: dist.Normal(u, s1), m1, obs=d1)
        observe("d2", lambda u: dist.Normal(u, s2), m2, obs=d2)

    return trace(model)


def two_observations_reverse_sorted_names(
    *, n=7, m=5, weight=1.75, s1=0.35, s2=0.85, seed=19
):
    """Like `two_observations`, but the two observed nodes' DECLARED order is
    the reverse of their ALPHABETICAL order.

    `two_observations` names its nodes "d1" then "d2" -- declared order and
    sorted order coincide there, so nothing built only from it can tell
    "rows/weights are in alphabetical order" apart from "rows/weights are in
    declaration order". Task 9's own mutation list names exactly this gap
    for `bayesmith.exact.fisher.dense_operator`'s `sorted(pushed)` and
    predicts it. Declaring "z_first" before "a_second" makes the two orders
    DISAGREE (declared: z_first, a_second; sorted: a_second, z_first).

    Used against two call sites in `fisher.py` that each build a
    `sorted(...)`-ordered concatenation: `dense_operator`'s `sorted(pushed)`
    and `fisher_information`'s `sorted(noise_std)`. Measured (see
    `tests/exact/test_fisher.py`): only the second is a live mutation risk
    on this fixture -- `pushed` is a `jax.linearize` tangent output and
    JAX's dict-pytree flattening already forces it into sorted-key order
    before `dense_operator` ever runs `sorted()` on it, so that particular
    swap is a no-op for ANY graph, not only this one. `noise_std` is a plain
    dict with no such round-trip, so its ordering genuinely depends on
    `sorted(...)` being called at all.
    """
    k1, k2 = jax.random.split(jax.random.key(seed))
    x1 = jnp.linspace(1.0, 3.0, n)
    x2 = jnp.linspace(-1.0, 1.0, m)
    d1 = weight * x1 + s1 * jax.random.normal(k1, (n,))
    d2 = 2.0 * weight * x2 + s2 * jax.random.normal(k2, (m,))

    def model():
        a = const("X1", x1)
        b = const("X2", x2)
        w = sample("w", lambda: dist.Normal(0.0, 4.5))
        m1 = det("mu1", lambda w_, x_: w_ * x_, w, a, linear_in=("w",))
        m2 = det("mu2", lambda w_, x_: 2.0 * w_ * x_, w, b, linear_in=("w",))
        observe("z_first", lambda u: dist.Normal(u, s1), m1, obs=d1)
        observe("a_second", lambda u: dist.Normal(u, s2), m2, obs=d2)

    return trace(model)


def plated_and_scalar_latents(*, n=4, tau=1.3, prior_std_w=2.7, sigma=0.35, seed=16):
    """One plated latent (`z`, `n` elements) and one scalar latent (`w`),
    solved JOINTLY -- HETEROGENEOUS per-member sizes in one block.

    `bayesmith.exact.fisher._spans`'s cumulative-offset arithmetic
    (`start += size`) is exercised by every other multi-member fixture in
    this suite only where every member has size 1 (`two_linear_latents`,
    `radiometer_group`), or by a single plated member with nothing placed
    after it (`plated_latent`) -- neither can distinguish a member's actual
    `size` from a hardcoded `1`, or from a DIFFERENT member's size. Here `z`
    occupies flat slots `[0, n)` and `w` occupies `[n, n+1)`, so a bug that
    reuses one member's size for the other's span produces a shape mismatch
    or a wrong span rather than a numerically-coincidental pass.

    `mu_i = z_i + w` (no covariate) is deliberately the simplest jointly
    affine function of a plate and a scalar: it keeps `A`'s structure
    legible for a pinned-value assertion -- the "z" columns of the design
    are exactly `I_n` and the "w" column is all ones.
    """
    key = jax.random.key(seed)
    z_true = jnp.linspace(-1.8, 1.8, n)
    w_true = 1.6
    truth = z_true + w_true
    data = truth + sigma * jax.random.normal(key, (n,))

    def model():
        obs = plate("obs", n)
        z = sample("z", lambda: dist.Normal(0.0, tau), plate=obs)
        w = sample("w", lambda: dist.Normal(0.0, prior_std_w))
        mu = det("mu", lambda z_, w_: z_ + w_, z, w, plate=obs, linear_in=("z", "w"))
        observe("d", lambda m: dist.Normal(m, sigma), mu, plate=obs, obs=data)

    return trace(model)


def prior_held_direction(*, n=6, sigma=0.4, seed=21):
    """One direction held by the prior alone, one held tightly by the data.

    ``loose`` reaches no observed node, so the only thing bounding the normal
    operator from below in its direction is its own prior curvature --
    ``lambda_min`` IS ``1/100**2``. ``tight`` has a narrow prior and does see
    data. That combination is the regime `condition_bound` was designed for,
    and it is the only shape on which the bound's guarantee can actually be
    tested: measured, the bound comes out at exactly 1.0000x the true
    condition number, while the tightest-prior mutation of it lands 1e-8x
    below -- a factor of 1e8 in the direction that would destroy the
    guarantee.

    `two_linear_latents` cannot do either job: its data constrains both
    directions far better than either prior, so the bound is 3676x loose and
    a tightest-prior mutation still sits 1875x ABOVE the true kappa.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 0.02 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        tight = sample("tight", lambda: dist.Normal(0.0, 0.01))
        sample("loose", lambda: dist.Normal(0.0, 100.0))
        mu = det("mu", lambda t, x_: t * x_, tight, xs, linear_in=("tight",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def plated_latent(*, n=6, sigma=0.4, tau=1.5, seed=5):
    """``z_i ~ N(0, tau)``, ``d_i ~ N(z_i, sigma)`` under one plate.

    The observed node's loc is the latent ITSELF, with no deterministic node
    on the path -- so the "every Deterministic on the path declares
    linear_in" rule holds vacuously, which the exact path must accept rather
    than trip over.
    """
    key = jax.random.key(seed)
    truth = tau * jax.random.normal(key, (n,))
    data = truth + sigma * jax.random.normal(jax.random.fold_in(key, 1), (n,))

    def model():
        obs = plate("obs", n)
        z = sample("z", lambda: dist.Normal(0.0, tau), plate=obs)
        observe("d", lambda z_: dist.Normal(z_, sigma), z, plate=obs, obs=data)

    return trace(model)


def plated_latent_through_deterministic(
    *, n=6, sigma=0.4, tau=1.5, gain=2.0, prior_mean=0.8, seed=15
):
    """Like `plated_latent`, but `d`'s loc reaches `z` through a Deterministic node.

    `plated_latent` deliberately has NO Deterministic node on the path (see
    its own docstring) -- which leaves `_env_before`'s Deterministic branch
    entirely unexercised whenever the block's sole member is plated, since
    that branch is simply never reached for that fixture. This one adds a
    plated `mu_i = gain * z_i` between `z` and `d`, still under one plate, so
    a block on `z` walks Const-free, member (plated), Deterministic (plated),
    observed -- the one combination `plated_latent` cannot cover.

    `z`'s prior mean is 0.8, not 0.0: `_env_before` evaluates every block
    member AT its prior mean, so a zero-mean prior makes `mu = gain * z`
    evaluate to zero too -- indistinguishable from a Deterministic branch
    that was replaced by a hardcoded zero. Measured directly: with
    `prior_mean=0.0` a mutation doing exactly that left this fixture's
    parametrization of `test_env_before_agrees_with_evaluate_on_every_node`
    green, `jnp.allclose(0.0, zeros((6,)))` being `True` regardless of which
    of the two produced it.
    """
    key = jax.random.key(seed)
    truth = prior_mean + tau * jax.random.normal(key, (n,))
    data = gain * truth + sigma * jax.random.normal(jax.random.fold_in(key, 1), (n,))

    def model():
        obs = plate("obs", n)
        z = sample("z", lambda: dist.Normal(prior_mean, tau), plate=obs)
        mu = det("mu", lambda z_: gain * z_, z, plate=obs, linear_in=("z",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, plate=obs, obs=data)

    return trace(model)


def radiometer(*, n=10, weight=3.0, kappa=0.05, floor=1e-3, seed=6):
    """``sigma_i = kappa |mu_i| + floor`` -- sigma tracks the prediction.

    The GLS / correction case. ``floor`` keeps sigma strictly positive where
    the prediction crosses zero, which the probe guard requires.
    """
    x = jnp.linspace(1.0, 5.0, n)
    truth = weight * x
    data = truth + (kappa * jnp.abs(truth) + floor) * jax.random.normal(
        jax.random.key(seed), (n,)
    )

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 10.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, kappa * jnp.abs(m) + floor),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


def one_sided_sigma(*, n=8, kappa=0.2, floor=1e-2, seed=12):
    """``sigma = kappa * max(mu, 0) + floor`` -- constant for every mu <= 0.

    A one-sided probe that happens to go negative reads sigma as constant and
    lets `depends_on_prediction=False` through. Two-sided does not.
    """
    x = jnp.linspace(1.0, 3.0, n)
    data = 2.0 * x + floor * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, kappa * jnp.maximum(m, 0.0) + floor),
            mu,
            obs=data,
        )

    return trace(model)


def plated_radiometer(*, n=6, tau=3.0, kappa=0.06, floor=2e-3, seed=13):
    """A plated latent whose sigma tracks its OWN prediction, elementwise.

    Structurally the shape Task 7 found unexercised for `gcr_sample`: one
    domain LEAF with several elements, rather than several scalar members.
    `radiometer`'s ``w`` is a scalar; ``z`` here is a plate of ``n``, and
    ``sigma_i = kappa|z_i| + floor`` depends on that SAME element -- so
    `iterative_gls`'s reweighting loop and `check_prediction_dependence`'s
    probe run on an array leaf instead of a dict of scalars. No Deterministic
    node sits between ``z`` and ``d``, matching `plated_latent`'s pattern.
    """
    key = jax.random.key(seed)
    truth = tau * jax.random.normal(key, (n,))
    data = truth + (kappa * jnp.abs(truth) + floor) * jax.random.normal(
        jax.random.fold_in(key, 1), (n,)
    )

    def model():
        obs = plate("obs", n)
        z = sample("z", lambda: dist.Normal(0.0, tau), plate=obs)
        observe(
            "d",
            lambda z_: dist.Normal(z_, kappa * jnp.abs(z_) + floor),
            z,
            plate=obs,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


def radiometer_group(
    *, n=9, m=6, a_true=1.5, b_true=-2.0, kappa=0.04, floor=2e-3, s2=0.25, seed=14
):
    """Two latents solved JOINTLY; one observed node's sigma tracks its
    prediction, the other's does not.

    Structurally the shape `check_prediction_dependence` and the reweighting
    loop have never been run against: MORE than one domain leaf (``a``,
    ``b``) and MORE than one codomain leaf (``d1`` prediction-dependent,
    ``d2`` constant) at once -- `radiometer` is scalar/single-leaf/
    single-observed on all three axes gls.py's own dict/pytree code is
    generic over.
    """
    x1 = jnp.linspace(1.0, 4.0, n)
    x2 = jnp.linspace(-1.5, 1.5, m)
    truth1 = a_true * x1 + b_true
    k1, k2 = jax.random.split(jax.random.key(seed))
    data1 = truth1 + (kappa * jnp.abs(truth1) + floor) * jax.random.normal(k1, (n,))
    data2 = b_true * x2 + s2 * jax.random.normal(k2, (m,))

    def model():
        xs1 = const("X1", x1)
        xs2 = const("X2", x2)
        a = sample("a", lambda: dist.Normal(0.0, 4.0))
        b = sample("b", lambda: dist.Normal(0.0, 6.0))
        mu1 = det(
            "mu1", lambda a_, b_, x_: a_ * x_ + b_, a, b, xs1, linear_in=("a", "b")
        )
        mu2 = det("mu2", lambda b_, x_: b_ * x_, b, xs2, linear_in=("b",))
        observe(
            "d1",
            lambda u: dist.Normal(u, kappa * jnp.abs(u) + floor),
            mu1,
            depends_on_prediction=True,
            obs=data1,
        )
        observe("d2", lambda u: dist.Normal(u, s2), mu2, obs=data2)

    return trace(model)


def indirect_ancestor(*, n=6, sigma=0.5, seed=8):
    """`tau` reaches `x`'s prior through a deterministic node, not directly.

    A direct-parent ancestry check passes this and is wrong: x's parents are
    ("width",), and `width` is a function of `tau`.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        width = det("width", lambda t: jnp.abs(t) + 0.1, tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def diamond_ancestor(*, n=6, sigma=0.5, seed=17):
    """``tau`` reaches ``x``'s prior along TWO paths, so ancestry must dedupe.

    ``tau -> upper -> width`` and ``tau -> lower -> width``. Walking the
    ancestry without a seen-set revisits ``tau`` once per path; on a wide DAG
    that is exponential rather than merely wasteful. Also the first fixture
    here with a diamond, which this project's extreme-value list asks for.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        upper = det("upper", lambda t: jnp.abs(t) + 0.1, tau)
        lower = det("lower", lambda t: jnp.abs(t) * 0.5 + 0.1, tau)
        width = det("width", lambda u, v: u + v, upper, lower)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def shared_ancestor(*, n=6, sigma=0.5, seed=7):
    """``tau`` is a latent AND an ancestor of the latent ``x``.

    Both nodes are Gaussian, so a naive classifier would put them in one
    block -- and the pair's joint distribution is not Gaussian, because x's
    own width is a function of tau. The block builder must refuse the pair.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        x = sample("x", lambda t: dist.Normal(0.0, jnp.abs(t)), tau)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def cubic_tail(*, n=6, curvature=1e-6, prior_std=1.0, sigma=0.5, seed=9):
    """``mu = (w + curvature w**3) X`` -- affine only for small ``|w|``.

    ``linear_in=("w",)`` is false, but detectably so only at probes large
    enough for the cubic term to matter -- and what sets that scale is the
    declared prior width. The SAME fn therefore passes with a narrow prior
    and fails with a wide one, which is the cleanest demonstration that the
    probe magnitude is read off the prior rather than fixed.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, prior_std))
        mu = det(
            "mu", lambda w_, x_: (w_ + curvature * w_**3) * x_, w, xs, linear_in=("w",)
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def affine_only_at_zero(*, n=6, sigma=0.4, seed=10):
    """``mu = (x + z**2 x**2) X`` -- affine in ``x`` only where ``z == 0``.

    ``z ~ N(3, 1)``, so a prior draw lands nowhere near zero. A check that
    probes only at the caller's ``at`` (with z pinned to 0) passes; a check
    that also probes at prior draws does not. That gap is the entire reason
    check_linearity takes several at-points.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        z = sample("z", lambda: dist.Normal(3.0, 1.0))
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        mu = det(
            "mu",
            lambda x_, z_, g_: (x_ + z_**2 * x_**2) * g_,
            x,
            z,
            xs,
            linear_in=("x",),
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def nan_at_negative_probes(*, n=4, sigma=0.5):
    """``mu = sqrt(w) X`` -- NaN wherever a probe goes negative.

    Half of every symmetric probe does. NaN must count as a FAILURE:
    `nan > rtol` is False, so a naive comparison reads an unusable probe as
    evidence of linearity.
    """
    x = jnp.linspace(1.0, 2.0, n)

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(4.0, 1.0))
        mu = det("mu", lambda w_, x_: jnp.sqrt(w_) * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=2.0 * x)

    return trace(model)


def overflowing_outside_latent(*, n=6, sigma=0.5, seed=11):
    """``mu = w X + exp(|z| / 50)`` -- `w` is genuinely affine; `z` can overflow.

    ``z ~ Cauchy(0, 1e6)`` is heavy-tailed enough that ``exp(|z| / 50)``
    overflows float32 on about 99.7% of draws (measured: overflow needs
    ``|z| > 4436``, and a Cauchy(0, 1e6) puts over 99% of its mass beyond
    that). Holding ``z`` fixed at ANY value, ``mu`` is affine in ``w`` --
    `linear_in=("w",)` is TRUE -- so a check that lands on an overflowing
    ``z`` (the usual outcome of a default prior draw here) and reports a
    linearity failure of ``w`` is misattributing the fault: the overflow is
    a statement about where ``z`` sits, not about ``w``. Pinning ``z`` at 0
    -- where nothing overflows -- and finding the SAME block accepted is
    what proves the declaration was true all along.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        z = sample("z", lambda: dist.Cauchy(0.0, 1e6))
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det(
            "mu",
            lambda w_, z_, x_: w_ * x_ + jnp.exp(jnp.abs(z_) / 50.0),
            w,
            z,
            xs,
            linear_in=("w",),
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def improper_outside_prior(*, n=6, sigma=0.5, seed=12):
    """An outside latent with an improper prior -- no sampler, by construction.

    ``dist.ImproperUniform`` carries infinite mass, so NumPyro's own
    ``.sample()`` on it raises ``NotImplementedError`` rather than returning
    a value. ``prior_at_points`` hits this trying to draw a default at-point
    for ``z``, which sits outside a block on the genuinely-affine ``w``.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        z = sample("z", lambda: dist.ImproperUniform(constraints.real, (), ()))
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda w_, z_, x_: w_ * x_ + z_, w, z, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def unconstrained_latent(*, n=5, sigma=0.5, seed=11):
    """``u`` reaches no observed node, so its posterior IS its prior.

    An extreme corner the solve must handle rather than divide by: A's column
    for ``u`` is exactly zero, so the normal operator there is the prior
    curvature alone and the answer is the prior mean. ``u``'s centre (1.37)
    and width (0.75) are meant to be distinct from every other number in the
    model, so a solve that returned zero, or the other latent's prior, could
    not pass -- and that claim is checked, not just asserted: measured,
    ``jnp.linspace(1.0, 2.0, 5)[1] == 1.25`` exactly, so an earlier version of
    this fixture that used 1.25 put ``u``'s centre exactly on ``X``'s own
    grid. No path from ``X`` to ``u`` exists (``u`` reaches no observed node
    at all), so nothing was actually exploitable -- but discipline #3 is not
    "reason about exploitability case by case", it is "the constant must not
    equal a value already present in the model", full stop. 1.37 is off that
    grid.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 2.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        sample("u", lambda: dist.Normal(1.37, 0.75))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)
