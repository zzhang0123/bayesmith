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

    Since the probe gained a RANDOM direction, this fixture also carries the
    measurement that keeps ``uniform`` in
    :data:`~bayesmith.exact.gls.DEPENDENCE_PATTERNS`: a random direction
    multiplies each signed magnitude by its own draw, so both probes land on
    the clipped half-space whenever the draws have the wrong signs. Measured
    with ``DEPENDENCE_PATTERNS = ("random",)`` over 400 keys, **105 of them
    (26%) read bitwise 0.0**. The deterministic anchor removes that failure
    mode by construction -- see
    `test_a_clipped_sigma_is_detected_at_every_key_because_of_the_anchor`.
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


def tunable_curvature(*, n=8, departure=0.0, sigma=0.5, prior_std=1.0, seed=14):
    """``mu = (w + departure * w**2 / prior_std) X``.

    ``departure`` is, to first order, the relative departure from affinity a
    one-sigma probe sees -- so sweeping it across check_linearity's rtol walks
    the accept/reject boundary directly, which is what
    `boundary-validation.md` asks for: evaluate BOTH sides at the threshold
    rather than trusting the dispatcher's own verdict.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = 1.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, prior_std))
        mu = det(
            "mu",
            lambda w_, x_: (w_ + departure * w_**2 / prior_std) * x_,
            w,
            xs,
            linear_in=("w",),
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def collinear_pair(*, n=8, sigma=0.4, prior_std=3.0, seed=13):
    """``mu = (a + b) X`` -- the data cannot tell ``a`` from ``b`` at all.

    Jointly affine, so check_linearity passes and the JOINT block is the right
    thing: the data fixes ``a + b``, the prior alone fixes ``a - b``, and the
    joint kappa reports honestly how much worse one direction is determined
    than the other. Alternating over two one-latent blocks instead would
    report a converged residual and a condition number of ~1 forever, which is
    rheplicant's recorded failure in its purest form.
    """
    x = jnp.linspace(1.0, 3.0, n)
    data = 2.0 * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, prior_std))
        b = sample("b", lambda: dist.Normal(0.0, prior_std))
        mu = det(
            "mu", lambda a_, b_, x_: (a_ + b_) * x_, a, b, xs, linear_in=("a", "b")
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def wide_plate(*, size, sigma=0.4, tau=1.5, seed=15):
    """``plated_latent`` at an arbitrary plate size, for the size sweep."""
    return plated_latent(n=size, sigma=sigma, tau=tau, seed=seed)


def many_observations(*, count, n=6, weight=1.5, sigma=0.4, seed=16):
    """One latent constrained by ``count`` observed nodes.

    Names are ``obs_0 ... obs_{count-1}``, whose sorted order is their
    declaration order only while ``count <= 10`` -- deliberately, so the
    codomain ordering is exercised rather than assumed.
    """
    key = jax.random.key(seed)
    grids = [jnp.linspace(1.0, 2.0 + index, n) for index in range(count)]
    data = [
        weight * grid + sigma * jax.random.normal(jax.random.fold_in(key, index), (n,))
        for index, grid in enumerate(grids)
    ]

    def model():
        w = sample("w", lambda: dist.Normal(0.0, 4.0))
        for index, (grid, values) in enumerate(zip(grids, data, strict=True)):
            xs = const(f"X_{index}", grid)
            mu = det(f"mu_{index}", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
            observe(f"obs_{index}", lambda m: dist.Normal(m, sigma), mu, obs=values)

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


def bright_and_faint_observations(*, n=6, bright=1e17, sigma_faint=0.01, w_true=0.8):
    """An honest BRIGHT node beside a lying FAINT one that dominates the posterior.

    `affinity_errors` normalised `departure` by a `variation` taken as a max
    over EVERY codomain leaf, so the bright node set the yardstick -- and the
    roundoff floor -- for the faint one. Measured before the fix:
    `check_linearity` returned 3.45e-14 and PASSED (float64; 2.57e-14 in
    float32, which is what the suite runs), while `mu2` alone was correctly
    refused at 4.93e+00, and the "exact" answer was off by 202 true posterior
    standard deviations.

    `bright=1e17` is not adversarial engineering; it is the dynamic range
    this package targets -- a foreground in K beside a signal in mK is 1e6,
    and an interferometric visibility against a monopole is far more.

    No `seed`: every array here is exact, with no noise draw to seed. A
    `seed=` kwarg would be dead, and a reader would reasonably expect
    changing it to change the data.
    """
    x1 = jnp.linspace(1.0, 2.0, n)
    x2 = jnp.linspace(1.0, 2.0, n)
    d1 = bright * w_true * x1
    d2 = (w_true + 0.5 * w_true**2) * x2

    def model():
        a = const("X1", x1)
        b = const("X2", x2)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        m1 = det("mu1", lambda w_, x_: bright * w_ * x_, w, a, linear_in=("w",))
        m2 = det("mu2", lambda w_, x_: (w_ + 0.5 * w_**2) * x_, w, b, linear_in=("w",))
        observe("d1", lambda u: dist.Normal(u, bright * 100.0), m1, obs=d1)
        observe("d2", lambda u: dist.Normal(u, sigma_faint), m2, obs=d2)

    return trace(model)


def faint_alone(*, n=6, sigma_faint=0.01, w_true=0.8):
    """`bright_and_faint_observations`'s faint node, with no bright sibling.

    The control the defect is read against: the SAME false claim on the SAME
    node, so a difference in verdict between this and its sibling above can
    only be the bright leaf's doing. Measured before the fix: refused here at
    4.93e+00 and accepted there at 3.45e-14.
    """
    x2 = jnp.linspace(1.0, 2.0, n)
    d2 = (w_true + 0.5 * w_true**2) * x2

    def model():
        b = const("X2", x2)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        m2 = det("mu2", lambda w_, x_: (w_ + 0.5 * w_**2) * x_, w, b, linear_in=("w",))
        observe("d2", lambda u: dist.Normal(u, sigma_faint), m2, obs=d2)

    return trace(model)


def bright_and_faint_channels(*, n=6, bright=1e17, sigma=0.01, w_true=0.8, lying=None):
    """The same dilution WITHIN ONE ARRAY -- one bright channel, five faint.

    Sibling of `bright_and_faint_observations`, and the more realistic of the
    two: a spectrum whose first channel carries a bright foreground and whose
    remaining channels carry the signal is one observed node, not two. A
    per-LEAF fix is not enough here -- only a per-ELEMENT comparison sees it,
    which is the same argument `check_gaussian`'s docstring already makes for
    its own elementwise probe.

    `C` is a `const` rather than a literal so the curvature can be switched
    off channel by channel, and so the honest twin of this graph differs from
    it in exactly one node -- see
    `test_an_affine_model_with_the_same_dynamic_range_still_passes`.

    Args:
        lying: how many of the trailing (faint) channels carry the false
            claim. Defaults to every one of them but the bright channel,
            which is the configuration the verdict tables were measured at.
            `lying=1` puts the defect in a MINORITY of entries, which is the
            only configuration that can tell a per-element `any` apart from
            an average over the array: with five of six channels lying, the
            mean of the per-element departures is within 6/5 of their
            maximum and dilution is invisible.
    """
    lying = n - 1 if lying is None else lying
    x = jnp.concatenate([jnp.array([bright]), jnp.linspace(1.0, 2.0, n - 1)])
    curvature = jnp.concatenate([jnp.zeros(n - lying), jnp.full(lying, 0.5)])
    truth = (w_true + curvature * w_true**2) * x
    scale = jnp.concatenate([jnp.array([bright * 100.0]), jnp.full(n - 1, sigma)])

    def model():
        xs = const("X", x)
        cs = const("C", curvature)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det(
            "mu", lambda w_, x_, c_: (w_ + c_ * w_**2) * x_, w, xs, cs, linear_in=("w",)
        )
        observe("d", lambda u: dist.Normal(u, scale), mu, obs=truth)

    return trace(model)


def bright_and_faint_pair(*, n=6, bright=1e17, sigma=0.01, a_true=0.8, b_true=-0.35):
    """`bright_and_faint_channels` with a TWO-member block: `mu = (a + b + C a b) X`.

    The dilution and the joint claim in one graph, which no other fixture
    covers: every other bright/faint fixture has a single-member block, so
    they cannot say whether the per-element comparison survives the probe
    scheme's per-member random directions. Each conditional here really is
    affine -- `bilinear_pair`'s point -- so only the JOINT claim is false,
    and only on the faint channels, where `C` is non-zero.

    The bright channel dilutes it exactly as before: its own departure is
    zero (`C[0] == 0`) while its variation is 1e17 times everyone else's, so
    a `max(departure) / max(variation)` taken across the array reports ~1e-17
    and passes.
    """
    x = jnp.concatenate([jnp.array([bright]), jnp.linspace(1.0, 2.0, n - 1)])
    curvature = jnp.concatenate([jnp.zeros(1), jnp.full(n - 1, 0.5)])
    truth = (a_true + b_true + curvature * a_true * b_true) * x
    scale = jnp.concatenate([jnp.array([bright * 100.0]), jnp.full(n - 1, sigma)])

    def model():
        xs = const("X", x)
        cs = const("C", curvature)
        a = sample("a", lambda: dist.Normal(0.0, 3.0))
        b = sample("b", lambda: dist.Normal(0.0, 2.0))
        mu = det(
            "mu",
            lambda a_, b_, x_, c_: (a_ + b_ + c_ * a_ * b_) * x_,
            a,
            b,
            xs,
            cs,
            linear_in=("a", "b"),
        )
        observe("d", lambda u: dist.Normal(u, scale), mu, obs=truth)

    return trace(model)


def roundoff_stress(*, big, sigma, n=6):
    """``mu = (w + big) X`` -- exactly affine in ``w``, with REAL roundoff.

    Every other honest fixture here is bitwise exact: ``g(probe)`` and
    ``baseline + tangent(probe)`` evaluate the same expression in the same
    association order, so the departure from affinity is identically zero and
    nothing bounds any tolerance from below at all. ``(w + big) * x`` breaks
    that tie honestly -- the primal computes ``(probe + big) * x``, the
    linearization computes ``big * x + probe * x``, and the two differ by
    ordinary float rounding of order ``eps * big * x``. The claim
    ``linear_in=("w",)`` is TRUE.

    ``big / sigma`` is the offset-to-noise ratio, which is what the
    sigma-weighted criterion has to survive: ``departure / sigma`` grows with
    that ratio rather than with curvature, so ungated it reaches 2.44e-02 at
    a ratio of 1e2 in float32 -- far above `WEIGHTED_RTOL` -- for a model
    with no curvature whatsoever. The per-element roundoff floor is what
    drives every one of those to exactly 0. Measured in Task 1's verdict
    tables; this fixture is where the suite keeps that measurement.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = (1.0 + big) * x

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda w_, x_: (w_ + big) * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def unusable_observed_scale(*, kind="zero", n=5, w_true=2.0):
    """An entirely affine model whose observed node produces an unusable sigma.

    `affinity_errors` divides each element's departure by that node's sigma,
    so a scale that is zero, negative or non-finite makes the weighted column
    unreadable -- and the failure surfaces as "latent 'w' is declared linear,
    but the prediction is not affine in it". **`mu = w * X` is exactly
    affine.** The fault is the scale expression on `d`, and blaming the
    modeller's `linear_in` sends them to rewrite the one part of the model
    that is correct.

    That is the same mis-attribution class `affinity_errors`'s non-finite
    BASELINE branch and its `at_description` argument were added to prevent
    (see `test_a_non_finite_baseline_is_attributed_to_the_outside_latent_not_the_member`),
    so the repair follows the pattern already in the file.

    ``kind``:
        ``"zero"``      -- sigma is 0 everywhere: 1/sigma**2 is an infinite weight.
        ``"one_zero"``  -- a single zero entry among honest ones. The elementwise
                           case: a per-leaf check that reduced before testing
                           would see four good entries and miss it.
        ``"negative"``  -- sigma is negative. **This one passed silently before
                           the guard**, because the weighted column takes
                           ``abs(sigma)`` and so cannot tell -0.5 from +0.5,
                           while `check_gaussian` refuses it by name. A guard
                           that only rejected non-finite values would leave
                           this hole open, which is why it is its own case.
        ``"nan"``       -- sigma is NaN.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = w_true * x
    scales = {
        "zero": jnp.zeros(n),
        "one_zero": jnp.asarray([0.4] * (n // 2) + [0.0] + [0.4] * (n - n // 2 - 1)),
        "negative": jnp.full(n, -0.5),
        "nan": jnp.full(n, jnp.nan),
    }
    scale = scales[kind]

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, scale), mu, obs=data)

    return trace(model)


def non_gaussian_observed_node(*, n=5, w_true=2.0, sigma=0.5):
    """Affine in `w`, but `d` is a Student-t -- so there IS no sigma to read.

    Pins a contract change that arrived silently. Before `affinity_errors`
    took a sigma argument, `check_linearity` read only each observed node's
    LOCATION and so was indifferent to the noise family; a Student-t
    likelihood over an affine mean checked fine. Now `check_linearity` calls
    `noise_std_at`, which reaches `gaussian_parts`, which raises
    `NotGaussian` -- so the entry point refuses this graph outright.

    That is defensible: the weighted criterion is stated in units of sigma,
    and a Student-t has no sigma to state it in. But it was undocumented and
    untested, and the distinction matters downstream -- P3b's dispatcher
    treats `NotGaussian` as a CLASSIFICATION outcome (route the block to
    NUTS) and `StructureError` as a FAULT, so which one comes out of here
    decides whether an ordinary non-conjugate model looks broken.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = w_true * x

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.StudentT(4.0, m, sigma), mu, obs=data)

    return trace(model)


def two_unusable_observed_scales(*, n=5, m=4, w_true=2.0):
    """TWO observed nodes with unusable scales, declared in reverse-sorted order.

    Sibling of `two_observations_reverse_sorted_names`, and here for the same
    reason: `noise_std_at` returns a plain dict built by comprehension over
    `graph.observed`, so it carries DECLARATION order, and any `sorted()`
    applied to it is load-bearing rather than decorative (P3a Task 9's
    criterion for telling the two apart).

    With only one bad node the ordering is unobservable, so
    `_refuse_unusable_scale`'s `sorted` could be deleted with nothing going
    red. Here `z_first` is declared first and `a_second` sorts first, so which
    node the message names distinguishes the two orderings.

    What this pins is the STABILITY of the message, not its correctness:
    either node is a legitimate thing to name, and both are genuinely broken.
    What would be a defect is the name changing because someone reordered
    their `observe()` calls.
    """
    x1 = jnp.linspace(1.0, 2.0, n)
    x2 = jnp.linspace(1.0, 3.0, m)

    def model():
        a = const("X1", x1)
        b = const("X2", x2)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        m1 = det("mu1", lambda w_, x_: w_ * x_, w, a, linear_in=("w",))
        m2 = det("mu2", lambda w_, x_: w_ * x_, w, b, linear_in=("w",))
        observe("z_first", lambda u: dist.Normal(u, jnp.zeros(n)), m1, obs=w_true * x1)
        observe("a_second", lambda u: dist.Normal(u, jnp.zeros(m)), m2, obs=w_true * x2)

    return trace(model)


def contrast_sigma_pair(*, n=200, a_true=1.0, b_true=-0.5, base=0.3, seed=24):
    """sigma depends on a CONTRAST of two members that the mean cannot separate.

    `check_prediction_dependence` moved every member by the same signed
    multiple of its own prior width, so its probe never left the level set of
    `a - b` and it measured a movement of exactly 0.0 -- bitwise. The
    dispatcher then reads "sigma is constant" and picks plain `gcr`.

    Both latents sit on the SAME regressor, so `a + b` is all the mean knows
    and `a - b` is determined entirely by sigma. Measured with the lockstep
    probe: the contrast came back +0.0000 (sd 1.4142) against a long-NUTS
    +1.6038 (sd 0.0486) -- 33 NUTS sd out, width inflated 29x.

    The worst part is not the size of the error. This graph is
    whole-graph-one-block, so it takes the iid-draws-no-chain row of the
    dispatch table: no r-hat, no k-hat, no ESS, nothing to diagnose.

    **Point or region, measured.** The quantity this fixture exists to
    produce -- `check_prediction_dependence`'s movement -- does not depend on
    ``n``, ``base``, ``seed`` or the true contrast AT ALL: sigma is
    ``base * exp(a - b)`` evaluated at probe points around the PRIOR centre
    (0, 0), the data never enters it, and the relative movement divides
    ``base`` out. Swept n in {2, 3, 5, 10, 200, 5000}, base over 1e-6..1e6,
    seed 0-18, and ``a_true - b_true`` in {0, +-1.5, +-10}: the fixed probe
    reads 6.389057 (``exp(2) - 1``) at every single one and the lockstep
    probe reads bitwise 0.0 at every single one. **In particular
    ``a_true == b_true``, contrast zero at the truth, is detected exactly as
    strongly** -- the probe starts at the prior mean, so where the truth sits
    is irrelevant to the guard, which is the whole reason a movement probe
    can run before any data is fitted.

    The axis the DEFECT actually lives on is the ratio of the two PRIOR
    WIDTHS, which this fixture holds at exactly 1.0 (both ``Normal(0, 1)``):
    the lockstep ray only stays inside the level set of ``a - b`` when the
    two members are displaced by equal amounts. Measured on a throwaway
    variant with the widths exposed (not a committed fixture), sd_a = 1 and
    sd_b = 1 + delta, lockstep probe, float64: movement is delta to four
    digits -- 0.0 at delta=0, 1.0e-8 at 1e-8, 9.995e-4 at 1e-3, 9.950e-3 at
    1e-2. So the blind spot is a REGION whose width is set by the consumer's
    threshold, not a knife edge: |delta| <~ 1e-3 for a dispatcher reading the
    returned number against 1e-3, but only |delta| <~ 1e-8 for
    `check_prediction_dependence`'s own ``rtol`` guard on a ``declared=False``
    node. In float32, the suite's default, delta below ~1e-7 is not even
    representable in the probe and every one of those reads bitwise 0.0.

    **Under the shipped probe** (``uniform`` plus a per-member ``random``
    direction) the same invariance holds and the reading is 3.639157: swept n,
    base, seed and the true contrast exactly as above, every cell reads
    3.639156-3.639157, the spread being float32 rounding alone. The 6.389
    this paragraph used to quote was the ``alternating`` pattern's reading,
    and ``alternating`` is no longer shipped -- it separated only positions of
    differing parity, which does not survive a third member.
    """
    x = jnp.linspace(0.5, 2.0, n)
    truth = (a_true + b_true) * x
    noise = base * jnp.exp(a_true - b_true)
    data = truth + noise * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, 1.0))
        b = sample("b", lambda: dist.Normal(0.0, 1.0))
        mu = det(
            "mu", lambda a_, b_, x_: (a_ + b_) * x_, a, b, xs, linear_in=("a", "b")
        )
        observe(
            "d",
            lambda m, a_, b_: dist.Normal(m, base * jnp.exp(a_ - b_)),
            mu,
            a,
            b,
            obs=data,
        )

    return trace(model)


def sum_sigma_pair(*, n=200, a_true=1.0, b_true=-0.5, base=0.3, seed=25):
    """The mirror of `contrast_sigma_pair`: sigma depends on the SUM.

    Here the mean determines ``a - b`` and sigma determines ``a + b``, so the
    direction sigma moves along is the LOCKSTEP one -- exactly the ray the
    original probe travelled.

    Exists to make the ``uniform`` entry of
    :data:`~bayesmith.exact.gls.DEPENDENCE_PATTERNS` load-bearing, and what
    it pins is the SIZE of the reading rather than the detection. ``uniform``
    reads ``exp(2) - 1 = 6.389`` here with no key involved; the ``random``
    entry detects the sum too but reads whatever ``z_a + z_b`` happened to be
    -- 8.399e-01 at the default key, and 1.342e-01 to 4.419e+01 swept over
    200 keys. A dispatcher thresholds the number, so the most common real
    dependence there is (a radiometer's sigma tracking its own prediction)
    gets a guaranteed floor rather than a distribution.

    This fixture is NOT a regression test for the contrast defect: it passes
    on the pre-fix lockstep probe too, by construction. It guards the fix
    from being "simplified" into the opposite blind spot.
    """
    x = jnp.linspace(0.5, 2.0, n)
    truth = (a_true - b_true) * x
    noise = base * jnp.exp(a_true + b_true)
    data = truth + noise * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, 1.0))
        b = sample("b", lambda: dist.Normal(0.0, 1.0))
        mu = det(
            "mu", lambda a_, b_, x_: (a_ - b_) * x_, a, b, xs, linear_in=("a", "b")
        )
        observe(
            "d",
            lambda m, a_, b_: dist.Normal(m, base * jnp.exp(a_ + b_)),
            mu,
            a,
            b,
            obs=data,
        )

    return trace(model)


def sigma_functional_block(*, weights, n=200, base=0.3, seed=26, mean_weights=None):
    """``sigma = base * exp(sum_i w_i theta_i)`` on a block of ``len(weights)``.

    The generalisation of `contrast_sigma_pair` and `sum_sigma_pair` to any
    number of members and any linear functional. Members are named ``a``,
    ``b``, ``c``, ... in order, so a member's POSITION IN THE SORTED NAMES is
    its index into ``weights`` -- which is the coordinate
    `~bayesmith.exact.gls.DEPENDENCE_PATTERNS`' deterministic signs are keyed
    to, so a probe pattern is exactly a sign vector to dot ``weights`` with.
    A pattern whose sign vector is ORTHOGONAL to ``weights`` leaves sigma
    bitwise constant and the guard reads "sigma does not move".

    All priors are ``Normal(0, 1)`` -- **equal widths are what makes the
    deterministic blind spots exact** rather than merely small, exactly as
    `contrast_sigma_pair` documents: a uniform ray only stays inside the
    level set of ``a - c`` when the two members are displaced by equal
    amounts. Unequal widths would turn every zero below into a small number
    whose size is set by the width ratio, which is a weaker fixture.

    Args:
        weights: the functional sigma depends on, one entry per member.
            ``(0.0,) * k`` gives a genuinely CONSTANT sigma on a k-member
            block -- the two-sided arm, where the guard must read no
            movement.
        mean_weights: the functional the MEAN depends on. Defaults to all
            ones, so the mean sees only the sum and every other direction is
            determined by sigma alone -- `contrast_sigma_pair`'s shape.

    Nothing here is crafted to have a root at any probe point: the movement
    each pattern measures is ``|exp(factor * (signs . weights)) - 1|``, which
    is zero exactly when the pattern's sign vector is orthogonal to
    ``weights`` and nonzero otherwise.
    """
    names = tuple(chr(ord("a") + index) for index in range(len(weights)))
    # The observed node is "y", not the "d" every other fixture here uses:
    # at four members the fourth is named "d" and the graph refuses the
    # duplicate. Members past "l" would collide with "mu" only if the naming
    # scheme changed; four is as wide as anything here goes.
    assert len(weights) <= 12, "member names would start colliding with the graph's own"
    mean_weights = (1.0,) * len(weights) if mean_weights is None else mean_weights
    x = jnp.linspace(0.5, 2.0, n)
    truth = sum(mean_weights) * x
    noise = base * jnp.exp(sum(weights))
    data = truth + noise * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        members = [sample(name, lambda: dist.Normal(0.0, 1.0)) for name in names]
        mu = det(
            "mu",
            lambda *args: (
                sum(w * t for w, t in zip(mean_weights, args[:-1])) * args[-1]
            ),
            *members,
            xs,
            linear_in=names,
        )
        observe(
            "y",
            lambda m, *thetas: dist.Normal(
                m, base * jnp.exp(sum(w * t for w, t in zip(weights, thetas)))
            ),
            mu,
            *members,
            obs=data,
        )

    return trace(model)


def element_contrast_sigma_plate(*, n=6, tau=3.0, base=0.3, seed=27):
    """sigma depends on a contrast between two ELEMENTS of one plated latent.

    The same defect shape as `contrast_sigma_pair`, one structural level
    down: there the level set was spanned by two MEMBERS of the block, here
    by two entries of a single array leaf. Every probe direction that
    displaces a leaf uniformly -- which is every per-member direction, random
    or not -- stays inside it, so the guard reads bitwise 0.0 and the
    dispatcher reads "sigma does not move". Measured: a per-member scalar
    draw reads **0.000000e+00**; the shipped per-element draw reads
    **1.730645e+01**.

    The observed node is deliberately NOT plated, unlike `plated_radiometer`.
    A ``plate=`` observed node is applied under ``jax.vmap``, so its
    ``dist_fn`` sees a SCALAR element and ``z_[0] - z_[1]`` raises
    ``IndexError: array is 0-dimensional`` -- the plate abstraction forbids
    cross-element dependence outright, which is exactly why this fixture has
    to reach for the whole array instead. Its latent is still plated, so the
    domain is one leaf of ``n`` elements, which is the dimension being
    probed.
    """
    key = jax.random.key(seed)
    truth = tau * jax.random.normal(key, (n,))
    data = truth + base * jax.random.normal(jax.random.fold_in(key, 1), (n,))

    def model():
        obs = plate("obs", n)
        z = sample("z", lambda: dist.Normal(0.0, tau), plate=obs)
        observe(
            "d",
            lambda z_: dist.Normal(
                z_, base * jnp.exp(z_[0] - z_[1]) * jnp.ones_like(z_)
            ),
            z,
            obs=data,
        )

    return trace(model)


def orphaned_child_latent(*, n=6, sigma=0.5, w_true=2.0, seed=25):
    """`w` is Gaussian and affine, but a DISQUALIFIED latent's density needs it.

    The partition rule's ejection clause originally read "z leaves if it is an
    ancestor of another QUALIFIED latent". `v` here is Student-t, so it fails
    criterion 1 and is not qualified -- and `w` therefore stayed in the block
    while the factor `p(v | w)` was dropped on the floor. `unchecked_operator`
    reads exactly two things, the block members' own priors and the observed
    nodes; every other density term in the graph is invisible to it.

    Measured with the qualified-only rule: mean(w) +0.4106 sd 1.7723 against
    a truth of +1.9759 / 0.4816 and a long-NUTS +2.0004 / 0.4809 -- 3.2 true
    sd out, width inflated 3.7x.

    `tests/exact/oracle.py::graph_oracle` reproduces the SAME wrong answer,
    because it reads the same two sources. The dense oracle cannot see this
    class of defect at all, which is why the guard has to be structural.
    """
    x = jnp.linspace(1.0, 2.0, n)
    key = jax.random.key(seed)
    data = w_true * x + sigma * jax.random.normal(key, (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        sample("v", lambda w_: dist.StudentT(3.0, w_, 0.4), w)
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def student_t_likelihood(*, n=6, sigma=0.5, w_true=1.5, seed=26):
    """The OBSERVED node is not Gaussian -- criterion 2.

    Every one of the 29 `observe()` calls in this module used `dist.Normal`
    before this fixture existed, so criterion 2 had no fixture and a
    classifier that simply never checked observed nodes would have passed the
    entire table. The latent-side criterion is already covered twice, by
    `overflowing_outside_latent`'s Cauchy and `improper_outside_prior`'s
    ImproperUniform.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.StudentT(4.0, m, sigma), mu, obs=data)

    return trace(model)


def lying_observed_node(*, n=6, sigma=0.5, w_true=1.5, seed=27):
    """The observed node's TYPE says Normal while its own log_prob does not.

    `LyingNormal` keeps `loc` and `scale` and changes the density, so
    introspection passes and the probe does not. The classifier must let the
    resulting StructureError THROUGH -- routing it to NUTS would hide a
    broken model behind an ordinary-looking fallback, which is precisely the
    distinction `errors.py` exists to preserve.

    Note this is the same exception TYPE that `check_linearity` raises for a
    false `linear_in`, and that one MUST be caught. The classifier therefore
    discriminates by raise SITE, not by exception type.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: LyingNormal(m, sigma), mu, obs=data)

    return trace(model)


def dangling_deterministic(*, n=6, sigma=0.55, w_true=2.2, seed=28):
    """A `Deterministic` child of the latent that NO observed node depends on.

    Criterion 3 quantifies over the `Deterministic` nodes on a path from the
    latent to an observed node's location. `audit` here is a child of `w`
    that leads nowhere: it declares no `linear_in` at all, and it is not an
    ancestor of `d`. Requiring a declaration from it would refuse an
    entirely honest model for a node the solve never evaluates -- so
    `_relevant_deterministics` intersects with the observed nodes'
    ancestors, and this fixture is what makes that intersection load-bearing.

    Before it existed the mutation "drop the `matters` intersection" left the
    whole suite green: every other fixture's Deterministic nodes are all on
    a path to an observed node, so the intersection was a no-op on all of
    them. `audit` is `w**2` rather than something affine so that the fixture
    is not accidentally harmless in some later, laxer reading of criterion 3.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 3.4))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        det("audit", lambda w_: w_**2, w)
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def three_latent_chain(*, n=6, sigma=0.45, y_true=1.9, seed=29):
    """`tau -> x -> y`: a chain of three latents, each setting the next's width.

    Every existing ancestry fixture (`indirect_ancestor`, `diamond_ancestor`,
    `shared_ancestor`) has exactly ONE ancestor and ONE descendant, so the
    ejection rule is only ever asked a single question there. A chain asks it
    a nested one: `tau` is an ancestor of `x` AND of `y`, `x` is an ancestor
    of `y`, and `y` is an ancestor of nothing.

    **What it separates, measured.** Eject only the FIRST ancestor found --
    a `break` in the loop, which is exactly the slip a set comprehension
    hides -- and this fixture comes out with the block `('x', 'y')`, a pair
    whose joint distribution is not Gaussian and which
    `_refuse_internal_ancestry` then refuses from two layers down. On
    `shared_ancestor`, `indirect_ancestor` and `mixed_radiometer` the same
    mutation is invisible: each has exactly one latent to eject, so stopping
    after the first IS stopping after all of them. Measured, all four.

    **What it does NOT separate, also measured.** "Runs once versus runs to a
    fixed point" turns out to be a distinction without a difference here, for
    a reason worth writing down rather than rediscovering: `_ancestors` is
    TRANSITIVE, so `tau` is an ancestor of `y` directly and not only by way
    of `x`. Re-reading the rule to quantify over the surviving candidates
    instead of over `graph.latents`, and running a single order-dependent
    pass in the least favourable order (`y`, then `x`, then `tau`), still
    ejects `tau` -- because `y` is still there and `tau` still reaches it.
    Measured: `('y',)` under both readings. The shipped quantifier over
    `graph.latents`, a set that never shrinks, is preferable anyway for being
    order-independent by construction rather than by transitivity, but this
    fixture is not evidence for it.

    Not a block-size-3 fixture either: the block here is `{y}` alone. Block
    size 3 is reached through `sigma_functional_block(weights=(...,) * 3)`,
    whose three members have no ancestry between them.
    """
    x_grid = jnp.linspace(1.0, 2.0, n)
    data = y_true * x_grid + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x_grid)
        tau = sample("tau", lambda: dist.Normal(2.4, 0.6))
        x = sample("x", lambda t: dist.Normal(0.0, jnp.abs(t) + 0.15), tau)
        y = sample("y", lambda x_: dist.Normal(0.0, jnp.abs(x_) + 0.35), x)
        mu = det("mu", lambda y_, g_: y_ * g_, y, xs, linear_in=("y",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def mixed_radiometer(*, n=10, weight=2.6, kappa=0.05, floor=1.3e-3, seed=30):
    """A prediction-dependent sigma on a block that is a PROPER SUBSET.

    `tau` is a latent ancestor of `w`'s own prior, so the ancestor rule
    ejects it: the block is `{w}` and `tau` goes to NUTS. Sigma still tracks
    the prediction, so the block is approximate -- and it is embedded in a
    Gibbs sweep rather than being the whole graph, which is spec section 5.3's
    path (B), the independent-proposal Metropolis step.

    Every other prediction-dependent fixture in this module (`radiometer`,
    `radiometer_group`, `plated_radiometer`, `contrast_sigma_pair`,
    `sum_sigma_pair`, `sigma_functional_block`) is whole-graph-one-block and
    therefore routes to path (A), self-normalised importance sampling.
    Without this fixture the `gcr+mh` arm of the method choice has no test at
    all and deleting it leaves the suite green.
    """
    x = jnp.linspace(1.0, 5.0, n)
    truth = weight * x
    data = truth + (kappa * jnp.abs(truth) + floor) * jax.random.normal(
        jax.random.key(seed), (n,)
    )

    def model():
        xs = const("X", x)
        tau = sample("tau", lambda: dist.Normal(4.2, 0.7))
        w = sample("w", lambda t: dist.Normal(0.0, jnp.abs(t) + 0.2), tau)
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda m: dist.Normal(m, kappa * jnp.abs(m) + floor),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


def observation_reused_downstream(*, n=6, s1=0.3, s2=0.7, w_true=1.6, seed=31):
    """A `Deterministic` whose only parent is an OBSERVED node.

    `d1` is data, so `mu2 = tanh(d1) / 2` is a CONSTANT with respect to every
    latent -- `evaluate` puts the observed value into the environment, so
    `isolate` never moves it. Criterion 3 must therefore not ask `mu2` for a
    `linear_in` declaration on `w`'s behalf: `w` does not reach `d2`'s
    location at all, and `d2` contributes a constant offset and nothing else.

    This is the one fixture in this module where a Probabilistic node is the
    PARENT of a Deterministic one, and it is what makes
    `bayesmith.dispatch.classify._relevant_deterministics`' "stop at every
    Probabilistic node" clause load-bearing. Measured: a forward walk that
    continues THROUGH Probabilistic nodes reaches `mu2` from `w` by
    `mu1 -> d1`, finds `linear_in=()`, and disqualifies an entirely honest
    `w`. On every other fixture in this module that mutation is a verdict
    no-op, because the only paths that leave a latent through a Probabilistic
    node leave it through a LATENT one -- and the ancestor rule ejects the
    latent for that anyway, reaching the same verdict by another route.
    """
    x = jnp.linspace(1.0, 2.0, n)
    k1, k2 = jax.random.split(jax.random.key(seed))
    first = w_true * x + s1 * jax.random.normal(k1, (n,))
    second = jnp.tanh(first) / 2.0 + s2 * jax.random.normal(k2, (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 2.8))
        mu1 = det("mu1", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        obs1 = observe("d1", lambda u: dist.Normal(u, s1), mu1, obs=first)
        mu2 = det("mu2", lambda r: jnp.tanh(r) / 2.0, obs1)
        observe("d2", lambda u: dist.Normal(u, s2), mu2, obs=second)

    return trace(model)


def steep_radiometer(
    *, n=6, kappa=0.5, floor=1e-2, prior_std=10.0, prior_mean=0.8, w_true=1.5, seed=32
):
    """A sigma steep enough that the frozen-sigma corrections are CHEAP to test.

    Not a cosmetic choice. Two separate guards need a fixture where freezing
    sigma is a VISIBLE approximation, and every other prediction-dependent
    graph in this module is too gentle to supply one.

    **The freeze point is pinned three ways, and the three are far apart.**
    Dense-oracle posterior sd of `w` with sigma frozen at

    ==================  ==========  ===================================
    freeze point        sd          what a mutation freezing there means
    ==================  ==========  ===================================
    `w = 0.8`           1.6561e-01  the prior mean -- what the factory declares
    `w = 0.0`           1.9317e-03  the block's ZERO, `graph_oracle`'s default
    `w = 3.0`           6.1355e-01  an arbitrary third point
    ==================  ==========  ===================================

    86x and 3.7x apart. `prior_mean` is **deliberately not 0.0**: at a
    zero-mean prior the first two rows coincide and a mutation that freezes
    sigma at the block's zero instead of at its prior mean is invisible --
    discipline 3's mutation face, and `graph_oracle`'s `sigma_at` really does
    default to the zero while `_env_before` really does centre members at the
    prior mean, so the two freeze points both exist in this repository.

    **The MH step's own guard.** With sigma frozen at the GLS fixed point the
    independence proposal is accepted 0.456-0.477 of the time, and one MH
    step applied to a quadrature-exact sample of `p(w | d)` leaves it where it
    was: +0.0 to +1.4 standard errors on the paired mean shift, -0.003 to
    +0.001 relative on the sd, measured over keys 4/13/99 and over the
    parameter sweep below. Spec section 5.3's ORIGINAL error -- rebuilding
    sigma at the current `x`, which makes the proposal adaptive and leaves an
    uncancelled `log det M` -- shifts that same statistic by **-16.1 to -17.0
    standard errors at 8000 draws**, narrows the sd by 13.0-17.3%, and does it
    at an acceptance rate of 0.591-0.614, i.e. one that looks HEALTHIER than
    the correct chain's. Two sigma is reached at **111-123 draws**. Not slow.

    On `radiometer()` -- kappa=0.05 rather than 0.5 -- the same mutation is
    -0.0 to -1.2 standard errors at 8000 draws (acceptance 0.980 correct vs
    0.981 mutated), so two sigma needs O(10^4) draws there and the guard would
    have to be marked `slow`, which is the fourth sub-criterion firing.

    **Point or region, measured.** Swept one parameter at a time at 2000
    draws, reporting the draws needed for two sigma on the mutation: the
    default 116, `prior_mean=-1.7` 119, `w_true=4.0` 120, `prior_std=1.0` 209,
    `seed=40` 113, `kappa=0.3` 394, `kappa=0.8` 73. So it is a region in
    `kappa`, `prior_mean`, `prior_std` and `w_true`. It is **not** a region in
    `n` or `seed`: `n=12` needs 5057 draws and `seed=41` needs 11586, because
    both happen to make the frozen-sigma proposal already accurate
    (acceptance 0.856 and 0.787 against the default's 0.477). The mechanism is
    the same one in both directions -- the mutation is exactly as visible as
    the proposal is bad -- so the defaults here are chosen at an acceptance
    rate near 0.5, and the correct step's invariance holds at every cell
    above.
    """
    x = jnp.linspace(1.0, 3.0, n)
    truth = w_true * x
    scale = kappa * jnp.abs(truth) + floor
    data = truth + scale * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(prior_mean, prior_std))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe(
            "d",
            lambda u: dist.Normal(u, kappa * jnp.abs(u) + floor),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


def high_snr_curvature(*, n=12, amplitude=1e-3, sigma=2e-6, w_true=1.0):
    """``mu = X (w + A (cos w - 1))`` -- curvature only the SIGMA units can see.

    The counterexample that pinned :data:`~bayesmith.exact.linearity.
    WEIGHTED_FLOOR_FACTOR`. ``linear_in=("w",)`` is FALSE: the second term is
    ``-A w**2 / 2 + O(w**4)``. But ``A`` is small enough that at the prior
    width the RELATIVE departure stays under ``rtol = 1e4 eps`` in float32,
    so only the sigma-weighted criterion can refuse it -- and that criterion
    was switched off at every signal-to-noise ratio above 0.84 while both
    columns shared the relative column's ``1e4 eps`` floor.

    Measured at the defaults, float32: the departure is 6.03e+03 eps of the
    prediction's own magnitude, worth 8.10e+02 noise widths. Under the shared
    floor the whole check reported ``0.00e+00`` at every probe, `compile()`
    chose ``gcr``, and 4000 iid draws came back at 0.99954033 against grid
    quadrature's 1.00000000 +/- 5.733e-07 -- **802 posterior standard
    deviations wrong, with `unreliable=False`.**

    Four cells were reproduced, spanning SNR 5e2 to 5e5 and two amplitudes:
    ``(A, sigma)`` in ``{(3e-4, 2e-3), (3e-4, 2e-5), (1e-3, 2e-5),
    (1e-3, 2e-6)}``. All four read ``nuts`` in float64 before any fix and
    ``gcr`` in float32, which is what identified the dtype as the variable.

    ``cos`` rather than a bare ``w**2`` so the curvature is bounded: at the
    widest probe (1e3 prior widths) ``cos w - 1`` stays in ``[-2, 0]`` and the
    departure does NOT grow with the probe, which is what keeps the relative
    criterion blind there. A quadratic would be caught by ``rtol`` at the
    widest probe and would not isolate the sigma-weighted half at all.

    ``X`` spans 0.8 to 1.2 -- deliberately narrow and away from zero, so no
    element is exempted by a vanishing covariate and every one of the twelve
    carries the same lie. No ``seed``: the data are noiseless at ``w_true``,
    because what is under test is the linearity check, not an inference.
    """
    x = jnp.linspace(0.8, 1.2, n)
    data = x * (w_true + amplitude * (jnp.cos(w_true) - 1.0))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det(
            "mu",
            lambda w_, x_: x_ * (w_ + amplitude * (jnp.cos(w_) - 1.0)),
            w,
            xs,
            linear_in=("w",),
        )
        observe("d", lambda u: dist.Normal(u, sigma), mu, obs=data)

    return trace(model)


def cancelling_sum(*, cancel, n=16, big=1e3, sigma=1e-2, m=4, seed=32):
    """``mu_j = sum_k A_jk (w + big)`` where the row sums CANCEL by ``cancel``.

    The lower bound on :data:`~bayesmith.exact.linearity.
    WEIGHTED_FLOOR_FACTOR`, and the only fixture that supplies one at a
    magnitude ``roundoff_stress`` cannot reach. ``linear_in=("w",)`` is
    exactly TRUE: the map is a fixed matrix applied to ``w`` plus a constant.

    ``roundoff_stress`` bounds the floor at about 1 eps, because its
    prediction is two operations deep. A prediction that is a near-cancelling
    sum -- an interferometric visibility against a monopole, a contrast
    channel, any difference of large numbers -- carries relative roundoff of
    order ``cancel * eps`` instead, and that is the honest model a floor set
    too low refuses. Each row of ``A`` is shifted to sum to
    ``sum|A_j| / cancel``, so ``cancel`` IS the ratio of the terms' magnitude
    to the result's.

    Measured, float32, max ``departure / (eps |mu|)`` over the whole probe
    grid: 2.19 at ``cancel=1``, 54.8 at 1e2, 3.50e+03 at 1e4, 4.15e+05 at
    1e6. Verdicts at ``cancel=1e2`` are REFUSE for a weighted floor factor of
    1e0 or 1e1 and pass at 1e2 -- which is what pins the shipped 1e2 from
    below. float64 accepts every cell at every factor tried, so this fixture
    only says anything at the dtype the suite runs.
    """
    a = jax.random.normal(jax.random.key(seed), (m, n))
    a = a - jnp.mean(a, axis=1, keepdims=True)
    a = a + (jnp.sum(jnp.abs(a), axis=1, keepdims=True) / cancel) / n
    data = jnp.sum(a * (1.0 + big), axis=1)

    def model():
        matrix = const("A", a)
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det(
            "mu",
            lambda w_, a_: jnp.sum(a_ * (w_ + big), axis=1),
            w,
            matrix,
            linear_in=("w",),
        )
        observe("d", lambda u: dist.Normal(u, sigma), mu, obs=data)

    return trace(model)


def hinged_sigma_beyond_the_probe(
    *, n=40, a_true=6.0, prior_std=1.0, base=0.05, slope=0.3, hinge=3.0, seed=33
):
    """``sigma = base + slope * max(mu - hinge, 0)`` with the hinge out of reach.

    The counterexample to reading
    :data:`~bayesmith.exact.gls.DEPENDENCE_PROBES` alone as "does sigma move
    with the block". Those probes displace by ``1.0`` and ``-0.5`` prior
    widths from the PRIOR centre, so with ``a ~ N(0, 1)`` they reach
    ``a in {1.0, -0.5}`` and ``mu`` never gets past 1. The hinge sits at
    ``mu = 3`` and the data put the posterior at ``a ~ 6.1``, so every probe
    lands on the flat side and the movement reads **bitwise 0.0** -- at every
    key, since a flat sigma is flat in every direction.

    That is the magnitude gap :data:`~bayesmith.exact.gls.DEPENDENCE_PROBES`'
    own docstring names and says no pattern can repair ("only a larger
    magnitude would, at the cost of probing where the posterior will never
    go"). This fixture is where the posterior DOES go, which is why the
    dispatcher probes sigma at the data-informed point as well.

    Measured, before that second probe existed: classified ``gcr``, which
    applies no correction at all, and ``sample()`` returned mean 6.173466 sd
    **0.007764** against grid quadrature of ``log_joint`` over 400001 points
    on [-2, 12] at mean 6.101637 sd **0.133736** -- a posterior 17.2x too
    narrow, reported with ``ess=4000.0``, ``log_weights=None``,
    ``unreliable=False`` and a plan printing "sigma does not move with the
    block".

    ``n`` is 40 rather than a handful so the likelihood is sharp enough to
    pull the posterior clear of the hinge: at n=40 the posterior sits 3.1
    sigma past ``mu = 3`` in its own units.
    """
    truth = a_true * jnp.ones(n)
    width = base + slope * jnp.maximum(truth - hinge, 0.0)
    data = truth + width * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        ones = const("ones", jnp.ones(n))
        a = sample("a", lambda: dist.Normal(0.0, prior_std))
        mu = det("mu", lambda a_, o_: a_ * o_, a, ones, linear_in=("a",))
        observe(
            "d",
            lambda m: dist.Normal(m, base + slope * jnp.maximum(m - hinge, 0.0)),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


def plated_student_t_latent(*, n=5, sigma=0.45, w_true=1.7, spread=0.9, seed=34):
    """A latent that is BOTH plated and not Gaussian.

    Every other non-Gaussian latent in this module is scalar
    (``overflowing_outside_latent``'s Cauchy, ``improper_outside_prior``'s
    ImproperUniform, ``orphaned_child_latent``'s and
    ``student_t_likelihood``'s Student-t), and every plated latent is
    Gaussian -- so
    :func:`~bayesmith.dispatch.classify._latent_centre`'s plate arm, the one
    that broadcasts a distribution's own ``mean`` out to the plate's size, had
    no fixture at all. A NumPyro distribution declared under a plate does not
    know it: ``dist.StudentT(6.0, 0.4, 0.9).shape()`` is ``()``, and only the
    plate says there are ``n`` of them.

    ``u``'s prior centre is 0.4, not 0.0, for the reason
    ``plated_latent_through_deterministic``'s is 0.8: an arm that returned
    ``jnp.zeros(shape)`` regardless -- which is what the two ``except``
    branches immediately below it do -- is indistinguishable from one that
    read the distribution, if the distribution's mean is zero.

    ``u`` is disqualified by criterion 1 and leaves; ``w`` is Gaussian, affine
    and stays, so this also exercises the degraded-at-points note (``u`` has
    no draw ``_prior_draw`` will take, and is held at its centre).
    """
    x = jnp.linspace(1.0, 3.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        obs = plate("obs", n)
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, 2.5))
        u = sample("u", lambda: dist.StudentT(6.0, 0.4, spread), plate=obs)
        mu = det(
            "mu",
            lambda w_, u_, x_: w_ * x_ + u_,
            w,
            u,
            xs,
            plate=obs,
            linear_in=("w", "u"),
        )
        observe("d", lambda m: dist.Normal(m, sigma), mu, plate=obs, obs=data)

    return trace(model)


def lying_block_member(*, n=6, sigma=0.5, w_true=1.4, seed=35):
    """The LYING density is on a block MEMBER, not on an observed node.

    Every other use of `LyingNormal` in this module and in
    `tests/exact/test_gaussian.py` puts it on an `observe()` node, so
    `unchecked_operator`'s per-observed-node probe was the only one any
    fixture could reach. `_env_before`'s per-MEMBER probe -- the second call
    site `probe_gaussian` has to switch off, and the whole reason that keyword
    is threaded through two functions -- had no fixture at all.

    Measured before this existed: deleting `check_gaussian` from
    `_env_before` entirely left the suite at 581 passed. It only LOOKED
    covered, because under `jax.jit` every `check_gaussian` raises whatever it
    is handed, so the trace-safety test is equally satisfied by whichever call
    site happens to run first -- and `_env_before`'s runs first. A genuinely
    non-Gaussian member is caught one line later by `gaussian_parts`, so the
    member probe's only unique contribution is exactly this: a member whose
    TYPE reads Gaussian while its own `log_prob` does not.
    """
    x = jnp.linspace(1.0, 2.0, n)
    data = w_true * x + sigma * jax.random.normal(jax.random.key(seed), (n,))

    def model():
        xs = const("X", x)
        w = sample("w", lambda: LyingNormal(0.0, 2.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def flagged_line(
    *,
    n=8,
    slope=1.5,
    intercept=-3.0,
    sigma=0.4,
    prior_std=5.0,
    garbage=1e3,
    flagged=(2, 5),
    seed=40,
):
    """``d ~ N(a X + b, sigma)`` with two channels flagged and full of garbage.

    **The model's arithmetic is written once and wrapped twice**, which is the
    point of returning both graphs from one function rather than building them
    in the test. A masking bug and a fixture that quietly describes two
    different models look identical from the assertion's side, and this
    package has paid for that once already (a graph fixture missing an
    additive offset read exactly like a solver defect).

    Returns ``(masked, kept, mask)``:

    * ``masked`` sees all ``n`` samples and DECLARES which were taken;
    * ``kept`` sees only the samples that were taken and declares nothing;
    * ``mask`` is the boolean, ``True`` = taken.

    The two must give the same posterior. The flagged entries carry
    ``garbage`` -- three orders of magnitude off the line -- so that ignoring
    the mask is not a small error but a visible one: a guard that cannot tell
    a masked solve from an unmasked one is not a guard.
    """
    x = jnp.linspace(-2.0, 2.0, n)
    clean = slope * x + intercept + sigma * jax.random.normal(
        jax.random.key(seed), (n,)
    )
    mask = jnp.ones((n,), dtype=bool).at[jnp.asarray(flagged)].set(False)
    data = jnp.where(mask, clean, garbage)

    def build(xs, obs, node_mask):
        def model():
            coordinate = const("X", xs)
            a = sample("a", lambda: dist.Normal(0.0, prior_std))
            b = sample("b", lambda: dist.Normal(0.0, prior_std))
            mu = det(
                "mu",
                lambda a_, b_, x_: a_ * x_ + b_,
                a,
                b,
                coordinate,
                linear_in=("a", "b"),
            )
            observe("d", lambda m: dist.Normal(m, sigma), mu, obs=obs, mask=node_mask)

        return trace(model)

    return build(x, data, mask), build(x[mask], data[mask], None), mask
