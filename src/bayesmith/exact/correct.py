"""Correcting a frozen-sigma proposal back to the exact conditional.

Freezing sigma makes each step a linear-Gaussian problem, which is the whole
reason GCR applies -- and it is also exactly the step :mod:`bayesmith.exact.gls`
names as the reason its fixed point is the GLS optimum rather than the
likelihood's: "the log-determinant's dependence on the solution is held fixed
rather than differentiated". The importance weight below contains precisely
the two terms that were frozen. This is not a patch on an approximation; it is
putting a recorded discrepancy back.

**What is being corrected, quantitatively.** ``gls.py``'s own docstring
measures the gap on ``radiometer()``: 0.08% at its default ``kappa=0.05``,
19.5% at ``kappa=1``, and ~50% with the two estimates on OPPOSITE sides of
the truth by ``kappa=3.5``. That is the quantity these weights exist to
remove, and it is why the diagnostics below matter as much as the weight
itself -- a correction whose effective sample size is 1 has not corrected
anything, it has just relabelled one draw.

**Where the diagnostics run.** Nowhere near a trace. :func:`self_normalise`
is two jittable lines, but :func:`khat` and :func:`unreliable` are Python:
``_psis_khat`` sorts and then indexes with a data-dependent boolean mask, and
``unreliable`` branches on the result. That costs nothing, because SNIS has no
loop to jit -- the expensive part is generating the draws, which happens
before any of this.
"""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.exact.block import LinearBlock, variance_parts
from bayesmith.exact.solve import _weights, normal_operator
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.graph import Graph

#: PSIS shape parameters at which the importance-sampling estimator loses a
#: moment. ``k < 0.5``: finite variance. ``k < 1``: finite MEAN. These are
#: **moment-existence** bounds and are not the same thing as the reliability
#: threshold :func:`unreliable` applies -- an earlier draft conflated 0.7
#: (empirical, sample-size dependent) with 1.0 (where the mean stops
#: existing), which reads as a far weaker guarantee than it is.
FINITE_VARIANCE_KHAT: float = 0.5
FINITE_MEAN_KHAT: float = 1.0

#: The cap in Vehtari et al. (2024)'s ``min(1 - 1/log10(N), 0.7)``. The two
#: branches cross at ``N = 10 ** (1/0.3) = 2154``: below it the formula is
#: what binds and a hard-wired 0.7 is optimistic, above it the cap is.
KHAT_THRESHOLD_CAP: float = 0.7

#: Below this many draws ``1 - 1/log10(N)`` is not usable: it is <= 0 for
#: ``N <= 10``, undefined at ``N = 1``, and -- worse -- comes out at ``+1.0``
#: for ``N = 0`` because ``log10(0)`` is ``-inf``, which would report the
#: emptiest possible sample as the most reliable one.
KHAT_MIN_DRAWS: int = 10


def log_weight(
    graph: Graph,
    block: LinearBlock,
    x: dict[str, Any],
    *,
    at: dict[str, Any],
    noise_std: dict[str, Any],
    mu: dict[str, Any],
) -> jax.Array:
    """``log p(x, z, d) - log q(x)``, up to a constant common to every draw.

    ``q = N(mu, M^-1)`` with ``M = A^T N^-1 A + S^-1`` at the FROZEN sigma, so

        log w = log_joint(graph, {**at, **x}) + 1/2 (x-mu)^T M (x-mu) + C ,
        C = -1/2 log det M + (n/2) log 2pi .

    ``C`` is dropped. It is identical for every draw **because sigma is frozen
    at a value that does not depend on x**: ``noise_std`` is a decided dict,
    not a rule, so ``M`` -- and therefore ``log det M`` -- is the same operator
    for every draw this function is called on. Both consumers cancel it:
    self-normalisation because ``softmax`` is shift-invariant, an MH ratio
    because it takes a difference.

    The sign matters and is easy to get backwards. Since
    ``log q(y) = -1/2 r^T M r + 1/2 log det M - (n/2) log 2pi``, subtracting it
    makes the quadratic term and the log-det term carry **OPPOSITE** signs --
    so a ``C`` written with ``+1/2 log det M`` is wrong under every convention
    rather than merely a different one. Both P3's spec and P3b's first draft
    inverted both signs; measured against a dense reference, the draft's ``C``
    is off by +2.396e-01 on a 4-dimensional case while the corrected one is
    bitwise zero. ``test_the_documented_constant_makes_the_weight_the_log_evidence``
    is what keeps that claim live: ``C`` does not appear in this function's
    body, so nothing else in the suite can check the sign the docstring states.

    Args:
        graph: the model. ``log p`` is its own ``log_joint``, so this cannot
            drift from what NUTS targets.
        block: the block ``q`` was built for, from
            :func:`~bayesmith.exact.linearity.linear_operator`.
        x: ``{member: draw}`` over the block's domain. ``vmap`` over the draw
            axis for a whole sample; nothing here branches on a value.
        at: the latents OUTSIDE the block, exactly as when ``block`` was
            built. **Unverifiable here** -- a ``LinearBlock`` does not record
            the ``at`` it was built at, so passing a different one silently
            weights draws against a ``q`` from a different conditional.
        noise_std: the FROZEN ``{observed: sigma}`` that ``q`` was drawn at --
            for a prediction-dependent model, ``GLSResult.noise_std``. Not
            recomputed at ``x``: recomputing it is precisely the dependence
            this weight exists to correct, and doing it here would put the
            correction on both sides and cancel it.
        mu: ``q``'s centre, i.e. the GLS/Wiener solution at that same sigma.

    Returns:
        A scalar, up to ``C``. Only differences between draws are meaningful.

    Note:
        Costs ONE application of ``M`` (a JVP plus a VJP) and one graph scan
        per draw. The plan's draft called ``operator(delta)`` inside the
        summation, i.e. once per block member, which is the same answer at
        ``len(block.names)`` times the cost -- a five-member block paid five
        JVP/VJP pairs to use one leaf of each.
    """
    operator = normal_operator(block, _weights(noise_std), variance_parts(block))
    delta = jax.tree.map(jnp.subtract, x, mu)
    pushed = operator(delta)
    quadratic = 0.5 * sum(jnp.sum(delta[name] * pushed[name]) for name in delta)
    return log_joint(graph, {**at, **x}) + quadratic


def self_normalise(log_weights: Any) -> tuple[jax.Array, jax.Array]:
    """``(normalised weights, Kish ESS)``. ESS is ``1 / sum(w^2)``.

    ``softmax`` rather than ``exp``-then-divide, and that is load-bearing
    rather than tidy: importance weights arrive with an arbitrary additive
    constant (see :func:`log_weight`, which drops one), so a naive
    normaliser overflows on exactly the input this function is defined to be
    invariant to. ``test_the_weight_constant_cancels_between_draws`` shifts
    by +800, where ``exp`` is ``inf`` in float64 and softmax is unmoved.

    Kish's ESS is the count of equally-weighted draws carrying the same
    information: ``N`` when every weight is equal, ``1`` when one draw has all
    of it. An SNIS mean's uncertainty is ``variance / ESS``, i.e. ``sd /
    sqrt(ESS)`` -- so ESS is what turns a weighted mean into an error bar, and
    an ESS of 1 says the answer is one draw wearing a sample's clothes.

    **It degrades exponentially in the number of mismatched coordinates**, not
    gracefully. Measured with ``A = I``, ``sigma_i = 0.3|x_i| + 0.05``, the
    proposal at the GLS fixed point, N=40000: n=1 gives ESS 2509; n=25 gives
    124; n=50 gives 4.5; n=100 gives 6.0; and by n=500 it is 1.00 and stays
    there at n=4000. That is not a pathological input -- it is a mild
    radiometer -- which is why a dispatcher has to read this number rather
    than assume the correction worked.

    Args:
        log_weights: unnormalised log weights, any shape ``softmax`` accepts.
            Normalisation runs over every axis.

    Returns:
        ``(weights, ess)``. ``weights`` sums to 1; ``ess`` is a scalar.
    """
    weights = jax.nn.softmax(log_weights)
    return weights, 1.0 / jnp.sum(weights**2)


def khat(log_weights: Any) -> float | None:
    """PSIS k-hat, or ``None`` if numpyro's private entry point is gone.

    ``numpyro.infer.importance._psis_khat`` is private, so its existence is
    pinned by a test rather than trusted, and its absence degrades to "no
    diagnostic" rather than to a traceback out of a dispatcher --
    ``test_khat_is_none_rather_than_an_exception_when_the_private_entry_is_gone``.

    It cannot be jitted, and not for the reason its signature suggests:
    returning a Python ``float`` is a symptom. The cause is that it does
    ``np.sort`` and then indexes with a boolean mask (``log_weights >
    lw_cutoff``) whose output shape depends on the data, which has no jit
    expression at all. Rewriting the return type would not help.

    Args:
        log_weights: unnormalised log weights, 1-D. ``_psis_khat`` subtracts
            its own maximum, so the dropped ``C`` of :func:`log_weight` does
            not reach the fit. Not promoted to float64 first: measured on the
            same samples cast both ways, f32 and f64 inputs agree to within
            2.3e-7 in ``k``, so the suite's float32 default costs nothing
            here.

    Returns:
        ``k``, or ``None``. Calibration, so a caller knows what "small" is:
        for Gaussian log weights of sd 0.5 at N=2000, 20 seeds gave mean
        0.076, sd 0.066, span -0.017..0.184 -- and it is strongly N-dependent
        at fixed sd (N=200 -> 0.275, N=2000 -> 0.156, N=20000 -> -0.028), so
        a bare point value is not a property of the estimator. Raising the sd
        to 3.0 moves it to 1.0..1.24 across seeds, i.e. past
        :data:`FINITE_MEAN_KHAT`.
    """
    try:
        from numpyro.infer.importance import _psis_khat
    except (ImportError, AttributeError):
        return None
    return float(_psis_khat(np.asarray(log_weights)))


def unreliable(khat_value: float | None, n: int) -> bool:
    """PSIS's reliability threshold, which is sample-size dependent.

    ``min(1 - 1/log10(N), 0.7)`` (Vehtari et al. 2024) -- **below 0.7 for
    every N under 2154**, so a hard-wired 0.7 is optimistic exactly where SNIS
    is run without a chain and N is the caller's own choice. At N=100 the
    threshold is 0.5; at N=1000, 0.667.

    Note also that the moment bands are :data:`FINITE_VARIANCE_KHAT` for a
    finite variance and :data:`FINITE_MEAN_KHAT` for a finite MEAN; 0.7 is an
    empirical reliability threshold, not where the mean stops existing.

    Args:
        khat_value: from :func:`khat`. ``None`` -- the diagnostic being
            unavailable -- returns ``False``, which is **abstention, not
            endorsement**: the caller must still read the Kish ESS, which is
            always available and is the cheaper of the two guards anyway.
        n: number of draws the k-hat was computed from. At or below
            :data:`KHAT_MIN_DRAWS` the formula is unusable and every k-hat is
            reported unreliable -- ``1 - 1/log10(N)`` is <= 0 there, so that
            branch is the formula's own limit and not an extra rule. Clamping
            ``n`` upward instead (``max(n, 11)``, as the plan's draft did)
            inverts the answer in the one corner it exists for: it reports
            k=0.02 at N=1 as RELIABLE.

    Returns:
        Whether ``khat_value`` is at or above the threshold for this ``n``.
        A real ``bool``, not a ``numpy.bool_``, so a caller may write
        ``is True`` -- the comparison is against a Python float only when
        :func:`khat` produced the value, and a caller passing a numpy scalar
        would otherwise get something that fails an identity test.
    """
    if khat_value is None:
        return False
    if n <= KHAT_MIN_DRAWS:
        return True
    return bool(khat_value >= min(1.0 - 1.0 / math.log10(n), KHAT_THRESHOLD_CAP))
