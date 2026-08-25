"""The `gibbs_fn` numpyro calls, and the independence-proposal Metropolis step.

Four of the guards below run inside ``with jax.enable_x64(True):`` and six do
not, and the split is not arbitrary -- but it is also not what a first guess
would say, so it was measured. Rerunning the whole module with the context
manager neutered, **14 of 17 rows still pass**: the three that compare a drawn
scatter against the dense oracle are nowhere near the float32 floor. Measured
on ``steep_radiometer`` at ``tol=1e-10``, float32 ``gcr_sample`` returns a
relative residual of 1.82e-07 against a condition bound of 3.646e+03, so the
bound on the relative ERROR is 6.6e-04 -- two decades under those tests' 3%
and 5% tolerances. What x64 buys them is that margin, which depends on the
condition bound (1.299e+04 on ``contrast_sigma_pair``'s ``{a}`` block at
``b = 2.0``) and would be eaten by a fixture change nobody connected to it.

For ``test_the_mh_step_leaves_the_exact_conditional_invariant`` x64 is
load-bearing, for a reason that is not precision at all. ``jax.random.normal``
returns **different values**, not merely rounded ones, at the two dtypes:
``steep_radiometer``'s data is ``[2.017, 0.831, 2.087, 4.469, 2.629, -0.269]``
in float64 and ``[2.593, 3.922, 5.360, 2.234, 6.095, 7.314]`` in float32. So
the two dtypes are two DATASETS, the acceptance rate is a property of the
dataset (0.456-0.477 in float64, 0.726-0.734 in float32), and every number in
``steep_radiometer``'s docstring -- including the 111-123 draws the section
5.3 mutation needs -- was measured on the float64 one. That is the row that
fails when the context manager is removed, and it fails on the acceptance
band alone: invariance itself still holds, because the float32 sigma-hat is a
different but still x-INDEPENDENT choice, which is the "correctness does not
depend on sigma-hat being any good" claim happening by accident.

The remaining six assert a *structure* -- a signature, a set of returned keys,
finiteness under trace, a refusal -- that no dtype affects, and pay nothing
for float32.

Every x64 test traces its model INSIDE the ``with`` block: ``const`` and
``observe`` call ``jnp.asarray`` at trace time, so a graph built outside stays
float32 no matter where it is solved.
"""

from __future__ import annotations

import inspect
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.gibbs import assemble, gibbs_factory
from bayesmith.graph.evaluate import log_joint
from tests.exact.models import (
    contrast_sigma_pair,
    mixed_radiometer,
    steep_radiometer,
    three_latent_chain,
)
from tests.exact.oracle import graph_oracle

#: Where `test_gibbs_fn_uses_the_hmc_sites_it_was_handed` freezes sigma on
#: `mixed_radiometer`. Stated on BOTH sides of that test -- handed to the
#: factory as `precision=` and to the oracle as `sigma_at=` -- so the two
#: agree on the frozen covariance without either reading the other's choice.
#: Not 0.0 (the block's prior mean and zero), not 2.6 (`weight`), not 4.2
#: (`tau`'s prior mean): a value distinct from every parameter in the fixture.
MIXED_FREEZE = 8.0


def _quadrature(graph, name, lo=-4.0, hi=8.0, points=2001):
    """``(grid, density, mean, sd)`` of a ONE-dimensional posterior.

    Independent of every linear-algebra path in this package: it evaluates
    ``log_joint`` on a grid and integrates by trapezium, so nothing it
    produces can move when ``unchecked_operator``, ``normal_operator``,
    ``cg`` or ``log_weight``'s quadratic form moves. That independence is the
    whole reason it is here -- an MH kernel checked against the same frozen
    Gaussian it proposes from is checked against itself.

    Insensitive to both knobs, measured on ``steep_radiometer()``: mean
    1.39090776 / sd 0.29251207 at ``(-4, 8, 801)``, ``(-4, 8, 4001)`` and
    ``(-4, 8, 40001)`` alike, against 1.39090928 / 0.29253306 on
    ``(-20, 30, 100001)`` -- seven digits of agreement over a 4x wider window
    and a 50x finer grid.
    """
    grid = jnp.linspace(lo, hi, points)
    log_p = jax.vmap(lambda value: log_joint(graph, {name: value}))(grid)
    density = jnp.exp(log_p - jnp.max(log_p))
    density = density / jnp.trapezoid(density, grid)
    mean = float(jnp.trapezoid(density * grid, grid))
    second = float(jnp.trapezoid(density * grid**2, grid))
    return np.asarray(grid), np.asarray(density), mean, math.sqrt(second - mean**2)


def _quantile_sample(grid, density, count):
    """``count`` midpoint quantiles of the tabulated density.

    Deterministic on purpose. The invariance test compares the step's OUTPUT
    against its own INPUT, so any Monte-Carlo error in the input is error the
    comparison has to carry; a quantile grid has essentially none. Measured on
    ``steep_radiometer()`` at 8000 points: mean 1.390885 against quadrature's
    1.390908, sd 0.292235 against 0.292512.
    """
    cell = (density[1:] + density[:-1]) / 2 * np.diff(grid)
    cdf = np.concatenate([[0.0], np.cumsum(cell)])
    cdf = cdf / cdf[-1]
    return np.interp((np.arange(count) + 0.5) / count, cdf, grid)


@pytest.mark.parametrize(
    "build,names,free,fixed,values,freeze,draws",
    [
        (three_latent_chain, ("y",), "x", {"tau": 2.4}, (0.05, 4.0), None, 4000),
        (mixed_radiometer, ("w",), "tau", {}, (0.02, 9.5), MIXED_FREEZE, 4000),
    ],
)
def test_gibbs_fn_uses_the_hmc_sites_it_was_handed(
    build, names, free, fixed, values, freeze, draws
):
    """Pins `at`, without the oracle sharing the choice.

    A direct "draws agree with the oracle at this `at`" assertion inherits
    P3a's shared-layer blind spot -- `graph_oracle` and `unchecked_operator`
    both go through `_env_before`, `isolate` and `observation_parts`, so a
    mutation to that layer moves BOTH sides and the comparison never moves.
    It also adds a NEW shared parameter: the test would have to hand the
    oracle the same `at` it fixed `hmc_sites` to.

    So this asserts a DIFFERENCE instead: run `gibbs_fn` at two values of one
    outside latent, NEITHER of them that latent's prior mean, and check the
    drawn mean moves by what the oracle predicts for that shift. A `gibbs_fn`
    that ignores `hmc_sites` and rebuilds `at` from prior means -- which is
    what compiling the plan at the prior environment tells the implementer to
    do, so the confusion is pre-installed -- produces a difference of exactly
    zero and fails here.

    **Why neither value may be the prior mean.** `mixed_radiometer`'s `tau`
    is `Normal(4.2, 0.7)`, not zero-mean, and the prior-mean rebuild lands at
    a posterior mean of 2.658 -- which is 0.006 away from the tau=9.5 arm's
    2.664. One arm compared against its own oracle would therefore pass with
    the mutation in place. The DIFFERENCE is 0.662 against the mutation's
    0.000, which is why the difference is what is asserted.

    Both rows use the same key for both arms, so the two draw sets share
    their white noise and the difference is far more accurate than either
    mean: measured relative error 0.16% (row 1) and 0.15% (row 2) against a
    2% tolerance, where the UNPAIRED standard error would be 1.7% and 0.9%.

    Row 2 freezes sigma explicitly at `MIXED_FREEZE`. `mixed_radiometer`'s
    sigma is `kappa |mu| + floor` and its block's prior mean is 0.0, so the
    factory's own freeze point puts sigma at `floor` = 1.3e-3 -- a likelihood
    30 million times tighter than the prior, under which `tau` (which reaches
    the block ONLY through `w`'s prior width) moves the posterior mean by
    0.008 posterior sd and would need ~60,000 draws. Handing the factory an
    x-independent sigma of the caller's own choosing is a documented use of
    `precision=`, and it is also the claim that correctness does not depend
    on sigma-hat being any good being exercised rather than asserted.
    """
    with jax.enable_x64(True):
        graph = build()
        precision = (
            None
            if freeze is None
            else precision_at(
                graph,
                {
                    latent: jnp.asarray(
                        freeze if latent in names else fixed.get(latent, values[0])
                    )
                    for latent in graph.latents
                },
            )
        )
        fn = gibbs_factory(graph, names, tol=1e-10, precision=precision)
        sigma_at = None if freeze is None else {n: jnp.asarray(freeze) for n in names}
        drawn, predicted = [], []
        for value in values:
            at = {k: jnp.asarray(v) for k, v in fixed.items()}
            at[free] = jnp.asarray(value)
            keys = jax.random.split(jax.random.key(0), draws)
            out = jax.vmap(
                lambda key, where=at: fn(
                    rng_key=key,
                    gibbs_sites={n: jnp.asarray(0.0) for n in names},
                    hmc_sites=where,
                )[names[0]]
            )(keys)
            drawn.append(float(jnp.mean(out)))
            predicted.append(
                float(graph_oracle(graph, names, at=at, sigma_at=sigma_at).mean[0])
            )
        assert drawn[0] - drawn[1] == pytest.approx(
            predicted[0] - predicted[1], rel=2e-2
        )


@pytest.mark.parametrize("draws,key_seed", [(6000, 1), (6000, 17)])
def test_gibbs_fn_freezes_sigma_where_it_says_it_does(draws, key_seed):
    """Pins the freeze point, with BOTH wrong arms named.

    `graph_oracle`'s `sigma_at` defaults to the block's ZERO while
    `_env_before` centres members at their PRIOR MEAN -- two different freeze
    points, both live in this repository. Whatever the test passes for
    `sigma_at` is what makes the comparison agree, so a mutation that freezes
    sigma at the wrong point is absorbed unless the wrong arms are asserted
    too.

    `steep_radiometer`'s prior mean is **0.8, not 0.0**, precisely so that the
    block's zero is a genuinely different freeze point: the posterior sd there
    is 1.93e-3 against the declared point's 1.656e-01, 86x apart. The second
    wrong arm, `w = 3.0`, is 3.7x the other way, so a mutation cannot land
    between them by accident.

    This runs `method="gcr"` on a graph whose sigma DOES move with the block
    -- deliberately. `"gcr"` is exactly the claim "sigma is frozen and the
    draw is exact at that sigma", and what is under test is WHERE it is
    frozen, not whether freezing was warranted; `steep_radiometer` is the
    fixture where the three candidate points are far enough apart to tell.
    """
    with jax.enable_x64(True):
        graph = steep_radiometer()
        fn = gibbs_factory(graph, ("w",), tol=1e-10)
        keys = jax.random.split(jax.random.key(key_seed), draws)
        out = jax.vmap(
            lambda key: fn(
                rng_key=key, gibbs_sites={"w": jnp.asarray(0.0)}, hmc_sites={}
            )["w"]
        )(keys)
        got = float(jnp.std(out))
        # Written out here rather than read off the implementation, so a
        # mutation that moves the freeze point cannot move the expectation
        # with it. 0.8 is `prior_mean`; the factory's documented choice is the
        # block's prior mean given `at`.
        declared = graph_oracle(graph, ["w"], sigma_at={"w": jnp.asarray(0.8)})
        assert got == pytest.approx(float(np.sqrt(declared.covariance[0, 0])), rel=3e-2)
        for wrong_point in (0.0, 3.0):
            wrong = graph_oracle(graph, ["w"], sigma_at={"w": jnp.asarray(wrong_point)})
            assert got != pytest.approx(
                float(np.sqrt(wrong.covariance[0, 0])), rel=3e-2
            )


@pytest.mark.parametrize("outside,draws", [(0.5, 4000), (2.0, 4000)])
def test_noise_std_is_rebuilt_when_sigma_depends_on_a_latent_outside_the_block(
    outside, draws
):
    """`check_prediction_dependence` is structurally blind to this case.

    It only ever moves BLOCK members. On `contrast_sigma_pair` split as block
    `{a}` with `b` outside, sigma is `base * exp(a - b)`, so the probe reads
    **1.71828183e+00 -- `e - 1`, bitwise identical at `b = 0.5` and at
    `b = 2.0`** -- while sigma itself moves by a factor `exp(1.5) = 4.48`
    between those two. The number a dispatcher thresholds cannot see `b` at
    all, which is why `sigma_needs_rebuild` is a STRUCTURAL criterion (does
    any observed node's scale have a latent ancestor) and not a movement
    measurement. `_sigma_needs_rebuild(graph, ("a",))` is `True` here.

    Measured cost of getting it wrong: hoisting sigma to the graph's prior
    centre gives a drawn sd of 0.064471 at BOTH values of `b` -- it is
    b-independent by construction -- against oracles of 0.039295 and
    0.008774, i.e. **1.64x and 7.35x too wide**.

    Asserted on the sd, not the mean: the same hoist moves the mean by only
    0.4% (-1.4345 against -1.4400 at `b = 2.0`), which a 5% tolerance would
    wave through. The covariance is where a frozen sigma goes wrong.
    """
    with jax.enable_x64(True):
        graph = contrast_sigma_pair(n=12)
        fn = gibbs_factory(graph, ("a",), tol=1e-10, sigma_rebuild=True)
        at = {"b": jnp.asarray(outside)}
        keys = jax.random.split(jax.random.key(2), draws)
        out = jax.vmap(
            lambda key: fn(
                rng_key=key, gibbs_sites={"a": jnp.asarray(0.0)}, hmc_sites=at
            )["a"]
        )(keys)
        oracle = graph_oracle(graph, ["a"], at=at)
        assert float(jnp.std(out)) == pytest.approx(
            float(np.sqrt(oracle.covariance[0, 0])), rel=5e-2
        )


@pytest.mark.parametrize("draws,key_seed", [(6000, 4), (6000, 13), (6000, 99)])
def test_the_mh_step_leaves_the_exact_conditional_invariant(draws, key_seed):
    """Spec section 5.3's whole point, against an oracle that shares nothing.

    Invariance is the property an MH kernel has to have and the one a
    frozen-sigma draw does NOT: apply one step to a sample of `p(w | d)` and
    the sample must still be one. The starting sample comes from 1-D
    quadrature of `log_joint`, which touches no operator, no CG and no
    importance weight, so the comparison cannot be satisfied by two readings
    of the same wrong linear algebra.

    Compared PAIRED -- output minus its own input -- because rejection
    returns the input bitwise, which makes the two enormously correlated and
    the difference far more sensitive than either sample's mean. Measured
    over keys 4/13/99 at 8000 draws: +0.0 to +0.9 standard errors, and the
    sd changes by -0.15% to -0.34%.

    **What the three clauses each kill.**

    * The shift clause kills spec section 5.3's original error -- sigma-hat
      rebuilt at the CURRENT state, which makes the proposal adaptive, makes
      ``M' != M``, and leaves an uncancelled ``1/2 log det M`` that a
      matrix-free method cannot even compute. Measured -16.1 to -17.0
      standard errors at 8000 draws; two sigma at 111-123 draws.
    * The sd clause kills the same thing from the other side: that mutation
      narrows the sd by 13.0-17.3%, against a 4% tolerance and a 0.9%
      standard error at these draw counts.
    * The acceptance clause kills a kernel that is the IDENTITY. Dropping the
      reverse-density term ``log_weight(now)`` from the ratio leaves
      ``log alpha = log w(x')``, which on this fixture is so negative that
      **not one proposal in 96,000 was accepted** -- and an identity kernel
      passes an invariance test trivially. The correct step accepts
      0.456-0.477 of the time.

    Run on `steep_radiometer`, whose block spans the whole graph, so `at` is
    empty and quadrature is available. The classifier would route that graph
    to `gcr+snis`; `method="gcr+mh"` is asked for explicitly here because the
    kernel is what is under test, and this is the only shape on which an
    independent 1-D reference exists.
    """
    with jax.enable_x64(True):
        graph = steep_radiometer()
        fn = gibbs_factory(graph, ("w",), tol=1e-10, method="gcr+mh")
        grid, density, _, _ = _quadrature(graph, "w")
        start = jnp.asarray(_quantile_sample(grid, density, draws))
        keys = jax.random.split(jax.random.key(key_seed), draws)
        out = jax.vmap(
            lambda key, current: fn(
                rng_key=key, gibbs_sites={"w": current}, hmc_sites={}
            )["w"]
        )(keys, start)

        shift = out - start
        standard_error = float(jnp.std(shift)) / math.sqrt(draws)
        assert abs(float(jnp.mean(shift))) < 4.0 * standard_error
        assert float(jnp.std(out)) == pytest.approx(float(jnp.std(start)), rel=4e-2)
        accepted = float(jnp.mean(out != start))
        assert 0.25 < accepted < 0.70


def test_gibbs_fn_survives_a_trace_and_the_probe_still_bites():
    """`probe_gaussian=False`'s whole reason, exercised. Milliseconds, no MCMC.

    A test that runs `gibbs_fn` on concrete values is GREEN whether the probe
    is disabled or not -- concrete `check_gaussian` works fine. It calls
    `bool(jnp.all(...))`, so only opening a trace tells the two apart, and
    then it is a `TracerBoolConversionError` rather than a slow path.
    """
    graph = mixed_radiometer()
    fn = gibbs_factory(graph, ("w",), tol=1e-8)
    out = jax.jit(
        lambda key, tau: fn(
            rng_key=key,
            gibbs_sites={"w": jnp.asarray(0.0)},
            hmc_sites={"tau": tau},
        )
    )(jax.random.key(0), jnp.asarray(3.1))
    assert bool(jnp.isfinite(out["w"]))


@pytest.mark.parametrize("method", ["gcr", "gcr+mh"])
def test_gibbs_fn_is_called_by_keyword_and_returns_exactly_the_block(method):
    """Pins numpyro's contract, which is a keyword one.

    `HMCGibbs.sample` calls `self._gibbs_fn(rng_key=..., gibbs_sites=...,
    hmc_sites=...)`, so the PARAMETER NAMES are the contract and their order
    is free -- and a positional-only `/` in the signature raises TypeError.
    Measured on numpyro 0.21.0: returning a SUBSET raises an `AssertionError`
    with an empty message that names nothing, and returning an EXTRA key
    naming a NUTS latent is silently accepted and ignored (the run finishes
    and `get_samples()` still has `tau` from NUTS). So "exactly" is asserted
    rather than "at least".

    Both methods are exercised because they RETURN from different places --
    `gcr` from the `gcr_sample` branch, `gcr+mh` from `_mh_step`'s
    `jnp.where`. A mutation that leaks an extra key into one of them is
    invisible to a test that only runs the other, and `at` is sitting right
    there in `_mh_step`'s namespace waiting to be merged in.

    [EXPERIMENTAL INTERFACE] upstream: pinning it makes a change there a
    deliberate decision instead of a surprise.
    """
    graph = mixed_radiometer()
    fn = gibbs_factory(graph, ("w",), tol=1e-8, method=method)
    params = inspect.signature(fn).parameters
    assert set(params) == {"rng_key", "gibbs_sites", "hmc_sites"}
    assert all(p.kind is not p.POSITIONAL_ONLY for p in params.values())
    out = fn(
        rng_key=jax.random.key(0),
        gibbs_sites={"w": jnp.asarray(0.0)},
        hmc_sites={"tau": jnp.asarray(3.1)},
    )
    assert set(out) == {"w"}


def test_vectorized_chains_are_refused_with_a_reason():
    """numpyro 0.21.0's `HMCGibbs.init` splits `rng_key` unconditionally.

    `HMC.init` has an `rng_key.ndim` branch; `HMCGibbs.init` does not, so
    under `chain_method="vectorized"` MCMC hands it a batched key and it
    raises `ValueError: split accepts a single key, but was given a key array
    of shape (2,) != ()` -- before `gibbs_fn` is ever reached. Measured on
    this repository's own assembly: `sequential` and `parallel` both complete.

    Refusing here turns an upstream stack trace into a sentence, and pins the
    limitation so a numpyro upgrade that fixes it shows up as a deliberate
    decision rather than a silent behaviour change.
    """
    graph = mixed_radiometer()
    with pytest.raises(NotImplementedError, match="vectorized"):
        assemble(
            graph,
            ("w",),
            tol=1e-8,
            method="gcr+mh",
            sigma_rebuild=True,
            num_chains=2,
            chain_method="vectorized",
        )


@pytest.mark.parametrize("chain_method", ["sequential", "parallel"])
def test_assemble_runs_the_sweep_end_to_end(chain_method):
    """The two chain methods that do work, through numpyro rather than around it.

    Everything above calls `gibbs_fn` directly, which is how it can be
    checked against an oracle -- but nothing above would notice if
    `gibbs_sites` were spelled wrong, if the inner kernel were handed the
    unwrapped model, or if the returned dict were keyed by something numpyro
    does not recognise. This is short (50 warmup, 100 samples) because its
    job is to reach the end, not to converge.
    """
    graph = mixed_radiometer()
    mcmc = assemble(
        graph,
        ("w",),
        tol=1e-8,
        method="gcr+mh",
        sigma_rebuild=True,
        num_warmup=50,
        num_samples=100,
        num_chains=2,
        chain_method=chain_method,
    )
    mcmc.run(jax.random.key(0))
    samples = mcmc.get_samples()
    assert set(samples) >= {"w", "tau"}
    assert samples["w"].shape == (200,)
    assert bool(jnp.all(jnp.isfinite(samples["w"])))


def test_assemble_refuses_a_block_that_leaves_nothing_for_nuts():
    """A whole-graph block has no HMC site, and HMCGibbs is then a lie.

    `steep_radiometer`'s only latent is `w`. Handing that to `HMCGibbs` builds
    an inner NUTS with an empty latent space; the sweep would consist of the
    Gibbs step alone, with the frozen-sigma approximation never corrected.
    That graph's row in the dispatch table is `gcr+snis` -- one
    self-normalised reweighting of iid draws, no chain at all -- so the
    refusal names it rather than letting the wrong path start.
    """
    graph = steep_radiometer()
    with pytest.raises(ValueError, match="every latent"):
        assemble(graph, ("w",), tol=1e-8)


def test_an_unknown_method_is_refused_by_name():
    """`method` selects between two corrections and there is no third.

    A typo would otherwise fall through to whichever branch is written as the
    `else`, which is the MH one -- so a graph asking for plain `gcr` would
    silently pay for and receive a Metropolis correction, or the reverse.
    """
    graph = mixed_radiometer()
    with pytest.raises(ValueError, match="gcr\\+mh"):
        gibbs_factory(graph, ("w",), tol=1e-8, method="snis")
