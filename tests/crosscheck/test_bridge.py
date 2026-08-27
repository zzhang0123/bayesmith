"""``numpyro_bridge.py`` against ``bridge/``: the §四 4.2 row, and the last
one blocking §六.

The row does not ask for a numerical comparison -- both sides hand the same
joint to the same sampler -- but for **three rheplicant-specific things** to
be carried:

1. an ``init_to_declared`` equivalent, where *"带过去的是教训不是代码"* --
   the lesson, not the code;
2. ``predict_from_samples``' shape guard;
3. the Jeffreys factor site's "the density is added only once".

Each is measured here on this side rather than assumed to transfer, and two
of the three turned out to be reachable failures.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import init_to_value

pytestmark = pytest.mark.crosscheck

N_CHANNEL = 24
#: A needle: the amplitude's prior is 1e4 wide and its posterior 0.4.
TRUE_AMPLITUDE, TRUE_INDEX = 5000.0, -2.5
AMPLITUDE_PRIOR_STD = 1e4
CHANNEL_SIGMA = 2.0

WARMUP, DRAWS, CHAINS = 400, 400, 2


def _needle_graph():
    """A power law whose posterior is ~1e4 times narrower than its prior."""
    from bayesmith import const, det, observe, sample, trace

    freq = jnp.linspace(60.0, 85.0, N_CHANNEL)
    truth = TRUE_AMPLITUDE * (freq / 70.0) ** TRUE_INDEX
    data = truth + CHANNEL_SIGMA * jax.random.normal(jax.random.key(2), (N_CHANNEL,))

    def model():
        x = const("nu", freq)
        amp = sample("amp", lambda: dist.Normal(0.0, AMPLITUDE_PRIOR_STD))
        beta = sample("beta", lambda: dist.Normal(0.0, 10.0))
        mu = det("mu", lambda a, b, v: a * (v / 70.0) ** b, amp, beta, x)
        observe(
            "d",
            lambda m: dist.Normal(m, CHANNEL_SIGMA).to_event(1),
            mu,
            depends_on_prediction=False,
            obs=data,
        )

    return trace(model)


def _rhat_and_ess(samples, name):
    """r-hat and ESS, with a non-finite ESS read as the WORST case.

    numpyro returns ``nan`` for a chain that never moved, and ``nan < 10.0``
    is ``False`` -- so an assertion that a badly-initialised chain mixes
    badly is failed by the chain mixing as badly as it is possible to mix.
    Measured: on arm64 macOS the needle posterior's default init gives a
    small finite ESS here, and on the x86_64 CI runner it gives ``nan``.

    1.0 rather than 0.0 or an exception, because that is already this
    package's convention: :func:`~bayesmith.dispatch.execute.chain_ess` fixes
    a non-finite ESS at 1.0, on the grounds that one draw's worth of
    information is the least a non-empty sample can carry. Inventing a second
    rule for the same quantity is how the two drift.
    """
    stacked = np.asarray(samples[name]).reshape(CHAINS, -1)
    ess = float(effective_sample_size(stacked))
    return (
        float(np.max(split_gelman_rubin(stacked))),
        ess if math.isfinite(ess) else 1.0,
    )


@pytest.mark.slow
def test_the_init_strategy_lesson_transfers_and_the_remedy_is_reachable():
    """§四 4.2's first item: the LESSON, not the code.

    rheplicant ships ``init_to_declared`` because NumPyro's default
    ``init_to_uniform`` draws in the unconstrained space with no knowledge
    of the declaration, and on its ring toy that is ``r_hat = 840`` with an
    effective sample size of 2 against ``r_hat = 1.002`` and 1327.

    There is no ``Latent(init=...)`` on a graph to port, so the question is
    whether the failure is reachable here at all. It is, and the shape is
    the same. Measured on this fixture, two chains of 400:

    ==============================================  ======  ======
    init                                            r_hat   ESS
    ==============================================  ======  ======
    default (what ``nuts()`` ships)                  ~1609    ~1.0
    ``init_to_value`` at the declared values         ~1.006  ~138.6
    ==============================================  ======  ======

    And the remedy needs no new code: ``nuts_options`` forwards straight to
    the ``NUTS`` kernel, and ``init_strategy`` is one of its keywords. What
    was missing was the sentence saying so, which is now in ``nuts``'s
    docstring with these numbers.

    Bounds, not pins, so a NumPyro release that changes the arithmetic does
    not fail this -- but a release that closes the gap will, and that is the
    moment to delete the paragraph this guards.
    """
    from bayesmith.bridge.numpyro_bridge import nuts

    graph = _needle_graph()
    shipped = nuts(
        graph,
        jax.random.key(0),
        num_warmup=WARMUP,
        num_samples=DRAWS,
        num_chains=CHAINS,
    )
    guided = nuts(
        graph,
        jax.random.key(0),
        num_warmup=WARMUP,
        num_samples=DRAWS,
        num_chains=CHAINS,
        nuts_options={
            "init_strategy": init_to_value(
                values={
                    "amp": jnp.array(TRUE_AMPLITUDE),
                    "beta": jnp.array(TRUE_INDEX),
                }
            )
        },
    )
    bad_rhat, bad_ess = _rhat_and_ess(shipped, "amp")
    good_rhat, good_ess = _rhat_and_ess(guided, "amp")
    assert bad_rhat > 100.0, ("the default no longer fails; delete the note", bad_rhat)
    assert bad_ess < 10.0, bad_ess
    assert good_rhat < 1.05, good_rhat
    assert good_ess > 50.0, good_ess


def test_rheplicant_still_ships_the_strategy_this_side_reaches_by_keyword():
    """The correspondence, so the record cannot go stale on either side.

    ``init_to_declared`` is a thin wrapper over ``init_to_value`` at the
    space's declared values -- which is exactly what a caller passes here
    through ``nuts_options``. If it ever becomes something else, this goes
    red and the paragraph in ``nuts``'s docstring needs re-reading.
    """
    from rheplicant.inference.numpyro_bridge import init_to_declared

    assert callable(init_to_declared)
    # The behavioural claim, and only it: `x or True` was here first, which
    # is an assertion that cannot fail and reads as one that can.
    from rheplicant.inference import Bind, Latent, ParameterSpace

    space = ParameterSpace(
        latents=[Latent("g", init=jnp.array(2.0))],
        bindings=[Bind("g", into=lambda p: p)],
    )
    strategy = init_to_declared(space)
    assert getattr(strategy, "func", strategy).__name__ == "init_to_value"


# --------------------------------------------------------------------------
# Item 2: predict_from_samples' shape guard, and the hole BOTH sides have.
# --------------------------------------------------------------------------


def _square_graph():
    """A length-3 latent, so a 3-draw stack is SQUARE and transposable."""
    from bayesmith import const, det, observe, sample, trace

    freq = jnp.linspace(1.0, 3.0, 3)

    def model():
        x = const("x", freq)
        c = sample("c", lambda: dist.Normal(jnp.zeros(3), 10.0).to_event(1))
        mu = det("mu", lambda v, xx: v * xx, c, x)
        observe("d", lambda z: dist.Normal(z, 1.0).to_event(1), mu, obs=jnp.zeros(3))

    return trace(model)


def test_a_transposed_sample_stack_is_refused_here_as_it_is_upstream():
    """§四 4.2's second item, ported because the failure is reachable.

    rheplicant checks each stack's PER-SAMPLE shape, and its own comment
    says why: *"Checking only the NAME lets a wrong-shaped stack broadcast
    into the leaf and return a finite, correctly-shaped, wrong
    predictive."*

    Measured on this side before the port, through NumPyro's ``Predictive``
    directly: a non-square transposition raises a broadcast ``TypeError``
    from three layers down that names neither the site nor the axis, and a
    SQUARE one raises nothing at all. ``bayesmith.predict`` now carries the
    same check, reading each latent's declared shape off its own
    ``dist_fn``.
    """
    from bayesmith.bridge.numpyro_bridge import predict
    from bayesmith.errors import GraphError

    with jax.enable_x64(True):
        graph = _square_graph()
        draws = jnp.arange(9.0).reshape(3, 3)
        good = np.asarray(predict(graph, {"c": draws})["mu"])
        # `(3, 2)`: two draws stacked along the WRONG axis, so the
        # per-sample shape reads `(2,)` against a latent of `(3,)`. A first
        # draft wrote `draws[:, :2].T`, which is `(2, 3)` -- per-sample
        # shape `(3,)`, correct, and correctly NOT refused. The bad stack
        # has to be bad in the axis the guard reads.
        with pytest.raises(GraphError, match="per-sample shape"):
            predict(graph, {"c": draws[:2].T})
        with pytest.raises(GraphError, match="missing latent"):
            predict(graph, {})
        # Two latents, so the draw counts have something to disagree about.
        # A first draft put an extra KEY in a one-latent graph and did not
        # raise: the guard walks `graph.latents`, so a name the graph does
        # not declare is never inspected -- which matches upstream, where
        # `predict_from_samples` also selects by `space.names`.
        two = _needle_graph()
        with pytest.raises(GraphError, match="disagree about the number"):
            predict(two, {"amp": jnp.zeros(3), "beta": jnp.zeros(5)})
    assert good.tolist() == [[0.0, 2.0, 6.0], [3.0, 8.0, 15.0], [6.0, 14.0, 24.0]]


def test_a_square_transposition_is_invisible_to_both_guards():
    """The limit, asserted so the guard is not read as complete.

    A shape check cannot separate a stack from its own transpose when the
    draw count equals the latent's size. **Neither package can**: rheplicant
    compares ``jnp.shape(samples[name])[1:]`` against the latent's shape,
    which is exactly what is preserved by transposing a square. Measured
    here, both stacks accepted, both finite, both correctly shaped, and
    every entry different:

        correct     [[0, 2, 6], [3, 8, 15], [6, 14, 24]]
        transposed  [[0, 6, 18], [1, 8, 21], [2, 10, 24]]

    Stated because a guard whose limits are not written down gets read as
    coverage. The remedy is not a better shape check -- there is no shape
    left to look at -- it is not building the stack by hand.
    """
    from bayesmith.bridge.numpyro_bridge import predict

    with jax.enable_x64(True):
        graph = _square_graph()
        draws = jnp.arange(9.0).reshape(3, 3)
        upright = np.asarray(predict(graph, {"c": draws})["mu"])
        flipped = np.asarray(predict(graph, {"c": draws.T})["mu"])
    assert upright.shape == flipped.shape == (3, 3)
    assert np.all(np.isfinite(flipped))
    assert not np.allclose(upright, flipped)
    assert flipped.tolist() == [[0.0, 6.0, 18.0], [1.0, 8.0, 21.0], [2.0, 10.0, 24.0]]


# --------------------------------------------------------------------------
# Item 3: the Jeffreys factor site, added once.
# --------------------------------------------------------------------------


def _jeffreys_graph(flat: bool):
    """A radiometer power law whose two latents are covered by the prior.

    ``flat=True`` declares them improper, which is what a joint prior over
    them requires; ``flat=False`` leaves their own proper densities in
    place, which is the double count.
    """
    from bayesmith import const, det, observe, sample, trace

    freq = jnp.linspace(60.0, 85.0, N_CHANNEL)
    truth = 3000.0 * (freq / 70.0) ** -2.5
    data = truth * (1.0 + 0.05 * jax.random.normal(jax.random.key(4), (N_CHANNEL,)))

    def prior_for(name):
        if flat:
            return lambda: dist.ImproperUniform(dist.constraints.real, (), ())
        return lambda: dist.Normal(0.0, 1e4)

    def model():
        x = const("nu", freq)
        amp = sample("fg_log_amp", prior_for("fg_log_amp"))
        beta = sample("fg_beta", prior_for("fg_beta"))
        mu = det("mu", lambda a, b, v: jnp.exp(a) * (v / 70.0) ** b, amp, beta, x)
        observe(
            "d",
            lambda m: dist.Normal(m, 0.05 * jnp.abs(m) + 1e-9),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return trace(model)


def test_a_joint_prior_over_a_latent_that_already_has_one_is_refused():
    """§四 4.2's third item: the density is added ONCE.

    The double count is the shape nothing downstream reports, because each
    density on its own is correct and their product is a proper density and
    a plausible chain. bayesmith refuses it by name at the point the prior
    is evaluated, and says what to do instead.
    """
    from bayesmith.diagnose.priors import JeffreysPrior
    from bayesmith.errors import GraphError

    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        at = {"fg_log_amp": jnp.log(3000.0), "fg_beta": jnp.array(-2.5)}
        with pytest.raises(GraphError, match="two priors on one quantity"):
            prior.log_density(_jeffreys_graph(flat=False), at)
        # Declared flat, it evaluates -- and to a finite number.
        value = float(prior.log_density(_jeffreys_graph(flat=True), at))
    assert np.isfinite(value)


def test_the_factor_site_adds_the_jeffreys_term_exactly_once():
    """The arithmetic behind the refusal, on the model that is legal.

    ``numpyro.factor`` is how the prior reached a NUTS run when this was
    written. **That parenthetical is now spent**: declaring it ON the graph
    landed on 2026-08-27 (the G13 wiring, then rheplicant's bridge switch), so
    `to_numpyro` emits the factor from `graph.joint_prior` and no caller writes
    one by hand. This test keeps the hand-written form on purpose -- it is the
    independent construction the emitted one is checked against, and a test
    that used the emitted factor to check the emitted factor would be checking
    nothing.

    The claim worth checking is not that the factor works but that the joint
    moves by the Jeffreys term and by nothing else -- so the term is measured
    independently and subtracted.
    """
    import numpyro
    from numpyro.infer.util import log_density as numpyro_log_density

    from bayesmith.bridge.numpyro_bridge import to_numpyro
    from bayesmith.diagnose.priors import JeffreysPrior

    with jax.enable_x64(True):
        graph = _jeffreys_graph(flat=True)
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        at = {"fg_log_amp": jnp.log(3000.0), "fg_beta": jnp.array(-2.5)}
        term = float(prior.log_density(graph, at))
        base_model = to_numpyro(graph)

        def with_factor():
            env = base_model()
            numpyro.factor(
                "joint_prior",
                prior.log_density(
                    graph,
                    {"fg_log_amp": env["fg_log_amp"], "fg_beta": env["fg_beta"]},
                ),
            )
            return env

        without, _ = numpyro_log_density(base_model, (), {}, at)
        withit, _ = numpyro_log_density(with_factor, (), {}, at)
    assert float(withit - without) == pytest.approx(term, rel=1e-12), (
        float(withit - without),
        term,
    )
