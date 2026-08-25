"""Whether a graph's exact block can be run one epoch at a time.

The evidence layer was complete, dense-oracled and cross-checked bitwise
against rheplicant, and nothing in ``dispatch/`` called it. This module is
the caller: it answers the one question ``compress_campaign`` cannot answer
for itself -- WHICH plate is the epoch plate -- and it answers it by trying,
not by guessing.

The load-bearing test here is
``test_the_route_s_verdict_is_the_one_compress_campaign_actually_gives``. A
route that reports "available" and then raises would be worse than no route,
and every other test in this file could pass while it did.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest

from bayesmith import const, det, observe, plate, sample, trace
from bayesmith.dispatch.streaming import streaming_route
from bayesmith.errors import StructureError
from bayesmith.evidence import compress_campaign

N_EPOCH, TAU, SIGMA, GAIN, PRIOR_MEAN = 4, 1.3, 0.55, 2.0, 0.4


def _campaign(n_epoch=N_EPOCH, seed=0):
    """One survivor ``g``, one per-epoch nuisance ``n``, one datum per epoch."""
    data = np.random.default_rng(seed).normal(size=n_epoch)

    def model():
        epoch = plate("epoch", n_epoch)
        g = sample("g", lambda: ndist.Normal(0.0, 3.0))
        n = sample("n", lambda: ndist.Normal(PRIOR_MEAN, TAU), plate=epoch)
        mu = det(
            "mu", lambda a, b: GAIN * a + b, g, n, plate=epoch, linear_in=("g", "n")
        )
        observe(
            "d", lambda m: ndist.Normal(m, SIGMA), mu, plate=epoch,
            obs=jnp.asarray(data),
        )

    return trace(model)


def _unplated():
    def model():
        x = sample("x", lambda: ndist.Normal(0.0, 2.0))
        observe("d", lambda v: ndist.Normal(v, 0.5), x, obs=jnp.asarray([1.0, 2.0]))

    return trace(model)


def _every_latent_inside_the_plate(n_epoch=N_EPOCH):
    """No survivor: each epoch is its own analysis, so nothing accumulates."""
    data = np.random.default_rng(1).normal(size=n_epoch)

    def model():
        epoch = plate("epoch", n_epoch)
        n = sample("n", lambda: ndist.Normal(PRIOR_MEAN, TAU), plate=epoch)
        observe(
            "d", lambda m: ndist.Normal(m, SIGMA), n, plate=epoch,
            obs=jnp.asarray(data),
        )

    return trace(model)


class TestTheRouteIsDerivedByTrying:
    def test_a_campaign_graph_offers_its_epoch_plate(self):
        with jax.enable_x64(True):
            route = streaming_route(_campaign())
        assert route.available
        assert route.plate == "epoch"
        assert route.survivors == ("g",)
        assert route.per_epoch == ("n",)
        assert route.refused == ()

    def test_an_unplated_graph_offers_nothing_and_refuses_nothing(self):
        """No plates means no candidates -- not a refusal, an absence."""
        with jax.enable_x64(True):
            route = streaming_route(_unplated())
        assert not route.available
        assert route.plate is None
        assert route.survivors == ()
        assert route.per_epoch == ()
        assert route.refused == ()

    def test_a_plate_with_no_survivor_is_refused_in_the_layer_s_own_words(self):
        """The refusal is `factorize`'s, forwarded -- not restated here.

        A second spelling of a refusal is the copy that goes stale, and this
        one is load-bearing: it is the sentence that tells a user what to
        change about their model.
        """
        with jax.enable_x64(True):
            route = streaming_route(_every_latent_inside_the_plate())
        assert not route.available
        assert dict(route.refused).keys() == {"epoch"}
        assert "nothing survives an epoch" in dict(route.refused)["epoch"]


class TestTheRouteDoesNotLie:
    @pytest.mark.parametrize(
        "build", [_campaign, _unplated, _every_latent_inside_the_plate]
    )
    def test_the_route_s_verdict_is_the_one_compress_campaign_actually_gives(
        self, build
    ):
        """``available`` must mean it runs, and not-available must mean it raises.

        This is the anti-lying clause. ``factorize`` is only ONE of the two
        structural checks ``epoch_terms`` makes -- the other is that exactly
        one observed node is plated on the epoch axis -- so a route built
        from ``factorize`` alone would report a plate that then raises. Both
        directions are asserted, because a route that always said "no" would
        satisfy the first half alone.
        """
        with jax.enable_x64(True):
            graph = build()
            route = streaming_route(graph)
            if route.available:
                term = compress_campaign(graph, route.plate)
                assert term.names == route.survivors
            else:
                for name in {p.name for p in graph.plates}:
                    with pytest.raises(StructureError):
                        compress_campaign(graph, name)


def _two_plates():
    """A campaign plate plus an unrelated one -- both are legitimate models."""
    d1 = np.random.default_rng(0).normal(size=N_EPOCH)

    def model():
        epoch = plate("epoch", N_EPOCH)
        night = plate("night", 3)
        g = sample("g", lambda: ndist.Normal(0.0, 3.0))
        n = sample("n", lambda: ndist.Normal(PRIOR_MEAN, TAU), plate=epoch)
        mu = det(
            "mu", lambda a, b: GAIN * a + b, g, n, plate=epoch, linear_in=("g", "n")
        )
        observe(
            "d", lambda m: ndist.Normal(m, SIGMA), mu, plate=epoch,
            obs=jnp.asarray(d1),
        )
        b = sample("b", lambda: ndist.Normal(0.0, 1.0), plate=night)
        observe("e", lambda m: ndist.Normal(m, 0.5), b, plate=night, obs=jnp.zeros(3))

    return trace(model)


class TestASecondPlateIsRefusedInTheLayersOwnWords:
    """Measured, and the reason it is tested here rather than assumed.

    The first draft of ``streaming_route`` caught ``StructureError`` alone
    and promised in its own docstring that it never raises. Both plates of
    this graph made ``factorize`` raise a bare jax ``ValueError`` instead --
    ``epoch_leakage`` probes with one ``jacfwd`` over survivors AND per-epoch
    latents together, and a survivor plated on the other axis has the wrong
    rank for the vmap. So the promise was false, and ``compile()`` would have
    begun raising on a graph it plans perfectly well today.

    ``factorize`` now refuses this early and BY NAME, so what reaches the
    route is an ordinary refusal it forwards like any other. The route's
    ``ValueError`` clause is kept as containment rather than as the handler
    for this case -- see its own comment.
    """

    def test_the_route_declines_without_raising(self):
        with jax.enable_x64(True):
            route = streaming_route(_two_plates())
        assert not route.available
        assert dict(route.refused).keys() == {"epoch", "night"}

    def test_the_refusal_names_the_latent_and_the_other_plate(self):
        """A verdict the author can act on, not a vmap message.

        This is the half that regressed silently before: the model WAS
        refused, and the sentence said "rank should be at least 1, but is
        only 0", which names no latent, no plate, and nothing to change.
        """
        with jax.enable_x64(True):
            why = dict(streaming_route(_two_plates()).refused)["epoch"]
        assert "'b'" in why
        assert "night" in why
        assert "vmap" not in why

    def test_it_reads_as_a_refusal_not_as_a_failure_to_analyse(self):
        """The two are different facts and only one is a judgement."""
        with jax.enable_x64(True):
            refused_here = dict(streaming_route(_two_plates()).refused)["epoch"]
            refused_there = dict(
                streaming_route(_every_latent_inside_the_plate()).refused
            )["epoch"]
        assert "could not be analysed" not in refused_here
        assert "could not be analysed" not in refused_there

    def test_compile_still_plans_this_graph(self):
        """The regression the narrow catch exists to prevent."""
        from bayesmith import compile as compile_graph

        with jax.enable_x64(True):
            plan = compile_graph(_two_plates())
        assert plan.blocks


def _two_clean_plates():
    """Two plates that BOTH pass every structural check -- the ambiguity case.

    Finding this shape took three tries and each failure was informative:

    * latents in both plates -> ``epoch_leakage``'s ``jacfwd`` isolates them
      together and the other plate's survivor has the wrong rank for the vmap;
    * latents in NEITHER plate, dets parented only on the unplated survivor
      -> ``GraphError``, "none of its parents are [in the plate]";
    * so each plate needs a PLATED parent that is not a latent. A plated
      ``const`` is one, and then both plates factorize, both have exactly one
      observation, and both evaluate.
    """

    def model():
        e1 = plate("epoch", 4)
        e2 = plate("night", 3)
        c1 = const("c1", jnp.arange(4.0), plate=e1)
        c2 = const("c2", jnp.arange(3.0), plate=e2)
        g = sample("g", lambda: ndist.Normal(0.0, 3.0))
        m1 = det("m1", lambda a, c: a * c, g, c1, plate=e1, linear_in=("g",))
        m2 = det("m2", lambda a, c: a * c, g, c2, plate=e2, linear_in=("g",))
        observe("d1", lambda v: ndist.Normal(v, SIGMA), m1, plate=e1, obs=jnp.zeros(4))
        observe("d2", lambda v: ndist.Normal(v, SIGMA), m2, plate=e2, obs=jnp.zeros(3))

    return trace(model)


class TestTwoQualifyingPlatesAreDeclinedNotGuessed:
    def test_the_route_declines_and_names_both_candidates(self):
        with jax.enable_x64(True):
            route = streaming_route(_two_clean_plates())
        assert not route.available
        assert route.plate is None
        assert dict(route.refused).keys() == {"epoch", "night"}
        assert "which one indexes epochs" in dict(route.refused)["epoch"]

    def test_declining_is_a_choice_because_both_folds_really_do_run(self):
        """The anti-vacuity clause.

        If neither plate could be folded, declining would be free and this
        branch would be indistinguishable from a refusal. Both DO fold, and
        to different answers -- which is exactly why picking one silently
        would be wrong rather than merely arbitrary.
        """
        with jax.enable_x64(True):
            graph = _two_clean_plates()
            by_epoch = compress_campaign(graph, "epoch")
            by_night = compress_campaign(graph, "night")
            assert by_epoch.names == ("g",)
            assert by_night.names == ("g",)
            here = float(by_epoch.information().reshape(()))
            there = float(by_night.information().reshape(()))
        assert here != there, (here, there)


def _plated_det_with_no_plated_parent():
    """Passes factorize AND the observation check, and still cannot be folded.

    The fixture that caught the route lying. ``factorize`` probes leakage
    only ``for name in per_epoch``, and this campaign has no per-epoch
    latent, so nothing evaluated the graph and nothing noticed that ``m`` is
    plated with no plated parent. The route reported ``available`` and
    ``compress_campaign`` then raised ``GraphError``.
    """

    def model():
        epoch = plate("epoch", N_EPOCH)
        g = sample("g", lambda: ndist.Normal(0.0, 3.0))
        m = det("m", lambda a: GAIN * a, g, plate=epoch, linear_in=("g",))
        observe(
            "d", lambda v: ndist.Normal(v, SIGMA), m, plate=epoch, obs=jnp.zeros(N_EPOCH)
        )

    return trace(model)


ALL_FIXTURES = [
    _campaign,
    _unplated,
    _every_latent_inside_the_plate,
    _two_plates,
    _two_clean_plates,
    _plated_det_with_no_plated_parent,
]


@pytest.mark.parametrize("build", ALL_FIXTURES, ids=lambda f: f.__name__.lstrip("_"))
def test_no_fixture_makes_the_route_disagree_with_the_fold(build):
    """The total oracle: the route offers a plate exactly when ONE folds.

    Every fixture in this module, swept. The narrower
    ``TestTheRouteDoesNotLie`` above lists three by hand and was green while
    the route was lying about a fourth -- a guard that COULD have caught the
    bug and was not pointed at it. This one takes the list, so a fixture
    added below is covered by construction.

    Stated as a COUNT rather than as "available iff the fold runs", because
    those are not the same claim and the weaker one is false here:
    ``_two_clean_plates`` folds on both of its plates and the route still
    offers neither. Declining to guess between two working folds is a
    verdict, not a failure -- so what must hold is that the route offers a
    plate exactly when the number of foldable plates is one, and that the
    plate it offers is that one.

    The exception types are the three the layer really raises, measured:
    ``StructureError`` for its own verdicts, ``GraphError`` from evaluating a
    plated node, ``ValueError`` from a vmap the second plate breaks. Listing
    them is the point -- a bare ``Exception`` would accept a ``TypeError``
    from a future refactor and call it a refusal.
    """
    from bayesmith.errors import GraphError

    with jax.enable_x64(True):
        graph = build()
        route = streaming_route(graph)
        foldable = []
        for name in sorted(p.name for p in graph.plates):
            try:
                term = compress_campaign(graph, name)
            except (StructureError, GraphError, ValueError):
                continue
            foldable.append((name, term.names))

    assert route.available == (len(foldable) == 1), (route.plate, foldable)
    if route.available:
        assert route.plate == foldable[0][0]
        assert route.survivors == foldable[0][1]


class TestThePlanCarriesTheRoute:
    """``compile()`` MEASURES the route rather than offering a live query.

    ``Block``'s own docstring says why: "a plan is meant to be a record of
    what was measured, not a live query". The cost is affordable and flat in
    the campaign -- measured 0.278 s at E=4 and 0.342 s at E=512 against
    ``compile()``'s own 1.0-1.3 s -- and a graph with no plates pays nothing,
    because the loop over plates does not run.
    """

    def test_a_campaign_plan_names_the_plate_it_could_stream(self):
        from bayesmith import compile as compile_graph

        with jax.enable_x64(True):
            plan = compile_graph(_campaign())
        assert plan.streaming is not None
        assert plan.streaming.available
        assert plan.streaming.plate == "epoch"
        assert "campaign fold available on plate 'epoch'" in str(plan)

    def test_a_graph_that_is_not_a_campaign_prints_exactly_what_it_used_to(self):
        """No line at all -- not an empty one, and not "unavailable".

        The regression guard for the twenty existing assertions on
        ``str(plan)``: they were written against a form with no streaming
        row, and a row saying nothing would still have moved every line
        index and every trailing-newline count under them.
        """
        from bayesmith import compile as compile_graph

        with jax.enable_x64(True):
            plan = compile_graph(_unplated())
        assert plan.streaming is not None
        assert not plan.streaming.available
        assert plan.streaming.line() == ""
        assert "streaming" not in str(plan)
        assert str(plan).splitlines()[-1].startswith("execution: ")

    def test_the_plan_agrees_with_the_route_it_was_built_from(self):
        """No second derivation -- the plan carries the route, not a copy."""
        from bayesmith import compile as compile_graph

        with jax.enable_x64(True):
            graph = _campaign()
            plan = compile_graph(graph)
            direct = streaming_route(graph)
        assert plan.streaming.plate == direct.plate
        assert plan.streaming.survivors == direct.survivors
        assert plan.streaming.per_epoch == direct.per_epoch
