"""G10 -- D14's three additions to the factor execution surface.

(i) a per-sweep hook, (ii) a sweep-form estimate, (iii) a declared partition.
All three land ON ``sample_factors`` rather than beside it: v2 of the
migration plan believed there was no multi-block executor here and was wrong,
and starting a second one would break this package's own one-implementation
rule from the inside.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.dispatch.factor import (
    SweepReport,
    declared_partition,
    estimate_factors,
    factor_partition,
    sample_factors,
)
from bayesmith.errors import GraphError, StructureError
from bayesmith.graph.evaluate import log_joint
from tests.exact.models import (
    bilinear_pair,
    collinear_pair,
    cubic_tail,
    straight_line,
    two_linear_latents,
)

# ------------------------------------------------------- (iii) declared ------


class TestTheDeclaredEntry:
    """``declared_partition`` accepts a caller's block table and runs no
    probe. The semantics are "you declare, you are responsible", so the tests
    are mostly about what it still refuses -- a declaration this package
    cannot check is exactly the place a bookkeeping mistake becomes silent."""

    def test_a_declared_plan_sweeps_and_lands_where_the_derived_one_does(self):
        graph = bilinear_pair()
        derived = factor_partition(graph)
        declared = declared_partition(
            graph, [(("gain",), "gcr"), (("t_ant",), "gcr")]
        )
        assert [b.latents for b in declared.blocks] == [
            b.latents for b in derived.blocks
        ]
        a = sample_factors(graph, derived, jax.random.key(1), num_warmup=20,
                           num_samples=40)
        b = sample_factors(graph, declared, jax.random.key(1), num_warmup=20,
                           num_samples=40)
        for name in graph.latents:
            assert np.allclose(np.asarray(a[name]), np.asarray(b[name]), rtol=1e-5)

    def test_it_runs_no_affinity_probe(self):
        """The point of the entry, and the thing that makes it different from
        `factor_partition` rather than a rename of it.

        Counted rather than argued: `check_linearity` is entered zero times.
        """
        import bayesmith.dispatch.factor as module

        calls = {"n": 0}
        real = module.check_linearity

        def spy(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        module.check_linearity = spy
        try:
            declared_partition(bilinear_pair(), [(("gain", "t_ant"), "gcr")])
        finally:
            module.check_linearity = real
        assert calls["n"] == 0

    def test_it_accepts_a_block_the_probe_would_have_refused(self):
        """`cubic_tail` falls entirely to NUTS under `factor_partition`; a
        caller who says it is affine gets a gcr block, and owns that."""
        graph = cubic_tail()
        assert factor_partition(graph).nuts == ("w",)
        declared = declared_partition(graph, [(("w",), "gcr")])
        assert declared.exact and declared.exact[0].method == "gcr"

    def test_the_reason_says_who_decided(self):
        """A plan is a record of what was measured, and here nothing was.
        `str(plan)` and the block's own reason have to say so, or a declared
        plan read later is indistinguishable from a derived one.
        """
        declared = declared_partition(bilinear_pair(), [(("gain", "t_ant"), "gcr")])
        assert "declared" in declared.blocks[0].reason.lower()
        assert "not probed" in declared.blocks[0].reason.lower()

    def test_an_incomplete_cover_is_refused_rather_than_swept_into_nuts(self):
        """`factor_partition` puts what it could not solve into a NUTS block.
        Doing that here would invent a decision the caller did not make --
        the opposite of "you declare, you are responsible" -- so the
        uncovered latents are named instead.
        """
        with pytest.raises(StructureError, match="t_ant"):
            declared_partition(bilinear_pair(), [(("gain",), "gcr")])

    def test_a_latent_in_two_blocks_is_refused(self):
        with pytest.raises(StructureError, match="more than one"):
            declared_partition(
                bilinear_pair(), [(("gain",), "gcr"), (("gain", "t_ant"), "gcr")]
            )

    def test_a_name_that_is_not_a_latent_is_refused(self):
        with pytest.raises(StructureError, match="nope"):
            declared_partition(
                bilinear_pair(), [(("gain", "t_ant"), "gcr"), (("nope",), "nuts")]
            )

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(StructureError, match="method"):
            declared_partition(bilinear_pair(), [(("gain", "t_ant"), "lbfgs")])

    def test_two_nuts_blocks_are_refused(self):
        """`FactorPlan.nuts` returns the FIRST one, so a second is a block
        that would silently never be sampled."""
        with pytest.raises(StructureError, match="one 'nuts' block"):
            declared_partition(
                bilinear_pair(), [(("gain",), "nuts"), (("t_ant",), "nuts")]
            )

    def test_an_empty_block_is_refused(self):
        with pytest.raises(StructureError, match="empty"):
            declared_partition(
                bilinear_pair(), [((), "gcr"), (("gain", "t_ant"), "gcr")]
            )


# ------------------------------------------------------ (i) the sweep hook ---


class TestThePerSweepHook:
    def test_it_is_called_once_per_sweep_with_the_index_and_phase(self):
        graph = bilinear_pair()
        plan = factor_partition(graph)
        seen: list[SweepReport] = []
        sample_factors(
            graph, plan, jax.random.key(0),
            num_warmup=3, num_samples=5, on_sweep=seen.append,
        )
        assert [r.index for r in seen] == list(range(8))
        assert [r.warmup for r in seen] == [True] * 3 + [False] * 5

    def test_the_report_carries_the_joint_at_that_sweep(self):
        """The chi-square trajectory D14 asks for, in the general spelling:
        for a Gaussian model ``-2 log_joint`` IS the chi-square up to an
        additive constant, and the joint is the quantity that also exists for
        a model that is not."""
        graph = bilinear_pair()
        plan = factor_partition(graph)
        seen: list[SweepReport] = []
        sample_factors(
            graph, plan, jax.random.key(0),
            num_warmup=2, num_samples=3, on_sweep=seen.append,
        )
        for report in seen:
            assert float(report.log_joint) == pytest.approx(
                float(log_joint(graph, report.values)), rel=1e-6
            )

    def test_the_report_carries_a_residual_per_exact_block(self):
        """`gcr_sample` returns one and this executor used to drop it. A
        relative CG residual is not an accuracy -- multiply by the block's
        kappa for that -- but it is the only per-solve number there is."""
        graph = bilinear_pair()
        plan = factor_partition(graph)
        seen: list[SweepReport] = []
        sample_factors(
            graph, plan, jax.random.key(0),
            num_warmup=1, num_samples=2, on_sweep=seen.append,
        )
        for report in seen:
            assert set(report.residuals) == {b.latents for b in plan.exact}
            assert all(float(v) >= 0.0 for v in report.residuals.values())

    def test_the_values_are_the_ones_that_were_kept(self):
        """Anti-vacuity: a hook handed a stale or a copied-too-early
        environment would still be called the right number of times with
        plausible numbers."""
        graph = bilinear_pair()
        plan = factor_partition(graph)
        seen: list[SweepReport] = []
        drawn = sample_factors(
            graph, plan, jax.random.key(0),
            num_warmup=2, num_samples=3, on_sweep=seen.append,
        )
        kept = [r for r in seen if not r.warmup]
        for row, report in enumerate(kept):
            for name in graph.latents:
                assert np.allclose(
                    np.asarray(drawn[name][row]), np.asarray(report.values[name])
                )

    def test_it_is_refused_on_a_plan_with_a_nuts_remainder(self):
        """MEASURED, not assumed: with a NUTS block the sweep becomes
        HMCGibbs's `gibbs_fn`, which numpyro TRACES -- entered twice at the
        Python level for five sweeps of a two-block plan. A Python callback
        there fires once, at trace time, and reports a sweep that never
        happened. Refused by name rather than called wrongly.
        """
        graph = cubic_tail()
        plan = declared_partition(
            graph, [(("w",), "nuts")]
        )
        with pytest.raises(GraphError):
            # all-NUTS is refused for its own reason; use a mixed plan below.
            sample_factors(graph, plan, jax.random.key(0), num_samples=2)

    def test_the_refusal_names_the_tracing_reason(self):
        graph = two_linear_latents()
        plan = declared_partition(graph, [(("a",), "gcr"), (("b",), "nuts")])
        with pytest.raises(StructureError, match="TRACES"):
            sample_factors(
                graph, plan, jax.random.key(0),
                num_warmup=1, num_samples=1, on_sweep=lambda r: None,
            )

    def test_a_mixed_plan_still_runs_without_the_hook(self):
        """The other direction: the refusal must be about the HOOK and not
        about mixed plans, which have always worked."""
        graph = two_linear_latents()
        plan = declared_partition(graph, [(("a",), "gcr"), (("b",), "nuts")])
        drawn = sample_factors(
            graph, plan, jax.random.key(0), num_warmup=5, num_samples=5
        )
        assert drawn["a"].shape == (5,) and drawn["b"].shape == (5,)


# ------------------------------------------------- (ii) the sweep estimate ---


class TestTheSweepEstimate:
    """Block coordinate descent, which for a Gaussian conditional is exact
    coordinate ASCENT on the joint: the conditional mean is the conditional
    mode, so a sweep of Wiener solves cannot decrease the joint density."""

    def test_a_two_block_sweep_reaches_the_joint_wiener_answer(self):
        """The oracle is a DIFFERENT ALGORITHM on a DIFFERENT partition: one
        joint solve over both latents, against alternating one-latent solves.

        `two_linear_latents` has an orthogonal design, so coordinate ascent
        is exact after one pass -- measured at max|err| = **4.4e-16** at 5
        sweeps and unchanged at 300. The correlated case is the test below,
        and it is a different result.
        """
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator
        from bayesmith.exact.solve import wiener_solve

        with jax.enable_x64(True):
            graph = two_linear_latents()
            centres = prior_environment(graph)
            joint, _ = wiener_solve(
                linear_operator(graph, ("a", "b"), at={}),
                precision=precision_at(
                    graph, {n: centres[n] for n in graph.latents}
                ),
            )
            plan = declared_partition(graph, [(("a",), "gcr"), (("b",), "gcr")])
            swept = estimate_factors(graph, plan, sweeps=5)
        for name in ("a", "b"):
            assert float(swept.values[name]) == pytest.approx(
                float(joint[name]), rel=1e-8
            )

    def test_alternating_on_a_collinear_pair_does_not_reach_the_joint_answer(self):
        """The hazard `collinear_pair`'s own docstring records, measured.

        The data fixes ``a + b`` and the prior alone fixes ``a - b``, so
        coordinate ascent crawls along a nearly flat ridge. Measured against
        the joint solve's ``a = b = 1.024136``:

        =======  =================  ===============================
        sweeps   max abs error      joint moved from -12.051890 by
        =======  =================  ===============================
        5        1.0195             0.000931
        50       0.9745             0.010900
        300      **0.7583**         0.052515
        =======  =================  ===============================

        **Monotone the whole way**, in every row. Five hundredths of a nat
        over 300 sweeps while the parameters are still three quarters of a
        unit from the mode -- against a first sweep that moved the joint from
        -476.54 to -12.05. That is the shape of the failure, and it is why
        this is a counter-example rather than a slow success: every number a
        caller could look at says converged.

        The remedy is the partition, not the sweep count: `linear_operator`
        over BOTH latents solves it in one, which is what
        `factor_partition`'s pairwise grouping produces for this model.
        """
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator
        from bayesmith.exact.solve import wiener_solve

        with jax.enable_x64(True):
            # Built inside the block: `const` and `observe` capture their
            # arrays at trace time, so a graph traced outside carries float32
            # constants into a float64 solve and the numbers below move.
            graph = collinear_pair()
            centres = prior_environment(graph)
            joint, _ = wiener_solve(
                linear_operator(graph, ("a", "b"), at={}),
                precision=precision_at(
                    graph, {n: centres[n] for n in graph.latents}
                ),
            )
            split = declared_partition(graph, [(("a",), "gcr"), (("b",), "gcr")])
            crawling = estimate_factors(graph, split, sweeps=50)
            together = declared_partition(graph, [(("a", "b"), "gcr")])
            solved = estimate_factors(graph, together, sweeps=1)
        error = max(
            abs(float(crawling.values[n]) - float(joint[n])) for n in ("a", "b")
        )
        assert error > 0.5, error
        # ... and the trajectory looks perfectly healthy while it happens.
        history = np.asarray(crawling.history)
        assert np.all(np.diff(history) >= -1e-9)
        assert float(history[-1] - history[0]) < 0.02
        # One block, one sweep, done -- the partition is the remedy.
        for name in ("a", "b"):
            assert float(solved.values[name]) == pytest.approx(
                float(joint[name]), rel=1e-6
            )

    def test_the_joint_never_decreases_across_sweeps(self):
        """The property that makes this coordinate ASCENT rather than a loop
        that happens to end somewhere: a Gaussian conditional's mean IS its
        mode, so each block update maximises the joint over its own members
        and cannot lower it.

        This is the assertion a stale-environment bug fails -- conditioning
        on the values from the START of the sweep rather than the latest
        still converges on an orthogonal model and violates this on a
        correlated one.

        Improvement is measured against the STARTING point, not against
        ``history[0]``: the first sweep does nearly all the work (measured
        -476.5 -> -12.05 here), so comparing the first recorded entry with
        the last would be comparing two nearly identical numbers.
        """
        from bayesmith.dispatch.classify import prior_environment

        with jax.enable_x64(True):
            # Built inside the block: `const` and `observe` capture their
            # arrays at trace time, so a graph traced outside carries float32
            # constants into a float64 solve and the numbers below move.
            graph = collinear_pair()
            centres = prior_environment(graph)
            start = float(
                log_joint(graph, {n: centres[n] for n in graph.latents})
            )
            plan = declared_partition(graph, [(("a",), "gcr"), (("b",), "gcr")])
            swept = estimate_factors(graph, plan, sweeps=30)
        joints = np.asarray(swept.history)
        assert np.all(np.diff(joints) >= -1e-9), np.diff(joints).min()
        assert float(joints[0]) > start + 1.0

    def test_a_partly_exact_graph_is_estimated_rather_than_refused(self):
        """The gap D14 names: ``run_estimate`` refuses a graph that is not
        exact throughout. Here the exact blocks are solved and the remainder
        is stepped by `fit`, which is why G2 came first.
        """
        graph = two_linear_latents()
        plan = declared_partition(graph, [(("a",), "gcr"), (("b",), "nuts")])
        swept = estimate_factors(graph, plan, sweeps=40, steps=200,
                                 learning_rate=0.05)
        assert set(swept.values) == set(graph.latents)
        assert float(swept.history[-1]) > float(swept.history[0])

    def test_the_gradient_remainder_lands_where_a_joint_solve_does(self):
        """Anti-vacuity for the test above: "it ran" is not "it is right".

        Both latents of `two_linear_latents` are affine, so a joint Wiener
        solve knows the answer; declaring one of them a NUTS block and
        stepping it by gradient must reach the same point.
        """
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator
        from bayesmith.exact.solve import wiener_solve

        graph = two_linear_latents()
        centres = prior_environment(graph)
        joint, _ = wiener_solve(
            linear_operator(graph, ("a", "b"), at={}),
            precision=precision_at(graph, {n: centres[n] for n in graph.latents}),
        )
        plan = declared_partition(graph, [(("a",), "gcr"), (("b",), "nuts")])
        swept = estimate_factors(graph, plan, sweeps=60, steps=400,
                                 learning_rate=0.05)
        for name in ("a", "b"):
            assert float(swept.values[name]) == pytest.approx(
                float(joint[name]), rel=1e-3
            )

    def test_it_reports_the_residual_of_the_last_sweep(self):
        graph = bilinear_pair()
        plan = factor_partition(graph)
        swept = estimate_factors(graph, plan, sweeps=10)
        assert set(swept.residuals) == {b.latents for b in plan.exact}

    def test_a_single_exact_block_matches_its_own_wiener_solve(self):
        """The simplest oracle, and the one that would catch a sweep that
        drew instead of solving: one block, one sweep, the posterior MEAN."""
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator
        from bayesmith.exact.solve import wiener_solve

        graph = straight_line()
        centres = prior_environment(graph)
        exact, _ = wiener_solve(
            linear_operator(graph, ("w",), at={}),
            precision=precision_at(graph, {"w": centres["w"]}),
        )
        swept = estimate_factors(graph, factor_partition(graph), sweeps=3)
        assert float(swept.values["w"]) == pytest.approx(float(exact["w"]), rel=1e-5)

    def test_it_is_deterministic(self):
        """An estimate takes no key. A sweep that drew rather than solved
        would still land near the mode and would differ run to run."""
        graph = bilinear_pair()
        plan = factor_partition(graph)
        one = estimate_factors(graph, plan, sweeps=20)
        two = estimate_factors(graph, plan, sweeps=20)
        for name in graph.latents:
            assert jnp.array_equal(one.values[name], two.values[name])

    def test_a_non_positive_sweep_count_is_refused(self):
        with pytest.raises(StructureError, match="sweeps"):
            estimate_factors(bilinear_pair(), factor_partition(bilinear_pair()),
                             sweeps=0)


# ------------------------------------------- G12: sigma frozen where it is ---


def _moving_sigma_pair(*, n=10, kappa=0.25, floor=1e-3, seed=41):
    """``mu = (a + b) X``, ``sigma = kappa |mu| + floor``.

    Two latents, each affine in the prediction, and a sigma that moves with
    BOTH of them -- so each is outside the other's block and the rebuild
    branch is reached. `factor_partition` refuses this pair on the movement
    gate; declaring it is the whole subject of G12.
    """
    import numpyro.distributions as dist

    from bayesmith import const, det, observe, sample, trace

    x = jnp.linspace(1.0, 4.0, n)
    truth = 2.0 * x
    data = truth + (kappa * jnp.abs(truth) + floor) * jax.random.normal(
        jax.random.key(seed), (n,)
    )

    def model():
        xs = const("X", x)
        a = sample("a", lambda: dist.Normal(0.0, 3.0))
        b = sample("b", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda a_, b_, x_: (a_ + b_) * x_, a, b, xs,
                 linear_in=("a", "b"))
        observe("d", lambda m: dist.Normal(m, kappa * jnp.abs(m) + floor), mu,
                depends_on_prediction=True, obs=data)

    return trace(model)


class TestG12SigmaFrozenAtTheBlocksCurrentValue:
    """The semantics D8 stages in, exposed through the declared path rather
    than written a second time. Everything asserted here is about WHICH point
    the covariance is evaluated at, which is the only thing that separates
    this approximation from the hoisted one."""

    def test_the_movement_gate_is_what_the_declaration_bypasses(self):
        """The premise, measured rather than assumed: `factor_partition`
        sends this pair to NUTS, naming the movement."""
        graph = _moving_sigma_pair()
        plan = factor_partition(graph)
        assert set(plan.nuts) == {"a", "b"}
        nuts = next(b for b in plan.blocks if b.method == "nuts")
        assert "sigma moves with block" in nuts.reason

    def test_a_two_block_declaration_reaches_the_rebuild_branch(self):
        """`_sigma_needs_rebuild` asks about an OUTSIDE latent, so this is
        the arrangement in which the rebuild happens at all."""
        from bayesmith.dispatch.classify import _sigma_needs_rebuild

        graph = _moving_sigma_pair()
        assert _sigma_needs_rebuild(graph, ("a",)) is True
        assert _sigma_needs_rebuild(graph, ("b",)) is True

    def test_one_block_over_the_whole_model_does_not(self):
        """The other half of the same fact, and the reason the note in
        `declared_partition` names the condition instead of just saying
        "frozen at the current value".

        With nothing outside it, the block is HOISTED -- sigma frozen at the
        prior centre, which is a different approximation with a different
        error. Measured on the same model.
        """
        from bayesmith.dispatch.classify import _sigma_needs_rebuild

        graph = _moving_sigma_pair()
        assert _sigma_needs_rebuild(graph, ("a", "b")) is False

    def test_the_covariance_is_evaluated_at_the_blocks_own_latest_value(self):
        """The claim itself, and the only test here that could catch it being
        false.

        Spied at `precision_at`: for a rebuilt block the values it is handed
        must contain that block's OWN latest draw -- the one from the previous
        sweep -- and not its prior centre. A version that passed only the
        outside latents would give a sigma frozen somewhere no latent ever
        was, and every downstream number would still be finite and plausible.
        """
        import bayesmith.dispatch.factor as module

        graph = _moving_sigma_pair()
        plan = declared_partition(graph, [(("a",), "gcr"), (("b",), "gcr")])
        seen: list[dict] = []
        real = module.precision_at

        def spy(source, values):
            seen.append({k: float(v) for k, v in values.items()})
            return real(source, values)

        module.precision_at = spy
        try:
            drawn = sample_factors(graph, plan, jax.random.key(3),
                                   num_warmup=0, num_samples=4)
        finally:
            module.precision_at = real

        # Two blocks per sweep, four sweeps; neither block is hoisted.
        assert len(seen) == 8
        assert all(set(row) == {"a", "b"} for row in seen)

        # The assertion that matters, and the one the first version of this
        # test could not make. Mutation W8 substituted each block's PRIOR
        # CENTRE for its own current value in the rebuilt precision, and this
        # test passed: the OTHER latent was still moving, so "both names are
        # present" and "the points differ" were both still true. A guard whose
        # fixture cannot tell right from plausibly-wrong is the recurring
        # defect here, and it survived a mutation to prove it.
        #
        # Calls alternate block a, block b, per sweep. In sweep k the call for
        # block a must see a = that block's draw from sweep k-1, and the call
        # for block b must see a = the draw JUST taken in sweep k and b = its
        # own from sweep k-1.
        a_draws = [float(v) for v in drawn["a"]]
        b_draws = [float(v) for v in drawn["b"]]
        for k in range(1, 4):
            assert seen[2 * k]["a"] == pytest.approx(a_draws[k - 1], rel=1e-12)
            assert seen[2 * k + 1]["a"] == pytest.approx(a_draws[k], rel=1e-12)
            assert seen[2 * k + 1]["b"] == pytest.approx(b_draws[k - 1], rel=1e-12)

    def test_gcr_plus_mh_is_refused_at_construction_with_its_own_reason(self):
        """Not the generic unknown-method message: a caller reaching for
        'gcr+mh' has a real model in mind and needs the argument, not a list
        of tokens."""
        graph = _moving_sigma_pair()
        with pytest.raises(StructureError, match="gcr\\+mh") as caught:
            declared_partition(graph, [(("a",), "gcr+mh"), (("b",), "gcr")])
        message = str(caught.value)
        assert "single-block" in message.lower() or "SINGLE-block" in message
        assert "signature" in message
        assert "assemble" in message

    def test_the_declared_sweep_runs_and_recovers_the_sum(self):
        """The approximation is offered to be USED, so it has to work: the
        data fixes ``a + b`` at 2.0 and the sweep must find it.

        This is not a correctness proof for the kernel and the docstring says
        so -- it is the check that the path is wired, which a note alone
        would not give.
        """
        graph = _moving_sigma_pair()
        plan = declared_partition(graph, [(("a",), "gcr"), (("b",), "gcr")])
        drawn = sample_factors(graph, plan, jax.random.key(5), num_warmup=200,
                               num_samples=400)
        total = np.asarray(drawn["a"]) + np.asarray(drawn["b"])
        assert float(total.mean()) == pytest.approx(2.0, abs=0.3)


class TestTheThreeRoutesEstimatedTogether:
    """The gap D14 (ii) names, at full width: ``InferencePlan.estimate``
    refuses a graph that is not exact throughout, and the ordinary model is
    not. Here one plan carries a ``gcr`` block, a ``log-gcr`` block and a
    ``nuts`` remainder, and all three are estimated in the same sweep."""

    def test_a_gcr_a_log_gcr_and_a_nuts_block_are_estimated_in_one_sweep(self):
        """Measured against the truths the fixture was simulated at, which no
        part of this code can see: ``log_gain`` 0.470 and ``centre`` 0.100.

        The two are reached by different machinery -- a Wiener solve on the
        log-transformed graph and a gradient step on the joint -- so a single
        assertion here would not tell them apart; both are named.
        """
        from tests.dispatch.test_factor import TestMixedRouting

        graph, log_gain_true, centre_true, _ = TestMixedRouting._mixed()
        plan = factor_partition(graph)
        methods = {b.latents: b.method for b in plan.blocks}
        assert methods == {
            ("coef",): "gcr", ("log_gain",): "log-gcr", ("centre",): "nuts"
        }
        swept = estimate_factors(graph, plan, sweeps=8, steps=150,
                                 learning_rate=0.05)
        assert float(swept.values["log_gain"]) == pytest.approx(
            log_gain_true, abs=0.01
        )
        assert float(swept.values["centre"]) == pytest.approx(
            centre_true, abs=0.01
        )

    def test_the_same_plan_is_refused_by_the_whole_graph_estimator(self):
        """The anti-vacuity clause, and the reason (ii) is a gap rather than
        a convenience: the existing estimator does not accept this plan at
        all, so the test above is not duplicating a route that already
        worked."""
        from bayesmith.dispatch.execute import _refuse_unless_whole_graph_exact
        from tests.dispatch.test_factor import TestMixedRouting

        graph, *_ = TestMixedRouting._mixed()
        compiled = __import__(
            "bayesmith.dispatch.plan", fromlist=["compile"]
        ).compile(graph)
        with pytest.raises(NotImplementedError) as caught:
            _refuse_unless_whole_graph_exact(compiled)
        # And the refusal now names this route. It used to say the package
        # "does not ship" an optimiser (P5) -- true when written and made
        # false by G2 and this batch, which is the shape a claim spelled in
        # two places always takes.
        assert "estimate_factors" in str(caught.value)
