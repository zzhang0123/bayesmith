"""Factor partition -- the multilinear case the single exact block drops.

The headline fixture is bilinear, ``d = g (B t + tone)``, with BOTH latents
declared ``linear_in``. :func:`~bayesmith.dispatch.classify.partition` sends
it whole to NUTS -- its joint check correctly refuses the pair -- and this
file first PINS that, so "factor_partition does better" stays a measured
claim about a real gap rather than a comparison with a strawman.

Truth-recovery assertions use generous bounds (4-5 posterior widths): the
point of an end-to-end test here is that the sweep targets the right
posterior, not a re-derivation of the closed forms already pinned in
``tests/exact``.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import det, observe, sample, trace
from bayesmith.dispatch import partition
from bayesmith.dispatch.factor import (
    FactorPlan,
    factor_partition,
    first_fit,
    sample_factors,
)
from bayesmith.errors import GraphError

N = 12
B = jnp.linspace(1.0, 2.0, N)
TONE = jnp.zeros(N).at[3].set(400.0)
G_TRUE, T_TRUE = 1.5, 300.0
SIGMA = 0.5


def _bilinear():
    def model(data):
        g = sample("g", lambda: dist.Normal(1.5, 0.5))
        t = sample("t", lambda: dist.Normal(0.0, 1000.0))
        mu = det(
            "mu", lambda g_, t_: g_ * (B * t_ + TONE), g, t, linear_in=("g", "t")
        )
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

    data = G_TRUE * (B * T_TRUE + TONE) + SIGMA * jax.random.normal(
        jax.random.key(0), (N,)
    )
    return trace(model, data)


class TestFirstFit:
    def test_groups_are_maximal_under_the_predicate(self):
        incompatible = {frozenset({"a", "c"}), frozenset({"b", "c"})}
        groups = first_fit(
            ["a", "b", "c", "d"],
            lambda x, y: frozenset({x, y}) not in incompatible,
        )
        assert groups == [["a", "b", "d"], ["c"]]

    def test_no_pair_is_asked_twice(self):
        asked = []

        def compatible(x, y):
            asked.append(frozenset({x, y}))
            return True

        first_fit(list("abcde"), compatible)
        assert len(asked) == len(set(asked))

    def test_an_always_false_predicate_gives_singletons(self):
        assert first_fit(["x", "y", "z"], lambda *_: False) == [["x"], ["y"], ["z"]]


class TestTheBilinearHeadline:
    def test_the_single_block_partition_drops_the_whole_model_to_nuts(self):
        """The gap, pinned. If this ever starts producing an exact block,
        the comparison below is against a strawman and must be rethought."""
        classified = partition(_bilinear())
        assert classified.exact == ()
        assert classified.method == "nuts"
        assert "JOINTLY" in classified.reason

    def test_factor_partition_finds_one_gcr_block_per_factor(self):
        plan = factor_partition(_bilinear())
        assert [(b.latents, b.method) for b in plan.blocks] == [
            (("g",), "gcr"),
            (("t",), "gcr"),
        ]

    def test_the_pure_exact_sweep_recovers_the_truth(self):
        graph = _bilinear()
        plan = factor_partition(graph)
        draws = sample_factors(
            graph, plan, jax.random.key(1), num_warmup=200, num_samples=400
        )
        for name, truth in (("g", G_TRUE), ("t", T_TRUE)):
            mean = float(jnp.mean(draws[name]))
            spread = float(jnp.std(draws[name]))
            assert spread > 0.0
            assert abs(mean - truth) < 5.0 * spread, (name, mean, spread)

    def test_the_blocks_carry_their_own_measured_tolerances(self):
        plan = factor_partition(_bilinear())
        for block in plan.exact:
            assert block.tol is not None and block.tol > 0.0
            assert block.kappa is not None
            assert block.epsilon is not None


class TestMixedRouting:
    @staticmethod
    def _mixed():
        """Additive calibration + multiplicative science + one true nonlinear."""
        n = 24
        x = jnp.linspace(-1.0, 1.0, n)
        basis = -jnp.linalg.qr(jnp.stack([x**k for k in range(3)], axis=1))[0]
        coef = jnp.array([2800.0, -160.0, 35.0])
        log_g, f, centre_true = float(jnp.log(1.6)), 4.05e-3, 0.10

        def model(d_cal, d_sky):
            c = sample("coef", lambda: dist.Normal(coef, 300.0).to_event(1))
            cal = det("cal", lambda cc: basis @ cc, c, linear_in=("coef",))
            observe("d_cal", lambda m: dist.Normal(m, 2.0), cal, obs=d_cal)
            lam = sample("log_gain", lambda: dist.Normal(0.3, 0.5))
            centre = sample("centre", lambda: dist.Normal(0.0, 0.4))
            line = det(
                "line",
                lambda ce: 1.0 + 0.5 * jnp.exp(-0.5 * ((x - ce) / 0.3) ** 2),
                centre,
            )
            mu = det("mu", lambda l, li: jnp.exp(l) * 2500.0 * li, lam, line)
            observe("d_sky", lambda m: dist.Normal(m, f * m), mu, obs=d_sky)

        line_true = 1.0 + 0.5 * jnp.exp(-0.5 * ((x - centre_true) / 0.3) ** 2)
        d_cal = basis @ coef + 2.0 * jax.random.normal(jax.random.key(0), (n,))
        d_sky = jnp.exp(log_g) * 2500.0 * line_true * (
            1.0 + f * jax.random.normal(jax.random.key(5), (n,))
        )
        return trace(model, d_cal, d_sky), log_g, centre_true, coef

    def test_the_three_routes_are_assigned_in_one_plan(self):
        graph, *_ = self._mixed()
        plan = factor_partition(graph)
        methods = {b.latents: b.method for b in plan.blocks}
        assert methods[("coef",)] == "gcr"
        assert methods[("log_gain",)] == "log-gcr"
        assert methods[("centre",)] == "nuts"
        assert plan.log_space is not None
        assert plan.log_space.kind == {"d_sky": "multiplicative"}
        assert set(plan.log_space.skipped) == {"d_cal"}

    def test_the_nuts_reason_names_each_latent_and_why(self):
        graph, *_ = self._mixed()
        plan = factor_partition(graph)
        nuts = next(b for b in plan.blocks if b.method == "nuts")
        assert "'centre'" in nuts.reason
        assert "not affine" in nuts.reason

    @pytest.mark.slow
    def test_the_mixed_sweep_recovers_all_three(self):
        graph, log_g, centre_true, coef = self._mixed()
        plan = factor_partition(graph)
        draws = sample_factors(
            graph, plan, jax.random.key(1), num_warmup=300, num_samples=500
        )
        assert (
            abs(float(jnp.mean(draws["log_gain"])) - log_g)
            < 5.0 * float(jnp.std(draws["log_gain"]))
        )
        assert (
            abs(float(jnp.mean(draws["centre"])) - centre_true)
            < 5.0 * float(jnp.std(draws["centre"]))
        )
        recovered = jnp.mean(draws["coef"], axis=0)
        assert jnp.all(jnp.abs(recovered - coef) < 5.0 * jnp.std(draws["coef"], axis=0))


class TestHierarchy:
    """Example 2 of docs/factor-partition-examples.md, pinned.

    A hyperparameter ``y`` setting a latent field ``w1``'s statistics is
    ejected from every exact block by the ancestry rule, in BOTH the linear
    and the nonlinear parameterisation -- an exact block solves only against
    observed nodes, so admitting ``y`` would drop the ``p(w1 | y)`` factor
    silently. The doc page's partition printouts are these assertions.
    """

    N_H = 32

    @classmethod
    def _graph(cls, kind: str):
        n = cls.N_H
        xi = jnp.linspace(-1.0, 1.0, n)
        a = jnp.stack([xi, xi**2 - jnp.mean(xi**2)], axis=1)
        b = jnp.stack([jnp.ones(n), xi, xi**2], axis=1)
        c = jnp.stack([xi, xi**3], axis=1)
        d_mat = jnp.eye(3)
        y_t = jnp.array([5.0, 1.0, 0.8])
        f = 4.05e-3

        def model(data):
            y = sample("y", lambda: dist.Normal(y_t, 1.0).to_event(1))
            if kind == "linear":
                w1 = sample(
                    "w1", lambda y_: dist.Normal(d_mat @ y_, 0.3).to_event(1), y
                )
            else:
                w1 = sample(
                    "w1",
                    lambda y_: dist.Normal(
                        y_t, 0.3 * jnp.exp(0.2 * (y_ - y_t))
                    ).to_event(1),
                    y,
                )
            x = sample("x", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
            z = sample("z", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
            s = det(
                "s", lambda w_, z_: b @ w_ + jnp.exp(c @ z_), w1, z, linear_in=("w1",)
            )
            mu = det("mu", lambda x_, s_: jnp.exp(a @ x_) * s_, x, s, linear_in=("s",))
            observe("d", lambda m: dist.Normal(m, f * m), mu, obs=data)

        truth = jnp.exp(a @ jnp.array([0.3, -0.2])) * (
            b @ y_t + jnp.exp(c @ jnp.array([0.5, -0.3]))
        )
        data = truth * (1.0 + f * jax.random.normal(jax.random.key(0), (n,)))
        return trace(model, data)

    @pytest.mark.parametrize("kind", ["linear", "nonlinear"])
    def test_the_hyperparameter_is_ejected_in_both_parameterisations(self, kind):
        plan = factor_partition(self._graph(kind))
        methods = {b.latents: b.method for b in plan.blocks}
        assert methods[("x",)] == "log-gcr"
        nuts = next(b for b in plan.blocks if b.method == "nuts")
        assert "y" in nuts.latents and "w1" in nuts.latents and "z" in nuts.latents
        assert "ancestor of another latent's distribution" in nuts.reason

    def test_the_two_parameterisations_produce_the_same_partition(self):
        """The ejection is structural, so the linear case may not sneak into
        an exact block on the strength of a conjugacy the dispatcher does not
        yet cut -- a slower right answer over a fast wrong one."""
        shape = lambda plan: [(b.latents, b.method) for b in plan.blocks]
        assert shape(factor_partition(self._graph("linear"))) == shape(
            factor_partition(self._graph("nonlinear"))
        )

    @pytest.mark.slow
    def test_the_hierarchical_sweep_recovers_the_hyperparameter(self):
        graph = self._graph("linear")
        plan = factor_partition(graph)
        draws = sample_factors(
            graph, plan, jax.random.key(1), num_warmup=600, num_samples=900
        )
        y_t = jnp.array([5.0, 1.0, 0.8])
        pull = jnp.abs(jnp.mean(draws["y"], axis=0) - y_t) / jnp.std(
            draws["y"], axis=0
        )
        assert float(jnp.max(pull)) < 5.0


class TestGenerality:
    """The partition is a function of STRUCTURE, refuted three ways.

    These three tests jointly pin that no routing decision is ad hoc: any
    branch keyed on a latent's name or declaration order dies on the renaming
    test; any baked-in "this slot is the log-linear one" dies on the role
    swap; any assumption that the noise is always multiplicative dies on the
    structure edit. What survives all three is a rule that reads the graph.
    """

    N_G = 24

    @classmethod
    def _worked_example(cls, names=("x", "y", "z"), order=("x", "y", "z"),
                        swap_exponential=False, additive=False):
        """Example 1 of the docs, with its structure and labels adjustable.

        ``names`` relabels the three roles (gain-exponent, linear
        coefficients, summed exponent); ``order`` declares them in any order;
        ``swap_exponential`` moves the outer exponential from the first role
        to the third; ``additive`` replaces the multiplicative noise with a
        constant sigma.
        """
        n = cls.N_G
        xi = jnp.linspace(-1.0, 1.0, n)
        a = jnp.stack([xi, xi**2 - jnp.mean(xi**2)], axis=1)
        b = jnp.stack([jnp.ones(n), xi, xi**2], axis=1)
        c = jnp.stack([xi, xi**3], axis=1)
        f = 4.05e-3
        gain_name, linear_name, summed_name = names

        def declare(name):
            if name == gain_name:
                return sample(name, lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
            if name == linear_name:
                # Width 0.5, NOT 2.0, and the number is load-bearing: at 2.0
                # the prior puts real mass on a NEGATIVE summed sky, where the
                # log route genuinely fails -- so the verdict then depends on
                # which prior draws the probe happens to take, and the first
                # run of the renaming test below caught exactly that (renaming
                # reseeds the draws through the sorted-names fold). A prior
                # that keeps the sky positive is what makes log-linearity a
                # property of the MODEL rather than of the probe's luck.
                return sample(
                    name,
                    lambda: dist.Normal(jnp.array([5.0, 1.0, 0.8]), 0.5).to_event(1),
                )
            return sample(name, lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))

        def model(data):
            made = {name: declare(name) for name in order}
            gain, linear, summed = (
                made[gain_name], made[linear_name], made[summed_name],
            )
            if swap_exponential:
                s_node = det(
                    "s", lambda l, g: b @ l + jnp.exp(a @ g), linear, gain,
                    linear_in=(linear_name,),
                )
                mu = det(
                    "mu", lambda z_, s_: jnp.exp(c @ z_) * s_, summed, s_node,
                    linear_in=("s",),
                )
            else:
                s_node = det(
                    "s", lambda l, z_: b @ l + jnp.exp(c @ z_), linear, summed,
                    linear_in=(linear_name,),
                )
                mu = det(
                    "mu", lambda g, s_: jnp.exp(a @ g) * s_, gain, s_node,
                    linear_in=("s",),
                )
            if additive:
                observe("d", lambda m: dist.Normal(m, 0.05), mu, obs=data)
            else:
                observe("d", lambda m: dist.Normal(m, f * m), mu, obs=data)

        truth = jnp.exp(a @ jnp.array([0.3, -0.2])) * (
            b @ jnp.array([5.0, 1.0, 0.8]) + jnp.exp(c @ jnp.array([0.5, -0.3]))
        )
        if additive:
            data = truth + 0.05 * jax.random.normal(jax.random.key(0), (n,))
        else:
            data = truth * (1.0 + f * jax.random.normal(jax.random.key(0), (n,)))
        return trace(model, data)

    @staticmethod
    def _shape(plan):
        return {block.latents: block.method for block in plan.blocks}

    def test_the_partition_is_invariant_under_renaming_and_reordering(self):
        """Kills any branch keyed on a latent's NAME or declaration order."""
        original = self._shape(factor_partition(self._worked_example()))
        renamed = self._shape(
            factor_partition(
                self._worked_example(
                    names=("alpha", "beta", "gamma"),
                    order=("gamma", "alpha", "beta"),
                )
            )
        )
        mapping = {"x": "alpha", "y": "beta", "z": "gamma"}
        translated = {
            tuple(sorted(mapping[m] for m in members)): method
            for members, method in original.items()
        }
        assert {
            tuple(sorted(members)): method for members, method in renamed.items()
        } == translated

    def test_the_log_route_follows_the_exponential_not_the_slot(self):
        """Kills any baked-in "the first latent is the log-linear one"."""
        plain = self._shape(factor_partition(self._worked_example()))
        swapped = self._shape(
            factor_partition(self._worked_example(swap_exponential=True))
        )
        assert plain[("x",)] == "log-gcr"
        assert ("z",) not in plain or plain.get(("z",)) != "log-gcr"
        # After the swap, z multiplies through the outer exponential and x
        # moved inside the sum -- the verdicts swap with the structure.
        assert swapped[("z",)] == "log-gcr"
        assert ("x",) not in swapped or swapped.get(("x",)) != "log-gcr"

    def test_the_routing_responds_to_the_noise_structure(self):
        """Kills any assumption that the noise is always multiplicative.

        With a CONSTANT sigma: the log route closes for everyone (log space
        would not simplify additive noise), so the gain-exponent latent loses
        its log-gcr block -- and the linear latent GAINS a plain gcr block,
        because the sigma-movement gate that refused it under multiplicative
        noise now measures zero.
        """
        additive = self._shape(factor_partition(self._worked_example(additive=True)))
        assert additive[("y",)] == "gcr"
        nuts = next(k for k in additive if additive[k] == "nuts")
        assert "x" in nuts and "z" in nuts


class TestRefusalsAndEdges:
    def test_a_radiometer_sigma_that_moves_with_the_block_is_named_not_swept(self):
        """The one method this module refuses: gcr+mh. The latent lands in
        NUTS with the movement number and the remedy in its reason."""

        def model(data):
            # sigma carries a positive FLOOR so the affinity check's own
            # covariance stays positive definite with the block at zero --
            # without it the solo check refuses on the covariance and the
            # latent never reaches the gate this test is about. The floor
            # keeps sigma genuinely moving with the block, which is the
            # property under test.
            t = sample("t", lambda: dist.Normal(300.0, 0.2))
            mu = det("mu", lambda t_: B * t_, t, linear_in=("t",))
            observe("d", lambda m: dist.Normal(m, 0.5 + 0.05 * m), mu, obs=data)

        data = B * T_TRUE + (0.5 + 0.05 * B * T_TRUE) * jax.random.normal(
            jax.random.key(0), (N,)
        )
        plan = factor_partition(trace(model, data))
        nuts = [b for b in plan.blocks if b.method == "nuts"]
        assert nuts and "gcr+mh" in nuts[0].reason
        assert plan.exact == ()

    def test_an_all_nuts_plan_refuses_to_pretend_to_sweep(self):
        def model(data):
            z = sample("z", lambda: dist.Normal(0.0, 1.0))
            mu = det("mu", lambda z_: jnp.tanh(z_) * B, z)
            observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data)

        graph = trace(model, jnp.tanh(0.4) * B)
        plan = factor_partition(graph)
        assert plan.exact == ()
        with pytest.raises(GraphError) as caught:
            sample_factors(graph, plan, jax.random.key(0))
        assert "bayesmith.nuts" in str(caught.value)

    def test_the_plan_prints_one_line_per_block(self):
        plan = factor_partition(_bilinear())
        text = str(plan)
        assert text.splitlines()[0].startswith("block 0")
        assert "gcr" in text

    def test_factor_plan_is_a_module_with_stable_properties(self):
        plan = factor_partition(_bilinear())
        assert isinstance(plan, FactorPlan)
        assert plan.nuts == ()
        assert {b.latents for b in plan.exact} == {("g",), ("t",)}
