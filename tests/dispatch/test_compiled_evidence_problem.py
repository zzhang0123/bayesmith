"""The prior/likelihood split, over every fixture this package ships.

Nested sampling needs ``log L(theta)`` separate from ``log pi(theta)``, and
nothing in bayesmith produces that separation: ``graph/evaluate.py``'s
``log_joint`` sums every ``Probabilistic`` node into ONE running total and then
adds ``joint_prior`` and every ``evidence_terms`` entry in the same loop.
Building the split is a compiler pass, and this module grades it.

**What is asserted, and what deliberately is not.** The obvious assertion --
``log_prior + log_likelihood == log_joint`` bitwise -- is NOT a property of the
partition. Splitting one accumulator into two reorders a float sum, and float
addition is not associative, so the identity holds or fails according to the
draw. Measured over 200 prior draws per fixture: ``straight_line`` 200/200,
``diamond_ancestor`` 200/200, ``three_latent_chain`` 200/200, but
``two_observations`` 186/200 (worst gap 1.8e-12) and
``observation_reused_downstream`` 182/200 (worst 2.8e-14). A planning census
that sampled one draw reported 46/46 and would have shipped an assertion whose
red depends on which draw the implementer picked -- and again differently on
Linux, where the ``log_prob`` kernels contract differently.

So two assertions replace it, and neither pins a summation order:

1. **Per term, bitwise.** Every ``Probabilistic`` node and every graph-level
   term appears EXACTLY ONCE, on the side the partition names, and each side
   equals an independently accumulated reference over exactly those terms. This
   is what catches a constant that moved sides, which is the whole failure mode
   the evidence layer exists to prevent, and it is exact by construction rather
   than by luck.
2. **The recomposition, to a derived band**, over a DECLARED seed set rather
   than one draw. The band is ``n * eps * max(|log_joint|, 1)``: ``n`` terms
   accumulated, each carrying at most one rounding of the running total, and a
   sum of logs carries absolute error proportional to its largest term rather
   than relative error -- the same form D107 took for the dense-agreement band,
   and for the same reason.

Design basis: R5 plan sections 0.2 and 0.17.
"""

from __future__ import annotations

import dataclasses
import inspect
import math

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith.dispatch.evidence import (
    CompiledEvidenceProblem,
    compile_evidence_problem,
)
from bayesmith.graph.evaluate import apply_probabilistic, evaluate, log_joint
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic
from bayesmith.graph.reduction import as_graph
from tests.exact import models

#: The seed set is part of the fixture (spec section 9.3), not a number the
#: test picks per run. Five draws, declared here so that a failure names which
#: one moved rather than reporting "a draw".
SEEDS = (0, 1, 2, 3, 4)

#: The five fixtures in ``models.py`` that take constructor arguments, with the
#: arguments this census uses. Without them the census silently covers 49 of
#: the 54 rows, and a census whose denominator is unstated is one the next
#: reader re-derives differently.
PARAMETERISED = {
    "cancelling_sum": {"cancel": 1e2},
    "many_observations": {"count": 3},
    "roundoff_stress": {"big": 1e6, "sigma": 1e-3},
    "sigma_functional_block": {"weights": (1.0, 0.0, -1.0)},
    "wide_plate": {"size": 4},
}


def _graphs():
    """Every graph this package ships, with its denominator stated.

    ``flagged_line`` returns three objects, two of which are graphs, so the
    count is over GRAPHS and not over fixture functions. Two independent
    censuses of this differed -- 15 and 13 for one class -- because one skipped
    the tuple return entirely and neither said which it was counting.
    """
    for name, fn in sorted(vars(models).items()):
        if not inspect.isfunction(fn) or name.startswith("_"):
            continue
        if fn.__module__ != models.__name__:
            continue
        required = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.default is inspect.Parameter.empty
        ]
        if required and name not in PARAMETERISED:
            raise AssertionError(
                f"{name} takes required arguments and is not in PARAMETERISED; "
                "a census that skips it covers less than it claims"
            )
        built = fn(**PARAMETERISED.get(name, {}))
        candidates = built if isinstance(built, tuple) else (built,)
        for index, candidate in enumerate(candidates):
            try:
                graph = as_graph(candidate)
                graph.nodes  # noqa: B018 - reading it is the check
            except (AttributeError, TypeError):
                # `flagged_line` returns three objects and the third is an
                # array. Narrow on purpose: a BayesmithError here would be a
                # fixture that cannot build, which this census must NOT skip.
                continue
            label = name if len(candidates) == 1 else f"{name}[{index}]"
            yield label, graph


GRAPHS = list(_graphs())


#: The one shipped graph whose prior cannot be drawn from at all. Declared here
#: rather than discovered by an `except: continue`, because a census that skips
#: a row silently covers less than it claims -- and this is the row a dedicated
#: test asserts RAISES, so it is an expectation and not a gap.
UNSAMPLABLE_PRIOR = frozenset({"improper_outside_prior"})


def _needs_a_draw(label):
    if label.split("[")[0] in UNSAMPLABLE_PRIOR:
        pytest.skip(
            f"{label} declares a prior with no sample; that it RAISES is "
            "asserted by TestAnImproperPriorCannotBeSampled. THIS IS NOT A PASS "
            "of the property below."
        )


def _prior_draw(graph, seed):
    """One draw of every latent, from the graph's own prior."""
    from numpyro import handlers

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    trace = handlers.trace(
        handlers.seed(to_numpyro(graph), jax.random.key(seed))
    ).get_trace(observed={})
    latents = set(graph.latents)
    return {name: node["value"] for name, node in trace.items() if name in latents}


def _reference_terms(graph, values):
    """Per-term contributions, keyed by the name the partition must file them
    under. Computed here from the graph rather than read off the compiled
    problem, so that agreement is a check and not a tautology."""
    env = evaluate(graph, values)
    prior: dict[str, jax.Array] = {}
    likelihood: dict[str, jax.Array] = {}
    for node in graph.nodes:
        if not isinstance(node, Probabilistic):
            continue
        term = apply_probabilistic(graph, node, env).log_prob(env[node.name])
        if node.observed_mask is not None:
            term = jnp.where(node.observed_mask, term, 0.0)
        total = jnp.sum(term)
        if node.observed is None:
            prior[node.name] = total
        else:
            likelihood[node.name] = total
    if graph.joint_prior is not None:
        prior["joint_prior"] = jnp.asarray(
            graph.joint_prior.log_density(
                graph, {name: env[name] for name in graph.latents}
            )
        )
    for index, term in enumerate(graph.evidence_terms):
        likelihood[f"evidence_terms[{index}]"] = jnp.asarray(
            term.log_density(graph, {name: env[name] for name in term.over})
        )
    return prior, likelihood


class TestEveryTermIsFiledExactlyOnce:
    """The assertion that catches a constant which moved sides."""

    @pytest.mark.parametrize("label,graph", GRAPHS, ids=[g[0] for g in GRAPHS])
    def test_the_two_sides_partition_the_graphs_terms(self, label, graph):
        _needs_a_draw(label)
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            prior, likelihood = _reference_terms(graph, _prior_draw(graph, SEEDS[0]))
        assert set(problem.prior_terms) == set(prior), (
            f"{label}: the prior side files {sorted(problem.prior_terms)}; the "
            f"graph's latent terms are {sorted(prior)}"
        )
        assert set(problem.likelihood_terms) == set(likelihood), (
            f"{label}: the likelihood side files "
            f"{sorted(problem.likelihood_terms)}; the graph's observed terms "
            f"are {sorted(likelihood)}"
        )
        both = set(problem.prior_terms) & set(problem.likelihood_terms)
        assert not both, f"{label}: {sorted(both)} filed on both sides"

    @pytest.mark.parametrize("label,graph", GRAPHS, ids=[g[0] for g in GRAPHS])
    def test_each_side_equals_its_own_terms_bitwise(self, label, graph):
        _needs_a_draw(label)
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            values = _prior_draw(graph, SEEDS[0])
            prior, likelihood = _reference_terms(graph, values)
            got_prior = float(problem.log_prior(values))
            got_likelihood = float(problem.log_likelihood(values))
            # Inside the x64 context on purpose: summing float64 arrays outside
            # it downcasts them to float32, which reads as a bitwise failure of
            # the partition and is a failure of the test's own scoping.
            want_prior = float(sum(prior[name] for name in problem.prior_terms))
            want_likelihood = float(
                sum(likelihood[name] for name in problem.likelihood_terms)
            )
        if math.isfinite(want_prior):
            assert got_prior == want_prior, f"{label}: prior side"
        if math.isfinite(want_likelihood):
            assert got_likelihood == want_likelihood, f"{label}: likelihood side"


class TestTheRecompositionSitsInsideADerivedBand:
    """Not bitwise -- see this module's docstring for the measurement."""

    @pytest.mark.parametrize("label,graph", GRAPHS, ids=[g[0] for g in GRAPHS])
    def test_over_a_declared_seed_set(self, label, graph):
        _needs_a_draw(label)
        eps = float(jnp.finfo(jnp.float64).eps)
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            for seed in SEEDS:
                values = _prior_draw(graph, seed)
                total = float(log_joint(graph, values))
                recomposed = float(problem.log_prior(values)) + float(
                    problem.log_likelihood(values)
                )
                if not math.isfinite(total):
                    continue
                terms = len(problem.prior_terms) + len(problem.likelihood_terms)
                band = terms * eps * max(abs(total), 1.0)
                assert abs(recomposed - total) <= band, (
                    f"{label} at seed {seed}: recomposed {recomposed!r} against "
                    f"log_joint {total!r}, gap {recomposed - total:.3e}, band "
                    f"{band:.3e}"
                )


class TestTheSplitLocalisesWhatLogJointCannot:
    """Both `nan` fixtures put the `nan` on the likelihood side.

    ``log_joint`` reports one `nan` and cannot say which half produced it. The
    split can, and that is a reason to prefer it rather than a side effect.
    """

    @pytest.mark.parametrize(
        "fixture", ["unusable_observed_scale", "two_unusable_observed_scales"]
    )
    def test_the_nan_is_on_the_likelihood_side(self, fixture):
        graph = as_graph(getattr(models, fixture)())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            values = _prior_draw(graph, SEEDS[0])
            assert math.isnan(float(log_joint(graph, values)))
            assert math.isfinite(float(problem.log_prior(values)))
            assert math.isnan(float(problem.log_likelihood(values)))


class TestAnImproperPriorCannotBeSampled:
    """Stricter than integrating to one, and it is the sampler's requirement.

    A nested sampler draws from the prior. ``improper_outside_prior`` declares
    one whose mass diverges, and the honest outcome is a raise rather than a
    number -- R4 already refuses it as ``evidence_prior_proper``, and this
    asserts the ordering is load-bearing rather than tidy.
    """

    def test_prior_sampling_raises_rather_than_returning_a_number(self):
        graph = as_graph(models.improper_outside_prior())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            with pytest.raises(NotImplementedError):
                problem.prior_sample(jax.random.key(0))


class TestTheProblemRefusesAContradictoryPartition:
    def test_a_name_cannot_be_both_eliminated_and_residual(self):
        """``InferencePlanRecord`` already refuses this; the compiled problem
        must too, or the two can disagree about which parameters a backend is
        being handed."""
        with pytest.raises(ValueError, match="both"):
            CompiledEvidenceProblem(
                log_prior=lambda values: jnp.zeros(()),
                log_likelihood=lambda values: jnp.zeros(()),
                prior_sample=lambda key: {},
                exact_elimination=("w",),
                residual_parameters=("w",),
                shapes=(("w", ()),),
                prior_terms=(),
                likelihood_terms=(),
            )

    def test_no_field_is_or_publicly_contains_a_graph(self):
        """Design line 192: a compiled problem may not make a backend
        re-interpret the Graph.

        Asserted over the PUBLIC surface -- each field, and into tuples, lists
        and dicts -- because a later field could hold a graph inside a
        container and a top-level type check would not see it.
        """
        graph = as_graph(models.straight_line())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)

        def public_reach(value, depth=0):
            if depth > 4:
                return
            yield value
            if isinstance(value, (tuple, list, set, frozenset)):
                for item in value:
                    yield from public_reach(item, depth + 1)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from public_reach(item, depth + 1)

        for field in dataclasses.fields(problem):
            for value in public_reach(getattr(problem, field.name)):
                assert not (
                    hasattr(value, "nodes") and hasattr(value, "latents")
                ), f"{field.name} publicly exposes a graph"

    def test_the_densities_DO_close_over_the_graph_and_that_is_the_contract(self):
        """The honest statement, asserted positively rather than denied.

        An earlier version of this test walked only the top-level field values
        for a ``.nodes`` attribute. Those values are functions, so it passed
        trivially while the graph was reachable through EVERY one of the three
        closures -- a guard that read a spelling, which is red line 7's subject
        and which this batch has now produced in its own test file.

        What design line 192 forbids is a BACKEND re-interpreting the graph.
        bayesmith compiling the densities itself, by closing over the graph, is
        that contract being kept rather than broken: the adapter receives three
        callables and cannot reach a graph without walking CPython closure
        internals. So the capture is asserted here, deliberately, and Task 6
        asserts the half that actually binds -- that the adapter's module never
        imports ``Graph`` and its signature accepts only a compiled problem.
        """
        graph = as_graph(models.straight_line())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)

        def captured(fn, depth=0, seen=None):
            seen = set() if seen is None else seen
            if id(fn) in seen or depth > 6:
                return None
            seen.add(id(fn))
            if hasattr(fn, "nodes") and hasattr(fn, "latents"):
                return fn
            for cell in getattr(fn, "__closure__", None) or ():
                try:
                    inner = cell.cell_contents
                except ValueError:  # pragma: no cover - an empty cell
                    continue
                hit = captured(inner, depth + 1, seen)
                if hit is not None:
                    return hit
            return None

        for name in ("log_prior", "log_likelihood", "prior_sample"):
            assert captured(getattr(problem, name)) is graph, (
                f"{name} no longer closes over the graph it was compiled from; "
                "if that is deliberate, this test and the docstring both move"
            )


# --------------------------------------------------- the constant dimension

#: **No shipped fixture carries a ``joint_prior`` or an ``evidence_terms``
#: entry.** Measured: zero of 54. So every mutation of where the graph-level
#: terms are filed -- move ``joint_prior`` to the likelihood side, move
#: ``evidence_terms`` to the prior side, drop either -- survives the entire
#: fixture set with nothing red, because the code path is never entered.
#:
#: That is the failure shape red line 1 names: a test suite grades the
#: dimensions its fixtures happen to VARY, and these two are constant (absent)
#: across the whole family. It was found by asking which dimensions take the
#: same value in every fixture, which is answerable by grep and needs no
#: understanding of the change -- the form the rule takes for exactly this
#: reason. The graphs below supply the missing variance.


class _GaussianTerm:
    """A graph-level factor with a real density, over declared latents."""

    def __init__(self, *over: str, scale: float = 1.4):
        self.over = over
        self._scale = scale

    def log_density(self, graph, values):
        del graph
        total = jnp.zeros(())
        for name in self.over:
            total = total + jnp.sum(
                dist.Normal(0.0, self._scale).log_prob(values[name])
            )
        return total


def _bare_latent(name, loc=0.0, scale=1.0):
    return Probabilistic(
        name=name,
        parents=(),
        plate=(),
        dist_fn=lambda: dist.Normal(loc, scale),
        observed=None,
    )


def _graph_with_a_joint_prior():
    """One latent whose ONLY prior is the graph-level one.

    The node carries ``ImproperUniform`` -- log-density zero -- so the graph
    level supplies the whole prior. Giving the node a proper density AS WELL
    makes the prior side a PRODUCT of two densities, and the normalisation
    oracle below caught exactly that in this fixture's first draft: it
    integrated to 0.23188, which is ``1 / sqrt(2 pi (1 + 1.4**2))``, the
    convolution of the two widths. The oracle earned its place before it
    graded any production code.
    """
    node = Probabilistic(
        name="x",
        parents=(),
        plate=(),
        dist_fn=lambda: dist.ImproperUniform(
            dist.constraints.real, batch_shape=(), event_shape=()
        ),
        observed=None,
    )
    return Graph(nodes=(node,), plates=(), joint_prior=_GaussianTerm("x"))


def _graph_with_an_evidence_term():
    """One latent, and a graph-level LIKELIHOOD factor over it."""
    return Graph(
        nodes=(_bare_latent("x"),), plates=(), evidence_terms=(_GaussianTerm("x"),)
    )


class TestTheGraphLevelTermsAreFiledOnTheRightSide:
    def test_a_joint_prior_goes_to_the_prior_side(self):
        problem = compile_evidence_problem(_graph_with_a_joint_prior())
        assert "joint_prior" in problem.prior_terms
        assert "joint_prior" not in problem.likelihood_terms

    def test_an_evidence_term_goes_to_the_likelihood_side(self):
        problem = compile_evidence_problem(_graph_with_an_evidence_term())
        assert "evidence_terms[0]" in problem.likelihood_terms
        assert "evidence_terms[0]" not in problem.prior_terms

    @pytest.mark.parametrize(
        "build", [_graph_with_a_joint_prior, _graph_with_an_evidence_term]
    )
    def test_the_term_actually_contributes_to_its_side(self, build):
        """Membership in the tuple is bookkeeping; this is the number moving.

        A partition that FILES a term correctly and then does not ADD it is a
        dropped constant, which is the failure the evidence layer exists to
        prevent, and the membership assertion above cannot see it.
        """
        graph = build()
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            values = {"x": jnp.asarray(0.7)}
            total = float(log_joint(graph, values))
            both = float(problem.log_prior(values)) + float(
                problem.log_likelihood(values)
            )
        assert math.isclose(both, total, rel_tol=1e-12), (
            f"recomposed {both!r} against log_joint {total!r}; a graph-level "
            "term was filed but not summed"
        )


class TestThePriorSideIntegratesToOne:
    """The oracle that makes the partition CORRECT rather than consistent.

    Every other assertion here compares the compiled split against a reference
    written in this file, and the two would have to be wrong the same way to
    agree -- which a single wrong idea about where a term belongs would make
    them. This one does not compare implementations at all: a prior integrates
    to one, and a likelihood factor filed on the prior side takes it away from
    one. Quadrature over a scalar latent, which is what every graph here has.
    """

    @pytest.mark.parametrize(
        "label,build",
        [
            ("node prior only", lambda: Graph(nodes=(_bare_latent("x"),), plates=())),
            ("graph-level joint_prior", _graph_with_a_joint_prior),
            ("with an evidence term", _graph_with_an_evidence_term),
        ],
    )
    def test_exp_log_prior_integrates_to_one(self, label, build):
        graph = build()
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            grid = jnp.linspace(-40.0, 40.0, 40001)
            density = jnp.exp(
                jnp.asarray([problem.log_prior({"x": point}) for point in grid])
            )
            mass = float(jnp.trapezoid(density, grid))
        assert math.isclose(mass, 1.0, rel_tol=1e-6), (
            f"{label}: the prior side integrates to {mass!r}, not 1. A "
            "likelihood factor filed on the prior side is exactly what this "
            "looks like"
        )
