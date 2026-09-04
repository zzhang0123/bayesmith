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

#: Fixtures whose joint is non-finite at EVERY declared seed, so the band test
#: grades nothing for them. Declared, because the alternative is a row that
#: passes having asserted nothing and looks identical to one that asserted
#: something -- measured at 15 of 265 rows silently ungraded, three fixtures at
#: 0 of 5. The band test asserts this set is exactly right in both directions.
NEVER_FINITE = frozenset(
    {
        "overflowing_outside_latent",
        "unusable_observed_scale",
        "two_unusable_observed_scales",
    }
)


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
        """What this test CANNOT do, stated because a draft of it overclaimed.

        It grades one thing: that splitting a running float sum into two and
        adding them back lands within the rounding that reordering can produce.
        The band is therefore sized by the largest term, because that is what
        bounds the absolute error of a sum of logs.

        **It cannot detect a small perturbation of the smaller side, and no
        reformulation of the band fixes that.** On ``high_snr_curvature`` the
        likelihood side is about 8.8e12 nats and the prior side about 1.9, so
        any band that admits the reordering also admits a 0.2% error in the
        prior. Measured twice: with the band as ``n eps max(|total|, 1)`` this
        test passed at all five seeds under ``log_prior * (1 + 1e-7)``, and it
        passed again after the band was rewritten as the sum of the two sides'
        own magnitudes -- 3.9e-3 nats either way, against a 1.9e-7 shift. The
        second form is kept because it is the tighter statement, not because it
        repaired anything.

        **The mutant is caught by the per-side bitwise test above**, which is
        exact and does not depend on the sides' relative size. A reviewer called
        that test the tautological one; it is the one that kills M7, and this is
        why both exist.

        Separately: ``if not math.isfinite(total): continue`` silently ungraded
        15 of 265 rows, three fixtures at 0 of 5 seeds -- a guaranteed pass that
        looks exactly like a graded one. The count is now asserted in both
        directions.
        """
        _needs_a_draw(label)
        eps = float(jnp.finfo(jnp.float64).eps)
        graded = 0
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            terms = len(problem.prior_terms) + len(problem.likelihood_terms)
            for seed in SEEDS:
                values = _prior_draw(graph, seed)
                total = float(log_joint(graph, values))
                prior = float(problem.log_prior(values))
                likelihood = float(problem.log_likelihood(values))
                if not (
                    math.isfinite(total)
                    and math.isfinite(prior)
                    and math.isfinite(likelihood)
                ):
                    continue
                graded += 1
                # Each side carries its own rounding; neither buys slack for
                # the other.
                band = terms * eps * (max(abs(prior), 1.0) + max(abs(likelihood), 1.0))
                assert abs(prior + likelihood - total) <= band, (
                    f"{label} at seed {seed}: recomposed {prior + likelihood!r} "
                    f"against log_joint {total!r}, gap "
                    f"{prior + likelihood - total:.3e}, band {band:.3e}"
                )
        if label.split("[")[0] in NEVER_FINITE:
            assert graded == 0, (
                f"{label} is declared as never producing a finite joint and "
                f"graded {graded} rows; move it out of NEVER_FINITE"
            )
        else:
            # ALL of them, not "at least one". `graded > 0` lets a fixture drift
            # to 4 of 5 non-finite -- 80% of its grading gone -- with nothing
            # red and no output difference. Measured over the 265 rows: every
            # fixture outside NEVER_FINITE grades 5 of 5 today, so the strict
            # form costs nothing and the loose one bought nothing.
            assert graded == len(SEEDS), (
                f"{label} graded {graded} of {len(SEEDS)} seeds. A row that "
                "grades fewer than all of them passed without asserting what it "
                "claims to; declare the fixture in NEVER_FINITE or give it "
                "seeds that produce a finite joint."
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


# ------------------------------- the dimensions the first draft never varied

#: Coverage of the two-route shape comparison, accumulated across the
#: parametrized run and asserted once at the end. A comparison whose `except`
#: branch quietly grows is a comparison that stops comparing without the output
#: changing -- the same failure as the 15 silently ungraded band rows.
_SHAPE_COMPARISON_HITS: set = set()
_SHAPE_COMPARISON_SKIPS: set = set()


def test_the_shape_comparison_actually_compared_something():
    """Runs after the parametrized sweep above (alphabetical file order puts it
    last within its module for `-p no:randomly`; under random order it asserts
    only what has accumulated, which is why it asserts a RATIO and a named
    skip rather than an exact hit count)."""
    if not (_SHAPE_COMPARISON_HITS or _SHAPE_COMPARISON_SKIPS):
        pytest.skip("the parametrized shape sweep has not run in this session")
    skips = {name for _label, name in _SHAPE_COMPARISON_SKIPS}
    assert skips <= {"z"}, (
        f"node_shape was skipped for {sorted(_SHAPE_COMPARISON_SKIPS)}; the "
        "only latent it cannot size is improper_outside_prior's `z`, which "
        "declares an ImproperUniform and so has no `loc`"
    )
    assert len(_SHAPE_COMPARISON_HITS) > 20 * len(_SHAPE_COMPARISON_SKIPS), (
        f"{len(_SHAPE_COMPARISON_HITS)} compared against "
        f"{len(_SHAPE_COMPARISON_SKIPS)} skipped; the except branch has grown "
        "and this test is no longer comparing much"
    )


class TestTheDimensionsTheFirstDraftHeldConstant:
    """Three mutation survivors, and each was a dimension no test varied.

    Every test above calls ``compile_evidence_problem(graph)`` with the default
    ``exact_elimination=()``, and none looks at ``shapes`` or constructs a
    problem with overlapping term tuples. So deleting the term-overlap check,
    reporting every parameter as a scalar, and ignoring ``exact_elimination``
    all survived the whole file. Found by enumerating what the fixtures hold
    CONSTANT rather than by reading the diff -- red line 1's form.

    One of the three was not a test gap at all: ``shapes`` really was always
    ``()``, because the code read ``graph.shape`` behind a
    ``hasattr(graph, "shape")`` guard and ``Graph`` has no such attribute. The
    mutant and the code agreed, which is why no mutation of that line could
    have shown it and why "SURVIVED" had to be diagnosed rather than counted.
    """

    def test_a_plated_latent_reports_its_real_shape(self):
        """``plated_latent`` declares a plate of 6. Reporting `()` for it is
        a parameter layout that tells a backend the wrong dimension."""
        graph = as_graph(models.plated_latent())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
        assert dict(problem.shapes)["z"] == (6,), (
            f"shapes reports {problem.shapes}; the plate is 6 wide"
        )

    @pytest.mark.parametrize("label,graph", GRAPHS, ids=[g[0] for g in GRAPHS])
    def test_the_layout_agrees_with_node_shape_where_node_shape_applies(
        self, label, graph
    ):
        """Two routes to a latent's shape, pinned together.

        ``compile_evidence_problem`` uses ``batch_shape + event_shape``
        broadcast with the plate, because it must answer for an
        ``ImproperUniform`` latent, which has no ``loc`` for ``node_shape`` to
        read. ``node_shape`` is what ``to_numpyro`` is pinned against. Where
        both are defined they must agree, or the compiled layout describes a
        different space from the one a sampler would open.
        """
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import node_shape

        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            env = prior_environment(graph)
            for name, shape in problem.shapes:
                try:
                    reference = tuple(node_shape(graph, graph.node(name), env))
                except (AttributeError, TypeError):
                    # node_shape reads `loc`; this latent has none. Recorded
                    # rather than skipped silently -- an `except: continue` that
                    # grows is how a comparison stops comparing without saying
                    # so. Measured over all 54 graphs: 78 parameters compared,
                    # exactly 1 skipped, and it is improper_outside_prior's `z`.
                    _SHAPE_COMPARISON_SKIPS.add((label, name))
                    continue
                _SHAPE_COMPARISON_HITS.add((label, name))
                assert shape == reference, (
                    f"{label}: layout says {name}{shape}, node_shape says "
                    f"{reference}"
                )

    def test_a_latent_with_a_non_scalar_batch_shape_reports_it(self):
        """``batch_shape + event_shape`` is dead on the whole fixture family.

        Measured: **0 of 54** shipped latents have
        ``batch_shape + event_shape != ()``. Every non-``()`` layout in the
        suite comes from the PLATE broadcast alone, so the expression the last
        repair introduced can be replaced by the literal ``()`` and the file
        still passes -- three mutants lived in that gap, in the code the repair
        was written to fix.

        A `Normal(loc=zeros(3))` latent compiles here and must be laid out as
        `(3,)`. The fixture family does not contain one; this supplies it.
        """
        graph = Graph(
            nodes=(
                Probabilistic(
                    name="v",
                    parents=(),
                    plate=(),
                    dist_fn=lambda: dist.Normal(jnp.zeros(3), 1.0),
                    observed=None,
                ),
            ),
            plates=(),
        )
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            drawn = problem.prior_sample(jax.random.key(0))
        assert dict(problem.shapes)["v"] == (3,), (
            f"a Normal(loc=zeros(3)) latent is laid out as "
            f"{dict(problem.shapes)['v']}"
        )
        assert jnp.shape(drawn["v"]) == (3,)

    def test_a_latent_with_an_event_shape_reports_it(self):
        """The other half: ``event_shape``, which a multivariate latent has and
        no shipped fixture does."""
        graph = Graph(
            nodes=(
                Probabilistic(
                    name="v",
                    parents=(),
                    plate=(),
                    dist_fn=lambda: dist.MultivariateNormal(
                        jnp.zeros(3), jnp.eye(3)
                    ),
                    observed=None,
                ),
            ),
            plates=(),
        )
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            drawn = problem.prior_sample(jax.random.key(0))
        assert dict(problem.shapes)["v"] == (3,)
        assert jnp.shape(drawn["v"]) == (3,)

    def test_a_scalar_latent_still_reports_the_empty_shape(self):
        graph = as_graph(models.straight_line())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
        assert dict(problem.shapes)["w"] == ()

    def test_the_residual_order_follows_the_graphs_latent_order(self):
        """M19. ``shapes`` is a LAYOUT: a backend that flattens parameters in
        this order and unflattens in another gets a different model, silently.
        The order is the graph's own, minus what was eliminated."""
        graph = as_graph(models.two_linear_latents())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
        expected = tuple(n for n in graph.latents)
        assert problem.residual_parameters == expected, (
            f"residual order {problem.residual_parameters}, graph order {expected}"
        )
        assert [name for name, _ in problem.shapes] == list(expected), (
            "the layout must be in the same order as the parameters it sizes"
        )

    def test_exact_elimination_removes_names_from_the_residual(self):
        """The parameter Tasks 6 and 7 depend on, exercised here because no
        other test passes it a non-empty value."""
        graph = as_graph(models.two_linear_latents())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph, exact_elimination=("a",))
        assert problem.exact_elimination == ("a",)
        assert problem.residual_parameters == ("b",)
        assert [name for name, _ in problem.shapes] == ["b"], (
            "shapes must cover the residual parameters and only those; a "
            "backend sized from this layout would allocate for an eliminated "
            "parameter"
        )

    def test_a_term_cannot_be_filed_on_both_sides(self):
        """``__post_init__``'s second check, which nothing else constructs."""
        with pytest.raises(ValueError, match="both the prior and the likelihood"):
            CompiledEvidenceProblem(
                log_prior=lambda values: jnp.zeros(()),
                log_likelihood=lambda values: jnp.zeros(()),
                prior_sample=lambda key: {},
                exact_elimination=(),
                residual_parameters=("w",),
                shapes=(("w", ()),),
                prior_terms=("w",),
                likelihood_terms=("w",),
            )


class TestPriorSampleIsGradedRatherThanOnlyRaised:
    """Five mutants lived here because only the RAISE was ever asserted.

    ``prior_sample`` had exactly one test -- that an improper prior raises --
    and no test called it on a graph it can draw from. So ignoring the key
    (every "draw" identical, which for a nested sampler is fatal), returning the
    observed sites as well, and returning zeros all survived the whole file.
    """

    def test_two_keys_give_two_different_draws(self):
        """M16. A sampler fed identical live points explores nothing, and the
        evidence it reports would be confidently wrong rather than noisy."""
        graph = as_graph(models.two_linear_latents())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            first = problem.prior_sample(jax.random.key(0))
            second = problem.prior_sample(jax.random.key(1))
        assert set(first) == set(second)
        assert any(
            not jnp.array_equal(first[name], second[name]) for name in first
        ), "two different keys produced identical draws"

    def test_the_same_key_gives_the_same_draw(self):
        """Reproducibility is the other half: a run record names its seed."""
        graph = as_graph(models.two_linear_latents())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            first = problem.prior_sample(jax.random.key(7))
            second = problem.prior_sample(jax.random.key(7))
        for name in first:
            assert jnp.array_equal(first[name], second[name])

    def test_it_returns_exactly_the_latents_and_no_observed_site(self):
        """M17. An observed site handed back as a parameter would be integrated
        over, which is a different model."""
        graph = as_graph(models.two_observations())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            drawn = problem.prior_sample(jax.random.key(0))
        assert set(drawn) == set(graph.latents)
        assert not (set(drawn) & set(graph.observed))

    #: Draws, and the band's width in standard errors. Both DECLARED, and the
    #: band is derived from them rather than picked (spec section 9.3).
    #:
    #: The sample standard deviation of ``n`` draws from ``N(0, sigma)`` has
    #: standard error ``sigma / sqrt(2n)`` for large ``n``. At ``n = 8000`` that
    #: is ``0.0237``, so a ``k = 5`` band is ``3 * (1 +- 5/sqrt(2n))`` =
    #: ``(2.881, 3.119)``, with a two-sided false-positive rate of ``5.7e-7``.
    #:
    #: ``n`` is 8000 and not 4000 because the band has to KILL something. A
    #: first version used 4000 draws and a hand-picked ``(2.7, 3.3)``, which is
    #: 8.9 standard errors wide -- a false-positive rate of 4e-19, and a
    #: mutation scaling every draw by 1.05 gives 3.150 and SURVIVES it. At
    #: n=4000 even a 5-sigma band (2.832, 3.168) admits it; 8000 is the point
    #: where 5 sigma and a 5% error separate. A tolerance chosen for comfort
    #: rather than derived is red line 6, and this one was chosen for comfort.
    PRIOR_DRAWS = 8000
    BAND_SIGMAS = 5.0

    def test_the_draws_come_from_the_declared_prior(self):
        """M18. Zeros are a valid-looking dict of the right keys and shapes."""
        sigma = 3.0  # student_t_likelihood declares w ~ Normal(0, 3)
        graph = as_graph(models.student_t_likelihood())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            draws = jnp.asarray(
                [
                    problem.prior_sample(jax.random.key(seed))["w"]
                    for seed in range(self.PRIOR_DRAWS)
                ]
            )
        spread = float(jnp.std(draws))
        half = self.BAND_SIGMAS * sigma / math.sqrt(2 * self.PRIOR_DRAWS)
        assert abs(spread - sigma) <= half, (
            f"{self.PRIOR_DRAWS} prior draws of w ~ N(0, {sigma}) have spread "
            f"{spread}; the derived {self.BAND_SIGMAS}-sigma band is "
            f"({sigma - half:.4f}, {sigma + half:.4f})"
        )

    def test_the_shapes_match_the_declared_layout(self):
        """A plated latent must draw at its plate width, not as a scalar."""
        graph = as_graph(models.plated_latent())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            drawn = problem.prior_sample(jax.random.key(0))
        for name, shape in problem.shapes:
            assert jnp.shape(drawn[name]) == shape, (
                f"{name} drawn at {jnp.shape(drawn[name])}, layout says {shape}"
            )

    def test_each_latent_gets_its_OWN_draw_on_a_multi_latent_graph(self):
        """Two mutants lived here, and one of them is sampler-fatal.

        Every value assertion above uses a SINGLE-latent graph; the multi-latent
        tests checked only key sets and key-determinism. So on a graph with more
        than one latent, ``prior_sample`` could drop a latent, or hand every
        latent the FIRST one's value, with nothing red -- the same class of
        fatality the ignore-the-key mutant was written to prevent, one dimension
        over.

        Graded by the only thing that separates the latents: their declared
        widths. ``a`` and ``b`` are given widths differing by 8x, so a draw
        copied from one to the other shows up in the spread.
        """
        sigma_a, sigma_b = 0.5, 4.0
        graph = Graph(
            nodes=(
                _bare_latent("a", scale=sigma_a),
                _bare_latent("b", scale=sigma_b),
            ),
            plates=(),
        )
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            drawn = [
                problem.prior_sample(jax.random.key(seed))
                for seed in range(self.PRIOR_DRAWS)
            ]
        assert all(set(d) == {"a", "b"} for d in drawn), "a latent was dropped"
        for name, sigma in (("a", sigma_a), ("b", sigma_b)):
            spread = float(jnp.std(jnp.asarray([d[name] for d in drawn])))
            half = self.BAND_SIGMAS * sigma / math.sqrt(2 * self.PRIOR_DRAWS)
            assert abs(spread - sigma) <= half, (
                f"{name} drawn with spread {spread}; its declared width is "
                f"{sigma}, derived band ({sigma - half:.4f}, {sigma + half:.4f})"
            )
        first, second = drawn[0]["a"], drawn[0]["b"]
        assert not jnp.array_equal(first, second), (
            "both latents drew the identical value; a sampler handed this "
            "explores one dimension while reporting two"
        )


class TestAGraphLevelTermSeesOnlyItsDeclaredBlock:
    """M14: routing a term ``graph.latents`` instead of its own ``over``.

    ``Graph._check_evidence_term`` makes ``over`` an enforceable dependency
    boundary. Passing the full latent environment erases it, and survives every
    other test here because the reference in this file makes the same call.
    """

    def test_the_term_receives_exactly_its_over_block(self):
        seen = {}

        class _Recording:
            over = ("a",)

            def log_density(self, graph, values):
                del graph
                seen["names"] = tuple(sorted(values))
                return jnp.zeros(())

        graph = Graph(
            nodes=(_bare_latent("a"), _bare_latent("b")),
            plates=(),
            evidence_terms=(_Recording(),),
        )
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            problem.log_likelihood({"a": jnp.asarray(0.1), "b": jnp.asarray(0.2)})
        assert seen["names"] == ("a",), (
            f"the term was handed {seen['names']}; its declared over is ('a',)"
        )

    def test_a_non_scalar_graph_level_term_is_refused_as_log_joint_refuses_it(self):
        """The scalar requirement `_graph_term_value`'s docstring claimed and
        did not enforce: measured, `log_joint` raised and this returned a
        vector."""
        from bayesmith.errors import GraphError

        class _Vector:
            over = ("a",)

            def log_density(self, graph, values):
                del graph, values
                return jnp.asarray([-1.0, -2.0])

        graph = Graph(
            nodes=(_bare_latent("a"),), plates=(), evidence_terms=(_Vector(),)
        )
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            with pytest.raises(GraphError) as compiled:
                problem.log_likelihood({"a": jnp.asarray(0.1)})
            with pytest.raises(GraphError) as direct:
                log_joint(graph, {"a": jnp.asarray(0.1)})
        # The SLOT, not just "one scalar". Matching only the shared phrase is a
        # guard reading a spelling weak enough to admit the divergence it exists
        # to prevent: this refusal named the term's CLASS while log_joint named
        # the slot, so two same-class terms were indistinguishable and a
        # joint_prior could not be told from an evidence_terms[0].
        assert "evidence_terms[0]" in str(compiled.value), str(compiled.value)
        assert "one scalar" in str(compiled.value)
        assert str(compiled.value) == str(direct.value), (
            f"compiled: {compiled.value}\nlog_joint: {direct.value}"
        )

    def test_a_non_scalar_joint_prior_names_the_joint_prior_slot(self):
        """The other slot, which the class-name label could not distinguish."""
        from bayesmith.errors import GraphError

        class _Vector:
            over = ("a",)

            def log_density(self, graph, values):
                del graph, values
                return jnp.asarray([-1.0, -2.0])

        graph = Graph(nodes=(_bare_latent("a"),), plates=(), joint_prior=_Vector())
        with jax.enable_x64(True):
            problem = compile_evidence_problem(graph)
            with pytest.raises(GraphError) as compiled:
                problem.log_prior({"a": jnp.asarray(0.1)})
            with pytest.raises(GraphError) as direct:
                log_joint(graph, {"a": jnp.asarray(0.1)})
        assert "joint_prior" in str(compiled.value), str(compiled.value)
        assert str(compiled.value) == str(direct.value)
