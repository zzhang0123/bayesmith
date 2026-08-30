"""P4 -- graph reduction and the graph-level evidence-factor slot.

The dangerous failure is a graph that keeps the likelihood nodes after their
marginal likelihood has been attached.  Its arrays all have the right shape,
and NumPyro will sample it without complaint; only an absolute-density check
can say that the data entered twice.  The last two tests therefore compare to
a dense integral and pressure-test the same assertion on that mutant.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density as numpyro_log_density

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic
from bayesmith.graph.trace import observe, plate, sample, trace


class _QuadraticTerm(eqx.Module):
    """Small graph-level density with the protocol P4 promises."""

    over: tuple[str, ...] = eqx.field(static=True)
    centre: jax.Array
    scale: jax.Array
    offset: jax.Array

    def log_density(self, graph: Graph, values: dict[str, Any]) -> jax.Array:
        del graph
        total = self.offset
        for name in self.over:
            total = total + dist.Normal(self.centre, self.scale).log_prob(values[name])
        return total


def _term(*over: str, offset: float = 0.0) -> _QuadraticTerm:
    return _QuadraticTerm(
        over=over,
        centre=jnp.asarray(0.25),
        scale=jnp.asarray(1.4),
        offset=jnp.asarray(offset),
    )


class _MissingLogDensity:
    over = ("x",)


class _MissingOver:
    def log_density(self, graph, values):
        del graph, values
        return jnp.zeros(())


class _VectorTerm:
    over = ("x",)

    def log_density(self, graph, values):
        del graph, values
        return jnp.asarray([-1.0, -2.0])


def _latent(name: str, loc: float = 0.0, scale: float = 1.0) -> Probabilistic:
    return Probabilistic(
        name=name,
        parents=(),
        plate=(),
        dist_fn=lambda: dist.Normal(loc, scale),
        observed=None,
    )


def test_evidence_terms_default_empty_and_a_well_formed_term_is_accepted():
    bare = Graph(nodes=(_latent("x"),), plates=())
    evidence = _term("x")
    declared = Graph(nodes=bare.nodes, plates=(), evidence_terms=(evidence,))

    assert bare.evidence_terms == ()
    assert declared.evidence_terms[0] is evidence


def test_evidence_terms_itself_is_a_tuple_not_a_second_mutable_node_list():
    with pytest.raises(GraphError, match="evidence_terms must be a tuple"):
        Graph(
            nodes=(_latent("x"),),
            plates=(),
            evidence_terms=[_term("x")],
        )

    assert Graph(
        nodes=(_latent("x"),), plates=(), evidence_terms=(_term("x"),)
    ).evidence_terms


def test_an_evidence_term_without_the_density_protocol_is_refused_with_a_route():
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*log_density.*Provide an object with both",
    ):
        Graph(
            nodes=(_latent("x"),),
            plates=(),
            evidence_terms=(_MissingLogDensity(),),
        )

    # The neighbouring valid shape proves this is a protocol check rather
    # than a blanket refusal of graph-level likelihood terms.
    assert Graph(
        nodes=(_latent("x"),), plates=(), evidence_terms=(_term("x"),)
    ).evidence_terms


def test_an_evidence_term_without_over_is_refused_and_the_protocol_is_reachable():
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*over.*Provide an object with both",
    ):
        Graph(
            nodes=(_latent("x"),),
            plates=(),
            evidence_terms=(_MissingOver(),),
        )

    assert Graph(
        nodes=(_latent("x"),), plates=(), evidence_terms=(_term("x"),)
    ).evidence_terms


@pytest.mark.parametrize(
    "bad, phrase",
    [
        (type("NotCallable", (), {"over": ("x",), "log_density": 3})(), "callable"),
        (
            type(
                "ListOver",
                (),
                {"over": ["x"], "log_density": lambda self, graph, values: 0.0},
            )(),
            "tuple",
        ),
        (
            type(
                "NonStringOver",
                (),
                {
                    "over": ([1],),
                    "log_density": lambda self, graph, values: 0.0,
                },
            )(),
            "non-string",
        ),
        (_term("x", "x"), "repeats"),
    ],
)
def test_the_evidence_protocol_refuses_ambiguous_shapes(bad, phrase):
    with pytest.raises(GraphError, match=phrase):
        Graph(nodes=(_latent("x"),), plates=(), evidence_terms=(bad,))

    assert Graph(
        nodes=(_latent("x"),), plates=(), evidence_terms=(_term("x"),)
    ).evidence_terms


def test_an_evidence_term_over_a_nonlatent_is_refused_with_a_route():
    observed = Probabilistic(
        name="d",
        parents=("x",),
        plate=(),
        dist_fn=lambda x: dist.Normal(x, 1.0),
        observed=jnp.asarray(0.3),
    )
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*not latents.*Keep evidence over remaining latents",
    ):
        Graph(
            nodes=(_latent("x"), observed),
            plates=(),
            evidence_terms=(_term("d"),),
        )

    assert Graph(
        nodes=(_latent("x"), observed),
        plates=(),
        evidence_terms=(_term("x"),),
    ).evidence_terms


def test_evidence_and_the_joint_prior_must_be_disjoint():
    nodes = (_latent("x"), _latent("y"))
    prior = _term("x")

    accepted = Graph(
        nodes=nodes,
        plates=(),
        joint_prior=prior,
        evidence_terms=(_term("y"),),
    )
    assert accepted.joint_prior is prior

    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*overlaps joint_prior.*Keep the two blocks disjoint",
    ):
        Graph(
            nodes=nodes,
            plates=(),
            joint_prior=prior,
            evidence_terms=(_term("x"),),
        )


def test_log_joint_adds_every_evidence_term_from_the_graph_declaration():
    bare = Graph(nodes=(_latent("x", loc=-0.4, scale=2.1),), plates=())
    first = _term("x", offset=0.7)
    second = _term("x", offset=-1.2)
    graph = Graph(
        nodes=bare.nodes,
        plates=(),
        evidence_terms=(first, second),
    )
    at = {"x": jnp.asarray(0.8)}

    expected = (
        log_joint(bare, at)
        + first.log_density(graph, at)
        + second.log_density(graph, at)
    )
    assert float(log_joint(graph, at)) == pytest.approx(float(expected), rel=1e-7)


def test_a_non_scalar_evidence_density_is_refused_by_both_graph_scans():
    graph = Graph(
        nodes=(_latent("x"),),
        plates=(),
        evidence_terms=(_VectorTerm(),),
    )
    at = {"x": jnp.asarray(0.2)}

    with pytest.raises(GraphError, match=r"evidence_terms\[0\].*scalar.*shape \(2,\)"):
        log_joint(graph, at)
    with pytest.raises(GraphError, match=r"evidence_terms\[0\].*scalar.*shape \(2,\)"):
        numpyro_log_density(to_numpyro(graph), (), {}, at)


def test_the_bridge_emits_each_evidence_factor_outside_every_plate():
    def model():
        obs = plate("obs", 4)
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "d",
            lambda value: dist.Normal(value, 0.6),
            x,
            obs=jnp.asarray([0.1, 0.2, -0.4, 0.8]),
            plate=obs,
        )

    bare = trace(model)
    graph = Graph(
        nodes=bare.nodes,
        plates=bare.plates,
        evidence_terms=(_term("x", offset=-0.9), _term("x", offset=0.3)),
    )
    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(4))
    ).get_trace()

    assert tuple(model_trace["evidence_0"]["cond_indep_stack"]) == ()
    assert tuple(model_trace["evidence_1"]["cond_indep_stack"]) == ()

    at = {"x": jnp.asarray(0.35)}
    bridged, _ = numpyro_log_density(to_numpyro(graph), (), {}, at)
    assert float(bridged) == pytest.approx(float(log_joint(graph, at)), rel=1e-6)


def test_a_factor_inside_a_plate_is_silently_multiplied_by_its_size():
    """Characterise the upstream rule that makes placement load-bearing."""
    size = 5
    term = jnp.asarray(-1.7)

    def outside():
        numpyro.factor("evidence", term)

    def inside():
        with numpyro.plate("obs", size):
            numpyro.factor("evidence", term)

    outside_density, _ = numpyro_log_density(outside, (), {}, {})
    inside_density, _ = numpyro_log_density(inside, (), {}, {})
    assert float(inside_density / outside_density) == pytest.approx(float(size))


def test_an_evidence_factor_name_cannot_collide_with_a_graph_node_name():
    graph = Graph(
        nodes=(_latent("evidence_0"),),
        plates=(),
        evidence_terms=(_term("evidence_0"),),
    )
    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(9))
    ).get_trace()

    assert model_trace["evidence_0"]["type"] == "sample"
    assert model_trace["_evidence_0"]["type"] == "sample"


def test_an_evidence_factor_name_cannot_collide_with_a_plate_name():
    def model():
        group = plate("evidence_0", 3)
        sample("x", lambda: dist.Normal(0.0, 1.0), plate=group)

    bare = trace(model)
    graph = Graph(
        nodes=bare.nodes,
        plates=bare.plates,
        evidence_terms=(_term(offset=-0.4),),
    )
    model_trace = numpyro.handlers.trace(
        numpyro.handlers.seed(to_numpyro(graph), jax.random.key(10))
    ).get_trace()

    assert model_trace["evidence_0"]["type"] == "plate"
    assert model_trace["_evidence_0"]["type"] == "sample"


class _IntegratedGaussianTerm(eqx.Module):
    """The exact integral over ``x`` in :func:`_collapsible_graph`.

    The dense integral used as the oracle below does not call this method.
    Keeping the two constructions separate is what makes the test capable of
    catching either a graph-wiring error or an error in this closed form.
    """

    over: tuple[str, ...] = eqx.field(static=True)
    data: jax.Array
    basis: jax.Array
    trend: jax.Array
    x_loc: jax.Array
    x_scale: jax.Array
    sigma: jax.Array

    def log_density(self, graph: Graph, values: dict[str, Any]) -> jax.Array:
        del graph
        gain = values["gain"]
        offset = values["offset"]
        # The graph's global fixtures are float32 arrays while this absolute
        # density check deliberately runs in float64. The products happen in
        # the graph's order (gain*basis, offset*trend) and are then promoted:
        # doing the outer product in float32 first loses 1.8e-5 nats at the
        # endpoint and turns a precision test into a dtype test.
        direction = (gain * self.basis).astype(self.sigma.dtype)
        baseline = (offset * self.trend).astype(self.sigma.dtype)
        data = self.data.astype(self.sigma.dtype)
        mean = baseline + direction * self.x_loc
        covariance = self.sigma**2 * jnp.eye(data.size)
        covariance = covariance + self.x_scale**2 * jnp.outer(direction, direction)
        residual = data - mean
        sign, logdet = jnp.linalg.slogdet(covariance)
        quadratic = residual @ jnp.linalg.solve(covariance, residual)
        return -0.5 * (
            data.size * jnp.log(2.0 * jnp.pi)
            + jnp.where(sign > 0, logdet, jnp.nan)
            + quadratic
        )


X_LOC = 0.35
X_SCALE = 1.7
SIGMA = 0.55
BASIS = jnp.asarray([-0.8, 0.2, 1.1, 1.7])
TREND = jnp.asarray([1.0, 0.5, -0.25, 1.25])
DATA = jnp.asarray([0.4, -0.7, 1.2, 2.3])


def _integrated_term() -> _IntegratedGaussianTerm:
    return _IntegratedGaussianTerm(
        over=("gain", "offset"),
        data=DATA,
        basis=BASIS,
        trend=TREND,
        x_loc=jnp.asarray(X_LOC),
        x_scale=jnp.asarray(X_SCALE),
        sigma=jnp.asarray(SIGMA),
    )


def _collapsible_graph(*, unrelated_branch: bool = False) -> Graph:
    nodes: list[Any] = [
        _latent("x", loc=X_LOC, scale=X_SCALE),
        _latent("gain", loc=0.1, scale=1.3),
        _latent("offset", loc=-0.2, scale=0.8),
        Const(name="basis", parents=(), plate=(), value=BASIS),
        Const(name="trend", parents=(), plate=(), value=TREND),
        Deterministic(
            name="mu",
            parents=("x", "gain", "offset", "basis", "trend"),
            plate=(),
            fn=lambda x, gain, offset, basis, trend: offset * trend + gain * basis * x,
            linear_in=("x",),
        ),
        Probabilistic(
            name="d",
            parents=("mu",),
            plate=(),
            dist_fn=lambda mu: dist.Normal(mu, SIGMA).to_event(1),
            observed=DATA,
        ),
    ]
    if unrelated_branch:
        nodes.extend(
            [
                _latent("z", loc=0.5, scale=1.6),
                Deterministic(
                    name="z_mu",
                    parents=("z",),
                    plate=(),
                    fn=lambda z: 2.0 * z,
                    linear_in=("z",),
                ),
                Probabilistic(
                    name="z_data",
                    parents=("z_mu",),
                    plate=(),
                    dist_fn=lambda z_mu: dist.Normal(z_mu, 0.9),
                    observed=jnp.asarray(1.1),
                ),
            ]
        )
    return Graph(nodes=tuple(nodes), plates=())


def _reduce(
    graph: Graph,
    term: Any,
    *,
    absorb_observed: tuple[str, ...] = ("d",),
    nuts_latents: tuple[str, ...] = ("gain", "offset"),
) -> Any:
    from bayesmith.graph.reduction import reduce_with_evidence

    return reduce_with_evidence(
        graph,
        remove_latents=("x",),
        absorb_observed=absorb_observed,
        evidence_term=term,
        nuts_latents=nuts_latents,
    )


def test_reduction_and_attachment_are_one_graph_result_and_keep_live_order():
    original = _collapsible_graph(unrelated_branch=True)
    term = _integrated_term()
    reduced = _reduce(
        original,
        term,
        nuts_latents=("gain", "offset", "z"),
    )

    assert reduced.names == (
        "gain",
        "offset",
        "basis",
        "trend",
        "z",
        "z_mu",
        "z_data",
    )
    assert reduced.evidence_terms[-1] is term
    assert original.names == (
        "x",
        "gain",
        "offset",
        "basis",
        "trend",
        "mu",
        "d",
        "z",
        "z_mu",
        "z_data",
    )
    assert original.evidence_terms == ()

    # Reach the retained child, rather than treating constructor success as
    # evidence that the reduced topology is executable (D23).
    value = log_joint(
        reduced,
        {"gain": jnp.asarray(0.2), "offset": jnp.asarray(-0.1), "z": jnp.asarray(0.4)},
    )
    assert jnp.isfinite(value)


def test_the_reduction_result_is_nuts_only_and_generic_compile_refuses_it():
    from bayesmith import compile as compile_graph
    from bayesmith.graph.reduction import ReducedGraph

    reduced = _reduce(_collapsible_graph(), _integrated_term())
    assert isinstance(reduced, ReducedGraph)
    with pytest.raises(
        GraphError,
        match="NUTS-only.*to_numpyro.*nuts.*generic compile",
    ):
        compile_graph(reduced)

    assert jnp.isfinite(
        log_joint(
            reduced,
            {"gain": jnp.asarray(0.2), "offset": jnp.asarray(-0.1)},
        )
    )


def test_an_unwrapped_reduced_graph_cannot_enter_a_public_exact_block_builder():
    from bayesmith import linear_operator

    reduced = _reduce(
        _collapsible_graph(unrelated_branch=True),
        _integrated_term(),
        nuts_latents=("gain", "offset", "z"),
    )
    raw = reduced.as_graph()
    at = {"offset": jnp.asarray(-0.1), "z": jnp.asarray(0.4)}
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*outside the NUTS block.*keep.*explicit",
    ):
        linear_operator(
            raw,
            ("gain",),
            at=at,
            at_points=(at,),
            scales=(1.0,),
        )

    term_free = Graph(nodes=raw.nodes, plates=raw.plates)
    assert linear_operator(
        term_free,
        ("gain",),
        at=at,
        at_points=(at,),
        scales=(1.0,),
    ).names == ("gain",)


def test_partition_and_compile_check_raw_evidence_against_the_derived_nuts_block():
    from bayesmith import compile as compile_graph
    from bayesmith.dispatch.classify import partition

    bare = _collapsible_graph()
    unsafe = Graph(
        nodes=bare.nodes,
        plates=bare.plates,
        evidence_terms=(_term("x"),),
    )
    with pytest.raises(
        GraphError,
        match=(
            r"evidence_terms\[0\].*outside the NUTS block.*"
            r"Exact and conditional.*put.*NUTS.*keep.*explicit"
        ),
    ):
        partition(unsafe)
    with pytest.raises(
        GraphError,
        match=(
            r"evidence_terms\[0\].*outside the NUTS block.*"
            r"Exact and conditional.*put.*NUTS.*keep.*explicit"
        ),
    ):
        compile_graph(unsafe)

    safe = Graph(
        nodes=bare.nodes,
        plates=bare.plates,
        evidence_terms=(_term("gain"),),
    )
    assert "gain" in partition(safe).nuts
    sampled = compile_graph(safe).sampled
    assert sampled is not None and "gain" in sampled.latents


def test_factor_dispatch_checks_raw_graph_evidence_at_plan_and_execution_edges():
    from bayesmith.dispatch.factor import (
        declared_partition,
        estimate_factors,
        factor_partition,
        sample_factors,
    )

    bare = _collapsible_graph()
    unsafe = Graph(
        nodes=bare.nodes,
        plates=bare.plates,
        evidence_terms=(_term("x"),),
    )
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*outside the NUTS block.*keep.*explicit",
    ):
        factor_partition(unsafe)

    safe = Graph(
        nodes=bare.nodes,
        plates=bare.plates,
        evidence_terms=(_term("gain"),),
    )
    assert "gain" in factor_partition(safe).nuts

    declared = (("x",), "gcr"), (("gain", "offset"), "nuts")
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*outside the NUTS block.*keep.*explicit",
    ):
        declared_partition(unsafe, declared, measure=False)
    assert "gain" in declared_partition(safe, declared, measure=False).nuts

    # A plan can be cached or supplied by a caller. Reusing one derived from
    # the term-free graph must not bypass either execution boundary.
    old_plan = factor_partition(bare)
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*outside the NUTS block.*keep.*explicit",
    ):
        sample_factors(
            unsafe,
            old_plan,
            jax.random.key(12),
            num_warmup=0,
            num_samples=1,
        )
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*outside the NUTS block.*keep.*explicit",
    ):
        estimate_factors(unsafe, old_plan, sweeps=1, steps=1)


def test_a_live_probabilistic_descendant_of_the_removed_block_is_refused():
    graph = _collapsible_graph()
    with pytest.raises(
        GraphError,
        match="probabilistic descendant.*'d'.*Add it to absorb_observed.*do not remove",
    ):
        _reduce(graph, _integrated_term(), absorb_observed=())

    assert "d" not in _reduce(graph, _integrated_term()).names


def test_an_absorbed_observation_must_descend_from_the_removed_frontier():
    graph = _collapsible_graph(unrelated_branch=True)
    with pytest.raises(
        GraphError,
        match=(
            r"absorb_observed.*\['z_data'\].*not descendants.*"
            r"Keep independent likelihoods explicit.*separate reduction"
        ),
    ):
        _reduce(
            graph,
            _integrated_term(),
            absorb_observed=("d", "z_data"),
            nuts_latents=("gain", "offset", "z"),
        )

    accepted = _reduce(
        graph,
        _integrated_term(),
        nuts_latents=("gain", "offset", "z"),
    )
    assert "z_data" in accepted.names


def test_every_existing_or_new_evidence_term_must_be_over_the_nuts_block():
    bare = _collapsible_graph()
    graph = Graph(
        nodes=bare.nodes,
        plates=bare.plates,
        evidence_terms=(_term("offset"),),
    )
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*non-NUTS.*'offset'.*Add it to nuts_latents.*likelihood explicit",
    ):
        _reduce(graph, _term("gain"), nuts_latents=("gain",))
    with pytest.raises(
        GraphError,
        match=r"evidence_terms\[0\].*non-NUTS.*'offset'.*Add it to nuts_latents.*likelihood explicit",
    ):
        _reduce(bare, _term("offset"), nuts_latents=("gain",))

    accepted = _reduce(
        graph,
        _term("gain"),
        nuts_latents=("gain", "offset"),
    )
    assert len(accepted.evidence_terms) == 2


def test_duplicate_reduction_arguments_are_refused_but_unique_ones_are_accepted():
    from bayesmith.graph.reduction import reduce_with_evidence

    graph = _collapsible_graph()
    common = {"graph": graph, "evidence_term": _integrated_term()}
    cases = (
        {
            "remove_latents": ("x", "x"),
            "absorb_observed": ("d",),
            "nuts_latents": ("gain", "offset"),
            "message": "remove_latents repeats",
        },
        {
            "remove_latents": ("x",),
            "absorb_observed": ("d", "d"),
            "nuts_latents": ("gain", "offset"),
            "message": "absorb_observed repeats",
        },
        {
            "remove_latents": ("x",),
            "absorb_observed": ("d",),
            "nuts_latents": ("gain", "gain", "offset"),
            "message": "nuts_latents repeats",
        },
    )
    for case in cases:
        message = case.pop("message")
        with pytest.raises(GraphError, match=message):
            reduce_with_evidence(**common, **case)

    assert _reduce(graph, _integrated_term()).names


def test_reduction_arguments_must_name_nodes_of_the_declared_role():
    from bayesmith.graph.reduction import reduce_with_evidence

    graph = _collapsible_graph()
    with pytest.raises(GraphError, match=r"remove_latents.*\['d'\].*not latent"):
        reduce_with_evidence(
            graph,
            remove_latents=("d",),
            absorb_observed=(),
            evidence_term=_integrated_term(),
            nuts_latents=("x", "gain", "offset"),
        )
    with pytest.raises(GraphError, match=r"absorb_observed.*\['mu'\].*not observed"):
        reduce_with_evidence(
            graph,
            remove_latents=("x",),
            absorb_observed=("mu",),
            evidence_term=_integrated_term(),
            nuts_latents=("gain", "offset"),
        )
    with pytest.raises(GraphError, match=r"nuts_latents.*\['x'\].*not retained"):
        reduce_with_evidence(
            graph,
            remove_latents=("x",),
            absorb_observed=("d",),
            evidence_term=_integrated_term(),
            nuts_latents=("x", "gain", "offset"),
        )

    assert _reduce(graph, _integrated_term()).names


PARAMETER_POINTS = (
    {"gain": jnp.asarray(-1.2), "offset": jnp.asarray(-1.1)},
    {"gain": jnp.asarray(-1.2), "offset": jnp.asarray(0.75)},
    {"gain": jnp.asarray(1.4), "offset": jnp.asarray(-1.1)},
    {"gain": jnp.asarray(1.4), "offset": jnp.asarray(0.75)},
    {"gain": jnp.asarray(0.1), "offset": jnp.asarray(-0.2)},
)


def _dense_integral_over_x(graph: Graph, point: dict[str, jax.Array]) -> float:
    grid = jnp.linspace(X_LOC - 12.0 * X_SCALE, X_LOC + 12.0 * X_SCALE, 60_001)
    log_values = jax.vmap(lambda x: log_joint(graph, {**point, "x": x}))(grid)
    peak = jnp.max(log_values)
    integral = jnp.trapezoid(jnp.exp(log_values - peak), grid)
    return float(peak + jnp.log(integral))


def _absolute_density_vectors(
    original: Graph, candidate: Any
) -> tuple[np.ndarray, np.ndarray]:
    from bayesmith.graph.reduction import ReducedGraph

    oracle = np.asarray(
        [_dense_integral_over_x(original, point) for point in PARAMETER_POINTS]
    )
    if isinstance(candidate, ReducedGraph) or "x" not in candidate.latents:
        found = np.asarray(
            [float(log_joint(candidate, point)) for point in PARAMETER_POINTS]
        )
    else:
        found = np.asarray(
            [_dense_integral_over_x(candidate, point) for point in PARAMETER_POINTS]
        )
    return oracle, found


def _assert_absolute_density_is_preserved(original: Graph, candidate: Graph) -> None:
    oracle, found = _absolute_density_vectors(original, candidate)
    np.testing.assert_allclose(found, oracle, rtol=0.0, atol=2.0e-9)


def test_reduced_absolute_density_matches_a_dense_integral_at_five_points():
    """All four corners exercise both endpoints of both retained parameters."""
    with jax.enable_x64(True):
        original = _collapsible_graph()
        reduced = _reduce(original, _integrated_term())
        _assert_absolute_density_is_preserved(original, reduced)


def test_the_absolute_density_guard_really_catches_an_unreduced_graph_mutant():
    """Attach the term without reducing; the production guard must turn red."""
    with jax.enable_x64(True):
        original = _collapsible_graph()
        mutant = Graph(
            nodes=original.nodes,
            plates=original.plates,
            evidence_terms=(_integrated_term(),),
        )
        with pytest.raises(AssertionError):
            _assert_absolute_density_is_preserved(original, mutant)

        oracle, doubled = _absolute_density_vectors(original, mutant)
        assert np.min(np.abs(doubled - oracle)) > 1.0
