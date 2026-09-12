"""Automatic proposals are real plans, with truthful fallback and MH records."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import compile_task, execute_task, observe, sample, trace
from bayesmith.artifacts import (
    ComputeBudget,
    DiagnosticPolicy,
    PosteriorTask,
    ProposalBlockPolicy,
    Refusal,
    model_ref_from_callable,
    new_task_meta,
    proposal_options,
)
from bayesmith.dispatch.proposal_plan import ProposalRuntimePlan
from examples.inference import multiplicative_noise as demo


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64():
        yield


def _task(**kw):
    return PosteriorTask(meta=new_task_meta(), diagnostics=DiagnosticPolicy(enabled=False), **kw)


def _demo_graph():
    u, a = demo.design_matrices(jnp.linspace(0.0, 1.0, 24))
    return trace(demo.model, u, a, demo.mean_signal(u, a, jnp.zeros(2), jnp.ones(2)))


def _compile(graph, task, model=demo.model):
    result = compile_task(graph, task, model_ref=model_ref_from_callable(model, identifier="test.auto_proposals"))
    assert not isinstance(result, Refusal), result
    return result


def test_default_auto_selects_both_demo5_blocks_and_preserves_task():
    task = _task()
    planned = _compile(_demo_graph(), task)
    assert planned.task is task and task.proposals == ()
    assert planned.runtime_plan.selection == "automatic"
    assert [(b.names, b.method) for b in planned.record.blocks] == [
        (("p_g",), "bias_corrected_log_linear+mh"),
        (("p_n",), "iterative_gls+mh"),
    ]
    for block in planned.record.blocks:
        assert "automatic_proposal_selection" in block.reason_codes
        assert dict(block.approximation.details)["proposal_mh_correction"] is True
    assert planned.record.fallback_policy is None
    selection = next(f for f in planned.analysis.findings if f.code == "automatic_proposals")
    assert dict(selection.measurements)["prior_action"] == "unchanged"


@pytest.mark.parametrize("overrides", [
    {"backend_options": (("auto_proposals", False),)},
    {"initialization": None},
    {"block_order": (("p_g", "p_n"),)},
])
def test_legacy_controls_preserve_nuts(overrides):
    plan = _compile(_demo_graph(), _task(**overrides))
    assert [b.method for b in plan.record.blocks] == ["nuts"]


def test_explicit_policy_takes_precedence():
    task = _task(backend_options=proposal_options(ProposalBlockPolicy(("p_n",), "iterative_gls")))
    plan = _compile(_demo_graph(), task)
    assert plan.runtime_plan.selection == "explicit"
    assert [b.names for b in plan.record.blocks] == [("p_n",), ("p_g",)]


@pytest.mark.parametrize("mode", ["nonlinear", "variance_only", "masked", "budget"])
def test_unsuitable_candidates_retain_nuts_with_reasons(mode):
    def model():
        theta = sample("theta", lambda: dist.Uniform(jnp.full(65 if mode == "budget" else 1, 0.2), 2.0))
        def law(t):
            mean = t**2 + 1 if mode == "nonlinear" else jnp.ones_like(t) if mode == "variance_only" else t
            sigma = t * 0.02 if mode == "variance_only" else mean * 0.02
            return dist.Normal(mean, sigma)
        observe("y", law, theta, obs=jnp.ones(65 if mode == "budget" else 1),
                mask=jnp.array([False]) if mode == "masked" else None)

    planned = _compile(trace(model), _task(), model)
    assert [b.method for b in planned.record.blocks] == ["nuts"]
    decisions = dict(next(f for f in planned.analysis.findings if f.code == "automatic_proposals").measurements)["decisions"]
    assert any(row[1] == "nuts" for row in decisions)


def test_graph_faults_are_not_downgraded_to_applicability(monkeypatch):
    from bayesmith.dispatch import auto_proposals
    from bayesmith.errors import GraphError

    def broken(*args):
        raise GraphError("broken model")
    monkeypatch.setattr(auto_proposals, "compile_proposals", broken)
    with pytest.raises(GraphError, match="broken model"):
        _compile(_demo_graph(), _task())


def test_existing_gcr_route_keeps_its_executor():
    def model():
        theta = sample("theta", lambda: dist.Normal(0.0, 1.0))
        observe("y", lambda x: dist.Normal(x, 0.2), theta, obs=0.1)

    plan = _compile(trace(model), _task(), model)
    assert not isinstance(plan.runtime_plan, ProposalRuntimePlan)
    assert [b.method for b in plan.record.blocks] == ["gcr"]


def test_symbolic_zero_noise_dependence_does_not_change_fixed_covariance_route():
    def model():
        theta = sample("theta", lambda: dist.Uniform(-1.0, 1.0))
        observe("y", lambda x: dist.Normal(x, 0.3 + 0.0 * x), theta, obs=0.1)

    plan = _compile(trace(model), _task(), model)
    assert [b.method for b in plan.record.blocks] == ["nuts"]


def test_composed_process_merges_instance_and_background():
    from examples.inference import composed_process

    x = jnp.linspace(0, 1, 24, endpoint=False)
    k, _, response, b, u = composed_process.design_matrices(x)
    graph = trace(composed_process.model, x, k, response, b, u, jnp.full(24, 3.0))
    planned = _compile(graph, _task(), composed_process.model)
    first, remainder = planned.record.blocks
    assert first.names == ("instance", "background")
    assert first.method == "iterative_gls+mh"
    assert sum(np.size(planned.runtime_plan.compiled.initial_values[n]) for n in first.names) == 14
    assert set(remainder.names) == {"power_amplitude", "gain", "nonlinear", "sigma_w"}
    assert remainder.method == "nuts"
    assert any(d[2] == "joint_affinity_validated_merged"
               for d in planned.runtime_plan.selection_decisions)


def test_separately_linear_product_is_not_merged():
    def model():
        a = sample("a", lambda: dist.Uniform(0.5, 2.0))
        b = sample("b", lambda: dist.Uniform(0.5, 2.0))
        observe("y", lambda x, y: dist.Normal(x * y, 0.02 * x * y), a, b, obs=1.0)

    planned = _compile(trace(model), _task(), model)
    assert [b.names for b in planned.record.blocks] == [("a",), ("b",)]
    assert any(d[1] == "merge_refused" for d in planned.runtime_plan.selection_decisions)


@pytest.mark.parametrize("size", [2, 64])
def test_existing_block_is_indivisible_and_aggregate_budget_is_checked(size):
    from bayesmith.dispatch.auto_proposals import select_proposals
    from bayesmith.dispatch.plan import Block

    def model():
        a = sample("a", lambda: dist.Normal(jnp.ones(size - 1), 0.1))
        s = sample("s", lambda: dist.Normal(1.0, 0.1))
        b = sample("b", lambda: dist.Uniform(0.5, 2.0))
        def law(x, y, z):
            mean = jnp.concatenate((x, jnp.atleast_1d(y))) + z
            return dist.Normal(mean, 0.02 * jnp.abs(mean))
        observe("y", law, a, s, b, obs=jnp.full(size, 2.0))

    graph = trace(model)
    legacy = SimpleNamespace(blocks=(Block(("a", "s"), "gcr+mh", "test seed"),
                                    Block(("b",), "nuts", "test remainder")))
    planned, decisions = select_proposals(graph, _task(), legacy)
    if size == 64:
        assert planned is legacy
        assert any(d[1] == "merge_refused" and "budget" in d[2] for d in decisions)
    else:
        assert planned.blocks[0].latents == ("a", "s", "b")


def test_auto_option_is_boolean_and_not_forwarded_to_sampler():
    graph = _demo_graph()
    bad = compile_task(graph, _task(backend_options=(("auto_proposals", "yes"),)),
                       model_ref=model_ref_from_callable(demo.model, identifier="test.auto_proposals"))
    assert isinstance(bad, Refusal)
    task = _task(budget=ComputeBudget(draws=32, warmup=16, chains=1),
                 backend_options=(("auto_proposals", True),))
    planned = _compile(graph, task)
    result = execute_task(planned, key=jax.random.key(42))
    assert not isinstance(result, Refusal)
    for draw in result.representation.draws:
        assert np.all(np.isfinite(draw.value))
    records = dict(result.run.sampling_details)["proposal_diagnostics"]
    assert len(records) == 2
    assert all(dict(row)["attempted"] == 32 for row in records)
    assert planned.record.meta.fingerprints.compilation != _compile(
        graph, replace(task, backend_options=(("auto_proposals", False),))
    ).record.meta.fingerprints.compilation
