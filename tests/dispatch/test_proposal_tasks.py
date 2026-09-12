"""Explicit proposal tasks retain model identity, policies and chain semantics."""

import dataclasses

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def diagnostic_precision():
    import jax

    # Numerical rank diagnostics already require model construction in x64.
    with jax.enable_x64(True):
        yield


def test_proposal_policy_roundtrip_and_task_identity():
    from bayesmith.artifacts import (
        PosteriorTask,
        ProposalBlockPolicy,
        new_task_meta,
        proposal_options,
        task_fingerprint,
    )
    from bayesmith.artifacts._codec import canonical_dumps, canonical_loads

    block = ProposalBlockPolicy(("beta",), "iterative_gls")
    task = PosteriorTask(meta=new_task_meta(), backend_options=proposal_options(block))
    assert task.proposals == (block,)
    assert canonical_loads(canonical_dumps(task)) == task
    assert task_fingerprint(task) != task_fingerprint(
        dataclasses.replace(task, backend_options=())
    )
    # Reuse the existing backend-options storage; legacy task schemas do not
    # acquire a new required field and their default fingerprints stay intact.
    assert "proposals" not in {f.name for f in dataclasses.fields(task)}
    assert dataclasses.replace(task, backend_options=()).proposals == ()


def test_default_task_fingerprint_retains_pre_extension_identity():
    from bayesmith.artifacts import PosteriorTask, new_task_meta, task_fingerprint

    # Recorded before introducing ProposalBlockPolicy. Task ids/timestamps are
    # intentionally excluded from this semantic fingerprint.
    assert task_fingerprint(PosteriorTask(meta=new_task_meta())).digest == (
        "25d3cf93fd153d8959db19489ccc7151997bfe71092416793b109824df342cd0"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"names": ()},
        {"names": ("a", "a")},
        {"method": "typo"},
        {"steps": 0},
        {"iterations": True},
        {"scale": 0},
        {"damping": np.nan},
        {"max_parameters": 0},
        {"max_matrix_elements": -1},
    ],
)
def test_proposal_policy_rejects_invalid_configuration(changes):
    from bayesmith.artifacts import ProposalBlockPolicy

    with pytest.raises((TypeError, ValueError)):
        ProposalBlockPolicy(**{"names": ("a",), "method": "gauss_newton", **changes})


def test_proposal_options_validate_disjoint_blocks_at_task_boundary():
    from bayesmith.artifacts import (
        PosteriorTask,
        ProposalBlockPolicy,
        new_task_meta,
        proposal_options,
    )

    with pytest.raises(ValueError, match="overlap|duplicate"):
        PosteriorTask(
            meta=new_task_meta(),
            backend_options=proposal_options(
                ProposalBlockPolicy(("a",), "gauss_newton"),
                ProposalBlockPolicy(("a",), "iterative_gls"),
            ),
        )


def proposal_task(**overrides):
    from bayesmith.artifacts import (
        ComputeBudget,
        PosteriorTask,
        ProposalBlockPolicy,
        new_task_meta,
        proposal_options,
    )

    return PosteriorTask(
        **{
            "meta": new_task_meta(),
            "budget": ComputeBudget(draws=16, warmup=8, chains=2),
            "backend_options": proposal_options(
                ProposalBlockPolicy(("x",), "iterative_gls")
            ),
            **overrides,
        }
    )


def compile_fixture(task):
    from bayesmith import compile_task
    from tests.dispatch.test_task_protocol import model_ref
    from tests.exact.test_proposals import gaussian_graph

    return compile_task(gaussian_graph(bounded=True), task, model_ref=model_ref())


def test_proposal_task_executes_real_chain_with_support_and_counters():
    import jax

    from bayesmith import execute_task
    from bayesmith.artifacts import PosteriorResult, Refusal

    task = proposal_task()
    planned = compile_fixture(task)
    assert not isinstance(planned, Refusal), planned
    assert planned.record.blocks[0].method == "iterative_gls+mh"
    assert planned.record.fallback_policy is None
    details = dict(planned.record.blocks[0].approximation.details)
    assert details["proposal_iterations"] == 5
    assert details["correction"] == "original_target_mh"
    findings = {f.code: f for f in planned.analysis.findings}
    assert any(f.code == "proposal_block" for f in planned.analysis.findings)
    assert findings["block_0_jeffreys"].conclusion == "flat"
    result = execute_task(planned, key=jax.random.key(414))
    assert isinstance(result, PosteriorResult), result
    assert result.representation.method == "proposal_mh"
    assert result.representation.chain_shape == (2, 16)
    draws = result.representation.draws[0].value
    assert draws.shape == (32,)
    assert np.all((-1 <= draws) & (draws <= 1))
    counters = dict(dict(result.run.sampling_details)["proposal_diagnostics"][0])
    assert counters["attempted"] == 32
    assert 0 < counters["accepted"] <= 32
    assert counters["support_rejected"] > 0


def test_proposal_numerical_policy_is_in_compilation_identity():
    from bayesmith.artifacts import ProposalBlockPolicy, proposal_options

    a = proposal_task()
    b = dataclasses.replace(
        a,
        backend_options=proposal_options(
            ProposalBlockPolicy(("x",), "iterative_gls", scale=1.5),
        ),
    )
    assert compile_fixture(a).record.meta.fingerprints.compilation != (
        compile_fixture(b).record.meta.fingerprints.compilation
    )


@pytest.mark.parametrize(
    "options",
    [
        {"chain_method": "parallel"},
        {"solver_maxiter": 10},
        {"block_order": (("unknown",),)},
    ],
)
def test_unsupported_proposal_execution_options_refuse_at_compile(options):
    from bayesmith.artifacts import Refusal

    assert isinstance(compile_fixture(proposal_task(**options)), Refusal)


def test_explicit_false_collapse_option_is_executable():
    import jax

    from bayesmith import execute_task
    from bayesmith.artifacts import PosteriorResult

    task = proposal_task()
    task = dataclasses.replace(
        task, backend_options=task.backend_options + (("collapse", False),)
    )
    assert isinstance(
        execute_task(compile_fixture(task), key=jax.random.key(3)), PosteriorResult
    )


def test_mh_correction_defaults_on_and_legacy_policy_payloads_read_on():
    import json

    from bayesmith.artifacts import PosteriorTask, ProposalBlockPolicy, new_task_meta
    from bayesmith.artifacts._codec import canonical_dumps, canonical_loads

    block = ProposalBlockPolicy(("x",), "gauss_newton")
    assert block.mh_correction is True
    legacy_options = tuple(
        (k, v) for k, v in block.as_options() if k != "mh_correction"
    )
    task = PosteriorTask(
        meta=new_task_meta(), backend_options=(("proposals", (legacy_options,)),)
    )
    assert task.proposals[0].mh_correction is True
    payload = json.loads(canonical_dumps(block))
    payload["fields"] = [
        item for item in payload["fields"] if item[0] != "mh_correction"
    ]
    loaded = canonical_loads(json.dumps(payload).encode())
    assert loaded == block
    payload["fields"].append(["unknown_setting", True])
    with pytest.raises(ValueError, match="fields"):
        canonical_loads(json.dumps(payload).encode())


@pytest.mark.parametrize("value", [0, "false", None])
def test_mh_correction_requires_an_actual_boolean(value):
    from bayesmith.artifacts import ProposalBlockPolicy

    with pytest.raises(TypeError, match="mh_correction"):
        ProposalBlockPolicy(("x",), "gauss_newton", mh_correction=value)


def test_mh_disabled_policy_roundtrips_and_changes_explicit_task_identity():
    from bayesmith.artifacts import (
        ProposalBlockPolicy,
        proposal_options,
        task_fingerprint,
    )
    from bayesmith.artifacts._codec import canonical_dumps, canonical_loads

    off = ProposalBlockPolicy(("x",), "iterative_gls", mh_correction=False)
    task = proposal_task(backend_options=proposal_options(off))
    assert canonical_loads(canonical_dumps(task)) == task
    assert task.proposals[0].mh_correction is False
    on = dataclasses.replace(
        task,
        backend_options=proposal_options(dataclasses.replace(off, mh_correction=True)),
    )
    assert task_fingerprint(task) != task_fingerprint(on)


def test_mh_disabled_plan_and_result_are_explicitly_approximate():
    import jax

    from bayesmith import execute_task
    from bayesmith.artifacts import (
        PosteriorResult,
        ProposalBlockPolicy,
        Refusal,
        TargetFidelity,
        proposal_options,
    )

    task = proposal_task(
        backend_options=proposal_options(
            ProposalBlockPolicy(("x",), "iterative_gls", mh_correction=False)
        )
    )
    planned = compile_fixture(task)
    assert not isinstance(planned, Refusal), planned
    block = planned.record.blocks[0]
    assert block.method == "iterative_gls+uncorrected"
    assert block.approximation.target_fidelity == TargetFidelity.APPROXIMATE
    assert dict(block.approximation.details)["correction"] == "none_approximate"
    assert "original_target_mh" not in block.reason_codes
    assert "approximate" in str(planned.runtime_plan).lower()
    assert planned.runtime_plan.target_fidelity == TargetFidelity.APPROXIMATE
    result = execute_task(planned, key=jax.random.key(414))
    assert isinstance(result, PosteriorResult), result
    assert result.representation.method == "proposal_approximate"
    assert result.run.approximation.target_fidelity == TargetFidelity.APPROXIMATE
    counters = dict(dict(result.run.sampling_details)["proposal_diagnostics"][0])
    assert counters["mh_correction"] is False
    assert counters["mh_attempted"] == counters["mh_accepted"] == 0
    assert counters["mh_acceptance_rate"] is None
    assert counters["accepted"] > 0
    assert counters["support_rejected"] > 0
