"""Research-demo integration, separate from portable compiler regressions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith import trace
from tests.dispatch.test_auto_proposals import _compile, _task


@pytest.fixture(autouse=True)
def x64():
    with jax.enable_x64():
        yield


def test_composed_process_merges_instance_and_background():
    from examples.inference import composed_process

    x = jnp.linspace(0, 1, 24, endpoint=False)
    k, _, response, b, u = composed_process.design_matrices(x)
    graph = trace(composed_process.model, x, k, response, b, u, jnp.full(24, 3.0))
    planned = _compile(graph, _task(), composed_process.model)
    first, remainder = planned.record.blocks
    assert first.names == ("instance", "background")
    assert first.method == "iterative_gls+mh"
    assert (
        sum(
            np.size(planned.runtime_plan.compiled.initial_values[n])
            for n in first.names
        )
        == 14
    )
    assert set(remainder.names) == {"power_amplitude", "gain", "nonlinear", "sigma_w"}
    assert remainder.method == "nuts"
    assert any(
        d[2] == "joint_affinity_validated_merged"
        for d in planned.runtime_plan.selection_decisions
    )
