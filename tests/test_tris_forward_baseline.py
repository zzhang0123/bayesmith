"""P1 parity: the graph's forward baseline against an independent recomputation."""

import jax
import numpy as np
import pytest

from bayesmith import trace
from bayesmith.graph.evaluate import log_joint
from examples.inference.tris_forward_baseline import (
    batch_mean_mcse,
    log_joint_components,
    predict_from_bundle,
    predict_from_inputs,
    summarize_draws,
)
from examples.inference.tris_rsb_sky import model, model_inputs


def tiny_bundle():
    return {
        "template_k": np.array([10.0, 15.0, 20.0]),
        "region": np.array([0, 1, 2], dtype=np.int32),
        "response_0": np.array([[1.0, 0.2, 0.0], [0.1, 0.4, 0.3]]),
        "response_1": np.array([[0.2, 0.1, 0.8], [0.3, 0.5, 0.1]]),
        "offset_response_0": np.array([1.0, 1.0]),
        "offset_response_1": np.array([1.0, 1.0]),
        "whitened_data_0": np.array([8.0, 9.0]),
        "whitened_data_1": np.array([4.0, 5.0]),
        "frequency_mhz": np.array([600.5, 817.8]),
        "cmb_k": np.array([2.71, 2.706]),
        "reference_cmb_k": np.array(2.715),
    }


def tiny_external():
    return {
        "frequency_mhz": np.array([40.0, 3200.0]),
        "temperature_rj_k": np.array([5792.0, 2.75]),
        "sigma_independent_rj_k": np.array([960.0, 0.008]),
        "tau_rj_k": np.array([10.0, 0.005]),
        "survey_code": np.array([0, 1], dtype=np.int32),
        "survey": np.array(["LWA", "ARCADE"]),
    }


def values(include_rsb=True):
    base = {
        "amplitude": np.array([1.0, 1.1, 0.9]),
        "beta": np.array([-2.7, -2.8, -2.9]),
        "zero_standard": np.array([0.1, -0.2]),
        "haslam_monopole_K": np.array(0.2),
        "calibration_standard": np.array([0.2, -0.3]),
    }
    if include_rsb:
        base.update(rsb_amplitude=np.array(0.5), rsb_beta=np.array(-2.6))
    return base


@pytest.mark.parametrize("include_rsb", [True, False])
def test_graph_log_joint_equals_independent_components(include_rsb):
    with jax.enable_x64(True):
        bundle, external = tiny_bundle(), tiny_external()
        chosen = values(include_rsb)
        inputs = model_inputs(bundle, external, ("LWA", "ARCADE"), include_rsb)
        graph = trace(model, *inputs)
        direct = float(log_joint(graph, chosen))
        parts = log_joint_components(
            bundle, external, ("LWA", "ARCADE"), chosen, include_rsb
        )
        assert parts["support_valid"] is True
        assert parts["log_joint"] == pytest.approx(direct, abs=1e-9)
        assert (
            parts["log_prior"]
            + parts["log_likelihood_tris"]
            + parts["log_likelihood_external"]
        ) == pytest.approx(parts["log_joint"], abs=1e-12)


@pytest.mark.parametrize("include_rsb", [True, False])
def test_raw_bundle_recomputation_matches_the_graph_designs(include_rsb):
    # x64 is required: in float32 the graph's own designs are rounded first and
    # the comparison would measure that rounding, not a real disagreement.
    with jax.enable_x64(True):
        bundle, external = tiny_bundle(), tiny_external()
        chosen = values(include_rsb)
        raw_tris, raw_external = predict_from_bundle(
            bundle, external, ("LWA", "ARCADE"), chosen, include_rsb
        )
        inputs = model_inputs(bundle, external, ("LWA", "ARCADE"), include_rsb)
        input_tris, input_external = predict_from_inputs(inputs, chosen)
        np.testing.assert_allclose(raw_tris, input_tris, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(raw_external, input_external, rtol=1e-12, atol=1e-12)


def test_negative_template_is_support_failure_not_a_silent_number():
    bundle, external = tiny_bundle(), tiny_external()
    chosen = values(include_rsb=True)
    chosen["haslam_monopole_K"] = np.array(-50.0)
    parts = log_joint_components(
        bundle, external, ("LWA", "ARCADE"), chosen, include_rsb=True
    )
    assert parts["support_valid"] is False
    assert parts["log_joint"] == -np.inf
    # The prior and the likelihoods stay finite, so the failure is attributable.
    assert np.isfinite(parts["log_prior"])
    assert np.isfinite(parts["log_likelihood_tris"])


def test_batch_mean_mcse_scales_like_inverse_sqrt_draws():
    rng = np.random.default_rng(0)
    small = rng.normal(size=(2, 100))
    large = rng.normal(size=(2, 4000))
    assert np.allclose(small.mean(axis=1), large.mean(axis=1), atol=10)
    mcse_small = batch_mean_mcse(small, batches=10)
    mcse_large = batch_mean_mcse(large, batches=10)
    assert mcse_large < mcse_small
    summary = summarize_draws({"x": rng.normal(size=(2, 200, 3))})
    assert np.shape(summary["x"]["mcse"]) == (3,)
