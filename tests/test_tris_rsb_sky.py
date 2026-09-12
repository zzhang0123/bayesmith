"""Joint TRIS-map and external-background graph behavior."""

import jax
import numpy as np
import pytest
from numpyro.infer.util import log_density

from bayesmith import trace
from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.graph.evaluate import log_joint
from examples.inference.tris_rsb_sky import (
    calibrated_external_mean,
    galactic_template,
    model,
    model_inputs,
    rsb_temperature,
    valid_galactic_template,
)


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


def graph_and_values(include_rsb=True):
    inputs = model_inputs(tiny_bundle(), tiny_external(), ("LWA", "ARCADE"), include_rsb)
    values = {
        "amplitude": np.array([1.0, 1.1, 0.9]),
        "beta": np.array([-2.7, -2.8, -2.9]),
        "zero_standard": np.array([0.1, -0.2]),
        "haslam_monopole_K": np.array(0.2),
        "calibration_standard": np.array([0.2, -0.3]),
    }
    if include_rsb:
        values.update(rsb_amplitude=np.array(0.5), rsb_beta=np.array(-2.6))
    return trace(model, *inputs), values


def test_galactic_template_subtracts_rsb_without_clipping():
    template = np.array([10.0, 20.0])
    np.testing.assert_allclose(galactic_template(template, 1.5, 4.0), [7.5, 17.5])
    assert not valid_galactic_template(galactic_template(template, -1.0, 9.5))


def test_rsb_and_calibration_follow_declared_reference_units():
    assert rsb_temperature(1.2, -2.6, 1000.0) == pytest.approx(1.2)
    mean = calibrated_external_mean(
        np.array([2.7, 2.6]), np.array([4.0, 0.1]), np.array([.5, -.25]),
        np.array([10.0, .005]), np.array([0, 1]),
    )
    np.testing.assert_allclose(mean, [11.7, 2.69875])


def test_invalid_galactic_template_has_negative_infinite_graph_density():
    with jax.enable_x64(True):
        graph, values = graph_and_values(include_rsb=True)
        values.update(haslam_monopole_K=np.array(-20.0), rsb_amplitude=np.array(4.0), rsb_beta=np.array(-2.6))
        assert np.isneginf(float(log_joint(graph, values)))


def test_graph_and_numpyro_bridge_agree_for_valid_joint_density():
    with jax.enable_x64(True):
        graph, values = graph_and_values(include_rsb=True)
        direct = float(log_joint(graph, values))
        bridged, _ = log_density(to_numpyro(graph), (), {}, values)
        assert np.isfinite(direct)
        assert float(bridged) == pytest.approx(direct, abs=1e-10)


def test_no_rsb_graph_omits_only_the_two_rsb_latents():
    with jax.enable_x64(True):
        graph, values = graph_and_values(include_rsb=False)
        assert graph.latents == (
            "amplitude", "beta", "zero_standard", "haslam_monopole_K", "calibration_standard"
        )
        assert np.isfinite(float(log_joint(graph, values)))


def test_model_inputs_reject_unknown_or_empty_survey_selection():
    with pytest.raises(ValueError, match="nonempty"):
        model_inputs(tiny_bundle(), tiny_external(), (), True)
    with pytest.raises(ValueError, match="unknown"):
        model_inputs(tiny_bundle(), tiny_external(), ("OTHER",), True)
