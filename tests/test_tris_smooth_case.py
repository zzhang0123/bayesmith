"""Smooth-foreground case helpers: basis, inputs and the numpy prediction."""

import numpy as np
import pytest

from examples.inference.tris_smooth_case import (
    _repair_positive_template,
    _smooth_whitened_mean,
    _tris_mean,
    basis_matrix,
    heldout_deltas,
    smooth_inputs,
)
from tests.test_tris_forward_baseline import tiny_bundle, tiny_external


def latitude_bundle():
    return {**tiny_bundle(), "galactic_latitude_deg": np.array([5.0, 25.0, 45.0])}


def test_basis_matrix_is_bounded_and_has_one_column_per_function():
    bundle = {**tiny_bundle(), "galactic_latitude_deg": np.linspace(-90.0, 90.0, 181)}
    basis = basis_matrix(bundle, 4)
    assert basis.shape == (181, 4)
    assert np.all(basis >= 0.0)
    assert np.all(basis <= 1.0 + 1e-12)
    assert np.all(basis.max(axis=0) > 0.9)


def test_smooth_inputs_orders_and_shapes():
    bundle = latitude_bundle()
    basis = basis_matrix(bundle, 3)
    inputs = smooth_inputs(bundle, tiny_external(), ("LWA", "ARCADE"), basis, False)
    assert np.shape(inputs[0]) == (3, 3)  # basis
    assert np.shape(inputs[1]) == (2, 3)  # response_0
    assert np.shape(inputs[5]) == (3,)  # template
    assert np.shape(inputs[8]) == (4,)  # frequency index
    assert np.shape(inputs[9]) == (4,)  # whitened data
    assert np.shape(inputs[10]) == (2,)  # external frequency, both surveys
    assert int(inputs[15]) == 2  # calibration coordinates
    assert bool(inputs[16]) is False


def test_smooth_inputs_selects_only_the_requested_survey():
    bundle = latitude_bundle()
    basis = basis_matrix(bundle, 3)
    one = smooth_inputs(bundle, tiny_external(), ("ARCADE",), basis, True)
    assert np.shape(one[10]) == (1,)
    assert int(one[15]) == 1
    assert bool(one[16]) is True


def test_constant_basis_reproduces_a_uniform_sky():
    bundle = latitude_bundle()
    basis = np.ones((3, 1))
    amplitude, beta = 1.5, -2.7
    x = np.array([np.log(amplitude), beta, 0.0, 0.0, 0.0, 0.0, 0.0])
    prediction, external_mean, mask = _smooth_whitened_mean(
        bundle, tiny_external(), ("LWA", "ARCADE"), basis, x, False
    )
    template = np.asarray(bundle["template_k"], dtype=float)
    chunks = []
    for index, frequency in enumerate(np.asarray(bundle["frequency_mhz"], dtype=float)):
        response = np.asarray(bundle[f"response_{index}"], dtype=float)
        sky = template * amplitude * (frequency / 408.0) ** beta + float(
            bundle["cmb_k"][index]
        )
        chunks.append(response @ sky)
    np.testing.assert_allclose(prediction, np.concatenate(chunks))
    assert mask.tolist() == [True, True]
    assert np.all(np.isfinite(external_mean))


def test_jax_tris_mean_concatenates_the_two_responses():
    import jax.numpy as jnp

    sky = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    response_0 = jnp.array([[1.0, 0.0, 0.0]])
    response_1 = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    offset_0 = jnp.zeros(1)
    offset_1 = jnp.zeros(2)
    level = jnp.array([0.5, 0.25])
    result = np.asarray(
        _tris_mean(sky, response_0, response_1, offset_0, offset_1, level)
    )
    np.testing.assert_allclose(result, [1.0, 5.0, 6.0])



def test_heldout_deltas_gate_a_nonconverged_refit():
    rows = [
        {"train": "LWA", "heldout": "ARCADE", "variant": "no_rsb", "passed": True,
         "log_predictive_density": -1.0, "mcse": 0.1, "divergences": 0},
        {"train": "LWA", "heldout": "ARCADE", "variant": "rsb", "passed": True,
         "log_predictive_density": -0.5, "mcse": 0.2, "divergences": 0},
        {"train": "ARCADE", "heldout": "LWA", "variant": "no_rsb", "passed": True,
         "log_predictive_density": -2.0, "mcse": 0.1, "divergences": 0},
        {"train": "ARCADE", "heldout": "LWA", "variant": "rsb", "passed": False,
         "log_predictive_density": -1.5, "mcse": 0.2, "divergences": 54},
    ]
    deltas = {entry["train"]: entry for entry in heldout_deltas(rows)}
    assert deltas["LWA"]["valid"] is True
    assert deltas["LWA"]["delta_M1_minus_M0"] == pytest.approx(0.5)
    assert deltas["ARCADE"]["valid"] is False
    assert deltas["ARCADE"]["delta_M1_minus_M0"] is None
    assert deltas["ARCADE"]["status"] == "invalid_nonconverged_refit"
    assert deltas["ARCADE"]["invalid_refits"] == ["rsb"]


def test_heldout_deltas_mark_a_missing_refit():
    rows = [
        {"train": "LWA", "heldout": "ARCADE", "variant": "no_rsb", "passed": True,
         "log_predictive_density": -1.0, "mcse": 0.1},
    ]
    deltas = {entry["train"]: entry for entry in heldout_deltas(rows)}
    assert deltas["LWA"]["valid"] is False
    assert deltas["LWA"]["status"] == "missing_refit"
    assert deltas["LWA"]["missing_refits"] == ["rsb"]



def test_repair_positive_template_moves_the_monopole_off_the_support_failure():
    import numpy as np

    bundle = {"template_k": np.array([10.0, 20.0, 30.0])}
    repaired = _repair_positive_template(bundle, -50.0, None, None, False)
    assert repaired > -10.0
    assert np.all(bundle["template_k"] + repaired > 0.0)
    # An admissible monopole is left alone.
    assert _repair_positive_template(bundle, 1.0, None, None, False) == 1.0


def test_repair_positive_template_accounts_for_the_rsb_shift():
    import numpy as np

    bundle = {"template_k": np.array([10.0, 20.0, 30.0])}
    repaired = _repair_positive_template(bundle, -50.0, 1.0, -2.6, True)
    background = 1.0 * 0.408 ** -2.6
    assert np.all(bundle["template_k"] + repaired - background > 0.0)
