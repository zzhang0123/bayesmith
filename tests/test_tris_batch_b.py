"""Batch B comparison helpers: posterior shift, residual rows, real-bundle parity."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest

from examples.inference.tris_batch_b import (
    frequency_rows,
    log_joint_parity,
    parameter_comparison,
    posterior_mean_values,
)
from tests.test_tris_forward_baseline import tiny_bundle, tiny_external
from tests.test_tris_forward_baseline import values as tiny_values


def synthetic_draws(seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    draws = {
        "amplitude": rng.normal(size=(2, 50, 3)),
        "beta": rng.normal(size=(2, 50, 3)),
        "zero_standard": rng.normal(size=(2, 50, 2)),
        "haslam_monopole_K": rng.normal(size=(2, 50)),
        "calibration_standard": rng.normal(size=(2, 50, 2)),
        "rsb_amplitude": rng.normal(size=(2, 50)),
        "rsb_beta": rng.normal(size=(2, 50)),
    }
    if shift:
        draws["amplitude"] = draws["amplitude"] + shift
    return draws


def test_posterior_mean_values_flattens_the_chain_and_draw_axes():
    draws = synthetic_draws()
    chosen = posterior_mean_values(draws, True)
    assert np.shape(chosen["amplitude"]) == (3,)
    assert np.shape(chosen["zero_standard"]) == (2,)
    assert np.ndim(chosen["rsb_amplitude"]) == 0
    np.testing.assert_allclose(
        chosen["amplitude"], draws["amplitude"].mean(axis=(0, 1))
    )
    without = posterior_mean_values(draws, False)
    assert "rsb_amplitude" not in without


def test_parameter_comparison_delta_and_combined_mcse_are_consistent():
    fixed = synthetic_draws(seed=1)
    audited = synthetic_draws(seed=2, shift=0.3)
    rows = {row["parameter"]: row for row in parameter_comparison(fixed, audited)}
    assert set(rows) == set(fixed)
    row = rows["amplitude"]
    delta = np.asarray(row["delta_mean"])
    np.testing.assert_allclose(delta, audited["amplitude"].mean(axis=(0, 1)) - fixed["amplitude"].mean(axis=(0, 1)))
    combined = np.sqrt(
        np.asarray(row["mcse_fixed"]) ** 2 + np.asarray(row["mcse_audited"]) ** 2
    )
    np.testing.assert_allclose(
        np.asarray(row["delta_over_combined_mcse"]), delta / combined, rtol=1e-12
    )
    # The shift is real only in the sense that scaling it scales the ratio.
    doubled = {
        row["parameter"]: row
        for row in parameter_comparison(fixed, synthetic_draws(seed=2, shift=0.6))
    }["amplitude"]
    np.testing.assert_allclose(
        np.asarray(doubled["delta_mean"]) - delta,
        np.full(3, 0.3),
        atol=1e-12,
    )


def test_frequency_rows_select_only_the_adequacy_fields():
    report = {
        "frequencies": [
            {
                "frequency_mhz": 600.5,
                "residual_rms_k": 0.69,
                "chi_square_per_observation": 2292.3,
                "ppc_tail_probability": 0.0,
                "adequacy": "mismatch",
                "residual_k": [1, 2, 3],
            }
        ]
    }
    assert frequency_rows(report) == [
        {
            "frequency_mhz": 600.5,
            "residual_rms_k": 0.69,
            "chi_square_per_observation": 2292.3,
            "ppc_tail_probability": 0.0,
            "adequacy": "mismatch",
        }
    ]


@pytest.mark.parametrize("variant", ["no_rsb", "rsb"])
def test_log_joint_parity_on_a_bundle_closes(variant):
    with jax.enable_x64(True):
        common = SimpleNamespace(tris_bundle=tiny_bundle(), external=tiny_external())
        chosen = tiny_values(variant == "rsb")
        parity = log_joint_parity(common, variant, chosen)
        assert parity["support_valid"] is True
        assert parity["difference"] == pytest.approx(0.0, abs=1e-9)
        assert parity["log_joint"] == pytest.approx(
            parity["log_prior"]
            + parity["log_likelihood_tris"]
            + parity["log_likelihood_external"],
            abs=1e-12,
        )
