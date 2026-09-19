"""Property tests for the P0 numerical budget.

The budget's deterministic defects were: a flat-prefix alm truncation that
ignored healpy's m-major packing, a Gaussian log-likelihood "delta" that added
back the constants a difference cancels, a 0.000 K row selector hard-coded to
index 0 of one frequency, and a parameter shift compared with the prior width
instead of the posterior SD.  These tests assert the properties those defects
violated, not the spelling of the repair.
"""

import numpy as np
import pytest

from examples.inference.tris_audit_report import linearized_region_shift
from examples.inference.tris_p0_budget import (
    adjacent_budget,
    alm_ell,
    data_weighted_likelihood,
    lmax_from_alm_size,
    passes_budget,
    truncate_alm,
    two_frequency_budget,
    vector_budget,
    zero_sigma_row_budget,
)


def packed_alm(lmax, seed=0):
    rng = np.random.default_rng(seed)
    size = (lmax + 1) * (lmax + 2) // 2
    return rng.normal(size=size) + 1j * rng.normal(size=size)


def test_lmax_from_alm_size_round_trips():
    for lmax in (0, 1, 2, 3, 47, 383):
        size = (lmax + 1) * (lmax + 2) // 2
        assert lmax_from_alm_size(size) == lmax
    with pytest.raises(ValueError):
        lmax_from_alm_size(9)


def test_truncate_alm_keeps_only_low_ell_and_preserves_values():
    pytest.importorskip("healpy")
    lmax, cutoff = 383, 47
    alm = packed_alm(lmax)
    ell = alm_ell(alm)
    out = truncate_alm(alm, cutoff)
    kept = ell <= cutoff
    assert np.all(out[~kept] == 0.0)
    np.testing.assert_array_equal(out[kept], alm[kept])
    # The old flat-prefix truncation of an all-ones lmax-383 array kept 1008
    # coefficients with ell > cutoff; the survivor count must be exact.
    assert int(np.count_nonzero(out)) == int(np.count_nonzero(kept))


def test_truncate_alm_single_high_and_low_modes():
    pytest.importorskip("healpy")
    lmax = 383
    ell = alm_ell(np.zeros((lmax + 1) * (lmax + 2) // 2, dtype=complex))
    high = int(np.flatnonzero(ell > 47)[0])
    low = int(np.flatnonzero(ell <= 47)[-1])
    for index, survives in ((high, False), (low, True)):
        alm = np.zeros(ell.size, dtype=complex)
        alm[index] = 3.0 + 4.0j
        out = truncate_alm(alm, 47)
        if survives:
            assert out[index] == alm[index]
            assert int(np.count_nonzero(out)) == 1
        else:
            assert out[index] == 0.0
            assert int(np.count_nonzero(out)) == 0


def test_truncate_alm_is_a_copy_and_an_exact_no_op_above_the_source_lmax():
    pytest.importorskip("healpy")
    alm = packed_alm(47)
    before = alm.copy()
    truncate_alm(alm, 10)
    np.testing.assert_array_equal(alm, before)
    np.testing.assert_array_equal(truncate_alm(alm, 95), alm)


def test_identical_predictions_have_exactly_zero_likelihood_delta():
    operator = np.arange(12, dtype=float).reshape(3, 4) / 10.0
    sky = np.linspace(1.0, 2.0, 4)
    sigma = np.linspace(0.1, 0.3, 3)
    report = vector_budget(operator, operator, sky, sigma)
    assert report["gaussian_log_likelihood_delta"] == 0.0
    assert report["delta_chi_square"] == 0.0
    assert report["prediction_rms_k"] == 0.0


def test_gaussian_log_likelihood_delta_is_a_difference_of_two_likelihoods():
    control = np.zeros((3, 4))
    trial = np.zeros((3, 4))
    trial[0, 0] = 1.0
    report = vector_budget(control, trial, np.ones(4), np.full(3, 0.5))
    np.testing.assert_allclose(report["gaussian_log_likelihood_delta"], -2.0)
    assert report["gaussian_log_likelihood_delta"] < 0.0


def test_two_frequency_budget_sums_the_documented_quantities():
    rng = np.random.default_rng(2)
    control = rng.normal(size=(4, 3))
    trial = control + 0.01
    first = vector_budget(control, trial, np.ones(3), np.full(4, 0.5))
    second = vector_budget(control, trial, np.full(3, 2.0), np.full(4, 0.25))
    combined = two_frequency_budget(
        control, trial, np.ones(3), np.full(4, 0.5), np.full(3, 2.0), np.full(4, 0.25)
    )
    np.testing.assert_allclose(
        combined["combined_delta_chi_square"],
        first["delta_chi_square"] + second["delta_chi_square"],
    )


def test_data_weighted_likelihood_is_zero_for_equal_predictions():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(5, 4))
    data = rng.normal(size=5)
    sky = rng.normal(size=4)
    assert data_weighted_likelihood(a, a, data, sky, np.full(5, 0.2)) == 0.0


def test_passes_budget_thresholds_are_exclusive_upper_bounds():
    assert passes_budget({"prediction_rms_sigma": 0.09, "prediction_max_abs_sigma": 0.29})
    assert not passes_budget({"prediction_rms_sigma": 0.1, "prediction_max_abs_sigma": 0.29})
    assert not passes_budget({"prediction_rms_sigma": 0.09, "prediction_max_abs_sigma": 0.3})


def synthetic_payload(seed=11):
    rng = np.random.default_rng(seed)
    payload = {"frequency_mhz": np.array([600.5, 817.8])}
    for index in range(2):
        payload[f"operator_{index}"] = rng.normal(size=(6, 4))
        payload[f"data_k_{index}"] = rng.normal(size=6)
        payload[f"sigma_k_{index}"] = np.linspace(0.1, 0.3, 6)
        payload[f"prior_k_{index}"] = rng.normal(size=4)
    return payload


def test_zero_sigma_row_budget_drops_the_named_row_at_each_frequency():
    payload = synthetic_payload()
    report = zero_sigma_row_budget(payload, rows={0: 3, 1: 5})
    assert report["per_frequency"]["0"]["row_index"] == 3
    assert report["per_frequency"]["1"]["row_index"] == 5
    assert report["per_frequency"]["0"]["frequency_mhz"] == 600.5
    assert report["per_frequency"]["1"]["frequency_mhz"] == 817.8
    # The reported sigma is the bundle sigma at the named row, not at row 0.
    assert report["per_frequency"]["0"]["sigma_k"] == payload["sigma_k_0"][3]
    assert report["per_frequency"]["1"]["sigma_k"] == payload["sigma_k_1"][5]
    combined = (
        report["per_frequency"]["0"]["delta_chi_square"]
        + report["per_frequency"]["1"]["delta_chi_square"]
    )
    assert report["delta_chi_square"] == pytest.approx(combined)
    assert report["chi_square_keep"] == pytest.approx(
        report["per_frequency"]["0"]["chi_square_keep"]
        + report["per_frequency"]["1"]["chi_square_keep"]
    )


def test_zero_sigma_row_budget_rejects_an_out_of_range_row():
    with pytest.raises(ValueError):
        zero_sigma_row_budget(synthetic_payload(), rows={0: 99})


def test_adjacent_budget_compares_the_two_operators_it_is_given():
    rng = np.random.default_rng(5)
    control = rng.normal(size=(4, 3))
    trial = control + 1e-3
    sky_600 = rng.normal(size=3)
    sky_820 = rng.normal(size=3)
    entry = adjacent_budget(
        control,
        trial,
        sky_600=sky_600,
        sigma_600=np.full(4, 0.5),
        data_600=rng.normal(size=4),
        sky_820=sky_820,
        sigma_820=np.full(4, 0.5),
        data_820=rng.normal(size=4),
        template=rng.uniform(5.0, 50.0, size=3),
        region=np.array([0, 1, 2]),
    )
    direct = vector_budget(control, trial, sky_600, np.full(4, 0.5))
    assert entry["600_5"]["prediction_rms_sigma"] == pytest.approx(
        direct["prediction_rms_sigma"]
    )
    assert entry["data_weighted_log_likelihood_delta"] == pytest.approx(
        entry["data_weighted_log_likelihood_delta_600_5"]
        + entry["data_weighted_log_likelihood_delta_817_8"]
    )
    assert entry["budget_pass"] == (
        entry["budget_adjacent_pass"] and entry["budget_posterior_sd_pass"]
    )
    # The shift is tested against the posterior SD, and the prior cannot be
    # tighter than the posterior under Gaussian updating.
    shift = np.asarray(entry["shift"]["delta_amplitude"] + entry["shift"]["delta_beta"])
    posterior_sd = np.asarray(entry["shift"]["posterior_sd"])
    np.testing.assert_allclose(
        entry["shift"]["max_delta_over_posterior_sd"],
        float(np.max(np.abs(shift / posterior_sd))),
    )
    assert np.all(posterior_sd <= np.asarray(entry["shift"]["prior_width"]) + 1e-12)


def test_linearized_region_shift_reports_posterior_sd_not_prior_width():
    rng = np.random.default_rng(13)
    a = rng.normal(size=(10, 6))
    sigma = np.full(10, 0.1)
    template = rng.uniform(10.0, 50.0, size=6)
    region = np.array([0, 0, 1, 1, 2, 2])
    delta = rng.normal(size=10)
    report = linearized_region_shift(a, sigma, template, region, 600.5, delta)
    posterior_sd = np.asarray(report["posterior_sd"])
    prior_width = np.asarray(report["prior_width"])
    assert posterior_sd.shape == (6,)
    assert np.all(posterior_sd > 0.0)
    assert np.all(posterior_sd <= prior_width + 1e-12)
    shift = np.asarray(report["delta_amplitude"] + report["delta_beta"])
    np.testing.assert_allclose(
        report["max_delta_over_posterior_sd"],
        float(np.max(np.abs(shift / posterior_sd))),
    )
