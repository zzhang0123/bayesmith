"""Batch D/D2 helpers: held-out extraction, reweighting and extension bases."""

import numpy as np
import pytest
from scipy.stats import norm

from examples.inference.tris_batch_d import delta_rows, heldout_rows
from examples.inference.tris_batch_d2 import (
    equal_area_bins,
    importance,
    log_normal,
    raised_cosine_basis,
    region_fields,
    reweighted_summary,
    smooth_fields,
    zero820_symmetric_ratio,
)
from tests.test_tris_forward_baseline import tiny_bundle


def test_log_normal_matches_scipy():
    x = np.array([-2.0, -0.5, 0.0, 1.5])
    for sigma in (0.5, 1.0, 3.0):
        np.testing.assert_allclose(log_normal(x, sigma), norm.logpdf(x, 0.0, sigma))


def test_importance_weights_sum_to_one_and_full_ess_when_flat():
    rng = np.random.default_rng(0)
    weights, ess = importance(rng.normal(size=1000))
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12)
    np.testing.assert_allclose(ess, 1.0 / np.sum(weights**2), rtol=1e-12)
    _flat, flat_ess = importance(np.zeros(500))
    np.testing.assert_allclose(flat_ess, 500.0, rtol=1e-12)


def test_reweighting_widens_a_narrow_sample_toward_a_wider_prior():
    rng = np.random.default_rng(1)
    values = rng.normal(size=(4000, 1))
    weights, ess = importance(
        log_normal(values[:, 0], 3.0) - log_normal(values[:, 0], 1.0)
    )
    summary = reweighted_summary(weights, values)
    assert float(summary["sd"][0]) > 1.2
    assert float(summary["sd"][0]) < 3.0
    assert ess > 100


def test_zero820_ratio_is_finite_on_both_sign_branches():
    z = np.array([-2.0, -0.5, 0.5, 2.0])
    ratio = zero820_symmetric_ratio(z, 0.365)
    assert ratio.shape == (4,)
    assert np.all(np.isfinite(ratio))


def test_equal_area_bins_are_equal_area_on_the_sphere():
    # Equal area means equal counts for a sample uniform on the sphere.  A
    # sample uniform in latitude is NOT that test: it deliberately over-weights
    # the wide polar bins.
    rng = np.random.default_rng(3)
    cosine = rng.uniform(-1.0, 1.0, size=200000)
    latitude = 90.0 - np.rad2deg(np.arccos(cosine))
    for regions in (3, 6):
        codes = equal_area_bins(latitude, regions)
        assert set(codes.tolist()) == set(range(regions))
        counts = np.bincount(codes, minlength=regions)
        assert counts.min() > 0.95 * counts.max()


def test_raised_cosine_basis_is_bounded_and_non_negative():
    latitude = np.linspace(-90.0, 90.0, 301)
    basis = raised_cosine_basis(latitude, functions=5)
    assert basis.shape == (301, 5)
    assert np.all(basis >= 0.0)
    assert np.all(basis <= 1.0 + 1e-12)
    # Each basis function peaks somewhere, so none is identically zero.
    assert np.all(basis.max(axis=0) > 0.9)


def test_extension_field_decoders_agree_at_constant_coefficients():
    bundle = tiny_bundle()
    region = np.array([0, 1, 2])
    amplitude, beta = region_fields(
        bundle, {"log_amplitude": np.log([1.5, 1.5, 1.5]), "beta": [-2.7, -2.7, -2.7]}, region
    )
    np.testing.assert_allclose(amplitude, 1.5)
    np.testing.assert_allclose(beta, -2.7)
    basis = np.eye(3)
    smooth_amplitude, smooth_beta = smooth_fields(
        bundle,
        {"log_amplitude": np.log([1.5, 1.5, 1.5]), "beta": [-2.7, -2.7, -2.7]},
        basis,
    )
    np.testing.assert_allclose(smooth_amplitude, 1.5)
    np.testing.assert_allclose(smooth_beta, -2.7)


def _comparison():
    return {
        "heldout_scores": [
            {
                "train": "LWA",
                "heldout": "ARCADE",
                "models": {
                    "no_rsb": {"log_predictive_density": -14.8, "mcse": 0.0},
                    "rsb": {"log_predictive_density": -14.775, "mcse": 0.0005},
                },
                "delta_M1_minus_M0": 0.0201,
                "delta_mcse": 0.0005,
            }
        ]
    }


def test_heldout_rows_flatten_both_models():
    rows = heldout_rows(_comparison())
    assert [row["variant"] for row in rows] == ["no_rsb", "rsb"]
    assert rows[1]["log_predictive_density"] == pytest.approx(-14.775)
    deltas = delta_rows(_comparison())
    assert deltas[0]["delta_M1_minus_M0"] == pytest.approx(0.0201)
