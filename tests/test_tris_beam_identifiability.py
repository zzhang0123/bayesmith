"""Beam-width identifiability helpers: cut deformation, whitening and Jacobian."""

import numpy as np
import pytest

from examples.inference.tris_beam_identifiability import (
    analyze,
    noise_sigma,
    parameter_vector,
    resample_cuts,
    ring_prediction,
    set_parameter,
    sky_field,
    step_stability,
    support_diagnostics,
)
from examples.inference.tris_forward_baseline import zero_levels
from tests.test_tris_forward_baseline import tiny_bundle
from tests.test_tris_forward_baseline import values as tiny_values


def _ramp_cuts():
    angle = np.linspace(0.0, 180.0, 181)
    return angle, -angle, -angle


def _tiny_operators():
    """Five (2, 3) sky-to-sample matrices with a nonzero width derivative."""
    center = np.asarray(tiny_bundle()["response_0"], dtype=float)
    return {
        "center": center,
        "e_plus": 1.10 * center,
        "e_minus": 0.90 * center,
        "h_plus": 1.05 * center,
        "h_minus": 0.95 * center,
    }


def _tiny_sigma():
    return np.array([0.01, 0.02, 0.03, 0.04])


def test_resample_cuts_is_the_identity_at_zero_width():
    angle, e_plane, h_plane = _ramp_cuts()
    _angle, h_out, e_out = resample_cuts(angle, e_plane, h_plane, 0.0, 0.0)
    np.testing.assert_allclose(e_out, e_plane)
    np.testing.assert_allclose(h_out, h_plane)


def test_resample_cuts_narrows_a_plane_under_positive_log_width():
    angle, e_plane, h_plane = _ramp_cuts()
    _angle, h_out, e_out = resample_cuts(angle, e_plane, h_plane, np.log(2.0), 0.0)
    # A ramp makes the deformation exact: e(a / 2) = -a / 2.
    np.testing.assert_allclose(e_out, -angle / 2.0)
    # The untouched plane is unchanged.
    np.testing.assert_allclose(h_out, h_plane)


def test_parameter_vector_packing_order():
    chosen = {
        "amplitude": np.array([1.0, 1.1, 1.2]),
        "beta": np.array([-2.7, -2.8, -2.9]),
        "zero_standard": np.array([0.1, 0.2]),
        "haslam_monopole_K": np.asarray(0.3),
    }
    np.testing.assert_allclose(
        parameter_vector(chosen), [1.0, 1.1, 1.2, -2.7, -2.8, -2.9, 0.1, 0.2, 0.3]
    )


def test_set_parameter_shifts_one_coordinate_and_leaves_the_input():
    chosen = {
        "amplitude": np.array([1.0, 1.1, 1.2]),
        "beta": np.array([-2.7, -2.8, -2.9]),
        "zero_standard": np.array([0.1, 0.2]),
        "haslam_monopole_K": np.asarray(0.3),
    }
    shifted = set_parameter(chosen, 4, 0.5)
    np.testing.assert_allclose(shifted["beta"], [-2.7, -2.3, -2.9])
    np.testing.assert_allclose(chosen["beta"], [-2.7, -2.8, -2.9])
    scalar_shift = set_parameter(chosen, 8, 1.0)
    assert float(np.asarray(scalar_shift["haslam_monopole_K"])) == pytest.approx(1.3)
    with pytest.raises(IndexError):
        set_parameter(chosen, 9, 1.0)


def test_sky_field_matches_the_hand_expression():
    bundle = tiny_bundle()
    chosen = tiny_values(False)
    frequency = 600.5
    field = sky_field(bundle, chosen, frequency)
    region = np.asarray(bundle["region"], dtype=int)
    template = np.asarray(bundle["template_k"], dtype=float)
    amplitude = np.asarray(chosen["amplitude"])
    beta = np.asarray(chosen["beta"])
    expected = (template + float(np.asarray(chosen["haslam_monopole_K"]))) * (
        amplitude[region] * (frequency / 408.0) ** beta[region]
    )
    np.testing.assert_allclose(field, expected)


def test_ring_prediction_concatenates_the_two_rings():
    bundle = tiny_bundle()
    chosen = tiny_values(False)
    # One shared matrix for both rings, exactly as the prepared bundle stores it.
    operator = np.asarray(bundle["response_0"], dtype=float)
    result = ring_prediction(bundle, chosen, operator)
    assert result.shape == (4,)
    levels = zero_levels(chosen["zero_standard"])
    first = operator @ sky_field(bundle, chosen, 600.5) - levels[0]
    second = operator @ sky_field(bundle, chosen, 817.8) - levels[1]
    np.testing.assert_allclose(result, np.concatenate([first, second]))


def test_noise_sigma_concatenates_the_two_rings_in_prediction_order():
    bundle = {"sigma_k_0": np.array([0.01, 0.02]), "sigma_k_1": np.array([0.03, 0.04])}
    np.testing.assert_allclose(noise_sigma(bundle), [0.01, 0.02, 0.03, 0.04])


def test_analyze_whitens_by_sigma_so_the_fisher_scales_as_one_over_sigma_squared():
    bundle, chosen = tiny_bundle(), tiny_values(False)
    operators = _tiny_operators()
    sigma = _tiny_sigma()
    base = analyze(bundle, chosen, operators, sigma=sigma, width_prior_sd=0.05)
    doubled = analyze(bundle, chosen, operators, sigma=2.0 * sigma, width_prior_sd=0.05)
    # The scaled Jacobian is J / sigma * prior_sd, so its singular values are
    # exactly halved and the Fisher information is quartered.
    np.testing.assert_allclose(
        np.asarray(doubled["singular_values"]) * 2.0,
        np.asarray(base["singular_values"]),
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        np.asarray(doubled["fisher_eigenvalues"]) * 4.0,
        np.asarray(base["fisher_eigenvalues"]),
        rtol=1e-12,
        atol=1e-15,
    )
    # More noise cannot increase the posterior precision.
    assert np.all(
        np.asarray(doubled["posterior_over_prior_sd"])
        >= np.asarray(base["posterior_over_prior_sd"]) - 1e-12
    )


def test_analyze_rejects_a_sigma_of_the_wrong_length_or_sign():
    bundle, chosen = tiny_bundle(), tiny_values(False)
    with pytest.raises(ValueError):
        analyze(bundle, chosen, _tiny_operators(), sigma=np.ones(3))
    with pytest.raises(ValueError):
        analyze(bundle, chosen, _tiny_operators(), sigma=-_tiny_sigma())


def test_k_to_mk_rescale_is_invariant_with_the_temperature_priors():
    # A consistent mK analysis multiplies every kelvin quantity -- the template,
    # the monopole parameter, the zero-level conversion and the per-point sigma
    # -- by 1000, and scales the priors of the temperature-valued coordinates
    # with it.  The whitened, prior-scaled Jacobian is then unchanged, so no
    # posterior/prior ratio depends on the temperature unit.
    bundle = tiny_bundle()
    bundle["template_k"] = np.asarray(bundle["template_k"], dtype=float) * 1e-4
    chosen = tiny_values(False)
    chosen["haslam_monopole_K"] = np.asarray(2e-4)
    chosen["zero_standard"] = np.array([1e-4, -2e-4])
    operators = _tiny_operators()
    sigma = _tiny_sigma()
    scale = 1000.0
    scaled_bundle = dict(bundle)
    scaled_bundle["template_k"] = np.asarray(bundle["template_k"], dtype=float) * scale
    scaled_values = {name: np.array(value, dtype=float, copy=True) for name, value in chosen.items()}
    scaled_values["haslam_monopole_K"] = scaled_values["haslam_monopole_K"] * scale
    scaled_values["zero_standard"] = scaled_values["zero_standard"] * scale
    prior_scale = np.ones(11)
    prior_scale[6:9] = scale
    base = analyze(bundle, chosen, operators, sigma=sigma, width_prior_sd=0.05)
    scaled = analyze(
        scaled_bundle,
        scaled_values,
        operators,
        sigma=sigma * scale,
        width_prior_sd=0.05,
        prior_scale=prior_scale,
    )
    np.testing.assert_allclose(
        scaled["posterior_over_prior_sd"],
        base["posterior_over_prior_sd"],
        rtol=1e-7,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        scaled["singular_values"], base["singular_values"], rtol=1e-8, atol=1e-12
    )


def test_support_diagnostics_reports_the_zero_level_kink_distance():
    chosen = tiny_values(False)
    chosen["zero_standard"] = np.array([0.0, 0.25])
    support = support_diagnostics(chosen)
    assert support["zero_standard_1_kink_distance_prior_sd"] == pytest.approx(0.25)
    assert support["zero_level_kink_at_zero_standard_1"] is True
    assert support["amplitude_lower_margin_prior_sd"][0] == pytest.approx(
        (1.0 - 0.2) / 0.808290384, rel=1e-6
    )


def test_step_stability_is_tight_on_a_quadratic_prediction():
    bundle, chosen = tiny_bundle(), tiny_values(False)
    table = step_stability(
        bundle, chosen, _tiny_operators(), _tiny_sigma(), width_prior_sd=0.05
    )
    assert table["steps"] == ["1e-06", "1e-05", "1e-04"]
    assert table["max_relative_spread"] < 1e-6


def test_surrogate_cache_keys_do_not_merge_small_derivative_steps():
    from examples.inference.tris_beam_surrogate import key_of

    assert len({key_of(step, 0.) for step in (0., .0001, .0002, -.0001)}) == 4

    from examples.inference.tris_beam_surrogate import _box_passes

    table = {}
    for e, h in ((.01,.01),(.01,-.01),(-.01,.01),(-.01,-.01),(.005,0.)):
        budget = {"prediction_rms_sigma": .01, "prediction_max_abs_sigma": .02}
        table[key_of(e,h)] = {"log_width_e": e, "log_width_h": h,
                              "600_5": dict(budget), "817_8": dict(budget)}
    assert _box_passes(table, .01)
    table[key_of(.005,0.)]["600_5"]["prediction_rms_sigma"] = .2
    assert not _box_passes(table, .01)
