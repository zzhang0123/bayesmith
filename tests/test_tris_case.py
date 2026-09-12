"""Independent checks for real-data map inference; no downloaded data needed."""

import numpy as np
import pytest

from examples.inference.tris_maps import compress_map


def test_failed_chain_report_renders_unavailable_diagnostics(monkeypatch):
    """A stuck real-data run must still display its failed-diagnostics report."""
    from examples.inference import presentation, tris_presentation

    monkeypatch.setattr(presentation, "dag_svg", lambda *args, **kwargs: "<svg></svg>")
    summary = {
        "frequency_mhz": 600.5, "residual_rms_k": 1., "chi_square_per_observation": 100.,
        "ppc_tail_probability": 0., "ppc_replicates": 8,
    }
    report = {
        "checks": {"chain_diagnostics": {"passed": False, "divergences": 2,
                    "sites": {"amplitude": {"r_hat": None, "ess": 0.},
                              "beta": {"r_hat": 1.4, "ess": 2.}}}},
        "data_manifest": {"maps": [{"map_pixels": 768}], "archive_url": "https://example.org/archive",
                          "upstream_model": {"url": "https://example.org/model", "revision": "abc"}},
        "method": "nuts", "seed": 0, "warmup": 4, "chain_shape": [2, 4],
        "frequencies": [summary], "preflight": [], "execution": {},
        "dag": [], "model_source": "recorded-source-only", "model_adequacy": "mismatch",
        "parameters": [{"name": f"zero_level_K[{i}]", "parameter": "zero_level_K", "region": "test",
                        "mean": .1, "lower": 0., "upper": .2} for i in range(2)],
    }
    rendered = tris_presentation.render_case(report, "tris_haslam")
    assert "unavailable" in rendered
    assert "Sampling: diagnostics failed" in rendered
    assert "采样：诊断未通过" in rendered
    assert "recorded-source-only" in rendered
    assert "Sampling: converged" not in rendered


def reconstruction(a, data, sigma, prior, prior_sigma):
    precision = a.T @ np.diag(sigma**-2) @ a + np.diag(prior_sigma**-2)
    cov = np.linalg.inv(precision)
    m = np.linalg.solve(precision, a.T @ (data / sigma**2) + prior / prior_sigma**2)
    return compress_map(m, prior, a, sigma, cov)


def test_map_likelihood_preserves_correlated_noise_response_and_prior_invariance():
    a = np.array([[1.0, 0.4, 0.0, 0.1], [0.1, 0.8, 0.5, 0.0], [0.3, 0.2, 0.9, 0.2]])
    data = np.array([2.2, 1.1, 1.8])
    sigma = np.array([0.1, 0.2, 0.15])
    sky0, sky1 = np.array([1.0, 2.0, 1.0, 0.5]), np.array([1.3, 1.6, 0.8, 0.4])
    ring_delta = -0.5 * (
        np.sum(((data - a @ sky1 - 0.12) / sigma) ** 2)
        - np.sum(((data - a @ sky0 + 0.03) / sigma) ** 2)
    )
    for prior, width in [
        (np.zeros(4), np.ones(4)),
        (np.array([5.0, -2.0, 3.0, 1.0]), np.array([0.2, 3.0, 0.4, 2.0])),
    ]:
        c = reconstruction(a, data, sigma, prior, width)
        r0 = c.data - c.response @ sky0 + 0.03 * c.offset_response
        r1 = c.data - c.response @ sky1 - 0.12 * c.offset_response
        assert -0.5 * (r1 @ r1 - r0 @ r0) == pytest.approx(ring_delta, rel=1e-10)
        noise_cov = (c.weights * sigma) @ (c.weights * sigma).T
        assert np.max(np.abs(noise_cov - np.diag(np.diag(noise_cov)))) > 0.001
        np.testing.assert_allclose(c.noise_sigma**2, np.diag(noise_cov), rtol=1e-12)


def test_null_modes_do_not_create_observations():
    a = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    c = reconstruction(a, np.array([1.0, 2.0]), np.ones(2), np.zeros(3), np.ones(3))
    assert c.data.shape == (1,)
    np.testing.assert_allclose(c.response[:, 1:], 0, atol=1e-15)


@pytest.mark.parametrize("sigma", [np.array([0.0, 1.0]), np.array([np.nan, 1.0])])
def test_invalid_uncertainty_is_rejected(sigma):
    with pytest.raises(ValueError):
        compress_map(np.ones(2), np.ones(2), np.eye(2), sigma, np.eye(2))


def test_haslam_spectrum_and_asymmetric_sky_correction_match_independent_equation():
    import jax

    from examples.inference.tris_sky import map_mean, spectral_scaling, zero_levels

    with jax.enable_x64(True):
        amp = np.array([1.2, 0.8, 1.1])
        beta = np.array([-2.5, -3.0, -2.8])
        frequencies = np.array([600.5, 817.8])
        design = np.array([[20.0, 5.0, 1.0], [1.0, 3.0, 10.0]])
        for z820 in [-1.0, 1.0]:
            correction = np.asarray(zero_levels(np.array([1.0, z820])))
            np.testing.assert_allclose(
                correction, [0.066, -0.300 if z820 < 0 else 0.430]
            )
            scale = spectral_scaling(amp, beta, frequencies)
            result = map_mean(
                scale,
                correction,
                design,
                np.array([2.71, 2.706]),
                np.ones(2),
                np.array([0, 1]),
            )
            expected = []
            for i, nu in enumerate(frequencies):
                sky = sum(
                    design[i, r] * amp[r] * np.exp(beta[r] * np.log(nu / 408.0))
                    for r in range(3)
                )
                expected.append(sky + [2.71, 2.706][i] - correction[i])
            np.testing.assert_allclose(result, expected, rtol=1e-13)
            # In the zero-noise limit, d + correction = true sky: positive
            # uncertainty is +.430 K, not its accidentally reflected -.430 K.
            np.testing.assert_allclose(
                np.asarray(result) + correction,
                np.array(expected) + correction,
                rtol=1e-13,
            )


def test_graph_likelihood_has_unit_noise_and_all_parameter_dependencies():
    import jax

    from bayesmith import trace
    from bayesmith.graph.evaluate import log_joint
    from examples.inference.tris_sky import model

    with jax.enable_x64(True):
        design = np.array([[20.0, 5.0, 1.0], [1.0, 3.0, 10.0], [2.0, 7.0, 4.0]])
        cmb = np.array([2.71, 2.706, 2.71])
        response = np.array([2.0, 3.0, 2.0])
        channel = np.array([0, 1, 0])
        frequencies = np.array([600.5, 817.8])
        data = np.array([5.0, 6.0, 7.0])
        graph = trace(model, design, cmb, response, channel, frequencies, data)
        for amp, beta, z in [
            ([1.0, 1.2, 0.9], [-2.5, -3.0, -2.7], [0.2, -0.4]),
            ([1.3, 0.8, 1.1], [-2.7, -2.8, -3.0], [-0.3, 0.6]),
        ]:
            amp, beta, z = map(np.array, (amp, beta, z))
            correction = z * [0.066, 0.300 if z[1] < 0 else 0.430]
            means = np.array(
                [
                    sum(
                        design[i, r] * amp[r] * (frequencies[c] / 408.0) ** beta[r]
                        for r in range(3)
                    )
                    + cmb[i]
                    - response[i] * correction[c]
                    for i, c in enumerate(channel)
                ]
            )
            independent = (
                -0.5 * np.sum((data - means) ** 2)
                - 1.5 * np.log(2 * np.pi)
                - 3 * np.log(2.8)
                - 3 * np.log(2.5)
                - 0.5 * np.sum(z**2)
                - np.log(2 * np.pi)
            )
            actual = float(
                log_joint(graph, {"amplitude": amp, "beta": beta, "zero_standard": z})
            )
            assert actual == pytest.approx(independent, abs=1e-10)


def test_posterior_maps_preserve_regions_reference_cmb_and_uncertainty():
    from examples.inference.tris_products import posterior_sky_products

    bundle = {"region": np.array([2, 0, 1, 0]),
              "template_k": np.array([20., 30., -2., 50.]), "reference_cmb_k": 2.71}
    amplitude = np.array([[1., 2., 3.], [2., 3., 4.], [3., 4., 5.]])
    beta = np.array([[-2., -3., -4.], [-3., -4., -5.], [-4., -5., -6.]])
    products = posterior_sky_products(bundle, amplitude, beta)
    np.testing.assert_allclose(products["amplitude_mean"], [4., 2., 3., 2.])
    np.testing.assert_allclose(products["beta_mean"], [-5., -3., -4., -3.])
    np.testing.assert_allclose(products["amplitude_sd"], 1.)
    np.testing.assert_allclose(products["haslam_reference_k"], [22.71, 32.71, .71, 52.71])
    np.testing.assert_allclose(products["haslam_mean"], [82.71, 62.71, -3.29, 102.71])
    np.testing.assert_allclose(products["haslam_sd"], [20., 30., 2., 50.])
    np.testing.assert_allclose(products["haslam_change_mean"], [60., 30., -4., 50.])
    assert np.all(products["haslam_lower"] <= products["haslam_upper"])
    # Spectral slope and instrument corrections cannot alter the 408-MHz sky.
    changed = posterior_sky_products({**bundle, "zero_level_K": [100., 200.]}, amplitude, beta - 2)
    np.testing.assert_array_equal(products["haslam_mean"], changed["haslam_mean"])
    unity = posterior_sky_products(bundle, np.ones_like(amplitude), beta)
    np.testing.assert_allclose(unity["haslam_mean"], products["haslam_reference_k"])
    np.testing.assert_allclose(unity["haslam_sd"], 0., atol=1e-14)


@pytest.mark.parametrize("bad_region", [np.array([-1, 0]), np.array([0, 3]), np.array([0., 1.])])
def test_posterior_maps_reject_invalid_pixel_region_lookup(bad_region):
    from examples.inference.tris_products import posterior_sky_products

    with pytest.raises(ValueError, match="Region indices"):
        posterior_sky_products({"region": bad_region, "template_k": np.ones(2), "reference_cmb_k": 2.71},
                               np.ones((2, 3)), np.ones((2, 3)))


def test_real_setup_displays_mixed_priors_and_recorded_chain_initialization():
    from examples.inference.case_panels import diagnostics_panel, sampling_panel

    report = {
        "case": "tris_haslam", "kind": "real_observations", "chain_shape": [2, 8],
        "requested_draws_per_chain": 8, "warmup": 5,
        "blocking": {"priors": [
            {"latent": "amplitude", "family": "Uniform", "coordinates": 3,
             "bounds": {"lower": [.2] * 3, "upper": [3.] * 3}},
            {"latent": "zero_standard", "family": "Normal", "coordinates": 2},
        ], "executed": {"blocks": [{"latents": ["amplitude", "beta", "zero_standard"], "method": "nuts"}]}},
        "preflight": [{"code": "block_0_jeffreys", "conclusion": "nonflat",
                       "scope": {"name": "actual_joint_scope"}, "measurements": {"half_logdet_change": -.8}, "grounds": []}],
        "execution": {"stopping_policy": {"mode": "fixed"}, "initial_values": {"zero_standard": [[.1, .2], [-.2, -.3]]},
                      "sampling": {"stop_reason": "fixed_budget", "divergences": 0,
                                   "checkpoints": [{"draws_per_chain": 8, "ess_min": 6., "rhat_max": 1.3, "passed": False}]}},
    }
    diagnostic = diagnostics_panel(report, "tris_haslam", {})
    assert "Normal(0, 1)" in diagnostic and "actual_joint_scope" in diagnostic
    assert "Root parameters use finite Uniform" not in diagnostic
    assert "Non-flat" in diagnostic
    sampling = sampling_panel(report, "tris_haslam", {}, show_prediction=False)
    assert "[0.1, 0.2], [-0.2, -0.3]" in sampling
    assert "fixed_budget" in sampling and "Recorded diagnostic checks" in sampling
    assert "Failed" in sampling
    assert "inference.png" not in sampling  # A real case has no simulated recovery product.
