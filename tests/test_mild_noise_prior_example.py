"""Independent density and simulation checks for the noise-prior counterpart."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import truncnorm

from bayesmith import trace
from bayesmith.graph.evaluate import log_joint
from examples.inference import composed_process as demo
from examples.inference.common import simulate_fixed
from examples.inference.noise_scale_reference import known_mean_reference


def test_mild_prior_changes_only_normalized_noise_density_and_preserves_fixed_simulation():
    with jax.enable_x64(True):
        x = jnp.linspace(0, 1, 32, endpoint=False)
        k, _, response, background, gain = demo.design_matrices(x)
        args = (x, k, response, background, gain, jnp.full_like(x, 3.0))
        baseline, mild = (trace(model, *args) for model in (demo.model, demo.mild_model))
        values = {
            "power_amplitude": jnp.array(0.5), "instance": jnp.zeros(demo.MODES),
            "nonlinear": jnp.array([0.45, 0.09]), "background": jnp.array([3.0, 0.2]),
            "gain": jnp.array([0.08, -0.05]), "sigma_w": jnp.array(0.015),
        }
        difference = jax.jit(lambda sigma: log_joint(mild, {**values, "sigma_w": sigma})
                             - log_joint(baseline, {**values, "sigma_w": sigma}))
        law = truncnorm(-1.7, 4.0, loc=0.02, scale=0.01)
        for sigma in (0.004, 0.015, 0.04, 0.059):
            np.testing.assert_allclose(difference(sigma), law.logpdf(sigma) + np.log(0.057), atol=2e-11)
            np.testing.assert_allclose(jax.jit(jax.grad(difference))(sigma), -(sigma-0.02)/0.01**2, rtol=2e-11)
        for sigma in (0.002, 0.061):
            # NumPyro log_prob assumes valid inputs by default. The NUTS support
            # transform enforces the declared interval during inference.
            assert not demo.mild_white_noise_prior().support.check(jnp.array(sigma))
        first, _ = simulate_fixed(baseline, demo.model, values, jax.random.key(7))
        second, _ = simulate_fixed(mild, demo.mild_model, values, jax.random.key(7))
        np.testing.assert_array_equal(first, second)


def test_known_mean_scale_reference_matches_integrated_sigma_density():
    residual = np.array([0.01, -0.02, 0.005, 0.015, -0.01, 0.012])
    actual = known_mean_reference(1 + residual, np.ones(6), 0.015)
    energy = np.sum(residual**2)
    mode = np.sqrt(energy / residual.size)
    peak = -residual.size * np.log(mode) - energy / (2 * mode**2)

    def density(sigma):
        return np.exp(-residual.size * np.log(sigma) - energy / (2*sigma**2) - peak)

    normalizer = quad(density, 0.003, 0.06, epsabs=1e-13)[0]
    expected = [brentq(lambda sigma, p=p: quad(density, 0.003, sigma, epsabs=1e-13)[0] / normalizer - p,
                       0.003, 0.06, xtol=1e-14) for p in (0.005, 0.5, 0.995)]
    np.testing.assert_allclose(actual["uniform_sigma_posterior_quantiles"], expected, rtol=1e-10)


def test_comparison_reads_flat_archives_and_refuses_unmatched_data(tmp_path):
    import json

    from examples.inference.case_panels import noise_scale_panel, prior_variant_note
    from examples.inference.noise_scale_reference import compare_saved_runs

    report = {
        "seed": 0, "requested_draws_per_chain": 4, "warmup": 5,
        "chain_shape": [2, 4], "interval_mass": 0.99, "max_sd_ratio": 0.5,
        "passed": False,
        "parameters": [{"name": "sigma_w", "truth": 0.015, "mean": 0.014,
                        "posterior_sd": 0.001, "lower": 0.011, "upper": 0.0149,
                        "covered": False}],
        "checks": {"chain_diagnostics": {"passed": True, "divergences": 0}},
        "signal": {"x": [0, 1, 2], "data": [0.99, 1.02, 1.01], "truth": [1, 1, 1]},
        "variant": {"kind": "mild_noise_prior", "loc": 0.02, "scale": 0.01,
                    "support": [0.003, 0.06]},
    }
    chain = np.array([[0.012, 0.013, 0.014, 0.0148], [0.013, 0.014, 0.015, 0.016]])
    paths = [tmp_path / name for name in ("baseline", "mild")]
    for index, path in enumerate(paths):
        path.mkdir()
        (path / "result.json").write_text(json.dumps({**report, "variant": report["variant"] if index else None}))
        np.savez(path / "posterior.npz", sigma_w=chain.reshape(-1))
    comparison = compare_saved_runs(*paths)
    np.testing.assert_allclose(comparison["rows"][0]["chain_upper_995"], np.quantile(chain, 0.995, axis=1))
    assert comparison["rows"][0]["draws_above_truth"] == [0, 1]
    assert all(comparison["matched"].values())
    assert "2 chains × 4 retained" in noise_scale_panel({"noise_scale_comparison": comparison})
    assert "Jeffreys" not in prior_variant_note(report)
    (paths[0] / "result.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Uniform baseline"):
        compare_saved_runs(*paths)
    (paths[0] / "result.json").write_text(json.dumps({**report, "variant": None}))
    report["signal"]["data"][0] = 0.98
    (paths[1] / "result.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="matched simulation"):
        compare_saved_runs(*paths)
