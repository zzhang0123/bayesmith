"""Regression tests for the convergence review, independent of saved fit results."""

import json

import jax
import numpy as np
import pytest

from bayesmith import trace
from bayesmith.graph.evaluate import log_joint
from examples.inference.tris_rsb_data import survey_covariance
from examples.inference.tris_rsb_initialization import (
    data_initialization,
    fitting_problem,
)
from examples.inference.tris_rsb_sky import model, model_inputs
from tests.test_tris_rsb_sky import tiny_bundle, tiny_external


@pytest.mark.parametrize("rsb", [False, True])
def test_initialization_objective_retains_original_joint_density(rsb):
    with jax.enable_x64(True):
        inputs = model_inputs(tiny_bundle(), tiny_external(), ("LWA", "ARCADE"), rsb)
        graph = trace(model, *inputs)
        decode, residual, _, _ = fitting_problem(graph, inputs)
        x = np.array([1., 1.1, .9, -2.7, -2.8, -2.9, .1, -.2, 8., .2, -.3]
                     + ([.5, -2.6] if rsb else []))
        y = x.copy()
        y[[0, 3, 6, 8, 9]] += [.2, .1, -.4, .7, .2]
        if rsb:
            y[-2:] += [.2, -.1]
        delta = -.5 * (np.sum(np.asarray(residual(y)) ** 2) - np.sum(np.asarray(residual(x)) ** 2))
        assert float(log_joint(graph, decode(y)) - log_joint(graph, decode(x))) == pytest.approx(delta, abs=1e-10)


def test_fit_starts_are_distinct_finite_and_reproducible():
    with jax.enable_x64(True):
        inputs = model_inputs(tiny_bundle(), tiny_external(), ("LWA", "ARCADE"), True)
        graph = trace(model, *inputs)
        policy, report = data_initialization(graph, inputs, chains=4, seed=17)
        again, _ = data_initialization(graph, inputs, chains=4, seed=17)
        assert report["posterior_target_changed"] is False
        assert len(report["fits"]) == 3
        for first, second in zip(policy.values, again.values, strict=True):
            np.testing.assert_array_equal(first.value, second.value)
        values = {item.name: item.value for item in policy.values}
        assert not np.array_equal(values["amplitude"][0], values["amplitude"][1])
        for index in range(4):
            assert np.isfinite(log_joint(graph, {name: value[index] for name, value in values.items()}))


@pytest.mark.parametrize("shape", [(), (2, 2), (1, 2, 1)])
def test_covariance_rejects_non_vector_inputs(shape):
    with pytest.raises(ValueError, match="one-dimensional"):
        survey_covariance(np.ones(shape), np.zeros(shape), np.ones(shape))


def test_joint_sky_products_keep_paired_draws_and_separate_instrument_offsets():
    from examples.inference.tris_rsb_diagnostics import posterior_checks

    bundle = tiny_bundle()
    for i in range(2):
        # A coherent reconstruction with known unequal noise, not a test
        # that compares two copies of the same hard-coded chi-square formula.
        operator = bundle[f"response_{i}"]
        sigma = np.array([.2, .4])
        bundle.update({f"operator_{i}": operator, f"sigma_k_{i}": sigma,
                       f"data_k_{i}": np.array([8., 9.]), f"ra_deg_{i}": np.array([0., 180.]),
                       f"weights_{i}": np.array([[1., 0.], [0., 1.], [.5, .5]]),
                       f"bias_{i}": np.zeros(3), f"response_{i}": operator / sigma[:, None],
                       f"offset_response_{i}": 1 / sigma, f"whitened_data_{i}": np.array([8., 9.]) / sigma})
    draws = {"amplitude": np.array([[.8, 1., 1.2], [1.2, 1., .8]]),
             "beta": np.full((2, 3), -2.6), "zero_standard": np.zeros((2, 2)),
             "haslam_monopole_K": np.array([1., 2.]),
             "rsb_amplitude": np.array([.2, .3]), "rsb_beta": np.array([-2.5, -2.7])}
    diagnostics, products = posterior_checks(bundle, draws, seed=9)
    expected = []
    for row in range(2):
        b408 = draws["rsb_amplitude"][row] * .408 ** draws["rsb_beta"][row]
        expected.append(draws["amplitude"][row] * (bundle["template_k"] + draws["haslam_monopole_K"][row] - b408)
                        + bundle["reference_cmb_k"] + b408)
    np.testing.assert_allclose(products["haslam_mean"], np.mean(expected, axis=0))
    assert diagnostics["map_likelihood"]["passed"]
    shifted, changed = posterior_checks(bundle, {**draws, "zero_standard": np.ones((2, 2))}, seed=9)
    np.testing.assert_array_equal(changed["haslam_mean"], products["haslam_mean"])
    np.testing.assert_allclose(changed["map_mean_0"], products["map_mean_0"] - .066)
    assert shifted["map_likelihood"]["passed"]


def test_heldout_scores_use_directional_refits_and_fail_closed(tmp_path):
    from examples.inference.tris_rsb_compare import assemble_comparison

    external = tiny_external()
    for variant in ("no_rsb", "rsb"):
        folder = tmp_path / f"tris_haslam_{variant}"
        folder.mkdir()
        report = {"variant": variant, "passed": True, "included_surveys": ["LWA", "ARCADE"],
                  "data_manifest": {"common_input_sha256": "common"}, "chain_shape": [2, 8]}
        (folder / "result.json").write_text(json.dumps(report))
        np.savez_compressed(folder / "external.npz", **external)
        for survey in ("LWA", "ARCADE"):
            fold = tmp_path / "heldout" / f"{variant}_train_{survey}"
            fold.mkdir(parents=True)
            (fold / "result.json").write_text(json.dumps({**report, "included_surveys": [survey]}))
            np.savez_compressed(fold / "posterior.npz", rsb_amplitude=np.linspace(.1, .8, 16), rsb_beta=np.full(16, -2.6))
    result = assemble_comparison(tmp_path)
    assert result["comparison_status"] == "available"
    assert len(result["heldout_scores"]) == 2
    assert len(result["data_manifest"]["source_refit_sha256"]) == 8
    for row in result["heldout_scores"]:
        assert row["models"]["no_rsb"]["mcse"] == 0
        assert row["models"]["rsb"]["chain_shape"] == [2, 8]
    path = tmp_path / "heldout" / "rsb_train_LWA" / "result.json"
    fold = json.loads(path.read_text())
    path.write_text(json.dumps({**fold, "passed": False}))
    assert assemble_comparison(tmp_path)["comparison_status"] == "blocked_nonconverged_refits"
    path.write_text(json.dumps({**fold, "included_surveys": ["LWA", "ARCADE"]}))
    with pytest.raises(ValueError, match="training survey"):
        assemble_comparison(tmp_path)


def test_unseen_no_rsb_calibration_is_exactly_marginalized():
    from scipy.stats import multivariate_normal

    from examples.inference.tris_rsb_compare import heldout_log_predictive_density
    from examples.inference.tris_rsb_data import to_rj_temperature

    external = tiny_external()
    external["survey_code"] = np.array([0, 0])
    # Analytically integrating c~N(0,1) adds the outer product tau tau^T.
    # Standardize before the independent eigensolve oracle: this fixture spans
    # K-to-mK scales, which scipy's default relative rank tolerance conflates.
    scale = external["sigma_independent_rj_k"]
    mean = np.array([to_rj_temperature(2.725, nu) for nu in external["frequency_mhz"]])
    expected = multivariate_normal.logpdf(
        (external["temperature_rj_k"] - mean) / scale,
        mean=np.zeros(2), cov=np.eye(2) + np.outer(external["tau_rj_k"] / scale, external["tau_rj_k"] / scale),
    ) - np.log(scale).sum()
    result = heldout_log_predictive_density({"amplitude": np.ones((16, 3))}, external,
                                            include_rsb=False, chain_shape=(2, 8))
    assert result["log_predictive_density"] == pytest.approx(expected)
    assert result["mcse"] == 0
