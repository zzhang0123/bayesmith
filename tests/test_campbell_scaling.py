"""Independent checks of the scalable sampler and the two evidence estimators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from bayesmith.reweight import PosteriorReweighting
from examples.inference.campbell_sky import (
    SkyConfig,
    beam_matrix,
    beam_rows,
    coefficient_logpdf,
    connected_cumulant,
    make_graph,
)
from examples.inference.campbell_sky_scaling import (
    draw_bank,
    make_scores,
    maximize_grid,
)


def test_selected_beam_rows_match_dense_and_scale_to_large_maps():
    small = SkyConfig(shape=(4, 5))
    np.testing.assert_array_equal(
        beam_rows(small, [0, 1, 19]), beam_matrix(small)[[0, 1, 19]]
    )
    large = SkyConfig(shape=(256, 256))
    # Nonzero cross-cumulants survive periodic wraparound without an n² matrix.
    assert connected_cumulant(large, 2.0, (0, 0, 255)) == pytest.approx(
        connected_cumulant(small, 2.0, (0, 0, 4))
    )


def test_gaussian_reference_draws_have_conditional_mean_variance_and_independence():
    observed = np.array([-3.0, 0.0, 2.0])
    bank = np.asarray(draw_bank(jax.random.key(17), observed, 40000))
    expected_variance = 0.01 / 1.01
    # 6 standard errors, derived from Gaussian mean/covariance sampling variance.
    np.testing.assert_allclose(
        bank.mean(axis=0),
        observed / 1.01,
        atol=6 * np.sqrt(expected_variance / len(bank)),
        rtol=0,
    )
    covariance = np.cov(bank.T)
    np.testing.assert_allclose(
        covariance,
        np.eye(3) * expected_variance,
        atol=6 * expected_variance * np.sqrt(2 / (len(bank) - 1)),
        rtol=0,
    )
    lag = np.mean((bank[1:] - observed / 1.01) * (bank[:-1] - observed / 1.01), axis=0)
    assert np.max(np.abs(lag)) < 6 * expected_variance / np.sqrt(len(bank) - 1)
    assert not np.array_equal(
        bank, np.asarray(draw_bank(jax.random.key(18), observed, 40000))
    )


@pytest.mark.parametrize("rate", [0.5, 2.0, 8.0])
def test_streaming_estimators_match_independent_numpy_density_ratios(rate):
    config = SkyConfig(shape=(8, 8))
    bank = np.random.default_rng(3).normal(size=(128, 64))
    ratio = coefficient_logpdf(bank, rate, config) - norm.logpdf(bank)
    local_norm = logsumexp(ratio, axis=0)
    local_ess = np.exp(2 * local_norm - logsumexp(2 * ratio, axis=0))
    joint = ratio.sum(axis=1)
    joint_norm = logsumexp(joint)
    expected = [
        np.sum(local_norm - np.log(len(bank))),
        joint_norm - np.log(len(bank)),
        np.exp(2 * joint_norm - logsumexp(2 * joint)),
        np.exp(joint.max() - joint_norm),
        local_ess.min(),
        local_ess.mean(),
    ]
    for chunk in (8, 64):
        np.testing.assert_allclose(
            make_scores(bank, config, chunk)(jnp.asarray(rate)),
            expected,
            rtol=3e-5,
            atol=2e-5,
        )


def test_streaming_joint_is_the_existing_full_posterior_reweighting_objective():
    config = SkyConfig(shape=(8, 8))
    data = np.linspace(-2, 2, 64).reshape(config.shape)
    bank = draw_bank(jax.random.key(4), data.ravel(), 256)
    problem = PosteriorReweighting.from_graphs(
        make_graph(data, config),
        {"coefficients": bank.reshape((256,) + config.shape)},
        lambda rate: make_graph(data, config, rate),
    )
    rate = jnp.asarray(1.7)
    reference = problem.estimate(rate)
    streamed = make_scores(bank, config)(rate)
    np.testing.assert_allclose(
        streamed[1:4],
        [reference.log_mean_weight, reference.ess, reference.max_weight],
        rtol=2e-5,
        atol=2e-5,
    )


def test_independent_cell_permutations_preserve_only_factorized_estimator():
    config = SkyConfig(shape=(8, 8))
    bank = np.random.default_rng(1).normal(size=(64, 64))
    shifted = np.column_stack([np.roll(bank[:, j], j) for j in range(64)])
    before = np.asarray(make_scores(bank, config)(jnp.asarray(0.6)))
    after = np.asarray(make_scores(shifted, config)(jnp.asarray(0.6)))
    np.testing.assert_allclose(
        before[[0, 4, 5]], after[[0, 4, 5]], rtol=2e-5, atol=1e-5
    )
    assert abs(before[1] - after[1]) > 0.1


def test_grid_maximization_uses_boundary_and_interior_modes():
    grid = np.linspace(0.5, 8, 61)
    assert maximize_grid(lambda r: r, grid, grid) == 8
    mode = maximize_grid(lambda r: -((r - 2.13) ** 2), grid, -((grid - 2.13) ** 2))
    assert mode == pytest.approx(2.13, abs=2e-4)


def test_campbell_chapter_joins_existing_navigation_and_preserves_saved_evidence(
    tmp_path,
):
    # The child below runs examples/inference/plot_results.py, whose own
    # docstring says it requires optional matplotlib. The child is started with
    # `sys.executable`, so the parent's ability to import it is the child's.
    pytest.importorskip("matplotlib")

    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path

    from examples.inference.campbell_notebook import install
    from examples.inference.presentation import gallery_navigation

    source = tmp_path / "input"
    source.mkdir()
    report = {
        "case": "campbell_sky",
        "kind": "analytic_validation",
        "config": {"shape": [8, 8], "rate": 2.0, "count_max": 48},
        "draws": 128,
        "scaling": [
            {
                "side": 8,
                "cells": 64,
                "draws": 128,
                "cold_seconds": 0.1,
                "warm_seconds_median": 0.01,
                "bank_mib": 0.1,
            }
        ],
        "banks": [
            {
                "seed": 71,
                "draw_seconds": 0.01,
                "score_compile_seconds": 0.2,
                "profile_and_fit_seconds": 1.0,
                "factorized_map": 2.0,
                "joint_map": 2.1,
                "joint_ess_at_truth": 30.0,
                "local_min_ess_at_fit": 120.0,
                "factorized_log_evidence_error_at_fit": 0.02,
            }
        ],
        "environment": {"platform": "test"},
        "oracle_map": 2.0,
        "oracle_interval95": [1.5, 2.5],
        "poisson_tail_union_bound": 1e-20,
        "cutoff_log_evidence_difference": 0.0,
    }
    report["grid"] = [0.5, 2.0, 4.0, 8.0]
    report["oracle_profile"] = [-10.0, 0.0, -10.0, -20.0]
    report["banks"][0]["factorized_profile"] = [-10.0, 0.02, -10.0, -20.0]
    report["banks"][0]["joint_profile"] = [-11.0, -0.5, -11.0, -21.0]
    (source / "result.json").write_text(json.dumps(report))
    for name in ("maps.npz", "sky.png", "scaling.png", "scaling.pdf"):
        (source / name).write_bytes(b"test fixture, rendering does not read pixels")
    np.savez(
        source / "maps.npz",
        coefficients=np.zeros((8, 8)),
        observed_coefficients=np.zeros((8, 8)),
        sky=np.zeros((8, 8)),
        observed_sky=np.zeros((8, 8)),
        analytic_mean=np.zeros((8, 8)),
    )
    before = hashlib.sha256((source / "result.json").read_bytes()).hexdigest()
    notebook = tmp_path / "notebook"
    page = install(source, notebook).read_text()
    assert page.count('id="campbell_sky"') == 1
    assert "Campbell 定理" in page and "Campbell’s theorem" in page
    assert all(
        f'id="campbell_sky-{stage}"' in page
        for stage in ("model", "methods", "diagnostics", "sampling", "recovery")
    )
    assert 'data-select-case="campbell_sky"' in page
    assert 'data-reading-layout="continuous"' in page
    for section in ("distributions", "campbell", "priors", "edgeworth"):
        assert f'id="campbell_sky-{section}"' in page
    assert 'class="step-button"' not in page.split('id="campbell_sky"', 1)[1]
    for name in (
        "distributions",
        "edgeworth",
        "edgeworth_accuracy",
        "posterior_density",
    ):
        assert f"campbell_sky/{name}.png" in page
        assert (notebook / "campbell_sky" / f"{name}.png").stat().st_size > 1000
    assert "campbell_sky/sky.png" in page
    provenance = json.loads((notebook / "presentation.json").read_text())
    assert provenance["inputs"]["campbell_sky"]["result.json"] == before
    assert "campbell_presentation.py" in provenance["renderer_sha256"]
    assert hashlib.sha256((source / "result.json").read_bytes()).hexdigest() == before
    # Existing six-case catalogue stays available when this chapter is present.
    navigation = gallery_navigation(
        [({"case": "linear_gaussian"}, "linear_gaussian"), (report, "campbell_sky")],
        notebook,
    )
    assert 'data-select-case="linear_gaussian"' in navigation
    assert 'data-select-case="campbell_sky"' in navigation
    counterpart = notebook / "proposals"
    counterpart.mkdir()
    (notebook / "linear_gaussian").mkdir()
    (notebook / "linear_gaussian/result.json").write_text("{}")
    assert "../index.html#case=campbell_sky" in gallery_navigation([], counterpart)

    # Exercise the documented direct-script entry (different sys.path than -m).
    root = Path(__file__).resolve().parents[1]
    (notebook / "linear_gaussian/result.json").unlink()
    done = subprocess.run(
        [
            sys.executable,
            str(root / "examples/inference/plot_results.py"),
            "--input",
            str(notebook),
        ],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert (
        hashlib.sha256((notebook / "campbell_sky/result.json").read_bytes()).hexdigest()
        == before
    )
