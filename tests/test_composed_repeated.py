"""Repeated datasets, not posterior draws, are the replication unit."""

import hashlib
import json

import numpy as np
import pytest
from scipy.stats import binom

from examples.inference.composed_repeated import execute_one
from examples.inference.repeated_summary import (
    coverage_uncertainty,
    mean_uncertainty,
    summarize_coordinate,
    summarize_study,
)


def test_repeated_statistics_use_dataset_errors_and_include_misses():
    errors = np.array([-2.0, 0.0, 1.0, 3.0])
    rows = [{"mean": 10+e, "truth": 10, "posterior_sd": 2,
             "intervals": {"0.95": [10+e-1.5, 10+e+1.5]}} for e in errors]
    actual = summarize_coordinate(rows)
    assert actual["replicates"] == 4
    assert actual["bias"] == 0.5
    assert actual["bias_mcse"] == pytest.approx(np.sqrt(13/3)/2)
    assert actual["rmse"] == pytest.approx(np.sqrt(3.5))
    assert actual["coverage"]["0.95"]["covered"] == 2
    assert actual["coverage"]["0.95"]["total"] == 4
    assert actual["bias_interval_95"][0] < 0 < actual["bias_interval_95"][1]
    with pytest.raises(ValueError, match="two finite"):
        mean_uncertainty([1])


@pytest.mark.parametrize("covered", [[True]*32, [False]*32, [True]*29+[False]*3])
def test_coverage_interval_inverts_binomial_tails_even_at_boundaries(covered):
    result = coverage_uncertainty(covered)
    low, high = result["interval_95"]
    n, k = len(covered), sum(covered)
    assert low <= k/n <= high
    if k:
        assert binom.sf(k-1, n, low) == pytest.approx(0.025, abs=2e-14)
    else:
        assert low == 0
    if k < n:
        assert binom.cdf(k, n, high) == pytest.approx(0.025, abs=2e-14)
    else:
        assert high == 1
        assert low < 0.9  # 32 perfect outcomes do not establish 99% coverage.


def test_worker_launch_and_record_write_failures_remain_attempts(tmp_path, monkeypatch):
    import examples.inference.composed_repeated as module

    def refuse(*args, **kwargs):
        raise OSError("simulated launch failure")

    monkeypatch.setattr(module.subprocess, "run", refuse)
    record = execute_one(tmp_path, 1, "uniform", {"draws": 4, "warmup": 4})
    assert not record["completed"] and record["error"]["type"] == "OSError"
    assert json.loads((tmp_path / "seed-0001/uniform/attempt.json").read_text()) == record
    monkeypatch.setattr(module, "write_json", refuse)
    record = execute_one(tmp_path, 2, "uniform", {"draws": 4, "warmup": 4})
    assert not record["completed"] and record["attempt_write_error"]


def test_paired_study_keeps_diagnostic_failures_and_refuses_unpaired_data(tmp_path):
    config = {"kind": "fixed_root_repeated_simulation", "seeds": [1, 2],
              "priors": ["uniform", "mild"], "chains": 2, "draws": 4, "warmup": 4,
              "fixed_roots": {"sigma_w": 0.015}, "interval_masses": [0.68, 0.95, 0.99],
              "mild_prior": {"loc": 0.02, "scale": 0.01, "low": 0.003, "high": 0.06}}
    (tmp_path / "registered.json").write_text(json.dumps(config))
    (tmp_path / "completion.json").write_text(json.dumps({"source_unchanged": True}))
    attempts = []
    for seed in config["seeds"]:
        for prior in config["priors"]:
            path = tmp_path / f"{seed}-{prior}"
            path.mkdir()
            draws = 0.015+(seed-1.5)*0.002+(prior == "mild")*0.0001+np.linspace(-0.0005, 0.0005, 8)
            np.savez(path / "posterior.npz", sigma_w=draws)
            report = {"seed": seed, "chain_shape": [2, 4], "warmup": 4,
                      "variant": {"kind": "mild_noise_prior", "loc": 0.02, "scale": 0.01,
                                  "support": [0.003, 0.06]} if prior == "mild" else None,
                      "parameters": [{"name": "sigma_w", "truth": 0.015, "mean": float(draws.mean()),
                                      "posterior_sd": float(draws.std(ddof=1))}],
                      "checks": {"chain_diagnostics": {"divergences": 0, "sites": {"sigma_w": {}}}},
                      "signal": {"x": [1, 2], "data": [seed, 3], "truth": [1, 1], "view": {}},
                      "method": "test"}
            result = path / "result.json"
            result.write_text(json.dumps(report))
            attempts.append({"seed": seed, "prior": prior, "completed": True,
                             "source_matches_registration": True, "diagnostics_passed": seed == 1,
                             "recovery_passed": False, "elapsed_seconds": 1,
                             "result": str(result.relative_to(tmp_path)),
                             "result_sha256": hashlib.sha256(result.read_bytes()).hexdigest()})
    (tmp_path / "progress.json").write_text(json.dumps(attempts))
    summary = summarize_study(tmp_path)
    from examples.inference.repeated_panels import render_repeated

    markup = render_repeated(summary, "../repeated-composed")
    assert "2 registered datasets" in markup and "2 completed pairs" in markup
    assert "diagnostics failed, retained" in markup
    assert 'src="../repeated-composed/sigma-repeats.png"' in markup
    assert summary["groups"]["uniform"]["completed"] == 2
    assert summary["groups"]["uniform"]["diagnostics_passed"] == 1
    assert summary["groups"]["uniform"]["parameters_all_completed"]["sigma_w"]["replicates"] == 2
    assert summary["paired_mild_minus_uniform"]["sigma_w"]["mean_change"]["mean"] == pytest.approx(0.0001)
    posterior_path = (tmp_path / attempts[-1]["result"]).with_name("posterior.npz")
    archive = posterior_path.read_bytes()
    posterior_path.unlink()
    missing = summarize_study(tmp_path)
    assert missing["groups"]["mild"]["completed"] == 1
    assert len(missing["failed_attempts"]) == 1
    assert "artifact_read_error" in missing["failed_attempts"][0]
    posterior_path.write_bytes(archive[:len(archive)//2])
    truncated = summarize_study(tmp_path)
    assert truncated["groups"]["mild"]["completed"] == 1
    assert "Unreadable posterior archive" in truncated["failed_attempts"][0]["artifact_read_error"]
    posterior_path.write_bytes(archive)
    result = tmp_path / attempts[-1]["result"]
    report = json.loads(result.read_text())
    report["signal"]["data"][0] += 1
    result.write_text(json.dumps(report))
    attempts[-1]["result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    (tmp_path / "progress.json").write_text(json.dumps(attempts))
    with pytest.raises(ValueError, match="identical simulated data"):
        summarize_study(tmp_path)
