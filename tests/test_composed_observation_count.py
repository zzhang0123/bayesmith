"""Observation count changes data size, without changing the generating process."""

import copy
import json

import jax
import numpy as np
import pytest

from examples.inference import composed_process as demo
from examples.inference.observation_count_reference import compare_observation_counts


def test_larger_grid_preserves_process_truth_and_default_simulation(monkeypatch):
    captured = []
    simulate = demo.simulate_fixed

    class SimulationCaptured(Exception):
        pass

    def capture(graph, model, truths, key):
        data, result = simulate(graph, model, truths, key)
        captured.append((np.asarray(data), {k: np.asarray(v) for k, v in truths.items()}))
        return data, result

    def stop_before_initialization(graph):
        raise SimulationCaptured

    monkeypatch.setattr(demo, "simulate_fixed", capture)
    monkeypatch.setattr(demo, "data_initialization", stop_before_initialization)
    with jax.enable_x64(True):
        for kwargs in ({}, {"observations": 3840}, {"observations": 384}):
            with pytest.raises(SimulationCaptured):
                demo.run(**kwargs)
    assert [data.shape for data, _ in captured] == [(3840,), (3840,), (384,)]
    np.testing.assert_array_equal(captured[0][0], captured[1][0])
    for _, truths in captured[1:]:
        assert truths.keys() == captured[0][1].keys()
        for name, truth in truths.items():
            np.testing.assert_array_equal(truth, captured[0][1][name])
    assert captured[2][1]["instance"].shape == (12,)
    for invalid in (0, -1, 3, 384.0, True):
        with pytest.raises(ValueError, match="observations"):
            demo.run(observations=invalid)


def test_size_comparison_refuses_changed_truth_prior_budget_and_metadata(tmp_path):
    baseline = {
        "seed": 0, "chain_shape": [2, 4], "warmup": 5,
        "interval_mass": 0.99, "max_sd_ratio": 0.5, "variant": None,
        "parameters": [{"name": "sigma_w", "truth": 0.015}],
        "signal": {"data": [1.0] * 4, "view": {"noise_realization": {}}},
        "checks": {"recovery": True, "chain_diagnostics": {"passed": True, "divergences": 0}},
        "method": "iterative_gls+mh+nuts",
    }
    larger = copy.deepcopy(baseline)
    larger["signal"]["data"] = [1.0] * 40
    larger["variant"] = {"kind": "observation_count", "prior": "bounded_uniform_roots", "observations": 40}
    paths = [tmp_path / name for name in ("baseline", "larger")]
    for path, report in zip(paths, (baseline, larger), strict=True):
        path.mkdir()
        (path / "result.json").write_text(json.dumps(report))
    actual = compare_observation_counts(*paths)
    assert [r["observations"] for r in actual["rows"]] == [4, 40]
    assert actual["same_process_and_parameter_truths"] and not actual["noise_is_nested"]
    assert len(actual["inputs_sha256"]) == 2
    for change in (
        {"variant": None}, {"warmup": 6},
        {"parameters": [{"name": "sigma_w", "truth": 0.02}]},
        {"variant": {"prior": "bounded_uniform_roots", "observations": 41}},
    ):
        (paths[1] / "result.json").write_text(json.dumps({**larger, **change}))
        with pytest.raises(ValueError):
            compare_observation_counts(*paths)
