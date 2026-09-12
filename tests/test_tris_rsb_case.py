"""Pure artifact and held-out-prediction checks for the RSB real-data case."""

import hashlib
import json

import numpy as np
import pytest

from examples.inference.tris_rsb_case import (
    build_common_manifest,
    load_common_inputs,
    summarize,
)
from examples.inference.tris_rsb_compare import (
    assemble_comparison,
    heldout_log_predictive_density,
)


def write_inputs(tmp_path):
    tris = tmp_path / "tris"
    external = tmp_path / "external"
    tris.mkdir()
    external.mkdir()
    np.savez_compressed(
        tris / "maps.npz",
        template_k=np.array([10.0, 12.0, 14.0]), region=np.array([0, 1, 2]),
        response_0=np.eye(3), response_1=np.eye(3), offset_response_0=np.ones(3),
        offset_response_1=np.ones(3), whitened_data_0=np.ones(3),
        whitened_data_1=np.ones(3), frequency_mhz=np.array([600.5, 817.8]),
        cmb_k=np.array([2.71, 2.706]), reference_cmb_k=np.array(2.715),
    )
    tris_hash = hashlib.sha256((tris / "maps.npz").read_bytes()).hexdigest()
    (tris / "manifest.json").write_text(json.dumps({"schema": "bayesmith.tris.maps.v1", "input_sha256": tris_hash}))
    np.savez_compressed(
        external / "external.npz", frequency_mhz=np.array([40.0, 3200.0]),
        temperature_rj_k=np.array([5792.0, 2.75]), sigma_rj_k=np.array([963.0, .01]),
        sigma_independent_rj_k=np.array([960.0, .008]), tau_rj_k=np.array([10.0, .005]),
        survey_code=np.array([0, 1]), survey=np.array(["LWA", "ARCADE"]),
    )
    external_hash = hashlib.sha256((external / "external.npz").read_bytes()).hexdigest()
    (external / "external_manifest.json").write_text(json.dumps({"schema": "bayesmith.tris.rsb.external.v1", "input_sha256": external_hash, "rows": [{"survey": "LWA"}, {"survey": "ARCADE"}]}))
    return tris, external


def test_common_inputs_bind_both_archives_and_make_one_ordered_manifest(tmp_path):
    tris, external = write_inputs(tmp_path)
    common = load_common_inputs(tris, external)
    manifest = build_common_manifest(common)
    assert manifest["external_surveys"] == ["LWA", "ARCADE"]
    assert manifest["external_rows"] == 2
    assert manifest["common_input_sha256"] == build_common_manifest(common)["common_input_sha256"]


def test_common_input_loader_rejects_tampered_archive_hash(tmp_path):
    tris, external = write_inputs(tmp_path)
    (external / "external_manifest.json").write_text(json.dumps({"schema": "bayesmith.tris.rsb.external.v1", "input_sha256": "0" * 64, "rows": []}))
    with pytest.raises(ValueError, match="hash"):
        load_common_inputs(tris, external)


def test_summary_uses_conditional_95_percent_intervals():
    result = summarize(np.array([0.0, 1.0, 2.0, 3.0]))
    assert result["mean"] == pytest.approx(1.5)
    assert result["lower"] == pytest.approx(.075)
    assert result["upper"] == pytest.approx(2.925)


def test_heldout_score_integrates_unseen_survey_calibration_and_reports_mcse():
    heldout = {
        "frequency_mhz": np.array([3200.0, 3410.0]),
        "temperature_rj_k": np.array([2.75, 2.74]),
        "sigma_independent_rj_k": np.array([.008, .008]),
        "tau_rj_k": np.array([.005, .005]),
        "survey_code": np.array([0, 0]),
    }
    draws = {"rsb_amplitude": np.linspace(.4, .8, 8), "rsb_beta": np.full(8, -2.6)}
    score = heldout_log_predictive_density(draws, heldout, include_rsb=True)
    assert np.isfinite(score["log_predictive_density"])
    assert score["mcse"] >= 0.0
    assert score["draws"] == 8


def test_comparison_refuses_full_models_with_different_common_input_hashes(tmp_path):
    for name, digest in (("tris_haslam_no_rsb", "a"), ("tris_haslam_rsb", "b")):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps({"data_manifest": {"common_input_sha256": digest}}))
    with pytest.raises(ValueError, match="common input"):
        assemble_comparison(tmp_path)


def test_comparison_renderer_explains_predictive_limit():
    from examples.inference.tris_rsb_presentation import render_case

    html = render_case({
        "case": "tris_haslam_rsb_comparison", "kind": "real_observations",
        "title": "TRIS + Haslam + RSB", "passed": True,
        "models": {"no_rsb": {"parameters": []}, "rsb": {"parameters": []}},
        "comparison_policy": "Cross-survey posterior predictive scores only; no Bayes factor.",
        "note": "Predictive comparison does not establish a physical origin.",
        "data_manifest": {"common_input_sha256": "abc"},
    }, "tris_haslam_rsb_comparison")
    assert "ARCADE 2" in html and "LWA" in html
    assert "Bayes factor" in html and "physical origin" in html
    assert html.count('data-stage=') == 5
    assert 'class="previous"' in html and 'class="next"' in html
    assert 'data-language-choice="zh"' in html
    assert 'M0 diagnostics' in html and 'M1 diagnostics' in html
    assert 'tris-metrics rsb-metrics' in html
