"""P2 injection checks: self-consistency, misspecification hook, recovery table."""

import json

import numpy as np

from examples.inference.tris_injection import (
    inject,
    recovery_rows,
    shifted_regions,
    truth_values,
    write_inputs,
)
from tests.test_tris_forward_baseline import tiny_bundle, tiny_external


def injectable_bundle():
    """A tiny bundle whose whitened data is exactly the ring data over sigma."""
    bundle = tiny_bundle()
    sigma = (np.array([0.1, 0.2]), np.array([0.15, 0.25]))
    for index, width in enumerate(sigma):
        operator = np.asarray(bundle[f"response_{index}"], dtype=float) * width[:, None]
        bundle[f"operator_{index}"] = operator
        bundle[f"sigma_k_{index}"] = width
        bundle[f"ring_projection_{index}"] = np.eye(width.size)
        bundle[f"offset_response_{index}"] = 1.0 / width
        bundle[f"data_k_{index}"] = np.zeros(width.size)
        bundle[f"whitened_data_{index}"] = np.zeros(width.size)
    return bundle


def test_inject_keeps_whitened_data_equal_to_the_projection():
    bundle, external = injectable_bundle(), tiny_external()
    rng = np.random.default_rng(0)
    injected, _ = inject(
        bundle, external, ("LWA", "ARCADE"), truth_values(), False, rng
    )
    for index in (0, 1):
        projection = bundle[f"ring_projection_{index}"]
        sigma = bundle[f"sigma_k_{index}"]
        expected = projection @ (injected[f"data_k_{index}"] / sigma)
        np.testing.assert_allclose(
            injected[f"whitened_data_{index}"], expected, rtol=1e-12
        )
        assert not np.allclose(
            injected[f"data_k_{index}"], bundle[f"data_k_{index}"]
        )


def test_inject_with_a_different_region_split_changes_the_data():
    bundle, external = injectable_bundle(), tiny_external()
    generator = {
        **bundle,
        "region": shifted_regions(
            np.array([12.0, 25.0, 45.0]), (10.0, 30.0)
        ),
    }
    assert generator["region"].tolist() == [1, 1, 2]
    assert generator["region"].tolist() != bundle["region"].tolist()
    # A per-region truth is required: with one shared amplitude and index the
    # region assignment cannot change the prediction.
    per_region = {
        "amplitude": np.array([1.0, 1.6, 2.2]),
        "beta": np.array([-2.9, -2.75, -2.6]),
        "zero_standard": np.zeros(2),
        "haslam_monopole_K": np.asarray(0.0),
        "calibration_standard": np.zeros(2),
    }
    original = inject(
        bundle, external, ("LWA", "ARCADE"), per_region, False,
        np.random.default_rng(1),
    )
    shifted = inject(
        bundle, external, ("LWA", "ARCADE"), per_region, False,
        np.random.default_rng(1), generator_bundle=generator,
    )
    assert not np.allclose(
        original[0]["data_k_0"], shifted[0]["data_k_0"]
    )


def test_recovery_rows_report_truth_mean_and_pull():
    truth = {"amplitude": np.array([1.0, 2.0, 3.0])}
    draws = {"amplitude": np.stack([np.full((10, 3), 1.0), np.full((10, 3), 1.5)])}
    rows = recovery_rows(truth, draws, False)
    assert len(rows) == 3
    assert rows[0]["truth"] == 1.0
    assert rows[0]["mean"] == 1.25
    # Zero sd must not divide by zero.
    flat = {"amplitude": np.stack([np.full((4, 3), 2.0), np.full((4, 3), 2.0)])}
    assert all(row["pull"] == 0.0 for row in recovery_rows(truth, flat, False))


def test_write_inputs_produces_a_self_consistent_manifest(tmp_path):
    from examples.inference.tris_prepare import sha256

    bundle, external = injectable_bundle(), tiny_external()
    directory = write_inputs(
        tmp_path / "input",
        bundle,
        external,
        {"schema": "bayesmith.tris.maps.v1", "input_sha256": "stale"},
        {"schema": "bayesmith.tris.rsb.external.v1", "input_sha256": "stale"},
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    external_manifest = json.loads((directory / "external_manifest.json").read_text())
    assert manifest["input_sha256"] == sha256(directory / "maps.npz")
    assert external_manifest["input_sha256"] == sha256(directory / "external.npz")
    assert manifest["synthetic"] is True
