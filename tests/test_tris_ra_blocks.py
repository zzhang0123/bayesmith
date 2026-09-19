"""RA-block holdout helpers: block definitions, scoring and the delta gate."""

import numpy as np

from examples.inference.tris_ra_blocks import (
    block_deltas,
    ra_blocks,
    ring_log_scores,
)


def test_ra_blocks_are_contiguous_and_drop_the_buffer_from_training():
    blocks = ra_blocks(12, folds=3, buffer=1)
    assert [block["heldout"] for block in blocks] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
    assert 4 not in blocks[0]["train"]
    assert 3 not in blocks[1]["train"]
    assert 8 not in blocks[1]["train"]
    assert set(blocks[0]["train"]) == set(range(5, 11))


def test_ra_blocks_with_zero_buffer_partition_the_samples():
    blocks = ra_blocks(10, folds=5, buffer=0)
    flattened = [row for block in blocks for row in block["heldout"]]
    assert sorted(flattened) == list(range(10))
    assert all(len(block["heldout"]) == 2 for block in blocks)


def _ring_bundle():
    template = np.array([10.0, 20.0, 30.0])
    operator = np.array(
        [[1.0, 0.2, 0.0], [0.1, 0.4, 0.3], [0.2, 0.1, 0.8], [0.3, 0.5, 0.1]]
    )
    frequency = np.array([600.5, 817.8])
    cmb = np.array([2.71, 2.706])
    sky0 = template * (600.5 / 408.0) ** -2.8 + cmb[0]
    sky1 = template * (817.8 / 408.0) ** -2.8 + cmb[1]
    return {
        "template_k": template,
        "frequency_mhz": frequency,
        "cmb_k": cmb,
        "operator_0": operator,
        "operator_1": operator,
        "sigma_k_0": np.array([0.1, 0.2, 0.3, 0.4]),
        "sigma_k_1": np.array([0.2, 0.1, 0.3, 0.4]),
        "data_k_0": operator @ sky0,
        "data_k_1": operator @ sky1,
    }


def _draws():
    return {
        "log_amplitude_coefficients": np.zeros((2, 1)),
        "beta_coefficients": np.full((2, 1), -2.8),
        "zero_standard": np.zeros((2, 2)),
        "haslam_monopole_K": np.zeros(2),
    }


def test_ring_log_scores_peak_at_the_generating_sky():
    bundle = _ring_bundle()
    basis = np.ones((3, 1))
    scores = ring_log_scores(bundle, _draws(), basis, False, np.array([0, 1, 2, 3]))
    assert scores.shape == (2,)
    # Both draws are the generating sky, so they agree exactly.
    assert scores[0] == scores[1]
    # A wrong amplitude must score worse.
    bad = {
        "log_amplitude_coefficients": np.full((2, 1), np.log(1.5)),
        "beta_coefficients": np.full((2, 1), -2.8),
        "zero_standard": np.zeros((2, 2)),
        "haslam_monopole_K": np.zeros(2),
    }
    worse = ring_log_scores(bundle, bad, basis, False, np.array([0, 1, 2, 3]))
    assert worse[0] < scores[0]


def test_block_deltas_gate_a_nonconverged_refit():
    rows = [
        {"fold": 0, "variant": "no_rsb", "passed": True, "log_predictive_density": -1.0},
        {"fold": 0, "variant": "rsb", "passed": True, "log_predictive_density": -0.5},
        {"fold": 1, "variant": "no_rsb", "passed": True, "log_predictive_density": -2.0},
        {"fold": 1, "variant": "rsb", "passed": False, "log_predictive_density": -1.5},
    ]
    deltas = {entry["fold"]: entry for entry in block_deltas(rows)}
    assert deltas[0]["valid"] is True
    assert deltas[0]["delta_M1_minus_M0"] == 0.5
    assert deltas[1]["valid"] is False
    assert deltas[1]["delta_M1_minus_M0"] is None
    assert deltas[1]["status"] == "invalid_nonconverged_refit"


def test_ring_score_includes_both_frequency_normalizers():
    from scipy.stats import norm

    bundle = _ring_bundle()
    rows = np.array([0, 2])
    score = ring_log_scores(bundle, _draws(), np.ones((3, 1)), False, rows)
    expected = sum(np.sum(norm.logpdf(np.zeros(rows.size), scale=bundle[f'sigma_k_{i}'][rows])) for i in (0, 1))
    np.testing.assert_allclose(score, expected, atol=1e-12)


def test_ra_buffer_wraps_at_midnight():
    blocks = ra_blocks(12, folds=3, buffer=1)
    assert 11 not in blocks[0]['train']
    assert 0 not in blocks[-1]['train']
