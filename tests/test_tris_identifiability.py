"""P2 identifiability checks: Jacobian, prior scaling and posterior correlation."""

import jax
import numpy as np
import pytest

from examples.inference.tris_identifiability import (
    analyze,
    coordinate_names,
    degeneracy_rows,
    from_vector,
    jacobian,
    prior_sd,
    to_vector,
)
from tests.test_tris_forward_baseline import tiny_bundle, tiny_external
from tests.test_tris_forward_baseline import values as tiny_values


def test_prior_sd_has_one_entry_per_scalar_coordinate():
    assert prior_sd(False, calibration_coordinates=2).shape == (11,)
    assert prior_sd(True, calibration_coordinates=2).shape == (13,)
    assert prior_sd(True, calibration_coordinates=3).shape == (14,)
    assert np.all(prior_sd(True, calibration_coordinates=2) > 0)


def test_vector_roundtrip_preserves_the_block_layout():
    chosen = tiny_values(True)
    theta = to_vector(chosen, True)
    back = from_vector(theta, True, calibration_coordinates=2)
    np.testing.assert_allclose(to_vector(back, True), theta, rtol=1e-12)
    assert back["zero_standard"].shape == (2,)
    assert np.ndim(back["haslam_monopole_K"]) == 0


def test_from_vector_rejects_a_wrong_length():
    with pytest.raises(ValueError):
        from_vector(np.zeros(5), False, calibration_coordinates=2)


def test_coordinate_names_match_the_parameter_vector():
    assert len(coordinate_names(False, calibration_coordinates=2)) == 11
    assert len(coordinate_names(True, calibration_coordinates=2)) == 13
    assert coordinate_names(False, calibration_coordinates=2)[-1] == (
        "calibration_standard[1]"
    )


def test_zero_level_jacobian_matches_the_analytic_column():
    with jax.enable_x64(True):
        bundle, external = tiny_bundle(), tiny_external()
        names = coordinate_names(False, calibration_coordinates=2)
        matrix = jacobian(
            bundle, external, ("LWA", "ARCADE"), tiny_values(False), False
        )
        column = matrix[:, names.index("zero_standard[0]")]
        expected = np.concatenate(
            [
                -0.066 * np.asarray(bundle["offset_response_0"], dtype=float),
                np.zeros(2),  # the 817.8 MHz rows do not depend on zero_standard[0]
                np.zeros(2),  # the external rows do not either
            ]
        )
        np.testing.assert_allclose(column, expected, rtol=1e-6, atol=1e-9)


def test_analyze_is_internally_consistent():
    with jax.enable_x64(True):
        bundle, external = tiny_bundle(), tiny_external()
        analysis = analyze(
            bundle, external, ("LWA", "ARCADE"), tiny_values(False), False
        )
        assert len(analysis["names"]) == 11
        assert analysis["observation_count"] == 4 + 2  # 4 TRIS rows, 2 external
        correlation = np.asarray(analysis["posterior_correlation"])
        np.testing.assert_allclose(np.diag(correlation), 1.0, rtol=1e-9)
        ratios = np.asarray(analysis["posterior_over_prior_sd"])
        assert np.all(ratios > 0)
        assert np.all(ratios <= 1.0 + 1e-9)
        rows = degeneracy_rows(analysis, top=3)
        assert len(rows) == 3
        assert all(abs(row["correlation"]) <= 1.0 + 1e-9 for row in rows)
