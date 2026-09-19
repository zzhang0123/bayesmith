"""Published ARCADE 2/LWA rows used by the joint RSB analysis."""

import hashlib
import json

import numpy as np
import pytest

from examples.inference.tris_rsb_data import (
    TAU_K,
    likelihood_arrays,
    load_declared_rows,
    prepare_external_data,
    survey_covariance,
    to_rj_sigma,
    to_rj_temperature,
)


def test_declared_rows_are_exactly_the_approved_eleven_and_exclude_haslam():
    rows = load_declared_rows()
    assert [(row.survey, row.frequency_mhz) for row in rows] == [
        ("LWA", 40.0),
        ("LWA", 50.0),
        ("LWA", 60.0),
        ("LWA", 70.0),
        ("LWA", 80.0),
        ("ARCADE", 3200.0),
        ("ARCADE", 3410.0),
        ("ARCADE", 7970.0),
        ("ARCADE", 8330.0),
        ("ARCADE", 9720.0),
        ("ARCADE", 10490.0),
    ]
    assert all(row.frequency_mhz not in {22.0, 45.0, 408.0, 1420.0} for row in rows)


def test_transcription_pins_all_source_table_values_and_convention():
    rows = load_declared_rows()
    assert [
        (row.temperature_thermodynamic_k, row.sigma_thermodynamic_k)
        for row in rows
    ] == [
        (5792.0, 963.0),
        (3443.0, 526.0),
        (2363.0, 365.0),
        (1505.0, 208.0),
        (1188.0, 112.0),
        (2.792, 0.010),
        (2.771, 0.009),
        (2.765, 0.014),
        (2.741, 0.016),
        (2.732, 0.006),
        (2.732, 0.006),
    ]
    assert {row.convention for row in rows} == {"thermodynamic_temperature"}
    assert {row.table for row in rows if row.survey == "LWA"} == {"Dowell & Taylor (2018), Table 2"}
    assert {row.table for row in rows if row.survey == "ARCADE"} == {"Fixsen et al. (2009), arXiv:0901.0555v1, Table 4"}


def test_rj_conversion_is_identity_at_low_frequency_and_has_analytic_sigma_derivative():
    assert to_rj_temperature(5792.0, 40.0) == pytest.approx(5792.0, rel=2e-7)
    temperature, sigma, frequency = 2.792, 0.010, 3200.0
    numerical = (
        to_rj_temperature(temperature + 1e-6, frequency)
        - to_rj_temperature(temperature - 1e-6, frequency)
    ) / (2e-6)
    assert to_rj_sigma(temperature, sigma, frequency) == pytest.approx(
        abs(numerical) * sigma, rel=1e-7
    )


def test_shared_calibration_keeps_published_diagonals_and_adds_only_within_survey_covariance():
    arrays = likelihood_arrays(load_declared_rows())
    covariance = survey_covariance(
        arrays["sigma_independent_rj_k"],
        arrays["survey_code"],
        arrays["tau_rj_k"],
    )
    np.testing.assert_allclose(np.diag(covariance), arrays["sigma_rj_k"] ** 2)
    assert covariance[0, 1] == pytest.approx(
        to_rj_sigma(5792.0, TAU_K["LWA"], 40.0)
        * to_rj_sigma(3443.0, TAU_K["LWA"], 50.0)
    )
    assert covariance[5, 6] == pytest.approx(
        to_rj_sigma(2.792, TAU_K["ARCADE"], 3200.0)
        * to_rj_sigma(2.771, TAU_K["ARCADE"], 3410.0)
    )
    assert covariance[0, 5] == 0.0
    assert np.all(arrays["sigma_independent_rj_k"] > 0)


def test_prepare_records_pdf_and_output_hashes_without_copying_source_pdfs(tmp_path):
    arcade = tmp_path / "0901.0555.pdf"
    lwa = tmp_path / "1804.08581.pdf"
    arcade.write_bytes(b"arcade source")
    lwa.write_bytes(b"lwa source")
    output = tmp_path / "prepared"

    manifest = prepare_external_data(arcade, lwa, output)

    assert (output / "external.npz").exists()
    saved = json.loads((output / "external_manifest.json").read_text())
    assert manifest == saved
    assert saved["source_pdfs"]["ARCADE"]["sha256"] == hashlib.sha256(
        arcade.read_bytes()
    ).hexdigest()
    assert saved["source_pdfs"]["LWA"]["sha256"] == hashlib.sha256(
        lwa.read_bytes()
    ).hexdigest()
    assert not (output / arcade.name).exists()
    with np.load(output / "external.npz", allow_pickle=False) as archive:
        assert archive["frequency_mhz"].shape == (11,)
        assert archive["source_temperature_thermodynamic_k"].shape == (11,)
