"""Independent density checks for the final ARCADE inclusive covariance."""

import hashlib
import json

import jax
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from bayesmith import trace
from bayesmith.graph.evaluate import evaluate, log_joint
from examples.inference import tris_rsb_external as external
from examples.inference.tris_rsb_initialization import fitting_problem
from examples.inference.tris_rsb_sky import model, model_inputs
from tests.test_tris_rsb_sky import tiny_bundle


def test_final_data_and_covariance_preserve_published_correlations():
    arrays, declared = external.final2011_arrays()
    np.testing.assert_array_equal(
        arrays["frequency_mhz"], [3200, 3410, 7980, 8330, 9720, 10490]
    )
    np.testing.assert_array_equal(
        arrays["source_temperature_thermodynamic_k"],
        [2.787, 2.770, 2.761, 2.743, 2.731, 2.738],
    )
    covariance = arrays["source_covariance_thermodynamic_k2"]
    assert covariance[2, 3] == pytest.approx(86e-6)
    assert covariance[0, 1] == pytest.approx(6.78e-6)
    assert covariance[4, 5] == pytest.approx(1.19e-6)
    assert declared["known_source_count_subtraction"] is False
    assert "tau_rj_k" not in arrays
    selected = external.covariance_for_rows(arrays, [4, 0, 2])
    np.testing.assert_array_equal(
        selected, arrays["covariance_rj_k2"][np.ix_([4, 0, 2], [4, 0, 2])]
    )


def test_correlated_likelihood_matches_scipy_and_has_no_duplicate_calibration():
    with jax.enable_x64(True):
        arrays, _ = external.final2011_arrays()
        inputs = model_inputs(tiny_bundle(), arrays, ("ARCADE",), True)
        graph = trace(model, *inputs)
        assert "calibration_standard" not in graph.latents
        decode, residual, _, _ = fitting_problem(graph, inputs)
        x = np.array([1.0, 1.1, 0.9, -2.7, -2.8, -2.9, 0.1, -0.2, 8.0, 0.5, -2.6])
        y = x.copy()
        y[[0, 3, 6, 8, 9]] += [0.2, 0.1, -0.4, 0.7, 0.2]
        delta = -0.5 * (
            np.sum(np.asarray(residual(y)) ** 2) - np.sum(np.asarray(residual(x)) ** 2)
        )
        assert float(
            log_joint(graph, decode(y)) - log_joint(graph, decode(x))
        ) == pytest.approx(delta, abs=2e-10)
        diagonal = {
            **arrays,
            "covariance_rj_k2": np.diag(np.diag(arrays["covariance_rj_k2"])),
        }
        diagonal_graph = trace(
            model, *model_inputs(tiny_bundle(), diagonal, ("ARCADE",), True)
        )
        mean = np.asarray(evaluate(graph, decode(x))["external_background_mean"])
        expected = multivariate_normal.logpdf(
            arrays["temperature_rj_k"], mean, arrays["covariance_rj_k2"]
        ) - multivariate_normal.logpdf(
            arrays["temperature_rj_k"], mean, diagonal["covariance_rj_k2"]
        )
        assert abs(expected) > 1
        assert float(
            log_joint(graph, decode(x)) - log_joint(diagonal_graph, decode(x))
        ) == pytest.approx(expected, abs=2e-10)


def test_versioned_preparation_checks_source_and_refuses_overwrite(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"test source bytes")
    declared = json.loads(external.DATA_PATH.read_text())
    declared["source_pdf"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    declaration = tmp_path / "declaration.json"
    declaration.write_text(json.dumps(declared))
    monkeypatch.setattr(external, "DATA_PATH", declaration)
    output = tmp_path / "prepared"
    external.prepare_final2011(source, output)
    arrays, manifest = external.load_external(output)
    assert manifest["additional_calibration_latents"] is False
    assert arrays["covariance_rj_k2"].shape == (6, 6)
    with pytest.raises(ValueError, match="fresh"):
        external.prepare_final2011(source, output)
    source.write_bytes(b"different publication")
    with pytest.raises(ValueError, match="PDF"):
        external.prepare_final2011(source, tmp_path / "wrong")
    with (output / "external.npz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        external.load_external(output)


@pytest.mark.parametrize(
    "covariance",
    [
        np.eye(2),
        [[1.0, 2.0], [2.0, 1.0]],
        [[1.0, 0.1], [0.2, 1.0]],
        [[1.0, 0.0], [0.0, np.nan]],
    ],
)
def test_malformed_covariance_is_rejected(covariance):
    with pytest.raises(ValueError):
        external.validate_covariance(
            covariance, 3 if np.array_equal(covariance, np.eye(2)) else 2
        )
