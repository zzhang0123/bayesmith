"""Versioned ARCADE background with its published, already inclusive covariance.

The default product is the six final-2011 ARCADE channels. Combining this with
LWA or a reused Haslam template does not establish independent foreground errors.
Legacy eleven-point products are read only with their explicit legacy schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.inference.tris_rsb_data import (
    _rj_derivative,
    sha256,
    survey_covariance,
    to_rj_temperature,
)

DATA_PATH = Path(__file__).with_name("data") / "tris_rsb_background_v2.json"
SCHEMA = "bayesmith.tris.rsb.external.v2"


def validate_covariance(covariance, count):
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (count, count) or not np.all(np.isfinite(covariance)):
        raise ValueError("external covariance must be a finite aligned square matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=0):
        raise ValueError("external covariance must be symmetric")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("external covariance must be positive definite") from error
    return covariance


def final2011_arrays():
    declared = json.loads(DATA_PATH.read_text())
    if declared.get("schema") != "bayesmith.tris.rsb.background.v2":
        raise ValueError("unsupported background declaration")
    rows = declared["rows"]
    frequency = np.asarray([r["frequency_mhz"] for r in rows])
    temperature = np.asarray([r["temperature_thermodynamic_k"] for r in rows])
    if (
        not len(rows)
        or not np.all(np.isfinite(frequency) & (frequency > 0))
        or not np.all(np.diff(frequency) > 0)
        or not np.all(np.isfinite(temperature) & (temperature > 0))
    ):
        raise ValueError(
            "positive finite temperatures and ordered frequencies required"
        )
    covariance = validate_covariance(declared["covariance_thermodynamic_k2"], len(rows))
    jacobian = np.array([_rj_derivative(t, f) for t, f in zip(temperature, frequency)])
    rj_covariance = covariance * np.outer(jacobian, jacobian)
    return {
        "frequency_mhz": frequency,
        "temperature_rj_k": np.array(
            [to_rj_temperature(t, f) for t, f in zip(temperature, frequency)]
        ),
        "covariance_rj_k2": rj_covariance,
        "sigma_rj_k": np.sqrt(np.diag(rj_covariance)),
        "source_temperature_thermodynamic_k": temperature,
        "source_covariance_thermodynamic_k2": covariance,
        "survey": np.array([r["survey"] for r in rows]),
        "survey_code": np.zeros(len(rows), dtype=np.int32),
    }, declared


def prepare_final2011(source_pdf, output):
    arrays, declared = final2011_arrays()
    source_pdf, output = Path(source_pdf), Path(output)
    if (
        not source_pdf.is_file()
        or sha256(source_pdf) != declared["source_pdf"]["sha256"]
    ):
        raise ValueError("source PDF does not match the audited 2011 final journal PDF")
    if output.exists():
        raise ValueError(
            "use a fresh output directory; historical inputs are immutable"
        )
    output.mkdir(parents=True)
    np.savez_compressed(output / "external.npz", **arrays)
    manifest = {
        **declared,
        "schema": SCHEMA,
        "source_table_sha256": sha256(DATA_PATH),
        "input_sha256": sha256(output / "external.npz"),
        "likelihood": "multivariate_normal; covariance already includes foreground/instrument terms",
        "conversion": "absolute Planck temperature to RJ; covariance J C J^T at published temperatures",
        "independent_of_haslam_template": False,
        "additional_calibration_latents": False,
    }
    (output / "external_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def load_external(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "external_manifest.json").read_text())
    if manifest.get("schema") not in (SCHEMA, "bayesmith.tris.rsb.external.v1"):
        raise ValueError("unsupported external-background schema")
    if sha256(directory / "external.npz") != manifest.get("input_sha256"):
        raise ValueError("external archive hash differs from manifest")
    with np.load(directory / "external.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    required = {"frequency_mhz", "temperature_rj_k", "survey", "survey_code"}
    if required - arrays.keys():
        raise ValueError("external input lacks required arrays")
    count = len(arrays["frequency_mhz"])
    if not count or any(arrays[k].shape != (count,) for k in required):
        raise ValueError("external rows must be nonempty and aligned")
    if not np.all(
        np.isfinite(arrays["frequency_mhz"]) & (arrays["frequency_mhz"] > 0)
    ) or not np.all(np.isfinite(arrays["temperature_rj_k"])):
        raise ValueError("invalid external frequency or temperature")
    if manifest["schema"] == SCHEMA:
        if "covariance_rj_k2" not in arrays:
            raise ValueError("v2 requires the inclusive covariance")
        if manifest.get("additional_calibration_latents") is not False:
            raise ValueError(
                "v2 must not double-count calibration already in covariance"
            )
        validate_covariance(arrays["covariance_rj_k2"], count)
    else:
        needed = {"sigma_independent_rj_k", "tau_rj_k"}
        if needed - arrays.keys():
            raise ValueError("legacy input lacks uncertainty arrays")
        survey_covariance(
            arrays["sigma_independent_rj_k"], arrays["survey_code"], arrays["tau_rj_k"]
        )
    return arrays, manifest


def covariance_for_rows(arrays, selection):
    """Select the covariance submatrix, not a conditional precision submatrix."""
    indices = np.arange(len(arrays["frequency_mhz"]))[selection]
    covariance = arrays.get("covariance_rj_k2")
    if covariance is None:
        covariance = survey_covariance(
            arrays["sigma_independent_rj_k"], arrays["survey_code"], arrays["tau_rj_k"]
        )
    validate_covariance(covariance, len(arrays["frequency_mhz"]))
    return np.asarray(covariance)[np.ix_(indices, indices)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arcade-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare_final2011(args.arcade_pdf, args.output)
    print(json.dumps({"release": manifest["release"], "rows": len(manifest["rows"])}))


if __name__ == "__main__":
    main()
