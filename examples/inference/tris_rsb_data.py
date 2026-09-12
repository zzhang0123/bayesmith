"""Prepare published ARCADE 2 and LWA background temperatures offline.

The checked-in transcription is deliberately small and explicit.  The command
requires local source PDFs solely to bind a real run to immutable source bytes;
it never downloads or copies them.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

HC_OVER_K_MHZ_K = 4.799243073e-5
TAU_K = {"LWA": 10.0, "ARCADE": 0.005}
SURVEY_CODES = {"LWA": 0, "ARCADE": 1}
DATA_PATH = Path(__file__).with_name("data") / "tris_rsb_background_v1.json"


@dataclasses.dataclass(frozen=True)
class ExternalRow:
    survey: str
    frequency_mhz: float
    temperature_thermodynamic_k: float
    sigma_thermodynamic_k: float
    table: str
    row: str
    convention: str
    source_arxiv: str
    source_pdf_sha256: str | None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_declared_rows() -> tuple[ExternalRow, ...]:
    payload = json.loads(DATA_PATH.read_text())
    if payload.get("schema") != "bayesmith.tris.rsb.background.v1":
        raise ValueError("unsupported external-background data schema")
    rows = tuple(ExternalRow(**item) for item in payload["rows"])
    if not rows or any(
        row.survey not in SURVEY_CODES
        or row.convention != "thermodynamic_temperature"
        or not np.isfinite(
            [row.frequency_mhz, row.temperature_thermodynamic_k, row.sigma_thermodynamic_k]
        ).all()
        or row.frequency_mhz <= 0
        or row.temperature_thermodynamic_k <= 0
        or row.sigma_thermodynamic_k <= 0
        for row in rows
    ):
        raise ValueError("external-background rows must be finite declared measurements")
    if len(rows) != 11 or len({(row.survey, row.frequency_mhz) for row in rows}) != len(rows):
        raise ValueError("expected eleven unique approved external-background rows")
    if any(row.frequency_mhz in {22.0, 45.0, 408.0, 1420.0} for row in rows):
        raise ValueError("external-background transcription must not reuse literature survey rows")
    return rows


def _rj_derivative(value_k: float, frequency_mhz: float) -> float:
    if not np.isfinite([value_k, frequency_mhz]).all() or value_k <= 0 or frequency_mhz <= 0:
        raise ValueError("RJ conversion requires positive finite temperature and frequency")
    a = HC_OVER_K_MHZ_K * frequency_mhz
    x = a / value_k
    return float(a * np.exp(x) * a / value_k**2 / np.expm1(x) ** 2)


def to_rj_temperature(value_k: float, frequency_mhz: float) -> float:
    if not np.isfinite([value_k, frequency_mhz]).all() or value_k <= 0 or frequency_mhz <= 0:
        raise ValueError("RJ conversion requires positive finite temperature and frequency")
    x = HC_OVER_K_MHZ_K * frequency_mhz / value_k
    return float(value_k * x / np.expm1(x))


def to_rj_sigma(value_k: float, sigma_k: float, frequency_mhz: float) -> float:
    if not np.isfinite(sigma_k) or sigma_k <= 0:
        raise ValueError("RJ uncertainty conversion requires a positive finite uncertainty")
    return _rj_derivative(value_k, frequency_mhz) * sigma_k


def likelihood_arrays(rows: tuple[ExternalRow, ...]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("at least one external-background row is required")
    temperature = np.asarray(
        [to_rj_temperature(row.temperature_thermodynamic_k, row.frequency_mhz) for row in rows]
    )
    sigma = np.asarray(
        [to_rj_sigma(row.temperature_thermodynamic_k, row.sigma_thermodynamic_k, row.frequency_mhz) for row in rows]
    )
    tau = np.asarray(
        [to_rj_sigma(row.temperature_thermodynamic_k, TAU_K[row.survey], row.frequency_mhz) for row in rows]
    )
    if np.any(sigma <= tau):
        raise ValueError("published uncertainty must exceed declared shared calibration scale")
    return {
        "frequency_mhz": np.asarray([row.frequency_mhz for row in rows]),
        "source_temperature_thermodynamic_k": np.asarray(
            [row.temperature_thermodynamic_k for row in rows]
        ),
        "source_sigma_thermodynamic_k": np.asarray(
            [row.sigma_thermodynamic_k for row in rows]
        ),
        "temperature_rj_k": temperature,
        "sigma_rj_k": sigma,
        "tau_rj_k": tau,
        "sigma_independent_rj_k": np.sqrt(sigma**2 - tau**2),
        "survey_code": np.asarray([SURVEY_CODES[row.survey] for row in rows], dtype=np.int32),
        "survey": np.asarray([row.survey for row in rows]),
    }


def survey_covariance(
    sigma_independent_rj_k: np.ndarray,
    survey_code: np.ndarray,
    tau_rj_k: np.ndarray,
) -> np.ndarray:
    sigma, code, tau = map(np.asarray, (sigma_independent_rj_k, survey_code, tau_rj_k))
    if sigma.ndim != code.ndim != tau.ndim or sigma.shape != code.shape or sigma.shape != tau.shape:
        raise ValueError("external covariance arrays must be one-dimensional and aligned")
    if not np.all(np.isfinite(sigma) & np.isfinite(tau) & (sigma > 0) & (tau >= 0)):
        raise ValueError("external covariance scales must be finite and nonnegative")
    return np.diag(sigma**2) + (code[:, None] == code[None, :]) * np.outer(tau, tau)


def prepare_external_data(arcade_pdf: Path, lwa_pdf: Path, output: Path) -> dict:
    arcade_pdf, lwa_pdf, output = map(Path, (arcade_pdf, lwa_pdf, output))
    if not arcade_pdf.is_file() or not lwa_pdf.is_file():
        raise ValueError("--arcade-pdf and --lwa-pdf must name local source files")
    rows = load_declared_rows()
    arrays = likelihood_arrays(rows)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "external.npz", **arrays)
    manifest = {
        "schema": "bayesmith.tris.rsb.external.v1",
        "source_table_sha256": sha256(DATA_PATH),
        "source_pdfs": {
            "ARCADE": {"basename": arcade_pdf.name, "sha256": sha256(arcade_pdf), "arxiv": "0901.0555"},
            "LWA": {"basename": lwa_pdf.name, "sha256": sha256(lwa_pdf), "arxiv": "1804.08581"},
        },
        "conversion": "T_RJ=T_th*x/expm1(x), x=(h/k)*nu/T_th; sigma by analytic derivative",
        "calibration_tau_thermodynamic_k": TAU_K,
        "rows": [dataclasses.asdict(row) for row in rows],
        "input_sha256": sha256(output / "external.npz"),
    }
    (output / "external_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arcade-pdf", type=Path, required=True)
    parser.add_argument("--lwa-pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_external_data(args.arcade_pdf, args.lwa_pdf, args.output)


if __name__ == "__main__":
    main()
