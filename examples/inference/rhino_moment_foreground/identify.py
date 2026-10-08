"""Which foreground moment combinations the waterfall constrains (Demo C, item 4).

Runs in the twin's venv. The manuscript's two reductions ("What the
synthetic SED constrains"), applied to the twin's per-LST model:

1. **Exact identities.** For the continuum kernel every field-derivative
   column is a combination of energy-derivative columns of the same cell
   (``basis.py``), so the ``K C(N+2, 2)`` columns with ``r + s <= N`` reduce
   to ``K (N + 1)`` without changing any spectrum the model can make. The
   count is reported from the basis file; the identity residuals are in
   ``basis_<scenario>.json``.
2. **The noise threshold.** In the declared scaling (each cell's columns
   divided by that cell's zeroth-order amplitude in a least-squares fit of the
   LST-median spectrum, so that a coefficient is a dimensionless moment and
   the zeroth one is of order one), the per-LST response ``Phi`` (``n_freq``
   channels at ``sigma`` = 10 mK) has the noise-whitened SVD
   ``Phi / sigma = U D V^T``; combination ``beta_i = V_i^T a`` has noise
   standard deviation ``1 / D_ii``. The manuscript retains ``1 / D_ii <= 0.1``;
   both that count and the count at 0.01 are reported.

One LST's channels are the unit, because the coefficients are free per LST;
sharing them across the 96 LSTs would divide every standard deviation by
``sqrt(96)`` and is also reported.

Writes ``identifiability_<scenario>.json`` beside the recovery it read.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import recover

THRESHOLDS = (0.1, 0.01)


def cell_amplitudes(grouped: np.ndarray, median_spectrum: np.ndarray) -> np.ndarray:
    """Least-squares amplitude of each cell's zeroth-order column on the LST-median spectrum."""
    zeroth = grouped[:, :, 0]
    amp, *_ = np.linalg.lstsq(zeroth, median_spectrum, rcond=None)
    floor = 1e-3 * np.max(np.abs(amp))
    return np.where(np.abs(amp) < floor, floor, np.abs(amp))


def scaled_response(grouped: np.ndarray, order: int, amplitudes: np.ndarray, beam_spectra=None) -> np.ndarray:
    """``(n_freq, K (N+1) max(n_beam, 1))`` in K per unit dimensionless coefficient, cell-major."""
    columns = grouped[:, :, : order + 1] * amplitudes[None, :, None]
    columns = columns.reshape(columns.shape[0], -1)
    if beam_spectra is not None and beam_spectra.shape[1] > 0:
        # Beam spectra carry the shape of the normalised beam; their scale is
        # the first spectrum's, so the factor keeps the coefficients comparable.
        spectra = beam_spectra / np.max(np.abs(beam_spectra[:, :1]))
        columns = (columns[:, :, None] * spectra[:, None, :]).reshape(columns.shape[0], -1)
    return columns


def whitened_svd(response: np.ndarray, sigma: float) -> dict:
    _, d, vt = np.linalg.svd(response / sigma, full_matrices=False)
    sd = 1.0 / d
    return {"singular_values": d.tolist(), "combination_sd": sd.tolist(),
            "retained": {str(t): int(np.sum(sd <= t)) for t in THRESHOLDS},
            "retained_shared_lst": {str(t): int(np.sum(sd / np.sqrt(common.N_TIME) <= t)) for t in THRESHOLDS},
            "vectors": vt.tolist()}  # fmt: skip


def column_labels(n_cells: int, order: int, n_beam: int) -> list[str]:
    labels = []
    for c in range(n_cells):
        for r in range(order + 1):
            for b in range(max(n_beam, 1)):
                labels.append(f"c{c} r{r}" + (f" b{b}" if n_beam else ""))
    return labels


def analyse_scenario(case: common.Scenario, kind: str, out: Path, quick: bool) -> dict:
    results = common.results_dir(out, quick)
    recovery = json.loads((results / f"recovery_{case.name}.json").read_text())
    chosen = recovery["grid"]["chosen"]
    basis = recover.load_basis(case, kind, out)
    waterfall = np.load(case.sim_dir / "waterfall.npy")
    median = np.median(waterfall, axis=0)
    spectra = np.load(case.sim_dir / "beam_svd_spectra.npy")
    record = {"scenario": case.name, "channel_kind": kind, "chosen": chosen, "sigma_k": common.NOISE_SIGMA_K, "cases": {}}
    for label, n_cells, order, n_beam in (("chosen", chosen["cells"], chosen["N"], chosen["beam_terms"]),
                                          ("chosen_cells_N_max", chosen["cells"], common.N_MAX, chosen["beam_terms"]),
                                          ("single_reference_N_max", 1, common.N_MAX, chosen["beam_terms"])):  # fmt: skip
        grouped = basis[f"cells{n_cells}_{kind}_grouped"]
        amplitudes = cell_amplitudes(grouped, median)
        response = scaled_response(grouped, order, amplitudes, spectra[:, :n_beam] if n_beam else None)
        n_full = n_cells * comb(order + 2, 2) * max(n_beam, 1)
        record["cases"][label] = {
            "cells": n_cells, "N": order, "beam_terms": n_beam,
            "columns_full_rs": n_full, "columns_after_identities": int(response.shape[1]),
            "cell_amplitudes_k": amplitudes.tolist(), "labels": column_labels(n_cells, order, n_beam),
            **whitened_svd(response, common.NOISE_SIGMA_K),
        }  # fmt: skip
        brief = record["cases"][label]
        print(case.name, label, "full", n_full, "grouped", response.shape[1], "retained", brief["retained"], flush=True)
    common.write_json(results / f"identifiability_{case.name}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario", nargs="*", choices=list(common.SCENARIOS))
    parser.add_argument("--channel", choices=("tophat", "centre"), default="tophat")
    parser.add_argument("--quick", action="store_true", help="read the quick path's recovery")
    parser.add_argument("--out", type=Path, default=common.RUNS)
    args = parser.parse_args()
    for name in args.scenario or list(common.SCENARIOS):
        analyse_scenario(common.SCENARIOS[name], args.channel, args.out, args.quick)


if __name__ == "__main__":
    main()
