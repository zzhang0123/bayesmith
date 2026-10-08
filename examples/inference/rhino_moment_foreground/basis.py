"""The physical foreground basis: channel-integrated moment responses from SyncMoments.

Runs in the scratch venv that holds SyncMoments 0.4.0 (``README.md``). For
each scenario it writes ``runs/rhino-moment-foreground/basis_<scenario>.npz``
holding, for every cell count ``K`` of :data:`common.CELL_COUNTS`:

* ``cells{K}_grouped``: ``(n_freq, K, N_MAX + 1)``, the brightness-temperature
  response of cell ``c`` at Taylor order ``r`` in the energy displacement,
  ``C_{r0}`` of SyncMoments's ``I_basis`` divided by ``nu^2``;
* ``cells{K}_full``: ``(n_freq, K, n_full)``, every ``(r, s)`` column with
  ``r + s <= N_MAX`` (energy and field derivatives), for the identity check
  and the identifiability figure, with ``full_rs`` naming the columns;
* ``cells{K}_gamma``, ``cells{K}_nu_c_hz``: the pivots.

**The reference population.** One ordered field ``B0`` (5 microgauss),
isotropic pitch and field directions (SyncMoments's continuum kernel,
projected on ``k = 0`` in the field--line-of-sight cosine), no Faraday
rotation (Stokes I only). The energy axis is split into ``K`` cells whose
pivots ``gamma_c`` have critical frequencies ``a_B gamma_c^2`` geometric
between ``nu_min / PIVOT_PAD`` and ``nu_max PIVOT_PAD``; ``K = 1`` is the
band's geometric centre. The manuscript's own broad-population test
(main text, "fig: powerlaw order band") splits a truncated power law into
fixed log-energy cells for the same reason: a single Taylor pivot does not
represent a population whose emission spans the band. The coefficients are
free in the fit, so no power-law index enters the basis; the index names
the population the cells stand for and nothing else.

**The scaling identity.** For the continuum kernel
``d_lnB K = K + d_lngamma K / 2`` (manuscript eq. "continuum scaling"), so
every field-derivative column lies in the span of the energy-derivative
columns of the same cell. :func:`grouping_check` measures that: the
first-order identity ``C_01 = C_00 + C_10 / 2`` (scales ``s_gamma = gamma_c``,
``s_B = B0``) and the projection residual of every ``s >= 1`` column onto
the ``s = 0`` span. Demo C fits the grouped columns.

**Independent check.** :func:`scipy_kernel` evaluates the same kernel with
scipy's Bessel function and the isotropic-direction average, and
:func:`derivative_check` compares ``C_00`` and ``C_10`` with it and with a
central difference in ``ln gamma``. Both numbers are recorded, not asserted.

**Channels.** The twin evaluates its waterfall at the channel centres
(``limtod_jax``), so the basis is built twice: integrated over a top-hat of
the channel width (``tophat``, the talk's "integrate over the channel first")
and at the centre (``centre``, a 1 kHz top-hat). The span difference between
the two is measured and recorded; ``recover.py`` reads one of them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy import integrate, special
from syncmoments.model.basis import build_basis
from syncmoments.model.channels import Channels
from syncmoments.model.index import Truncation
from syncmoments.model.kernels import ContinuumKernel
from syncmoments.model.moments import Reference, Support

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

CHANNEL_KINDS = ("tophat", "centre")
CENTRE_HALF_WIDTH_HZ = 1e3
N_NU = 16  # Gauss-Legendre nodes per channel
N_ETA = 24  # nodes of the field-direction projection
SUPPORT_SPAN = 4.0  # declared support gamma_c / 4 .. 4 gamma_c, B0 / 4 .. 4 B0


def pivots(freqs_mhz, n_cells: int) -> tuple[np.ndarray, np.ndarray]:
    """``(gamma_c, nu_c_hz)`` of ``n_cells`` cells (module docstring)."""
    lo, hi = float(freqs_mhz.min()) * 1e6, float(freqs_mhz.max()) * 1e6
    if n_cells == 1:
        nu_c = np.array([np.sqrt(lo * hi)])
    else:
        nu_c = np.geomspace(lo / common.PIVOT_PAD, hi * common.PIVOT_PAD, n_cells)
    return np.sqrt(nu_c / common.a_b_hz(common.B0_GAUSS)), nu_c


def channels(freqs_mhz, width_mhz: float, kind: str) -> Channels:
    centres = np.asarray(freqs_mhz, dtype=np.float64) * 1e6
    half = 0.5 * width_mhz * 1e6 if kind == "tophat" else CENTRE_HALF_WIDTH_HZ
    return Channels.tophat(centres_hz=centres, widths_hz=np.full(centres.size, half),
                           normalisation="unit_integral", n_nu=N_NU)  # fmt: skip


def cell_columns(ch: Channels, gamma_c: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Every ``(r, s)`` column, ``r + s <= N_MAX``, of one cell: ``(n_freq, n_full)``."""
    reference = Reference(gamma_c, common.B0_GAUSS, depth_ref=0.0, scales=(gamma_c, common.B0_GAUSS, 1.0))
    support = Support(gamma=(gamma_c / SUPPORT_SPAN, gamma_c * SUPPORT_SPAN),
                      B=(common.B0_GAUSS / SUPPORT_SPAN, common.B0_GAUSS * SUPPORT_SPAN), depth=(0.0, 1.0))  # fmt: skip
    basis = build_basis(ContinuumKernel(n_eta=N_ETA), ch, Truncation(0, 0, common.N_MAX, depth_degree=0),
                        reference, support=support, convergence=False, allow_nonsmooth=True)  # fmt: skip
    rows = [(row[2], row[3]) for row in basis.index.h0]  # (l, k, r, s, b) -> (r, s)
    return np.asarray(basis.I_basis, dtype=np.float64), rows


def to_kelvin(columns: np.ndarray, freqs_mhz) -> np.ndarray:
    """Rayleigh-Jeans: ``T = I c^2 / (2 k nu^2)`` per electron column (units absorbed by the fit)."""
    nu = np.asarray(freqs_mhz, dtype=np.float64)[:, None] * 1e6
    return columns * common.C_CGS**2 / (2.0 * common.K_B_CGS * nu**2)


def grouped(columns: np.ndarray, rows: list[tuple[int, int]]) -> np.ndarray:
    """The ``s = 0`` columns in order ``r = 0 .. N_MAX``: ``(n_freq, N_MAX + 1)``."""
    return np.stack([columns[:, rows.index((r, 0))] for r in range(common.N_MAX + 1)], axis=1)


def grouping_check(columns: np.ndarray, rows: list[tuple[int, int]]) -> dict:
    """The scaling identity: field columns inside the energy-column span."""
    c = {rs: columns[:, i] for i, rs in enumerate(rows)}
    first = c[(0, 1)] - (c[(0, 0)] + 0.5 * c[(1, 0)])
    q, _ = np.linalg.qr(grouped(columns, rows))
    residuals = {}
    for rs in rows:
        if rs[1] >= 1:
            col = c[rs]
            residuals[f"{rs[0]},{rs[1]}"] = float(np.linalg.norm(col - q @ (q.T @ col)) / np.linalg.norm(col))
    return {"first_order_identity_rel": float(np.linalg.norm(first) / np.linalg.norm(c[(0, 1)])),
            "field_column_outside_energy_span_rel": residuals}  # fmt: skip


def synch_f(x: float) -> float:
    """``F(x) = x int_x^inf K_5/3``."""
    return x * integrate.quad(lambda t: special.kv(5.0 / 3.0, t), x, np.inf)[0]


def scipy_kernel(nu_hz: float, gamma: float) -> float:
    """``int_0^1 d eta sqrt(1 - eta^2) F(nu / (a_B(B0 sqrt(1 - eta^2)) gamma^2))``, per electron, arbitrary scale."""
    def integrand(eta):
        sin = np.sqrt(1.0 - eta**2)
        return sin * synch_f(nu_hz / (common.a_b_hz(common.B0_GAUSS * sin) * gamma**2)) if sin > 0 else 0.0
    return integrate.quad(integrand, 0.0, 1.0, limit=200)[0]


def derivative_check(ch_centre: Channels, gamma_c: float, freqs_mhz, columns: np.ndarray, rows) -> dict:
    """``C_00`` and ``C_10`` of the centre basis against scipy and a central difference."""
    nu = np.asarray(freqs_mhz, dtype=np.float64) * 1e6
    ref = np.array([scipy_kernel(v, gamma_c) for v in nu])
    h = 1e-3
    up = np.array([scipy_kernel(v, gamma_c * np.exp(h)) for v in nu])
    down = np.array([scipy_kernel(v, gamma_c * np.exp(-h)) for v in nu])
    dlng = (up - down) / (2.0 * h)
    c00, c10 = columns[:, rows.index((0, 0))], columns[:, rows.index((1, 0))]
    scale = float(c00 @ ref / (ref @ ref))  # one constant: per-electron units and the eta weight
    return {"c00_vs_scipy_max_rel": float(np.max(np.abs(c00 - scale * ref) / np.abs(scale * ref))),
            "c10_vs_central_difference_max_rel": float(np.max(np.abs(c10 - scale * dlng)) / np.max(np.abs(c10))),
            "finite_difference_step_lngamma": h}  # fmt: skip


def span_difference(a: np.ndarray, b: np.ndarray) -> float:
    """Largest relative residual of ``b``'s columns off ``a``'s span."""
    q, _ = np.linalg.qr(a)
    r = b - q @ (q.T @ b)
    return float(np.max(np.linalg.norm(r, axis=0) / np.linalg.norm(b, axis=0)))


def build_scenario(case: common.Scenario, out: Path) -> dict:
    freqs = common.load_freqs_mhz(case)
    arrays: dict[str, np.ndarray] = {"freqs_mhz": freqs}
    record: dict = {"scenario": case.name, "channel_width_mhz": case.channel_width_mhz, "cells": {}}
    chans = {kind: channels(freqs, case.channel_width_mhz, kind) for kind in CHANNEL_KINDS}
    t0 = time.perf_counter()
    for n_cells in common.CELL_COUNTS:
        gamma_c, nu_c = pivots(freqs, n_cells)
        entry: dict = {"gamma": gamma_c.tolist(), "nu_c_mhz": (nu_c / 1e6).tolist(), "checks": {}}
        for kind in CHANNEL_KINDS:
            full, groups = [], []
            for c, g in enumerate(gamma_c):
                cols, rows = cell_columns(chans[kind], float(g))
                kelvin = to_kelvin(cols, freqs)
                full.append(kelvin)
                groups.append(grouped(kelvin, rows))
                if c == 0:
                    entry["checks"][kind] = {"identity": grouping_check(cols, rows)}
                    if kind == "centre":
                        entry["checks"][kind]["derivatives"] = derivative_check(chans[kind], float(g), freqs, cols, rows)
            arrays[f"cells{n_cells}_{kind}_full"] = np.stack(full, axis=1)
            arrays[f"cells{n_cells}_{kind}_grouped"] = np.stack(groups, axis=1)
        arrays[f"cells{n_cells}_gamma"], arrays[f"cells{n_cells}_nu_c_hz"] = gamma_c, nu_c
        g_t, g_c = (arrays[f"cells{n_cells}_{k}_grouped"].reshape(freqs.size, -1) for k in CHANNEL_KINDS)
        entry["checks"]["tophat_columns_off_centre_span_rel"] = span_difference(g_c, g_t)
        entry["checks"]["centre_columns_off_tophat_span_rel"] = span_difference(g_t, g_c)
        record["cells"][str(n_cells)] = entry
        print(case.name, "cells", n_cells, "identity", entry["checks"]["tophat"]["identity"]["first_order_identity_rel"],
              "span", entry["checks"]["tophat_columns_off_centre_span_rel"], flush=True)  # fmt: skip
    arrays["full_rs"] = np.array(rows)
    record["full_rs"] = [list(rs) for rs in rows]
    record["seconds"] = round(time.perf_counter() - t0, 1)
    record["provenance"] = {
        "syncmoments": common.versions(("syncmoments", "jax", "equinox", "numpy", "scipy")),
        "python": sys.version.split()[0],
        "freqs_mhz_sha256": common.sha256(case.sim_dir / "freqs_mhz.npy"),
        "B0_gauss": common.B0_GAUSS, "pivot_pad": common.PIVOT_PAD, "n_max": common.N_MAX,
        "n_nu": N_NU, "n_eta": N_ETA, "support_span": SUPPORT_SPAN, "centre_half_width_hz": CENTRE_HALF_WIDTH_HZ,
        "kernel": "syncmoments.model.kernels.ContinuumKernel (isotropic pitch, k=0 field-direction projection)",
    }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"basis_{case.name}.npz", **arrays)
    common.write_json(out / f"basis_{case.name}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario", nargs="*", choices=list(common.SCENARIOS))
    parser.add_argument("--out", type=Path, default=common.RUNS)
    args = parser.parse_args()
    for name in args.scenario or list(common.SCENARIOS):
        build_scenario(common.SCENARIOS[name], args.out)


if __name__ == "__main__":
    main()
