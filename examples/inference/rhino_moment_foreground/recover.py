"""Demo C's order grid, the 21 cm posterior at each truncation order, and its scores.

Runs in the twin's venv (``README.md``). For one scenario:

1. **The grid.** Every candidate basis ``(K cells, Taylor order N, n_beam
   beam-SVD spectra)`` is scored exactly as Demo A's order grid is
   (``global21cm.strategies._moment_rows``): the whole waterfall's chi^2 with
   the per-LST coefficients fitted, minimised over the 40000-draw prior bank
   of 21 cm curves, its p-value on ``n_data - n_time n_basis - 7`` degrees of
   freedom, the fit check at ``p >= 0.01``, and BIC. The rule is Demo A's
   ``bic_among_passing``. Each candidate also carries the sampler-free
   amplitude forecast (``collapse.amplitude_forecast``): the bias that
   foreground misfit alone puts on the 21 cm amplitude, and its sigma.
2. **The sweeps.** At the chosen ``(K, n_beam)`` and at ``K = 1`` (the
   brief's single reference population), every order ``N = 0 .. N_MAX`` is
   fitted: the coefficients are integrated out exactly
   (``collapse.per_lst_basis``, the flat-prior limit of the Gaussian block
   that ``bayesmith.compile`` names GCR-exact in ``plan.txt``), and the seven
   latents are sampled with the twin's tempered SMC. The scores are the
   twin's figure of merit (``scoring.score``, ``goodness_of_fit``,
   ``laplace_trace``) against the same truth and prior bank.
3. **The oracle and Demos A and B** are read from the twin's saved
   ``results/analysis/posteriors.npz`` and ``fom.json`` when present, so the
   comparison columns are the published ones.

Writes ``recovery_<scenario>.json`` and ``posteriors_<scenario>.npz`` under
``runs/rhino-moment-foreground/full/`` (or ``quick/``). ``--quick`` runs one
SMC seed of 4000 particles per posterior instead of four of 20000; the grid
is the same on both paths.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from global21cm import (
    analyse,
    collapse,
    scenario,
    scoring,
    signal21,
    smc,
    strategies,
)
from global21cm.foreground_beamconv import orthonormal_span

BEAM_TERMS = (0, 4)
QUICK = {"seeds": (1,), "n": 4000, "n_move": 20}
FULL = {"seeds": (1, 2, 3, 4), "n": 20000, "n_move": 30}
TWIN_MODELS = ("oracle", "beamconv", "physical")


def load_basis(case: common.Scenario, kind: str, out: Path) -> dict:
    with np.load(out / f"basis_{case.name}.npz") as z:
        wanted = (f"_{kind}_grouped", f"_{kind}_full", "_gamma", "_nu_c_hz")
        return {k: z[k] for k in z.files if k.endswith(wanted) or k in ("freqs_mhz", "full_rs")}


def physical_basis(grouped: np.ndarray, order: int, beam_spectra=None) -> np.ndarray:
    """Unit-norm columns ``(n_freq, K (N+1) max(n_beam, 1))``: cell-major, order, beam spectrum minor."""
    columns = grouped[:, :, : order + 1].reshape(grouped.shape[0], -1)
    if beam_spectra is not None and beam_spectra.shape[1] > 0:
        columns = (columns[:, :, None] * beam_spectra[:, None, :]).reshape(columns.shape[0], -1)
    return columns / np.linalg.norm(columns, axis=0)[None, :]


def candidate_row(waterfall, curves, basis, n_cells, order, n_beam, noiseless, t21, sigma=collapse.SIGMA0) -> dict:
    """One grid cell: Demo A's statistics (``strategies._moment_rows``) plus the amplitude forecast."""
    n_time, n_freq = waterfall.shape
    q = orthonormal_span(basis)
    perp = np.eye(n_freq) - q @ q.T
    chi2 = (np.sum((waterfall @ perp) ** 2) - 2 * curves @ (perp @ waterfall.sum(axis=0))
            + n_time * np.einsum("nf,fg,ng->n", curves, perp, curves))  # fmt: skip
    best = float(chi2.min()) / sigma**2
    n_par = n_time * q.shape[1] + len(signal21.THETA_NAMES)
    dof = waterfall.size - n_par
    p = float(stats.chi2.sf(best, dof))
    item = collapse.per_lst_basis(waterfall, q)
    return {"cells": n_cells, "N": order, "beam_terms": n_beam, "n_basis": int(q.shape[1]), "chi2_min": best,
            "dof": dof, "p": p, "passes": p >= strategies.FIT_CHECK_P, "chi2_per_datum": best / waterfall.size,
            "bic": best + n_par * np.log(waterfall.size), "rank": item.rank,
            "forecast": collapse.amplitude_forecast(item, t21, noiseless),
            "compressed_chi2_truth": float(item.chi2(t21)[0])}  # fmt: skip


def grid(sim: dict, basis: dict, curves) -> list[dict]:
    rows = []
    n_freq = sim["waterfall"].shape[1]
    for n_cells in common.CELL_COUNTS:
        grouped = basis[f"cells{n_cells}_{basis['kind']}_grouped"]
        for n_beam in range(BEAM_TERMS[0], BEAM_TERMS[1] + 1):
            spectra = sim["beam_svd_spectra"][:, :n_beam] if n_beam else None
            for order in range(common.N_MAX + 1):
                if n_cells * (order + 1) * max(n_beam, 1) > n_freq - 3:
                    continue
                b = physical_basis(grouped, order, spectra)
                rows.append(candidate_row(sim["waterfall"], curves, b, n_cells, order, n_beam, sim["noiseless"], sim["t21_truth"]))
                print("grid", json.dumps({k: rows[-1][k] for k in ("cells", "N", "beam_terms", "n_basis", "p", "bic")}), flush=True)
    return rows


def posterior(item: collapse.Collapsed, freqs, fine, settings: dict) -> tuple[dict, np.ndarray, list, np.ndarray]:
    runs = [smc.run(smc.log_likelihood(item.design, item.data, freqs), seed, n=settings["n"], n_move=settings["n_move"])
            for seed in settings["seeds"]]  # fmt: skip
    u = np.concatenate([r["u"] for r in runs])
    loglik = np.concatenate([r["loglik"] for r in runs])
    per_seed = [{"seed": s, "log_z": r["log_z"], "steps": r["steps"]} for r, s in zip(runs, settings["seeds"])]
    return analyse._draws(u, freqs, fine), loglik, per_seed, u


def score_sweep(sim, basis, context, n_cells, n_beam, settings, store: dict) -> list[dict]:
    """Every order at one ``(K, n_beam)``: collapse, SMC, the twin's scores."""
    freqs, fine, truth, prior = context["freqs"], context["fine"], context["truth"], context["prior"]
    grouped = basis[f"cells{n_cells}_{basis['kind']}_grouped"]
    spectra = sim["beam_svd_spectra"][:, :n_beam] if n_beam else None
    rows = []
    for order in range(common.N_MAX + 1):
        b = physical_basis(grouped, order, spectra)
        if b.shape[1] > freqs.size - 3:
            rows.append({"N": order, "skipped": f"{b.shape[1]} columns on {freqs.size} channels"})
            continue
        item = collapse.per_lst_basis(sim["waterfall"], orthonormal_span(b))
        t0 = time.perf_counter()
        draws, loglik, per_seed, u = posterior(item, freqs, fine, settings)
        seconds = time.perf_counter() - t0
        gof = scoring.goodness_of_fit(item, draws, loglik, seed=5)
        entry = {"N": order, "cells": n_cells, "beam_terms": n_beam, "n_basis": int(b.shape[1]), "rank": item.rank,
                 **scoring.score(draws, truth, prior, fine), "gof": gof,
                 "forecast": collapse.amplitude_forecast(item, truth["curve"], sim["noiseless"]),
                 "compressed_chi2_truth": float(item.chi2(truth["curve"])[0]),
                 "marginal_chi2_truth": item.marginal_chi2(truth["curve"]),
                 "laplace_trace": scoring.laplace_trace(item.design, freqs, truth["u"]),
                 "tail": analyse.tail_decomposition(draws, fine),
                 "smc": {"runs": per_seed, "seconds": round(seconds, 1), "particles": len(u)}}  # fmt: skip
        entry["calibrated"] = analyse.calibrated_verdict(entry["calibrated_signal"], gof)
        store[f"cells{n_cells}_beam{n_beam}_N{order}__u"] = u
        store[f"cells{n_cells}_beam{n_beam}_N{order}__loglik"] = loglik
        rows.append(entry)
        brief = {k: entry[k] for k in ("N", "n_basis", "rank", "ser_over_prior", "depth_mk", "calibrated")}
        brief["bias_sigma"] = entry["forecast"]["bias_sigma"]
        print("sweep", json.dumps(brief), flush=True)
    return rows


def twin_results(case: common.Scenario, freqs, fine, context, store: dict) -> dict:
    """The oracle's and Demos A and B's saved particles and scores, when the twin has them."""
    analysis = common.TWIN / "results" / "analysis"
    out: dict = {"source": str(analysis)}
    fom_path, post_path = analysis / "fom.json", analysis / "posteriors.npz"
    if not (fom_path.is_file() and post_path.is_file()):
        out["missing"] = True
        return out
    fom = json.loads(fom_path.read_text())[case.name]
    with np.load(post_path) as z:
        for model in TWIN_MODELS:
            u = z[f"{case.name}__{model}__u"]
            store[f"{model}__u"] = u
            out[model] = {k: fom[model][k] for k in ("ser", "ser_over_prior", "variance_over_prior", "bias_fraction",
                                                        "z2_tail", "depth_quantile", "position_quantile", "depth_mk",
                                                        "position_mhz", "calibrated", "laplace_trace", "tail")}  # fmt: skip
            out[model]["forecast"] = fom[model]["forecast"]
            out[model]["gof_passes"] = fom[model]["gof"]["passes"]
            for key in ("eta_laplace", "eta_smc", "eta_smc_core"):
                if key in fom[model]:
                    out[model][key] = fom[model][key]
    out["fom_sha256"], out["posteriors_sha256"] = common.sha256(fom_path), common.sha256(post_path)
    return out


def run_scenario(case: common.Scenario, kind: str, settings: dict, out: Path, quick: bool) -> dict:
    t_start = time.perf_counter()
    results = common.results_dir(out, quick)
    results.mkdir(parents=True, exist_ok=True)
    sim = {n: np.load(case.sim_dir / f"{n}.npy") for n in ("waterfall", "noiseless", "fg_truth", "t21_truth",
                                                           "freqs_mhz", "beam_svd_spectra")}  # fmt: skip
    basis = {**load_basis(case, kind, out), "kind": kind}
    freqs = sim["freqs_mhz"]
    twin_case = scenario.SCENARIOS[case.name]
    fine = scoring.fine_freqs(twin_case)
    bank_u, bank_curves = strategies.prior_bank(tuple(float(v) for v in freqs))
    context = {"freqs": freqs, "fine": fine, "truth": analyse._truth(twin_case, fine),
               "prior": analyse._draws(bank_u[:4000], freqs, fine)}  # fmt: skip
    rows = grid(sim, basis, bank_curves)
    chosen = strategies.choose_order(rows, "bic_among_passing")
    store: dict = {"freqs": freqs, "fine": fine, "prior__u": bank_u[:4000]}
    sweeps = {"chosen": score_sweep(sim, basis, context, chosen["cells"], chosen["beam_terms"], settings, store)}
    if chosen["cells"] != 1:
        sweeps["single_reference"] = score_sweep(sim, basis, context, 1, chosen["beam_terms"], settings, store)
    record = {
        "scenario": case.name, "channel_kind": kind, "settings": settings,
        "grid": {"rule": "bic_among_passing", "fit_check_p": strategies.FIT_CHECK_P, "table": rows,
                 "chosen": {k: chosen[k] for k in ("cells", "N", "beam_terms", "n_basis", "p", "bic", "passes")},
                 "n_passing": int(sum(r["passes"] for r in rows)), "n_candidates": len(rows)},
        "sweeps": sweeps,
        "twin": twin_results(case, freqs, fine, context, store),
        "truth": {"depth_k": context["truth"]["depth"], "where_mhz": context["truth"]["where"]},
        "provenance": {
            "inputs_sha256": {n: common.sha256(case.sim_dir / f"{n}.npy") for n in sim},
            "basis_sha256": common.sha256(out / f"basis_{case.name}.npz"),
            "packages": common.versions(("rheplicant", "bayesmith", "jax", "numpy", "scipy", "numpyro", "global21cm-jax", "limtod")),
            "rheplicant_git_head": common.git_head(common.TWIN), "python": sys.version.split()[0],
            "bank": {"n": strategies.N_BANK, "seed": strategies.BANK_SEED}, "smc_seeds": list(settings["seeds"]),
        },
        "seconds": round(time.perf_counter() - t_start, 1),
    }  # fmt: skip
    np.savez(results / f"posteriors_{case.name}.npz", **store)
    common.write_json(results / f"recovery_{case.name}.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario", nargs="*", choices=list(common.SCENARIOS))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--channel", choices=("tophat", "centre"), default="tophat")
    parser.add_argument("--out", type=Path, default=common.RUNS)
    args = parser.parse_args()
    settings = QUICK if args.quick else FULL
    for name in args.scenario or list(common.SCENARIOS):
        record = run_scenario(common.SCENARIOS[name], args.channel, settings, args.out, args.quick)
        print(name, "chosen", json.dumps(record["grid"]["chosen"]), "seconds", record["seconds"])


if __name__ == "__main__":
    main()
