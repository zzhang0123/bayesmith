"""Demo C as a bayesmith graph, and the plan ``bayesmith.compile`` derives for it.

Runs in the twin's venv (``README.md``). The forward model of the waterfall
``d`` (``n_time`` LSTs by ``n_freq`` channels):

    a ~ Normal(0, PRIOR_SD^2) per coefficient      (n_time, n_basis)
    u ~ Normal(0, I)                                (7,)
    theta = box(u)                                  the emulator's training box
    d = a Phi^T + 1 c(theta)^T + n,  n ~ Normal(0, sigma^2 I)

with ``Phi`` the orthonormalised physical basis of ``basis.py`` at the order
``recover.py`` chose (or one given on the command line), ``c`` the 21cmVAE
curve through ``global21cm_jax``, and ``sigma`` the twin's 10 mK. The graph
is written with ``sample`` / ``det`` / ``observe``; nothing tells the
compiler which block is affine.

``bayesmith.compile`` partitions the graph, measures the exact block's
conditioning and prints the plan. The printout is saved verbatim as
``plan_<scenario>.txt`` and rendered as ``plan_<scenario>.svg`` and ``.png``
in monospace on a light background. ``--sample n`` also runs the plan's own
sampler for ``n`` draws and records the wall time per draw, so that the
report can say what the bayesmith-side run costs; the recovery figures use
the twin's collapse-and-SMC route (``recover.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import recover
from global21cm import signal21
from global21cm.foreground_beamconv import orthonormal_span

from bayesmith import compile as compile_graph
from bayesmith import declared_partition, det, observe, sample, trace

#: Coefficient prior width, K, on unit-norm columns (the foreground is 1e2-1e4 K).
PRIOR_SD = 1e5


def graph_for(case: common.Scenario, basis_columns: np.ndarray, waterfall: np.ndarray, prior_sd: float = PRIOR_SD):
    """``trace`` the model on this waterfall; returns the graph."""
    phi = jnp.asarray(orthonormal_span(basis_columns))
    freqs = jnp.asarray(common.load_freqs_mhz(case))
    low, high = (jnp.asarray(v) for v in signal21.prior_box())
    n_time, n_basis = waterfall.shape[0], phi.shape[1]

    def model(observed):
        a = sample("fg_coeff", lambda: dist.Normal(jnp.zeros((n_time, n_basis)), prior_sd).to_event(2))
        u = sample("u", lambda: dist.Normal(jnp.zeros(7), 1.0).to_event(1))
        fg = det("foreground", lambda a_: a_ @ phi.T, a)
        theta = det("theta", lambda u_: signal21.box_from_unit_normal(u_, low, high), u)
        t21 = det("t21", lambda th: signal21.curve_kelvin(th, freqs), theta)
        sky = det("sky", lambda f, c: f + c[None, :], fg, t21)
        observe("waterfall", lambda m: dist.Normal(m, common.NOISE_SIGMA_K).to_event(2), sky, obs=observed)

    return trace(model, jnp.asarray(waterfall))


def render(text: str, path: Path, title: str) -> None:
    """Monospace on a light background, the talk's ink colour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lines = text.rstrip("\n").split("\n")
    width = max(len(line) for line in lines)
    fig_w = max(8.0, 0.0905 * width + 0.8)
    fig_h = 0.26 * (len(lines) + 3) + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#fbfaf7")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("#fbfaf7")
    ax.text(0.02, 0.97, title, transform=ax.transAxes, va="top", ha="left", fontsize=11,
            family="Arial", color=common.PALETTE["blue"], weight="bold")  # fmt: skip
    ax.text(0.02, 0.90, text, transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
            family="monospace", color=common.PALETTE["ink"], linespacing=1.35)  # fmt: skip
    for ext in ("svg", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def sample_smoke(plan, n: int, seed: int) -> dict:
    """``n`` draws of the plan's own sampler; the cost per draw and the latents' spread.

    A failure is recorded as its error text, not raised: the smoke measures
    what the plan's own sampler does on this graph, whatever that is.
    """
    t0 = time.perf_counter()
    try:
        posterior = plan.sample(jax.random.key(seed), num_warmup=n, num_samples=n, progress_bar=False)
    except Exception as error:  # noqa: BLE001 - recorded, see the docstring
        return {"draws": n, "warmup": n, "seconds": round(time.perf_counter() - t0, 1),
                "error": f"{type(error).__name__}: {str(error)[:600]}"}  # fmt: skip
    seconds = time.perf_counter() - t0
    draws = posterior.draws if hasattr(posterior, "draws") else posterior
    u = np.asarray(draws["u"]) if "u" in draws else None
    return {"draws": n, "warmup": n, "seconds": round(seconds, 1), "seconds_per_draw": round(seconds / (2 * n), 3),
            "u_mean": None if u is None else u.reshape(-1, 7).mean(axis=0).tolist(),
            "u_sd": None if u is None else u.reshape(-1, 7).std(axis=0).tolist(),
            "posterior_type": type(posterior).__name__}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scenario", nargs="*", choices=list(common.SCENARIOS))
    parser.add_argument("--channel", choices=("tophat", "centre"), default="tophat")
    parser.add_argument("--cells", type=int, help="override the recovery's chosen cell count")
    parser.add_argument("--order", type=int, help="override the chosen Taylor order")
    parser.add_argument("--beam", type=int, help="override the chosen number of beam spectra")
    parser.add_argument("--sample", type=int, default=0, help="also run the plan's sampler for this many draws")
    parser.add_argument("--tag", default="", help="suffix of the output names, e.g. a bayesmith variant")
    parser.add_argument("--prior-sd", type=float, default=PRIOR_SD, help="coefficient prior width, K")
    parser.add_argument("--out", type=Path, default=common.RUNS)
    args = parser.parse_args()
    tag = f"_{args.tag}" if args.tag else ""
    for name in args.scenario or list(common.SCENARIOS):
        case = common.SCENARIOS[name]
        chosen = {"cells": 1, "N": 2, "beam_terms": 0}
        for quick in (False, True):  # the grid's choice is the same on both paths
            recovery = common.results_dir(args.out, quick) / f"recovery_{name}.json"
            if recovery.is_file():
                chosen = json.loads(recovery.read_text())["grid"]["chosen"]
                break
        pick = {"cells": args.cells or chosen["cells"], "N": args.order if args.order is not None else chosen["N"],
                "beam_terms": args.beam if args.beam is not None else chosen["beam_terms"]}  # fmt: skip
        basis = recover.load_basis(case, args.channel, args.out)
        spectra = np.load(case.sim_dir / "beam_svd_spectra.npy")[:, : pick["beam_terms"]] if pick["beam_terms"] else None
        columns = recover.physical_basis(basis[f"cells{pick['cells']}_{args.channel}_grouped"], pick["N"], spectra)
        waterfall = np.load(case.sim_dir / "waterfall.npy")
        t0 = time.perf_counter()
        graph = graph_for(case, columns, waterfall, args.prior_sd)
        plan = compile_graph(graph)
        text = str(plan)
        seconds = time.perf_counter() - t0
        (args.out / f"plan_{name}{tag}.txt").write_text(text + "\n")
        subtitle = (f"Demo C, {name} scenario, {pick['cells']} cell(s), N = {pick['N']}, "
                    f"{pick['beam_terms']} beam spectra, {columns.shape[1]} coefficients per LST")  # fmt: skip
        render(text, args.out / f"plan_{name}{tag}.svg", f"bayesmith.compile(graph) — {subtitle}")
        # The same graph with the block table declared by the modeller: the
        # entry that skips discovery and says so in every block's reason.
        t1 = time.perf_counter()
        declared = declared_partition(graph, [(("fg_coeff",), "gcr"), (("u",), "nuts")])
        declared_text = str(declared)
        declared_seconds = time.perf_counter() - t1
        (args.out / f"plan_{name}{tag}_declared.txt").write_text(declared_text + "\n")
        render(declared_text, args.out / f"plan_{name}{tag}_declared.svg",
               f"bayesmith.declared_partition(graph, [fg_coeff: gcr, u: nuts]) — {subtitle}")  # fmt: skip
        record = {"scenario": name, "basis": pick, "n_basis": int(columns.shape[1]), "n_time": int(waterfall.shape[0]),
                  "prior_sd_k": args.prior_sd, "compile_seconds": round(seconds, 1), "plan": text,
                  "blocks": [{"latents": list(b.latents), "method": b.method} for b in plan.blocks],
                  "declared": {"plan": declared_text, "seconds": round(declared_seconds, 1)},
                  "bayesmith_file": __import__("bayesmith").__file__,
                  "packages": common.versions(("bayesmith", "rheplicant", "jax", "numpyro"))}  # fmt: skip
        if args.sample:
            record["sample_smoke"] = sample_smoke(plan, args.sample, seed=0)
        common.write_json(args.out / f"plan_{name}{tag}.json", record)
        print(text)
        print("--- declared ---")
        print(declared_text)
        print(name, "compile seconds", round(seconds, 1), json.dumps(record.get("sample_smoke", {})))


if __name__ == "__main__":
    main()
