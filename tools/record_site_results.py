"""Record tutorial results and figures from the downloadable snippets.

Run with JAX_ENABLE_X64=1 .venv/bin/python tools/record_site_results.py.
--check verifies provenance, figure hashes, and closed-form reference columns;
it does not rerun sampling or require platform-identical Monte Carlo output.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import runpy
import warnings
from pathlib import Path

from record_site_plans import ROOT, SNIPPETS, _require_x64, environment

OUTPUT = ROOT / "site/assets/results"
SOURCES = ("first_model.py", "five_tasks.py", "nonlinear.py")
BLUE, ORANGE, GRAY = "#126c88", "#b65a21", "#596779"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_level():
    return 12 / 12.25, math.sqrt(1 / 12.25)


def check(directory=OUTPUT):
    record = json.loads((directory / "run.json").read_text())
    problems = []
    if record.get("recorder_sha256") != digest(Path(__file__)):
        problems.append("Recorder changed; regenerate tutorial results")
    for name in SOURCES:
        if record["sources"].get(name) != digest(SNIPPETS / name):
            problems.append(f"Source changed: {name}")
    for name, checksum in record["figures"].items():
        path = directory / name
        if not path.is_file() or digest(path) != checksum:
            problems.append(f"Figure changed or missing: {name}")
    partial = ROOT / "site/assets/plans/partial.json"
    if record["partial_record_sha256"] != digest(partial):
        problems.append("Partial-structure record changed; regenerate its figure")
    for row, value in zip(
        record["tables"]["first"]["rows"], exact_level(), strict=True
    ):
        if not math.isclose(row["exact"], value, rel_tol=1e-12):
            problems.append(f"Incorrect closed-form {row['quantity']}")
    return problems


def record():
    _require_x64()
    import jax
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bayesmith.artifacts import ComputeBudget
    from bayesmith.evaluation import check_posterior

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": GRAY,
            "text.color": GRAY,
            "svg.fonttype": "none",
            "svg.hashsalt": "bayesmith-tutorials",
            "figure.facecolor": "white",
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stored = {
        "environment": environment(),
        "sources": {name: digest(SNIPPETS / name) for name in SOURCES},
        "recorder_sha256": digest(Path(__file__)),
        "figures": {},
        "tables": {},
        "stdout": {},
        "data": {},
    }

    def run(name, function="main"):
        namespace = runpy.run_path(str(SNIPPETS / name))
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = namespace[function]()
        stored["stdout"][name] = stream.getvalue()
        print(f"Executed {name}", flush=True)
        return result

    def save(fig, name):
        fig.savefig(OUTPUT / name, bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
        stored["figures"][name] = digest(OUTPUT / name)

    def table(name, columns, rows, settings):
        stored["tables"][name] = {
            "columns": columns,
            "rows": rows,
            "settings": settings,
        }

    def density(x, mean, sd):
        return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    first = np.asarray(run("first_model.py").samples["level"])
    mean, sd = exact_level()
    stored["data"]["first_draws"] = first.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    x = np.linspace(-1, 2.5, 500)
    ax.hist(
        first, bins=22, density=True, color=BLUE, alpha=0.25, label="256 exact draws"
    )
    ax.plot(x, density(x, mean, sd), color=BLUE, lw=2, label="Closed-form posterior")
    ax.plot(x, density(x, 0, 2), color=GRAY, ls="--", label="Prior")
    ax.plot([0.8, 1, 1.2], [0, 0, 0], "|", color=ORANGE, ms=15, label="Measurements")
    ax.set(
        xlabel="Latent level",
        ylabel="Probability density",
        title="Data narrow the uncertainty in the level",
    )
    ax.legend(fontsize=10)
    save(fig, "first-posterior.svg")
    table(
        "first",
        [
            {"key": "quantity", "label": "Quantity"},
            {"key": "exact", "label": "Closed form", "format": ".4f"},
            {"key": "sampled", "label": "Recorded draws", "format": ".4f"},
        ],
        [
            {
                "quantity": "Posterior mean",
                "exact": mean,
                "sampled": float(first.mean()),
            },
            {"quantity": "Posterior sd", "exact": sd, "sampled": float(first.std())},
        ],
        {"seed": 7, "draws": 256},
    )

    partial_path = ROOT / "site/assets/plans/partial.json"
    partial = json.loads(partial_path.read_text())
    stored["partial_record_sha256"] = digest(partial_path)
    rows = partial["tables"]["recovery"]["rows"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), layout="constrained")
    y = np.arange(len(rows))
    ax.errorbar(
        [r["exact_mean"] for r in rows],
        y + 0.12,
        xerr=[r["exact_sd"] for r in rows],
        fmt="o",
        color=ORANGE,
        capsize=3,
        label="Closed form: mean ± 1 sd",
    )
    ax.errorbar(
        [r["sampled_mean"] for r in rows],
        y - 0.12,
        xerr=[r["sampled_sd"] for r in rows],
        fmt="o",
        color=BLUE,
        capsize=3,
        label="Recorded draws: mean ± 1 sd",
    )
    ax.scatter(
        [r["truth"] for r in rows], y, marker="x", color=GRAY, label="Generating value"
    )
    ax.set(
        yticks=y,
        yticklabels=[r["name"] for r in rows],
        xlabel="Parameter value",
        title="Conditional exact draws agree with the joint Gaussian reference",
    )
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.15))
    save(fig, "partial-recovery.svg")

    results = run("five_tasks.py", "run_workflow")
    latent = np.asarray(results["posterior"].representation.draws[0].value).ravel()
    predicted = np.asarray(results["predictive"].replicated_draws[0].value)
    simulated = np.asarray(results["simulation"].observation_draws[0].value)
    point = float(np.asarray(results["point"].values[0].value))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gate = check_posterior(
            results["graph"],
            results["posterior"],
            key=jax.random.key(90),
            budget=ComputeBudget(draws=512),
            model_ref=results["model_ref"],
        )
    stored["check_warnings"] = list(dict.fromkeys(str(item.message) for item in caught))
    stored["data"].update(
        {
            "posterior": latent.tolist(),
            "predictive": predicted.tolist(),
            "simulation": simulated.tolist(),
        }
    )
    table(
        "workflow",
        [
            {"key": "result", "label": "Result"},
            {"key": "value", "label": "Recorded output"},
        ],
        [
            {
                "result": "PosteriorResult",
                "value": f"{latent.size} level draws; mean {latent.mean():.4f}",
            },
            {"result": "PointEstimateResult", "value": f"Posterior mean {point:.4f}"},
            {
                "result": "PredictiveResult",
                "value": f"Replicated signal shape {predicted.shape}",
            },
            {
                "result": "SimulationResult",
                "value": f"Prior simulation shape {simulated.shape}",
            },
            {
                "result": "EvidenceResult",
                "value": f"log p(data) = {results['evidence'].log_evidence:.6f}",
            },
            {
                "result": "Artifact round trip",
                "value": "Identity preserved (asserted during execution)",
            },
        ],
        {"seed": 7, "posterior_draws": 256, "prior_draws": 32},
    )
    table(
        "gate",
        [
            {"key": "quantity", "label": "Quantity"},
            {"key": "value", "label": "Recorded output"},
        ],
        [
            {"quantity": "Operational status", "value": str(gate.status)},
            {"quantity": "Verdict", "value": str(gate.verdict)},
            *[
                {"quantity": finding.code, "value": finding.message}
                for finding in gate.findings
            ],
            *[
                {"quantity": "Diagnostic warning", "value": warning}
                for warning in stored["check_warnings"]
            ],
        ],
        {"key": 90, "requested_draws": 512},
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    bins = np.linspace(-6, 6, 45)
    ax.hist(
        simulated.ravel(),
        bins=bins,
        density=True,
        histtype="step",
        color=GRAY,
        lw=1.5,
        label="Prior simulations (32 scalar observations)",
    )
    ax.hist(
        predicted.ravel(),
        bins=bins,
        density=True,
        color=BLUE,
        alpha=0.35,
        label="Posterior predictions (256 datasets)",
    )
    ax.plot([0.8, 1, 1.2], [0, 0, 0], "|", color=ORANGE, ms=18, label="Observed data")
    ax.set(
        xlabel="Signal in a replicated measurement",
        ylabel="Marginal density",
        title="Conditioning changes which observations the model predicts",
    )
    ax.legend(fontsize=9)
    save(fig, "predictive-simulation.svg")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    bins = np.linspace(-0.8, 2.8, 35)
    ax.hist(
        latent,
        bins=bins,
        density=True,
        histtype="step",
        lw=2,
        color=ORANGE,
        label="Latent level: posterior draws",
    )
    ax.hist(
        predicted.ravel(),
        bins=bins,
        density=True,
        color=BLUE,
        alpha=0.3,
        label="New measurements: posterior predictions",
    )
    ax.set(
        xlabel="Value",
        ylabel="Marginal density",
        title="A new measurement also carries observation noise",
    )
    ax.legend(fontsize=9)
    save(fig, "posterior-predictive.svg")

    decay = run("nonlinear.py")
    rates = np.asarray(decay.representation.draws[0].value).ravel()
    grid = np.linspace(0, 4, 200)
    curves = np.exp(-rates[:, None] * grid)
    low, median, high = np.quantile(curves, [0.05, 0.5, 0.95], axis=0)
    stored["data"]["decay_rates"] = rates.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.fill_between(
        grid, low, high, color=BLUE, alpha=0.2, label="90% interval for mean signal"
    )
    ax.plot(grid, median, color=BLUE, label="Median mean signal")
    ax.errorbar(
        np.arange(5),
        [1.02, 0.57, 0.30, 0.21, 0.12],
        yerr=0.1,
        fmt="o",
        color=ORANGE,
        capsize=3,
        label="Data ± known noise sd",
    )
    ax.set(
        xlabel="Time",
        ylabel="Signal",
        title="Nonlinear decay: a short NUTS demonstration",
    )
    ax.legend(fontsize=9)
    save(fig, "decay-fit.svg")
    table(
        "decay",
        [
            {"key": "quantity", "label": "Quantity"},
            {"key": "value", "label": "Recorded output"},
        ],
        [
            {"quantity": "Rate mean", "value": f"{rates.mean():.4f}"},
            {"quantity": "Rate sd", "value": f"{rates.std():.4f}"},
            {
                "quantity": "Backend",
                "value": f"{decay.run.backend.name} {decay.run.backend.version}",
            },
            {"quantity": "Termination", "value": decay.run.termination.message},
        ],
        {"compile_key": 1, "sample_key": 2, "warmup": 100, "draws": 100, "chains": 1},
    )
    (OUTPUT / "run.json").write_text(json.dumps(stored, indent=2) + "\n")
    print(
        f"Recorded {len(stored['figures'])} figures and {len(stored['tables'])} tables in {OUTPUT}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=OUTPUT,
        help="directory to verify with --check",
    )
    args = parser.parse_args()
    if args.check:
        problems = check(args.record_dir)
        if problems:
            raise SystemExit("\n".join(problems))
        print("Tutorial result provenance, figures and closed-form columns verified.")
    else:
        record()


if __name__ == "__main__":
    main()
