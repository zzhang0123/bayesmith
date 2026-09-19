"""Build the D15 record from actual run artifacts; no acceptance from log prose."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from examples.inference.tris_bright_sources import cas_a_flux_jy, rj_integral


def record(root, rho):
    b = root / "runs/tris-local-fields-20260915"
    r = rho / "runs/tris-local-fields-20260915"
    entries = {}

    def read(path):
        raw = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "record": json.loads(raw),
        }

    for key, relative in (
        ("a4em2_1970_screen", "preflight-a4em2-q16/summary.json"),
        ("a8em4_1970_screen", "preflight-a8em4-q16/summary.json"),
        ("a8em4_1978_screen", "preflight-sparse-a8em4-q32-1978/summary.json"),
        ("a4em2_simulation", "simulation-a4em2-q16/summary.json"),
        ("a8em4_1970_interrupted", "simulation-a8em4-q32/summary.json"),
        ("a8em4_1978_simulation", "simulation-sparse-a8em4-q32-1978/summary.json"),
        ("quadrature16_32", "quadrature-a8em4.json"),
        ("quadrature32_64", "quadrature-sparse-q32-q64-1978.json"),
    ):
        entries[key] = read(r / relative)
        if "simulation" in key or "interrupted" in key:
            directory = (r / relative).parent
            completed = len(list(directory.glob("chain*.npz")))
            entries[key]["actual_saved_chains"] = completed
            entries[key]["pilot_present"] = (directory / "pilot_samples.npz").exists()
            if (directory / "pilot.json").exists():
                entries[key]["pilot"] = read(directory / "pilot.json")
            claimed = entries[key]["record"].get("completed_chains")
            if claimed is not None and claimed != completed:
                raise ValueError(
                    "recorded chain count differs from actual saved chains"
                )
    entries["d13_screen"] = read(
        rho / "runs/tris-spatial-modes-20260914/real16-prescreen/summary.json"
    )
    for name in ("ppc.json", "posterior_quadrature.json", "failed_chain_audit.json", "leapfrog_audit.json"):
        path = r / "simulation-sparse-a8em4-q32-1978" / name
        if path.exists():
            entries[name] = read(path)
    stress = r / "fixed-step-stress/summary.json"
    if stress.exists():
        entries["fixed_step_stress"] = read(stress)
    tests = {}
    for directory, name in (
        (r, "rho-regression"), (r, "rho-fixed-step"), (r, "rho-count"),
        (b, "bayes-regression"),
    ):
        path = directory / "tests" / f"{name}.xml"
        if path.exists():
            suites = list(ET.parse(path).getroot().iter("testsuite"))
            tests[name] = {
                k: sum(int(s.attrib.get(k, 0)) for s in suites)
                for k in ("tests", "failures", "errors", "skipped")
            }
            tests[name].update(
                exit_code=int(path.with_suffix(".exit").read_text()),
                junit=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
    manifest = read(b / "prepared-sparse-a8em4-q32-1978/manifest.json")
    with np.load(
        b / "prepared-sparse-a8em4-q32-1978/source_operators.npz", allow_pickle=False
    ) as f:
        h, response = f["haslam"], f["scan"]
    frequencies = np.array([408.0, 600.5, 817.8, 1420.0, 2427.8])
    scenarios = []
    for epoch, segment, role in (
        (1970.0, 1, "historical control, not CasA observation epoch"),
        (1978 + 7 / 12, 2, "August boundary"),
        (1978.7, 2, "conditional center"),
        (1978 + 9 / 12, 2, "September boundary"),
        (1998.0, 3, "TRIS early boundary"),
        (2000.0, 3, "TRIS conditional center"),
        (2001.0, 3, "TRIS late boundary"),
        (2016.0, 3, "reference comparison"),
    ):
        flux = cas_a_flux_jy(frequencies, epoch, segment=segment)
        q = rj_integral(frequencies, flux)
        scenarios.append(
            {
                "epoch": epoch,
                "segment": segment,
                "role": role,
                "flux_jy": flux.tolist(),
                "reference_haslam_cell408_k": float(np.max(h * q[0])),
                "reference_tris_max_k": np.max(response[:, None] * q, axis=0).tolist(),
            }
        )
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (b / "sources").glob("*.pdf")
    }
    return {
        "schema": "tris.local_fields_study.v1",
        "scientific_certification": False,
        "runs": entries,
        "tests": tests,
        "main_input": manifest,
        "source_pdf_sha256": sources,
        "cas_a_scenarios": scenarios,
        "frequency_mhz": frequencies.tolist(),
        "program_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def plot(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["D13\n53", "Local\n260*", "Local\n980*", "Local\n980"]
    screens = [
        result["runs"][name]["record"]["screen"]
        for name in ("a4em2_1970_screen", "a8em4_1970_screen", "a8em4_1978_screen")
    ]
    d13 = result["runs"]["d13_screen"]["record"]["initial_screen"]
    rms = [d13["haslam"]["rms"]] + [s["haslam"]["rms"] for s in screens]
    gain = [d13["optical_gain_log_prior_sd"]] + [
        s["optical_gain_log_prior_sd"] for s in screens
    ]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    ax[0, 0].bar(labels, rms, color=["#b55648", "#d49248", "#48979e", "#26777f"])
    ax[0, 0].set(
        ylabel="Haslam optimized residual RMS [K]",
        title="Expression checks, not posterior evidence",
    )
    ax[0, 1].bar(labels, gain, color="#688ea9")
    ax[0, 1].axhline(-5, color="gray", linestyle="--")
    ax[0, 1].axhline(5, color="gray", linestyle="--")
    ax[0, 1].set(ylabel="Optical log gain / prior SD", title="Calibration pressure")
    selected = [result["cas_a_scenarios"][i] for i in (0, 2, 5, 7)]
    ax[1, 0].bar(
        [str(s["epoch"]) for s in selected],
        [s["flux_jy"][0] / 1000 for s in selected],
        color="#91806a",
    )
    ax[1, 0].set(
        ylabel="Conditional Cas A flux at408MHz [kJy]",
        xlabel="Epoch",
        title="Explicit temporal segments; no added flux data",
    )
    q = [
        result["runs"][name]["record"]
        for name in ("quadrature16_32", "quadrature32_64")
    ]
    ax[1, 1].plot(
        ["16 to32", "32 to64"],
        [x["scan"]["rms"][1] * 1000 for x in q],
        "o-",
        label="600.5MHz",
    )
    ax[1, 1].plot(
        ["16 to32", "32 to64"],
        [x["scan"]["rms"][2] * 1000 for x in q],
        "s-",
        label="817.8MHz",
    )
    ax[1, 1].set(
        ylabel="Prediction difference RMS [mK]",
        xlabel="Quadrature NSIDE",
        title="One fixed injected field, unchanged observations",
    )
    ax[1, 1].legend()
    fig.suptitle(
        "D15: local fields and variable Galactic sources\n*1970 source epoch retained only as historical controls",
        fontsize=13,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    result = record(root, root.parent / "rheplicant")
    output = root / "examples/inference/data/tris_local_fields_20260915"
    output.with_suffix(".json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    plot(result, output.with_suffix(".png"))
    if "failed_chain_audit.json" in result["runs"]:
        plot_failed_chain(result, output.with_name(output.name + "_sampling.png"))


def plot_failed_chain(result, path):
    """Display rejected-chain diagnostics without presenting posterior intervals."""
    import matplotlib.pyplot as plt

    run = result["runs"]["a8em4_1978_simulation"]
    directory = Path(run["path"]).parent
    with np.load(directory / "chain0.npz", allow_pickle=False) as f:
        eta, zero = f["log_em_coordinate"][:, 0], f["zero_h"]
    with np.load(directory / "extra0.npz", allow_pickle=False) as f:
        divergent = f["diverging"].astype(bool)
    draw = np.arange(len(zero))
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    for axis, values, label in (
        (ax[0, 0], eta, "Optical EM monopole coordinate"),
        (ax[0, 1], zero, "Haslam zero [K]"),
    ):
        axis.plot(draw, values, lw=0.8, color="#26777f")
        axis.scatter(draw[divergent], values[divergent], color="#b55648", s=13,
                     label="Divergence flag at saved state", zorder=3)
        axis.set(xlabel="Saved draw", ylabel=label)
    ax[0, 0].legend(fontsize=8)
    ax[1, 0].scatter(eta, zero, c="#26777f", alpha=0.25, s=5)
    ax[1, 0].scatter(eta[divergent], zero[divergent], c="#b55648", s=15)
    ax[1, 0].set(xlabel="Optical EM monopole coordinate", ylabel="Haslam zero [K]",
                title="Failed-chain coupling; no posterior interpretation")
    audit = result["runs"]["leapfrog_audit.json"]["record"]
    for step in sorted({r["step"] for r in audit["trajectories"]}, reverse=True):
        rows = [r for r in audit["trajectories"] if r["step"] == step]
        ax[1, 1].plot([str(r["draw"]) for r in rows],
                      [max(r["max_abs_delta_h"]) for r in rows], "o-",
                      label=f"Step {step:.4f}")
    ax[1, 1].axhline(1000, color="gray", linestyle="--", lw=0.8)
    ax[1, 1].set(yscale="log", xlabel="Saved starting state",
                ylabel="Largest |energy error| across four momenta",
                title="Matched momenta and integration time")
    ax[1, 1].legend(fontsize=8)
    fig.suptitle("D15 simulation stopped: 40 divergences / 1000 draws\n"
                 "Diagnostic states only; not accepted posterior samples", fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
