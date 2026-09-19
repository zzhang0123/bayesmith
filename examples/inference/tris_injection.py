"""P2 synthetic injection and recovery for the TRIS sky/background model.

Generates whitened data from a declared truth (optionally with a different
region split, to test misattribution) and measures how well a same-budget fit
recovers it.  The generator is an independent numpy path, not the graph, and
the fit is the production graph, so a recovery failure is a real statement
about the model rather than about one shared function.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tris_identifiability import NAMES_M0, NAMES_M1, to_vector

__all__ = [
    "inject",
    "recovery_rows",
    "shifted_regions",
    "truth_values",
    "write_inputs",
]


def truth_values(
    *,
    amplitude=1.6,
    beta=-2.75,
    zero_standard=(0.0, 0.0),
    haslam_monopole_K=0.0,
    calibration_standard=(0.0, 0.0),
    rsb_amplitude=None,
    rsb_beta=None,
):
    """A parameter block in the model's own ordering."""
    values = {
        "amplitude": np.full(3, float(amplitude)),
        "beta": np.full(3, float(beta)),
        "zero_standard": np.asarray(zero_standard, dtype=float),
        "haslam_monopole_K": np.asarray(float(haslam_monopole_K)),
        "calibration_standard": np.asarray(calibration_standard, dtype=float),
    }
    if rsb_amplitude is not None:
        values["rsb_amplitude"] = np.asarray(float(rsb_amplitude))
        values["rsb_beta"] = np.asarray(float(rsb_beta))
    return values


def shifted_regions(latitude_deg, boundaries):
    """Region codes from |galactic latitude| with different boundaries."""
    return np.digitize(np.abs(np.asarray(latitude_deg, dtype=float)), list(boundaries))


def inject(
    bundle,
    external,
    included_surveys,
    truth,
    include_rsb,
    rng,
    *,
    generator_bundle=None,
):
    """Return (bundle, external) whose data are the truth plus unit noise.

    generator_bundle lets the data come from a different region split while the
    fit still uses the original one, which is the misattribution test.
    """
    from .tris_forward_baseline import (
        predict_from_inputs,
        rsb_temperature,
        zero_levels,
    )
    from .tris_rsb_sky import model_inputs

    generator = bundle if generator_bundle is None else generator_bundle
    region = np.asarray(generator["region"], dtype=int)
    template = np.asarray(bundle["template_k"], dtype=float)
    frequencies = np.asarray(bundle["frequency_mhz"], dtype=float)
    amplitude = np.asarray(truth["amplitude"], dtype=float)
    beta = np.asarray(truth["beta"], dtype=float)
    levels = zero_levels(truth["zero_standard"])
    monopole = float(np.asarray(truth["haslam_monopole_K"]))
    if include_rsb:
        rsb_amplitude = float(np.asarray(truth["rsb_amplitude"]))
        rsb_beta = float(np.asarray(truth["rsb_beta"]))
        background_408 = float(rsb_temperature(rsb_amplitude, rsb_beta, np.array([408.0]))[0])
    else:
        rsb_amplitude = rsb_beta = 0.0
        background_408 = 0.0

    # Inject in RING space and project, because the whitened data must equal the
    # compression of the ring data or the bundle fails its own parity check.
    new_bundle = {name: np.array(value, copy=True) for name, value in bundle.items()}
    for index, frequency in enumerate(frequencies):
        operator = np.asarray(bundle[f"operator_{index}"], dtype=float)
        sigma = np.asarray(bundle[f"sigma_k_{index}"], dtype=float)
        projection = np.asarray(bundle[f"ring_projection_{index}"], dtype=float)
        scaling = amplitude[region] * (frequency / 408.0) ** beta[region]
        background = (
            float(rsb_temperature(rsb_amplitude, rsb_beta, frequency)) if include_rsb else 0.0
        )
        sky = (template + monopole - background_408) * scaling + float(
            bundle["cmb_k"][index]
        ) + background
        ring = operator @ sky - levels[index] + rng.normal(scale=sigma)
        new_bundle[f"data_k_{index}"] = ring
        new_bundle[f"whitened_data_{index}"] = projection @ (ring / sigma)

    inputs = model_inputs(bundle, external, included_surveys, include_rsb)
    _tris_inputs, external_mean = predict_from_inputs(inputs, truth)

    labels = np.asarray(external["survey"])
    mask = np.isin(labels, list(included_surveys))
    sigma = np.asarray(external["sigma_independent_rj_k"], dtype=float)[mask]
    injected = np.asarray(external["temperature_rj_k"], dtype=float).copy()
    injected[mask] = external_mean + rng.normal(scale=sigma)
    new_external = {name: np.array(value, copy=True) for name, value in external.items()}
    new_external["temperature_rj_k"] = injected
    return new_bundle, new_external


def write_inputs(
    directory,
    bundle,
    external,
    source_manifest,
    source_external_manifest,
):
    """Persist a synthetic bundle with self-consistent provenance hashes."""
    from .tris_prepare import sha256

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "maps.npz", **bundle)
    manifest = dict(source_manifest)
    manifest["input_sha256"] = sha256(directory / "maps.npz")
    manifest["synthetic"] = True
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(directory / "external.npz", **external)
    external_manifest = dict(source_external_manifest)
    external_manifest["input_sha256"] = sha256(directory / "external.npz")
    external_manifest["synthetic"] = True
    (directory / "external_manifest.json").write_text(
        json.dumps(external_manifest, indent=2) + "\n"
    )
    return directory


def recovery_rows(truth, draws, include_rsb):
    """Per-coordinate truth, posterior mean, sd, and pull."""
    names = NAMES_M1 if include_rsb else NAMES_M0
    rows = []
    for name in names:
        if name not in truth or name not in draws:
            continue
        target = np.atleast_1d(np.asarray(truth[name], dtype=float)).ravel()
        values = np.asarray(draws[name], dtype=float)
        # Draws arrive as (chains, draws, ...); collapse the sample axes.
        values = (
            values.reshape(-1, *values.shape[2:]) if values.ndim >= 2 else values.reshape(-1)
        )
        mean = np.atleast_1d(values.mean(axis=0))
        sd = np.atleast_1d(values.std(axis=0, ddof=1))
        lower = np.atleast_1d(np.quantile(values, 0.025, axis=0))
        upper = np.atleast_1d(np.quantile(values, 0.975, axis=0))
        for index in range(target.size):
            pull = (mean[index] - target[index]) / sd[index] if sd[index] > 0 else 0.0
            rows.append(
                {
                    "parameter": name if target.size == 1 else f"{name}[{index}]",
                    "truth": float(target[index]),
                    "mean": float(mean[index]),
                    "sd": float(sd[index]),
                    "pull": float(pull),
                    "covered_95": bool(lower[index] <= target[index] <= upper[index]),
                }
            )
    return rows


def truth_vector(truth, include_rsb):
    return to_vector(truth, include_rsb)
