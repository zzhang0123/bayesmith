"""Independent angular and spectral checks of continuous TRIS field fits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from examples.inference.tris_continuous import harmonic_basis


def direct_means(dsds, ds, maps, archive, parameters, *, nside=1024):
    """Direct continuous local sky, independently rotated with spherical formulas."""
    import healpy as hp
    from examples.TRIS.fixed_background import BASELINE
    from examples.TRIS.joint_sky_beam import cmb_rj_temperature, zero_levels
    from limTOD.tris import read_tris_beam_cuts
    from limTOD.tris.beam import tris_cut_beam_response
    from limTOD.tris.geometry import tris_zenith_geometry

    bundle = dict(np.load(maps, allow_pickle=False))
    h = hp.read_map(dsds, dtype=float)
    source = hp.read_map(ds, dtype=float) - h
    h -= float(bundle["reference_cmb_k"])
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    parent = hp.ang2pix(hp.get_nside(h), theta, phi)
    max_l = max(int(np.sqrt(len(p["beta_coeff"]))) - 1 for p in parameters)
    max_l = max(
        max_l, max(int(np.sqrt(len(p["log_amplitude_coeff"]))) - 1 for p in parameters)
    )
    basis = harmonic_basis(theta, phi, max_l)
    nu = np.array([600.5, 817.8])
    b408 = BASELINE[0] * 0.408 ** BASELINE[1] - source.mean()
    spectra = []
    zeros = []
    for p in parameters:
        a, b = p["log_amplitude_coeff"], p["beta_coeff"]
        scale = np.exp(
            (basis[:, : len(a)] @ a)[:, None]
            + (basis[:, : len(b)] @ b)[:, None] * np.log(nu / 408.0)
        )
        g = h[parent] + p["monopole_k"] - b408
        if g.min() <= 0:
            raise ValueError("direct validation encountered nonpositive template")
        sky = (
            g[:, None] * scale
            + (source[parent] - source.mean())[:, None] * (nu / 408.0) ** -2.7
        )
        sky += (
            np.asarray(cmb_rj_temperature(nu))
            + BASELINE[0] * (nu / 1000.0) ** BASELINE[1]
        )
        spectra.append(sky)
        zeros.append(np.asarray(zero_levels(p["zero_standard"])))
    del basis, parent, h, source
    fields = np.concatenate(spectra, axis=1)
    del spectra
    theta, phi = hp.Rotator(coord=["G", "C"])(theta, phi)
    dec = np.pi / 2 - theta
    geometry = tris_zenith_geometry(bundle["ra_deg_0"])
    latitude = np.deg2rad(geometry.latitude_deg)
    cuts = read_tris_beam_cuts(archive / "TRIS_Beam_Profile.txt")
    result = []
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    for row, lst in enumerate(geometry.lst_deg):
        hour = phi - np.deg2rad(lst)
        cos_hour = np.cos(hour)
        up = sin_dec * sin_lat + cos_dec * cos_lat * cos_hour
        north = sin_dec * cos_lat - cos_dec * sin_lat * cos_hour
        east = cos_dec * np.sin(hour)
        roll = np.deg2rad(geometry.selfrot_deg[row])
        e = -north * np.cos(roll) + east * np.sin(roll)
        h = east * np.cos(roll) + north * np.sin(roll)
        angle = np.rad2deg(np.arccos(np.clip(up, -1, 1)))
        azimuth = np.rad2deg(np.arctan2(h, e))
        weights = np.where(up >= 0, tris_cut_beam_response(cuts, angle, azimuth), 0.0)
        result.append(weights @ fields / weights.sum())
        if row % 20 == 0:
            print(
                f"independent continuous oracle nside={nside} row={row}/120", flush=True
            )
    return (
        np.asarray(result).reshape(len(result), len(parameters), 2).transpose(1, 0, 2)
        - np.asarray(zeros)[:, None, :]
    )


def predict_parameters(archive, parameters, lmax):
    from examples.TRIS.continuous_background import (
        continuous_prediction,
        prepare_operator,
    )
    from examples.TRIS.joint_sky_beam import zero_levels

    operator = prepare_operator(archive, lmax)
    result = []
    for p in parameters:
        a, b = p["log_amplitude_coeff"], p["beta_coeff"]
        ba = jnp.asarray(
            harmonic_basis(operator["theta"], operator["phi"], int(np.sqrt(len(a))) - 1)
        )
        bb = jnp.asarray(
            harmonic_basis(operator["theta"], operator["phi"], int(np.sqrt(len(b))) - 1)
        )
        result.append(
            np.asarray(
                continuous_prediction(
                    operator,
                    ba,
                    bb,
                    jnp.asarray(a),
                    jnp.asarray(b),
                    p["monopole_k"],
                    zero_levels(p["zero_standard"]),
                )
            )
        )
    return np.asarray(result)


def error_summary(delta, sigma):
    return {
        "max_abs_k": float(np.max(np.abs(delta))),
        "max_abs_archive_sigma": float(np.max(np.abs(delta) / sigma)),
        "rms_archive_sigma": float(np.sqrt(np.mean((delta / sigma) ** 2))),
    }


def main():
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "response512",
        "response1024",
        "ds",
        "dsds",
        "maps",
        "archive",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--fit", type=Path, action="append", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("use a fresh output directory")
    args.output.mkdir(parents=True)
    a512 = dict(np.load(args.response512, allow_pickle=False))
    a1024 = dict(np.load(args.response1024, allow_pickle=False))
    rows, means = [], []
    names = []
    for fit in args.fit:
        samples = dict(np.load(fit / "samples.npz", allow_pickle=False))
        use = ("log_amplitude_coeff", "beta_coeff", "monopole_k", "zero_standard")
        flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in samples.items() if k in use}
        indices = np.linspace(0, len(flat["monopole_k"]) - 1, 32, dtype=int)
        parameters = [{k: v[i] for k, v in flat.items()} for i in indices]
        low = predict_parameters(a512, parameters, 12)
        high = predict_parameters(a512, parameters, 20)
        finer = predict_parameters(a1024, parameters, 20)
        row = {
            "fit": str(fit),
            "draws": indices.tolist(),
            "response_12_vs_20": error_summary(low - high, a512["sigma"]),
            "quadrature_512_vs_1024": error_summary(high - finer, a512["sigma"]),
        }
        rows.append(row)
        means.append({k: v.mean(0) for k, v in flat.items()})
        names.append(str(fit))
        print(json.dumps(row), flush=True)
    oracle = direct_means(
        args.dsds, args.ds, args.maps, args.archive, means, nside=1024
    )
    surrogate = predict_parameters(a1024, means, 20)
    for i, row in enumerate(rows):
        row["independent_spherical_direct_mean"] = error_summary(
            surrogate[i] - oracle[i], a512["sigma"]
        )
        row["spectral_pass"] = row["response_12_vs_20"]["max_abs_archive_sigma"] < 0.1
        row["direct_pass"] = (
            row["independent_spherical_direct_mean"]["max_abs_archive_sigma"] < 0.1
        )
        row["angular_pass"] = (
            row["quadrature_512_vs_1024"]["max_abs_archive_sigma"] < 0.3
        )
    paths = [
        Path(__file__),
        args.response512,
        args.response1024,
        args.ds,
        args.dsds,
        args.maps,
    ]
    record = {
        "rows": rows,
        "pass": all(
            r["spectral_pass"] and r["direct_pass"] and r["angular_pass"] for r in rows
        ),
        "scope": "32 retained positions per fit plus independent posterior-mean sky; not uniform prior certification",
        "sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }
    np.savez_compressed(
        args.output / "oracle.npz", direct=oracle, surrogate=surrogate, names=names
    )
    (args.output / "validation.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
