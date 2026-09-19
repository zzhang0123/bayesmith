"""Smooth local spectra, projected through the native Haslam/TRIS response.

Real harmonics have unit full-sky RMS (B00=1), ordered by ell then m=-ell..ell.
The exponential field is transformed on a Gauss-Legendre/longitude grid. Beam
moments retain the native brightness map; their truncation must be validated
against a higher response order and direct integration at posterior positions.
This first comparison conditions on the nominal archived beam, not a fitted beam.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import sph_harm_y


def harmonic_basis(theta, phi, lmax):
    if isinstance(lmax, bool) or not isinstance(lmax, (int, np.integer)) or lmax < 0:
        raise ValueError("lmax must be a nonnegative integer")
    theta, phi = np.broadcast_arrays(theta, phi)
    if not np.all(np.isfinite(theta) & np.isfinite(phi)) or np.any(
        (theta < 0) | (theta > np.pi)
    ):
        raise ValueError("finite angles and colatitude in [0, pi] required")
    columns = []
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            y = sph_harm_y(ell, abs(m), theta, phi)
            columns.append(
                np.sqrt(4 * np.pi if m == 0 else 8 * np.pi)
                * (y.imag if m < 0 else y.real)
            )
    return np.stack(columns, axis=-1)


def spectral_rule(lmax, order=None):
    """Quadrature normalized to sky average; exact on basis products."""
    order = 2 * lmax + 4 if order is None else order
    if order <= lmax:
        raise ValueError("quadrature order must exceed response lmax")
    z, w = np.polynomial.legendre.leggauss(order)
    theta, phi = np.meshgrid(
        np.arccos(z), np.arange(2 * order) * np.pi / order, indexing="ij"
    )
    weights = np.broadcast_to(w[:, None] / (4 * order), theta.shape).ravel()
    basis = harmonic_basis(theta.ravel(), phi.ravel(), lmax)
    return theta.ravel(), phi.ravel(), weights, basis


def harmonic_scales(lmax, monopole_sd, spatial_sd):
    """Each ell contributes decreasing total pointwise prior variance."""
    ell = np.concatenate([np.full(2 * l + 1, l) for l in range(lmax + 1)])
    return np.where(
        ell == 0,
        monopole_sd,
        spatial_sd / np.sqrt((2 * ell + 1) * np.maximum(ell, 1) ** 2),
    )


def map_moments(values, lmax):
    """Sum(values * B_lm) on native RING cells, via the uniterated transform."""
    import healpy as hp

    alm = hp.map2alm(values, lmax=lmax, iter=0, pol=False, use_weights=False)
    area = hp.nside2pixarea(hp.get_nside(values))
    result = []
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            value = alm[hp.Alm.getidx(lmax, ell, abs(m))]
            result.append(
                np.sqrt(4 * np.pi if m == 0 else 8 * np.pi)
                * (-value.imag if m < 0 else value.real)
                / area
            )
    return np.asarray(result)


def prepare(args):
    import healpy as hp
    from limTOD.tris import read_tris_beam_cuts
    from limTOD.tris.beam import tris_cut_beam_response
    from limTOD.tris.geometry import tris_zenith_geometry

    from examples.inference.tris_audit import _pointing_frame

    if args.output.exists():
        raise ValueError("use a fresh output directory")
    args.output.mkdir(parents=True)
    maps = dict(np.load(args.maps, allow_pickle=False))
    h = hp.read_map(args.dsds, dtype=float) - float(maps["reference_cmb_k"])
    source = hp.read_map(args.ds, dtype=float) - h - float(maps["reference_cmb_k"])
    if (
        source.shape != h.shape
        or not np.all(np.isfinite(h))
        or not np.all(np.isfinite(source))
    ):
        raise ValueError("finite aligned native Haslam maps required")
    native_nside = hp.get_nside(h)
    if args.nside < native_nside:
        raise ValueError("quadrature must retain native brightness pixels")
    theta, phi = hp.pix2ang(args.nside, np.arange(hp.nside2npix(args.nside)))
    parent = hp.ang2pix(native_nside, theta, phi)
    theta_c, phi_c = hp.Rotator(coord=["G", "C"])(theta, phi)
    directions = hp.ang2vec(theta_c, phi_c)
    ra_key = "ra_deg_0" if "ra_deg_0" in maps else "ra_deg"
    ra = maps[ra_key]
    if "frequency_mhz" not in maps or not np.array_equal(
        maps["frequency_mhz"], [600.5, 817.8]
    ):
        raise ValueError(
            "continuous TRIS response requires 600.5/817.8 MHz in that order"
        )
    if "ra_deg_1" in maps and not np.array_equal(ra, maps["ra_deg_1"]):
        raise ValueError("shared-beam response requires aligned frequency RA grids")
    geometry = tris_zenith_geometry(ra)
    axes = _pointing_frame(
        geometry.lst_deg,
        geometry.latitude_deg,
        geometry.azimuth_deg,
        geometry.elevation_deg,
        geometry.selfrot_deg,
    )
    cuts = read_tris_beam_cuts(args.archive / "TRIS_Beam_Profile.txt")
    kh, k1, ks = [], [], []
    for row in range(len(ra)):
        bore, axis_e, axis_h, zenith = (axis[row] for axis in axes)
        angle = np.rad2deg(np.arccos(np.clip(directions @ bore, -1, 1)))
        azimuth = np.rad2deg(np.arctan2(directions @ axis_h, directions @ axis_e))
        response = np.where(
            directions @ zenith >= 0, tris_cut_beam_response(cuts, angle, azimuth), 0.0
        )
        if (
            not np.all(np.isfinite(response))
            or response.min() < 0
            or response.sum() <= 0
        ):
            raise ValueError("invalid beam")
        response /= response.sum()
        kh.append(map_moments(response * h[parent], args.lmax))
        k1.append(map_moments(response, args.lmax))
        ks.append(response @ source[parent])
        if row % 20 == 0:
            print(
                f"continuous response nside={args.nside} row={row}/{len(ra)}",
                flush=True,
            )
    theta_q, phi_q, weights, basis = spectral_rule(args.lmax)
    arrays = {
        "kh": kh,
        "k1": k1,
        "source": ks,
        "source_mean": source.mean(),
        "template_min": h.min(),
        "theta": theta_q,
        "phi": phi_q,
        "weights": weights,
        "basis": basis,
        "ra_deg": ra,
        "data": np.stack([maps[f"data_k_{i}"] for i in range(2)], axis=1),
        "sigma": np.stack([maps[f"sigma_k_{i}"] for i in range(2)], axis=1),
        "lmax": args.lmax,
        "nside": args.nside,
    }
    output = args.output / "response.npz"
    np.savez_compressed(output, **arrays)
    import inspect

    paths = [
        Path(inspect.getfile(tris_cut_beam_response)),
        Path(inspect.getfile(tris_zenith_geometry)),
        Path(inspect.getfile(_pointing_frame)),
        args.maps,
        args.ds,
        args.dsds,
        args.archive / "TRIS_Beam_Profile.txt",
        Path(__file__),
    ]
    manifest = {
        "schema": "tris.continuous.nominal_beam.v1",
        "response_lmax": args.lmax,
        "quadrature_nside": args.nside,
        "native_nside": native_nside,
        "boresight_declination_deg": float(geometry.latitude_deg),
        "frequency_mhz": maps["frequency_mhz"].tolist(),
        "beam": "fixed nominal cut beam; exact horizon; no beam parameters fitted",
        "freefree": "not separated: conditional Haslam residual model; limitation retained",
        "source": "signed DS-DSDS, beta=-2.7; subtract its mean from total fixed RSB",
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "response_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "certified": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("maps", "ds", "dsds", "archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--lmax", type=int, default=16)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
