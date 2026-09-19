"""Freeze D12 observations while refining integration of fixed smooth fields."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.inference.tris_continuous import harmonic_basis, harmonic_scales
from examples.inference.tris_halpha import optical_selection
from examples.inference.tris_physical_prepare import CYGNUS_A_RA_DEC, read_nested


def prepare(
    observation_dir, quadrature_dir, optical_dir, output, *, lmax_a=4, lmax_em=2
):
    import healpy as hp
    from astropy.coordinates import SkyCoord

    if min(lmax_a, lmax_em) < 0:
        raise ValueError("nonnegative field orders required")
    with np.load(observation_dir / "data.npz", allow_pickle=False) as f:
        data = {k: np.asarray(f[k]) for k in f.files}
    with np.load(quadrature_dir / "data.npz", allow_pickle=False) as f:
        fine = {k: np.asarray(f[k]) for k in f.files}
    nobs, nquad = int(data["nside"]), int(fine["nside"])
    if nquad < nobs or nquad > 512 or nquad < 2 * max(lmax_a, lmax_em, 2):
        raise ValueError(
            "quadrature must refine observations and resolve requested modes"
        )
    theta, phi = hp.pix2ang(nquad, np.arange(hp.nside2npix(nquad)))
    parent = hp.ang2pix(nobs, theta, phi)
    cell_weight = np.full(parent.size, (nobs / nquad) ** 2)
    intensity = read_nested(optical_dir / "halpha.fits", "R")
    error = read_nested(optical_dir / "halpha_error.fits", "R")
    flags = read_nested(optical_dir / "halpha_mask.fits", "")
    ebv = read_nested(optical_dir / "ebv.fits", "magnitudes")
    theta_native, _ = hp.pix2ang(512, np.arange(intensity.size))
    valid = optical_selection(
        intensity, error, flags, ebv, 90 - np.rad2deg(theta_native)
    )
    fraction = hp.ud_grade(valid.astype(float), nobs)
    attenuation = np.exp(-0.4 * np.log(10) * 2.51 * ebv / 3)
    optical_weight = (
        hp.ud_grade(valid * attenuation, nquad)
        * cell_weight
        / np.maximum(fraction[parent], 1e-30)
    )
    # Unobserved cells carry no optical likelihood. Their weights may be zero.
    check = np.bincount(parent, weights=optical_weight, minlength=len(data["haslam"]))
    if not np.allclose(
        check[data["ha_index"]], data["ha_attenuation"], rtol=2e-12, atol=1e-12
    ):
        raise ValueError(
            "optical mask/dust convention differs from frozen observations"
        )
    source = SkyCoord(*CYGNUS_A_RA_DEC, unit="deg", frame="icrs").galactic
    data.update(
        parent=parent,
        cell_weight=cell_weight,
        optical_weight=optical_weight,
        beam=fine["beam"],
        source_response=fine["source_response"],
        ha_total_sigma=np.sqrt(data["ha_sigma"] ** 2 + (0.2 * data["ha_data"]) ** 2),
        quadrature_nside=np.array(nquad),
        source_em_basis=harmonic_basis(np.pi / 2 - source.b.rad, source.l.rad, lmax_em),
    )
    for name, lmax, mean, mono_sd, angular_sd in (
        ("a", lmax_a, np.log(30.0), 1.5, 1.0),
        ("em", lmax_em, np.log(10.0), 2.0, 1.5),
        ("beta", 2, -2.8, 0.3, 0.15),
    ):
        center = np.zeros((lmax + 1) ** 2)
        center[0] = mean
        data[f"{name}_basis"] = harmonic_basis(theta, phi, lmax)
        data[f"{name}_mean"] = center
        data[f"{name}_sd"] = harmonic_scales(lmax, mono_sd, angular_sd)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "data.npz", **data)
    inputs = [observation_dir / "data.npz", quadrature_dir / "data.npz"] + [
        optical_dir / name
        for name in ("halpha.fits", "halpha_error.fits", "halpha_mask.fits", "ebv.fits")
    ]
    record = {
        "schema": "tris.physical_modes_input.v1",
        "kind": "real",
        "observation_nside": nobs,
        "quadrature_nside": nquad,
        "lmax_a": lmax_a,
        "lmax_em": lmax_em,
        "lmax_beta": 2,
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "data_sha256": hashlib.sha256((output / "data.npz").read_bytes()).hexdigest(),
        "prior": "independent fixed angular harmonic coefficients of log A/log EM/beta; not observed-map priors",
        "measurement_policy": "fixed NSIDE8 observation set/errors; no extra observations on quadrature refinement",
        "approximations": "diffuse survey PSF mixing neglected; nominal TRIS beam; monochromatic; kappa=0; fixed RSB; dust mixing1/3",
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observation", "quadrature", "optical", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--lmax-a", type=int, default=4)
    parser.add_argument("--lmax-em", type=int, default=2)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.observation,
                args.quadrature,
                args.optical,
                args.output,
                lmax_a=args.lmax_a,
                lmax_em=args.lmax_em,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
