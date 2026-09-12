"""Prepare real TRIS maps offline. Run with limTOD, healpy and astropy installed.

python -m examples.inference.tris_prepare --archive /path/to/TRIS \
    --haslam /path/to/haslam408_dsds_Remazeilles2014.fits --output runs/tris-input
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import numpy as np

if __package__:
    from .tris_maps import compress_map
else:
    from tris_maps import compress_map

UPSTREAM = "https://github.com/BellaNasirudin/bayesian_skymap"
UPSTREAM_REVISION = "e7b8cb34e1a872d11a219f6215e38791058f4e14"
ARCHIVE_URL = "https://lambda.gsfc.nasa.gov/product/tris/tris_prod_table.html"
HASLAM_URL = "https://lambda.gsfc.nasa.gov/data/foregrounds/haslam_2014/haslam408_dsds_Remazeilles2014.fits"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(archive, haslam, output, *, nside=8):
    import healpy as hp
    import limTOD
    from limTOD.HPW_filter import wiener_filter_map
    from limTOD.tris import (
        build_tris_mapmaking_inputs,
        cmb_monopole_rj_k,
        read_tris_beam_cuts,
        read_tris_point_set,
        read_tris_ring,
    )

    if not hp.isnsideok(nside) or nside < 8 or nside > 16:
        raise ValueError("This dense case supports nside 8 or 16")
    output.mkdir(parents=True, exist_ok=True)
    archive_names = [
        "TRIS_absolute_600.txt",
        "TRIS_absolute_820.txt",
        "TRIS_absolute_2500MHz.txt",
        "TRIS_Beam_Profile.txt",
    ]
    cuts = read_tris_beam_cuts(archive / archive_names[3])
    rings = [read_tris_ring(archive / name) for name in archive_names[:2]]
    points = read_tris_point_set(archive / archive_names[2])
    raw = hp.read_map(haslam, dtype=np.float64)
    if np.any(~np.isfinite(raw)) or np.any(raw == hp.UNSEEN):
        raise ValueError("Haslam template must cover the full sky with finite values")
    # Haslam is Galactic, RING, antenna K. Subtract only the RJ CMB at 408 MHz;
    # all remaining emission is one phenomenological component, not pure synchrotron.
    gal = hp.ud_grade(raw - float(cmb_monopole_rj_k(408.0)), max(64, nside))
    template = hp.ud_grade(hp.Rotator(coord=["G", "C"]).rotate_map_pixel(gal), nside)
    pixels = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pixels)
    theta_g, _ = hp.Rotator(coord=["C", "G"])(theta, phi)
    latitude = 90 - np.rad2deg(theta_g)
    region = np.digitize(np.abs(latitude), [10.0, 30.0])
    freqs = np.array([r.effective_frequency_mhz for r in rings])
    payload = {
        "nside": np.array(nside),
        "template_k": template,
        "reference_cmb_k": np.array(cmb_monopole_rj_k(408.0)),
        "region": region,
        "galactic_latitude_deg": latitude,
        "ra_pixel_deg": np.rad2deg(phi),
        "dec_pixel_deg": 90 - np.rad2deg(theta),
        "frequency_mhz": freqs,
        "cmb_k": np.array([cmb_monopole_rj_k(f) for f in freqs]),
        "points_ra_deg": points.ra_deg,
        "points_temperature_k": points.temperature_k,
        "points_frequency_mhz": np.array(points.effective_frequency_mhz),
        "points_zero_level_k": np.array(points.zero_level_uncertainty_k),
        "beam_angle_deg": cuts.angle_deg,
        "beam_e_db": cuts.e_plane_db,
        "beam_h_db": cuts.h_plane_db,
    }
    evidence = []
    for i, ring in enumerate(rings):
        floor = float(
            np.min(ring.statistical_uncertainty_k[ring.statistical_uncertainty_k > 0])
        )
        print(
            f"Prepare {ring.effective_frequency_mhz} MHz, nside={nside}, floor={floor} K",
            flush=True,
        )
        inputs = build_tris_mapmaking_inputs(
            ring,
            nside=nside,
            cuts=cuts,
            pixel_indices=pixels,
            uncertainty_floor_k=floor,
            nside_hires=64,
        )
        a, sigma = inputs.operator, inputs.noise.statistical_sigma_k
        prior = (
            template * (ring.effective_frequency_mhz / 408.0) ** -2.8
            + payload["cmb_k"][i]
        )
        prior_sigma = np.hypot(0.5 * prior, 3.0)
        # Same public solver used by TRISMapMakingInputs.solve, requesting full C.
        m, posterior_sigma, cov = wiener_filter_map(
            inputs.data_k,
            a,
            noise_variance=sigma**2,
            prior_inv_cov=prior_sigma**-2,
            guess=prior,
            return_full_cov=True,
        )
        cov = 0.5 * (cov + cov.T)  # remove inverse round-off asymmetry only
        c = compress_map(m, prior, a, sigma, cov)
        roundtrip = c.data - c.ring_projection @ (inputs.data_k / sigma)
        # Do not carry a numerically unreliable map transform into inference.
        if np.max(np.abs(roundtrip)) > 0.01:
            raise ValueError(
                f"map whitening round-trip exceeds 0.01 noise sigma: {roundtrip}"
            )
        arrays = {
            "map_k": m,
            "posterior_sigma_k": posterior_sigma,
            "posterior_covariance": cov,
            "noise_sigma_k": c.noise_sigma,
            "weights": c.weights,
            "bias": c.bias,
            "prior_k": prior,
            "prior_sigma_k": prior_sigma,
            "operator": a,
            "data_k": inputs.data_k,
            "sigma_k": sigma,
            "ra_deg": ring.ra_deg,
            "whitened_data": c.data,
            "response": c.response,
            "offset_response": c.offset_response,
            "ring_projection": c.ring_projection,
            "singular_values": c.singular_values,
            "retained": c.retained,
            "coverage": inputs.beam_coverage,
        }
        payload.update({f"{key}_{i}": value for key, value in arrays.items()})
        evidence.append(
            {
                "frequency_mhz": float(freqs[i]),
                "rows": len(ring.ra_deg),
                "noise_floor_k": floor,
                "floored_rows": int(np.sum(ring.statistical_uncertainty_k < floor)),
                "map_pixels": len(pixels),
                "modes": int(c.retained.sum()),
                "max_whitening_error_sigma": float(np.max(np.abs(roundtrip))),
                "min_beam_coverage": float(inputs.beam_coverage.min()),
            }
        )
        print(evidence[-1], flush=True)
    np.savez_compressed(output / "maps.npz", **payload)
    sources = output / "archive"
    sources.mkdir(exist_ok=True)
    for name in archive_names:
        shutil.copy2(archive / name, sources / name)
    limtod_root = Path(limTOD.__file__).parent
    manifest = {
        "schema": "bayesmith.tris.maps.v1",
        "archive_url": ARCHIVE_URL,
        "haslam_url": HASLAM_URL,
        "source_files": {name: sha256(archive / name) for name in archive_names},
        "haslam_sha256": sha256(haslam),
        "input_sha256": sha256(output / "maps.npz"),
        "upstream_model": {
            "url": UPSTREAM,
            "revision": UPSTREAM_REVISION,
            "functions": ["apply_beam", "calc_model_spectral"],
        },
        "limtod_version": importlib.metadata.version("limTOD"),
        "limtod_source_sha256": {
            str(p.relative_to(limtod_root)): sha256(p)
            for p in sorted(limtod_root.rglob("*.py"))
        },
        "preparation_source_sha256": {
            p.name: sha256(p)
            for p in [Path(__file__), Path(__file__).with_name("tris_maps.py")]
        },
        "nside": nside,
        "ordering": "RING",
        "coordinates": "equatorial; archive epoch unspecified",
        "mapmaking_prior": "Haslam extrapolation beta=-2.8, SD=hypot(0.5 * mean, 3 K)",
        "haslam_convention": "Remazeilles2014 minus RJ CMB(408 MHz); no extra monopole subtraction",
        "reference_frequency_mhz": 408.0,
        "reference_cmb_k": float(payload["reference_cmb_k"]),
        "region_boundaries_abs_galactic_latitude_deg": [10.0, 30.0],
        "maps": evidence,
        "whitening_relative_cutoff": 1e-8,
        "points_use": "display only: no statistical uncertainty supplied; column 3 is common zero level",
        "assumptions": [
            "Published statistical errors treated as independent; sample covariance not published",
            "820 MHz zero level uses two half normals with equal side mass, scales 0.300/0.430 K",
            "820 MHz asymmetric sky-temperature correction already incorporates astrophysical constraints (TRIS I section 5); raw systematic was 0.660 K",
            "Haslam template fixed; calibration/zero-level/template uncertainty not marginalized",
            "Three latitude-region amplitudes and indices, no curvature or separate free-free component",
            "nside 8/16 pixelization and measured-cut beam interpolation remain approximations",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--haslam", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/tris-input"))
    parser.add_argument("--nside", type=int, default=8)
    args = parser.parse_args()
    prepare(args.archive, args.haslam, args.output, nside=args.nside)


if __name__ == "__main__":
    main()
