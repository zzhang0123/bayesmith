"""Direct solid-angle integration of the TRIS cut beam and fixed sky features.

The sky is a piecewise constant HEALPix map. Refining quadrature subdivides
those pixels without changing their brightness or region membership. Downgrading
averages *features*, never integer region labels. This separates integration
error from replacing the sky template. Output columns are integrals, not pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from examples.inference.tris_audit import _pointing_frame
from examples.inference.tris_beam_identifiability import deformed_cuts


def feature_maps(template, region):
    template = np.asarray(template, float)
    region = np.asarray(region)
    if template.ndim != 1 or region.shape != template.shape:
        raise ValueError("template and region must be one-dimensional maps")
    if not np.all(np.isfinite(template)) or not np.issubdtype(region.dtype, np.integer):
        raise ValueError("finite template and integer regions required")
    count = int(region.max()) + 1
    if not np.array_equal(np.unique(region), np.arange(count)):
        raise ValueError("region labels must be consecutive from zero")
    mask = np.eye(count)[region]
    return np.concatenate((template[:, None] * mask, mask, np.ones((len(region), 1))), axis=1)


def integrate_features(cuts, geometry, template, region, *, nside, coord="C",
                       widths=((0.0, 0.0),), beam_nside=None, interpolate=False,
                       frame="vector", progress=False):
    """Integrate an unchanged sky over equal-area cells, with an exact horizon.

    ``beam_nside=None`` evaluates the continuous cut interpolation directly.
    Otherwise sample a discretized beam by nearest pixel or bilinear interpolation.
    ``frame='spherical'`` is an independent zenith-only geometry implementation.
    """
    return integrate_feature_maps(cuts, geometry, feature_maps(template, region),
                                  nside=nside, coord=coord, widths=widths,
                                  beam_nside=beam_nside, interpolate=interpolate,
                                  frame=frame, progress=progress)


def integrate_feature_maps(cuts, geometry, features, *, nside, coord="C",
                           widths=((0.0, 0.0),), beam_nside=None,
                           interpolate=False, frame="vector", progress=False):
    """Integrate finite native HEALPix scalar fields, retaining their pixel definition."""
    import healpy as hp
    from limTOD.tris.beam import tris_cut_beam_map, tris_cut_beam_response

    if coord not in ("C", "G") or frame not in ("vector", "spherical"):
        raise ValueError("unknown coordinate system or frame implementation")
    features = np.asarray(features, float)
    if features.ndim != 2 or not np.all(np.isfinite(features)):
        raise ValueError("finite (pixel, feature) fields required")
    source_nside = hp.npix2nside(len(features))
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    if nside < source_nside:
        features = np.stack([hp.ud_grade(x, nside) for x in features.T], axis=1)
        parent = None
    else:
        parent = hp.ang2pix(source_nside, theta, phi)
    if coord == "G":
        theta, phi = hp.Rotator(coord=["G", "C"])(theta, phi)
    directions = np.asarray(hp.ang2vec(theta, phi))
    axes = _pointing_frame(geometry.lst_deg, geometry.latitude_deg,
                           geometry.azimuth_deg, geometry.elevation_deg,
                           geometry.selfrot_deg)
    deformed = [deformed_cuts(cuts, *width) for width in widths]
    beams = None if beam_nside is None else [
        tris_cut_beam_map(cut, nside=beam_nside, normalization="none") for cut in deformed
    ]
    result = np.empty((len(widths), len(geometry.lst_deg), features.shape[1]))
    for row, lst in enumerate(geometry.lst_deg):
        if frame == "vector":
            bore, axis_e, axis_h, zenith = (axis[row] for axis in axes)
            cos_theta = np.clip(directions @ bore, -1, 1)
            angle = np.rad2deg(np.arccos(cos_theta))
            azimuth = np.rad2deg(np.arctan2(directions @ axis_h, directions @ axis_e))
            above = directions @ zenith >= 0
        else:
            if not np.allclose(geometry.elevation_deg, 90) or not np.allclose(geometry.azimuth_deg, 0):
                raise ValueError("spherical oracle supports zenith scans only")
            latitude = np.deg2rad(geometry.latitude_deg)
            dec = np.pi / 2 - theta
            hour = phi - np.deg2rad(lst)
            up = np.sin(dec) * np.sin(latitude) + np.cos(dec) * np.cos(latitude) * np.cos(hour)
            north = np.sin(dec) * np.cos(latitude) - np.cos(dec) * np.sin(latitude) * np.cos(hour)
            east = np.cos(dec) * np.sin(hour)
            roll = np.deg2rad(geometry.selfrot_deg[row])
            # At zenith E is south and H is east; positive roll is Rodrigues
            # about the outward zenith, hence these two projections.
            e = -north * np.cos(roll) + east * np.sin(roll)
            h = east * np.cos(roll) + north * np.sin(roll)
            angle = np.rad2deg(np.arccos(np.clip(up, -1, 1)))
            azimuth = np.rad2deg(np.arctan2(h, e))
            above = up >= 0
        selected = np.flatnonzero(above)
        cells = features[selected if parent is None else parent[selected]]
        angle, azimuth = angle[selected], azimuth[selected]
        for index, cut in enumerate(deformed):
            if beams is None:
                weights = tris_cut_beam_response(cut, angle, azimuth)
            elif interpolate:
                weights = hp.get_interp_val(beams[index], np.deg2rad(angle), np.deg2rad(azimuth))
            else:
                weights = beams[index][hp.ang2pix(beam_nside, np.deg2rad(angle), np.deg2rad(azimuth))]
            if np.any(weights < 0) or not np.all(np.isfinite(weights)) or weights.sum() <= 0:
                raise ValueError("invalid beam integration weights")
            result[index, row] = weights @ cells / weights.sum()
        if progress and row % 20 == 0:
            print(f"nside={nside} row={row}/{len(geometry.lst_deg)}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--haslam", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nside", type=int, nargs="+", default=[64, 128, 256, 512])
    parser.add_argument("--beam-nside", type=int)
    parser.add_argument("--interpolate", action="store_true")
    parser.add_argument("--frame", choices=["vector", "spherical"], default="vector")
    parser.add_argument("--basis", action="store_true")
    partition = parser.add_mutually_exclusive_group()
    partition.add_argument("--hemispheres", action="store_true")
    partition.add_argument("--longitude-sectors", action="store_true")
    args = parser.parse_args(argv)
    import healpy as hp
    from limTOD.tris import read_tris_beam_cuts, read_tris_ring
    from limTOD.tris.geometry import tris_zenith_geometry

    args.output.mkdir(parents=True, exist_ok=True)
    bundle = dict(np.load(args.maps, allow_pickle=False))
    template, region, coord = bundle["template_k"], bundle["region"], "C"
    if args.haslam:
        template = hp.read_map(args.haslam, dtype=float) - float(bundle["reference_cmb_k"])
        theta, longitude = hp.pix2ang(hp.get_nside(template), np.arange(len(template)))
        latitude = 90 - np.rad2deg(theta)
        region = np.digitize(np.abs(latitude), [10.0, 30.0])
        if args.hemispheres:
            region = region + 3 * (latitude < 0)
        if args.longitude_sectors:
            region = region + 3 * (np.cos(longitude) < 0)
        coord = "G"
    elif args.hemispheres:
        region = region + 3 * (bundle["galactic_latitude_deg"] < 0)
    elif args.longitude_sectors:
        parser.error("longitude sectors require --haslam to define the native Galactic grid")
    geometry = tris_zenith_geometry(read_tris_ring(args.archive / "TRIS_absolute_600.txt").ra_deg)
    cuts = read_tris_beam_cuts(args.archive / "TRIS_Beam_Profile.txt")
    widths = [(0.0, 0.0)]
    if args.basis:
        widths += [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001),
                   (0.008, -0.008), (-0.01, -0.01), (-0.01, 0.01),
                   (0.01, -0.01), (0.01, 0.01)]
    paths = [args.maps, args.archive / "TRIS_Beam_Profile.txt",
             args.archive / "TRIS_absolute_600.txt", Path(__file__)]
    if args.haslam:
        paths.append(args.haslam)
    metadata = {
        "schema": "tris.feature_quadrature.v1", "coord": coord,
        "sky_nside": hp.npix2nside(len(template)), "template_min": float(template.min()),
        "regions": int(region.max()) + 1, "widths": widths, "width_step": 0.001,
        "beam_nside": args.beam_nside, "interpolate": args.interpolate, "frame": args.frame,
        "partition": ("inner-outer longitude x latitude" if args.longitude_sectors else
                      "north-south x latitude" if args.hemispheres else "absolute latitude"),
        "inputs": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "sky_definition": "piecewise constant native pixels; averaging feature maps when coarsened",
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(args.output / "sky.npz", template_k=template, region=region)
    for nside in args.nside:
        started = time.time()
        columns = integrate_features(cuts, geometry, template, region, nside=nside,
                                     coord=coord, widths=widths, beam_nside=args.beam_nside,
                                     interpolate=args.interpolate, frame=args.frame, progress=True)
        np.savez_compressed(args.output / f"features-{nside}.npz", columns=columns,
                            widths=np.asarray(widths), width_step=np.asarray(0.001),
                            template_min=np.asarray(template.min()), nside=np.asarray(nside))
        print(json.dumps({"nside": nside, "seconds": time.time() - started}), flush=True)


if __name__ == "__main__":
    main()
