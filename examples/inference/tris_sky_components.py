"""Explicit, conditional foreground components with common 408-MHz bookkeeping.

DS-DSDS is a signed processing template, not a positive point-source catalogue.
Commander free-free is an externally conditioned shape and carries no independent
absolute zero-level information. All component amplitudes below are antenna K.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


def free_free_optical_depth(frequency_ghz, emission_measure, electron_temperature):
    """Planck 2015 X, free-free law (Table 4); EM in cm^-6 pc, Te in K."""
    frequency, em, te = np.broadcast_arrays(
        frequency_ghz, emission_measure, electron_temperature
    )
    if (
        not np.all(np.isfinite(frequency))
        or not np.all(frequency > 0)
        or not np.all(np.isfinite(em))
        or not np.all(em >= 0)
        or not np.all(np.isfinite(te))
        or not np.all(te > 0)
    ):
        raise ValueError(
            "finite positive frequency/temperature and nonnegative EM required"
        )
    gaunt = np.logaddexp(
        5.960 - np.sqrt(3) / np.pi * np.log(frequency * (te / 1e4) ** -1.5), 1
    )
    return 0.05468 * te**-1.5 * frequency**-2 * em * gaunt


def free_free_temperature(frequency_ghz, emission_measure, electron_temperature):
    """RJ emission from an isothermal slab; this does not attenuate other fields."""
    tau = free_free_optical_depth(
        frequency_ghz, emission_measure, electron_temperature
    )
    return -np.asarray(electron_temperature) * np.expm1(-tau)


def smooth_weights(latitude_deg, longitude_rad):
    """Partition of unity for a smooth spatial mixture of six power laws.

    Transitions have 5-degree latitude scale and 0.2 scale in Cartesian x.
    Using sin(latitude)^2 avoids a cusp at the Galactic equator.
    """
    b = np.deg2rad(np.asarray(latitude_deg))
    longitude = np.asarray(longitude_rad)
    gates = [
        expit(
            (np.sin(b) ** 2 - np.sin(np.deg2rad(edge)) ** 2)
            / (2 * np.sin(np.deg2rad(edge)) * np.cos(np.deg2rad(edge)) * np.deg2rad(5))
        )
        for edge in (10, 30)
    ]
    # At high latitude the wider outer transition can cross the inner tail.
    # Ordered cumulative gates preserve positivity without clipping pixel values.
    outer = gates[0] * gates[1]
    lat = np.stack([1 - gates[0], gates[0] - outer, outer], axis=-1)
    east = expit(np.cos(b) * np.cos(longitude) / 0.2)
    return np.concatenate([east[..., None] * lat, (1 - east[..., None]) * lat], axis=-1)


def component_fields(template, weights, source_difference, freefree):
    h, w, s, f = map(np.asarray, (template, weights, source_difference, freefree))
    if (
        h.ndim != 1
        or w.shape != (len(h), 6)
        or s.shape != h.shape
        or f.shape != (3, len(h))
        or not all(np.all(np.isfinite(v)) for v in (h, w, s, f))
        or np.any(w < 0)
        or not np.allclose(w.sum(-1), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError(
            "aligned finite component maps and positive partition weights required"
        )
    return np.concatenate(
        [
            h[:, None] * w,
            w,
            f[0, :, None] * w,
            s[:, None],
            f[1:].T,
            np.ones((len(h), 1)),
        ],
        axis=1,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ds", "dsds", "freefree", "maps", "archive", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--nside", type=int, default=512)
    args = parser.parse_args(argv)
    import healpy as hp
    from limTOD.tris import read_tris_beam_cuts, read_tris_ring
    from limTOD.tris.geometry import tris_zenith_geometry

    from examples.inference.tris_beam_grid import fit_coefficients, width_nodes
    from examples.inference.tris_quadrature import integrate_feature_maps

    if args.output.exists():
        raise ValueError("use a fresh output directory")
    args.output.mkdir(parents=True)
    ds, dsds = (hp.read_map(p, dtype=float) for p in (args.ds, args.dsds))
    if ds.shape != dsds.shape:
        raise ValueError(
            "source templates must have matching resolution and coordinates"
        )
    with np.load(args.maps, allow_pickle=False) as archive:
        cmb = float(archive["reference_cmb_k"])
    h = dsds - cmb
    em, te = hp.read_map(args.freefree, field=(1, 4), dtype=float)
    f = np.array(
        [
            hp.ud_grade(free_free_temperature(nu, em, te), hp.get_nside(h))
            for nu in (0.408, 0.6005, 0.8178)
        ]
    )
    theta, longitude = hp.pix2ang(hp.get_nside(h), np.arange(len(h)))
    w = smooth_weights(90 - np.rad2deg(theta), longitude)
    s = ds - dsds
    fields = component_fields(h, w, s, f)
    np.savez_compressed(args.output / "fields.npz", fields=fields)
    scales = np.array([0.0, 0.25, 0.5, 1.0])
    minima = np.array([np.min(h - scale * f[0]) for scale in scales])
    nodes = width_nodes(args.degree)
    widths = np.array([(e, h) for e in nodes for h in nodes])
    paths = [
        args.ds,
        args.dsds,
        args.freefree,
        args.maps,
        args.archive / "TRIS_Beam_Profile.txt",
        args.archive / "TRIS_absolute_600.txt",
        Path(__file__),
    ]
    cuts = read_tris_beam_cuts(paths[-3])
    geometry = tris_zenith_geometry(read_tris_ring(paths[-2]).ra_deg)
    columns = integrate_feature_maps(
        cuts,
        geometry,
        fields,
        nside=args.nside,
        coord="G",
        widths=widths,
        progress=True,
    )
    columns = columns.reshape(len(nodes), len(nodes), *columns.shape[1:])
    fit_coefficients(nodes, columns, region_count=6)
    output = args.output / "grid.npz"
    np.savez_compressed(
        output,
        nodes=nodes,
        columns=columns,
        regions=6,
        template_min=h.min(),
        ff_scales=scales,
        minimum_h_minus_ff=minima,
        source_mean=s.mean(),
    )
    manifest = {
        "schema": "tris.conditional_components.v1",
        "regions": 6,
        "coord": "G",
        "native_nside": hp.get_nside(h),
        "quadrature_nside": args.nside,
        "degree": args.degree,
        "columns": "H*W(6), W(6), F408*W(6), DS-DSDS, F600, F820, 1",
        "background": "total non-CMB isotropic; subtract source template mean before adding template",
        "source_template": "signed DS-DSDS; no clipping; processing sensitivity, not a physical catalogue",
        "source_mean_k": float(s.mean()),
        "source_negative_fraction": float(np.mean(s < 0)),
        "source_negative_mean_k": float(np.minimum(s, 0).mean()),
        "freefree": "Commander EM_MEAN and TEMP_MEAN, HEALPix RING conversion; Planck 2015 X law",
        "freefree_source": "https://arxiv.org/html/1502.01588v2",
        "freefree_scales": scales.tolist(),
        "minimum_h_minus_ff_k": minima.tolist(),
        "freefree_negative_synch_fraction": float(np.mean(h - f[0] < 0)),
        "reference_cmb_rj_k": cmb,
        "sky_definition": "smooth positive partition of six power laws, not one local power law",
        "certified": False,
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "grid_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
