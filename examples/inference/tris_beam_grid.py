"""Piecewise Chebyshev interpolation of directly integrated TRIS beam features.

Resampling tabulated cuts has a derivative change at zero log-width. Each
quadrant is therefore interpolated separately; a polynomial spanning zero is
not the same approximation. Independent direct integrals must certify any grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.chebyshev import chebvander2d


def width_nodes(degree=4, bound=0.01):
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 2:
        raise ValueError("degree must be an integer >= 2")
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("positive finite width bound required")
    half = (1 - np.cos(np.pi * np.arange(degree + 1) / degree)) * bound / 2
    return np.concatenate((-half[:0:-1], half))


def fit_coefficients(nodes, columns, *, region_count=None):
    nodes, columns = np.asarray(nodes, float), np.asarray(columns, float)
    if nodes.ndim != 1:
        raise ValueError("one-dimensional width nodes required")
    degree = (len(nodes) - 1) // 2
    if (
        degree < 2
        or len(nodes) != 2 * degree + 1
        or not np.all(np.isfinite(nodes))
        or not np.all(np.diff(nodes) > 0)
        or nodes[degree] != 0
        or nodes[0] != -nodes[-1]
    ):
        raise ValueError("symmetric ordered width nodes with a shared zero required")
    if columns.ndim != 4 or columns.shape[:2] != (len(nodes), len(nodes)):
        raise ValueError("columns must have (E node, H node, row, feature) axes")
    if not np.all(np.isfinite(columns)):
        raise ValueError("finite features required")
    count = (columns.shape[-1] - 1) // 2 if region_count is None else region_count
    if (
        count < 1
        or columns.shape[-1] < 2 * count + 1
        or (region_count is None and columns.shape[-1] != 2 * count + 1)
    ):
        raise ValueError("features must contain template, region and constant columns")
    if not np.allclose(columns[..., -1], 1, rtol=0, atol=1e-10) or not np.allclose(
        columns[..., count : 2 * count].sum(-1), 1, rtol=0, atol=1e-10
    ):
        raise ValueError("features must conserve the constant sky and region partition")
    coefficients = []
    for e_side in range(2):
        for h_side in range(2):
            es = slice(e_side * degree, (e_side + 1) * degree + 1)
            hs = slice(h_side * degree, (h_side + 1) * degree + 1)
            x = 2 * nodes[es] / nodes[-1] + (1 if e_side == 0 else -1)
            y = 2 * nodes[hs] / nodes[-1] + (1 if h_side == 0 else -1)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            design = chebvander2d(xx.ravel(), yy.ravel(), [degree, degree])
            values = columns[es, hs].reshape((degree + 1) ** 2, -1)
            coefficients.append(
                np.linalg.solve(design, values).reshape(
                    degree + 1, degree + 1, *columns.shape[2:]
                )
            )
    return np.stack(coefficients)


class BeamFeatureGrid(eqx.Module):
    coefficients: jnp.ndarray
    bound: float = eqx.field(static=True)

    def __call__(self, width_e, width_h):
        width_e, width_h = jnp.asarray(width_e), jnp.asarray(width_h)
        e_side, h_side = width_e >= 0, width_h >= 0
        x = 2 * width_e / self.bound + jnp.where(e_side, -1, 1)
        y = 2 * width_h / self.bound + jnp.where(h_side, -1, 1)
        coefficient = self.coefficients[2 * e_side.astype(int) + h_side.astype(int)]
        tx, ty = [jnp.ones_like(x), x], [jnp.ones_like(y), y]
        for _ in range(2, coefficient.shape[0]):
            tx.append(2 * x * tx[-1] - tx[-2])
            ty.append(2 * y * ty[-1] - ty[-2])
        value = jnp.einsum("i,j,ijrk->rk", jnp.stack(tx), jnp.stack(ty), coefficient)
        valid = (jnp.abs(width_e) <= self.bound) & (jnp.abs(width_h) <= self.bound)
        return jnp.where(valid, value, jnp.nan)

    def rows(self, indices):
        return BeamFeatureGrid(self.coefficients[:, :, :, indices, :], self.bound)


def load_grid(path):
    with np.load(path, allow_pickle=False) as archive:
        nodes, columns = archive["nodes"], archive["columns"]
        minimum = float(archive["template_min"])
        count = int(archive["regions"]) if "regions" in archive else None
    if not np.isfinite(minimum):
        raise ValueError("finite template minimum required")
    coefficients = fit_coefficients(nodes, columns, region_count=count)
    return BeamFeatureGrid(jnp.asarray(coefficients), float(nodes[-1])), minimum


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sky", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--nside", type=int, default=512)
    args = parser.parse_args(argv)
    from limTOD.tris import read_tris_beam_cuts, read_tris_ring
    from limTOD.tris.geometry import tris_zenith_geometry

    from examples.inference.tris_quadrature import integrate_features

    if args.output.exists():
        raise ValueError("use a fresh grid output path")
    nodes = width_nodes(args.degree)
    widths = np.array([(e, h) for e in nodes for h in nodes])
    with np.load(args.sky, allow_pickle=False) as archive:
        template, region = archive["template_k"], archive["region"]
    paths = [
        args.sky,
        args.archive / "TRIS_Beam_Profile.txt",
        args.archive / "TRIS_absolute_600.txt",
        Path(__file__),
    ]
    cuts = read_tris_beam_cuts(paths[1])
    geometry = tris_zenith_geometry(read_tris_ring(paths[2]).ra_deg)
    columns = integrate_features(
        cuts,
        geometry,
        template,
        region,
        nside=args.nside,
        coord="G",
        widths=widths,
        progress=True,
    )
    columns = columns.reshape(len(nodes), len(nodes), *columns.shape[1:])
    fit_coefficients(nodes, columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        columns=columns,
        nodes=nodes,
        template_min=np.asarray(template.min()),
        nside=args.nside,
    )
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "schema": "tris.beam_feature_grid.v1",
                "degree": args.degree,
                "bound": 0.01,
                "quadrature_nside": args.nside,
                "regions": int(region.max()) + 1,
                "definition": "four quadrant tensor Chebyshev; direct normalized pixel integrals",
                "certified": False,
                "inputs": {
                    str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in paths
                },
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
