"""Prepare only fixed geometry for the exact D19 amplitude-coordinate translation."""

import argparse
import hashlib
import json
from pathlib import Path

import healpy as hp
import numpy as np

from examples.inference.tris_local_fields import field_basis


def prepare(input_path, output):
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    nside = int(arrays['a_node_nside'])
    count = hp.nside2npix(nside)
    if count != len(arrays['haslam']) or nside != int(arrays['nside']):
        raise ValueError('reference nodes must match the frozen Haslam observation grid')
    theta, phi = hp.pix2ang(nside, np.arange(count))
    loading = np.column_stack((np.ones(count), arrays['a_node_loading']))
    projection = np.linalg.solve(loading, np.eye(count))
    em_basis = field_basis(int(arrays['em_node_nside']), theta, phi, arrays['em_node_loading'])
    identity_error = float(np.max(np.abs(projection @ loading - np.eye(count))))
    if identity_error > 1e-10 or not np.isfinite(projection).all():
        raise ValueError('amplitude node-mode transform is numerically unresolved')
    arrays.update(a_reference_projection=projection, em_at_a_reference_basis=em_basis)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'data.npz', **arrays)
    metadata = {
        'kind': 'amplitude_reference_geometry', 'scientific_certification': False,
        'input': str(input_path.resolve()), 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256((output / 'data.npz').read_bytes()).hexdigest(),
        'node_count': count, 'projection_condition': float(np.linalg.cond(loading)),
        'identity_max_error': identity_error, 'added_arrays': ['a_reference_projection', 'em_at_a_reference_basis'],
        'interpretation': 'Only coordinate geometry added; existing observations, physical nodes, priors and quadrature unchanged.',
        'source_sha256': {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name('tris_local_fields.py'))
        },
    }
    (output / 'reference.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output), indent=2))
