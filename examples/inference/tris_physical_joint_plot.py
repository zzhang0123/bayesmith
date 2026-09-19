"""Plot forward-resolution checks and explicitly unvalidated chain diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def plot(root, fit, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    grids = [json.loads((root / name / 'manifest.json').read_text())
             for name in ('prepared8-v2', 'prepared16', 'prepared32')]
    with np.load(fit / 'samples.npz') as a:
        beta, zero_h = a['beta'][:, :, 0], a['zero_h']
    with np.load(fit / 'predictive.npz') as a:
        high_data, high_pred = a['heldout2428_data'], a['heldout2428']
        high_ra = a['heldout_ra_deg']
    colors = ['#007a87', '#bd593c', '#6152a3', '#768340']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    nsides = [m['nside'] for m in grids]
    errors = [m['coarse_diffuse_projection_error_408_k'] for m in grids]
    axes[0, 0].loglog(nsides, [e['rms'] for e in errors], 'o-', color=colors[0], label='RMS')
    axes[0, 0].loglog(nsides, [e['max_abs'] for e in errors], 's--', color=colors[1], label='Maximum')
    axes[0, 0].set(xticks=nsides, xticklabels=nsides, xlabel='Sky NSIDE', ylabel='408 MHz error [K]',
                   title='Coarse sky vs native-map response')
    axes[0, 0].xaxis.set_minor_locator(NullLocator())
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=.2)
    for chain in range(len(beta)):
        axes[0, 1].plot(beta[chain], color=colors[chain % 4], linewidth=.8, label=f'Chain {chain+1}')
        axes[1, 0].plot(zero_h[chain], color=colors[chain % 4], linewidth=.8)
    axes[0, 1].set(xlabel='Retained draw', ylabel=r'$\beta_{00}$', title='Index trace: unvalidated chains')
    axes[0, 1].legend(frameon=False)
    axes[1, 0].set(xlabel='Retained draw', ylabel='Haslam observed − physical offset [K]',
                   title='Calibration trace: unvalidated chains')
    axes[1, 1].plot(high_ra, high_data, 'ko', label='Archived 2427.8 MHz')
    # The report selects equally spaced pooled draws, preserving chain order.
    for chain, subset in enumerate(np.array_split(high_pred, len(beta))):
        axes[1, 1].plot(high_ra, subset.mean(0), '.-', color=colors[chain % 4],
                        label=f'Chain {chain+1} mean prediction')
    axes[1, 1].set(xlabel='RA [deg]', ylabel='Total RJ temperature [K]',
                   title='Held-out measurements and raw-chain means')
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle('D12 physical pilot — sampling and scientific validation remain open', fontsize=14)
    fig.supxlabel('Raw-chain plots diagnose sampling; they are not accepted posterior constraints.', fontsize=10)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('choose a new figure path')
    plot(args.root, args.fit, args.output)


if __name__ == '__main__':
    main()
