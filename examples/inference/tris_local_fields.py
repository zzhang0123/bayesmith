"""Fixed HEALPix-node log fields, with local interpolation and angular priors.

The constant mode is the node mean, not exactly the integral over the sphere.
All geometry and prior factors are computed before JAX inference. The kernel
is a sum of squared-exponential kernels of chordal distance (positive definite
on the sphere), not an unvalidated Gaussian of geodesic distance.
"""

import numpy as np
from scipy.linalg import helmert


def node_loading(nside, scales):
    import healpy as hp

    if not hp.isnsideok(nside) or nside > 8:
        raise ValueError("prototype requires valid node NSIDE <= 8")
    scales = np.asarray(scales, dtype=float)
    if (
        scales.ndim != 2
        or scales.shape[1] != 2
        or scales.shape[0] == 0
        or not np.all(np.isfinite(scales))
        or np.any(scales <= 0)
        or np.any(scales[:, 1] > 90)
    ):
        raise ValueError("scales must be positive (log SD, angle degrees) pairs")
    xyz = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    half_chord_squared = np.maximum(1 - xyz @ xyz.T, 0)
    covariance = sum(
        sd**2 * np.exp(-half_chord_squared / np.deg2rad(angle) ** 2)
        for sd, angle in scales
    )
    contrast = helmert(len(xyz), full=False).T
    # No eigenvalue flooring: an unresolved/ill-conditioned prior is an error.
    factor = np.linalg.cholesky(contrast.T @ covariance @ contrast)
    return contrast @ factor, covariance


def interpolation_entries(nside, theta, phi):
    import healpy as hp

    theta, phi = np.broadcast_arrays(np.asarray(theta), np.asarray(phi))
    if (
        not hp.isnsideok(nside)
        or not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(phi))
        or np.any((theta < 0) | (theta > np.pi))
    ):
        raise ValueError("finite sphere coordinates and valid NSIDE required")
    indices, weights = hp.get_interp_weights(nside, theta.ravel(), phi.ravel())
    # Ring boundaries can return ~1ULP negative barycentric weights. Repair
    # only floating-point roundoff, never clip latent fields or observations.
    roundoff = 64 * np.finfo(float).eps
    if np.any(weights < -roundoff) or np.any(np.abs(weights.sum(0) - 1) > roundoff):
        raise ValueError("invalid HEALPix barycentric weights")
    weights = np.maximum(weights, 0)
    weights /= weights.sum(0)
    return indices.T, weights.T


def interpolation(nside, theta, phi):
    import healpy as hp

    indices, weights = interpolation_entries(nside, theta, phi)
    matrix = np.zeros((len(indices), hp.nside2npix(nside)))
    for index, weight in zip(indices.T, weights.T, strict=True):
        np.add.at(matrix, (np.arange(len(indices)), index), weight)
    if np.any(matrix < 0) or not np.allclose(matrix.sum(1), 1, atol=1e-14, rtol=0):
        raise ValueError("interpolation must preserve constants and positivity")
    return matrix


def field_basis(nside, theta, phi, loading):
    weights = interpolation(nside, theta, phi)
    loading = np.asarray(loading)
    if loading.shape != (weights.shape[1], weights.shape[1] - 1):
        raise ValueError("node loading shape mismatch")
    return np.column_stack((np.ones(len(weights)), weights @ loading))
