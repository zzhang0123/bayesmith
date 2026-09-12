"""A Wiener map as an observation, with its response and sampling covariance.

This module has no astronomy dependencies. The map posterior covariance is
NOT the covariance of repeated noisy reconstructions of a fixed sky.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MapLikelihood:
    data: np.ndarray
    response: np.ndarray
    offset_response: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    noise_sigma: np.ndarray
    singular_values: np.ndarray
    retained: np.ndarray
    ring_projection: np.ndarray


def compress_map(map_k, prior_k, operator, sigma_k, covariance, *, rtol=1e-8):
    """Whiten ``m = b + W d`` on supported modes of ``W sqrt(N)``.

    ``covariance`` is the FULL Wiener posterior covariance supplied by limTOD.
    With W = Cpost A.T N^-1 and b = m0 - W A m0, repeated reconstructions
    have noise Cnoise = W N W.T and mean b + W A s. SVD whitening retains
    the same likelihood as the measured ring, up to parameter-independent
    terms when discarded modes have no model response. The caller must check
    truncation evidence for its actual model, not assume that it is harmless.
    """
    m, prior, a, sigma, cov = [
        np.asarray(x, dtype=float)
        for x in (map_k, prior_k, operator, sigma_k, covariance)
    ]
    if a.ndim != 2:
        raise ValueError("operator must have sample and pixel axes")
    n, p = a.shape
    if (
        m.shape != (p,)
        or prior.shape != (p,)
        or sigma.shape != (n,)
        or cov.shape != (p, p)
    ):
        raise ValueError("incompatible map, noise or covariance shape")
    if not all(np.isfinite(x).all() for x in (m, prior, a, sigma, cov)):
        raise ValueError("map inputs must be finite")
    if np.any(sigma <= 0) or not 0 < rtol < 1:
        raise ValueError("positive noise and 0 < rtol < 1 are required")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12):
        raise ValueError("posterior covariance must be symmetric")
    np.linalg.cholesky(cov)
    weights = (cov @ a.T) / sigma**2
    bias = prior - weights @ (a @ prior)
    left, singular, right = np.linalg.svd(weights * sigma, full_matrices=False)
    keep = singular > rtol * singular[0]
    if not keep.any():
        raise ValueError("map has no measured modes")
    projection = right[keep]
    return MapLikelihood(
        data=(left[:, keep].T @ (m - bias)) / singular[keep],
        # Algebraically U.T W A / s; this form avoids small-singular-value
        # cancellation when constructing the response, not when reading data.
        response=projection @ (a / sigma[:, None]),
        offset_response=projection @ (1 / sigma),
        weights=weights,
        bias=bias,
        noise_sigma=np.sqrt(np.sum((weights * sigma) ** 2, axis=1)),
        singular_values=singular,
        retained=keep,
        ring_projection=projection,
    )
