"""Host-side preparation of spatial information without a common map zero.

This is an observation transform, not a HartRAO likelihood admission. The caller
must supply the instrument-matched response and an explicit unprojected noise
covariance. No beam, polarization, gain or uncertainty prescription is inferred.
Apply it after selecting valid pixels; do not supply filled or already centered
maps with their singular covariance.
"""

from __future__ import annotations

import numpy as np


def spatial_contrast_basis(count: int) -> np.ndarray:
    """Return n-1 orthonormal Helmert contrasts, with L @ 1 = 0.

    This discards one arbitrary additive measurement zero. It is not integration
    against a bounded or informative zero prior, and cannot retain that prior's
    absolute-temperature information. A spatially varying attenuated background
    is not necessarily removed.
    """
    if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 2:
        raise ValueError("at least two integer measurement rows required")
    basis = np.zeros((count - 1, count), dtype=float)
    for k in range(1, count):
        scale = np.sqrt(float(k) * (k + 1))
        basis[k - 1, :k] = 1 / scale
        basis[k - 1, k] = -k / scale
    return basis


def prepare_contrasts(observed, covariance, response) -> dict[str, np.ndarray]:
    """Return Ld, LCLᵀ and LR for data d = R sky + z 1 + noise.

    The returned covariance is full rank in n-1 dimensions. Its determinant,
    correlations and normalization belong to that same basis. The normalized
    contrast density equals sqrt(n) times the integral of the original Gaussian
    over a shared flat zero; it is not an absolute model evidence with a proper
    zero prior. A fitted gain still multiplies LR and is not projected away.

    This dense host implementation is for a preselected small set of map rows,
    not a full-resolution all-sky covariance. It copies inputs and adds no jitter.
    """
    if any(np.iscomplexobj(value) for value in (observed, covariance, response)):
        raise ValueError("real-valued observations, covariance and response required")
    observed = np.asarray(observed, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    response = np.asarray(response, dtype=float)
    if observed.ndim != 1 or not np.all(np.isfinite(observed)):
        raise ValueError("observed must be one finite vector")
    count = observed.size
    basis = spatial_contrast_basis(count)
    if covariance.shape != (count, count) or not np.all(np.isfinite(covariance)):
        raise ValueError("covariance must be finite and aligned with observed")
    if not np.allclose(covariance, covariance.T, atol=0, rtol=1e-12):
        raise ValueError("covariance must be symmetric")
    if response.ndim != 2 or response.shape[0] != count or response.shape[1] < 1:
        raise ValueError("response must have one row per observation and at least one column")
    if not np.all(np.isfinite(response)):
        raise ValueError("response must be finite")
    covariance = covariance * 0.5 + covariance.T * 0.5
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("unprojected covariance must be positive definite") from error
    try:
        with np.errstate(over="raise", invalid="raise"):
            transformed = basis @ covariance @ basis.T
            transformed = transformed * 0.5 + transformed.T * 0.5
            contrast_data = basis @ observed
            contrast_response = basis @ response
    except FloatingPointError as error:
        raise ValueError("contrast arithmetic overflowed; rescale explicit input units") from error
    if not all(np.all(np.isfinite(value)) for value in
               (transformed, contrast_data, contrast_response)):
        raise ValueError("contrast arithmetic produced nonfinite values")
    try:
        np.linalg.cholesky(transformed)
    except np.linalg.LinAlgError as error:
        raise ValueError("contrast covariance lost positive definiteness") from error
    return {
        "basis": basis,
        "observed": contrast_data,
        "covariance": transformed,
        "response": contrast_response,
    }
