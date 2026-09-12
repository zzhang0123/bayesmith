"""Posterior sky fields on the prepared HEALPix grid (NumPy only).

These are global extrapolations of three regional parameters, not independent
pixel fits. Their uncertainty is conditional on the fixed Haslam template.
"""

import numpy as np


def posterior_sky_products(bundle, amplitude, beta):
    """Summarize a(p), beta(p), and total 408-MHz RJ sky from joint draws.

    The CMB removed during preparation is restored without recalibration.
    TRIS instrument zero-level corrections do not belong to the Haslam sky.
    """
    amplitude, beta = np.asarray(amplitude), np.asarray(beta)
    region, template = np.asarray(bundle["region"]), np.asarray(bundle["template_k"])
    if amplitude.ndim != 2 or amplitude.shape[1] != 3 or len(amplitude) < 2:
        raise ValueError("Need at least two posterior draws of three amplitudes")
    if beta.shape != amplitude.shape:
        raise ValueError("Amplitude and beta draws must have matching shapes")
    if region.ndim != 1 or region.shape != template.shape or not region.size:
        raise ValueError("Region and Haslam template must share a nonempty pixel grid")
    if not np.issubdtype(region.dtype, np.integer) or np.any((region < 0) | (region > 2)):
        raise ValueError("Region indices must be integers in [0, 2]")
    if "reference_cmb_k" not in bundle:
        raise ValueError("Re-run tris_prepare to record reference_cmb_k before exporting Haslam")
    cmb = np.asarray(bundle["reference_cmb_k"])
    if cmb.ndim != 0 or any(not np.all(np.isfinite(x)) for x in (amplitude, beta, template, cmb)):
        raise ValueError("Sky fields require finite draws, template and scalar reference CMB")
    calibrated = amplitude[:, region] * template + cmb
    result = {"haslam_reference_k": template + cmb}
    for name, values in (
        ("amplitude", amplitude[:, region]),
        ("beta", beta[:, region]),
        ("haslam", calibrated),
        ("haslam_change", calibrated - result["haslam_reference_k"]),
    ):
        result[f"{name}_mean"] = values.mean(axis=0)
        result[f"{name}_sd"] = values.std(axis=0, ddof=1)
        result[f"{name}_lower"], result[f"{name}_upper"] = np.quantile(values, [.025, .975], axis=0)
    return result
