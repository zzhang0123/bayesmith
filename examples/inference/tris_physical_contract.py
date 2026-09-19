"""Admission checks and explicit calibration assumptions for physical-sky work."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.special import log_ndtr


def check_independent_products(products, *, joint_calibration_groups=()):
    """Refuse duplicated observations or unmodelled shared calibration.

    Each product supplies ``id``, ``role``, ``observation_ancestors`` and
    ``calibration_groups``. Declaring a joint calibration does not make two
    products of the same raw observations independent. Conditional templates
    and summary products cannot enter this independent-data likelihood.
    This is a declared-lineage check, not automatic literature provenance.
    """
    allowed = set(joint_calibration_groups)
    ids = [p['id'] for p in products]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate product')
    for product in products:
        if product['role'] != 'measurement' or not product['observation_ancestors']:
            raise ValueError(f"{product['id']}: not an admitted independent measurement")
    for a, b in combinations(products, 2):
        common = set(a['observation_ancestors']) & set(b['observation_ancestors'])
        if common:
            raise ValueError(f"{a['id']} / {b['id']}: shared observations {sorted(common)}")
        common = set(a['calibration_groups']) & set(b['calibration_groups'])
        if common - allowed:
            raise ValueError(f"shared calibration requires a joint model: {sorted(common - allowed)}")


def tris820_offset_bounds(branch):
    """Bounds on observed-minus-physical K, opposite the old correction sign.

    Uniform density *inside* these bounds is an implementation sensitivity,
    not a measured probability distribution. Lab magnitude uses Paper I's
    rounded 0.660 K; Paper III quotes 0.659 K. Astrophysical bounds use the
    same scan and restrictive spectral assumptions, so are conditional only.
    """
    if branch == 'laboratory_uniform':
        return -0.660, 0.660
    if branch == 'astrophysical_uniform_conditional':
        return -0.430, 0.300
    raise ValueError('explicit TRIS 820 calibration branch required')


def bounded_offset_logpdf(residual, covariance, *, lower, upper):
    """Integrate one common Uniform(lower,upper) offset over all residual rows.

    ``residual=observed-physical``. The covariance contains statistical and
    other separately identified errors; it must not contain this same zero
    mode again. No TRIS 2428 per-row sigma is supplied or inferred here.
    """
    r, c = np.asarray(residual, float), np.asarray(covariance, float)
    if (r.ndim != 1 or not r.size or c.shape != (r.size, r.size)
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(c))
            or not np.allclose(c, c.T, rtol=0, atol=1e-14)
            or not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower):
        raise ValueError('finite residual, symmetric covariance and ordered bounds required')
    factor = cho_factor(c, lower=True)
    ones = np.ones(r.size)
    precision_one = cho_solve(factor, ones)
    precision = ones @ precision_one
    mean = r @ precision_one / precision
    centered = r - mean
    quad = centered @ cho_solve(factor, centered)
    lo, hi = (np.array([lower, upper]) - mean) * np.sqrt(precision)
    # Use survival probabilities in the positive tail to avoid subtracting 1-1.
    if lo > 0:
        lo, hi = -hi, -lo
    log_hi, log_lo = log_ndtr(hi), log_ndtr(lo)
    interval = log_hi + np.log(-np.expm1(log_lo - log_hi))
    logdet = 2 * np.log(np.diag(factor[0])).sum()
    return float(-0.5 * ((r.size - 1) * np.log(2 * np.pi) + logdet
                         + np.log(precision) + quad) + interval - np.log(upper - lower))
