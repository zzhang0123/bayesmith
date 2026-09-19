"""P0 audit for the TRIS ring analysis: map algebra and discretisation error.

Two independent questions:

* **P0b** -- is the compressed Wiener map an honest likelihood?  The map
  posterior covariance is not the sampling covariance of a noisy reconstruction;
  examples/inference/tris_maps.compress_map says the right covariance is
  W N W.T.  This module builds an explicit dense oracle from the Wiener
  equations and checks the compressed likelihood against it, then checks the
  sampling covariance by Monte Carlo and the truncation error direction by
  direction.
* **P0c** -- how much of the residual is the coarse beam?  The production path
  builds the beam at nside 8 and only then upgrades it to nside 64 before the
  harmonic transform, so the beam the transform sees is a nearest-neighbour
  copy of a map that never had sub-nside-8 structure.  The ladder here rebuilds
  the beam directly from the archive cuts at higher nside and measures the
  prediction change in kelvin, in ring sigma, and in log-likelihood.

The P0b core needs only numpy and imports cleanly in the bayesmith environment.
The P0c driver imports limTOD and healpy lazily, inside the functions that need
them, so this module can be imported (and unit-tested) without them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "DenseWiener",
    "beam_resolution_ladder",
    "control_operator",
    "dense_wiener",
    "operator_delta_report",
    "map_residual_norm2",
    "monte_carlo_sampling_covariance",
    "pointed_operator_from_alm",
    "pointed_operator_from_beam_map",
    "ring_residual_norm2",
    "truncated_response_fraction",
    "whitening_report",
]


# ---------------------------------------------------------------------------
# P0b: the map likelihood, checked against a dense oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseWiener:
    """Explicit dense Wiener quantities, built without limTOD.

    With W = Cpost A.T N^-1, b = m0 - W A m0 and
    Cpost = (A.T N^-1 A + S^-1)^-1:

    mean
        b + W d.
    covariance
        Cpost, the posterior covariance.
    weights
        W.
    bias
        b.
    noise_covariance
        W N W.T, the covariance of repeated noisy reconstructions.  This is
        the covariance the map likelihood needs, and it is not Cpost.
    """

    mean: np.ndarray
    covariance: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    noise_covariance: np.ndarray


def dense_wiener(
    a: np.ndarray,
    data: np.ndarray,
    sigma: np.ndarray,
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
) -> DenseWiener:
    """Solve the Wiener problem with explicit matrix inverses.

    Deliberately naive: no Cholesky, no Woodbury, no shared helper.  It is the
    oracle for compress_map and limTOD.wiener_filter_map, so it must not be able
    to inherit either one's bug.
    """
    a = np.asarray(a, dtype=float)
    data = np.asarray(data, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_sigma = np.asarray(prior_sigma, dtype=float)
    n, p = a.shape
    if data.shape != (n,) or sigma.shape != (n,):
        raise ValueError("data and sigma must have one entry per sample")
    if prior_mean.shape != (p,) or prior_sigma.shape != (p,):
        raise ValueError("prior mean and sigma must have one entry per parameter")
    inverse_variance = 1.0 / sigma**2
    precision = (a * inverse_variance[:, None]).T @ a + np.diag(prior_sigma**-2.0)
    covariance = np.linalg.inv(precision)
    # Cpost is symmetric by construction.  Its inverse is symmetric only to
    # round-off, and at 768 pixels with a large condition number that round-off
    # reached 1e-9 relative -- above compress_map's own 1e-10 symmetry
    # tolerance.  Symmetrizing here is numerical hygiene, not a model choice;
    # the production path does the same thing after wiener_filter_map.
    covariance = 0.5 * (covariance + covariance.T)
    weights = covariance @ (a.T * inverse_variance)
    rhs = a.T @ (data * inverse_variance) + prior_mean / prior_sigma**2.0
    bias = prior_mean - weights @ (a @ prior_mean)
    noise_covariance = (weights * sigma) @ (weights * sigma).T
    return DenseWiener(
        mean=covariance @ rhs,
        covariance=covariance,
        weights=weights,
        bias=bias,
        noise_covariance=noise_covariance,
    )


def compress_map(mean, prior_mean, a, sigma, covariance):
    """Thin wrapper so callers here use the production compress_map."""
    from examples.inference.tris_maps import compress_map as _compress

    return _compress(mean, prior_mean, a, sigma, covariance)


def whitening_report(likelihood) -> dict:
    """Rank, singular values and whitened-data moments of a MapLikelihood."""
    values = np.asarray(likelihood.singular_values, dtype=float)
    data = np.asarray(likelihood.data, dtype=float)
    retained = np.asarray(likelihood.retained, dtype=bool)
    return {
        "modes": int(values.size),
        "retained": int(np.count_nonzero(retained)),
        "singular_values": values.tolist(),
        "largest": float(values[0]),
        "smallest_retained": float(values[retained][-1]),
        "condition": float(values[0] / values[retained][-1]),
        "whitened_mean": float(data.mean()),
        "whitened_second_moment": float(np.mean(data**2)),
        "whitened_rms": float(np.sqrt(np.mean(data**2))),
    }


def monte_carlo_sampling_covariance(
    a: np.ndarray,
    sigma: np.ndarray,
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
    sky: np.ndarray,
    *,
    draws: int = 400,
    seed: int = 20260912,
) -> dict:
    """Draw noisy rings, compress them, and compare moments to the algebra.

    The model is d = A sky + eps with eps ~ N(0, diag(sigma^2)).  The
    reconstruction is m = b + W d.  Everything below is measured from the
    draws; nothing is read back from the oracle.
    """
    from examples.inference.tris_maps import compress_map as _compress

    a = np.asarray(a, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_sigma = np.asarray(prior_sigma, dtype=float)
    sky = np.asarray(sky, dtype=float)
    dense = dense_wiener(a, np.zeros(a.shape[0]), sigma, prior_mean, prior_sigma)
    noise_covariance = dense.noise_covariance
    rng = np.random.default_rng(seed)
    reconstructions = np.empty((draws, a.shape[1]))
    whitened_draws = []
    for index in range(draws):
        sample = a @ sky + rng.normal(scale=sigma)
        reconstruction = dense.bias + dense.weights @ sample
        likelihood = _compress(
            reconstruction, prior_mean, a, sigma, dense.covariance
        )
        reconstructions[index] = reconstruction
        whitened_draws.append(np.asarray(likelihood.data, dtype=float))
    whitened = np.asarray(whitened_draws)
    centered = whitened - whitened.mean(axis=0)
    empirical = centered.T @ centered / (draws - 1)
    diagonal = np.diag(empirical)
    reconstruction_covariance = np.cov(reconstructions, rowvar=False)
    off_diagonal = empirical - np.diag(diagonal)
    reconstruction_error = reconstruction_covariance - noise_covariance
    # With more parameters than draws the empirical reconstruction covariance is
    # rank limited, and W N W.T is itself rank deficient: a whole-pixel
    # Frobenius ratio is dominated by noise in directions that carry no
    # sampling variance at all.  Restrict the comparison to the pixels the
    # sampling covariance actually constrains.
    sampling_diagonal = np.diag(noise_covariance)
    scale = float(np.max(sampling_diagonal)) if sampling_diagonal.size else 0.0
    measured = sampling_diagonal > 1e-6 * scale if scale > 0 else np.zeros_like(
        sampling_diagonal, dtype=bool
    )
    if measured.any():
        masked_error = reconstruction_error[np.ix_(measured, measured)]
        masked_reference = noise_covariance[np.ix_(measured, measured)]
        relative_frobenius = float(
            np.linalg.norm(masked_error) / np.linalg.norm(masked_reference)
        )
        diagonal_relative_max = float(
            np.max(
                np.abs(np.diag(reconstruction_error))[measured]
                / sampling_diagonal[measured]
            )
        )
    else:
        relative_frobenius = 0.0
        diagonal_relative_max = 0.0
    return {
        "draws": int(draws),
        "seed": int(seed),
        "measured_pixel_fraction": float(measured.mean()),
        # Maxima over ~7000 whitened entries are dominated by the extreme tail
        # of the sample-covariance noise; the RMS is the stable statistic.
        "empirical_whitened_diagonal_max_deviation": float(
            np.max(np.abs(diagonal - 1.0))
        ),
        "empirical_whitened_offdiagonal_rms": float(
            np.sqrt(np.mean(off_diagonal**2))
        ),
        "sampling_covariance_offdiagonal_max": float(
            np.max(np.abs(noise_covariance - np.diag(np.diag(noise_covariance))))
        ),
        "reconstruction_covariance_relative_frobenius": relative_frobenius,
        "reconstruction_covariance_diagonal_relative_max": diagonal_relative_max,
        "coverage_within_1sigma": float(
            np.mean(np.abs(centered / np.sqrt(diagonal)) <= 1.0)
        ),
        "coverage_within_2sigma": float(
            np.mean(np.abs(centered / np.sqrt(diagonal)) <= 2.0)
        ),
        "whitened_mean_norm": float(whitened.mean(axis=0) @ whitened.mean(axis=0)),
    }


def truncated_response_fraction(likelihood, a, sigma, directions) -> dict:
    """Fraction of each whitened model direction the compression discards.

    directions is (n_parameters, k).  The full whitened response is
    A theta / sigma; the retained subspace is spanned by the rows of
    ring_projection.  A direction whose discarded fraction is not negligible is
    information the ring likelihood has and the map likelihood does not, so the
    two are not interchangeable for that model.
    """
    a = np.asarray(a, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    directions = np.asarray(directions, dtype=float)
    projection = np.asarray(likelihood.ring_projection, dtype=float)
    fractions = []
    for column in directions.T:
        whitened = a @ column / sigma
        retained = projection.T @ (projection @ whitened)
        full = float(whitened @ whitened)
        lost = float((whitened - retained) @ (whitened - retained))
        fractions.append(0.0 if full == 0.0 else lost / full)
    return {
        "discarded_fraction": fractions,
        "max_discarded_fraction": float(max(fractions, default=0.0)),
    }


def ring_residual_norm2(a, data, sigma, sky, zero_level=0.0) -> float:
    """Ring chi-square for the model mean A sky - zero_level."""
    residual = (
        np.asarray(data, float)
        - np.asarray(a, float) @ np.asarray(sky, float)
        + float(zero_level)
    ) / np.asarray(sigma, float)
    return float(residual @ residual)


def map_residual_norm2(likelihood, sky, zero_level=0.0) -> float:
    """Compressed-map chi-square for the same model mean."""
    residual = (
        np.asarray(likelihood.data, float)
        - np.asarray(likelihood.response, float) @ np.asarray(sky, float)
        + float(zero_level) * np.asarray(likelihood.offset_response, float)
    )
    return float(residual @ residual)


# ---------------------------------------------------------------------------
# P0c: beam-construction and sky-grid convergence
# ---------------------------------------------------------------------------


def beam_map_from_cuts(cuts, nside: int, *, blend: str = "db") -> np.ndarray:
    """Build the principal-plane cut beam directly at nside."""
    from limTOD.tris.beam import tris_cut_beam_map

    return tris_cut_beam_map(cuts, nside=nside, blend=blend, normalization="peak")


def pointed_operator_from_alm(
    beam_alm,
    geometry,
    pixels,
    nside_target: int,
    *,
    horizontal_mask=None,
    normalize: bool = True,
) -> np.ndarray:
    """Point a beam alm at every sample and sample the result on pixels."""
    from limTOD.simulator import pointing_beam_in_eq_sys

    pixels = np.asarray(pixels, dtype=int)
    samples = geometry.lst_deg.size
    operator = np.empty((samples, pixels.size))
    for index in range(samples):
        pointed = pointing_beam_in_eq_sys(
            beam_alm,
            float(geometry.lst_deg[index]),
            float(geometry.latitude_deg),
            float(geometry.azimuth_deg[index]),
            float(geometry.elevation_deg[index]),
            float(geometry.selfrot_deg[index]),
            nside=nside_target,
            normalize=normalize,
            horizontal_mask=horizontal_mask,
        )
        operator[index] = pointed[pixels]
    return operator


def pointed_operator_from_beam_map(
    beam_map,
    geometry,
    pixels,
    nside_target: int,
    *,
    horizontal_mask=None,
    normalize: bool = True,
) -> np.ndarray:
    """Build the beam alm at its own resolution, then point it."""
    import healpy as hp

    return pointed_operator_from_alm(
        hp.map2alm(beam_map),
        geometry,
        pixels,
        nside_target,
        horizontal_mask=horizontal_mask,
        normalize=normalize,
    )


def control_operator(
    beam_map,
    geometry,
    pixels,
    nside_target: int,
    *,
    nside_hires: int = 64,
    horizontal_mask=None,
) -> np.ndarray:
    """The production path: upgrade the coarse beam, then transform."""
    from limTOD.simulator import generate_sky2sys_projection

    return generate_sky2sys_projection(
        beam_map,
        geometry.lst_deg,
        geometry.latitude_deg,
        geometry.azimuth_deg,
        geometry.elevation_deg,
        geometry.selfrot_deg,
        np.asarray(pixels, dtype=int),
        horizontal_mask=horizontal_mask,
        normalize_beam=True,
        nside_hires=nside_hires,
        nside_target=nside_target,
    )


def beam_resolution_ladder(
    cuts,
    geometry,
    pixels,
    nsides: Sequence[int],
    *,
    nside_target: int,
    horizontal_mask=None,
    blend: str = "db",
) -> dict:
    """Point the cut beam built directly at each nside in nsides."""
    ladder = {}
    for nside in nsides:
        beam = beam_map_from_cuts(cuts, nside, blend=blend)
        ladder[nside] = pointed_operator_from_beam_map(
            beam,
            geometry,
            pixels,
            nside_target,
            horizontal_mask=horizontal_mask,
        )
    return ladder


def operator_delta_report(reference, trial, sky, sigma) -> dict:
    """Summary of (trial - reference) @ sky in K and in ring sigma."""
    delta = np.asarray(trial, float) - np.asarray(reference, float)
    prediction = delta @ np.asarray(sky, float)
    sigma = np.asarray(sigma, float)
    return {
        "operator_max_abs": float(np.max(np.abs(delta))),
        "operator_frobenius": float(np.linalg.norm(delta)),
        "prediction_rms_k": float(np.sqrt(np.mean(prediction**2))),
        "prediction_max_abs_k": float(np.max(np.abs(prediction))),
        "prediction_rms_sigma": float(np.sqrt(np.mean((prediction / sigma) ** 2))),
        "prediction_max_abs_sigma": float(np.max(np.abs(prediction / sigma))),
        "delta_chi_square": float(np.sum((prediction / sigma) ** 2)),
    }


def jsonable(value):
    """Recursively convert numpy scalars and arrays for json.dumps."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def pointed_prediction(
    beam_alm,
    geometry,
    sky,
    nside_target: int,
    *,
    horizontal_mask=None,
) -> np.ndarray:
    """Beam-weighted integral of a full-sky map at every sample.

    This is the same normalization the operator path uses (sum over the
    nside_target sphere, horizon mask applied before normalization), but it
    never materialises the operator, so the sky grid can be raised further than
    the operator's pixel count would allow.
    """
    from limTOD.simulator import pointing_beam_in_eq_sys

    sky = np.asarray(sky, dtype=float)
    out = np.empty(geometry.lst_deg.size)
    for index in range(geometry.lst_deg.size):
        pointed = pointing_beam_in_eq_sys(
            beam_alm,
            float(geometry.lst_deg[index]),
            float(geometry.latitude_deg),
            float(geometry.azimuth_deg[index]),
            float(geometry.elevation_deg[index]),
            float(geometry.selfrot_deg[index]),
            nside=nside_target,
            normalize=True,
            horizontal_mask=horizontal_mask,
        )
        out[index] = float(pointed @ sky)
    return out



# ---------------------------------------------------------------------------
# Independent direct integration: pixel-space, no limTOD rotation helper
# ---------------------------------------------------------------------------


def _equatorial_frame(lst_deg, lat_deg):
    """Zenith, north and east unit vectors in the equatorial frame.

    x is RA = 0, y is RA = 90 deg, z is the north celestial pole.  This is a
    from-scratch spherical-astronomy construction, not a call into limTOD."""
    lst = np.deg2rad(np.asarray(lst_deg, dtype=float))
    lat = np.deg2rad(float(lat_deg))
    z_loc = np.stack(
        [np.cos(lat) * np.cos(lst), np.cos(lat) * np.sin(lst), np.full_like(lst, np.sin(lat))],
        axis=-1,
    )
    north = np.stack(
        [-np.sin(lat) * np.cos(lst), -np.sin(lat) * np.sin(lst), np.full_like(lst, np.cos(lat))],
        axis=-1,
    )
    east = np.stack([-np.sin(lst), np.cos(lst), np.zeros_like(lst)], axis=-1)
    return z_loc, north, east


def _pointing_frame(lst_deg, lat_deg, azimuth_deg, elevation_deg, selfrot_deg):
    """Boresight and the two beam-frame axes at every sample."""
    z_loc, north, east = _equatorial_frame(lst_deg, lat_deg)
    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=float))
    elevation = np.deg2rad(np.asarray(elevation_deg, dtype=float))
    roll = np.deg2rad(np.asarray(selfrot_deg, dtype=float))
    cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)
    cos_el, sin_el = np.cos(elevation), np.sin(elevation)
    boresight = (
        (cos_el * cos_az)[:, None] * north
        + (cos_el * sin_az)[:, None] * east
        + sin_el[:, None] * z_loc
    )
    # phi = 0 points toward increasing elevation, phi = 90 deg toward
    # increasing azimuth, both tangent to the sphere at the boresight.
    elevation_axis = (
        (-sin_el * cos_az)[:, None] * north
        + (-sin_el * sin_az)[:, None] * east
        + cos_el[:, None] * z_loc
    )
    azimuth_axis = (-sin_az)[:, None] * north + cos_az[:, None] * east
    # Rodrigues about the boresight (the axes are tangent, so no parallel term).
    cos_roll, sin_roll = np.cos(roll)[:, None], np.sin(roll)[:, None]
    elevation_axis_rolled = (
        cos_roll * elevation_axis + sin_roll * np.cross(boresight, elevation_axis)
    )
    azimuth_axis_rolled = (
        cos_roll * azimuth_axis + sin_roll * np.cross(boresight, azimuth_axis)
    )
    return boresight, elevation_axis_rolled, azimuth_axis_rolled, z_loc


def independent_pointed_beam(
    beam_map,
    geometry,
    nside_target,
    *,
    beam_nside,
    min_elevation_deg=0.0,
):
    """Pointed, masked, normalized beam on the target grid, by pixel lookup.

    For every target pixel this looks up the beam value at that pixel's
    coordinates in the beam frame and masks it below the horizon, instead of
    rotating the beam alm with limTOD/healpy's rotation helper.  It is the
    independent direct integration the P0 audit needs, and it shares no code
    with pointing_beam_in_eq_sys."""
    import healpy as hp

    beam_map = np.asarray(beam_map, dtype=float)
    npix = hp.nside2npix(nside_target)
    theta, phi = hp.pix2ang(nside_target, np.arange(npix))
    direction = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
    )
    boresight, elevation_axis, azimuth_axis, z_loc = _pointing_frame(
        geometry.lst_deg,
        geometry.latitude_deg,
        geometry.azimuth_deg,
        geometry.elevation_deg,
        geometry.selfrot_deg,
    )
    operator = np.empty((boresight.shape[0], npix))
    for index in range(boresight.shape[0]):
        cos_angle = np.clip(direction @ boresight[index], -1.0, 1.0)
        theta_beam = np.arccos(cos_angle)
        phi_beam = np.arctan2(
            direction @ azimuth_axis[index], direction @ elevation_axis[index]
        )
        values = beam_map[hp.ang2pix(beam_nside, theta_beam, phi_beam)]
        elevation = np.rad2deg(np.arcsin(np.clip(direction @ z_loc[index], -1.0, 1.0)))
        values = np.where(elevation >= min_elevation_deg, values, 0.0)
        total = values.sum()
        if not np.isfinite(total) or np.abs(total) < np.finfo(float).tiny:
            raise ValueError('pointed beam sums to zero for sample ' + str(index))
        operator[index] = values / total
    return operator


def independent_operator(beam_map, geometry, pixels, nside_target, *, beam_nside, min_elevation_deg=0.0):
    """Independent operator restricted to the target pixels the analysis keeps."""
    full = independent_pointed_beam(
        beam_map, geometry, nside_target, beam_nside=beam_nside, min_elevation_deg=min_elevation_deg
    )
    return full[:, np.asarray(pixels, dtype=int)]

