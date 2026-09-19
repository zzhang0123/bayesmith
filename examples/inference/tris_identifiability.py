"""P2 identifiability: the whitened Jacobian, its SVD, and prior versus data.

The plan asks for a noise-whitened Jacobian at representative prior and
posterior points, scaled by each parameter's prior width, with an SVD that
separates the directions the data constrain from the directions the prior
supplies.  This module computes exactly that from the model means, in plain
numpy, so it can be checked against finite differences and read without
running a sampler.

The whitened observation vector is (TRIS whitened map, external background in
units of its own sigma), so the likelihood is unit-variance in every row and
the Fisher information is J^T J.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NAMES_M0",
    "NAMES_M1",
    "analyze",
    "coordinate_names",
    "degeneracy_rows",
    "jacobian",
    "parameter_names",
    "prior_sd",
    "to_vector",
    "whitened_mean",
]

NAMES_M0 = (
    "amplitude",
    "beta",
    "zero_standard",
    "haslam_monopole_K",
    "calibration_standard",
)
NAMES_M1 = NAMES_M0 + ("rsb_amplitude", "rsb_beta")


def parameter_names(include_rsb):
    return NAMES_M1 if include_rsb else NAMES_M0


def coordinate_names(include_rsb, *, calibration_coordinates):
    """One label per scalar coordinate, in the packed order."""
    labels = []
    for name, size in (
        ("amplitude", 3),
        ("beta", 3),
        ("zero_standard", 2),
        ("haslam_monopole_K", 1),
        ("calibration_standard", int(calibration_coordinates)),
    ):
        labels.extend([name] if size == 1 else [f"{name}[{i}]" for i in range(size)])
    if include_rsb:
        labels.extend(["rsb_amplitude", "rsb_beta"])
    return labels


def prior_sd(include_rsb, *, calibration_coordinates=2):
    """Marginal prior standard deviations, in the model's own units."""
    uniform = 1.0 / np.sqrt(12.0)
    values = [
        *([2.8 * uniform] * 3),  # amplitude, three regions, Uniform(0.2, 3)
        *([2.5 * uniform] * 3),  # beta, three regions, Uniform(-4, -1.5)
        *([1.0] * 2),  # zero_standard Normal(0, 1)
        3.0,  # haslam_monopole_K Normal(0, 3)
        *([1.0] * calibration_coordinates),  # calibration_standard Normal(0, 1)
    ]
    if include_rsb:
        values.extend([5.0 * uniform, 2.5 * uniform])  # Uniform(0,5), Uniform(-4,-1.5)
    return np.asarray(values, dtype=float)


def to_vector(values, include_rsb):
    names = parameter_names(include_rsb)
    return np.concatenate(
        [np.atleast_1d(np.asarray(values[name], dtype=float)).ravel() for name in names]
    )


def from_vector(theta, include_rsb, *, calibration_coordinates):
    theta = np.asarray(theta, dtype=float)
    sizes = {
        "amplitude": 3,
        "beta": 3,
        "zero_standard": 2,
        "haslam_monopole_K": 1,
        "calibration_standard": int(calibration_coordinates),
        "rsb_amplitude": 1,
        "rsb_beta": 1,
    }
    values, offset = {}, 0
    for name in parameter_names(include_rsb):
        size = sizes[name]
        block = theta[offset : offset + size]
        if block.size != size:
            raise ValueError(f"parameter vector is too short for {name}")
        values[name] = float(block[0]) if size == 1 else block
        offset += size
    if offset != theta.size:
        raise ValueError("parameter vector has trailing entries")
    return values


def whitened_mean(bundle, external, included_surveys, values, include_rsb):
    """The full whitened model mean: TRIS map plus external/sigma."""
    from .tris_forward_baseline import predict_from_inputs
    from .tris_rsb_sky import model_inputs

    inputs = model_inputs(bundle, external, included_surveys, include_rsb)
    tris_mean, external_mean = predict_from_inputs(inputs, values)
    sigma = np.asarray(inputs[11], dtype=float)
    return np.concatenate([tris_mean, external_mean / sigma])


def jacobian(bundle, external, included_surveys, values, include_rsb, *, step=1e-5):
    """Central-difference Jacobian of the whitened mean, (n_observations, n_params)."""
    theta = to_vector(values, include_rsb)
    calibration = int(np.size(np.asarray(values["calibration_standard"])))
    columns = []
    for index in range(theta.size):
        width = step * max(1.0, abs(theta[index]))
        upper, lower = theta.copy(), theta.copy()
        upper[index] += width
        lower[index] -= width
        columns.append(
            (
                whitened_mean(
                    bundle,
                    external,
                    included_surveys,
                    from_vector(upper, include_rsb, calibration_coordinates=calibration),
                    include_rsb,
                )
                - whitened_mean(
                    bundle,
                    external,
                    included_surveys,
                    from_vector(lower, include_rsb, calibration_coordinates=calibration),
                    include_rsb,
                )
            )
            / (2.0 * width)
        )
    return np.array(columns).T


def analyze(bundle, external, included_surveys, values, include_rsb, *, step=1e-5):
    """SVD of the prior-scaled Jacobian plus the posterior correlation."""
    calibration = int(np.size(np.asarray(values["calibration_standard"])))
    names = coordinate_names(include_rsb, calibration_coordinates=calibration)
    sigma_prior = prior_sd(include_rsb, calibration_coordinates=calibration)
    raw = jacobian(bundle, external, included_surveys, values, include_rsb, step=step)
    scaled = raw * sigma_prior[None, :]
    left, singular, right = np.linalg.svd(scaled, full_matrices=False)
    del left
    fisher = scaled.T @ scaled
    # The standardized prior precision is the identity, so the standardized
    # posterior precision is the Fisher information plus one.
    posterior_scaled = np.linalg.inv(fisher + np.eye(sigma_prior.size))
    posterior_sd = np.sqrt(np.diag(posterior_scaled)) * sigma_prior
    correlation = posterior_scaled / np.outer(
        np.sqrt(np.diag(posterior_scaled)), np.sqrt(np.diag(posterior_scaled))
    )
    return {
        "names": list(names),
        "prior_sd": sigma_prior.tolist(),
        "singular_values": singular.tolist(),
        "directions_with_snr_above_one": int(np.count_nonzero(singular > 1.0)),
        "fisher_eigenvalues": (singular**2).tolist(),
        "posterior_sd": posterior_sd.tolist(),
        "posterior_over_prior_sd": (posterior_sd / sigma_prior).tolist(),
        "posterior_correlation": correlation.tolist(),
        "right_vectors": right.tolist(),
        "observation_count": int(raw.shape[0]),
    }


def degeneracy_rows(analysis, *, top=8):
    """Largest absolute posterior correlations, as (a, b, correlation) rows."""
    correlation = np.asarray(analysis["posterior_correlation"], dtype=float)
    names = analysis["names"]
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append((names[i], names[j], float(correlation[i, j])))
    rows.sort(key=lambda row: abs(row[2]), reverse=True)
    return [{"a": a, "b": b, "correlation": c} for a, b, c in rows[:top]]
