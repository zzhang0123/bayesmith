"""P1 forward baseline: independent prediction and log-density components.

Batch B of the TRIS handoff plan asks for a fixed-beam forward baseline whose
per-parameter log joint can be checked component by component, for both the
frozen operator and the P0-audited operator.  This module recomputes, in plain
numpy, what the bayesmith graph computes in JAX:

* the whitened TRIS map mean and the external-background mean,
* the log prior, split into its uniform and normal parts,
* the TRIS and external log likelihoods,
* the positive-template support indicator,

so that a mismatch can be attributed to a named component instead of showing up
as one number that disagrees.  It is an independent implementation, not a
wrapper: it never calls the graph.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "batch_mean_mcse",
    "cmb_rj_temperature",
    "log_joint_components",
    "predict_from_bundle",
    "predict_from_inputs",
    "summarize_draws",
]

_HALF_LOG_2PI = 0.5 * np.log(2.0 * np.pi)
_CMB_THERMODYNAMIC_K = 2.725
_HC_OVER_K_MHZ_K = 4.799243073e-5


def cmb_rj_temperature(frequency_mhz):
    """Rayleigh-Jeans brightness temperature of the CMB monopole, in K."""
    frequency = np.asarray(frequency_mhz, dtype=float)
    x = _HC_OVER_K_MHZ_K * frequency / _CMB_THERMODYNAMIC_K
    return _CMB_THERMODYNAMIC_K * x / np.expm1(x)


def zero_levels(zero_standard):
    zero = np.asarray(zero_standard, dtype=float)
    return zero * np.array([0.066, 0.300 if zero[1] < 0 else 0.430])


def spectral_scaling(amplitude, beta, frequency_mhz):
    amplitude = np.asarray(amplitude, dtype=float)
    beta = np.asarray(beta, dtype=float)
    frequency = np.asarray(frequency_mhz, dtype=float)
    return amplitude[None, :] * (frequency[:, None] / 408.0) ** beta[None, :]


def rsb_temperature(amplitude, beta, frequency_mhz):
    return float(amplitude) * (np.asarray(frequency_mhz, dtype=float) / 1000.0) ** float(beta)


def _backgrounds(values, include_rsb, tris_frequency, external_frequency):
    if include_rsb:
        amplitude = float(values["rsb_amplitude"])
        beta = float(values["rsb_beta"])
        return (
            rsb_temperature(amplitude, beta, tris_frequency),
            rsb_temperature(amplitude, beta, external_frequency),
            rsb_temperature(amplitude, beta, np.array([408.0]))[0],
        )
    return (
        np.zeros_like(np.asarray(tris_frequency, dtype=float)),
        np.zeros_like(np.asarray(external_frequency, dtype=float)),
        0.0,
    )


def _galactic(values, template_k, background_408):
    return (
        np.asarray(template_k, dtype=float)
        + float(values["haslam_monopole_K"])
        - float(background_408)
    )


def predict_from_inputs(inputs, values):
    """Recompute both model means from the arrays the graph was given."""
    (
        template_design,
        region_design,
        cmb_response,
        background_response,
        offset_response,
        frequency_index,
        tris_frequency,
        _whitened_data,
        _template_k,
        external_frequency,
        _external_data,
        _external_sigma,
        external_tau,
        external_survey,
        _calibration_coordinates,
        include_rsb,
    ) = inputs
    template_design = np.asarray(template_design, dtype=float)
    region_design = np.asarray(region_design, dtype=float)
    cmb_response = np.asarray(cmb_response, dtype=float)
    background_response = np.asarray(background_response, dtype=float)
    offset_response = np.asarray(offset_response, dtype=float)
    frequency_index = np.asarray(frequency_index, dtype=int)
    tris_frequency = np.asarray(tris_frequency, dtype=float)
    external_frequency = np.asarray(external_frequency, dtype=float)
    external_tau = np.asarray(external_tau, dtype=float)
    external_survey = np.asarray(external_survey, dtype=int)

    tris_background, external_background, background_408 = _backgrounds(
        values, include_rsb, tris_frequency, external_frequency
    )
    scaling = spectral_scaling(values["amplitude"], values["beta"], tris_frequency)
    template_shift = float(values["haslam_monopole_K"]) - float(background_408)
    foreground = np.sum(
        (template_design + template_shift * region_design) * scaling[frequency_index],
        axis=1,
    )
    tris_mean = (
        foreground
        + cmb_response
        + background_response * tris_background[frequency_index]
        - offset_response * zero_levels(values["zero_standard"])[frequency_index]
    )
    external_mean = (
        cmb_rj_temperature(external_frequency)
        + external_background
        + np.asarray(values["calibration_standard"], dtype=float)[external_survey]
        * external_tau
    )
    return tris_mean, external_mean


def predict_from_bundle(bundle, external, included_surveys, values, include_rsb):
    """Recompute both model means straight from the raw bundle and external arrays.

    This is the half of the parity check that does not reuse the graph's
    precomputed designs, so a wrong mask, region assignment or response
    multiplication shows up here.
    """
    from .tris_rsb_sky import model_inputs

    template = np.asarray(bundle["template_k"], dtype=float)
    region = np.asarray(bundle["region"], dtype=int)
    tris_frequency = np.asarray(bundle["frequency_mhz"], dtype=float)
    amplitude = np.asarray(values["amplitude"], dtype=float)
    beta = np.asarray(values["beta"], dtype=float)
    tris_background, _external_background, background_408 = _backgrounds(
        values,
        include_rsb,
        tris_frequency,
        np.asarray(external["frequency_mhz"], dtype=float),
    )
    tris_mean = []
    for index in range(len(tris_frequency)):
        response = np.asarray(bundle[f"response_{index}"], dtype=float)
        scaling = amplitude[region] * (tris_frequency[index] / 408.0) ** beta[region]
        # The graph multiplies (template + shift) by the channel scaling, so the
        # monopole and RSB shift carry the same regional index.  Reproduce that
        # exactly; it is a property of the model, not an approximation here.
        sky = template + (float(values["haslam_monopole_K"]) - float(background_408))
        sky = sky * scaling
        level = zero_levels(values["zero_standard"])[index]
        tris_mean.append(
            response @ sky
            + response.sum(axis=1)
            * (float(bundle["cmb_k"][index]) + tris_background[index])
            - np.asarray(bundle[f"offset_response_{index}"], dtype=float) * level
        )
    inputs = model_inputs(bundle, external, included_surveys, include_rsb)
    _tris_mean_inputs, external_mean = predict_from_inputs(inputs, values)
    return np.concatenate(tris_mean), external_mean


def log_joint_components(bundle, external, included_surveys, values, include_rsb):
    """Independent log prior, log likelihoods, support indicator and total."""
    from .tris_rsb_sky import model_inputs

    inputs = model_inputs(bundle, external, included_surveys, include_rsb)
    (
        _template_design,
        _region_design,
        _cmb_response,
        _background_response,
        _offset_response,
        _frequency_index,
        _tris_frequency,
        whitened_data,
        template_k,
        _external_frequency,
        external_data,
        external_sigma,
        _external_tau,
        _external_survey,
        _calibration_coordinates,
        _include_rsb,
    ) = inputs
    tris_mean, external_mean = predict_from_inputs(inputs, values)

    amplitude = np.asarray(values["amplitude"], dtype=float)
    beta = np.asarray(values["beta"], dtype=float)
    zero = np.asarray(values["zero_standard"], dtype=float)
    calibration = np.asarray(values["calibration_standard"], dtype=float)
    valid = bool(np.all(np.isfinite(tris_mean)))

    prior = 0.0
    uniform_terms = [(-np.log(3.0 - 0.2), np.all((amplitude >= 0.2) & (amplitude <= 3.0))),
                     (-np.log(-1.5 - -4.0), np.all((beta >= -4.0) & (beta <= -1.5)))]
    for log_norm, inside in uniform_terms:
        prior += 3 * log_norm if inside else -np.inf
    prior += float(np.sum(-0.5 * zero**2 - _HALF_LOG_2PI))
    prior += -0.5 * (float(values["haslam_monopole_K"]) / 3.0) ** 2 - np.log(3.0) - _HALF_LOG_2PI
    prior += float(np.sum(-0.5 * calibration**2 - _HALF_LOG_2PI))
    if include_rsb:
        rsb_amplitude = float(values["rsb_amplitude"])
        rsb_beta = float(values["rsb_beta"])
        prior += -np.log(5.0) if 0.0 <= rsb_amplitude <= 5.0 else -np.inf
        prior += -np.log(2.5) if -4.0 <= rsb_beta <= -1.5 else -np.inf

    residual_tris = np.asarray(whitened_data, dtype=float) - tris_mean
    log_likelihood_tris = float(
        -0.5 * np.sum(residual_tris**2) - residual_tris.size * _HALF_LOG_2PI
    )
    residual_external = (
        np.asarray(external_data, dtype=float) - external_mean
    ) / np.asarray(external_sigma, dtype=float)
    log_likelihood_external = float(
        -0.5 * np.sum(residual_external**2)
        - np.sum(np.log(np.asarray(external_sigma, dtype=float)))
        - np.asarray(external_sigma).size * _HALF_LOG_2PI
    )

    # The positive-template support multiplies the TRIS density by zero.  It is
    # reported separately so a -inf total can be attributed.
    galactic = _galactic(values, template_k, _background_408_of(values, include_rsb))
    support_valid = bool(np.all(np.isfinite(galactic) & (galactic > 0.0)))
    total = prior + log_likelihood_tris + log_likelihood_external
    if not support_valid:
        total = -np.inf
    return {
        "log_prior": float(prior),
        "log_likelihood_tris": log_likelihood_tris,
        "log_likelihood_external": log_likelihood_external,
        "support_valid": support_valid,
        "finite_prediction": valid,
        "log_joint": float(total),
    }


def _background_408_of(values, include_rsb):
    if include_rsb:
        return rsb_temperature(
            float(values["rsb_amplitude"]), float(values["rsb_beta"]), np.array([408.0])
        )[0]
    return 0.0


def batch_mean_mcse(values, *, batches=20):
    """Batch-means Monte Carlo standard error of the mean, per parameter.

    values has shape (chains, draws, ...).  Each chain is split into batches;
    the standard error is the spread of the pooled batch means over the square
    root of their count, which is conservative but needs no arviz.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim < 2:
        raise ValueError("values must have chain and draw axes")
    chains, draws = values.shape[:2]
    width = draws // batches
    if width < 1:
        raise ValueError("not enough draws for the requested batch count")
    trimmed = values[:, : width * batches]
    shape = (chains, batches, width) + values.shape[2:]
    batch_means = trimmed.reshape(shape).mean(axis=2).reshape(-1, *values.shape[2:])
    return batch_means.std(axis=0, ddof=1) / np.sqrt(batch_means.shape[0])


def summarize_draws(draws):
    """Mean, sd and MCSE for each parameter in a samples dict."""
    summary = {}
    for name, values in draws.items():
        values = np.asarray(values, dtype=float)
        summary[name] = {
            "mean": values.mean(axis=(0, 1)).tolist(),
            "sd": values.std(axis=(0, 1), ddof=1).tolist(),
            "mcse": batch_mean_mcse(values).tolist(),
        }
    return summary
