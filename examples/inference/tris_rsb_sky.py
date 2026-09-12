"""Joint TRIS × Haslam sky and ARCADE 2/LWA background model."""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints

from bayesmith import const, det, observe, sample

from .tris_sky import amplitude_prior, beta_prior, zero_levels, zero_prior

CMB_THERMODYNAMIC_K = 2.725
HC_OVER_K_MHZ_K = 4.799243073e-5


class PositiveTemplateNormal(dist.Distribution):
    """Unit normal that assigns no density outside the physical sky domain."""

    arg_constraints: ClassVar = {}
    support = constraints.real

    def __init__(self, location, valid, validate_args=None):
        self.location = jnp.asarray(location)
        self.valid = jnp.asarray(valid)
        super().__init__(batch_shape=self.location.shape, validate_args=validate_args)

    def log_prob(self, value):
        normal = dist.Normal(self.location, jnp.ones_like(self.location))
        return jnp.where(self.valid, normal.log_prob(value), -jnp.inf)

    def sample(self, key, sample_shape=()):
        return dist.Normal(self.location, jnp.ones_like(self.location)).sample(key, sample_shape)


def positive_template_normal(location, valid):
    return PositiveTemplateNormal(location, valid)


def rsb_temperature(amplitude, beta, frequency_mhz):
    return amplitude * (jnp.asarray(frequency_mhz) / 1000.0) ** beta


def galactic_template(template_k, haslam_monopole_k, background_408_k):
    return jnp.asarray(template_k) + haslam_monopole_k - background_408_k


def valid_galactic_template(template_k):
    template = jnp.asarray(template_k)
    return jnp.all(jnp.isfinite(template) & (template > 0))


def cmb_rj_temperature(frequency_mhz):
    frequency = jnp.asarray(frequency_mhz)
    x = HC_OVER_K_MHZ_K * frequency / CMB_THERMODYNAMIC_K
    return CMB_THERMODYNAMIC_K * x / jnp.expm1(x)


def calibrated_external_mean(
    cmb_rj_k, rsb_k, calibration_standard, tau_rj_k, survey_code
):
    return cmb_rj_k + rsb_k + calibration_standard[survey_code] * tau_rj_k


def spectral_scaling(amplitude, beta, frequency_mhz):
    return amplitude[None, :] * (frequency_mhz[:, None] / 408.0) ** beta[None, :]


def map_mean(
    scaling,
    template_shift,
    zero_level,
    template_design,
    region_design,
    cmb_response,
    background_response,
    background_frequency,
    offset_response,
    frequency_index,
):
    foreground = jnp.sum(
        (template_design + template_shift * region_design) * scaling[frequency_index],
        axis=1,
    )
    return (
        foreground
        + cmb_response
        + background_response * background_frequency[frequency_index]
        - offset_response * zero_level[frequency_index]
    )


def model(
    template_design,
    region_design,
    cmb_response,
    background_response,
    offset_response,
    frequency_index,
    tris_frequency_mhz,
    whitened_tris_data,
    template_k,
    external_frequency_mhz,
    external_data_rj_k,
    external_sigma_independent_rj_k,
    external_tau_rj_k,
    external_survey_code,
    calibration_coordinates,
    include_rsb,
):
    design = const("Haslam_template_region_response", template_design)
    region = const("Haslam_region_response", region_design)
    cmb = const("RJ_CMB_response", cmb_response)
    background_response_node = const("isotropic_background_response", background_response)
    offset = const("common_offset_response", offset_response)
    channel = const("frequency_index", frequency_index)
    tris_frequency = const("tris_frequency_mhz", tris_frequency_mhz)
    template = const("Haslam_template_k", template_k)
    external_frequency = const("external_frequency_mhz", external_frequency_mhz)
    external_sigma = const("external_sigma_independent_rj_k", external_sigma_independent_rj_k)
    external_tau = const("external_tau_rj_k", external_tau_rj_k)
    external_survey = const("external_survey_code", external_survey_code)
    external_cmb = const("RJ_CMB_external_k", cmb_rj_temperature(external_frequency_mhz))
    amp = sample("amplitude", amplitude_prior)
    beta = sample("beta", beta_prior)
    zero = sample("zero_standard", zero_prior)
    haslam_monopole = sample("haslam_monopole_K", lambda: dist.Normal(0.0, 3.0))
    calibration = sample(
        "calibration_standard",
        lambda: dist.Normal(jnp.zeros(calibration_coordinates), jnp.ones(calibration_coordinates)).to_event(1),
    )
    if include_rsb:
        rsb_amplitude = sample("rsb_amplitude", lambda: dist.Uniform(0.0, 5.0))
        rsb_beta = sample("rsb_beta", lambda: dist.Uniform(-4.0, -1.5))
        background_tris = det("rsb_tris_K", rsb_temperature, rsb_amplitude, rsb_beta, tris_frequency)
        background_external = det("rsb_external_K", rsb_temperature, rsb_amplitude, rsb_beta, external_frequency)
        background_408 = det(
            "rsb_408_K", lambda amplitude, beta: rsb_temperature(amplitude, beta, 408.0), rsb_amplitude, rsb_beta
        )
    else:
        background_tris = const("rsb_tris_K", jnp.zeros_like(tris_frequency_mhz))
        background_external = const("rsb_external_K", jnp.zeros_like(external_frequency_mhz))
        background_408 = const("rsb_408_K", 0.0)
    galactic = det("galactic_template_408_K", galactic_template, template, haslam_monopole, background_408)
    valid = det("positive_galactic_template", valid_galactic_template, galactic)
    scaling = det("frequency_scaling", spectral_scaling, amp, beta, tris_frequency)
    level = det("zero_level_K", zero_levels, zero)
    template_shift = det("galactic_template_shift_K", lambda z, b: z - b, haslam_monopole, background_408)
    tris_mean = det(
        "whitened_map_mean",
        map_mean,
        scaling,
        template_shift,
        level,
        design,
        region,
        cmb,
        background_response_node,
        background_tris,
        offset,
        channel,
    )
    external_mean = det(
        "external_background_mean",
        calibrated_external_mean,
        external_cmb,
        background_external,
        calibration,
        external_tau,
        external_survey,
    )
    observe("tris_obs", positive_template_normal, tris_mean, valid, obs=whitened_tris_data)
    observe("external_obs", lambda mean, sigma: dist.Normal(mean, sigma), external_mean, external_sigma, obs=external_data_rj_k)


def model_inputs(bundle, external, included_surveys, include_rsb):
    included = tuple(included_surveys)
    if not included:
        raise ValueError("included surveys must be nonempty")
    available = {str(name) for name in np.asarray(external["survey"]).tolist()}
    unknown = set(included) - available
    if unknown:
        raise ValueError(f"unknown external survey selection: {sorted(unknown)}")
    if len(set(included)) != len(included):
        raise ValueError("included surveys must be unique")
    template, region = np.asarray(bundle["template_k"]), np.asarray(bundle["region"])
    if template.ndim != 1 or region.shape != template.shape or set(region.tolist()) != {0, 1, 2}:
        raise ValueError("expected a three-region Haslam template")
    if not np.all(np.isfinite(template) & (template > 0)):
        raise ValueError("Haslam foreground template must be positive and finite")
    required = ("frequency_mhz", "cmb_k", "reference_cmb_k")
    if any(name not in bundle for name in required):
        raise ValueError("TRIS input lacks required frequency/CMB metadata")
    masks = region[:, None] == np.arange(3)
    template_basis, region_basis = template[:, None] * masks, masks.astype(float)
    template_design, region_design, cmbs, backgrounds, offsets, channels, data = [], [], [], [], [], [], []
    for index in range(2):
        response = np.asarray(bundle[f"response_{index}"])
        if response.ndim != 2 or response.shape[1] != len(template):
            raise ValueError("TRIS response must map the Haslam pixel grid")
        template_design.append(response @ template_basis)
        region_design.append(response @ region_basis)
        cmbs.append(response.sum(axis=1) * np.asarray(bundle["cmb_k"])[index])
        backgrounds.append(response.sum(axis=1))
        offsets.append(np.asarray(bundle[f"offset_response_{index}"]))
        channels.append(np.full(response.shape[0], index, dtype=np.int32))
        data.append(np.asarray(bundle[f"whitened_data_{index}"]))
    mask = np.isin(np.asarray(external["survey"]), included)
    if not mask.any():
        raise ValueError("included surveys select no external observations")
    labels = np.asarray(external["survey"])[mask]
    remapped = np.asarray([included.index(str(label)) for label in labels], dtype=np.int32)
    external_arrays = [
        np.asarray(external[name])[mask]
        for name in ("frequency_mhz", "temperature_rj_k", "sigma_independent_rj_k", "tau_rj_k")
    ]
    if any(array.ndim != 1 or not np.all(np.isfinite(array)) for array in external_arrays):
        raise ValueError("external data must be finite one-dimensional arrays")
    if np.any(external_arrays[2] <= 0) or np.any(external_arrays[3] < 0):
        raise ValueError("external uncertainty scales are invalid")
    return tuple(
        jnp.asarray(value)
        for value in (
            np.concatenate(template_design),
            np.concatenate(region_design),
            np.concatenate(cmbs),
            np.concatenate(backgrounds),
            np.concatenate(offsets),
            np.concatenate(channels),
            np.asarray(bundle["frequency_mhz"]),
            np.concatenate(data),
            template,
            *external_arrays,
            remapped,
            len(included),
            bool(include_rsb),
        )
    )
