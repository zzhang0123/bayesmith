"""Haslam amplitude × frequency power law, specialized to three sky regions.

Forward-equation provenance: BellaNasirudin/bayesian_skymap, revision
e7b8cb34e1a872d11a219f6215e38791058f4e14, apply_beam/calc_model_spectral.
This is an independent JAX implementation; no upstream sampler is imported.
TRIS's measured beam replaces that repository's frequency-scaled Gaussian.
"""

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import const, det, observe, sample

REGIONS = ("plane |b| < 10°", "transition 10°–30°", "high latitude |b| ≥ 30°")


def amplitude_prior():
    return dist.Uniform(jnp.full(3, 0.2), jnp.full(3, 3.0)).to_event(1)


def beta_prior():
    return dist.Uniform(jnp.full(3, -4.0), jnp.full(3, -1.5)).to_event(1)


def zero_prior():
    return dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1)


def spectral_scaling(amplitude, beta, frequency_mhz):
    return amplitude[None, :] * (frequency_mhz[:, None] / 408.0) ** beta[None, :]


def zero_levels(zero_standard):
    """Corrections added to measured temperatures, shared over each ring.

    820 MHz: an explicit equal-side-mass, two-piece Gaussian convention for
    the published -0.300/+0.430 K scales. The archive does not specify a PDF.
    """
    return zero_standard * jnp.array(
        [0.066, jnp.where(zero_standard[1] < 0, 0.300, 0.430)]
    )


def map_mean(
    scaling, zero_level, design, cmb_response, offset_response, frequency_index
):
    return (
        jnp.sum(design * scaling[frequency_index], axis=1)
        + cmb_response
        - offset_response * zero_level[frequency_index]
    )


def unit_normal(location):
    return dist.Normal(location, 1.0)


def model(design, cmb_response, offset_response, frequency_index, frequency_mhz, data):
    d = const("Haslam_region_response", design)
    cmb = const("RJ_CMB_response", cmb_response)
    offset = const("common_offset_response", offset_response)
    channel = const("frequency_index", frequency_index)
    frequency = const("frequency_mhz", frequency_mhz)
    amp = sample("amplitude", amplitude_prior)
    beta = sample("beta", beta_prior)
    zero = sample("zero_standard", zero_prior)
    scaling = det(
        "frequency_scaling",
        spectral_scaling,
        amp,
        beta,
        frequency,
        linear_in=("amplitude",),
    )
    level = det("zero_level_K", zero_levels, zero)
    mean = det("whitened_map_mean", map_mean, scaling, level, d, cmb, offset, channel)
    observe("obs", unit_normal, mean, obs=data)


def model_inputs(bundle):
    template, region = bundle["template_k"], bundle["region"]
    if (
        template.ndim != 1
        or region.shape != template.shape
        or set(region.tolist()) != {0, 1, 2}
    ):
        raise ValueError(
            "expected a Haslam template and three nonempty latitude regions"
        )
    if not np.all(np.isfinite(template) & (template > 0)):
        raise ValueError("Haslam foreground template must be positive and finite")
    basis = template[:, None] * (region[:, None] == np.arange(3))
    designs, cmbs, offsets, channels, data = [], [], [], [], []
    for i in range(2):
        response = bundle[f"response_{i}"]
        designs.append(response @ basis)
        cmbs.append(response.sum(axis=1) * bundle["cmb_k"][i])
        offsets.append(bundle[f"offset_response_{i}"])
        channels.append(np.full(len(response), i, dtype=np.int32))
        data.append(bundle[f"whitened_data_{i}"])
    return tuple(
        jnp.asarray(x)
        for x in (
            np.concatenate(designs),
            np.concatenate(cmbs),
            np.concatenate(offsets),
            np.concatenate(channels),
            bundle["frequency_mhz"],
            np.concatenate(data),
        )
    )


def sky_draws(bundle, amplitude, beta):
    """Unsmoothed sky on the declared grid, draws × frequency × pixel."""
    scale = (
        np.asarray(amplitude)[:, None, :]
        * (bundle["frequency_mhz"][None, :, None] / 408.0)
        ** np.asarray(beta)[:, None, :]
    )
    return (
        scale[:, :, bundle["region"]] * bundle["template_k"][None, None, :]
        + bundle["cmb_k"][None, :, None]
    )
