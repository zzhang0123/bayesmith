"""Joint TRIS × Haslam sky and ARCADE 2/LWA background model."""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.truncated import LeftTruncatedDistribution

from bayesmith import const, det, joint_prior, observe, sample

from .tris_sky import amplitude_prior, beta_prior, zero_levels, zero_prior

CMB_THERMODYNAMIC_K = 2.725
HC_OVER_K_MHZ_K = 4.799243073e-5


MONOPOLE_SCALE_K = 3.0


class StableLeftTruncatedNormal(LeftTruncatedDistribution):
    """A left-truncated normal whose log density does not underflow.

    numpyro's LeftTruncatedDistribution (what dist.TruncatedNormal returns for
    a low bound) computes its normalizer from the CDF and returns +inf once the
    truncated tail is below double precision -- measured at low about 151 K,
    where the true log density is about -14.  That made the joint density +inf
    at a prior-allowed point.  This uses the stable log survival instead; the
    support transform is inherited unchanged.
    """

    def sample(self, key, sample_shape=()):
        """Tail-safe rejection sampling, without subtracting rounded CDFs."""
        shape = sample_shape + self.batch_shape
        low = jnp.broadcast_to((self.low - self.base_dist.loc) / self.base_dist.scale, shape)
        alpha = 0.5 * (jnp.maximum(low, 0.0) + jnp.hypot(low, 2.0))
        initial = (key, jnp.zeros(shape), jnp.zeros(shape, dtype=bool))

        def step(state):
            key, value, accepted = state
            key, normal_key, exp_key, accept_key = jax.random.split(key, 4)
            normal = jax.random.normal(normal_key, shape)
            tail = low + jax.random.exponential(exp_key, shape) / alpha
            proposal = jnp.where(low >= 0.0, tail, normal)
            log_accept = -0.5 * (tail - alpha)**2
            accept = jnp.where(low >= 0.0,
                               jnp.log(jax.random.uniform(accept_key, shape)) < log_accept,
                               normal > low)
            update = accept & ~accepted
            return key, jnp.where(update, proposal, value), accepted | accept

        _, value, _ = jax.lax.while_loop(lambda state: ~jnp.all(state[2]), step, initial)
        return self.base_dist.loc + self.base_dist.scale * value

    def log_prob(self, value):
        loc = self.base_dist.loc
        scale = self.base_dist.scale
        z = (value - loc) / scale
        low_z = (self.low - loc) / scale
        return (
            dist.Normal(0.0, 1.0).log_prob(z)
            - jnp.log(scale)
            - jax.scipy.special.log_ndtr(-low_z)
        )


def stable_left_truncated_normal(low, scale=MONOPOLE_SCALE_K):
    """The monopole distribution: Normal(0, scale) truncated to (low, inf)."""
    return StableLeftTruncatedNormal(dist.Normal(0.0, scale), low=low)


def log_positive_template_survival(lower, scale=MONOPOLE_SCALE_K):
    """log P(Normal(0, scale) > lower), stable for large lower.

    The previous (cdf(lower + 30) - cdf(lower)) underflowed to -inf at
    lower = 25 K, where the correct value is about -37.775.  log_ndtr(-x) is
    the log survival function and has no cancellation.
    """
    return jax.scipy.special.log_ndtr(-jnp.asarray(lower) / scale)


class PositiveTemplatePrior(eqx.Module):
    """The RSB prior that moves the positive-template constraint into a SUPPORT.

    G408 = template + monopole - B408 must be positive. Enforcing that with
    only the ``-inf`` gate in :class:`PositiveTemplateNormal` leaves the
    constraint as a wall in an unconstrained monopole coordinate: every NUTS
    trajectory that crosses diverges, dual averaging collapses the step size,
    and the chain freezes (measured on the RA folds: 2786-4000 divergences and
    an ESS of about 2 for every site). The monopole now carries a
    TruncatedNormal prior whose lower bound is ``-min(template) + B408``, so the
    constraint lives in the sampler's transform. A truncated normal
    renormalizes by ``Z(B408) = P(Normal(0, 3) > lower)``, which depends on the
    RSB parameters; this graph-level prior multiplies that back
    (``log U + log U + log Z - log C``) so the target is the ORIGINAL
    unnormalized prior ``Uniform(A) Uniform(beta) Normal(0, 3)(m)`` restricted
    to ``G408 > 0``. The RSB latents are declared flat (ImproperUniform); this
    declaration is their whole prior.
    """

    over: tuple[str, ...] = eqx.field(static=True)
    has_rsb: bool = eqx.field(static=True)
    template_min_k: float = eqx.field(static=True)
    monopole_scale: float = eqx.field(static=True)
    amplitude_bounds: tuple[float, float] = eqx.field(static=True)
    beta_bounds: tuple[float, float] = eqx.field(static=True)

    def log_density(self, graph, values):
        """The graph-level prior that makes ``log_joint`` the original target.

        With RSB (``has_rsb``): ``log U(A) + log U(beta) + log Z(B408)``.
        Together with the monopole's TruncatedNormal, whose density carries
        ``-log Z``, the sum is the original unnormalized
        ``Uniform(A) Uniform(beta) Normal(0, 3)(m)`` exactly.

        Without RSB: ``log Normal(0, 3)(m)``, the monopole's whole prior when
        the monopole itself is declared flat and constrained to the domain.

        The box normalization ``C`` is deliberately NOT subtracted: the
        posterior is proportional to the target, so the constant is
        irrelevant to sampling, and dropping it keeps ``log_joint`` equal to
        the original rather than equal up to a constant.
        """
        if not self.has_rsb:
            return jnp.sum(
                dist.Normal(0.0, self.monopole_scale).log_prob(
                    values["haslam_monopole_K"]
                )
            )
        amplitude = values["rsb_amplitude"]
        beta = values["rsb_beta"]
        lower = -self.template_min_k + amplitude * 0.408**beta
        # Stable log survival.  The previous (cdf(lower + 30) - cdf(lower))
        # underflowed to log(0) = -inf at lower = 25 K, where the correct value
        # is about -37.775; log_ndtr(-lower / scale) has no cancellation.
        log_z = log_positive_template_survival(lower, self.monopole_scale)
        low, high = self.amplitude_bounds
        beta_low, beta_high = self.beta_bounds
        total = (
            jnp.log(1.0 / (high - low))
            + jnp.log(1.0 / (beta_high - beta_low))
            + log_z
        )
        return jnp.sum(total)


def positive_template_prior(template_min_k):
    """The graph-level RSB prior that cancels the monopole truncation."""
    return PositiveTemplatePrior(
        over=("rsb_amplitude", "rsb_beta"),
        has_rsb=True,
        template_min_k=float(template_min_k),
        monopole_scale=MONOPOLE_SCALE_K,
        amplitude_bounds=(0.0, 5.0),
        beta_bounds=(-4.0, -1.5),
    )


def positive_template_monopole_prior():
    """The monopole's whole prior when no RSB is sampled."""
    return PositiveTemplatePrior(
        over=("haslam_monopole_K",),
        has_rsb=False,
        template_min_k=0.0,
        monopole_scale=MONOPOLE_SCALE_K,
        amplitude_bounds=(0.0, 5.0),
        beta_bounds=(-4.0, -1.5),
    )


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
    template_min_k = float(np.min(template_k))
    if include_rsb:
        # Flat latents: positive_template_prior is their whole prior.
        rsb_amplitude = sample("rsb_amplitude", lambda: dist.ImproperUniform(constraints.interval(0.0, 5.0), (), ()))
        rsb_beta = sample("rsb_beta", lambda: dist.ImproperUniform(constraints.interval(-4.0, -1.5), (), ()))
        background_tris = det("rsb_tris_K", rsb_temperature, rsb_amplitude, rsb_beta, tris_frequency)
        background_external = det("rsb_external_K", rsb_temperature, rsb_amplitude, rsb_beta, external_frequency)
        background_408 = det(
            "rsb_408_K", lambda amplitude, beta: rsb_temperature(amplitude, beta, 408.0), rsb_amplitude, rsb_beta
        )
        joint_prior(positive_template_prior(template_min_k))
    else:
        background_tris = const("rsb_tris_K", jnp.zeros_like(tris_frequency_mhz))
        background_external = const("rsb_external_K", jnp.zeros_like(external_frequency_mhz))
        background_408 = const("rsb_408_K", 0.0)
        joint_prior(positive_template_monopole_prior())
    if include_rsb:
        haslam_monopole = sample(
            "haslam_monopole_K",
            lambda template_value, background: stable_left_truncated_normal(
                -jnp.min(template_value) + background
            ),
            template,
            background_408,
        )
    else:
        haslam_monopole = sample(
            "haslam_monopole_K",
            lambda: dist.ImproperUniform(
                constraints.greater_than(-template_min_k),
                (),
                (),
            ),
        )
    if calibration_coordinates:
        calibration = sample(
            "calibration_standard",
            lambda: dist.Normal(jnp.zeros(calibration_coordinates), jnp.ones(calibration_coordinates)).to_event(1),
        )
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
    if calibration_coordinates:
        external_mean = det(
            "external_background_mean", calibrated_external_mean, external_cmb,
            background_external, calibration, external_tau, external_survey,
        )
    else:
        external_mean = det("external_background_mean", lambda cmb, background: cmb + background,
                            external_cmb, background_external)
    observe("tris_obs", positive_template_normal, tris_mean, valid, obs=whitened_tris_data)
    if external_sigma_independent_rj_k.ndim == 2:
        observe("external_obs", lambda mean, covariance: dist.MultivariateNormal(
            mean, covariance_matrix=covariance), external_mean, external_sigma, obs=external_data_rj_k)
    else:
        observe("external_obs", lambda mean, sigma: dist.Normal(mean, sigma),
                external_mean, external_sigma, obs=external_data_rj_k)


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
    inclusive_covariance = "covariance_rj_k2" in external
    if inclusive_covariance:
        from examples.inference.tris_rsb_external import covariance_for_rows

        external_arrays = [np.asarray(external[name])[mask]
                           for name in ("frequency_mhz", "temperature_rj_k")]
        external_arrays += [covariance_for_rows(external, mask), np.zeros(mask.sum())]
    else:
        external_arrays = [
            np.asarray(external[name])[mask]
            for name in ("frequency_mhz", "temperature_rj_k", "sigma_independent_rj_k", "tau_rj_k")
        ]
        if np.any(external_arrays[2] <= 0) or np.any(external_arrays[3] < 0):
            raise ValueError("external uncertainty scales are invalid")
    if any(not np.all(np.isfinite(array)) or array.ndim != (2 if inclusive_covariance and i == 2 else 1)
           for i, array in enumerate(external_arrays)):
        raise ValueError("external data must be finite aligned arrays")
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
            0 if inclusive_covariance else len(included),
            bool(include_rsb),
        )
    )
