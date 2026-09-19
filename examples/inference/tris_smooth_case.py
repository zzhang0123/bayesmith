"""Smooth-foreground M0/M1 case, to test whether the RSB preference survives.

Batch D2 showed by MAP that a smooth latitude basis fits the TRIS profiles far
better than the rigid three regions.  If that extra flexibility is what the RSB
term has been absorbing, the M1 preference should shrink.  This module samples
the smooth foreground with the same external data, priors and budget as the
3-region case, with and without the RSB term, and reports the same diagnostics.

The foreground is amplitude(p) = exp(phi(p) @ a) and beta(p) = phi(p) @ b with
phi a raised-cosine basis in Galactic latitude, so positivity is structural
rather than a support check on the amplitude.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from scipy.optimize import least_squares

from bayesmith import (
    compile_task,
    const,
    det,
    execute_task,
    joint_prior,
    observe,
    sample,
    trace,
)
from bayesmith.artifacts import (
    ComputeBudget,
    DrawsPosterior,
    InitializationPolicy,
    NamedArray,
    PosteriorResult,
    PosteriorTask,
    Refusal,
    StoppingPolicy,
    dump_artifact,
    model_ref_from_callable,
    new_task_meta,
)
from examples.inference.common import diagnostic_checks, require_result, samples
from examples.inference.tris_batch_d2 import raised_cosine_basis
from examples.inference.tris_rsb_compare import heldout_log_predictive_density
from examples.inference.tris_rsb_sky import (
    calibrated_external_mean,
    cmb_rj_temperature,
    positive_template_monopole_prior,
    positive_template_normal,
    positive_template_prior,
    rsb_temperature,
    stable_left_truncated_normal,
    valid_galactic_template,
)
from examples.inference.tris_sky import zero_levels

SURVEYS = ("LWA", "ARCADE")
FUNCTIONS = 5
# How far inside the positive-template domain a repaired start is placed, in K.
# The support is a hard -inf cliff; see _repair_positive_template.
POSITIVE_TEMPLATE_START_MARGIN_K = 1.0


def basis_matrix(bundle, functions=FUNCTIONS):
    return raised_cosine_basis(
        np.asarray(bundle["galactic_latitude_deg"], dtype=float), functions=functions
    )


def smooth_inputs(bundle, external, included_surveys, basis, include_rsb):
    """Arrays for the smooth model, in a fixed order."""
    template = np.asarray(bundle["template_k"], dtype=float)
    cmb_k = np.asarray(bundle["cmb_k"], dtype=float)
    frequency = np.asarray(bundle["frequency_mhz"], dtype=float)
    designs, offsets, channels, data = [], [], [], []
    for index in range(len(frequency)):
        response = np.asarray(bundle[f"response_{index}"], dtype=float)
        designs.append(response)
        offsets.append(np.asarray(bundle[f"offset_response_{index}"], dtype=float))
        channels.append(np.full(response.shape[0], index, dtype=np.int32))
        data.append(np.asarray(bundle[f"whitened_data_{index}"], dtype=float))
    included = tuple(included_surveys)
    labels = np.asarray(external["survey"])
    mask = np.isin(labels, list(included))
    if not mask.any():
        raise ValueError("included surveys select no external observations")
    selected = np.asarray([included.index(str(label)) for label in labels[mask]], dtype=np.int32)
    external_arrays = [
        np.asarray(external[name])[mask]
        for name in ("frequency_mhz", "temperature_rj_k", "sigma_independent_rj_k", "tau_rj_k")
    ]
    return tuple(
        jnp.asarray(value)
        for value in (
            np.asarray(basis, dtype=float),
            designs[0],
            designs[1],
            offsets[0],
            offsets[1],
            template,
            cmb_k,
            frequency,
            np.concatenate(channels),
            np.concatenate(data),
            *external_arrays,
            selected,
            len(included),
            bool(include_rsb),
        )
    )


def _sky_per_frequency(frequencies, log_amplitude, beta_field, template, shift, cmb, background):
    amplitude = jnp.exp(log_amplitude)[None, :]
    scale = (frequencies[:, None] / 408.0) ** beta_field[None, :]
    return (template[None, :] + shift) * amplitude * scale + cmb[:, None] + background[:, None]


def _tris_mean(sky, response_0, response_1, offset_0, offset_1, level):
    mean0 = sky[0] @ response_0.T - offset_0 * level[0]
    mean1 = sky[1] @ response_1.T - offset_1 * level[1]
    return jnp.concatenate([mean0, mean1])


def model(
    basis,
    response_0,
    response_1,
    offset_0,
    offset_1,
    template_k,
    cmb_k,
    frequency_mhz,
    frequency_index,
    whitened_tris_data,
    external_frequency_mhz,
    external_data_rj_k,
    external_sigma_independent_rj_k,
    external_tau_rj_k,
    external_survey_code,
    calibration_coordinates,
    include_rsb,
):
    functions = int(np.shape(basis)[1])
    basis_node = const("smooth_basis", basis)
    response_0_node = const("response_0", response_0)
    response_1_node = const("response_1", response_1)
    offset_0_node = const("offset_response_0", offset_0)
    offset_1_node = const("offset_response_1", offset_1)
    template = const("Haslam_template_k", template_k)
    cmb = const("cmb_k", cmb_k)
    frequencies = const("frequency_mhz", frequency_mhz)
    external_frequency = const("external_frequency_mhz", external_frequency_mhz)
    # cmb_rj_temperature is evaluated on the raw argument, as in the RSB case:
    # calling it on a const node would hand a NodeRef to jnp.asarray.
    external_cmb = const("RJ_CMB_external_k", cmb_rj_temperature(external_frequency_mhz))
    external_sigma = const("external_sigma_independent_rj_k", external_sigma_independent_rj_k)
    external_tau = const("external_tau_rj_k", external_tau_rj_k)
    external_survey = const("external_survey_code", external_survey_code)
    coefficients = sample(
        "log_amplitude_coefficients",
        lambda: dist.Normal(
            jnp.full(functions, np.log(1.6)), jnp.full(functions, 0.5)
        ).to_event(1),
    )
    beta_coefficients = sample(
        "beta_coefficients",
        lambda: dist.Normal(
            jnp.full(functions, -2.75), jnp.full(functions, 0.5)
        ).to_event(1),
    )
    zero = sample("zero_standard", lambda: dist.Normal(jnp.zeros(2), jnp.ones(2)).to_event(1))
    template_min_k = float(np.min(template_k))
    if include_rsb:
        # Flat latents: positive_template_prior is their whole prior.
        rsb_amplitude = sample("rsb_amplitude", lambda: dist.ImproperUniform(dist.constraints.interval(0.0, 5.0), (), ()))
        rsb_beta = sample("rsb_beta", lambda: dist.ImproperUniform(dist.constraints.interval(-4.0, -1.5), (), ()))
        background_tris = det(
            "rsb_tris_K", rsb_temperature, rsb_amplitude, rsb_beta, frequencies
        )
        background_external = det(
            "rsb_external_K",
            rsb_temperature,
            rsb_amplitude,
            rsb_beta,
            external_frequency,
        )
        background_408 = det(
            "rsb_408_K",
            lambda amplitude, beta: rsb_temperature(amplitude, beta, 408.0),
            rsb_amplitude,
            rsb_beta,
        )
        joint_prior(positive_template_prior(template_min_k))
    else:
        background_tris = const("rsb_tris_K", jnp.zeros_like(frequency_mhz))
        background_external = const("rsb_external_K", jnp.zeros_like(external_frequency_mhz))
        background_408 = const("rsb_408_K", 0.0)
        joint_prior(positive_template_monopole_prior())
    if include_rsb:
        monopole = sample(
            "haslam_monopole_K",
            lambda template_value, background: stable_left_truncated_normal(
                -jnp.min(template_value) + background
            ),
            template,
            background_408,
        )
    else:
        monopole = sample(
            "haslam_monopole_K",
            lambda: dist.ImproperUniform(
                dist.constraints.greater_than(-template_min_k),
                (),
                (),
            ),
        )
    calibration = sample(
        "calibration_standard",
        lambda: dist.Normal(
            jnp.zeros(calibration_coordinates), jnp.ones(calibration_coordinates)
        ).to_event(1),
    )
    # The basis must be an ARGUMENT of the det function, not a closed-over const
    # node: only arguments are evaluated to traced values before the call.
    log_amplitude = det(
        "log_amplitude_field",
        lambda basis, coefficients: basis @ coefficients,
        basis_node,
        coefficients,
    )
    beta_field = det(
        "beta_field",
        lambda basis, coefficients: basis @ coefficients,
        basis_node,
        beta_coefficients,
    )
    shift = det("template_shift_K", lambda z, b: z - b, monopole, background_408)
    galactic = det("galactic_template_408_K", lambda t, s: t + s, template, shift)
    valid = det("positive_galactic_template", valid_galactic_template, galactic)
    level = det("zero_level_K", zero_levels, zero)
    sky = det(
        "sky_per_frequency",
        _sky_per_frequency,
        frequencies,
        log_amplitude,
        beta_field,
        template,
        shift,
        cmb,
        background_tris,
    )
    tris_mean = det(
        "whitened_map_mean",
        _tris_mean,
        sky,
        response_0_node,
        response_1_node,
        offset_0_node,
        offset_1_node,
        level,
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
    observe(
        "tris_obs", positive_template_normal, tris_mean, valid, obs=whitened_tris_data
    )
    observe(
        "external_obs",
        lambda mean, sigma: dist.Normal(mean, sigma),
        external_mean,
        external_sigma,
        obs=external_data_rj_k,
    )


def _smooth_whitened_mean(bundle, external, surveys, basis, x, include_rsb):
    functions = basis.shape[1]
    amplitude = np.exp(basis @ x[:functions])
    beta = basis @ x[functions : 2 * functions]
    zero = x[2 * functions : 2 * functions + 2]
    monopole = x[2 * functions + 2]
    calibration = x[2 * functions + 3 : 2 * functions + 3 + len(surveys)]
    offset = 2 * functions + 3 + len(surveys)
    if include_rsb:
        rsb_amplitude = x[offset]
        rsb_beta = x[offset + 1]
    else:
        rsb_amplitude = rsb_beta = 0.0
    background_408 = float(np.asarray(rsb_temperature(rsb_amplitude, rsb_beta, np.array([408.0])))[0])
    levels = np.asarray(zero_levels(zero))
    template = np.asarray(bundle["template_k"], dtype=float)
    prediction = []
    for index, frequency in enumerate(np.asarray(bundle["frequency_mhz"], dtype=float)):
        response = np.asarray(bundle[f"response_{index}"], dtype=float)
        background = float(np.asarray(rsb_temperature(rsb_amplitude, rsb_beta, np.array([frequency])))[0])
        sky = (template + monopole - background_408) * amplitude * (frequency / 408.0) ** beta
        sky = sky + float(bundle["cmb_k"][index]) + background
        prediction.append(
            response @ sky
            - np.asarray(bundle[f"offset_response_{index}"], dtype=float) * levels[index]
        )
    labels = np.asarray(external["survey"])
    mask = np.isin(labels, list(surveys))
    # The calibration vector is indexed by the SELECTED survey, so the stored
    # survey code has to be remapped, exactly as model_inputs does.
    code = np.asarray(
        [list(surveys).index(str(label)) for label in labels[mask]], dtype=int
    )
    frequency = np.asarray(external["frequency_mhz"], dtype=float)[mask]
    tau = np.asarray(external["tau_rj_k"], dtype=float)[mask]
    external_mean = (
        np.asarray(cmb_rj_temperature(frequency))
        + np.asarray(rsb_temperature(rsb_amplitude, rsb_beta, frequency))
        + calibration[code] * tau
    )
    return np.concatenate(prediction), external_mean, mask


def map_start(bundle, external, surveys, basis, include_rsb, *, seed):
    functions = basis.shape[1]
    tris_data = np.concatenate(
        [
            np.asarray(bundle[f"whitened_data_{index}"], dtype=float)
            for index in range(2)
        ]
    )
    labels = np.asarray(external["survey"])
    mask = np.isin(labels, list(surveys))
    external_data = np.asarray(external["temperature_rj_k"], dtype=float)[mask]
    external_sigma = np.asarray(external["sigma_independent_rj_k"], dtype=float)[mask]
    start = np.concatenate(
        [
            np.full(functions, np.log(1.6)),
            np.full(functions, -2.75),
            np.zeros(2),
            np.zeros(1),
            np.zeros(len(surveys)),
            ([0.3, -2.6] if include_rsb else []),
        ]
    )
    prior_sd = np.concatenate(
        [
            np.full(functions, 0.5),
            np.full(functions, 0.5),
            np.ones(2),
            np.full(1, 3.0),
            np.ones(len(surveys)),
            ([1.443, 0.722] if include_rsb else []),
        ]
    )
    centre = np.concatenate(
        [
            np.full(functions, np.log(1.6)),
            np.full(functions, -2.75),
            np.zeros(2),
            np.zeros(1),
            np.zeros(len(surveys)),
            ([0.3, -2.6] if include_rsb else []),
        ]
    )

    def residual(x):
        prediction, external_mean, _ = _smooth_whitened_mean(
            bundle, external, surveys, basis, x, include_rsb
        )
        return np.concatenate(
            [
                tris_data - prediction,
                (external_data - external_mean) / external_sigma,
                (x - centre) / prior_sd,
            ]
        )

    fit = least_squares(residual, start, max_nfev=4000, ftol=1e-12, xtol=1e-12)
    return fit.x, prior_sd, centre


def initialization(graph, bundle, external, surveys, basis, include_rsb, *, chains, seed):
    centre, prior_sd, _ = map_start(bundle, external, surveys, basis, include_rsb, seed=seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 313]))
    functions = basis.shape[1]
    starts = []
    for _chain in range(chains):
        point = centre + 0.1 * prior_sd * rng.normal(size=centre.size)
        values = {}
        values["log_amplitude_coefficients"] = point[:functions]
        values["beta_coefficients"] = point[functions : 2 * functions]
        values["zero_standard"] = point[2 * functions : 2 * functions + 2]
        values["haslam_monopole_K"] = point[2 * functions + 2]
        values["calibration_standard"] = point[
            2 * functions + 3 : 2 * functions + 3 + len(surveys)
        ]
        if include_rsb:
            offset = 2 * functions + 3 + len(surveys)
            # The perturbation can push the amplitude below its Uniform(0,5)
            # support; an out-of-support start is refused, so clip it in.
            values["rsb_amplitude"] = float(np.clip(point[offset], 0.01, 4.99))
            values["rsb_beta"] = float(np.clip(point[offset + 1], -3.99, -1.51))
        values["haslam_monopole_K"] = _repair_positive_template(
            bundle, values["haslam_monopole_K"], values.get("rsb_amplitude"),
            values.get("rsb_beta"), include_rsb,
        )
        starts.append(values)
    policy = InitializationPolicy(
        values=tuple(
            NamedArray(
                name,
                np.stack([start[name] for start in starts]),
                ("chain",) + tuple(f"axis{i}" for i in range(np.ndim(starts[0][name]))),
            )
            for name in graph.latents
        )
    )
    return policy, centre


def _repair_positive_template(bundle, monopole, rsb_amplitude, rsb_beta, include_rsb):
    """Move the monopole to the nearest value whose galactic template is positive.

    The positive-template support multiplies the TRIS density by zero, so a
    start that violates it is refused; the scipy MAP residual does not contain
    that constraint, so on a small RA subset it can land outside.  Repair the
    start rather than relying on the MAP to respect a constraint it cannot see.
    """
    template = np.asarray(bundle["template_k"], dtype=float)
    if include_rsb and rsb_amplitude is not None:
        background = float(
            np.asarray(rsb_temperature(rsb_amplitude, rsb_beta, np.array([408.0])))[0]
        )
    else:
        background = 0.0
    # The support is a hard -inf cliff at G408 = 0, so a start one mK above
    # it lets every early NUTS trajectory cross the cliff.  Start a fixed
    # fraction of the monopole prior SD inside the domain instead; the RA
    # block refits that produced 2786-4000 divergences all started a hair
    # above the boundary.
    margin = POSITIVE_TEMPLATE_START_MARGIN_K
    floor = background - float(template.min()) + margin
    return float(max(float(monopole), floor))


def residual_rows(bundle, draws, basis, include_rsb, *, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 731]))
    coefficients = np.asarray(draws["log_amplitude_coefficients"])
    beta = np.asarray(draws["beta_coefficients"])
    zero = np.asarray(draws["zero_standard"])
    monopole = np.asarray(draws["haslam_monopole_K"])
    count = coefficients.shape[0]
    background_408 = (
        np.asarray(draws["rsb_amplitude"]) * 0.408 ** np.asarray(draws["rsb_beta"])
        if include_rsb
        else np.zeros(count)
    )
    template = np.asarray(bundle["template_k"], dtype=float)
    rows = []
    for index, frequency in enumerate(np.asarray(bundle["frequency_mhz"], dtype=float)):
        operator = np.asarray(bundle[f"operator_{index}"], dtype=float)
        sigma = np.asarray(bundle[f"sigma_k_{index}"], dtype=float)
        observed = np.asarray(bundle[f"data_k_{index}"], dtype=float)
        amplitude = np.exp(coefficients @ basis.T)
        slope = beta @ basis.T
        level = zero * np.where(
            np.arange(2)[None, :] == 0, 0.066, np.where(zero[:, 1:2] < 0, 0.300, 0.430)
        )
        background = (
            np.asarray(draws["rsb_amplitude"]) * (frequency / 1000.0)
            ** np.asarray(draws["rsb_beta"])
            if include_rsb
            else np.zeros(count)
        )
        sky = (
            (template[None, :] + monopole[:, None] - background_408[:, None])
            * amplitude
            * (frequency / 408.0) ** slope
            + float(bundle["cmb_k"][index])
            + background[:, None]
        )
        mean = sky @ operator.T - level[:, index : index + 1]
        residual = observed - mean.mean(axis=0)
        d_obs = np.sum(((observed[None, :] - mean) / sigma[None, :]) ** 2, axis=1)
        replicated = mean + sigma[None, :] * rng.normal(size=mean.shape)
        d_rep = np.sum(((replicated - mean) / sigma[None, :]) ** 2, axis=1)
        rows.append(
            {
                "frequency_mhz": float(frequency),
                "residual_rms_k": float(np.sqrt(np.mean(residual**2))),
                "chi_square_per_observation": float(np.sum((residual / sigma) ** 2) / residual.size),
                "chi_square_at_posterior_mean": float(np.sum((residual / sigma) ** 2)),
                "ppc_tail_probability": float(np.mean(d_rep >= d_obs)),
                "draws": int(count),
            }
        )
    return rows


def run_smooth(bundle, external, surveys, include_rsb, output, *, basis, seed, draws, warmup, chains):
    if not jax.config.jax_enable_x64:
        raise ValueError("the smooth TRIS case requires jax.enable_x64(True)")
    inputs = smooth_inputs(bundle, external, surveys, basis, include_rsb)
    graph = trace(model, *inputs)
    policy, centre = initialization(
        graph, bundle, external, surveys, basis, include_rsb, chains=chains, seed=seed
    )
    # 0.99 rather than 0.95: the positive-template support is a hard boundary
    # and the failing RA refits diverged on essentially every draw, which a
    # smaller adapted step size is the direct remedy for.
    kernel_options = (("dense_mass", True), ("target_accept_prob", 0.99))
    task = PosteriorTask(
        meta=new_task_meta(
            label=f"smooth TRIS + Haslam {'RSB' if include_rsb else 'no RSB'}"
        ),
        budget=ComputeBudget(draws=draws, warmup=warmup, chains=chains),
        chain_method="sequential",
        initialization=policy,
        nuts_on_collapse=False,
        stopping=StoppingPolicy(rhat_max=1.01, ess_min=400),
        backend_options=(("progress_bar", False), ("nuts_options", kernel_options)),
    )
    plan_key, sample_key = jax.random.split(jax.random.key(seed))
    plan = compile_task(
        graph, task, model_ref=model_ref_from_callable(model, identifier=model.__name__), key=plan_key
    )
    if isinstance(plan, Refusal):
        raise RuntimeError(f"smooth case refused: {plan}")  # noqa: TRY004
    posterior = require_result(execute_task(plan, key=sample_key), PosteriorResult)
    if not isinstance(posterior.representation, DrawsPosterior):
        raise TypeError("smooth case requires unweighted draws")
    posterior_samples = samples(posterior)
    diagnostics = diagnostic_checks(posterior)
    report = {
        "case": "tris_haslam_smooth",
        "variant": "rsb" if include_rsb else "no_rsb",
        "included_surveys": list(surveys),
        "functions": int(basis.shape[1]),
        "seed": seed,
        "warmup": warmup,
        "chain_shape": posterior.representation.chain_shape,
        "passed": diagnostics["passed"],
        "checks": {"chain_diagnostics": diagnostics},
        "frequencies": residual_rows(bundle, posterior_samples, basis, include_rsb, seed=seed),
        "map_center": centre.tolist(),
        "priors": {
            "log_amplitude_coefficients": "independent Normal(log 1.6, 0.5)",
            "beta_coefficients": "independent Normal(-2.75, 0.5)",
            "zero_standard": "two independent Normal(0,1)",
            "haslam_monopole_K": "Normal(0,3 K)",
            "calibration_standard": "Normal(0,1) per survey",
            "rsb_amplitude": "Uniform(0,5) K at 1 GHz" if include_rsb else None,
            "rsb_beta": "Uniform(-4,-1.5)" if include_rsb else None,
        },
        "initialization": (
            "scipy MAP plus 10 percent prior-SD dispersion, monopole repaired to "
            "POSITIVE_TEMPLATE_START_MARGIN_K inside the positive-template domain"
        ),
        "execution": {
            "termination": posterior.run.termination.reason.value,
            "sampling": dict(posterior.run.sampling_details),
            "wall_clock_seconds": posterior.run.timing.wall_clock_seconds,
            "nuts_options": dict(kernel_options),
        },
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    np.savez_compressed(output / "posterior.npz", **posterior_samples)
    dump_artifact(posterior, output / "posterior.artifact.json")
    return report


def heldout_deltas(rows):
    """M1-minus-M0 predictive deltas, WITH a convergence gate.

    A refit whose chain diagnostics failed is not a posterior, so its
    predictive score cannot enter a model comparison.  The delta is emitted
    as None with valid=False rather than as a number a reader could mistake
    for evidence; the rows keep the raw score so the failure stays visible.
    """
    deltas = []
    for fold, train in enumerate(SURVEYS):
        selected = {row["variant"]: row for row in rows if row["train"] == train}
        missing = [variant for variant in ("no_rsb", "rsb") if variant not in selected]
        invalid = [variant for variant, row in selected.items() if not row["passed"]]
        valid = not missing and not invalid
        entry = {
            "train": train,
            "heldout": SURVEYS[1 - fold],
            "valid": bool(valid),
            "invalid_refits": invalid,
            "missing_refits": missing,
        }
        if valid:
            entry["delta_M1_minus_M0"] = (
                selected["rsb"]["log_predictive_density"]
                - selected["no_rsb"]["log_predictive_density"]
            )
            entry["delta_mcse"] = float(
                np.hypot(selected["rsb"]["mcse"], selected["no_rsb"]["mcse"])
            )
            entry["status"] = "available"
        else:
            entry["delta_M1_minus_M0"] = None
            entry["delta_mcse"] = None
            entry["status"] = "invalid_nonconverged_refit" if invalid else "missing_refit"
        deltas.append(entry)
    return deltas


def heldout_phase(
    bundle, external, basis, output, *, seed, draws, warmup, chains, variants,
    skip_existing=False,
):
    """Train on one survey and score the other, for both models."""
    rows = []
    for fold, train in enumerate(SURVEYS):
        held = SURVEYS[1 - fold]
        labels = np.asarray(external["survey"])
        mask = labels == held
        heldout = {
            name: np.asarray(external[name])[mask]
            for name in (
                "frequency_mhz",
                "temperature_rj_k",
                "sigma_independent_rj_k",
                "tau_rj_k",
                "survey_code",
            )
        }
        for index, variant in enumerate(variants):
            directory = output / "heldout" / f"smooth_{variant}_train_{train}"
            include_rsb = variant == "rsb"
            if skip_existing and (directory / "result.json").is_file():
                report = json.loads((directory / "result.json").read_text())
            else:
                report = run_smooth(
                    bundle,
                    external,
                    (train,),
                    include_rsb,
                    directory,
                    basis=basis,
                    seed=seed + 10 + 2 * fold + index,
                    draws=draws,
                    warmup=warmup,
                    chains=chains,
                )
            with np.load(directory / "posterior.npz", allow_pickle=False) as archive:
                saved = {name: archive[name] for name in archive.files}
            score = heldout_log_predictive_density(
                saved, heldout, include_rsb=include_rsb, chain_shape=report["chain_shape"]
            )
            score.pop("score_draws", None)
            divergences = (
                report.get("checks", {})
                .get("chain_diagnostics", {})
                .get("divergences")
            )
            rows.append(
                {
                    "train": train,
                    "heldout": held,
                    "variant": variant,
                    "log_predictive_density": score["log_predictive_density"],
                    "mcse": score["mcse"],
                    "passed": bool(report["passed"]),
                    "divergences": divergences,
                }
            )
    return rows, heldout_deltas(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-b", type=Path, required=True)
    parser.add_argument("--external-input", type=Path, default=Path("runs/tris-rsb-input"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--functions", type=int, default=FUNCTIONS)
    parser.add_argument("--variants", nargs="+", default=["no_rsb", "rsb"])
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()
    with np.load(args.batch_b / "audited-input" / "maps.npz", allow_pickle=False) as archive:
        bundle = {name: archive[name] for name in archive.files}
    with np.load(args.external_input / "external.npz", allow_pickle=False) as archive:
        external = {name: archive[name] for name in archive.files}
    basis = basis_matrix(bundle, args.functions)
    summary = {}
    with jax.enable_x64(True):
        for index, variant in enumerate(args.variants):
            directory = args.output / f"tris_haslam_smooth_{variant}"
            if args.skip_existing and (directory / "result.json").is_file():
                report = json.loads((directory / "result.json").read_text())
            else:
                report = run_smooth(
                    bundle,
                    external,
                    SURVEYS,
                    variant == "rsb",
                    directory,
                    basis=basis,
                    seed=args.seed + index,
                    draws=args.draws,
                    warmup=args.warmup,
                    chains=args.chains,
                )
            summary[variant] = {
                "passed": report["passed"],
                "frequencies": report["frequencies"],
                "wall_clock_seconds": report["execution"]["wall_clock_seconds"],
            }
    if args.heldout:
        with jax.enable_x64(True):
            rows, deltas = heldout_phase(
                bundle,
                external,
                basis,
                args.output,
                seed=args.seed,
                draws=args.draws,
                warmup=args.warmup,
                chains=args.chains,
                variants=args.variants,
                skip_existing=args.skip_existing,
            )
        summary["heldout"] = {"rows": rows, "deltas": deltas}
    (args.output / "smooth_summary.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n"
    )
    print(json.dumps({"output": str(args.output), "elapsed_s": round(time.time() - started, 2)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
