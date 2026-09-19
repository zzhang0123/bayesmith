"""Audit numerical support, predictive Monte Carlo error and sky/calibration modes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_triangular

from examples.inference.tris_continuous import harmonic_basis
from examples.inference.tris_continuous_validation import (
    error_summary,
    predict_parameters,
)


def mean_information(fit, samples, archive):
    from examples.TRIS.continuous_background import (
        continuous_prediction,
        error_covariance,
        prepare_operator,
        ra_kernel,
    )
    from examples.TRIS.joint_sky_beam import zero_levels

    operator = prepare_operator(archive)
    na, nb = (fit["l_amplitude"] + 1) ** 2, (fit["l_beta"] + 1) ** 2
    ba = jnp.asarray(
        harmonic_basis(operator["theta"], operator["phi"], fit["l_amplitude"])
    )
    bb = jnp.asarray(harmonic_basis(operator["theta"], operator["phi"], fit["l_beta"]))
    mean = {k: v.mean((0, 1)) for k, v in samples.items()}
    center = np.r_[
        mean["log_amplitude_coeff"],
        mean["beta_coeff"],
        mean["monopole_k"],
        mean["zero_standard"],
    ]
    train = np.asarray(fit["train"])

    def predict(x):
        p = continuous_prediction(
            operator, ba, bb, x[:na], x[na : na + nb], x[-3], zero_levels(x[-2:])
        )
        return p[train].T

    jac = np.asarray(jax.jacfwd(predict)(jnp.asarray(center)))
    cov = np.asarray(
        error_covariance(
            archive["sigma"],
            mean.get("noise_floor_k", np.zeros(2)),
            mean.get("correlated_sd_k", np.zeros(2)),
            ra_kernel(archive["ra_deg"]),
        )
    )
    whitened = np.concatenate(
        [
            solve_triangular(
                np.linalg.cholesky(cov[f][np.ix_(train, train)]), jac[f], lower=True
            )
            for f in range(2)
        ]
    )
    u, s, _ = np.linalg.svd(whitened[:, -3:], full_matrices=False)
    rank = int(np.sum(s > np.finfo(float).eps * max(whitened.shape) * s.max()))
    residual = whitened[:, : na + nb] - u[:, :rank] @ (
        u[:, :rank].T @ whitened[:, : na + nb]
    )
    scales = np.r_[fit["prior_sd_a"], fit["prior_sd_b"]]
    singular = np.linalg.svd(residual * scales, compute_uv=False)
    coefficients = np.concatenate(
        [samples["log_amplitude_coeff"], samples["beta_coeff"]], axis=-1
    ).reshape(-1, na + nb)
    posterior = np.linalg.eigvalsh(np.cov((coefficients / scales).T))
    return {
        "mean_likelihood_singular_values_after_calibration_projection": singular.tolist(),
        "profiled_calibration_rank": rank,
        "joint_coefficient_posterior_covariance_in_prior_units_eigenvalues": posterior.tolist(),
        "interpretation": "local mean-only likelihood, noise hyperparameters fixed at posterior means; monopole and two zero responses projected out without calibration priors; not global identifiability proof",
    }


def main():
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--response1024", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from examples.TRIS.continuous_background import predictive_report, ra_kernel

    registry = json.loads(args.registry.read_text())
    higher = dict(np.load(args.response1024, allow_pickle=False))
    paths = {**registry["full"], **registry.get("heldout", {})}
    results = {}
    for name, path in paths.items():
        path = Path(path)
        record = json.loads((path / "result.json").read_text())
        if not record["sampling_pass"]:
            raise ValueError(f"unaccepted fit: {path}")
        config = json.loads((path / "configuration.json").read_text())
        archive = dict(np.load(config["response"], allow_pickle=False))
        pred = dict(np.load(path / "predictions.npz", allow_pickle=False))
        samples = dict(np.load(path / "samples.npz", allow_pickle=False))
        checks = {}
        selected = {
            k: v.reshape(-1, *v.shape[2:])
            for k, v in samples.items()
            if k in ("log_amplitude_coeff", "beta_coeff", "monopole_k", "zero_standard")
        }
        indices = np.linspace(0, len(selected["monopole_k"]) - 1, 32, dtype=int)
        positions = [{k: v[i] for k, v in selected.items()} for i in indices]
        low = predict_parameters(archive, positions, 12)
        high = predict_parameters(archive, positions, 20)
        finer = predict_parameters(higher, positions, 20)
        checks["spectral"] = error_summary(low - high, archive["sigma"])
        checks["angular"] = error_summary(high - finer, archive["sigma"])
        checks["pass"] = (
            checks["spectral"]["max_abs_archive_sigma"] < 0.1
            and checks["angular"]["max_abs_archive_sigma"] < 0.3
        )
        report, _ = predictive_report(
            pred["data"],
            pred["sigma"],
            pred["pred"],
            pred["noise_floor"],
            pred["correlated_sd"],
            ra_kernel(archive["ra_deg"]),
            np.asarray(config["train"]),
            np.asarray(config["held"]),
            config["seed"] + 1,
        )
        np.testing.assert_allclose(
            report["pointwise_log_predictive_sum"],
            record["predictive"]["pointwise_log_predictive_sum"],
            atol=1e-9,
            rtol=0,
        )
        results[str(path.resolve())] = {
            "name": name,
            "numerical": checks,
            "predictive": report,
            "information": mean_information(config, samples, archive),
            "result_sha256": hashlib.sha256(
                (path / "result.json").read_bytes()
            ).hexdigest(),
        }
        print(
            json.dumps(
                {
                    "finished": name,
                    "numerical_pass": checks["pass"],
                    "mcse": report["monte_carlo_error"],
                }
            ),
            flush=True,
        )
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
