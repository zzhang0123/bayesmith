"""Deterministic morphology audit before expanding the physical sky model.

Fits to log survey brightness are compression diagnostics, not priors or
component-separated skies. Optical extrapolation outside its mask is untrusted.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.inference.tris_continuous import harmonic_basis
from examples.inference.tris_halpha import flux_to_rj_integral


def source_and_coverage(data, residual):
    """A 2016 spectrum as a scale check, never an applied map correction."""
    import healpy as hp

    lon, lat = 111.7376, -2.1345  # SIMBAD Galactic coordinates, J2000.
    nside = int(data["nside"])
    pixel = int(hp.ang2pix(nside, lon, lat, lonlat=True))
    x = np.log10(data["frequency_mhz"] / 1000)
    # Perley & Butler arXiv1609.05940v1, Table5, CasA 2016 spectrum.
    flux = 10 ** np.polynomial.polynomial.polyval(x, [3.3584, -0.7518, -0.035, -0.071])
    q = flux_to_rj_integral(data["frequency_mhz"], flux)
    vectors = np.array(hp.pix2vec(512, np.arange(hp.nside2npix(512)))).T
    angle = np.arccos(np.clip(vectors @ hp.ang2vec(lon, lat, lonlat=True), -1, 1))
    width = np.deg2rad(56 / 60) / np.sqrt(8 * np.log(2))
    psf = np.exp(-0.5 * (angle / width) ** 2)
    psf /= psf.sum() * hp.nside2pixarea(512)
    coarse = hp.ud_grade(psf, nside) * q[0]
    observed = np.zeros(len(data["haslam"]), dtype=bool)
    observed[data["ha_index"]] = True
    missing_weight = data["beam"] @ (~observed[data["parent"]])
    return {
        "casa_galactic_deg": [lon, lat],
        "casa_pixel": pixel,
        "classification": "Galactic SNR; no subtraction from isotropic extragalactic RSB budget",
        "spectrum_epoch": 2016,
        "source_epoch_mapping_complete": False,
        "source_urls": [
            "https://simbad.cds.unistra.fr/simbad/sim-id?Ident=Cassiopeia+A",
            "https://arxiv.org/pdf/1609.05940v1",
        ],
        "frequency_mhz": data["frequency_mhz"].tolist(),
        "flux2016_jy": flux.tolist(),
        "q2016_k_sr": q.tolist(),
        "source_pixel_contribution_408_2016_k": float(coarse[pixel]),
        "source_pixel_d13_residual_408_k": float(residual[pixel]),
        "source_pixel_q_fraction": float(
            (residual[pixel] / data["haslam_sigma"][pixel]) ** 2
            / np.sum((residual / data["haslam_sigma"]) ** 2)
        ),
        "largest_residual_cell_contains_casa": bool(
            pixel == np.argmax(np.abs(residual))
        ),
        "not_an_estimated_source_correction": True,
        "beam_fraction_without_optical_quantiles": np.quantile(
            missing_weight[:120], [0, 0.5, 1]
        ).tolist(),
        "beam_rows_more_than_half_without_optical": int(
            np.sum(missing_weight[:120] > 0.5)
        ),
    }


def compress_log_map(value, sigma, basis, index, evaluate_index):
    # Equal weight in log brightness measures shape representation. Report
    # residuals back in native units; it is not a Gaussian likelihood fit.
    design = basis[index]
    coeff, _, rank, singular = np.linalg.lstsq(design, np.log(value), rcond=1e-10)
    prediction = np.exp(design @ coeff)
    residual = value - prediction
    full_log = basis[evaluate_index] @ coeff
    return {
        "coefficients": coeff.tolist(),
        "rank": int(rank),
        "columns": design.shape[1],
        "condition_number": float(singular[0] / singular[-1]),
        "rms": float(np.sqrt(np.mean(residual**2))),
        "q_per_observation": float(np.mean((residual / sigma) ** 2)),
        "max_abs": float(np.max(np.abs(residual))),
        "unobserved_log_prediction_range": [
            float(full_log.min()),
            float(full_log.max()),
        ],
    }


def audit(input_path, optimized_path):
    import healpy as hp
    from scipy.optimize import least_squares

    with np.load(input_path, allow_pickle=False) as f:
        data = {k: np.asarray(f[k]) for k in f.files}
    with np.load(optimized_path, allow_pickle=False) as f:
        optimized = {k: np.asarray(f[k]) for k in f.files}
    nside = int(data["nside"])
    theta, phi = hp.pix2ang(nside, np.arange(len(data["haslam"])))
    latitude = 90 - np.rad2deg(theta)
    residual = data["haslam"] - optimized["haslam"]
    optical_residual = data["ha_data"] - optimized["halpha"]
    q = (residual / data["haslam_sigma"]) ** 2
    regions = {}
    for low, high in ((0, 10), (10, 30), (30, 91)):
        selected = (np.abs(latitude) >= low) & (np.abs(latitude) < high)
        optical = selected[data["ha_index"]]
        regions[f"abs_b_{low}_{high}"] = {
            "haslam_n": int(selected.sum()),
            "haslam_rms_k": float(np.sqrt(np.mean(residual[selected] ** 2))),
            "haslam_q_fraction": float(q[selected].sum() / q.sum()),
            "halpha_n": int(optical.sum()),
            "halpha_rms_r": float(np.sqrt(np.mean(optical_residual[optical] ** 2)))
            if optical.any()
            else None,
        }
    unobserved = np.ones(len(latitude), dtype=bool)
    unobserved[data["ha_index"]] = False
    representation = {"haslam_total": {}, "halpha_observed": {}}
    for lmax in (2, 4, 6, 8, 12, 16):
        basis = harmonic_basis(theta, phi, lmax)
        representation["haslam_total"][str(lmax)] = compress_log_map(
            data["haslam"],
            data["haslam_sigma"],
            basis,
            np.arange(len(latitude)),
            np.arange(len(latitude)),
        )
        representation["halpha_observed"][str(lmax)] = compress_log_map(
            data["ha_data"],
            data["ha_total_sigma"],
            basis,
            data["ha_index"],
            np.flatnonzero(unobserved),
        )
    # A second diagnostic optimizes the Haslam residual in Kelvin directly.
    # This avoids mistaking equal log-weight compression for a lower bound on
    # achievable map RMS. It still fits TOTAL survey brightness, not A_syn.
    native_unit_fits = {}
    for lmax in (4, 8, 12, 16, 20, 23):
        basis = harmonic_basis(theta, phi, lmax)
        value, sigma = data["haslam"], data["haslam_sigma"]
        weight = value / sigma
        initial = np.linalg.lstsq(
            basis * weight[:, None], np.log(value) * weight, rcond=1e-10
        )[0]

        def residual_fn(coeff, basis=basis, value=value, sigma=sigma):
            return (np.exp(basis @ coeff) - value) / sigma

        def jacobian(coeff, basis=basis, sigma=sigma):
            return np.exp(basis @ coeff)[:, None] * basis / sigma[:, None]

        result = least_squares(
            residual_fn,
            initial,
            jac=jacobian,
            method="trf",
            max_nfev=40,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-8,
        )
        native_unit_fits[str(lmax)] = {
            "success": bool(result.success),
            "evaluations": int(result.nfev),
            "rms_k": float(np.sqrt(np.mean((result.fun * sigma) ** 2))),
            "q_per_observation": float(np.mean(result.fun**2)),
            "optimality": float(result.optimality),
            "min_prediction_k": float(np.min(np.exp(basis @ result.x))),
        }
    order = np.argsort(-q)[:15]
    record = {
        "schema": "tris.spatial_residual_audit.v1",
        "interpretation": "deterministic optimized-point and log-brightness compression diagnostics; not a posterior or data-derived prior",
        "regions": regions,
        "source_and_coverage": source_and_coverage(data, residual),
        "representation": representation,
        "haslam_native_unit_fits": native_unit_fits,
        "top15_haslam_residuals": [
            {
                "pixel": int(i),
                "l_deg": float(np.rad2deg(phi[i])),
                "b_deg": float(latitude[i]),
                "observed_k": float(data["haslam"][i]),
                "residual_k": float(residual[i]),
                "optical_observed": bool(not unobserved[i]),
            }
            for i in order
        ],
        "optical_unobserved_pixels": int(unobserved.sum()),
        "haslam_q_fraction_without_optical": float(q[unobserved].sum() / q.sum()),
        "compression_policy": "equal log-brightness weights; pseudoinverse rcond1e-10; evaluate on measurement pixel centres, not the physical beam/PSF model",
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (input_path, optimized_path)
        },
    }
    return record, {
        "longitude": np.rad2deg(phi),
        "latitude": latitude,
        "haslam_residual": residual,
        "optical_residual": optical_residual,
        "optical_index": data["ha_index"],
    }


def plot(record, maps, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    artist = axes[0, 0].scatter(
        (maps["longitude"] + 180) % 360 - 180,
        maps["latitude"],
        c=maps["haslam_residual"],
        s=13,
        cmap="RdBu_r",
        vmin=-20,
        vmax=20,
    )
    axes[0, 0].annotate(
        "Cas A cell",
        xy=(111.7376, -2.1345),
        xytext=(80, 40),
        arrowprops={"arrowstyle": "->"},
        fontsize=8,
    )
    axes[0, 0].set(
        xlabel="Galactic longitude [deg]",
        ylabel="Galactic latitude [deg]",
        title="D13 optimized-point Haslam residual",
    )
    fig.colorbar(
        artist, ax=axes[0, 0], label="Observed minus predicted [K]", extend="both"
    )
    keys = list(record["regions"])
    axes[0, 1].bar(
        ["|b|<10", "10<=|b|<30", "|b|>=30"],
        [record["regions"][k]["haslam_q_fraction"] for k in keys],
        color="#bd7938",
    )
    axes[0, 1].set(
        ylabel="Fraction of total Haslam residual quadratic",
        title="Where the mismatch is",
    )
    for key, label in (
        ("haslam_total", "Haslam total"),
        ("halpha_observed", "H-alpha observed"),
    ):
        rows = record["representation"][key]
        axes[1, 0].plot(
            [int(k) for k in rows],
            [r["q_per_observation"] for r in rows.values()],
            "o-",
            label=label,
        )
    axes[1, 0].set(
        xlabel="Log-brightness harmonic lmax",
        ylabel="Residual quadratic / N (working sigma)",
        yscale="log",
        title="Shape compression only; not a sky-component fit",
    )
    axes[1, 0].legend(fontsize=8)
    rows = record["representation"]["halpha_observed"]
    axes[1, 1].plot(
        [int(k) for k in rows],
        [r["condition_number"] for r in rows.values()],
        "o-",
        color="#b44a3e",
    )
    axes[1, 1].set(
        xlabel="Optical log-brightness harmonic lmax",
        ylabel="Masked basis condition number",
        yscale="log",
        title="Fine optical modes lose spatial support",
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "optimized", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    record, maps = audit(args.input, args.optimized)
    record["analysis_source_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    plot(record, maps, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
