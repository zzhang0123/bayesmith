"""Render accepted continuous-field fits, with uncertainty and holdout evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.inference.tris_continuous import harmonic_basis


def read_fit(path):
    path = Path(path)
    record = json.loads((path / "result.json").read_text())
    if not record["sampling_pass"]:
        raise ValueError(f"unaccepted sampling diagnostics: {path}")
    return {
        "path": str(path),
        "result": record,
        "configuration": json.loads((path / "configuration.json").read_text()),
        "samples": dict(np.load(path / "samples.npz", allow_pickle=False)),
        "prediction": dict(np.load(path / "predictions.npz", allow_pickle=False)),
    }


def field_summary(samples, theta, phi, l_amplitude, l_beta):
    ba = harmonic_basis(theta, phi, l_amplitude)
    bb = harmonic_basis(theta, phi, l_beta)
    a = samples["log_amplitude_coeff"].reshape(-1, (l_amplitude + 1) ** 2)
    b = samples["beta_coeff"].reshape(-1, (l_beta + 1) ** 2)
    keep = np.linspace(0, len(a) - 1, min(len(a), 512), dtype=int)
    amplitude = np.exp(ba @ a[keep].T)
    covariance = np.atleast_2d(np.cov(b.T))
    return {
        "amplitude_mean": amplitude.mean(1),
        "amplitude_sd": amplitude.std(1, ddof=1),
        "beta_mean": bb @ b.mean(0),
        "beta_sd": np.sqrt(np.maximum(np.einsum("pi,ij,pj->p", bb, covariance, bb), 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    full = {name: read_fit(path) for name, path in registry["full"].items()}
    held = {name: read_fit(path) for name, path in registry.get("heldout", {}).items()}
    diagnostics = json.loads(args.diagnostics.read_text()) if args.diagnostics else {}
    for fit in [*full.values(), *held.values()]:
        extra = diagnostics.get(str(Path(fit["path"]).resolve()))
        if extra:
            if not extra["numerical"]["pass"]:
                raise ValueError(f"numerical check failed: {fit['path']}")
            fit["result"]["predictive"] = extra["predictive"]
            fit["supplemental"] = extra
    rows = []
    for name, fit in full.items():
        r = fit["result"]
        prior = r["coefficient_sd_over_prior"]
        order = r["parameter_order"]
        ia = [i for i, k in enumerate(order) if k.startswith("log_amplitude_coeff")]
        ib = [i for i, k in enumerate(order) if k.startswith("beta_coeff")]
        correlation = fit["prediction"]["correlation"]
        cross = correlation[np.ix_(ia, ib)]
        calibration = [
            i
            for i, k in enumerate(order)
            if k.startswith(("monopole_k", "zero_standard"))
        ]
        calibration_pairs = sorted(
            [
                {
                    "sky": order[i],
                    "calibration": order[j],
                    "correlation": float(correlation[i, j]),
                }
                for i in ia + ib
                for j in calibration
            ],
            key=lambda item: abs(item["correlation"]),
            reverse=True,
        )[:5]
        rows.append(
            {
                "name": name,
                "path": fit["path"],
                "predictive": r["predictive"],
                "max_abs_amplitude_beta_correlation": float(np.max(np.abs(cross))),
                "coefficient_sd_over_prior": prior,
                "strongest_sky_calibration_correlations": calibration_pairs,
                "posterior": r["posterior"],
                "supplemental": fit.get("supplemental"),
                "max_rhat": max(d["r_hat_max"] for d in r["diagnostics"].values()),
                "min_bulk_ess": min(
                    d["bulk_ess_min"] for d in r["diagnostics"].values()
                ),
            }
        )
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, layout="constrained")
    for name, fit in full.items():
        if fit["configuration"]["noise"] != "floor":
            continue
        prediction = fit["prediction"]
        with np.load(fit["configuration"]["response"], allow_pickle=False) as response:
            ra = response["ra_deg"]
        label = f"L_A={fit['configuration']['l_amplitude']}"
        for f in range(2):
            axes[0, f].plot(ra, prediction["sky_mean"][:, f], label=label, lw=1.4)
            axes[1, f].plot(
                ra,
                prediction["data"][:, f] - prediction["sky_mean"][:, f],
                label=label,
                lw=1.2,
            )
    example = next(iter(full.values()))["prediction"]
    for f, nu in enumerate((600.5, 817.8)):
        axes[0, f].scatter(ra, example["data"][:, f], s=8, c="black", label="TRIS")
        axes[0, f].set_title(f"{nu} MHz")
        axes[0, f].set_ylabel("Antenna temperature [K]")
        axes[0, f].legend(fontsize=8)
        axes[1, f].set_ylabel("Data - sky prediction [K]")
        axes[1, f].set_xlabel("RA [deg]")
        axes[1, f].axhline(0, color="black", lw=0.5)
    fig.suptitle(
        "Continuous beta (L=2), fixed RSB and nominal beam; inferred white floor"
    )
    fig.savefig(args.output / "profiles.png", dpi=170)
    plt.close(fig)
    longitude = np.linspace(-np.pi, np.pi, 181)
    latitude = np.linspace(-np.pi / 2, np.pi / 2, 91)
    lon, lat = np.meshgrid(longitude, latitude)
    theta, phi = np.pi / 2 - lat.ravel(), lon.ravel()
    # Same C->G rotation and zenith latitude as the response builder.
    import healpy as hp

    with np.load(
        next(iter(full.values()))["configuration"]["response"], allow_pickle=False
    ) as response:
        ring_ra = response["ra_deg"]
    ring_theta, ring_phi = hp.Rotator(coord=["C", "G"])(
        np.full(len(ring_ra), np.deg2rad(90 - (42 + 26 / 60))), np.deg2rad(ring_ra)
    )
    ring_lon = (ring_phi + np.pi) % (2 * np.pi) - np.pi
    ring_lat = np.pi / 2 - ring_theta
    field_products = {}
    map_fits = {
        k: v for k, v in full.items() if v["configuration"]["noise"] == "correlated"
    }
    if not map_fits:
        map_fits = full
    map_noise = ", ".join(
        sorted({fit["configuration"]["noise"] for fit in map_fits.values()})
    )
    products_by_name = {
        name: field_summary(
            fit["samples"],
            theta,
            phi,
            fit["configuration"]["l_amplitude"],
            fit["configuration"]["l_beta"],
        )
        for name, fit in map_fits.items()
    }
    specifications = [
        ("amplitude_mean", "Amplitude correction mean", "viridis"),
        ("amplitude_sd", "Amplitude posterior SD", "magma"),
        ("beta_mean", "Beta mean", "coolwarm"),
        ("beta_sd", "Beta posterior SD", "magma"),
    ]
    bounds = {}
    for key, _, _ in specifications:
        values = np.concatenate([v[key] for v in products_by_name.values()])
        bounds[key] = (
            0.0 if key.endswith("sd") else float(values.min()),
            float(values.max()),
        )
    fig = plt.figure(figsize=(16, 3.2 * len(map_fits)), layout="constrained")
    for row, (name, fit) in enumerate(map_fits.items()):
        config = fit["configuration"]
        products = products_by_name[name]
        for key, value in products.items():
            field_products[f"{name}_{key}"] = value.reshape(lon.shape)
        for col, (key, title, cmap) in enumerate(specifications):
            ax = fig.add_subplot(
                len(map_fits), 4, row * 4 + col + 1, projection="mollweide"
            )
            im = ax.pcolormesh(
                lon,
                lat,
                products[key].reshape(lon.shape),
                shading="auto",
                cmap=cmap,
                vmin=bounds[key][0],
                vmax=bounds[key][1],
            )
            ax.grid(alpha=0.2)
            ax.scatter(
                ring_lon, ring_lat, s=3, c="black", edgecolors="white", linewidths=0.2
            )
            ax.tick_params(labelsize=6)
            ax.set_title(f"L_A={config['l_amplitude']}: {title}", fontsize=10)
            fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.8)
    fig.suptitle(
        "Conditional full-sky extrapolations with posterior uncertainty\n"
        f"Galactic coordinates; fixed RSB / nominal beam; error model: {map_noise}\n"
        "Dots: nominal TRIS boresights; the broad beam extends beyond this line",
        fontsize=12,
    )
    fig.savefig(args.output / "fields.png", dpi=160)
    plt.close(fig)
    np.savez_compressed(
        args.output / "field_maps.npz",
        longitude=longitude,
        latitude=latitude,
        boresight_longitude=ring_lon,
        boresight_latitude=ring_lat,
        **field_products,
    )
    fold_rows = []
    for name, fit in held.items():
        fold_rows.append(
            {
                "name": name,
                "path": fit["path"],
                "fold": fit["configuration"]["fold"],
                "l_amplitude": fit["configuration"]["l_amplitude"],
                "noise_posterior": {
                    k: fit["result"]["posterior"][k]
                    for k in ("noise_floor_k", "correlated_sd_k")
                    if k in fit["result"]["posterior"]
                },
                **fit["result"]["predictive"],
            }
        )
    if held:
        fig, axes = plt.subplots(4, 2, figsize=(12, 10), layout="constrained")
        for fit in held.values():
            config, prediction = fit["configuration"], fit["prediction"]
            ell, fold = config["l_amplitude"], config["fold"]
            if ell not in (0, 2):
                continue
            ids = np.asarray(config["held"])
            x = ring_ra[ids]
            color = "C0" if ell == 0 else "C1"
            for f in range(2):
                ax = axes[fold, f]
                ax.plot(
                    x,
                    prediction["sky_mean"][ids, f],
                    color=color,
                    ls="--",
                    lw=1,
                    label=f"A{ell}: sky",
                )
                ax.plot(
                    x,
                    prediction["predictive_center"][:, f],
                    color=color,
                    lw=1.2,
                    label=f"A{ell}: conditional mean",
                )
                ax.fill_between(
                    x,
                    prediction["lower"][:, f],
                    prediction["upper"][:, f],
                    color=color,
                    alpha=0.13,
                    label=f"A{ell}: 95% predictive",
                )
                if ell == 0:
                    ax.scatter(
                        x,
                        prediction["data"][ids, f],
                        color="black",
                        s=9,
                        label="Held TRIS",
                    )
                ax.set_title(f"Fold {fold}: {(600.5, 817.8)[f]} MHz", fontsize=10)
                ax.set_ylabel("Temperature [K]")
                if fold == 3:
                    ax.set_xlabel("RA [deg]")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
        fig.suptitle(
            "Buffered holdout: sky prediction and conditional residual process are distinct"
        )
        fig.savefig(args.output / "holdout-profiles.png", dpi=170)
        plt.close(fig)
    aggregates = []
    for ell in sorted({r["l_amplitude"] for r in fold_rows}):
        folds = sorted(
            [r for r in fold_rows if r["l_amplitude"] == ell], key=lambda r: r["fold"]
        )
        mc = [r.get("monte_carlo_error", {}).get("pointwise_sum_total") for r in folds]
        aggregates.append(
            {
                "l_amplitude": ell,
                "pointwise_ranking_reliable": all(
                    r.get("monte_carlo_error", {}).get(
                        "pointwise_ranking_reliable", False
                    )
                    for r in folds
                ),
                "crps_sum_k": sum(r["sample_scores"]["crps_sum_k"] for r in folds),
                "crps_sum_mcse_k": float(
                    np.linalg.norm(
                        [r["sample_scores"]["crps_sum_mcse_k"][0] for r in folds]
                    )
                ),
                "joint_energy_sum_k": sum(
                    r["sample_scores"]["joint_energy_k"] for r in folds
                ),
                "joint_energy_sum_mcse_k": float(
                    np.linalg.norm(
                        [r["sample_scores"]["joint_energy_mcse_k"][0] for r in folds]
                    )
                ),
                "fold_scores": [sum(r["pointwise_log_predictive_sum"]) for r in folds],
                "pointwise_total": sum(
                    sum(r["pointwise_log_predictive_sum"]) for r in folds
                ),
                "mcse_total": float(np.linalg.norm(mc))
                if all(x is not None for x in mc)
                else None,
            }
        )
    if aggregates:
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        for ell in (1, 2):
            model = [r for r in fold_rows if r["l_amplitude"] == ell]
            delta, errors = [], []
            for row in sorted(model, key=lambda r: r["fold"]):
                base = next(
                    r
                    for r in fold_rows
                    if r["l_amplitude"] == 0 and r["fold"] == row["fold"]
                )
                delta.append(
                    row["sample_scores"]["joint_energy_k"]
                    - base["sample_scores"]["joint_energy_k"]
                )
                errors.append(
                    np.hypot(
                        row["sample_scores"]["joint_energy_mcse_k"][0],
                        base["sample_scores"]["joint_energy_mcse_k"][0],
                    )
                )
            ax.errorbar(
                np.arange(len(delta)) + (ell - 1.5) * 0.08,
                delta,
                yerr=errors,
                marker="o",
                capsize=3,
                label=f"L_A={ell} minus global amplitude",
            )
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(range(4), [f"Fold {i}" for i in range(4)])
        ax.set_ylabel("Joint energy score difference [K]; lower is better")
        ax.set_title("Buffered RA holdout; bars show Monte Carlo error only")
        ax.legend()
        fig.savefig(args.output / "holdout.png", dpi=180)
        plt.close(fig)
    summary = {
        "full": rows,
        "heldout": fold_rows,
        "heldout_aggregate": aggregates,
        "registry": registry,
        "scope": "conditional Haslam residual model; no new external maps; no fitted beam/RSB",
        "error_model": "floor/RA process can include missing sky structure; not identified instrument noise",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# 连续 amplitude／β 场执行结果",
        "",
        "**文档状态：record。** T-001，2026-09-14。",
        "",
        "固定RSB与归档名义beam；β统一L=2，amplitude分别L=0/1/2。Haslam原生空间结构保留，",
        "平滑的是乘性修正。源差图和背景均值预算沿用既有约定，free-free尚未独立分离。",
        "",
        "|模型|RMS 600/818 K|95%覆盖|残差lag1|最大 amplitude–β 相关|",
        "|---|---|---|---|---|",
    ]

    def pair(values):
        return " / ".join(f"{x:.3f}" for x in values)

    for r in rows:
        p = r["predictive"]
        lines.append(
            f"|{r['name']}|{pair(p['sky_residual_rms_k'])}|{pair(p['predictive_95_coverage'])}|"
            f"{pair(p['sky_residual_lag1'])}|{r['max_abs_amplitude_beta_correlation']:.3f}|"
        )
    lines.extend(
        [
            "",
            "![Profiles](profiles.png)",
            "",
            "![Conditional fields](fields.png)",
            "",
            "地图包含依赖基底和先验的全天外推；同时保存amplitude和β不确定性，不能把像素数视为独立约束数。",
            "",
            "## 带缓冲RA留出",
            "",
            "每折70点训练、30点留出、20点缓冲；误差超参数仅用训练数据拟合。",
            "相关误差预测使用训练—留出协方差；同时报告仅天空均值的残差，避免把误差插值当作天空恢复。",
            "",
            "|折|L_A|天空RMS K|条件预测RMS K|95%覆盖|逐点log预测密度之和|联合块混合有效draw数|",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for r in fold_rows:
        lines.append(
            f"|{r['fold']}|{r['l_amplitude']}|{pair(r['sky_residual_rms_k'])}|"
            f"{pair(r['predictive_center_rms_k'])}|{pair(r['predictive_95_coverage'])}|"
            f"{pair(r['pointwise_log_predictive_sum'])}|{pair(r['block_mixture_effective_draws'])}|"
        )
    lines.extend(
        [
            "",
            "|L_A|四折逐点log评分|MCSE，仅参考|排名可靠|CRPS总和 K±MCSE|联合energy总和 K±MCSE|",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in aggregates:
        error = "未计算" if row["mcse_total"] is None else f"{row['mcse_total']:.3f}"
        lines.append(
            f"|{row['l_amplitude']}|{row['pointwise_total']:.3f}|{error}|"
            f"{row['pointwise_ranking_reliable']}|{row['crps_sum_k']:.3f}±{row['crps_sum_mcse_k']:.3f}|"
            f"{row['joint_energy_sum_k']:.3f}±{row['joint_energy_sum_mcse_k']:.3f}|"
        )
    if aggregates:
        lines.extend(
            [
                "",
                "![Holdout score changes](holdout.png)",
                "",
                "![Held profiles and predictive intervals](holdout-profiles.png)",
            ]
        )
    lines.extend(
        [
            "",
            "逐点评分之和不等于联合块评分。两者均须检查混合有效样本数；第4折逐点密度也由稀有样本主导，汇总log-score不用于排名，其delta MCSE亦不可靠。",
            "CRPS与joint energy基于预测样本而非稀有密度贡献，越低越好。CRPS总和覆盖两频率120个留出RA点；energy为每折60维物理K向量评分的四折和。误差条仅为Monte Carlo误差，不是跨天空区域的统计显著性。",
            "每折记录、后验样本、配置和源码快照的位置见summary.json。",
            "",
        ]
    )
    (args.output / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
