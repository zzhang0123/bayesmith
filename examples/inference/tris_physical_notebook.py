"""Build an executed physical-model notebook from verified local TRIS records."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from examples.inference.tris_notebook import execute, render_html, validate


def field_log_moments(samples, loading, indices, weights, prior_sd, log_scale_sd=0.5):
    """Exact empirical log-field moments; prior integrates the angular hyperprior."""
    samples, loading, indices, weights, prior_sd = map(
        np.asarray, (samples, loading, indices, weights, prior_sd)
    )
    if (
        samples.ndim != 2 or samples.shape[0] < 2
        or loading.shape != (samples.shape[1], samples.shape[1] - 1)
        or indices.shape != weights.shape or indices.ndim != 2
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0) or np.any(indices >= loading.shape[0])
        or np.any(weights < 0) or not np.allclose(weights.sum(1), 1, rtol=0, atol=1e-13)
        or prior_sd.shape != (samples.shape[1],) or np.any(prior_sd <= 0)
        or not all(np.isfinite(x).all() for x in (samples, loading, weights, prior_sd))
        or not np.isfinite(log_scale_sd) or log_scale_sd < 0
    ):
        raise ValueError("Finite aligned coefficients and positive normalized interpolation required")
    transform = np.column_stack((np.ones(len(loading)), loading))
    node_mean = transform @ samples.mean(0)
    covariance = transform @ np.cov(samples, rowvar=False) @ transform.T
    prior_variance = prior_sd**2 * np.r_[1, np.full(len(prior_sd)-1, np.exp(2*log_scale_sd**2))]
    prior_covariance = (transform * prior_variance) @ transform.T
    variance = np.zeros(len(indices))
    prior = np.zeros(len(indices))
    for i in range(indices.shape[1]):
        for j in range(indices.shape[1]):
            pair_weight = weights[:, i] * weights[:, j]
            variance += pair_weight * covariance[indices[:, i], indices[:, j]]
            prior += pair_weight * prior_covariance[indices[:, i], indices[:, j]]
    if np.any(variance <= 0) or np.any(prior <= 0):
        raise ValueError("Positive field variance required")
    return np.sum(node_mean[indices] * weights, axis=1), np.sqrt(variance), np.sqrt(prior)


SETUP = '''import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from examples.inference.tris_continuous import harmonic_basis
from examples.inference.tris_physical_notebook import field_log_moments

RHO = Path(os.environ["TRIS_PHYSICAL_RHO"])
ROOT = RHO / "runs/tris-variance-fine-20260915"
REAL_FIT = next((name for name in ("real-refined", "real-adaptive")
                 if (ROOT / (name+"-execution-contract.json")).is_file()), "real")
OUTPUT = Path(os.environ["TRIS_NOTEBOOK_OUTPUT"])
print("Executed snapshot UTC:", datetime.now(UTC).isoformat())
ASSETS = OUTPUT / "assets"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": .2})

def read(path):
    return json.loads(path.read_text()) if path.is_file() else None

def save(fig, name):
    fig.savefig(ASSETS / name, bbox_inches="tight")
    preview = OUTPUT / "previews"
    preview.mkdir(exist_ok=True)
    fig.savefig(preview / (Path(name).stem+".png"), bbox_inches="tight", dpi=120)
    plt.close(fig)

def decorate_sky(ax):
    # Longitude increases to the left; the projection coordinate is -longitude.
    ax.set_xticks(np.deg2rad([-120,-60,0,60,120]))
    ax.set_xticklabels(["120°","60°","0°","300°","240°"], fontsize=7)
    ax.set_yticks(np.deg2rad([-60,-30,0,30,60]))
    ax.tick_params(axis="y",labelsize=7)

fits = {}
for label, directory in (
    ("coarse simulation", RHO / "runs/tris-variance-gibbs-20260915/simulation/assessment"),
    ("fine simulation", ROOT / "simulation/assessment"),
    ("real observations", ROOT / REAL_FIT / "assessment"),
):
    record = read(directory / "summary.json")
    if record is None:
        active = read(directory.parent / "summary.json")
        print(label, "not assessed", None if active is None else
              {k: active.get(k) for k in ("status", "draws", "sampling_pass")})
        continue
    if record.get("status") != "sampled":
        raise ValueError("Unexpected assessment state")
    for path, sha in record["assessment_input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("Assessed raw chain or data changed")
    print(label, "sampling pass:", record["sampling_pass"], "draws:", record["draws"])
    if record["sampling_pass"]:
        with np.load(directory / "samples.npz", allow_pickle=False) as f:
            samples = dict(f)
        for chain in range(4):
            with np.load(directory.parent / f"chain{chain}.npz", allow_pickle=False) as f:
                if set(samples) != set(f.files) or any(not np.array_equal(samples[k][chain],f[k]) for k in f.files):
                    raise ValueError("Pooled samples differ from assessed raw chains")
        fits[label] = (directory, record, samples)
'''

SAMPLING = '''for label, (directory, record, samples) in fits.items():
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), constrained_layout=True)
    for ax, name in zip(axes.flat, ("zero_h", "gain_h", "a_angular_scale", "em_angular_scale")):
        for chain, values in enumerate(samples[name]):
            ax.plot(values, lw=.45, alpha=.65, label=f"chain {chain+1}")
        ax.set_title(name)
        ax.set_xlabel("retained draw")
    axes[0,0].set_ylabel("K")
    axes[0,0].legend(ncol=2, fontsize=8)
    fig.suptitle(label + ": accepted sampling, separate numerical/science gates")
    save(fig, label.replace(" ", "_")+"_traces.svg")
    print(label)
    for name in ("zero_h", "gain_h", "beta", "a_angular_scale", "em_angular_scale"):
        x = samples[name]
        print(name, "mean", x.mean((0,1)).tolist(), "SD", x.std((0,1),ddof=1).tolist())
    print("sampler:", record["sampler"])
    print("failures:", record["failures"])
    if "recovery_pass" in record:
        print("Fixed-injection recovery:",record["recovery_pass"],"(not SBC)")
'''

WARMUP_DIAGNOSTICS = '''for probe_name in ("adaptive-warm-numerics", "refined-warm-numerics"):
    probe = read(ROOT / (probe_name+".json"))
    if probe is not None:
        for path, sha in probe["input_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError("Saved warm-state numerical input changed")
        print(probe_name, "warm states only; no posterior admission")
        for state in probe["states"]:
            print("chain",state["chain"],{name:{"pass":r["prediction_budget_pass"],
                  "max_sigma":{k:v["max_abs_sigma"] for k,v in r["prediction_metrics"].items()}}
                  for name,r in state["comparisons"].items()})
active = read(ROOT / REAL_FIT / "summary.json")
if active is None or not active.get("warmup_chunks"):
    print("No saved real warmup chunks are available.")
else:
    names = ("gain_h", "zero_h", "a_angular_scale", "em_angular_scale")
    collected = {name: [] for name in names}
    count = 0
    for row in active["warmup_chunks"]:
        path = ROOT / REAL_FIT / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Saved warmup chunk changed")
        with np.load(path, allow_pickle=False) as chunk:
            size = chunk["sample_gain_h"].shape[1]
            for name in names:
                values = chunk["sample_"+name]
                if values.shape != (4, size):
                    raise ValueError("Warmup diagnostic layout differs")
                collected[name].append(values)
        count += size
        if row["iteration"] != count:
            raise ValueError("Warmup diagnostic iteration count differs")
    if count != active["warmup_iteration"]:
        raise ValueError("Saved warmup count differs from snapshot")
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), constrained_layout=True)
    for ax, name in zip(axes.flat, names):
        values = np.concatenate(collected[name], axis=1)
        for chain, curve in enumerate(values):
            ax.plot(np.arange(1,count+1),curve,lw=.65,label=f"chain {chain+1}")
        ax.set_title(name)
        ax.set_xlabel("warmup iteration")
    axes[0,0].legend(ncol=2, fontsize=8)
    iterations = [r["iteration"] for r in active["warmup_chunks"]]
    acceptance = np.array([r["mean_acceptance_per_chain"] for r in active["warmup_chunks"]])
    for chain in range(4):
        axes[2,1].plot(iterations,acceptance[:,chain],".-",lw=.7,label=f"chain {chain+1}")
    axes[2,1].set_title("Block mean acceptance")
    if all("step_size_per_chain" in r for r in active["warmup_chunks"]):
        steps = np.array([r["step_size_per_chain"] for r in active["warmup_chunks"]])
        for chain in range(4):
            axes[2,0].plot(iterations,steps[:,chain],".-",lw=.7)
        axes[2,0].set_yscale("log")
        axes[2,0].set_title("Step size after each block")
    else:
        axes[2,0].text(.1,.5,"Fixed step size: "+str(active.get("fixed_step_size")))
    for ax in axes[2]:
        ax.set_xlabel("warmup iteration")
    fig.suptitle("Real warmup diagnostics; these are not posterior draws")
    save(fig,"real_warmup_diagnostics.svg")
    divergences = np.sum([r["divergences_per_chain"] for r in active["warmup_chunks"]],axis=0)
    print("Execution:",REAL_FIT,"status:",active["status"],"warmup:",count,"/",active["warmup"])
    print("Warmup divergences per chain:",divergences.tolist())
    print("Saved posterior draws per chain:",active["draws"],"sampling pass:",active["sampling_pass"])
    print("Warmup and posterior samples are assessed separately; a warmup plot certifies neither mixing nor physics.")
'''

NUMERICS = '''numerical_records = {}
conditional_reviews = {}
for label, filename in (("fine simulation", "posterior-quadrature.json"),
                        ("real observations", REAL_FIT+"-posterior-quadrature.json")):
    numerical = read(ROOT / filename)
    if numerical is None:
        print(label, "posterior numerical assessment has not completed.")
        continue
    for path, sha in numerical["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("Numerical assessment input changed")
    correction_name = REAL_FIT if label == "real observations" else "simulation"
    correction = read(ROOT / (correction_name+"-collapsed-offset-review.json"))
    if correction is None or correction.get("status") != "completed":
        numerical = dict(numerical, numerical_budget_pass=False)
        print(label, "Conditional-zero820 numerical review is pending; numerical gate is not final.")
    else:
        for path, sha in correction["input_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
                raise ValueError("Conditional-offset review input changed")
        for name, sha in correction["source_sha256"].items():
            if hashlib.sha256((RHO / "examples/TRIS" / name).read_bytes()).hexdigest() != sha:
                raise ValueError("Conditional-offset helper changed")
        if correction["target_signature"] != fits[label][1]["target_signature"]:
            raise ValueError("Conditional-offset review belongs to another target")
        conditional_reviews[label] = correction
        numerical = dict(numerical, numerical_budget_pass=correction["numerical_budget_pass"])
        print(label, correction["policy"])
        print("Corrected conditional-zero820 numerical rows:", correction["numerical"])
    numerical_records[label] = numerical
    print(label, "numerical status:", numerical["status"], "budget pass:",numerical["numerical_budget_pass"])
    for key, result in numerical["results"].items():
        print(key, result["density"], "failures", result["failures"])
    results = numerical["results"]
    if results:
        descriptions = {
            "sky128_to_sky256": "Sky NSIDE 128 → 256",
            "sky128_to_beam512_sky128": "Beam NSIDE 256 → 512 at sky 128",
            "sky256_to_beam512_sky256": "Beam NSIDE 256 → 512 at sky 256",
            "sky128_to_beam512_sky256": "Sky and beam refined together",
            "beam512_sky256_to_beam512_sky512": "Sky 256 → 512 at beam 512",
            "sky256_to_beam512_sky512": "Sky and beam 256 → 512",
        }
        short_labels = {"sky128_to_sky256": "sky", "sky128_to_beam512_sky128": "beam at 128",
                        "sky256_to_beam512_sky256": "beam at 256", "sky128_to_beam512_sky256": "both",
                        "beam512_sky256_to_beam512_sky512": "sky at beam512",
                        "sky256_to_beam512_sky512": "both 256→512"}
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        for key, result in results.items():
            delta = np.asarray(result["delta_log_density"])
            axes[0].hist(delta-delta.mean(), bins=25, histtype="step", label=descriptions.get(key, key))
        axes[0].set_xlabel("centered log-density change")
        axes[0].set_ylabel("selected draws")
        axes[0].legend(fontsize=7)
        keys = list(results)
        axes[1].bar(np.arange(len(keys)), [results[k]["density"]["weight_ess_fraction"] for k in keys])
        axes[1].set_xticks(np.arange(len(keys)), [short_labels.get(k, k) for k in keys], fontsize=8)
        axes[1].set_xlabel("refinement")
        axes[1].set_ylabel("finite-state weight ESS fraction")
        axes[1].set_title("Weight concentration; not MCMC ESS", fontsize=9)
        fig.suptitle(label)
        save(fig, label.replace(" ", "_")+"_numerical_refinement.svg")
sensitivity = read(ROOT / (REAL_FIT+"-sensitivity.json"))
if sensitivity is None:
    print("Real uncertainty/prior sensitivity has not completed.")
else:
    for path, sha in sensitivity["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("Sensitivity input changed")
    print("Real sensitivity:", sensitivity["status"], sensitivity["policy"])
    for label, result in sensitivity["scenarios"].items():
        print(label, "finite weight ESS fraction:", result["finite_weight_ess_fraction"],
              "independent refit required:", result["requires_independent_refit"])
        for name, value in result["groups"].items():
            if name == "zero820" and label == "tris_archive_sigma_times2":
                corrected = conditional_reviews.get("real observations", {}).get("uncertainty", {}).get(label)
                print(name, "conditional correction:", corrected if corrected is not None else "pending")
            else:
                print(name, value)
review = read(ROOT / (REAL_FIT+"-sensitivity-review.json"))
if review is None or review.get("status") != "completed":
    print("Sensitivity-tail and real Cyg A convention review has not completed.")
else:
    for path, sha in review["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("Sensitivity review input changed")
    print(review["policy"])
    print("Stronger overlap checks:",review["uncertainty"])
    source_review = dict(review["source_convention"])
    source_review["groups"] = dict(source_review["groups"])
    source_review["groups"]["zero820"] = conditional_reviews.get("real observations", {}).get(
        "source_convention", "pending conditional correction")
    print("Cyg A convention:",source_review)
    print("Unresolved independent refits:",review["unresolved_refits"])
'''

PPC = '''for label, (directory, record, samples) in fits.items():
    ppc = read(directory / "ppc.json")
    if ppc is None:
        print(label, "PPC absent; no fit-adequacy conclusion")
        continue
    print(label, json.dumps(ppc, indent=2))
    with np.load(directory / "data.npz", allow_pickle=False) as f:
        data = dict(f)
    with np.load(directory / "predictive_means.npz", allow_pickle=False) as f:
        prediction = dict(f)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    ra = data["all_ra_deg"][:120] / 15
    for band in range(2):
        curves = prediction["scan_physical"][:, :120, band+1]
        interval = np.quantile(curves,[.025,.5,.975],axis=0)
        ax = axes[0,band]
        observation_label = "archive" if label == "real observations" else "simulated observation"
        ax.errorbar(ra,data["data"][:,band],yerr=data["sigma"][:,band],fmt=".",ms=3,label=observation_label)
        ax.plot(ra,interval[1],label="sky prediction, before common offset")
        ax.fill_between(ra,interval[0],interval[2],alpha=.2,label="95% conditional sky band")
        ax.set_title(label+f": {data['frequency_mhz'][band+1]:.1f} MHz")
        ax.set_ylabel("RJ K")
        ax.legend(fontsize=7)
        axes[1,band].plot(ra,data["data"][:,band]-interval[1],".")
        axes[1,band].axhline(0,color="grey",lw=.8)
        axes[1,band].set_xlabel("RA [hours]")
        axes[1,band].set_ylabel("data minus sky [K]")
    save(fig,label.replace(" ","_")+"_sky_profiles.svg")
    print("The curves exclude fitted common offsets. The separate covariance PPC includes them;")
    print("archive error bars exclude the fitted floor and RA-correlated discrepancy.")
deletion = read(ROOT / (REAL_FIT+"-ra-deletion.json"))
if deletion is None or deletion.get("status") != "completed":
    print("Buffered real RA deletion diagnostic has not completed.")
else:
    for path, sha in deletion["input_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError("RA deletion input changed")
    print(deletion["policy"])
    for fold in deletion["folds"]:
        print("fold",fold["fold"],"finite weight ESS fraction",fold["finite_weight_ess_fraction"],
              "Pareto k",fold["pareto_k"],"normalizer MCSE",fold["relative_normalizer_batch_mcse"],
              "refit required",fold["requires_independent_refit"])
        corrected_offsets = {x["fold"]: x["zero820"] for x in
                             conditional_reviews.get("real observations", {}).get("ra_deletion", [])}
        print("Conditional-zero820 shift:",corrected_offsets.get(fold["fold"],"pending correction"))
        if fold["predictive_intervals_accepted"]:
            print("95% predictive coverage out of12 rows per band:",fold["covered_per_band"],fold["interval_scope"])
'''

BETA = '''real_numerics = numerical_records.get("real observations", {})
if ("real observations" in fits and real_numerics.get("status") == "completed"
        and real_numerics.get("numerical_budget_pass") is True):
    directory, record, samples = fits["real observations"]
    with np.load(directory / "data.npz",allow_pickle=False) as f:
        prior_sd = f["beta_sd"]
    longitude, latitude = np.meshgrid(np.linspace(-np.pi,np.pi,241),np.linspace(-np.pi/2,np.pi/2,121))
    basis = harmonic_basis(np.pi/2-latitude, np.mod(longitude,2*np.pi), 2)
    beta = samples["beta"].reshape(-1,9)
    mean = basis @ beta.mean(0)
    covariance = np.cov(beta,rowvar=False)
    sd = np.sqrt(np.einsum("...i,ij,...j->...",basis,covariance,basis))
    prior_map_sd = np.sqrt(np.sum((basis*prior_sd)**2,axis=-1))
    fig, axes = plt.subplots(1,3,figsize=(13,4),subplot_kw={"projection":"mollweide"})
    for ax, values, title in zip(axes,(mean,sd,sd/prior_map_sd),
                                ("conditional synchrotron beta","posterior SD","posterior / prior SD")):
        m=ax.pcolormesh(-longitude,latitude,values,shading="auto",cmap="viridis")
        decorate_sky(ax)
        ax.set_title(title,fontsize=10)
        fig.colorbar(m,ax=ax,orientation="horizontal",pad=.12,shrink=.8)
    fig.suptitle("lmax=2 conditional reconstruction; weakly updated regions remain prior-dependent")
    save(fig,"conditional_beta.svg")
else:
    print("Real sampling and numerical gates have not both passed; no real beta map is displayed.")
'''

COMPONENTS = '''for label, (directory, record, samples) in fits.items():
    components = read(directory / "components.json")
    if components is None:
        print(label,"component report not available yet")
        continue
    path = directory / "components.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != components["artifact_sha256"]:
        raise ValueError("Component artifact changed")
    print(label,"component closure [scan, Haslam] K:",components["closure_max_k"])
    with np.load(path,allow_pickle=False) as f:
        values = {k:f["scan_"+k] for k in components["component_names"]}
    with np.load(directory/"data.npz",allow_pickle=False) as f:
        ra = f["all_ra_deg"][:120]/15
    fig, axes = plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for band,ax in enumerate(axes):
        for name,v in values.items():
            ax.plot(ra,np.median(v[:,:120,band+1],axis=0),label=name)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-4)
        ax.set_xlabel("RA [hours]")
        ax.set_ylabel("RJ K")
        ax.set_title(label+f": {components['frequency_mhz'][band+1]:.1f} MHz")
        ax.legend(fontsize=8,ncol=2)
    save(fig,label.replace(" ","_")+"_components.svg")
    print("Curves are marginal component medians. Closure is checked per draw, before taking medians.")
'''

FIELDS = '''geometry_record = read(ROOT / "plot-geometry/summary.json")
if geometry_record is None:
    print("Fixed plot geometry is unavailable.")
else:
    geometry_path = ROOT / "plot-geometry/data.npz"
    if hashlib.sha256(geometry_path.read_bytes()).hexdigest() != geometry_record["artifact_sha256"]:
        raise ValueError("Plot geometry changed")
    with np.load(geometry_path, allow_pickle=False) as f:
        geometry = dict(f)
    longitude, latitude = geometry["longitude"], geometry["latitude"]
    for label, (directory, record, samples) in fits.items():
        if label == "real observations" and not (
            real_numerics.get("status") == "completed" and real_numerics.get("numerical_budget_pass") is True
        ):
            print("Real field maps require the numerical gate as well as accepted sampling.")
            continue
        with np.load(directory / "data.npz", allow_pickle=False) as f:
            data = dict(f)
        for name, site, unit in (("a", "log_a", "K at 408 MHz"),
                                  ("em", "log_em_modes", "pc cm^-6")):
            mean, sd, prior = field_log_moments(
                samples[site].reshape(-1, samples[site].shape[-1]),
                data[name+"_node_loading"], geometry[name+"_index"],
                geometry[name+"_weight"], data[name+"_sd"], .5)
            fig, axes = plt.subplots(1,3,figsize=(13,4),subplot_kw={"projection":"mollweide"})
            for ax, value, title in zip(axes, (np.exp(mean), sd, sd/prior),
                                        ("geometric mean ["+unit+"]", "SD of log field", "posterior / prior log SD")):
                m = ax.pcolormesh(-longitude, latitude, value.reshape(longitude.shape), shading="auto")
                decorate_sky(ax)
                ax.set_title(title, fontsize=10)
                fig.colorbar(m,ax=ax,orientation="horizontal",pad=.12,shrink=.8)
            fig.suptitle(label+": "+name+"; spatial prior and fixed background are analysis conditions")
            save(fig,label.replace(" ","_")+"_"+name+"_field.svg")
            print(label,name,"all-draw empirical log-field moments; prior SD integrates LogNormal(0,.5) angular scale")
            print("Pixel-scale SD patterns also reflect the fixed node/interpolation geometry; display resolution is not inference resolution.")
        if label == "real observations":
            gain = samples["gain_h"].reshape(-1,1)
            zero = samples["zero_h"].reshape(-1,1)
            corrected = (data["haslam"][None,:]-zero)/gain
            fig, axes = plt.subplots(1,2,figsize=(10,4),subplot_kw={"projection":"mollweide"})
            for ax, value, title in zip(axes,(corrected.mean(0), corrected.std(0,ddof=1)),
                                        ("calibrated observed total sky [K]", "calibration-induced SD [K]")):
                m=ax.pcolormesh(-longitude,latitude,value[geometry["haslam_index"]],shading="auto")
                decorate_sky(ax)
                ax.set_title(title,fontsize=10)
                fig.colorbar(m,ax=ax,orientation="horizontal",pad=.12,shrink=.8)
            save(fig,"calibrated_haslam_total.svg")
            print("Haslam correction is (observed-zero)/gain. SD propagates joint calibration uncertainty,")
            print("not new per-pixel noise or a latent synchrotron-only map; NSIDE8 observations are shown.")
'''


def cells():
    return [
        ("markdown", ("# TRIS、Haslam 与 Hα：一致物理模型的推断\n\n"
         "本 notebook 从原始链和独立验收记录重新读取结果。模拟恢复、真实观测、采样质量、"
         "数值积分与模型适配分别呈现；尚未通过的阶段会明确显示。")),
        ("code", SETUP),
        ("markdown", ("## 观测对象与物理分量\n\n"
         "Haslam 是 408 MHz 总亮温测量，包含同步辐射、free-free、CMB、河外背景和保留的亮源。"
         "A408 是待推断的正同步辐射场；Haslam 只通过自己的增益、零点和观测算子约束它，"
         "不被当成无误差的纯同步辐射模板。Haslam采用 Hobs = gH R_H(T408) + zH；"
         "zH是加在模型上的仪器零点，不能直接当作待加回地图的修正值。\n\n"
         "TRIS 的 600.5/817.8 MHz 数据是已校正、经过波束卷积的扫描轮廓。"
         "Hα 是含尘埃/掩膜、光学校准及示踪误差的独立 EM 约束。"
         "TRIS 2427.8 MHz 和 Stockert/Villa-Elisa 1420 MHz 的源历元/响应契约尚未完成，"
         "本次没有把它们计入 likelihood。\n\n"
         "TRIS采用 dν = Rν(Tν) + zν + εν。600 MHz共同零点按Normal(0,0.066 K)"
         "精确边缘化；820 MHz按实验室范围Uniform(−0.660,0.660 K)处理。"
         "这个Uniform密度是明示的工作先验，不是论文给出了同样的概率分布。\n\n"
         "同步辐射采用 A408 exp[β log(ν/408)]；κ=0。A 与 EM 是连续的局部节点场，"
         "分别有独立的角尺度超参数；β 用 lmax=2 的 9 个谐波系数表示。"
         "RSB 固定为24.1(ν/310 MHz)^−2.599 K，CMB 从2.7255 K转换为RJ亮温。"
         "这套分析不拟合或宣称检测 RSB。\n\n"
         "free-free 发射及背景的吸收使用同一 EM 与 Te；当前同步辐射前景比例固定为1。"
         "Cyg A 显式加回并从所用河外平均背景中一致扣除其平均贡献；Cas A采用银河源和"
         "已到达地球的经验流量约定，不再施加第二次吸收。")),
        ("markdown", ("## 采样与固定注入恢复\n\n"
         "链轨迹来自通过独立采样验收的目标。粗网格的成功不意味着积分精度通过；"
         "一幅固定注入天空的恢复也不等于SBC覆盖率校准。")),
        ("code", SAMPLING),
        ("markdown", "## 真实运行的预热诊断\n\n预热轨迹、步长和接受率用于检查采样过程；这里的点不计入后验或天空重建。"),
        ("code", WARMUP_DIAGNOSTICS),
        ("markdown", ("## 天空积分与波束积分\n\n"
         "比较固定物理场和观测下的天空积分及native256→512波束积分；模拟使用128→256，最新真实候选使用256→512。"
         "权重集中度仅用于数值敏感性；不把它称为MCMC有效样本数。"
         "解析积分zero820后，其敏感性使用两种目标各自的条件均值；旧零点行由补充记录取代。")),
        ("code", NUMERICS),
        ("markdown", ("## 预测、残差与不确定度\n\n"
         "TRIS归档误差不是不可质疑的完整噪声模型。当前模型包含额外白噪声floor、"
         "RA相关项及每频率共同零点；PPC结论始终以这些约定为条件。"
         "Haslam采用每个NSIDE8观测单元1 K工作误差、增益LogNormal(0,0.1)、零点Normal(0,3 K)；"
         "这些不是由Haslam提供的完整误差协方差。角尺度超先验为LogNormal(0,0.5)，"
         "相关长度保持固定；需另外检查这些假设的敏感性。")),
        ("code", PPC),
        ("markdown", "## 各物理分量\n\n分量之和逐样本核对；图中显示每个分量的边际中位数。"),
        ("code", COMPONENTS),
        ("markdown", ("## 条件β重建\n\n"
         "全天重建由数据和空间先验共同决定，不能把两个扫描圈解释为直接测量了全天的细节。"
         "posterior/prior SD帮助识别更新较弱的区域。")),
        ("code", BETA),
        ("markdown", ("## 振幅、EM与Haslam校准\n\n"
         "A和EM显示几何均值exp(mean log field)，以及log场的标准差；不假设后验为lognormal。"
         "先验标准差已对角尺度超先验积分，不能与固定尺度先验混用。"
         "Haslam校准图显示总天空观测的(数据−零点)/增益，图中SD仅传播联合校准不确定性。"
         "模拟场仅用于验证；真实场仍须通过数值门槛。")),
        ("code", FIELDS),
    ]


def build(output, rho):
    output.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [dict(cell_type=kind, source=source.splitlines(keepends=True), metadata={},
                    **({"execution_count": None, "outputs": []} if kind == "code" else {}))
               for kind, source in cells()],
        "metadata": {"kernelspec": {"display_name": "python3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5,
    }
    environment = {"TRIS_NOTEBOOK_OUTPUT": str(output.resolve()), "TRIS_PHYSICAL_RHO": str(rho.resolve())}
    previous = {k: os.environ.get(k) for k in environment}
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = hashlib.sha256((str(index)+"".join(cell["source"])).encode()).hexdigest()[:16]
    try:
        sections = execute(notebook, environment=environment)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    for cell, section in zip(notebook["cells"], sections, strict=True):
        for name in section["images"]:
            cell["outputs"].append({"output_type": "display_data", "metadata": {},
                                        "data": {"image/svg+xml": (output / "assets" / name).read_text()}})
    path = output / "tris_physical_inference.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n")
    render_html(sections, output / "index.html")
    problems = validate(output / "index.html", output / "assets")
    if problems:
        raise ValueError(problems)
    record = {"status": "executed", "cells": len(sections), "validation_problems": problems,
                  "snapshot_at_utc": datetime.now(UTC).isoformat(),
                  "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "notebook_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "html_sha256": hashlib.sha256((output / "index.html").read_bytes()).hexdigest(),
                  "figures": sum(len(s["images"]) for s in sections)}
    (output / "execution.json").write_text(json.dumps(record, indent=2)+"\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rheplicant", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.rheplicant)))
