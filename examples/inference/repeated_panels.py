"""Bilingual presentation of repeated-simulation evidence, separate from one run."""

import html
import json

if __package__:
    from .methodology_guide import bi, detail, paragraph
    from .symbols import inline_symbol
else:
    from methodology_guide import bi, detail, paragraph
    from symbols import inline_symbol


LABELS = {"uniform": ("Bounded Uniform", "有界 Uniform"),
          "mild": ("Mild noise prior", "温和噪声先验")}


def coverage_text(value):
    lo, hi = value["interval_95"]
    return f'{value["covered"]}/{value["total"]} · {100*value["rate"]:.1f}% <small>[{100*lo:.1f}%, {100*hi:.1f}%]</small>'


def render_repeated(summary, folder):
    config = summary["registered"]
    n = len(config["seeds"])
    obs = sorted({r["observations"] for r in summary["runs"]})
    text = '<section class="variant-note repeated-study"><h3>' + bi(
        f"Repeated simulations · {n} registered datasets", f"重复模拟 · 预先固定的 {n} 组数据"
    ) + '</h3>' + paragraph(
        f"Fixed root truths; redraw both the process instance and observation noise for each seed. Each dataset has {', '.join(map(str, obs))} observations. Both priors use the same data, {config['chains']} chains × {config['draws']:,} retained draws and {config['warmup']:,} warmup steps per chain. The registered seed list is retained in the protocol.",
        f"固定顶层参数真值；每个种子重新抽取过程实例和观测噪声。每组含 {', '.join(map(str, obs))} 个观测。两种先验使用相同数据、{config['chains']} 条链各 {config['draws']:,} 个保留样本、每条链 {config['warmup']:,} 步预热。预先固定的种子列表保存在实验方案中。",
    )
    text += paragraph(
        "These are fixed-root recovery experiments, not prior-predictive SBC. All completed runs enter the primary summaries, including diagnostic failures; failed attempts and the diagnostic-passing subset remain visible. No seed replacement or tuning to recovery.",
        "这是固定顶层真值的恢复实验，不是从先验生成参数的 SBC。所有完成的运行都计入主要汇总，包括诊断失败；未完成的尝试及诊断通过的子集单独保留。不替换种子，不为通过恢复而调参。",
    )
    text += '<div class="table-scroll"><table><thead><tr>' + ''.join('<th>'+bi(*x)+'</th>' for x in (
        ("Prior", "先验"), ("Completed / attempted", "完成 / 尝试"),
        ("Diagnostics passed", "诊断通过"), ("σ_w bias ± MCSE", "σ_w 偏差 ± MCSE"),
        ("σ_w RMSE", "σ_w RMSE"),
    )) + '</tr></thead><tbody>'
    for prior, group in summary["groups"].items():
        value = group["parameters_all_completed"].get("sigma_w")
        text += '<tr><th>'+bi(*LABELS[prior])+f'</th><td>{group["completed"]}/{group["attempted"]}</td><td>{group["diagnostics_passed"]}/{group["completed"]}</td>'
        text += (f'<td>{value["bias"]:+.6f} ± {value["bias_mcse"]:.6f}</td><td>{value["rmse"]:.6f}</td>' if value else '<td>—</td><td>—</td>')+'</tr>'
    text += '</tbody></table></div><div class="table-scroll"><table><thead><tr><th>'+bi("σ_w coverage", "σ_w 覆盖率")+'</th>'
    text += ''.join(f'<th>{100*mass:g}%</th>' for mass in config["interval_masses"])+'</tr></thead><tbody>'
    for prior, group in summary["groups"].items():
        value = group["parameters_all_completed"].get("sigma_w")
        text += '<tr><th>'+bi(*LABELS[prior])+'</th>'+''.join('<td>'+coverage_text(value["coverage"][str(mass)])+'</td>' if value else '<td>—</td>' for mass in config["interval_masses"])+'</tr>'
    text += '</tbody></table></div>'+paragraph(
        f"Bracketed ranges are 95% binomial confidence intervals for the measured coverage rate. With {n} datasets, even {n}/{n} covers gives a lower bound of {100*0.025**(1/n):.1f}%; it cannot establish precise 99% coverage. Bias MCSE measures variation across datasets and includes finite-chain error. These are pointwise checks at one root-parameter setting.",
        f"方括号为实测覆盖率的 95% 二项置信区间。只有 {n} 组数据时，即使 {n}/{n} 覆盖，下界也仅为 {100*0.025**(1/n):.1f}%，无法精确验证 99% 覆盖率。偏差 MCSE 衡量跨数据集的波动，并包含有限链误差。这些是在一组顶层参数设置下的逐项检查。",
    )
    sigma = [group["parameters_all_completed"].get("sigma_w") for group in summary["groups"].values()]
    if all(value and value["bias_interval_95"][0] <= 0 <= value["bias_interval_95"][1] for value in sigma):
        text += paragraph(
            "Both 95% intervals for mean bias include zero. This pilot does not resolve a systematic σ_w mean bias; it does not prove unbiasedness. Recovering every coordinate in every dataset is a different criterion.",
            "两种先验下平均偏差的 95% 区间均包含零。这批试验没有分辨出 σ_w 后验均值的系统性偏差，但不能证明无偏。要求每次数据的全部坐标都恢复，是另一项标准。",
        )
    for prior, group in summary["groups"].items():
        text += '<p>'+bi(*LABELS[prior])+': '+bi(
            f"all-coordinate single-dataset recovery {group['all_coordinates_recovered']}/{group['completed']}; chain diagnostics {group['diagnostics_passed']}/{group['completed']}.",
            f"单次数据的全部坐标恢复 {group['all_coordinates_recovered']}/{group['completed']}；链诊断 {group['diagnostics_passed']}/{group['completed']}。",
        )+'</p>'
    paired = summary["paired_mild_minus_uniform"].get("sigma_w")
    if paired:
        delta = paired["mean_change"]
        absolute = paired["absolute_error_change"]
        text += paragraph(
            f"{paired['pairs']} completed pairs, mild minus Uniform (± MCSE): mean estimate changes by {delta['mean']:+.6f} ± {delta['mcse']:.6f}; absolute error changes by {absolute['mean']:+.6f} ± {absolute['mcse']:.6f}. A positive mean shift alone does not establish smaller bias or error.",
            f"{paired['pairs']} 组完成的配对，温和先验减去 Uniform（± MCSE）：平均估计改变 {delta['mean']:+.6f} ± {delta['mcse']:.6f}；绝对误差改变 {absolute['mean']:+.6f} ± {absolute['mcse']:.6f}。均值上移本身不能证明偏差或误差减小。",
        )
    prefix = html.escape(folder, quote=True)
    text += f'<figure><img src="{prefix}/sigma-repeats.png" alt="Noise-scale posterior intervals / 噪声尺度后验区间" loading="lazy"><figcaption>'+bi("Dots: posterior means; bars: 95% posterior intervals; dashed line: generating σ_w. Red ×: diagnostics failed, retained. Each seed redraws the full process and noise.", "圆点：后验均值；横线：95% 后验区间；虚线：生成 σ_w。红色 ×：诊断失败，仍予保留。每个种子重新抽取整个过程与噪声。")+f' <a href="{prefix}/sigma-repeats.svg" download>SVG</a></figcaption></figure>'
    rows = '<div class="table-scroll"><table><thead><tr><th>'+bi("Parameter", "参数")+'</th><th>'+bi("Prior", "先验")+'</th><th>'+bi("Bias ± MCSE", "偏差 ± MCSE")+'</th><th>RMSE</th><th>'+bi("99% coverage", "99% 覆盖率")+'</th></tr></thead><tbody>'
    for prior, group in summary["groups"].items():
        for name, value in group["parameters_all_completed"].items():
            rows += '<tr><th>'+inline_symbol("composed_process", name.split("[")[0])+f' <code>{html.escape(name)}</code></th><td>'+bi(*LABELS[prior])+f'</td><td>{value["bias"]:+.6f} ± {value["bias_mcse"]:.6f}</td><td>{value["rmse"]:.6f}</td><td>'+coverage_text(value["coverage"]["0.99"])+'</td></tr>'
    text += detail(("All coordinates and priors", "全部坐标与先验"), rows+'</tbody></table></div>')
    text += detail(("Diagnostics, unsuccessful attempts and registered protocol", "诊断、未完成尝试与预先固定的方案"), '<pre>'+html.escape(json.dumps({
        "registered": config, "completion": summary["completion"], "failed_attempts": summary["failed_attempts"],
        "diagnostic_passing_subsets": {k:v["parameters_diagnostics_passed"] for k,v in summary["groups"].items()},
    }, indent=2))+'</pre>')
    text += '<p>'+bi("More independent observations generally improve σ_w precision. In the known-mean Gaussian limit, SE(σ_w) ≈ σ_w / √(2n). Increasing MCMC draws instead improves numerical precision for the existing dataset. Denser measurements of one process instance do not create new independent process modes.", "更多独立观测通常提高 σ_w 的精度。在均值已知的高斯极限下，SE(σ_w) ≈ σ_w / √(2n)。增加 MCMC 样本则提高现有数据的计算精度。更密地观测同一个过程实例，并不会产生新的独立过程模式。")+'</p>'
    return text+f'<p><a href="{prefix}/summary.json" download>'+bi("Download study summary and per-run statistics", "下载实验汇总与逐次统计")+'</a></p></section>'
