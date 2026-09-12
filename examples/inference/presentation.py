"""Bilingual walkthrough of saved demos; no inference or plotting dependencies."""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from urllib.parse import quote

if __package__:
    from .case_content import CASES as CASE_CONTENT
    from .case_content import LEGACY_CASES
    from .comparison_policy import (
        COMPARISON_GALLERIES,
        RETIRED_GALLERIES,
        comparison_visible,
    )
    from .symbols import inline_symbol, parameter_symbols, symbol
else:
    from case_content import CASES as CASE_CONTENT
    from case_content import LEGACY_CASES
    from comparison_policy import (
        COMPARISON_GALLERIES,
        RETIRED_GALLERIES,
        comparison_visible,
    )
    from symbols import inline_symbol, parameter_symbols, symbol

CASES = {name: {key: value[0] if isinstance(value, tuple) else value
                for key, value in data.items()} for name, data in CASE_CONTENT.items()}

LABELS = {
    "x": "Input position",
    "X": "Design matrix",
    "beta": "Coefficients",
    "channel": "Channel index",
    "time": "Observation time",
    "group_index": "Group index",
    "slope": "Slope",
    "intercept": "Intercept",
    "rate": "Decay rate",
    "amplitude": "Amplitude",
    "offset": "Background",
    "population": "Population mean",
    "groups": "Group locations",
    "basis": "Decay basis",
    "location": "Mean signal",
    "logits": "Log odds",
    "obs": "Observations",
    "U": "Gain design",
    "A": "Signal design",
    "p_g": "Gain parameters",
    "p_n": "Signal parameters",
    "gain": "Exponential gain",
    "signal": "Linear signal",
    "mu": "Mean signal",
}
ROLES = {
    "constant": "Known input",
    "latent": "Random operator · latent",
    "deterministic": "Deterministic operator",
    "observed": "Random operator · observed",
}
ROLES_ZH = {
    "constant": "已知输入", "latent": "随机算子 · 潜变量",
    "deterministic": "确定性算子", "observed": "随机算子 · 观测",
}
NODE_LABELS = {
    "alpha": ("Power-law index", "幂律指数"),
    "tail_index": ("Pareto tail index", "Pareto 尾指数"),
    "x": ("Input position", "输入位置"), "X": ("Design matrix", "设计矩阵"),
    "beta": ("Coefficients", "系数"), "time": ("Observation time", "观测时间"),
    "channel": ("Channel index", "通道索引"), "group_index": ("Group index", "组索引"),
    "rate": ("Decay rate", "衰减率"), "amplitude": ("Amplitude", "振幅"),
    "offset": ("Background", "背景"), "population": ("Population mean", "总体均值"),
    "groups": ("Random group locations", "随机组位置"),
    "basis": ("Decay basis", "衰减基函数"), "location": ("Mean signal", "均值信号"),
    "logits": ("Log odds", "对数几率"), "obs": ("Observed data", "观测数据"),
    "U": ("Gain design", "增益设计矩阵"), "A": ("Signal design", "信号设计矩阵"),
    "p_g": ("Gain coefficients", "增益系数"), "p_n": ("Signal coefficients", "信号系数"),
    "signal": ("Linear signal", "线性信号"), "mu": ("Mean signal", "均值信号"),
    "power_amplitude": ("Power amplitude", "功率振幅"),
    "frequencies": ("Fourier frequencies", "Fourier 频率"),
    "power": ("Power spectrum", "功率谱"), "instance": ("Random Fourier instance", "随机 Fourier 实例"),
    "response": ("Linear response matrix", "线性响应矩阵"),
    "response_signal": ("Response to the instance", "实例的线性响应"),
    "nonlinear": ("Bump center and width", "凸起中心与宽度"),
    "nonlinear_shape": ("Gaussian bump", "高斯凸起"),
    "combined": ("Response plus bump", "响应加凸起"),
    "background": ("Background coefficients", "背景系数"),
    "background_design": ("Background design", "背景设计矩阵"),
    "gain_design": ("Gain design", "增益设计矩阵"),
    "additive_signal": ("Combined additive signal", "合成加性信号"),
    "sigma_w": ("White-noise scale", "白噪声尺度"),
}
STEPS = (
    ("model", "Model", "Operators · DAG"),
    ("simulation", "Simulate", "Set truth · Draw data"),
    ("inference", "Infer", "Hide truth · Fit data"),
    ("recovery", "Check recovery", "Reveal truth · Check posterior"),
)


def esc(value):
    return html.escape(str(value), quote=True)


def parameter_value(value, row):
    """Resolve narrow intervals without adding noisy digits to broad ones."""
    sd = row["posterior_sd"]
    decimals = max(3, min(8, 1 - math.floor(math.log10(sd)))) if sd > 0 else 3
    return f"{value:.{decimals}f}"


def parameter_label(name):
    return LABELS.get(name.split("[", 1)[0], name)


def dag_svg(nodes, prefix, *, labels=None, symbols=None):
    """Right-align dependencies: edges in these demos span adjacent columns.

    Both nodes and edges come from result.json, including the stochastic edge
    population -> groups. Labels are explanatory; operator names stay intact.
    """
    children = {node["name"]: [] for node in nodes}
    for node in nodes:
        for parent in node["parents"]:
            children[parent].append(node["name"])
    distance = {}
    for node in reversed(nodes):
        distance[node["name"]] = 1 + max(
            (distance[c] for c in children[node["name"]]), default=-1
        )
    maximum = max(distance.values())
    layers = {}
    for node in nodes:
        layers.setdefault(maximum - distance[node["name"]], []).append(node)
    width, height = (maximum + 1) * 264 - 32, max(map(len, layers.values())) * 136 + 20
    positions = {}
    for layer, members in layers.items():
        for index, node in enumerate(members):
            positions[node["name"]] = (
                layer * 264 + 4,
                (index + 0.5) * (height / len(members)) - 52,
            )
    arrow = esc(prefix + "-arrow")
    parts = [
        f'<svg class="dag" style="width:{width}px" role="img" aria-labelledby="{esc(prefix)}-dag-title" viewBox="0 0 {width} {height}">',
        f'<title id="{esc(prefix)}-dag-title">Traced DAG with {len(nodes)} nodes. Arrows point from inputs to the nodes that depend on them.</title>',
        f'<defs><marker id="{arrow}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8c98a3"/></marker></defs>',
    ]
    for node in nodes:
        x, y = positions[node["name"]]
        for parent in node["parents"]:
            px, py = positions[parent]
            start, end = px + 224, x - 6
            middle = (start + end) / 2
            parts.append(
                f'<path class="edge" data-parent="{esc(parent)}" data-child="{esc(node["name"])}" d="M {start} {py + 52} C {middle} {py + 52}, {middle} {y + 52}, {end} {y + 52}" marker-end="url(#{arrow})"/>'
            )
    for node in nodes:
        x, y = positions[node["name"]]
        name = node["name"]
        parts.append(
            f'<g class="node {esc(node["kind"])}" data-node="{esc(name)}"><title>{esc(name)} · {esc(ROLES[node["kind"]])} · operator: {esc(node["operator"])}</title><rect x="{x}" y="{y}" width="224" height="104" rx="8"/>'
        )
        label = (labels or {}).get(name, NODE_LABELS.get(name, (name, name)))
        if name == "gain":
            label = ("Gain coefficients", "增益系数") if prefix == "composed_process" else ("Exponential gain", "指数增益")
        if prefix == "power_law" and name == "basis":
            label = ("Power-law basis", "幂律基函数")
        for language, role, description in zip(("en", "zh"), (ROLES[node["kind"]], ROLES_ZH[node["kind"]]), label):
            parts.append(f'<text x="{x + 13}" y="{y + 20}" class="node-role translated" lang="{language}">{esc(role)}</text>')
            parts.append(f'<text x="{x + 13}" y="{y + 69}" class="node-description translated" lang="{language}">{esc(description)}</text>')
        node_symbol = (symbols or {}).get(name, symbol(prefix, name)[1])
        parts.append(f'<text x="{x + 13}" y="{y + 47}" class="node-symbol">{esc(node_symbol)}</text>')
        parts.append(f'<text x="{x + 13}" y="{y + 90}" class="node-name">{esc(name)}</text></g>')
    return "".join(parts) + "</svg>"


def operator_table(nodes, case=""):
    if __package__:
        from .methodology_guide import bi
    else:
        from methodology_guide import bi
    rows = "".join(
        f'<tr><th scope="row">{inline_symbol(case, n["name"])} <code>{esc(n["name"])}</code></th><td>{bi(ROLES[n["kind"]], ROLES_ZH[n["kind"]])}</td><td><code>{esc(n["operator"])}</code></td><td>{parameter_symbols(case, n["parents"]) if n["parents"] else "—"}</td></tr>'
        for n in nodes
    )
    return (
        '<details><summary>' + bi("Operators and dependencies", "算子与依赖关系") + '</summary><div class="table-scroll"><table><thead><tr><th>' + bi("Symbol / node", "符号 / 节点") + '</th><th>' + bi("Role", "角色") + '</th><th>' + bi("Declared operator", "声明的算子") + '</th><th>' + bi("Depends on", "依赖于") + '</th></tr></thead><tbody>'
        + rows
        + "</tbody></table></div></details>"
    )


def route_text(report):
    method = report["method"]
    if method == "gcr":
        return (
            "Exact Gaussian (GCR)",
            "The whole graph is linear Gaussian. Draw independent posterior samples directly.",
        )
    if method == "gcr+nuts":
        return (
            "Exact updates + NUTS",
            "GCR updates linear coefficients; NUTS updates the nonlinear rate.",
        )
    if method == "nuts":
        return "NUTS sampling", "Use MCMC to explore the joint parameter posterior."
    return method, "See the saved execution plan for dispatch details."


def figure(folder, name, caption):
    path = quote(folder, safe="") + "/" + name
    return f'<figure><img src="{path}.png" alt="{esc(caption)}" width="1700" height="710" loading="lazy"><figcaption>{esc(caption)} <a href="{path}.svg" download>Download SVG</a></figcaption></figure>'


def blocking_panel(report, *, heading="Automatic blocking"):
    """Display returned blocks and refusal reasons, never infer labels from a case name."""
    if __package__:
        from .case_panels import method_label
    else:
        from case_panels import method_label
    audit = report.get("blocking")
    if audit is None:
        return '<p class="reading-note">This saved run predates structured blocking records. See its execution plan below.</p>'
    parts = [
        f'<section class="blocking"><h3>{esc(heading)}</h3><p class="blocking-intro">Strategy, evidence and outcome recorded from this model.</p>'
    ]
    for key, label in (
        ("executed", "Executed strategy"),
        ("factor_comparison", "Alternative automatic strategy · inspected only"),
    ):
        result = audit[key]
        if key == "executed" and not result["executed"]:
            label = "Compiled strategy · not executed"
        if key == "factor_comparison":
            parts.append(
                '<details class="factor-comparison"><summary>Compare factor_partition on the same graph</summary>'
            )
        parts.append(
            f'<p class="strategy-label">{label}</p><code class="strategy-name">{esc(result["strategy"])}</code><p class="strategy-description">{esc(result["description"])}</p>'
        )
        if result["status"] == "refused":
            parts.append(
                f'<div class="partition-refusal"><strong>No partition returned · {esc(result["exception"])}</strong><p>{esc(result["reason"])}</p><span>No alternative samples were generated.</span></div>'
            )
        else:
            parts.append(
                f'<p class="partition-outcome">Result: {len(result["blocks"])} block(s) · {"used for the saved posterior" if result["executed"] else "inspection only; not executed"}</p>'
            )
            parts.append(
                '<div class="table-scroll"><table class="blocks-table"><thead><tr><th>Block / parameters</th><th>Selected method</th><th>Returned reason</th></tr></thead><tbody>'
            )
            for block in result["blocks"]:
                display_index = block["index"] + 1 if type(block["index"]) is int else block["index"]
                outside = ", ".join(block["conditional_on"]) or "none (joint update)"
                evidence = ""
                if block["linearity_evidence"] is not None:
                    evidence = f"<details><summary>Numerical evidence</summary><p>Condition bound: {esc(block['kappa'])}<br>Solver tolerance: {esc(block['solver_tolerance'])}</p><pre>{esc(block['linearity_evidence'])}</pre></details>"
                parts.append(
                    f'<tr><th scope="row"><span class="block-index">Block {esc(display_index)}</span>{parameter_symbols(report.get("case", ""), block["latents"], show_names=True)}<small>{esc(block["coordinates"])} scalar coordinates<br>Condition on: {esc(outside)}</small></th><td><span class="method-badge">{method_label(block["method"])}</span></td><td>{esc(block["reason"])}{evidence}</td></tr>'
                )
            parts.append("</tbody></table></div>")
            parts.append(
                f"<details><summary>Raw plan returned by the compiler</summary><pre>{esc(result['plan_text'])}</pre></details>"
            )
        for warning in result.get("warnings", []):
            parts.append(
                f'<p class="partition-refusal">Compiler warning: {esc(warning)}</p>'
            )
        if result.get("log_space_kind"):
            parts.append(
                f"<p>Log-space reading: <code>{esc(result['log_space_kind'])}</code>. Multiplicative Gaussian noise uses a small-noise approximation.</p>"
            )
        if key == "factor_comparison":
            parts.append("</details>")
    parts.append(
        '<details><summary>Input priors seen by the partitioner</summary><div class="table-scroll"><table><thead><tr><th>Latent</th><th>Prior family</th><th>Coordinates</th></tr></thead><tbody>'
    )
    parts.extend(
        f'<tr><th scope="row"><code>{esc(prior["latent"])}</code></th><td>{esc(prior["family"])}</td><td>{esc(prior["coordinates"])}</td></tr>'
        for prior in audit["priors"]
    )
    parts.append("</tbody></table></div></details></section>")
    for probe in report.get("blocking_probes", []):
        parts.append(
            f'<details class="scale-probe" open><summary>Scale check: {esc(probe["title"])} · compilation only</summary><p>{esc(probe["observations"])} simulated observations · seed {esc(probe["seed"])} · {esc(probe["simulation_method"])}</p><p>{esc(probe["design"])}</p><p class="reading-note">The compiler was run on this larger graph. No posterior sampling or parameter-recovery check was run for it; the plots below belong to the four-coordinate example.</p>'
        )
        parts.append(blocking_panel(probe, heading="Partition returned for the larger graph"))
        parts.append("</details>")
    return "".join(parts)


def render_case(report, folder, index):
    if report.get("kind") == "real_observations" and report["case"] == "tris_haslam":
        if __package__:
            from .tris_presentation import render_case as render
        else:
            from tris_presentation import render_case as render
        return render(report, folder, index)
    if report.get("kind") == "real_observations" and report["case"] == "tris_haslam_rsb_comparison":
        if __package__:
            from .tris_rsb_presentation import render_case as render
        else:
            from tris_rsb_presentation import render_case as render
        return render(report, folder, index)
    if __package__:
        from .case_panels import render_case as render
    else:
        from case_panels import render_case as render
    return render(report, folder, index)


def gallery_navigation(reports, directory):
    """Use one baseline catalogue on every sibling gallery, including variants."""
    import os

    if __package__:
        from .methodology_guide import EXAMPLES, bi
    else:
        from methodology_guide import EXAMPLES, bi

    local = {report["case"]: report for report, _ in reports}
    is_comparison = directory.name in {*COMPARISON_GALLERIES, *RETIRED_GALLERIES} and any(
        (directory.parent / name / "result.json").exists() for name in CASES
    )
    catalogue = directory.parent if is_comparison else directory
    titles_zh = {key: title[1] for key, title, *_ in EXAMPLES}

    def href(target, case, step="methods"):
        path = Path(os.path.relpath(target / "index.html", directory)).as_posix()
        return f'{path}#case={quote(case)}&step={step}'

    entries = []
    for index, case in enumerate(CASES, 1):
        if not (catalogue / case / "result.json").exists() and case not in local:
            continue
        label = f'<span>{index:02d}</span>{bi(CASES[case]["subtitle"], titles_zh[case])}'
        if not is_comparison and case in local:
            entry = f'<button type="button" class="case-button" data-select-case="{esc(case)}" aria-controls="{esc(case)}" aria-pressed="false">{label}</button>'
        elif (catalogue / case / "result.json").exists():
            entry = f'<a class="case-button" data-gallery-link data-preserve-step href="{esc(href(catalogue, case))}">{label}</a>'
        else:
            entry = f'<span class="case-button">{label}</span>'
        for kind, title in COMPARISON_GALLERIES.items():
            if not comparison_visible(kind, case):
                continue
            variant = catalogue / kind
            if not (variant / case / "result.json").exists():
                continue
            if variant == directory and case in local:
                entry += f'<button type="button" class="case-variant-link" data-select-case="{esc(case)}" aria-controls="{esc(case)}" aria-pressed="false">{bi(*title)}</button>'
            else:
                entry += f'<a class="case-variant-link" data-gallery-link data-preserve-step href="{esc(href(variant, case))}">{bi(*title)}</a>'
        if comparison_visible("jeffreys", case) and (catalogue / "jeffreys" / case / "status.json").exists():
            entry += '<span class="case-variant-link">' + bi("Marginal Jeffreys: not implemented · no samples", "边缘 Jeffreys：未实现 · 无样本") + '</span>'
        entries.append(entry)
    for case in local.keys() & LEGACY_CASES.keys():
        if not comparison_visible(directory.name, case):
            continue
        entries.append(f'<button type="button" class="case-variant-link" data-select-case="{esc(case)}" aria-controls="{esc(case)}" aria-pressed="false">{bi(*LEGACY_CASES[case]["subtitle"])}</button>')
    return "".join(entries)


def observation_navigation(reports, directory):
    """Real data have their own catalogue, including on comparison pages."""
    if __package__:
        from .methodology_guide import bi
    else:
        from methodology_guide import bi
    entries = []
    for case, label in (("tris_haslam", bi("TRIS × Haslam sky", "TRIS × Haslam 天空")), ("tris_haslam_rsb_comparison", bi("TRIS + Haslam + RSB", "TRIS + Haslam + RSB"))):
        if any(report.get("case") == case for report, _ in reports):
            entries.append(f'<button type="button" class="case-button" data-select-case="{case}" aria-controls="{case}" aria-pressed="false">{label}</button>')
        elif directory.name in {*COMPARISON_GALLERIES, *RETIRED_GALLERIES} and (directory.parent / case / "result.json").exists():
            entries.append(f'<a class="case-button" data-gallery-link data-preserve-step href="../index.html#case={case}&step=model">{label}</a>')
    if not entries:
        return ""
    return '<section class="sidebar-observations"><h2>' + bi("Real observations", "真实观测") + '</h2><nav aria-label="Real observations">' + "".join(entries) + '</nav></section>'


def write_gallery(reports, directory):
    if __package__:
        from .methodology import render_methodology
        from .methodology_guide import bi
        from .tris_presentation import SCRIPT as tris_script
        from .tris_presentation import STYLE as tris_style
    else:
        from methodology import render_methodology
        from methodology_guide import bi
        from tris_presentation import SCRIPT as tris_script
        from tris_presentation import STYLE as tris_style

    reports = [(report, folder) for report, folder in reports
               if comparison_visible(directory.name, report["case"])]
    root = Path(__file__).parent
    comparison_inputs = {}
    probe_path = directory / "structure-probes.json"
    probes = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    if probes:
        comparison_inputs[probe_path.name] = hashlib.sha256(probe_path.read_bytes()).hexdigest()
    display_reports = []
    for report, folder in reports:
        shown = dict(report)
        probe_key = {"multiplicative_noise": "demo5", "composed_process": "demo6"}.get(report["case"])
        if probe_key and not report.get("variant") and not report.get("proposal_policies"):
            shown["structure_probe"] = probes.get(probe_key)
        links = []
        for subfolder, label in COMPARISON_GALLERIES.items():
            if not comparison_visible(subfolder, report["case"]):
                continue
            result_path = directory / subfolder / report["case"] / "result.json"
            status_path = result_path.with_name("status.json")
            if result_path.exists():
                links.append({"href": f'{subfolder}/index.html#case={quote(report["case"])}&step=methods', "label": label, "kind": subfolder})
                comparison_inputs[str(result_path.relative_to(directory))] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            elif status_path.exists():
                shown["jeffreys_unavailable"] = json.loads(status_path.read_text())
                comparison_inputs[str(status_path.relative_to(directory))] = hashlib.sha256(status_path.read_bytes()).hexdigest()
        if report.get("variant") or report.get("signal", {}).get("view", {}).get("jeffreys_prior") or (report.get("proposal_policies") and report.get("proposal_selection") != "automatic"):
            baseline = directory.parent / report["case"] / "result.json"
            if baseline.exists():
                links.append({"href": f'../index.html#case={quote(report["case"])}&step=methods', "label": ("Original demo", "原始 demo"), "kind": "baseline"})
        shown["comparison_links"] = links
        display_reports.append((shown, folder))
    design_html, design_sources = render_methodology(directory, [(r, f) for r, f in reports if r.get("kind") != "real_observations"])
    style = (root / "gallery.css").read_text() + (root / "methodology.css").read_text() + tris_style
    script = (root / "gallery.js").read_text() + "\n" + (root / "methodology.js").read_text() + tris_script
    navigation = gallery_navigation(display_reports, directory)
    observations = observation_navigation(display_reports, directory)
    comparison = ""
    if any(report.get("proposal_policies") and report.get("proposal_selection") != "automatic" for report, _ in reports):
        comparison = f'<p class="demo-caption">{bi("Explicit proposal schedules", "显式指定的提议方案")}</p>'
    page = '<!doctype html><html lang="en" data-guide-language="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>bayesmith · An inference walkthrough</title><link rel="icon" href="data:,">'
    page += (
        f'<link rel="stylesheet" href="vendor/katex/katex.min.css"><style>{style}</style></head><body><div class="app">'
        '<aside class="sidebar"><a class="brand" href="#">bayesmith<span>inference notebook</span></a>'
        f'<button type="button" class="design-entry" data-open-design aria-controls="methodology" aria-pressed="false">{bi("Design & diagnostics", "设计与诊断")}</button>'
        '<section class="sidebar-demos" aria-labelledby="verified-demos">'
        f'<h2 id="verified-demos">{bi("Verified with demos", "Demo 验证")}</h2>'
        f'<p class="demo-caption">{bi("Forward simulation & posterior recovery", "先模拟，再推断；以已知真值检查恢复")}</p>'
        f'<nav aria-label="Examples">{navigation}</nav>{comparison}</section>'
        f'{observations}'
        '<section class="sidebar-credits" aria-label="Project credits / 项目署名">'
        f'<p><span>{bi("Designed by", "设计")}</span><strong>Zheng Zhang</strong></p>'
        f'<p><span>{bi("Assistance from", "辅助支持")}</span><strong class="assistance-credit">{bi("OpenAI Academic Researcher plan, Claude Code, and all developers who have contributed to humanity’s knowledge base", "OpenAI Academic Researcher plan, Claude Code, 以及曾向人类知识库做出贡献的所有开发者")}</strong></p>'
        '</section></aside>'
        f'<main><div class="topline"><span>{bi("Models, methods and their evidence", "模型、方法及其证据")}</span><span>{bi("Inference notebook", "推断笔记本")}</span></div>'
    )
    page += design_html
    page += "".join(
        render_case(report, folder, index)
        for index, (report, folder) in enumerate(display_reports)
    )
    page += f'</main></div><noscript><p>JavaScript is disabled. All examples and guide panels are shown in reading order; expand source sections to read the full design.</p></noscript><script src="vendor/katex/katex.min.js"></script><script>{script}</script></body></html>'
    (directory / "index.html").write_text(page)
    # Rendering provenance is separate from the numerical run's original hashes.
    sources = [
        root / name
        for name in (
            "plot_results.py", "presentation.py", "case_content.py", "case_panels.py", "comparison_policy.py", "noise_scale_reference.py", "observation_count_reference.py", "repeated_panels.py", "repeated_summary.py", "plot_repeated.py", "symbols.py", "gallery.css", "gallery.js",
            "methodology.py", "methodology.html", "methodology.css", "methodology.js",
            "methodology_guide.py", "tris_presentation.py",
            "requirements-presentation.txt",
        )
    ]
    sources.extend(path for path in (root / "vendor").rglob("*") if path.is_file())
    (directory / "presentation.json").write_text(
        json.dumps(
            {
                "renderer_sha256": {
                    path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sources
                },
                "design_sources_sha256": design_sources,
                "comparison_inputs_sha256": comparison_inputs,
                "inputs": {
                    folder: {
                        name: hashlib.sha256(
                            (directory / folder / name).read_bytes()
                        ).hexdigest()
                        for name in ("result.json", "posterior.npz", "maps.npz", "predictions.npz", "manifest.json")
                        if (directory / folder / name).exists()
                    }
                    for _, folder in reports
                },
                "note": "Display only. Original inference results and their provenance are unchanged.",
            },
            indent=2,
        )
        + "\n"
    )
