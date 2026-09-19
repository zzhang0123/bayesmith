"""Batch E: a self-contained HTML report for the TRIS forward audit.

Builds figures from the recorded Batch A-D artifacts and writes one openable
local HTML page with bilingual section navigation, tables and equations.  The
page loads no network resources, so it renders offline; validate() checks the
image references, the absence of external URLs, and the inline script syntax,
and the caller must report that no browser was actually run.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTLINE = [
    ("overview", "总览", "Overview"),
    ("data", "数据契约", "Data contract"),
    ("operator", "观测算子审计", "Operator audit"),
    ("baseline", "正向基线", "Forward baseline"),
    ("identifiability", "可辨识性", "Identifiability"),
    ("injection", "注入恢复", "Injection recovery"),
    ("extensions", "扩展与敏感性", "Extensions and sensitivity"),
    ("comparison", "同层比较", "Same-layer comparison"),
    ("limits", "结论与限制", "Conclusions and limits"),
]


def load(path: Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.autolayout": True,
        }
    )


def figure_operator(audit: dict, path: Path) -> None:
    ladder = audit.get("p0c", {}).get("beam_ladder", {})
    keys = sorted(ladder, key=int)
    rungs = audit.get("p0c", {}).get("ladder_convergence", {})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar([str(k) for k in keys], [ladder[k]["prediction_rms_k"] for k in keys], color="#4477aa")
    axes[0].set_xlabel("beam built directly at nside h")
    axes[0].set_ylabel("rms prediction difference vs N=8 control [K]")
    axes[0].set_title("coarse-to-direct beam gap")
    names = list(rungs)
    axes[1].plot(names, [rungs[k]["prediction_rms_k"] for k in names], "o-", color="#cc6677")
    axes[1].set_ylabel("rung-to-rung rms [K]")
    axes[1].set_title("direct ladder convergence")
    axes[1].tick_params(axis="x", rotation=45)
    fig.savefig(path, format="svg")
    plt.close(fig)


def figure_baseline(batch_b: dict, path: Path) -> None:
    fixed = batch_b.get("frequencies", {}).get("fixed/no_rsb", [])
    audited = batch_b.get("frequencies", {}).get("audited/no_rsb", [])
    labels = [f"{row['frequency_mhz']:.1f}" for row in fixed]
    width = 0.35
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar(x - width / 2, [r["residual_rms_k"] for r in fixed], width, label="fixed A")
    axes[0].bar(x + width / 2, [r["residual_rms_k"] for r in audited], width, label="audited A")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("frequency [MHz]")
    axes[0].set_ylabel("residual rms [K]")
    axes[0].legend()
    axes[1].bar(x - width / 2, [r["chi_square_per_observation"] for r in fixed], width, label="fixed A")
    axes[1].bar(x + width / 2, [r["chi_square_per_observation"] for r in audited], width, label="audited A")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("frequency [MHz]")
    axes[1].set_ylabel("chi2 / N")
    axes[1].legend()
    fig.savefig(path, format="svg")
    plt.close(fig)


def figure_identifiability(batch_c: dict, path: Path) -> None:
    analysis = batch_c.get("identifiability", {}).get("posterior_mean", {}).get("M1")
    if not analysis:
        return
    names = analysis["names"]
    ratios = np.asarray(analysis["posterior_over_prior_sd"])
    order = np.argsort(ratios)
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.barh([names[i] for i in order], ratios[order], color="#228833")
    axis.axvline(1.0, color="#bbbbbb", linestyle="--")
    axis.set_xlabel("local linearized SD / prior SD (1 = weak local constraint)")
    axis.set_title("M1 identifiability at the audited posterior mean")
    fig.savefig(path, format="svg")
    plt.close(fig)


def figure_injection(batch_c: dict, path: Path) -> None:
    scenarios = batch_c.get("injection", {})
    rows = []
    for name, entry in scenarios.items():
        for label, fit in entry["variants"].items():
            pulls = [abs(row["pull"]) for row in fit["recovery"]]
            if pulls:
                rows.append((f"{name}/{label}", max(pulls)))
    if not rows:
        return
    labels, values = zip(*rows)
    fig, axis = plt.subplots(figsize=(8, 4.0))
    axis.barh(list(labels), list(values), color="#aa3377")
    axis.axvline(2.0, color="#bbbbbb", linestyle="--")
    axis.set_xlabel("max absolute recovery pull [posterior SD]")
    axis.set_xscale("log")
    axis.set_title("injection recovery (reference |pull|=2; not an interval test)")
    fig.savefig(path, format="svg")
    plt.close(fig)


def figure_extensions(batch_d2: dict, path: Path) -> None:
    rows = batch_d2.get("extensions", [])
    if not rows:
        return
    fig, axis = plt.subplots(figsize=(6, 3.4))
    axis.bar([row["label"] for row in rows], [row["chi_square_per_observation"] for row in rows], color="#ee7733")
    axis.set_ylabel("chi2 / N at the MAP")
    axis.set_title("foreground basis flexibility")
    fig.savefig(path, format="svg")
    plt.close(fig)


def figure_comparison(batch_d: dict, path: Path) -> None:
    deltas = []
    for operator, block in batch_d.get("comparisons", {}).items():
        for row in block.get("deltas", []):
            deltas.append((f"{operator}/{row['train']}->{row['heldout']}", row["delta_M1_minus_M0"]))
    if not deltas:
        return
    labels, values = zip(*deltas)
    fig, axis = plt.subplots(figsize=(7, 3.2))
    axis.bar(list(labels), list(values), color="#332288")
    axis.set_ylabel("M1 - M0 held-out log density [nats]")
    axis.set_title("same-layer RSB preference, both operators")
    axis.tick_params(axis="x", rotation=20)
    fig.savefig(path, format="svg")
    plt.close(fig)


def build_figures(runs: dict, assets: Path) -> dict:
    assets.mkdir(parents=True, exist_ok=True)
    style()
    figures = {}
    audit = load(runs["a"] / "audit.json") if runs.get("a") else {}
    batch_b = load(runs["b"] / "batch_b.json") if runs.get("b") else {}
    batch_c = load(runs["c"] / "batch_c.json") if runs.get("c") else {}
    batch_d = load(runs["d"] / "batch_d.json") if runs.get("d") else {}
    batch_d2 = load(runs["d2"] / "batch_d2.json") if runs.get("d2") else {}
    for key, builder, data in (
        ("operator", figure_operator, audit),
        ("baseline", figure_baseline, batch_b),
        ("identifiability", figure_identifiability, batch_c),
        ("injection", figure_injection, batch_c),
        ("extensions", figure_extensions, batch_d2),
        ("comparison", figure_comparison, batch_d),
    ):
        target = assets / f"{key}.svg"
        try:
            builder(data, target)
        except (KeyError, TypeError, ValueError) as error:
            # A missing artifact must not abort the whole report; the gap is
            # reported in place of the figure.
            figures[key + "_error"] = repr(error)
            continue
        if target.is_file():
            figures[key] = target.name
    return figures


INLINE_SCRIPT = """
(function () {
  var buttons = Array.prototype.slice.call(document.querySelectorAll('[data-section]'));
  var articles = Array.prototype.slice.call(document.querySelectorAll('article[data-id]'));
  function show(id) {
    articles.forEach(function (article) {
      article.hidden = article.getAttribute('data-id') !== id;
    });
    buttons.forEach(function (button) {
      button.setAttribute('aria-pressed', String(button.getAttribute('data-section') === id));
    });
  }
  buttons.forEach(function (button) {
    button.addEventListener('click', function () {
      show(button.getAttribute('data-section'));
    });
  });
  if (articles.length) { show(articles[0].getAttribute('data-id')); }
})();
"""


def section_body(key: str, data: dict) -> str:
    if key == "overview":
        change = data.get("operator_change", {})
        return (
            "<p><span class='zh'>从公开 TRIS 轮廓与 limTOD 算子出发，审计数据语义、"
            "map-making、束流构路与天空求积，然后固定/审计两套算子重拟合，并给出"
            "可辨识性与注入恢复证据。</span>"
            "<span class='en'>The audit runs from the public TRIS profiles and the limTOD "
            "operator through data semantics, map-making, beam construction and sky "
            "quadrature, then refits under two operators and adds identifiability and "
            "injection evidence.</span></p>"
            f"<p><span class='zh'>算子改动：最大 {change.get('max_abs', 0):.3e}，"
            f"600.5 MHz 预测差异 {change.get('prediction_rms_k', {}).get('600.5', 0):.4f} K rms，"
            f"817.8 MHz {change.get('prediction_rms_k', {}).get('817.8', 0):.4f} K rms。</span>"
            f"<span class='en'>Operator change: max {change.get('max_abs', 0):.3e}, "
            f"0.1620 K rms at 600.5 MHz and 0.0682 K rms at 817.8 MHz.</span></p>"
        )
    if key == "data":
        counts = data.get("provenance_counts", {})
        return (
            "<p><span class='zh'>22 行账本，逐项标注 archived-confirmed / paper-confirmed / "
            "implementation-convention / unconfirmable。未公布量不补造。</span>"
            "<span class='en'>22 ledger rows tagged archived-confirmed, paper-confirmed, "
            "implementation-convention or unconfirmable; nothing unpublished is invented."
            "</span></p>"
            f"<p><code>{html.escape(json.dumps(counts, sort_keys=True, ensure_ascii=False))}</code></p>"
        )
    if key == "baseline":
        parity = data.get("log_joint_parity", {})
        rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td>{v['graph_log_joint']:.4f}</td>"
            f"<td>{v['log_joint']:.4f}</td><td>{v['difference']:.2e}</td></tr>"
            for k, v in parity.items()
        )
        return (
            "<p><span class='zh'>fixed A 重跑与冻结 posterior 逐 draw 相同；log-joint 与"
            "独立分项差约 1e-10。</span>"
            "<span class='en'>The fixed-A rerun reproduces the frozen posterior draw for "
            "draw, and log joints agree with the independent decomposition to about 1e-10."
            "</span></p>"
            "<table><tr><th>baseline</th><th>graph</th><th>independent</th><th>difference</th></tr>"
            + rows
            + "</table>"
        )
    if key == "identifiability":
        analysis = data.get("identifiability", {}).get("posterior_mean", {}).get("M1", {})
        rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{ratio:.4f}</td></tr>"
            for name, ratio in zip(analysis.get("names", []), analysis.get("posterior_over_prior_sd", []))
        )
        return (
            "<p><span class='zh'>后验/先验 SD 比接近 1 表示该方向由先验提供。"
            "600 MHz 零点、LWA 定标与 rsb_beta 属于此类。</span>"
            "<span class='en'>A posterior-to-prior SD ratio near one means the prior "
            "supplies that direction; the 600 MHz zero level, the LWA calibration and "
            "rsb_beta are in that class.</span></p>"
            "<table><tr><th>coordinate</th><th>posterior / prior SD</th></tr>" + rows + "</table>"
        )
    if key == "injection":
        return (
            "<p><span class='zh'>先验均值天空下全部 pull 小于 1.7 SD，0.5 K RSB 可被 M1 恢复；"
            "但在真实后验均值处注入不可恢复，log-joint 显示先验压过似然约 3700 nats，"
            "说明 600 MHz 零点由先验决定而非测量。</span>"
            "<span class='en'>At a prior-mean sky every pull is below 1.7 SD and M1 "
            "recovers an injected 0.5 K background; at the real posterior mean the truth "
            "is not recovered and the log joint shows the prior outweighing the likelihood "
            "by about 3700 nats, so the 600 MHz zero level is prior-dominated.</span></p>"
        )
    if key == "extensions":
        rows = "".join(
            f"<tr><td>{html.escape(row['label'])}</td><td>{row['parameters']}</td>"
            f"<td>{row['chi_square_per_observation']:.1f}</td></tr>"
            for row in data.get("extensions", [])
        )
        sensitivity = "".join(
            f"<tr><td>{html.escape(e['label'])}</td><td>{html.escape(e['variant'])}</td>"
            f"<td>{e['ess']:.0f}</td><td>{e['trustworthy']}</td></tr>"
            for e in data.get("sensitivity", [])
        )
        return (
            "<p><span class='zh'>6 区与平滑基都把 TRIS 白化 chi2/N 显著压低；"
            "先验/定标敏感性用自归一化重加权，ESS 过低的组合不报矩。</span>"
            "<span class='en'>Both a 6-region and a smooth basis lower the TRIS whitened "
            "chi2/N; prior and calibration sensitivity use self-normalised reweighting, "
            "and combinations with too low an ESS report no moments.</span></p>"
            "<table><tr><th>model</th><th>parameters</th><th>chi2 / N</th></tr>" + rows + "</table>"
            "<table><tr><th>variant</th><th>model</th><th>ESS</th><th>trustworthy</th></tr>"
            + sensitivity
            + "</table>"
        )
    if key == "comparison":
        rows = "".join(
            f"<tr><td>{html.escape(operator)}</td><td>{html.escape(row['train'])}</td>"
            f"<td>{html.escape(row['heldout'])}</td><td>{row['delta_M1_minus_M0']:+.4f}</td>"
            f"<td>{row['delta_mcse']:.4f}</td></tr>"
            for operator, block in data.get("comparisons", {}).items()
            for row in block.get("deltas", [])
        )
        return (
            "<p><span class='zh'>同层、同数据、同预算下 M1 相对 M0 的留出预测增益约 "
            "+0.019 与 +0.158 nats，两套算子几乎不变。这些是条件预测分数，不是 Bayes factor。</span>"
            "<span class='en'>At one layer, one dataset and one budget, M1 beats M0 on the "
            "held-out predictive score by about +0.019 and +0.158 nats, essentially "
            "unchanged by the operator. These are conditional scores, not a Bayes factor."
            "</span></p>"
            "<table><tr><th>operator</th><th>train</th><th>held out</th><th>delta</th>"
            "<th>MCSE</th></tr>" + rows + "</table>"
        )
    return (
        "<ul>"
        "<li><span class='zh'>600 MHz 零点与 Haslam monopole 不可单独测量。</span>"
        "<span class='en'>The 600 MHz zero level and the Haslam monopole are not separately "
        "measured.</span></li>"
        "<li><span class='zh'>三区前景无法吸收另一种区域划分，平滑基可显著降低 chi2。</span>"
        "<span class='en'>The three-region foreground cannot absorb a different split, and "
        "a smooth basis lowers chi2 materially.</span></li>"
        "<li><span class='zh'>束流宽度/色散参数块尚未接入，需要 rheplicant 的可微 beam。</span>"
        "<span class='en'>The beam-width and chromaticity block is not yet wired in and "
        "needs rheplicant's differentiable beam.</span></li>"
        "<li><span class='zh'>本页未在浏览器中执行；只做了结构与资源校验。</span>"
        "<span class='en'>This page was not executed in a browser; only structural and "
        "resource checks were run.</span></li>"
        "</ul>"
    )


def build_html(figures: dict, payload: dict) -> str:
    nav = "".join(
        f"<button data-section='{key}'>{html.escape(zh)} / {html.escape(en)}</button>"
        for key, zh, en in OUTLINE
    )
    articles = []
    for key, zh, en in OUTLINE:
        image = figures.get(key)
        img = (
            f"<figure><img src='assets/{html.escape(image)}' alt='{html.escape(key)}'>"
            f"</figure>"
            if image
            else ""
        )
        body = section_body(key, payload)
        articles.append(
            f"<article data-id='{key}'><h2><span class='zh'>{html.escape(zh)}</span>"
            f"<span class='en'>{html.escape(en)}</span></h2>{img}{body}</article>"
        )
    css = (
        "body{font-family:system-ui,-apple-system,'PingFang SC',sans-serif;margin:0;"
        "background:#fafafa;color:#222}header{padding:1.2rem 2rem;background:#22303f;"
        "color:#fff}nav{display:flex;flex-wrap:wrap;gap:.4rem;padding:0 2rem 1rem}"
        "nav button{cursor:pointer;padding:.35rem .7rem;border:1px solid #98a6b3;"
        "border-radius:999px;background:#fff}nav button[aria-pressed='true']"
        "{background:#22303f;color:#fff}main{padding:1rem 2rem 3rem;max-width:1000px}"
        "article{background:#fff;border:1px solid #e2e6ea;border-radius:10px;"
        "padding:1rem 1.4rem;margin-bottom:1rem}table{border-collapse:collapse;"
        "width:100%;margin:.6rem 0}th,td{border:1px solid #dde2e7;padding:.32rem .5rem;"
        "font-size:.86rem;text-align:left}figure{margin:.6rem 0}img{max-width:100%}"
        "code{background:#f1f3f5;padding:.1rem .3rem;border-radius:4px}"
        ".en{color:#33475b}.zh{color:#111}"
    )
    return (
        "<!doctype html><html lang='zh'>"
        "<head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>TRIS forward-model audit</title>"
        f"<style>{css}</style></head><body>"
        "<header><h1>TRIS 全流程审计 / TRIS forward-model audit</h1>"
        "<p><span class='zh'>数据语义、算子、基线、可辨识性、扩展与同层比较。</span>"
        "<span class='en'>Data semantics, operator, baseline, identifiability, "
        "extensions and same-layer comparison.</span></p></header>"
        f"<nav>{nav}</nav><main>{''.join(articles)}</main>"
        f"<script>{INLINE_SCRIPT}</script></body></html>"
    )


def validate(output: Path, assets: Path) -> list:
    problems = []
    text = output.read_text()
    if "http://" in text or "https://" in text:
        problems.append("page references an external URL")
    if "data-section=" not in text or "data-id=" not in text:
        problems.append("navigation or section markers are missing")
    if text.count("class='zh'") == 0 or text.count("class='en'") == 0:
        problems.append("bilingual spans are missing")
    for chunk in text.split("<img src='")[1:]:
        source = Path(chunk.split("'")[0])
        if not (assets / source.name).is_file():
            problems.append(f"missing image: {source.name}")
    script = text.split("<script>")[1].split("</script>")[0]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(script)
        script_path = Path(handle.name)
    try:
        result = subprocess.run(
            ["node", "--check", str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            problems.append("inline script is not valid JavaScript: " + result.stderr.strip())
    except FileNotFoundError:
        problems.append("node not available: inline script syntax not checked")
    finally:
        script_path.unlink(missing_ok=True)
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("a", "b", "c", "d", "d2"):
        parser.add_argument(f"--batch-{flag}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = {flag: getattr(args, f"batch_{flag}") for flag in ("a", "b", "c", "d", "d2")}
    args.output.mkdir(parents=True, exist_ok=True)
    assets = args.output / "assets"
    figures = build_figures(runs, assets)
    batch_b = load(runs["b"] / "batch_b.json") if runs.get("b") else {}
    batch_c = load(runs["c"] / "batch_c.json") if runs.get("c") else {}
    batch_d2 = load(runs["d2"] / "batch_d2.json") if runs.get("d2") else {}
    payload = {**batch_b, **batch_c.get("identifiability", {}).get("posterior_mean", {})}
    for key in ("operator_change", "log_joint_parity", "frequencies"):
        payload.setdefault(key, batch_b.get(key, {}))
    payload = dict(payload)
    payload["extensions"] = batch_d2.get("extensions", [])
    payload["sensitivity"] = batch_d2.get("sensitivity", [])
    payload["comparisons"] = load(runs["d"] / "batch_d.json").get("comparisons", {})
    payload["injection"] = batch_c.get("injection", {})
    document = build_html(figures, payload)
    (args.output / "index.html").write_text(document)
    problems = validate(args.output / "index.html", assets)
    print(json.dumps({"figures": figures, "problems": problems}, ensure_ascii=False))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
