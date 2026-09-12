"""Bilingual saved-artifact presentation for the RSB comparison."""

from __future__ import annotations

import html
from urllib.parse import quote


def _esc(value):
    return html.escape(str(value), quote=True)


def render_case(report, folder, index=0):
    """Render saved comparison evidence; this function never runs inference."""
    models = report["models"]
    available = report.get("comparison_status") == "available"
    parameter_rows = models.get("rsb", {}).get("parameters", []) if available else []
    rows = "".join(
        f"<tr><td>{_esc(row.get('name', '—'))}</td><td>{_esc(row.get('mean', '—'))}</td><td>{_esc(row.get('lower', '—'))}–{_esc(row.get('upper', '—'))}</td></tr>"
        for row in parameter_rows
    ) or "<tr><td colspan=\"3\">Saved posterior summaries are required for the final comparison.</td></tr>"
    status = "Validated comparison / 已验证的比较" if available else "Blocked: chains did not converge / 阻塞：链未收敛"
    status_reason = report.get("comparison_status")
    finding = (
        "Directional held-out prediction is available only after converged refits."
        if available else
        "Both full-data chains failed the registered convergence criteria; posterior intervals and cross-survey scores are intentionally withheld."
        if status_reason == "blocked_nonconverged_chains" else
        "Full-data chains passed, but the directional held-out refits and scores have not been produced; posterior intervals and cross-survey scores are intentionally withheld."
    )
    return (
        '<article class="case methodology tris-case" id="tris_haslam_rsb_comparison" data-case="tris_haslam_rsb_comparison" data-case-kind="real_observations">'
        '<header class="case-heading"><span class="status-tag">Real observations · joint analysis / 真实观测 · 联合分析</span><p class="question">' + _esc(status) + '</p>'
        '<h1>TRIS + Haslam + RSB</h1><p class="question">Can an isotropic spectrum improve cross-survey prediction?</p></header>'
        '<section class="stage"><h2>Model / 模型</h2><p>TRIS maps use their saved full response and noise whitening. ARCADE 2 and LWA contribute published background temperatures with survey-wide calibration terms.</p>'
        '<p>ARCADE 2: 3.20–10.49 GHz; LWA: 40–80 MHz. The 408-MHz literature row is excluded so Haslam is not counted twice.</p></section>'
        '<section class="stage"><h2>Methods, diagnostics & priors / 方法、诊断与先验</h2><p>The RSB model adds a positive 1-GHz amplitude, a finite spectral-index prior, and a Haslam monopole correction. Draws with a non-positive 408-MHz Galactic template have zero posterior density.</p>'
        '<p>Quoted per-row variances are preserved while calibration is correlated within each survey.</p></section>'
        '<section class="stage"><h2>Findings / 发现与解释</h2><p>' + _esc(finding) + '</p><p>'
        + _esc(report["comparison_policy"])
        + '</p><p>' + _esc(report["note"]) + '</p>'
        + '<table><thead><tr><th>RSB parameter</th><th>Mean</th><th>95% interval</th></tr></thead><tbody>' + rows + '</tbody></table></section>'
        '<footer class="artifacts"><a href="' + quote(folder, safe="") + '/result.json">Comparison artifact</a><span>Input fingerprint: ' + _esc(report["data_manifest"]["common_input_sha256"]) + '</span></footer></article>'
    )
