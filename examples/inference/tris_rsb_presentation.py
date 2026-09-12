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
    status_reason = report.get("comparison_status")
    status = "Validated comparison / 已验证的比较" if available else "Blocked: chains did not converge / 阻塞：链未收敛"
    finding = (
        "Directional held-out prediction is available only after converged refits."
        if available else
        "Both full-data chains failed the registered convergence criteria; posterior intervals and cross-survey scores are intentionally withheld."
        if status_reason == "blocked_nonconverged_chains" else
        "Full-data chains passed, but the directional held-out refits and scores have not been produced; posterior intervals and cross-survey scores are intentionally withheld."
    )
    controls = "".join(
        f'<button type="button" data-step="{key}" aria-pressed="false">{label}</button>'
        for key, label in (
            ("model", "Model / 模型"), ("methods", "Methods & blocks / 方法与参数块"),
            ("diagnostics", "Diagnostics & priors / 诊断与先验"),
            ("sampling", "Sampling / 采样"), ("recovery", "Findings / 发现与解释"),
        )
    )
    return (
        '<article class="case methodology tris-case" id="tris_haslam_rsb_comparison" data-case="tris_haslam_rsb_comparison" data-case-kind="real_observations">'
        '<header class="case-heading"><span class="status-tag">Real observations · joint analysis / 真实观测 · 联合分析</span><p class="question">' + _esc(status) + '</p>'
        '<h1>TRIS + Haslam + RSB</h1><p class="question">Can an isotropic spectrum improve cross-survey prediction?</p></header>'
        '<div class="steps" role="tablist" aria-label="RSB analysis chapters">' + controls + '</div>'
        '<section class="stage" id="tris_haslam_rsb_comparison-model" data-stage="model"><h2>Model / 模型</h2><p>TRIS maps retain their saved beam/map response and noise whitening. ARCADE 2 and LWA contribute published background temperatures with survey-wide calibration terms.</p>'
        '<p>ARCADE 2 spans 3.20–10.49 GHz and LWA spans 40–80 MHz. The 408-MHz literature row is excluded so Haslam is not counted twice.</p></section>'
        '<section class="stage" id="tris_haslam_rsb_comparison-methods" data-stage="methods"><h2>Methods & parameter blocks / 方法与参数块</h2><p>M0 fits the TRIS–Haslam sky with no isotropic excess. M1 adds a positive RSB amplitude at 1 GHz and a spectral index, jointly with the three regional Haslam amplitudes and indices, two TRIS zero corrections, a Haslam monopole correction, and one calibration standard per external survey.</p>'
        '<p>The comparison is directional: fit one external survey with TRIS, then predict the other. It is not a Bayes-factor calculation.</p></section>'
        '<section class="stage" id="tris_haslam_rsb_comparison-diagnostics" data-stage="diagnostics"><h2>Diagnostics & priors / 诊断与先验</h2><p>RSB amplitude has a Uniform(0, 5 K) prior at 1 GHz and its spectral index has Uniform(−4, −1.5). The Haslam monopole has Normal(0, 3 K); regional sky and zero-level priors match the recorded TRIS analysis.</p>'
        '<p>Draws with a non-positive 408-MHz Galactic template have zero posterior density. Quoted per-row variances are retained while calibration is correlated within LWA and ARCADE 2.</p></section>'
        '<section class="stage" id="tris_haslam_rsb_comparison-sampling" data-stage="sampling"><h2>Posterior sampling / 后验采样</h2><p>Each variant uses two sequential NUTS chains and the registered R-hat, ESS, and divergence checks. The saved diagnostic artifacts are linked below.</p>'
        '<p>Convergence is a requirement for both posterior summaries and held-out prediction; completing the requested number of draws alone is not sufficient.</p></section>'
        '<section class="stage" id="tris_haslam_rsb_comparison-recovery" data-stage="recovery"><h2>Findings / 发现与解释</h2><p>' + _esc(finding) + '</p><p>'
        + _esc(report["comparison_policy"])
        + '</p><p>' + _esc(report["note"]) + '</p>'
        + '<div class="table-wrap"><table><thead><tr><th>RSB parameter</th><th>Mean</th><th>95% interval</th></tr></thead><tbody>' + rows + '</tbody></table></div></section>'
        '<div class="step-footer"><button type="button" class="previous">Previous / 上一步</button><span class="step-position" aria-live="polite"></span><button type="button" class="next">Next / 下一步</button></div>'
        '<footer class="artifacts"><a href="' + quote(folder, safe="") + '/result.json">Comparison artifact</a><span>Input fingerprint: ' + _esc(report["data_manifest"]["common_input_sha256"]) + '</span></footer></article>'
    )
