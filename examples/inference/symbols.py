"""Shared mathematical symbols for DAG nodes, parameter blocks and tables."""

from __future__ import annotations

import html

SYMBOLS = {
    "alpha": (r"\alpha", "α"), "tail_index": (r"\alpha_j-1", "αⱼ − 1"),
    "x": ("x", "x"), "X": ("X", "X"), "time": ("t", "t"),
    "channel": ("j", "j"), "group_index": ("g_i", "gᵢ"),
    "beta": (r"\beta", "β"), "amplitude": ("a", "a"),
    "offset": ("b", "b"), "rate": ("r", "r"),
    "population": ("m", "m"), "groups": (r"\eta", "η"),
    "p_g": ("p_g", "p_g"), "p_n": ("p_n", "p_n"),
    "U": ("U", "U"), "A": ("A", "A"),
    "mu": (r"\mu", "μ"), "obs": ("y", "y"),
    "location": (r"\mu", "μ"), "logits": (r"X\beta", "Xβ"),
    "basis": ("e^{-rt}", "exp(−rt)"),
    "power_amplitude": ("a", "a"), "frequencies": ("k", "k"),
    "power": ("P_k(a)", "Pₖ(a)"), "instance": ("s", "s"),
    "response": ("R", "R"), "response_signal": ("Rs", "Rs"),
    "nonlinear": (r"\theta=(c,w)", "θ = (c, w)"),
    "nonlinear_shape": (r"h(x;\theta)", "h(x; θ)"),
    "combined": (r"Rs+h(x;\theta)", "Rs + h(x; θ)"),
    "background": ("b", "b"), "background_design": ("B", "B"),
    "gain_design": ("U", "U"), "additive_signal": ("v", "v"),
    "sigma_w": (r"\sigma_w", "σ_w"),
}


def symbol(case, name):
    if case in {"tris_haslam", "tris_haslam_no_rsb", "tris_haslam_rsb"}:
        return {
            "amplitude": (r"a_r", "aᵣ"),
            "beta": (r"\beta_r", "βᵣ"),
            "zero_standard": (r"z_\nu", "zᵥ"),
            "haslam_monopole_K": (r"z_H", "zᴴ"),
            "calibration_standard": (r"c_s", "cₛ"),
            "rsb_amplitude": (r"A_{\rm RSB}", "Aᴿˢᴮ"),
            "rsb_beta": (r"\beta_{\rm RSB}", "βᴿˢᴮ"),
        }.get(name, SYMBOLS.get(name, (name, name)))
    if case == "power_law" and name == "amplitude":
        return ("A", "A")
    if case == "power_law" and name == "basis":
        return (r"x^{\alpha_j}", "x^αⱼ")
    if name == "gain":
        return ("g", "g") if case == "composed_process" else ("e^{Up_g}", "exp(U p_g)")
    if name == "signal":
        return ("Ap_n", "A p_n")
    return SYMBOLS.get(name, (r"\mathrm{" + name.replace("_", r"\_") + "}", name))


def inline_symbol(case, name):
    tex, plain = symbol(case, name)
    return f'<span class="parameter-symbol" data-tex="{html.escape(tex, quote=True)}">{html.escape(plain)}</span>'


def parameter_symbols(case, names, *, show_names=False):
    return '<span class="parameter-symbols">' + "".join(
        '<span class="symbol-item">' + inline_symbol(case, name)
        + (f'<code>{html.escape(name)}</code>' if show_names else "") + '</span>'
        for name in names
    ) + '</span>'
