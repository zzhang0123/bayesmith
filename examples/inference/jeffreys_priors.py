"""Exact, bounded Jeffreys priors for the registered small inference demos.

These are model-specific formulas, not a determinant estimated at probe points.
The full observed likelihood defines the nonhierarchical priors. For the hierarchy
only the population hyperparameter receives Jeffreys' prior, computed from the
observed likelihood after integrating the Gaussian group effects. Their original
conditional law remains part of the model.

Rank on the original support interiors follows analytically:

* Power-law regression: known Gaussian white noise, positive amplitudes and
  distinct positive inputs give full-rank amplitude/index Fisher blocks.
* Decay: for r>0, {exp(-rt), t exp(-rt), 1} is an extended Chebyshev system;
  three distinct times identify each channel. The amplitude supplies one extra
  multiplicative factor, with zero density at amplitude=0.
* Logistic: the rectangular covariate grid identifies {1,x1,x2,x1*x2}; every
  Bernoulli variance is strictly positive for finite coefficients.
* Multiplicative: multiplying H=[U,A/(An)] by positive s=An gives columns in
  {sin(wk), k sin(wk), cos(wk), k cos(wk), 1, k}. These six sampled sequences
  form a confluent Vandermonde system with distinct roots 1, exp(+iw), exp(-iw)
  of multiplicity two, where w=4*pi/127. They are independent; n0,n1>0 then
  implies that all four original coefficients in a null combination vanish.
* Hierarchical: independent group averages have variance tau^2+sigma^2/40,
  so I(population)=8/(tau^2+sigma^2/40)>0 and is constant.

Power-law regression uses a weighted log-input variance in log space; the other
multivariate formulas use thin QR. Neither adds a ridge or floors an eigenvalue.
The declaration retains canonical
bounded Uniform root distributions as constant density and support carriers;
one graph-level factor changes their product to the bounded Jeffreys shape,
up to a global constant. This is posterior sampling, not an evidence estimate.
This helper refuses different operators/designs instead of reusing the formulas
for a model for which the above rank arguments do not apply.
"""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith import trace
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic

from .priors import TOP_LEVEL_BOUNDS

SUPPORTED_CASES = ("power_law", "exponential_decay", "bernoulli", "multiplicative_noise", "hierarchical")

_FORMULAS = {
    "power_law": (
        "joint_observed_likelihood",
        "pi(A,alpha) ∝ product_j A_j sqrt(S0_j S2_j - S1_j²) / sigma²; S_k=Σ_i x_i^(2 alpha_j) (log x_i)^k",
        ("All four amplitudes and indices jointly, including their cross terms. "
         "Independent additive Gaussian noise has known sigma=0.2. Positive amplitude "
         "bounds and distinct positive inputs give full rank throughout the prior box."),
    ),
    "exponential_decay": (
        "joint_observed_likelihood",
        ("pi(a,r,b) ∝ product_j |a_j| sqrt(det(B_jᵀ B_j)) / sigma³; "
         "B_j=[exp(-r_j t), -t exp(-r_j t), 1], sigma=0.08"),
        ("All six amplitude, rate and offset coordinates; the channel product "
         "is the determinant of the full block-diagonal Fisher matrix."),
    ),
    "bernoulli": (
        "joint_observed_likelihood",
        "pi(beta) ∝ sqrt(det(Xᵀ diag(p_i(1-p_i)) X)); p_i=sigmoid((X beta)_i)",
        "All four logits coefficients jointly; the expectation is over binary observations.",
    ),
    "multiplicative_noise": (
        "joint_observed_likelihood",
        ("pi(p_g,p_n) ∝ sqrt(det((f⁻²+2) HᵀH)); "
         "H=[U, A / (A p_n)[:,None]], f=0.001"),
        ("All four coordinates jointly, including gain/signal cross terms and "
         "the information from parameter-dependent variance."),
    ),
    "hierarchical": (
        "marginal_observed_likelihood_for_population",
        "I(population)=8/(0.6²+0.25²/40); pi(population) ∝ 1 on [-3,3]",
        ("Integrate the Gaussian group effects to obtain the population information. "
         "Keep groups | population ~ Normal(population, 0.6²) unchanged; "
         "this is not a joint Jeffreys density over population and realized groups."),
    ),
}

_LATEX = {
    "power_law": r"\pi_J(A,\alpha)\propto\prod_{j=1}^2\frac{A_j}{\sigma^2}\sqrt{S_{0j}S_{2j}-S_{1j}^2},\quad S_{kj}=\sum_i x_i^{2\alpha_j}(\log x_i)^k",
    "exponential_decay": r"\pi(a,r,b)\propto\prod_{j=1}^{2}|a_j|\sqrt{\det(B_j^T B_j)}/\sigma^3,\quad B_j=[e^{-r_jt},-t e^{-r_jt},1]",
    "bernoulli": r"\pi(\beta)\propto\sqrt{\det[X^T\operatorname{diag}\{p_i(1-p_i)\}X]},\quad p_i=\operatorname{sigmoid}((X\beta)_i)",
    "multiplicative_noise": r"\pi(p_g,p_n)\propto\sqrt{\det[(f^{-2}+2)H^T H]},\quad H=[U,\operatorname{diag}\{(Ap_n)^{-1}\}A]",
    "hierarchical": r"I(m)=\frac{8}{0.6^2+0.25^2/40},\quad\pi(m)\propto 1\quad(-3\leq m\leq3)",
}

UNSUPPORTED_VARIANTS = {
    "composed_process": {
        "kind": "jeffreys_prior",
        "label": "with Jeffreys prior",
        "baseline_case": "composed_process",
        "status": "not_constructed",
        "reason": "marginal_fisher_unresolved",
        "information_scope": "marginal_observed_likelihood_for_root_parameters",
        "formula": None,
        "explanation": "The observed likelihood must integrate the latent Gaussian field "
        "through a nonlinear mean and parameter-dependent noise. An exact marginal Fisher "
        "formula is not available here. Products of conditional block determinants would "
        "define a different prior and could change the field's conditional generative law. "
        "No Jeffreys posterior samples were constructed.",
    },
}


def metadata_for(case):
    """Serializable interpretation of the prior actually used by a counterpart."""
    if case in UNSUPPORTED_VARIANTS:
        return deepcopy(UNSUPPORTED_VARIANTS[case])
    scope, formula, note = _FORMULAS[case]
    return {
        "kind": "jeffreys_prior", "label": "with Jeffreys prior",
        "baseline_case": case, "status": "implemented",
        "coordinates": list(TOP_LEVEL_BOUNDS[case]),
        "information_scope": scope, "formula": formula, "formula_latex": _LATEX[case],
        "support": deepcopy(TOP_LEVEL_BOUNDS[case]),
        "normalization": "One global normalization constant is omitted for posterior sampling. "
        "Bounded Uniform root sites contribute only constant density inside this support.",
        "recovery_scale": "Unchanged baseline bounded-Uniform prior standard deviations; "
        "these are reference scales, not the marginal SDs of the Jeffreys prior.",
        "equivalent_to_baseline": case == "hierarchical",
        "conditional_law_note": note,
        "regularization": "None: no ridge, eigenvalue floor or product of conditional priors.",
    }


def _log_volume(design):
    _, triangular = jnp.linalg.qr(design, mode="reduced")
    return jnp.log(jnp.abs(jnp.diag(triangular))).sum()


class AnalyticDemoJeffreys(eqx.Module):
    """One joint-prior factor, restricted to a verified registered demo graph."""

    case: str = eqx.field(static=True)
    over: tuple[str, ...] = eqx.field(static=True)
    inputs: tuple[jax.Array, ...]
    bounds: tuple[tuple[jax.Array, jax.Array], ...]

    def log_density(self, graph, values):
        if self.case == "power_law":
            logx, sigma = self.inputs
            exponent = 2 * values["alpha"][:, None] * logx
            total = jax.scipy.special.logsumexp(exponent, axis=-1)
            weights = jnp.exp(exponent - total[:, None])
            center = weights @ logx
            variance = jnp.sum(weights * (logx-center[:, None])**2, axis=-1)
            result = jnp.sum(jnp.log(jnp.abs(values["amplitude"])) - 2*jnp.log(sigma)
                             + total + 0.5*jnp.log(variance))
        elif self.case == "exponential_decay":
            time, sigma = self.inputs
            densities = []
            for channel in range(2):
                exponential = jnp.exp(-values["rate"][channel] * time)
                design = jnp.column_stack((exponential, -time * exponential,
                                           jnp.ones_like(time))) / sigma
                densities.append(jnp.log(jnp.abs(values["amplitude"][channel]))
                                 + _log_volume(design))
            result = sum(densities)
        elif self.case == "bernoulli":
            (X,) = self.inputs
            logits = X @ values["beta"]
            weights = jnp.exp(-0.5 * (jax.nn.softplus(logits) + jax.nn.softplus(-logits)))
            result = _log_volume(X * weights[:, None])
        elif self.case == "multiplicative_noise":
            U, A, fraction = self.inputs
            H = jnp.concatenate((U, A / (A @ values["p_n"])[:, None]), axis=1)
            result = _log_volume(H) + 0.5 * H.shape[1] * jnp.log(fraction**-2 + 2)
        else:
            (information,) = self.inputs
            result = 0.5 * jnp.log(information)
        inside = jnp.asarray(True)
        for name, (low, high) in zip(self.over, self.bounds, strict=True):
            inside = inside & jnp.all((values[name] >= low) & (values[name] <= high))
        return jnp.where(inside, result, -jnp.inf)


def _registered_graph(case, graph):
    """Verify the formulas' operator, design, support and observation premises."""
    module = import_module(f"examples.inference.{case}")
    if case == "power_law":
        arguments = module.design()
        x = arguments[0][:module.OBSERVATIONS]
        sigma = jnp.asarray(module.gaussian_observation(0.).scale)
        if (np.any(np.asarray(x) <= 0) or np.ptp(np.asarray(x)) == 0 or float(sigma) <= 0):
            raise ValueError("The regression Jeffreys prior requires distinct positive x and positive noise SD.")
        inputs = (jnp.log(x), sigma)
    elif case == "exponential_decay":
        time = jnp.tile(jnp.linspace(0.0, 5.0, 120), 2)
        arguments = (time, jnp.repeat(jnp.arange(2), 120))
        inputs = (time[:120], jnp.asarray(module.gaussian_observation(0.0).scale))
    elif case == "bernoulli":
        axis = jnp.linspace(-2.0, 2.0, 20)
        x1, x2 = jnp.meshgrid(axis, axis, indexing="xy")
        X = module.design_matrix(x1.ravel(), x2.ravel())
        arguments, inputs = (X,), (X,)
    elif case == "multiplicative_noise":
        U, A = module.design_matrices(jnp.linspace(0.0, 1.0, 128))
        arguments = (U, A)
        inputs = (U, A, jnp.asarray(module.relative_gaussian(1.0).scale))
    else:
        arguments = (jnp.repeat(jnp.arange(8), 40),)
        sigma = jnp.asarray(module.gaussian_observation(0.0).scale)
        tau = jnp.sqrt(module.group_prior(0.0).variance[0])
        inputs = (8 / (tau**2 + sigma**2 / 40),)
    expected = trace(module.model, *arguments, graph.node("obs").observed)
    if len(graph.nodes) != len(expected.nodes) or graph.plates != expected.plates:
        raise ValueError("The analytic prior requires the registered demo graph.")
    for node, reference in zip(graph.nodes, expected.nodes, strict=True):
        if (type(node) is not type(reference) or node.name != reference.name
                or node.parents != reference.parents or node.plate != reference.plate):
            raise ValueError("The analytic prior requires the registered graph structure.")
        if isinstance(node, Const) and not np.array_equal(node.value, reference.value):
            raise ValueError("The analytic rank proof requires the registered design.")
        if isinstance(node, Deterministic) and node.fn is not reference.fn:
            raise ValueError("The analytic prior requires the registered mean operators.")
        if isinstance(node, Probabilistic):
            if node.observed_mask is not None:
                raise ValueError("The analytic prior does not cover masked observations.")
            if node.parents and node.dist_fn is not reference.dist_fn:
                raise ValueError("The analytic prior requires the registered conditional laws.")
    bounds = []
    for name, (low, high) in TOP_LEVEL_BOUNDS[case].items():
        node = graph.node(name)
        if node.parents or not node.is_latent:
            raise ValueError("The Jeffreys prior covers independent root parameters only.")
        law = node.dist_fn()
        while type(law) is dist.Independent:
            if "log_prob" in vars(law):
                raise ValueError("A changed root density is not a constant Uniform carrier.")
            law = law.base_dist
        if (type(law) is not dist.Uniform or "log_prob" in vars(law)
                or not np.array_equal(law.low, low) or not np.array_equal(law.high, high)):
            raise ValueError("The analytic prior requires the original bounded Uniform roots.")
        bounds.append((jnp.asarray(low), jnp.asarray(high)))
    if any(not np.all(np.isfinite(value)) for value in inputs):
        raise ValueError("The analytic prior requires finite design and scale inputs.")
    return inputs, tuple(bounds)


def with_jeffreys_prior(case, graph):
    """Retain the model and support; attach one exact analytic prior factor."""
    if case not in SUPPORTED_CASES:
        raise ValueError(f"{case}: an exact marginal Fisher Jeffreys prior is not implemented.")
    if graph.joint_prior is not None or graph.evidence_terms:
        raise ValueError("A prior or reduced likelihood is already declared on this graph.")
    if not jax.config.x64_enabled:
        raise ValueError("These analytic Jeffreys demos require jax.enable_x64(True).")
    inputs, bounds = _registered_graph(case, graph)
    prior = AnalyticDemoJeffreys(case, tuple(TOP_LEVEL_BOUNDS[case]), inputs, bounds)
    return Graph(nodes=graph.nodes, plates=graph.plates, joint_prior=prior)
