"""Data-informed starts for the rigid TRIS sky model; never a posterior fit.

The temporary optimization coordinate is the minimum Galactic temperature.
It makes the physical support a simple positive bound. Sampling still uses
the original variables, priors, and likelihood, with no truncation renormalization.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from bayesmith.artifacts import InitializationPolicy, NamedArray
from bayesmith.graph.evaluate import evaluate, log_joint


def fitting_problem(graph, inputs):
    """Return residuals whose squared norm differs from -2 log p only by a constant."""
    include_rsb = "rsb_amplitude" in graph.latents
    survey_count = int(inputs[-2])
    minimum = jnp.min(inputs[8])
    end = 9 + survey_count

    def decode(x):
        values = {"amplitude": x[:3], "beta": x[3:6], "zero_standard": x[6:8],
                  "calibration_standard": x[9:end]}
        background = 0.0
        if include_rsb:
            values.update(rsb_amplitude=x[end], rsb_beta=x[end + 1])
            background = x[end] * .408 ** x[end + 1]
        values["haslam_monopole_K"] = x[8] - minimum + background
        return values

    def residual(x):
        values = decode(x)
        env = evaluate(graph, values)
        return jnp.concatenate((
            inputs[7] - env["whitened_map_mean"],
            (inputs[10] - env["external_background_mean"]) / inputs[11],
            values["zero_standard"], values["calibration_standard"],
            jnp.atleast_1d(values["haslam_monopole_K"] / 3.0),
        ))

    lower = np.array([.2] * 3 + [-4.] * 3 + [-np.inf] * 2 + [1e-8]
                     + [-np.inf] * survey_count + ([1e-9, -4.] if include_rsb else []))
    upper = np.array([3.] * 3 + [-1.5] * 3 + [np.inf] * (3 + survey_count)
                     + ([5., -1.5] if include_rsb else []))
    return decode, jax.jit(residual), lower, upper


def data_initialization(graph, inputs, *, chains, seed):
    """Audit three dispersed fits, then seed distinct support-interior chains.

    Interior margins and perturbations affect initialization ONLY. Uniform
    prior endpoints can be MAP optima but are invalid logistic-transform starts.
    The fitted inverse curvature scales perturbations, not prior uncertainties.
    """
    decode, residual, lower, upper = fitting_problem(graph, inputs)
    jacobian = jax.jit(jax.jacfwd(residual))
    include_rsb = "rsb_amplitude" in graph.latents
    survey_count = int(inputs[-2])
    fits = []
    for index, (a, beta, floor) in enumerate(((.8, -3.2, 8.), (1., -2.7, 15.), (1.4, -2.2, 25.))):
        start = np.array([a] * 3 + [beta] * 3 + [0., 0., floor] + [0.] * survey_count
                         + ([(.3, 1.2, 2.)[index], (-2., -2.6, -3.2)[index]] if include_rsb else []))
        fit = least_squares(
            residual, start, jac=jacobian, bounds=(lower, upper), x_scale="jac",
            max_nfev=3000, ftol=1e-12, xtol=1e-12, gtol=1e-7,
        )
        fits.append(fit)
    good = [fit for fit in fits if fit.success and np.isfinite(fit.cost)]
    if not good:
        raise RuntimeError("None of the three TRIS data-initialization fits succeeded")
    best = min(good, key=lambda fit: fit.cost)
    _, singular, vt = np.linalg.svd(best.jac, full_matrices=False)
    # A boundary MAP may leave the RSB slope nearly unidentified. Its prior
    # range, not the near-singular local curvature, bounds the initial spread.
    inverse = vt.T / np.maximum(singular, 1.0)
    bounded = np.isfinite(lower) & np.isfinite(upper)
    margin = 1e-4 * (upper[bounded] - lower[bounded])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 941]))
    starts = []
    for chain in range(chains):
        point = best.x + 3.0 * inverse @ rng.normal(size=inverse.shape[1])
        point[bounded] = np.clip(point[bounded], lower[bounded] + margin, upper[bounded] - margin)
        point[8] = max(point[8], .01)
        if include_rsb and best.x[-2] < 1e-6:
            point[-1] = -3.8 + 2.1 * (chain + .5) / chains
        values = decode(jnp.asarray(point))
        if not np.isfinite(float(log_joint(graph, values))):
            raise RuntimeError("Data fit produced a non-finite chain initialization")
        starts.append(values)
    policy = InitializationPolicy(values=tuple(
        NamedArray(name, np.stack([value[name] for value in starts]),
                   ("chain",) + tuple(f"axis{i}" for i in range(np.ndim(starts[0][name]))))
        for name in graph.latents
    ))
    record = {
        "method": "three bounded least-squares starts; curvature-scaled perturbations",
        "truth_used": False, "posterior_target_changed": False,
        "optimization_coordinate": "min Galactic template = min(H-CMB) + z_H - B408",
        "uniform_interior_margin_fraction": 1e-4,
        "fits": [{"success": bool(fit.success), "cost": float(fit.cost),
                  "evaluations": int(fit.nfev), "optimality": float(fit.optimality),
                  "message": str(fit.message)} for fit in fits],
        "initial_log_density": [float(log_joint(graph, value)) for value in starts],
    }
    return policy, record
