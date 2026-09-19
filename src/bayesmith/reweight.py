"""Fit hyperparameters by recycling a fixed reference posterior sample.

For draws x_i from p_0(x | d), log weights are
log p_eta(x_i, d) - log p_0(x_i, d). Their log mean estimates
log Z_eta - log Z_0. Add log p(eta) and maximize to obtain a Monte Carlo
marginal hyperparameter MAP, integrating out x rather than maximizing x.

The reference need not be Gaussian. A fixed linear-Gaussian reference is
particularly useful because its draws can come from GCR. This module owns
the density-ratio objective; sampling remains with the existing compiler and
optimization with bayesmith.optimize.minimize. See docs/reweight.md.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpyro.distributions import (
    Delta,
    Distribution,
    MixtureGeneral,
    MixtureSameFamily,
    constraints,
)

from bayesmith.distributions import ComplexNormal
from bayesmith.exact.correct import self_normalise
from bayesmith.graph.evaluate import apply_probabilistic, evaluate, log_joint
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic
from bayesmith.optimize import minimize

__all__ = ["HyperparameterFit", "ImportanceEstimate", "PosteriorReweighting"]


class ImportanceEstimate(NamedTuple):
    """Raw Monte Carlo normalization estimate and weight concentration.

    ``log_mean_weight`` is log(Z_target/Z_reference) when both callbacks
    score joint densities. With a normalized proposal denominator it instead
    estimates log Z_target. It is NOT an absolute log evidence by default.
    Kish ``ess`` measures weight concentration only, not MCMC autocorrelation
    or unseen target mass. No field certifies overlap or convergence.
    """

    log_mean_weight: jax.Array
    weights: jax.Array
    ess: jax.Array
    ess_fraction: jax.Array
    max_weight: jax.Array


class HyperparameterFit(NamedTuple):
    """Point reached after a fixed step budget, without a convergence verdict.

    Objective and history use the MAXIMIZATION sign. ``parameters`` are in
    exactly the coordinates supplied to ``fit``. ``gradient_norm`` is the
    Euclidean norm in those coordinates. ``importance`` is evaluated at the
    returned point, so its weights describe the plug-in target posterior.
    """

    parameters: Any
    log_objective: jax.Array
    history: jax.Array
    importance: ImportanceEstimate
    gradient_norm: jax.Array


def _scalar(value: Any, name: str) -> jax.Array:
    value = jnp.asarray(value)
    if value.shape != () or jnp.iscomplexobj(value):
        raise ValueError(f"{name} must return one real scalar per draw/event")
    return value.astype(jnp.result_type(value, float))


def _special_measure(distribution: Distribution, shape: tuple[int, ...]) -> str | None:
    """Preserve atomic/complex measures hidden by distribution wrappers."""
    base = distribution
    while hasattr(base, "base_dist"):
        base = base.base_dist
    if isinstance(base, ComplexNormal):
        return "complex"
    if isinstance(base, Delta):
        return "atomic"
    if isinstance(base, MixtureSameFamily):
        return _special_measure(base.component_distribution, shape)
    if isinstance(base, MixtureGeneral):
        kinds = {
            _measure(component, shape)[0] for component in base.component_distributions
        }
        if len(kinds) != 1:
            raise ValueError(
                "mixture components use different measures; use explicit density callbacks"
            )
        kind = kinds.pop()
        return kind if kind in {"atomic", "complex"} else None
    return None


def _measure(
    distribution: Distribution, shape: tuple[int, ...]
) -> tuple[str, tuple[int, ...]]:
    """Identify supported coordinate measures, without requiring equal bounds.

    Equal support declarations cannot prove proposal coverage. These signatures
    only reject known incompatible counting, real, complex and manifold domains.
    Unknown constraints require the explicit callback interface.
    """
    special = _special_measure(distribution, shape)
    if special is not None:
        return special, shape
    support = distribution.support
    while isinstance(support, constraints.independent):
        support = support.base_constraint
    if support.is_discrete:
        return "counting", shape
    full_dimensional = (
        type(constraints.real),
        type(constraints.positive),
        type(constraints.nonnegative),
        constraints.greater_than,
        constraints.greater_than_eq,
        constraints.less_than,
        constraints.less_than_eq,
        constraints.interval,
        constraints.open_interval,
        type(constraints.unit_interval),
        type(constraints.ordered_vector),
        type(constraints.positive_ordered_vector),
    )
    if type(support) in full_dimensional:
        return "real", shape
    manifolds = (
        constraints.simplex,
        constraints.lower_cholesky,
        constraints.positive_definite,
        constraints.corr_matrix,
        constraints.corr_cholesky,
    )
    if type(support) in tuple(type(item) for item in manifolds):
        return type(support).__name__, shape
    raise ValueError(
        f"unsupported graph support measure {type(support).__name__}; use "
        "PosteriorReweighting's explicit density callbacks with a common coordinate measure"
    )


def _graph_domain(
    graph: Graph, draw: Mapping[str, jax.Array]
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], jax.Array]:
    """Check declared coordinates and support at one complete graph draw."""
    env = evaluate(graph, draw)
    measures = {}
    supported = jnp.asarray(True)
    for node in graph.nodes:
        if not isinstance(node, Probabilistic):
            continue
        distribution = apply_probabilistic(graph, node, env)
        value = jnp.asarray(env[node.name])
        batch = distribution.batch_shape
        if node.plate:
            batch = jnp.broadcast_shapes(batch, (graph.plate_size(node.plate[0]),))
        expected = batch + distribution.event_shape
        if node.is_latent and value.shape != expected:
            raise ValueError(
                f"latent {node.name!r} has draw shape {value.shape}; "
                f"its declared shape is {expected}"
            )
        if (
            not node.is_latent
            and node.plate
            and (value.ndim == 0 or value.shape[0] != graph.plate_size(node.plate[0]))
        ):
            raise ValueError(
                f"node {node.name!r} needs an explicit plate observation axis; "
                "broadcast the observations before graph-based reweighting"
            )
        # Observations may legitimately broadcast a scalar law over explicit
        # data. Compare the resulting coordinates, not just the raw data shape:
        # a scalar datum scored by a vector law is a different conditioning law.
        coordinates = jnp.broadcast_shapes(value.shape, expected)
        measure = _measure(distribution, coordinates)
        if jnp.iscomplexobj(value) and measure[0] != "complex":
            raise ValueError(f"node {node.name!r} has complex values in a real measure")
        measures[node.name] = measure
        valid = distribution.support(value)
        if node.observed_mask is not None:
            valid = jnp.where(node.observed_mask, valid, True)
        supported = supported & jnp.all(valid)
    return measures, supported


def _graph_score(
    graph: Graph,
    draw: Mapping[str, jax.Array],
    reference_measures: dict[str, tuple[str, tuple[int, ...]]],
) -> jax.Array:
    measures, supported = _graph_domain(graph, draw)
    if measures != reference_measures:
        raise ValueError("target and reference must use the same coordinate measures")
    # Unlike where(supported, log_joint(...), -inf), cond keeps derivatives of
    # undefined out-of-support terms out of the result, including under vmap.
    score_dtype = jax.eval_shape(lambda: log_joint(graph, draw)).dtype
    return jax.lax.cond(
        supported,
        lambda _: log_joint(graph, draw),
        lambda _: jnp.asarray(-jnp.inf, dtype=score_dtype),
        operand=None,
    )


class PosteriorReweighting(eqx.Module):
    """A fixed sample bank, cached reference scores and a target family.

    ``samples`` is an array or pytree with one common leading draw axis;
    combine chain and draw axes explicitly before construction. The bank
    must contain unweighted draws from the declared reference law.
    ``reference_log_density(draw)`` may score a normalized proposal or an
    unnormalized reference joint. ``target_log_density(parameters, draw)``
    scores the FULL target joint, including every parameter-dependent
    normalizer. Only terms identical in both models can be canceled.

    The bank and cached denominator are held fixed while optimizing.
    The reference must cover the target support; this cannot be established
    from a finite bank. Callbacks must be deterministic and JAX compatible.
    """

    samples: Any
    reference_log_density: jax.Array
    target_log_density: Callable
    n_samples: int = eqx.field(static=True)

    def __init__(
        self,
        samples: Any,
        *,
        reference_log_density: Callable[[Any], jax.Array],
        target_log_density: Callable[[Any, Any], jax.Array],
    ):
        samples = jax.tree.map(jnp.asarray, samples)
        leaves = jax.tree.leaves(samples)
        if not leaves or any(x.ndim == 0 or x.shape[0] == 0 for x in leaves):
            raise ValueError("samples must have a nonempty leading sample axis")
        self.n_samples = leaves[0].shape[0]
        if any(x.shape[0] != self.n_samples for x in leaves):
            raise ValueError("all sample leaves must share their leading draw axis")
        self.samples = jax.tree.map(
            lambda x: jax.lax.stop_gradient(
                eqx.error_if(x, jnp.any(~jnp.isfinite(x)), "samples must be finite")
            ),
            samples,
        )
        reference = jax.vmap(
            lambda x: _scalar(reference_log_density(x), "reference_log_density")
        )(self.samples)
        self.reference_log_density = jax.lax.stop_gradient(
            eqx.error_if(
                reference,
                jnp.any(~jnp.isfinite(reference)),
                "reference density must be finite at every supplied draw",
            )
        )
        self.target_log_density = target_log_density

    @classmethod
    def from_graphs(
        cls,
        reference: Graph,
        samples: Mapping[str, jax.Array],
        target: Callable[[Any], Graph],
    ) -> PosteriorReweighting:
        """Use complete graph joints, preserving all likelihood/prior terms.

        ``target(parameters)`` builds the non-Gaussian graph. Both graphs
        must have the same latent names and condition on the same observed
        values and masks. All nuisance latents are included in each draw.
        A conditional block at held nuisance values is not a sample of this
        full reference posterior. The caller owns reference sampling and its
        convergence; passing a Gaussian approximation's draws while scoring
        some other reference joint is invalid.

        Latent value shapes and supported coordinate measures must agree;
        batch/event regrouping and same-measure support restrictions are allowed.
        Draws outside target support receive zero weight. Unsupported custom
        constraints require the explicit callback constructor. These checks do
        not establish proposal coverage or a graph-level factor's normalization.
        """
        if not isinstance(samples, Mapping) or set(samples) != set(reference.latents):
            raise ValueError("samples must contain exactly the reference graph latents")
        bank = {name: jnp.asarray(value) for name, value in samples.items()}
        if not bank or any(
            value.ndim == 0 or value.shape[0] == 0 for value in bank.values()
        ):
            raise ValueError("samples must have a nonempty leading sample axis")
        first = {name: value[0] for name, value in bank.items()}
        reference_measures, _ = _graph_domain(reference, first)

        def target_score(parameters, draw):
            graph = target(parameters)
            if set(graph.latents) != set(reference.latents):
                raise ValueError("target and reference must have the same latents")
            if set(graph.observed) != set(reference.observed):
                raise ValueError(
                    "target and reference must condition on the same observations"
                )
            score = _graph_score(graph, draw, reference_measures)
            for name in reference.observed:
                old, new = reference.node(name), graph.node(name)
                for before, after in (
                    (old.observed, new.observed),
                    (old.observed_mask, new.observed_mask),
                ):
                    if before is None and after is None:
                        continue
                    if before is None or after is None or before.shape != after.shape:
                        raise ValueError(
                            "target and reference observations/masks must agree"
                        )
                    score = eqx.error_if(
                        score,
                        ~jnp.array_equal(before, after, equal_nan=True),
                        "target and reference observations/masks must agree",
                    )
            return score

        return cls(
            bank,
            reference_log_density=lambda draw: _graph_score(
                reference, draw, reference_measures
            ),
            target_log_density=target_score,
        )

    def log_weights(self, parameters: Any) -> jax.Array:
        """Full target minus fixed reference scores; zeros remain -inf.

        NaN, positive infinity, and an all-zero-weight bank are refused,
        including under JIT. In particular, a negative Edgeworth correction
        cannot be clipped or silently dropped to make optimization proceed.
        """
        target = jax.vmap(
            lambda x: _scalar(
                self.target_log_density(parameters, x), "target_log_density"
            )
        )(self.samples)
        weights = target - self.reference_log_density
        return eqx.error_if(
            weights,
            jnp.any(jnp.isnan(weights) | jnp.isposinf(weights))
            | ~jnp.any(jnp.isfinite(weights)),
            "invalid target log density or no overlap with the reference sample bank",
        )

    def estimate(self, parameters: Any) -> ImportanceEstimate:
        """Evaluate the raw log mean and normalized posterior weights."""
        log_weights = self.log_weights(parameters)
        weights, ess = self_normalise(log_weights)
        return ImportanceEstimate(
            logsumexp(log_weights) - jnp.log(self.n_samples),
            weights,
            ess,
            ess / self.n_samples,
            jnp.max(weights),
        )

    def log_objective(
        self,
        parameters: Any,
        *,
        log_hyperprior: Callable[[Any], jax.Array] | None = None,
    ) -> jax.Array:
        """Log mean weight + log hyperprior, up to a fixed additive constant.

        This is a Monte Carlo marginal likelihood objective, not the mean
        log weight, a joint MAP over the field, or an EM lower bound.
        ``log_hyperprior`` defines the MAP coordinates. For eta=exp(u),
        log p_eta(exp(u)) finds a mode in eta; adding u changes the density
        to one in u and generally changes that mode. No Jacobian is added
        automatically. Without a hyperprior this is marginal ML.
        """
        score = logsumexp(self.log_weights(parameters)) - jnp.log(self.n_samples)
        if log_hyperprior is not None:
            hyperprior = _scalar(log_hyperprior(parameters), "log_hyperprior")
            hyperprior = eqx.error_if(
                hyperprior,
                jnp.isnan(hyperprior) | jnp.isposinf(hyperprior),
                "invalid hyperprior log density",
            )
            score = score + hyperprior
        return score

    def fit(
        self,
        at: Any,
        *,
        log_hyperprior: Callable[[Any], jax.Array] | None = None,
        min_ess: float | None = None,
        **optimizer_options: Any,
    ) -> HyperparameterFit:
        """Maximize on this fixed bank using ``optimize.minimize`` settings.

        Optional ``min_ess`` is an explicit final Kish-ESS floor, not a
        certification of reliability. A finite objective and gradient are
        required at entry and exit. Steps do not redraw the reference or
        clip/smooth the importance ratios. Validate with a fresh bank and
        reference-chain diagnostics before trusting an optimized result.
        """
        if min_ess is not None and not 1 <= min_ess <= self.n_samples:
            raise ValueError("min_ess must be between 1 and the sample count")
        at = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.result_type(x, float)), at)

        if not jax.tree.leaves(at):
            raise ValueError("at must contain at least one hyperparameter")

        def objective(parameters):
            return self.log_objective(parameters, log_hyperprior=log_hyperprior)

        def checked_point(point):
            value, gradient = jax.value_and_grad(objective)(point)
            finite = jnp.isfinite(value) & jnp.all(
                jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(gradient)])
            )
            point = eqx.error_if(
                point, ~finite, "hyperparameter objective and gradient must be finite"
            )
            norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(gradient)))
            return point, norm

        at, _ = checked_point(at)
        fitted = minimize(lambda point: -objective(point), at, **optimizer_options)
        point, gradient_norm = checked_point(fitted.values)
        estimate = self.estimate(point)
        result = HyperparameterFit(
            point, -fitted.objective, -fitted.history, estimate, gradient_norm
        )
        # Attach final checks to every result leaf. Guarding only parameters
        # lets JIT discard the check when a caller reads only log_objective.
        result = eqx.error_if(
            result,
            ~jnp.isfinite(result.log_objective) | ~jnp.isfinite(gradient_norm),
            "hyperparameter objective and gradient must be finite",
        )
        if min_ess is not None:
            result = eqx.error_if(
                result,
                estimate.ess < min_ess,
                "importance ESS is below min_ess; improve the reference or enlarge the bank",
            )
        return result
