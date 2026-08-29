"""A graph-native posterior mode, with refusals that belong to MAP.

The objective is exactly ``-log_joint(graph, values)`` over every latent in
declaration order.  It is not prior sensitivity's likelihood-only
counterfactual and deliberately does not inherit that diagnostic's three
admission rules: hierarchical priors may depend on other latents, densities
need not be diagonal Gaussians, and the likelihood alone need not anchor a
direction that the full posterior anchors legitimately.

This remains a local optimisation result.  A successful mode says nothing
about other basins, posterior mass, or funnel geometry; a Laplace diagnostic
centred here inherits those blind spots.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.diagnose.local import (
    check_differentiable,
    flat_view,
    latent_values,
    refuse_ambient_float32,
    unflatten,
)
from bayesmith.diagnose.sensitivity import NEWTON_TOL, _newton
from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.graph import Graph


@dataclasses.dataclass(frozen=True)
class MapEstimate(Mapping[str, jax.Array]):
    """A converged local posterior mode and its optimisation evidence.

    The object is itself a mapping, so it can be passed directly where graph
    diagnostics expect ``{latent: value}``, including
    ``local_block(graph, names, estimate, priors=True)``.
    """

    point: dict[str, jax.Array]
    objective: float
    gradient_norm: float
    steps: int
    verdict: ClassVar[str] = "measured"

    def __getitem__(self, name: str) -> jax.Array:
        return self.point[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self.point)

    def __len__(self) -> int:
        return len(self.point)


@dataclasses.dataclass(frozen=True)
class Refused:
    """The graph has no trustworthy finite MAP result from this solve."""

    reason: str
    verdict: ClassVar[str] = "refused"


@dataclasses.dataclass(frozen=True)
class NotApplicable:
    """The graph contains no unknown quantity for MAP to estimate."""

    reason: str
    verdict: ClassVar[str] = "not-applicable"


MapResult = MapEstimate | Refused | NotApplicable


def _precision_refusal() -> Refused | None:
    """Name the caller-controlled x64 boundary as a MAP refusal."""
    try:
        refuse_ambient_float32(doing="map_estimate's posterior optimisation")
    except GraphError as error:
        return Refused(
            f"{error} Build the graph and call map_estimate inside "
            "`with jax.enable_x64(True):`; widening only the call leaves "
            "already-created constants and observations at float32."
        )
    return None


def map_estimate(
    graph: Graph,
    *,
    at: dict[str, jax.Array] | None = None,
) -> MapResult:
    """Find a local mode of the graph's complete posterior.

    Args:
        graph: the declared model.  Every probabilistic term, observed mask,
            and graph-level joint prior enters through :func:`log_joint`.
        at: optional Newton starting point.  Missing latents start at their
            declared centres; keys and shapes are validated by name.

    Returns:
        :class:`MapEstimate` on convergence, :class:`Refused` when the
        arithmetic or local optimisation cannot support a MAP, and
        :class:`NotApplicable` for a graph with no latents.  Every refusal
        includes an actionable alternative rather than leaking the last
        iterate as if it had converged.

    Raises:
        GraphError: for a malformed ``at`` mapping or a latent with no
            real-valued continuous derivative.  Those are caller/graph
            contract errors, not optimisation verdicts.
    """
    if not graph.latents:
        return NotApplicable(
            "map_estimate is not applicable: the graph has no latent nodes. "
            "There is no posterior parameter to optimise; evaluate the graph "
            "directly instead."
        )
    precision_refusal = _precision_refusal()
    if precision_refusal is not None:
        return precision_refusal

    names = graph.latents
    values0 = latent_values(graph, at)
    check_differentiable(graph, names, values0)
    x0, shapes, spans = flat_view(values0, names)
    if x0.dtype != jnp.float64:
        return Refused(
            f"map_estimate's starting point is {x0.dtype}, not float64. The "
            "graph was built with single-precision latent centres even though "
            "the call is now wide. Rebuild the graph inside `with "
            "jax.enable_x64(True):` and retry."
        )

    def objective(x: jax.Array) -> jax.Array:
        values = {**values0, **unflatten(x, names, shapes, spans)}
        return -log_joint(graph, values)

    mode, steps, converged = _newton(objective, x0)
    if not converged:
        return Refused(
            f"map_estimate did not converge after {steps} damped Newton "
            f"steps to max|dx|/(1+|x|) < {NEWTON_TOL:g}. The last iterate is "
            "not returned because it cannot be distinguished from a mode. "
            "Try a different basin with at=, reparameterise the model, or "
            "use NUTS when the posterior is not locally quadratic."
        )

    value = objective(mode)
    gradient = jax.grad(objective)(mode)
    hessian = jax.hessian(objective)(mode)
    if value.dtype != jnp.float64 or gradient.dtype != jnp.float64:
        return Refused(
            "map_estimate's objective or derivative came back single "
            "precision even though the latent vector is float64. Remove "
            "float32 casts/constants from the graph, rebuild it inside "
            "`jax.enable_x64(True)`, and retry."
        )
    finite = bool(
        jnp.isfinite(value)
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(jnp.isfinite(hessian))
    )
    if not finite:
        return Refused(
            "map_estimate reached a non-finite objective, gradient, or "
            "Hessian. Try a finite at= closer to the posterior mass, rescale "
            "or transform the offending latent, or use NUTS with an "
            "appropriate constrained parameterisation."
        )

    eigenvalues = np.linalg.eigvalsh(np.asarray(hessian))
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    curvature_floor = (
        np.finfo(np.asarray(hessian).dtype).eps
        * max(abs(largest), 1.0)
        * max(mode.size, 1)
    )
    if not smallest > curvature_floor:
        return Refused(
            "map_estimate found a stationary point whose negative-log-"
            f"posterior Hessian is not positive above roundoff (smallest "
            f"eigenvalue {smallest:.3e}, floor {curvature_floor:.3e}). It is "
            "a saddle or a flat ridge, not an isolated MAP. Add a proper "
            "prior to the unanchored direction, remove a redundant latent, "
            "or reparameterise before using a Laplace diagnostic."
        )

    gradient_norm = float(jnp.max(jnp.abs(gradient)))
    # `_newton` certifies a small relative step.  Record the gradient rather
    # than imposing a condition-number ceiling here: a very sharp but valid
    # funnel mode can have a large Hessian, and this function never inverts it.
    found = unflatten(mode, names, shapes, spans)
    return MapEstimate(
        point={name: found[name] for name in names},
        objective=float(value),
        gradient_norm=gradient_norm,
        steps=int(steps),
    )


__all__ = [
    "MapEstimate",
    "Refused",
    "NotApplicable",
    "MapResult",
    "map_estimate",
]
