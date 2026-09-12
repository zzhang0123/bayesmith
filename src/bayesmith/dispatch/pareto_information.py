"""Expected information for canonical Pareto observations with fixed cutoffs."""

import weakref

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions.transforms import (
    AffineTransform,
    ExpTransform,
    _InverseTransform,
)

from bayesmith.diagnose.local import (
    check_differentiable,
    flat_view,
    refuse_single_precision,
    unflatten,
)
from bayesmith.diagnose.structure import derivative_semantics_evidence
from bayesmith.errors import GraphError
from bayesmith.exact.block import _ancestors
from bayesmith.graph.evaluate import apply_probabilistic, evaluate


def _canonical_transform(transform, allowed):
    # Instance method overrides can change the Jacobian or inverse while the
    # transform's exact type is unchanged. Also validate the inverse cache.
    if set(vars(transform)) - allowed or transform.domain is not dist.constraints.real:
        return False
    cached = vars(transform).get("_inv")
    if cached is None:
        return True
    if not isinstance(cached, weakref.ReferenceType):
        return False
    inverse = cached()
    return inverse is None or (type(inverse) is _InverseTransform
                               and set(vars(inverse)) == {"_inv"}
                               and inverse._inv is transform)


def pareto_parameters(graph, name, env):
    node = graph.node(name)
    law = apply_probabilistic(graph, node, env)
    shape = law.batch_shape
    if node.plate:
        shape = jnp.broadcast_shapes(shape, (graph.plate_size(node.plate[0]),))
    shape = jnp.broadcast_shapes(shape + law.event_shape, jnp.shape(node.observed))
    if node.observed_mask is not None and law.event_shape:
        raise GraphError("Pareto information does not support event-reduced masks.")
    while type(law) in (dist.Independent, dist.ExpandedDistribution):
        if "log_prob" in vars(law):
            raise GraphError("Custom Pareto wrapper density is unsupported.")
        law = law.base_dist
    if (type(law) is not dist.Pareto or "log_prob" in vars(law)
            or type(law.base_dist) is not dist.Exponential
            or "log_prob" in vars(law.base_dist)
            or len(law.transforms) != 2
            or type(law.transforms[0]) is not ExpTransform
            or type(law.transforms[1]) is not AffineTransform
            or not _canonical_transform(law.transforms[0], {"_domain", "_inv"})
            or not _canonical_transform(law.transforms[1], {"loc", "scale", "_domain", "_inv"})):
        raise GraphError("Expected information requires canonical Pareto factors only.")
    # Read the actual density's rate and cutoff, not potentially stale attributes.
    return tuple(jnp.broadcast_to(p, shape) for p in
                 (law.base_dist.rate, law.transforms[1].scale, law.transforms[1].loc))


def pareto_information(graph, names, values):
    """J_log_shape.T J_log_shape; support must be independent of this block."""
    check_differentiable(graph, names, values)
    if any(_ancestors(graph, n).intersection(graph.observed) for n in graph.observed):
        raise GraphError("Observation-dependent Pareto parameters are unsupported.")
    point, shapes, spans = flat_view(values, names)

    def parameters(vector):
        env = evaluate(graph, {**values, **unflatten(vector, names, shapes, spans)})
        return tuple(part for n in graph.observed for part in pareto_parameters(graph, n, env))

    evidence = derivative_semantics_evidence(parameters, point)
    degrees = evidence.get("output_degrees", ())
    if not evidence["trusted"] or any(degrees[1::3] + degrees[2::3]):
        raise GraphError("Regular Pareto Fisher requires a structurally fixed cutoff "
                         "and trusted derivatives; parameter-dependent support is unsupported.")
    evaluated = parameters(point)
    for parameter in evaluated[::3] + evaluated[1::3]:
        refuse_single_precision(parameter, doing="automatic Pareto likelihood geometry")
    if any(not np.all(np.isfinite(v) & (np.asarray(v) > 0))
           for v in evaluated[::3] + evaluated[1::3]):
        raise GraphError("Pareto shapes and cutoffs must be finite and positive.")
    if any(not np.all(np.asarray(v) == 0) for v in evaluated[2::3]):
        raise GraphError("Canonical Pareto information requires zero affine offsets.")

    def log_shapes(vector):
        return jnp.concatenate([jnp.log(v).ravel() for v in parameters(vector)[::3]])

    jacobian = np.asarray(jax.jacfwd(log_shapes)(point))
    refuse_single_precision(jacobian, doing="automatic Pareto likelihood geometry")
    weights = []
    for name, alpha in zip(graph.observed, evaluated[::3], strict=True):
        mask = graph.node(name).observed_mask
        weights.append(np.ones(alpha.size) if mask is None
                       else np.broadcast_to(mask, alpha.shape).ravel())
    if not np.all(np.isfinite(jacobian)):
        raise GraphError("Pareto shape derivatives must be finite.")
    matrix = jacobian.T @ (np.concatenate(weights)[:, None] * jacobian)
    return (matrix + matrix.T) / 2
