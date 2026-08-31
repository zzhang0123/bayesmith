"""Collapsed-target sampling: integrate the exact block out of the NUTS target (P6).

**Scope, narrowed before it is widened.** Collapse is a small-block specialty,
not a general accelerator. Measured in this checkout, float64, mu = th . (B x):

* at k=8 collapse already wins for c >= ~0.95; at k=512 it needs c >= ~0.9995;
* the truncated trace-log does NOT rescue large k: it is 25-33% slower than
  QR by scalar-order count and 95-130% slower by CERTIFIED order (gradient
  error 2.803e-06). The reason is not numerical: rho(X) reads 0.85-0.93 on
  every fixture, so the order m has to be 135-212, and the hundreds of
  operator applications spend everything that never materialising the matrix
  saved. The root cause is that P = A S A^T is simply NOT small next to
  Lambda = N, and there is no physically grounded generic Lambda selector
  (D83).

So the collapsed route materialises the dense design and does one QR per
gradient -- which is exactly the side of the package exact/solve.py exists to
avoid -- and it is only worth it when the coupling is strong enough that the
Gibbs sweep's tau(c) amplification outweighs that cost. Nothing here decides
when; the cost scoreboard (dispatch/costs.py) reports the numbers, and the
caller chooses.

The arithmetic was verified end to end before this module existed:

    unchecked_operator(probe_gaussian=False) -> dense_operator -> compress
    -> SqrtInfo.combine(nuisance_prior) -> marginalise_arrays

against a dense slogdet oracle gives 1.8e-14 on the value and 2.6e-08 on the
gradient, and traces cleanly under jit and hessian. The
-sum(log pivots[:n_block]) folded into the offset by marginalise_arrays IS
0.5 * logdet(F_bb).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.fisher import dense_operator
from bayesmith.exact.gaussian import precision_at
from bayesmith.graph.graph import Graph
from bayesmith.graph.reduction import ReducedGraph, reduce_with_evidence
from bayesmith.marginal.compress import compress, nuisance_prior
from bayesmith.marginal.sqrtinfo import SqrtInfo, marginalise_arrays


def _size(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def observed_descendants(graph: Graph, remove: Iterable[str]) -> tuple[str, ...]:
    """The observed nodes reachable from the removed latents, transitively.

    These, and only these, have their likelihood moved into the marginal
    evidence term. An observed node independent of the removed block stays an
    explicit likelihood in the reduced graph; absorbing it would count its
    density twice.
    """
    children: dict[str, set[str]] = {}
    for node in graph.nodes:
        for parent in node.parents:
            children.setdefault(parent, set()).add(node.name)
    seen: set[str] = set()
    frontier = set(remove)
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        frontier.update(children.get(name, ()))
    return tuple(name for name in graph.observed if name in seen)



def pivots_are_finite(pivots: jax.Array) -> jax.Array:
    """Whether every pivot of the marginal QR is finite (D97).

    A non-finite pivot means the re-triangularisation itself carried nan/inf,
    so the marginal log-density is not a number. Under the eager
    marginalise() this is the finiteness check; here it is traced and handed
    to eqx.error_if by marginal_log_density.
    """
    return jnp.all(jnp.isfinite(pivots))


def pivots_constrain_block(pivots: jax.Array, n_block: int) -> jax.Array:
    """Whether the block's pivots all sit above the relative floor (D98).

    The floor is sqrt(eps) times the largest pivot -- the same relative floor
    the eager marginalise() applies. A pivot at or below it means the block
    does not constrain one of its own directions, so the Gaussian integral
    diverges and finite arithmetic would hand back a large plausible number
    instead of an infinity anyone notices.
    """
    scale = jnp.max(pivots)
    floor = jnp.sqrt(jnp.finfo(pivots.dtype).eps) * scale
    return jnp.all(pivots[:n_block] > floor)


def marginal_log_density(
    graph: Graph, exact_names: tuple[str, ...], values: dict[str, Any]
) -> jax.Array:
    """log p(d | theta) -- the exact block integrated out of the joint.

    The exact block is affine given the outside latents, so its marginal
    likelihood is a Gaussian integral in square-root form: whiten the dense
    design, append the block's prior rows, and drop the leading columns. The
    offset that falls out of that drop is the marginal log-density,
    normalisation included.
    """
    block = unchecked_operator(graph, exact_names, at=values, probe_gaussian=False)
    centre = {name: block.prior_mean[name] for name in block.names}
    precision = precision_at(graph, {**values, **centre})

    shapes = {name: block.shape[name] for name in block.names}
    spans: dict[str, tuple[int, int]] = {}
    column = 0
    for name in block.names:
        spans[name] = (column, column + _size(block.shape[name]))
        column += _size(block.shape[name])

    design = dense_operator(block)  # (n_data, n_block), rows sorted(observed)
    terms: list[SqrtInfo] = []
    row = 0
    for observed in sorted(block.data):
        data = jnp.ravel(block.data[observed])
        width = data.shape[0]
        block_columns = design[row : row + width]
        by_name = {
            name: block_columns[:, spans[name][0] : spans[name][1]]
            for name in block.names
        }
        terms.append(
            compress(
                by_name,
                data,
                precision[observed],
                shapes,
                offset_prediction=jnp.ravel(block.offset[observed]),
            )
        )
        row += width

    joint = terms[0]
    for term in terms[1:]:
        joint = SqrtInfo.combine(joint, term)

    prior = nuisance_prior(
        block.names,
        shapes,
        {name: block.prior_std[name] for name in block.names},
        {name: block.prior_mean[name] for name in block.names},
        (),
        {},
    )
    joint = SqrtInfo.combine(joint, prior)

    _factor, _target, offset, pivots = marginalise_arrays(
        joint.factor, joint.target, joint.offset, column
    )

    offset = eqx.error_if(
        offset,
        ~pivots_are_finite(pivots),
        "collapse: the re-triangularisation of the collapsed block is not "
        "finite, so the marginal log-density is not a number.",
    )
    offset = eqx.error_if(
        offset,
        ~pivots_constrain_block(pivots, column),
        "collapse: the exact block does not constrain one of its own "
        "directions at this point, so the Gaussian integral over it diverges. "
        "Give the block a proper prior -- the prior rows must be part of the "
        "model, not an optional regulariser -- or do not collapse it.",
    )
    return offset


class CollapsedEvidence(eqx.Module):
    """The marginal likelihood log p(d | theta) as a graph evidence term.

    Carries the ORIGINAL graph and the integrated block, because the reduced
    graph it is attached to no longer contains them; log_density rebuilds the
    block operator from the stored graph at the passed values.
    """

    graph: Graph
    exact_names: tuple[str, ...] = eqx.field(static=True)
    over: tuple[str, ...] = eqx.field(static=True)

    def log_density(self, graph: Graph, values: dict[str, Any]) -> jax.Array:
        del graph  # the reduced graph; the operator comes from the stored one
        return marginal_log_density(self.graph, self.exact_names, values)


def collapse_graph(
    graph: Graph, exact_names: Iterable[str], nuts_names: Iterable[str]
) -> ReducedGraph:
    """Reduce the graph: exact block integrated out, marginal term attached.

    Atomic by construction -- reduce_with_evidence returns the reduced graph
    and the evidence term as one value, so the data cannot be evaluated twice.
    nuts_names is the explicit NUTS witness the reduced graph is sampled over.
    """
    exact = tuple(exact_names)
    nuts = tuple(nuts_names)
    observed = observed_descendants(graph, exact)
    term = CollapsedEvidence(graph=graph, exact_names=exact, over=nuts)
    return reduce_with_evidence(
        graph,
        remove_latents=exact,
        absorb_observed=observed,
        evidence_term=term,
        nuts_latents=nuts,
    )


__all__ = [
    "CollapsedEvidence",
    "collapse_graph",
    "marginal_log_density",
    "observed_descendants",
    "pivots_are_finite",
    "pivots_constrain_block",
]
