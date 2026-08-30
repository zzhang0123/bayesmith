"""Atomically remove an integrated block and attach its evidence density.

There is intentionally no public operation here that performs only one half.
A caller either receives one :class:`ReducedGraph` whose underlying graph has
reduced ``nodes`` and whose ``evidence_terms`` carry the replacement density,
or receives no result at all. Keeping those facts in two fields of one returned
value is the structural guard against evaluating the data twice.

The wrapper is deliberately NUTS-only. Generic compilation starts by reading
``latents`` or ``nodes`` and is refused before an exact conditional sampler can
silently omit the graph-level term.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Never

import equinox as eqx

from bayesmith.errors import GraphError
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic


class ReducedGraph(eqx.Module):
    """An atomically reduced graph that may only enter evidence-aware scans.

    ``log_joint`` and the NumPyro bridge unwrap this value explicitly. The
    ``nodes`` and ``evidence_terms`` remain inspectable as the two atomic
    products. The ``latents`` query that starts generic dispatch instead raises
    with the safe route, because those consumers do not read ``evidence_terms``.
    """

    _graph: Graph

    @staticmethod
    def _refuse_generic(attribute: str) -> Never:
        raise GraphError(
            f"ReducedGraph is NUTS-only: generic compile tried to read "
            f"{attribute}. Pass this result directly to log_joint, "
            "to_numpyro, or nuts; do not pass it to generic compile, whose "
            "exact and conditional paths do not read graph-level evidence "
            "terms."
        )

    @property
    def nodes(self) -> tuple[Any, ...]:
        """Retained nodes, in their original topological order."""
        return self._graph.nodes

    @property
    def latents(self) -> tuple[str, ...]:
        self._refuse_generic("latents")

    @property
    def names(self) -> tuple[str, ...]:
        """Retained node names, in their original topological order."""
        return self._graph.names

    @property
    def evidence_terms(self) -> tuple[Any, ...]:
        """The evidence declarations atomically attached by reduction."""
        return self._graph.evidence_terms

    def as_graph(self) -> Graph:
        """Expose the underlying graph to explicitly evidence-aware code."""
        return self._graph


def as_graph(graph: Graph | ReducedGraph) -> Graph:
    """Unwrap a reduced graph for an explicitly evidence-aware consumer."""
    if isinstance(graph, ReducedGraph):
        return graph.as_graph()
    return graph


def _names(values: Iterable[str], *, argument: str) -> tuple[str, ...]:
    names = tuple(values)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise GraphError(
            f"{argument} repeats {duplicates}. Name each node once; repeated "
            "declarations make the reduction boundary ambiguous without "
            "changing the graph."
        )
    return names


def check_evidence_nuts_boundary(
    graph: Graph, nuts_latents: Iterable[str]
) -> None:
    """Refuse a graph-level likelihood that an exact block would omit."""
    nuts = set(nuts_latents)
    for index, term in enumerate(graph.evidence_terms):
        outside = [name for name in term.over if name not in nuts]
        if outside:
            raise GraphError(
                f"evidence_terms[{index}] covers non-NUTS latents {outside}, "
                f"outside the NUTS block {sorted(nuts)}; its full block is "
                f"{list(term.over)}. Exact and conditional samplers do not "
                "read graph-level density terms, so they would silently omit "
                "this likelihood. Add it to nuts_latents (put it in NUTS), "
                "or keep the likelihood explicit and do not absorb those "
                "observations."
            )


def reduce_with_evidence(
    graph: Graph,
    *,
    remove_latents: Iterable[str],
    absorb_observed: Iterable[str],
    evidence_term: Any,
    nuts_latents: Iterable[str],
) -> ReducedGraph:
    """Return the reduced graph with its replacement likelihood attached.

    ``remove_latents`` names the variables integrated out by
    ``evidence_term``. ``absorb_observed`` names the data whose likelihood is
    already inside that term. Deterministic descendants of either set are
    removed automatically; a probabilistic descendant must be named in the
    corresponding set, because silently pruning one would integrate a density
    the term did not declare.

    ``nuts_latents`` is an explicit witness supplied by the plan that will run
    the returned graph. The graph layer cannot import dispatch without
    reversing the package dependency, but it must still refuse a term over a
    non-NUTS block: the exact conditional samplers do not read graph-level
    density terms.
    """
    remove = _names(remove_latents, argument="remove_latents")
    absorbed = _names(absorb_observed, argument="absorb_observed")
    nuts = _names(nuts_latents, argument="nuts_latents")

    latent_names = set(graph.latents)
    wrong_remove = [name for name in remove if name not in latent_names]
    if wrong_remove:
        raise GraphError(
            f"remove_latents names {wrong_remove}, which are not latent nodes "
            f"of this graph; its latents are {list(graph.latents)}. Put only "
            "integrated latent names in remove_latents. Observations whose "
            "likelihood moved into the term belong in absorb_observed."
        )

    observed_names = set(graph.observed)
    wrong_absorbed = [name for name in absorbed if name not in observed_names]
    if wrong_absorbed:
        raise GraphError(
            f"absorb_observed names {wrong_absorbed}, which are not observed "
            f"probabilistic nodes of this graph; its observations are "
            f"{list(graph.observed)}. Name only likelihood nodes here; their "
            "deterministic descendants are removed automatically."
        )

    dropped = set(remove)
    unreached_absorbed = set(absorbed)
    for node in graph.nodes:
        if node.name in dropped:
            continue
        downstream = any(parent in dropped for parent in node.parents)
        if node.name in unreached_absorbed:
            if downstream:
                dropped.add(node.name)
                unreached_absorbed.remove(node.name)
            continue
        if not downstream:
            continue
        if isinstance(node, Probabilistic):
            destination = "remove_latents" if node.is_latent else "absorb_observed"
            density = "latent distribution" if node.is_latent else "likelihood"
            raise GraphError(
                f"probabilistic descendant {node.name!r} still depends on the "
                f"removed region, so its {density} would lose a parent. Add it "
                f"to {destination} only if evidence_term already contains "
                "that density, or do not remove this block."
            )
        dropped.add(node.name)

    if unreached_absorbed:
        unrelated = sorted(unreached_absorbed)
        raise GraphError(
            f"absorb_observed names {unrelated}, which are not descendants "
            "of remove_latents or another dropped node. Keep independent "
            "likelihoods explicit, or collapse them in a separate reduction "
            "whose evidence_term includes those observations."
        )

    reduced = Graph(
        nodes=tuple(node for node in graph.nodes if node.name not in dropped),
        plates=graph.plates,
        joint_prior=graph.joint_prior,
        evidence_terms=(*graph.evidence_terms, evidence_term),
    )

    retained = set(reduced.latents)
    unknown_nuts = [name for name in nuts if name not in retained]
    if unknown_nuts:
        raise GraphError(
            f"nuts_latents names {unknown_nuts}, which are not retained "
            f"latents of the reduced graph; its latents are "
            f"{list(reduced.latents)}. Pass the NUTS block of the returned "
            "graph, after the integrated block has been removed."
        )

    check_evidence_nuts_boundary(reduced, nuts)

    return ReducedGraph(_graph=reduced)
