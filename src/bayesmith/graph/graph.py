"""The graph container, and the structural rules it refuses to violate.

Declaration order **is** topological order. A node may only name parents
already declared, which is automatic when the graph comes from ``trace`` --
you cannot pass a ``NodeRef`` you have not created -- and is checked here so
that a hand-built graph obeys the same rule.
"""

from __future__ import annotations

import equinox as eqx

from bayesmith.errors import GraphError
from bayesmith.graph.nodes import Node, Probabilistic


class Plate(eqx.Module):
    """A named repetition axis. ``size`` is fixed at construction."""

    name: str = eqx.field(static=True)
    size: int = eqx.field(static=True)


class Graph(eqx.Module):
    """A static DAG of nodes in topological order, plus its plates."""

    nodes: tuple[Node, ...]
    plates: tuple[Plate, ...]

    def __check_init__(self) -> None:
        plate_names: set[str] = set()
        for p in self.plates:
            if p.name in plate_names:
                raise GraphError(
                    f"duplicate plate name {p.name!r}: plate names must be "
                    "unique within a graph."
                )
            plate_names.add(p.name)

        seen: set[str] = set()
        for node in self.nodes:
            if node.name in seen:
                raise GraphError(
                    f"duplicate node name {node.name!r}: node names must be "
                    "unique within a graph."
                )
            for parent in node.parents:
                if parent not in seen:
                    raise GraphError(
                        f"node {node.name!r} names parent {parent!r}, which is not "
                        "declared before it. Declaration order is topological "
                        "order: declare the parent first."
                    )
            if len(node.plate) > 1:
                raise GraphError(
                    f"node {node.name!r} is in plates {node.plate}: nested plates "
                    "are not supported yet. Flatten the two axes into one plate."
                )
            for plate in node.plate:
                if plate not in plate_names:
                    raise GraphError(
                        f"node {node.name!r} is in plate {plate!r}, which the graph "
                        f"does not declare. Known plates: {sorted(plate_names)}."
                    )
            seen.add(node.name)

    @property
    def names(self) -> tuple[str, ...]:
        """Every node name, in topological order."""
        return tuple(node.name for node in self.nodes)

    @property
    def latents(self) -> tuple[str, ...]:
        """Names of the probabilistic nodes that are inferred."""
        return tuple(
            n.name for n in self.nodes if isinstance(n, Probabilistic) and n.is_latent
        )

    @property
    def observed(self) -> tuple[str, ...]:
        """Names of the probabilistic nodes that are conditioned on."""
        return tuple(
            n.name
            for n in self.nodes
            if isinstance(n, Probabilistic) and not n.is_latent
        )

    def node(self, name: str) -> Node:
        """The node called ``name``.

        Raises:
            GraphError: if no node has that name.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise GraphError(f"no node named {name!r}. Known: {list(self.names)}.")

    def plate_size(self, name: str) -> int:
        """The declared size of plate ``name``.

        Raises:
            GraphError: if no plate has that name.
        """
        for plate in self.plates:
            if plate.name == name:
                return plate.size
        raise GraphError(
            f"no plate named {name!r}. Known: {[p.name for p in self.plates]}."
        )
