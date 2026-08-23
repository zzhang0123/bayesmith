"""The three node types a bayesmith graph is built from.

A node is an :class:`equinox.Module`, so a graph is a pytree and the
parameters carried by its operators are differentiable leaves for free.

**Why ``fn`` and ``dist_fn`` are not static fields.** They must accept an
``eqx.Module`` -- a whole rheplicant ``Pipeline`` is the motivating case --
whose array leaves have to stay traceable so gradients reach the parameters
inside them. A plain lambda placed in the same field simply becomes a
non-array leaf, which ``eqx.filter_jit`` routes to the static side. Marking
these fields ``static=True`` would break the Module case outright: equinox
refuses a JAX array in a static field.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax


class Node(eqx.Module):
    """Identity and position of one node in the graph.

    Attributes:
        name: unique within a graph; how parents refer to it.
        parents: names of the nodes whose values this one consumes, in the
            order ``fn`` / ``dist_fn`` expects them.
        plate: names of the plates this node lives inside. Empty means the
            node is scalar with respect to every plate.
    """

    name: str = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)
    plate: tuple[str, ...] = eqx.field(static=True)


class Const(Node):
    """A fixed input: data, coordinates, anything the model conditions on.

    The value is an ordinary array field rather than a closure capture, so it
    is one traced leaf of the graph instead of a constant baked into a
    compiled function.
    """

    value: jax.Array


class Deterministic(Node):
    """A mapping that propagates dependence but contributes no density.

    Attributes:
        fn: called with the parents' values, in ``parents`` order.
        linear_in: the parents this node claims to be linear in. This is a
            **claim about the model**, not a hint -- it decides whether an
            exact conjugate solve may be used -- so it is checked rather than
            trusted before any such solve runs. Nothing in P1 reads it; it is
            recorded here so the declaration exists from the start.
    """

    fn: Callable[..., Any]
    linear_in: tuple[str, ...] = eqx.field(static=True, default=())


class Probabilistic(Node):
    """A conditional distribution: contributes one term to the log-density.

    Attributes:
        dist_fn: called with the parents' values; returns a NumPyro
            distribution. Sampling and ``log_prob`` therefore come from one
            object and cannot disagree -- the failure mode of every
            hand-written ``data + sigma * normal`` sitting beside a likelihood
            that carries its own sigma.
        observed: the data this node is conditioned on, or ``None`` if the
            node is latent.
    """

    dist_fn: Callable[..., Any]
    observed: jax.Array | None

    @property
    def is_latent(self) -> bool:
        """Whether this node is inferred rather than conditioned on."""
        return self.observed is None
