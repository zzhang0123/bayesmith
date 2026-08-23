"""The three node types a bayesmith graph is built from.

A node is an :class:`equinox.Module`, so a graph is a pytree and the
parameters carried by its operators are differentiable leaves for free.

**Why ``fn`` and ``dist_fn`` are not static fields.** They must accept an
``eqx.Module`` -- a whole rheplicant ``Pipeline`` is the motivating case --
whose array leaves have to stay traceable so gradients reach the parameters
inside them. A plain lambda placed in the same field simply becomes a
non-array leaf, which ``eqx.filter_jit`` routes to the static side. Marking
these fields ``static=True`` would not raise on the Module case: equinox
does not refuse a JAX array in a static field, it only warns. Construction
succeeds, the whole module is absorbed into pytree aux data, and
``eqx.filter_grad`` then silently returns each parameter's *original* value
in place of a gradient -- nothing raises, the answer is simply wrong. That
is the stronger argument for keeping these fields non-static: not a
constructor error to catch, but a silent wrong answer.

**A second gradient-loss trap, independent of the first.** Even with
``dist_fn`` correctly non-static, *how* a parameterised module is handed to
it matters, and equinox does not raise on the wrong choice either. Measured
against equinox 0.13.8 (this package's pinned version -- re-measure if you
upgrade, since the second bullet below is equinox-internal behaviour, not a
bayesmith guarantee):

* ``dist_fn=model`` (the module itself, invoked through its ``__call__``)
  -- gradient reaches every leaf of ``model``. This is the case the first
  trap above is about, and the case
  ``test_a_module_dist_fn_exposes_its_parameters_as_traceable_leaves``
  pins.
* ``dist_fn=model.some_method`` (a *bound method*, obtained by ordinary
  attribute access on the module instance) -- **also correct**, but only
  because ``equinox.Module.__getattribute__`` intercepts non-dunder method
  access and substitutes its own ``equinox.BoundMethod``: itself an
  ``eqx.Module`` that stores ``__self__`` (``model``) as a genuine,
  non-static subnode. ``model``'s array leaves stay reachable through the
  bound method exactly as they would through the model itself. Reaching
  for the dunder explicitly (``model.__call__``) does **not** get this
  treatment -- equinox's wrapping excludes dunder names -- so that
  spelling behaves like the next bullet instead, not like this one.
* a **plain closure** over the module (``def dist_fn(loc): return
  model(loc)``, or an equivalent lambda) -- **silently loses the
  gradient**. A closure is an ordinary ``types.FunctionType``: JAX's
  pytree machinery has no rule for looking inside its ``__closure__``
  cells, so the whole closure is one opaque, non-array leaf.
  ``eqx.is_inexact_array`` is ``False`` for it, so ``eqx.partition`` routes
  it to the static side and ``eqx.filter_grad`` never differentiates
  through it -- the corresponding position in the returned gradient
  pytree comes back ``None``.

In every case the *forward* value is byte-identical, because the closure
still executes with the model's real (correct) value -- only the gradient
differs, silently. The parameter simply never updates during fitting,
forever, and nothing signals it. The tests immediately after
``test_a_module_dist_fn_exposes_its_parameters_as_traceable_leaves`` in
``tests/test_nodes.py`` pin all three spellings, so a future change to
equinox's bound-method handling, or to this calling convention, shows up
as a deliberate decision instead of a surprise.
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


class Support(eqx.Module):
    """Marker for a :class:`Probabilistic` node's support.

    Structural axis for P4's discrete-enumeration dispatch (design doc §1.1:
    ``support: Support = Continuous | Discrete(n)``). All-static, like
    :class:`~bayesmith.graph.graph.Plate` -- a closed, hashable marker type
    rather than an unconstrained value, so a caller cannot accidentally put
    something unhashable (worst case, a JAX array) into
    :attr:`Probabilistic.support`, which is exactly the kind of static-field
    misuse this package's own docstrings elsewhere warn does not raise.
    """


class Continuous(Support):
    """The node's support is the reals, or an interval of them."""


class Discrete(Support):
    """The node's support is ``n`` known, finite states."""

    n: int = eqx.field(static=True)


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
        support: :class:`Continuous`, :class:`Discrete` with a known state
            count, or ``None`` if undeclared. A **claim about the model**,
            in the same sense as :attr:`Deterministic.linear_in` -- nothing
            in P1 reads it; it is recorded here so the declaration exists
            from the start. Defaults to ``None`` rather than guessing
            ``Continuous``: every dist_fn in this package's own test suite
            today happens to be continuous, but encoding that as a default
            claim would be exactly the kind of unverified assertion this
            package's dispatch axes exist to never make. A future
            dispatcher must treat ``None`` as ineligible for any
            support-specific method and fall through to NUTS, not as a
            claim of continuity.
        depends_on_prediction: whether this node's distribution depends on
            the value it is predicting (ported from rheplicant's
            ``NoiseModel.depends_on_prediction``: ``False`` means the
            iterative reweighting loop can be skipped). A claim about the
            model, like ``support`` above -- nothing in P1 reads it.
            Defaults to ``True`` (assume dependence), not ``False``, so an
            undeclared node can never cause a future dispatcher to skip a
            step it actually needed -- the same "claim nothing, unlock no
            shortcut" reasoning behind ``linear_in``'s empty-tuple default
            and ``support``'s ``None`` default above, applied to a boolean
            gate instead of a name set or a closed type.
    """

    dist_fn: Callable[..., Any]
    observed: jax.Array | None
    support: Support | None = eqx.field(static=True, default=None)
    depends_on_prediction: bool = eqx.field(static=True, default=True)

    @property
    def is_latent(self) -> bool:
        """Whether this node is inferred rather than conditioned on."""
        return self.observed is None
