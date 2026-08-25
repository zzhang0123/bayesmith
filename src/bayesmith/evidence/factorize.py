"""Which latents survive an epoch, and which are integrated away inside it.

rheplicant declares this: a user tags each latent ``scope="global"``,
``"per_epoch"`` or ``"linked"``, and a ``Factorization`` partitions the space
by those tags. The migration spec's argument for rewriting B11 rather than
transplanting it is that bayesmith's graph already carries plates, so the
partition can be **derived** -- "which kills the whole error class
`factorize.py` exists for (the same space declared twice)".

**That argument is two-thirds right, and this module is written to the
measured version of it.** ``docs/probes/probe_10_b11_scope_derivation.py``:

* ``global`` and ``per_epoch`` really are a plate-membership question --
  ``node.plate`` is empty for one and names the epoch plate for the other.
* ``linked`` is expressible, and reaches the exact path in its NON-CENTRED
  form (iid innovations through a deterministic linear recursion). But in
  that form the innovations are per-epoch and the chain lives in a
  ``Deterministic`` node, so **``node.plate`` cannot tell a linked latent
  from an ordinary per-epoch one.**

Plate membership is therefore a hypothesis, not the answer, and this module
tests it rather than trusting it. :func:`epoch_leakage` asks the question the
fold actually depends on -- *does this latent touch any epoch but its own* --
and that is the same question for a linked latent, a mis-plated one, and a
model whose author simply did not think about it.

**What a leak can actually look like, measured.** Three facts about plates
narrow it, and each was found by building a fixture that did not work:

* a plated ``Deterministic``'s function sees a SCALAR for a plated parent, so
  a cross-epoch map cannot be written as a plated node at all;
* a plated node whose parents are ALL unplated is refused by the graph itself
  ("nothing to map over");
* so a leak needs a plated consumer with BOTH a plated parent and an unplated
  ``(E,)`` map. Then it broadcasts, and every epoch's observation depends on
  every epoch's latent.

The other reachable shape is an observation that is not plate-tagged at all:
its axis is an epoch axis only by the author's intention, and nothing in the
graph says so, which is exactly why it cannot be sliced. Both are refused,
and ``tests/evidence/test_factorize.py`` carries one fixture for each -- the
guard has two reachable branches and neither is decoration.

**Why it matters, measured.** A per-epoch nuisance is integrated exactly once,
inside its own epoch. One that also touches its neighbours is integrated E
times instead, and the result is a finite, plausible, WRONG evidence: on a
four-epoch fixture, folding a latent that leaks into the next epoch moves the
answer by 0.15 nats, and at sixteen and thirty-two epochs the gap runs to
several and then tens of nats with **no consistent sign**. At zero leakage
the fold is exact to the bit. Nothing downstream tests for it, which is why
this is a refusal rather than a diagnostic.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.errors import StructureError
from bayesmith.exact.block import isolate
from bayesmith.graph.graph import Graph

#: Relative leakage above which a latent is refused as not epoch-local. The
#: comparison is against the latent's OWN largest sensitivity, so a latent the
#: model barely uses is judged on the same scale as one it leans on.
LEAK_RTOL: float = 1e-10


class Factorization(eqx.Module):
    """A campaign's latents, split by the extent of data they are constant over.

    Attributes:
        epoch_plate: the plate whose axis indexes epochs.
        survivors: latents outside that plate -- constant over the whole
            campaign, so they are what a streamed term is over.
        per_epoch: latents inside it -- integrated away within their own
            epoch, and therefore never accumulated.
    """

    epoch_plate: str = eqx.field(static=True)
    survivors: tuple[str, ...] = eqx.field(static=True)
    per_epoch: tuple[str, ...] = eqx.field(static=True)


def epoch_leakage(
    graph: Graph,
    name: str,
    epoch_plate: str,
    at: dict[str, Any],
) -> float:
    """How far outside its own epoch a per-epoch latent reaches, relatively.

    ``0.0`` means the latent touches only its own epoch's observations, which
    is what makes integrating it once per epoch the same as integrating it
    once. Anything else means it does not.

    The measurement is one forward-mode jacobian of the observed predictions
    with respect to the latent -- the same map :func:`compress` will build a
    design from -- so it asks about the model the fold will actually see,
    not about a declaration.

    Returns:
        ``max |off-epoch| / max |any|`` over every observed node, or ``0.0``
        if the latent moves nothing at all. Relative to the latent's OWN
        largest sensitivity, so a latent the model barely uses is judged on
        the same scale as one it leans on -- an absolute floor would wave
        through a leak in a weakly-used nuisance and refuse a well-used one
        for roundoff.
    """
    size = graph.plate_size(epoch_plate)
    outside = {
        latent: jnp.zeros(_domain_shape(graph, latent, epoch_plate, size))
        for latent in graph.latents
        if latent != name
    }
    outside.update({k: jnp.asarray(v) for k, v in at.items() if k != name})
    forward = isolate(graph, (name,), outside)
    jacobian = jax.jacfwd(forward)(
        {name: jnp.zeros(_domain_shape(graph, name, epoch_plate, size))}
    )

    worst, largest = 0.0, 0.0
    for observed, block in jacobian.items():
        entry = jnp.asarray(block[name])
        if not bool(jnp.all(jnp.isfinite(entry))):
            # Refused by NAME rather than folded into the score, because
            # Python's `max` LOSES a NaN: `max(0.0, nan)` is `0.0`, since it
            # returns the first argument when `nan > 0.0` is False. Measured
            # on a model whose per-epoch map is `1/eps`, evaluated at zero --
            # the jacobian is `nan`, both running maxima stayed `0.0`, and the
            # latent was reported as perfectly epoch-local and accepted.
            raise StructureError(
                f"the sensitivity of {observed!r} to {name!r} is not finite at "
                "the point this check evaluates, so whether the latent is "
                "epoch-local cannot be decided -- and a non-finite entry is "
                "LOST by a running maximum rather than propagated, which would "
                "report it as perfectly local. The model's map is singular "
                "there; `check_linearity` is what settles whether it can be "
                "folded at all."
            )
        node = graph.node(observed)
        if epoch_plate not in node.plate:
            # An observed node outside the epoch plate is touched by every
            # epoch at once; a per-epoch latent reaching it is leakage by
            # definition, and there is no diagonal to compare against.
            largest = max(largest, float(jnp.max(jnp.abs(entry))))
            worst = max(worst, float(jnp.max(jnp.abs(entry))))
            continue
        # Row epoch axis first, column epoch axis last: `isolate` returns the
        # observed node's own shape, and `jacfwd` appends the latent's.
        flat = jnp.reshape(entry, (size, -1, size))
        rows = jnp.arange(size)
        diagonal = flat[rows, :, rows]
        off = jnp.sum(jnp.abs(flat)) - jnp.sum(jnp.abs(diagonal))
        largest = max(largest, float(jnp.max(jnp.abs(entry))))
        worst = max(worst, float(off))
    return 0.0 if largest == 0.0 else worst / largest


def _domain_shape(
    graph: Graph, name: str, epoch_plate: str, size: int
) -> tuple[int, ...]:
    node = graph.node(name)
    return (size,) if epoch_plate in node.plate else ()


def factorize(
    graph: Graph,
    epoch_plate: str,
    *,
    at: dict[str, Any] | None = None,
    rtol: float = LEAK_RTOL,
) -> Factorization:
    """Derive the campaign's factorization from the graph, and check it.

    Plate membership gives the partition; :func:`epoch_leakage` decides
    whether it is TRUE. Both halves are needed and the second is the one that
    can fail: a latent under the epoch plate that reaches its neighbours'
    data would be integrated once per epoch instead of once, and the wrong
    answer is finite and plausible.

    Args:
        graph: the model.
        epoch_plate: the plate whose axis indexes epochs.
        at: values for the latents this check holds fixed. Defaults to zeros,
            which is enough because the leakage of an affine model does not
            depend on the point -- and a model that is not affine in its
            per-epoch latents cannot be folded at all, which
            :func:`~bayesmith.exact.linearity.check_linearity` is what says.
        rtol: relative leakage above which a latent is refused.

    Raises:
        StructureError: if the plate is not in the graph; if no latent
            survives the epoch, which would leave nothing to accumulate; or
            if a per-epoch latent is not epoch-local.
    """
    if epoch_plate not in {p.name for p in graph.plates}:
        raise StructureError(
            f"graph has no plate {epoch_plate!r}; it has "
            f"{sorted(p.name for p in graph.plates)}. The epoch plate is what "
            "makes a campaign a campaign -- without one there is nothing to "
            "stream over."
        )
    survivors = tuple(
        name for name in graph.latents if epoch_plate not in graph.node(name).plate
    )
    per_epoch = tuple(
        name for name in graph.latents if epoch_plate in graph.node(name).plate
    )
    if not survivors:
        raise StructureError(
            f"every latent of this graph lives inside {epoch_plate!r}, so "
            "nothing survives an epoch and a streamed term would be over no "
            "parameters. A campaign needs at least one latent the epochs share "
            "-- otherwise each epoch is its own independent analysis and there "
            "is nothing to accumulate."
        )

    # BEFORE the leakage probe, because the probe is what cannot survive this.
    # `epoch_leakage` takes one `jacfwd` of an `isolate` over survivors AND
    # per-epoch latents together, and a survivor plated on some other axis has
    # the wrong rank for that vmap. The model was already refused -- by
    # `ValueError: vmap was requested to map its argument along axis 0 ...`
    # raised out of `graph/evaluate.py`, naming no latent, no plate and
    # nothing the author could act on. Right verdict, unusable reason: the
    # same shape `check_precision` was caught in when it refused an
    # indefinite kernel via NaN and reported "the log-density is not
    # quadratic" about a perfectly quadratic density.
    #
    # Note this is NOT "the graph may have only one plate". A second plate
    # with no latent of its own is fine and `tests/dispatch/test_streaming.py`
    # depends on it -- that fixture is how the ambiguity branch is reachable
    # at all. What cannot be folded is a survivor that is itself plated.
    elsewhere = [
        (name, graph.node(name).plate) for name in survivors if graph.node(name).plate
    ]
    if elsewhere:
        listed = "; ".join(
            f"{name!r} is plated on {list(plates)}" for name, plates in elsewhere
        )
        raise StructureError(
            f"a campaign streamed over {epoch_plate!r} needs its survivors to be "
            f"constant across the whole campaign, but {listed}. The leakage check "
            "evaluates the survivors and the per-epoch latents together over the "
            "epoch axis, and a latent carrying a second plate's axis cannot be "
            "mapped over this one. Fold the other plate into this latent's own "
            "shape, or analyse the two plates as separate campaigns."
        )

    for name in per_epoch:
        leak = epoch_leakage(graph, name, epoch_plate, at or {})
        # `not <=` so a NaN is refused too. **A SHAPE, not a live guard, and
        # the mutation survives** -- the finiteness check inside
        # `epoch_leakage` refuses every non-finite entry before a score is
        # formed, so `leak > rtol` is equivalent here. Written the strong way
        # anyway: this is the fifth refusal in this family (rheplicant counted
        # three, `sqrtinfo.marginalise` was the fourth), and a weak one is how
        # the next reader copies the weak shape somewhere it IS reachable.
        # Recorded rather than papered over.
        if not leak <= rtol:
            raise StructureError(
                f"latent {name!r} sits inside {epoch_plate!r} but reaches other "
                f"epochs' observations: relative leakage {leak:.3e} against "
                f"rtol {rtol:.1e}. A per-epoch latent is integrated exactly "
                "ONCE, inside its own epoch; one that touches its neighbours "
                "would be integrated once per epoch instead, and the resulting "
                "evidence is finite, plausible and wrong -- measured at 0.15 "
                "nats over four epochs and tens of nats over thirty-two, with "
                "no consistent sign. Nothing downstream tests for it.\\n\\n"
                "The usual cause is a LINKED latent written non-centred: iid "
                "innovations through a deterministic recursion look exactly "
                "like a per-epoch nuisance to the plate, and reach every later "
                "epoch. Such a chain is a survivor, not a per-epoch latent -- "
                "declare it outside the epoch plate."
            )
    return Factorization(
        epoch_plate=epoch_plate, survivors=survivors, per_epoch=per_epoch
    )
