"""Whether a graph's exact block can be executed one epoch at a time.

``evidence/`` answers "fold this campaign"; it does not answer "is this a
campaign, and on which axis". That second question is a dispatch question --
the same shape as "which latents does an exact method apply to" that
:mod:`~bayesmith.dispatch.classify` already answers -- and until this module
existed nothing asked it. The evidence layer was complete, validated against
dense oracles at every level and cross-checked bitwise against rheplicant's
``inference.sqrtinfo``, with **no caller anywhere in this package**.

**Not a new method, and not a new kind.** Measured on the four-epoch scalar
campaign: ``compile()`` already accepts such a graph and plans one joint
``GCR exact`` block over ``{g, n}`` -- the survivor and all E copies of the
per-epoch nuisance at once. The streamed route solves the SAME problem, so
the plan gains a note rather than a branch::

    compile()->estimate()       g = -0.10616253398516946   (8.8e-15 from oracle)
    compress_campaign + prior   g = -0.1061625339851607    (5.6e-17 from oracle)
    dense marginal oracle       g = -0.10616253398516065

What differs is scaling, not the answer: the joint block's solve grows with
``1 + E`` latents, and the fold holds one epoch at a time.

**The route is derived by TRYING, never by guessing.** A plate is an epoch
plate only if the evidence layer's own structural checks accept it, so this
module runs those checks and forwards their refusals verbatim. It deliberately
declines to pick when more than one plate qualifies: which axis indexes
epochs is a modelling fact the graph does not carry, and a dispatcher that
guessed would be right until it silently was not.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.block import isolate, unchecked_operator
from bayesmith.graph.graph import Graph
from bayesmith.marginal.campaign import epoch_observation
from bayesmith.marginal.factorize import factorize

__all__ = ["StreamingRoute", "streaming_route"]


class StreamingRoute(eqx.Module):
    """Whether a campaign fold applies to this graph, and on which plate.

    Attributes:
        plate: the epoch plate, or ``None`` when no plate qualifies or more
            than one does. ``None`` is never "we could not be bothered" --
            :attr:`refused` says which it is.
        survivors: latents outside that plate, which a streamed term is over.
            ``()`` when there is no route.
        per_epoch: latents inside it, integrated away within their own epoch.
        refused: ``(plate name, why not)`` per rejected candidate, in plate
            order, **in the evidence layer's own words**. A graph with no
            plates refuses nothing and this is empty -- an absence of
            candidates, not a rejection of any.
    """

    plate: str | None = eqx.field(static=True)
    survivors: tuple[str, ...] = eqx.field(static=True)
    per_epoch: tuple[str, ...] = eqx.field(static=True)
    refused: tuple[tuple[str, str], ...] = eqx.field(static=True)

    @property
    def available(self) -> bool:
        """Whether ``compress_campaign(graph, route.plate)`` will run.

        Both directions are pinned by
        ``tests/dispatch/test_streaming.py::TestTheRouteDoesNotLie`` against
        the real call, because a route that reports a plate the fold then
        refuses is worse than no route at all.
        """
        return self.plate is not None

    def line(self) -> str:
        """The one line :class:`~bayesmith.dispatch.plan.InferencePlan` prints.

        Empty when there is nothing to say, so a graph that is not a campaign
        prints byte-identically to what it printed before this module
        existed -- which is what keeps the twenty existing assertions on
        ``str(plan)`` measuring what they were written to measure.
        """
        if self.available:
            return (
                f"streaming: campaign fold available on plate {self.plate!r} "
                f"over {list(self.survivors)}; "
                f"{len(self.per_epoch)} per-epoch latent(s) integrated in place"
            )
        return ""


def _forward_runs(graph: Graph, names: tuple[str, ...]) -> None:
    """Evaluate the graph once, the way ``epoch_terms`` does before folding.

    THE THIRD structural check, and the one that had to be found by trying:
    ``factorize`` probes leakage ``for name in per_epoch``, so a campaign
    with NO per-epoch latent evaluates nothing at all and an
    evaluation-time refusal reaches no one. Measured -- a plated ``det``
    whose only parent is the unplated survivor passes ``factorize`` and
    ``epoch_observation`` cleanly, and then ``compress_campaign``
    raises ``GraphError: deterministic node 'm' is in plate 'epoch' but none
    of its parents are``. Before this ran, ``streaming_route`` reported that
    graph as ``available`` and the fold refused it.

    Deliberately the forward pass ALONE -- no ``jacfwd`` -- because the
    jacobian is the expensive half and every refusal reachable here is
    structural, raised while the plated nodes are applied.
    """
    domain = unchecked_operator(graph, names, {})
    zeros = {
        name: jnp.zeros(domain.shape[name], domain.dtype[name]) for name in names
    }
    isolate(graph, names, {})(zeros)


def streaming_route(graph: Graph, *, at: dict[str, Any] | None = None) -> StreamingRoute:
    """Try every plate, and report which one the evidence layer accepts.

    Runs the two STRUCTURAL checks ``epoch_terms`` makes before it folds --
    :func:`~bayesmith.marginal.factorize.factorize` (plate membership, then
    ``epoch_leakage`` to test it) and "exactly one observed node is plated on
    this axis". Running only the first would produce a route that reports a
    plate ``compress_campaign`` then refuses, which is the failure this
    module's own tests are built around.

    It does NOT run the fold, so the one refusal it cannot anticipate is
    ``_refuse_unconstrained_epochs`` -- a rank verdict on the assembled
    pivots, which exists only once the arithmetic has been done. That is
    stated rather than hidden: :attr:`StreamingRoute.available` promises the
    structural checks pass, which is what a plan can honestly record.

    Cost is flat in the campaign length -- measured 0.278 s at E=4 and 0.342 s
    at E=512, against ``compile()``'s own 1.0-1.3 s -- because the leakage
    probe is one ``jacfwd`` of the whole plate, not one per epoch. A graph
    with no plates costs nothing: the loop below does not run.

    Args:
        graph: the model.
        at: values the leakage check holds fixed, forwarded to ``factorize``.
            Defaults to zeros, which is enough because an affine model's
            leakage does not depend on the point, and a model that is not
            affine in its per-epoch latents cannot be folded at all.

    Returns:
        A :class:`StreamingRoute`. Does not raise for a graph the evidence
        layer declines OR one it cannot analyse: a refusal is the ANSWER
        here, not an error, because ``compile()`` calls this on every graph
        and must not start rejecting graphs it accepts today. The two are
        recorded distinguishably -- see the ``ValueError`` clause below.
    """
    accepted: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    refused: list[tuple[str, str]] = []
    for name in sorted(p.name for p in graph.plates):
        try:
            found = factorize(graph, name, at=at)
            epoch_observation(graph, name)
            _forward_runs(graph, found.survivors + found.per_epoch)
        except (StructureError, GraphError) as refusal:
            # Forwarded verbatim, not restated. This sentence is what tells a
            # user what to change about their model, and a second spelling of
            # it here is the copy that would go stale.
            refused.append((name, str(refusal)))
        except ValueError as unanalysable:
            # CONTAINMENT, and no longer the handler for the case that
            # prompted it. It was added because a second plate made
            # `epoch_leakage`'s jacfwd raise a bare jax `ValueError` -- it
            # isolates survivors and per-epoch latents together and a
            # survivor plated on another axis has the wrong rank for the
            # vmap. `factorize` now refuses that early and BY NAME, so what
            # arrives here is a `StructureError` like any other refusal, and
            # this branch has no known reachable input.
            #
            # Kept anyway, and this is the whole argument: `compile()` calls
            # `streaming_route` on EVERY graph, so one escaping exception
            # stops the dispatcher planning a model it otherwise plans
            # perfectly well -- which is exactly the regression this branch
            # was written to fix, and it was found by building the graph
            # rather than by reasoning about it. Probed for a second route
            # in (a vector nuisance, a vector survivor): both factorize
            # cleanly, so the search for one came up empty rather than
            # unattempted.
            #
            # A SHAPE, not a live guard -- the same status
            # `sqrtinfo.marginalise` and `factorize` record for their
            # NaN-safe pivot spellings, and recorded here for the same
            # reason: an undocumented unreachable branch is the one a later
            # reader deletes or, worse, copies somewhere it matters.
            # Deliberately narrow: `Exception` here would swallow a
            # `TypeError` from a future refactor and report it as a model's
            # problem.
            why = (
                "could not be analysed by the evidence layer, which reported "
                f"this in jax's words rather than its own: {unanalysable}. "
                "No known model reaches this: every refusal the layer has a "
                "sentence for arrives as a StructureError instead. Please "
                "report the model, because the layer owes it a verdict of "
                "its own."
            )
            refused.append((name, why))
        else:
            accepted.append((name, found.survivors, found.per_epoch))

    if len(accepted) == 1:
        plate_name, survivors, per_epoch = accepted[0]
        return StreamingRoute(
            plate=plate_name,
            survivors=survivors,
            per_epoch=per_epoch,
            refused=tuple(refused),
        )
    if len(accepted) > 1:
        # DECLINED, not chosen. Which axis indexes epochs is a modelling fact
        # the graph does not carry: two plates can both factorize cleanly and
        # only one of them is the campaign. Picking the first would be right
        # on every fixture anyone happens to write and wrong in the field,
        # with a finite plausible answer to show for it -- the same failure
        # mode `epoch_leakage` exists to catch one level down.
        names = [name for name, _, _ in accepted]
        why = (
            f"{len(names)} plates factorize cleanly ({names}); which one indexes "
            "epochs is a modelling fact this graph does not carry. Name it "
            "yourself: compress_campaign(graph, <plate>)."
        )
        return StreamingRoute(
            plate=None,
            survivors=(),
            per_epoch=(),
            refused=tuple(refused) + tuple((name, why) for name in names),
        )
    return StreamingRoute(
        plate=None, survivors=(), per_epoch=(), refused=tuple(refused)
    )
