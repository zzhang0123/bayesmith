"""Log space as a graph transform: multiplicative noise made conjugate.

Two observation models produce data whose LOG is Gaussian about an affine
prediction, and they arrive in different clothes:

* **multiplicative Gaussian** -- ``d = mu (1 + f w)``, ``w ~ N(0, 1)``: the
  radiometer equation's own form, declared as ``Normal(mu, f |mu|)``. Taking
  logs, ``log d = log mu + log(1 + f w)`` and ``log(1 + f w) -> N(-f^2/2,
  f^2)`` to first order -- an approximation whose size is measured below.
* **log-Gaussian** -- ``d = mu exp(s w)``, declared as ``LogNormal(ell, s)``
  with ``ell`` the log-location: here ``log d = ell + s w`` EXACTLY, with no
  correction term and no threshold.

Either way, in log space the noise is additive AND its scale no longer
depends on the prediction -- so a latent the prediction is ``exp``-affine in
(a gain entered as ``exp(log_gain)``) has an exact conjugate block there,
where in the original space it had only NUTS; and the reweighting fixed point
:func:`~bayesmith.exact.gls.iterative_gls` exists to find has nothing left to
iterate on.

**The implementation is a graph-to-graph transform, and that is the design.**
:func:`log_space` rewrites each observed node into an ordinary ``Normal`` in
log space -- ``loc`` becomes ``log`` of the old prediction (or the LogNormal's
own ``ell``, which already is one), ``scale`` becomes the constant ``f`` (or
``s``), ``observed`` becomes ``log d`` (plus ``f^2/2`` in the first-order
case), and ``depends_on_prediction`` becomes ``False`` because it genuinely
is. Everything downstream is then STOCK: the per-element affinity check with
its two criteria and roundoff floors, ``unchecked_operator``,
``wiener_solve``/``gcr_sample``, ``precision_at``, the Gibbs machinery -- none
of it knows or needs to know that the graph it was handed carries logs. One
transform, zero re-implementations, and the checking machinery that catches a
false claim is the same one that catches it in linear space.

**Which scenario a node is, is probed rather than declared.** The transform
evaluates each observed node's distribution at two points of the latents'
own priors: a ``LogNormal`` is the exact case by type; a ``Normal`` whose
``scale / loc`` is the same at both points (while ``loc`` itself moved) is
multiplicative with that ratio as ``f``; a ``Normal`` whose scale did not
track the prediction is additive, and log space is refused for it --
:class:`~bayesmith.errors.NotLogLinear` -- because a transform that does not
simplify the noise is just a different, wrong likelihood.

**The first-order approximation, and its size.** ``E[log(1 + f w)] =
-f^2/2`` to leading order and ``Var[log(1 + f w)]`` exceeds ``f^2``.
Measured over 2e7 draws:

======  ==================  =====================
``f``   ``Var / f^2 - 1``   ``mean / (-f^2/2)``
======  ==================  =====================
0.004   below the MC floor  1.00
0.06    0.0088              1.006
0.10    0.0258              1.016
0.30    0.3983              1.185
======  ==================  =====================

:data:`FIRST_ORDER_MAX_FRACTIONAL` refuses above 0.06, where the variance
taken as ``f^2`` is 1 % wrong. A radiometer at 61 kHz x 1 s has
``f = 4.05e-3`` -- fifteen times below the refusal -- so the threshold is
not a limit met by observing; reaching it means a channel width or an
integration time is not what was intended. The ``LogNormal`` route has no
threshold at all, which is the reason to declare one when the noise really
is log-Gaussian.

**Positivity is refused eagerly, never propagated.** ``log`` of a
non-positive value is NaN, and ``nan > rtol`` is ``False`` -- so a NaN
departure would read as PASSING the affinity check, and a NaN residual as a
converged solve. Data is checked at transform time on concrete values;
a prediction that goes non-positive at a probe surfaces through the stock
non-finite-baseline refusal.

Ported from ``rheplicant.inference.loglinear`` and re-founded on the graph:
what was there a parallel implementation (its own probe loop, its own
operator export) is here one transform in front of the existing machinery --
which also upgrades the checking, since the stock check is per-element with
a sigma-weighted second criterion and the rheplicant original was neither.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith.errors import NotLogLinear, StructureError
from bayesmith.exact.block import LinearBlock, _env_before, unchecked_operator
from bayesmith.exact.gaussian import node_shape, unwrap
from bayesmith.exact.linearity import check_linearity
from bayesmith.graph.evaluate import apply_probabilistic, evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic

#: Largest fractional noise for which the first-order log-space equivalence
#: is used. Set where the log-space VARIANCE, taken to be ``f^2``, is 1 %
#: wrong: measured 0.0088 at 0.06 against 0.0161 at 0.08. Above it the
#: correction this module applies is itself leading-order and stops being
#: enough -- at f = 0.3 the variance is 40 % low and a 90 % interval covers
#: 83 %. The ``LogNormal`` scenario is exact and never consults this.
FIRST_ORDER_MAX_FRACTIONAL: float = 0.06

#: Probe magnitudes for the log-space affinity check, as multiples of each
#: latent's own prior width. Deliberately NOT
#: :data:`~bayesmith.exact.linearity.DEFAULT_SCALES`, whose top entry is
#: ``1e3``: a log-latent's prior width is order 0.1-1, so a 1e3-sigma probe
#: puts hundreds into an exponential, which overflows float32 near 88 -- and
#: the check would then refuse a map that is exactly affine in log space,
#: reporting the dtype rather than the model. Four decades of probe are kept,
#: which is what the stock check's own docstring says the sweep is for.
LOG_DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)

#: Relative tolerance for "the same fractional level at both probe points".
#: Generous against roundoff in ``scale / loc`` and far below any real
#: departure: an additive-noise node's ratio changes by the same factor the
#: prediction does, which the probe moves by order one.
_FRACTIONAL_RTOL: float = 1e-6


@dataclasses.dataclass(frozen=True)
class LogSpace:
    """A graph taken to log space, with the reading that justified it.

    Attributes:
        graph: the transformed model. Every observed node is an ordinary
            ``Normal`` whose ``loc`` is the log-prediction, whose ``scale``
            is constant in the latents (``depends_on_prediction=False``, and
            genuinely so), and whose ``observed`` is the log data, shifted by
            ``f^2/2`` where the first-order route applied. Latents, priors,
            deterministic structure and plates are untouched -- which is why
            every downstream consumer takes this graph verbatim.
        kind: ``{observed: "lognormal" | "multiplicative"}`` -- which scenario
            each node was read as. ``"lognormal"`` is exact;
            ``"multiplicative"`` carries the first-order caveat the module
            docstring sizes.
        fractional: ``{observed: f}`` for the multiplicative nodes, absent
            for the exact ones. Broadcast to the node's shape, so a
            channel-dependent bandwidth is representable.
        skipped: ``{observed: why}`` for nodes left UNTRANSFORMED -- additive
            noise, a distribution the route does not read. A skipped node
            keeps its original form in :attr:`graph`, which is harmless to a
            log block that does not reach it (its design column there is
            zero) and disqualifying for one that does --
            :func:`~bayesmith.dispatch.factor.factor_partition` checks
            reachability against exactly this dict.
    """

    graph: Graph
    kind: dict[str, str]
    fractional: dict[str, jax.Array]
    skipped: dict[str, str]


def multiplicative_log_data(
    observed: jax.Array, fractional: jax.Array | float
) -> tuple[jax.Array, jax.Array]:
    """``(log d + f^2/2, f)`` -- the first-order transform, on bare arrays.

    The two halves of one claim about the noise, returned together so a
    caller cannot take the shift without the sigma that goes with it. The
    ``f^2/2`` adds back ``E[log(1 + f w)]`` to leading order; it is a
    CONSTANT -- the same for every sample, independent of the prediction --
    so adding it is exact arithmetic rather than an estimate.

    Positivity of ``observed`` is the CALLER's obligation here, stated
    rather than checked: this is the seam another package's own data
    handling (masks, flags) delegates its arithmetic to, and a check on this
    side would force that caller to pre-fill values it is about to weight by
    zero anyway. :func:`log_space`, which owns whole graphs, does check.
    """
    fractional = jnp.asarray(fractional)
    y = jnp.log(observed) + fractional**2 / 2.0
    return y, jnp.broadcast_to(fractional.astype(y.dtype), jnp.shape(y))


def _refuse_bad_data(name: str, observed: jax.Array) -> None:
    """Refuse data ``log`` cannot be taken of, on concrete values, by name."""
    values = jnp.asarray(observed)
    bad = int(jnp.sum(~(values > 0) | ~jnp.isfinite(values)))
    if bad:
        worst = float(jnp.min(jnp.where(jnp.isfinite(values), values, jnp.inf)))
        raise NotLogLinear(
            f"observed node {name!r} has {bad} of {int(jnp.size(values))} values "
            f"that are non-positive or non-finite (smallest finite {worst:.6g}), "
            "so log of the data does not exist there. log() of such a value is "
            "NaN, and NaN fails every comparison -- a NaN departure would read "
            "as PASSING the affinity check -- so this is refused here, on "
            "concrete values, rather than propagated. A model whose data can "
            "genuinely reach zero is not log-Gaussian; one whose bad samples "
            "are unobserved should drop or impute them before the transform.",
            reason="data_not_positive",
            node=name,
        )


def _two_environments(graph: Graph) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluation environments at the prior centres, and displaced from them.

    The displacement is 0.7 of each latent's own prior width -- inside the
    prior's bulk, so the probe asks about the model where inference will
    actually run, and big enough that a prediction which depends on the
    latents at all moves measurably between the two.
    """
    _, domain = _env_before(graph, tuple(sorted(graph.latents)), {})
    centres = {name: domain[name][2] for name in domain}
    displaced = {
        name: domain[name][2] + 0.7 * domain[name][3] for name in domain
    }
    return evaluate(graph, centres), evaluate(graph, displaced)


def _log_normal_node(node: Probabilistic) -> Probabilistic:
    """The exact scenario: ``LogNormal(ell, s)`` becomes ``Normal(ell, s)``.

    ``ell`` IS the log-location, so no log is taken of the prediction --
    log-linearity of this node is affinity of ``ell`` itself, which is what
    the stock check will now measure.
    """
    old = node.dist_fn

    def dist_fn(*values: Any) -> Any:
        found = unwrap(old(*values))
        return dist.Normal(found.loc, found.scale)

    return Probabilistic(
        name=node.name,
        parents=node.parents,
        plate=node.plate,
        dist_fn=dist_fn,
        observed=jnp.log(jnp.asarray(node.observed)),
        support=None,
        depends_on_prediction=False,
        # Carried, not re-derived. The log transform changes the DATA and the
        # scale; it does not change which samples were taken, and a node that
        # silently lost its mask here would solve with the flagged channels
        # back in, finitely and wrongly.
        observed_mask=node.observed_mask,
    )


def _multiplicative_node(
    node: Probabilistic, fractional: jax.Array
) -> Probabilistic:
    """The first-order scenario: ``Normal(mu, f mu)`` becomes ``Normal(log mu, f)``."""
    old = node.dist_fn

    def dist_fn(*values: Any) -> Any:
        found = unwrap(old(*values))
        return dist.Normal(jnp.log(found.loc), fractional)

    y, _ = multiplicative_log_data(jnp.asarray(node.observed), fractional)
    return Probabilistic(
        name=node.name,
        parents=node.parents,
        plate=node.plate,
        dist_fn=dist_fn,
        observed=y,
        support=None,
        depends_on_prediction=False,
        observed_mask=node.observed_mask,
    )


def _read_scenario(
    graph: Graph,
    node: Probabilistic,
    first: dict[str, Any],
    second: dict[str, Any],
) -> tuple[str, jax.Array | None]:
    """Which log-space scenario ``node`` is, probed at two environments.

    Raises :class:`~bayesmith.errors.NotLogLinear` -- naming the reason, as
    that class promises -- where there is none.
    """
    found = unwrap(apply_probabilistic(graph, node, first))

    if isinstance(found, dist.LogNormal):
        return "lognormal", None

    if not isinstance(found, dist.Normal):
        raise NotLogLinear(
            f"observed node {node.name!r} returns "
            f"{type(found).__name__}; the log-space route reads a "
            "multiplicative Normal -- Normal(mu, f mu) -- or a LogNormal. "
            "Anything else has no log-Gaussian reading to transform to.",
            reason="not_gaussian_family",
            node=node.name,
            found=type(found).__name__,
        )

    shape = node_shape(graph, node, first)
    loc_1 = jnp.broadcast_to(jnp.asarray(found.loc), shape)
    scale_1 = jnp.broadcast_to(jnp.asarray(found.scale), shape)
    again = unwrap(apply_probabilistic(graph, node, second))
    loc_2 = jnp.broadcast_to(jnp.asarray(again.loc), shape)
    scale_2 = jnp.broadcast_to(jnp.asarray(again.scale), shape)

    moved = float(jnp.max(jnp.abs(loc_1 - loc_2)))
    magnitude = float(jnp.max(jnp.abs(loc_1)))
    if not moved > 1e-12 * max(magnitude, 1e-300):
        raise NotLogLinear(
            f"observed node {node.name!r}: the prediction does not move with "
            "the latents between two points of their own priors, so whether "
            "its scale tracks the prediction cannot be measured -- and a "
            "block whose prediction ignores the latents has nothing for any "
            "exact method, log-space or otherwise, to solve.",
            reason="prediction_static",
            node=node.name,
        )
    if not (bool(jnp.all(loc_1 > 0)) and bool(jnp.all(loc_2 > 0))):
        raise NotLogLinear(
            f"observed node {node.name!r}: the prediction is not strictly "
            "positive at points of the latents' own priors, so log of it "
            "does not exist where inference would run. A log-linear model "
            "predicts exp(affine), which is positive everywhere -- a sign "
            "change says this model is not one.",
            reason="prediction_not_positive",
            node=node.name,
        )

    ratio_1 = scale_1 / loc_1
    ratio_2 = scale_2 / loc_2
    same = bool(
        jnp.all(
            jnp.abs(ratio_1 - ratio_2)
            <= _FRACTIONAL_RTOL * jnp.maximum(jnp.abs(ratio_1), 1e-300)
        )
    )
    if not same:
        constant = bool(
            jnp.all(
                jnp.abs(scale_1 - scale_2)
                <= _FRACTIONAL_RTOL * jnp.maximum(jnp.abs(scale_1), 1e-300)
            )
        )
        why = (
            "its scale is CONSTANT while the prediction moves -- additive "
            "Gaussian noise. Log space would not simplify it; the transform "
            "would just state a different likelihood from the one declared"
            if constant
            else "its scale moves with the latents but not proportionally to "
            "the prediction, so it is neither additive nor multiplicative "
            "and no log-space Gaussian describes it"
        )
        raise NotLogLinear(
            f"observed node {node.name!r}: {why}.",
            # The same branch that chose the sentence chooses the reason, so
            # the two cannot disagree -- which they would if the reason were
            # re-derived from `constant` a second time further down.
            reason="noise_additive" if constant else "noise_neither",
            node=node.name,
        )

    worst = float(jnp.max(ratio_1))
    if worst > FIRST_ORDER_MAX_FRACTIONAL:
        raise NotLogLinear(
            f"observed node {node.name!r} is multiplicative with fractional "
            f"level f = {worst:.4g}, above the {FIRST_ORDER_MAX_FRACTIONAL} "
            "at which the first-order log-space equivalence is still good to "
            "1 %: log(1 + f w) is treated as N(-f^2/2, f^2), and measured, "
            "its variance is 0.9 % above f^2 at f = 0.06 and 40 % above at "
            "f = 0.3, where a 90 % interval covers 83 %. For reference, "
            "f = 1/sqrt(channel_width x integration_time) is 4.05e-3 for a "
            "61 kHz channel at 1 s -- reaching this refusal by observing is "
            "not possible, so a noise declaration is mis-specified. Declare "
            "the noise LogNormal if it really is log-Gaussian; that route is "
            "exact and has no threshold.",
            reason="fractional_too_large",
            node=node.name,
            fractional=worst,
        )
    return "multiplicative", ratio_1


def log_space(graph: Graph) -> LogSpace:
    """Take the graph to log space, deciding each observed node's scenario.

    See the module docstring for the two scenarios and for why the result is
    an ordinary graph. The probe environments are two points of the latents'
    own priors, so the scenario is read where inference will actually run.

    A node with no log-Gaussian reading -- additive noise, a distribution
    the route does not read -- is left UNTRANSFORMED and recorded in
    ``skipped`` with its reason, rather than refusing the whole transform:
    on a graph with several observations, a log block only ever needs the
    ones ITS latents reach, and a caller (``factor_partition`` does) checks
    that reachability against ``skipped``. Non-positive DATA on an otherwise
    multiplicative node still refuses loudly -- that is a contradiction in
    the model, not a scenario mismatch.

    Raises:
        NotLogLinear: if NO observed node could be transformed -- then there
            is no log space to hand back at all, and the aggregated per-node
            reasons say why; or if a multiplicative node's data is
            non-positive, which no routing can repair.
        StructureError: propagated from the graph machinery where the graph
            itself is broken, which is a different statement from "no
            log-linear structure" and deliberately keeps its own class.
    """
    first, second = _two_environments(graph)
    kind: dict[str, str] = {}
    fractional: dict[str, jax.Array] = {}
    skipped: dict[str, str] = {}
    # The per-node REASONS, kept beside their sentences. Collapsing the two
    # into one string is the shape G11 exists to remove: the graph-level
    # refusal below is the only thing a caller ever sees here, so a per-node
    # verdict that survives only inside its own prose is a verdict the caller
    # would have to parse back out.
    skipped_reasons: dict[str, str] = {}
    nodes: list[Any] = []
    for node in graph.nodes:
        if not (isinstance(node, Probabilistic) and node.observed is not None):
            nodes.append(node)
            continue
        try:
            scenario, ratio = _read_scenario(graph, node, first, second)
        except NotLogLinear as refused:
            skipped[node.name] = str(refused)
            skipped_reasons[node.name] = refused.reason
            nodes.append(node)
            continue
        _refuse_bad_data(node.name, node.observed)
        kind[node.name] = scenario
        if scenario == "lognormal":
            nodes.append(_log_normal_node(node))
        else:
            assert ratio is not None
            fractional[node.name] = ratio
            nodes.append(_multiplicative_node(node, ratio))
    if not kind:
        raise NotLogLinear(
            "no observed node of this graph has a log-Gaussian reading, so "
            "there is no log space to transform to. Per node: "
            + "; ".join(f"{name}: {why}" for name, why in sorted(skipped.items())),
            reason="no_node_qualifies",
            # No `node=`: this verdict is about the graph. The per-node
            # reasons stay readable as data, which is what a caller deciding
            # "fix one node, or give up on log space" actually needs -- the
            # sentences are in the message above, for whoever is reading.
            per_node=skipped_reasons,
        )
    return LogSpace(
        # Carried, not defaulted. Only OBSERVED nodes are rebuilt above, so
        # the latents -- and therefore the block any `joint_prior` is over --
        # are the same set on both sides of the transform, and the field
        # survives `__check_init__` unchanged. It has to be named explicitly
        # because it DEFAULTS to None: `Graph(nodes=..., plates=...)` is a
        # legal call that drops it silently, and `__check_init__` inspects the
        # field only when it is not None, so nothing downstream complains. A
        # graph that lost it here would solve in log space against a different
        # posterior than the one declared -- and a block prior moves the
        # DENSITY while barely moving the mean and width that every consumer
        # of `LogSpace` reports, so the loss would not show where anyone looks.
        graph=Graph(
            nodes=tuple(nodes),
            plates=graph.plates,
            joint_prior=graph.joint_prior,
        ),
        kind=kind,
        fractional=fractional,
        skipped=skipped,
    )


def check_log_linearity(
    graph: Graph,
    names: Iterable[str],
    at: dict[str, Any] | None = None,
    *,
    scales: tuple[float, ...] = LOG_DEFAULT_SCALES,
    rtol: float | None = None,
    at_points: Any = None,
    key: jax.Array | None = None,
) -> dict[int, dict[float, float]]:
    """Verify ``log`` of every prediction is affine in a block -- or a group.

    :func:`log_space` decides the scenarios, then the STOCK
    :func:`~bayesmith.exact.linearity.check_linearity` runs on the
    transformed graph -- same per-element criteria, same roundoff floors,
    same at-point sweep over the outside latents' priors. There is no
    separate log-space probe loop to drift from the linear one.

    Args and returns as for the stock check, with one difference:
    ``scales`` defaults to :data:`LOG_DEFAULT_SCALES`, for the overflow
    reason its docstring gives.

    Raises:
        NotLogLinear: if no log-space reading exists (from
            :func:`log_space`), or if log of the prediction departs from
            affinity in these latents -- the inner check's refusal, re-raised
            under the blameless class with its message carried verbatim, so
            a dispatcher probing many latents can catch "no" narrowly.
        GraphError: propagated -- a malformed ``names`` or ``at`` is misuse,
            not a classification outcome.
    """
    ls = log_space(graph)
    try:
        return check_linearity(
            ls.graph, names, at, scales=scales, rtol=rtol, at_points=at_points, key=key
        )
    except StructureError as refused:
        raise NotLogLinear(
            "log(prediction) is not affine in "
            f"{sorted(tuple(names) if not isinstance(names, str) else (names,))}: "
            f"{refused}",
            reason="log_not_affine",
        ) from refused


def log_linear_operator(
    graph: Graph,
    names: Iterable[str],
    at: dict[str, Any] | None = None,
    *,
    scales: tuple[float, ...] = LOG_DEFAULT_SCALES,
    rtol: float | None = None,
    at_points: Any = None,
    key: jax.Array | None = None,
) -> tuple[LinearBlock, LogSpace]:
    """Check the log-linearity claim, then export the block. The entry point.

    The returned block is an ordinary
    :class:`~bayesmith.exact.block.LinearBlock` over the TRANSFORMED graph:
    its ``offset`` is the log-prediction with the block at zero, its ``data``
    is the log data (shifted by ``f^2/2`` where first-order), and the
    matching noise is ``precision_at(returned.graph, values)`` -- constant in
    the latents, which is the point. The :class:`LogSpace` comes back with
    the block because every further step (a Gibbs sweep's rebuilds, the
    precision, a NUTS comparison) runs against the transformed graph, and
    handing back the block alone would leave the caller to reconstruct it.
    """
    ls = log_space(graph)
    try:
        check_linearity(
            ls.graph, names, at, scales=scales, rtol=rtol, at_points=at_points, key=key
        )
    except StructureError as refused:
        raise NotLogLinear(
            "log(prediction) is not affine in "
            f"{sorted(tuple(names) if not isinstance(names, str) else (names,))}: "
            f"{refused}",
            reason="log_not_affine",
        ) from refused
    return unchecked_operator(ls.graph, names, at), ls


__all__ = [
    "FIRST_ORDER_MAX_FRACTIONAL",
    "LOG_DEFAULT_SCALES",
    "LogSpace",
    "check_log_linearity",
    "log_linear_operator",
    "log_space",
    "multiplicative_log_data",
]
