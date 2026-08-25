"""Which latents an exact method applies to, and how they group.

The three structural axes P1 recorded -- ``linear_in``, ``support``,
``depends_on_prediction`` -- are read here for the first time. P3a *verified*
``linear_in`` (it measured affinity); this module *reads* it, which is a
different thing: it walks the declaration along every path from a latent to
every observed node's location parameter.

**Nothing here samples, and nothing here is jittable.** Every guard it calls
-- :func:`~bayesmith.exact.gaussian.check_gaussian`,
:func:`~bayesmith.exact.linearity.check_linearity`,
:func:`~bayesmith.exact.gls.check_prediction_dependence` -- runs on concrete
values and raises, so this whole module runs once, at compile time, outside
any trace.

**Exceptions are discriminated by raise SITE, not by type.**
:class:`~bayesmith.errors.NotGaussian` is a verdict about an ordinary model
and is caught wherever it appears. :class:`~bayesmith.errors.StructureError`
is caught at exactly one call site -- ``check_linearity``'s, where it means a
false ``linear_in`` and the block falls to NUTS naming its members -- and is
deliberately allowed through everywhere else, because from ``check_gaussian``
it means a node whose own ``log_prob`` contradicts the ``loc``/``scale`` read
off it, and routing that to NUTS would hide a broken model behind an
ordinary-looking fallback.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError, NotGaussian, StructureError
from bayesmith.exact.block import (
    LinearBlock,
    _ancestors,
    domain_centre,
    unchecked_operator,
)
from bayesmith.exact.gaussian import check_gaussian, gaussian_parts, node_shape
from bayesmith.exact.gls import check_prediction_dependence, sigma_from_graph
from bayesmith.exact.linearity import DEFAULT_AT_POINTS, check_linearity
from bayesmith.exact.precision import diagonal_from
from bayesmith.exact.solve import wiener_solve
from bayesmith.graph.evaluate import apply_deterministic, apply_probabilistic
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic

SIGMA_RTOL = 1e-8
"""Relative sigma movement above which a block counts as prediction-dependent.

Matches :func:`~bayesmith.exact.gls.check_prediction_dependence`'s own
``rtol`` default, so the number a declaration is judged against and the
number a method is chosen against are the same number. **Covered at the ends,
not at the boundary**: every fixture in this package's suite reads either
bitwise ``0.0`` or at least 3.639e+00 -- eight decades above 1e-8, with
nothing anywhere near the threshold itself. (The figure of 1e+02 this
docstring carried before was never right: ``sigma_functional_block`` at a
two-member contrast has read 3.639e+00 since it was written.) That is the
same position ``check_prediction_dependence``'s docstring takes and for the
same reason -- this is a coarse yes/no movement detector, not a
numeric dispatcher between two methods that must agree at a threshold.
"""


@dataclasses.dataclass(frozen=True)
class Classification:
    """The partition of one graph, and the evidence behind it.

    Attributes:
        exact: the single exact block, sorted. Empty if there is none.
        nuts: every other latent, sorted.
        method: ``"gcr"``, ``"gcr+snis"``, ``"gcr+mh"`` or ``"nuts"``.
        reason: why -- named members on a refusal, so a user one declaration
            away from an exact solve can see that from the plan.
        linearity: ``check_linearity``'s per-at-point errors, or ``None``.
        sigma_movement: the largest relative movement of sigma with the
            block, or ``None`` if there is no block. Measured from TWO
            anchors -- prior-scale probes around the prior centre, and the
            block's own Wiener solution, which is where the data put the
            posterior. See :func:`_sigma_movement` for why one of them is not
            enough.
        sigma_needs_rebuild: whether any observed node's scale has a latent
            ANCESTOR outside the block, in which case ``noise_std`` must be
            recomputed every sweep rather than hoisted. Distinct from
            ``sigma_movement``, which only sees movement with the BLOCK.
    """

    exact: tuple[str, ...]
    nuts: tuple[str, ...]
    method: str
    reason: str
    linearity: dict[int, dict[float, float]] | None
    sigma_movement: float | None
    sigma_needs_rebuild: bool


def _latent_centre(graph: Graph, node: Probabilistic, env: dict[str, Any]) -> jax.Array:
    """One latent's centre, for an environment the classifier can run in.

    A Gaussian latent's centre is its ``loc``, read the same way the block
    machinery reads it. A latent that is not Gaussian has no ``loc`` at all --
    and the classifier still needs a value for it, because a DISQUALIFIED
    latent can sit upstream of an observed node whose Gaussianity is exactly
    what is being checked. Its own distribution's ``mean`` is the natural
    second choice, and zero is the third: ``Cauchy`` has no finite mean and
    ``ImproperUniform`` has no mean at all (measured -- ``NotImplementedError``
    from NumPyro 0.21). Both of those are live fixtures here
    (``overflowing_outside_latent``, ``improper_outside_prior``) and both are
    LEGAL models the classifier must not refuse.

    A NumPyro distribution does not know the plate its ``sample()`` named --
    ``dist.StudentT(6.0, 0.4, 0.9).shape()`` is ``()`` either way -- so that
    second choice has to be broadcast out to the plate's size by hand.
    ``plated_student_t_latent`` is the fixture; without the broadcast
    ``apply_deterministic`` raises out of ``vmap`` rather than returning a
    wrong centre, so nothing downstream papers over it.
    """
    try:
        loc, _ = gaussian_parts(graph, node, env)
        return jnp.broadcast_to(loc, node_shape(graph, node, env))
    except NotGaussian:
        pass
    distribution = apply_probabilistic(graph, node, env)
    shape = tuple(distribution.shape())
    if node.plate:
        shape = tuple(jnp.broadcast_shapes(shape, (graph.plate_size(node.plate[0]),)))
    try:
        centre = jnp.broadcast_to(jnp.asarray(distribution.mean), shape)
    except (NotImplementedError, AttributeError, TypeError, ValueError):
        return jnp.zeros(shape)
    return centre if bool(jnp.all(jnp.isfinite(centre))) else jnp.zeros(shape)


def prior_environment(graph: Graph) -> dict[str, Any]:
    """Every node's value, with each latent at the centre of its own prior.

    Repeats :func:`~bayesmith.graph.evaluate.evaluate`'s isinstance ladder for
    the same reason :func:`~bayesmith.exact.block._env_before` does: the
    values it substitutes have to be derived DURING the scan, from what is
    already in hand. A latent whose prior width is another latent's value --
    ``shared_ancestor``, ``mixed_radiometer`` -- cannot be centred before its
    ancestor has been.

    Public because :mod:`bayesmith.dispatch.plan` must anchor the block it
    BUILDS at exactly the point :func:`partition` classified it at. Two
    independent spellings of "the prior mean" would let the plan solve a
    different problem from the one that was checked.
    """
    env: dict[str, Any] = {}
    for node in graph.nodes:
        if isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = apply_deterministic(graph, node, env)
        elif isinstance(node, Probabilistic):
            env[node.name] = (
                _latent_centre(graph, node, env) if node.is_latent else node.observed
            )
        else:  # pragma: no cover - defensive, mirrors evaluate()
            raise GraphError(f"unknown node type {type(node).__name__}")
    return env


def block_at(
    graph: Graph, names: tuple[str, ...], *, env: dict[str, Any] | None = None
) -> dict[str, Any]:
    """``{outside latent: prior centre}`` -- where a block is built.

    A block is affine *given* the latents outside it, so it has to be
    anchored somewhere, and the prior's centre is the "before seeing
    anything" point. Anchoring at zero instead is not equivalent and is not
    merely a different starting guess: ``affine_only_at_zero``'s prediction is
    affine exactly where its outside latent is zero, so a zero anchor accepts
    at the caller's own at-point a model that the prior's centre refuses
    there.
    """
    env = prior_environment(graph) if env is None else env
    members = set(names)
    return {name: env[name] for name in graph.latents if name not in members}


def _all_to_nuts(latents: list[str], reason: str) -> Classification:
    """No exact block at all: every latent goes to NUTS, with the reason kept.

    Three call sites reach this -- a non-Gaussian observed node, an empty
    qualified set, and a block whose ``linear_in`` claim turned out false --
    and the six fields they would each have to spell out are the same six
    every time. One place to read them off is one place for them to be wrong.
    """
    return Classification((), tuple(sorted(latents)), "nuts", reason, None, None, False)


def _observed_ancestors(graph: Graph) -> set[str]:
    """Every node that some observed node depends on."""
    out: set[str] = set()
    for name in graph.observed:
        out |= _ancestors(graph, name) | {name}
    return out


def _relevant_deterministics(graph: Graph, name: str) -> dict[str, set[str]]:
    """``{det_name: parents the path arrives by}`` for criterion 3.

    Walks FORWARD from ``name`` through ``Deterministic`` nodes only, and
    stops at every ``Probabilistic`` one. Stopping there is the point: a path
    ``x -> v -> mu -> d`` does not make ``x`` reach ``d``'s location
    deterministically. With ``v`` latent, ``isolate`` holds ``v`` fixed; with
    ``v`` observed, ``v`` is data. Either way ``x`` does not reach ``d`` that
    way at all. (The latent case is not harmless, but it is :func:`partition`'s
    ancestor rule that handles it, not this -- which is why the fixture that
    makes this clause load-bearing, ``observation_reused_downstream``, has to
    use an OBSERVED intermediate: on a latent one the two rules reach the same
    verdict by different routes and the mutation is invisible.)

    Restricted to deterministics that are themselves ancestors of some
    observed node. One that leads nowhere contributes nothing to any
    prediction, so requiring a declaration from it would refuse a model for a
    node the solve never evaluates -- ``dangling_deterministic``.
    """
    matters = _observed_ancestors(graph)
    reached = {name}
    arrivals: dict[str, set[str]] = {}
    for node in graph.nodes:  # declaration order IS topological order
        if not isinstance(node, Deterministic):
            continue
        incoming = {p for p in node.parents if p in reached}
        if not incoming:
            continue
        reached.add(node.name)
        if node.name in matters:
            arrivals[node.name] = incoming
    return arrivals


def _declares_linear_in(graph: Graph, name: str) -> tuple[bool, str]:
    """Criterion 3: EVERY Deterministic on EVERY path declares its in-edges.

    A universal quantifier, so it is VACUOUSLY true two different ways, and
    both must be accepted: for a latent with no ``Deterministic`` anywhere on
    its path (``plated_latent``), and for one with no path to any observed
    node at all (``unconstrained_latent``). The obvious spelling -- "some
    Deterministic names this latent" -- rejects both.
    """
    for det_name, incoming in _relevant_deterministics(graph, name).items():
        undeclared = sorted(incoming - set(graph.node(det_name).linear_in))
        if undeclared:
            return False, (
                f"{det_name!r} declares linear_in="
                f"{graph.node(det_name).linear_in!r}, which does not name "
                f"{undeclared} -- and that node is on a path from {name!r} to "
                "an observed node's location"
            )
    return True, ""


def _is_gaussian(graph: Graph, name: str, env: dict[str, Any]) -> tuple[bool, str]:
    """Criterion 1 and 2. ``NotGaussian`` is a verdict; ``StructureError`` is not.

    ``NotGaussian`` means "this node is simply not a diagonal Gaussian",
    which is an ordinary property of an ordinary model and routes the node to
    NUTS. ``StructureError`` from :func:`check_gaussian` means the node's own
    ``log_prob`` contradicts the ``loc``/``scale`` read off it -- a broken
    model -- and is deliberately NOT caught here.

    **This asks a CAPABILITY question, and that is why it calls
    ``check_gaussian`` rather than
    :func:`~bayesmith.exact.gaussian.check_observed`**, which the build path
    does. The two are different questions: "is this node's declared density
    sound?" and "can the exact path solve a block containing it?". For a
    correlated node the first answer is now yes and the second is still no --
    the block builder's data and loc walks are diagonal-only -- so accepting
    it here promises an exact solve that cannot be delivered.

    Measured, by wiring ``check_observed`` here and running the suite: on a
    well-formed ``CirculantNormal`` graph, ``compile()`` stopped routing to
    NUTS and raised ``NotGaussian`` from deeper in the block builder instead.
    All 748 tests stayed green through it, because no fixture compiles a
    correlated graph -- ``test_a_correlated_graph_still_compiles_to_nuts`` is
    what closes that, and this comment is what stops the next reader
    "tidying" the two call sites into one.

    When the exact path can build a correlated block, this becomes
    ``check_observed`` and that test changes with it. Not before.

    Only the message's FIRST SENTENCE is kept: the rest says which solve is
    not implemented and that this is a verdict rather than a defect, neither
    of which belongs in a one-line dispatch reason. Split on ``". "`` and not
    on ``"."``, because that first sentence names the one wrapper that IS
    accepted -- ``.to_event(...)`` -- and a bare ``"."`` cuts at that
    wrapper's own leading dot, leaving the user reading "or one wrapped by".
    """
    try:
        check_gaussian(graph, graph.node(name), env)
    except NotGaussian as exc:
        return False, str(exc).partition(". ")[0]
    return True, ""


def _prior_draw(
    graph: Graph, node: Probabilistic, env: dict[str, Any], key: jax.Array
) -> jax.Array | None:
    """One draw from a Gaussian latent's own prior, or ``None`` if it has none."""
    try:
        loc, scale = gaussian_parts(graph, node, env)
    except NotGaussian:
        return None
    shape = node_shape(graph, node, env)
    noise = jax.random.normal(key, shape, dtype=loc.dtype)
    return jnp.broadcast_to(loc, shape) + jnp.broadcast_to(scale, shape) * noise


def _at_points(
    graph: Graph, names: tuple[str, ...], env: dict[str, Any], key: jax.Array
) -> tuple[list[dict[str, Any]] | None, str]:
    """At-points for ``check_linearity``, drawn only where a prior can be drawn.

    ``check_linearity``'s own default draws every outside latent from the
    graph's prior through the NumPyro bridge, and two LEGAL models in this
    package's suite die on it: ``overflowing_outside_latent``'s Cauchy(0, 1e6)
    overflows the prediction on ~99.7% of draws, and
    ``improper_outside_prior``'s ImproperUniform has no sampler at all. So
    only the Gaussian outside latents are drawn here; the rest stay at their
    centre, and the degradation is named in the reason.

    **The one thing that must not be done instead is passing a single
    at-point.** That turns the check into the moderate-parameter probe its
    own docstring names as the failure mode to avoid --
    ``affine_only_at_zero`` is the fixture that catches it.

    Returns ``(points, note)``; ``points`` is ``None`` when the block spans
    every latent, where ``check_linearity``'s default already short-circuits
    to repeated empty dicts without touching the bridge.
    """
    outside = [name for name in graph.latents if name not in set(names)]
    if not outside:
        return None, ""
    at = block_at(graph, names, env=env)
    points, degraded = [at], []
    for index in range(1, DEFAULT_AT_POINTS):
        root = jax.random.fold_in(key, index)
        point: dict[str, Any] = {}
        for position, name in enumerate(outside):
            drawn = _prior_draw(
                graph, graph.node(name), env, jax.random.fold_in(root, position)
            )
            if drawn is None:
                degraded.append(name)
            point[name] = at[name] if drawn is None else drawn
        points.append(point)
    note = (
        ""
        if not degraded
        else (
            f"; the linearity check held {sorted(set(degraded))} at their prior "
            "centre rather than drawing them -- they are not Gaussian, so their "
            "prior has no draw this module will take"
        )
    )
    return points, note


def _sigma_needs_rebuild(graph: Graph, names: tuple[str, ...]) -> bool:
    """Whether an observed node depends on a latent OUTSIDE the block.

    A deliberate OVER-approximation. ``loc`` and ``scale`` come out of one
    ``dist_fn`` and cannot be told apart structurally, so an outside latent
    reaching the observed node only through its location -- ``indirect_ancestor``
    -- reports ``True`` as well. ``False`` is the dangerous verdict, because it
    is what authorises hoisting ``noise_std`` out of a Gibbs sweep, so the
    approximation is on the side that costs work rather than correctness.
    """
    members = set(names)
    latents = set(graph.latents)
    return any(
        (_ancestors(graph, observed) & latents) - members for observed in graph.observed
    )


def _accepted_reason(
    block: tuple[str, ...], method: str, movement: float, note: str
) -> str:
    """Why an accepted block got the method it got."""
    if method == "gcr":
        moved = (
            f"sigma does not move with the block ({movement:.3e} relative, "
            f"rtol={SIGMA_RTOL:.1e}), so the covariance is fixed and one "
            "Wiener solve is exact"
        )
    else:
        tail = (
            "the block spans every latent, so the correction is one "
            "self-normalised importance reweighting"
            if method == "gcr+snis"
            else "the block is a proper subset, so the correction is an "
            "independent-proposal Metropolis step inside the sweep"
        )
        moved = (
            f"sigma moves {movement:.3e} relative with the block "
            f"(rtol={SIGMA_RTOL:.1e}), so the GLS fixed point is only a "
            f"proposal; {tail}"
        )
    return f"exact block {list(block)}: {moved}{note}"


def _relative_movement(
    baseline: dict[str, jax.Array], moved: dict[str, jax.Array]
) -> float:
    """Largest relative change of sigma, in ``check_prediction_dependence``'s units.

    The same denominator that function uses -- the largest ``|sigma|`` at the
    anchor, floored at 1e-300 -- so a movement obtained here and a movement
    obtained there mean the same thing and can be combined with ``max``. Two
    spellings of "relative" would make the number the dispatcher thresholds
    depend on which probe happened to be the largest.
    """
    worst = 0.0
    for observed, value in moved.items():
        scale = max(float(jnp.max(jnp.abs(baseline[observed]))), 1e-300)
        worst = max(worst, float(jnp.max(jnp.abs(value - baseline[observed]))) / scale)
    return worst


def _data_informed_point(
    operator: LinearBlock, precision: dict[str, Any]
) -> dict[str, jax.Array] | None:
    """Where this block's posterior actually sits: one Wiener solve.

    Solved at the PRIOR CENTRE's sigma, which is the only sigma available
    before the question "does sigma move" has been answered. That makes it
    the first step of the reweighting :func:`~bayesmith.exact.gls.
    iterative_gls` would run, and it is not iterated here: this is a probe
    LOCATION, not an answer, and it only has to land in the region the chain
    will occupy.

    ``require_convergence=None`` for the same reason. The guard costs
    ``POWER_ITERATIONS`` extra operator applications and, worse, RAISES on an
    ill-conditioned block -- which would turn a merely awkward model into a
    compile-time failure over a probe whose accuracy does not matter. What
    does matter is that the point is finite, so that is checked directly and
    ``None`` comes back if it is not, leaving the prior-scale probes as the
    whole measurement.
    """
    solution, _ = wiener_solve(operator, precision=precision, require_convergence=None)
    finite = all(bool(jnp.all(jnp.isfinite(value))) for value in solution.values())
    return solution if finite else None


def _sigma_movement(
    graph: Graph, operator: LinearBlock, at: dict[str, Any], key: jax.Array
) -> float:
    """How much sigma moves with the block -- probed from BOTH ends.

    A MEASUREMENT, not a declaration check: ``declared=True`` never raises,
    and the method is chosen from the number this returns rather than from
    what any node claims. Policing a ``depends_on_prediction=False`` that is
    false belongs where the claim is USED --
    ``iterative_gls(depends_on_prediction=...)``.

    **Two anchors, because one probe answers only half the question.**
    :func:`~bayesmith.exact.gls.check_prediction_dependence` displaces the
    block by :data:`~bayesmith.exact.gls.DEPENDENCE_PROBES`' 1.0 and -0.5
    prior widths from the PRIOR centre, along a deterministic ray and a
    random one. That covers DIRECTION -- a random direction detects any
    non-zero linear functional with probability 1 -- and its own docstring
    says what it does not cover: "The remaining gap is MAGNITUDE, not
    direction. The probe displaces by O(1) prior width from the prior centre,
    so a sigma that is exactly flat there and hinges further out reads
    bitwise constant however the direction is chosen ... only a larger
    magnitude would, at the cost of probing where the posterior will never
    go."

    So sigma is probed once more at the one larger magnitude that costs
    nothing, because it is where the posterior goes by construction: the
    block's own Wiener solution. ``hinged_sigma_beyond_the_probe`` is the
    fixture -- ``a ~ N(0, 1)``, a hinge at ``mu = 3``, and data that put the
    posterior at ``a ~ 6.1`` --
    where the prior-scale probes read **bitwise 0.0** at every key and this
    reads 1.904e+01, key-free. Classified from the prior probes alone it took
    the bare ``gcr`` arm, which applies no correction of any kind, and
    ``sample()`` returned a posterior 17.2x narrower than grid quadrature
    with ``ess=4000.0``, ``log_weights=None`` and ``unreliable=False``.

    **Cost**: one CG solve plus two ``sigma_of`` evaluations, at compile
    time, on blocks that have already passed ``check_linearity`` -- and only
    there. (One of the two is a re-read of the baseline
    ``check_prediction_dependence`` already took and does not return.)
    Measured, ``compile()`` over eight fixtures goes from 756 ms to 1030 ms,
    +36%, once per graph. Over the twenty-eight classified fixtures that
    predate it no verdict moves and three numbers rise (``plated_radiometer`` 1.133e+02 ->
    1.482e+02, ``one_sided_sigma`` 6.000e+01 -> 1.200e+02,
    ``element_contrast_sigma_plate`` 1.731e+01 -> 2.497e+02); every
    constant-sigma row stays bitwise 0.0, which it must, a constant being
    constant wherever it is read.

    **What is NOT done here, measured rather than assumed.** Re-running the
    whole probe battery ANCHORED at the solution -- four more ``sigma_of``
    calls -- was prototyped and adds nothing this returns does not already
    have. It fires only where sigma leaves its plateau and returns to exactly
    the plateau's value at the solution while still varying there, i.e. a
    root placed at the probe point, which is the fixture-crafting
    :data:`~bayesmith.exact.gls.DEPENDENCE_PATTERNS` refuses. On the hinge
    fixture it reads 2.994e-01 against this function's 1.904e+01, and on
    every other fixture it is dominated too.
    """
    sigma_of = sigma_from_graph(graph, at)
    movement = check_prediction_dependence(
        operator, sigma_of, declared=True, rtol=SIGMA_RTOL, key=key
    )
    baseline = sigma_of(domain_centre(operator))
    # `baseline` is read BOTH ways here, three lines apart: as the OPERATOR
    # the probe solve is weighted by, and as the sigma VALUES the movement is
    # measured against below. Converting the variable would break the second;
    # the conversion belongs at the call that wants an operator.
    solution = _data_informed_point(operator, diagonal_from(baseline))
    if solution is None:
        return movement
    at_the_data = _relative_movement(baseline, sigma_of(solution))
    return max(movement, at_the_data) if math.isfinite(at_the_data) else movement


def _classify_block(
    graph: Graph,
    block: tuple[str, ...],
    latents: list[str],
    env: dict[str, Any],
    key: jax.Array,
) -> Classification:
    """Check the block's ``linear_in`` claim, then pick its method.

    Split out of :func:`partition` only to keep both under this project's
    50-line function guideline; the two halves are one algorithm.
    """
    at = block_at(graph, block, env=env)
    at_points, note = _at_points(graph, block, env, key)
    try:
        linearity = check_linearity(graph, block, at, at_points=at_points, key=key)
    except StructureError as exc:
        # The ONE call site where StructureError is a verdict rather than a
        # fault: `linear_in` was declared and is false, which the spec routes
        # to NUTS with the members named. Caught here by SITE -- an `except`
        # placed any wider would swallow `check_gaussian`'s, which must not be.
        return _all_to_nuts(latents, f"exact block {list(block)} falls together: {exc}")
    operator = unchecked_operator(graph, block, at)
    movement = _sigma_movement(graph, operator, at, key)
    if movement <= SIGMA_RTOL:
        method = "gcr"
    else:
        method = "gcr+snis" if len(block) == len(latents) else "gcr+mh"
    return Classification(
        block,
        tuple(sorted(set(latents) - set(block))),
        method,
        _accepted_reason(block, method, movement, note),
        linearity,
        movement,
        _sigma_needs_rebuild(graph, block),
    )


def partition(graph: Graph, *, key: jax.Array | None = None) -> Classification:
    """Derive the exact block and its method from the graph's structure."""
    key = jax.random.key(0) if key is None else key
    latents = list(graph.latents)
    env = prior_environment(graph)

    for observed in graph.observed:
        ok, why = _is_gaussian(graph, observed, env)
        if not ok:
            return _all_to_nuts(
                latents,
                f"observed node {observed!r} is not a diagonal Gaussian: {why}",
            )

    qualified, why_not = [], {}
    for name in latents:
        ok, why = _is_gaussian(graph, name, env)
        if not ok:
            why_not[name] = why
            continue
        ok, why = _declares_linear_in(graph, name)
        if not ok:
            why_not[name] = why
            continue
        qualified.append(name)

    # Ejection: ancestor of ANY latent, qualified or not. The "qualified"
    # reading drops the factor p(child | member) silently -- see
    # `orphaned_child_latent`, and note the dense oracle reproduces the same
    # wrong answer because it reads the same two sources the operator does.
    # One pass and no `break`: a chain `tau -> x -> y` has TWO to eject, and
    # stopping at the first leaves a block whose members are ancestors of each
    # other -- `three_latent_chain` measures that. One pass suffices because
    # the quantifier ranges over `latents`, which never shrinks.
    ejected = {
        z
        for z in qualified
        if any(z in _ancestors(graph, other) for other in latents if other != z)
    }
    for z in ejected:
        why_not[z] = f"{z!r} is an ancestor of another latent's distribution"
    block = tuple(sorted(set(qualified) - ejected))

    if not block:
        return _all_to_nuts(
            latents, "; ".join(f"{n!r}: {w}" for n, w in sorted(why_not.items()))
        )
    return _classify_block(graph, block, latents, env, key)
