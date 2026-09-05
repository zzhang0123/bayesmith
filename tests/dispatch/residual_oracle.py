"""Deterministic quadrature oracles for a residual evidence, with a certificate.

**Which side is the oracle, said once and in the module that defines both,
because getting it backwards is how R4's square-root route fooled its own
test.**

===================== ============================================ ======
name                  what it integrates                           tier
===================== ============================================ ======
``oracle_joint``      ``log_joint`` over **every** latent, the       1
                      residual block and the exact block together
``oracle_collapsed``  ``log_joint`` of the **reduced** graph over    5
                      the residual latents alone
===================== ============================================ ======

``oracle_joint`` shares only ``log_joint`` and ``evaluate`` with production --
the model's own definition -- so it is independent of the elimination and may
grade it. ``oracle_collapsed`` calls ``collapse_graph`` and
``marginal_log_density``, which ARE the elimination, so it is self-consistency
with respect to it: it may grade a sampler running on the collapsed problem and
it may never grade the collapse. In every comparison this module builds, the
uncollapsed side is the oracle and the collapsed side is what is being graded.

**Why a certificate, and why it can fail.** A gap between two quadratures where
one has not converged is indistinguishable from a defect in the other. The R5
plan walked into exactly that: a first grid on ``shared_ancestor`` reported a
gap of 6.55 nats and the uncollapsed side was simply drifting, not wrong.
``test_residual_oracle.py`` pins that span as a regression and requires an
ABSTAIN. So :func:`quadrature` never returns a bare number. It returns a
:class:`Quadrature` whose ``value`` is ``None`` unless three conditions hold,
each of which is a bound this module computes rather than a level it was
handed:

1. **The increments shrink.** With the interval count doubled at each step, the
   last two movements give ``ratio = |delta_k| / |delta_(k-1)|``. ``ratio >= 1``
   means the value is not settling; on a span whose interior contains
   ``shared_ancestor``'s degenerate conditional the movement is a CONSTANT
   ``log 4 = 1.386`` per doubling, forever, which is what an unconverged
   trapezoid looks like from the outside.
2. **The geometric tail of those increments reaches the demanded resolution.**
   Summing the remaining movements as a geometric series -- the idiom
   ``dispatch/evidence.py::_tail_sum`` already uses for the prior audit's D109
   -- bounds what is left to gain by refining at
   ``|delta_k| * ratio / (1 - ratio)``, and the caller declares the level that
   bound must reach. **The resolution is declared and not inferred, because a
   band assembled from each side's own residual error otherwise WIDENS to
   contain whatever it is asked about.** An earlier draft of this module stopped
   at ``n_eval * eps * max(1, |value|)``, the naive-summation bound; in two
   dimensions ``n_eval`` is ``n**2``, so that floor QUADRUPLED with every
   refinement while the error it was chased with fell by four, and the test got
   easier the less work was done. The floor is now the pairwise-summation bound
   ``ceil(log2(n_eval)) * eps * max(1, |value|)`` -- numpy reduces with pairwise
   summation, whose relative error over ``N`` non-negative terms is
   ``O(log2 N) * eps``; a relative error on ``Z`` is an absolute error on
   ``log Z``, and ``max(1, |value|)`` carries the representation error of the
   logarithm itself -- and it enters the BAND rather than the stopping rule.
3. **The integrand is a number everywhere on the grid.** ``nan`` and ``+inf``
   are abstains with their own sentence, never a value that propagates.
   ``shared_ancestor`` is where this bites, and the reason is worth the line it
   takes: ``np.linspace(-3, 7, 101)`` lands EXACTLY on ``tau = 0`` -- where
   ``Normal(0, 0).log_prob`` is ``nan`` -- while ``jnp.linspace`` over the same
   interval misses it by ``5.55e-16`` and drifts instead. Two grids that read
   identically in a test, one ``nan`` and one a plausible number, chosen by
   which library built the axis.
4. **The integrand is decaying at every edge of every span.** Per axis and per
   side, ``rho = (outermost trapezoid cell) / (its neighbour)``. ``rho >= 1``
   says the mass is still growing as the span runs out, so the mass is outside
   the span and the number is an integral over the wrong region. ``rho < 1``
   models the tail past the edge as geometric and bounds the mass it holds at
   ``cell * rho / (1 - rho)``.

The three bounds are also the AGREEMENT BAND (**D111**): two quadratures agree
when the gap between them is no larger than the sum, over both sides, of
``refinement tail + float floor + truncation``, **and never less than**
:data:`AGREEMENT_FLOOR` ``* max(1, |log Z|)``. The FORM is derived -- each
summand is a named error source computed from the run -- and the one declared
LEVEL is :data:`AGREEMENT_FLOOR`, which is measured. That is D107's provenance
exactly: derived form, measured level.

**Why the band needs a floor, since it was green without one.** On
``indirect_ancestor`` both sides converge spectrally, so the refinement tail
falls to ``1e-16`` and the band collapses onto the arithmetic: a measured gap of
``1.95e-14`` against a band of ``5.45e-14``, which is 22 ULP of the answer
against a bound of 34. That is a comparison of one machine's rounding, and red
line 9 has cost this project four release tags. The floor makes the assertion
say what it means -- these two integrals are the same number -- at a level a
differently-rounding LAPACK cannot reach, and the price is quantified rather
than assumed:
``test_the_agreement_floor_is_far_below_the_defect_it_must_catch`` measures the
``dense_operator`` scaling against it and finds five to seven orders of margin
left. **A band assembled from
error bounds could in principle swallow a defect, so it is not asserted, it is
demonstrated**: ``test_the_oracle_is_not_blind_to_a_scaled_dense_operator``
scales ``dense_operator`` by 1.03 and the same comparison goes red with five to
seven orders of margin. R4's close-out records the square-root cross-route test
surviving exactly that mutation, and this module exists so that it does not
happen twice.

**What the geometric tail model is worth, measured rather than asserted in
either direction.** ``rho`` is estimated from the outermost pair of cells, so
the bound is exact for a geometric tail, high for anything decaying faster and
low for anything decaying slower. Measured on ``overflowing_outside_latent`` at
``|z| <= 60`` -- the span the R5 plan used while planning -- against the
``|z| <= 100`` value: the bound reads ``1.48e-03`` for an actual ``1.27e-03``,
so it BOUNDS the error, by 17 per cent, and the factor is stable to three
digits over six grids from ``n = 401`` to ``n = 12801``.
``test_the_truncation_bound_measures_the_mass_a_short_span_left_out`` pins that.
〔An earlier draft of this paragraph said the bound UNDERestimates, by 1.75x.
That came of comparing a bound this module reports for BOTH edges against a
measurement taken one edge at a time -- 7.45e-04 per side, and there are two.
The correction is recorded rather than swapped in silently, because the
direction of a bound's error is the only interesting thing about it.〕
The claim is measured on an exponentially-cut tail and is not proven for a
polynomial one; ``tests/exact/residual_models.py``'s Cauchy fixture is where
that case is put to it.

**A span is a domain, not a claim about the support.** ``shared_ancestor``
declares ``tau ~ N(2, 0.5)`` and ``x ~ N(0, |tau|)``, so ``p(x | tau)`` is
degenerate at ``tau = 0``, which is INTERIOR to the support. A span that spans
it does not merely lose accuracy: with ``tau = 0`` on the grid,
``marginal_log_density`` aborts through ``eqx.error_if``.  So the caller
declares a finite span, :func:`excluded_prior_mass` records how much declared
prior mass that span leaves out, and the number is never reported without it.
The excluded PRIOR mass is recorded and is NOT the abstain criterion --
``overflowing_outside_latent`` on ``|z| <= 100`` excludes 99.99 per cent of a
``Cauchy(0, 1e6)`` prior and is right to four hundred digits of the integrand,
because the likelihood and not the prior decides where the integrand's mass is.

**The GAP is the quantity these comparisons pin; the values are not.** A
twelve-digit trapezoid sum is a platform-dependent reduction, and red line 9
forbids pinning one.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.dispatch.classify import prior_environment
from bayesmith.dispatch.collapse import collapse_graph
from bayesmith.exact.gaussian import apply_probabilistic
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.graph import Graph

#: Double precision is a premise of every number here, never an accident of the
#: ambient config.  The caller is expected to be inside ``jax.enable_x64(True)``
#: and :func:`quadrature` refuses if it is not.
EPS = float(np.finfo(np.float64).eps)

#: **D111.** The relative floor under the agreement band, and the resolution the
#: refinement is asked to reach. One number, used for both, because they are the
#: same statement: the oracle stops refining when what is left to gain is below
#: the level at which it is willing to claim two integrals are equal.
#:
#: Derived FORM -- relative to ``max(1, |log Z|)``, because the precision a log
#: evidence can be represented in scales with its magnitude. Measured LEVEL, and
#: both halves of the measurement are pinned as tests rather than asserted here:
#: it is about four orders ABOVE the float-level disagreement between the two
#: routes on this platform (worst measured ``1.95e-14`` on ``indirect_ancestor``,
#: 22 ULP), so a platform whose LAPACK rounds a thousand times worse still
#: certifies; and seven orders BELOW the smallest defect the comparison exists
#: to catch (scaling ``dense_operator`` by 1.03 moves the gap to ``2e-02``).
AGREEMENT_FLOOR = 1e-9

#: How many grid points a single vmapped call evaluates.  Bounds peak memory at
#: a few megabytes whatever the dimension, so the cost of a high-dimensional
#: grid shows up as TIME -- which is what Task 3.2's stop-rule measures -- and
#: not as an allocation failure that says nothing about convergence.
_CHUNK = 1 << 15


@dataclasses.dataclass(frozen=True, slots=True)
class Span:
    """One integration axis: which latent, and the finite interval used."""

    name: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not self.upper > self.lower:
            raise ValueError(
                f"span for {self.name!r} is {(self.lower, self.upper)}, which "
                "is not an interval; an oracle over an empty or inverted span "
                "returns a number with no integrand under it"
            )

    def __str__(self) -> str:
        return f"{self.name} in ({self.lower:g}, {self.upper:g})"


@dataclasses.dataclass(frozen=True, slots=True)
class EdgeDecay:
    """What the integrand is doing at one end of one axis."""

    axis: str
    side: str
    ratio: float
    fraction: float

    @property
    def decaying(self) -> bool:
        return self.ratio < 1.0

    def __str__(self) -> str:
        return f"{self.axis}/{self.side}: rho={self.ratio:.4g} tail={self.fraction:.3e}"


@dataclasses.dataclass(frozen=True, slots=True)
class Certificate:
    """Everything the quadrature knows about its own error, reported always.

    ``refused`` is the whole point: a certificate that cannot fail certifies
    nothing.  It carries the reason as a sentence, so a test asserting an
    ABSTAIN can assert WHICH abstain it got rather than merely that the value
    was ``None``.
    """

    history: tuple[tuple[int, float], ...]
    increment_ratio: float | None
    refinement_tail: float
    float_floor: float
    truncation: float
    edges: tuple[EdgeDecay, ...]
    refused: str | None

    @property
    def bound(self) -> float:
        """This side's total error bound: refinement, arithmetic and truncation."""
        return self.refinement_tail + self.float_floor + self.truncation

    @property
    def increments(self) -> tuple[float, ...]:
        return tuple(
            self.history[index + 1][1] - self.history[index][1]
            for index in range(len(self.history) - 1)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Quadrature:
    """A quadrature result. ``value is None`` means ABSTAIN, and says why."""

    value: float | None
    spans: tuple[Span, ...]
    certificate: Certificate
    excluded_prior_mass: Mapping[str, float | None]

    @property
    def certified(self) -> bool:
        return self.value is not None

    def describe(self) -> str:
        where = ", ".join(str(span) for span in self.spans)
        if self.value is None:
            return f"ABSTAIN over [{where}]: {self.certificate.refused}"
        return (
            f"{self.value!r} over [{where}], bound {self.certificate.bound:.3e} "
            f"(refinement {self.certificate.refinement_tail:.3e}, float "
            f"{self.certificate.float_floor:.3e}, truncation "
            f"{self.certificate.truncation:.3e}), excluded prior mass "
            f"{dict(self.excluded_prior_mass)}"
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Agreement:
    """Two quadratures compared. ``gap`` is the quantity; the values are not."""

    left: Quadrature
    right: Quadrature
    gap: float | None
    band: float | None

    @property
    def agree(self) -> bool:
        return (
            self.gap is not None
            and self.band is not None
            and abs(self.gap) <= self.band
        )

    @property
    def margin(self) -> float:
        """How many times the band the gap is. Above 1 the comparison is red."""
        if self.gap is None or not self.band:
            return math.inf
        return abs(self.gap) / self.band

    def describe(self) -> str:
        if self.gap is None:
            return (
                "no comparison: "
                + ("left ABSTAINed; " if not self.left.certified else "")
                + ("right ABSTAINed; " if not self.right.certified else "")
                + f"left={self.left.describe()} right={self.right.describe()}"
            )
        return (
            f"gap {self.gap:+.6e} against band {self.band:.6e} "
            f"(margin {self.margin:.3e}); left {self.left.describe()}; "
            f"right {self.right.describe()}"
        )


def _require_x64() -> None:
    if jnp.zeros(()).dtype != jnp.float64:
        raise RuntimeError(
            "the residual oracle runs in double precision only; wrap the call "
            "in `with jax.enable_x64(True):`. A float32 trapezoid reaches its "
            "float floor four decades early and certifies a value that has "
            "not converged"
        )


def _evaluate(
    log_density: Callable[[dict[str, Any]], Any],
    names: Sequence[str],
    grids: Sequence[Any],
) -> np.ndarray:
    """``log_density`` on the full product grid, in memory-bounded chunks."""
    shape = tuple(int(grid.shape[0]) for grid in grids)
    total = int(np.prod(shape))
    stacked = jnp.stack(
        [jnp.asarray(grid, dtype=jnp.float64) for grid in grids]
    )  # (d, n)

    @jax.jit
    def block(indices):
        coordinates = []
        remainder = indices
        for size in reversed(shape):
            coordinates.append(remainder % size)
            remainder = remainder // size
        coordinates = coordinates[::-1]
        points = jnp.stack(
            [stacked[axis][coordinates[axis]] for axis in range(len(shape))], axis=1
        )
        return jax.vmap(
            lambda point: log_density(
                {name: point[axis] for axis, name in enumerate(names)}
            )
        )(points)

    pieces = [
        block(jnp.arange(start, min(start + _CHUNK, total)))
        for start in range(0, total, _CHUNK)
    ]
    return np.asarray(jnp.concatenate(pieces), dtype=float).reshape(shape)


def _log_trapezoid(log_values: np.ndarray, axes: Sequence[np.ndarray]) -> float:
    """``log`` of the trapezoid integral of ``exp(log_values)``, peak-shifted."""
    peak = float(np.max(log_values))
    if not math.isfinite(peak):
        return peak if peak == -math.inf else math.nan
    density = np.exp(log_values - peak)
    for axis in range(len(axes) - 1, -1, -1):
        density = np.trapezoid(density, axes[axis], axis=axis)
    return peak + float(np.log(density))


def _marginal(density: np.ndarray, axes: Sequence[np.ndarray], keep: int) -> np.ndarray:
    """Integrate every axis except ``keep`` out of ``density``."""
    reduced = density
    for axis in range(len(axes) - 1, -1, -1):
        if axis == keep:
            continue
        reduced = np.trapezoid(reduced, axes[axis], axis=axis)
    return reduced


def _edges(
    log_values: np.ndarray, axes: Sequence[np.ndarray], names: Sequence[str]
) -> tuple[EdgeDecay, ...]:
    """The decay of the integrand at both ends of every axis.

    Read off the per-axis MARGINAL rather than off a slice: a corner cell can be
    negligible while the edge of the axis as a whole is not, and it is the axis
    that the span is a statement about.
    """
    peak = float(np.max(log_values))
    if not math.isfinite(peak):
        return tuple(
            EdgeDecay(name, side, math.inf, math.inf)
            for name in names
            for side in ("lower", "upper")
        )
    density = np.exp(log_values - peak)
    found: list[EdgeDecay] = []
    for axis, name in enumerate(names):
        marginal = np.asarray(_marginal(density, axes, axis), dtype=float)
        grid = np.asarray(axes[axis], dtype=float)
        cells = (marginal[1:] + marginal[:-1]) / 2.0 * np.diff(grid)
        total = float(cells.sum())
        for side, (outer, inner) in (("lower", (0, 1)), ("upper", (-1, -2))):
            edge = float(cells[outer])
            neighbour = float(cells[inner])
            if edge <= 0.0:
                found.append(EdgeDecay(name, side, 0.0, 0.0))
                continue
            if neighbour <= 0.0 or total <= 0.0:
                found.append(EdgeDecay(name, side, math.inf, math.inf))
                continue
            ratio = edge / neighbour
            if ratio >= 1.0:
                found.append(EdgeDecay(name, side, ratio, math.inf))
                continue
            found.append(
                EdgeDecay(name, side, ratio, edge * ratio / (1.0 - ratio) / total)
            )
    return tuple(found)


def excluded_prior_mass(graph: Graph, spans: Sequence[Span]) -> dict[str, float | None]:
    """How much declared prior mass each span leaves out.

    ``None`` where the question has no one-dimensional answer: a latent whose
    parents are latent has no marginal prior to integrate -- that is precisely
    what the R5 plan's 0.15 says makes a graph class (b) -- and a distribution
    without a ``cdf`` cannot be asked.  ``None`` is reported rather than a zero,
    because "no mass excluded" and "not asked" are the distinction this
    repository has paid for most often.
    """
    environment = prior_environment(graph)
    latents = frozenset(graph.latents)
    out: dict[str, float | None] = {}
    for span in spans:
        if _ancestors(graph, span.name) & latents:
            out[span.name] = None
            continue
        try:
            distribution = apply_probabilistic(
                graph, graph.node(span.name), dict(environment)
            )
            below = float(distribution.cdf(jnp.asarray(span.lower)))
            above = 1.0 - float(distribution.cdf(jnp.asarray(span.upper)))
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            out[span.name] = None
            continue
        out[span.name] = max(0.0, below) + max(0.0, above)
    return out


def _ancestors(graph: Graph, name: str) -> frozenset[str]:
    """Every node reachable upstream of ``name``, deduped."""
    seen: set[str] = set()
    stack = list(graph.node(name).parents)
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(graph.node(current).parents)
    return frozenset(seen)


def quadrature(
    log_density: Callable[[dict[str, Any]], Any],
    spans: Sequence[Span],
    *,
    resolution: float,
    start: int = 201,
    refinements: int = 8,
    graph: Graph | None = None,
) -> Quadrature:
    """Trapezoid over ``spans``, refined until it certifies or abstains.

    ``start`` is the point count of the first grid and each refinement doubles
    the intervals, ``n -> 2n - 1``, so every earlier grid is a subset of every
    later one and the movement between them is the discretisation error alone.

    ``resolution`` is relative: the refinement stops when the geometric tail of
    the increments is at or below ``resolution * max(1, |value|)``, and ABSTAINs
    if ``refinements`` grids do not get there. It has no default. A caller that
    could omit it would be choosing the band by choosing how long to run, which
    is the failure mode the module docstring's condition 2 describes.
    """
    _require_x64()
    if not resolution > 0.0:
        raise ValueError(
            f"resolution={resolution!r}; the refinement needs a positive level "
            "to reach, and a non-positive one is unreachable rather than strict"
        )
    names = tuple(span.name for span in spans)
    if len(set(names)) != len(names):
        raise ValueError(
            f"{names} names an axis twice; a product grid needs distinct axes"
        )
    history: list[tuple[int, float]] = []
    edges: tuple[EdgeDecay, ...] = ()
    count = start
    ratio: float | None = None
    tail = math.inf
    floor = math.inf
    demanded = math.inf
    truncation = math.inf
    for _ in range(refinements):
        axes = [
            np.linspace(span.lower, span.upper, count, dtype=float) for span in spans
        ]
        log_values = _evaluate(log_density, names, [jnp.asarray(axis) for axis in axes])
        unusable = int(np.count_nonzero(np.isnan(log_values) | (log_values == np.inf)))
        if unusable:
            where = np.argwhere(np.isnan(log_values) | (log_values == np.inf))[0]
            at = ", ".join(
                f"{name}={axes[axis][index]!r}"
                for axis, (name, index) in enumerate(zip(names, where, strict=True))
            )
            return Quadrature(
                value=None,
                spans=tuple(spans),
                certificate=Certificate(
                    history=tuple(history),
                    increment_ratio=None,
                    refinement_tail=math.inf,
                    float_floor=math.inf,
                    truncation=math.inf,
                    edges=(),
                    refused=(
                        f"the integrand is not a number at {unusable} of "
                        f"{log_values.size} grid points on the n={count} grid, "
                        f"the first at {at}; a span whose interior holds a "
                        "degenerate conditional is not a span this quadrature "
                        "can be asked about"
                    ),
                ),
                excluded_prior_mass=(
                    excluded_prior_mass(graph, spans) if graph is not None else {}
                ),
            )
        value = _log_trapezoid(log_values, axes)
        history.append((count, value))
        edges = _edges(log_values, axes, names)
        truncation = sum(edge.fraction for edge in edges)
        cells = int(np.prod([len(axis) for axis in axes]))
        scale = max(1.0, abs(value)) if math.isfinite(value) else math.inf
        floor = math.ceil(math.log2(cells)) * EPS * scale
        demanded = resolution * scale
        if len(history) >= 3:
            previous = history[-2][1] - history[-3][1]
            latest = history[-1][1] - history[-2][1]
            if latest == 0.0:
                ratio, tail = 0.0, 0.0
            elif previous == 0.0:
                ratio, tail = math.inf, math.inf
            else:
                ratio = abs(latest) / abs(previous)
                tail = abs(latest) * ratio / (1.0 - ratio) if ratio < 1.0 else math.inf
            settled = math.isfinite(tail) and tail <= demanded
            if settled and math.isfinite(truncation):
                break
        count = 2 * count - 1
    reasons: list[str] = []
    if not math.isfinite(truncation):
        growing = [str(edge) for edge in edges if not edge.decaying]
        reasons.append(
            "the integrand is not decaying at "
            + ", ".join(growing)
            + " -- the mass is outside this span, so the value integrates the "
            "wrong region"
        )
    if not (math.isfinite(tail) and tail <= demanded):
        steps = [
            f"{history[index + 1][1] - history[index][1]:.3e}"
            for index in range(len(history) - 1)
        ]
        reasons.append(
            f"the value has not reached the demanded resolution: after "
            f"{len(history)} grids up to n={history[-1][0]} the increments are "
            f"{steps} (ratio {ratio!r}), whose geometric tail is {tail:.3e} "
            f"against a demanded {demanded:.3e}"
        )
    certificate = Certificate(
        history=tuple(history),
        increment_ratio=ratio,
        refinement_tail=tail,
        float_floor=floor,
        truncation=truncation,
        edges=edges,
        refused="; ".join(reasons) if reasons else None,
    )
    return Quadrature(
        value=None if reasons else history[-1][1],
        spans=tuple(spans),
        certificate=certificate,
        excluded_prior_mass=(
            excluded_prior_mass(graph, spans) if graph is not None else {}
        ),
    )


def oracle_joint(graph: Graph, spans: Sequence[Span], **options: Any) -> Quadrature:
    """Tier 1. ``log_joint`` over EVERY latent -- residual and exact together.

    Refuses a subset, because integrating some of the latents and calling the
    answer an evidence is the error this whole module is arranged to prevent,
    and it would read as a perfectly ordinary call.
    """
    names = tuple(span.name for span in spans)
    if set(names) != set(graph.latents):
        raise ValueError(
            f"oracle_joint needs a span for every latent; got {sorted(names)} "
            f"for latents {sorted(graph.latents)}. Integrating a SUBSET yields "
            "a conditional, not an evidence, and nothing downstream of here "
            "could tell the difference"
        )
    return quadrature(
        lambda values: log_joint(graph, dict(values)), spans, graph=graph, **options
    )


def oracle_collapsed(
    graph: Graph,
    exact_names: Iterable[str],
    spans: Sequence[Span],
    **options: Any,
) -> Quadrature:
    """Tier 5. The REDUCED graph's ``log_joint`` over the residual latents.

    Self-consistency with respect to the elimination: it calls
    ``collapse_graph`` and ``marginal_log_density``, which are the production
    path.  It grades a sampler; it never grades the collapse.
    """
    exact = tuple(exact_names)
    residual = tuple(name for name in graph.latents if name not in frozenset(exact))
    names = tuple(span.name for span in spans)
    if set(names) != set(residual):
        raise ValueError(
            f"oracle_collapsed needs a span for every residual latent; got "
            f"{sorted(names)} for residual {sorted(residual)}"
        )
    reduced = collapse_graph(graph, exact, residual)
    return quadrature(
        lambda values: log_joint(reduced, dict(values)), spans, graph=graph, **options
    )


def agreement(left: Quadrature, right: Quadrature) -> Agreement:
    """Compare two quadratures within the sum of what each says about itself."""
    if not (left.certified and right.certified):
        return Agreement(left=left, right=right, gap=None, band=None)
    scale = max(1.0, abs(left.value), abs(right.value))
    return Agreement(
        left=left,
        right=right,
        gap=left.value - right.value,
        band=max(
            left.certificate.bound + right.certificate.bound,
            AGREEMENT_FLOOR * scale,
        ),
    )


__all__ = [
    "AGREEMENT_FLOOR",
    "Agreement",
    "Certificate",
    "EdgeDecay",
    "Quadrature",
    "Span",
    "agreement",
    "excluded_prior_mass",
    "oracle_collapsed",
    "oracle_joint",
    "quadrature",
]
