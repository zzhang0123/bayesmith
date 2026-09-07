"""The exact log evidence, assembled as terms that can be audited one by one.

``log Z = log p(d | M) = INT p(d | theta, M) p(theta | M) d theta`` is what this
module builds, and the only thing that separates it from a posterior is that
every theta-independent constant is invisible in the second and load-bearing in
the first. A dropped constant does not make an evidence obviously wrong; it
makes it finite, plausible and wrong by a fixed number of nats.

So nothing here returns one scalar. The exact block's contribution is five
named terms, each derived from the block's own inputs.

**What checks them, stated precisely, because the first draft of this sentence
overclaimed.** Comparing this module's total against the square-root route in
``marginal_log_density`` is NOT a check against an independent derivation: the
two share every upstream seam -- ``unchecked_operator``, ``precision_at``,
``dense_operator`` and ``observed_descendants`` -- and ``dense_operator`` is
what builds the design matrix both of them integrate. An adversarial review
priced that: scaling ``dense_operator``'s return by 1.03 leaves the
cross-route test **completely blind**, 5 passed, exit 0, while the hand
derivation fails 5 and an independent quadrature fails 14.

The cross-route comparison is worth keeping -- it catches everything
downstream of the shared design, which is where the constants live -- but the
check §9.1 actually asks for is the one that rebuilds the design from the
model's own parameters. That is ``_by_hand`` in
``tests/dispatch/test_evidence.py``, and the genuinely third route is
quadrature of the graph's own ``log_joint``.

The decomposition is the determinant lemma written out. With ``A`` the design
over the exact block, ``N`` the observation covariance, ``S`` the block's prior
covariance, ``m`` its prior mean, ``c`` the constant part of the prediction and
``C = A S A^T + N``::

    log p(d) = -1/2 [ r^T C^-1 r + logdet C + n log 2pi ],  r = d - A m - c
    logdet C = logdet N + logdet S + logdet F,              F = A^T N^-1 A + S^-1

which splits into

===========================  ==========================================
``data_log_normaliser``      ``-1/2 logdet(2 pi N)``
``residual_quadratic``       ``-1/2 r^T C^-1 r``
``prior_log_normaliser``     ``-1/2 logdet(2 pi S)``
``integral_log_two_pi``      ``+k/2 log 2pi``
``block_log_determinant``    ``-1/2 logdet F``
===========================  ==========================================

The third and fourth do NOT cancel. Only their ``2 pi`` halves do -- the prior
contributes one row per block degree of freedom, so the ``k`` in both is the
same ``k`` -- and what is left is ``-sum_j log s_j``, the declared prior widths.
That residue is exactly zero at unit width and nowhere else: measured,
``-0.5306`` at ``s = 1.7``, ``+5.9915`` at ``k = 2, s = 0.05``, ``-12.2830`` at
``k = 3, s = 60``.

An earlier draft of this paragraph said they cancel exactly, which is what unit
priors look like, and it contradicted a test in its own batch --
``test_the_prior_normaliser_is_blind_at_unit_width`` asserts precisely that the
residue vanishes there and nowhere else. Corrected after an adversarial review
measured it. The two are reported separately because they come from different
places and a change to either alone is a real change.

**This module is dense on purpose.** R4's admitted structure class is a
whole-graph-exact linear-Gaussian block small enough that ``dense_operator``
already materialises its design on the collapse path. A matrix-free route
cannot produce ``logdet F`` at all -- ``exact/gibbs.py`` says so in its own
prose -- and the determinant is the term this module exists to report.

Dependency direction: this module reads ``dispatch``, ``exact`` and ``graph``
and is read by ``evaluation``. It does not import ``evaluation``, and
``tests/test_layering.py`` holds that.
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import jax.numpy as jnp
import numpy as np

from bayesmith.artifacts._codec import register_artifact_type
from bayesmith.artifacts.results import EvidenceComponent
from bayesmith.bridge.numpyro_bridge import to_numpyro as _to_numpyro
from bayesmith.compiled import CompiledEvidenceProblem
from bayesmith.dispatch.classify import prior_environment
from bayesmith.dispatch.collapse import observed_descendants
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.fisher import dense_operator
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.precision import per_sample_sigma
from bayesmith.graph.evaluate import apply_probabilistic
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic

__all__ = [
    "EVIDENCE_COMPONENT_NAMES",
    "CompiledEvidenceProblem",
    "compile_evidence_problem",
    "increments_converge",
    "mass_is_normalised",
    "ConditionalPriorRange",
    "ExactAssembly",
    "PriorAudit",
    "PriorVerdict",
    "assemble_exact",
    "audit_graph_priors",
    "audit_prior",
    "conditional_prior_range_report",
    "conditional_prior_verdicts",
    "residual_backend",
]

#: The closed set of names an exact assembly may report.
#:
#: Closed so that a provenance guard can assert MEMBERSHIP rather than match a
#: string it hopes about. Two guards in this repository have been walked past
#: by writing the forbidden thing a second way -- a duplicate re-introduced
#: under an import alias, and a threshold spelled ``int("40")`` -- and a
#: component name validated by ``_text`` rather than ``_code`` may legally
#: contain whitespace, so "the same name with a space in it" is available to
#: anyone routing on the spelling.
EVIDENCE_COMPONENT_NAMES: tuple[str, ...] = (
    "data_log_normaliser",
    "residual_quadratic",
    "prior_log_normaliser",
    "integral_log_two_pi",
    "block_log_determinant",
)

#: What each component says about how it was obtained. Never the empty string:
#: ``EvidenceComponent.method`` defaults to ``""`` and ``_text`` refuses it, so
#: the default is unconstructible and every construction site must supply one.
_METHODS: dict[str, str] = {
    "data_log_normaliser": "exact_gaussian_normaliser",
    "residual_quadratic": "exact_dense_solve",
    "prior_log_normaliser": "exact_gaussian_normaliser",
    "integral_log_two_pi": "exact_gaussian_integral",
    "block_log_determinant": "exact_dense_slogdet",
}


@dataclasses.dataclass(frozen=True, slots=True)
class ExactAssembly:
    """A log evidence and the terms it is made of.

    ``log_evidence`` is the assembly's own total, computed from the closed form
    rather than by adding the components up. ``log_evidence := sum(components)``
    is an identity a dropped term survives, because dropping one removes it from
    both sides at once.

    **That mutant survives the test suite, and the honest thing is to say so.**
    Measured in an isolated worktree at ``ef19106``: ten mutations of this
    module, nine killed, and the one that lives is exactly
    ``log_evidence=float(sum(terms.values()))`` -- 15 passed. It cannot be
    killed by any test of CORRECT code, because when every term is right the
    two expressions are the same number; here they differ by one ULP
    (``-4.284669145867042`` against ``-4.284669145867041``), and an assertion
    that they differ bitwise would be a fixture pinning one machine's
    arithmetic, which is the failure that cost this project four release tags.

    What it is worth was measured rather than argued. Compounding it with each
    single fault changes nothing: ``M5 + drop the prior normaliser`` fails the
    same 10 tests as that fault alone, ``M5 + flip the determinant`` the same
    11, ``M5 + skip the descendant filter`` the same 1. The per-term grading
    against a hand derivation is what catches those, and it catches them
    whichever way the total is spelled. So the separation is defence in depth
    over a defence that already holds, kept because it costs one line and
    because §0.3 asks for it -- not because a test would notice its absence.
    """

    log_evidence: float
    components: tuple[EvidenceComponent, ...]

    def __post_init__(self) -> None:
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError(
                f"an exact assembly reported a term twice: {names}. Each "
                f"constant enters log Z once, and a duplicate is the shape of "
                f"the defect R4 Task 1 repaired one layer down."
            )
        # EQUALITY, not containment. An earlier version subtracted the closed
        # set from the reported names, which is empty for any SUBSET -- so an
        # assembly reporting four of the five terms, or none, constructed
        # cleanly. A dropped term is the defect this class exists to make
        # visible, and it was the one shape the check could not see. Found by
        # an adversarial review building it.
        if set(names) != set(EVIDENCE_COMPONENT_NAMES):
            missing = sorted(set(EVIDENCE_COMPONENT_NAMES) - set(names))
            unknown = sorted(set(names) - set(EVIDENCE_COMPONENT_NAMES))
            raise ValueError(
                f"an exact assembly must report every term of log Z and no "
                f"others; missing {missing}, unknown {unknown}. A provenance "
                f"guard reads membership of EVIDENCE_COMPONENT_NAMES, so a "
                f"name it does not know is a term nothing checks -- and a name "
                f"it does not receive is a constant nothing reports."
            )


@dataclasses.dataclass(frozen=True, slots=True)
class _DenseBlock:
    """The exact block as dense numpy, with nothing JAX left in it."""

    design: np.ndarray
    data: np.ndarray
    offset: np.ndarray
    variance: np.ndarray
    prior_std: np.ndarray
    prior_mean: np.ndarray


def _dense_block(
    graph: Graph, exact_names: tuple[str, ...], values: Mapping[str, Any]
) -> _DenseBlock:
    """The exact block's design, data, noise and prior, as dense numpy.

    Read through exactly the seam ``marginal_log_density`` reads, including its
    descendant filter: an observation the block does not reach is not part of
    the block's marginal likelihood, and compressing it here would reintroduce
    the double count Task 1 removed.
    """
    block = unchecked_operator(graph, exact_names, at=values, probe_gaussian=False)
    centre = {name: block.prior_mean[name] for name in block.names}
    precision = precision_at(graph, {**dict(values), **centre})
    absorbed = frozenset(observed_descendants(graph, exact_names))

    design = np.asarray(dense_operator(block), dtype=float)
    rows: list[np.ndarray] = []
    data: list[np.ndarray] = []
    offset: list[np.ndarray] = []
    variance: list[np.ndarray] = []
    row = 0
    for observed in sorted(block.data):
        width = int(jnp.ravel(block.data[observed]).shape[0])
        if observed not in absorbed:
            row += width
            continue
        rows.append(design[row : row + width])
        data.append(np.asarray(jnp.ravel(block.data[observed]), dtype=float))
        offset.append(np.asarray(jnp.ravel(block.offset[observed]), dtype=float))
        variance.append(
            np.asarray(_variance_of(precision, observed, width), dtype=float)
        )
        row += width

    if not rows:
        # `marginal_log_density` returns 0.0 here -- the integral is over the
        # prior alone and a normalised prior integrates to one. Reaching
        # np.concatenate([]) instead would report that as "need at least one
        # array to concatenate", which names neither the model nor the fix.
        raise NotImplementedError(
            f"no observation reaches the block {list(block.names)}, so its "
            f"marginal likelihood is the prior's own integral and carries no "
            f"data term. `marginal_log_density` answers 0.0 for this; an "
            f"evidence DECOMPOSITION has no data normaliser and no residual to "
            f"report, so R4 refuses rather than filing empty components."
        )

    prior_mean = np.concatenate(
        [np.atleast_1d(np.asarray(block.prior_mean[n], dtype=float)) for n in block.names]
    )
    # An ASSERTION, not a broadcast. An earlier version wrote
    # `np.broadcast_to(std, (_width_of(shape),))`, which reads as if a scalar
    # prior std were being expanded to the block's width -- and measured, it
    # never is: `unchecked_operator` already returns one std per component in
    # every reachable shape. Removing the broadcast changed no test, which is
    # what dead defensive code looks like. What the line was actually doing was
    # checking a width, so it now says so and fails with the two numbers rather
    # than with a numpy broadcast error from inside a comprehension.
    prior_std_parts: list[np.ndarray] = []
    for name in block.names:
        part = np.atleast_1d(np.asarray(block.prior_std[name], dtype=float))
        expected = _width_of(block.shape[name])
        if part.size != expected:
            raise NotImplementedError(
                f"latent {name!r} declares {part.size} prior width(s) for a "
                f"block component of width {expected}. An evidence needs one "
                f"declared scale per degree of freedom -- `-sum(log s)` is a "
                f"sum over components -- and this package has no rule for "
                f"spreading fewer across more."
            )
        prior_std_parts.append(part)
    prior_std = np.concatenate(prior_std_parts)
    return _DenseBlock(
        design=np.concatenate(rows, axis=0),
        data=np.concatenate(data),
        offset=np.concatenate(offset),
        variance=np.concatenate(variance),
        prior_std=prior_std,
        prior_mean=prior_mean,
    )


def _width_of(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _variance_of(precision: dict[str, Any], observed: str, width: int) -> np.ndarray:
    """The per-sample observation variance, or a refusal that says why not.

    R4's admitted class is finite diagonal observation noise. A correlated
    covariance has a perfectly good exact evidence -- ``compress`` reads it
    through ``Precision`` with no special case -- but it has no DENSE ORACLE
    here, and §9.1 does not allow shipping a number whose only check is the
    route that produced it. So the class is refused rather than assembled
    ungated.

    **``per_sample_sigma is None`` is not enough, and this function is where
    that was learned.** It reads as a consequence check and is a TYPE check: a
    :class:`~bayesmith.exact.precision.MaskedPrecision` answers it, reporting
    ``inf`` for a sample that was never observed rather than declining to
    answer. So a masked observation passed the guard, gave ``slogdet`` an
    infinite variance, and died in ``EvidenceComponent``'s validator with
    ``log_value must be finite; got -inf`` -- a crash from two modules away
    naming neither the cause nor the remedy, and the module's own
    read-the-spelling disease inside the function written to avoid it.

    The finiteness test is the consequence, and it holds whatever the class is
    called. A masked evidence is a real capability and R4 does not have it:
    a sample that was not taken changes the DIMENSION of the data, so two
    models with different masks are not comparable by Bayes factor either.
    """
    sigmas = per_sample_sigma({observed: precision[observed]})
    if sigmas is None:
        raise NotImplementedError(
            "an exact evidence assembly needs a per-sample observation "
            "variance to write its dense oracle against. This observation's "
            "noise is correlated, which the square-root route integrates "
            "exactly and this dense decomposition has no independent "
            "reference for; R4 refuses the class rather than reporting a "
            "number nothing grades."
        )
    sigma = np.atleast_1d(np.asarray(sigmas[observed], dtype=float))
    if not np.all(np.isfinite(sigma)):
        raise NotImplementedError(
            f"observation {observed!r} declares a non-finite per-sample sigma, "
            f"which is how this package spells a sample that was never taken. "
            f"An evidence over a masked observation is a real quantity and R4 "
            f"does not assemble one: the mask changes the dimension of the "
            f"data, so the number would not be comparable with an unmasked "
            f"model's even if it were right."
        )
    return np.broadcast_to(sigma, (width,)) ** 2


def assemble_exact(
    graph: Graph,
    exact_names: tuple[str, ...],
    values: Mapping[str, Any] | None = None,
) -> ExactAssembly:
    """``log p(d | values)`` over the exact block, and the five terms of it.

    ``values`` holds the latents OUTSIDE the block, at which the block's
    marginal likelihood is evaluated; for a whole-graph-exact model it is
    empty and the number returned is the model's log evidence.
    """
    block = _dense_block(graph, tuple(exact_names), dict(values or {}))
    design = block.design

    n = block.data.size
    k = block.prior_std.size
    noise = np.diag(block.variance)
    prior_covariance = np.diag(block.prior_std**2)
    covariance = design @ prior_covariance @ design.T + noise
    residual = block.data - (design @ block.prior_mean + block.offset)
    fisher = design.T @ np.linalg.solve(noise, design) + np.linalg.inv(
        prior_covariance
    )

    _, logdet_noise = np.linalg.slogdet(noise)
    _, logdet_prior = np.linalg.slogdet(prior_covariance)
    _, logdet_fisher = np.linalg.slogdet(fisher)

    terms = {
        "data_log_normaliser": -0.5 * (logdet_noise + n * math.log(2.0 * math.pi)),
        "residual_quadratic": -0.5
        * float(residual @ np.linalg.solve(covariance, residual)),
        "prior_log_normaliser": -0.5 * (logdet_prior + k * math.log(2.0 * math.pi)),
        "integral_log_two_pi": +0.5 * k * math.log(2.0 * math.pi),
        "block_log_determinant": -0.5 * logdet_fisher,
    }

    # The total is the closed form, computed once and not by adding the terms
    # up: `log_evidence := sum(components)` is an identity a dropped term
    # survives, because dropping one removes it from both sides.
    _, logdet_covariance = np.linalg.slogdet(covariance)
    total = -0.5 * (
        float(residual @ np.linalg.solve(covariance, residual))
        + logdet_covariance
        + n * math.log(2.0 * math.pi)
    )

    return ExactAssembly(
        log_evidence=float(total),
        components=tuple(
            EvidenceComponent(
                name=name,
                log_value=float(terms[name]),
                standard_error=None,
                method=_METHODS[name],
            )
            for name in EVIDENCE_COMPONENT_NAMES
        ),
    )


# --------------------------------------------------------- the prior audit


@register_artifact_type
class PriorVerdict(StrEnum):
    """What an audit can say about a declared prior's own mass.

    Four values and not two, because the questions genuinely differ.
    ``PROPER`` says the density integrates to something finite; whether that
    something is one is the separate ``normalised`` field.
    ``IMPROPER`` says the integral diverges, so there is no ``Z``.
    ``UNVERIFIABLE`` says this audit could not tell -- §2.4's rule that a method
    which does not apply must not be dressed as a failure.
    ``UNDECLARED`` says the latent has no node-level prior to audit because a
    graph-level one covers it (§0.6).
    """

    PROPER = "proper"
    IMPROPER = "improper"
    UNVERIFIABLE = "unverifiable"
    UNDECLARED = "undeclared"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class PriorAudit:
    """One latent's prior, and the two things an evidence needs of it.

    Registered like its own verdict enum. Without it the verdict serialised and
    the record holding it did not, which is the half of a pair going stale that
    this repository has spent the most time repairing.

    **The mass check needs float64 and does not open a context to get one.**
    Its only production caller is behind evidence_requires_x64, which
    refuses a float32 environment before any of this runs; a direct call from
    outside that gate integrates at float32 and carries about 1e-7 of error
    against a 1e-6 normalisation tolerance. One decision, one home: the gate
    owns the dtype and this does not re-decide it.
    """

    latent: str
    verdict: PriorVerdict
    normalised: bool | None
    mass: float | None
    reason: str


#: Half-widths, in the density's own scale, at which the mass is measured.
#:
#: Not a threshold and not registered as one: the point is the SEQUENCE, not
#: any member of it. A proper density's mass stops moving as the window widens
#: and an improper one's keeps growing with it, so what is read is the shape of
#: the sequence -- which is what "improper" means, rather than a level anything
#: is compared against.
_WINDOWS: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0, 128.0)

#: Gauss-Legendre nodes per PANEL, and panels per unit of the density's own
#: scale. Fixed, so the audit is deterministic and gives the same answer on
#: every machine -- §9.3's rule for anything a test's verdict depends on.
#:
#: Composite rather than one high-order rule over the whole window, and that is
#: not a performance choice. Gauss-Legendre clusters its nodes at the ends of
#: the interval, so a single rule over +/-128 scale units samples the middle --
#: where a peaked density actually lives -- most thinly. Panels of one scale
#: unit put the same resolution everywhere the window reaches.
_NODES: int = 24
_PANELS_PER_SCALE: int = 2

#: The increment ratio below which a window sequence counts as convergent.
#:
#: Derived, not tuned. A flat density doubles its mass every time the window
#: doubles, so its increment ratio is exactly 2; the log-divergent ``1/x`` tail
#: adds a constant ``log 2`` each time, ratio exactly 1. Every convergent
#: density falls strictly below 1, and measured over the stock numpyro priors
#: the largest is Cauchy at 1.005. Anything in (1.006, 2) separates the two
#: families; 1.0 is the value the algebra names, and the margin below is the
#: one measurement rather than the threshold.
_CONVERGENT_INCREMENT_RATIO: float = 0.995

#: Below this the integral has found nothing, and "nothing is here" cannot be
#: told from "the window is elsewhere". Not a statistical threshold: any prior
#: whose total mass is 1e-12 is unnormalised by twelve orders of magnitude, so
#: no correct answer is lost by declining to name which of the two it is.
_NEGLIGIBLE_MASS: float = 1e-12

#: How close an extrapolated mass must be to one before the prior counts as
#: normalised, when the extrapolation's own error is smaller than this (D110).
#: The floor rather than the whole tolerance: a heavy tail's extrapolation
#: carries more error than this and supplies its own.
_NORMALISED_TOLERANCE: float = 1e-6


def _moments(distribution: Any) -> tuple[float, float]:
    """A centre and a scale to place the quadrature window on.

    The centre is looked for in four places, in order: the distribution's
    ``mean``, a declared ``loc``, a finite bound of the declared support, and
    finally zero. The fallbacks are not decoration -- a heavy-tailed density
    has no finite mean BY DEFINITION, so reading only ``mean`` and giving up
    would refuse Cauchy and HalfCauchy, and reading only ``mean`` and
    DEFAULTING to zero is worse: measured, ``dist.Cauchy(1e5, 1.0)`` was
    audited PROPER with ``mass = 8.15e-9``, a confident answer obtained by
    integrating empty space eight orders of magnitude from the peak.

    A misplaced window is caught downstream by consequence rather than here by
    inspection: a window that finds essentially no mass is reported
    UNVERIFIABLE, because "the density is zero" and "the window is in the
    wrong place" look identical from inside the integral.
    """
    centre: float | None = None
    scale = 1.0
    try:
        mean = float(np.asarray(distribution.mean))
        if np.isfinite(mean):
            centre = mean
    except Exception:  # noqa: BLE001 - numpyro raises six kinds here
        centre = None
    if centre is None:
        # A location the density itself declares, where there is one. Reached
        # for Cauchy and StudentT, whose `mean` is nan by definition.
        for attribute in ("loc", "low", "concentration"):
            try:
                value = float(np.asarray(getattr(distribution, attribute)))
            except Exception:  # noqa: BLE001, S112 - absent is the answer
                continue
            if np.isfinite(value):
                centre = value
                break
    if centre is None:
        # A finite edge of the declared support, which is where a
        # half-bounded density lives. HalfCauchy's `mean` is `inf` and it has
        # no `loc`; its support starts at zero.
        low, high, _ = _support_bounds(distribution)
        for bound in (low, high):
            if np.isfinite(bound):
                centre = float(bound)
                break
    if centre is None:
        centre = 0.0
    try:
        variance = float(np.asarray(distribution.variance))
        if np.isfinite(variance) and variance > 0.0:
            scale = float(np.sqrt(variance))
    except Exception:  # noqa: BLE001, S110 - a missing variance is recoverable
        pass
    if scale <= 0.0 or not np.isfinite(scale):
        scale = 1.0
    for attribute in ("scale",):
        try:
            declared = float(np.asarray(getattr(distribution, attribute)))
        except Exception:  # noqa: BLE001, S112 - absent is the answer
            continue
        if np.isfinite(declared) and declared > 0.0:
            scale = max(scale, declared)
    return centre, scale


def _support_bounds(distribution: Any) -> tuple[float, float, bool]:
    """``(lower, upper, resolved)`` for the declared support.

    ``resolved`` is ``False`` when the unwrapping loop hit its cap without
    finding a leaf constraint. Returning ``(-inf, inf)`` silently in that case
    turns a bounded proper prior into an IMPROPER verdict, which is the shape
    of failure this module is otherwise about.

    **The bounds cannot be inferred from the density**, which is why this reads
    the declaration. Measured: ``dist.Uniform(-2, 5).log_prob(100.0)`` is
    ``-log 7``, not ``-inf``, and ``ImproperUniform`` over the same interval
    returns ``0.0`` at every point on the line. NumPyro's ``log_prob`` does not
    mask its own support, so integrating a bounded density over a wide window
    without clipping reports EVERY bounded prior as improper.

    ``base_constraint`` is unwrapped in a loop rather than special-cased,
    because ``ImproperUniform``'s support arrives wrapped in an
    ``IndependentConstraint`` and a check for that class by name would be the
    spelling guard this module refuses to write.
    """
    support = getattr(distribution, "support", None)
    seen = 0
    while hasattr(support, "base_constraint"):
        if seen >= 8:
            return -np.inf, np.inf, False
        support = support.base_constraint
        seen += 1
    lower = getattr(support, "lower_bound", -np.inf)
    upper = getattr(support, "upper_bound", np.inf)
    try:
        lower = float(np.asarray(lower))
    except (TypeError, ValueError):
        lower = -np.inf
    try:
        upper = float(np.asarray(upper))
    except (TypeError, ValueError):
        upper = np.inf
    return lower, upper, True


@functools.cache
def _rule() -> tuple[np.ndarray, np.ndarray]:
    """The Gauss-Legendre nodes and weights, computed once per process."""
    return np.polynomial.legendre.leggauss(_NODES)


def _mass_on(distribution: Any, lower: float, upper: float, panels: int) -> float:
    """``INT exp(log_prob(x)) dx`` over ``[lower, upper]``, composite GL."""
    nodes, weights = _rule()
    edges = np.linspace(lower, upper, panels + 1)
    half = 0.5 * (edges[1:] - edges[:-1])
    middle = 0.5 * (edges[1:] + edges[:-1])
    points = (middle[:, None] + half[:, None] * nodes[None, :]).ravel()
    # The ambient dtype, not a demanded one: asking for float64 outside an x64
    # context gets a truncation warning and a float32 array anyway, and the
    # caller who must care about that is the evidence gate, which refuses the
    # environment by name rather than quietly integrating at half the digits.
    values = np.asarray(distribution.log_prob(jnp.asarray(points)), dtype=float)
    # A density this large overflows the sum rather than the exponential, and
    # that overflow is a RESULT -- it is how `audit_prior` learns it cannot
    # resolve the mass -- so it is caught and returned, not warned about.
    with np.errstate(over="ignore", invalid="ignore"):
        density = np.where(np.isfinite(values), np.exp(values), 0.0)
        density = density.reshape(panels, _NODES)
        return float(np.sum(half * np.sum(weights[None, :] * density, axis=1)))


def increments_converge(ratio: float, threshold: float) -> bool:
    """Whether a window sequence's increments are shrinking (D109).

    The one predicate that decides PROPER against IMPROPER, extracted so the
    threshold has a name, a boundary grid and a mutation -- the same shape
    ``evaluation/sbc.py``'s ``replicates_meet_floor`` has, and for the same
    reason: a comparison written inline inside a hundred-line function is a
    threshold no grid can reach.

    A flat density doubles its mass with its window, so its ratio is exactly 2;
    a ``1/x`` tail adds a constant ``log 2``, ratio exactly 1; every convergent
    density falls strictly below 1. The threshold separates those families and
    carries no units, which is why it is a ratio and not a difference.
    """
    return not ratio >= threshold


def mass_is_normalised(total: float, uncertainty: float) -> bool:
    """Whether an extrapolated prior mass is one, to what can be resolved (D110).

    The tolerance is ``max(1e-6, uncertainty)`` where the second term is the
    extrapolation's own measured error. A fixed tolerance refuses Cauchy for
    being 3.6e-06 from one and InverseGamma(1,1) for 3.7e-04, and both of those
    are the rule's error rather than the prior's.
    """
    return bool(abs(total - 1.0) <= max(_NORMALISED_TOLERANCE, uncertainty))


def _tail_sum(increment: float, ratio: float) -> float:
    """What a geometric tail with this ratio adds beyond the last window."""
    if ratio <= 0.0 or ratio >= 1.0:
        return 0.0
    return increment * ratio / (1.0 - ratio)


def _undecidable(latent: str, reason: str, mass: float | None = None) -> PriorAudit:
    """UNVERIFIABLE, which is not IMPROPER and must never be read as it.

    One says the prior has no finite mass; the other says this audit could not
    tell. §2.4's rule that a method which does not apply must not be dressed as
    a failure, one layer down from the reports it was written for.
    """
    return PriorAudit(
        latent=latent,
        verdict=PriorVerdict.UNVERIFIABLE,
        normalised=None,
        mass=mass,
        reason=reason,
    )


def _out_of_scope(distribution: Any, latent: str) -> PriorAudit | None:
    """What this quadrature cannot decide, refused before it tries.

    Each arm is a CONSEQUENCE of the density's own declaration rather than a
    type name, and each was found by an adversarial review returning a
    confident wrong answer or an uncaught exception:

    * a non-scalar shape -- ``MultivariateNormal``, ``Dirichlet``,
      ``Normal(zeros(3), 1)``, ``Independent(..., 1)`` -- raised ``TypeError``
      out of ``log_prob`` on a 1-D grid. The rule is one-dimensional and says so.
    * a discrete support -- the mass of a PMF is a SUM and this is an integral.
      ``Bernoulli(0.3)`` was reported IMPROPER; ``Poisson(50)`` was reported
      PROPER and normalised, which was right by coincidence, since the
      Euler-Maclaurin error happens to vanish at large lambda. Nothing in the
      output distinguished the two.
    * an unplaceable window -- ``Cauchy(1e5, 1)``'s ``mean`` is ``nan``, the
      centre defaulted to zero, and the audit integrated empty space eight
      orders of magnitude from the peak to report ``mass = 8.15e-9``.
    * an unresolved support -- see :func:`_support_bounds`.
    """
    event = tuple(getattr(distribution, "event_shape", ()) or ())
    batch = tuple(getattr(distribution, "batch_shape", ()) or ())
    if event or batch:
        return _undecidable(
            latent,
            f"this prior has event shape {event} and batch shape {batch}; the "
            f"mass check integrates one dimension and cannot speak for a "
            f"joint density. An evidence over it is a real quantity and this "
            f"audit is not the thing that decides it",
        )
    support = getattr(distribution, "support", None)
    seen = 0
    while hasattr(support, "base_constraint") and seen < 8:
        support = support.base_constraint
        seen += 1
    if bool(getattr(support, "is_discrete", False)):
        return _undecidable(
            latent,
            "this prior is declared over a discrete support, where the total "
            "mass is a sum and not an integral. Quadrature over it returns a "
            "number, and the number is not the mass",
        )
    _lower, _upper, resolved = _support_bounds(distribution)
    if not resolved:
        return _undecidable(
            latent,
            "the declared support is wrapped more deeply than this audit "
            "unwraps, so its bounds were not established. Reporting the "
            "unbounded default here would turn a bounded proper prior into an "
            "improper verdict",
        )
    return None


def audit_prior(distribution: Any, latent: str = "") -> PriorAudit:
    """Whether this declared density has finite mass, and whether it is one.

    Decided by integrating the density the model declares, never by reading a
    type name. ``isinstance(d, dist.ImproperUniform)`` is walked past by any
    user subclass whose ``log_prob`` returns a constant -- built and run in
    ``tests/dispatch/test_evidence_audit.py`` -- and ``unwrap`` strips only
    ``Independent``, so nothing else in the package looks through one either.

    **The method is the INCREMENTS of the window sequence, and the first
    version of it read the differences instead.** That version asked whether
    ``|mass(W) - mass(W/2)|`` had fallen below ``1e-6 * max(mass, 1)``, which
    is wrong in both directions and an adversarial review measured both:

    * too tight for a polynomial tail. ``HalfCauchy(1)`` -- the standard
      weakly-informative scale prior -- reaches 0.995 of its mass by the last
      window and was reported IMPROPER. So were ``Cauchy``, ``StudentT(1.5)``,
      ``Pareto``, ``LogNormal(0,3)`` and ``InverseGamma(0.5,1)``: thirty-one
      stock priors in all.
    * floored, so the verdict depended on the UNITS. The same flat improper
      prior on a length was IMPROPER in metres and PROPER in nanometres,
      because ``max(..., 1.0)`` cannot be tripped by a mass of 1e-8 however
      fast it doubles.

    What separates the two cases cleanly is whether the increments SHRINK. A
    convergent series has a ratio below one; a flat density doubles its mass
    with its window, ratio exactly two; the log-divergent ``1/sigma`` adds a
    constant ``log 2`` each time, ratio exactly one. That is a property of the
    density and carries no units at all.

    Where the increments shrink geometrically the tail is extrapolated rather
    than truncated, so ``normalised`` is right for a heavy tail instead of
    being short by the part the last window did not reach.
    """
    out_of_scope = _out_of_scope(distribution, latent)
    if out_of_scope is not None:
        return out_of_scope

    lower, upper, _ = _support_bounds(distribution)
    centre, scale = _moments(distribution)

    masses: list[float] = []
    last_low = last_high = 0.0
    last_panels = 0
    for width in _WINDOWS:
        low = max(lower, centre - width * scale)
        high = min(upper, centre + width * scale)
        if not (high > low):
            return _undecidable(
                latent,
                "the declared support and the density's own scale leave no "
                "interval to integrate over, so this audit cannot say whether "
                "the prior has finite mass",
            )
        panels = max(8, int(_PANELS_PER_SCALE * (high - low) / scale))
        mass = _mass_on(distribution, low, high, panels)
        if not np.isfinite(mass):
            return _undecidable(
                latent,
                f"the mass over [{low:.6g}, {high:.6g}] is not a finite "
                f"number, so this audit cannot say whether the prior is "
                f"proper. That is not the same as improper: one is a property "
                f"of the prior, the other is a limit of this check",
            )
        masses.append(mass)
        last_low, last_high, last_panels = low, high, panels

    settled = masses[-1]
    increments = [b - a for a, b in itertools.pairwise(masses)]
    tail = increments[-1]
    previous = increments[-2]
    if tail <= 0.0 or abs(tail) <= 1e-12 * max(abs(settled), 1e-300):
        ratio = 0.0
    elif abs(previous) <= 0.0:
        ratio = 1.0
    else:
        ratio = abs(tail) / abs(previous)

    # Growth decides BEFORE level, and the order is the point. A flat density
    # at `log_prob = -50` has a mass of 5e-20 over the widest window, which is
    # negligible by any absolute measure -- and it DOUBLES every time the
    # window doubles, which is what improper means. Asking "is this
    # negligible?" first would have abstained on a prior whose divergence is
    # unambiguous, and abstaining is the wrong answer when there is a right one.
    if not increments_converge(ratio, _CONVERGENT_INCREMENT_RATIO):
        return PriorAudit(
            latent=latent,
            verdict=PriorVerdict.IMPROPER,
            normalised=None,
            mass=None,
            reason=(
                f"the mass added by each doubling of the window is not "
                f"shrinking (ratio {ratio:.6g}), so the integral diverges and "
                f"p(d) is undefined for this model. A flat density gives "
                f"exactly 2 and a 1/x tail exactly 1; a convergent one falls "
                f"below {_CONVERGENT_INCREMENT_RATIO}"
            ),
        )

    # The increments do not diverge, so a mass is about to be reported -- and
    # before reporting one, ask whether the RULE agrees with itself. Halving
    # the panel width at the widest window is a different quadrature of the
    # same integral; a density with structure finer than the node spacing moves
    # under it. The window sequence tests the DOMAIN and nothing was testing
    # the rule, which is how a normalised bimodal prior came back with
    # `mass = 9.02e-31` and five identical windows behind it.
    #
    # It is also what separates a tail from a SINGULARITY. `Gamma(0.5, 1)` has
    # an integrable x^-0.5 pole at zero; its window masses creep upward, the
    # growth test reads that as tail mass, and the creep is this rule's own
    # discretisation error -- measured, doubling the panels moves it by
    # 3.5e-03 and it is still 1.2% short of one at sixty-four times the panels.
    # Reporting `normalised=False` there would refuse a perfectly proper prior.
    refined = _mass_on(distribution, last_low, last_high, last_panels * 2)
    if abs(refined - settled) > 1e-6 * max(abs(settled), abs(refined), 1e-300):
        return _undecidable(
            latent,
            f"the quadrature has not converged at the widest window: "
            f"{settled:.9g} against {refined:.9g} at twice the panel count. "
            f"The declared density has structure finer than this rule "
            f"resolves, so any mass it reported would be about the grid",
        )

    if settled <= _NEGLIGIBLE_MASS:
        # Not growing, and essentially nothing found. "The density is zero
        # here" and "the window is somewhere the density is not" look
        # identical from inside the integral, and the second is reachable: a
        # density peaked far from every location it declares puts the window in
        # empty space. Measured before this arm existed, `dist.Cauchy(1e5, 1)`
        # came back PROPER with mass 8.15e-9.
        return _undecidable(
            latent,
            f"the density integrates to {settled:.6g} over every window tried "
            f"and is not growing. Either it is zero on its own support or the "
            f"window is not where the density is, and this audit cannot tell "
            f"which",
            mass=float(settled),
        )

    # A geometric tail sums to `tail * r / (1 - r)`. Truncating instead is what
    # made every heavy tail look unnormalised.
    total = settled + _tail_sum(tail, ratio)

    # And the extrapolation has an error of its own, because a real tail is
    # only approximately geometric. It is measured rather than assumed: the
    # same extrapolation is run one window earlier and the two totals compared.
    # Without this, `normalised` was False for Cauchy (off by 3.6e-06) and for
    # InverseGamma(1,1) (off by 3.7e-04) -- correct priors refused for being
    # 0.0004 away from a mass this rule cannot resolve to better than that.
    earlier_ratio = (
        abs(increments[-2]) / abs(increments[-3])
        if len(increments) >= 3 and abs(increments[-3]) > 0.0
        else ratio
    )
    earlier_total = masses[-2] + _tail_sum(increments[-2], earlier_ratio)
    uncertainty = abs(total - earlier_total)
    normalised = mass_is_normalised(total, uncertainty)
    return PriorAudit(
        latent=latent,
        verdict=PriorVerdict.PROPER,
        normalised=normalised,
        mass=float(total),
        reason=(
            f"the window increments shrink with ratio {ratio:.6g}; the mass "
            f"reaches {settled:.9g} by the last window and extrapolates to "
            f"{total:.9g} +/- {uncertainty:.3g}"
            + ("" if normalised else ", which is not one")
        ),
    )


def residual_backend() -> Any | None:
    """The adapter that runs a residual integral, or ``None`` if there is none.

    R5 widens the admitted structure class before it chooses a backend, and the
    two are separate steps on purpose: the bake-off's legal answers include
    "no candidate passed". So the widened classes must not be admitted with
    nothing behind them, and this is where that is decided -- one function,
    returning ``None`` today, so that a graph which now passes every premise
    about the MODEL is refused for the reason that is actually true of it,
    which is a missing capability rather than a defect in the model.

    A function rather than a module constant because a constant read at import
    time cannot answer "is the optional extra installed" without importing it,
    and importing an optional dependency to find out whether it is optional is
    the failure the extra exists to avoid.
    """
    return None


def _prior_centre(graph: Graph) -> tuple[dict[str, Any] | None, str]:
    """Every node at its prior centre, or ``None`` and the reason why not.

    Split out so that a graph which cannot be centred keeps R4's answer with
    the failure NAMED, rather than the audit raising from inside a loop over
    latents that have nothing to do with it.
    """
    try:
        return prior_environment(graph), ""
    except Exception as error:  # noqa: BLE001 -- any failure is the same answer
        return None, f"{type(error).__name__}: {error}"


def _environment_at(graph: Graph, pinned: Mapping[str, Any]) -> dict[str, Any]:
    """The prior centre with ``pinned`` latents held at the given values.

    Repeats :func:`~bayesmith.dispatch.classify.prior_environment`'s scan for
    the reason that function's own docstring gives: the substituted values have
    to be derived DURING the scan, because a latent whose prior width is
    another latent's value cannot be centred before its ancestor has been. The
    only difference here is that a pinned latent takes the caller's value
    instead of its centre, and everything downstream of it is then evaluated at
    that value rather than at the centre.
    """
    from bayesmith.dispatch.classify import _latent_centre
    from bayesmith.graph.evaluate import apply_deterministic
    from bayesmith.graph.nodes import Const, Deterministic

    env: dict[str, Any] = {}
    for node in graph.nodes:
        if isinstance(node, Const):
            env[node.name] = node.value
        elif isinstance(node, Deterministic):
            env[node.name] = apply_deterministic(graph, node, env)
        elif isinstance(node, Probabilistic):
            if node.name in pinned:
                env[node.name] = jnp.asarray(pinned[node.name])
            elif node.is_latent:
                env[node.name] = _latent_centre(graph, node, env)
            else:
                env[node.name] = node.observed
    return env


def conditional_prior_verdicts(
    graph: Graph, at: Mapping[str, Any]
) -> dict[str, PriorVerdict]:
    """One verdict per latent WITH parents, with ``at``'s latents pinned.

    The verdict for a latent whose conditional cannot be built or cannot be
    integrated is ``UNVERIFIABLE``, which is the same answer R4's rule gives to
    a density it cannot resolve: not proper, and not a claim that it diverges.
    """
    env = _environment_at(graph, at)
    verdicts: dict[str, PriorVerdict] = {}
    for name in graph.latents:
        node = graph.node(name)
        if not tuple(node.parents):
            continue
        try:
            verdicts[name] = audit_prior(
                apply_probabilistic(graph, node, env), latent=name
            ).verdict
        except Exception:  # noqa: BLE001 -- any failure is the same answer
            verdicts[name] = PriorVerdict.UNVERIFIABLE
    return verdicts


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class ConditionalPriorRange:
    """Where each conditional prior was evaluated, where it was not proper, and
    **where the check declined to run**.

    Three states, not two, and the third is why this class was rewritten. An
    adversarial review found ``degenerate == ()`` carrying two meanings --
    "swept, and every cell was a density" and "never swept at all" -- with
    nothing downstream able to tell them apart. A refusal built on the first
    meaning then asserted a coverage the second did not have. That is the
    failure family ``CLAUDE.md`` opens with, reproduced inside a guard written
    to enforce it.

    So ``unresolved`` is a separate field from ``degenerate``, and
    :meth:`covers` is the question most callers actually have.

    Every field is a tuple. ``at_points`` was a ``dict``, which made this the
    only one of this package's registered artifact types carrying a mutable
    container. ``register_artifact_type`` refuses a mutable dataclass -- but it
    reads ``__dataclass_params__.frozen``, a SPELLING, and a ``dict`` field on
    a frozen dataclass walks straight past it. Measured: the instance was
    editable after construction and was the only registered artifact that was
    unhashable. Red line 7's subject exactly.
    """

    #: ``(parent, ((component, ...), ...))`` -- every probe, in full. The
    #: components are the raveled probe rather than its first element, because
    #: a parent with a shape is displaced element-wise and reading ``[0]``
    #: reports one corner of it as though it were the whole.
    at_points: tuple[tuple[str, tuple[tuple[float, ...], ...]], ...]
    #: ``(latent, parents, values, verdict)`` for every cell whose conditional
    #: was not a proper density. EVERY cell: a report that kept only the first
    #: could not be told from one that kept them all. ``values`` carries each
    #: pinned parent's FULL raveled probe rather than its first component --
    #: measured, a plated parent whose declared centres vary across the plate
    #: is probed at ``(0.5, 4.5, 8.5, 12.5)``, and quoting ``0.5`` names one
    #: element of the cell as though it were the cell.
    degenerate: tuple[
        tuple[str, tuple[str, ...], tuple[tuple[float, ...], ...], str], ...
    ]
    #: ``(latent, reason)`` for a conditional the sweep could not cover at all.
    unresolved: tuple[tuple[str, str], ...]

    def covers(self, latent: str) -> bool:
        """Was ``latent``'s conditional actually swept?"""
        return latent not in {name for name, _reason in self.unresolved}

    def points_for(self, parent: str) -> tuple[tuple[float, ...], ...]:
        return dict(self.at_points).get(parent, ())


def _sweep_points(
    graph: Graph, parent: str, centre: Mapping[str, Any]
) -> tuple[tuple[Any, ...], str]:
    """Where to probe ``parent``, and the reason there is nowhere if there is not.

    Tries the package's own grid first --
    :func:`~bayesmith.dispatch.plan._probe_values`, plus and minus one and
    three prior widths about the centre -- so a Gaussian hyperprior is probed
    at exactly the points the rest of the compiler probes it at.

    **It returns ``None`` for any non-Gaussian prior, and that used to end the
    sweep silently.** Measured by an adversarial review: ``StudentT``,
    ``Laplace``, ``Uniform``, ``LogNormal`` and ``Cauchy`` all fall through it,
    so an ordinary model with a StudentT hyperprior had **zero** points
    evaluated while the report still read as "no degeneracy found". The corner
    it missed carried 0.058 of the prior.

    The fallback is not a second grid: it is the same offsets read through
    :func:`_moments`, which is what ``audit_prior`` already uses to place its
    own window, clipped into the declared support. A non-Gaussian hyperprior is
    probed on the same footing rather than skipped, and only a prior with no
    usable window at all -- a discrete support, or one that cannot be built --
    is reported as unswept.
    """
    from bayesmith.dispatch.plan import KAPPA_PROBE_SIGMAS, _probe_values

    probes = _probe_values(graph, parent, centre)
    if probes is not None:
        return tuple(probes), ""
    try:
        declared = apply_probabilistic(graph, graph.node(parent), centre)
    except Exception as error:  # noqa: BLE001 -- any failure is the same answer
        return (), f"its prior could not be built ({type(error).__name__}: {error})"
    # `_support_bounds` returns `(lower, upper, RESOLVED)`. The third element
    # says whether the unwrapping loop reached a leaf constraint -- it is NOT a
    # discreteness flag, and reading it as one sent every non-Gaussian
    # hyperprior down the unswept path: safe, in that they refused rather than
    # being admitted, and wrong, because the fallback this exists for then
    # never ran once and an ordinary StudentT model was refused for a reason
    # that was not true of it.
    try:
        low, high, resolved = _support_bounds(declared)
    except Exception as error:  # noqa: BLE001
        return (), f"its support could not be read ({type(error).__name__}: {error})"
    if not resolved:
        return (), (
            "its declared support could not be unwrapped to a leaf constraint, "
            "so there is no interval to place probe points inside"
        )
    if bool(getattr(getattr(declared, "support", None), "is_discrete", False)):
        return (), (
            "its support is discrete, so a probe point between two atoms is a "
            "value this parent cannot take and a conditional evaluated there "
            "would be a corner the integral never reaches"
        )
    try:
        loc, scale = _moments(declared)
    except Exception as error:  # noqa: BLE001
        return (), f"its window could not be placed ({type(error).__name__}: {error})"
    seen: list[float] = []
    for sigmas in KAPPA_PROBE_SIGMAS:
        value = float(np.clip(loc + sigmas * scale, low, high))
        if np.isfinite(value) and value not in seen:
            seen.append(value)
    if not seen:
        return (), "no finite probe point lies inside its declared support"
    return tuple(jnp.asarray(value) for value in seen), ""


def conditional_prior_range_report(graph: Graph) -> ConditionalPriorRange:
    """Is each conditional prior proper across a declared range of its parents?

    ``p(x | tau)`` proper at the prior centre does not make it proper at every
    ``tau`` the outer integral visits, and the identity the residual route
    rests on --
    ``Z = INT p(tau) [ INT p(x|tau) p(d|x,tau) dx ] dtau`` -- needs the second.

    **The range is over the PARENTS, and the verdict is read off the DENSITY.**
    There is no coordinate on ``tau`` that separates a proper conditional from
    a degenerate one: measured at ``tau = 0``, ``shared_ancestor``'s ``x`` is
    degenerate while ``three_latent_chain``'s and ``mixed_radiometer``'s are
    ordinary, because each declares a different function of ``tau`` as its
    width. Any threshold on ``tau`` would have to admit all three or refuse all
    three. So this pins the parents and asks the one-dimensional rule what the
    realised conditional is.

    **Parents, plural, and the PRODUCT of their grids.** This swept one parent
    at a time and left the others at their centres. An adversarial review built
    the two-parent version of the same fault --
    ``jnp.where((a < 1) & (b < 1), inf, 1)`` -- and it sailed through: both
    ``a = 0.5`` and ``b = 0.5`` were listed as visited, the corner where they
    are low TOGETHER was never asked about, and it carries 5.18e-04 of the
    prior with ``marginal_log_density`` returning ``-inf`` on it. Every shipped
    fixture has exactly one parent per conditioned latent, so nothing separated
    "pin the parents that matter" from "pin one and hope".

    The cost is the product of the grids over ONE conditioned latent's own
    latent ancestors, so it is exponential in that number rather than in the
    graph's size. Measured over the shipped graphs the largest is 4 cells, and
    the largest reachable with today's four-point grid and two ancestors is 16.
    The count is not capped, because a cap is a number and R5 pre-authorises
    none here; a graph that ever makes this expensive is a measurement to act
    on rather than a threshold to guess at now.

    **What it does not do, recorded rather than gated.** A degeneracy strictly
    between the grid's points, or outside its ends, is not found -- closing
    that needs a claim about the MEASURE of the degenerate set, which is a
    threshold. A conditional whose parents cannot be probed goes to
    ``unresolved`` rather than passing quietly.
    """
    from bayesmith.exact.block import _ancestors

    centre, error = _prior_centre(graph)
    conditioned = tuple(
        name for name in graph.latents if tuple(graph.node(name).parents)
    )
    if centre is None:
        return ConditionalPriorRange(
            at_points=(),
            degenerate=(),
            unresolved=tuple(
                (name, f"this graph has no evaluable prior centre ({error})")
                for name in conditioned
            ),
        )

    ancestry = {
        child: tuple(
            name for name in graph.latents if name in _ancestors(graph, child)
        )
        for child in conditioned
    }
    grids: dict[str, tuple[Any, ...]] = {}
    reasons: dict[str, str] = {}
    for parent in {p for parents in ancestry.values() for p in parents}:
        points, reason = _sweep_points(graph, parent, centre)
        if points:
            grids[parent] = points
        else:
            reasons[parent] = reason

    degenerate: list[tuple[str, tuple[str, ...], tuple[float, ...], str]] = []
    unresolved: list[tuple[str, str]] = []
    used: dict[str, tuple[tuple[float, ...], ...]] = {}
    for child in conditioned:
        parents = ancestry[child]
        if not parents:
            unresolved.append(
                (
                    child,
                    (
                        "none of its parents is a latent, so this sweep has "
                        "nothing to vary and its conditional is a fixed density"
                    ),
                )
            )
            continue
        missing = [name for name in parents if name not in grids]
        if missing:
            unresolved.append(
                (child, f"{missing[0]} could not be probed: {reasons[missing[0]]}")
            )
            continue
        for combination in itertools.product(*(grids[name] for name in parents)):
            pins = dict(zip(parents, combination, strict=True))
            verdict = conditional_prior_verdicts(graph, pins).get(child)
            if verdict is not None and verdict is not PriorVerdict.PROPER:
                degenerate.append(
                    (
                        child,
                        parents,
                        tuple(
                            tuple(
                                float(component)
                                for component in np.asarray(pin).ravel()
                            )
                            for pin in combination
                        ),
                        verdict.value,
                    )
                )
        for parent in parents:
            used[parent] = tuple(
                tuple(float(component) for component in np.asarray(point).ravel())
                for point in grids[parent]
            )
    return ConditionalPriorRange(
        at_points=tuple(sorted(used.items())),
        degenerate=tuple(degenerate),
        unresolved=tuple(unresolved),
    )


def audit_graph_priors(graph: Graph) -> tuple[PriorAudit, ...]:
    """One :class:`PriorAudit` per latent, in the graph's own order.

    A latent inside ``graph.joint_prior.over`` is reported ``UNDECLARED`` and
    not audited. Its node-level density is ``ImproperUniform`` BY REQUIREMENT --
    ``diagnose/priors.py`` refuses a ``JeffreysPrior`` over a latent that also
    declares a proper prior of its own, because the graph-level term IS the
    declaration. Auditing it would report this package's mandated
    configuration as a user error and hand back a remedy that undoes it (§0.6).

    **A latent with parents is audited through its CONDITIONAL, which is what
    R5 changed and why.** R4 answered ``UNVERIFIABLE`` here, on the ground that
    a hierarchical prior is not a fixed density -- ``p(w)`` is defined only once
    ``s`` is integrated out. That is true of the MARGINAL, and it is the wrong
    question to ask of ``w``: nothing ever integrates against ``p(w)``. The
    joint prior factorises along the graph as ``prod p(theta_i | parents_i)``,
    every factor of that product is a conditional density, and whether each one
    is proper is both answerable and the thing that decides whether ``Z``
    exists. So the conditional is built at the prior centre -- the same anchor
    :func:`~bayesmith.dispatch.classify.prior_environment` gives the classifier,
    rather than a second spelling of "where a latent sits" -- and handed to the
    same one-dimensional rule a root prior gets.

    **A latent with NO parents goes through that rule exactly as it did**, and
    measured over every shipped graph the only verdicts this restatement moves
    are the seven latents that have parents, spread over six graphs --
    ``three_latent_chain`` carries two of them, which is why the two counts are
    different and why saying "six latents" was wrong. The exact-class census R4
    closed
    on has none, so its census is untouched -- which is asserted rather than
    argued, in ``tests/dispatch/test_evidence_audit.py``.

    〔R5 Task 7 write-back. The R5 plan's section 0.15 restates propriety as a
    property of *the residual block's* joint prior. That cannot be evaluated
    here: this function is a PRE-compile refusal and no block partition exists
    yet, and moving the audit after ``compile_plan`` would turn
    ``lying_block_member``'s ``evidence_prior_normalised`` refusal into a raised
    ``StructureError``, measured. It is also weaker than the graph's own
    factorisation: three of the four fixtures the plan names as its target carry
    the offending latent in the EXACT block, so a residual-only audit would
    leave them unexamined rather than admitted for a reason. The plan's section
    0.20 then rules that such a latent "still passes R4's one-dimensional rule";
    measured, it did not and could not, because that rule short-circuits on
    ``node.parents`` and never reaches the density its reasoning describes.〕

    The environment is built once, not per latent, and a graph whose centre
    cannot be built keeps R4's answer with the failure named -- an audit that
    cannot run is not a verdict that the prior is improper.
    """
    covered: frozenset[str] = frozenset(
        getattr(graph.joint_prior, "over", ()) if graph.joint_prior is not None else ()
    )
    centre, centre_error = _prior_centre(graph)
    audits: list[PriorAudit] = []
    for name in graph.latents:
        if name in covered:
            audits.append(
                PriorAudit(
                    latent=name,
                    verdict=PriorVerdict.UNDECLARED,
                    normalised=None,
                    mass=None,
                    reason=(
                        "this latent has no node-level prior to audit: the "
                        "graph's joint_prior covers it, and that term is the "
                        "declaration. An evidence task needs a proper prior "
                        "here, which a graph-level reference prior is not"
                    ),
                )
            )
            continue
        node = graph.node(name)
        conditioned = bool(tuple(node.parents))
        if conditioned and centre is None:
            audits.append(
                _undecidable(
                    name,
                    f"this latent's prior is parameterised by "
                    f"{list(node.parents)}, and the conditional could not be "
                    f"built because this graph has no evaluable prior centre "
                    f"({centre_error}). p({name} | parents) is the density an "
                    f"evidence integrates against, and it was not reached",
                )
            )
            continue
        try:
            declared = apply_probabilistic(graph, node, centre if conditioned else {})
        except Exception as error:  # noqa: BLE001 -- any failure is the same answer
            audits.append(
                _undecidable(
                    name,
                    "this latent's declared prior could not be built"
                    + (
                        " at its parents' prior centre"
                        if conditioned
                        else " without an environment"
                    )
                    + f" ({type(error).__name__}: {error}), so its "
                    f"mass was not established",
                )
            )
            continue
        try:
            audits.append(audit_prior(declared, latent=name))
        except Exception as error:  # noqa: BLE001
            audits.append(
                _undecidable(
                    name,
                    f"integrating this prior raised "
                    f"{type(error).__name__}: {error}. An audit that cannot "
                    f"run is not a verdict that the prior is improper",
                )
            )
    return tuple(audits)


# The compiled residual problem lives at the leaf, in `bayesmith.compiled`, so
# `bridge/` can import it at module scope without closing a
# `bridge -> dispatch -> bridge` cycle. Re-exported here, where it was written
# and where callers look for it; that module carries the ruling.


def _graph_term_value(
    graph: Graph, term: Any, env: Mapping[str, Any], names, *, label: str
) -> Any:
    """One graph-level term's density, read the way ``log_joint`` reads it.

    Including the scalar requirement and the SLOT NAME in the refusal, both of
    which earlier versions of this function got wrong in turn.

    The first claimed the scalar check in its docstring and did not make it: a
    term returning shape ``(2,)`` made ``log_joint`` raise ``GraphError`` while
    this returned the vector. Calling ``graph_density`` fixed that and left a
    smaller divergence in place -- the label. ``log_joint`` passes the SLOT
    (``joint_prior``, ``evidence_terms[0]``); this passed the term's class name,
    so with two same-class evidence terms the refusal could not say which one
    failed, and could not tell a ``joint_prior`` from an ``evidence_terms[0]``
    at all. The only test matched ``"one scalar"``, which both spellings
    satisfy -- a guard reading a spelling weak enough to admit the divergence it
    existed to prevent.

    The slot is now passed in by the caller, which is the only place that knows
    it, and a test asserts the two messages name the same slot.
    """
    from bayesmith.graph.evaluate import graph_density

    return graph_density(graph, term, dict(env), label=label, names=names)



def _latent_shape(graph: Graph, name: str, env: Mapping[str, Any]) -> tuple[int, ...]:
    """A latent's value shape, defined for every distribution."""
    node = graph.node(name)
    distribution = apply_probabilistic(graph, node, dict(env))
    shape = tuple(distribution.batch_shape) + tuple(distribution.event_shape)
    if node.plate:
        shape = tuple(
            jnp.broadcast_shapes(shape, (graph.plate_size(node.plate[0]),))
        )
    return shape


def compile_evidence_problem(
    graph: Graph,
    *,
    exact_elimination: tuple[str, ...] = (),
) -> CompiledEvidenceProblem:
    """Split a graph's joint density into a prior and a likelihood.

    The partition: latent ``Probabilistic`` nodes and ``joint_prior`` on the
    prior side; observed ``Probabilistic`` nodes and every ``evidence_terms``
    entry on the likelihood side.

    **The identity this is graded against is not the obvious one.**
    ``log_prior + log_likelihood == log_joint`` bitwise is a property of the
    SUMMATION ORDER, not of the partition: ``log_joint`` accumulates one
    running total and splitting it into two reorders that sum. Measured over
    200 prior draws, ``two_observations`` is bitwise 186/200 and
    ``observation_reused_downstream`` 182/200, while single-observation
    fixtures are 200/200. So the test asserts each term is filed exactly once
    -- which is what catches a constant that moved sides -- grades the
    recomposition against a derived band over a declared seed set, and settles
    CORRECTNESS with an oracle that compares no implementations at all: the
    prior side must integrate to one.
    """
    from bayesmith.graph.evaluate import evaluate

    latents = frozenset(graph.latents)
    prior_nodes = tuple(
        node.name
        for node in graph.nodes
        if isinstance(node, Probabilistic) and node.observed is None
    )
    likelihood_nodes = tuple(
        node.name
        for node in graph.nodes
        if isinstance(node, Probabilistic) and node.observed is not None
    )
    prior_terms = list(prior_nodes)
    if graph.joint_prior is not None:
        prior_terms.append("joint_prior")
    likelihood_terms = list(likelihood_nodes)
    likelihood_terms.extend(
        f"evidence_terms[{index}]" for index in range(len(graph.evidence_terms))
    )

    def _side(values: Mapping[str, Any], *, observed: bool) -> Any:
        env = evaluate(graph, dict(values))
        total = jnp.zeros(())
        for node in graph.nodes:
            if not isinstance(node, Probabilistic):
                continue
            if (node.observed is not None) is not observed:
                continue
            term = apply_probabilistic(graph, node, env).log_prob(env[node.name])
            if node.observed_mask is not None:
                term = jnp.where(node.observed_mask, term, 0.0)
            total = total + jnp.sum(term)
        if observed:
            for index, term in enumerate(graph.evidence_terms):
                total = total + _graph_term_value(
                    graph, term, env, term.over, label=f"evidence_terms[{index}]"
                )
        elif graph.joint_prior is not None:
            total = total + _graph_term_value(
                graph, graph.joint_prior, env, graph.latents, label="joint_prior"
            )
        return total

    def log_prior(values: Mapping[str, Any]) -> Any:
        return _side(values, observed=False)

    def log_likelihood(values: Mapping[str, Any]) -> Any:
        return _side(values, observed=True)

    def prior_sample(key: Any) -> dict[str, Any]:
        """A draw from the declared prior, ancestrally.

        Routed through ``to_numpyro`` with ``observed={}`` -- the bridge's own
        prior-predictive convention -- rather than walked here. A
        ``Probabilistic`` node's ``dist_fn`` returns one object that owns both
        ``sample`` and ``log_prob``, and its docstring gives that as the reason
        the field exists: a hand-written ancestral walk is the second
        implementation those two could disagree across.

        A prior with no ``sample`` -- an improper one -- raises here rather than
        returning a number. That is the outcome R4 already refuses as
        ``evidence_prior_proper``, and a nested sampler needs to be told before
        it starts rather than after.
        """
        from numpyro import handlers

        traced = handlers.trace(handlers.seed(_to_numpyro(graph), key)).get_trace(
            observed={}
        )
        return {name: site["value"] for name, site in traced.items() if name in latents}

    eliminated = frozenset(exact_elimination)
    residual = tuple(name for name in graph.latents if name not in eliminated)

    # The parameter layout, from the DISTRIBUTION's own shape broadcast against
    # the plate.
    #
    # Not a `graph.shape` lookup: `Graph` has no `.shape` attribute at all, so
    # the first draft -- guarded by `hasattr(graph, "shape")` -- reported every
    # parameter as a scalar, silently and for every graph. No mutation of that
    # line could show it, because the mutant and the code agreed.
    #
    # And not `node_shape`, which is the obvious answer and reads `loc` off the
    # distribution: an `ImproperUniform` latent has no `loc`, so a residual
    # problem carrying one would raise here rather than report its layout.
    # `batch_shape + event_shape`, broadcast with the plate, is what
    # `to_numpyro` opens the site at and is defined for every distribution;
    # `test_the_layout_agrees_with_node_shape_where_node_shape_applies` pins
    # the two together on the fixtures where both are defined.
    env = prior_environment(graph)
    shapes = tuple((name, _latent_shape(graph, name, env)) for name in residual)

    return CompiledEvidenceProblem(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        prior_sample=prior_sample,
        exact_elimination=tuple(exact_elimination),
        residual_parameters=residual,
        shapes=shapes,
        prior_terms=tuple(prior_terms),
        likelihood_terms=tuple(likelihood_terms),
    )
