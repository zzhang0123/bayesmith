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
    "ExactAssembly",
    "PriorAudit",
    "PriorVerdict",
    "assemble_exact",
    "audit_graph_priors",
    "audit_prior",
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


def audit_graph_priors(graph: Graph) -> tuple[PriorAudit, ...]:
    """One :class:`PriorAudit` per latent, in the graph's own order.

    A latent inside ``graph.joint_prior.over`` is reported ``UNDECLARED`` and
    not audited. Its node-level density is ``ImproperUniform`` BY REQUIREMENT --
    ``diagnose/priors.py`` refuses a ``JeffreysPrior`` over a latent that also
    declares a proper prior of its own, because the graph-level term IS the
    declaration. Auditing it would report this package's mandated
    configuration as a user error and hand back a remedy that undoes it (§0.6).

    **A latent with parents is UNVERIFIABLE, and that arm exists because six of
    this package's own shipped fixtures crashed without it.** The first version
    called ``apply_probabilistic(graph, node, {})`` on every latent; that
    function reads ``env[parent]`` for each parent, so a hierarchical prior --
    ``s ~ HalfNormal(1); w ~ Normal(0, s)`` -- raised ``KeyError``.
    ``diamond_ancestor``, ``indirect_ancestor``, ``mixed_radiometer``,
    ``orphaned_child_latent``, ``shared_ancestor`` and ``three_latent_chain``
    all did. The plan's step 4.3 required exactly this measurement before
    widening and it was not run; an adversarial review ran it.

    The refusal is not a limitation of the arithmetic. A hierarchical prior is
    not a fixed density at all -- ``p(w)`` is only defined after ``s`` is
    integrated out -- so there is no single ``p(theta)`` for this audit to
    weigh, and answering would mean answering a different question.
    """
    covered: frozenset[str] = frozenset(
        getattr(graph.joint_prior, "over", ()) if graph.joint_prior is not None else ()
    )
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
        if tuple(node.parents):
            audits.append(
                _undecidable(
                    name,
                    f"this latent's prior is parameterised by "
                    f"{list(node.parents)}, so it is not a fixed density: "
                    f"p({name}) is defined only once those are integrated out. "
                    f"There is no single p(theta) here for a mass check to "
                    f"weigh",
                )
            )
            continue
        try:
            declared = apply_probabilistic(graph, node, {})
        except Exception as error:  # noqa: BLE001 -- any failure is the same answer
            audits.append(
                _undecidable(
                    name,
                    f"this latent's declared prior could not be built without "
                    f"an environment ({type(error).__name__}: {error}), so its "
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


# ------------------------------------------------- the compiled residual problem


@dataclasses.dataclass(frozen=True, slots=True)
class CompiledEvidenceProblem:
    """A residual evidence problem, with its prior and likelihood separated.

    Nested sampling needs ``log L(theta)`` apart from ``log pi(theta)``, and
    nothing else in this package produces that separation:
    :func:`~bayesmith.graph.evaluate.log_joint` sums every ``Probabilistic``
    node into ONE running total and then adds ``joint_prior`` and every
    ``evidence_terms`` entry in the same loop. Building the split is a compiler
    pass and bayesmith owns it -- design section 1.5's first clause is graph
    semantics, structure discovery, premise validation and task-aware
    compilation.

    **No FIELD is a Graph, and the densities close over one.** Both halves are
    asserted, because stating only the first is how a guard that reads a
    spelling gets written: an earlier test here walked the field values for a
    ``.nodes`` attribute, and they are functions, so it passed while the graph
    was reachable through every one of the three closures.

    Design line 192 says a compiled problem may contain the residual log
    density, transforms, constant terms and a reconstruction map, "but it may
    not re-interpret the Graph". What that forbids is a BACKEND doing the
    re-interpreting. bayesmith compiling the densities itself, by closing over
    the graph, is the contract kept rather than broken -- the adapter receives
    callables and cannot reach a graph without walking closure internals. The
    half that binds an adapter is asserted where an adapter exists: its module
    imports no ``Graph`` and its signature takes only a compiled problem.

    The parent type design line 482 names, ``CompiledProblem``, does not exist
    in this package -- the name appears six times in the design and nowhere in
    the source -- so this is built standalone and to line 192's contract, in
    order that the variant relation is a refactor rather than a redesign when
    the parent is written.

    Attributes:
        log_prior: ``theta -> log pi(theta)``, the latent nodes' own densities
            plus the graph-level ``joint_prior``.
        log_likelihood: ``theta -> log L(theta)``, the observed nodes' densities
            (honouring ``observed_mask``) plus every graph-level
            ``evidence_terms`` entry. Those hold graph-level LIKELIHOOD factors
            despite the field's name, which the R4 PLAN records and declines to
            rename (`2026-09-04-r4-evidence.md:650`; the close-out does not
            mention the field at all, and an earlier version of this line cited
            it). The assignment is asserted by consequence -- the prior side
            integrates to one -- and never by the name.
        prior_sample: ``key -> theta``, a draw from the prior. Strictly
            stronger than a prior that integrates to one, and it is what a
            nested sampler actually needs: ``improper_outside_prior`` raises
            here rather than returning a number.
        exact_elimination: latents already integrated in closed form.
        residual_parameters: latents the backend must integrate. Disjoint from
            ``exact_elimination`` -- ``InferencePlanRecord`` already refuses a
            name in both, and the two would otherwise disagree about which
            parameters a backend was handed.
        shapes: the parameter layout, ``(name, shape)`` per residual parameter.
        prior_terms, likelihood_terms: which term went to which side. Carried
            as data so that a test can assert every term is filed exactly once
            without re-deriving the partition it is grading.
    """

    log_prior: Any
    log_likelihood: Any
    prior_sample: Any
    exact_elimination: tuple[str, ...]
    residual_parameters: tuple[str, ...]
    shapes: tuple[tuple[str, tuple[int, ...]], ...]
    prior_terms: tuple[str, ...]
    likelihood_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        both = sorted(set(self.exact_elimination) & set(self.residual_parameters))
        if both:
            raise ValueError(
                f"{both} are named as both eliminated and residual; an "
                "eliminated parameter is precisely one the problem does not "
                "carry"
            )
        shared = sorted(set(self.prior_terms) & set(self.likelihood_terms))
        if shared:
            raise ValueError(
                f"{shared} are filed on both the prior and the likelihood "
                "side; a term counted twice is the failure an evidence layer "
                "exists to prevent"
            )


def _graph_term_value(graph: Graph, term: Any, env: Mapping[str, Any], names) -> Any:
    """One graph-level term's density, read the way ``log_joint`` reads it.

    Including the scalar requirement, which an earlier version of this function
    claimed in that sentence and did not enforce: measured, a term returning
    shape ``(2,)`` made ``log_joint`` raise ``GraphError`` while this returned
    the vector. ``graph_density`` is called rather than re-derived, so the two
    cannot drift.
    """
    from bayesmith.graph.evaluate import graph_density

    return graph_density(graph, term, dict(env), label=_term_label(term), names=names)


def _term_label(term: Any) -> str:
    return getattr(term, "__class__", type(term)).__name__



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
            for term in graph.evidence_terms:
                total = total + _graph_term_value(graph, term, env, term.over)
        elif graph.joint_prior is not None:
            total = total + _graph_term_value(
                graph, graph.joint_prior, env, graph.latents
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
