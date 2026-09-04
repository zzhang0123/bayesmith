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
import math
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import jax.numpy as jnp
import numpy as np

from bayesmith.artifacts._codec import register_artifact_type
from bayesmith.artifacts.results import EvidenceComponent
from bayesmith.dispatch.collapse import observed_descendants
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.fisher import dense_operator
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.precision import per_sample_sigma
from bayesmith.graph.evaluate import apply_probabilistic
from bayesmith.graph.graph import Graph

__all__ = [
    "EVIDENCE_COMPONENT_NAMES",
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


@dataclasses.dataclass(frozen=True, slots=True)
class PriorAudit:
    """One latent's prior, and the two things an evidence needs of it."""

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


def _moments(distribution: Any) -> tuple[float, float]:
    """A centre and a scale to place the quadrature window on.

    Read off the distribution when it has finite moments, and defaulted to
    ``(0, 1)`` when it does not -- a flat density has no mean, and the whole
    point of the window sequence is that it does not need one.
    """
    centre, scale = 0.0, 1.0
    try:
        mean = float(np.asarray(distribution.mean))
        if np.isfinite(mean):
            centre = mean
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        pass
    try:
        variance = float(np.asarray(distribution.variance))
        if np.isfinite(variance) and variance > 0.0:
            scale = float(np.sqrt(variance))
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        pass
    return centre, scale


def _support_bounds(distribution: Any) -> tuple[float, float]:
    """The declared support's bounds, as floats, ``(-inf, inf)`` when unbounded.

    Read off the constraint's own ``lower_bound`` / ``upper_bound`` rather than
    from the constraint's TYPE: an interval constraint carries its numbers, and
    a constraint that carries none is unbounded whatever it is called.

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
    while hasattr(support, "base_constraint") and seen < 8:
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
    return lower, upper


@functools.cache
def _rule() -> tuple[np.ndarray, np.ndarray]:
    """The Gauss-Legendre nodes and weights, computed once per process."""
    return np.polynomial.legendre.leggauss(_NODES)


def _mass_on(
    distribution: Any, lower: float, upper: float, panels: int
) -> float:
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


def audit_prior(distribution: Any, latent: str = "") -> PriorAudit:
    """Whether this declared density has finite mass, and whether it is one.

    Decided by integrating the density the model declares, never by reading a
    type name. ``isinstance(d, dist.ImproperUniform)`` is walked past by any
    user subclass whose ``log_prob`` returns a constant -- built and run in
    ``tests/dispatch/test_evidence_audit.py`` -- and ``unwrap`` strips only
    ``Independent``, so nothing else in the package looks through one either.

    The window sequence is the whole method. A proper density's mass stops
    moving as the window widens; an improper one's grows with it. That is what
    the two words MEAN, so reading the sequence is reading the property rather
    than a proxy for it.
    """
    lower, upper = _support_bounds(distribution)
    centre, scale = _moments(distribution)

    masses: list[float] = []
    for width in _WINDOWS:
        low = max(lower, centre - width * scale)
        high = min(upper, centre + width * scale)
        if not (high > low):
            return PriorAudit(
                latent=latent,
                verdict=PriorVerdict.UNVERIFIABLE,
                normalised=None,
                mass=None,
                reason=(
                    "the declared support and the density's own scale leave no "
                    "interval to integrate over, so this audit cannot say "
                    "whether the prior has finite mass"
                ),
            )
        panels = max(8, int(_PANELS_PER_SCALE * (high - low) / scale))
        mass = _mass_on(distribution, low, high, panels)
        if not np.isfinite(mass):
            return PriorAudit(
                latent=latent,
                verdict=PriorVerdict.UNVERIFIABLE,
                normalised=None,
                mass=None,
                reason=(
                    f"the mass over [{low:.6g}, {high:.6g}] is not a finite "
                    f"number, so this audit cannot say whether the prior is "
                    f"proper. That is not the same as improper: one is a "
                    f"property of the prior, the other is a limit of this check"
                ),
            )
        masses.append(mass)

    settled = masses[-1]
    previous = masses[-2]
    if settled <= 0.0:
        return PriorAudit(
            latent=latent,
            verdict=PriorVerdict.UNVERIFIABLE,
            normalised=None,
            mass=float(settled),
            reason=(
                "the density integrates to zero over every window tried, so "
                "there is nothing to normalise and nothing to call improper"
            ),
        )
    grew = abs(settled - previous) > 1e-6 * max(abs(settled), 1.0)
    if grew:
        return PriorAudit(
            latent=latent,
            verdict=PriorVerdict.IMPROPER,
            normalised=None,
            mass=None,
            reason=(
                f"the mass keeps growing as the window widens "
                f"({previous:.6g} -> {settled:.6g}), so the integral diverges "
                f"and p(d) is undefined for this model"
            ),
        )
    normalised = bool(abs(settled - 1.0) <= 1e-6)
    return PriorAudit(
        latent=latent,
        verdict=PriorVerdict.PROPER,
        normalised=normalised,
        mass=float(settled),
        reason=(
            f"the mass settles at {settled:.9g}"
            + ("" if normalised else ", which is finite but not one")
        ),
    )


def audit_graph_priors(graph: Graph) -> tuple[PriorAudit, ...]:
    """One :class:`PriorAudit` per latent, in the graph's own order.

    A latent inside ``graph.joint_prior.over`` is reported ``UNDECLARED`` and
    not audited. Its node-level density is ``ImproperUniform`` BY REQUIREMENT --
    ``diagnose/priors.py`` refuses a ``JeffreysPrior`` over a latent that also
    declares a proper prior of its own, because the graph-level term IS the
    declaration. Auditing it would report this package's mandated configuration
    as a user error and hand back a remedy that undoes it (§0.6).
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
        audits.append(audit_prior(apply_probabilistic(graph, node, {}), latent=name))
    return tuple(audits)
