"""The exact log evidence, assembled as terms that can be audited one by one.

``log Z = log p(d | M) = INT p(d | theta, M) p(theta | M) d theta`` is what this
module builds, and the only thing that separates it from a posterior is that
every theta-independent constant is invisible in the second and load-bearing in
the first. A dropped constant does not make an evidence obviously wrong; it
makes it finite, plausible and wrong by a fixed number of nats.

So nothing here returns one scalar. The exact block's contribution is five
named terms, each derived from the block's own inputs, and the total is checked
against the square-root information route in ``marginal_log_density`` -- two
derivations that share no linear algebra, because §9.1 forbids proving a
constant with a second entry point to the arithmetic that produced it.

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

The third and fourth cancel exactly -- the prior contributes one row per block
degree of freedom, so the ``k`` in both is the same ``k``. They are reported
separately anyway. They come from different places (the prior's own normaliser,
and the Gaussian integral's), a change to either alone is a real change, and a
pre-cancelled pair is a decomposition that has already decided what the reader
is allowed to see.

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
import math
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from bayesmith.artifacts.results import EvidenceComponent
from bayesmith.dispatch.collapse import observed_descendants
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.fisher import dense_operator
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.precision import per_sample_sigma
from bayesmith.graph.graph import Graph

__all__ = [
    "EVIDENCE_COMPONENT_NAMES",
    "ExactAssembly",
    "assemble_exact",
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
        unknown = sorted(set(names) - set(EVIDENCE_COMPONENT_NAMES))
        if unknown:
            raise ValueError(
                f"an exact assembly reported component names outside the "
                f"closed set: {unknown}. A provenance guard reads membership "
                f"of EVIDENCE_COMPONENT_NAMES, so a name it does not know is a "
                f"term nothing checks."
            )
        if len(names) != len(set(names)):
            raise ValueError(
                f"an exact assembly reported a term twice: {names}. Each "
                f"constant enters log Z once, and a duplicate is the shape of "
                f"the defect R4 Task 1 repaired one layer down."
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

    prior_mean = np.concatenate(
        [np.atleast_1d(np.asarray(block.prior_mean[n], dtype=float)) for n in block.names]
    )
    prior_std = np.concatenate(
        [
            np.broadcast_to(
                np.atleast_1d(np.asarray(block.prior_std[n], dtype=float)),
                (_width_of(block.shape[n]),),
            )
            for n in block.names
        ]
    )
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

    R4's admitted class is diagonal observation noise. A correlated covariance
    has a perfectly good exact evidence -- ``compress`` reads it through
    ``Precision`` with no special case -- but it has no DENSE ORACLE here, and
    §9.1 does not allow shipping a number whose only check is the route that
    produced it. So the class is refused rather than assembled ungated.
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
