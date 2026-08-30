"""Local coupling between two latent blocks, measured from precision.

The number controlling a two-block Gaussian Gibbs sweep is the largest
canonical correlation, not the largest entry of a correlation matrix.  This
module measures it at a caller-supplied point and also reports the condition
numbers paid by conditional, marginalised, and joint dynamics.

The implementation never forms a covariance.  If ``F`` is the local Fisher
information plus prior precision and ``L_x L_x.T = F_xx`` and
``L_t L_t.T = F_tt``, the canonical correlations are the singular values of
``L_x^-1 F_xt L_t^-T``.  The marginal precision is measured as
``L_t (I - M.T M) L_t.T``; it is not inferred from the largest correlation.

``block_coupling`` itself is an eager design-time diagnostic.  A future
traced consumer must build affine blocks with
``unchecked_operator(..., probe_gaussian=False)`` after an eager Gaussian
check.  ``local_block(..., priors=True)`` performs a concrete Gaussian probe
and raises ``TracerBoolConversionError`` under ``jit`` (M3).

This is a Laplace quantity: it describes one tangent and can miss multimodal
or scale-dependent geometry completely.  ``blind_to`` therefore travels in
every report rather than living only in this prose.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import ClassVar

import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_triangular

from bayesmith.diagnose.local import (
    check_differentiable,
    check_observed_have_locs,
    latent_values,
    local_block,
    refuse_ambient_float32,
    refuse_single_precision,
    resolve_names,
)
from bayesmith.errors import GraphError
from bayesmith.exact.fisher import dense_operator, fisher_information
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.gls import precision_from_graph
from bayesmith.graph.graph import Graph


@dataclasses.dataclass(frozen=True)
class Measured:
    """A numerical coupling that stands above its arithmetic noise floor."""

    value: float
    floor: float
    n_correlations: int
    verdict: ClassVar[str] = "measured"


@dataclasses.dataclass(frozen=True)
class Refused:
    """A diagnostic whose arithmetic cannot support a numerical verdict."""

    reason: str
    verdict: ClassVar[str] = "refused"


CorrelationVerdict = Measured | Refused


@dataclasses.dataclass(frozen=True)
class CouplingReport:
    """Canonical coupling and the three condition numbers at one point.

    Arrays are NumPy values on purpose.  A frozen dataclass can carry them as
    ordinary eager results; putting them in an ``eqx.Module`` static field
    makes a second trace compare array-valued pytree metadata.
    """

    first: tuple[str, ...]
    second: tuple[str, ...]
    correlation: CorrelationVerdict
    canonical_correlations: np.ndarray
    kappa_cond: float
    kappa_marg: float
    kappa_joint: float
    blind_to: tuple[str, ...] = ("gaussian-only",)


def _classify_correlation(
    value: float, *, floor: float, n_correlations: int
) -> CorrelationVerdict:
    """Apply the closed noise-floor boundary (D74)."""
    if not np.isfinite(value):
        return Refused(
            "the canonical correlation is non-finite. Check the local "
            "parameterisation and evaluate at a finite point before using "
            "this diagnostic."
        )
    if not np.isfinite(floor):
        return Refused(
            "the whitening noise floor is non-finite, so the local precision "
            "does not retain enough arithmetic to classify coupling. "
            "Reparameterise the blocks or evaluate at a better-conditioned "
            "point."
        )
    if value <= floor:
        return Refused(
            f"canonical correlation {value:.3e} is at or below its "
            f"{floor:.3e} whitening noise floor. A comfortable-looking small "
            "number from ill-conditioned whitening is not evidence of low "
            "coupling; reparameterise the blocks or evaluate at a better-"
            "conditioned point."
        )
    if value >= 1.0 - floor:
        return Refused(
            f"canonical correlation {value:.3e} is at or within its "
            f"{floor:.3e} whitening noise floor of one. Arithmetic cannot "
            "distinguish this from a singular cross-block ridge; "
            "reparameterise or combine the redundant directions before "
            "choosing a sampler."
        )
    return Measured(
        value=float(value),
        floor=float(floor),
        n_correlations=int(n_correlations),
    )


def _condition_number(matrix: np.ndarray) -> float:
    """Spectral condition number of a symmetric positive matrix."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if not np.isfinite(smallest) or not np.isfinite(largest):
        return float("nan")
    if smallest <= 0.0:
        return float("inf")
    return largest / smallest


def block_coupling(
    graph: Graph,
    first: Sequence[str] | str,
    second: Sequence[str] | str,
    *,
    at: dict[str, jnp.ndarray],
) -> CouplingReport:
    """Measure local coupling and three strategy condition numbers.

    Args:
        graph: model whose observed densities and declared priors define the
            local posterior precision.
        first, second: disjoint, non-empty latent blocks.  A bare name is
            accepted for a one-latent block.
        at: the linearisation point.  It is deliberately required: this
            module does not choose a mode.  Missing latent values use their
            declared centres in the same way as other local diagnostics.

    Returns:
        A frozen :class:`CouplingReport`.  ``correlation`` is two-valued; a
        value at either whitening boundary is :class:`Refused`, never a claim
        of low coupling or a roundoff-perfect ridge.

    Raises:
        GraphError: if the blocks overlap, or from the graph/local diagnostic
            guards for invalid names, dtypes, points, or likelihoods.
        NotGaussian: if a selected latent has no Gaussian prior curvature.
    """
    refuse_ambient_float32(doing="block_coupling's local posterior geometry")
    first_names = resolve_names(graph, first)
    second_names = resolve_names(graph, second)
    overlap = sorted(set(first_names) & set(second_names))
    if overlap:
        raise GraphError(
            f"block_coupling needs two disjoint latent blocks, but {overlap} "
            "appears in both. Put each latent in exactly one block."
        )

    names = first_names + second_names
    values = latent_values(graph, at)
    check_differentiable(graph, names, values)
    check_observed_have_locs(graph, values)

    block = local_block(graph, names, values, priors=True)
    refuse_single_precision(
        dense_operator(block), doing="block_coupling's local design"
    )
    selected = {name: values[name] for name in names}
    outside = {name: value for name, value in values.items() if name not in names}
    precision = precision_at(graph, values)
    matrix = fisher_information(
        block,
        precision=precision,
        include_prior=True,
        precision_of=precision_from_graph(graph, outside),
        centre=selected,
    )
    dense = np.asarray(matrix.values)

    split = matrix.spans[len(first_names) - 1][1]
    f_xx = dense[:split, :split]
    f_xt = dense[:split, split:]
    f_tt = dense[split:, split:]
    try:
        l_x = np.linalg.cholesky(f_xx)
        l_t = np.linalg.cholesky(f_tt)
    except np.linalg.LinAlgError as error:
        raise GraphError(
            "block_coupling needs positive-definite within-block posterior "
            "precision at `at`; a Cholesky factor did not exist. Add a proper "
            "prior, remove a redundant latent, or choose a finite point."
        ) from error

    left_whitened = solve_triangular(l_x, f_xt, lower=True)
    whitened = solve_triangular(l_t, left_whitened.T, lower=True).T
    correlations = np.linalg.svd(whitened, compute_uv=False)

    kappa_x = _condition_number(f_xx)
    kappa_cond = _condition_number(f_tt)
    identity = np.eye(f_tt.shape[0], dtype=dense.dtype)
    marginal = l_t @ (identity - whitened.T @ whitened) @ l_t.T
    # Symmetrise the roundoff from the two matrix products before eigvalsh.
    marginal = 0.5 * (marginal + marginal.T)
    kappa_marg = _condition_number(marginal)
    kappa_joint = _condition_number(dense)

    floor = np.sqrt(kappa_x * kappa_cond) * np.finfo(dense.dtype).eps
    maximum = float(correlations[0])
    verdict = _classify_correlation(
        maximum,
        floor=float(floor),
        n_correlations=int(correlations.size),
    )
    return CouplingReport(
        first=first_names,
        second=second_names,
        correlation=verdict,
        canonical_correlations=np.asarray(correlations),
        kappa_cond=float(kappa_cond),
        kappa_marg=float(kappa_marg),
        kappa_joint=float(kappa_joint),
    )


__all__ = [
    "Measured",
    "Refused",
    "CorrelationVerdict",
    "CouplingReport",
    "block_coupling",
]
