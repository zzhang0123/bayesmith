"""Reviewed production targets for the two-sided gate mutations.

Declarations use both scanner family and normalized syntax.  Resolution must
find exactly one manifest-pinned target already owned by the gate registry;
source drift therefore fails rather than silently selecting a neighbour.
"""

from __future__ import annotations

from dataclasses import dataclass

from tests.numerical_gates.boundary_core import GateSide
from tests.numerical_gates.mutation_harness import (
    ComparisonThresholdSide,
    MutationDirection,
    MutationSpec,
    MutationStrategy,
)
from tests.numerical_gates.registry import GATE_REGISTRY, MutationMode
from tests.numerical_gates.source_manifest import EXPECTED_SOURCE_MANIFEST


@dataclass(frozen=True, slots=True)
class TargetDeclaration:
    """A line-independent but exact checked-in production mutation target."""

    family: str
    syntax: str
    strategy: MutationStrategy
    true_side: GateSide | None = None
    threshold_side: ComparisonThresholdSide | None = None
    occurrence: int | None = None
    threshold_replacement: float | None = None


_MANIFEST_SYNTAX = {
    item.candidate_id: item.syntax for item in EXPECTED_SOURCE_MANIFEST
}
_ENTRY_BY_GATE = {entry.gate_id: entry for entry in GATE_REGISTRY}


def _both(
    family: str,
    syntax: str,
    strategy: MutationStrategy,
    *,
    true_side: GateSide | None = None,
    threshold_side: ComparisonThresholdSide | None = None,
    occurrence: int | None = None,
    threshold_replacement: float | None = None,
) -> tuple[TargetDeclaration, TargetDeclaration]:
    declaration = TargetDeclaration(
        family,
        syntax,
        strategy,
        true_side,
        threshold_side,
        occurrence,
        threshold_replacement,
    )
    return declaration, declaration


DIAGNOSE_GRAPH_DECLARATIONS = {
    "COUPLING:_classify_correlation:value-finite": _both(
        "decision_predicate",
        "not np.isfinite(value)",
        MutationStrategy.FORCE_GATE_SIDE,
        true_side=GateSide.REFUSED,
    ),
    "COUPLING:_classify_correlation:floor-finite": _both(
        "decision_predicate",
        "not np.isfinite(floor)",
        MutationStrategy.FORCE_GATE_SIDE,
        true_side=GateSide.REFUSED,
    ),
    "COUPLING:_classify_correlation:lower-noise-floor": _both(
        "decision_predicate",
        "value <= floor",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.REFUSED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "COUPLING:_classify_correlation:upper-noise-floor": _both(
        "decision_predicate",
        "value >= 1.0 - floor",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.REFUSED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "COUPLING:_condition_number:finite-spectrum": _both(
        "decision_predicate",
        "not np.isfinite(smallest) or not np.isfinite(largest)",
        MutationStrategy.FORCE_GATE_SIDE,
        true_side=GateSide.REFUSED,
    ),
    "COUPLING:_condition_number:positive-spectrum": _both(
        "decision_predicate",
        "smallest <= 0.0",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.REFUSED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "COUPLING:block_coupling:f-xx-spd": _both(
        "linalg_call_atom",
        "np.linalg.cholesky(f_xx)",
        MutationStrategy.LINALG_BOUNDARY,
    ),
    "COUPLING:block_coupling:f-tt-spd": _both(
        "linalg_call_atom",
        "np.linalg.cholesky(f_tt)",
        MutationStrategy.LINALG_BOUNDARY,
    ),
    "GRAPH:_names:duplicate-multiplicity": _both(
        "compare",
        "names.count(name) > 1",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.REFUSED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "MAP:map_estimate:finite-derivative-payload": _both(
        "decision_predicate",
        "bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)) & "
        "jnp.all(jnp.isfinite(hessian)))",
        MutationStrategy.FORCE_GATE_SIDE,
        true_side=GateSide.ADMITTED,
    ),
    "MAP:map_estimate:stationarity-floor": _both(
        "decision_predicate",
        "gradient_norm > gradient_floor",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.REFUSED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "MAP:map_estimate:relative-positive-curvature": _both(
        "compare",
        "smallest > curvature_floor",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "MAP:map_estimate:absolute-curvature": _both(
        "compare",
        "largest > absolute_curvature_floor",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
}


COLLAPSE_DECLARATIONS = {
    "COLLAPSE:pivots:finite": _both(
        "decision_predicate",
        "jnp.all(jnp.isfinite(pivots))",
        MutationStrategy.FORCE_GATE_SIDE,
        true_side=GateSide.ADMITTED,
    ),
    "COLLAPSE:pivots:relative-floor": _both(
        "compare",
        "pivots[:n_block] > floor",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
}


PILOT_DECLARATIONS = {
    "PILOT:quadratic_cc_crosses_floor:sampling-floor": _both(
        "decision_predicate",
        "quadratic_cc > floor",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "PILOT:ratio_exceeds_declared_multiple:declared-multiple": _both(
        "decision_predicate",
        "ratio > DECLARED_MULTIPLE",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
}


COSTS_DECLARATIONS = {
    "COSTS:share_is_dominant:dominance-share": _both(
        "decision_predicate",
        "share > DOMINANCE_SHARE",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "COSTS:gap_is_contested:contested-bandwidth": _both(
        "decision_predicate",
        "gap < CONTESTED_BANDWIDTH",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "COSTS:timing_noise_in_domain:proper-fraction": _both(
        "decision_predicate",
        "tol < 1.0",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
    "COSTS:cg_tol_positive:strictly-positive": _both(
        "decision_predicate",
        "tol > 0.0",
        MutationStrategy.SHIFT_COMPARISON,
        true_side=GateSide.ADMITTED,
        threshold_side=ComparisonThresholdSide.RIGHT,
    ),
}


EAGER_TARGETS = {
    "EAGER:LadderConfig:integer-threshold-domain": (
        "compare",
        "value < 0",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.REFUSED,
        ComparisonThresholdSide.RIGHT,
    ),
    "EAGER:LadderConfig:low-rank-fraction-domain": (
        "decision_predicate",
        "not 0.0 <= self.low_rank_fraction <= 1.0",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:LadderConfig:structure-tolerance-domain": (
        "compare",
        "self.structure_rtol < 0.0",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.REFUSED,
        ComparisonThresholdSide.RIGHT,
    ),
    "EAGER:array-normalization:shape-and-finiteness": (
        "decision_predicate",
        "not np.all(np.isfinite(array))",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:LogDetProblem:lambda-spd": (
        "decision_predicate",
        "not _is_positive_definite(lam)",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:factor-balance:exact-power-of-two-reversibility": (
        "decision_predicate",
        (
            "np.all(np.isfinite(scaled_left)) and "
            "np.all(np.isfinite(scaled_right)) and "
            "np.array_equal(restored_left, left_column) and "
            "np.array_equal(restored_right, right_column)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "EAGER:factor-reconstruction:layout-exactness": (
        "decision_predicate",
        "np.array_equal(canonical, value)",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "EAGER:factor-reduced:diagonal-certificate": (
        "decision_predicate",
        (
            "reduced_sign > 0.0 and "
            "np.all(np.isfinite(relative_diagonal_error)) and "
            "np.all(relative_diagonal_error < 1.0)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "EAGER:factor-reduced:qr-certificate": (
        "decision_predicate",
        (
            "reduced_sign > 0.0 and np.isfinite(reduced_eta) and "
            "(0.0 <= reduced_eta < 1.0) and "
            "np.isfinite(orthogonality_eta) and "
            "(0.0 <= orthogonality_eta < 1.0) and "
            "np.all(reduced_r_diagonal != 0.0)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "EAGER:factor-reduced:acceptance-budget": (
        "decision_predicate",
        ("np.isfinite(total_log_error_bound) and total_log_error_bound <= ceiling"),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "EAGER:symmetry:tolerant-representative": (
        "decision_predicate",
        "not _is_symmetric(value, rtol=rtol, atol=atol)",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:dense-condition:strict-dtype-ceiling": (
        "compare",
        "condition < ceiling",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.ADMITTED,
        ComparisonThresholdSide.RIGHT,
    ),
    "EAGER:lambda-logdet:subnormal-rescale": (
        "compare",
        "maximum < float(np.finfo(lam.dtype).tiny)",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.ADMITTED,
        ComparisonThresholdSide.RIGHT,
    ),
    "EAGER:finite:newton-stability-rho": (
        "compare",
        "rho <= 1.0",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.ADMITTED,
        ComparisonThresholdSide.RIGHT,
    ),
    "EAGER:state-space:block-chain-exactness": (
        "decision_predicate",
        "not _is_block_chain(dense, block_size, rtol=rtol, atol=atol)",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:state-space:payload-domain": (
        "decision_predicate",
        "not _is_symmetric(dense, rtol=rtol, atol=atol)",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:structured:exact-shape-and-spectrum": (
        "decision_predicate",
        (
            "reconstructed.shape != dense.shape or "
            "not np.array_equal(reconstructed, dense)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:spectral-radius:finite-measurement": (
        "decision_predicate",
        "not np.all(np.isfinite(x))",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:trace:certificate-domain": (
        "decision_predicate",
        "not 0.0 <= certificate < 1.0",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:trace:certificate-upper-bound": (
        "compare",
        "actual_rho > certificate",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.REFUSED,
        ComparisonThresholdSide.RIGHT,
    ),
    "EAGER:trace:tail-domain-and-order": (
        "decision_predicate",
        "not 0.0 <= rho < 1.0",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "EAGER:trace:exact-power-trace-evidence": (
        "decision_predicate",
        "np.all(np.isfinite(supplied)) and np.array_equal(supplied, derived)",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "EAGER:frozen-probes:identity-width-order": (
        "decision_predicate",
        "vectors.shape[1] != _n(lam)",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
}


EAGER_DECLARATIONS = {
    gate_id: _both(
        values[0],
        values[1],
        values[2],
        true_side=values[3],
        threshold_side=values[4],
    )
    for gate_id, values in EAGER_TARGETS.items()
}
EAGER_DECLARATIONS["EAGER:factor-projection:whitened-positive-spectrum"] = (
    TargetDeclaration(
        "compare",
        "smallest_eigenvalue <= 0.0",
        MutationStrategy.REPLACE_COMPARISON_THRESHOLD,
        GateSide.REFUSED,
        ComparisonThresholdSide.RIGHT,
        threshold_replacement=float.fromhex("0x1p-53"),
    ),
    TargetDeclaration(
        "compare",
        "smallest_eigenvalue <= 0.0",
        MutationStrategy.REPLACE_COMPARISON_THRESHOLD,
        GateSide.REFUSED,
        ComparisonThresholdSide.RIGHT,
        threshold_replacement=float.fromhex("-0x1.0000000000001p-52"),
    ),
)


PLAN_TARGETS = {
    "PLAN:multiplicity:index-and-gamma-domain": (
        "decision_predicate",
        "multiplicity >= _RHO_MULTIPLICITY_LIMIT",
    ),
    "PLAN:certificate:rho-domain-and-coverage": (
        "decision_predicate",
        "not 0.0 <= self.certified_rho < 1.0",
    ),
    "PLAN:certificate:error-budget-domain": (
        "decision_predicate",
        (
            "self.margin < 0.0 or self.tolerance <= 0.0 or "
            "(not 0.0 < self.tail_tolerance < self.tolerance)"
        ),
    ),
    "PLAN:certificate:optional-scale-domain": (
        "compare",
        "self.max_abs_lambda_logdet < 0.0",
    ),
    "PLAN:certificate:order-is-derived": (
        "decision_predicate",
        "self.order != expected_order",
    ),
    "PLAN:warmup:rho-inputs-and-margin": (
        "decision_predicate",
        "not np.isfinite(margin) or margin < 0.0",
    ),
    "PLAN:warmup:tail-fraction": (
        "decision_predicate",
        "not np.isfinite(tail_fraction) or not 0.0 < tail_fraction < 1.0",
    ),
    "PLAN:warmup:lambda-scale-inputs": (
        "decision_predicate",
        "not np.isfinite(lambda_logdet_margin) or lambda_logdet_margin < 0.0",
    ),
    "PLAN:warmup:x-norm-inputs": (
        "decision_predicate",
        "not np.isfinite(x_operator_norm_margin) or x_operator_norm_margin < 0.0",
    ),
    "PLAN:warmup:rho-roundoff-ceiling": (
        "decision_predicate",
        "not certified < 1.0",
    ),
    "PLAN:audit:retained-rho": (
        "compare",
        "value > certificate.certified_rho",
    ),
    "PLAN:audit:retained-lambda-scale": (
        "compare",
        "value > certificate.max_abs_lambda_logdet",
    ),
    "PLAN:audit:retained-x-norm": (
        "compare",
        "value > certificate.max_x_operator_norm",
    ),
    "PLAN:audit:retained-trace-evidence": (
        "decision_predicate",
        (
            "problem.trace_order != certificate.order or "
            "_retained_rank_exceeds_certificate(problem, certificate) or "
            "problem.exact_power_traces is None or "
            "(not _checked_power_traces_match(problem.lambda_matrix, "
            "problem.perturbation, problem.exact_power_traces, certificate.order))"
        ),
    ),
    "PLAN:measurement:x-norm-finite": (
        "decision_predicate",
        "not np.all(np.isfinite(x)) or not np.isfinite(actual_norm)",
    ),
    "PLAN:factory-certificate:order-and-rank": (
        "decision_predicate",
        "certificate.multiplicity < required_multiplicity",
    ),
    "PLAN:factory-certificate:lambda-scale": (
        "decision_predicate",
        "actual_base_scale > certificate.max_abs_lambda_logdet",
    ),
    "PLAN:factory-certificate:x-norm": (
        "decision_predicate",
        "actual_norm > certificate.max_x_operator_norm",
    ),
    "PLAN:canonical-probes:runtime-finite": (
        "decision_predicate",
        "not np.all(np.isfinite(values))",
    ),
    "PLAN:outward-arithmetic:positive-underflow": (
        "decision_predicate",
        "value == 0.0 or not np.isfinite(value)",
    ),
    "PLAN:frozen:probe-energy-range": (
        "decision_predicate",
        "not np.isfinite(total_energy) or total_energy > maximum",
    ),
    "PLAN:runtime-range:product": (
        "decision_predicate",
        "not np.isfinite(result) or result > maximum",
    ),
    "PLAN:runtime-range:sum": (
        "decision_predicate",
        "not np.isfinite(result) or result > maximum",
    ),
    "PLAN:gamma:operation-count-domain": (
        "decision_predicate",
        "product >= 1.0",
    ),
    "PLAN:frozen:x-bound-runtime-range": (
        "decision_predicate",
        "x_bound > maximum",
    ),
    "PLAN:frozen:intermediate-runtime-range": (
        "numerical_premise_call",
        (
            "_runtime_range_product(correction_bound, addition_factor, maximum, "
            "runtime_dtype, 'the frozen correction accumulation')"
        ),
    ),
    "PLAN:runtime:sigma-finite-and-positive": (
        "decision_predicate",
        "np.any(sigma <= 0.0)",
    ),
    "PLAN:runtime:expected-and-ulp-finite": (
        "decision_predicate",
        "ulp > certificate.tolerance",
    ),
    "PLAN:runtime:base-scale-range": (
        "decision_predicate",
        "base_scale > maximum_runtime_value",
    ),
    "PLAN:runtime:frozen-prerequisites-and-series": (
        "decision_predicate",
        "not np.isfinite(series_scale)",
    ),
    "PLAN:runtime:total-error-budget": (
        "decision_predicate",
        "total_error_bound > certificate.tolerance",
    ),
    "PLAN:runtime-call:scalar-and-dtype": (
        "decision_predicate",
        (
            "expected.itemsize > 4 and (not jax.config.x64_enabled) or "
            "any((dtype.kind != 'f' or dtype.itemsize < expected.itemsize "
            "for dtype in actual))"
        ),
    ),
    "PLAN:trace-factory:exact-evidence": (
        "decision_predicate",
        (
            "problem.exact_power_traces is None or not "
            "_checked_power_traces_match(problem.lambda_matrix, "
            "problem.perturbation, problem.exact_power_traces, certificate.order)"
        ),
    ),
    "PLAN:frozen-factory:probe-presence-width": (
        "decision_predicate",
        "problem.frozen_probes.values.shape[1] != _n(problem.lambda_matrix)",
    ),
}

_PLAN_SHIFT_THRESHOLD_SIDES = {
    "PLAN:multiplicity:index-and-gamma-domain": ComparisonThresholdSide.RIGHT,
    "PLAN:certificate:optional-scale-domain": ComparisonThresholdSide.RIGHT,
    "PLAN:audit:retained-rho": ComparisonThresholdSide.RIGHT,
    "PLAN:audit:retained-lambda-scale": ComparisonThresholdSide.RIGHT,
    "PLAN:audit:retained-x-norm": ComparisonThresholdSide.RIGHT,
    "PLAN:factory-certificate:order-and-rank": ComparisonThresholdSide.LEFT,
    "PLAN:factory-certificate:lambda-scale": ComparisonThresholdSide.RIGHT,
    "PLAN:factory-certificate:x-norm": ComparisonThresholdSide.RIGHT,
    "PLAN:frozen:x-bound-runtime-range": ComparisonThresholdSide.RIGHT,
    "PLAN:runtime:expected-and-ulp-finite": ComparisonThresholdSide.RIGHT,
    "PLAN:runtime:base-scale-range": ComparisonThresholdSide.RIGHT,
    "PLAN:runtime:total-error-budget": ComparisonThresholdSide.RIGHT,
}
_PLAN_NUMERIC_TARGETS = {
    "PLAN:frozen:intermediate-runtime-range",
}


def _plan_declarations() -> dict[
    str, tuple[TargetDeclaration, TargetDeclaration]
]:
    declarations: dict[str, tuple[TargetDeclaration, TargetDeclaration]] = {}
    for gate_id, (family, syntax) in PLAN_TARGETS.items():
        threshold_side = _PLAN_SHIFT_THRESHOLD_SIDES.get(gate_id)
        if gate_id in _PLAN_NUMERIC_TARGETS:
            declarations[gate_id] = _both(
                family,
                syntax,
                MutationStrategy.NUMERIC_BOUNDARY,
            )
        elif threshold_side is not None:
            declarations[gate_id] = _both(
                family,
                syntax,
                MutationStrategy.SHIFT_COMPARISON,
                true_side=GateSide.REFUSED,
                threshold_side=threshold_side,
            )
        else:
            declarations[gate_id] = _both(
                family,
                syntax,
                MutationStrategy.FORCE_GATE_SIDE,
                true_side=(
                    GateSide.ADMITTED
                    if gate_id == "PLAN:outward-arithmetic:positive-underflow"
                    else GateSide.REFUSED
                ),
            )
    return declarations


PLAN_DECLARATIONS = _plan_declarations()


LADDER_TARGETS = {
    "LADDER:sigma:payload-symmetry": (
        "decision_predicate",
        (
            "sigma.ndim == 1 or np.array_equal(sigma, sigma.T) or (not "
            "_is_symmetric(sigma, rtol=config.structure_rtol, "
            "atol=config.structure_atol))"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:sigma:finite-two-sum": (
        "decision_predicate",
        "not sigma_formation_valid",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.REFUSED,
        None,
    ),
    "LADDER:structure:compact-diagonal-positive": (
        "compare",
        "sigma > 0.0",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.ADMITTED,
        ComparisonThresholdSide.RIGHT,
    ),
    "LADDER:structure:diagonal-tolerance": (
        "predicate_call_atom",
        (
            "_is_diagonal(sigma, rtol=config.structure_rtol, "
            "atol=config.structure_atol)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
        1,
    ),
    "LADDER:structure:circulant-tolerance-spectrum": (
        "decision_predicate",
        (
            "_is_circulant(sigma, rtol=config.structure_rtol, "
            "atol=config.structure_atol)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:structure:toeplitz-tolerance": (
        "decision_predicate",
        (
            "_is_toeplitz(sigma, rtol=config.structure_rtol, "
            "atol=config.structure_atol)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:structure:kronecker-evidence": (
        "decision_predicate",
        "reconstructed.shape == sigma.shape and np.array_equal(reconstructed, sigma)",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:sigma:symmetry-spd-condition": (
        "decision_predicate",
        (
            "sigma_symmetric and _is_positive_definite(sigma, "
            "rtol=config.structure_rtol, atol=config.structure_atol)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:rank:evidence": (
        "decision_predicate",
        "rank_evidence_valid",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:rho:measurement": (
        "numerical_premise_call",
        "spectral_radius(lam, perturb)",
        MutationStrategy.NUMERIC_BOUNDARY,
        None,
        None,
    ),
    "LADDER:finite:payload-rho": (
        "compare",
        "finite_payload_rho <= 1.0",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.ADMITTED,
        ComparisonThresholdSide.RIGHT,
    ),
    "LADDER:determinant-lemma:payload": (
        "decision_predicate",
        (
            "problem.low_rank_factors is not None and rank_evidence_valid and "
            "sigma_formation_valid and sigma_exactly_symmetric"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:rung0:base": (
        "decision_predicate",
        (
            "sigma_formation_valid and bool(np.array_equal(sigma, lam)) and "
            "dense_arithmetic_resolved"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:rung1:low-rank-size": (
        "boolean_atom",
        "rank <= config.low_rank_fraction * n",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.ADMITTED,
        ComparisonThresholdSide.RIGHT,
    ),
    "LADDER:rung2:chain": (
        "decision_predicate",
        "chain_structure and sigma_spd and condition_resolved",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:rung3:structured": (
        "gate_qualifier",
        "structured",
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:rung4:dense": (
        "boolean_atom",
        "n <= config.dense_max_n",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.ADMITTED,
        ComparisonThresholdSide.RIGHT,
    ),
    "LADDER:rung5:finite-size": (
        "compare",
        "n <= config.finite_max_n",
        MutationStrategy.SHIFT_COMPARISON,
        GateSide.ADMITTED,
        ComparisonThresholdSide.RIGHT,
    ),
    "LADDER:rung5:finite-executable": (
        "decision_predicate",
        (
            "finite_size_qualified and finite_payload_stable and sigma_spd and "
            "(determinant_lemma_payload or dense_arithmetic_resolved)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:rung6:trace": (
        "decision_predicate",
        (
            "sigma_formation_valid and traces_verified and measured_rho_converges "
            "and rho_covers_input and (0.0 <= rho < 1.0)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
    "LADDER:rung7:frozen": (
        "decision_predicate",
        (
            "sigma_formation_valid and frozen_width_valid and "
            "(problem.trace_order is not None) and (problem.trace_order >= 0) "
            "and measured_rho_converges and rho_covers_input and "
            "(0.0 <= rho < 1.0)"
        ),
        MutationStrategy.FORCE_GATE_SIDE,
        GateSide.ADMITTED,
        None,
    ),
}


LADDER_DECLARATIONS = {
    gate_id: _both(
        values[0],
        values[1],
        values[2],
        true_side=values[3],
        threshold_side=values[4],
        occurrence=(values[5] if len(values) == 6 else None),
    )
    for gate_id, values in LADDER_TARGETS.items()
}


def _resolve_target(gate_id: str, declaration: TargetDeclaration) -> str:
    entry = _ENTRY_BY_GATE[gate_id]
    matches = sorted(
        {
            target_id
            for target_id in entry.mutation_target_ids
            if f"::{declaration.family}::" in target_id
            and _MANIFEST_SYNTAX[target_id] == declaration.syntax
            and (
                declaration.occurrence is None
                or target_id.endswith(f"::{declaration.occurrence}")
            )
        }
    )
    if len(matches) != 1:
        raise AssertionError(
            f"{gate_id} mutation declaration resolves {len(matches)} targets: "
            f"family={declaration.family!r}, syntax={declaration.syntax!r}"
        )
    return matches[0]


def _resolve_specs(
    declarations: dict[str, tuple[TargetDeclaration, TargetDeclaration]],
) -> tuple[MutationSpec, ...]:
    specs: list[MutationSpec] = []
    for gate_id in sorted(declarations):
        if _ENTRY_BY_GATE[gate_id].mutation_mode is not MutationMode.TWO_SIDED:
            continue
        tighten, loosen = declarations[gate_id]
        for direction, declaration in (
            (MutationDirection.TIGHTEN, tighten),
            (MutationDirection.LOOSEN, loosen),
        ):
            specs.append(
                MutationSpec(
                    gate_id=gate_id,
                    direction=direction,
                    target_id=_resolve_target(gate_id, declaration),
                    strategy=declaration.strategy,
                    true_side=declaration.true_side,
                    threshold_side=declaration.threshold_side,
                    threshold_replacement=declaration.threshold_replacement,
                )
            )
    return tuple(specs)


DIAGNOSE_GRAPH_MUTATION_SPECS = _resolve_specs(DIAGNOSE_GRAPH_DECLARATIONS)
COSTS_MUTATION_SPECS = _resolve_specs(COSTS_DECLARATIONS)
PILOT_MUTATION_SPECS = _resolve_specs(PILOT_DECLARATIONS)
COLLAPSE_MUTATION_SPECS = _resolve_specs(COLLAPSE_DECLARATIONS)
EAGER_MUTATION_SPECS = _resolve_specs(EAGER_DECLARATIONS)
LADDER_MUTATION_SPECS = _resolve_specs(LADDER_DECLARATIONS)
PLAN_MUTATION_SPECS = _resolve_specs(PLAN_DECLARATIONS)


__all__ = [
    "COLLAPSE_DECLARATIONS",
    "COLLAPSE_MUTATION_SPECS",
    "COSTS_DECLARATIONS",
    "COSTS_MUTATION_SPECS",
    "DIAGNOSE_GRAPH_DECLARATIONS",
    "DIAGNOSE_GRAPH_MUTATION_SPECS",
    "EAGER_DECLARATIONS",
    "EAGER_MUTATION_SPECS",
    "EAGER_TARGETS",
    "LADDER_DECLARATIONS",
    "LADDER_MUTATION_SPECS",
    "LADDER_TARGETS",
    "PILOT_DECLARATIONS",
    "PILOT_MUTATION_SPECS",
    "PLAN_DECLARATIONS",
    "PLAN_MUTATION_SPECS",
    "PLAN_TARGETS",
    "TargetDeclaration",
]
