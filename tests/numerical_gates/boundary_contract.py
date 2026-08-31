"""Reviewed per-gate direct-call contract for Task-3 boundary providers.

The provider modules do not own this allowlist.  Keeping it separate prevents a
suite from declaring every callable and thereby making the runtime
``undeclared_calls`` check vacuous.  Names are the concrete call labels retained
by observations; the required subset comes from the two reviewed Task-3 call
maps, while allowed extras are only the neighbours named by those maps.
"""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

_required: dict[str, frozenset[str]] = {}
_allowed: dict[str, frozenset[str]] = {}


def _add(
    gate_ids: Iterable[str],
    required: Iterable[str],
    allowed_extras: Iterable[str] = (),
) -> None:
    required_set = frozenset(required)
    allowed_set = required_set | frozenset(allowed_extras)
    if not required_set:
        raise AssertionError("a direct-call contract cannot be empty")
    for gate_id in gate_ids:
        if gate_id in _required:
            raise AssertionError(f"duplicate direct-call contract for {gate_id}")
        _required[gate_id] = required_set
        _allowed[gate_id] = allowed_set


_add(
    (
        "EAGER:LadderConfig:integer-threshold-domain",
        "EAGER:LadderConfig:low-rank-fraction-domain",
        "EAGER:LadderConfig:structure-tolerance-domain",
    ),
    ("LadderConfig.__post_init__",),
)
_add(
    ("EAGER:array-normalization:shape-and-finiteness",),
    ("_read_only_array",),
)
_add(
    ("EAGER:LogDetProblem:lambda-spd",),
    ("LogDetProblem.__init__",),
    ("_is_positive_definite",),
)
_add(
    ("EAGER:factor-balance:exact-power-of-two-reversibility",),
    ("_balanced_factor_columns", "low_rank_logdet", "dense_cholesky_logdet"),
)
_add(
    ("EAGER:factor-reconstruction:layout-exactness",),
    (
        "_matching_factor_reconstruction",
        "low_rank_logdet",
        "dense_cholesky_logdet",
    ),
)
_add(
    ("EAGER:symmetry:tolerant-representative",),
    ("_is_symmetric", "_is_positive_definite"),
    ("LogDetProblem.__init__",),
)
_add(
    ("EAGER:dense-condition:strict-dtype-ceiling",),
    ("_condition_certificate",),
)
_add(
    ("EAGER:lambda-logdet:subnormal-rescale",),
    ("lambda_logdet", "dense_cholesky_logdet"),
)
_add(
    ("EAGER:finite:newton-stability-rho",),
    ("_newton_stability", "finite_perturbation_logdet", "dense_cholesky_logdet"),
)
_add(
    ("EAGER:state-space:block-chain-exactness",),
    ("_is_block_chain", "state_space_logdet", "dense_cholesky_logdet"),
)
_add(
    ("EAGER:state-space:payload-domain",),
    ("state_space_logdet", "dense_cholesky_logdet"),
)
_add(
    ("EAGER:structured:exact-shape-and-spectrum",),
    (
        "_is_diagonal",
        "_is_circulant",
        "_is_toeplitz",
        "_circulant_eigenvalues",
        "structured_logdet",
        "dense_cholesky_logdet",
    ),
)
_add(
    ("EAGER:spectral-radius:finite-measurement",),
    ("spectral_radius",),
)
_add(
    (
        "EAGER:trace:certificate-domain",
        "EAGER:trace:certificate-upper-bound",
    ),
    ("_validate_strict_rho", "truncated_trace_logdet", "dense_cholesky_logdet"),
)
_add(
    ("EAGER:trace:tail-domain-and-order",),
    ("trace_log_tail_bound", "whole_trace_log_tail_bound", "choose_trace_order"),
)
_add(
    ("EAGER:trace:exact-power-trace-evidence",),
    ("_power_traces_match", "truncated_trace_logdet", "dense_cholesky_logdet"),
)
_add(
    ("EAGER:frozen-probes:identity-width-order",),
    ("frozen_hutchinson_trace_logdet", "dense_cholesky_logdet"),
)
_add(
    (
        "COUPLING:_classify_correlation:value-finite",
        "COUPLING:_classify_correlation:floor-finite",
        "COUPLING:_classify_correlation:lower-noise-floor",
        "COUPLING:_classify_correlation:upper-noise-floor",
    ),
    ("_classify_correlation",),
    ("block_coupling",),
)
_add(
    (
        "COUPLING:_condition_number:finite-spectrum",
        "COUPLING:_condition_number:positive-spectrum",
    ),
    ("_condition_number",),
    ("block_coupling",),
)
_add(
    (
        "COUPLING:block_coupling:f-xx-spd",
        "COUPLING:block_coupling:f-tt-spd",
    ),
    ("block_coupling",),
)
_add(
    (
        "MAP:map_estimate:finite-derivative-payload",
        "MAP:map_estimate:stationarity-floor",
        "MAP:map_estimate:relative-positive-curvature",
        "MAP:map_estimate:absolute-curvature",
    ),
    ("map_estimate",),
)
_add(("GRAPH:_names:duplicate-multiplicity",), ("_names",))

_add(("COSTS:gap_is_contested:contested-bandwidth",), ("gap_is_contested",))
_add(("COSTS:timing_noise_in_domain:proper-fraction",), ("timing_noise_in_domain",))
_add(("COSTS:cg_tol_positive:strictly-positive",), ("cg_tol_positive",))

_add(
    ("LADDER:sigma:payload-symmetry",),
    (
        "ladder._sigma_payload",
        "ladder.check_logdet_premises",
        "eager.dense_cholesky_logdet",
    ),
    (
        "eager.low_rank_logdet",
        "eager.state_space_logdet",
        "eager.structured_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:sigma:finite-two-sum",),
    (
        "ladder.check_logdet_premises",
        "eager._two_sum_error",
        "eager.finite_perturbation_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:structure:compact-diagonal-positive",),
    (
        "ladder._structure_request",
        "eager.structured_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:structure:diagonal-tolerance",),
    (
        "ladder._structure_request",
        "eager._is_diagonal",
        "eager.structured_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:structure:circulant-tolerance-spectrum",),
    (
        "ladder._structure_request",
        "eager._is_circulant",
        "eager._circulant_eigenvalues",
        "eager.structured_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:structure:toeplitz-tolerance",),
    (
        "ladder._structure_request",
        "eager._is_toeplitz",
        "eager.structured_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:structure:kronecker-evidence",),
    (
        "ladder._structure_request",
        "eager._is_positive_definite",
        "eager.structured_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:sigma:symmetry-spd-condition",),
    (
        "ladder.check_logdet_premises",
        "eager._is_symmetric",
        "eager._is_positive_definite",
        "eager._condition_certificate",
        "eager.dense_cholesky_logdet",
        "eager.finite_perturbation_logdet",
    ),
)
_add(
    ("LADDER:rank:evidence",),
    (
        "ladder.check_logdet_premises",
        "eager._algebraic_rank_bound",
        "eager._factor_projection_certificate",
        "eager.low_rank_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:rho:measurement",),
    (
        "ladder.check_logdet_premises",
        "eager.spectral_radius",
        "eager.truncated_trace_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:finite:payload-rho",),
    (
        "ladder.check_logdet_premises",
        "eager.spectral_radius",
        "eager.finite_perturbation_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
_add(
    ("LADDER:determinant-lemma:payload",),
    (
        "ladder.check_logdet_premises",
        "eager.low_rank_logdet",
        "eager.dense_cholesky_logdet",
    ),
)
for _gate_id, _methods in (
    ("LADDER:rung0:base", ("eager.lambda_logdet", "eager.dense_cholesky_logdet")),
    ("LADDER:rung1:low-rank-size", ("eager.low_rank_logdet", "eager.dense_cholesky_logdet")),
    ("LADDER:rung2:chain", ("eager.state_space_logdet", "eager.dense_cholesky_logdet")),
    ("LADDER:rung3:structured", ("eager.structured_logdet", "eager.dense_cholesky_logdet")),
    ("LADDER:rung4:dense", ("eager.dense_cholesky_logdet", "eager.finite_perturbation_logdet")),
    ("LADDER:rung5:finite-size", ("eager.finite_perturbation_logdet", "eager.dense_cholesky_logdet")),
    ("LADDER:rung5:finite-executable", ("eager.finite_perturbation_logdet", "eager.dense_cholesky_logdet")),
    ("LADDER:rung6:trace", ("eager.truncated_trace_logdet", "eager.dense_cholesky_logdet")),
    ("LADDER:rung7:frozen", ("eager.frozen_hutchinson_trace_logdet", "eager.dense_cholesky_logdet")),
):
    _add((_gate_id,), ("ladder.check_logdet_premises", *_methods))

_add(
    ("PLAN:multiplicity:index-and-gamma-domain",),
    ("plan._normalize_rho_multiplicity",),
)
_add(
    (
        "PLAN:certificate:error-budget-domain",
        "PLAN:certificate:optional-scale-domain",
        "PLAN:certificate:order-is-derived",
    ),
    ("plan.RhoCertificate",),
)
_add(
    (
        "PLAN:warmup:lambda-scale-inputs",
        "PLAN:warmup:x-norm-inputs",
    ),
    ("plan.certify_warmup_rho",),
)
for _gate_id, _method in (
    ("PLAN:audit:retained-rho", "plan.audit_retained_rho"),
    ("PLAN:audit:retained-lambda-scale", "plan.audit_retained_lambda_logdet"),
    ("PLAN:audit:retained-x-norm", "plan.audit_retained_operator_norm"),
    ("PLAN:audit:retained-trace-evidence", "plan.audit_retained_power_traces"),
    ("PLAN:measurement:x-norm-finite", "plan._checked_x_operator_norm"),
):
    _add((_gate_id,), (_method,))
_add(
    (
        "PLAN:factory-certificate:order-and-rank",
        "PLAN:factory-certificate:lambda-scale",
        "PLAN:factory-certificate:x-norm",
    ),
    ("plan._validate_plan_certificate",),
    ("plan.make_trace_log_plan", "plan.make_frozen_trace_log_plan"),
)
_add(
    ("PLAN:outward-arithmetic:positive-underflow",),
    (
        "plan._outward_nonnegative",
        "plan._outward_product",
        "plan._outward_sum",
        "plan._outward_quotient",
    ),
)
for _gate_id, _method in (
    ("PLAN:frozen:probe-energy-range", "plan._frozen_probe_energy_bounds"),
    ("PLAN:runtime-range:product", "plan._runtime_range_product"),
    ("PLAN:runtime-range:sum", "plan._runtime_range_sum"),
    ("PLAN:gamma:operation-count-domain", "plan._gamma_for_count"),
    ("PLAN:frozen:x-bound-runtime-range", "plan._validate_frozen_runtime_range"),
    (
        "PLAN:frozen:intermediate-runtime-range",
        "plan._validate_frozen_runtime_range",
    ),
):
    _add((_gate_id,), (_method,))
_add(
    (
        "PLAN:runtime:sigma-finite-and-positive",
        "PLAN:runtime:expected-and-ulp-finite",
        "PLAN:runtime:base-scale-range",
        "PLAN:runtime:total-error-budget",
    ),
    ("plan._validate_runtime_precision",),
)
_add(
    ("PLAN:runtime:frozen-prerequisites-and-series",),
    ("plan._validate_runtime_precision",),
    ("plan._outward_power_series",),
)
_add(
    ("PLAN:runtime-call:scalar-and-dtype",),
    ("plan._require_runtime_precision",),
    ("plan.TraceLogPlan.__call__", "plan.FrozenTraceLogPlan.__call__"),
)
_add(
    ("PLAN:trace-factory:exact-evidence",),
    ("plan.make_trace_log_plan",),
)
_add(
    ("PLAN:frozen-factory:probe-presence-width",),
    ("plan.make_frozen_trace_log_plan",),
)


REQUIRED_DIRECT_CALLS = MappingProxyType(dict(_required))
ALLOWED_DIRECT_CALLS = MappingProxyType(dict(_allowed))


__all__ = ["ALLOWED_DIRECT_CALLS", "REQUIRED_DIRECT_CALLS"]
