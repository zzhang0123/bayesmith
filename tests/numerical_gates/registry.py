"""Typed semantic registry for every numerical gate in the Phase-Two scope."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from tests.numerical_gates.source_scan import (
    CandidateClassification,
    CandidateFamily,
    ManifestEntry,
    SourceCandidate,
    index_source_text,
)


class ThresholdProvenance(str, Enum):
    DERIVED = "derived"
    BORROWED = "borrowed"
    MAGIC = "magic"
    EXACT_DOMAIN = "exact_or_domain"
    API_CONTRACT = "api_contract"


class MutationMode(str, Enum):
    TWO_SIDED = "two_sided"
    STATIC_ONLY = "static_only"
    POLICY_ONLY = "policy_only"


class FixtureScalePolicy(str, Enum):
    NON_UNIT_REQUIRED = "non_unit_required"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class SourceAnchor:
    module: str
    qualname: str
    family: CandidateFamily


@dataclass(frozen=True, slots=True)
class AxisRange:
    name: str
    low: str
    endpoints: tuple[str, str]
    high: str
    extreme: str


@dataclass(frozen=True, slots=True)
class GateMetadata:
    """Literal, reviewable boundary semantics for one registered gate."""

    quantity: str
    threshold: str
    provenance: ThresholdProvenance
    admitted_outcome: str
    refused_outcome: str
    oracle: str
    axes: tuple[AxisRange, ...]
    fixture_scale_policy: FixtureScalePolicy


@dataclass(frozen=True, slots=True)
class GateEntry:
    gate_id: str
    source_candidate_ids: tuple[str, ...]
    mutation_target_ids: tuple[str, ...]
    conjunction_atom_ids: tuple[str, ...]
    dependencies: tuple[str, ...]
    expected_source_syntax: tuple[str, ...]
    source_classifications: tuple[CandidateClassification, ...]
    source_anchors: tuple[SourceAnchor, ...]
    module: str
    quantity: str
    threshold: str
    provenance: ThresholdProvenance
    admitted_outcome: str
    refused_outcome: str
    oracle: str
    axes: tuple[AxisRange, ...]
    fixture_scale_policy: FixtureScalePolicy
    mutation_mode: MutationMode
    tighten_witness: str
    loosen_witness: str
    static_reason: str | None
    static_atom_reasons: Mapping[str, str]
    atom_isolation_ambiguities: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SourceLocation:
    candidate_id: str
    module: str
    qualname: str
    family: CandidateFamily
    lineno: int
    syntax: str


class RegistryValidationError(ValueError):
    """Raised when semantic registry data is incomplete or stale."""


@dataclass(frozen=True, slots=True)
class _Seed:
    gate_id: str
    module: str
    qualname: str
    family: CandidateFamily


EAGER = "src/bayesmith/marginal/_logdet_eager.py"
LADDER = "src/bayesmith/marginal/_logdet_ladder.py"
PLAN = "src/bayesmith/marginal/_logdet_plan.py"
COUPLING = "src/bayesmith/diagnose/coupling.py"
MAP = "src/bayesmith/diagnose/map.py"
GRAPH = "src/bayesmith/graph/reduction.py"
COSTS = "src/bayesmith/dispatch/costs.py"
COLLAPSE = "src/bayesmith/dispatch/collapse.py"


def _seed_group(
    module: str,
    qualname: str,
    family: CandidateFamily,
    gate_ids: str,
) -> tuple[_Seed, ...]:
    return tuple(
        _Seed(gate_id, module, f"<module>.{qualname}", family)
        for gate_id in gate_ids.split()
    )


_SEEDS = (
    *_seed_group(
        EAGER,
        "LadderConfig.__post_init__",
        CandidateFamily.COMPARE,
        """EAGER:LadderConfig:integer-threshold-domain
        EAGER:LadderConfig:low-rank-fraction-domain
        EAGER:LadderConfig:structure-tolerance-domain""",
    ),
    *_seed_group(
        EAGER,
        "_read_only_array",
        CandidateFamily.FINITE_PREDICATE,
        "EAGER:array-normalization:shape-and-finiteness",
    ),
    *_seed_group(
        EAGER,
        "LogDetProblem.__init__",
        CandidateFamily.COMPARE,
        "EAGER:LogDetProblem:lambda-spd",
    ),
    *_seed_group(
        EAGER,
        "_balanced_factor_columns",
        CandidateFamily.COMPARE,
        "EAGER:factor-balance:exact-power-of-two-reversibility",
    ),
    *_seed_group(
        EAGER,
        "_factor_projection_certificate",
        CandidateFamily.COMPARE,
        """EAGER:factor-reconstruction:layout-exactness
        EAGER:factor-projection:finite-qr-arithmetic
        EAGER:factor-projection:whitened-positive-spectrum
        EAGER:factor-projection:error-budget
        EAGER:factor-base:condition-ceiling
        EAGER:factor-base:error-budget
        EAGER:factor-reduced:diagonal-certificate
        EAGER:factor-reduced:qr-certificate
        EAGER:factor-reduced:acceptance-budget""",
    ),
    *_seed_group(
        EAGER,
        "_is_positive_definite",
        CandidateFamily.COMPARE,
        "EAGER:symmetry:tolerant-representative",
    ),
    *_seed_group(
        EAGER,
        "_condition_certificate",
        CandidateFamily.COMPARE,
        "EAGER:dense-condition:strict-dtype-ceiling",
    ),
    *_seed_group(
        EAGER,
        "lambda_logdet",
        CandidateFamily.COMPARE,
        "EAGER:lambda-logdet:subnormal-rescale",
    ),
    *_seed_group(
        EAGER,
        "_newton_logdet",
        CandidateFamily.COMPARE,
        "EAGER:finite:newton-stability-rho",
    ),
    *_seed_group(
        EAGER,
        "_is_block_chain",
        CandidateFamily.COMPARE,
        "EAGER:state-space:block-chain-exactness",
    ),
    *_seed_group(
        EAGER,
        "state_space_logdet",
        CandidateFamily.FINITE_PREDICATE,
        "EAGER:state-space:payload-domain",
    ),
    *_seed_group(
        EAGER,
        "structured_logdet",
        CandidateFamily.COMPARE,
        "EAGER:structured:exact-shape-and-spectrum",
    ),
    *_seed_group(
        EAGER,
        "spectral_radius",
        CandidateFamily.FINITE_PREDICATE,
        "EAGER:spectral-radius:finite-measurement",
    ),
    *_seed_group(
        EAGER,
        "_validate_strict_rho",
        CandidateFamily.COMPARE,
        """EAGER:trace:actual-rho-strict
        EAGER:trace:certificate-domain
        EAGER:trace:certificate-upper-bound""",
    ),
    *_seed_group(
        EAGER,
        "choose_trace_order",
        CandidateFamily.COMPARE,
        "EAGER:trace:tail-domain-and-order",
    ),
    *_seed_group(
        EAGER,
        "truncated_trace_logdet",
        CandidateFamily.COMPARE,
        "EAGER:trace:exact-power-trace-evidence",
    ),
    *_seed_group(
        EAGER,
        "frozen_hutchinson_trace_logdet",
        CandidateFamily.COMPARE,
        "EAGER:frozen-probes:identity-width-order",
    ),
    *_seed_group(
        LADDER,
        "_sigma_payload",
        CandidateFamily.COMPARE,
        "LADDER:sigma:payload-symmetry",
    ),
    *_seed_group(
        LADDER,
        "check_logdet_premises",
        CandidateFamily.COMPARE,
        "LADDER:sigma:finite-two-sum",
    ),
    *_seed_group(
        LADDER,
        "_structure_request",
        CandidateFamily.COMPARE,
        """LADDER:structure:compact-diagonal-positive
        LADDER:structure:diagonal-tolerance
        LADDER:structure:circulant-tolerance-spectrum
        LADDER:structure:toeplitz-tolerance
        LADDER:structure:kronecker-evidence""",
    ),
    *_seed_group(
        LADDER,
        "check_logdet_premises",
        CandidateFamily.COMPARE,
        """LADDER:sigma:symmetry-spd-condition
        LADDER:rank:evidence
        LADDER:rho:measurement
        LADDER:finite:payload-rho
        LADDER:determinant-lemma:payload
        LADDER:rung0:base
        LADDER:rung1:low-rank-size
        LADDER:rung2:chain
        LADDER:rung3:structured
        LADDER:rung4:dense
        LADDER:rung5:finite-size
        LADDER:rung5:finite-executable
        LADDER:rung6:trace
        LADDER:rung7:frozen""",
    ),
    *_seed_group(
        PLAN,
        "_normalize_rho_multiplicity",
        CandidateFamily.COMPARE,
        "PLAN:multiplicity:index-and-gamma-domain",
    ),
    *_seed_group(
        PLAN,
        "RhoCertificate.__post_init__",
        CandidateFamily.COMPARE,
        """PLAN:certificate:rho-domain-and-coverage
        PLAN:certificate:error-budget-domain
        PLAN:certificate:optional-scale-domain
        PLAN:certificate:order-is-derived""",
    ),
    *_seed_group(
        PLAN,
        "certify_warmup_rho",
        CandidateFamily.COMPARE,
        """PLAN:warmup:rho-inputs-and-margin
        PLAN:warmup:tail-fraction
        PLAN:warmup:lambda-scale-inputs
        PLAN:warmup:x-norm-inputs
        PLAN:warmup:rho-roundoff-ceiling""",
    ),
    *_seed_group(
        PLAN,
        "audit_retained_rho",
        CandidateFamily.COMPARE,
        "PLAN:audit:retained-rho",
    ),
    *_seed_group(
        PLAN,
        "audit_retained_lambda_logdet",
        CandidateFamily.COMPARE,
        "PLAN:audit:retained-lambda-scale",
    ),
    *_seed_group(
        PLAN,
        "audit_retained_operator_norm",
        CandidateFamily.COMPARE,
        "PLAN:audit:retained-x-norm",
    ),
    *_seed_group(
        PLAN,
        "audit_retained_power_traces",
        CandidateFamily.COMPARE,
        "PLAN:audit:retained-trace-evidence",
    ),
    *_seed_group(
        PLAN,
        "_checked_x_operator_norm",
        CandidateFamily.FINITE_PREDICATE,
        "PLAN:measurement:x-norm-finite",
    ),
    *_seed_group(
        PLAN,
        "_checked_lambda_logdet_scale",
        CandidateFamily.FINITE_PREDICATE,
        "PLAN:measurement:lambda-logdet-finite",
    ),
    *_seed_group(
        PLAN,
        "_validate_plan_certificate",
        CandidateFamily.COMPARE,
        """PLAN:factory-certificate:order-and-rank
        PLAN:factory-certificate:lambda-scale
        PLAN:factory-certificate:x-norm
        PLAN:factory-certificate:strict-rho""",
    ),
    *_seed_group(
        PLAN,
        "_canonical_runtime_probes",
        CandidateFamily.FINITE_PREDICATE,
        "PLAN:canonical-probes:runtime-finite",
    ),
    *_seed_group(
        PLAN,
        "_outward_nonnegative",
        CandidateFamily.COMPARE,
        "PLAN:outward-arithmetic:positive-underflow",
    ),
    *_seed_group(
        PLAN,
        "_frozen_probe_energy_bounds",
        CandidateFamily.COMPARE,
        "PLAN:frozen:probe-energy-range",
    ),
    *_seed_group(
        PLAN,
        "_runtime_range_product",
        CandidateFamily.COMPARE,
        "PLAN:runtime-range:product",
    ),
    *_seed_group(
        PLAN,
        "_runtime_range_sum",
        CandidateFamily.COMPARE,
        "PLAN:runtime-range:sum",
    ),
    *_seed_group(
        PLAN,
        "_gamma_for_count",
        CandidateFamily.COMPARE,
        "PLAN:gamma:operation-count-domain",
    ),
    *_seed_group(
        PLAN,
        "_validate_frozen_runtime_range",
        CandidateFamily.COMPARE,
        "PLAN:frozen:x-bound-runtime-range",
    ),
    *_seed_group(
        PLAN,
        "_validate_frozen_runtime_range",
        CandidateFamily.RAISE,
        "PLAN:frozen:intermediate-runtime-range",
    ),
    *_seed_group(
        PLAN,
        "_validate_runtime_precision",
        CandidateFamily.COMPARE,
        """PLAN:runtime:sigma-finite-and-positive
        PLAN:runtime:expected-and-ulp-finite
        PLAN:runtime:base-scale-range
        PLAN:runtime:frozen-prerequisites-and-series
        PLAN:runtime:total-error-budget""",
    ),
    *_seed_group(
        PLAN,
        "_require_runtime_precision",
        CandidateFamily.COMPARE,
        "PLAN:runtime-call:scalar-and-dtype",
    ),
    *_seed_group(
        PLAN,
        "make_trace_log_plan",
        CandidateFamily.COMPARE,
        "PLAN:trace-factory:exact-evidence",
    ),
    *_seed_group(
        PLAN,
        "make_frozen_trace_log_plan",
        CandidateFamily.COMPARE,
        "PLAN:frozen-factory:probe-presence-width",
    ),
    *_seed_group(
        COUPLING,
        "_classify_correlation",
        CandidateFamily.FINITE_PREDICATE,
        """COUPLING:_classify_correlation:value-finite
        COUPLING:_classify_correlation:floor-finite""",
    ),
    *_seed_group(
        COUPLING,
        "_classify_correlation",
        CandidateFamily.COMPARE,
        """COUPLING:_classify_correlation:lower-noise-floor
        COUPLING:_classify_correlation:upper-noise-floor""",
    ),
    *_seed_group(
        COUPLING,
        "_condition_number",
        CandidateFamily.FINITE_PREDICATE,
        "COUPLING:_condition_number:finite-spectrum",
    ),
    *_seed_group(
        COUPLING,
        "_condition_number",
        CandidateFamily.COMPARE,
        "COUPLING:_condition_number:positive-spectrum",
    ),
    *_seed_group(
        COUPLING,
        "block_coupling",
        CandidateFamily.LINALG_PREMISE,
        """COUPLING:block_coupling:f-xx-spd
        COUPLING:block_coupling:f-tt-spd""",
    ),
    *_seed_group(
        MAP,
        "map_estimate",
        CandidateFamily.BITWISE_FINITE_CONJUNCTION,
        "MAP:map_estimate:finite-derivative-payload",
    ),
    *_seed_group(
        MAP,
        "map_estimate",
        CandidateFamily.COMPARE,
        "MAP:map_estimate:stationarity-floor",
    ),
    *_seed_group(
        MAP,
        "map_estimate",
        CandidateFamily.COMPARE,
        """MAP:map_estimate:relative-positive-curvature
        MAP:map_estimate:absolute-curvature""",
    ),
    *_seed_group(
        GRAPH,
        "_names",
        CandidateFamily.COMPARE,
        "GRAPH:_names:duplicate-multiplicity",
    ),
    *_seed_group(
        COSTS,
        "gap_is_contested",
        CandidateFamily.DECISION_PREDICATE,
        "COSTS:gap_is_contested:contested-bandwidth",
    ),
    *_seed_group(
        COSTS,
        "timing_noise_in_domain",
        CandidateFamily.DECISION_PREDICATE,
        "COSTS:timing_noise_in_domain:proper-fraction",
    ),
    *_seed_group(
        COSTS,
        "cg_tol_positive",
        CandidateFamily.DECISION_PREDICATE,
        "COSTS:cg_tol_positive:strictly-positive",
    ),
    *_seed_group(
        COLLAPSE,
        "pivots_are_finite",
        CandidateFamily.DECISION_PREDICATE,
        "COLLAPSE:pivots:finite",
    ),
    *_seed_group(
        COLLAPSE,
        "pivots_constrain_block",
        CandidateFamily.DECISION_PREDICATE,
        "COLLAPSE:pivots:relative-floor",
    ),
)


_GATE_SOURCE_LINKS: dict[str, tuple[tuple[str, str], ...]] = {
    "COUPLING:_classify_correlation:floor-finite": (
        (
            "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::decision_predicate::45b422c98efa4d2b::0",
            "not np.isfinite(floor)",
        ),
    ),
    "COUPLING:_classify_correlation:lower-noise-floor": (
        (
            "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::decision_predicate::bb065f25cc0fb571::0",
            "value <= floor",
        ),
    ),
    "COUPLING:_classify_correlation:upper-noise-floor": (
        (
            "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::decision_predicate::c96b8c0f89a8f758::0",
            "value >= 1.0 - floor",
        ),
    ),
    "COUPLING:_classify_correlation:value-finite": (
        (
            "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::decision_predicate::2776f1f61aa53ddd::0",
            "not np.isfinite(value)",
        ),
    ),
    "COUPLING:_condition_number:finite-spectrum": (
        (
            "src/bayesmith/diagnose/coupling.py::<module>._condition_number::decision_predicate::15ffb02e90908ffa::0",
            "not np.isfinite(smallest) or not np.isfinite(largest)",
        ),
    ),
    "COUPLING:_condition_number:positive-spectrum": (
        (
            "src/bayesmith/diagnose/coupling.py::<module>._condition_number::decision_predicate::1c27e6520e2bee10::0",
            "smallest <= 0.0",
        ),
    ),
    "COUPLING:block_coupling:f-xx-spd": (
        (
            "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::linalg_call_atom::d0ca43b45b317458::0",
            "np.linalg.cholesky(f_xx)",
        ),
    ),
    "COUPLING:block_coupling:f-tt-spd": (
        (
            "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::linalg_call_atom::1e90cfb77ca31791::0",
            "np.linalg.cholesky(f_tt)",
        ),
    ),
    "EAGER:LadderConfig:integer-threshold-domain": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::decision_predicate::08e52b1635749332::0",
            "any((value < 0 for value in integer_fields))",
        ),
    ),
    "EAGER:LadderConfig:low-rank-fraction-domain": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::decision_predicate::f29cb1f137f9131a::0",
            "not 0.0 <= self.low_rank_fraction <= 1.0",
        ),
    ),
    "EAGER:LadderConfig:structure-tolerance-domain": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::decision_predicate::50468fbb991e3009::0",
            "not np.isfinite(self.structure_rtol) or not np.isfinite(self.structure_atol) or self.structure_rtol < 0.0 or (self.structure_atol < 0.0)",
        ),
    ),
    "EAGER:LogDetProblem:lambda-spd": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::decision_predicate::2b61f5f86861b6c3::0",
            "not _is_positive_definite(lam)",
        ),
    ),
    "EAGER:array-normalization:shape-and-finiteness": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::0020fd8302214f01::0",
            "ndim is not None and array.ndim != ndim",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::4b9b4c4f077e83f4::0",
            "array.ndim not in (1, 2)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::decision_predicate::a5d3b66b1f22170f::0",
            "not np.all(np.isfinite(array))",
        ),
    ),
    "EAGER:dense-condition:strict-dtype-ceiling": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::decision_predicate::3e9d8ac524caed2b::0",
            "np.isfinite(condition) and condition < ceiling",
        ),
    ),
    "EAGER:factor-balance:exact-power-of-two-reversibility": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::decision_predicate::0a4170fb4dffee95::0",
            "np.all(np.isfinite(scaled_left)) and np.all(np.isfinite(scaled_right)) and np.array_equal(restored_left, left_column) and np.array_equal(restored_right, right_column)",
        ),
    ),
    "EAGER:factor-base:condition-ceiling": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::3e2925807fea5201::0",
            "np.isfinite(base_condition)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::8979ff3006924922::0",
            "base_condition < base_condition_ceiling",
        ),
    ),
    "EAGER:factor-base:error-budget": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::f5cf9fa84c4b9ba2::0",
            "np.isfinite(base_solve_eta) and 0.0 <= base_solve_eta < 1.0",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::0520942938172abf::0",
            "np.isfinite(base_log_error_bound)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::7eb5677a4b7f505c::0",
            "base_log_error_bound <= ceiling",
        ),
    ),
    "EAGER:factor-projection:error-budget": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::22296f4e1c2def63::0",
            "np.isfinite(eta) and eta < 1.0 and np.isfinite(log_error_bound) and (log_error_bound <= ceiling)",
        ),
    ),
    "EAGER:factor-projection:finite-qr-arithmetic": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::ec16f32088da528c::0",
            "not (np.all(np.isfinite(left_basis)) and np.all(np.isfinite(right_basis)) and np.all(np.isfinite(core)) and np.all(np.isfinite(projected)))",
        ),
    ),
    "EAGER:factor-projection:whitened-positive-spectrum": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::3f267babe0743274::0",
            "smallest_eigenvalue <= 0.0 or not np.isfinite(smallest_eigenvalue)",
        ),
    ),
    "EAGER:factor-reconstruction:layout-exactness": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._matching_factor_reconstruction::decision_predicate::b678026387cb59b1::0",
            "np.array_equal(canonical, value)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._matching_factor_reconstruction::decision_predicate::cba17196a4c6f789::0",
            "np.array_equal(reconstructed, value)",
        ),
    ),
    "EAGER:factor-reduced:acceptance-budget": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::14853fc7855c8120::0",
            "np.isfinite(total_log_error_bound) and total_log_error_bound <= ceiling",
        ),
    ),
    "EAGER:factor-reduced:diagonal-certificate": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::70ad0922d1407890::0",
            "reduced_sign > 0.0 and np.all(np.isfinite(relative_diagonal_error)) and np.all(relative_diagonal_error < 1.0)",
        ),
    ),
    "EAGER:factor-reduced:qr-certificate": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::a1dcce8aab2a2ac4::0",
            "reduced_sign > 0.0 and np.isfinite(reduced_eta) and (0.0 <= reduced_eta < 1.0) and np.isfinite(orthogonality_eta) and (0.0 <= orthogonality_eta < 1.0) and np.all(reduced_r_diagonal != 0.0)",
        ),
    ),
    "EAGER:finite:newton-stability-rho": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_stability::numerical_premise_call::9f18757e939336dc::0",
            "spectral_radius(lambda_matrix, perturbation)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_stability::compare::87baab435b8b4904::0",
            "rho <= 1.0",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._newton_logdet::decision_predicate::11d4f5fe8aeca2b4::0",
            "not stable",
        ),
    ),
    "EAGER:frozen-probes:identity-width-order": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::decision_predicate::a463584319db77e0::0",
            "type(probes) is not FrozenProbes",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::decision_predicate::5fe9ad3578e110cc::0",
            "vectors.shape[1] != _n(lam)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::decision_predicate::7f499b566536192a::0",
            "order < 0",
        ),
    ),
    "EAGER:lambda-logdet:subnormal-rescale": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::decision_predicate::411d89f67a1cb014::0",
            "maximum < float(np.finfo(lam.dtype).tiny)",
        ),
    ),
    "EAGER:spectral-radius:finite-measurement": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::decision_predicate::454612947ebca50f::0",
            "not np.all(np.isfinite(x))",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::decision_predicate::599684cb5a2bcc16::0",
            "not np.all(np.isfinite(eigenvalues))",
        ),
    ),
    "EAGER:state-space:block-chain-exactness": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::decision_predicate::5323d72464a08437::0",
            "block_size < 1 or n % block_size",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::decision_predicate::5bc915df4c4a49d1::0",
            "np.any(piece != 0.0)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::05d181e4f7161009::0",
            "not _is_block_chain(dense, block_size, rtol=rtol, atol=atol)",
        ),
    ),
    "EAGER:state-space:payload-domain": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::da99982e585a6a0c::0",
            "not _is_symmetric(dense, rtol=rtol, atol=atol)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::fd1f83756118c8ce::0",
            "not _is_positive_definite(dense)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::30d95146f3bde552::0",
            "_require_resolved_dense_condition(dense, 'block-LDL')",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::decision_predicate::f517e52fd3adde59::0",
            "not math.isfinite(total)",
        ),
    ),
    "EAGER:structured:exact-shape-and-spectrum": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_diagonal::predicate_call_atom::1df7bc50a873a119::0",
            "np.array_equal(matrix, np.diag(np.diag(matrix)))",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_circulant::predicate_call_atom::eeb4af8582c3e1f5::0",
            "np.array_equal(matrix, expected)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_toeplitz::compare::164edfdc38712d7d::0",
            "diagonal == diagonal[0]",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._circulant_eigenvalues::compare::4ac2ef8d233a8860::0",
            "eigenvalues <= 0.0",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::decision_predicate::ecd14e44b0d8b37f::0",
            "reconstructed.shape != dense.shape or not np.array_equal(reconstructed, dense)",
        ),
    ),
    "EAGER:symmetry:tolerant-representative": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::8b0c93739c0c0f85::0",
            "bool(np.all(value > 0.0))",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::compare::ede5c27a1b4dedf3::0",
            "value > 0.0",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_symmetric::compare::c3355ac3d163349a::0",
            "difference <= tolerance",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::f0ef79aec301dc19::0",
            "not _is_symmetric(value, rtol=rtol, atol=atol)",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::decision_predicate::469f1eb6fda2ed51::0",
            "np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues > 0.0)",
        ),
    ),
    "EAGER:trace:actual-rho-strict": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::decision_predicate::19690c86ac3c4081::0",
            "not actual_rho < 1.0",
        ),
    ),
    "EAGER:trace:certificate-domain": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::decision_predicate::84b35b0be0b6b1d7::0",
            "not 0.0 <= certificate < 1.0",
        ),
    ),
    "EAGER:trace:certificate-upper-bound": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::decision_predicate::0e23a71d8847891c::0",
            "actual_rho > certificate",
        ),
    ),
    "EAGER:trace:exact-power-trace-evidence": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::decision_predicate::6bcb5682fc2262b5::0",
            "order < 0 or len(traces) < order",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::decision_predicate::24c12791d2599c57::0",
            "np.all(np.isfinite(supplied)) and np.array_equal(supplied, derived)",
        ),
    ),
    "EAGER:trace:tail-domain-and-order": (
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::decision_predicate::96a33b68a27091f1::0",
            "not 0.0 <= rho < 1.0",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::decision_predicate::7f499b566536192a::0",
            "order < 0",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.whole_trace_log_tail_bound::decision_predicate::cb1557568a2a2887::0",
            "multiplicity < 1",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::decision_predicate::db7fa083851799e3::0",
            "not np.isfinite(tolerance) or tolerance <= 0.0",
        ),
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::decision_predicate::be8bebac0fc5f981::0",
            "whole_trace_log_tail_bound(rho, order, multiplicity) > tolerance",
        ),
    ),
    "GRAPH:_names:duplicate-multiplicity": (
        (
            "src/bayesmith/graph/reduction.py::<module>._names::compare::762259ce967b894b::0",
            "names.count(name) > 1",
        ),
    ),
    "COSTS:gap_is_contested:contested-bandwidth": (
        (
            "src/bayesmith/dispatch/costs.py::<module>.gap_is_contested::decision_predicate::2ef630995dace887::0",
            "gap < CONTESTED_BANDWIDTH",
        ),
    ),
    "COSTS:timing_noise_in_domain:proper-fraction": (
        (
            "src/bayesmith/dispatch/costs.py::<module>.timing_noise_in_domain::decision_predicate::3ae6110a138fcde9::0",
            "tol < 1.0",
        ),
    ),
    "COSTS:cg_tol_positive:strictly-positive": (
        (
            "src/bayesmith/dispatch/costs.py::<module>.cg_tol_positive::decision_predicate::135e313228ca2cb8::0",
            "tol > 0.0",
        ),
    ),
    "COLLAPSE:pivots:finite": (
        (
            "src/bayesmith/dispatch/collapse.py::<module>.pivots_are_finite::decision_predicate::10b0270e03e50a15::0",
            "jnp.all(jnp.isfinite(pivots))",
        ),
    ),
    "COLLAPSE:pivots:relative-floor": (
        (
            "src/bayesmith/dispatch/collapse.py::<module>.pivots_constrain_block::compare::fa67000d1d7f01d8::0",
            "pivots[:n_block] > floor",
        ),
    ),
    "LADDER:determinant-lemma:payload": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::8f473f76dc3f6936::0",
            "problem.low_rank_factors is not None and rank_evidence_valid and sigma_formation_valid and sigma_exactly_symmetric",
        ),
    ),
    "LADDER:finite:payload-rho": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::d3068b908ee92dee::0",
            "spectral_radius(lam, finite_payload_perturbation)",
        ),
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::057b7b726f746fbf::0",
            "finite_payload_rho_measurement_valid and finite_payload_rho <= 1.0",
        ),
    ),
    "LADDER:rank:evidence": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::32b77c3c606be73a::0",
            "_algebraic_rank_bound(perturb)",
        ),
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::fe4f7ab1beecdb99::0",
            "_factor_projection_certificate(perturb, problem.low_rank_factors, lam)",
        ),
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::c515480aa003dc6d::0",
            "rank_evidence_valid",
        ),
    ),
    "LADDER:rho:measurement": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::a443ff27e05df6b5::0",
            "spectral_radius(lam, perturb)",
        ),
    ),
    "LADDER:rung0:base": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7fbc415aee154691::0",
            "sigma_formation_valid and bool(np.array_equal(sigma, lam)) and dense_arithmetic_resolved",
        ),
    ),
    "LADDER:rung1:low-rank-size": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::efa1c96a45b8a99d::0",
            "rank_evidence_valid and (compact_diagonal_payload or determinant_lemma_payload) and sigma_spd and (rank <= config.low_rank_max) and (rank <= config.low_rank_fraction * n)",
        ),
    ),
    "LADDER:rung2:chain": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7d56f970127e454a::0",
            "chain_structure and sigma_spd and condition_resolved",
        ),
    ),
    "LADDER:rung3:structured": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::gate_qualifier::2ea8744b9588c9b7::0",
            "structured",
        ),
    ),
    "LADDER:rung4:dense": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::3ca3ab1fd1c3fc27::0",
            "n <= config.dense_max_n and condition_resolved and sigma_spd",
        ),
    ),
    "LADDER:rung5:finite-executable": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::ed70d0b90f2a707b::0",
            "finite_size_qualified and finite_payload_stable and sigma_spd and (determinant_lemma_payload or dense_arithmetic_resolved)",
        ),
    ),
    "LADDER:rung5:finite-size": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::6d8dda4ea60f3e75::0",
            "n <= config.finite_max_n or ((compact_diagonal_payload or determinant_lemma_payload) and rank <= config.finite_max_rank)",
        ),
    ),
    "LADDER:rung6:trace": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7f581995d7238d06::0",
            "sigma_formation_valid and traces_verified and measured_rho_converges and rho_covers_input and (0.0 <= rho < 1.0)",
        ),
    ),
    "LADDER:rung7:frozen": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::1f75b0ed3ab0d827::0",
            "sigma_formation_valid and frozen_width_valid and (problem.trace_order is not None) and (problem.trace_order >= 0) and measured_rho_converges and rho_covers_input and (0.0 <= rho < 1.0)",
        ),
    ),
    "LADDER:sigma:finite-two-sum": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::4025d304e4487b7c::0",
            "_two_sum_error(lam, perturb)",
        ),
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::ad0571b6919536d8::0",
            "not sigma_formation_valid",
        ),
    ),
    "LADDER:sigma:payload-symmetry": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::numerical_premise_call::fae93f08468cafbe::0",
            "_sigma_payload(sigma, config)",
        ),
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::decision_predicate::aa5e8dd848a0018b::0",
            "sigma.ndim == 1 or np.array_equal(sigma, sigma.T) or (not _is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol))",
        ),
    ),
    "LADDER:sigma:symmetry-spd-condition": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::e990d931da5e772e::0",
            "sigma.ndim == 1 or _is_symmetric(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
        ),
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::7ce742d10265b420::0",
            "sigma_symmetric and _is_positive_definite(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
        ),
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::d69a1ae0cf7be478::0",
            "not condition_resolved",
        ),
    ),
    "LADDER:structure:circulant-tolerance-spectrum": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::7ead0b7ae54d3a4a::0",
            "_is_circulant(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
        ),
    ),
    "LADDER:structure:compact-diagonal-positive": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::89bf95eccb3933be::0",
            "bool(np.all(sigma > 0.0))",
        ),
    ),
    "LADDER:structure:diagonal-tolerance": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::e4f1323628f87cb8::0",
            "kind is None and _is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
        ),
    ),
    "LADDER:structure:kronecker-evidence": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::d611227f5fbefbc9::0",
            "reconstructed.shape == sigma.shape and np.array_equal(reconstructed, sigma)",
        ),
    ),
    "LADDER:structure:toeplitz-tolerance": (
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::6a0aad293d04c15d::0",
            "_is_toeplitz(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
        ),
    ),
    "MAP:map_estimate:absolute-curvature": (
        (
            "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::4811937db4eb0e1e::0",
            "largest > absolute_curvature_floor",
        ),
    ),
    "MAP:map_estimate:finite-derivative-payload": (
        (
            "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::087e9fc8a8556c2f::0",
            "bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(hessian)))",
        ),
        (
            "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::8f9dae6ee5f5a679::0",
            "jnp.isfinite(value)",
        ),
        (
            "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::c602a9209c1f710a::0",
            "jnp.isfinite(gradient)",
        ),
        (
            "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::5ae94c7be7ed7e6a::0",
            "jnp.isfinite(hessian)",
        ),
    ),
    "MAP:map_estimate:relative-positive-curvature": (
        (
            "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::d64931ad93baeec2::0",
            "smallest > curvature_floor",
        ),
    ),
    "MAP:map_estimate:stationarity-floor": (
        (
            "src/bayesmith/diagnose/map.py::<module>.map_estimate::decision_predicate::ed87ffcc51d9fe06::0",
            "gradient_norm > gradient_floor",
        ),
    ),
    "PLAN:audit:retained-lambda-scale": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_lambda_logdet::compare::07d13ba7ca0d0c09::0",
            "value > certificate.max_abs_lambda_logdet",
        ),
    ),
    "PLAN:audit:retained-rho": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_rho::compare::f7bd0534d701df1c::0",
            "value > certificate.certified_rho",
        ),
    ),
    "PLAN:audit:retained-trace-evidence": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::decision_predicate::34984b001958dd9c::0",
            "problem.trace_order != certificate.order or _retained_rank_exceeds_certificate(problem, certificate) or problem.exact_power_traces is None or (not _checked_power_traces_match(problem.lambda_matrix, problem.perturbation, problem.exact_power_traces, certificate.order))",
        ),
    ),
    "PLAN:audit:retained-x-norm": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_operator_norm::compare::fd7fdbc56498b458::0",
            "value > certificate.max_x_operator_norm",
        ),
    ),
    "PLAN:canonical-probes:runtime-finite": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::decision_predicate::87d9424687bc11bb::0",
            "not np.all(np.isfinite(values))",
        ),
    ),
    "PLAN:certificate:error-budget-domain": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::6993e0eaa405cc88::0",
            "self.margin < 0.0 or self.tolerance <= 0.0 or (not 0.0 < self.tail_tolerance < self.tolerance)",
        ),
    ),
    "PLAN:certificate:optional-scale-domain": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::11003df0a5e25fa8::0",
            "self.max_abs_lambda_logdet is not None and (not np.isfinite(self.max_abs_lambda_logdet) or self.max_abs_lambda_logdet < 0.0)",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::a0a6c3092291c7f1::0",
            "self.max_x_operator_norm is not None and (not np.isfinite(self.max_x_operator_norm) or self.max_x_operator_norm < 0.0)",
        ),
    ),
    "PLAN:certificate:order-is-derived": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::5278515c7ba5de86::0",
            "self.order != expected_order",
        ),
    ),
    "PLAN:certificate:rho-domain-and-coverage": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::ad10c538bb5d448c::0",
            "not 0.0 <= self.measured_max < 1.0",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::26baf29e9d74ae01::0",
            "not 0.0 <= self.certified_rho < 1.0",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::1c74be788b92b3e1::0",
            "self.certified_rho < self.measured_max",
        ),
    ),
    "PLAN:factory-certificate:lambda-scale": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::1943e05d2c7883c5::0",
            "actual_base_scale > certificate.max_abs_lambda_logdet",
        ),
    ),
    "PLAN:factory-certificate:order-and-rank": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::4ae94ae7e3ef3f98::0",
            "problem.trace_order != certificate.order",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::ffbfe426624c1b2c::0",
            "certificate.multiplicity < required_multiplicity",
        ),
    ),
    "PLAN:factory-certificate:strict-rho": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::b0e44e8a17533440::0",
            "_validate_strict_rho(problem.lambda_matrix, problem.perturbation, certificate.certified_rho)",
        ),
    ),
    "PLAN:factory-certificate:x-norm": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::compare::0cff16fe3f2e1490::0",
            "actual_norm > certificate.max_x_operator_norm",
        ),
    ),
    "PLAN:frozen-factory:probe-presence-width": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::decision_predicate::655cc937a61c4cc6::0",
            "problem.frozen_probes is None",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::compare::c9ac3a5056f615b8::0",
            "problem.frozen_probes.values.shape[1] != _n(problem.lambda_matrix)",
        ),
    ),
    "PLAN:frozen:intermediate-runtime-range": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::numerical_premise_call::6d9734b76eb352db::0",
            "_runtime_range_product(correction_bound, addition_factor, maximum, runtime_dtype, 'the frozen correction accumulation')",
        ),
    ),
    "PLAN:frozen:probe-energy-range": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::decision_predicate::d1640f41a06efe07::0",
            "not np.isfinite(total_energy) or total_energy > maximum",
        ),
    ),
    "PLAN:frozen:x-bound-runtime-range": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::compare::c10e15779304e86d::0",
            "x_bound > maximum",
        ),
    ),
    "PLAN:gamma:operation-count-domain": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._gamma_for_count::compare::8da450fabbb947ef::0",
            "product >= 1.0",
        ),
    ),
    "PLAN:measurement:lambda-logdet-finite": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::decision_predicate::ae1c882bb97c3ce8::0",
            "not np.isfinite(actual_scale)",
        ),
    ),
    "PLAN:measurement:x-norm-finite": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::decision_predicate::b0b3d49680c202d3::0",
            "not np.all(np.isfinite(x)) or not np.isfinite(actual_norm)",
        ),
    ),
    "PLAN:multiplicity:index-and-gamma-domain": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::decision_predicate::b26d8d6f4a246b6c::0",
            "isinstance(value, (bool, np.bool_))",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::decision_predicate::cb1557568a2a2887::0",
            "multiplicity < 1",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::numerical_premise_call::b987512875e7714f::0",
            "operator.index(value)",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::compare::c19d43bd613dda15::0",
            "multiplicity >= _RHO_MULTIPLICITY_LIMIT",
        ),
    ),
    "PLAN:outward-arithmetic:positive-underflow": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::decision_predicate::21d0c003ac8233e2::0",
            "value == 0.0 or not np.isfinite(value)",
        ),
    ),
    "PLAN:runtime-call:scalar-and-dtype": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::decision_predicate::f3da34068224aef0::0",
            "expected.itemsize > 4 and (not jax.config.x64_enabled) or any((dtype.kind != 'f' or dtype.itemsize < expected.itemsize for dtype in actual))",
        ),
    ),
    "PLAN:runtime-range:product": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::decision_predicate::958e52d096caca96::0",
            "not np.isfinite(left) or not np.isfinite(right) or left > maximum / right",
        ),
    ),
    "PLAN:runtime-range:sum": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::decision_predicate::3f0c849b23f9cef1::0",
            "not np.isfinite(left) or not np.isfinite(right) or right > maximum or (left > maximum - right)",
        ),
    ),
    "PLAN:runtime:base-scale-range": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::2e3fba3ea6b5f4f0::0",
            "base_scale > maximum_runtime_value",
        ),
    ),
    "PLAN:runtime:expected-and-ulp-finite": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::d8dc3ffb0ede7b65::0",
            "not np.isfinite(expected)",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::ce4340efc14e88e2::0",
            "not np.isfinite(rounded) or not np.isfinite(ulp)",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::4e9bf2538eccc387::0",
            "ulp > certificate.tolerance",
        ),
    ),
    "PLAN:runtime:frozen-prerequisites-and-series": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::17d15162104ee643::0",
            "not np.isfinite(series_scale)",
        ),
    ),
    "PLAN:runtime:sigma-finite-and-positive": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::8fe5d7e22a754fc2::0",
            "np.any(sigma <= 0.0)",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::18513ce0c5e61279::0",
            "sign <= 0.0",
        ),
    ),
    "PLAN:runtime:total-error-budget": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::b7c85688cf179fbc::0",
            "total_error_bound > certificate.tolerance",
        ),
    ),
    "PLAN:trace-factory:exact-evidence": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.make_trace_log_plan::decision_predicate::a05753de1a88953b::0",
            "problem.exact_power_traces is None or not _checked_power_traces_match(problem.lambda_matrix, problem.perturbation, problem.exact_power_traces, certificate.order)",
        ),
    ),
    "PLAN:warmup:lambda-scale-inputs": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::d52eae775e943fd0::0",
            "not bases or any((not np.isfinite(value) for value in bases))",
        ),
    ),
    "PLAN:warmup:rho-inputs-and-margin": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::57c724e170630987::0",
            "not values or any((not np.isfinite(value) or value < 0.0 for value in values))",
        ),
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::83c56a26785d2596::0",
            "not np.isfinite(margin) or margin < 0.0",
        ),
    ),
    "PLAN:warmup:rho-roundoff-ceiling": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::cd3f603b2fb8ab4d::0",
            "not certified < 1.0",
        ),
    ),
    "PLAN:warmup:tail-fraction": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::df7a4a82222ab703::0",
            "not np.isfinite(tail_fraction) or not 0.0 < tail_fraction < 1.0",
        ),
    ),
    "PLAN:warmup:x-norm-inputs": (
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::793fd4e8d52e933e::0",
            "not norms or any((not np.isfinite(value) or value < 0.0 for value in norms))",
        ),
    ),
}
_DECLARED_SOURCE_ANCHORS: dict[str, tuple[SourceAnchor, ...]] = {
    "COUPLING:_classify_correlation:floor-finite": (
        SourceAnchor(
            "src/bayesmith/diagnose/coupling.py",
            "<module>._classify_correlation",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COUPLING:_classify_correlation:lower-noise-floor": (
        SourceAnchor(
            "src/bayesmith/diagnose/coupling.py",
            "<module>._classify_correlation",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COUPLING:_classify_correlation:upper-noise-floor": (
        SourceAnchor(
            "src/bayesmith/diagnose/coupling.py",
            "<module>._classify_correlation",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COUPLING:_classify_correlation:value-finite": (
        SourceAnchor(
            "src/bayesmith/diagnose/coupling.py",
            "<module>._classify_correlation",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COUPLING:_condition_number:finite-spectrum": (
        SourceAnchor(
            "src/bayesmith/diagnose/coupling.py",
            "<module>._condition_number",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COUPLING:_condition_number:positive-spectrum": (
        SourceAnchor(
            "src/bayesmith/diagnose/coupling.py",
            "<module>._condition_number",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COUPLING:block_coupling:f-xx-spd": (
        SourceAnchor(
            "src/bayesmith/diagnose/coupling.py",
            "<module>.block_coupling",
            CandidateFamily.LINALG_ATOM,
        ),
    ),
    "COUPLING:block_coupling:f-tt-spd": (
        SourceAnchor(
            "src/bayesmith/diagnose/coupling.py",
            "<module>.block_coupling",
            CandidateFamily.LINALG_ATOM,
        ),
    ),
    "EAGER:LadderConfig:integer-threshold-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.LadderConfig.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:LadderConfig:low-rank-fraction-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.LadderConfig.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:LadderConfig:structure-tolerance-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.LadderConfig.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:LogDetProblem:lambda-spd": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.LogDetProblem.__init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:array-normalization:shape-and-finiteness": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._read_only_array",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._read_only_array",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._read_only_array",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:dense-condition:strict-dtype-ceiling": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._condition_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:factor-balance:exact-power-of-two-reversibility": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._balanced_factor_columns",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:factor-base:condition-ceiling": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.BOOLEAN_ATOM,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.BOOLEAN_ATOM,
        ),
    ),
    "EAGER:factor-base:error-budget": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.BOOLEAN_ATOM,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.BOOLEAN_ATOM,
        ),
    ),
    "EAGER:factor-projection:error-budget": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:factor-projection:finite-qr-arithmetic": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:factor-projection:whitened-positive-spectrum": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:factor-reconstruction:layout-exactness": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._matching_factor_reconstruction",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._matching_factor_reconstruction",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:factor-reduced:acceptance-budget": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:factor-reduced:diagonal-certificate": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:factor-reduced:qr-certificate": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._factor_projection_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:finite:newton-stability-rho": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._newton_stability",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._newton_stability",
            CandidateFamily.COMPARE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._newton_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:frozen-probes:identity-width-order": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.frozen_hutchinson_trace_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.frozen_hutchinson_trace_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.frozen_hutchinson_trace_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:lambda-logdet:subnormal-rescale": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.lambda_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:spectral-radius:finite-measurement": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.spectral_radius",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.spectral_radius",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:state-space:block-chain-exactness": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_block_chain",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_block_chain",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.state_space_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:state-space:payload-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.state_space_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.state_space_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.state_space_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.state_space_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:structured:exact-shape-and-spectrum": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_diagonal",
            CandidateFamily.PREDICATE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_circulant",
            CandidateFamily.PREDICATE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_toeplitz",
            CandidateFamily.COMPARE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._circulant_eigenvalues",
            CandidateFamily.COMPARE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.structured_logdet",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:symmetry:tolerant-representative": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_positive_definite",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_positive_definite",
            CandidateFamily.COMPARE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_symmetric",
            CandidateFamily.COMPARE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_positive_definite",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._is_positive_definite",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:trace:actual-rho-strict": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._validate_strict_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:trace:certificate-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._validate_strict_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:trace:certificate-upper-bound": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._validate_strict_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:trace:exact-power-trace-evidence": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._power_traces_match",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>._power_traces_match",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "EAGER:trace:tail-domain-and-order": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.trace_log_tail_bound",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.trace_log_tail_bound",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.whole_trace_log_tail_bound",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.choose_trace_order",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_eager.py",
            "<module>.choose_trace_order",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "GRAPH:_names:duplicate-multiplicity": (
        SourceAnchor(
            "src/bayesmith/graph/reduction.py",
            "<module>._names",
            CandidateFamily.COMPARE,
        ),
    ),
    "COSTS:gap_is_contested:contested-bandwidth": (
        SourceAnchor(
            "src/bayesmith/dispatch/costs.py",
            "<module>.gap_is_contested",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COSTS:timing_noise_in_domain:proper-fraction": (
        SourceAnchor(
            "src/bayesmith/dispatch/costs.py",
            "<module>.timing_noise_in_domain",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COSTS:cg_tol_positive:strictly-positive": (
        SourceAnchor(
            "src/bayesmith/dispatch/costs.py",
            "<module>.cg_tol_positive",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COLLAPSE:pivots:finite": (
        SourceAnchor(
            "src/bayesmith/dispatch/collapse.py",
            "<module>.pivots_are_finite",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "COLLAPSE:pivots:relative-floor": (
        SourceAnchor(
            "src/bayesmith/dispatch/collapse.py",
            "<module>.pivots_constrain_block",
            CandidateFamily.COMPARE,
        ),
    ),
    "LADDER:determinant-lemma:payload": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:finite:payload-rho": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rank:evidence": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rho:measurement": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
    ),
    "LADDER:rung0:base": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rung1:low-rank-size": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rung2:chain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rung3:structured": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.GATE_QUALIFIER,
        ),
    ),
    "LADDER:rung4:dense": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rung5:finite-executable": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rung5:finite-size": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rung6:trace": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:rung7:frozen": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:sigma:finite-two-sum": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:sigma:payload-symmetry": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>._sigma_payload",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:sigma:symmetry-spd-condition": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>.check_logdet_premises",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:structure:circulant-tolerance-spectrum": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>._structure_request",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:structure:compact-diagonal-positive": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>._structure_request",
            CandidateFamily.PREDICATE_CALL,
        ),
    ),
    "LADDER:structure:diagonal-tolerance": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>._structure_request",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:structure:kronecker-evidence": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>._structure_request",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "LADDER:structure:toeplitz-tolerance": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_ladder.py",
            "<module>._structure_request",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "MAP:map_estimate:absolute-curvature": (
        SourceAnchor(
            "src/bayesmith/diagnose/map.py",
            "<module>.map_estimate",
            CandidateFamily.COMPARE,
        ),
    ),
    "MAP:map_estimate:finite-derivative-payload": (
        SourceAnchor(
            "src/bayesmith/diagnose/map.py",
            "<module>.map_estimate",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/diagnose/map.py",
            "<module>.map_estimate",
            CandidateFamily.FINITE_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/diagnose/map.py",
            "<module>.map_estimate",
            CandidateFamily.FINITE_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/diagnose/map.py",
            "<module>.map_estimate",
            CandidateFamily.FINITE_PREDICATE,
        ),
    ),
    "MAP:map_estimate:relative-positive-curvature": (
        SourceAnchor(
            "src/bayesmith/diagnose/map.py",
            "<module>.map_estimate",
            CandidateFamily.COMPARE,
        ),
    ),
    "MAP:map_estimate:stationarity-floor": (
        SourceAnchor(
            "src/bayesmith/diagnose/map.py",
            "<module>.map_estimate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:audit:retained-lambda-scale": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.audit_retained_lambda_logdet",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:audit:retained-rho": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.audit_retained_rho",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:audit:retained-trace-evidence": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.audit_retained_power_traces",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:audit:retained-x-norm": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.audit_retained_operator_norm",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:canonical-probes:runtime-finite": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._canonical_runtime_probes",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:certificate:error-budget-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.RhoCertificate.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:certificate:optional-scale-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.RhoCertificate.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.RhoCertificate.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:certificate:order-is-derived": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.RhoCertificate.__post_init__",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:certificate:rho-domain-and-coverage": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.RhoCertificate.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.RhoCertificate.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.RhoCertificate.__post_init__",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:factory-certificate:lambda-scale": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_plan_certificate",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:factory-certificate:order-and-rank": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_plan_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_plan_certificate",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:factory-certificate:strict-rho": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_plan_certificate",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:factory-certificate:x-norm": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_plan_certificate",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:frozen-factory:probe-presence-width": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.make_frozen_trace_log_plan",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.make_frozen_trace_log_plan",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:frozen:intermediate-runtime-range": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_frozen_runtime_range",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
    ),
    "PLAN:frozen:probe-energy-range": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._frozen_probe_energy_bounds",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:frozen:x-bound-runtime-range": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_frozen_runtime_range",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:gamma:operation-count-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._gamma_for_count",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:measurement:lambda-logdet-finite": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._checked_lambda_logdet_scale",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:measurement:x-norm-finite": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._checked_x_operator_norm",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:multiplicity:index-and-gamma-domain": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._normalize_rho_multiplicity",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._normalize_rho_multiplicity",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._normalize_rho_multiplicity",
            CandidateFamily.NUMERICAL_PREMISE_CALL,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._normalize_rho_multiplicity",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:outward-arithmetic:positive-underflow": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._outward_nonnegative",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:runtime-call:scalar-and-dtype": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._require_runtime_precision",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:runtime-range:product": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._runtime_range_product",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:runtime-range:sum": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._runtime_range_sum",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:runtime:base-scale-range": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_runtime_precision",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:runtime:expected-and-ulp-finite": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_runtime_precision",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_runtime_precision",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_runtime_precision",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:runtime:frozen-prerequisites-and-series": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_runtime_precision",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:runtime:sigma-finite-and-positive": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_runtime_precision",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_runtime_precision",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:runtime:total-error-budget": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>._validate_runtime_precision",
            CandidateFamily.COMPARE,
        ),
    ),
    "PLAN:trace-factory:exact-evidence": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.make_trace_log_plan",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:warmup:lambda-scale-inputs": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.certify_warmup_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:warmup:rho-inputs-and-margin": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.certify_warmup_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.certify_warmup_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:warmup:rho-roundoff-ceiling": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.certify_warmup_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:warmup:tail-fraction": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.certify_warmup_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
    "PLAN:warmup:x-norm-inputs": (
        SourceAnchor(
            "src/bayesmith/marginal/_logdet_plan.py",
            "<module>.certify_warmup_rho",
            CandidateFamily.DECISION_PREDICATE,
        ),
    ),
}
_DECLARED_SOURCE_CLASSIFICATIONS: dict[str, tuple[CandidateClassification, ...]] = {
    "COUPLING:_classify_correlation:floor-finite": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "COUPLING:_classify_correlation:lower-noise-floor": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "COUPLING:_classify_correlation:upper-noise-floor": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "COUPLING:_classify_correlation:value-finite": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "COUPLING:_condition_number:finite-spectrum": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "COUPLING:_condition_number:positive-spectrum": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "COUPLING:block_coupling:f-xx-spd": (
        CandidateClassification.NUMERICAL_SAFETY,
    ),
    "COUPLING:block_coupling:f-tt-spd": (
        CandidateClassification.NUMERICAL_SAFETY,
    ),
    "EAGER:LadderConfig:integer-threshold-domain": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:LadderConfig:low-rank-fraction-domain": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:LadderConfig:structure-tolerance-domain": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:LogDetProblem:lambda-spd": (CandidateClassification.NUMERICAL_GATE,),
    "EAGER:array-normalization:shape-and-finiteness": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:dense-condition:strict-dtype-ceiling": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:factor-balance:exact-power-of-two-reversibility": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:factor-base:condition-ceiling": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:factor-base:error-budget": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:factor-projection:error-budget": (CandidateClassification.NUMERICAL_GATE,),
    "EAGER:factor-projection:finite-qr-arithmetic": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:factor-projection:whitened-positive-spectrum": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:factor-reconstruction:layout-exactness": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:factor-reduced:acceptance-budget": (CandidateClassification.NUMERICAL_GATE,),
    "EAGER:factor-reduced:diagonal-certificate": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:factor-reduced:qr-certificate": (CandidateClassification.NUMERICAL_GATE,),
    "EAGER:finite:newton-stability-rho": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:frozen-probes:identity-width-order": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:lambda-logdet:subnormal-rescale": (CandidateClassification.NUMERICAL_GATE,),
    "EAGER:spectral-radius:finite-measurement": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:state-space:block-chain-exactness": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:state-space:payload-domain": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:structured:exact-shape-and-spectrum": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:symmetry:tolerant-representative": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:trace:actual-rho-strict": (CandidateClassification.NUMERICAL_GATE,),
    "EAGER:trace:certificate-domain": (CandidateClassification.NUMERICAL_GATE,),
    "EAGER:trace:certificate-upper-bound": (CandidateClassification.NUMERICAL_GATE,),
    "EAGER:trace:exact-power-trace-evidence": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "EAGER:trace:tail-domain-and-order": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "GRAPH:_names:duplicate-multiplicity": (CandidateClassification.NUMERICAL_GATE,),
    "COSTS:gap_is_contested:contested-bandwidth": (CandidateClassification.NUMERICAL_GATE,),
    "COSTS:timing_noise_in_domain:proper-fraction": (CandidateClassification.NUMERICAL_GATE,),
    "COSTS:cg_tol_positive:strictly-positive": (CandidateClassification.NUMERICAL_GATE,),
    "COLLAPSE:pivots:finite": (CandidateClassification.NUMERICAL_GATE,),
    "COLLAPSE:pivots:relative-floor": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:determinant-lemma:payload": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:finite:payload-rho": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "LADDER:rank:evidence": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "LADDER:rho:measurement": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung0:base": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung1:low-rank-size": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung2:chain": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung3:structured": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung4:dense": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung5:finite-executable": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung5:finite-size": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung6:trace": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:rung7:frozen": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:sigma:finite-two-sum": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "LADDER:sigma:payload-symmetry": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "LADDER:sigma:symmetry-spd-condition": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "LADDER:structure:circulant-tolerance-spectrum": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "LADDER:structure:compact-diagonal-positive": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "LADDER:structure:diagonal-tolerance": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:structure:kronecker-evidence": (CandidateClassification.NUMERICAL_GATE,),
    "LADDER:structure:toeplitz-tolerance": (CandidateClassification.NUMERICAL_GATE,),
    "MAP:map_estimate:absolute-curvature": (CandidateClassification.NUMERICAL_GATE,),
    "MAP:map_estimate:finite-derivative-payload": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "MAP:map_estimate:relative-positive-curvature": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "MAP:map_estimate:stationarity-floor": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:audit:retained-lambda-scale": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:audit:retained-rho": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:audit:retained-trace-evidence": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:audit:retained-x-norm": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:canonical-probes:runtime-finite": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:certificate:error-budget-domain": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:certificate:optional-scale-domain": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:certificate:order-is-derived": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:certificate:rho-domain-and-coverage": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:factory-certificate:lambda-scale": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:factory-certificate:order-and-rank": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:factory-certificate:strict-rho": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:factory-certificate:x-norm": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:frozen-factory:probe-presence-width": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:frozen:intermediate-runtime-range": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:frozen:probe-energy-range": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:frozen:x-bound-runtime-range": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:gamma:operation-count-domain": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:measurement:lambda-logdet-finite": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:measurement:x-norm-finite": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:multiplicity:index-and-gamma-domain": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:outward-arithmetic:positive-underflow": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:runtime-call:scalar-and-dtype": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:runtime-range:product": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:runtime-range:sum": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:runtime:base-scale-range": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:runtime:expected-and-ulp-finite": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:runtime:frozen-prerequisites-and-series": (
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:runtime:sigma-finite-and-positive": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:runtime:total-error-budget": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:trace-factory:exact-evidence": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:warmup:lambda-scale-inputs": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:warmup:rho-inputs-and-margin": (
        CandidateClassification.NUMERICAL_GATE,
        CandidateClassification.NUMERICAL_GATE,
    ),
    "PLAN:warmup:rho-roundoff-ceiling": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:warmup:tail-fraction": (CandidateClassification.NUMERICAL_GATE,),
    "PLAN:warmup:x-norm-inputs": (CandidateClassification.NUMERICAL_GATE,),
}
_GATE_ATOM_LINKS: dict[str, tuple[str, ...]] = {
    "COUPLING:_classify_correlation:floor-finite": (
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::predicate_call_atom::b410e8e4c610d6de::0",
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::finite_predicate::b410e8e4c610d6de::0",
    ),
    "COUPLING:_classify_correlation:lower-noise-floor": (
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::compare::bb065f25cc0fb571::0",
    ),
    "COUPLING:_classify_correlation:upper-noise-floor": (
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::compare::c96b8c0f89a8f758::0",
    ),
    "COUPLING:_classify_correlation:value-finite": (
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::predicate_call_atom::6149324691a26641::0",
        "src/bayesmith/diagnose/coupling.py::<module>._classify_correlation::finite_predicate::6149324691a26641::0",
    ),
    "COUPLING:_condition_number:finite-spectrum": (
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::predicate_call_atom::334dace9ea4ca226::0",
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::finite_predicate::334dace9ea4ca226::0",
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::predicate_call_atom::42b1ecb26daa2662::0",
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::finite_predicate::42b1ecb26daa2662::0",
    ),
    "COUPLING:_condition_number:positive-spectrum": (
        "src/bayesmith/diagnose/coupling.py::<module>._condition_number::compare::1c27e6520e2bee10::0",
    ),
    "EAGER:LadderConfig:integer-threshold-domain": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::compare::d56ec5b9fc884113::0",
    ),
    "EAGER:LadderConfig:low-rank-fraction-domain": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::compare::fa2fc329c066f2fc::0",
    ),
    "EAGER:LadderConfig:structure-tolerance-domain": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::predicate_call_atom::433cf901a0d411f0::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::finite_predicate::433cf901a0d411f0::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::predicate_call_atom::b0aea59f157c766f::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::finite_predicate::b0aea59f157c766f::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::compare::38550887364ae38c::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LadderConfig.__post_init__::compare::baa72a1b6357ac36::0",
    ),
    "EAGER:LogDetProblem:lambda-spd": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.LogDetProblem.__init__::predicate_call_atom::2f5d27d8275ca569::0",
    ),
    "EAGER:array-normalization:shape-and-finiteness": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::compare::e6cd3e456539473f::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::compare::0023146ac52e33f1::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::compare::4b9b4c4f077e83f4::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::predicate_call_atom::c06551febb069e8b::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::predicate_call_atom::35b81c96644141fb::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._read_only_array::finite_predicate::35b81c96644141fb::0",
    ),
    "EAGER:dense-condition:strict-dtype-ceiling": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::predicate_call_atom::fc2f1b3e6eabcfd1::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::finite_predicate::fc2f1b3e6eabcfd1::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._condition_certificate::compare::5c3d08ac757d1ce7::0",
    ),
    "EAGER:factor-balance:exact-power-of-two-reversibility": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::508f419c8ad7bceb::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::633c4ba7bc098a19::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::finite_predicate::633c4ba7bc098a19::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::ff9f64daa7a7b618::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::09a90b48970dbb22::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::finite_predicate::09a90b48970dbb22::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::c62727bc99ba15f1::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._balanced_factor_columns::predicate_call_atom::78d67bade0ec5240::0",
    ),
    "EAGER:factor-base:condition-ceiling": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::3e2925807fea5201::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::8979ff3006924922::0",
    ),
    "EAGER:factor-base:error-budget": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::f077a4f9072830b5::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::607d8e28d211c1e8::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::0520942938172abf::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::boolean_atom::7eb5677a4b7f505c::0",
    ),
    "EAGER:factor-projection:error-budget": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::2223debe39afb9d2::1",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::2223debe39afb9d2::1",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::5dfa833658bbcfa3::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::8bfb56ffb90db4b0::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::8bfb56ffb90db4b0::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::c30cd037d5180310::0",
    ),
    "EAGER:factor-projection:finite-qr-arithmetic": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::decision_predicate::faa44f657bf8cd00::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::bdef49efc73fa379::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::46056fb4c1718479::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::46056fb4c1718479::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::db5916e25d36ef64::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::854832823b821fe4::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::854832823b821fe4::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::c2294f94e70821ca::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::f26acd7566369a93::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::f26acd7566369a93::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::039c47cc071f1e8e::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::6e2471591892211c::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::6e2471591892211c::0",
    ),
    "EAGER:factor-projection:whitened-positive-spectrum": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::386f873e56933cc1::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::92740bdbc4d0e19b::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::92740bdbc4d0e19b::0",
    ),
    "EAGER:factor-reconstruction:layout-exactness": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matching_factor_reconstruction::predicate_call_atom::b678026387cb59b1::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._matching_factor_reconstruction::predicate_call_atom::cba17196a4c6f789::0",
    ),
    "EAGER:factor-reduced:acceptance-budget": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::32ca60671f93bf91::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::32ca60671f93bf91::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::db227b7e3fb87859::0",
    ),
    "EAGER:factor-reduced:diagonal-certificate": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::0f7dbea9e949f40c::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::27dc5722290ed8d0::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::5312fdea367be588::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::5312fdea367be588::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::1b033f6c18a10c46::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::4fa526a712819567::0",
    ),
    "EAGER:factor-reduced:qr-certificate": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::0f7dbea9e949f40c::1",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::10bd024c3347566e::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::10bd024c3347566e::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::5df229387047e996::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::88a2e3998c62924a::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::finite_predicate::88a2e3998c62924a::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::c85350bec59e195e::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::predicate_call_atom::663cd7e7e74271e5::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._factor_projection_certificate::compare::b5bc8c97b914c9dd::0",
    ),
    "EAGER:frozen-probes:identity-width-order": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::compare::a463584319db77e0::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::compare::5fe9ad3578e110cc::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.frozen_hutchinson_trace_logdet::compare::7f499b566536192a::0",
    ),
    "EAGER:lambda-logdet:subnormal-rescale": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.lambda_logdet::compare::411d89f67a1cb014::0",
    ),
    "EAGER:spectral-radius:finite-measurement": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::predicate_call_atom::bfc361b4701f87e4::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::predicate_call_atom::67c48bccbcafebff::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::finite_predicate::67c48bccbcafebff::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::predicate_call_atom::ce0ccc48257ad825::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::predicate_call_atom::6ccd49715b3f8858::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.spectral_radius::finite_predicate::6ccd49715b3f8858::0",
    ),
    "EAGER:state-space:block-chain-exactness": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::compare::cbe13cb99f8f4f4e::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::predicate_call_atom::5bc915df4c4a49d1::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_block_chain::compare::3b7e7b0a86cea352::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::predicate_call_atom::038ec51d46ad9e6e::0",
    ),
    "EAGER:state-space:payload-domain": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::predicate_call_atom::59cf226300ca4fe8::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::predicate_call_atom::b838824f5fcd600a::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::predicate_call_atom::5662962bf2f78843::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::finite_predicate::5662962bf2f78843::0",
    ),
    "EAGER:structured:exact-shape-and-spectrum": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::compare::43a7e748e57b393e::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.structured_logdet::predicate_call_atom::1bb392e6480f6eed::0",
    ),
    "EAGER:symmetry:tolerant-representative": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::489cb72b71678a88::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::ce0ccc48257ad825::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::6ccd49715b3f8858::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::finite_predicate::6ccd49715b3f8858::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::predicate_call_atom::f9c84ba9f4f55168::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._is_positive_definite::compare::6873fea2c5a9a753::0",
    ),
    "EAGER:trace:actual-rho-strict": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::compare::23509a12d46634b8::0",
    ),
    "EAGER:trace:certificate-domain": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::compare::2ff2fddab1a04ef2::0",
    ),
    "EAGER:trace:certificate-upper-bound": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._validate_strict_rho::compare::0e23a71d8847891c::0",
    ),
    "EAGER:trace:exact-power-trace-evidence": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::compare::7f499b566536192a::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::compare::2ee9c826b03cae0b::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::predicate_call_atom::10e304f715b34f36::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::predicate_call_atom::7664e7df114ecb98::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::finite_predicate::7664e7df114ecb98::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>._power_traces_match::predicate_call_atom::d1ce1fecca04a454::0",
    ),
    "EAGER:trace:tail-domain-and-order": (
        "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::compare::6f6bf7e28f2b4e29::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.trace_log_tail_bound::compare::7f499b566536192a::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.whole_trace_log_tail_bound::compare::cb1557568a2a2887::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::predicate_call_atom::181c8e652b10fae7::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::finite_predicate::181c8e652b10fae7::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::compare::d42259ba0e4e85a6::0",
        "src/bayesmith/marginal/_logdet_eager.py::<module>.choose_trace_order::compare::be8bebac0fc5f981::0",
    ),
    "LADDER:determinant-lemma:payload": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::14820d49ac03ef64::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c515480aa003dc6d::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::91367abf12d9ecf4::0",
    ),
    "LADDER:finite:payload-rho": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::4a956d9542c5c0c2::0",
    ),
    "LADDER:rho:measurement": (),
    "LADDER:rung0:base": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::d9082e7801c0ca71::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::0daa42baad4d8b5a::0",
    ),
    "LADDER:rung1:low-rank-size": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::df59227e94deb60c::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::44f3815eb0ce7bcd::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::59f66427dc73ee94::0",
    ),
    "LADDER:rung3:structured": (),
    "LADDER:rung4:dense": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::f0ba3f1026fc300c::0",
    ),
    "LADDER:rung5:finite-executable": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::6a781d45dfcc23af::0",
    ),
    "LADDER:rung5:finite-size": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::5b64ae2cb778a757::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::554111cb2ca3295d::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::decision_predicate::df59227e94deb60c::1",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::80d02c27a4b17ad0::0",
    ),
    "LADDER:rung6:trace": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::3",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::bc108dc8b78fe54f::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5d6aaa0c807a1bf8::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::75bdf97ba8300ea7::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6f6bf7e28f2b4e29::0",
    ),
    "LADDER:rung7:frozen": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::4",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c087bc09f45f17ee::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c62b0e04d76df049::1",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5a38bc5f8f0f2d61::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::5d6aaa0c807a1bf8::1",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::75bdf97ba8300ea7::1",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6f6bf7e28f2b4e29::1",
    ),
    "LADDER:sigma:payload-symmetry": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::compare::3204fd05446ce318::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::predicate_call_atom::3916090d68cfe569::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._sigma_payload::predicate_call_atom::5092fdf71a6d9964::0",
    ),
    "LADDER:sigma:symmetry-spd-condition": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::compare::3204fd05446ce318::1",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::5092fdf71a6d9964::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::predicate_call_atom::ded5826f515f9d49::0",
    ),
    "LADDER:structure:circulant-tolerance-spectrum": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::7ead0b7ae54d3a4a::0",
    ),
    "LADDER:structure:compact-diagonal-positive": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::d003bc8bdc0751d7::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::d1a71af8a65a9965::0",
    ),
    "LADDER:structure:diagonal-tolerance": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::cf19d788dfb142fc::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::3b501ceeac5159d6::0",
    ),
    "LADDER:structure:kronecker-evidence": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::8cea1b9d73a001b5::0",
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::2aaa6008086ab43f::0",
    ),
    "LADDER:structure:toeplitz-tolerance": (
        "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::6a0aad293d04c15d::0",
    ),
    "MAP:map_estimate:finite-derivative-payload": (
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::087e9fc8a8556c2f::0",
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::8f9dae6ee5f5a679::0",
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::8f9dae6ee5f5a679::0",
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::c100b2a3318e04ae::0",
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::c602a9209c1f710a::0",
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::c602a9209c1f710a::0",
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::e2015a81ed24bc08::0",
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::predicate_call_atom::5ae94c7be7ed7e6a::0",
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::finite_predicate::5ae94c7be7ed7e6a::0",
    ),
    "MAP:map_estimate:stationarity-floor": (
        "src/bayesmith/diagnose/map.py::<module>.map_estimate::compare::ed87ffcc51d9fe06::0",
    ),
    "PLAN:audit:retained-trace-evidence": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::compare::4ae94ae7e3ef3f98::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.audit_retained_power_traces::compare::e56c637394ef295f::0",
    ),
    "PLAN:canonical-probes:runtime-finite": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::predicate_call_atom::161234ef6c373d98::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::predicate_call_atom::b992e6871aa5ff32::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._canonical_runtime_probes::finite_predicate::b992e6871aa5ff32::0",
    ),
    "PLAN:certificate:error-budget-domain": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::502e1e1ca5864853::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::08cf8960c456a2ca::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::d73054a840b6cacf::0",
    ),
    "PLAN:certificate:optional-scale-domain": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::07025b3c333759d1::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::98328934d03bd7b3::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::predicate_call_atom::d326b31a3a3e51a1::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::finite_predicate::d326b31a3a3e51a1::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::271377e809b7b124::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::f270ed94716d3f21::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::88d32009a53226a5::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::predicate_call_atom::c3c046963e6ed35a::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::finite_predicate::c3c046963e6ed35a::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::0ac44251ac6cc986::0",
    ),
    "PLAN:certificate:order-is-derived": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::decision_predicate::5278515c7ba5de86::0",
    ),
    "PLAN:certificate:rho-domain-and-coverage": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::cfd9ed6ff01db72b::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::ddc2a9a5cd66ebe3::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.RhoCertificate.__post_init__::compare::1c74be788b92b3e1::0",
    ),
    "PLAN:factory-certificate:lambda-scale": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::1943e05d2c7883c5::0",
    ),
    "PLAN:factory-certificate:order-and-rank": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::ffbfe426624c1b2c::0",
    ),
    "PLAN:factory-certificate:x-norm": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_plan_certificate::decision_predicate::0cff16fe3f2e1490::0",
    ),
    "PLAN:frozen-factory:probe-presence-width": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_frozen_trace_log_plan::decision_predicate::c9ac3a5056f615b8::0",
    ),
    "PLAN:frozen:probe-energy-range": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::predicate_call_atom::ec9578c6de121206::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::finite_predicate::ec9578c6de121206::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._frozen_probe_energy_bounds::compare::0fcd61fc2c50502a::0",
    ),
    "PLAN:frozen:x-bound-runtime-range": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_frozen_runtime_range::decision_predicate::c10e15779304e86d::0",
    ),
    "PLAN:gamma:operation-count-domain": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._gamma_for_count::decision_predicate::8da450fabbb947ef::0",
    ),
    "PLAN:measurement:lambda-logdet-finite": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::predicate_call_atom::d14485cd4ec6815a::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_lambda_logdet_scale::finite_predicate::d14485cd4ec6815a::0",
    ),
    "PLAN:measurement:x-norm-finite": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::predicate_call_atom::bfc361b4701f87e4::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::predicate_call_atom::67c48bccbcafebff::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::finite_predicate::67c48bccbcafebff::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::predicate_call_atom::e4236c24a78c1cde::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._checked_x_operator_norm::finite_predicate::e4236c24a78c1cde::0",
    ),
    "PLAN:multiplicity:index-and-gamma-domain": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._normalize_rho_multiplicity::decision_predicate::c19d43bd613dda15::0",
    ),
    "PLAN:outward-arithmetic:positive-underflow": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::compare::def44deb716d6bcc::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::predicate_call_atom::6149324691a26641::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._outward_nonnegative::finite_predicate::6149324691a26641::0",
    ),
    "PLAN:runtime-call:scalar-and-dtype": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::compare::3224ec9d7d999c0f::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::decision_predicate::afa73a661affbab4::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::compare::4df8c4a204ea9933::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::decision_predicate::066e0595e1ad7835::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._require_runtime_precision::compare::29d03c938376b0b3::0",
    ),
    "PLAN:runtime-range:product": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::predicate_call_atom::9d81c8aa824df72e::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::finite_predicate::9d81c8aa824df72e::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::predicate_call_atom::a24bd6de02c70ffc::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::finite_predicate::a24bd6de02c70ffc::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::compare::e3b414eecff3506e::0",
    ),
    "PLAN:runtime-range:sum": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::predicate_call_atom::9d81c8aa824df72e::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::finite_predicate::9d81c8aa824df72e::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::predicate_call_atom::a24bd6de02c70ffc::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::finite_predicate::a24bd6de02c70ffc::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::compare::1b346425452c0d94::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::compare::aa4ae7fb0f99df59::0",
    ),
    "PLAN:runtime:base-scale-range": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::2e3fba3ea6b5f4f0::0",
    ),
    "PLAN:runtime:expected-and-ulp-finite": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::4e9bf2538eccc387::0",
    ),
    "PLAN:runtime:frozen-prerequisites-and-series": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::predicate_call_atom::2368d001f3e9a883::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::finite_predicate::2368d001f3e9a883::0",
    ),
    "PLAN:runtime:sigma-finite-and-positive": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::predicate_call_atom::8fe5d7e22a754fc2::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::compare::0ca59d408e4f04f6::0",
    ),
    "PLAN:runtime:total-error-budget": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>._validate_runtime_precision::decision_predicate::b7c85688cf179fbc::0",
    ),
    "PLAN:trace-factory:exact-evidence": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.make_trace_log_plan::compare::e56c637394ef295f::0",
    ),
    "PLAN:warmup:lambda-scale-inputs": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::6149324691a26641::1",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::6149324691a26641::1",
    ),
    "PLAN:warmup:rho-inputs-and-margin": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::7b5386c0f5e68697::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::6149324691a26641::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::6149324691a26641::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::ceb972d0b12db89c::0",
    ),
    "PLAN:warmup:rho-roundoff-ceiling": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::2d66551cc373e8a1::0",
    ),
    "PLAN:warmup:tail-fraction": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::d7756508c0ebfa8c::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::d7756508c0ebfa8c::0",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::6d40d228cfffeb1e::0",
    ),
    "PLAN:warmup:x-norm-inputs": (
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::7b5386c0f5e68697::1",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::6149324691a26641::2",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::6149324691a26641::2",
        "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::ceb972d0b12db89c::1",
    ),
}

_GATE_ATOM_LINKS.update(
    {
        "LADDER:rung0:base": (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::48196fb8375388cd::2",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::d9082e7801c0ca71::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::969c43f564c8d1e8::0",
        ),
        "LADDER:rung1:low-rank-size": (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::c515480aa003dc6d::1",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::772b10fa9b76b884::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::fee2daab0ef0a42e::1",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2acb41d6956562c0::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::44f3815eb0ce7bcd::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::59f66427dc73ee94::0",
        ),
        "LADDER:rung2:chain": (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2df45b3b0a16f43b::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2acb41d6956562c0::1",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6bb5018fea5f3016::1",
        ),
        "LADDER:rung4:dense": (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::f0ba3f1026fc300c::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::6bb5018fea5f3016::2",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2acb41d6956562c0::2",
        ),
        "LADDER:rung5:finite-executable": (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::57439eb4dffa740a::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::cbe2295ff59158de::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::2acb41d6956562c0::3",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::fee2daab0ef0a42e::3",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::boolean_atom::969c43f564c8d1e8::1",
        ),
    }
)


_GATE_SOURCE_LINKS.update(
    {
        "LADDER:structure:diagonal-tolerance": (
            (
                "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::e4f1323628f87cb8::0",
                "kind is None and _is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
            ),
            (
                "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::3b501ceeac5159d6::1",
                "_is_diagonal(sigma, rtol=config.structure_rtol, atol=config.structure_atol)",
            ),
        ),
        "LADDER:structure:kronecker-evidence": (
            (
                "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::eb776e7e35594b34::0",
                "all((_is_positive_definite(factor) for factor in problem.structure.factors))",
            ),
            (
                "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::d815c74c8c74ed82::0",
                "not factors_spd",
            ),
            (
                "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::decision_predicate::d611227f5fbefbc9::0",
                "reconstructed.shape == sigma.shape and np.array_equal(reconstructed, sigma)",
            ),
        ),
        "PLAN:warmup:lambda-scale-inputs": (
            (
                "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::d52eae775e943fd0::0",
                "not bases or any((not np.isfinite(value) for value in bases))",
            ),
            (
                "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::d619e405863b0f28::0",
                "not np.isfinite(lambda_logdet_margin) or lambda_logdet_margin < 0.0",
            ),
        ),
        "PLAN:warmup:x-norm-inputs": (
            (
                "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::793fd4e8d52e933e::0",
                "not norms or any((not np.isfinite(value) or value < 0.0 for value in norms))",
            ),
            (
                "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::f64f954de34e0c17::0",
                "not np.isfinite(x_operator_norm_margin) or x_operator_norm_margin < 0.0",
            ),
        ),
        "PLAN:runtime-range:product": (
            (
                "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::decision_predicate::958e52d096caca96::0",
                "not np.isfinite(left) or not np.isfinite(right) or left > maximum / right",
            ),
            (
                "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::decision_predicate::2a13a4e35f93ecfe::0",
                "not np.isfinite(result) or result > maximum",
            ),
        ),
        "PLAN:runtime-range:sum": (
            (
                "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::decision_predicate::3f0c849b23f9cef1::0",
                "not np.isfinite(left) or not np.isfinite(right) or right > maximum or (left > maximum - right)",
            ),
            (
                "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::decision_predicate::2a13a4e35f93ecfe::0",
                "not np.isfinite(result) or result > maximum",
            ),
        ),
    }
)

_GATE_ATOM_LINKS.update(
    {
        "LADDER:structure:kronecker-evidence": (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::75f6a0da8f1923cb::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::compare::8cea1b9d73a001b5::0",
            "src/bayesmith/marginal/_logdet_ladder.py::<module>._structure_request::predicate_call_atom::2aaa6008086ab43f::0",
        ),
        "PLAN:warmup:lambda-scale-inputs": (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::6149324691a26641::1",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::6149324691a26641::1",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::b741736709d56096::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::b741736709d56096::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::f4f39dc0de81efbf::0",
        ),
        "PLAN:warmup:x-norm-inputs": (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::decision_predicate::7b5386c0f5e68697::1",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::6149324691a26641::2",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::6149324691a26641::2",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::ceb972d0b12db89c::1",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::predicate_call_atom::79f7278fa4fdf288::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::finite_predicate::79f7278fa4fdf288::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.certify_warmup_rho::compare::b6b21b85bcabf315::0",
        ),
        "PLAN:runtime-range:product": (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::predicate_call_atom::9d81c8aa824df72e::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::finite_predicate::9d81c8aa824df72e::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::predicate_call_atom::a24bd6de02c70ffc::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::finite_predicate::a24bd6de02c70ffc::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::compare::e3b414eecff3506e::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::predicate_call_atom::3cf98a58825628a4::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::finite_predicate::3cf98a58825628a4::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_product::compare::9b9d8f993db72706::0",
        ),
        "PLAN:runtime-range:sum": (
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::predicate_call_atom::9d81c8aa824df72e::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::finite_predicate::9d81c8aa824df72e::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::predicate_call_atom::a24bd6de02c70ffc::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::finite_predicate::a24bd6de02c70ffc::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::compare::1b346425452c0d94::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::compare::aa4ae7fb0f99df59::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::predicate_call_atom::3cf98a58825628a4::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::finite_predicate::3cf98a58825628a4::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>._runtime_range_sum::compare::9b9d8f993db72706::0",
        ),
    }
)

_DECLARED_SOURCE_ANCHORS.update(
    {
        "LADDER:structure:diagonal-tolerance": (
            SourceAnchor(
                LADDER,
                "<module>._structure_request",
                CandidateFamily.DECISION_PREDICATE,
            ),
            SourceAnchor(
                LADDER, "<module>._structure_request", CandidateFamily.PREDICATE_CALL
            ),
        ),
        "LADDER:structure:kronecker-evidence": (
            SourceAnchor(
                LADDER,
                "<module>._structure_request",
                CandidateFamily.DECISION_PREDICATE,
            ),
            SourceAnchor(
                LADDER,
                "<module>._structure_request",
                CandidateFamily.DECISION_PREDICATE,
            ),
            SourceAnchor(
                LADDER,
                "<module>._structure_request",
                CandidateFamily.DECISION_PREDICATE,
            ),
        ),
        "PLAN:warmup:lambda-scale-inputs": (
            SourceAnchor(
                PLAN, "<module>.certify_warmup_rho", CandidateFamily.DECISION_PREDICATE
            ),
            SourceAnchor(
                PLAN, "<module>.certify_warmup_rho", CandidateFamily.DECISION_PREDICATE
            ),
        ),
        "PLAN:warmup:x-norm-inputs": (
            SourceAnchor(
                PLAN, "<module>.certify_warmup_rho", CandidateFamily.DECISION_PREDICATE
            ),
            SourceAnchor(
                PLAN, "<module>.certify_warmup_rho", CandidateFamily.DECISION_PREDICATE
            ),
        ),
        "PLAN:runtime-range:product": (
            SourceAnchor(
                PLAN,
                "<module>._runtime_range_product",
                CandidateFamily.DECISION_PREDICATE,
            ),
            SourceAnchor(
                PLAN,
                "<module>._runtime_range_product",
                CandidateFamily.DECISION_PREDICATE,
            ),
        ),
        "PLAN:runtime-range:sum": (
            SourceAnchor(
                PLAN, "<module>._runtime_range_sum", CandidateFamily.DECISION_PREDICATE
            ),
            SourceAnchor(
                PLAN, "<module>._runtime_range_sum", CandidateFamily.DECISION_PREDICATE
            ),
        ),
    }
)

_DECLARED_SOURCE_CLASSIFICATIONS.update(
    {
        "LADDER:structure:diagonal-tolerance": (
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_GATE,
        ),
        "LADDER:structure:kronecker-evidence": (
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_GATE,
        ),
        "PLAN:warmup:lambda-scale-inputs": (
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_GATE,
        ),
        "PLAN:warmup:x-norm-inputs": (
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_GATE,
        ),
        "PLAN:runtime-range:product": (
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_GATE,
        ),
        "PLAN:runtime-range:sum": (
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_GATE,
        ),
    }
)


_GATE_EXTRA_TARGETS = {
    "COUPLING:block_coupling:f-xx-spd": (
        "src/bayesmith/diagnose/coupling.py::<module>.block_coupling::linalg_exception_premise::dc1b89c1f16106a5::0",
    ),
}


def _dependencies(gate_id: str) -> tuple[str, ...]:
    if gate_id in {
        "LADDER:structure:diagonal-tolerance",
        "LADDER:structure:circulant-tolerance-spectrum",
        "LADDER:structure:toeplitz-tolerance",
    }:
        return ("EAGER:structured:exact-shape-and-spectrum",)
    if gate_id == "EAGER:LogDetProblem:lambda-spd":
        return ("EAGER:symmetry:tolerant-representative",)
    if gate_id == "LADDER:sigma:symmetry-spd-condition":
        return (
            "EAGER:symmetry:tolerant-representative",
            "EAGER:dense-condition:strict-dtype-ceiling",
        )
    if gate_id == "LADDER:rank:evidence":
        return ("EAGER:factor-reduced:acceptance-budget",)
    if gate_id in {
        "LADDER:rung6:trace",
        "LADDER:rung7:frozen",
    }:
        return ("EAGER:trace:actual-rho-strict",)
    if gate_id == "LADDER:finite:payload-rho":
        return ("EAGER:finite:newton-stability-rho",)
    if gate_id == "LADDER:rho:measurement":
        return ("EAGER:spectral-radius:finite-measurement",)
    if gate_id == "LADDER:determinant-lemma:payload":
        return (
            "EAGER:factor-reconstruction:layout-exactness",
            "EAGER:factor-reduced:acceptance-budget",
        )
    if gate_id == "PLAN:factory-certificate:strict-rho":
        return (
            "EAGER:trace:actual-rho-strict",
            "EAGER:trace:certificate-domain",
            "EAGER:trace:certificate-upper-bound",
        )
    if gate_id in {
        "PLAN:certificate:order-is-derived",
        "PLAN:factory-certificate:order-and-rank",
    }:
        return ("EAGER:trace:tail-domain-and-order",)
    return ()


def _metadata(
    *,
    quantity: str,
    threshold: str,
    provenance: ThresholdProvenance,
    admitted_outcome: str,
    refused_outcome: str,
    oracle: str,
    axis_name: str,
    low: str,
    endpoints: tuple[str, str],
    high: str,
    extreme: str,
    fixture_scale_policy: FixtureScalePolicy,
) -> GateMetadata:
    """Construct one literal metadata row without deriving its semantics."""

    return GateMetadata(
        quantity=quantity,
        threshold=threshold,
        provenance=provenance,
        admitted_outcome=admitted_outcome,
        refused_outcome=refused_outcome,
        oracle=oracle,
        axes=(
            AxisRange(
                name=axis_name,
                low=low,
                endpoints=endpoints,
                high=high,
                extreme=extreme,
            ),
        ),
        fixture_scale_policy=fixture_scale_policy,
    )


GATE_METADATA: dict[str, GateMetadata] = {
    "EAGER:LadderConfig:integer-threshold-domain": _metadata(
        quantity="each of low_rank_max, dense_max_n, finite_max_n, and finite_max_rank; refuse when any integer is < 0.",
        threshold="closed integer domain value >= 0; API contract.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="admit construction and retain the four cutoffs",
        refused_outcome="Refuse LadderConfig construction with the non-negative-threshold ValueError",
        oracle="direct Python integer comparison for each field; neighbouring cases are identical configs differing in one field only.",
        axis_name="Boundary cells for each of low_rank_max, dense_max_n, finite_max_n, and finite_max_rank; refuse when any integer is < 0.",
        low="1",
        endpoints=("0 admits", "-1 refuses"),
        high="sys.maxsize",
        extreme="one negative field among cutoffs (13, 24, 7, 19); bool tested separately",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:LadderConfig:low-rank-fraction-domain": _metadata(
        quantity="low_rank_fraction; refuse unless 0.0 <= fraction <= 1.0.",
        threshold="closed real interval [0, 1]; API contract.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="admit configuration at both endpoints",
        refused_outcome="Refuse before any ladder routing outside the interval or for NaN",
        oracle="Decimal(str(value)) interval comparison, with explicit NaN handling.",
        axis_name="Boundary cells for low_rank_fraction; refuse unless 0.0 <= fraction <= 1.0.",
        low="0.125",
        endpoints=(
            "0.0, negative subnormal nextafter(0, -inf)",
            "1.0, nextafter(1, +inf)",
        ),
        high="4.0",
        extreme="NaN, +/-inf, smallest positive subnormal",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:LadderConfig:structure-tolerance-domain": _metadata(
        quantity="structure_rtol and structure_atol; both must be finite and non-negative.",
        threshold="product domain [0,+inf) intersected with IEEE finite values; API contract (the default magnitudes are chosen policy, not a derived error theorem).",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="admit config and pass tolerances to symmetry/structure checks",
        refused_outcome="Refuse construction if either component is negative or non-finite",
        oracle="scalar math.isfinite(x) and x >= 0 applied independently to both fields.",
        axis_name="Boundary cells for structure_rtol and structure_atol; both must be finite and non-negative.",
        low="(1e-11,1e-13) and non-unit (0.3,0.07)",
        endpoints=("0.0", "negative subnormal nextafter(0,-inf) in each coordinate"),
        high="finfo.max",
        extreme="NaN, infinities, subnormal, and one bad coordinate with the other valid",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:array-normalization:shape-and-finiteness": _metadata(
        quantity="requested ndim, actual rank, and every normalized array entry; accept only requested rank (when supplied), rank in {1,2}, and all-finite data.",
        threshold="discrete shape contract plus IEEE finite domain; API contract/exact domain.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="return a contiguous read-only float32/float64 vector or matrix",
        refused_outcome="Refuse with the dimension/matrix-input/finiteness ValueError before numerical work",
        oracle="array.ndim, exact shape comparison, and scalar math.isfinite over a flattened copy.",
        axis_name="Boundary cells for requested ndim, actual rank, and every normalized array entry; accept only requested rank (when supplied), rank in {1,2}, and all-finite data.",
        low="non-unit vector [1.3,-2.4] and 2x2 matrix",
        endpoints=(
            "rank 1/rank 2 (admit)",
            "scalar/rank 3 (refuse), last finite versus +inf",
        ),
        high="large finite shape",
        extreme="empty 1-D/2-D arrays, NaN, infinities, float16 promotion, integer promotion",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:LogDetProblem:lambda-spd": _metadata(
        quantity="compact diagonal Lambda entries or dense symmetric Lambda eigenvalues; _is_positive_definite(lam) must be true.",
        threshold="symmetry plus strict minimum eigenvalue/diagonal entry > 0; mathematical domain/API contract.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="construct immutable LogDetProblem",
        refused_outcome="Refuse immediately with Lambda must be symmetric positive definite",
        oracle="diagonal minimum for vectors; for dense fixtures use exact symmetry plus high-precision LDL inertia or analytic eigenvalues (and Cholesky as a neighbouring method).",
        axis_name="Boundary cells for compact diagonal Lambda entries or dense symmetric Lambda eigenvalues; _is_positive_definite(lam) must be true.",
        low="diagonal [1.3,2.4] or SPD [[2.4,.2],[.2,1.3]]",
        endpoints=(
            "minimum eigenvalue/entry positive subnormal nextafter(0, +inf)",
            "0, negative subnormal nextafter(0, -inf)",
        ),
        high="ill-scaled but SPD diagonal",
        extreme="asymmetric, NaN/inf (normally caught upstream), repeated eigenvalues, subnormal positive pivot",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:factor-balance:exact-power-of-two-reversibility": _metadata(
        quantity="for every nonzero factor column, both ldexp-scaled sides must remain finite and inverse ldexp must reproduce both original columns bit-for-bit.",
        threshold="exact IEEE representability under a power-of-two gauge; derived numerical-safety boundary.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="return balanced columns (zeroing a pair when either side is an exact zero column)",
        refused_outcome="Refuse before QR on underflow, overflow, or irreversible scaling",
        oracle="compare hexadecimal float representations before/after integer exponent shifts; for small fixtures verify the outer product with Decimal.",
        axis_name="Boundary cells for for every nonzero factor column, both ldexp-scaled sides must remain finite and inverse ldexp must reproduce both original columns bit-for-bit.",
        low="balanced non-unit columns near 1.3",
        endpoints=(
            "exponent shifts whose smallest nonzero entry is finfo.smallest_subnormal",
            "one shift that flushes it to zero, and finfo.max versus overflow",
        ),
        high="exponent imbalance",
        extreme="one-sided zero, signed zero, mixed subnormal/maximal entries, float32 and float64",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:factor-reconstruction:layout-exactness": _metadata(
        quantity="authoritative perturbation versus left @ right.T under the four supported C/F storage combinations; one product must satisfy np.array_equal.",
        threshold="exact elementwise equality, including shape; exact algebraic-evidence contract.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="mark reconstruction evidence true and use the matching product",
        refused_outcome="Otherwise retain the canonical product and later refuse factor-rank evidence",
        oracle="explicit triple-loop dot products with Decimal/Fraction for small matrices and byte/shape equality.",
        axis_name="Boundary cells for authoritative perturbation versus left @ right.T under the four supported C/F storage combinations; one product must satisfy np.array_equal.",
        low="exact non-unit rank-one product",
        endpoints=(
            "exact bitwise-identical product",
            "one output entry moved by one ULP",
        ),
        high="multi-column cancellation",
        extreme="signed zero, C/F layouts, empty rank, subnormals, large finite factors",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:factor-projection:finite-qr-arithmetic": _metadata(
        quantity="QR bases, core, and projected perturbation; all entries must be finite and QR/matmul must not raise supported arithmetic errors.",
        threshold="IEEE finite-result safety domain; derived implementation certificate.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="continue to whitening and error certification",
        refused_outcome="Refuse with the finite-QR-arithmetic ValueError",
        oracle="high-precision modified Gram-Schmidt and matrix multiplication for small fixtures, followed by explicit finiteness and Q.T@Q checks.",
        axis_name="Boundary cells for QR bases, core, and projected perturbation; all entries must be finite and QR/matmul must not raise supported arithmetic errors.",
        low="well-scaled independent columns near 1.3",
        endpoints=("largest finite QR product", "next exponent produces overflow"),
        high="nearly dependent columns",
        extreme="zero rank, exact dependence, subnormal/maximum finite, NaN/Inf intermediates",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:factor-projection:whitened-positive-spectrum": _metadata(
        quantity="smallest eigenvalue of symmetrized Lambda^-1/2 Sigma Lambda^-1/2; it must be finite and strictly positive before eta=residual_norm/lambda_min is meaningful.",
        threshold="strict SPD boundary lambda_min > 0; exact mathematical domain.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="form finite eta",
        refused_outcome="Otherwise set eta=inf, causing projection certification and factor route refusal",
        oracle="analytic eigenvalues for diagonal/2x2 fixtures or high-precision symmetric eigensolver/LDL inertia.",
        axis_name="Boundary cells for smallest eigenvalue of symmetrized Lambda^-1/2 Sigma Lambda^-1/2; it must be finite and strictly positive before eta=residual_norm/lambda_min is meaningful.",
        low="eigenvalues (1.3,2.4)",
        endpoints=(
            "lambda_min=positive subnormal nextafter(0, +inf)",
            "0, negative subnormal nextafter(0, -inf)",
        ),
        high="large condition number while positive",
        extreme="NaN/inf spectrum, repeated minimum, subnormal positive eigenvalue",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:factor-projection:error-budget": _metadata(
        quantity="eta=||whitened_residual||_2/lambda_min and log_error_bound=-n*log1p(-eta); require finite eta < 1, finite bound, and bound <= sqrt(eps(target_dtype)).",
        threshold="strict perturbation radius at eta=1 plus closed derived log-error ceiling; derived forward-error certificate.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="certify authoritative-P projection",
        refused_outcome="Otherwise refuse factor evidence with eta/bound/ceiling diagnostics",
        oracle="high-precision spectral norm and Decimal/mpmath -n*log1p(-eta) compared with sqrt(eps).",
        axis_name="Boundary cells for eta=||whitened_residual||_2/lambda_min and log_error_bound=-n*log1p(-eta); require finite eta < 1, finite bound, and bound <= sqrt(eps(target_dtype)).",
        low="eta=0.1 with bound below ceiling",
        endpoints=(
            "bound nextafter(ceiling,-inf), ceiling, nextafter(ceiling,+inf)",
            "eta nextafter(1,0), 1",
        ),
        high="large residual",
        extreme="eta 0, NaN/inf, subnormal residual, dimension 1 versus large n",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:factor-base:condition-ceiling": _metadata(
        quantity="dense Lambda 2-norm condition base_condition; it must be finite and strictly < 1/sqrt(eps(work_dtype)) (diagonal compact path deliberately has infinite ceiling).",
        threshold="base condition < 1/sqrt(eps(work dtype)) (strict)",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="allow determinant-lemma base factorization/solve certificate",
        refused_outcome="Refuse factor route with measured condition and ceiling",
        oracle="singular-value ratio in higher precision or analytic diagonal condition, compared with 1/sqrt(eps).",
        axis_name="Boundary cells for dense Lambda 2-norm condition base_condition; it must be finite and strictly < 1/sqrt(eps(work_dtype)) (diagonal compact path deliberately has infinite ceiling).",
        low="condition 2.4/1.3",
        endpoints=("nextafter(ceiling, 0)", "ceiling, nextafter(ceiling, +inf)"),
        high="singular/near-singular matrix",
        extreme="identity, repeated singular values, float32/float64 ceilings, NaN/inf condition",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:factor-base:error-budget": _metadata(
        quantity="base solve eta=gamma_(3n)*condition, its factorization/log-roundoff bound, and base_log_error_bound; require finite 0 <= eta < 1, finite bound, and bound <= sqrt(eps(target_dtype)).",
        threshold="derived floating-point forward-error certificate; eta upper endpoint is strict, log-error ceiling closed.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="mark base arithmetic valid",
        refused_outcome="Otherwise refuse determinant-lemma route with condition and bound diagnostics",
        oracle="compute gamma and log bounds with Decimal/mpmath from n, eps, condition, and factor diagonal logs.",
        axis_name="Boundary cells for base solve eta=gamma_(3n)*condition, its factorization/log-roundoff bound, and base_log_error_bound; require finite 0 <= eta < 1, finite bound, and bound <= sqrt(eps(target_dtype)).",
        low="well-conditioned non-unit base",
        endpoints=("eta just below/at 1", "bound just below/at/above ceiling"),
        high="n or condition driving gamma",
        extreme="diagonal compact case, identity, subnormal pivots, NaN/inf bound",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:factor-reduced:diagonal-certificate": _metadata(
        quantity="diagonal reduced determinant sign and componentwise relative formation error; require positive product sign, every error finite, and every error < 1.",
        threshold="exact positive-determinant domain plus strict derived relative-error radius.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="compute diagonal formation/log-roundoff bound",
        refused_outcome="Otherwise set reduced log-error bound to infinity and refuse reduced certificate",
        oracle="exact sign product plus high-precision componentwise (formation_error/abs(diagonal)) and -log1p(-error) sum.",
        axis_name="Boundary cells for diagonal reduced determinant sign and componentwise relative formation error; require positive product sign, every error finite, and every error < 1.",
        low="positive diagonal (1.3,2.4) with errors 0.1",
        endpoints=(
            "relative error nextafter(1, 0), 1, and nextafter(1, +inf)",
            "determinant sign +1, 0, and -1",
        ),
        high="mixed scales",
        extreme="zero diagonal, NaN/inf error, signed zeros, rank 0",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:factor-reduced:qr-certificate": _metadata(
        quantity="reduced determinant sign, formation/reconstruction reduced_eta, Q orthogonality orthogonality_eta, and nonzero R diagonal; sign positive, both finite etas in [0,1), and every R pivot nonzero.",
        threshold="exact determinant/pivot domain plus strict derived perturbation radii.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="compute matrix, orthogonality, and triangular-log bounds",
        refused_outcome="Otherwise assign infinite reduced bound and refuse reduced arithmetic",
        oracle="high-precision QR or SVD reconstruction, Q.TQ-I norm, exact R-pivot sign product, and mpmath log bounds.",
        axis_name="Boundary cells for reduced determinant sign, formation/reconstruction reduced_eta, Q orthogonality orthogonality_eta, and nonzero R diagonal; sign positive, both finite etas in [0,1), and every R pivot nonzero.",
        low="nonsingular non-unit 2x2 reduced matrix",
        endpoints=(
            "each eta just below/at 1",
            "R pivot smallest subnormal/zero, sign positive/zero/negative",
        ),
        high="ill-conditioned reduced matrix",
        extreme="rank 0/1, repeated singular values, NaN/inf, signed zero pivots",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:factor-reduced:acceptance-budget": _metadata(
        quantity="total_log_error_bound = projection + base + reduced; require finite total and total <= sqrt(eps(target_dtype)).",
        threshold="closed derived aggregate log-error ceiling.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="set overall factor certificate valid and expose executable determinant-lemma payload",
        refused_outcome="Refuse with combined-bound diagnostics",
        oracle="Decimal/mpmath sum of independently recomputed component bounds.",
        axis_name="Boundary cells for total_log_error_bound = projection + base + reduced; require finite total and total <= sqrt(eps(target_dtype)).",
        low="three nonzero components summing to half ceiling",
        endpoints=(
            "total nextafter(ceiling, -inf)",
            "exactly ceiling, nextafter(ceiling, +inf)",
        ),
        high="one dominant component",
        extreme="all zero, catastrophic cancellation is disallowed by nonnegative components, NaN/inf component",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:symmetry:tolerant-representative": _metadata(
        quantity="vector entries must all be >0; dense matrices must pass overflow-safe scaled symmetry difference <= scaled_atol + rtol*abs(scaled_transpose), yield finite eigenvalues, and have every eigenvalue >0.",
        threshold="configured tolerant symmetry plus exact strict-positive spectrum; API tolerance contract with derived overflow-safe normalization.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="report positive definite and enable SPD-dependent routes",
        refused_outcome="return false/refuse at caller on asymmetry, eigensolver failure, non-finite spectrum, or non-positive eigenvalue.",
        oracle="direct high-precision abs(a_ij-a_ji) <= atol+rtol*abs(a_ji) and analytic/MP eigenvalues; Cholesky/LDL inertia as neighbour.",
        axis_name="Boundary cells for vector entries must all be >0; dense matrices must pass overflow-safe scaled symmetry difference <= scaled_atol + rtol*abs(scaled_transpose), yield finite eigenvalues, and have every eigenvalue >0.",
        low="vector [1.3,2.4] and symmetric SPD 2x2",
        endpoints=(
            "one skew pair just below/at/above tolerance",
            "minimum eigenvalue positive subnormal nextafter(0,+inf), 0, negative",
        ),
        high="large-magnitude symmetric matrix",
        extreme="exact symmetry, signed zero, subnormal scale, NaN/inf, repeated eigenvalues",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:dense-condition:strict-dtype-ceiling": _metadata(
        quantity="condition of exact-power-of-two-normalized dense matrix; require finite condition < 1/eps(dtype).",
        threshold="strict derived recurrence/transform resolution ceiling.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="return (condition, ceiling, True) and allow dense/structured arithmetic",
        refused_outcome="Otherwise return unresolved and callers refuse their route",
        oracle="high-precision SVD ratio; power-of-two scaling must leave the condition unchanged.",
        axis_name="Boundary cells for condition of exact-power-of-two-normalized dense matrix; require finite condition < 1/eps(dtype).",
        low="condition near 2",
        endpoints=("nextafter(1/eps, 0)", "1/eps, above"),
        high="singular matrix",
        extreme="identity, subnormal/common power-of-two scaling, float32/float64, NaN/inf condition",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:lambda-logdet:subnormal-rescale": _metadata(
        quantity="maximum=max(abs(Lambda)) for dense Lambda; select exact power-of-two rescaling iff maximum < finfo(dtype).tiny.",
        threshold="strict dtype normal/subnormal boundary from finfo.tiny; derived numerical-safety selector.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="below boundary, rescale then Cholesky and subtract exact log-scale correction",
        refused_outcome="At/above boundary, factor Lambda directly. Failure of exact scaling or finite Cholesky refuses",
        oracle="float.hex, frexp/ldexp, and high-precision logdet; compare scaled and direct results where both are representable.",
        axis_name="Boundary cells for maximum=max(abs(Lambda)) for dense Lambda; select exact power-of-two rescaling iff maximum < finfo(dtype).tiny.",
        low="ordinary SPD scale 1.3",
        endpoints=("nextafter(tiny, 0)", "tiny, nextafter(tiny, +inf)"),
        high="normal/max-finite SPD",
        extreme="zero matrix (already non-SPD), smallest subnormal SPD diagonal, float32/float64",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:finite:newton-stability-rho": _metadata(
        quantity="measured spectral radius rho(Lambda^-1 P) for the generic finite e-polynomial route; stable = rho <= 1.0.",
        threshold="closed finite-polynomial stability policy at 1; derived from the chosen generic payload, not the strict trace-series certificate.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="execute finite e-polynomial when stable",
        refused_outcome="Otherwise raise and fall through to a stable exact rung",
        oracle="analytic eigenvalues for diagonal/2x2 Lambda^-1P or high-precision eigensolver.",
        axis_name="Boundary cells for measured spectral radius rho(Lambda^-1 P) for the generic finite e-polynomial route; stable = rho <= 1.0.",
        low="rho 0.4",
        endpoints=("nextafter(1, 0)", "exactly 1, nextafter(1, +inf)"),
        high="expansive rho 2.4",
        extreme="rho 0, repeated/complex eigenvalues, nonnormal matrix, NaN/inf measurement handled by spectral-radius gate",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:state-space:block-chain-exactness": _metadata(
        quantity="positive integer block_size dividing n and every block farther than the first off-diagonal exactly zero.",
        threshold="discrete divisibility plus exact sparsity pattern; exact structure/API contract (rtol/atol are intentionally ignored live).",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="proceed to block-LDL",
        refused_outcome="Refuse state_space_logdet as not a block chain",
        oracle="explicit block-index traversal and exact zero comparison; compare with a dense mask built from abs(row-column)<=1.",
        axis_name="Boundary cells for positive integer block_size dividing n and every block farther than the first off-diagonal exactly zero.",
        low="n=6, block_size=2 with non-unit tridiagonal blocks",
        endpoints=(
            "block_size 1, 0, divisor/non-divisor, forbidden entry exact 0",
            "smallest nonzero subnormal",
        ),
        high="many blocks",
        extreme="one block, empty/invalid size, signed zero, huge forbidden finite value",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:state-space:payload-domain": _metadata(
        quantity="block-chain matrix must be tolerantly symmetric, SPD, condition-resolved, every Schur update/pivot finite and SPD, and final pivot-logdet sum finite.",
        threshold="compound mathematical SPD domain plus derived finite/condition certificate.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="return block-LDL logdet including scale correction",
        refused_outcome="Refuse at the first failed symmetry, SPD, condition, Schur arithmetic/pivot, or final-sum premise",
        oracle="full high-precision dense slogdet/Cholesky and independently formed Schur complements; compare sum of pivot logdets with dense logdet.",
        axis_name="Boundary cells for block-chain matrix must be tolerantly symmetric, SPD, condition-resolved, every Schur update/pivot finite and SPD, and final pivot-logdet sum finite.",
        low="well-conditioned non-unit block tridiagonal SPD",
        endpoints=(
            "minimum Schur eigenvalue just positive/zero/negative",
            "condition just below/at 1/eps",
        ),
        high="many blocks/large scale",
        extreme="asymmetric link, singular pivot, overflowed update, NaN/inf, subnormal SPD pivots",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:structured:exact-shape-and-spectrum": _metadata(
        quantity="exact diagonal/circulant/Toeplitz layout, real strictly positive circulant FFT spectrum, or exact Kronecker reconstruction with matching shape.",
        threshold="exact structural evidence and strict positive spectrum; exact domain/API structure contract (live rtol/atol are ignored by structure helpers).",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="execute the requested exact structured logdet method",
        refused_outcome="Refuse the requested kind, nonpositive/complex spectrum, or mismatching Kronecker evidence",
        oracle="explicit index formulas for each structure, analytic DFT/MP FFT for circulant spectrum, and exact nested-loop Kronecker reconstruction; compare result with dense high-precision logdet.",
        axis_name="Boundary cells for exact diagonal/circulant/Toeplitz layout, real strictly positive circulant FFT spectrum, or exact Kronecker reconstruction with matching shape.",
        low="non-unit exact 2x2/3x3 structures",
        endpoints=(
            "exact layout",
            "one entry moved one ULP, smallest spectrum value positive-subnormal/zero/negative, matching versus one-row-wrong shape",
        ),
        high="multi-factor Kronecker",
        extreme="signed zero, repeated diagonals/eigenvalues, complex roundoff, empty/1x1 structure",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:spectral-radius:finite-measurement": _metadata(
        quantity="every entry of solved X=Lambda^-1P and every dense eigenvalue must be finite; solve/eigensolver must not signal supported arithmetic failure.",
        threshold="IEEE finite measurement domain; numerical-safety exact domain.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="return max(abs(X)) for diagonal X or max(abs(eigvals(X))) for dense",
        refused_outcome="Refuse with the resolved finite-measurement ValueError",
        oracle="analytic diagonal ratios/2x2 eigenvalues or high-precision solve and eigensolver.",
        axis_name="Boundary cells for every entry of solved X=Lambda^-1P and every dense eigenvalue must be finite; solve/eigensolver must not signal supported arithmetic failure.",
        low="finite non-unit X with rho 0.4",
        endpoints=("largest finite solve/eigenvalue", "overflow/non-finite"),
        high="ill-conditioned Lambda",
        extreme="zero perturbation, complex eigenpairs, defective matrix, NaN/inf, subnormal ratios",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:trace:actual-rho-strict": _metadata(
        quantity="independently measured actual_rho; require strict actual_rho < 1.",
        threshold="measured spectral radius rho < 1.0 (strict)",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="continue certificate checks and trace/frozen routing",
        refused_outcome="Refuse with measured-rho diagnostic",
        oracle="analytic/high-precision spectral radius of Lambda^-1P.",
        axis_name="Boundary cells for independently measured actual_rho; require strict actual_rho < 1.",
        low="0.4",
        endpoints=("nextafter(1, 0)", "1, nextafter(1, +inf)"),
        high="2.4",
        extreme="0, subnormal, repeated unit-modulus eigenvalues, non-finite handled upstream",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:trace:certificate-domain": _metadata(
        quantity="caller/warmup certificate; require 0 <= certificate < 1.",
        threshold="half-open API certificate domain [0,1).",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="continue to coverage comparison",
        refused_outcome="Refuse negative, unit, super-unit, or NaN certificate",
        oracle="direct Decimal interval check after explicit finite/NaN handling.",
        axis_name="Boundary cells for caller/warmup certificate; require 0 <= certificate < 1.",
        low="0.5",
        endpoints=("0, negative subnormal nextafter(0, -inf)", "nextafter(1, 0), 1"),
        high="2.4",
        extreme="NaN, infinities, smallest subnormal",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:trace:certificate-upper-bound": _metadata(
        quantity="certificate coverage margin certificate - actual_rho; require actual_rho <= certificate.",
        threshold="closed conservative-bound contract; API certificate obligation.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="return measured/certified pair",
        refused_outcome="Refuse an understating certificate",
        oracle="high-precision scalar comparison using independently measured rho.",
        axis_name="Boundary cells for certificate coverage margin certificate - actual_rho; require actual_rho <= certificate.",
        low="(actual=.4, cert=.6)",
        endpoints=(
            "certificate equals actual rho (admit)",
            "certificate nextafter(actual, 0) refuses; nextafter(actual, +inf) admits",
        ),
        high="wide conservative margin",
        extreme="both zero, both near 1, subnormal gap",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:trace:tail-domain-and-order": _metadata(
        quantity="rho in [0,1), integer order >=0, multiplicity >=1, finite tolerance >0, and the smallest order satisfying multiplicity*rho**(m+1)/((m+1)*(1-rho)) <= tolerance.",
        threshold="mathematical tail formula and discrete domain; derived convergence certificate, with tolerance as API input.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="compute scalar/whole-trace bound or return the first fixed order meeting tolerance",
        refused_outcome="Refuse invalid domains before looping",
        oracle="Decimal/mpmath evaluation of the closed formula plus integer linear search checked at m-1 and m.",
        axis_name="Boundary cells for rho in [0,1), integer order >=0, multiplicity >=1, finite tolerance >0, and smallest chosen order satisfying multiplicity*rho**(m+1)/((m+1)*(1-rho)) <= tolerance.",
        low="rho .4/order 3/multiplicity 2/tolerance .1",
        endpoints=(
            "rho 0",
            "just below/at 1, order 0/-1, multiplicity 1/0, bound equal tolerance and adjacent representable values",
        ),
        high="large multiplicity/tight tolerance",
        extreme="rho subnormal, tolerance subnormal/NaN/inf, large order without overflow in high precision",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "EAGER:trace:exact-power-trace-evidence": _metadata(
        quantity="supplied sequence covers every power through order, is finite, and equals the deterministically recomputed Tr(X**r) array exactly.",
        threshold="discrete coverage plus exact evidence equality; exact domain/API evidence contract.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="execute deterministic truncated trace-log",
        refused_outcome="Refuse missing, short, non-finite, or mismatching evidence",
        oracle="explicit high-precision matrix powers and traces (or exact diagonal powers) with exact length/value comparison.",
        axis_name="Boundary cells for supplied sequence covers every power through order, is finite, and equals the deterministically recomputed Tr(X**r) array exactly.",
        low="diagonal non-unit X with exact traces and order 2",
        endpoints=(
            "length exactly order",
            "order-1, exact value versus one ULP perturbation",
        ),
        high="larger order/matrix",
        extreme="order 0, negative order, empty sequence, NaN/inf trace, cancellation to signed zero",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "EAGER:frozen-probes:identity-width-order": _metadata(
        quantity="exact FrozenProbes runtime type, probe width equal to matrix dimension n, and integer order >=0.",
        threshold="capability/shape/order API contract with exact identity.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="execute deterministic frozen Hutchinson recurrence",
        refused_outcome="Refuse wrong probe object, wrong width, or negative order",
        oracle="exact type, tuple shape equality, and integer comparison; compare recurrence dimensions explicitly.",
        axis_name="Boundary cells for exact FrozenProbes runtime type, probe width equal to matrix dimension n, and integer order >=0.",
        low="non-unit immutable probes with width n and order 2",
        endpoints=("width n/n-1/n+1", "order 0/-1"),
        high="many probes/large n/order",
        extreme="subclass/wrapper rather than exact type, zero probes, zero-width, signed-zero values, huge finite probes",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:sigma:payload-symmetry": _metadata(
        quantity="Use original Sigma if compact 1-D, bitwise symmetric, or outside configured symmetric tolerance. Only tolerance-symmetric but nonexact dense input is replaced by the exact safe symmetric roundoff representative.",
        threshold="|a_ij-a_ji| <= atol + rtol*|a_ji|",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="Original payload",
        refused_outcome="representative; this is a payload-selection boundary, not an immediate refusal. The selected payload is what L2–L5 run",
        oracle="Independently evaluate both orientations of the elementwise tolerance and exact representative; compare L2/L3/L4/L5 with NumPy slogdet.",
        axis_name="Boundary cells for Use original Sigma if compact 1-D, bitwise symmetric, or outside configured symmetric tolerance. Only tolerance-symmetric but nonexact dense input is replaced by the exact safe symmetric roundoff representative.",
        low="exact symmetric non-unit",
        endpoints=(
            "rtol cell: asymmetry nextafter(T, 0), T, and nextafter(T, +inf) with atol=0",
            "atol cell: asymmetry nextafter(atol, 0), atol, and nextafter(atol, +inf) with rtol=0",
        ),
        high="clearly asymmetric",
        extreme="min-subnormal/near-max, both transpose orientations",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:sigma:finite-two-sum": _metadata(
        quantity="_two_sum_error(Lambda,P) returns finite sum and remainder.",
        threshold="IEEE representability",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Sigma facts and rungs may be evaluated",
        refused_outcome="every numerical rung false and dispatch ultimately refusing",
        oracle="Decimal/Fraction sum plus TwoSum reconstruction; direct L0–L7 and slogdet.",
        axis_name="Boundary cells for _two_sum_error(Lambda,P) returns finite sum and remainder.",
        low="ordinary non-unit finite",
        endpoints=("max-safe/nextafter-overflow", "cancellation boundaries"),
        high="overflowing addition",
        extreme="min-subnormal/cancellation/max, f32/f64; NaN/Inf non-finite cells",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:structure:compact-diagonal-positive": _metadata(
        quantity="Every compact Sigma entry is strictly positive.",
        threshold="Strict zero",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Compact diagonal structure evidence true",
        refused_outcome="structure false and fallthrough",
        oracle="Scalar sign and sum(log(sigma)); L3/L4 and slogdet.",
        axis_name="Boundary cells for Every compact Sigma entry is strictly positive.",
        low="positive compact diagonal entries at non-unit scale 1.3",
        endpoints=(
            "positive minimum subnormal entry",
            "zero or negative minimum-subnormal entry",
        ),
        high="ordinary negative entry",
        extreme="minimum subnormal/maximum finite entries; n=1 through 10000",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:structure:diagonal-tolerance": _metadata(
        quantity="Exact diagonal layout: every off-diagonal entry is bitwise zero.",
        threshold="Exact equality to zero; rtol and atol are ignored by the live helper.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="L3 diagonal candidate",
        refused_outcome="fallthrough or rejected explicit claim",
        oracle="Literal bitwise-zero off-diagonal mask; compare L3 with L4 and NumPy slogdet.",
        axis_name="Exact diagonal layout cells",
        low="exact diagonal with non-unit entries",
        endpoints=(
            "off-diagonal exact zero admits",
            "minimum-subnormal off-diagonal mismatch falls through",
        ),
        high="obvious dense violation",
        extreme="orientations, subnormal/max scale, condition near ceiling",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:structure:circulant-tolerance-spectrum": _metadata(
        quantity="Exact cyclic row shifts and a real, strictly positive FFT spectrum.",
        threshold="Exact row-shift equality plus every real FFT eigenvalue > 0.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="L3 circulant candidate",
        refused_outcome="noncirculant layout or nonpositive FFT spectrum rejects the request or falls through",
        oracle="Independently form cyclic rows and FFT eigenvalues; L3 versus L4 and the FFT log-eigenvalue sum.",
        axis_name="Exact circulant layout and spectrum cells",
        low="exact positive circulant",
        endpoints=(
            "exact cyclic shifts and positive-subnormal spectrum admit",
            "one-ULP shift mismatch or zero/negative spectrum falls through",
        ),
        high="noncirculant/nonpositive",
        extreme="complex roundoff, n/scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "LADDER:structure:toeplitz-tolerance": _metadata(
        quantity="Exact equality along every matrix diagonal.",
        threshold="Exact diagonal equality; rtol and atol are ignored by the live helper.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="L3 Toeplitz candidate",
        refused_outcome="non-Toeplitz diagonals reject an explicit claim or fall through auto-detection",
        oracle="Independently compare every diagonal; L3 versus L4 and slogdet.",
        axis_name="Exact Toeplitz diagonal-equality cells",
        low="exact non-unit SPD Toeplitz",
        endpoints=(
            "every diagonal is exactly constant",
            "one entry differs by one ULP or minimum subnormal",
        ),
        high="obvious violation",
        extreme="n low/high, subnormal/max, condition 1/eps±delta",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "LADDER:structure:kronecker-evidence": _metadata(
        quantity="Descriptor and factors present, every factor passes _is_positive_definite, reconstructed shape equals Sigma, and reconstruction is bitwise exact.",
        threshold="Exact identity and strict SPD zero",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="L3 Kronecker candidate",
        refused_outcome="missing descriptor, non-SPD factor, wrong shape, or inexact reconstruction falls through",
        oracle="Explicit reduce(np.kron), factor eigvalsh, and dense slogdet.",
        axis_name="Boundary cells for Descriptor and factors present, every factor SPD, reconstructed shape equals Sigma, and reconstruction is bitwise exact. The canonical registered root is the final shape/equality conjunction; the earlier checks are required live prerequisites of the same route.",
        low="exact non-unit factors",
        endpoints=("one-ULP reconstruction", "factor eigenvalue -delta,0,+delta"),
        high="shape mismatch",
        extreme="f16/longdouble normalization and extreme scales",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:sigma:symmetry-spd-condition": _metadata(
        quantity="Sigma is compact or configured-symmetric; the selected symmetric representative is SPD; dense condition is finite and strictly <1/eps(dtype).",
        threshold="Configured symmetry plus strict positivity and EAGER-derived dense ceiling",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Shared facts unlock L1–L5",
        refused_outcome="excluding SPD/condition-dependent rungs",
        oracle="Independent elementwise symmetry, eigvalsh, and singular-value condition; L2/L3/L4 versus slogdet.",
        axis_name="Boundary cells for Sigma is compact or configured-symmetric; the selected symmetric representative is SPD; dense condition is finite and strictly <1/eps(dtype).",
        low="well-conditioned SPD non-unit Sigma",
        endpoints=(
            "symmetry just within tolerance and positive minimum eigenvalue",
            "symmetry outside tolerance or minimum eigenvalue zero/negative",
        ),
        high="condition just below, at, and above 1/eps(dtype)",
        extreme="indefinite/unresolved Sigma; float32/float64 and scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:rank:evidence": _metadata(
        quantity="Without factors, the exact algebraic rank bound resolves. With factors, the EAGER projection certificate executes and is valid; otherwise rank becomes n.",
        threshold="Exact rank or borrowed factor error certificates",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Certified rank may unlock L1/L5",
        refused_outcome="factor routes excluded",
        oracle="SVD/exact structural rank plus independent reconstruction/Decimal certificate; L1/L5 versus slogdet.",
        axis_name="Boundary cells for Without factors, the exact algebraic rank bound resolves. With factors, the EAGER projection certificate executes and is valid; otherwise rank becomes n.",
        low="rank 0 or 1 with non-unit factors",
        endpoints=(
            "projection eta and log-error bounds just inside their ceilings",
            "one reconstruction or projection bound just outside its ceiling",
        ),
        high="rank 64, 128, and n",
        extreme="omitted factor direction, invalid certificate, layouts, gauges, subnormal/max",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:rho:measurement": _metadata(
        quantity="spectral_radius(Lambda,P) solves and returns finite eigenspectrum/rho.",
        threshold="Finite resolved arithmetic",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Rho-dependent L6/L7 facts available",
        refused_outcome="rho=inf/measurement invalid; exact rungs remain eligible",
        oracle="Independently form solve(Lambda,P) and eigvals.",
        axis_name="Boundary cells for spectral_radius(Lambda,P) solves and returns finite eigenspectrum/rho.",
        low="rho .5 non-unit",
        endpoints=("finite", "overflow boundary"),
        high="unresolved solve",
        extreme="nonnormal X, subnormal Lambda, f32/f64; NaN/Inf non-finite cells",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:finite:payload-rho": _metadata(
        quantity="The generic L5 payload symmetric_sigma-Lambda needs a resolved spectral radius rho <= 1; determinant_lemma_payload is an independent L5 alternative.",
        threshold="finite payload spectral radius rho <= 1.0 (closed endpoint; no certificate)",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Keep the generic finite-polynomial payload eligible for L5",
        refused_outcome="Exclude only the generic payload; L5 may still execute through determinant_lemma_payload",
        oracle="Measure eigvals of the generic Lambda^-1 P payload independently; compare generic L5, determinant-lemma L5, and dense slogdet",
        axis_name="Generic finite-polynomial payload stability, independent of the determinant-lemma alternative",
        low="generic rho .5 with determinant payload absent",
        endpoints=(
            "generic rho nextafter(1, 0) and exactly 1 remain eligible",
            "generic rho nextafter(1, +inf) or unresolved is excluded while a valid determinant_lemma_payload still permits L5",
        ),
        high="generic rho > 1 with determinant_lemma_payload independently true or false",
        extreme="unresolved generic solve, nonnormal payload, determinant route true/false, n/rank/scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "LADDER:determinant-lemma:payload": _metadata(
        quantity="Factors supplied AND rank evidence valid AND Sigma formation finite AND Sigma bitwise symmetric AND dense condition resolved.",
        threshold="Exact/domain conjunction with borrowed EAGER factor/condition certificates",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Factor determinant lemma may power L1/L5 independently of rho",
        refused_outcome="factors cannot be promised to the dispatched payload",
        oracle="Independently reconstruct factors, use slogdet, condition/SVD; compare L1/L5 factor paths.",
        axis_name="Boundary cells for Factors supplied AND rank evidence valid AND Sigma formation finite AND Sigma bitwise symmetric AND dense condition resolved.",
        low="all true non-unit",
        endpoints=("independently flip every conjunct", "condition 1/eps±delta"),
        high="multiple false",
        extreme="layouts/gauges/subnormal/max",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:rung0:base": _metadata(
        quantity="Finite Sigma, exact array equality Sigma == Lambda, and compact or resolved dense arithmetic",
        threshold="Exact zero perturbation and finite domain",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="First satisfied rung executes lambda_logdet",
        refused_outcome="continuing to L1+",
        oracle="lambda_logdet and dense slogdet.",
        axis_name="Boundary cells for Finite Sigma, exact array equality Sigma == Lambda, and compact or resolved dense arithmetic",
        low="finite non-unit Sigma equal to Lambda",
        endpoints=("P is exactly zero", "one P entry is the minimum subnormal"),
        high="ordinary nonzero perturbation",
        extreme="unresolved dense condition and dimension/scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:rung1:low-rank-size": _metadata(
        quantity="Rank evidence valid AND (compact diagonal OR executable determinant lemma) AND Sigma SPD AND rank <= config.low_rank_max AND rank <= config.low_rank_fraction * n.",
        threshold="Inclusive configured rank and fraction limits; rank evidence, payload availability, and Sigma SPD are borrowed facts",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="First qualifying L1 executes low_rank_logdet",
        refused_outcome="continuing L2+",
        oracle="L1 versus L2/L5 and dense slogdet.",
        axis_name="Boundary cells for Rank evidence valid AND (compact diagonal OR executable determinant lemma) AND Sigma SPD AND rank <=low_rank_max AND rank <=low_rank_fraction*n.",
        low="small certified rank",
        endpoints=("rank T-1 and T admit", "rank T+1 falls through"),
        high="rank 64, 128, and n",
        extreme="n=1, 8, 257, 10000; fraction 0/1; factor and compact payloads",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "LADDER:rung2:chain": _metadata(
        quantity="Finite dense 2-D Sigma, block size supplied, exact block-chain structure, Sigma SPD, and condition resolved.",
        threshold="Exact far-block zero/block-size contract plus borrowed SPD and strict 1/eps",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="L2 executes state_space_logdet",
        refused_outcome="continuing L3+",
        oracle="Explicit block-tridiagonal reconstruction; L2 versus L4 and slogdet.",
        axis_name="Boundary cells for Finite dense 2-D Sigma, block size supplied, exact block-chain structure, Sigma SPD, and condition resolved.",
        low="valid non-unit SPD block chain",
        endpoints=(
            "block_size 1 or an exact divisor, far block zero, and positive pivots",
            "block_size 0/non-divisor, far block minsubnormal, or zero/negative pivot",
        ),
        high="large valid block chain near the condition ceiling",
        extreme="pivot/eigenvalue/condition extremes and overflowed Schur update",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:rung3:structured": _metadata(
        quantity="Supported structure request verified AND Sigma SPD AND (diagonal OR condition resolved).",
        threshold="Delegated structure/SPD/condition gates",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="L3 executes the kind-specific structured_logdet",
        refused_outcome="continuing L4+",
        oracle="Each structure's independent oracle versus L4 and slogdet.",
        axis_name="Boundary cells for Supported structure request verified AND Sigma SPD AND (diagonal OR condition resolved).",
        low="one valid fixture per kind",
        endpoints=("each subordinate T-delta", "each subordinate T+delta"),
        high="invalid claim/SPD/condition",
        extreme="diagonal/circulant/Toeplitz/Kronecker and scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "LADDER:rung4:dense": _metadata(
        quantity="n <= config.dense_max_n, condition_resolved, and sigma_spd.",
        threshold="Inclusive configured n limit; condition resolution and Sigma SPD are borrowed facts",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="L4 dispatches dense_cholesky_logdet",
        refused_outcome="falls through to L5 finite e-polynomial",
        oracle="NumPy slogdet with independent eigvalsh/condition checks",
        axis_name="Boundary cells for n <= config.dense_max_n, condition_resolved, and sigma_spd.",
        low="small well-conditioned non-unit SPD Sigma",
        endpoints=(
            "n T-1 and T with positive eigenvalues and condition below 1/eps",
            "n T+1, zero/negative eigenvalue, or condition at/above 1/eps",
        ),
        high="too-large or condition-unresolved dense Sigma",
        extreme="float32/float64 and subnormal/maximum scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "LADDER:rung5:finite-size": _metadata(
        quantity="n<=finite_max_n OR ((compact OR determinant payload) AND rank <=finite_max_rank).",
        threshold="Inclusive configured n/rank policy",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="Size permits L5 evaluation",
        refused_outcome="continuing L6+ without a hidden dense fallback",
        oracle="L5 versus L6 and slogdet; inspect the actual selected payload route.",
        axis_name="Boundary cells for n<=finite_max_n OR ((compact OR determinant payload) AND rank <=finite_max_rank).",
        low="small dense/compact",
        endpoints=("n/rank each T-1", "T, T+1"),
        high="both limits exceeded",
        extreme="n/rank 1 through 10,000 and dense/compact/factor payloads",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "LADDER:rung5:finite-executable": _metadata(
        quantity="Finite-size qualified AND (generic payload rho stable OR determinant payload) AND Sigma SPD AND (determinant payload OR dense arithmetic resolved).",
        threshold="Compound borrowed exact/stability facts",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="L5 executes finite_perturbation_logdet with the exact generic payload or original factor payload",
        refused_outcome="continuing L6+",
        oracle="Both L5 paths versus L4/L6, slogdet, and Decimal.",
        axis_name="Boundary cells for Finite-size qualified AND (generic payload rho stable OR determinant payload) AND Sigma SPD AND (determinant payload OR dense arithmetic resolved).",
        low="all true",
        endpoints=(
            "independently flip size",
            "stability/SPD/determinant/dense resolution",
        ),
        high="all false",
        extreme="rho/condition/rank/n/scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "LADDER:rung6:trace": _metadata(
        quantity="Finite Sigma AND rho measurement valid AND traces/order supplied AND exact trace match AND actual rho <1 AND actual rho <=certificate AND certificate 0<=rho<1.",
        threshold="Analytic strict convergence/coverage plus exact evidence",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="L6 executes truncated_trace_logdet",
        refused_outcome="continuing L7",
        oracle="Independently recomputed dense traces, analytic tail, and slogdet; compare neighboring orders and L5.",
        axis_name="Boundary cells for Finite Sigma AND rho measurement valid AND traces/order supplied AND exact trace match AND actual rho <1 AND actual rho <=certificate AND certificate 0<=rho<1.",
        low="rho .5/exact traces",
        endpoints=(
            "rho 0, .99",
            "1±ULP, certificate measured±ULP, order/trace exact±ULP",
        ),
        high="missing/invalid evidence",
        extreme="n=1 through 10,000",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "LADDER:rung7:frozen": _metadata(
        quantity="Finite Sigma AND exact FrozenProbes present with width n AND order supplied and >=0 AND actual rho <1 and covered by certificate in [0,1).",
        threshold="Analytic convergence plus exact probe/order contract",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="L7 executes frozen_hutchinson_trace_logdet",
        refused_outcome="no deterministic rung and resampling refusal",
        oracle="Fixed-probe dense matrix powers/trace correction and slogdet; compare L6 where exact traces exist.",
        axis_name="Boundary cells for Finite Sigma AND exact FrozenProbes present with width n AND order supplied and >=0 AND actual rho <1 and covered by certificate in [0,1).",
        low="valid probes/order/rho",
        endpoints=("width n-1/n/n+1", "order -1/0/1, rho/certificate 1±ULP"),
        high="missing/wrong probes",
        extreme="probes 1/n/10000 and extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:multiplicity:index-and-gamma-domain": _metadata(
        quantity="Value is indexable as an integer, is not bool, and normalized multiplicity satisfies 1 <= m < 1/eps64.",
        threshold="Integer-index contract plus analytic gamma denominator domain",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Return canonical int(m)",
        refused_outcome="TypeError for bool/non-index or ValueError for nonpositive/too-large multiplicity",
        oracle="operator.index and exact check m*eps64<1; direct normalizer, RhoCertificate, and warmup factory.",
        axis_name="Boundary cells for Value is indexable as an integer, is not bool, and normalized multiplicity satisfies 1 <= m < 1/eps64.",
        low="m=1",
        endpoints=("-1,0,1", "limit-1,limit"),
        high="limit+1",
        extreme="bool, float, 10**1000",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:certificate:rho-domain-and-coverage": _metadata(
        quantity="Measured and certified rho each lie in [0,1), and certified rho is >= measured_max.",
        threshold="Analytic convergence and coverage",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Valid immutable certificate",
        refused_outcome="constructor ValueError naming the failed domain/coverage condition",
        oracle="Direct scalar predicates and independent nextafter comparisons; direct RhoCertificate.",
        axis_name="Boundary cells for Measured and certified rho each lie in [0,1), and certified rho is >= measured_max.",
        low="measured .4/certified .5",
        endpoints=(
            "domain: negative subnormal, 0, nextafter(1, 0), and 1",
            "coverage: nextafter(measured, 0), equality, and nextafter(measured, +inf)",
        ),
        high="uncovered or >=1",
        extreme="NaN/Inf",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:certificate:error-budget-domain": _metadata(
        quantity="Literal constructor predicate is margin < 0 OR tolerance <= 0 OR NOT(0 < tail_tolerance < tolerance). Do not claim direct-constructor margin finiteness: NaN margin is not rejected by this predicate; warmup rejects it.",
        threshold="Error-budget API contract",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="Certificate continues",
        refused_outcome="constructor ValueError",
        oracle="Direct scalar predicate and analytic tail-budget split; construct RhoCertificate directly.",
        axis_name="Boundary cells for Literal constructor predicate is margin < 0 OR tolerance <= 0 OR NOT(0 < tail_tolerance < tolerance). Do not claim direct-constructor margin finiteness: NaN margin is not rejected by this predicate; warmup rejects it.",
        low="positive margin/tolerance and interior tail",
        endpoints=("margin -ULP, 0, +ULP", "tolerance 0, +ULP, tail 0, tolerance"),
        high="negative/outer tail",
        extreme="NaN/Inf each field",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:certificate:optional-scale-domain": _metadata(
        quantity="Each supplied optional max_abs_lambda_logdet and max_x_operator_norm is finite and nonnegative; None is allowed.",
        threshold="Finite/nonnegative domain",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Store optional bound",
        refused_outcome="constructor ValueError for the named bound",
        oracle="np.isfinite plus scalar comparison; direct certificate.",
        axis_name="Boundary cells for Each supplied optional max_abs_lambda_logdet and max_x_operator_norm is finite and nonnegative; None is allowed.",
        low="None or 1.3",
        endpoints=("-ULP", "0, +ULP"),
        high="ordinary negative",
        extreme="minsubnormal/max/NaN/Inf",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:certificate:order-is-derived": _metadata(
        quantity="order == choose_trace_order(certified_rho,tail_tolerance,multiplicity).",
        threshold="Analytic tail-selected integer order",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Store certificate",
        refused_outcome="wrong-order constructor ValueError",
        oracle="Independently evaluate the whole-trace tail bound/search; direct certificate and L6 at neighboring orders.",
        axis_name="Boundary cells for order == choose_trace_order(certified_rho,tail_tolerance,multiplicity).",
        low="selected m",
        endpoints=("m-1", "m, m+1"),
        high="distant wrong order",
        extreme="rho 0/near1, tolerance/multiplicity extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:warmup:rho-inputs-and-margin": _metadata(
        quantity="Rho measurements are nonempty, finite, nonnegative; rho safety margin is finite and nonnegative.",
        threshold="Finite/nonnegative input domain",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Compute measured max/raw bound",
        refused_outcome="warmup raises ValueError for empty, non-finite, or negative rho values or margin",
        oracle="Tuple conversion, max, and scalar finite/sign predicates; direct certify_warmup_rho.",
        axis_name="Boundary cells for Rho measurements are nonempty, finite, nonnegative; rho safety margin is finite and nonnegative.",
        low="values 0/.5 and margin .01",
        endpoints=("empty, value/margin -ULP", "0, +ULP"),
        high="negative",
        extreme="NaN/Inf, long list, nextafter(1,0)",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:warmup:tail-fraction": _metadata(
        quantity="tail_fraction is finite and strictly 0 < f < 1.",
        threshold="User budget-split contract",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="Derive tail tolerance/order",
        refused_outcome="warmup raises ValueError unless finite tail_fraction lies strictly between zero and one",
        oracle="Literal scalar predicate and analytic tail; warmup factory.",
        axis_name="Boundary cells for tail_fraction is finite and strictly 0 < f < 1.",
        low="f=.5",
        endpoints=("0±ULP", "1±ULP"),
        high="outside interval",
        extreme="NaN/Inf",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:warmup:lambda-scale-inputs": _metadata(
        quantity="If supplied, lambda logdet measurements are nonempty and finite; lambda_logdet_margin is finite and nonnegative. Negative measurements remain valid because their absolute maximum is certified.",
        threshold="Finite/nonempty measurements and margin >= 0",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Produce outward max(abs(value))+margin bound",
        refused_outcome="warmup raises ValueError for empty/non-finite measurements or a negative/non-finite lambda margin",
        oracle="Exact abs/max/nextafter; factory then retained lambda audit.",
        axis_name="Warmup lambda measurements and margin cells",
        low="mixed finite ± non-unit",
        endpoints=(
            "empty or maximum finite measurement",
            "margin negative subnormal, 0, and positive subnormal",
        ),
        high="large negative/positive",
        extreme="minsubnormal/max/NaN/Inf and n=1/10000",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "PLAN:warmup:x-norm-inputs": _metadata(
        quantity="If supplied, X absolute-action norms are nonempty, finite, and nonnegative; x_operator_norm_margin is finite and nonnegative.",
        threshold="Finite/nonnegative measurements and margin >= 0",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Produce outward max-plus-margin bound",
        refused_outcome="warmup raises ValueError for empty/non-finite/negative norms or a negative/non-finite X-norm margin",
        oracle="Scalar max/nextafter; factory then retained X audit and frozen plan.",
        axis_name="Warmup X-norm measurements and margin cells",
        low="norms 0/1.3",
        endpoints=(
            "empty or maximum finite norm",
            "margin negative subnormal, 0, and positive subnormal",
        ),
        high="ordinary negative",
        extreme="minsubnormal, max, NaN/Inf, nonnormal actions",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:warmup:rho-roundoff-ceiling": _metadata(
        quantity="Binary64 evaluates raw_bound = float(measured_max + float(margin)); arithmetic_envelope = abs(raw_bound) * gamma_m; raw_certified = float(raw_bound + arithmetic_envelope); certified = nextafter(raw_certified, +inf); require certified < 1.",
        threshold="Strict convergence after the rounded binary64 bound, envelope, sum, and outward-nextafter stages",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Build RhoCertificate and choose its order from the outward certified rho",
        refused_outcome="Raise the warmup no-headroom ValueError when certified >= 1",
        oracle="Replay each binary64 addition and multiplication with float rounding, then apply math.nextafter; use Decimal only to check the enclosure",
        axis_name="Boundary cells for each rounded raw-bound, envelope, raw-certified, and outward-certified stage",
        low="measured rho .4, margin .01, and modest multiplicity",
        endpoints=(
            "raw_bound, arithmetic_envelope, and raw_certified at adjacent binary64 rounding cells",
            "certified nextafter(1, 0), exactly 1, and nextafter(1, +inf)",
        ),
        high="certified >= 1 after the outward step",
        extreme="multiplicity 1/limit-1, zero measurement, margin and near-one extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:audit:retained-rho": _metadata(
        quantity="Retained values are nonempty, finite, nonnegative; violation exactly when value > certificate.certified_rho.",
        threshold="Borrowed certificate bound, strict comparison",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Passed report with measured max",
        refused_outcome="failed report containing violating indices; malformed list raises ValueError",
        oracle="Literal enumerate/comparison and max; direct retained-rho audit.",
        axis_name="Boundary cells for Retained values are nonempty, finite, nonnegative; violation exactly when value > certificate.certified_rho.",
        low="all below bound",
        endpoints=("T-ULP", "T, T+ULP"),
        high="several violations",
        extreme="empty, negative, NaN/Inf, zero envelope",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:audit:retained-lambda-scale": _metadata(
        quantity="Certificate bound is present; retained logdets are nonempty/finite; violation when abs(value) > max_abs_lambda_logdet.",
        threshold="Borrowed scale certificate",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Return a passed retained-lambda audit report with the measured absolute maximum",
        refused_outcome="failed indices; missing bound or malformed list raises ValueError",
        oracle="Exact abs/max comparison and independent slogdet; direct audit.",
        axis_name="Boundary cells for Certificate bound is present; retained logdets are nonempty/finite; violation when abs(value) > max_abs_lambda_logdet.",
        low="finite values below",
        endpoints=("abs(value)=T-ULP", "T, T+ULP"),
        high="multiple violations",
        extreme="missing/empty, ±max, NaN/Inf, n=1/10000",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:audit:retained-x-norm": _metadata(
        quantity="X-norm certificate is present; retained ||abs(X)||_2 values are nonempty, finite, nonnegative; violation when value exceeds bound.",
        threshold="Borrowed X certificate",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Return a passed retained-X-norm audit report with the measured maximum",
        refused_outcome="failed indices; missing/malformed input ValueError",
        oracle="Independent solve then norm(abs(X),2) and literal comparison.",
        axis_name="Boundary cells for X-norm certificate is present; retained ||abs(X)||_2 values are nonempty, finite, nonnegative; violation when value exceeds bound.",
        low="below-bound norms",
        endpoints=("T-ULP", "T, T+ULP"),
        high="above-bound",
        extreme="missing/empty, negative/NaN/Inf, nonnormal and scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:audit:retained-trace-evidence": _metadata(
        quantity="Retained problems nonempty; each order equals certificate order, rank bound <= multiplicity, traces present, and exact arithmetic matches through the selected order. Helper arithmetic failure counts as mismatch.",
        threshold="Borrowed certificate order/rank plus exact evidence",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Return a passed retained-trace report after exact order, rank, and trace checks",
        refused_outcome="failed problem indices; empty input ValueError",
        oracle="Independently recompute dense matrix powers/traces and algebraic rank.",
        axis_name="Boundary cells for Retained problems nonempty; each order equals certificate order, rank bound <= multiplicity, traces present, and exact arithmetic matches through the selected order. Helper arithmetic failure counts as mismatch.",
        low="valid retained problems",
        endpoints=("order/rank T-1, T", "T+1, trace exact/±ULP"),
        high="missing/mismatched",
        extreme="empty, overflowed X, n/order/rho/scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:measurement:x-norm-finite": _metadata(
        quantity="Formed X=Lambda^-1 P and measured ||abs(X)||_2 (or compact max) are finite/resolved.",
        threshold="IEEE finite arithmetic",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Return actual norm",
        refused_outcome="raise normalized ValueError when solve or ||abs(X)||_2 is unresolved or non-finite",
        oracle="Independent solve and norm; direct helper and frozen factory.",
        axis_name="Boundary cells for Formed X=Lambda^-1 P and measured ||abs(X)||_2 (or compact max) are finite/resolved.",
        low="well-conditioned finite X",
        endpoints=("last finite", "first overflow"),
        high="unresolved solve",
        extreme="subnormal Lambda, nonnormal X, float max and f32/f64; NaN/Inf non-finite cells",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:measurement:lambda-logdet-finite": _metadata(
        quantity="abs(lambda_logdet(Lambda)) executes and is finite.",
        threshold="IEEE finite arithmetic and SPD logdet domain",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Return actual base scale",
        refused_outcome="raise normalized ValueError when Lambda logdet is unresolved or non-finite",
        oracle="Independent NumPy slogdet; direct helper and both factories.",
        axis_name="Boundary cells for abs(lambda_logdet(Lambda)) executes and is finite.",
        low="ordinary SPD non-unit",
        endpoints=("finite", "overflow log scale boundary"),
        high="unresolved/invalid",
        extreme="subnormal/max/cancellation and dimension extremes; NaN/Inf non-finite cells",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "PLAN:factory-certificate:order-and-rank": _metadata(
        quantity="Problem trace order exactly equals certificate order AND certificate multiplicity is at least the problem's algebraic rank bound.",
        threshold="Borrowed selected order and exact rank coverage",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Both trace and frozen factories continue after exact order and rank coverage checks",
        refused_outcome="order/rank ValueError",
        oracle="Independent algebraic rank and analytic tail order; both factories, direct L6 evidence.",
        axis_name="Boundary cells for Problem trace order exactly equals certificate order AND certificate multiplicity is at least the problem's algebraic rank bound.",
        low="matching order/covered rank",
        endpoints=("order/rank T-1", "T, T+1"),
        high="both wrong",
        extreme="rank 1…n/10000, factor layouts, order/rho extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:factory-certificate:lambda-scale": _metadata(
        quantity="Base-scale certificate exists and independently measured abs(logdet(Lambda)) <= bound exactly.",
        threshold="Borrowed warmup scale bound",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Factory accepts the independently measured Lambda logdet under its certificate bound",
        refused_outcome="missing/understated ValueError",
        oracle="NumPy slogdet; both factories and retained lambda audit.",
        axis_name="Boundary cells for Base-scale certificate exists and independently measured abs(logdet(Lambda)) <= bound exactly.",
        low="outward bound above actual",
        endpoints=("bound T-ULP", "T, T+ULP"),
        high="understated",
        extreme="missing, n/condition/minsubnormal/max scale",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:factory-certificate:x-norm": _metadata(
        quantity="When an X certificate is supplied, independently measured ||abs(X)||_2 <= bound exactly.",
        threshold="Borrowed warmup X bound",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Validation continues",
        refused_outcome="understated-bound ValueError; absent bound is allowed for trace plans but rejected later for frozen plans",
        oracle="Independent solve/norm; frozen factory and retained X audit.",
        axis_name="Boundary cells for When an X certificate is supplied, independently measured ||abs(X)||_2 <= bound exactly.",
        low="actual below bound",
        endpoints=("bound T-ULP", "T, T+ULP"),
        high="understated",
        extreme="None, nonnormal/rho/dimension/scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:factory-certificate:strict-rho": _metadata(
        quantity="Delegated eager validation: actual rho <1, certificate in [0,1), and actual rho <= certificate.",
        threshold="Borrowed EAGER analytic/domain/coverage gates",
        provenance=ThresholdProvenance.BORROWED,
        admitted_outcome="Factory validation continues",
        refused_outcome="eager ValueError",
        oracle="Independent solve/eigvals; both factories and direct L6/L7 versus slogdet.",
        axis_name="Boundary cells for Delegated eager validation: actual rho <1, certificate in [0,1), and actual rho <= certificate.",
        low="rho .5/covering cert",
        endpoints=("actual/cert -ULP,0,1±ULP", "coverage T±ULP"),
        high="uncovered/nonconvergent",
        extreme="nonnormal and dtype/scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:canonical-probes:runtime-finite": _metadata(
        quantity="Conversion of immutable probes to canonical JAX runtime dtype succeeds and every converted value is finite; underflow to captured zero is allowed, overflow is not.",
        threshold="Runtime dtype representability",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Return canonical immutable probes",
        refused_outcome="conversion ValueError",
        oracle="Literal NumPy dtype cast and byte/value comparison; frozen factory and runtime plan.",
        axis_name="Boundary cells for Conversion of immutable probes to canonical JAX runtime dtype succeeds and every converted value is finite; underflow to captured zero is allowed, overflow is not.",
        low="ordinary non-unit probes",
        endpoints=("largest representable/next value", "minsubnormal/zero underflow"),
        high="overflow",
        extreme="integer inputs, f32/f64, x64 on/off; NaN/Inf non-finite cells",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "PLAN:outward-arithmetic:positive-underflow": _metadata(
        quantity="_outward_nonnegative returns exact zero or nonfinite input unchanged; every nonzero finite proof quantity is rounded one representable step toward +infinity. Product/quotient helpers replace positive underflow-to-zero by the minimum positive float.",
        threshold="Directed-rounding proof rule",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Preserve zero/nonfinite branch",
        refused_outcome="outward-round positive finite branch; there is no immediate refusal",
        oracle="Decimal/Fraction exact scalar arithmetic plus math.nextafter.",
        axis_name="Boundary cells for _outward_nonnegative returns exact zero or nonfinite input unchanged; every nonzero finite proof quantity is rounded one representable step toward +infinity. Product/quotient helpers replace positive underflow-to-zero by the minimum positive float.",
        low="ordinary .5",
        endpoints=("zero/minsubnormal", "exact representable neighbors"),
        high="large finite",
        extreme="positive underflow, max, Inf/NaN",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:frozen:probe-energy-range": _metadata(
        quantity="Each abs(v)<=sqrt(finfo(runtime).max); every energy, total energy, and outward total is finite and <=finfo.max.",
        threshold="Derived dtype range for sum of squares",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Return total energy/maximum norm",
        refused_outcome="normalized range ValueError",
        oracle="Decimal sum of squares and dtype maximum; frozen factory.",
        axis_name="Boundary cells for Each abs(v)<=sqrt(finfo(runtime).max); every energy, total energy, and outward total is finite and <=finfo.max.",
        low="small non-unit probes",
        endpoints=("component sqrt(max)±ULP", "total max±ULP"),
        high="overflowing energy",
        extreme="zero/minsubnormal, many probes, f32/f64/max",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "PLAN:runtime-range:product": _metadata(
        quantity="Zero short-circuit; otherwise operands finite and left <= maximum/right; outward result must also be finite and <=maximum.",
        threshold="Derived overflow-safe product bound",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Return the finite outward product upper bound",
        refused_outcome="raise ValueError naming the product that exceeds runtime range",
        oracle="Decimal product against dtype max; direct helper and frozen factory.",
        axis_name="Boundary cells for Zero short-circuit; otherwise operands finite and left <= maximum/right; outward result must also be finite and <=maximum.",
        low="small finite product",
        endpoints=(
            "left equals maximum/right; outward nextafter may become +inf and is refused",
            "left one ULP below admits; one ULP above is refused before multiplication",
        ),
        high="overflowing product",
        extreme="zero, underflow, max, NaN/Inf, float32/float64",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:runtime-range:sum": _metadata(
        quantity="Operands finite, right<=maximum, left<=maximum-right; outward result finite and <=maximum.",
        threshold="Derived overflow-safe sum bound",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Return the finite outward sum upper bound",
        refused_outcome="raise ValueError naming the sum that exceeds runtime range",
        oracle="Decimal sum against dtype max; direct helper and both factories.",
        axis_name="Boundary cells for Operands finite, right<=maximum, left<=maximum-right; outward result finite and <=maximum.",
        low="small sum",
        endpoints=(
            "left equals maximum-right; outward nextafter may become +inf and is refused",
            "left one ULP below admits; one ULP above is refused before addition",
        ),
        high="overflow",
        extreme="zero/minsubnormal/max, NaN/Inf, f32/f64",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:gamma:operation-count-domain": _metadata(
        quantity="operation_count*eps <1; at or above one, gamma is infinity.",
        threshold="Standard gamma_n=n*eps/(1-n*eps) denominator",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Finite outward gamma",
        refused_outcome="infinity, which later causes range/error certification refusal",
        oracle="Decimal/Fraction gamma; helper and both factories.",
        axis_name="Boundary cells for operation_count*eps <1; at or above one, gamma is infinity.",
        low="small count",
        endpoints=("floor(1/eps)-1", "equality/first invalid"),
        high="above limit",
        extreme="order/probe/dimension extremes and f32/f64",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:frozen:x-bound-runtime-range": _metadata(
        quantity="Certified max_x_operator_norm <= finfo(runtime_dtype).max.",
        threshold="Runtime dtype range",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Continue frozen intermediate proof",
        refused_outcome="X-bound range ValueError",
        oracle="Literal dtype maximum; frozen factory.",
        axis_name="Boundary cells for Certified max_x_operator_norm <= finfo(runtime_dtype).max.",
        low="small bound",
        endpoints=("max-ULP", "max, max+ULP"),
        high="above range",
        extreme="zero, minsubnormal, Inf/NaN and f32/f64",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:frozen:intermediate-runtime-range": _metadata(
        quantity="The canonical registered root is the final correction_bound * addition_factor range proof. Earlier image, probe-image, dot, and reduction products are separate consumers of PLAN:runtime-range:product and must not be copied into this quantity.",
        threshold="correction_bound <= maximum/addition_factor",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Return certified final correction accumulation",
        refused_outcome="a ValueError naming that accumulation",
        oracle="Decimal multiplication against dtype maximum and actual fixed-probe JAX output; frozen factory.",
        axis_name="Boundary cells for The canonical registered root is the final correction_bound * addition_factor range proof. Earlier image, probe-image, dot, and reduction products are separate consumers of PLAN:runtime-range:product and must not be copied into this quantity.",
        low="non-unit correction/factor safely below max",
        endpoints=("quotient T-ULP", "T, T+ULP"),
        high="overflow",
        extreme="order 0/1/2/16/high, n/probes 1…10000, X/probe extremes",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "PLAN:runtime:sigma-finite-and-positive": _metadata(
        quantity="TwoSum Lambda+P has already remained finite; compact Sigma entries are all >0, or dense slogdet sign is >0 and its expected value computes.",
        threshold="compact Sigma entries > 0; dense slogdet sign > 0 after finite TwoSum formation",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Compute expected logdet",
        refused_outcome="normalized positive-definiteness or arithmetic ValueError",
        oracle="Decimal addition plus NumPy slogdet/eigvalsh; both factories.",
        axis_name="Boundary cells for TwoSum Lambda+P has already remained finite; compact Sigma entries are all >0, or dense slogdet sign is >0 and its expected value computes.",
        low="finite non-unit compact or dense SPD Sigma",
        endpoints=(
            "compact entry or dense minimum eigenvalue positive subnormal",
            "compact entry zero/negative or dense slogdet sign zero/negative",
        ),
        high="addition overflow or negative dense determinant sign",
        extreme="minimum subnormal/maximum scale, compact/dense, float32/float64",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "PLAN:runtime:expected-and-ulp-finite": _metadata(
        quantity="Expected logdet finite AND cast rounded=abs(expected) finite AND ulp=abs(spacing(rounded)) finite AND ulp<=certificate.tolerance.",
        threshold="Runtime representability and ULP budget",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Continue precision proof",
        refused_outcome="normalized expected/cast/spacing finite ValueError or explicit ULP-budget ValueError",
        oracle="Independent slogdet, literal dtype cast, and NumPy spacing; both factories.",
        axis_name="Boundary cells for Expected logdet finite AND cast rounded=abs(expected) finite AND ulp=abs(spacing(rounded)) finite AND ulp<=certificate.tolerance.",
        low="expected ordinary and ulp below tolerance",
        endpoints=("expected/cast finite boundary", "tolerance ulp-ULP,ulp,ulp+ULP"),
        high="ulp above budget",
        extreme="expected zero/subnormal/large, NaN/Inf, f32/f64",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:runtime:base-scale-range": _metadata(
        quantity="Certified base scale <=finfo(runtime_dtype).max.",
        threshold="Runtime dtype range",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Continue range proof",
        refused_outcome="base-scale range ValueError",
        oracle="Literal dtype maximum and independent slogdet; both factories.",
        axis_name="Boundary cells for Certified base scale <=finfo(runtime_dtype).max.",
        low="small base scale",
        endpoints=("max-ULP", "max, max+ULP"),
        high="above range",
        extreme="zero, minsubnormal, Inf/NaN, f32/f64 and extreme Lambda scales",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:runtime:frozen-prerequisites-and-series": _metadata(
        quantity="The canonical numerical root is that the outward frozen probe_energy * sum(x_bound**p/p) series scale is finite. A frozen plan also requires an X bound and canonical probes, but those prerequisite refusals are companion contract validations, not atoms of this root.",
        threshold="Derived finite series bound",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Continue frozen precision proof",
        refused_outcome="finite-series ValueError; absent X bound/probes produce their own prerequisite ValueErrors",
        oracle="Decimal/Fraction power series and probe energy; frozen factory.",
        axis_name="Boundary cells for the outward probe_energy * sum(x_bound**p/p) series scale.",
        low="small x/order/probes",
        endpoints=("last finite", "first overflowing series"),
        high="overflow",
        extreme="x=0/minsubnormal/max, order 0/1/2/16/high, probe/n extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:runtime:total-error-budget": _metadata(
        quantity="Outward analytic_tail + gamma_operation_count*(base_scale+series_scale) <= certificate.tolerance (analytic tail is zero for frozen plans).",
        threshold="Derived analytic-tail plus floating-roundoff proof",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="Plan accepted and constructed",
        refused_outcome="total-error-budget ValueError",
        oracle="Independent analytic tail, Decimal gamma/sums, actual plan output and slogdet; both factories.",
        axis_name="Boundary cells for Outward analytic_tail + gamma_operation_count*(base_scale+series_scale) <= certificate.tolerance (analytic tail is zero for frozen plans).",
        low="error well below tolerance",
        endpoints=("tolerance T-ULP", "T, T+ULP"),
        high="above budget",
        extreme="cancellation, rho/order/n/probes/dtype/base/series extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:runtime-call:scalar-and-dtype": _metadata(
        quantity="Base runtime argument must be scalar (companion ordinary contract); the canonical gate root requires x64 enabled for a >f32 certified plan and every dynamic dtype real floating with itemsize at least the certified dtype.",
        threshold="Runtime API precision contract",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="Execute JAX kernel",
        refused_outcome="scalar/dtype/context ValueError",
        oracle="Inspect canonical JAX dtypes and compare plan result with NumPy.",
        axis_name="Boundary cells for Base runtime argument must be scalar (companion ordinary contract); the canonical gate root requires x64 enabled for a >f32 certified plan and every dynamic dtype real floating with itemsize at least the certified dtype.",
        low="matching scalar f32/f64",
        endpoints=("itemsize just below/equal/above", "x64 off/on"),
        high="integer/complex/vector",
        extreme="float16/32/64, mixed values, context switch",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:trace-factory:exact-evidence": _metadata(
        quantity="Exact power traces are present and bitwise match independently recomputed traces through the certificate-selected order; arithmetic failure is mismatch.",
        threshold="Exact evidence identity",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="Construct TraceLogPlan",
        refused_outcome="exact-evidence ValueError",
        oracle="Dense matrix powers and traces plus runtime L6/slogdet.",
        axis_name="Boundary cells for Exact power traces are present and bitwise match independently recomputed traces through the certificate-selected order; arithmetic failure is mismatch.",
        low="exact traces",
        endpoints=("missing", "each trace exact/±ULP"),
        high="mismatched length/order",
        extreme="order -1/0/1/high, overflow, rho and scale extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "PLAN:frozen-factory:probe-presence-width": _metadata(
        quantity="FrozenProbes are present and values.shape[1] == n exactly.",
        threshold="Exact shape/API contract",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="Canonicalize probes and construct FrozenTraceLogPlan",
        refused_outcome="missing or wrong-width ValueError",
        oracle="Literal shape and captured-value identity; frozen factory and direct L7.",
        axis_name="Boundary cells for FrozenProbes are present and values.shape[1] == n exactly.",
        low="present width n",
        endpoints=("width n-1", "n, n+1"),
        high="gross mismatch",
        extreme="missing probes, n=1/16/10000, probe count 1/high, dtype extremes",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COUPLING:_classify_correlation:value-finite": _metadata(
        quantity="maximum canonical correlation value; require IEEE finite.",
        threshold="finite measurement domain; exact numerical-safety domain.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="proceed to noise-floor classification",
        refused_outcome="return actionable Refused for non-finite correlation.",
        oracle="scalar math.isfinite on the independently recomputed largest singular value.",
        axis_name="Boundary cells for maximum canonical correlation value; require IEEE finite.",
        low=".4",
        endpoints=("largest finite", "+inf"),
        high="near 1",
        extreme="NaN, +/-inf, zero, smallest subnormal",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COUPLING:_classify_correlation:floor-finite": _metadata(
        quantity="whitening floor sqrt(kappa_x*kappa_cond)*eps(dtype); require IEEE finite.",
        threshold="derived roundoff floor with finite-result safety boundary.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="classify correlation against the floor",
        refused_outcome="return actionable Refused when conditioning makes the floor non-finite.",
        oracle="high-precision product/square root from independent condition estimates and dtype eps.",
        axis_name="Boundary cells for whitening floor sqrt(kappa_x*kappa_cond)*eps(dtype); require IEEE finite.",
        low="finite kappas (2.4,3.1)",
        endpoints=("product below overflow", "next scale yielding inf"),
        high="ill-conditioned blocks",
        extreme="kappa 1, inf, NaN, subnormal eps surrogate",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COUPLING:_classify_correlation:lower-noise-floor": _metadata(
        quantity="canonical correlation versus whitening floor; measured only when value > floor.",
        threshold="closed refusal boundary value <= floor; derived D74 noise-floor policy.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="above floor continue/return Measured",
        refused_outcome="At or below floor return Refused rather than claim low coupling",
        oracle="independent SVD correlation and high-precision floor formula.",
        axis_name="Boundary cells for canonical correlation versus whitening floor; measured only when value > floor.",
        low="(value=.4,floor=.01)",
        endpoints=("value nextafter(floor, +inf)", "equal floor, nextafter(floor, 0)"),
        high="value .9",
        extreme="floor 0, subnormal gap, floor >=1, signed zero",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COUPLING:_classify_correlation:upper-noise-floor": _metadata(
        quantity="distance from one 1-value versus floor; measured only when value < 1-floor.",
        threshold="closed refusal boundary value >= 1-floor; derived D74 singular-ridge noise-floor policy.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="below boundary return Measured",
        refused_outcome="At/above boundary return Refused rather than claim a resolved perfect ridge",
        oracle="independent SVD correlation and high-precision subtraction/floor formula.",
        axis_name="Boundary cells for distance from one 1-value versus floor; measured only when value < 1-floor.",
        low="(value=.4,floor=.01)",
        endpoints=(
            "nextafter(1-floor, 0)",
            "exactly 1-floor, nextafter(1-floor, +inf)",
        ),
        high="1",
        extreme="floor 0/subnormal, value slightly above 1 from roundoff, signed zero",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COUPLING:_condition_number:finite-spectrum": _metadata(
        quantity="smallest and largest eigenvalues of a symmetric precision/marginal matrix; both must be finite.",
        threshold="IEEE finite spectral domain; exact numerical-safety boundary.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="continue condition calculation",
        refused_outcome="Otherwise return NaN condition so downstream floor/coupling classification refuses",
        oracle="analytic eigenvalues for 2x2 or high-precision symmetric eigensolver.",
        axis_name="Boundary cells for smallest and largest eigenvalues of a symmetric precision/marginal matrix; both must be finite.",
        low="eigenvalues (1.3,2.4)",
        endpoints=("largest finite", "inf and smallest finite versus NaN"),
        high="broad finite spectrum",
        extreme="repeated values, subnormal, NaN/inf matrix entries",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COUPLING:_condition_number:positive-spectrum": _metadata(
        quantity="smallest symmetric eigenvalue; require smallest > 0 before division.",
        threshold="strict SPD boundary; exact mathematical domain.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="return largest/smallest",
        refused_outcome="return infinity for zero/negative spectrum so whitening floor refuses.",
        oracle="LDL inertia/Cholesky or analytic eigenvalues independent of the condition calculation.",
        axis_name="Boundary cells for smallest symmetric eigenvalue; require smallest > 0 before division.",
        low="(1.3,2.4)",
        endpoints=("smallest positive subnormal", "zero, negative subnormal"),
        high="ill-conditioned positive spectrum",
        extreme="identity, singular, indefinite, repeated eigenvalues",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COUPLING:block_coupling:f-xx-spd": _metadata(
        quantity="first within-block posterior precision matrix f_xx; it must admit Cholesky (strict SPD).",
        threshold="strict SPD mathematical domain required by whitening.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="form the f_xx Cholesky factor and continue to f_tt",
        refused_outcome="raise actionable GraphError if the f_xx factor does not exist.",
        oracle="independent symmetric eigenvalue/LDL-inertia check for f_xx; analytic 1x1 principal minor.",
        axis_name="Boundary cells for f_xx with an independently valid f_tt companion.",
        low="non-unit positive f_xx with valid f_tt",
        endpoints=(
            "positive-subnormal minimum eigenvalue",
            "zero or negative minimum eigenvalue",
        ),
        high="negative non-unit f_xx with valid f_tt",
        extreme="maximum-magnitude negative f_xx with valid f_tt",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "COUPLING:block_coupling:f-tt-spd": _metadata(
        quantity="second within-block posterior precision matrix f_tt; it must admit Cholesky (strict SPD).",
        threshold="strict SPD mathematical domain required by whitening.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="form the f_tt Cholesky factor and whiten cross-block precision",
        refused_outcome="raise actionable GraphError if the f_tt factor does not exist.",
        oracle="independent symmetric eigenvalue/LDL-inertia check for f_tt; analytic 1x1 principal minor.",
        axis_name="Boundary cells for f_tt with an independently valid f_xx companion.",
        low="non-unit positive f_tt with valid f_xx",
        endpoints=(
            "positive-subnormal minimum eigenvalue",
            "zero or negative minimum eigenvalue",
        ),
        high="negative non-unit f_tt with valid f_xx",
        extreme="maximum-magnitude negative f_tt with valid f_xx",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "MAP:map_estimate:finite-derivative-payload": _metadata(
        quantity="objective scalar, every gradient entry, and every Hessian entry at the returned Newton point; all must be finite.",
        threshold="IEEE finite derivative domain; exact numerical-safety boundary.",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="proceed to stationarity/curvature certification",
        refused_outcome="return actionable Refused if any payload component is non-finite.",
        oracle="host NumPy scalar/array isfinite checks on independently evaluated objective/finite-difference derivatives for a small fixture.",
        axis_name="Boundary cells for objective scalar, every gradient entry, and every Hessian entry at the returned Newton point; all must be finite.",
        low="finite non-unit quadratic payload",
        endpoints=(
            "largest finite",
            "inf in value/gradient/Hessian one coordinate at a time",
        ),
        high="large finite derivatives",
        extreme="NaN, +/-inf, empty latent handled before this gate, signed zero/subnormal",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "MAP:map_estimate:stationarity-floor": _metadata(
        quantity="infinity norm gradient_norm=max(abs(gradient)) versus gradient_floor=sqrt(eps(gradient_dtype))*mode.size*||H||_2.",
        threshold="sqrt(eps(gradient dtype)) * mode.size * ||H||_2",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="continue to curvature tests",
        refused_outcome="return actionable non-stationary Refused when gradient exceeds the floor.",
        oracle="independently compute spectral Hessian norm and formula with high precision; finite-difference directional derivative as neighbour.",
        axis_name="Boundary cells for infinity norm gradient_norm=max(abs(gradient)) versus gradient_floor=sqrt(eps(gradient_dtype))*mode.size*||H||_2.",
        low="gradient norm at half the derived floor",
        endpoints=(
            "gradient norm equals the floor",
            "gradient norm one ULP above the floor",
        ),
        high="large nonstationary gradient",
        extreme="zero Hessian, mode sizes 1/large, float32/float64, subnormal gradient",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "MAP:map_estimate:relative-positive-curvature": _metadata(
        quantity="smallest Hessian eigenvalue versus curvature_floor=eps(H_dtype)*abs(lambda_max)*max(mode.size,1); require strict lambda_min > floor.",
        threshold="eps(H dtype) * abs(lambda_max) * max(mode.size, 1)",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="continue toward MapEstimate",
        refused_outcome="return degenerate/non-positive-curvature Refused at or below floor.",
        oracle="high-precision symmetric eigensolver/LDL inertia and independent formula evaluation.",
        axis_name="Boundary cells for smallest Hessian eigenvalue versus curvature_floor=eps(H_dtype)*abs(lambda_max)*max(mode.size,1); require strict lambda_min > floor.",
        low="lambda_min twice floor with non-unit spectrum",
        endpoints=("nextafter(floor, +inf)", "equal floor, nextafter(floor, 0)"),
        high="well-curved Hessian",
        extreme="mode size 1/large, repeated eigenvalues, subnormal positive/zero/negative lambda_min",
        fixture_scale_policy=FixtureScalePolicy.NON_UNIT_REQUIRED,
    ),
    "MAP:map_estimate:absolute-curvature": _metadata(
        quantity="largest Hessian eigenvalue versus absolute_curvature_floor=sqrt(eps(H_dtype)); require strict lambda_max > floor.",
        threshold="strict derived absolute scale floor preventing an underflow-flat stationary tail from being called a MAP.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="return MapEstimate only above the floor and after relative curvature passes",
        refused_outcome="Otherwise return curvature Refused",
        oracle="high-precision largest symmetric eigenvalue and direct sqrt(eps) computation.",
        axis_name="Boundary cells for largest Hessian eigenvalue versus absolute_curvature_floor=sqrt(eps(H_dtype)); require strict lambda_max > floor.",
        low="lambda_max 2*sqrt(eps)",
        endpoints=("nextafter(floor, +inf)", "equal floor, nextafter(floor, 0)"),
        high="order-one curvature",
        extreme="all-zero Hessian, negative spectrum, float32/float64, subnormal and repeated eigenvalues",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "GRAPH:_names:duplicate-multiplicity": _metadata(
        quantity="multiplicity names.count(name) for each requested node name; every name must occur exactly once (count <= 1).",
        threshold="discrete duplicate boundary at count 2; API declaration contract.",
        provenance=ThresholdProvenance.API_CONTRACT,
        admitted_outcome="return the original ordered tuple when unique",
        refused_outcome="raise GraphError listing sorted duplicates when any count exceeds one.",
        oracle="collections.Counter over the tuple, preserving a separate check that output order is unchanged.",
        axis_name="Boundary cells for multiplicity names.count(name) for each requested node name; every name must occur exactly once (count <= 1).",
        low="empty list iterable, distinct from the empty tuple T-1 cell",
        endpoints=("count 1 admits", "count 2 refuses"),
        high="many repetitions of several names",
        extreme="empty, one name, Unicode aliases, long tuple, widely separated duplicates",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COSTS:gap_is_contested:contested-bandwidth": _metadata(
        quantity="relative cost gap between two strategy rows; contested when the gap is below the contested bandwidth.",
        threshold="open refusal boundary gap < CONTESTED_BANDWIDTH (0.25); derived D93 contested-bandwidth policy.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="below the band, mark the row contested rather than a clear winner",
        refused_outcome="at or above the band, the gap is a decision, not a contest",
        oracle="direct float comparison gap < 0.25 with an independent high-precision gap formula.",
        axis_name="Boundary cells for the relative cost gap versus the contested bandwidth; contested when gap < 0.25.",
        low="0.01",
        endpoints=("nextafter(0.25, 0)", "0.25, nextafter(0.25, +inf)"),
        high="0.99",
        extreme="0, +/-inf, nan",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COSTS:timing_noise_in_domain:proper-fraction": _metadata(
        quantity="the timing noise tolerance; it must be a proper fraction (tol < 1.0) or the cost interval spread makes cost_lo non-positive.",
        threshold="open domain boundary tol < 1.0; derived D94 timing-noise policy.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="tol below 1.0 keeps the cost interval positive",
        refused_outcome="tol at or above 1.0 makes cost_lo <= 0 and the interval meaningless",
        oracle="direct float comparison tol < 1.0.",
        axis_name="Boundary cells for the timing noise tolerance; proper fraction when tol < 1.0.",
        low="0.01",
        endpoints=("nextafter(1.0, 0)", "1.0, nextafter(1.0, +inf)"),
        high="4.0",
        extreme="0, +/-inf, nan",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COSTS:cg_tol_positive:strictly-positive": _metadata(
        quantity="the CG tolerance in k_cg; it must be strictly positive or log(2 / tol) is undefined.",
        threshold="strict positivity tol > 0.0; derived D95 k_cg domain.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="tol > 0 names a finite CG iteration count",
        refused_outcome="tol <= 0 cannot be priced; raise ValueError",
        oracle="direct float comparison tol > 0.0.",
        axis_name="Boundary cells for the CG tolerance; strictly positive when tol > 0.0.",
        low="1e-6",
        endpoints=("smallest positive subnormal", "0.0, negative subnormal"),
        high="1.0",
        extreme="nan, +/-inf",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COLLAPSE:pivots:finite": _metadata(
        quantity="the pivots of the collapsed block's re-triangularisation; every one must be IEEE finite.",
        threshold="finite domain; exact numerical-safety boundary (a nan/inf pivot makes the marginal log-density not a number).",
        provenance=ThresholdProvenance.EXACT_DOMAIN,
        admitted_outcome="proceed to the relative-floor classification",
        refused_outcome="raise via eqx.error_if so a non-finite marginal never returns a plausible number",
        oracle="jnp.all(jnp.isfinite(pivots)) on the independently recomputed QR pivots.",
        axis_name="Boundary cells for the collapsed block's pivots; require every pivot IEEE finite.",
        low="positive ordinary pivots",
        endpoints=("largest finite pivot", "+inf pivot"),
        high="near-overflow pivots",
        extreme="nan, +/-inf, zero, smallest subnormal",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
    "COLLAPSE:pivots:relative-floor": _metadata(
        quantity="each block pivot versus the relative floor sqrt(eps) * max(pivots); the block must constrain itself.",
        threshold="open lower boundary pivots[:n_block] > floor; derived D98 relative-floor policy.",
        provenance=ThresholdProvenance.DERIVED,
        admitted_outcome="all block pivots above the floor; the Gaussian integral over the block is convergent",
        refused_outcome="a pivot at or below the floor means an unconstrained direction; raise rather than return a finite plausible number",
        oracle="direct high-precision comparison pivot_i > sqrt(eps) * max(pivots) on the independent QR pivots.",
        axis_name="Boundary cells for the collapsed block's pivots versus the relative floor; constrained when every block pivot is above sqrt(eps) * max(pivots).",
        low="well-constrained block (all pivots near max)",
        endpoints=("nextafter(floor, +inf)", "floor, nextafter(floor, -inf)"),
        high="well-separated pivots",
        extreme="zero pivot, nan pivot, single pivot, degenerate block",
        fixture_scale_policy=FixtureScalePolicy.NOT_APPLICABLE,
    ),
}


def _replace_axes(gate_id: str, *axes: AxisRange) -> None:
    """Replace a bundled prose axis with its real production input fields."""
    GATE_METADATA[gate_id] = replace(GATE_METADATA[gate_id], axes=tuple(axes))


def _ladder_numeric_axis(
    name: str,
    boundary: str,
    extreme: str,
) -> AxisRange:
    """Describe one scalar production input, including both adjacent faces."""
    return AxisRange(
        name,
        f"ordinary non-unit value safely on the admitted side of {boundary}",
        (
            f"nextafter({boundary}, admitted side)",
            f"exactly {boundary} and nextafter({boundary}, refused side)",
        ),
        f"ordinary non-unit value well on the refused side of {boundary}",
        extreme,
    )


def _ladder_discrete_axis(
    name: str,
    boundary: str,
    extreme: str,
) -> AxisRange:
    """Describe one integer production input without a derived proxy axis."""
    return AxisRange(
        name,
        f"small non-unit integer safely on the admitted side of {boundary}",
        (
            f"one integer below {boundary}",
            f"exactly {boundary} and one integer above it",
        ),
        f"large integer well on the refused side of {boundary}",
        extreme,
    )


def _ladder_state_axis(
    name: str,
    admitted: str,
    refused: str,
    extreme: str,
) -> AxisRange:
    """Describe one categorical/container input whose identity is observable."""
    return AxisRange(
        name,
        f"ordinary non-unit payload in the admitted state: {admitted}",
        (f"last admitted state: {admitted}", f"first refused state: {refused}"),
        f"non-unit payload materially in the refused state: {refused}",
        extreme,
    )


_LADDER_INPUT_AXES: Mapping[str, AxisRange] = {
    "payload_sigma_layout": _ladder_state_axis(
        "sigma_layout",
        "compact diagonal or bitwise-symmetric dense layout uses the original payload",
        "nonexact tolerance-symmetric dense layout uses the representative",
        "compact, exact dense, transposed asymmetry, empty, NaN, and infinity layouts",
    ),
    "fact_sigma_layout": _ladder_state_axis(
        "sigma_layout",
        "compact positive diagonal or finite symmetric dense layout",
        "dense asymmetric, singular, or indefinite layout",
        "compact, dense, n=1, empty, asymmetric, NaN, and infinity layouts",
    ),
    "sigma_asymmetry": AxisRange(
        "sigma_asymmetry",
        "ordinary non-unit asymmetry below the configured tolerance",
        (
            "asymmetry nextafter(atol + rtol*scale, 0)",
            "equality and nextafter(atol + rtol*scale, +inf)",
        ),
        "ordinary non-unit asymmetry well above the configured tolerance",
        "zero, minimum subnormal, signed orientations, near-maximum finite, NaN, and infinity",
    ),
    "structure_atol": _ladder_numeric_axis(
        "structure_atol",
        "the observed off-diagonal/asymmetry magnitude",
        "zero, negative, minimum subnormal, large finite, NaN, and infinity tolerances",
    ),
    "structure_rtol": _ladder_numeric_axis(
        "structure_rtol",
        "asymmetry divided by the opposite-orientation entry magnitude",
        "zero, negative, minimum subnormal, large finite, NaN, and infinity tolerances",
    ),
    "computation_dtype": AxisRange(
        "computation_dtype",
        "ordinary non-unit float64 input safely inside its dtype-specific boundary",
        (
            "float32 input at its adjacent representability/condition boundary",
            "float64 input at its adjacent representability/condition boundary",
        ),
        "ordinary non-unit narrow-dtype input beyond its dtype-specific boundary",
        "float32 and float64 minimum subnormal, maximum finite, overflow, and cancellation cases",
    ),
    "lambda_entry": _ladder_numeric_axis(
        "lambda_entry",
        "the finite TwoSum overflow/cancellation face",
        "minimum subnormal, signed zero, largest finite, NaN, and infinity entries",
    ),
    "perturbation_entry": _ladder_numeric_axis(
        "perturbation_entry",
        "the finite TwoSum overflow/cancellation face",
        "positive/negative minimum subnormal, largest finite, NaN, and infinity entries",
    ),
    "sigma_entry": _ladder_numeric_axis(
        "sigma_entry",
        "strict zero positivity boundary",
        "minimum positive subnormal, positive finite, zero, and negative entries",
    ),
    "diagonal_structure_request": AxisRange(
        "structure_request",
        "ordinary non-unit diagonal payload with an explicit diagonal request",
        (
            "None auto-detects an exact diagonal payload",
            "explicit diagonal admits; unsupported request refuses this payload",
        ),
        "non-unit diagonal payload paired with a contradicted structure request",
        "None, diagonal, circulant, Toeplitz, Kronecker, and unknown labels",
    ),
    "kronecker_structure_request": _ladder_state_axis(
        "structure_request",
        "explicit Kronecker request",
        "None, unsupported, or different structure request",
        "None, diagonal, circulant, Toeplitz, Kronecker, and unknown labels",
    ),
    "rung3_structure_request": AxisRange(
        "structure_request",
        "ordinary non-unit payload using one supported structure request",
        (
            "None auto-detects an exact diagonal payload",
            "diagonal, circulant, Toeplitz, and Kronecker requests admit when verified",
        ),
        "non-unit payload using an unknown or contradicted structure request",
        "None, every supported request, unknown labels, absent evidence, and malformed evidence",
    ),
    "structure_presence": _ladder_state_axis(
        "structure_presence",
        "KroneckerStructure evidence is present",
        "KroneckerStructure evidence is absent or malformed",
        "None, empty factors, one factor, many factors, malformed factors, NaN, and infinity",
    ),
    "off_diagonal": _ladder_numeric_axis(
        "off_diagonal",
        "bitwise zero",
        "positive/negative minimum subnormal, signed zero, and large finite values",
    ),
    "circulant_layout": _ladder_state_axis(
        "circulant_layout",
        "every row is the exact cyclic shift of the first",
        "one shifted entry differs",
        "n=1, repeated rows, one-ULP mismatch, NaN, infinity, and asymmetric layouts",
    ),
    "spectrum_scale": _ladder_numeric_axis(
        "spectrum_scale",
        "strictly positive finite FFT spectrum",
        "minimum positive subnormal, zero, negative, NaN, infinity, and near-maximum spectra",
    ),
    "toeplitz_layout": _ladder_state_axis(
        "toeplitz_layout",
        "every descending diagonal is exactly constant",
        "one diagonal entry differs",
        "n=1, one-ULP mismatch, asymmetric, NaN, infinity, and near-maximum layouts",
    ),
    "factor_spectrum": _ladder_numeric_axis(
        "factor_spectrum",
        "strictly positive finite factor eigenvalue",
        "minimum positive subnormal, zero, negative, repeated, NaN, and infinity eigenvalues",
    ),
    "factor_shape": _ladder_state_axis(
        "factor_shape",
        "square factor shapes whose product matches Sigma",
        "nonsquare or product-mismatched factor shapes",
        "empty, scalar, n=1, rectangular, permuted, and very unbalanced shapes",
    ),
    "reconstruction_value": _ladder_numeric_axis(
        "reconstruction_value",
        "the exact reconstructed Sigma entry",
        "signed zero, one-ULP mismatch, minimum subnormal, NaN, and infinity values",
    ),
    "sigma_symmetry": _ladder_numeric_axis(
        "sigma_symmetry",
        "the exact/tolerant symmetry boundary",
        "zero, signed minimum subnormal, large finite asymmetry, NaN, and infinity",
    ),
    "smallest_eigenvalue": _ladder_numeric_axis(
        "smallest_eigenvalue",
        "strict zero SPD boundary",
        "positive minimum subnormal, zero, negative, repeated, NaN, and infinity eigenvalues",
    ),
    "condition_scale": _ladder_numeric_axis(
        "condition_scale",
        "the configured finite condition ceiling",
        "one, minimum subnormal, largest finite, unresolved, NaN, and infinity condition scales",
    ),
    "rank_factor_presence": AxisRange(
        "factor_presence",
        "ordinary non-unit perturbation with factors absent uses valid algebraic rank evidence",
        (
            "factors absent admits through the algebraic-rank path",
            "compatible factors admit; present mismatched factors refuse factor evidence",
        ),
        "non-unit perturbation with present materially mismatched factor evidence",
        "None, empty rank, rank one, mismatched, aliased, and non-finite factor payloads",
    ),
    "determinant_factor_presence": _ladder_state_axis(
        "factor_presence",
        "both compatible low-rank factors are present",
        "factor evidence is absent or incomplete while the finite-polynomial alternative is disabled",
        "None, empty rank, rank one, mismatched, aliased, and non-finite factor payloads",
    ),
    "factor_reconstruction": _ladder_numeric_axis(
        "factor_reconstruction",
        "the exact perturbation reconstruction",
        "signed zero, one-ULP mismatch, minimum subnormal, NaN, and infinity values",
    ),
    "perturbation_rank": _ladder_discrete_axis(
        "perturbation_rank",
        "the algebraically reconstructed rank",
        "zero, one, repeated/deficient, full, and dimension-exceeding ranks",
    ),
    "factor_layout": _ladder_state_axis(
        "factor_layout",
        "compatible contiguous or strided factor arrays reconstruct the perturbation exactly",
        "shape-, orientation-, or memory-layout-sensitive evidence does not reconstruct it",
        "C/F order, transposed views, negative strides, empty rank, rank one, and wide factors",
    ),
    "factor_gauge": _ladder_numeric_axis(
        "factor_gauge",
        "the balanced rescaling range that preserves an exact factor product",
        "minimum subnormal, extreme reciprocal scales, cancellation, maximum finite, NaN, and infinity",
    ),
    "lambda_scale": _ladder_numeric_axis(
        "lambda_scale",
        "the resolved-rho representability/convergence face",
        "minimum subnormal, signed zero, near-maximum finite, NaN, and infinity scales",
    ),
    "perturbation_scale": _ladder_numeric_axis(
        "perturbation_scale",
        "the resolved-rho representability/convergence face",
        "positive/negative minimum subnormal, near-maximum finite, NaN, and infinity scales",
    ),
    "matrix_geometry": AxisRange(
        "matrix_geometry",
        "ordinary non-unit diagonal/normal geometry with a resolved spectral radius",
        (
            "diagonal geometry at the scalar ratio boundary",
            "nonnormal dense geometry at the same independently measured radius",
        ),
        "non-unit singular or strongly nonnormal geometry with unresolved/large radius",
        "diagonal, normal, defective, nearly singular, repeated-eigenvalue, NaN, and infinity geometries",
    ),
    "determinant_alternative": AxisRange(
        "determinant_alternative",
        "generic finite-polynomial payload is valid with determinant factors absent",
        (
            "generic rho exactly one remains valid without factors",
            "generic rho above one is rescued only by valid determinant-lemma factors",
        ),
        "generic rho above one with absent or invalid determinant factors",
        "absent, exact rank-one, rank-deficient, mismatched, NaN, and infinity factor evidence",
    ),
    "sigma_formation": _ladder_state_axis(
        "sigma_formation",
        "finite exact TwoSum formation",
        "overflowed or non-finite formation",
        "cancellation, signed zero, minimum subnormal, largest finite, NaN, and infinity",
    ),
    "sigma_lambda_equality": _ladder_state_axis(
        "sigma_lambda_equality",
        "Sigma bitwise equals Lambda",
        "one resolved entry has a one-ULP mismatch",
        "empty, signed-zero-only difference, one-ULP difference, NaN, and infinity",
    ),
    "dense_condition": _ladder_numeric_axis(
        "dense_condition",
        "the finite dense-arithmetic condition ceiling",
        "one, zero, largest finite, unresolved, NaN, and infinity conditions",
    ),
    "rank_evidence": _ladder_state_axis(
        "rank_evidence",
        "algebraic rank matches the supplied factor payload",
        "rank is missing, deficient, or mismatched",
        "rank zero, rank one, repeated columns, full rank, and dimension-exceeding evidence",
    ),
    "payload_capability": _ladder_state_axis(
        "payload_capability",
        "compact diagonal or valid determinant-lemma payload",
        "neither executable payload is available",
        "compact, dense, absent, malformed, non-finite, and rank-zero payloads",
    ),
    "sigma_spd": _ladder_state_axis(
        "sigma_spd",
        "finite symmetric positive-definite Sigma",
        "singular, indefinite, asymmetric, or unresolved Sigma",
        "positive, minimum-subnormal, zero, negative, NaN, and infinity spectra",
    ),
    "rank": _ladder_discrete_axis(
        "rank",
        "the active configured rank ceiling",
        "zero, one, maximum configured, full, and dimension-exceeding ranks",
    ),
    "dimension": _ladder_discrete_axis(
        "dimension",
        "the active configured matrix-size ceiling",
        "zero, one, exact ceiling, one above, and largest practical dimensions",
    ),
    "low_rank_max": _ladder_discrete_axis(
        "low_rank_max",
        "the supplied perturbation rank",
        "zero, one, exact rank, one below, and very large configured ceilings",
    ),
    "low_rank_fraction": _ladder_numeric_axis(
        "low_rank_fraction",
        "rank divided by dimension",
        "zero, minimum subnormal, one/dimension, one, above one, NaN, and infinity",
    ),
    "chain_block_size": _ladder_discrete_axis(
        "chain_block_size",
        "the exact supported chain block width",
        "zero, one, exact width, one above, odd, and dimension-sized widths",
    ),
    "chain_layout": _ladder_state_axis(
        "chain_layout",
        "exact symmetric block-tridiagonal chain",
        "one forbidden off-band block is nonzero",
        "n=1, one block, missing link, asymmetric link, cycle, NaN, and infinity",
    ),
    "structure_evidence": _ladder_state_axis(
        "structure_evidence",
        "requested structure is exactly verified",
        "request is absent, unknown, or contradicted",
        "diagonal, circulant, Toeplitz, Kronecker, chain, unknown, and malformed evidence",
    ),
    "dense_max_n": _ladder_discrete_axis(
        "dense_max_n",
        "the real matrix dimension",
        "zero, one, dimension minus one, exact dimension, and very large limits",
    ),
    "finite_max_n": _ladder_discrete_axis(
        "finite_max_n",
        "the real matrix dimension",
        "zero, one, dimension minus one, exact dimension, and very large limits",
    ),
    "finite_max_rank": _ladder_discrete_axis(
        "finite_max_rank",
        "the real perturbation rank",
        "zero, one, rank minus one, exact rank, full, and very large limits",
    ),
    "finite_payload_rho": _ladder_numeric_axis(
        "finite_payload_rho",
        "the finite-rung payload rho ceiling",
        "zero, minimum subnormal, nextafter(one,0), one, NaN, and infinity",
    ),
    "actual_rho": _ladder_numeric_axis(
        "actual_rho",
        "strict convergence value one and certificate coverage",
        "zero, minimum subnormal, nextafter(one,0), one, above one, NaN, and infinity",
    ),
    "trace_order": _ladder_discrete_axis(
        "trace_order",
        "the requested/certified trace order",
        "negative, zero, one, exact order, one short, and very large orders",
    ),
    "trace_evidence": _ladder_state_axis(
        "trace_evidence",
        "all retained power traces are present and exact",
        "one trace is missing or differs",
        "empty, one-short, one-ULP mismatch, signed zero, NaN, and infinity evidence",
    ),
    "certified_rho": _ladder_numeric_axis(
        "certified_rho",
        "actual rho coverage and strict certificate value one",
        "zero, minimum subnormal, equality, undercoverage, nextafter(one,0), one, NaN, and infinity",
    ),
    "probe_width": _ladder_discrete_axis(
        "probe_width",
        "the real matrix dimension",
        "zero, dimension minus one, exact dimension, one above, and very large widths",
    ),
    "probe_presence": _ladder_state_axis(
        "probe_presence",
        "FrozenProbes evidence is present",
        "FrozenProbes evidence is absent or malformed",
        "None, empty probes, one probe, many probes, malformed values, NaN, and infinity",
    ),
}


_LADDER_GATE_INPUTS: Mapping[str, tuple[str, ...]] = {
    "LADDER:sigma:payload-symmetry": (
        "payload_sigma_layout",
        "sigma_asymmetry",
        "structure_atol",
        "structure_rtol",
        "computation_dtype",
    ),
    "LADDER:sigma:finite-two-sum": (
        "lambda_entry",
        "perturbation_entry",
        "computation_dtype",
    ),
    "LADDER:structure:compact-diagonal-positive": ("sigma_entry",),
    "LADDER:structure:diagonal-tolerance": (
        "diagonal_structure_request",
        "off_diagonal",
    ),
    "LADDER:structure:circulant-tolerance-spectrum": (
        "circulant_layout",
        "spectrum_scale",
    ),
    "LADDER:structure:toeplitz-tolerance": ("toeplitz_layout",),
    "LADDER:structure:kronecker-evidence": (
        "kronecker_structure_request",
        "structure_presence",
        "factor_spectrum",
        "factor_shape",
        "reconstruction_value",
    ),
    "LADDER:sigma:symmetry-spd-condition": (
        "fact_sigma_layout",
        "sigma_symmetry",
        "smallest_eigenvalue",
        "condition_scale",
        "structure_rtol",
        "structure_atol",
        "computation_dtype",
    ),
    "LADDER:rank:evidence": (
        "rank_factor_presence",
        "factor_reconstruction",
        "perturbation_rank",
        "lambda_scale",
        "factor_layout",
        "factor_gauge",
        "computation_dtype",
    ),
    "LADDER:rho:measurement": (
        "lambda_scale",
        "perturbation_scale",
        "matrix_geometry",
        "computation_dtype",
    ),
    "LADDER:finite:payload-rho": (
        "lambda_scale",
        "perturbation_scale",
        "matrix_geometry",
        "computation_dtype",
        "determinant_alternative",
    ),
    "LADDER:determinant-lemma:payload": (
        "determinant_factor_presence",
        "factor_reconstruction",
        "sigma_formation",
        "sigma_symmetry",
        "condition_scale",
    ),
    "LADDER:rung0:base": (
        "sigma_formation",
        "sigma_lambda_equality",
        "dense_condition",
    ),
    "LADDER:rung1:low-rank-size": (
        "rank_evidence",
        "payload_capability",
        "sigma_spd",
        "rank",
        "dimension",
        "low_rank_max",
        "low_rank_fraction",
    ),
    "LADDER:rung2:chain": (
        "chain_block_size",
        "chain_layout",
        "sigma_formation",
        "sigma_spd",
        "condition_scale",
        "computation_dtype",
    ),
    "LADDER:rung3:structured": (
        "rung3_structure_request",
        "structure_evidence",
        "sigma_formation",
        "sigma_spd",
        "condition_scale",
    ),
    "LADDER:rung4:dense": (
        "dimension",
        "dense_max_n",
        "condition_scale",
        "sigma_spd",
        "computation_dtype",
    ),
    "LADDER:rung5:finite-size": (
        "dimension",
        "finite_max_n",
        "payload_capability",
        "rank",
        "finite_max_rank",
    ),
    "LADDER:rung5:finite-executable": (
        "dimension",
        "finite_max_n",
        "payload_capability",
        "rank",
        "finite_max_rank",
        "lambda_scale",
        "perturbation_scale",
        "sigma_formation",
        "smallest_eigenvalue",
        "sigma_symmetry",
        "determinant_factor_presence",
        "factor_reconstruction",
        "dense_condition",
        "computation_dtype",
    ),
    "LADDER:rung6:trace": (
        "sigma_formation",
        "actual_rho",
        "trace_order",
        "trace_evidence",
        "certified_rho",
    ),
    "LADDER:rung7:frozen": (
        "sigma_formation",
        "trace_order",
        "probe_presence",
        "probe_width",
        "actual_rho",
        "certified_rho",
    ),
}


for _gate_id, _axis_names in _LADDER_GATE_INPUTS.items():
    _replace_axes(_gate_id, *(_LADDER_INPUT_AXES[name] for name in _axis_names))


_replace_axes(
    "PLAN:certificate:rho-domain-and-coverage",
    AxisRange(
        "measured_max",
        "ordinary interior measurement",
        ("0 and nextafter(0,-inf)", "nextafter(1,0) and 1"),
        "uncovered/above-one measurement",
        "NaN and +/-Inf",
    ),
    AxisRange(
        "certified_rho",
        "ordinary interior certificate",
        ("0 and nextafter(measured,0)", "equality/nextafter(measured,+inf)"),
        "one or uncovered certificate",
        "NaN and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:certificate:error-budget-domain",
    AxisRange(
        "margin",
        "ordinary positive margin",
        ("nextafter(0,-inf)", "0 and nextafter(0,+inf)"),
        "large positive margin",
        "NaN and +/-Inf; NaN follows the literal admitted source behavior",
    ),
    AxisRange(
        "tolerance",
        "ordinary positive tolerance",
        ("nextafter(0,-inf) and 0", "nextafter(0,+inf)"),
        "large positive tolerance",
        "NaN and +/-Inf",
    ),
    AxisRange(
        "tail_tolerance",
        "strictly interior tail tolerance",
        ("nextafter(0,-inf), 0, nextafter(0,+inf)", "nextafter(tolerance,0), equality, nextafter(tolerance,+inf)"),
        "tail above tolerance",
        "NaN and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:certificate:optional-scale-domain",
    AxisRange(
        "max_abs_lambda_logdet",
        "None or ordinary positive bound",
        ("nextafter(0,-inf)", "0 and nextafter(0,+inf)"),
        "large finite bound",
        "None, NaN, and +/-Inf",
    ),
    AxisRange(
        "max_x_operator_norm",
        "None or ordinary positive bound",
        ("nextafter(0,-inf)", "0 and nextafter(0,+inf)"),
        "large finite bound",
        "None, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:certificate:order-is-derived",
    AxisRange(
        "order",
        "order well below the independently recomputed minimum",
        ("m-1", "m and m+1"),
        "order far above the recomputed minimum",
        "negative, zero, and very large integer order",
    ),
    AxisRange(
        "certified_rho",
        "ordinary strict-convergence radius",
        ("nextafter(0,+inf)", "nextafter(1,0)"),
        "radius close to one requiring large order",
        "zero, NaN, and +/-Inf",
    ),
    AxisRange(
        "tail_tolerance",
        "ordinary positive non-unit tail tolerance",
        ("nextafter(0,+inf)", "nextafter(tolerance,0)"),
        "tail tolerance close to total tolerance",
        "zero, minimum subnormal, NaN, and +/-Inf",
    ),
    AxisRange(
        "multiplicity",
        "small positive multiplicity",
        ("1", "2"),
        "large positive multiplicity",
        "zero, negative, and very large integer multiplicity",
    ),
)
_replace_axes(
    "PLAN:audit:retained-rho",
    AxisRange(
        "retained_value",
        "retained rho below the certificate",
        ("nextafter(certified_rho,0)", "equality and nextafter(certified_rho,+inf)"),
        "retained rho far above the certificate",
        "zero, NaN, and +/-Inf",
    ),
    AxisRange(
        "certified_rho",
        "certificate comfortably above retained rho",
        ("nextafter(retained_value,0)", "equality and nextafter(retained_value,+inf)"),
        "certificate far below retained rho",
        "zero, nextafter(1,0), NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:audit:retained-lambda-scale",
    AxisRange(
        "retained_value",
        "ordinary non-unit retained absolute lambda logdet",
        ("nextafter(max_abs_lambda_logdet,0)", "equality and nextafter(bound,+inf)"),
        "retained magnitude far above the certified bound",
        "zero, NaN, and +/-Inf",
    ),
    AxisRange(
        "max_abs_lambda_logdet",
        "certified bound comfortably above retained magnitude",
        ("nextafter(retained_value,0)", "equality and nextafter(retained_value,+inf)"),
        "certified bound far below retained magnitude",
        "None, zero, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:audit:retained-x-norm",
    AxisRange(
        "retained_value",
        "ordinary non-unit retained operator norm",
        ("nextafter(max_x_operator_norm,0)", "equality and nextafter(bound,+inf)"),
        "retained norm far above the certified bound",
        "zero, NaN, and +/-Inf",
    ),
    AxisRange(
        "max_x_operator_norm",
        "certified bound comfortably above retained norm",
        ("nextafter(retained_value,0)", "equality and nextafter(retained_value,+inf)"),
        "certified bound far below retained norm",
        "None, zero, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:factory-certificate:lambda-scale",
    AxisRange(
        "lambda_matrix",
        "ordinary non-unit positive diagonal Lambda",
        ("nextafter(max_abs_lambda_logdet,0)", "equality and nextafter(bound,+inf)"),
        "Lambda whose actual logdet magnitude is far above the certificate",
        "minimum positive subnormal and largest finite SPD diagonal entries",
    ),
    AxisRange(
        "max_abs_lambda_logdet",
        "certificate comfortably above the problem magnitude",
        ("nextafter(actual lambda logdet,0)", "equality and nextafter(actual lambda logdet,+inf)"),
        "certificate far below the problem magnitude",
        "None, zero, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:factory-certificate:x-norm",
    AxisRange(
        "perturbation",
        "ordinary non-unit perturbation with a measured operator norm",
        ("nextafter(max_x_operator_norm,0)", "equality and nextafter(bound,+inf)"),
        "perturbation whose actual operator norm is far above the certificate",
        "zero, minimum subnormal, and largest finite perturbations",
    ),
    AxisRange(
        "max_x_operator_norm",
        "certificate comfortably above the problem norm",
        ("nextafter(actual operator norm,0)", "equality and nextafter(actual operator norm,+inf)"),
        "certificate far below the problem norm",
        "None, zero, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:warmup:rho-inputs-and-margin",
    AxisRange(
        "rho_value",
        "ordinary nonnegative measurement",
        ("nextafter(0,-inf)", "0 and nextafter(0,+inf)"),
        "large finite measurement",
        "empty sequence, NaN, and +/-Inf",
    ),
    AxisRange(
        "margin",
        "ordinary nonnegative margin",
        ("nextafter(0,-inf)", "0 and nextafter(0,+inf)"),
        "large finite margin",
        "NaN and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:warmup:rho-roundoff-ceiling",
    AxisRange(
        "rho_value",
        "ordinary nonnegative raw rho measurement",
        ("nextafter(0,+inf)", "raw value whose outward envelope approaches one"),
        "raw measurement whose rounded envelope reaches one",
        "zero, NaN, and +/-Inf",
    ),
    AxisRange(
        "margin",
        "ordinary nonnegative roundoff margin",
        ("zero", "nextafter(one-rho,0) and equality"),
        "margin that pushes the outward certificate above one",
        "minimum subnormal, NaN, and +/-Inf",
    ),
    AxisRange(
        "multiplicity",
        "small positive envelope multiplicity",
        ("1", "2"),
        "large multiplicity amplifying the envelope",
        "zero, negative, and very large integer multiplicity",
    ),
)
_replace_axes(
    "PLAN:gamma:operation-count-domain",
    AxisRange(
        "operation_count",
        "small nonnegative operation count",
        ("0", "1"),
        "large count approaching the gamma denominator boundary",
        "negative, bool, and very large integer count",
    ),
    AxisRange(
        "epsilon",
        "ordinary positive runtime epsilon",
        ("nextafter(0,+inf)", "epsilon making count*epsilon approach one"),
        "epsilon for which count*epsilon reaches or exceeds one",
        "zero, minimum subnormal, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:outward-arithmetic:positive-underflow",
    AxisRange(
        "proof_value",
        "ordinary finite nonzero proof magnitude",
        ("exact zero", "minimum positive subnormal"),
        "large finite proof magnitude",
        "negative finite, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:warmup:lambda-scale-inputs",
    AxisRange(
        "lambda_value",
        "ordinary signed non-unit logdet measurement",
        ("small negative finite", "small positive finite"),
        "large-magnitude finite measurement",
        "empty sequence, NaN, and +/-Inf",
    ),
    AxisRange(
        "lambda_logdet_margin",
        "ordinary nonnegative margin",
        ("nextafter(0,-inf)", "0 and nextafter(0,+inf)"),
        "large finite margin",
        "NaN and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:warmup:x-norm-inputs",
    AxisRange(
        "x_norm_value",
        "ordinary nonnegative norm measurement",
        ("nextafter(0,-inf)", "0 and nextafter(0,+inf)"),
        "large finite norm",
        "empty sequence, NaN, and +/-Inf",
    ),
    AxisRange(
        "x_operator_norm_margin",
        "ordinary nonnegative margin",
        ("nextafter(0,-inf)", "0 and nextafter(0,+inf)"),
        "large finite margin",
        "NaN and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:measurement:x-norm-finite",
    AxisRange(
        "lambda_entry",
        "ordinary finite non-unit lambda entry",
        ("largest finite entry", "nextafter(largest,+inf) overflow"),
        "large finite lambda magnitude",
        "zero, minimum subnormal, NaN, and +/-Inf",
    ),
    AxisRange(
        "perturbation_entry",
        "ordinary finite non-unit perturbation entry",
        ("largest finite entry", "nextafter(largest,+inf) overflow"),
        "large finite perturbation magnitude",
        "zero, minimum subnormal, NaN, and +/-Inf",
    ),
    AxisRange(
        "matrix_path",
        "compact float32 matrix capability",
        ("compact float64", "dense singular or indefinite refusal"),
        "dense float32 matrix capability",
        "dense full-range and subnormal float64 matrices",
    ),
)
_replace_axes(
    "PLAN:measurement:lambda-logdet-finite",
    AxisRange(
        "lambda_entry",
        "ordinary finite non-unit lambda entry",
        ("largest finite entry", "nextafter(largest,+inf) overflow"),
        "large finite lambda magnitude",
        "zero, minimum subnormal, NaN, and +/-Inf",
    ),
    AxisRange(
        "matrix_path",
        "compact float32 matrix capability",
        ("compact float64", "dense singular or indefinite refusal"),
        "dense float32 matrix capability",
        "dense full-range and subnormal float64 matrices",
    ),
)
_replace_axes(
    "PLAN:audit:retained-trace-evidence",
    AxisRange(
        "problem_trace_order",
        "retained order below the certificate-selected order",
        ("T-1 (certificate order - 1)", "T (certificate order) and T+1"),
        "retained order far above the certificate-selected order",
        "negative, missing, and very large order",
    ),
    AxisRange(
        "perturbation",
        "perturbation whose independently computed algebraic rank is below multiplicity",
        ("T-1 (multiplicity - 1)", "T (multiplicity) and T+1"),
        "perturbation whose algebraic rank is far above multiplicity",
        "zero, rank-one, and dimension-rank perturbations",
    ),
    AxisRange(
        "trace_evidence_value",
        "exact non-unit power traces",
        ("bitwise exact evidence", "one-ULP mismatch"),
        "material trace mismatch",
        "missing, subnormal mismatch, NaN, and +/-Inf evidence",
    ),
)
_replace_axes(
    "PLAN:factory-certificate:order-and-rank",
    AxisRange(
        "problem_trace_order",
        "problem order below the certificate-selected order",
        ("T-1 (certificate order - 1)", "T (certificate order) and T+1"),
        "problem order far above the certificate-selected order",
        "negative, missing, and very large order",
    ),
    AxisRange(
        "perturbation",
        "perturbation whose independently computed algebraic rank is below certificate multiplicity",
        ("T-1 (multiplicity - 1)", "T (multiplicity) and T+1"),
        "perturbation whose algebraic rank is far above certificate multiplicity",
        "zero, rank-one, and dimension-rank perturbations",
    ),
)
_replace_axes(
    "PLAN:frozen:probe-energy-range",
    AxisRange(
        "probe_component",
        "ordinary finite non-unit probe component",
        (
            "finite component one ULP below the per-probe runtime limit",
            "finite component at and one ULP above the per-probe runtime limit",
        ),
        "finite component whose square exceeds the runtime maximum",
        "zero, minimum subnormal, and largest constructor-valid finite components",
    ),
    AxisRange(
        "probe_count",
        "small positive number of real frozen-probe rows",
        ("one row", "two rows"),
        "many rows whose finite component squares accumulate near overflow",
        "1, 2, 3, 257, and 10000 constructor-valid rows",
    ),
    AxisRange(
        "runtime_dtype",
        "supported runtime dtype able to accumulate the probe energy",
        ("narrow floating capability", "wide floating capability"),
        "runtime capability wider than probe storage",
        "supported narrow and wide floating runtime capabilities under their real contexts",
    ),
)
_replace_axes(
    "PLAN:runtime-range:product",
    AxisRange(
        "left",
        "zero or ordinary positive left operand",
        ("maximum/right one ULP below", "maximum/right and one ULP above"),
        "left operand far above maximum/right",
        "negative, NaN, and +/-Inf",
    ),
    AxisRange(
        "right",
        "zero or ordinary positive right operand",
        ("maximum/left one ULP below", "maximum/left and one ULP above"),
        "right operand far above maximum/left",
        "negative, NaN, and +/-Inf",
    ),
    AxisRange(
        "maximum",
        "maximum comfortably above the product",
        ("product one ULP below", "product and one ULP above"),
        "maximum far below the product",
        "zero, minimum subnormal, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:runtime-range:sum",
    AxisRange(
        "left",
        "zero or ordinary positive left addend",
        ("maximum-right one ULP below", "maximum-right and one ULP above"),
        "left addend far above maximum-right",
        "negative, NaN, and +/-Inf",
    ),
    AxisRange(
        "right",
        "zero or ordinary positive right addend",
        ("maximum-left one ULP below", "maximum-left and one ULP above"),
        "right addend far above maximum-left",
        "negative, NaN, and +/-Inf",
    ),
    AxisRange(
        "maximum",
        "maximum comfortably above the sum",
        ("sum one ULP below", "sum and one ULP above"),
        "maximum far below the sum",
        "zero, minimum subnormal, NaN, and +/-Inf",
    ),
)
_replace_axes(
    "PLAN:frozen:intermediate-runtime-range",
    AxisRange(
        "total_probe_energy",
        "ordinary finite non-unit total probe energy from real frozen probes",
        (
            "runtime correction one ULP below the maximum",
            "runtime correction at and one ULP above the maximum",
        ),
        "probe energy producing a correction far above the runtime maximum",
        "zero, minimum subnormal, and largest constructor-valid probe energies",
    ),
    AxisRange(
        "order",
        "small positive certificate-selected series order",
        (
            "order one below the first overflowing accumulation",
            "first overflowing order and one above",
        ),
        "large valid order with a wide roundoff factor",
        "zero, one, and largest practical certificate-selected orders",
    ),
)
_replace_axes(
    "PLAN:runtime:expected-and-ulp-finite",
    AxisRange(
        "lambda_entry",
        "ordinary positive finite non-unit Lambda entry",
        (
            "entry whose analytic logdet is representable in the runtime dtype",
            "finite entry whose analytic logdet reaches a runtime cast boundary",
        ),
        "large positive finite entry near the constructor/runtime limit",
        "minimum positive subnormal and largest finite constructor-valid entries; derived expected infinity would require about 4.8e35 float32 diagonal entries and is a resource-bound ambiguity",
    ),
    AxisRange(
        "perturbation_entry",
        "ordinary finite non-unit perturbation entry",
        (
            "finite entry preserving a representable positive sigma",
            "finite entry whose resolved sigma reaches the runtime boundary",
        ),
        "large finite perturbation near the resolution limit",
        "positive/negative minimum subnormal and largest finite perturbations",
    ),
    AxisRange(
        "runtime_dtype",
        "supported runtime dtype used for rounding and ULP measurement",
        ("narrow floating capability", "wide floating capability"),
        "runtime capability wider than the problem storage",
        "supported narrow and wide floating runtime capabilities under their real contexts",
    ),
    AxisRange(
        "tolerance",
        "ordinary positive finite certificate tolerance",
        (
            "minimum positive constructor-valid tolerance",
            "tolerance at the measured runtime ULP",
        ),
        "large finite tolerance",
        "minimum positive subnormal and largest finite constructor-valid tolerances",
    ),
)
_replace_axes(
    "PLAN:runtime:total-error-budget",
    AxisRange(
        "certified_rho",
        "ordinary strict certified rho producing a finite analytic tail",
        ("rho yielding total error one ULP below tolerance", "rho yielding equality and one ULP above"),
        "certified rho near one producing a tail far above tolerance",
        "zero, minimum subnormal, and nextafter(1,0)",
    ),
    AxisRange(
        "max_abs_lambda_logdet",
        "ordinary finite non-unit certified base-logdet scale",
        ("scale yielding total error one ULP below tolerance", "scale yielding equality and one ULP above"),
        "large certified base scale producing roundoff far above tolerance",
        "zero, minimum subnormal, and largest constructor-valid bound",
    ),
    AxisRange(
        "tolerance",
        "tolerance comfortably above total error",
        ("total error one ULP below", "equality and one ULP above"),
        "tolerance far below total error",
        "zero, minimum subnormal, and largest finite tolerance",
    ),
)
_replace_axes(
    "PLAN:trace-factory:exact-evidence",
    AxisRange(
        "problem_trace_order",
        "problem order below the certificate-selected order",
        ("T-1 (certificate order - 1)", "T (certificate order) and T+1"),
        "problem order far above the selected order",
        "negative, missing, and very large order",
    ),
    AxisRange(
        "trace_evidence_value",
        "exact non-unit power traces",
        ("bitwise exact evidence", "one-ULP mismatch"),
        "material trace mismatch",
        "missing, subnormal mismatch, NaN, and +/-Inf evidence",
    ),
)
_replace_axes(
    "PLAN:frozen-factory:probe-presence-width",
    AxisRange(
        "probe_presence",
        "present immutable FrozenProbes payload",
        ("payload present", "payload absent"),
        "present payload with many probe rows",
        "None, empty, and invalid container capability",
    ),
    AxisRange(
        "probe_width",
        "width below problem dimension",
        ("n-1", "n and n+1"),
        "width far above problem dimension",
        "zero, empty, and maximum practical width",
    ),
)
_replace_axes(
    "PLAN:canonical-probes:runtime-finite",
    AxisRange(
        "probe_scalar",
        "ordinary finite non-unit probe component",
        (
            "largest finite value representable by the runtime dtype",
            "finite wider-dtype value that overflows when cast to the runtime dtype",
        ),
        "large finite component near the runtime limit",
        "zero, minimum subnormal, largest finite input, and finite cast-overflow input that produces infinity only after the production cast",
    ),
    AxisRange(
        "probe_dtype",
        "finite probe storage dtype with a non-unit component",
        ("float32 capability", "float64 capability"),
        "wider finite probe storage capability",
        "supported floating storage dtypes at their finite magnitude extremes",
    ),
    AxisRange(
        "runtime_dtype",
        "runtime dtype able to represent the probe",
        ("narrow runtime capability", "wide runtime capability"),
        "runtime capability wider than probe storage",
        "supported narrow and wide floating runtime capabilities under their real contexts",
    ),
)
_replace_axes(
    "PLAN:frozen:x-bound-runtime-range",
    AxisRange(
        "max_x_operator_norm",
        "ordinary finite non-unit certified operator-norm bound",
        ("largest representable runtime bound", "outward overflow boundary"),
        "bound near the runtime dtype maximum",
        "None, zero, minimum subnormal, largest finite, and finite cast-overflow bounds",
    ),
    AxisRange(
        "runtime_dtype",
        "runtime dtype able to represent the certified bound",
        ("narrow runtime capability", "wide runtime capability"),
        "runtime capability wider than the certificate storage",
        "supported narrow and wide floating runtime capabilities under their real contexts",
    ),
)
_replace_axes(
    "PLAN:runtime:base-scale-range",
    AxisRange(
        "max_abs_lambda_logdet",
        "ordinary finite non-unit certified base-logdet bound",
        ("largest representable runtime bound", "outward overflow boundary"),
        "bound near the runtime dtype maximum",
        "None, zero, minimum subnormal, largest finite, and finite cast-overflow bounds",
    ),
    AxisRange(
        "runtime_dtype",
        "runtime dtype able to represent the certified base scale",
        ("narrow runtime capability", "wide runtime capability"),
        "runtime capability wider than the certificate storage",
        "supported narrow and wide floating runtime capabilities under their real contexts",
    ),
)
_replace_axes(
    "PLAN:runtime:sigma-finite-and-positive",
    AxisRange(
        "lambda_entry",
        "ordinary finite non-unit diagonal entry",
        (
            "smallest positive constructor-valid entry",
            "large positive entry whose finite perturbation reaches the sigma boundary",
        ),
        "large finite diagonal entry",
        "minimum positive subnormal and largest finite positive constructor-valid entries",
    ),
    AxisRange(
        "perturbation_entry",
        "ordinary finite non-unit perturbation entry",
        (
            "entry preserving strict positivity",
            "finite entry making resolved sigma exactly zero and singular",
        ),
        "large perturbation making sigma indefinite",
        "positive/negative minimum subnormal and largest finite perturbations that overflow TwoSum",
    ),
    AxisRange(
        "matrix_path",
        "compact non-unit diagonal construction",
        ("diagonal path", "dense symmetric path"),
        "larger dense symmetric construction",
        "singular, indefinite, asymmetric, and overflowed resolved matrices from finite inputs",
    ),
)
_replace_axes(
    "PLAN:runtime:frozen-prerequisites-and-series",
    AxisRange(
        "max_x_operator_norm",
        "ordinary certified operator-norm bound below one",
        ("nextafter(0,+inf)", "nextafter(1,0) and 1"),
        "bound at or above the strict-convergence limit",
        "zero, minimum subnormal, nextafter(1,0), and largest constructor-valid finite bound",
    ),
    AxisRange(
        "probe_component",
        "ordinary finite non-unit frozen probe component",
        ("smallest finite component", "largest finite safe component"),
        "component whose power trace exceeds the runtime range",
        "zero, minimum subnormal, and largest constructor-valid finite components",
    ),
    AxisRange(
        "order",
        "small positive frozen-series order",
        ("1", "2"),
        "large positive order",
        "smallest and largest orders independently derived by valid certificates",
    ),
)


_STATIC_GATE_REASONS = {
    "EAGER:factor-projection:whitened-positive-spectrum": (
        "Every production fixture that crosses this local spectrum predicate "
        "is already refused by the downstream projection certificate or the "
        "complete low-rank contract. Changing this predicate alone changes "
        "only the diagnostic path, not the selected method or final refusal."
    ),
    "EAGER:factor-projection:error-budget": (
        "The final total-error predicate uses the same ceiling on the sum of "
        "three nonnegative component bounds. A valid total therefore implies "
        "this projection predicate; changing it alone can only select a "
        "diagnostic reason, not a method or refusal."
    ),
    "EAGER:factor-projection:finite-qr-arithmetic": (
        "This post-QR finite-output predicate has no deterministic refused "
        "fixture on the supported finite-input path. Column balancing first "
        "establishes finite inputs; QR failures raise, and the core/projected "
        "products run under over/invalid='raise'. Reaching this predicate with "
        "a nonfinite result therefore requires a QR/library capability fault, "
        "so it cannot honestly claim two executable mutation witnesses."
    ),
    "EAGER:factor-reduced:diagonal-certificate": (
        "At the independently realizable one-sided boundary, the complete "
        "low-rank path is already refused by the reduced-log error budget. "
        "Changing this local certificate alone therefore cannot change a "
        "selected method or final refusal."
    ),
    "EAGER:factor-reduced:qr-certificate": (
        "Production fixtures that cross this local QR certificate are already "
        "refused by the projection certificate or another downstream reduced "
        "certificate. No isolated mutation changes the selected method or "
        "final refusal."
    ),
    "EAGER:factor-reduced:acceptance-budget": (
        "The local aggregate boundary has no independently executable factor "
        "fixture that also satisfies exact reconstruction and every preceding "
        "certificate. Mutating it changes only a dominated diagnostic path."
    ),
    "EAGER:trace:actual-rho-strict": (
        "On the successful path, the following certificate-domain and "
        "coverage checks prove actual_rho <= certificate < 1. Changing this "
        "earlier strict-rho check alone can only select a diagnostic reason, "
        "not a method or refusal."
    ),
    "EAGER:factor-base:error-budget": (
        "The final total-error predicate uses the same ceiling on the sum of "
        "three nonnegative component bounds. A valid total therefore implies "
        "this base-budget predicate; changing it alone can only select a "
        "diagnostic reason, not a method or refusal."
    ),
    "EAGER:factor-base:condition-ceiling": (
        "The dense condition comparison is strictly dominated at its boundary "
        "by the base log-error budget. With eta=gamma(3*n)*condition and "
        "condition nextafter(1/sqrt(eps), -inf), the bound "
        "-n*log1p(-eta) is already greater than sqrt(eps), so both one-ULP "
        "condition mutations leave the certificate refused and only change "
        "its diagnostic reason. The diagonal path uses an infinite condition "
        "ceiling and exposes no finite outcome boundary."
    ),
    "PLAN:factory-certificate:strict-rho": (
        "This expression is an unconsumed validation call. The callee owns the "
        "rho domain, strict-convergence, and coverage gates; its return value "
        "does not select a PLAN method or refusal."
    ),
    "PLAN:certificate:rho-domain-and-coverage": (
        "Loosening this predicate at certified_rho=1 is dominated by the real "
        "choose_trace_order convergence check, so the selected PLAN outcome "
        "cannot change independently."
    ),
    "PLAN:warmup:rho-inputs-and-margin": (
        "A negative margin is rejected by the real RhoCertificate constructor "
        "before it can independently change PLAN routing."
    ),
    "PLAN:warmup:tail-fraction": (
        "The endpoint tail fractions are rejected by the real RhoCertificate "
        "budget contract before this local predicate can control an outcome."
    ),
    "PLAN:warmup:rho-roundoff-ceiling": (
        "A certified rho at or above one is rejected by the real "
        "RhoCertificate contract, which dominates this local comparison."
    ),
    "PLAN:canonical-probes:runtime-finite": (
        "FrozenProbes rejects nonfinite values, while overflow during the "
        "supported runtime cast raises before this predicate. There is no real "
        "direct input that reaches its refused side independently."
    ),
    "PLAN:measurement:lambda-logdet-finite": (
        "The dense lambda_logdet payload either returns a finite value or "
        "raises before this predicate. On the compact finite-SPD path, "
        "overflow of sum(log(lambda)) would require about 3.8e36 float32 "
        "entries or 2.5e305 float64 entries, beyond constructable resources. "
        "No real direct input can both reach this predicate and produce its "
        "refused side; changing it only changes defensive/capability "
        "diagnostics."
    ),
}


_EAGER_BALANCE_PREFIX = (
    "src/bayesmith/marginal/_logdet_eager.py::<module>."
    "_balanced_factor_columns::"
)
_EAGER_STATE_PREFIX = (
    "src/bayesmith/marginal/_logdet_eager.py::<module>.state_space_logdet::"
)
_PLAN_FROZEN_ENERGY_PREFIX = (
    "src/bayesmith/marginal/_logdet_plan.py::<module>."
    "_frozen_probe_energy_bounds::"
)


_STATIC_ATOM_REASONS: Mapping[str, Mapping[str, str]] = {
    "EAGER:factor-balance:exact-power-of-two-reversibility": {
        f"{_EAGER_BALANCE_PREFIX}predicate_call_atom::508f419c8ad7bceb::0": (
            "Finite input columns and the computed power-of-two gauge do not "
            "provide a deterministic production fixture where only the "
            "scaled-left finiteness premise fails."
        ),
        f"{_EAGER_BALANCE_PREFIX}predicate_call_atom::633c4ba7bc098a19::0": (
            "Elementwise identity of the same unreachable scaled-left "
            "finiteness premise."
        ),
        f"{_EAGER_BALANCE_PREFIX}finite_predicate::633c4ba7bc098a19::0": (
            "Scanner alias of the same unreachable scaled-left finiteness "
            "premise."
        ),
        f"{_EAGER_BALANCE_PREFIX}predicate_call_atom::ff9f64daa7a7b618::0": (
            "Finite input columns and the computed power-of-two gauge do not "
            "provide a deterministic production fixture where only the "
            "scaled-right finiteness premise fails."
        ),
        f"{_EAGER_BALANCE_PREFIX}predicate_call_atom::09a90b48970dbb22::0": (
            "Elementwise identity of the same unreachable scaled-right "
            "finiteness premise."
        ),
        f"{_EAGER_BALANCE_PREFIX}finite_predicate::09a90b48970dbb22::0": (
            "Scanner alias of the same unreachable scaled-right finiteness "
            "premise."
        ),
    },
    "EAGER:state-space:payload-domain": {
        f"{_EAGER_STATE_PREFIX}predicate_call_atom::5662962bf2f78843::0": (
            "For the supported small block-chain payload, every component is "
            "finite and math.fsum cannot overflow without an impractically "
            "large allocation; no honest isolated finite-input fixture exists."
        ),
        f"{_EAGER_STATE_PREFIX}finite_predicate::5662962bf2f78843::0": (
            "Scanner alias of the same resource-dominated final-total "
            "finiteness premise."
        ),
    },
    "PLAN:frozen:probe-energy-range": {
        f"{_PLAN_FROZEN_ENERGY_PREFIX}predicate_call_atom::ec9578c6de121206::0": (
            "math.fsum overflow is caught before this premise and supported "
            "float32 casting can exceed the dtype ceiling while remaining "
            "finite, so this finiteness atom has no isolated false fixture."
        ),
        f"{_PLAN_FROZEN_ENERGY_PREFIX}finite_predicate::ec9578c6de121206::0": (
            "Scanner alias of the same unreachable probe-energy finiteness "
            "premise."
        ),
    },
}


_ATOM_ISOLATION_AMBIGUITIES: Mapping[str, Mapping[str, str]] = {}


def dynamic_atom_ids(entry: GateEntry) -> tuple[str, ...]:
    """Return conjunction atoms that admit a real isolated production case."""
    if entry.mutation_mode is not MutationMode.TWO_SIDED:
        return ()
    static_ids = set(entry.static_atom_reasons)
    return tuple(
        atom_id
        for atom_id in entry.conjunction_atom_ids
        if atom_id not in static_ids
    )


def isolatable_atom_ids(entry: GateEntry) -> tuple[str, ...]:
    """Return dynamic atoms with an honest finite production fixture."""
    ambiguous_ids = set(entry.atom_isolation_ambiguities)
    return tuple(
        atom_id
        for atom_id in dynamic_atom_ids(entry)
        if atom_id not in ambiguous_ids
    )


def _entry(seed: _Seed) -> GateEntry:
    slug = re.sub(r"[^a-z0-9]+", "_", seed.gate_id.lower()).strip("_")
    links = _GATE_SOURCE_LINKS[seed.gate_id]
    source_candidate_ids = tuple(candidate_id for candidate_id, _ in links)
    conjunction_atom_ids = _GATE_ATOM_LINKS.get(seed.gate_id, ())
    mutation_target_ids = tuple(
        dict.fromkeys(
            (
                *source_candidate_ids,
                *conjunction_atom_ids,
                *_GATE_EXTRA_TARGETS.get(seed.gate_id, ()),
            )
        )
    )
    anchors = _DECLARED_SOURCE_ANCHORS[seed.gate_id]
    source_classifications = _DECLARED_SOURCE_CLASSIFICATIONS[seed.gate_id]
    metadata = GATE_METADATA[seed.gate_id]
    static_reason = _STATIC_GATE_REASONS.get(seed.gate_id)
    static_atom_reasons = MappingProxyType(
        dict(_STATIC_ATOM_REASONS.get(seed.gate_id, {}))
    )
    atom_isolation_ambiguities = MappingProxyType(
        dict(_ATOM_ISOLATION_AMBIGUITIES.get(seed.gate_id, {}))
    )
    mutation_mode = (
        MutationMode.STATIC_ONLY
        if static_reason is not None
        else MutationMode.TWO_SIDED
    )
    return GateEntry(
        gate_id=seed.gate_id,
        source_candidate_ids=source_candidate_ids,
        mutation_target_ids=mutation_target_ids,
        conjunction_atom_ids=conjunction_atom_ids,
        dependencies=_dependencies(seed.gate_id),
        expected_source_syntax=tuple(syntax for _, syntax in links),
        source_classifications=source_classifications,
        source_anchors=anchors,
        module=seed.module,
        quantity=metadata.quantity,
        threshold=metadata.threshold,
        provenance=metadata.provenance,
        admitted_outcome=metadata.admitted_outcome,
        refused_outcome=metadata.refused_outcome,
        oracle=metadata.oracle,
        axes=metadata.axes,
        fixture_scale_policy=metadata.fixture_scale_policy,
        mutation_mode=mutation_mode,
        tighten_witness=(f"tighten_{slug}" if static_reason is None else ""),
        loosen_witness=(f"loosen_{slug}" if static_reason is None else ""),
        static_reason=static_reason,
        static_atom_reasons=static_atom_reasons,
        atom_isolation_ambiguities=atom_isolation_ambiguities,
    )


GATE_REGISTRY = tuple(_entry(seed) for seed in _SEEDS)


@dataclass(frozen=True, slots=True)
class NonGateEntry:
    site_id: str
    source_candidate_ids: tuple[str, ...]
    classification: CandidateClassification
    reason: str
    tighten_witness: None = None
    loosen_witness: None = None


NON_GATE_REGISTRY = (
    NonGateEntry(
        "POLICY:eager-unconditional-resampling",
        (
            "src/bayesmith/marginal/_logdet_eager.py::<module>.resampled_trace_logdet::raise::1f5ac479fb5ea756::0",
        ),
        CandidateClassification.POLICY_STATIC_REFUSAL,
        "Unconditional resampling refusal has no numeric predicate to move.",
    ),
    NonGateEntry(
        "POLICY:ladder-rung8-unconditional",
        (
            "src/bayesmith/marginal/_logdet_ladder.py::<module>.check_logdet_premises::policy_literal::bfdbc42f821a1db3::0",
        ),
        CandidateClassification.POLICY_STATIC_REFUSAL,
        "The final ladder refusal is unconditional after all live numeric routes fail.",
    ),
    NonGateEntry(
        "STATIC:plan-constructor-token",
        (
            "src/bayesmith/marginal/_logdet_plan.py::<module>.TraceLogPlan.__init__::decision_predicate::dbd899e0214ad067::0",
            "src/bayesmith/marginal/_logdet_plan.py::<module>.FrozenTraceLogPlan.__init__::decision_predicate::dbd899e0214ad067::0",
        ),
        CandidateClassification.POLICY_STATIC_REFUSAL,
        "Private capability-token identity is structural, not a numerical boundary.",
    ),
    NonGateEntry(
        "STATIC:map-mode-size-clamp",
        (
            "src/bayesmith/diagnose/map.py::<module>.map_estimate::clamp_selector::64f9a1dc656957b9::0",
        ),
        CandidateClassification.STATIC_SELECTOR,
        "max(mode.size, 1) is an unreachable empty-latent safeguard after an earlier return.",
    ),
)


_NON_GATE_SOURCE_SYNTAX = {
    "POLICY:eager-unconditional-resampling": (
        "raise ResamplingRefused('Per-call probe resampling makes the log determinant noisy, breaks HMC reversibility, and is always refused. Supply FrozenProbes instead.')",
    ),
    "POLICY:ladder-rung8-unconditional": ("False",),
    "STATIC:plan-constructor-token": ("token is not _PLAN_TOKEN",) * 2,
    "STATIC:map-mode-size-clamp": ("max(mode.size, 1)",),
}


def validate_non_gate_registry(
    entries: Sequence[NonGateEntry],
    candidates: Sequence[SourceCandidate],
    manifest: Sequence[ManifestEntry],
) -> None:
    """Validate stable, reasoned exclusions in both registry-to-manifest directions."""
    errors: list[str] = []
    site_ids = [entry.site_id for entry in entries]
    if len(site_ids) != len(set(site_ids)):
        errors.append("duplicate non-gate site IDs")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    manifest_by_id = {item.candidate_id: item for item in manifest}
    linked_policy: set[str] = set()
    for entry in entries:
        canonical_syntax = _NON_GATE_SOURCE_SYNTAX.get(entry.site_id)
        if canonical_syntax is None:
            errors.append(f"{entry.site_id}: unreasoned non-gate declaration")
            continue
        if not entry.reason.strip():
            errors.append(f"{entry.site_id}: non-gate reason is empty")
        if entry.tighten_witness is not None or entry.loosen_witness is not None:
            errors.append(f"{entry.site_id}: non-gate must not fabricate witnesses")
        if len(entry.source_candidate_ids) != len(canonical_syntax):
            errors.append(f"{entry.site_id}: non-gate source anchors are incomplete")
        for candidate_id, expected_syntax in zip(
            entry.source_candidate_ids, canonical_syntax
        ):
            candidate = candidate_by_id.get(candidate_id)
            classified = manifest_by_id.get(candidate_id)
            if candidate is None:
                errors.append(f"{entry.site_id}: non-gate source does not resolve")
            elif candidate.syntax != expected_syntax:
                errors.append(f"{entry.site_id}: non-gate source syntax changed")
            if (
                classified is None
                or classified.classification is not entry.classification
            ):
                errors.append(f"{entry.site_id}: non-gate classification differs")
            if entry.classification is CandidateClassification.POLICY_STATIC_REFUSAL:
                linked_policy.add(candidate_id)
    manifest_policy = {
        item.candidate_id
        for item in manifest
        if item.classification is CandidateClassification.POLICY_STATIC_REFUSAL
    }
    if linked_policy != manifest_policy:
        errors.append("policy/static manifest classification is not bidirectional")
    if errors:
        raise RegistryValidationError("\n".join(errors))


def resolve_source_locations(
    entry: GateEntry, candidates: Iterable[SourceCandidate]
) -> tuple[SourceLocation, ...]:
    """Resolve stable anchors to current, report-only line metadata."""
    candidate_tuple = tuple(candidates)
    locations: list[SourceLocation] = []
    for candidate_id in entry.source_candidate_ids:
        for candidate in candidate_tuple:
            if candidate.candidate_id == candidate_id:
                locations.append(
                    SourceLocation(
                        candidate_id=candidate.candidate_id,
                        module=candidate.module,
                        qualname=candidate.qualname,
                        family=candidate.family,
                        lineno=candidate.lineno,
                        syntax=candidate.syntax,
                    )
                )
    return tuple(locations)


def validate_registry(
    entries: Sequence[GateEntry],
    candidates: Sequence[SourceCandidate],
    manifest: Sequence[ManifestEntry],
) -> None:
    """Validate registry uniqueness, source ownership, and schema completeness."""
    errors: list[str] = []
    gate_ids = [entry.gate_id for entry in entries]
    gate_id_set = set(gate_ids)
    duplicate_ids = sorted(
        {gate_id for gate_id in gate_ids if gate_ids.count(gate_id) > 1}
    )
    if duplicate_ids:
        errors.append(f"duplicate registry IDs: {duplicate_ids}")

    manifest_ids = [item.candidate_id for item in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        errors.append("source candidate IDs are not uniquely classified")
    manifest_id_set = set(manifest_ids)
    manifest_by_id = {item.candidate_id: item for item in manifest}
    candidate_by_id = {item.candidate_id: item for item in candidates}
    repository_root = Path(__file__).resolve().parents[2]
    source_cache = {
        module: (repository_root / module).read_text()
        for module in {item.module for item in candidates}
    }
    source_indexes = {
        module: index_source_text(source, module)
        for module, source in source_cache.items()
    }
    claimed_candidate_ids: list[str] = []

    for entry in entries:
        canonical_links = _GATE_SOURCE_LINKS.get(entry.gate_id, ())
        canonical_source_ids = tuple(item[0] for item in canonical_links)
        canonical_syntax = tuple(item[1] for item in canonical_links)
        canonical_atoms = _GATE_ATOM_LINKS.get(entry.gate_id, ())
        canonical_targets = tuple(
            dict.fromkeys(
                (
                    *canonical_source_ids,
                    *canonical_atoms,
                    *_GATE_EXTRA_TARGETS.get(entry.gate_id, ()),
                )
            )
        )
        if entry.source_candidate_ids != canonical_source_ids:
            errors.append(f"{entry.gate_id}: canonical source roots differ")
        if entry.expected_source_syntax != canonical_syntax:
            errors.append(f"{entry.gate_id}: canonical source syntax differs")
        if entry.source_anchors != _DECLARED_SOURCE_ANCHORS.get(entry.gate_id, ()):
            errors.append(f"{entry.gate_id}: canonical source anchors differ")
        if entry.source_classifications != _DECLARED_SOURCE_CLASSIFICATIONS.get(
            entry.gate_id, ()
        ):
            errors.append(f"{entry.gate_id}: canonical source classifications differ")
        if entry.conjunction_atom_ids != canonical_atoms:
            errors.append(f"{entry.gate_id}: canonical conjunction atoms differ")
        canonical_static_atom_reasons = _STATIC_ATOM_REASONS.get(entry.gate_id, {})
        if dict(entry.static_atom_reasons) != dict(canonical_static_atom_reasons):
            errors.append(f"{entry.gate_id}: canonical static atom reasons differ")
        unknown_static_atoms = set(entry.static_atom_reasons) - set(
            entry.conjunction_atom_ids
        )
        if unknown_static_atoms:
            errors.append(
                f"{entry.gate_id}: static atom is not an owned conjunction atom: "
                f"{sorted(unknown_static_atoms)}"
            )
        if any(not reason.strip() for reason in entry.static_atom_reasons.values()):
            errors.append(f"{entry.gate_id}: static atom reason is empty")
        canonical_atom_ambiguities = _ATOM_ISOLATION_AMBIGUITIES.get(
            entry.gate_id, {}
        )
        if dict(entry.atom_isolation_ambiguities) != dict(
            canonical_atom_ambiguities
        ):
            errors.append(
                f"{entry.gate_id}: canonical atom isolation ambiguity reasons differ"
            )
        invalid_ambiguity_atoms = set(entry.atom_isolation_ambiguities) - (
            set(entry.conjunction_atom_ids) - set(entry.static_atom_reasons)
        )
        if invalid_ambiguity_atoms:
            errors.append(
                f"{entry.gate_id}: atom isolation ambiguity is not an owned "
                f"dynamic atom: {sorted(invalid_ambiguity_atoms)}"
            )
        if any(
            not reason.strip()
            for reason in entry.atom_isolation_ambiguities.values()
        ):
            errors.append(f"{entry.gate_id}: atom isolation ambiguity reason is empty")
        if entry.mutation_target_ids != canonical_targets:
            errors.append(f"{entry.gate_id}: canonical mutation targets differ")
        if not entry.gate_id:
            errors.append("registry entry has an empty gate ID")
        if not entry.source_anchors:
            errors.append(f"{entry.gate_id}: missing source anchors")
        if not entry.source_candidate_ids:
            errors.append(f"{entry.gate_id}: missing source candidate IDs")
        claimed_candidate_ids.extend(entry.source_candidate_ids)
        if any(anchor.module != entry.module for anchor in entry.source_anchors):
            errors.append(f"{entry.gate_id}: source ownership does not match module")
        locations = resolve_source_locations(entry, candidates)
        if not locations:
            errors.append(f"{entry.gate_id}: source anchor resolves to no candidate")
        location_by_id = {location.candidate_id: location for location in locations}
        if len(entry.expected_source_syntax) != len(entry.source_candidate_ids):
            errors.append(f"{entry.gate_id}: source syntax assertions are incomplete")
        if len(entry.source_anchors) != len(entry.source_candidate_ids):
            errors.append(f"{entry.gate_id}: declared source anchors are incomplete")
        if len(entry.source_classifications) != len(entry.source_candidate_ids):
            errors.append(
                f"{entry.gate_id}: declared source classifications are incomplete"
            )
        for candidate_id, expected_syntax in zip(
            entry.source_candidate_ids, entry.expected_source_syntax
        ):
            location = location_by_id.get(candidate_id)
            if location is not None and location.syntax != expected_syntax:
                errors.append(
                    f"{entry.gate_id}: normalized source syntax changed for {candidate_id}"
                )
        if any(location.candidate_id not in manifest_id_set for location in locations):
            errors.append(f"{entry.gate_id}: source anchor is not classified")
        for candidate_id, declared_anchor, declared_classification in zip(
            entry.source_candidate_ids,
            entry.source_anchors,
            entry.source_classifications,
        ):
            candidate = candidate_by_id.get(candidate_id)
            classified = manifest_by_id.get(candidate_id)
            if candidate is not None and (
                candidate.module != declared_anchor.module
                or candidate.qualname != declared_anchor.qualname
                or candidate.family is not declared_anchor.family
            ):
                errors.append(
                    f"{entry.gate_id}: candidate does not match its independently declared module/qualname/family anchor"
                )
            if (
                classified is not None
                and classified.classification is not declared_classification
            ):
                errors.append(
                    f"{entry.gate_id}: candidate classification differs from its independent declaration"
                )
            if classified is not None and classified.classification not in {
                CandidateClassification.NUMERICAL_GATE,
                CandidateClassification.NUMERICAL_SAFETY,
            }:
                errors.append(
                    f"{entry.gate_id}: declared anchor is not semantically classified as a numerical gate/safety premise"
                )
        if not entry.mutation_target_ids:
            errors.append(f"{entry.gate_id}: missing explicit mutation targets")
        if any(dependency not in gate_id_set for dependency in entry.dependencies):
            errors.append(f"{entry.gate_id}: dependency names an unknown gate")
        if any(
            candidate_id not in manifest_id_set
            for candidate_id in (
                *entry.mutation_target_ids,
                *entry.conjunction_atom_ids,
            )
        ):
            errors.append(
                f"{entry.gate_id}: mutation/conjunction target is unclassified"
            )
        for target_id in canonical_targets:
            target = candidate_by_id.get(target_id)
            classified = manifest_by_id.get(target_id)
            if target is None:
                errors.append(
                    f"{entry.gate_id}: canonical mutation target does not resolve"
                )
                continue
            indexed = source_indexes.get(target.module, {}).get(target_id)
            if indexed is None or indexed[0] != target:
                errors.append(
                    f"{entry.gate_id}: canonical mutation target identity differs"
                )
            if classified is None or classified.syntax != target.syntax:
                errors.append(
                    f"{entry.gate_id}: canonical mutation target syntax differs"
                )
            elif classified.classification not in {
                CandidateClassification.NUMERICAL_GATE,
                CandidateClassification.NUMERICAL_SAFETY,
            }:
                errors.append(
                    f"{entry.gate_id}: canonical mutation target classification differs"
                )
        root_nodes: list[tuple[SourceCandidate, object]] = []
        for candidate_id in entry.source_candidate_ids:
            candidate = candidate_by_id.get(candidate_id)
            if candidate is None:
                continue
            try:
                indexed_candidate, node = source_indexes[candidate.module][candidate_id]
            except KeyError:
                errors.append(
                    f"{entry.gate_id}: declared source target does not resolve exactly once"
                )
                continue
            if indexed_candidate != candidate:
                errors.append(
                    f"{entry.gate_id}: indexed candidate metadata differs from census"
                )
            root_nodes.append((candidate, node))
        for atom_id in entry.conjunction_atom_ids:
            atom = candidate_by_id.get(atom_id)
            if atom is None:
                continue
            try:
                _, atom_node = source_indexes[atom.module][atom_id]
            except KeyError:
                errors.append(
                    f"{entry.gate_id}: conjunction atom {atom_id} does not resolve exactly once"
                )
                continue
            contained = any(
                atom.module == root_candidate.module
                and atom.qualname == root_candidate.qualname
                and any(descendant is atom_node for descendant in ast.walk(root_node))
                for root_candidate, root_node in root_nodes
            )
            if not contained:
                errors.append(
                    f"{entry.gate_id}: conjunction atom {atom_id} is not an AST descendant of a declared root"
                )
        for location in locations:
            if not any(
                location.module == anchor.module
                and location.qualname == anchor.qualname
                and location.family is anchor.family
                for anchor in entry.source_anchors
            ):
                errors.append(
                    f"{entry.gate_id}: candidate ID violates its source anchor"
                )
        for field_name in (
            "quantity",
            "threshold",
            "admitted_outcome",
            "refused_outcome",
            "oracle",
        ):
            if not getattr(entry, field_name).strip():
                errors.append(f"{entry.gate_id}: empty {field_name}")
        if any(
            phrase in entry.threshold or phrase in entry.admitted_outcome
            for phrase in (
                "explicit source-domain boundary named by this gate",
                "named direct",
                "named ladder rung",
            )
        ):
            errors.append(
                f"{entry.gate_id}: generic/module-template metadata is forbidden"
            )
        if "bayesmith" in entry.oracle.lower():
            errors.append(f"{entry.gate_id}: oracle is not an independent oracle")
        if not entry.axes:
            errors.append(f"{entry.gate_id}: no relevant axes")
        for axis in entry.axes:
            if not all((axis.name, axis.low, *axis.endpoints, axis.high, axis.extreme)):
                errors.append(f"{entry.gate_id}: incomplete low/high/end/extreme range")
        if entry.mutation_mode is MutationMode.TWO_SIDED:
            if not entry.tighten_witness.strip():
                errors.append(f"{entry.gate_id}: missing tighten witness")
            if not entry.loosen_witness.strip():
                errors.append(f"{entry.gate_id}: missing loosen witness")
            if entry.static_reason is not None:
                errors.append(f"{entry.gate_id}: two-sided gate has a static reason")
        else:
            if not entry.static_reason:
                errors.append(f"{entry.gate_id}: static/policy entry needs a reason")
            if entry.tighten_witness or entry.loosen_witness:
                errors.append(
                    f"{entry.gate_id}: static/policy entry fabricated a witness"
                )
            if entry.static_atom_reasons:
                errors.append(
                    f"{entry.gate_id}: whole static gate must not duplicate static atom reasons"
                )
            if entry.atom_isolation_ambiguities:
                errors.append(
                    f"{entry.gate_id}: whole static gate cannot claim atom isolation ambiguities"
                )
        if (
            entry.fixture_scale_policy is FixtureScalePolicy.NON_UNIT_REQUIRED
            and not any(
                "non-unit" in axis.low or "non-unit" in axis.high for axis in entry.axes
            )
        ):
            errors.append(f"{entry.gate_id}: non-unit fixture policy lacks a range")

    duplicate_claims = sorted(
        {
            candidate_id
            for candidate_id in claimed_candidate_ids
            if claimed_candidate_ids.count(candidate_id) > 1
        }
    )
    if duplicate_claims:
        errors.append(
            f"source candidates claimed by multiple gates: {duplicate_claims}"
        )

    canonical_targets = {
        candidate_id for entry in entries for candidate_id in entry.mutation_target_ids
    }
    manifest_numerical = {
        item.candidate_id
        for item in manifest
        if item.classification
        in {
            CandidateClassification.NUMERICAL_GATE,
            CandidateClassification.NUMERICAL_SAFETY,
        }
    }
    if canonical_targets != manifest_numerical:
        errors.append(
            "bidirectional numerical manifest linkage differs from canonical registry targets"
        )

    if errors:
        raise RegistryValidationError("\n".join(errors))


if len(GATE_REGISTRY) != 104:
    raise RegistryValidationError(
        f"semantic registry expected 104 reviewed entries, found {len(GATE_REGISTRY)}"
    )
