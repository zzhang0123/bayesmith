"""Binary64 rounding boundaries for warmup rho certificates."""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

import bayesmith.marginal._logdet_plan as plan_module
from bayesmith.marginal.logdet import (
    FrozenProbes,
    LogDetProblem,
    RhoCertificate,
    audit_retained_power_traces,
    audit_retained_rho,
    certify_warmup_rho,
    lambda_logdet,
    make_frozen_trace_log_plan,
    make_trace_log_plan,
    spectral_radius,
    truncated_trace_logdet,
)

_NONUNIT_PRIOR_WIDTHS = np.array([1.3, 4.7])


def _remeasured_design_rho() -> float:
    """Remeasure the same rho through a dense, non-unit-width solve."""
    rotation, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(2, 2)))
    lambda_matrix = (rotation * _NONUNIT_PRIOR_WIDTHS**2) @ rotation.T
    perturbation = 0.2 * lambda_matrix
    action = np.linalg.solve(lambda_matrix.T, perturbation.T).T
    return float(np.max(np.abs(np.linalg.eigvals(action))))


def _remeasured_hadamard_rho() -> float:
    """Remeasure rho through the integration fixture's solve/eigensystem."""
    hadamard = np.ones((1, 1))
    while hadamard.shape[0] < 16:
        hadamard = np.block([[hadamard, hadamard], [hadamard, -hadamard]])
    lambda_matrix = 2.3 * np.eye(16)
    perturbation = lambda_matrix @ (0.125 * hadamard)
    action = np.linalg.solve(lambda_matrix.T, perturbation.T).T
    return float(np.max(np.abs(np.linalg.eigvals(action))))


def _remeasured_dense_qr_rho() -> float:
    """Remeasure rho through the independent n=100 integration grid."""
    n = 100
    widths = np.linspace(1.3, 2.3, n)
    basis, _ = np.linalg.qr(np.random.default_rng(8101).normal(size=(n, n)))
    eigenvalues = np.linspace(0.01, 0.05, n)
    square_root = np.diag(widths)
    lambda_matrix = np.diag(widths**2)
    perturbation = (
        square_root @ basis @ np.diag(eigenvalues) @ basis.T @ square_root
    )
    action = np.linalg.solve(lambda_matrix.T, perturbation.T).T
    return float(np.max(np.abs(np.linalg.eigvals(action))))


def _ill_conditioned_remeasurement() -> tuple[np.ndarray, np.ndarray, float]:
    """A symmetric payload whose solve drift is amplified by cond(Lambda)."""
    rng = np.random.default_rng(848)
    rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    lambda_eigenvalues = np.array([1.0, 1.0e6])
    lambda_matrix = (rotation * lambda_eigenvalues) @ rotation.T
    lambda_sqrt = (rotation * np.sqrt(lambda_eigenvalues)) @ rotation.T
    perturbation_basis, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    perturbation = (
        lambda_sqrt
        @ perturbation_basis
        @ np.diag([0.01, 0.2])
        @ perturbation_basis.T
        @ lambda_sqrt
    )
    perturbation = (perturbation + perturbation.T) / 2.0
    return lambda_matrix, perturbation, spectral_radius(lambda_matrix, perturbation)


def test_margin_zero_certificate_covers_ordinary_rho_remeasurement_roundoff():
    """Dropping the arithmetic envelope makes an ordinary remeasurement fail."""
    remeasured = _remeasured_design_rho()
    certificate = certify_warmup_rho(
        [0.2],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=_NONUNIT_PRIOR_WIDTHS.size,
    )

    assert certificate.certified_rho >= remeasured
    assert audit_retained_rho([remeasured], certificate).passed is True


def test_margin_zero_certificate_covers_hadamard_eigensolver_remeasurement():
    """The dimension-scaled envelope covers this ordinary dense solve."""
    remeasured = _remeasured_hadamard_rho()
    certificate = certify_warmup_rho(
        [0.5],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=16,
    )

    assert certificate.certified_rho >= remeasured
    assert audit_retained_rho([remeasured], certificate).passed is True


def test_margin_zero_certificate_covers_dense_qr_eigensolver_remeasurement():
    """A fixed ULP allowance undercovers the observed ambient dense solve."""
    remeasured = _remeasured_dense_qr_rho()
    certificate = certify_warmup_rho(
        [0.05],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=100,
    )

    assert certificate.certified_rho >= remeasured
    assert audit_retained_rho([remeasured], certificate).passed is True


def test_roundoff_envelope_is_the_standard_multiplicity_scaled_gamma_n():
    """A fixed relative allowance is not an arithmetic error bound."""
    measured = 0.2
    multiplicity = 16
    eps = np.finfo(np.float64).eps
    gamma_n = multiplicity * eps / (1.0 - multiplicity * eps)

    certificate = certify_warmup_rho(
        [measured],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=multiplicity,
    )

    expected = np.nextafter(measured + abs(measured) * gamma_n, np.inf)
    assert certificate.certified_rho == expected


@pytest.mark.parametrize("multiplicity", [1, np.int64(2)])
def test_certificate_normalizes_index_multiplicity_once(multiplicity):
    """Python and NumPy integers take one canonical path through the proof."""
    certificate = certify_warmup_rho(
        [0.0], margin=0.0, tolerance=1.0e-6, multiplicity=multiplicity
    )

    assert type(certificate.multiplicity) is int
    assert certificate.multiplicity == int(multiplicity)


@pytest.mark.parametrize(
    "multiplicity",
    [True, 1.0, 1.5, np.nan, np.inf],
    ids=("bool", "float-int", "fraction", "nan", "inf"),
)
def test_certificate_rejects_non_index_multiplicity(multiplicity):
    """A non-index is a type error, never a dimension-changing coercion."""
    with pytest.raises(TypeError, match="multiplicity"):
        certify_warmup_rho(
            [0.0], margin=0.0, tolerance=1.0e-6, multiplicity=multiplicity
        )


@pytest.mark.parametrize(
    "multiplicity", [0, -1, 2**52, 10**1000], ids=("zero", "negative", "limit", "huge")
)
def test_certificate_rejects_out_of_range_integer_multiplicity(multiplicity):
    """An integer outside gamma_n's domain is a value error."""
    with pytest.raises(ValueError, match="multiplicity"):
        certify_warmup_rho(
            [0.0], margin=0.0, tolerance=1.0e-6, multiplicity=multiplicity
        )


@pytest.mark.parametrize("multiplicity", [True, 1.0, 1.5, np.nan, np.inf])
def test_direct_rho_certificate_rejects_non_index_multiplicity(multiplicity):
    """Direct construction reports the same type error as the factory."""
    with pytest.raises(TypeError, match="multiplicity"):
        _direct_rho_certificate(multiplicity)


@pytest.mark.parametrize("multiplicity", [0, -1, 2**52, 10**1000])
def test_direct_rho_certificate_rejects_out_of_range_multiplicity(multiplicity):
    """Direct construction reports the same value error as the factory."""
    with pytest.raises(ValueError, match="multiplicity"):
        _direct_rho_certificate(multiplicity)


def test_runtime_plan_factories_reject_an_overflowing_expected_sigma_sum():
    """A finite Lambda/P pair cannot turn its expected-logdet ULP into NaN."""
    lam = np.array([1.4e308])
    perturbation = np.array([0.7e308])
    rho = 0.5
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=0.01,
        multiplicity=1,
        lambda_logdets=[lambda_logdet(lam)],
        x_operator_norms=[rho],
    )
    traces = tuple(rho**power for power in range(1, certificate.order + 1))
    trace_problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    frozen_problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=FrozenProbes([[1.0]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    for factory, problem in (
        (make_trace_log_plan, trace_problem),
        (make_frozen_trace_log_plan, frozen_problem),
    ):
        with (
            np.errstate(over="raise", invalid="raise"),
            pytest.raises(
                ValueError, match=r"runtime precision.*Lambda \+ perturbation"
            ),
        ):
            factory(problem, certificate)


def test_runtime_plan_factories_reject_an_understated_lambda_logdet_bound():
    """The runtime roundoff proof must cover the current Lambda base term."""
    n = 100
    rho = 0.2
    lam = np.full(n, 2.0)
    perturbation = rho * lam
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=n,
        lambda_logdets=[0.0],
        x_operator_norms=[rho],
    )
    x = perturbation / lam
    power = np.ones_like(x)
    traces = []
    for _ in range(certificate.order):
        power = power * x
        traces.append(float(np.sum(power)))
    trace_problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=traces,
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    frozen_problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=FrozenProbes([np.ones(n)]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    for factory, problem in (
        (make_trace_log_plan, trace_problem),
        (make_frozen_trace_log_plan, frozen_problem),
    ):
        with (
            jax.enable_x64(True),
            pytest.raises(
                ValueError, match="lambda-logdet scale certificate.*understates"
            ),
        ):
            factory(problem, certificate)


def test_trace_plan_rounds_the_analytic_tail_bound_outward(monkeypatch):
    """A raw tail equal to tolerance is an inward-bound acceptance mutant."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.0,
        tolerance=1.0,
        multiplicity=1,
        lambda_logdets=[0.0],
    )
    problem = LogDetProblem(
        np.array([1.0]),
        np.array([0.0]),
        exact_power_traces=(),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    raw_tail = certificate.tolerance
    assert np.nextafter(raw_tail, np.inf) > certificate.tolerance
    monkeypatch.setattr(
        plan_module, "whole_trace_log_tail_bound", lambda *_: raw_tail
    )

    with jax.enable_x64(True), pytest.raises(
        ValueError, match="analytic tail plus conservative roundoff"
    ):
        make_trace_log_plan(problem, certificate)


def test_frozen_plan_strictly_rejects_an_understated_x_norm_bound():
    """Consumer validation cannot grant a tolerance absent from the proof."""
    rho = 0.2
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[rho * (1.0 - 5.0e-13)],
    )
    problem = LogDetProblem(
        np.array([1.0]),
        np.array([rho]),
        frozen_probes=FrozenProbes([[1.0]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with (
        jax.enable_x64(True),
        pytest.raises(ValueError, match=r"\|X\| operator-norm.*understates"),
    ):
        make_frozen_trace_log_plan(problem, certificate)


def test_warmup_scale_bounds_cover_one_ulp_remeasurement_drift():
    """Producer-side outward rounding supports strict scale consumers."""
    rho = 0.2
    lam = np.array([2.0])
    perturbation = rho * lam
    actual_base_scale = abs(lambda_logdet(lam))
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=1,
        lambda_logdets=[np.nextafter(actual_base_scale, 0.0)],
        x_operator_norms=[np.nextafter(rho, 0.0)],
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=FrozenProbes([[1.0]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(True):
        plan = make_frozen_trace_log_plan(problem, certificate)

    assert plan.order == certificate.order


def test_runtime_plan_normalizes_lambda_logdet_measurement_failure(monkeypatch):
    """Base-scale measurement arithmetic must fail through the plan contract."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=1,
        lambda_logdets=[0.0],
    )
    problem = LogDetProblem(
        np.array([1.0]),
        np.array([0.0]),
        exact_power_traces=(),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    def fail_lambda_logdet(_):
        raise FloatingPointError("synthetic logdet overflow")

    monkeypatch.setattr(plan_module, "lambda_logdet", fail_lambda_logdet)
    with pytest.raises(ValueError, match="finite resolved lambda-logdet arithmetic"):
        make_trace_log_plan(problem, certificate)


def test_frozen_plan_normalizes_an_overflowing_x_measurement():
    """A subnormal Lambda division must be a refusal, never a raw FPE."""
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    lam = np.array([tiny])
    perturbation = np.array([1.0])
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=1,
        lambda_logdets=[float(np.log(tiny))],
        x_operator_norms=[1.0],
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=FrozenProbes([[1.0]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with (
        jax.enable_x64(True),
        np.errstate(all="raise"),
        pytest.raises(ValueError, match=r"\|X\| operator-norm.*finite"),
    ):
        make_frozen_trace_log_plan(problem, certificate)


def test_retained_trace_audit_marks_an_overflowing_x_as_a_violation():
    """Invalid retained trace arithmetic belongs in the audit report."""
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    lam = np.array([tiny])
    perturbation = np.array([1.0])
    certificate = certify_warmup_rho(
        [0.0], margin=0.0, tolerance=1.0e-6, multiplicity=1
    )
    problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=(),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with np.errstate(all="raise"):
        report = audit_retained_power_traces([problem], certificate)

    assert report.passed is False
    assert report.violations == (0,)


def test_trace_plan_normalizes_invalid_power_trace_arithmetic(monkeypatch):
    """The factory's trace-evidence check must translate arithmetic failure."""
    tiny = np.nextafter(np.float64(0.0), np.float64(1.0))
    certificate = certify_warmup_rho(
        [0.0], margin=0.0, tolerance=1.0e-6, multiplicity=1
    )
    problem = LogDetProblem(
        np.array([tiny]),
        np.array([1.0]),
        exact_power_traces=(),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )
    monkeypatch.setattr(plan_module, "_validate_plan_certificate", lambda *_: None)

    with (
        np.errstate(all="raise"),
        pytest.raises(ValueError, match="exact power-trace evidence"),
    ):
        make_trace_log_plan(problem, certificate)


def test_frozen_plan_normalizes_an_overflowing_series_scale():
    """A finite norm bound whose certified powers overflow is a refusal."""
    certificate = certify_warmup_rho(
        [0.5],
        margin=0.0,
        tolerance=0.1,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[1.0e308],
    )
    assert certificate.order >= 2
    problem = LogDetProblem(
        np.array([1.0]),
        np.array([0.5]),
        frozen_probes=FrozenProbes([[1.0]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with (
        jax.enable_x64(True),
        np.errstate(all="raise"),
        pytest.raises(ValueError, match="finite frozen series scale"),
    ):
        make_frozen_trace_log_plan(problem, certificate)


def test_frozen_plan_normalizes_overflowing_probe_energy():
    """Finite probes may still have an unrepresentable squared energy."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=1,
        lambda_logdets=[float(np.log(2.0))],
        x_operator_norms=[0.0],
    )
    problem = LogDetProblem(
        np.array([2.0]),
        np.array([0.0]),
        frozen_probes=FrozenProbes([[1.0e308]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with (
        jax.enable_x64(True),
        np.errstate(all="raise"),
        pytest.raises(ValueError, match="finite frozen probe energy"),
    ):
        make_frozen_trace_log_plan(problem, certificate)


def test_float32_frozen_plan_rejects_probe_overflow_during_canonical_capture():
    """A finite binary64 probe must not become infinity in the JAX runtime."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.999999,
        tolerance=120_000.0,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[0.0],
    )
    assert certificate.order == 16
    problem = LogDetProblem(
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        frozen_probes=FrozenProbes([[4.0e38]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with (
        jax.enable_x64(False),
        np.errstate(all="raise"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("error")
        with pytest.raises(
            ValueError, match=r"frozen probes.*finite.*runtime float32"
        ):
            make_frozen_trace_log_plan(problem, certificate)


def test_float32_frozen_plan_rejects_probe_energy_beyond_runtime_range():
    """A representable probe can still overflow ``v * (X @ v)``."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.999999,
        tolerance=120_000.0,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[0.5],
    )
    assert certificate.order == 16
    problem = LogDetProblem(
        np.array([1.0], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
        frozen_probes=FrozenProbes([[1.0e20]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False), pytest.raises(
        ValueError, match=r"runtime float32 range.*probe energy"
    ):
        make_frozen_trace_log_plan(problem, certificate)


def test_float32_frozen_plan_rejects_an_unrepresentable_certified_x_bound():
    """A dynamic X covered only in binary64 cannot enter a float32 plan."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.999999,
        tolerance=1.1e6,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[1.0e39],
    )
    assert certificate.order == 1
    problem = LogDetProblem(
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        frozen_probes=FrozenProbes([[1.0]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False), pytest.raises(
        ValueError, match=r"runtime float32 range.*X operator-norm"
    ):
        make_frozen_trace_log_plan(problem, certificate)


def test_float32_frozen_plan_rejects_a_later_image_overflow():
    """Tiny probe energy cannot hide growth of an intermediate ``X**p @ v``."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.999999,
        tolerance=800_000.0,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[1.0e38],
    )
    assert certificate.order == 2
    problem = LogDetProblem(
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        frozen_probes=FrozenProbes([[1.0e-37]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False), pytest.raises(
        ValueError, match=r"runtime float32 range.*frozen image at power 2"
    ):
        make_frozen_trace_log_plan(problem, certificate)


def test_outward_helpers_preserve_a_positive_underflow_bound():
    """A positive exact proof quantity cannot become a zero upper bound."""
    smallest = np.nextafter(0.0, 1.0)

    assert plan_module._outward_product(smallest, 0.5) == smallest
    assert plan_module._outward_quotient(smallest, 2) == smallest


def test_frozen_plan_rejects_growth_hidden_by_probe_energy_underflow():
    """A zero energy mutant admits a nilpotent runtime that produces NaN."""
    dimension = 7
    action_scale = 1.0e100
    action = np.zeros((dimension, dimension))
    action[np.arange(dimension - 1), np.arange(1, dimension)] = action_scale
    probes = np.zeros((1, dimension))
    probes[0, -1] = 1.0e-200
    certificate = certify_warmup_rho(
        [0.5],
        margin=0.0,
        tolerance=0.01,
        multiplicity=dimension,
        lambda_logdets=[0.0],
        x_operator_norms=[action_scale],
    )
    assert certificate.order >= dimension - 1
    problem = LogDetProblem(
        np.eye(dimension),
        action,
        frozen_probes=FrozenProbes(probes),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(True), pytest.raises(
        ValueError, match="finite frozen series scale|runtime float64 range"
    ):
        make_frozen_trace_log_plan(problem, certificate)


@pytest.mark.parametrize("factory", [make_trace_log_plan, make_frozen_trace_log_plan])
def test_float32_plan_rejects_an_unrepresentable_certified_base_scale(factory):
    """The final base addition must be finite for every certified runtime value."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.0,
        tolerance=1.0e40,
        multiplicity=1,
        lambda_logdets=[1.0e39],
        x_operator_norms=[0.0],
    )
    assert certificate.order == 0
    problem = LogDetProblem(
        np.array([1.0], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
        exact_power_traces=() if factory is make_trace_log_plan else None,
        frozen_probes=(
            FrozenProbes([[1.0]])
            if factory is make_frozen_trace_log_plan
            else None
        ),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False), pytest.raises(
        ValueError, match=r"runtime float32 range.*lambda-logdet"
    ):
        factory(problem, certificate)


@pytest.mark.parametrize(
    "probe_value",
    [1.0 + 2.0**-25, np.nextafter(0.0, 1.0), 1],
    ids=("rounded", "underflow-to-zero", "normalized-integer"),
)
def test_float32_frozen_plan_captures_canonical_probe_values(probe_value):
    """Probe conversion happens once, so JAX x64 mode cannot change the target."""
    rho = 0.25
    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=0.01,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[rho],
    )
    problem = LogDetProblem(
        np.array([1.0], dtype=np.float32),
        np.array([rho], dtype=np.float32),
        frozen_probes=FrozenProbes([[probe_value]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(False), np.errstate(all="raise"):
        plan = make_frozen_trace_log_plan(problem, certificate)
        float32_result = np.asarray(plan(np.float32(0.0), np.array([rho], np.float32)))
    with jax.enable_x64(True), np.errstate(all="raise"):
        float64_context_result = np.asarray(
            plan(np.float32(0.0), np.array([rho], np.float32))
        )

    expected = np.asarray([[probe_value]], dtype=np.float32)
    assert plan._probes.dtype == expected.dtype.str
    assert np.array_equal(plan._probes.values, expected)
    assert np.array_equal(float32_result, float64_context_result)


def test_zero_expected_logdet_spacing_survives_strict_numpy_errstate():
    """The minimum-subnormal ULP at zero is valid, not an underflow bug."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=1,
        lambda_logdets=[0.0],
        x_operator_norms=[0.0],
    )
    problem = LogDetProblem(
        np.array([1.0]),
        np.array([0.0]),
        frozen_probes=FrozenProbes([[1.0]]),
        trace_order=certificate.order,
        certified_rho=certificate.certified_rho,
    )

    with jax.enable_x64(True), np.errstate(all="raise"):
        plan = make_frozen_trace_log_plan(problem, certificate)

    assert plan.order == 0


def test_runtime_plan_factories_reject_a_nonpositive_stored_sigma():
    """ULP validation must not turn a negative determinant into log-absolute."""
    lam = np.array([[1.0, 0.9985], [0.9985, 1.0]])
    direction = np.array([1.0, -1.0]) / np.sqrt(2.0)
    perturbation = (
        -(1.0 - 3.45e-14) * 0.0015 * np.outer(direction, direction)
    )
    rho = spectral_radius(lam, perturbation)
    sigma_sign, _ = np.linalg.slogdet(lam + perturbation)
    assert rho < 1.0
    assert sigma_sign == -1.0

    certificate = certify_warmup_rho(
        [rho],
        margin=0.0,
        tolerance=1.0e16,
        multiplicity=2,
        lambda_logdets=[lambda_logdet(lam)],
        x_operator_norms=[1.0],
    )
    assert certificate.order == 0
    trace_problem = LogDetProblem(
        lam,
        perturbation,
        exact_power_traces=(),
        trace_order=0,
        certified_rho=certificate.certified_rho,
    )
    frozen_problem = LogDetProblem(
        lam,
        perturbation,
        frozen_probes=FrozenProbes([[1.0, 1.0]]),
        trace_order=0,
        certified_rho=certificate.certified_rho,
    )

    for factory, problem in (
        (make_trace_log_plan, trace_problem),
        (make_frozen_trace_log_plan, frozen_problem),
    ):
        with pytest.raises(ValueError, match="positive definite Sigma"):
            factory(problem, certificate)


def _direct_rho_certificate(multiplicity):
    return RhoCertificate(
        measured_max=0.0,
        margin=0.0,
        certified_rho=float(np.nextafter(0.0, np.inf)),
        order=0,
        tolerance=1.0e-6,
        tail_tolerance=0.5e-6,
        multiplicity=multiplicity,
    )


def test_direct_rho_certificate_normalizes_a_numpy_integer():
    """The stored field is consistent even when construction bypasses the factory."""
    certificate = RhoCertificate(
        measured_max=0.0,
        margin=0.0,
        certified_rho=float(np.nextafter(0.0, np.inf)),
        order=0,
        tolerance=1.0e-6,
        tail_tolerance=0.5e-6,
        multiplicity=np.int64(2),
    )

    assert type(certificate.multiplicity) is int
    assert certificate.multiplicity == 2


def test_gamma_n_requires_multiplicity_eps_strictly_below_one():
    """The last valid integer and first invalid integer pin the denominator."""
    largest = 2**52 - 1
    certificate = certify_warmup_rho(
        [0.0], margin=0.0, tolerance=1.0e-6, multiplicity=largest
    )

    assert certificate.multiplicity == largest
    assert certificate.certified_rho == np.nextafter(0.0, np.inf)
    with pytest.raises(ValueError, match=r"multiplicity.*eps < 1"):
        certify_warmup_rho(
            [0.0], margin=0.0, tolerance=1.0e-6, multiplicity=2**52
        )


def test_conditioning_drift_needs_an_explicit_margin_not_a_scalar_rho_floor():
    """Gamma_n covers ordinary arithmetic, not solve error amplified by kappa."""
    lambda_matrix, perturbation, remeasured = _ill_conditioned_remeasurement()
    no_margin = certify_warmup_rho(
        [0.2], margin=0.0, tolerance=1.0e-6, multiplicity=2
    )

    assert np.linalg.cond(lambda_matrix) == pytest.approx(1.0e6, rel=1.0e-9)
    assert remeasured - 0.2 > 1.0e-12
    assert audit_retained_rho([remeasured], no_margin).passed is False
    with pytest.raises(ValueError, match="understates measured rho"):
        truncated_trace_logdet(
            lambda_matrix,
            perturbation,
            exact_power_traces=[0.0],
            order=1,
            rho=no_margin.certified_rho,
        )

    drift = remeasured - 0.2
    explicit_margin = certify_warmup_rho(
        [0.2],
        margin=2.0 * max(drift, 1.0e-10),
        tolerance=1.0e-6,
        multiplicity=2,
    )
    assert audit_retained_rho([remeasured], explicit_margin).passed is True


def test_zero_measurement_envelope_does_not_cover_materially_nonzero_rho():
    """Replacing the arithmetic envelope with a magic tolerance overcovers zero."""
    certificate = certify_warmup_rho(
        [0.0],
        margin=0.0,
        tolerance=1.0e-6,
        multiplicity=_NONUNIT_PRIOR_WIDTHS.size,
    )

    assert certificate.certified_rho == np.nextafter(0.0, np.inf)
    report = audit_retained_rho([5.0e-15], certificate)
    assert report.passed is False
    assert report.violations == (0,)


def test_certificate_refuses_when_rounding_envelope_has_no_headroom_below_one():
    """Omitting the envelope incorrectly certifies the float immediately below one."""
    measured = float(np.nextafter(1.0, 0.0))

    with pytest.raises(ValueError, match="does not certify rho < 1"):
        certify_warmup_rho(
            [measured],
            margin=0.0,
            tolerance=4.0e16,
            multiplicity=_NONUNIT_PRIOR_WIDTHS.size,
        )


def test_nonunit_width_certificate_selects_order_from_inflated_bound():
    """Selecting from the raw bound understates the order at its exact boundary."""
    measured = 0.37
    raw_order_four_tail = (
        _NONUNIT_PRIOR_WIDTHS.size
        * measured**5
        / (5 * (1.0 - measured))
    )
    certificate = certify_warmup_rho(
        [measured],
        margin=0.0,
        tolerance=2.0 * raw_order_four_tail,
        multiplicity=_NONUNIT_PRIOR_WIDTHS.size,
    )

    assert certificate.certified_rho > measured
    assert certificate.order == 5
