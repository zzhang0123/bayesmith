"""Independent numerical oracles used by the boundary-grid harness.

This module deliberately imports no :mod:`bayesmith` implementation.  The
production calls live in :mod:`tests.numerical_gates.boundary_cases`; keeping
the two dependency directions separate makes accidental self-oracles visible
to both readers and the completeness tests.
"""

from __future__ import annotations

import cmath
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from decimal import Decimal, localcontext
from enum import Enum
from fractions import Fraction
from itertools import permutations

import numpy as np


class NumericalVerdict(str, Enum):
    """The Phase-Two numerical comparison bands."""

    OK = "OK"
    WARN = "WARN"
    BAD = "BAD"


def relative_error(actual: float, oracle: float) -> float:
    """Return the exact relative-error expression required by the protocol."""
    return abs(actual - oracle) / max(abs(actual), abs(oracle), 1e-300)


def numerical_verdict(actual: float, oracle: float) -> NumericalVerdict:
    """Grade a finite scalar comparison without concealing WARN/BAD values."""
    if not (math.isfinite(actual) and math.isfinite(oracle)):
        return NumericalVerdict.BAD
    error = relative_error(actual, oracle)
    if error < 1e-3:
        return NumericalVerdict.OK
    if error < 1e-1:
        return NumericalVerdict.WARN
    return NumericalVerdict.BAD


def diagonal_logdet(diagonal: Sequence[float]) -> float:
    """Analytic log determinant of a positive compact diagonal."""
    values = tuple(float(value) for value in diagonal)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("the diagonal oracle needs finite positive entries")
    return math.fsum(math.log(value) for value in values)


def slogdet_log(matrix: np.ndarray) -> float:
    """Dense NumPy oracle, with positive determinant required explicitly."""
    sign, value = np.linalg.slogdet(np.asarray(matrix))
    if sign <= 0.0 or not np.isfinite(value):
        raise ValueError("the dense oracle needs a finite positive determinant")
    return float(value)


def exact_two_by_two_logdet(matrix: np.ndarray) -> float:
    """Decimal determinant oracle for a positive 2x2 matrix."""
    value = np.asarray(matrix)
    if value.shape != (2, 2):
        raise ValueError("the exact determinant oracle is two-dimensional")
    with localcontext() as context:
        context.prec = 2500
        determinant = decimal_determinant(value)
        if determinant <= 0:
            raise ValueError("the exact determinant oracle needs positive determinant")
        return float(determinant.ln())


def decimal_determinant(matrix: np.ndarray) -> Decimal:
    """Exact determinant of a represented binary matrix up to order six."""
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("the exact determinant oracle needs a square matrix")
    dimension = value.shape[0]
    if not 1 <= dimension <= 6:
        raise ValueError("the exact determinant oracle supports order one to six")
    with localcontext() as context:
        context.prec = 2500
        converted = [Decimal.from_float(float(item)) for item in value.flat]

        def item(row: int, column: int) -> Decimal:
            return converted[row * dimension + column]

        determinant = Decimal(0)
        for permutation in permutations(range(dimension)):
            inversions = sum(
                permutation[left] > permutation[right]
                for left in range(dimension)
                for right in range(left + 1, dimension)
            )
            term = Decimal(1)
            for row, column in enumerate(permutation):
                term *= item(row, column)
            determinant += -term if inversions % 2 else term
        return determinant


def decimal_logdet(matrix: np.ndarray) -> float:
    """Exact-binary Decimal determinant followed by a high-precision log."""
    with localcontext() as context:
        context.prec = 2500
        determinant = decimal_determinant(matrix)
        if determinant <= 0:
            raise ValueError("the exact logdet oracle needs positive determinant")
        return float(determinant.ln())


def explicit_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Scalar-loop matrix multiplication, independent of BLAS layout paths."""
    first = np.asarray(left)
    second = np.asarray(right)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[0]:
        raise ValueError("incompatible explicit matrix product")
    result = np.empty(
        (first.shape[0], second.shape[1]), dtype=np.result_type(first, second)
    )
    for row in range(first.shape[0]):
        for column in range(second.shape[1]):
            result[row, column] = float(decimal_dot(first[row], second[:, column]))
    return result


def decimal_dot(left: Sequence[float], right: Sequence[float]) -> Decimal:
    """Exact dot product of represented binary scalars."""
    first = tuple(left)
    second = tuple(right)
    if len(first) != len(second):
        raise ValueError("incompatible exact dot product")
    with localcontext() as context:
        context.prec = 2500
        return sum(
            (
                Decimal.from_float(float(a)) * Decimal.from_float(float(b))
                for a, b in zip(first, second, strict=True)
            ),
            start=Decimal(0),
        )


def symmetric_two_by_two_is_positive(matrix: np.ndarray) -> bool:
    """Exact principal-minor SPD oracle for a represented 2x2 matrix."""
    value = np.asarray(matrix)
    if value.shape != (2, 2) or not np.array_equal(value, value.T):
        return False
    a = Decimal.from_float(float(value[0, 0]))
    determinant = decimal_determinant(value)
    return a > 0 and determinant > 0


def symmetric_is_positive_definite(
    matrix: np.ndarray,
    *,
    relative_tolerance: float = 0.0,
    absolute_tolerance: float = 0.0,
) -> bool:
    """Decimal diagonal/Sylvester/LDL SPD oracle independent of ``eigvalsh``."""
    value = np.asarray(matrix)
    if value.ndim == 1:
        return bool(
            all(math.isfinite(float(item)) and float(item) > 0.0 for item in value)
        )
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or not np.all(np.isfinite(value))
        or not tolerant_symmetry(
            value,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
    ):
        return False
    representative = np.array(value, copy=True)
    with localcontext() as context:
        context.prec = 2500
        for row in range(value.shape[0]):
            for column in range(row + 1, value.shape[1]):
                mean = (
                    Decimal.from_float(float(value[row, column]))
                    + Decimal.from_float(float(value[column, row]))
                ) / Decimal(2)
                rounded = float(mean)
                representative[row, column] = rounded
                representative[column, row] = rounded
        diagonal = np.diag(np.diag(representative))
        if np.array_equal(representative, diagonal):
            return all(
                Decimal.from_float(float(item)) > Decimal(0)
                for item in np.diag(representative)
            )
        if value.shape[0] > 6:
            converted = [
                [Decimal.from_float(float(item)) for item in row]
                for row in representative
            ]
            dimension = value.shape[0]
            unit_lower = [
                [Decimal(int(row == column)) for column in range(dimension)]
                for row in range(dimension)
            ]
            pivots = [Decimal(0) for _ in range(dimension)]
            for column in range(dimension):
                pivot = converted[column][column] - sum(
                    (
                        unit_lower[column][prior] ** 2 * pivots[prior]
                        for prior in range(column)
                    ),
                    start=Decimal(0),
                )
                if pivot <= 0:
                    return False
                pivots[column] = pivot
                for row in range(column + 1, dimension):
                    numerator = converted[row][column] - sum(
                        (
                            unit_lower[row][prior]
                            * unit_lower[column][prior]
                            * pivots[prior]
                            for prior in range(column)
                        ),
                        start=Decimal(0),
                    )
                    unit_lower[row][column] = numerator / pivot
            return True
        return all(
            decimal_determinant(representative[:size, :size]) > 0
            for size in range(1, value.shape[0] + 1)
        )


def tolerant_symmetry(
    matrix: np.ndarray, *, relative_tolerance: float, absolute_tolerance: float
) -> bool:
    """Direct Decimal evaluation of every oriented symmetry inequality."""
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        return False
    relative = Decimal.from_float(float(relative_tolerance))
    absolute = Decimal.from_float(float(absolute_tolerance))
    for row in range(value.shape[0]):
        for column in range(value.shape[1]):
            left = Decimal.from_float(float(value[row, column]))
            right = Decimal.from_float(float(value[column, row]))
            if abs(left - right) > absolute + relative * abs(right):
                return False
    return True


def is_block_chain(matrix: np.ndarray, block_size: int) -> bool:
    """Explicit block-index oracle for exact chain sparsity."""
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        return False
    dimension = value.shape[0]
    if block_size < 1 or dimension % block_size:
        return False
    for row in range(dimension):
        for column in range(dimension):
            if (
                abs(row // block_size - column // block_size) > 1
                and value[row, column] != 0.0
            ):
                return False
    return True


def is_diagonal(matrix: np.ndarray) -> bool:
    """Literal exact-zero off-diagonal oracle."""
    value = np.asarray(matrix)
    return bool(
        value.ndim == 2
        and value.shape[0] == value.shape[1]
        and all(
            row == column or value[row, column] == 0.0
            for row in range(value.shape[0])
            for column in range(value.shape[1])
        )
    )


def is_circulant(matrix: np.ndarray) -> bool:
    """Exact cyclic-row oracle without ``np.roll`` or stacking."""
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        return False
    dimension = value.shape[0]
    return all(
        value[row, column] == value[0, (column - row) % dimension]
        for row in range(dimension)
        for column in range(dimension)
    )


def is_toeplitz(matrix: np.ndarray) -> bool:
    """Exact diagonal-constant oracle without ``np.diag``."""
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        return False
    dimension = value.shape[0]
    return all(
        value[row, column] == value[row - 1, column - 1]
        for row in range(1, dimension)
        for column in range(1, dimension)
    )


def circulant_eigenvalues(first_row: Sequence[float]) -> tuple[complex, ...]:
    """Handwritten discrete Fourier transform of a circulant first row."""
    values = tuple(float(value) for value in first_row)
    dimension = len(values)
    return tuple(
        sum(
            value * cmath.exp(-2j * math.pi * frequency * index / dimension)
            for index, value in enumerate(values)
        )
        for frequency in range(dimension)
    )


def explicit_kronecker(factors: Sequence[np.ndarray]) -> np.ndarray:
    """Nested-loop Kronecker reconstruction independent of ``np.kron``."""
    materialized = tuple(np.asarray(factor) for factor in factors)
    if not materialized:
        raise ValueError("the Kronecker oracle needs at least one factor")
    result = materialized[0].copy()
    for factor in materialized[1:]:
        rows, columns = result.shape
        factor_rows, factor_columns = factor.shape
        expanded = np.empty(
            (rows * factor_rows, columns * factor_columns),
            dtype=np.result_type(result, factor),
        )
        for row in range(rows):
            for column in range(columns):
                for factor_row in range(factor_rows):
                    for factor_column in range(factor_columns):
                        expanded[
                            row * factor_rows + factor_row,
                            column * factor_columns + factor_column,
                        ] = result[row, column] * factor[factor_row, factor_column]
        result = expanded
    return result


def symmetric_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Independent symmetric-spectrum oracle for small deterministic fixtures."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("the spectrum oracle needs a square matrix")
    if not np.array_equal(value, value.T):
        raise ValueError("the spectrum oracle needs exact symmetry")
    return np.linalg.eigvalsh(value)


def exact_symmetric_two_by_two_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    """Decimal characteristic roots for a represented symmetric 2x2 matrix."""
    value = np.asarray(matrix)
    if value.shape != (2, 2) or not np.array_equal(value, value.T):
        raise ValueError("the exact spectrum oracle needs a symmetric 2x2 matrix")
    with localcontext() as context:
        context.prec = 2500
        first = Decimal.from_float(float(value[0, 0]))
        off_diagonal = Decimal.from_float(float(value[0, 1]))
        second = Decimal.from_float(float(value[1, 1]))
        trace = first + second
        root = ((first - second) ** 2 + Decimal(4) * off_diagonal**2).sqrt()
        return np.array(
            [float((trace - root) / 2), float((trace + root) / 2)]
        )


def spectral_condition(matrix: np.ndarray) -> float:
    """2-norm condition from independently obtained singular values."""
    singular = np.linalg.svd(np.asarray(matrix), compute_uv=False)
    smallest = float(singular[-1])
    return math.inf if smallest == 0.0 else float(singular[0]) / smallest


def spectral_radius(lambda_matrix: np.ndarray, perturbation: np.ndarray) -> float:
    """Analytic represented-binary radius for diagonal or 2x2 fixtures.

    The production implementation uses ``solve`` followed by ``eigvals``.
    This oracle instead performs the diagonal ratios or the full 2x2 inverse
    and characteristic roots in high-precision Decimal arithmetic.
    """
    lam = np.asarray(lambda_matrix)
    perturb = np.asarray(perturbation)
    if lam.ndim == 1:
        if perturb.shape != lam.shape:
            raise ValueError("diagonal spectral-radius operands must share shape")
        with localcontext() as context:
            context.prec = 2500
            ratios = (
                abs(
                    Decimal.from_float(float(numerator))
                    / Decimal.from_float(float(denominator))
                )
                for numerator, denominator in zip(
                    perturb,
                    lam,
                    strict=True,
                )
            )
            return float(max(ratios, default=Decimal(0)))
    if lam.shape != (2, 2) or perturb.shape != (2, 2):
        raise ValueError("dense analytic spectral-radius oracle supports 2x2 fixtures")
    with localcontext() as context:
        context.prec = 2500
        a, b, c, d = (
            Decimal.from_float(float(lam[row, column]))
            for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
        p00, p01, p10, p11 = (
            Decimal.from_float(float(perturb[row, column]))
            for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
        determinant = a * d - b * c
        x00 = (p00 * d - p01 * c) / determinant
        x01 = (-p00 * b + p01 * a) / determinant
        x10 = (p10 * d - p11 * c) / determinant
        x11 = (-p10 * b + p11 * a) / determinant
        trace = x00 + x11
        product = x00 * x11 - x01 * x10
        discriminant = trace * trace - Decimal(4) * product
        if discriminant >= 0:
            root = discriminant.sqrt()
            eigenvalues = ((trace + root) / 2, (trace - root) / 2)
            return float(max(abs(value) for value in eigenvalues))
        if product < 0:
            raise ArithmeticError("negative product with a complex-conjugate pair")
        return float(product.sqrt())


def exact_power_traces(
    lambda_matrix: np.ndarray, perturbation: np.ndarray, order: int
) -> tuple[float, ...]:
    """Exact traces of powers of the represented-binary ``P @ Lambda^-1``.

    Compact fixtures are evaluated as scalar Decimal ratios.  Dense boundary
    fixtures are deliberately restricted to 2x2 so their right solve and
    power recurrence can be written as scalar Decimal arithmetic instead of
    sharing NumPy's solve/matmul path with production.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    lam = np.asarray(lambda_matrix)
    perturb = np.asarray(perturbation)
    if lam.shape != perturb.shape:
        raise ValueError("lambda and perturbation fixtures must share shape")
    with localcontext() as context:
        context.prec = 2500
        if lam.ndim == 1:
            ratios = tuple(
                Decimal.from_float(float(numerator))
                / Decimal.from_float(float(denominator))
                for numerator, denominator in zip(perturb, lam, strict=True)
            )
            return tuple(
                float(sum((ratio**power for ratio in ratios), Decimal(0)))
                for power in range(1, order + 1)
            )
        if lam.shape != (2, 2):
            raise ValueError("dense exact-power-trace fixtures must be 2x2")
        a, b, c, d = (
            Decimal.from_float(float(lam[row, column]))
            for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
        p00, p01, p10, p11 = (
            Decimal.from_float(float(perturb[row, column]))
            for row, column in ((0, 0), (0, 1), (1, 0), (1, 1))
        )
        determinant = a * d - b * c
        if determinant == 0:
            raise ValueError("lambda fixture must be invertible")
        x00 = (p00 * d - p01 * c) / determinant
        x01 = (-p00 * b + p01 * a) / determinant
        x10 = (p10 * d - p11 * c) / determinant
        x11 = (-p10 * b + p11 * a) / determinant
        q00, q01, q10, q11 = Decimal(1), Decimal(0), Decimal(0), Decimal(1)
        traces: list[float] = []
        for _ in range(order):
            q00, q01, q10, q11 = (
                q00 * x00 + q01 * x10,
                q00 * x01 + q01 * x11,
                q10 * x00 + q11 * x10,
                q10 * x01 + q11 * x11,
            )
            traces.append(float(q00 + q11))
        return tuple(traces)


def trace_log_polynomial(base: float, traces: Sequence[float], order: int) -> float:
    """Independent fixed-order trace-log polynomial."""
    if order < 0 or len(traces) < order:
        raise ValueError("the trace sequence must cover the requested order")
    correction = math.fsum(
        ((-1.0) ** (power + 1)) * float(traces[power - 1]) / power
        for power in range(1, order + 1)
    )
    return math.fsum((float(base), correction))


def whole_trace_tail(rho: float, order: int, multiplicity: int) -> float:
    """Evaluate the analytic whole-trace remainder in high precision."""
    if not 0.0 <= rho < 1.0 or order < 0 or multiplicity < 1:
        raise ValueError("invalid trace-tail domain")
    with localcontext() as context:
        context.prec = 90
        radius = Decimal.from_float(float(rho))
        numerator = Decimal(multiplicity) * radius ** (order + 1)
        denominator = Decimal(order + 1) * (Decimal(1) - radius)
        return float(numerator / denominator)


def smallest_trace_order(rho: float, tolerance: float, multiplicity: int) -> int:
    """Linear high-precision search for the first certified order."""
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    order = 0
    while whole_trace_tail(rho, order, multiplicity) > tolerance:
        order += 1
    return order


def exact_product(left: float, right: float) -> Decimal:
    """Exact product of the represented binary64 inputs."""
    with localcontext() as context:
        context.prec = 2500
        return Decimal.from_float(float(left)) * Decimal.from_float(float(right))


def exact_sum(left: float, right: float) -> Decimal:
    """Exact sum of the represented binary64 inputs."""
    with localcontext() as context:
        context.prec = 2500
        return Decimal.from_float(float(left)) + Decimal.from_float(float(right))


def exact_quotient(numerator: float, denominator: int) -> Decimal:
    """Exact quotient of a represented binary64 numerator and integer."""
    with localcontext() as context:
        context.prec = 2500
        return Decimal.from_float(float(numerator)) / Decimal(denominator)


def outward_nonnegative_oracle(exact: Decimal | Fraction | float) -> float:
    """Smallest binary64 strictly above a positive finite proof quantity.

    Exact zero is kept at zero.  Positive underflow is promoted to the least
    positive binary64, matching the directed-rounding proof rule rather than
    ordinary round-to-nearest arithmetic.
    """
    decimal = (
        exact
        if isinstance(exact, Decimal)
        else Decimal(exact.numerator) / Decimal(exact.denominator)
        if isinstance(exact, Fraction)
        else Decimal.from_float(float(exact))
    )
    if decimal == 0:
        return 0.0
    if not decimal.is_finite():
        return float(decimal)
    rounded = float(decimal)
    if rounded == 0.0 and decimal > 0:
        return float(np.nextafter(0.0, math.inf))
    represented = Decimal.from_float(rounded)
    if represented <= decimal:
        return math.nextafter(rounded, math.inf)
    return rounded


def gamma_fraction(operation_count: int, epsilon: float) -> float:
    """Exact-rational ``gamma_n`` with the standard open denominator."""
    eps = Fraction.from_float(float(epsilon))
    product = operation_count * eps
    if product >= 1:
        return math.inf
    return float(product / (1 - product))


def classify_correlation_side(value: float, floor: float) -> str:
    """Literal independent oracle for the two closed CCA noise floors."""
    if not (math.isfinite(value) and math.isfinite(floor)):
        return "refused"
    if value <= floor or value >= 1.0 - floor:
        return "refused"
    return "measured"


def covariance_canonical_correlations(precision: np.ndarray, split: int) -> np.ndarray:
    """Independent covariance-route CCA oracle for a two-block precision.

    Production whitens the precision cross block.  This oracle deliberately
    inverts to covariance, whitens covariance principal blocks, and takes the
    singular values of the normalized covariance cross block instead.
    """
    matrix = np.asarray(precision, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("the covariance CCA oracle needs a square precision")
    if not 0 < split < matrix.shape[0]:
        raise ValueError("the covariance CCA split must be interior")
    covariance = np.linalg.inv(matrix)
    covariance_xx = covariance[:split, :split]
    covariance_tt = covariance[split:, split:]
    covariance_xt = covariance[:split, split:]
    factor_x = np.linalg.cholesky(covariance_xx)
    factor_t = np.linalg.cholesky(covariance_tt)
    left_whitened = np.linalg.solve(factor_x, covariance_xt)
    normalized = np.linalg.solve(factor_t, left_whitened.T).T
    return np.linalg.svd(normalized, compute_uv=False)


def whitening_floor(kappa_x: float, kappa_conditioned: float, epsilon: float) -> float:
    """High-precision square-root whitening floor."""
    with localcontext() as context:
        context.prec = 90
        first = Decimal.from_float(float(kappa_x))
        second = Decimal.from_float(float(kappa_conditioned))
        eps = Decimal.from_float(float(epsilon))
        product = first * second
        if product < 0:
            raise ValueError("condition estimates must be non-negative")
        return float(product.sqrt() * eps)


def quadratic_mode(information: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Analytic mode of ``0.5*x.T H x - b.T x``."""
    return np.linalg.solve(np.asarray(precision), np.asarray(information))


def quadratic_derivatives(
    point: np.ndarray, information: np.ndarray, precision: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Independent objective, gradient, and Hessian of a quadratic."""
    x = np.asarray(point)
    b = np.asarray(information)
    hessian = np.asarray(precision)
    objective = 0.5 * float(x @ hessian @ x) - float(b @ x)
    return objective, hessian @ x - b, hessian


def normal_quadratic_payload(
    point: np.ndarray, centre: np.ndarray, widths: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Handwritten negative-log Normal objective, gradient, and Hessian."""
    x = np.asarray(point, dtype=np.float64)
    mean = np.asarray(centre, dtype=np.float64)
    scale = np.asarray(widths, dtype=np.float64)
    if x.shape != mean.shape or x.shape != scale.shape or np.any(scale <= 0.0):
        raise ValueError("the Normal oracle needs matching positive-width arrays")
    displacement = x - mean
    diagonal = 1.0 / scale**2
    hessian = np.diag(diagonal)
    gradient = diagonal * displacement
    objective = math.fsum(
        0.5 * float(delta * delta * precision)
        + math.log(float(width))
        + 0.5 * math.log(2.0 * math.pi)
        for delta, precision, width in zip(displacement, diagonal, scale, strict=True)
    )
    return objective, gradient, hessian


def diagonal_quadratic_payload(
    point: np.ndarray,
    offset: float,
    linear_terms: np.ndarray,
    squared_coefficients: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact represented-binary payload for a sum of scalar quadratic terms.

    The executable fixture is ``offset + sum_j b_j dot x + sum_i c_i*x_i*x_i``.
    Decimal scalar arithmetic keeps this oracle independent of JAX autodiff
    while still exposing overflow when the exact represented result is cast
    back to the production float dtype.
    """
    x = np.asarray(point, dtype=np.float64)
    terms = np.asarray(linear_terms, dtype=np.float64)
    coefficients = np.asarray(squared_coefficients, dtype=np.float64)
    if x.ndim != 1 or terms.ndim != 2 or terms.shape[1:] != x.shape:
        raise ValueError("linear terms must be a two-dimensional stack over point")
    if coefficients.shape != x.shape:
        raise ValueError("squared coefficients must match point")
    with localcontext() as context:
        context.prec = 2500
        decimal_x = tuple(Decimal.from_float(float(value)) for value in x)
        objective = Decimal.from_float(float(offset))
        for term in terms:
            objective += sum(
                (
                    Decimal.from_float(float(coefficient)) * value
                    for coefficient, value in zip(term, decimal_x, strict=True)
                ),
                Decimal(0),
            )
        objective += sum(
            (
                Decimal.from_float(float(coefficient)) * value * value
                for coefficient, value in zip(
                    coefficients, decimal_x, strict=True
                )
            ),
            Decimal(0),
        )
        gradient = np.array(
            [
                float(
                    sum(
                        (
                            Decimal.from_float(float(term[index]))
                            for term in terms
                        ),
                        Decimal(0),
                    )
                    + Decimal(2)
                    * Decimal.from_float(float(coefficients[index]))
                    * decimal_x[index]
                )
                for index in range(x.size)
            ],
            dtype=np.float64,
        )
        diagonal = np.array(
            [
                float(Decimal(2) * Decimal.from_float(float(coefficient)))
                for coefficient in coefficients
            ],
            dtype=np.float64,
        )
    return float(objective), gradient, np.diag(diagonal)


def stationarity_floor(hessian: np.ndarray, dimension: int, epsilon: float) -> float:
    """Independent MAP gradient floor for the diagonal quadratic fixtures."""
    matrix = np.asarray(hessian)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("the stationarity oracle needs a square Hessian")
    if any(
        matrix[row, column] != 0.0
        for row in range(matrix.shape[0])
        for column in range(matrix.shape[1])
        if row != column
    ):
        raise ValueError("the stationarity oracle supports diagonal fixtures")
    norm = max((abs(float(matrix[index, index])) for index in range(matrix.shape[0])), default=0.0)
    return math.sqrt(float(epsilon)) * dimension * norm


def relative_curvature_floor(
    largest_eigenvalue: float, dimension: int, epsilon: float
) -> float:
    """Independent MAP relative-curvature floor, including the live clamp."""
    return (
        float(epsilon)
        * max(abs(float(largest_eigenvalue)), 1.0)
        * max(int(dimension), 1)
    )


def unique_names(values: Iterable[str]) -> tuple[str, ...]:
    """Counter-based oracle for the graph duplicate-multiplicity contract."""
    names = tuple(values)
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate names: {duplicates}")
    return names


__all__ = [
    "NumericalVerdict",
    "circulant_eigenvalues",
    "exact_symmetric_two_by_two_eigenvalues",
    "classify_correlation_side",
    "covariance_canonical_correlations",
    "decimal_determinant",
    "decimal_dot",
    "decimal_logdet",
    "diagonal_logdet",
    "diagonal_quadratic_payload",
    "is_block_chain",
    "exact_power_traces",
    "exact_product",
    "exact_quotient",
    "exact_sum",
    "exact_two_by_two_logdet",
    "explicit_kronecker",
    "explicit_matmul",
    "gamma_fraction",
    "is_circulant",
    "is_diagonal",
    "is_toeplitz",
    "normal_quadratic_payload",
    "numerical_verdict",
    "outward_nonnegative_oracle",
    "quadratic_derivatives",
    "quadratic_mode",
    "relative_curvature_floor",
    "relative_error",
    "slogdet_log",
    "smallest_trace_order",
    "spectral_condition",
    "spectral_radius",
    "stationarity_floor",
    "symmetric_two_by_two_is_positive",
    "symmetric_eigenvalues",
    "tolerant_symmetry",
    "trace_log_polynomial",
    "unique_names",
    "whitening_floor",
    "whole_trace_tail",
]
