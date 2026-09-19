"""Static Edgeworth bookkeeping, shared by scalar and field likelihoods."""

import math
import operator
from functools import lru_cache


def truncation_order(order: int) -> int:
    if isinstance(order, bool):
        raise TypeError("order must be a static integer >= 2, not a boolean")
    try:
        result = operator.index(order)
    except TypeError as error:
        raise TypeError("order must be a static integer >= 2") from error
    if result < 2:
        raise ValueError("order must be >= 2")
    return result


@lru_cache(maxsize=32)
def edgeworth_powers(weight: int) -> tuple[tuple[int, ...], ...]:
    """Nonconstant products with sum((r-2)*m_r) <= weight."""

    def visit(r, remaining, powers):
        if r > weight + 2:
            if any(powers):
                yield powers
            return
        for power in range(remaining // (r - 2) + 1):
            yield from visit(r + 1, remaining - (r - 2) * power, (*powers, power))

    return tuple(visit(3, weight, ()))


def edgeworth_terms(order: int) -> tuple[tuple[tuple[int, ...], int], ...]:
    """(cumulant orders, denominator), e.g. ((3, 3), 72)."""
    terms = []
    for powers in edgeworth_powers(truncation_order(order) - 2):
        orders = tuple(r for r, m in enumerate(powers, 3) for _ in range(m))
        denominator = math.prod(
            math.factorial(r) ** m * math.factorial(m) for r, m in enumerate(powers, 3)
        )
        terms.append((orders, denominator))
    return tuple(terms)
