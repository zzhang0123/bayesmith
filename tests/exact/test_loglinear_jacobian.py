"""R4 Task 7 -- the change of variables ``log_space`` performs, recorded.

``log_space`` replaces an observed node's data with its logarithm and its
distribution with a Normal in log space. That is a change of variables, and a
change of variables carries a Jacobian: for ``y = log d``,

    p_d(d) = p_y(log d) / d,   so   log p_d = log p_y - sum(log d)

The transformed graph's ``log_joint`` is the second term without the third, so
it differs from the original by exactly ``-sum(log d)`` -- a data-dependent
constant, invisible in every posterior shape and every diagnostic, and
load-bearing in an evidence.

**The correction is recorded on ``LogSpace`` and folded in by the evidence
assembler; the transformed graph's ``log_joint`` is left bitwise alone.**
Adding it there instead would shift every existing consumer's density by that
constant, which is precisely the class of silent change R4 exists to prevent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.exact.loglinear import log_space
from bayesmith.graph.evaluate import log_joint

DATA_VALUES = (1.4, 0.7, 2.2, 0.9, 1.8, 1.1)


def _lognormal_graph():
    """Built INSIDE the caller's x64 context, deliberately.

    A module-level ``jnp.asarray`` is traced at import, which is float32, and
    the constant then stays float32 through every later context -- so the
    Jacobian would be compared at 1e-9 against a number carrying 1e-7 of
    error. ``const`` and ``observe`` have to be reached at the wider dtype.
    """
    data = jnp.asarray(DATA_VALUES)
    basis = jnp.linspace(0.5, 1.5, 6)

    def model():
        w = sample("w", lambda: dist.Normal(0.2, 0.4))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * jnp.exp(w_), b, w)
        observe("d", lambda m: dist.LogNormal(jnp.log(m), 0.4).to_event(1), mu,
                obs=data)

    return trace(model)


def test_the_transform_records_the_jacobian_it_performs():
    """``sum(log d)``, measured against the two densities it separates."""
    with jax.enable_x64(True):
        graph = _lognormal_graph()
        transformed = log_space(graph)

        assert hasattr(transformed, "log_jacobian"), (
            "LogSpace carries (graph, kind, fractional, skipped) and nothing "
            "for the change of variables it performed"
        )
        expected = -float(np.sum(np.log(np.asarray(DATA_VALUES, dtype=float))))
        assert float(transformed.log_jacobian) == pytest.approx(expected, abs=1e-9)


def test_the_recorded_jacobian_is_the_gap_between_the_two_densities():
    """Not a formula copied into a field: the number that closes the gap.

    ``log_joint(original) - log_joint(transformed) == log_jacobian`` is what
    makes the field worth having, and it is asserted rather than derived twice.
    """
    with jax.enable_x64(True):
        graph = _lognormal_graph()
        transformed = log_space(graph)
        at = {"w": jnp.asarray(0.3)}

        original = float(log_joint(graph, at))
        in_log_space = float(log_joint(transformed.graph, at))
        # The correction an evidence ADDS, sign included: the field is the log
        # absolute Jacobian, not a magnitude the caller has to orient.
        assert in_log_space + float(transformed.log_jacobian) == pytest.approx(
            original, abs=1e-9
        )
        assert abs(float(transformed.log_jacobian)) > 0.1, (
            "the fixture must have a Jacobian big enough to distinguish"
        )


def test_the_transformed_graphs_own_density_is_untouched():
    """The half that must NOT change, asserted bitwise.

    Every downstream consumer takes ``LogSpace.graph`` verbatim. Folding the
    Jacobian into it would move each of them by a data-dependent constant that
    no posterior, no diagnostic and no predictive check can see.
    """
    with jax.enable_x64(True):
        graph = _lognormal_graph()
        transformed = log_space(graph)
        at = {"w": jnp.asarray(0.3)}
        first = log_joint(transformed.graph, at)
        second = log_joint(log_space(graph).graph, at)
        assert float(first) == float(second)
        # And it is NOT the original graph's density, which is the whole point.
        assert float(first) != float(log_joint(graph, at))
