"""Checking the linear_in claim -- at several scales and at several at-points."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import sample, trace
from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.linearity import affinity_errors, check_linearity, linear_operator
from tests.exact.models import (
    affine_only_at_zero,
    bilinear_pair,
    cubic_tail,
    nan_at_negative_probes,
    quadratic_claim,
    straight_line,
    two_linear_latents,
)


def test_a_genuinely_linear_block_passes_at_every_scale():
    graph = straight_line()
    errors = check_linearity(graph, ("w",), at={})
    assert errors  # one entry per at-point
    for per_point in errors.values():
        assert len(per_point) == 3
        assert all(err < 1e-3 for err in per_point.values())


def test_a_quadratic_claim_is_refused():
    graph = quadratic_claim()
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("w",), at={})


def test_a_bilinear_pair_passes_singly_and_fails_jointly():
    """rheplicant's motivating failure, caught by three forward evaluations.

    Each conditional genuinely IS affine, which is why checking one latent at
    a time cannot see this and why a hand-rolled alternating solve reports a
    CG residual of 1e-7 and a per-block condition number of ~1.5 while landing
    thousands of kelvin away. The claim that is false is the JOINT one.
    """
    graph = bilinear_pair()
    check_linearity(graph, ("gain",), at={"t_ant": jnp.asarray(2.0)})
    check_linearity(graph, ("t_ant",), at={"gain": jnp.asarray(1.0)})
    with pytest.raises(StructureError, match="JOINTLY"):
        check_linearity(graph, ("gain", "t_ant"), at={})


def test_the_probe_magnitude_is_read_off_the_declared_prior():
    """One fn, two declared prior widths, opposite verdicts.

    Nothing about the model changes between these two calls except the width
    the graph declares for `w`. If the probe magnitude were a fixed constant
    the two would agree, whichever way.

    **This test's REFUSED half depends on `DEFAULT_SCALES` reaching 1e3.**
    Measured directly against `cubic_tail(prior_std=1.0)` through
    `check_linearity` itself (the exact call this test makes): the relative
    departure from affinity is 0.00e+00 at scale=1e-3, 4.58e-06 at scale=1.0,
    and 7.45e-01 only at scale=1e3 -- against this fixture's float32 `rtol`
    of 1.19e-3, the first two scales are indistinguishable from a genuinely
    linear model and only the widest probe catches the cubic term. A change
    that narrows `DEFAULT_SCALES` to drop 1e3 (e.g. to `(1e-3, 1.0)`) makes
    this test go RED -- not because the probe-magnitude-from-prior logic
    broke, but because the fixture's departure is invisible at the scales
    that remain. Recorded here so that a future edit to `DEFAULT_SCALES`
    meets this explanation instead of a silent, surprising failure.
    """
    with pytest.raises(StructureError, match="affine"):
        check_linearity(cubic_tail(prior_std=1.0), ("w",), at={})
    check_linearity(cubic_tail(prior_std=1e-4), ("w",), at={})


def test_check_linearity_probes_more_than_the_caller_s_at_point():
    """The claim holds where the caller pinned z, and nowhere the prior goes.

    Pinned explicitly to a single at-point the check passes; left to its
    default -- the caller's at PLUS draws from the graph's own prior -- it
    does not. An implementation that probed one point would pass both.
    """
    graph = affine_only_at_zero()
    pinned = {"z": jnp.asarray(0.0)}
    check_linearity(graph, ("x",), at=pinned, at_points=[pinned])
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("x",), at=pinned)


def test_a_probe_that_returns_nan_counts_as_a_failure():
    """`nan_at_negative_probes` linearizes sqrt(w) AT w=0 -- itself a singular
    point of sqrt's derivative -- so this test's own raise is OVER-determined:
    a POSITIVE probe there produces a clean +inf departure that trips the
    ordinary `errors > rtol` comparison on its own (inf, unlike nan, compares
    fine), independent of whether NaN is separately masked. Measured
    directly: deleting the `not finite` branch this test is meant to guard
    still leaves it green, because at least one of its 9 (point, scale) grid
    cells lands on a positive probe under the fixed default key and the
    resulting +inf redundantly triggers the raise from a DIFFERENT point
    index than the one where every probe was negative. This test alone,
    therefore, does not prove the `not finite` branch is load-bearing --
    `test_affinity_errors_treats_nan_as_a_failure_in_isolation` below does,
    by construction, with no +inf anywhere to fall back on.
    """
    graph = nan_at_negative_probes()
    with pytest.raises(StructureError, match="affine"):
        check_linearity(graph, ("w",), at={})


def test_affinity_errors_treats_nan_as_a_failure_in_isolation():
    """Isolates the `not finite` branch from the redundant +inf pathway above.

    Linearizes sqrt(w) at w=1 -- an ORDINARY point, derivative 0.5, nothing
    singular -- then probes ENTIRELY on the negative side (w going to -0.5),
    so `actual` is NaN (sqrt of a negative number) while `predicted` stays
    finite (a regular derivative times a finite probe). The resulting
    departure is a clean NaN with no accompanying +inf anywhere: a naive
    `errors > rtol` comparison alone reads `nan > rtol` as False and would
    call this affine. Only the `not finite` branch catches it.
    """

    def g(x):
        return {"y": jnp.sqrt(x["w"])}

    zero = {"w": jnp.asarray(1.0)}

    def probe_at(index, scale):
        del index
        return {"w": jnp.asarray(-1.5 * scale)}

    errors, failed, _ = affinity_errors(g, zero, probe_at, (1.0,), None)
    assert not jnp.isfinite(errors[1.0])  # confirms this probe is the clean-NaN case
    assert failed == [1.0]


def test_linear_operator_checks_before_it_builds():
    """The safe name is the natural one; the unchecked primitive says so."""
    graph = quadratic_claim()
    with pytest.raises(StructureError):
        linear_operator(graph, ("w",), at={})
    unchecked_operator(graph, ("w",), at={})  # builds happily, and is wrong


def test_linear_operator_returns_the_block_the_primitive_would_have():
    graph = two_linear_latents()
    at = {"b": jnp.asarray(4.0)}
    checked = linear_operator(graph, ("a",), at=at)
    raw = unchecked_operator(graph, ("a",), at=at)
    assert jnp.allclose(checked.offset["d"], raw.offset["d"])
    probe = {"a": jnp.asarray(1.0)}
    assert jnp.allclose(checked.forward(probe)["d"], raw.forward(probe)["d"])
    assert jnp.allclose(checked.prior_std["a"], raw.prior_std["a"])


def test_a_graph_with_no_observed_node_is_refused():
    """There is nothing to condition on, so the posterior IS the prior.

    Refused by name rather than reaching affinity_errors, where an empty
    codomain has no dtype to take a tolerance from and the failure would
    arrive as an unrelated ValueError from two layers down -- measured
    directly: without the guard, `check_linearity` on this fixture raises
    ``ValueError: at least one array or dtype is required`` from
    `jnp.result_type()` seeing zero leaves.

    Checked through all three entry points, not just `unchecked_operator`.
    `linear_operator` calls `check_linearity` BEFORE `unchecked_operator`, so
    a guard living only in the latter is unreachable from the documented
    entry point -- `check_linearity` would hit the confusing ValueError
    above before `unchecked_operator`'s refusal is ever reached. That is why
    `_refuse_missing_observed` is a helper shared by both rather than a
    check inlined in `unchecked_operator` alone.
    """

    def model():
        sample("w", lambda: dist.Normal(0.0, 1.0))

    graph = trace(model)
    with pytest.raises(GraphError, match="observed"):
        unchecked_operator(graph, ("w",), at={})
    with pytest.raises(GraphError, match="observed"):
        check_linearity(graph, ("w",), at={})
    with pytest.raises(GraphError, match="observed"):
        linear_operator(graph, ("w",), at={})
