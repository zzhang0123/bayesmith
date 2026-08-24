"""The dense route -- and what it does and does not independently confirm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.exact.fisher import (
    FlatMatrix,
    dense_operator,
    fisher_information,
    parameter_covariance,
)
from bayesmith.exact.gaussian import noise_std_at
from bayesmith.exact.linearity import linear_operator
from tests.exact.models import (
    plated_and_scalar_latents,
    plated_latent,
    two_linear_latents,
    two_observations,
    two_observations_reverse_sorted_names,
)
from tests.exact.oracle import graph_oracle


def test_dense_operator_matches_the_probed_design_matrix():
    """R3 vs R2. jacfwd and a basis probe must agree on A."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        design = np.asarray(dense_operator(block))
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert design.shape == oracle.design.shape
    assert np.allclose(design, oracle.design, rtol=1e-8)


def test_fisher_with_the_prior_is_the_posterior_precision():
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        fisher = fisher_information(block, noise_std=sigma)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert fisher.kind == "posterior_precision"
    assert np.allclose(np.asarray(fisher.values), oracle.precision, rtol=1e-8)


def test_fisher_without_the_prior_is_the_likelihood_alone_and_says_so():
    """`F = J^T N^-1 J` is a different quantity, and the kind field records it."""
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        fisher = fisher_information(block, noise_std=sigma, include_prior=False)
        oracle = graph_oracle(graph, ("a", "b"), at={})
    assert fisher.kind == "fisher"
    expected = oracle.precision - np.diag(1.0 / oracle.prior_std**2)
    assert np.allclose(np.asarray(fisher.values), expected, rtol=1e-8)


def test_parameter_covariance_matches_the_oracle():
    with jax.enable_x64(True):
        graph = two_observations()
        block = linear_operator(graph, ("w",), at={})
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        covariance = parameter_covariance(fisher_information(block, noise_std=sigma))
        oracle = graph_oracle(graph, ("w",), at={})
    assert covariance.kind == "covariance"
    assert np.allclose(np.asarray(covariance.values), oracle.covariance, rtol=1e-8)


def test_a_flat_matrix_block_is_addressable_by_latent_name():
    """A six-dimensional plated block, so the spans are not all width one."""
    with jax.enable_x64(True):
        graph = plated_latent(n=6)
        block = linear_operator(graph, ("z",), at={})
        sigma = noise_std_at(graph, {"z": jnp.zeros(6)})
        fisher = fisher_information(block, noise_std=sigma)
    assert fisher.names == ("z",)
    assert fisher.spans == ((0, 6),)
    assert fisher.block("z").shape == (6, 6)
    with pytest.raises(KeyError, match="w"):
        fisher.block("w")


# --- Structural-dimension audit (beyond the plan's five tests above) ---
#
# fisher.py branches on three structural dimensions: the DOMAIN member count
# (`block.names`, walked by `_spans`/`_unravel` and by the prior-curvature
# concatenation), the ELEMENTS PER MEMBER (`_spans`' `size = prod(shape)`,
# which is what makes a plated member occupy more than one row/column), and
# the CODOMAIN member count and ORDER (the `sorted(pushed)` / `sorted(
# noise_std)` concatenations that decide which row belongs to which observed
# node). Individually, the five tests above DO take at least two values on
# each of the first two: domain member count is 2 in the first three tests
# and 1 in the last two; elements-per-member is 1 (scalar) everywhere except
# the last test's plated `z` (6). Codomain member count is 1 everywhere
# except `test_parameter_covariance_matches_the_oracle` (`two_observations`
# has "d1" and "d2").
#
# What no given test does is (a) call `dense_operator` itself on a plated
# block -- the plate-shaped test above only reaches `fisher_information` and
# `FlatMatrix`'s addressing, never compares against the oracle; (b) declare a
# block's members in an order that is NOT already alphabetical, so nothing
# tells `_spans`/`_unravel`/`dense_operator`'s COLUMN order (`block.names`)
# apart from a hypothetical sorted-column layout; (c) give the codomain's
# `sorted(...)` concatenations a case where sorted order actually disagrees
# with declared order -- both `two_linear_latents` (one observed node) and
# `two_observations` ("d1" before "d2") have sorted order equal to declared
# order, which is exactly the gap Task 9's own mutation list (mutation 2)
# names and predicts the given tests cannot catch; and (d) give one block
# members of DIFFERENT sizes, so `_spans`' running offset (`start += size`)
# is exercised anywhere a wrong stride (e.g. reusing one member's size for
# another's) would show as more than a coincidence. The five tests below
# close each gap in turn -- (c) takes two, one on `dense_operator` and one on
# `fisher_information`, because measuring them separately is what surfaces a
# real correction to Task 9's own mutation 2: `dense_operator`'s
# `sorted(pushed)` turns out to be UNFALSIFIABLE (`pushed` is a jax.linearize
# tangent output, and JAX's dict-pytree flattening already forces sorted-key
# order before this module ever sees it -- see the two tests' docstrings for
# the direct measurement), while `fisher_information`'s `sorted(noise_std)`,
# on a plain non-pytree-roundtripped dict, is a real and caught regression.


def test_dense_operator_matches_the_probed_design_matrix_on_a_plated_member():
    """R3 vs R2 on a plate, directly on `dense_operator` -- not just on the
    higher-level `fisher_information`/`FlatMatrix` addressing that
    `test_a_flat_matrix_block_is_addressable_by_latent_name` exercises.

    Not redundant with a Fisher-level plated check, measured: mutating
    `_unravel` to reverse each member's OWN slice before reshaping
    (`flat[start:stop][::-1]`) is invisible to
    `test_fisher_information_matches_the_oracle_with_heterogeneous_member_sizes`
    (12 tests pass, that one included) -- `plated_and_scalar_latents`' sigma
    is spatially uniform, so permuting a plated member's internal element
    order permutes `A^T N^-1 A`'s diagonal by a CONSTANT weight, which maps
    back to itself; the same mutation is caught HERE, directly, because
    `dense_operator`'s raw output for `plated_latent` is `I_6` and the
    mutation turns it into the anti-diagonal (reversed identity) -- as
    different from `I_6` as a matrix gets. A Fisher-level check alone would
    have missed this class of bug entirely.
    """
    with jax.enable_x64(True):
        graph = plated_latent(n=6)
        block = linear_operator(graph, ("z",), at={})
        design = np.asarray(dense_operator(block))
        oracle = graph_oracle(graph, ("z",), at={})
    assert design.shape == oracle.design.shape
    assert np.allclose(design, oracle.design, rtol=1e-8)


def test_dense_operator_column_order_follows_block_names_not_alphabetical_order():
    """Columns are `block.names`' own order -- not sorted -- and every other
    test in this module declares its block in already-alphabetical order
    (`("a", "b")`, `("w",)`, `("z",)`), so none of them could tell
    `block.names` order apart from a silently-sorted one. Declaring
    `two_linear_latents` as `("b", "a")` can.

    Measured against a mutation that makes `_spans`/`_unravel` flatten the
    domain in `sorted(block.names)` order instead of `block.names` order
    (both functions changed together, so the mutation is internally
    consistent rather than a shape-mismatching crash): this test catches it
    (the oracle comparison and the column-content checks below both fail).
    A narrower, single-function mutation -- only `_spans` sorted, `_unravel`
    left alone -- does NOT turn this test red: with both of
    `two_linear_latents`' members at size 1, `_spans`' returned span
    SEQUENCE is the same regardless of which name was visited when
    (`((0, 1), (1, 2))` either way), so nothing here actually depends on
    `_spans`' internal iteration order alone. See
    `test_fisher_information_matches_the_oracle_with_heterogeneous_member_sizes`
    for the fixture that DOES depend on it.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("b", "a"), at={})
        design = np.asarray(dense_operator(block))
        oracle = graph_oracle(graph, ("b", "a"), at={})
    assert block.names == ("b", "a")
    assert design.shape == oracle.design.shape
    assert np.allclose(design, oracle.design, rtol=1e-8)
    # Not a vacuous comparison: column 0 (d prediction / d b) is exactly 1
    # everywhere since mu = a*X + b; column 1 (d prediction / d a) is X,
    # which is not constant. If dense_operator silently re-sorted to
    # alphabetical order the two columns would swap, and the oracle
    # comparison above would already have failed -- these two lines confirm
    # that failure is actually reachable rather than the columns
    # coincidentally agreeing either way.
    assert np.allclose(design[:, 0], 1.0, atol=1e-8)
    assert not np.allclose(design[:, 1], 1.0, atol=1e-2)


def test_dense_operator_matches_the_oracle_when_sorted_order_reverses_declaration_order():
    """Confirms `dense_operator`'s rows really are ALPHABETICAL order, on a
    fixture where that disagrees with declaration order -- every other
    fixture in this file has the two coincide, so none of them could tell
    "rows are sorted" apart from "rows are declaration order".

    **Not a regression pin for Task 9 mutation 2, despite being built for
    that purpose -- measured, not assumed.** `sorted(pushed)` -> `list(
    pushed)` in `dense_operator` turns out to be a NO-OP on every fixture,
    including this one: `pushed = block.forward(...)` is the tangent
    function `jax.linearize` returns, and its output dict has been round-
    tripped through JAX's pytree flatten/unflatten, which sorts a plain
    dict's keys unconditionally (confirmed directly:
    `jax.tree_util.tree_unflatten(jax.tree_util.tree_flatten({'z_first':
    1, 'a_second': 2})[1], [1, 2])` comes back `{'a_second': ..., 'z_first':
    ...}`, and `block.forward(...)`'s own returned dict on THIS fixture
    prints as `['a_second', 'z_first']` for every input tried). So `pushed`
    is already sorted by the time `dense_operator` sees it, for ANY block --
    `list(pushed)` reads the same order `sorted(pushed)` would. No fixture,
    however constructed, can make the two diverge at this call site.

    This test still earns its place: it is the only place in this module
    that checks `dense_operator`'s row order directly against the oracle on
    a fixture where alphabetical and declared order differ, so it protects
    the DOCSTRING's claim ("rows ... in sorted name order") against a
    regression that changes the mechanism producing that order (e.g. a
    future refactor that stops routing `pushed` through `jax.linearize` and
    builds it by hand). The mutation that Task 9's own plan named is caught
    one function over instead -- see
    `test_fisher_information_matches_the_oracle_when_sorted_order_reverses_declaration_order`,
    where `noise_std`'s dict is NOT jax-pytree-roundtripped and the same
    `sorted(...)` -> `list(...)` swap is a real, measured regression.
    """
    with jax.enable_x64(True):
        graph = two_observations_reverse_sorted_names()
        block = linear_operator(graph, ("w",), at={})
        design = np.asarray(dense_operator(block))
        oracle = graph_oracle(graph, ("w",), at={})
    assert design.shape == oracle.design.shape
    assert np.allclose(design, oracle.design, rtol=1e-8)


def test_fisher_information_matches_the_oracle_when_sorted_order_reverses_declaration_order():
    """THE actual regression pin for Task 9 mutation 2's intent (see the
    docstring above for why the plan's own named call site,
    `dense_operator`'s `sorted(pushed)`, cannot be pinned this way).

    `fisher_information`'s `weight` is ALSO a `sorted(noise_std)`
    concatenation -- independent code, a separate call to `sorted(...)` on a
    separate dict. Unlike `pushed`, `noise_std` is never a JAX transform's
    output: `noise_std_at` -> `observation_parts` builds it with a plain
    `for name in graph.observed: scale[name] = ...` loop, so it carries true
    declaration order and was measured to do so directly (`list(noise_std.
    keys())` on this fixture prints `['z_first', 'a_second']`, NOT sorted).
    A `sorted(noise_std)` -> `list(noise_std)` mutation here is therefore a
    real, catchable misalignment of `weight[i]` against `design`'s row `i`
    -- confirmed: with that one-line mutation applied, this is the ONLY test
    in the module that goes red (11 pass, this one fails); reverting
    restores all 12.
    """
    with jax.enable_x64(True):
        graph = two_observations_reverse_sorted_names()
        block = linear_operator(graph, ("w",), at={})
        sigma = noise_std_at(graph, {"w": jnp.asarray(0.0)})
        fisher = fisher_information(block, noise_std=sigma)
        oracle = graph_oracle(graph, ("w",), at={})
    assert fisher.kind == "posterior_precision"
    assert np.allclose(np.asarray(fisher.values), oracle.precision, rtol=1e-8)


def test_fisher_information_matches_the_oracle_with_heterogeneous_member_sizes():
    """`_spans`' running offset (`start += size`), stress-tested where a
    block's members actually have DIFFERENT sizes -- every other multi-member
    fixture in this file has every member at size 1, and the plated fixture
    has exactly one member, so neither can distinguish a member's own `size`
    from a hardcoded `1` or from a different member's size.

    Measured against two mutations `test_dense_operator_column_order_...`
    above cannot fully separate, both caught here:

    * `_spans` alone visiting `sorted(block.names)` (`_unravel` left
      pairing against `block.names`): a genuine `size` mismatch between
      what `_spans` computed a span FOR and what `_unravel` assigns it TO,
      not just a reordering -- `_unravel` tries to reshape a 1-element flat
      slice into `z`'s `(4,)` shape and raises
      `TypeError: cannot reshape array of shape (1,) into shape (4,)`
      before any numeric comparison runs.
    * `_spans` AND `_unravel` both visiting `sorted(block.names)` (self-
      consistent, no crash): `fisher.spans` comes back `((0, 1), (1, 5))`
      instead of `((0, 4), (4, 5))`, caught by the explicit `spans`
      assertion below before the oracle comparison is even reached.
    """
    with jax.enable_x64(True):
        graph = plated_and_scalar_latents(n=4)
        block = linear_operator(graph, ("z", "w"), at={})
        sigma = noise_std_at(graph, {"z": jnp.zeros(4), "w": jnp.asarray(0.0)})
        fisher = fisher_information(block, noise_std=sigma)
        oracle = graph_oracle(graph, ("z", "w"), at={})
    assert fisher.spans == ((0, 4), (4, 5))
    assert fisher.block("z").shape == (4, 4)
    assert fisher.block("w").shape == (1, 1)
    assert np.allclose(np.asarray(fisher.values), oracle.precision, rtol=1e-8)


def test_std_refuses_a_precision():
    """Mutation-3 regression: `FlatMatrix.std()`'s `kind` guard had no test
    before this -- see the mutation testing note in the module report.

    The plan's own Step 5 text for this test asserted
    ``.std()["a"].shape == ()``. Measured false: `std()` returns
    `diagonal[start:stop]`, a plain 1-D slice of the flat diagonal with no
    reshape back to `block.shape[name]` -- `FlatMatrix` does not even carry
    `block.shape`, only `spans` (start, stop widths). For a truly scalar
    latent (`block.shape["a"] == ()`, confirmed directly) the span is still
    width 1, so the slice is `(1,)`, never `()`. Corrected here rather than
    reproduced; see the module report for this finding named in full.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        fisher = fisher_information(block, noise_std=sigma)
        with pytest.raises(ValueError, match="not an error bar"):
            fisher.std()
        assert parameter_covariance(fisher).std()["a"].shape == (1,)


def test_parameter_covariance_refuses_a_covariance():
    """The reciprocal guard: `parameter_covariance` refuses a `kind`
    that is already a covariance, exactly as `std()` refuses a `kind` that
    is not one. Not in the plan's own mutation list, but coverage confirms
    it: before this test, `parameter_covariance`'s `ValueError` branch was
    the one statement in this module no test reached (0/1, measured via
    `--cov-report=term-missing`) -- the same "guard exists but is untested"
    shape mutation 3 found for `std()`, one function over.
    """
    with jax.enable_x64(True):
        graph = two_linear_latents()
        block = linear_operator(graph, ("a", "b"), at={})
        sigma = noise_std_at(graph, {"a": jnp.asarray(0.0), "b": jnp.asarray(0.0)})
        covariance = parameter_covariance(fisher_information(block, noise_std=sigma))
    with pytest.raises(ValueError, match="was handed a covariance"):
        parameter_covariance(covariance)


# ---------------------------------------------------------------------------
# B2: the Cramer-Rao bound is only as good as the arithmetic that formed it.
# ---------------------------------------------------------------------------


def _precision(kappa: float, dtype, size: int = 4):
    """A ``FlatMatrix`` whose condition number is exactly ``kappa``.

    Built by construction rather than by finding a model that happens to be
    ill-conditioned: the gate is a statement about the matrix, and a fixture
    that reached a chosen kappa through a graph would be testing the graph.
    """
    values = jnp.asarray(
        np.diag(np.geomspace(1.0, 1.0 / kappa, size)).astype(
            np.float64 if dtype is jnp.float64 else np.float32
        ),
        dtype=dtype,
    )
    return FlatMatrix(
        values=values,
        names=("w",),
        spans=((0, size),),
        kind="fisher",
    )


def test_a_precision_past_the_arithmetic_half_life_is_refused():
    """The gate, in the dtype every caller here already uses."""
    with jax.enable_x64(True):
        past = _precision(1e10, jnp.float64)
        with pytest.raises(ValueError, match="condition"):
            parameter_covariance(past)


def test_a_precision_inside_the_ceiling_still_inverts():
    """The gate must not be a blanket refusal of anything interesting.

    ``1e6`` is genuinely ill-conditioned and genuinely fine in float64 --
    measured, the inverse is accurate to 1.08e-12 there -- so refusing it
    would be the conservative-bound mistake this project has already paid
    for once upstream.
    """
    with jax.enable_x64(True):
        found = parameter_covariance(_precision(1e6, jnp.float64))
        assert found.kind == "covariance"
        assert np.all(np.isfinite(np.asarray(found.values)))


def test_the_ceiling_follows_the_dtype_rather_than_being_hard_wired():
    """The whole design, in one comparison: ONE rule, read against the
    arithmetic actually in use.

    ``kappa = 1e5`` sits between the two ceilings -- above float32's
    ``1/sqrt(eps) = 2.90e3``, below float64's ``6.71e7``. The same matrix is
    therefore refused in float32 and accepted in float64, which is what makes
    this a statement about digits spent rather than a magic number. A
    hard-wired float64 ceiling would let the float32 case through silently,
    and that case is the defect: measured on a design with ``kappa(J) = 1e3``
    (so ``kappa(F) = 1e6``), float32 gets the covariance wrong by 2.4%.
    """
    with jax.enable_x64(True):
        assert parameter_covariance(_precision(1e5, jnp.float64)) is not None
    # Outside the context, so float32 is the arithmetic and stays it.
    with pytest.raises(ValueError, match="condition"):
        parameter_covariance(_precision(1e5, jnp.float32))


def test_the_refusal_says_which_arithmetic_and_where_to_widen_it():
    """A refusal a reader can act on names the fix, and names it correctly.

    The fix is NOT to widen the inverse. ``F = J^T N^-1 J`` squares the
    condition number, so the digits are gone when F is FORMED; the context
    has to be open around the graph. See
    ``test_widening_only_the_inverse_does_not_recover_the_bound``.
    """
    with pytest.raises(ValueError) as caught:
        parameter_covariance(_precision(1e5, jnp.float32))
    message = str(caught.value)
    assert "float32" in message
    assert "enable_x64" in message
    assert "1e+05" in message or "1.0e+05" in message.replace("1e+05", "1.0e+05")


def test_max_condition_none_disables_the_gate():
    """An escape hatch, because the gate is a default rather than a law.

    A caller who has already decided the number is untrustworthy -- a
    forecast sweep that expects some cells to be degenerate, say -- should
    not have to route around the function.
    """
    with jax.enable_x64(True):
        found = parameter_covariance(_precision(1e14, jnp.float64), max_condition=None)
    assert found.kind == "covariance"


def test_jitter_is_measured_after_it_is_applied():
    """Jitter is the remedy for exactly this, so the gate must see the cure.

    Measuring the raw matrix would refuse a caller who had already fixed the
    problem in the only way this function offers.
    """
    with jax.enable_x64(True):
        bad = _precision(1e12, jnp.float64)
        with pytest.raises(ValueError, match="condition"):
            parameter_covariance(bad)
        found = parameter_covariance(bad, jitter=1e-3)
    assert found.kind == "covariance"


def test_widening_only_the_inverse_does_not_recover_the_bound():
    """Why the refusal points at the graph and not at the ``inv``.

    The migration spec's B2 asked for the decomposition to be wrapped in
    ``with jax.enable_x64(True):``. Measured, that is a no-op in two separate
    ways, and this pins both so the "fix" cannot be reintroduced as an
    improvement:

    1. ``jnp.linalg.inv`` of a float32 array inside the context returns
       float32. The context governs what is TRACED under it, not arrays that
       already exist.
    2. Even forcing the upcast does not help, because ``F = J^T N^-1 J`` has
       already squared the condition number in float32. Measured on
       ``kappa(J) = 1e3``: all-float32 is 2.41e-02 wrong, upcast-at-the-
       inverse is 2.45e-02 wrong -- indistinguishable -- and all-float64 is
       1.08e-12. A guard that wrapped the inverse would have reported the
       defect fixed while the error bar stayed 2.4% wrong.
    """
    size = 4
    rng = np.random.default_rng(0)
    left, _ = np.linalg.qr(rng.normal(size=(40, size)))
    right, _ = np.linalg.qr(rng.normal(size=(size, size)))
    design = (left * np.geomspace(1.0, 1e-3, size)) @ right.T
    reference = np.linalg.inv(design.T @ design)

    single = jnp.asarray(design, dtype=jnp.float32)
    formed = single.T @ single
    assert formed.dtype == jnp.float32

    with jax.enable_x64(True):
        # (1) the context alone does not widen an existing array
        assert jnp.linalg.inv(formed).dtype == jnp.float32
        # (2) forcing it does not recover the digits either
        widened = jnp.linalg.inv(jnp.asarray(np.asarray(formed), dtype=jnp.float64))
        honest = jnp.linalg.inv(jnp.asarray(design).T @ jnp.asarray(design))

    def worst(matrix):
        got = np.asarray(matrix, dtype=float)
        return float(np.max(np.abs(got - reference) / np.abs(reference)))

    assert worst(widened) > 1e-2, worst(widened)
    assert worst(honest) < 1e-9, worst(honest)
    assert worst(widened) > 1e6 * worst(honest)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_a_precision_that_is_not_finite_is_refused_rather_than_inverted(bad):
    """The ``not measured <= ceiling`` spelling, which ``>`` would not give.

    ``jnp.linalg.cond`` returns NaN for a matrix carrying either a NaN or an
    inf, and NaN fails every comparison -- so ``measured > ceiling`` is False
    and a diverged fit's precision would sail through the gate and come back
    as a covariance full of NaN, labelled ``kind="covariance"`` like any
    other. Written as a comment first; the comment survived mutation and this
    is what convicts it.
    """
    size = 4
    with jax.enable_x64(True):
        diagonal = np.ones(size)
        diagonal[1] = bad
        matrix = FlatMatrix(
            values=jnp.asarray(np.diag(diagonal)),
            names=("w",),
            spans=((0, size),),
            kind="fisher",
        )
        with pytest.raises(ValueError, match="condition"):
            parameter_covariance(matrix)
        # ...and what returns without the gate, which is NOT the same story
        # for the two: a NaN propagates into the covariance, while an inf
        # inverts to a clean 0.0 and comes back looking like a parameter
        # measured with certainty. The second is the more dangerous of the
        # two and the one a finiteness check on the OUTPUT would miss, which
        # is why the gate reads the condition rather than the result.
        loose = np.asarray(parameter_covariance(matrix, max_condition=None).values)
    if np.isnan(bad):
        assert not bool(np.all(np.isfinite(loose)))
    else:
        assert bool(np.all(np.isfinite(loose)))
        assert float(loose[1, 1]) == 0.0
