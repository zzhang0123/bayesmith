"""Cross-block identifiability -- the rank test that sees ACROSS blocks.

The motivating failure is an alternating solve over ``gain x T_ant``: every
per-block guard passes at every sweep while the joint model is degenerate and
the solve sits thousands of kelvin from the truth. A rank test over the JOINT
Jacobian can say so, and these tests pin that it does -- including the part
that matters most, that it reports a DIFFERENT verdict for a good
parameterization than for a bad one.

Ported from rheplicant's ``tests/inference/test_identifiability.py``; the
per-fixture cross-check record is ``docs/migration/identifiability.md``. One
family does not port: rheplicant forces process-global x64 inside the
diagnostic, so its suite proves "a float32 caller still gets the float64
verdict". This package refuses instead, so the corresponding family proves
the refusal fires -- and computes the float32 counterfactual by hand to show
what the refusal is protecting.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.diagnose.identifiability import (
    DEFAULT_RANK_RTOL,
    identifiability,
)
from bayesmith.diagnose.local import latent_values, local_block
from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.fisher import dense_operator
from bayesmith.graph.evaluate import evaluate
from tests.diagnose.models import (
    N_TIME,
    TONE_KELVIN,
    basis_graph,
    free_graph,
    mixed_scale_graph,
    sort_trap_graph,
    zero_column_graph,
)

# --------------------------------------------------------------- the headline --


class TestTheMotivatingCase:
    """The four-row table, reproduced on the graph port of the fixture.

    A known calibration tone buys EXACTLY NOTHING against a free-per-cell
    antenna temperature and EVERYTHING against a frequency-smooth one. No
    other diagnostic in the package can say this.
    """

    def test_the_four_row_table(self):
        with jax.enable_x64(True):
            rows = {
                ("free", "on"): identifiability(free_graph(TONE_KELVIN)),
                ("free", "off"): identifiability(free_graph(0.0)),
                ("basis", "on"): identifiability(basis_graph(TONE_KELVIN)),
                ("basis", "off"): identifiability(basis_graph(0.0)),
            }
        table = {k: (r.n_par, r.rank, r.nullity) for k, r in rows.items()}
        assert table == {
            ("free", "on"): (72, 64, 8),
            ("free", "off"): (72, 64, 8),
            ("basis", "on"): (17, 17, 0),
            ("basis", "off"): (17, 16, 1),
        }, table

    def test_the_tone_buys_nothing_against_a_free_per_cell_temperature(self):
        """The free cell at the tone's channel absorbs the gain sample by
        sample, so nullity stays at n_time whether the tone is there or not."""
        with jax.enable_x64(True):
            on = identifiability(free_graph(TONE_KELVIN))
            off = identifiability(free_graph(0.0))
        assert on.nullity == off.nullity == N_TIME

    def test_the_tone_buys_everything_against_a_basis_temperature(self):
        """A delta at one channel is not in the span of three smooth frequency
        basis functions, so it cannot be reabsorbed: 1 -> 0."""
        with jax.enable_x64(True):
            off = identifiability(basis_graph(0.0))
            on = identifiability(basis_graph(TONE_KELVIN))
        assert (off.nullity, on.nullity) == (1, 0)

    def test_the_two_parameterizations_are_told_apart(self):
        """The discrimination, stated as a comparison rather than two numbers:
        an implementation returning a constant verdict cannot pass."""
        with jax.enable_x64(True):
            free = identifiability(free_graph(TONE_KELVIN))
            basis = identifiability(basis_graph(TONE_KELVIN))
        assert free.nullity > 0
        assert basis.nullity == 0
        assert free.nullity != basis.nullity

    def test_the_measured_spectra_straddle_the_threshold_by_many_decades(self):
        """The numbers the default rtol is chosen against, re-measured here.

        The weakest IDENTIFIED direction of the basis model sits at ~5e-5 of
        the largest; its null direction at ~7e-17. The default 1e-8 is within
        a decade of that gap's geometric centre, so it is about as far from
        flipping either verdict as a single number can be.
        """
        with jax.enable_x64(True):
            on = identifiability(basis_graph(TONE_KELVIN))
            off = identifiability(basis_graph(0.0))
        weakest_identified = on.weakest_identified
        largest_null = float(off.singular_values[off.rank] / off.singular_values[0])
        assert weakest_identified > 1e-3
        assert largest_null < 1e-13
        assert largest_null < DEFAULT_RANK_RTOL < weakest_identified

    def test_the_free_model_reports_the_ratio_its_geometry_fixes(self):
        """``weakest_identified`` is s[rank-1]/s[0]: the free model's Jacobian
        is surjective, so every computed singular value is identified and the
        number is 1/sqrt(2) exactly. Pinned because it is the one row where an
        implementation confusing "smallest computed" with "smallest
        identified" still looks right."""
        with jax.enable_x64(True):
            free = identifiability(free_graph(TONE_KELVIN))
        assert free.weakest_identified == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-6)
        # ... while the FULL spectrum, padded to n_par, ends in exact zeros,
        # because 72 parameters cannot be identified by 64 data points.
        assert free.singular_values.shape == (72,)
        assert float(free.singular_values[-1]) == 0.0
        assert free.n_data == 64


# ------------------------------------------------------- naming the null space --


def _prediction(graph, values):
    """The det node's output at ``values`` -- the model the report is about."""
    return evaluate(graph, values)["pred"]


class TestNamedNullDirections:
    """An unnamed null space tells a user they have a problem and nothing
    about which. The naming is the whole value."""

    def test_the_direction_names_both_latents_it_mixes(self):
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0))
        share = report.participation(0)
        assert set(share) == {"gain", "t_coeff"}
        # The bilinear degeneracy g -> a*g, T -> T/a puts comparable weight
        # on both. A direction living entirely in one latent would be a
        # different (and wrong) statement about the model.
        assert share["gain"] == pytest.approx(0.5, abs=0.05)
        assert share["t_coeff"] == pytest.approx(0.5, abs=0.05)
        assert sum(share.values()) == pytest.approx(1.0, abs=1e-6)

    def test_the_direction_is_shaped_like_the_latents(self):
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0))
        direction = report.direction(0)
        assert direction["gain"].shape == (N_TIME,)
        assert direction["t_coeff"].shape == (3, 3)

    def test_moving_along_the_direction_really_does_not_move_the_model(self):
        """The end-to-end statement, in the coordinates a caller acts in.

        ``direction`` is in RAW latent units, so adding a small multiple of
        it to the latents must leave the prediction unchanged to first order
        -- while a random direction of the same size must not. This is what
        pins that the per-name split is assembled correctly: a direction
        whose gain and t_coeff halves were swapped or mis-scaled would fail
        here even though every shape still matched.
        """
        with jax.enable_x64(True):
            graph = basis_graph(0.0)
            values = latent_values(graph, None)
            report = identifiability(graph)
            direction = report.direction(0)

            step = 1e-3
            base = _prediction(graph, values)
            along = _prediction(
                graph,
                {k: values[k] + step * jnp.asarray(direction[k]) for k in values},
            )
            key = jax.random.key(0)
            random = {
                k: values[k]
                + step
                * jax.random.normal(jax.random.fold_in(key, i), jnp.shape(values[k]))
                for i, k in enumerate(values)
            }
            away = _prediction(graph, random)

            moved_along = float(jnp.max(jnp.abs(along - base)))
            moved_away = float(jnp.max(jnp.abs(away - base)))
        assert moved_along < 1e-3 * moved_away, (moved_along, moved_away)

    def test_the_report_uses_DECLARATION_order_not_dict_order(self):
        """Flattening a ``{name: array}`` dict sorts the keys; a report that
        flattened one way and named the other would attribute a degeneracy to
        the wrong latent while every shape still checked out.

        The fixture is deliberately asymmetric: the degeneracy is between
        ``sky_scale`` and ``load_scale``, which sit at opposite ends of the
        declaration order and at DIFFERENT ends of the sorted order, while
        ``tone_amps`` -- the one in the middle when sorted -- carries none.
        """
        with jax.enable_x64(True):
            graph = sort_trap_graph()
            assert sorted(graph.latents) != list(graph.latents), (
                "fixture must not be sort-stable"
            )
            report = identifiability(graph)
        assert report.names == ("sky_scale", "tone_amps", "load_scale")
        assert report.nullity == 1
        share = report.participation(0)
        assert share["sky_scale"] == pytest.approx(0.5, abs=0.05)
        assert share["load_scale"] == pytest.approx(0.5, abs=0.05)
        assert share["tone_amps"] == pytest.approx(0.0, abs=1e-6)

    def test_an_out_of_range_direction_is_refused(self):
        """Indexing a JAX array out of range CLAMPS rather than raising, so
        ``direction(1)`` on a 1-dimensional null space would silently return
        direction 0 again."""
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0))
        assert report.nullity == 1
        with pytest.raises(StructureError, match="null direction"):
            report.direction(1)
        with pytest.raises(StructureError, match="null direction"):
            report.participation(-1)

    def test_a_fully_identified_model_has_no_directions_to_ask_for(self):
        with jax.enable_x64(True):
            report = identifiability(basis_graph(TONE_KELVIN))
        assert report.nullity == 0
        assert report.null_space.shape == (0, 17)
        with pytest.raises(StructureError, match="null direction"):
            report.direction(0)

    def test_the_null_space_is_whole_when_there_are_fewer_data_than_parameters(self):
        """The headline case, and the one an SVD shortcut silently empties.

        The free-per-cell model has 64 data points against 72 parameters.
        Taken with ``full_matrices=False`` the SVD returns a (64, 72) ``Vh``
        with no rows past index 64, so ``right[rank:]`` with ``rank = 64`` is
        EMPTY while ``nullity`` still reports 8 -- and every direction the
        report names becomes unreachable. Every OTHER model in this file is
        over-determined, where the two spellings agree, so nothing else here
        can tell them apart.
        """
        with jax.enable_x64(True):
            free = identifiability(free_graph(TONE_KELVIN))
        assert free.n_data < free.n_par, (free.n_data, free.n_par)
        assert (free.n_data, free.n_par, free.nullity) == (64, 72, N_TIME)
        assert free.null_space.shape == (free.nullity, free.n_par) == (N_TIME, 72)

        # The LAST direction has to be reachable too, not just the first: a
        # null space truncated to any size short of `nullity` fails here.
        for index in (0, free.nullity - 1):
            direction = free.direction(index)
            assert direction["gain"].shape == (N_TIME,)
            assert direction["t_ant"].shape == (N_TIME, 8)
            assert np.all(np.isfinite(direction["gain"])), direction
            assert np.all(np.isfinite(direction["t_ant"])), direction

    def test_moving_along_a_free_per_cell_direction_does_not_move_the_model(self):
        """The end-to-end statement, on the UNDER-determined model, where
        ``direction`` has to survive a null space the SVD only returns under
        ``full_matrices=True``."""
        with jax.enable_x64(True):
            graph = free_graph(TONE_KELVIN)
            values = latent_values(graph, None)
            report = identifiability(graph)
            direction = report.direction(0)

            step = 1e-3
            base = _prediction(graph, values)
            along = _prediction(
                graph,
                {k: values[k] + step * jnp.asarray(direction[k]) for k in values},
            )
            key = jax.random.key(0)
            random = {
                k: values[k]
                + step
                * jax.random.normal(jax.random.fold_in(key, i), jnp.shape(values[k]))
                for i, k in enumerate(values)
            }
            away = _prediction(graph, random)

            moved_along = float(jnp.max(jnp.abs(along - base)))
            moved_away = float(jnp.max(jnp.abs(away - base)))
        assert moved_along < 1e-3 * moved_away, (moved_along, moved_away)

    def test_a_report_whose_null_space_disagrees_with_its_nullity_is_refused(self):
        """``nullity`` and ``null_space`` are two records of one fact.

        ``_row``'s bounds check trusts the first; the lookup after it uses
        the second. A truncated null space satisfies the bounds check and
        then indexes off the end of the array, and numpy's bare
        ``IndexError`` names neither the cause nor the repair. Constructed
        here directly rather than by mutating the SVD, so the invariant is
        pinned as an invariant.
        """
        with jax.enable_x64(True):
            good = identifiability(free_graph(TONE_KELVIN))

        # Too FEW rows -- what full_matrices=False produces.
        truncated = dataclasses.replace(good, null_space=good.null_space[:0])
        assert (truncated.nullity, truncated.null_space.shape) == (N_TIME, (0, 72))
        with pytest.raises(StructureError, match="[Ii]nconsistent report"):
            truncated.direction(0)
        with pytest.raises(StructureError, match="[Ii]nconsistent report"):
            truncated.participation(0)

        # Too few COLUMNS -- the other half of the shape, which would
        # otherwise reach `row / column_norms` and die on a numpy broadcast
        # error instead.
        narrowed = dataclasses.replace(good, null_space=good.null_space[:, :5])
        with pytest.raises(StructureError, match="[Ii]nconsistent report"):
            narrowed.direction(0)

        # ... while the intact report is not disturbed by the same check.
        assert good.direction(0)["gain"].shape == (N_TIME,)


# ------------------------------------------------- conditional blocks (Gibbs) --


class TestConditionalBlocks:
    """``names=`` is the Gibbs-block question: is THIS block identified,
    given the others held fixed?"""

    def test_every_block_is_identified_while_the_joint_is_not(self):
        """The measured failure, in one assertion.

        Each conditional has full rank -- which is why every per-block guard
        passes, correctly -- and the joint does not. A diagnostic that could
        only see one block at a time would report clean bills of health on a
        model that is degenerate.
        """
        with jax.enable_x64(True):
            graph = basis_graph(0.0)
            gain_only = identifiability(graph, names=("gain",))
            t_only = identifiability(graph, names=("t_coeff",))
            joint = identifiability(graph)
        assert gain_only.nullity == 0
        assert t_only.nullity == 0
        assert joint.nullity == 1

    def test_a_subset_reports_only_its_own_parameters(self):
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0), names=("t_coeff",))
        assert report.names == ("t_coeff",)
        assert report.n_par == 9

    def test_names_may_be_given_in_any_order(self):
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0), names=("t_coeff", "gain"))
        assert report.names == ("t_coeff", "gain")
        assert report.nullity == 1
        assert report.participation(0)["gain"] == pytest.approx(0.5, abs=0.05)

    def test_a_bare_string_is_one_name_not_four_characters(self):
        """Without the normalisation ``names="gain"`` iterates into
        ``('g', 'a', 'i', 'n')`` and comes back as four undeclared latents."""
        with jax.enable_x64(True):
            one = identifiability(basis_graph(0.0), names="gain")
            tupled = identifiability(basis_graph(0.0), names=("gain",))
        assert one.names == ("gain",) == tupled.names
        assert one.n_par == tupled.n_par == N_TIME

    def test_at_moves_the_evaluation_point(self):
        """Identifiability is a LOCAL property, so a Gibbs sweep has to ask
        it where the sampler currently is -- not where the priors are
        centred.

        With the gain conditioned to zero the antenna-temperature block
        stops reaching the data at all: every one of its nine parameters
        becomes a null direction. At the prior-centre gain none of them is.
        """
        with jax.enable_x64(True):
            graph = basis_graph(0.0)
            here = identifiability(graph, names=("t_coeff",))
            there = identifiability(
                graph, names=("t_coeff",), at={"gain": jnp.zeros(N_TIME)}
            )
        assert here.nullity == 0
        assert there.nullity == 9
        # Nothing at all is identified there, so the headline ratio has no
        # meaning and must say so rather than divide zero by zero.
        assert there.rank == 0
        assert there.weakest_identified == 0.0
        assert here.weakest_identified > 0.0

    def test_at_rejects_an_unknown_name(self):
        with jax.enable_x64(True), pytest.raises(GraphError, match="not a latent"):
            identifiability(basis_graph(0.0), at={"nope": jnp.array(1.0)})


# ------------------------------------------------------- column normalisation --


class TestColumnNormalisation:
    """Without it the rank verdict reports UNITS rather than identifiability."""

    def test_a_1e10_scale_gap_does_not_manufacture_a_null_direction(self):
        with jax.enable_x64(True):
            report = identifiability(mixed_scale_graph())
        assert (report.n_par, report.rank, report.nullity) == (2, 2, 0)

    def test_the_same_model_without_normalisation_would_be_called_degenerate(self):
        """The counterfactual, computed here rather than asserted by
        assertion: the RAW Jacobian's singular values differ by exactly the
        1e10 scale gap (measured 1.000000e-10), which is below the default
        rtol -- so an implementation that skipped the column scaling would
        report nullity 1 for a model that has no null direction at all.
        """
        with jax.enable_x64(True):
            graph = mixed_scale_graph()
            values = latent_values(graph, None)
            raw = dense_operator(local_block(graph, ("big", "small"), values))
            raw_spectrum = jnp.linalg.svd(raw, compute_uv=False)
            raw_ratio = float(raw_spectrum[-1] / raw_spectrum[0])
            report = identifiability(graph)
        assert raw_ratio < DEFAULT_RANK_RTOL, raw_ratio
        assert report.nullity == 0
        assert report.weakest_identified > DEFAULT_RANK_RTOL

    def test_a_direction_through_a_zero_column_is_finite_not_NaN(self):
        """What storing the SAFE column norms is for.

        ``column_norms`` keeps the GUARDED norms -- 1.0 substituted for
        every exactly-zero column -- and :meth:`direction` divides by them
        to undo the normalisation. Store the raw norms instead and that
        division is 1/0 on the supported entry and 0/0 on the rest, so
        every direction through a dead latent comes back ``[nan, nan,
        inf]``; the unit-norm rescale cannot repair it (``norm`` is NaN,
        ``NaN > 0.0`` is False, the fallback divides by 1.0 and preserves
        it). :meth:`participation` never touches ``column_norms``, so a
        participation-only test passes against a raw-norm store -- only
        ``direction`` can catch it.

        A latent the prediction does not depend on at all is an ordinary
        way to discover a modelling mistake, so this is the path a user
        hits on the day the diagnostic is doing its job.
        """
        with jax.enable_x64(True):
            report = identifiability(zero_column_graph())
        assert (report.n_par, report.rank, report.nullity) == (5, 2, 3)

        # The stored norms are the safe ones: no zero survives to be
        # divided by...
        assert np.all(report.column_norms > 0.0), report.column_norms
        assert report.column_norms.shape == (5,)
        # ... and they are the REAL norms where the column is live, not 1.0
        # for everything -- a store that substituted 1.0 unconditionally
        # would undo the normalisation `direction` exists to reverse.
        assert report.column_norms[0] == pytest.approx(1e3 * np.sqrt(N_TIME), rel=1e-6)
        assert report.column_norms[1] == pytest.approx(1e-2 * np.sqrt(N_TIME), rel=1e-6)
        assert np.all(report.column_norms[2:] == 1.0)

        for index in range(report.nullity):
            direction = report.direction(index)
            assert direction["live"].shape == (2,)
            assert direction["flat"].shape == (3,)
            assert np.all(np.isfinite(direction["live"])), (index, direction)
            assert np.all(np.isfinite(direction["flat"])), (index, direction)
            # Documented contract: unit 2-norm over the flat vector.
            flat_vector = np.concatenate([direction["live"], direction["flat"]])
            assert float(np.linalg.norm(flat_vector)) == pytest.approx(1.0)
            # The degeneracy is the DEAD latent's, entirely -- and the live
            # latent's two very differently scaled halves are untouched.
            assert np.allclose(direction["live"], 0.0), (index, direction)
            assert float(np.linalg.norm(direction["flat"])) == pytest.approx(1.0)


# -------------------------------------------------------------- the threshold --


class TestRankThreshold:
    def test_the_threshold_is_exposed_and_reported(self):
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0))
        assert report.rtol == DEFAULT_RANK_RTOL
        assert report.threshold == pytest.approx(
            DEFAULT_RANK_RTOL * float(report.singular_values[0])
        )

    def test_raising_rtol_moves_the_verdict(self):
        """A silently chosen threshold that flips a verdict is the bug this
        package likes least, so the knob is real and its effect is pinned:
        the basis model's weakest identified direction sits at ~7e-2, and an
        rtol above that reclassifies it as null."""
        with jax.enable_x64(True):
            strict = identifiability(basis_graph(TONE_KELVIN), rtol=1e-8)
            loose = identifiability(basis_graph(TONE_KELVIN), rtol=0.5)
        assert strict.nullity == 0
        assert loose.nullity > 0
        assert loose.rtol == 0.5

    def test_the_suite_pins_this_constant_more_tightly_than_the_physics(self):
        """Where a retune of ``DEFAULT_RANK_RTOL`` will fail, and why -- in
        one place, with every number re-measured on THIS package's fixtures.

        The numerically justified window is wide: every tolerance between
        the SVD's own noise floor (~1e-13) and the basis model's weakest
        identified direction (4.822138e-5) returns the same verdict, 8.7
        decades of freedom. The SUITE allows 2.4, because two
        counterfactuals are stated against the default rather than a
        literal:

        * below **1.0e-10** the mixed-scale fixture's raw spectrum ratio
          stops falling under the tolerance, and the no-normalisation
          counterfactual demonstrates nothing;
        * above **3.116759e-8** the basis model's float32 null direction
          falls under the tolerance, single precision gets the verdict
          RIGHT, and the float32 refusal has nothing to protect.

        Both are real claims about the constant rather than drift catchers,
        so the narrow window is kept rather than loosened. What is added is
        that it is *findable*: a retuner who reads only the constant's
        docstring sees 8.7 decades of headroom and then gets failures in
        unrelated classes. This test names the window, and the constant's
        docstring points here.
        """
        justified_low, justified_high = 1e-13, 4.822138e-05
        pinned_low, pinned_high = 1.0e-10, 3.116759e-08

        assert justified_low < pinned_low, "the suite's floor is above the noise floor"
        assert pinned_low < DEFAULT_RANK_RTOL < pinned_high, DEFAULT_RANK_RTOL
        assert pinned_high < justified_high, "the suite's ceiling is the tighter one"

        with jax.enable_x64(True):
            # The lower end is the mixed-scale fixture's raw ratio, measured.
            graph = mixed_scale_graph()
            values = latent_values(graph, None)
            raw = dense_operator(local_block(graph, ("big", "small"), values))
            raw_spectrum = jnp.linalg.svd(raw, compute_uv=False)
            assert float(raw_spectrum[-1] / raw_spectrum[0]) == pytest.approx(
                pinned_low, rel=1e-3
            )
            # The upper end's own measurement lives in the float32
            # counterfactual below; the number it straddles is this model's.
            off = identifiability(basis_graph(0.0))
        assert off.weakest_identified == pytest.approx(justified_high, rel=1e-3)


# ------------------------------------------------------------------ precision --


class TestPrecision:
    """The verdict is a number at 1e-17. float32 cannot represent that as a
    signal, and this package's rule is that ``src/`` never opens an x64
    context of its own -- so the diagnostic REFUSES rather than forcing, and
    these tests pin both the refusal and what it protects."""

    def test_the_report_is_float64_numpy_so_it_survives_leaving_the_context(self):
        """A float64 JAX array that escapes an x64 context truncates -- with
        a warning -- the moment a default-precision caller touches it,
        throwing away exactly the precision the diagnostic went to trouble
        to obtain."""
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0))
        assert isinstance(report.singular_values, np.ndarray)
        assert report.singular_values.dtype == np.float64
        assert report.null_space.dtype == np.float64
        assert report.jacobian.dtype == np.float64
        assert isinstance(report.direction(0)["gain"], np.ndarray)
        # the operation that would have warned and truncated on a JAX array
        assert float(np.sum(report.singular_values**2)) > 0.0

    def test_an_ambient_float32_call_is_refused_by_name(self):
        """Outside the context the refusal must arrive BEFORE the
        linearization: a graph built in x64, called from outside it, would
        otherwise die inside jax.linearize with a bare dtype inconsistency
        naming neither the cause nor the remedy."""
        with jax.enable_x64(True):
            graph = basis_graph(0.0)
        with pytest.raises(GraphError, match="enable_x64"):
            identifiability(graph)

    def test_a_graph_that_pins_float32_is_refused_inside_the_context(self):
        """The other way to lose the digits: the ambient precision is right
        and some node casts. The message must blame the graph, not the
        caller."""
        import numpyro.distributions as dist

        from bayesmith import det, observe, sample, trace

        with jax.enable_x64(True):

            def pinned():
                gain = sample("gain", lambda: dist.Normal(0.0, 1.0))
                pred = det(
                    "pred", lambda g: (g * jnp.ones(4)).astype(jnp.float32), gain
                )
                observe(
                    "d",
                    lambda mu: dist.Normal(mu, 1.0).to_event(1),
                    pred,
                    obs=jnp.zeros(4),
                )

            graph32 = trace(pinned)
            with pytest.raises(GraphError, match="pins its\n?.*own arithmetic|float32"):
                identifiability(graph32)

    def test_float32_would_have_got_the_verdict_wrong(self):
        """What the refusal protects, measured rather than asserted.

        Computed in single precision the null direction of the basis model
        surfaces at ~3.117e-8 of the largest singular value -- ABOVE the
        default 1e-8 rtol -- so the same rank test would report the
        degenerate model as fully identified (rank 17 of 17). In float64
        the same number is ~7.5e-17. This runs OUTSIDE any x64 context, by
        hand, because the diagnostic itself refuses to.
        """
        graph = basis_graph(0.0)  # built float32, matching the ambient dtype
        values = latent_values(graph, None)
        jacobian = dense_operator(local_block(graph, ("gain", "t_coeff"), values))
        assert jacobian.dtype == jnp.float32
        norms = jnp.linalg.norm(jacobian, axis=0)
        spectrum = jnp.linalg.svd(jacobian / norms, compute_uv=False)
        single = float(spectrum[-1] / spectrum[0])
        assert single > DEFAULT_RANK_RTOL, single
        assert single == pytest.approx(3.116759e-08, rel=1e-2)
        assert int(jnp.sum(spectrum > DEFAULT_RANK_RTOL * spectrum[0])) == 17

        # ... while the x64 run of the same model reports the truth.
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0))
        assert report.nullity == 1
        assert float(report.singular_values[16] / report.singular_values[0]) < 1e-13


# ------------------------------------------------------------------- refusals --


class TestGuards:
    def test_an_unknown_latent_name_is_refused(self):
        with jax.enable_x64(True), pytest.raises(GraphError, match="not a latent"):
            identifiability(basis_graph(0.0), names=("nope",))

    def test_a_repeated_latent_name_is_refused(self):
        """Two copies of one latent are exactly degenerate with each other,
        so a repeat manufactures a null direction that says nothing about
        the model."""
        with jax.enable_x64(True), pytest.raises(GraphError, match="more than once"):
            identifiability(basis_graph(0.0), names=("gain", "gain"))

    def test_an_empty_selection_is_refused(self):
        """``names=()`` would report nullity 0 over nothing at all, which
        reads as a clean bill of health for an empty block."""
        with jax.enable_x64(True), pytest.raises(GraphError, match="at least one"):
            identifiability(basis_graph(0.0), names=())

    def test_a_complex_valued_latent_is_refused(self):
        """Matched on a phrase only the complex branch uses: a complex dtype
        is also not floating, so a deleted complex branch would drop the
        latent into the integer one and a test matching "complex" alone
        would still pass against the wrong message."""
        with jax.enable_x64(True):
            graph = basis_graph(0.0)
            with pytest.raises(GraphError, match="R-linear but not\n?.*C-linear|C-linear"):
                identifiability(graph, at={"gain": jnp.ones(N_TIME) + 0j})

    def test_an_integer_valued_latent_is_refused(self):
        with jax.enable_x64(True):
            graph = basis_graph(0.0)
            with pytest.raises(GraphError, match="not floating-point|continuous"):
                identifiability(graph, at={"gain": jnp.arange(N_TIME)})

    @staticmethod
    def _complex_block_graph():
        """A real gain beside a complex coefficient block that reaches the
        prediction through ``jnp.real`` -- the sky-``alm`` shape."""
        import numpyro.distributions as dist

        from bayesmith import det, observe, sample, trace

        data = jnp.ones((4, 4)) * 3.0

        def model():
            gain = sample("gain", lambda: dist.Normal(jnp.ones(4), 0.1).to_event(1))
            coeff = sample(
                "coeff",
                lambda: dist.Normal(jnp.ones(2), 1.0).to_event(1),
            )
            pred = det(
                "pred",
                lambda g, c: g[:, None]
                * jnp.real(jnp.sum(c * (1.0 + 0j)))
                * jnp.ones((4, 4)),
                gain,
                coeff,
            )
            observe("d", lambda mu: dist.Normal(mu, 1.0).to_event(2), pred, obs=data)

        return trace(model)

    def test_a_complex_latent_that_is_not_selected_is_not_refused(self):
        """The dtype rules are about the parameters being DIFFERENTIATED, so
        a complex latent held fixed outside the selection is no obstacle --
        the sky-``alm`` situation, where a complex block sits in another
        Gibbs block while this one is asked about."""
        with jax.enable_x64(True):
            graph = self._complex_block_graph()
            report = identifiability(
                graph,
                names=("gain",),
                at={"coeff": jnp.ones(2) + 0j},
            )
        assert report.names == ("gain",)
        assert report.nullity == 0
        # ... while SELECTING the complex block is refused with the R-linear
        # explanation, not silently analysed over C.
        with jax.enable_x64(True), pytest.raises(GraphError, match="C-linear"):
            identifiability(
                self._complex_block_graph(),
                at={"coeff": jnp.ones(2) + 0j},
            )

    def test_a_graph_with_no_observed_node_is_refused(self):
        import numpyro.distributions as dist

        from bayesmith import sample, trace

        with jax.enable_x64(True):

            def model():
                sample("x", lambda: dist.Normal(0.0, 1.0))

            graph = trace(model)
            with pytest.raises(GraphError, match="no observed node"):
                identifiability(graph)

    def test_an_observed_node_with_no_loc_is_refused_by_name(self):
        import numpyro.distributions as dist

        from bayesmith import det, observe, sample, trace

        with jax.enable_x64(True):

            def model():
                rate = sample("rate", lambda: dist.Normal(3.0, 1.0))
                pred = det("pred", lambda r: jnp.exp(r) * jnp.ones(4), rate)
                observe(
                    "d",
                    lambda mu: dist.Poisson(mu).to_event(1),
                    pred,
                    obs=jnp.ones(4),
                )

            graph = trace(model)
            with pytest.raises(GraphError, match="Poisson"):
                identifiability(graph)


# ------------------------------------------------- what the SVD is asked for --


class TestTheSVDAsksForWhatItUses:
    """``U`` is discarded and ``right`` is used only as ``right[rank:]``, so
    ``full_matrices=True`` materialises an ``(n_data, n_data)`` left factor
    nothing reads -- the dominant cost of the diagnostic on a realistic
    grid. The flag is load-bearing in exactly one regime, ``n_data <
    n_par``, and these tests pin that the call asks for it there and only
    there."""

    @staticmethod
    def _spy(monkeypatch):
        """Record the ``full_matrices`` every ``jnp.linalg.svd`` call gets.

        Patches the shared ``jax.numpy.linalg`` attribute, which is how the
        diagnostic resolves the name at call time; ``monkeypatch`` restores
        it. The spy delegates, so the report is the real one.
        """
        seen: list[bool] = []
        real = jnp.linalg.svd

        def spy(a, *args, **kwargs):
            if "full_matrices" in kwargs:
                seen.append(bool(kwargs["full_matrices"]))
            elif args:
                seen.append(bool(args[0]))
            else:  # the jnp default
                seen.append(True)
            return real(a, *args, **kwargs)

        monkeypatch.setattr(jnp.linalg, "svd", spy)
        return seen

    def test_the_full_left_factor_is_not_asked_for_when_the_jacobian_is_not_wide(
        self, monkeypatch
    ):
        """The basis model is over-determined -- 64 data against 17
        parameters -- so ``Vh`` is (17, 17) under either spelling and the
        full ``U`` is 64x64 of pure waste. Every other test in this file
        passes under either spelling, which is why this one reads the
        ARGUMENT: there is no output to compare."""
        seen = self._spy(monkeypatch)
        with jax.enable_x64(True):
            report = identifiability(basis_graph(0.0))
        assert report.n_data >= report.n_par
        assert seen == [False], seen

    def test_the_full_left_factor_is_still_asked_for_when_there_are_fewer_data(
        self, monkeypatch
    ):
        """The anti-vacuity partner, and the regime the flag exists for. It
        also proves the spy can observe a ``True``, so the ``[False]`` above
        is a measurement and not a silent no-op."""
        seen = self._spy(monkeypatch)
        with jax.enable_x64(True):
            report = identifiability(free_graph(TONE_KELVIN))
        assert report.n_data < report.n_par
        assert seen == [True], seen
        assert report.null_space.shape == (report.nullity, report.n_par)

    def test_the_two_spellings_agree_when_the_jacobian_is_not_wide(self):
        """Why the conditional is safe, as arithmetic rather than a claim.

        For ``n_data >= n_par`` both spellings return the same spectrum and
        the same ``(n_par, n_par)`` ``Vh``, so ``right[rank:]`` -- the only
        slice the report keeps -- is bitwise identical. Pinned with a matrix
        that HAS a null space (eight exactly dependent columns): an equality
        that only held at full rank would say nothing about the branch that
        matters. Taken inside the x64 context because that is where the
        production call runs; in float32 the dependent columns sit at ~1e-7,
        above the tolerance, and the matrix reads as full rank -- the one
        input that cannot tell the two spellings apart.
        """
        rng = np.random.default_rng(20260825)
        base = rng.standard_normal((200, 64))
        base[:, -8:] = base[:, :8]  # eight exactly dependent columns

        with jax.enable_x64(True):
            a = jnp.asarray(base / np.linalg.norm(base, axis=0))
            _, s_true, v_true = jnp.linalg.svd(a, full_matrices=True)
            _, s_false, v_false = jnp.linalg.svd(a, full_matrices=False)

            s_true = np.asarray(s_true, dtype=np.float64)
            rank = int(np.sum(s_true > DEFAULT_RANK_RTOL * s_true[0]))
            assert 0 < rank < 64, rank  # the null space is real

            assert np.array_equal(s_true, np.asarray(s_false, dtype=np.float64))
            assert v_true.shape == v_false.shape == (64, 64)
            assert np.array_equal(
                np.asarray(v_true[rank:], dtype=np.float64),
                np.asarray(v_false[rank:], dtype=np.float64),
            )
