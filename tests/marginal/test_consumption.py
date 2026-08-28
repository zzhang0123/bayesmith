"""The evidence consumption surface (G6).

Six names a campaign's wrapper calls and this package did not have:
:func:`residual_summary`, :func:`epoch_residuals`, :func:`template_modes`,
:func:`held_out_z`, :func:`shrinkage_power` / :func:`shrinkage_report` and
:func:`systematic_floor`.

**The oracles are independent of the implementations, and deliberately of
different kinds.** ``held_out_z`` is checked against a leave-one-out posterior
REFITTED FROM SCRATCH in numpy, which shares no code with the subtract-one-
summand recursion. ``residual_summary``'s chi-square is checked against its own
null distribution over hundreds of epochs -- a mean, not a formula.
``shrinkage_power`` is checked against exactly-power-law inputs, where the
answer is an integer ratio. ``systematic_floor``'s direction is checked against
a covariance built to make the coordinate reading wrong by four orders of
magnitude.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.exact.precision import DiagonalPrecision
from bayesmith.marginal.compress import compress, residual_summary
from bayesmith.marginal.diagnostics import (
    epoch_residuals,
    held_out_z,
    refuse_mixed_templates,
    shrinkage_power,
    shrinkage_report,
    systematic_floor,
    template_modes,
    tightest_direction,
)

# --- the fixture, in numpy -------------------------------------------------
#
# Two survivors and a LARGE per-epoch nuisance. The nuisance is large on
# purpose: it is what makes "did the projector include the nuisance columns"
# a visible question rather than a stylistic one.
N_DATA = 24
SIGMA = 0.5
X = np.linspace(0.0, 1.0, N_DATA)
GLOBAL = np.stack([np.ones(N_DATA), X], axis=1)
NUISANCE = np.stack([np.sin(2 * np.pi * X), np.cos(2 * np.pi * X), X**2], axis=1)
SHAPES = {"a": (2,), "n": (3,)}

#: A shape the design cannot explain -- 97.7 % of it is out of span, measured.
TEMPLATE = np.cos(2 * np.pi * 9 * X)
#: Another one, equally out of span, that does NOT match the injected fault.
OTHER = np.cos(2 * np.pi * 7 * X)


def precision():
    return DiagonalPrecision(sigma=jnp.full((N_DATA,), SIGMA))


def epoch_data(seed: int, amplitude: float = 0.0) -> jax.Array:
    rng = np.random.default_rng(seed)
    return jnp.asarray(
        GLOBAL @ rng.normal(size=2)
        + NUISANCE @ (rng.normal(size=3) * 4.0)
        + SIGMA * rng.normal(size=N_DATA)
        + amplitude * TEMPLATE
    )


def fitted_term(data, *, with_nuisance: bool = True):
    """`compress` over EVERY column the epoch fits -- what §9.3 needs."""
    if with_nuisance:
        design = {"a": jnp.asarray(GLOBAL), "n": jnp.asarray(NUISANCE)}
        return compress(design, data, precision(), SHAPES)
    return compress({"a": jnp.asarray(GLOBAL)}, data, precision(), {"a": (2,)})


def summaries(amplitude: float = 0.0, n: int = 120, templates=None):
    if templates is None:
        templates = {
            "right": jnp.asarray(TEMPLATE),
            "wrong": jnp.asarray(OTHER),
            "in_span": jnp.asarray(X),
        }
    return [
        residual_summary(
            fitted_term(epoch_data(seed, amplitude)), precision(), templates=templates
        )
        for seed in range(n)
    ]


class TestAFlaggedSampleCannotPoisonATerm:
    """`compress` selects on `seen` rather than trusting the zero weight.

    A flagged sample carries weight zero and contributes nothing -- but
    ``0.0 * nan`` is ``nan``, so whitening propagates the value the mask
    exists to discard. The failure was the quiet kind: ``information()``
    reads ``factor.T @ factor``, which stays finite and well conditioned when
    only the DATA is poisoned, so the campaign audits as healthy while every
    density it produces is NaN.
    """

    @staticmethod
    def _pieces():
        sigma = jnp.array([1.0, 1.0, jnp.inf, 1.0])
        return DiagonalPrecision(sigma=sigma), {
            "x": jnp.array([[1.0], [2.0], [3.0], [4.0]])
        }

    def test_a_nan_in_the_data_at_a_flagged_sample_changes_nothing(self):
        prec, design = self._pieces()
        clean = compress(design, jnp.array([1.0, 2.0, 3.0, 4.0]), prec, {"x": ()})
        poisoned = compress(
            design, jnp.array([1.0, 2.0, jnp.nan, 4.0]), prec, {"x": ()}
        )
        assert bool(jnp.all(clean.target == poisoned.target))
        assert bool(jnp.all(clean.factor == poisoned.factor))
        assert bool(jnp.all(jnp.isfinite(poisoned.target)))

    def test_a_nan_in_the_design_at_a_flagged_sample_changes_nothing(self):
        prec, design = self._pieces()
        clean = compress(design, jnp.array([1.0, 2.0, 3.0, 4.0]), prec, {"x": ()})
        poisoned = compress(
            {"x": jnp.array([[1.0], [2.0], [jnp.nan], [4.0]])},
            jnp.array([1.0, 2.0, 3.0, 4.0]),
            prec,
            {"x": ()},
        )
        assert bool(jnp.all(clean.factor == poisoned.factor))
        assert bool(jnp.all(jnp.isfinite(poisoned.factor)))

    def test_the_flagged_sample_really_is_being_masked(self):
        """The sibling: without it the two tests above would pass on a build
        that had simply stopped masking anything."""
        prec, design = self._pieces()
        masked = compress(design, jnp.array([1.0, 2.0, 3.0, 4.0]), prec, {"x": ()})
        everything = compress(
            design,
            jnp.array([1.0, 2.0, 3.0, 4.0]),
            DiagonalPrecision(sigma=jnp.ones(4)),
            {"x": ()},
        )
        assert float(masked.factor[2, 0]) == 0.0
        assert float(everything.factor[2, 0]) != 0.0
        assert float(masked.offset) != float(everything.offset)


class TestResidualSummary:
    def test_the_chi_square_has_its_own_degrees_of_freedom(self):
        """The null, checked as a MEAN over epochs rather than as a formula."""
        rows = [residual_summary(fitted_term(epoch_data(s)), precision()) for s in range(400)]
        dof = rows[0].dof
        assert dof == N_DATA - 5
        mean = float(np.mean([row.chi2 for row in rows]))
        # Measured 18.55 against dof 19; the standard error of the mean is
        # sqrt(2 * 19 / 400) = 0.31, so this is a three-sigma band.
        assert abs(mean - dof) < 1.0

    def test_leaving_the_nuisance_out_of_the_projector_detects_nothing_loudly(self):
        """Why ``fitted`` must carry every column the epoch fits.

        The nuisance's contribution stays in the residual, so the chi-square
        is inflated by whatever the nuisance explained while the dof is
        over-counted by its rank. No shape catches it. Measured over 400 clean
        epochs: including it gives dof 19 and a mean of 18.55; excluding it
        gives dof 22 and a mean of **1227**, a sixty-fold detection of nothing
        at all on data with no fault in it.
        """
        both = [residual_summary(fitted_term(epoch_data(s)), precision()) for s in range(400)]
        only = [
            residual_summary(fitted_term(epoch_data(s), with_nuisance=False), precision())
            for s in range(400)
        ]
        assert both[0].dof == 19 and only[0].dof == 22
        assert abs(np.mean([r.chi2 for r in both]) - 19) < 1.0
        assert np.mean([r.chi2 for r in only]) > 500.0

    def test_reduced_chi_square_is_nan_when_the_design_saturates_the_data(self):
        """Zero would read as a perfect fit."""
        prec = DiagonalPrecision(sigma=jnp.ones(3))
        saturated = compress(
            {"x": jnp.eye(3)}, jnp.array([1.0, 2.0, 3.0]), prec, {"x": (3,)}
        )
        summary = residual_summary(saturated, prec)
        assert summary.dof == 0
        assert np.isnan(summary.reduced_chi2)

    def test_a_template_inside_the_span_projects_to_exactly_zero(self):
        """And 'exactly' is load-bearing -- see the sibling below."""
        summary = residual_summary(
            fitted_term(epoch_data(1)),
            precision(),
            templates={"in_span": jnp.asarray(X)},
        )
        assert float(np.asarray(summary.projections)[0]) == 0.0

    def test_the_in_span_test_has_to_be_relative(self):
        """The measurement that put a ``sqrt(eps)`` cut there.

        ``X`` IS a design column, so in exact arithmetic nothing survives the
        projection. In floating point roundoff does -- measured at 6.0e-07 of
        the column's own norm -- and a ``norm > 0.0`` test then divides by that
        roundoff and returns an arbitrary unit vector's dot with the residual.
        This asserts the leftover is real and tiny, which is what makes the
        test above a statement about the CUT rather than about exact zeros.
        """
        term = fitted_term(epoch_data(1))
        design = np.asarray(term.factor)
        projector = design @ np.linalg.pinv(design)
        column = np.asarray(precision().whiten(jnp.asarray(X)))
        leftover = np.linalg.norm(column - projector @ column) / np.linalg.norm(column)
        assert 0.0 < leftover < np.sqrt(np.finfo(design.dtype).eps)

    def test_an_out_of_span_template_is_a_standard_normal_under_the_null(self):
        rows = summaries(0.0, n=400, templates={"right": jnp.asarray(TEMPLATE)})
        values = np.array([float(np.asarray(r.projections)[0]) for r in rows])
        assert abs(values.mean()) < 3.0 / np.sqrt(400)
        assert abs(values.std() - 1.0) < 0.15

    def test_a_template_of_the_wrong_length_is_refused(self):
        with pytest.raises(StructureError, match="one value per sample"):
            residual_summary(
                fitted_term(epoch_data(1)),
                precision(),
                templates={"short": jnp.ones(3)},
            )

    def test_a_non_finite_template_is_refused_rather_than_projected(self):
        broken = np.array(TEMPLATE, copy=True)
        broken[4] = np.nan
        with pytest.raises(StructureError, match="not finite"):
            residual_summary(
                fitted_term(epoch_data(1)),
                precision(),
                templates={"broken": jnp.asarray(broken)},
            )

    def test_a_template_may_be_non_finite_where_the_epoch_did_not_observe(self):
        """The sibling of the refusal above: the latitude the data itself gets."""
        sigma = jnp.array([1.0, 1.0, jnp.inf, 1.0])
        prec = DiagonalPrecision(sigma=sigma)
        term = compress(
            {"x": jnp.array([[1.0], [2.0], [3.0], [4.0]])},
            jnp.array([1.0, 2.0, 3.0, 4.0]),
            prec,
            {"x": ()},
        )
        summary = residual_summary(
            term, prec, templates={"t": jnp.array([0.0, 1.0, jnp.nan, -1.0])}
        )
        assert bool(np.isfinite(np.asarray(summary.projections)[0]))
        # ... and the mask was what made that legal.
        assert summary.dof == 3 - 1


class TestTheCampaignTable:
    def test_every_epoch_gets_a_row_in_the_order_given(self):
        rows = epoch_residuals(summaries(n=5))
        assert len(rows) == 5
        assert set(rows[0]) == {"chi2", "dof", "reduced_chi2", "templates"}
        assert set(rows[0]["templates"]) == {"right", "wrong", "in_span"}

    def test_mixed_template_lists_are_refused(self):
        mixed = summaries(n=2, templates={"a": jnp.asarray(TEMPLATE)})
        mixed += summaries(n=1, templates={"b": jnp.asarray(TEMPLATE)})
        with pytest.raises(StructureError, match="different systematic templates"):
            epoch_residuals(mixed)

    def test_an_empty_campaign_is_refused_rather_than_returning_nothing(self):
        with pytest.raises(StructureError, match="no epochs"):
            refuse_mixed_templates([])


class TestTemplateModes:
    """The named half of the common-mode question."""

    @pytest.fixture(scope="class")
    @classmethod
    def clean(cls):
        return template_modes(summaries(0.0))

    @pytest.fixture(scope="class")
    @classmethod
    def biased(cls):
        return template_modes(summaries(0.30))

    def test_the_named_template_fires_on_the_fault_it_names(self, clean, biased):
        # Measured: +0.59 clean, +23.28 biased -- a factor of thirty-nine.
        assert abs(clean["templates"]["right"]["z"]) < 3.0
        assert biased["templates"]["right"]["z"] > 12.0

    def test_a_template_that_names_the_wrong_shape_stays_silent(self, clean, biased):
        """What makes the test above a detection rather than a thermometer."""
        assert abs(clean["templates"]["wrong"]["z"]) < 3.0
        assert abs(biased["templates"]["wrong"]["z"]) < 3.0

    def test_an_in_span_template_says_exactly_nothing_either_way(self, clean, biased):
        assert clean["templates"]["in_span"]["z"] == 0.0
        assert biased["templates"]["in_span"]["z"] == 0.0

    def test_the_scatter_says_a_shift_is_a_shift(self, clean, biased):
        """A mean-level fault leaves the spread alone; an under-estimated noise
        model would raise both, and the two failing differently is what tells
        them apart.

        **Stated as a ratio of two effects rather than as a tolerance on one.**
        The first version of this test asserted the two scatters agreed to
        ``rel=1e-9``; measured, they agree to 3.3e-8 -- so it went red, and it
        deserved to. Both numbers come from float32 arithmetic over 120
        epochs, and 1e-9 there is the machine's luck written down as a bound,
        which is the shape `2026-08-27-ci-flat-chain.md` records costing five
        green local runs against five red remote ones. What is actually being
        claimed is that the fault moves the MEAN and not the SPREAD, and that
        is a ratio no rounding can reach.
        """
        clean_scatter = clean["templates"]["right"]["scatter"]
        biased_scatter = biased["templates"]["right"]["scatter"]
        moved_scatter = abs(biased_scatter - clean_scatter) / clean_scatter
        moved_mean = abs(
            biased["templates"]["right"]["mean"] - clean["templates"]["right"]["mean"]
        )
        assert moved_scatter < 1e-3
        assert moved_mean > 1.0
        assert abs(clean_scatter - 1.0) < 0.1

    def test_the_chi_square_z_uses_each_epochs_own_degrees_of_freedom(self):
        """A ragged campaign, which is what separates this from `coherent_mode`.

        ``Var(sum chi2_k) = 2 sum k``, so the SUM is what is chi-square
        distributed. Reading a row count off the first term -- which is all a
        term-based statistic can do -- gets the null wrong the moment the
        epochs differ in flagging.

        **The raggedness has to be LARGE, and the first version of this test
        got that wrong.** It used sixty epochs at dof 19 and one at dof 0, so
        summing and multiplying differ by 19 out of 1140 -- 1.7 % -- and a
        mutation that replaced the sum with ``n * summaries[0].dof`` moved
        ``chi2_z`` from -0.57 to -0.96 and SURVIVED, comfortably inside a
        band written for noise. Half the campaign at a third of the samples
        separates them by a factor: summed 660 against 1140 multiplied, which
        is the difference between a z of about zero and one of about -10.
        """
        wide = [
            residual_summary(fitted_term(epoch_data(seed)), precision())
            for seed in range(30)
        ]
        # Same five columns, a third of the samples: dof 8 - 5 = 3 against 19.
        narrow_x = np.linspace(0.0, 1.0, 8)
        narrow_design = {
            "a": jnp.asarray(np.stack([np.ones(8), narrow_x], axis=1)),
            "n": jnp.asarray(
                np.stack(
                    [np.sin(2 * np.pi * narrow_x), np.cos(2 * np.pi * narrow_x), narrow_x**2],
                    axis=1,
                )
            ),
        }
        narrow_prec = DiagonalPrecision(sigma=jnp.full((8,), SIGMA))
        narrow = []
        for seed in range(100, 130):
            rng = np.random.default_rng(seed)
            data = jnp.asarray(
                np.asarray(narrow_design["a"]) @ rng.normal(size=2)
                + np.asarray(narrow_design["n"]) @ (rng.normal(size=3) * 4.0)
                + SIGMA * rng.normal(size=8)
            )
            narrow.append(
                residual_summary(
                    compress(narrow_design, data, narrow_prec, SHAPES), narrow_prec
                )
            )

        assert wide[0].dof == 19 and narrow[0].dof == 3
        summed = 30 * 19 + 30 * 3
        multiplied = 60 * 19
        assert multiplied > 1.7 * summed, "the fixture must separate the two rules"

        report = template_modes([*wide, *narrow])
        assert report["chi2_dof"] is None, "a ragged campaign has no single dof"
        assert abs(report["chi2_z"]) < 4.0


class TestHeldOutZ:
    PRIOR = np.eye(2) * 0.25

    @staticmethod
    def _terms(n=12, rogue=None):
        with jax.enable_x64(True):
            prec = DiagonalPrecision(sigma=jnp.full((10,), 0.4, dtype=jnp.float64))
            grid = np.linspace(0.0, 1.0, 10)
            design = np.stack([np.ones(10), grid], axis=1)
            out = []
            for seed in range(n):
                rng = np.random.default_rng(seed)
                shift = rogue if (rogue is not None and seed == n - 1) else 0.0
                data = design @ np.array([1.0, -2.0]) + 0.4 * rng.normal(size=10) + shift
                out.append(
                    compress(
                        {"t": jnp.asarray(design)}, jnp.asarray(data), prec, {"t": (2,)}
                    )
                )
            return out

    def test_it_agrees_with_a_leave_one_out_posterior_refitted_from_scratch(self):
        """The independent oracle, and it shares no code with the recursion.

        ``held_out_z`` subtracts one summand from a total formed once;
        this refits the campaign without that epoch, from the terms, in numpy.
        Measured worst relative error over twelve epochs: 1.5e-16.
        """
        terms = self._terms()
        got = held_out_z(terms, self.PRIOR)
        factors = [np.asarray(t.factor, dtype=float) for t in terms]
        targets = [np.asarray(t.target, dtype=float) for t in terms]
        for left_out in range(len(terms)):
            fisher = self.PRIOR + sum(
                f.T @ f for i, f in enumerate(factors) if i != left_out
            )
            b = sum(
                f.T @ t
                for i, (f, t) in enumerate(zip(factors, targets, strict=True))
                if i != left_out
            )
            covariance = np.linalg.inv(fisher)
            residual = factors[left_out] @ (covariance @ b) - targets[left_out]
            spread = (
                np.eye(factors[left_out].shape[0])
                + factors[left_out] @ covariance @ factors[left_out].T
            )
            expected = float(residual @ np.linalg.solve(spread, residual))
            assert got[left_out]["chi2"] == pytest.approx(expected, rel=1e-10)

    def test_a_rogue_epoch_stands_out_by_a_factor_not_a_tolerance(self):
        scores = held_out_z(self._terms(rogue=6.0), self.PRIOR)
        # Measured: +457.6 against a largest-other of +8.0.
        assert scores[-1]["z"] > 20.0 * max(row["z"] for row in scores[:-1])

    def test_a_clean_campaign_scores_like_noise(self):
        scores = held_out_z(self._terms(), self.PRIOR)
        assert max(row["z"] for row in scores) < 8.0

    def test_a_prior_over_other_columns_is_refused(self):
        with pytest.raises(StructureError, match="prior_fisher has shape"):
            held_out_z(self._terms(), np.eye(3))

    def test_a_prior_mean_of_the_wrong_length_is_refused(self):
        with pytest.raises(StructureError, match="prior_mean has shape"):
            held_out_z(self._terms(), self.PRIOR, np.zeros(3))

    def test_epochs_over_different_latents_are_refused(self):
        terms = self._terms(n=2)
        other = compress(
            {"z": jnp.ones((10, 2))},
            jnp.zeros(10),
            DiagonalPrecision(sigma=jnp.ones(10)),
            {"z": (2,)},
        )
        with pytest.raises(StructureError, match="different latents"):
            held_out_z([*terms, other], self.PRIOR)

    def test_an_empty_campaign_is_refused(self):
        with pytest.raises(StructureError, match="no epochs"):
            held_out_z([], self.PRIOR)


class TestShrinkage:
    def test_an_exact_power_law_returns_its_exponent(self):
        assert shrinkage_power({1: 2.0, 4: 1.0, 16: 0.5, 64: 0.25}) == pytest.approx(
            -0.5, abs=1e-12
        )
        assert shrinkage_power({1: 1.0, 2: 0.5, 4: 0.25, 8: 0.125}) == pytest.approx(
            -1.0, abs=1e-12
        )

    def test_a_uniform_rescaling_cannot_move_it(self):
        """The intercept is free, which is why this number cannot detect a
        fault that scales every epoch's information the same way."""
        base = {1: 2.0, 4: 1.0, 16: 0.5, 64: 0.25}
        for factor in (1.0, 1.5, 0.7):
            scaled = {n: sigma * factor for n, sigma in base.items()}
            assert shrinkage_power(scaled) == pytest.approx(
                shrinkage_power(base), abs=1e-12
            )

    def test_it_pools_over_widths(self):
        pooled = shrinkage_power({1: [2.0, 4.0], 4: [1.0, 2.0], 16: [0.5, 1.0]})
        assert pooled == pytest.approx(-0.5, abs=1e-12)

    def test_one_campaign_size_is_refused(self):
        with pytest.raises(StructureError, match="at least two campaign sizes"):
            shrinkage_power({8: 1.0})

    def test_a_ragged_table_is_refused(self):
        with pytest.raises(StructureError, match="different numbers of widths"):
            shrinkage_power({1: [1.0, 2.0], 4: [1.0]})

    def test_a_non_positive_size_is_refused(self):
        with pytest.raises(StructureError, match="not positive"):
            shrinkage_power({0: 1.0, 4: 0.5})

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_width_that_is_not_a_width_is_refused(self, bad):
        with pytest.raises(StructureError, match="finite and strictly positive"):
            shrinkage_power({1: 1.0, 4: bad})

    def test_the_report_carries_the_caveat_in_the_same_object(self):
        report = shrinkage_report({1: 2.0, 4: 1.0, 16: 0.5})
        assert report["power"] == pytest.approx(-0.5, abs=1e-12)
        assert report["detects_coherent_bias"] is False
        assert report["n_values"] == (1, 4, 16)
        assert "no variance" in report["caveat"]


class TestSystematicFloor:
    """The tightest DIRECTION, which is not the tightest coordinate."""

    @staticmethod
    def _near_collinear():
        """Built INSIDE an x64 context, and every caller must stay inside one.

        A 1e-4 separation between two design columns is below float32's
        working precision for the inverse, and jax truncates a float64 array
        the moment it is operated on with x64 off -- with a warning nobody
        reads. Measured: the first version of this class built the term inside
        the context and read `information()` outside it.
        """
        grid = np.linspace(0.0, 1.0, 10)
        base = np.stack([np.ones(10), grid], axis=1)
        design = np.stack(
            [base[:, 0] + base[:, 1], base[:, 0] + 1.0001 * base[:, 1]], axis=1
        )
        prec = DiagonalPrecision(sigma=jnp.full((10,), 0.4, dtype=jnp.float64))
        data = design @ np.array([1.0, -2.0]) + 0.4 * np.random.default_rng(
            3
        ).normal(size=10)
        return compress(
            {"t": jnp.asarray(design)}, jnp.asarray(data), prec, {"t": (2,)}
        )

    def test_a_coordinate_reading_would_have_missed_it_by_four_orders(self):
        """The measurement the whole function turns on.

        Measured on this design: both coordinate widths are **702**, while the
        tightest direction is **0.0583** -- twelve thousand times narrower,
        along (0.707, 0.707). A campaign keyed on ``min(diag(covariance))``
        reports an error bar comfortably above a 0.1 floor while its actual
        tightest direction is well under it.
        """
        with jax.enable_x64(True):
            term = self._near_collinear()
            prior = np.eye(2) * 1e-6
            covariance = np.linalg.inv(np.asarray(term.information(), dtype=float) + prior)
            coordinates = np.sqrt(np.diag(covariance))
            tight, direction = tightest_direction(covariance)
            assert coordinates.min() > 500.0
            assert tight < 0.1
            assert coordinates.min() / tight > 1000.0
            assert direction is not None
            assert abs(abs(direction[0]) - abs(direction[1])) < 0.01

            report = systematic_floor(term, prior, {"t": 0.1}, n_epochs=1)
            assert report["t"]["below_floor"] is True
            assert report["t"]["sigma"] == pytest.approx(tight)
            assert not (coordinates.min() <= 0.1), "a coordinate reading would say False"

    def test_the_direction_sign_is_fixed_rather_than_lapacks_mood(self):
        block = np.array([[1.0, 0.9], [0.9, 1.0]])
        _, direction = tightest_direction(block)
        assert direction[int(np.argmax(np.abs(direction)))] > 0.0

    def test_a_poisoned_block_reports_nan_rather_than_raising(self):
        """NaN cannot be eigendecomposed -- ``eigh`` raises -- and a poisoned
        campaign has to report ``nan`` rather than a linear-algebra error, so
        that the caller's own comparison decides what to do about it."""
        width, direction = tightest_direction(np.array([[np.nan, 0.0], [0.0, 1.0]]))
        assert np.isnan(width) and direction is None

    def test_information_that_is_not_positive_definite_is_refused(self):
        """The reachable half of the poisoned-campaign question.

        ``systematic_floor``'s comparison is written ``not (sigma > floor)``
        rather than ``sigma <= floor``, because NaN is False for both and the
        second form would wave a poisoned campaign through while the same dict
        reported the nan. **Measured, that branch cannot be reached from
        here**: a non-finite information matrix makes ``cholesky`` raise
        first, and it raises long before the inverse could overflow to inf --
        an eigenvalue separation of 1e-160 is already refused while 1e-12
        still returns a finite covariance of order 1e12. So the NaN-safe form
        is defence against a future change in how the covariance is formed,
        not against an input, and a mutation between the two forms SURVIVES
        this suite for that reason. What is reachable, and what this pins, is
        the refusal itself.
        """
        with jax.enable_x64(True):
            term = self._near_collinear()
            with pytest.raises(StructureError, match="positive definite"):
                systematic_floor(term, -np.eye(2) * 1e9, {"t": 0.1}, n_epochs=1)

    def test_the_crossing_epoch_is_computed_and_it_is_reached(self):
        """Not quoted: the extrapolation is checked by actually growing the
        campaign to the epoch it names and measuring the width there."""
        with jax.enable_x64(True):
            grid = np.linspace(0.0, 1.0, 10)
            design = np.stack([np.ones(10), grid], axis=1)
            prec = DiagonalPrecision(sigma=jnp.full((10,), 0.4, dtype=jnp.float64))

            def campaign(n):
                total = None
                for seed in range(n):
                    rng = np.random.default_rng(seed)
                    data = design @ np.array([1.0, -2.0]) + 0.4 * rng.normal(size=10)
                    term = compress(
                        {"t": jnp.asarray(design)}, jnp.asarray(data), prec, {"t": (2,)}
                    )
                    total = (
                        term
                        if total is None
                        else type(term).combine(total, term)
                    )
                return total

            prior = np.eye(2) * 0.25
            start = campaign(2)
            floor = 0.05
            predicted = systematic_floor(start, prior, {"t": floor}, n_epochs=2)["t"]
            assert predicted["below_floor"] is False
            target = predicted["crossing_epoch"]
            assert isinstance(target, int) and target > 2
            reached = systematic_floor(campaign(target), prior, {"t": floor}, n_epochs=target)
            assert reached["t"]["below_floor"] is True
            # ... and one epoch earlier it was NOT, or the prediction is merely
            # a number that happens to be large enough.
            earlier = systematic_floor(
                campaign(target - 1), prior, {"t": floor}, n_epochs=target - 1
            )
            assert earlier["t"]["sigma"] > reached["t"]["sigma"]

    def test_a_floor_on_a_latent_the_campaign_is_not_over_is_refused(self):
        with jax.enable_x64(True):
            term = self._near_collinear()
            with pytest.raises(StructureError, match="not over"):
                systematic_floor(term, np.eye(2) * 1e-6, {"elsewhere": 0.1}, n_epochs=1)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_a_floor_that_is_not_a_width_is_refused(self, bad):
        with jax.enable_x64(True):
            term = self._near_collinear()
            with pytest.raises(StructureError, match="finite positive widths"):
                systematic_floor(term, np.eye(2) * 1e-6, {"t": bad}, n_epochs=1)

    def test_a_campaign_of_no_epochs_is_refused(self):
        with jax.enable_x64(True):
            term = self._near_collinear()
            with pytest.raises(StructureError, match="at least one epoch"):
                systematic_floor(term, np.eye(2) * 1e-6, {"t": 0.1}, n_epochs=0)
