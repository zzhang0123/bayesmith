"""Per-parameter r-hat and ESS, and why the threshold is not a constant.

B7 in the migration spec asks for per-parameter r-hat and ESS on the Gibbs
product, and warns that the per-parameter version needs **its own** threshold
argument -- rheplicant's 1.05 was argued for a single chain and a single joint
scalar, and may not be copied across.

That warning turns out to understate the problem. **A constant threshold on
r-hat is not a well-posed test at all**, whatever constant is chosen, because
split r-hat's null distribution is a function of how much independent
information the chain holds. Measured on this machine, with 4000 independent
coordinates per cell and 27 cells spanning chains in {1, 2, 4}, draws in
{200, 800, 3200} and AR(1) rho in {0, 0.5, 0.9}:

* ``(r_hat_99 - 1)`` scales as ``ESS ** -1.05`` -- a straight ``1 / ESS``,
  fitted across an ESS range of 14.8 to 12693, a factor of 860.
* ``(r_hat_99 - 1) * ESS`` therefore stays in a narrow band, 4.87 to 11.46 per
  coordinate, while ESS itself moves by that factor of 860.
* A fixed 1.05 threshold at ESS 15 fires on **45.9 %** of *converged* chains,
  and 1.01 fires on 69.9 %. At ESS 12693 both fire on **0.0 %** -- they cannot
  fire, so they are not tests.

A coin flip at one end and a no-op at the other, from the same constant. So
the ceiling here is ``1 + C / ESS``, and ``C`` is the measured constant rather
than the threshold. See :func:`~bayesmith.dispatch.execute.r_hat_ceiling`.

**Multiplicity is folded into C, not bolted on.** Coordinates are independent
under the null, so a family-wise 1 % alarm rate over ``P`` of them is the
per-coordinate quantile at ``0.99 ** (1/P)``. Measured maxima across the same
27 cells: 11.46 at P=1, 14.44 at P=10, 18.76 at P=100, 22.61 at P=1000 and
22.99 at P=10000 -- so it saturates, and a cap is honest rather than lazy.

The probes that produced these numbers are reproducible from the docstring of
:func:`~bayesmith.dispatch.execute.r_hat_ceiling`, which states the model they
sampled. They are not re-run here: 27 cells of 4000 coordinates is minutes,
and what the suite needs to hold is the CONSEQUENCE of those numbers, which
is cheap.
"""

import numpy as np
import pytest

from bayesmith.dispatch.execute import (
    SiteDiagnostic,
    chain_diagnostics,
    r_hat_ceiling,
)


def _converged(draws, coords, seed=0, rho=0.0):
    """A chain that IS sampling the target, optionally autocorrelated."""
    g = np.random.default_rng(seed)
    x = g.standard_normal((draws, coords))
    if rho:
        for t in range(1, draws):
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho**2) * x[t]
    return x


class TestTheCeilingIsAFunctionOfEssAndNotAConstant:
    """The threshold argument B7 demands, as assertions rather than prose."""

    def test_a_fixed_threshold_is_too_tight_at_low_ess(self):
        """1.05 at ESS 15 fires on 45.9 % of converged chains -- measured.

        So the ceiling must be ABOVE 1.05 there, or the diagnostic reports
        failure on chains that are sampling the target correctly.
        """
        assert r_hat_ceiling(15.0, coordinates=1) > 1.05

    def test_the_same_fixed_threshold_is_a_no_op_at_high_ess(self):
        """At ESS 12693 a converged chain never reaches 1.05, measured at
        0.0 % -- so 1.05 cannot fire and is not a test. The ceiling has to
        come down with ESS or it stops asking anything."""
        assert r_hat_ceiling(12693.0, coordinates=1) < 1.01

    def test_the_ceiling_falls_as_one_over_ess(self):
        """The measured scaling: ESS ** -1.05, which is 1/ESS to the accuracy
        the fit supports. Ten times the ESS, a tenth of the margin."""
        near = r_hat_ceiling(100.0, coordinates=1) - 1.0
        far = r_hat_ceiling(1000.0, coordinates=1) - 1.0
        assert near / far == pytest.approx(10.0, rel=1e-9)

    def test_more_coordinates_raise_the_ceiling(self):
        """Checking P coordinates and reporting the worst inflates the null.
        A report over 100 coordinates must be more forgiving per coordinate
        than a report over one, or the family-wise alarm rate climbs with the
        size of the model rather than with its health."""
        assert r_hat_ceiling(200.0, coordinates=100) > r_hat_ceiling(
            200.0, coordinates=1
        )

    def test_the_multiplicity_term_saturates_rather_than_growing_forever(self):
        """Measured: C is 22.61 at P=1000 and 22.99 at P=10000 -- it stops.
        An unbounded term would make a large model unfalsifiable."""
        assert r_hat_ceiling(200.0, coordinates=10**6) == pytest.approx(
            r_hat_ceiling(200.0, coordinates=10**4), rel=0.05
        )

    def test_the_constant_matches_what_was_measured_at_one_coordinate(self):
        """C(1) is pinned at the measured maximum, 11.46 rounded up. Pinned as
        a PRODUCT rather than as the ceiling at some ESS, because the product
        is the thing the probe measured and the ceiling is derived from it."""
        assert (r_hat_ceiling(50.0, coordinates=1) - 1.0) * 50.0 == pytest.approx(
            11.5, rel=1e-9
        )


class TestTheReportIsPerParameter:
    def test_every_site_gets_its_own_entry(self):
        report = chain_diagnostics(
            {"a": _converged(400, 3, seed=1), "b": _converged(400, 2, seed=2)}
        )
        assert set(report) == {"a", "b"}
        assert all(isinstance(v, SiteDiagnostic) for v in report.values())

    def test_the_entry_names_which_coordinate_decided_it(self):
        """Attribution is the whole point over the existing min-over-everything
        scalar: that number is the worst one, and says nothing about where it
        came from. A stuck coordinate in a 64-element site is a different
        morning's work depending on which element it is."""
        draws = _converged(400, 4, seed=3)
        draws[:, 2] = 0.0  # frozen, and not at either end
        report = chain_diagnostics({"alm": draws})
        assert report["alm"].worst == (2,)

    def test_a_scalar_site_reports_an_empty_coordinate(self):
        report = chain_diagnostics({"w": _converged(400, 1, seed=4).ravel()})
        assert report["w"].worst == ()


class TestAFrozenCoordinateCannotPassQuietly:
    """The trap this guard exists for, and it is not hypothetical.

    numpyro returns ``nan`` for both r-hat and ESS on a coordinate that never
    moved -- measured. And ``nan > ceiling`` is ``False``, so the most
    unconverged parameter possible would pass a naive comparison silently,
    while a caller reading the number saw ``nan`` and could not tell whether
    the check had run.
    """

    def test_a_frozen_coordinate_reads_as_unconverged(self):
        draws = _converged(400, 3, seed=5)
        draws[:, 1] = 2.5  # never moves
        report = chain_diagnostics({"g": draws})
        assert not report["g"].converged
        assert report["g"].worst == (1,)

    def test_its_r_hat_is_infinite_rather_than_nan(self):
        """Infinity is the honest value AND the one that survives comparison.
        nan would make the verdict depend on which side of the operator it
        landed on."""
        draws = _converged(400, 2, seed=6)
        draws[:, 0] = -1.0
        assert np.isinf(chain_diagnostics({"g": draws})["g"].r_hat)

    def test_a_wholly_frozen_site_is_unconverged_too(self):
        report = chain_diagnostics({"g": np.zeros((400, 2))})
        assert not report["g"].converged


class TestItSeparatesConvergedFromNot:
    def test_a_converged_chain_at_a_realistic_size_is_not_flagged(self):
        """The null case. If this is flaky the ceiling is too tight, which is
        the failure mode that makes people switch a diagnostic off."""
        report = chain_diagnostics({"x": _converged(2000, 20, seed=7)})
        assert report["x"].converged

    def test_an_autocorrelated_but_converged_chain_is_not_flagged_either(self):
        """rho=0.9 at 2000 draws is ESS of order 100. The chain is slow, not
        wrong, and the ceiling is supposed to know the difference -- that is
        precisely what tying it to ESS buys."""
        report = chain_diagnostics({"x": _converged(2000, 5, seed=8, rho=0.9)})
        assert report["x"].converged

    def test_two_halves_at_different_locations_are_flagged(self):
        """The thing r-hat is FOR: a chain that has not forgotten where it
        started. Split r-hat compares the halves, so a shifted second half is
        the canonical failure."""
        draws = _converged(400, 2, seed=9)
        draws[200:, 0] += 8.0
        report = chain_diagnostics({"x": draws})
        assert not report["x"].converged
        assert report["x"].worst == (0,)

    def test_the_healthy_neighbour_does_not_drag_the_verdict_down(self):
        """A site is judged by its worst coordinate, but a site with no bad
        coordinate must stay clean even when another site is filthy."""
        bad = _converged(400, 1, seed=10)
        bad[200:] += 8.0
        report = chain_diagnostics(
            {"good": _converged(400, 3, seed=11), "bad": bad}
        )
        assert report["good"].converged
        assert not report["bad"].converged


class TestTheEdgesNumpyroWillNotHandle:
    def test_too_few_draws_is_refused_by_name(self):
        """numpyro asserts below four draws, and a bare AssertionError from a
        library is not a diagnosis. Measured: three draws raises, four works."""
        with pytest.raises(ValueError, match="at least 4 draws"):
            chain_diagnostics({"x": _converged(3, 2, seed=12)})

    def test_four_draws_is_allowed_because_numpyro_allows_it(self):
        """The floor is numpyro's, not one invented here -- so it is checked
        from both sides rather than asserted from one."""
        assert "x" in chain_diagnostics({"x": _converged(4, 2, seed=13)})

    def test_the_draw_floor_is_applied_per_chain_and_not_to_the_stack(self):
        """``get_samples()`` concatenates chains along the draw axis, so the
        unstacking has to happen before anything counts draws.

        Written this way because it DISCRIMINATES. Six draws is comfortably
        above numpyro's floor of four as one chain, and is three per chain as
        two -- below it. An implementation that forgot to reshape would let
        this through, and one that reshapes cannot. The obvious version of
        this test, two displaced chains, passes either way and so proves
        nothing: as one long chain, split r-hat compares the first half to the
        second and finds the same disagreement.
        """
        stacked = _converged(6, 1, seed=14)
        assert "x" in chain_diagnostics({"x": stacked})           # 6 as one chain
        with pytest.raises(ValueError, match="at least 4 draws"):
            chain_diagnostics({"x": stacked}, num_chains=2)       # 3 per chain


class TestWhyTheEssFloorComesFirst:
    """The measurement that overturned this module's first design.

    The ceiling was written as the whole gate. It has essentially no power on
    its own, because a displaced chain drives ESS DOWN faster than it drives
    r-hat UP, so ``1 + C/ESS`` rises to meet the statistic it should reject.
    Measured over fifteen cells of two displaced chains: every one sneaks
    through. These two tests pin the worst of them, so the ordering cannot be
    quietly reversed by someone who reads the ceiling as sufficient.
    """

    def test_two_chains_stuck_apart_are_caught(self):
        a = _converged(200, 1, seed=16)
        b = _converged(200, 1, seed=17) + 8.0
        report = chain_diagnostics({"x": np.concatenate([a, b])}, num_chains=2)
        assert not report["x"].converged
        assert "effective sample size" in report["x"].reason

    def test_the_ceiling_alone_would_have_forgiven_it(self):
        """The number that made the floor necessary, kept executable.

        At separation 8 the measured pair was r-hat 9.25 against ESS 1.02 --
        and the ceiling at that ESS is above 12, so the ceiling ADMITS a
        9.25. If this ever stops being true the floor may be reconsidered;
        while it is true, r-hat cannot be the first question.
        """
        assert r_hat_ceiling(1.02, coordinates=1) > 9.25
