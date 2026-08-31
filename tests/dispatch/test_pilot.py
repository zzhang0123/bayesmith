"""The A4 pilot: the asymmetric verdict, and what it refuses to conclude.

These tests pin the two rules the plan states and nothing else. In
particular they never assert an absolute canonical correlation: the same
funnel reads 0.1018 to 0.2389 over twenty seeds of one construction and
0.619 under another, so an absolute value is a property of the estimator.
The **ratio** and the **verdict** are what is asserted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bayesmith.dispatch.pilot import (
    DECLARED_MULTIPLE,
    augment,
    canonical_correlation,
    pilot_from_samples,
    pilot_is_warranted,
    pilot_report,
    quadratic_cc_crosses_floor,
    ratio_exceeds_declared_multiple,
    resolve_switch,
    sampling_floor,
)

_DRAWS = 200_000


def _funnel(seed: int, draws: int = _DRAWS) -> tuple[np.ndarray, np.ndarray]:
    """Neal's funnel: v ~ N(0, 3), x ~ N(0, exp(v/2)).

    The Laplace correlation between v and x is exactly 0.0 -- the dependence
    is entirely in the second moment, which is the geometry the pilot exists
    to notice and a coupling diagnostic cannot.
    """
    rng = np.random.default_rng(seed)
    v = rng.normal(0.0, 3.0, size=draws)
    x = rng.normal(0.0, np.exp(v / 2.0))
    return v[:, None], x[:, None]


def _gaussian_pair(seed: int, draws: int = _DRAWS) -> tuple[np.ndarray, np.ndarray]:
    """A jointly Gaussian pair at rho=0.6: nothing for the squares to find."""
    rng = np.random.default_rng(1000 + seed)
    z = rng.normal(size=(draws, 2))
    a = z[:, 0]
    b = 0.6 * z[:, 0] + math.sqrt(1.0 - 0.36) * z[:, 1]
    return a[:, None], b[:, None]


def test_the_funnel_vetoes_the_switch_and_names_the_geometry():
    """Seed 0, 200 000 draws: linear 0.007992, quadratic 0.115687, ratio 14.48.

    The ratio is asserted against the declared multiple, not against 14.48:
    it is 8.36 at the worst of twenty seeds and 292 at the best, so pinning
    the number would pin the seed rather than the phenomenon.
    """
    report = pilot_report(*_funnel(0), n_eff=float(_DRAWS))

    assert report.ratio > DECLARED_MULTIPLE
    assert report.quadratic_cc > report.floor
    assert report.vetoed
    assert "funnel" in report.reason
    # A veto is not blind: it found the thing a Gaussian-only reading misses.
    assert report.blind_to == ()


def test_a_jointly_gaussian_pair_is_inconclusive_and_reports_its_blindness():
    """The null: squares add nothing, so the ratio sits at one.

    Measured over twenty seeds the null ratio never left
    [1.0000001, 1.0000508] -- seven times below the declared multiple.
    """
    report = pilot_report(*_gaussian_pair(0), n_eff=float(_DRAWS))

    assert report.ratio == pytest.approx(1.0, abs=1e-3)
    assert not report.vetoed
    assert report.blind_to == ("gaussian-only",)
    assert "abstention, not endorsement" in report.reason
    assert "blind_to=('gaussian-only',)" in report.line()


def test_an_inconclusive_pilot_leaves_the_decision_exactly_as_it_was():
    """**If the pilot abstains here, this test goes red.**

    The plan's rule, in its own words: no conclusion means A3's decision
    stands. A pilot that found no curvature has not earned the right to
    override a decision taken on other evidence -- and a pilot that answered
    "I do not know" by returning the published default would be a coin flip
    wearing a verdict's clothes, because it would silently reverse every
    switch it was run on.
    """
    report = pilot_report(*_gaussian_pair(0), n_eff=float(_DRAWS))

    assert not report.vetoed
    assert resolve_switch("split", "collapse", report) == "collapse"
    assert resolve_switch("collapse", "split", report) == "split"


def test_a_veto_returns_the_published_default_and_proposes_nothing():
    report = pilot_report(*_funnel(0), n_eff=float(_DRAWS))

    assert report.vetoed
    assert resolve_switch("split", "collapse", report) == "split"
    # It hands back the caller's own published default, never a third answer
    # of its own: the pilot's ESS cannot support a better c.
    assert resolve_switch("joint", "collapse", report) == "joint"


def test_the_verdict_survives_a_spread_of_absolute_readings():
    """Three seeds, three quite different absolute readings, one verdict.

    This is the measurement that forbids asserting an absolute value: the
    quadratic canonical correlation moves by more than a factor of 1.5 across
    these three seeds of one and the same distribution, while every one of
    them vetoes.
    """
    reports = [pilot_report(*_funnel(seed), n_eff=float(_DRAWS)) for seed in (0, 1, 2)]
    readings = [report.quadratic_cc for report in reports]

    assert max(readings) / min(readings) > 1.5
    assert all(report.vetoed for report in reports)


def test_too_little_information_cannot_veto_however_large_the_ratio():
    """The sampling floor is a real gate, not decoration.

    The same seed-0 funnel, declared to carry only 100 effective draws: the
    floor rises to sqrt(4/100) = 0.2, above the 0.1157 quadratic reading, and
    the pilot is inconclusive even though its ratio is 14.48. A pilot without
    the information to support a veto does not cast one.
    """
    report = pilot_report(*_funnel(0), n_eff=100.0)

    assert report.ratio > DECLARED_MULTIPLE
    assert report.floor == pytest.approx(0.2)
    assert report.quadratic_cc < report.floor
    assert not report.vetoed
    assert resolve_switch("split", "collapse", report) == "collapse"


def test_both_conditions_are_open_at_their_own_boundary():
    """Neither gate admits its own threshold; both are strict inequalities."""
    floor = sampling_floor(4, 200_000.0)

    assert not quadratic_cc_crosses_floor(floor, floor)
    assert quadratic_cc_crosses_floor(float(np.nextafter(floor, math.inf)), floor)
    assert not ratio_exceeds_declared_multiple(DECLARED_MULTIPLE)
    assert ratio_exceeds_declared_multiple(
        float(np.nextafter(DECLARED_MULTIPLE, math.inf))
    )


def test_the_sampling_floor_falls_as_the_root_of_the_information():
    """sqrt(p_aug / N_eff): four times the draws halves the floor."""
    assert sampling_floor(4, 200_000.0) == pytest.approx(0.0044721359549995795)
    assert sampling_floor(4, 800_000.0) == pytest.approx(sampling_floor(4, 200_000.0) / 2)
    assert sampling_floor(16, 200_000.0) == pytest.approx(
        2 * sampling_floor(4, 200_000.0)
    )


def test_the_quadratic_features_contain_the_linear_ones():
    """The squares are appended, never substituted.

    That containment is what makes the ratio meaningful: the quadratic
    reading is taken over a feature space that includes the linear one, so a
    ratio below one is sampling noise rather than a competing measurement.
    """
    block = np.arange(6.0).reshape(3, 2)
    augmented = augment(block)

    assert augmented.shape == (3, 4)
    assert np.array_equal(augmented[:, :2], block)
    assert np.array_equal(augmented[:, 2:], block * block)


def test_the_canonical_correlation_is_invariant_to_an_affine_rescale():
    """A change of units cannot move a canonical correlation.

    The counterpart of the MAP scale finding: a quantity that decides
    anything has to give the same answer in two unit systems.
    """
    left, right = _gaussian_pair(0, draws=20_000)
    plain = canonical_correlation(left, right)
    rescaled = canonical_correlation(left * 1e6 + 7.0, right * 1e-3 - 2.0)

    assert rescaled == pytest.approx(plain, rel=1e-9)


def test_the_pilot_runs_only_where_there_is_something_to_veto():
    """A stuck pilot costs zero because it is not reached at all."""
    assert pilot_is_warranted(switches_away=True, contested=False)
    assert pilot_is_warranted(switches_away=False, contested=True)
    assert not pilot_is_warranted(switches_away=False, contested=False)


def test_named_sites_are_stacked_with_the_draw_axis_leading():
    """A plated site contributes one column per coordinate, never a mean."""
    left, right = _funnel(0, draws=32)
    samples = {
        "v": left[:, 0],
        "x": np.tile(right, (1, 3)),
        "unused": np.full((32, 5), 9.0),
    }

    report = pilot_from_samples(samples, ("v",), ("x",), n_eff=32.0)

    assert report.p_aug == 2 + 6
    assert math.isfinite(report.linear_cc)
