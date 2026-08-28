"""What a campaign can and cannot see about a coherent error.

The load-bearing test here is not that `coherent_mode` detects something. It
is that an error of the SAME SIZE, placed inside the design's column space,
is invisible to it -- because that is what makes
`refuse_undeclared_coherent_error` a refusal rather than a missing feature.

Both injections are constructed, not sampled, so "detected" and "invisible"
are statements about the same known displacement rather than about two
different draws.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.marginal import (
    SqrtInfo,
    coherent_mode,
    epoch_chi_square,
    refuse_undeclared_coherent_error,
)

N_EPOCH, ROWS, WIDTH, SIGMA = 60, 4, 2, 0.5


def _campaign(shift=None, seed=0):
    """`N_EPOCH` whitened epochs sharing one design, with an optional common shift.

    The design REPEATS across epochs, which is what makes an error "coherent"
    -- one calibration solution applied to every night.
    """
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(ROWS, WIDTH)) / SIGMA
    truth = np.asarray([1.5, -0.75])
    terms = []
    for _ in range(N_EPOCH):
        noise = rng.normal(size=ROWS)
        target = design @ truth + noise
        if shift is not None:
            target = target + shift
        terms.append(
            SqrtInfo(
                factor=jnp.asarray(design),
                target=jnp.asarray(target),
                offset=jnp.zeros(()),
                names=("x",),
                shapes=((WIDTH,),),
            )
        )
    return terms, design, truth


def _least_squares(terms):
    """The campaign's own answer, folded and solved."""
    total = terms[0]
    for term in terms[1:]:
        total = SqrtInfo.combine(total, term)
    return np.linalg.solve(
        np.asarray(total.factor), np.asarray(total.target)
    ), total


class TestTheDetectableHalf:
    def test_a_clean_campaign_reads_clean(self):
        with jax.enable_x64(True):
            terms, _, _ = _campaign()
            found, _ = _least_squares(terms)
            report = coherent_mode(terms, {"x": jnp.asarray(found)})
        assert report["n_epochs"] == N_EPOCH
        assert report["chi2_dof"] == ROWS
        assert abs(report["chi2_z"]) < 4.0, report
        assert abs(report["scatter"] - 1.0) < 0.4, report

    def test_an_out_of_span_common_mode_is_detected(self):
        """A shift with a component the design cannot absorb leaves a residual.

        Reported as a z because a common mode moves a MEAN, and a mean over N
        epochs is resolved at sqrt(N) -- so the same shift is more visible in a
        longer campaign, which is the opposite of the in-span half below.

        Asserted against the ANALYTIC prediction rather than a guessed
        magnitude: a unit shift orthogonal to the design lands entirely in the
        residual, so the mean moves by exactly ``s**2`` and the z is
        ``s**2 / sqrt(2 dof / N)``. At ``s = 1.2``, ``dof = 4``, ``N = 60``
        that is 3.94, and the measured 4.50 is the noise on top of it.
        """
        with jax.enable_x64(True):
            _, design, _ = _campaign()
            # a direction in the DATA space orthogonal to the design's columns:
            # the trailing left-singular vectors, which need full_matrices
            left, _, _ = np.linalg.svd(design, full_matrices=True)
            null_direction = left[:, -1]
            assert np.allclose(design.T @ null_direction, 0.0, atol=1e-10)
            shifted, _, _ = _campaign(shift=1.2 * null_direction)
            found, _ = _least_squares(shifted)
            report = coherent_mode(shifted, {"x": jnp.asarray(found)})
        strength = 1.2**2
        assert report["chi2_mean"] == pytest.approx(ROWS + strength, rel=0.15), report
        predicted = strength / math.sqrt(2.0 * ROWS / N_EPOCH)
        assert report["chi2_z"] == pytest.approx(predicted, rel=0.35), (
            report,
            predicted,
        )
        assert report["chi2_z"] > 3.0, report

    def test_the_z_grows_with_the_campaign_but_the_mean_does_not(self):
        """The sqrt(N) that makes this findable at all."""
        global N_EPOCH
        original = N_EPOCH
        try:
            zs, means = [], []
            for length in (30, 120, 480):
                N_EPOCH = length
                with jax.enable_x64(True):
                    _, design, _ = _campaign()
                    direction = np.linalg.svd(design, full_matrices=True)[0][:, -1]
                    terms, _, _ = _campaign(shift=1.2 * direction)
                    found, _ = _least_squares(terms)
                    report = coherent_mode(terms, {"x": jnp.asarray(found)})
                zs.append(report["chi2_z"])
                means.append(report["chi2_mean"])
        finally:
            N_EPOCH = original
        # the mean displacement is a property of the shift, not of N
        assert max(means) - min(means) < 0.5 * means[0], means
        # the z is not: 16x the epochs is about 4x the z, because the mean is
        # resolved at sqrt(N). That factor IS the reason the statistic exists.
        assert zs[2] / zs[0] == pytest.approx(4.0, rel=0.4), zs


class TestTheHalfNoStatisticCanSee:
    """Why the refusal exists, measured on the same size of error."""

    def test_an_in_span_common_mode_is_invisible_and_biases_the_answer(self):
        """THE test this module is for.

        The shift is ``design @ bias`` -- inside the column space -- so the
        campaign absorbs it into the survivors identically in every epoch. The
        residual at the DISPLACED answer is exactly what it was, so every
        statistic here reads clean, and the answer is wrong by exactly
        ``bias``.
        """
        bias = np.asarray([0.3, -0.45])
        with jax.enable_x64(True):
            clean, design, _ = _campaign()
            clean_answer, _ = _least_squares(clean)
            clean_report = coherent_mode(clean, {"x": jnp.asarray(clean_answer)})

            biased, _, _ = _campaign(shift=design @ bias)
            biased_answer, _ = _least_squares(biased)
            biased_report = coherent_mode(biased, {"x": jnp.asarray(biased_answer)})

        # invisible: every reported number is the clean one
        assert biased_report["chi2_z"] == pytest.approx(
            clean_report["chi2_z"], rel=1e-9
        )
        assert biased_report["scatter"] == pytest.approx(
            clean_report["scatter"], rel=1e-9
        )
        # and the answer is displaced by exactly the injected bias
        assert np.allclose(biased_answer - clean_answer, bias, atol=1e-9)

    def test_the_two_halves_are_comparable_in_size(self):
        """So "invisible" is not "small".

        The point of the refusal is that a coherent error's undetectable half
        is the same order as its detectable one. Measured here as whitened
        norms of the two injections used above.
        """
        bias = np.asarray([0.3, -0.45])
        with jax.enable_x64(True):
            _, design, _ = _campaign()
            direction = np.linalg.svd(design, full_matrices=True)[0][:, -1]
            visible = 1.2 * direction
            hidden = design @ bias
        ratio = float(np.linalg.norm(hidden) / np.linalg.norm(visible))
        assert 0.2 < ratio < 5.0, ratio

    def test_a_longer_campaign_makes_the_hidden_half_worse_not_better(self):
        """The one error class that gets worse with more data.

        The bias does not shrink with N while the error bar does, so the
        answer becomes more confidently wrong. Measured as the bias in units
        of the posterior width.
        """
        global N_EPOCH
        original = N_EPOCH
        bias = np.asarray([0.3, -0.45])
        try:
            sigmas = []
            for length in (30, 480):
                N_EPOCH = length
                with jax.enable_x64(True):
                    clean, design, _ = _campaign()
                    clean_answer, total = _least_squares(clean)
                    biased, _, _ = _campaign(shift=design @ bias)
                    biased_answer, _ = _least_squares(biased)
                    covariance = np.linalg.inv(
                        np.asarray(total.factor).T @ np.asarray(total.factor)
                    )
                displacement = biased_answer - clean_answer
                sigmas.append(
                    float(
                        np.sqrt(displacement @ np.linalg.solve(covariance, displacement))
                    )
                )
        finally:
            N_EPOCH = original
        assert np.allclose(sigmas[1] / sigmas[0], 4.0, rtol=0.2), sigmas


class TestTheRefusal:
    def test_an_undeclared_campaign_is_refused(self):
        with pytest.raises(StructureError, match="has not been declared"):
            refuse_undeclared_coherent_error(False)

    def test_the_message_says_why_no_statistic_would_do(self):
        with pytest.raises(StructureError) as caught:
            refuse_undeclared_coherent_error(False)
        message = str(caught.value)
        assert "leaves NO residual" in message
        assert "does not shrink as epochs accumulate" in message
        assert "no statistic to improve" in message

    def test_a_declared_one_passes(self):
        refuse_undeclared_coherent_error(True)


class TestTheInputChecks:
    def test_an_empty_campaign_is_refused_rather_than_read_as_clean(self):
        with pytest.raises(StructureError, match="at least one epoch"):
            epoch_chi_square([], {"x": jnp.zeros(WIDTH)})

    def test_epochs_over_different_latents_are_refused(self):
        with jax.enable_x64(True):
            terms, _, _ = _campaign()
            odd = SqrtInfo(
                factor=terms[0].factor,
                target=terms[0].target,
                offset=terms[0].offset,
                names=("y",),
                shapes=((WIDTH,),),
            )
            with pytest.raises(StructureError, match="different latents"):
                coherent_mode(
                    [terms[0], odd], {"x": jnp.zeros(WIDTH), "y": jnp.zeros(WIDTH)}
                )

    def test_a_term_with_no_rows_is_refused(self):
        with jax.enable_x64(True):
            empty = SqrtInfo(
                factor=jnp.zeros((0, WIDTH)),
                target=jnp.zeros(0),
                offset=jnp.zeros(()),
                names=("x",),
                shapes=((WIDTH,),),
            )
            with pytest.raises(StructureError, match="no rows"):
                coherent_mode([empty], {"x": jnp.zeros(WIDTH)})
