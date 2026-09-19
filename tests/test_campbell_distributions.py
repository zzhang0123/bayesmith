"""Density plots must compare distributions in the same coordinates."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from examples.inference.campbell_distributions import edgeworth_pdf, mixture_logpdf


@pytest.mark.parametrize("sky", [False, True])
def test_plotted_mixture_normalization_and_moments_match_campbell(sky):
    values = []
    for order in range(4):
        value, error = quad(
            lambda x, order=order: x**order * np.exp(mixture_logpdf(x, 2.0, sky=sky)),
            -12,
            30,
            epsabs=1e-9,
        )
        assert error < 1e-7
        values.append(value)
    expected = [
        1,
        0,
        0.4 if sky else 1,
        0.8**1.5 / np.sqrt(2) * (0.6**3 + 4 * 0.1**3 if sky else 1),
    ]
    np.testing.assert_allclose(values, expected, atol=1e-7, rtol=0)


def test_signed_edgeworth_is_normalized_but_has_negative_density():
    integral, error = quad(lambda x: edgeworth_pdf(x, 2.0), -12, 12, epsabs=1e-10)
    assert abs(integral - 1) < 1e-9 and error < 1e-9
    assert edgeworth_pdf(-3.181, 2.0) < 0
    assert np.exp(mixture_logpdf(-3.181, 2.0)) > 0


def test_conditional_pdf_uses_the_noisy_evidence_normalizer():
    y = 1.2
    score = mixture_logpdf(y, 2.0, noise=0.1)
    integral, error = quad(
        lambda x: np.exp(mixture_logpdf(x, 2.0) + norm.logpdf(y, x, 0.1) - score),
        -3,
        5,
        epsabs=1e-9,
    )
    assert abs(integral - 1) < 1e-8 and error < 1e-8
