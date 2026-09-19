"""Check zero removal against independent Gaussian nuisance integration."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal

from examples.inference.tris_spatial_contrasts import (
    prepare_contrasts,
    spatial_contrast_basis,
)


def fixture():
    observed = np.array([1.3, 0.7, 1.8, 1.1])
    response = np.array([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9], [0.6, 0.4]])
    factor = np.array([[0.3, 0, 0, 0], [0.1, 0.4, 0, 0],
                       [-0.1, 0.1, 0.2, 0], [0.04, 0.05, -0.06, 0.3]])
    return observed, response, factor @ factor.T


def score(prepared, sky):
    return multivariate_normal.logpdf(
        prepared["observed"], mean=prepared["response"] @ sky,
        cov=prepared["covariance"],
    )


@pytest.mark.parametrize("count", [2, 4, 17, 651])
def test_basis_removes_exactly_the_common_measurement_mode(count):
    basis = spatial_contrast_basis(count)
    assert basis.shape == (count - 1, count)
    np.testing.assert_allclose(basis @ np.ones(count), 0, atol=3e-14)
    np.testing.assert_allclose(basis @ basis.T, np.eye(count - 1), atol=3e-14)
    # Also checks that no nonconstant spatial mode has disappeared.
    np.testing.assert_allclose(basis.T @ basis, np.eye(count) - 1/count, atol=3e-14)


def test_contrast_density_matches_integration_over_one_shared_zero():
    observed, response, covariance = fixture()
    prepared = prepare_contrasts(observed, covariance, response)
    for sky in (np.array([0.9, 1.3]), np.array([1.4, 0.5])):
        density = quad(
            lambda zero, sky=sky: multivariate_normal.pdf(
                observed, mean=response @ sky + zero, cov=covariance),
            -np.inf, np.inf, epsabs=1e-12, epsrel=1e-12,
        )[0]
        # Unit orthonormal contrasts use sqrt(n) times the flat-zero integral.
        assert score(prepared, sky) == pytest.approx(
            np.log(density) + 0.5*np.log(len(observed)), abs=2e-12,
        )


def test_correlated_covariance_and_operator_are_propagated_together():
    observed, response, covariance = fixture()
    prepared = prepare_contrasts(observed, covariance, response)
    sky = np.array([1.5, 0.4])
    baseline = score(prepared, sky)
    shifted = prepare_contrasts(observed+7.3, covariance+0.6*np.ones((4, 4)), response)
    assert score(shifted, sky) == pytest.approx(baseline, abs=2e-12)
    assert np.max(np.abs(prepared["covariance"] - np.diag(np.diag(prepared["covariance"])))) > 0.01
    wrong = dict(prepared, covariance=np.diag(np.diag(prepared["covariance"])))
    assert abs(score(wrong, sky) - baseline) > 0.1
    # A spatially varying calibration pattern is not a common zero.
    changed = prepare_contrasts(observed+np.array([0, 0, 1, 0]), covariance, response)
    assert abs(score(changed, sky)-baseline) > 1


def test_arbitrary_pixel_order_preserves_the_density():
    observed, response, covariance = fixture()
    permutation = [2, 0, 3, 1]
    original = prepare_contrasts(observed, covariance, response)
    shuffled = prepare_contrasts(observed[permutation], covariance[np.ix_(permutation, permutation)],
                                 response[permutation])
    assert score(original, np.array([1.2, 0.8])) == pytest.approx(
        score(shuffled, np.array([1.2, 0.8])), abs=2e-12,
    )


def test_sky_absorption_survives_when_the_instrument_response_is_not_constant():
    observed, _, covariance = fixture()
    transmission = np.array([0.8, 0.9, 0.6, 1.0])[:, None]
    prepared = prepare_contrasts(observed, covariance, transmission)
    assert np.linalg.norm(prepared["response"]) > 0.1
    assert abs(score(prepared, np.array([1.0]))-score(prepared, np.array([2.0]))) > 0.1


@pytest.mark.parametrize("count", [True, 1, 0, -1, 3.5])
def test_invalid_number_of_rows_is_rejected(count):
    with pytest.raises(ValueError):
        spatial_contrast_basis(count)


@pytest.mark.parametrize("defect", ["nan", "asymmetry", "indefinite", "singular", "shape", "response"])
def test_invalid_inputs_refuse_instead_of_using_pseudoinverse_or_jitter(defect):
    observed, response, covariance = fixture()
    if defect == "nan":
        observed[0] = np.nan
    elif defect == "asymmetry":
        covariance[0, 1] += 0.1
    elif defect == "indefinite":
        covariance[0, 0] = -1
    elif defect == "singular":
        covariance = np.ones((4, 4))
    elif defect == "shape":
        covariance = covariance[:3, :3]
    else:
        response = response[:3]
    with pytest.raises(ValueError):
        prepare_contrasts(observed, covariance, response)


@pytest.mark.parametrize("argument", [0, 1, 2])
def test_complex_input_is_not_silently_cast_to_real(argument):
    observed, response, covariance = fixture()
    inputs = [observed, covariance, response]
    inputs[argument] = inputs[argument].astype(complex) + 1j
    with pytest.raises(ValueError, match="real-valued"):
        prepare_contrasts(*inputs)


def test_finite_input_overflow_refuses_instead_of_returning_nan():
    observed = np.array([1.7e308, -1.7e308])
    with pytest.raises(ValueError, match="arithmetic"):
        prepare_contrasts(observed, np.eye(2), np.ones((2, 1)))
