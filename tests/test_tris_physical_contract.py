import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal

from examples.inference.tris_physical_contract import (
    bounded_offset_logpdf,
    check_independent_products,
    tris820_offset_bounds,
)


def product(name, ancestors, groups=(), role='measurement'):
    return {'id': name, 'observation_ancestors': ancestors,
            'calibration_groups': groups, 'role': role}


def test_raw_lineage_blocks_renamed_derived_maps_and_duplicate_haslam_products():
    a = product('haslam_ds', ['haslam1982'])
    b = product('renamed_map', ['haslam1982', 'planck'])
    with pytest.raises(ValueError, match='shared observations'):
        check_independent_products([a, b])
    with pytest.raises(ValueError, match='not an admitted'):
        check_independent_products([product('commander', ['planck'], role='conditional')])


def test_shared_calibration_requires_joint_nuisances_not_duplicate_data():
    a = product('north', ['stockert'], ['reich_scale'])
    b = product('south', ['villa_elisa'], ['reich_scale'])
    with pytest.raises(ValueError, match='joint model'):
        check_independent_products([a, b])
    check_independent_products([a, b], joint_calibration_groups=['reich_scale'])
    with pytest.raises(ValueError, match='duplicate'):
        check_independent_products([a, a], joint_calibration_groups=['reich_scale'])


def test_calibration_bounds_reverse_the_old_physical_minus_observed_correction():
    assert tris820_offset_bounds('laboratory_uniform') == (-0.660, 0.660)
    assert tris820_offset_bounds('astrophysical_uniform_conditional') == (-0.430, 0.300)
    with pytest.raises(ValueError):
        tris820_offset_bounds('default')


def test_common_bounded_zero_matches_independent_numerical_integration():
    covariance = np.array([[0.1, 0.02], [0.02, 0.3]])
    for residual in (np.array([0.2, 0.3]), np.array([2.0, 2.1]), np.array([-2.0, -2.1])):
        expected = quad(lambda z, residual=residual: multivariate_normal.pdf(residual - z, cov=covariance),
                        -0.430, 0.300, epsabs=1e-30)[0] / 0.730
        actual = bounded_offset_logpdf(residual, covariance, lower=-0.430, upper=0.300)
        np.testing.assert_allclose(actual, np.log(expected), rtol=2e-13)
    # A bounded common offset cannot explain a contrast between rows.
    same = bounded_offset_logpdf([0.2, 0.2], np.eye(2)*0.01, lower=-0.3, upper=0.3)
    different = bounded_offset_logpdf([-0.2, 0.2], np.eye(2)*0.01, lower=-0.3, upper=0.3)
    assert different < same
