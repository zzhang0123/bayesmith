import numpy as np
import pytest

from examples.inference.tris_physical_audit import (
    omission_experiment,
    pointset_feasibility,
    read_stockert_map,
)


@pytest.mark.parametrize('emission_measure', [100.0, 300.0])
def test_omitted_freefree_can_mimic_curvature_despite_tiny_residuals(emission_measure):
    result = omission_experiment(emission_measure)
    correct = result['fits']['physical']
    wrong = result['fits']['omit_freefree']
    np.testing.assert_allclose(correct['parameters'], [np.log(25), -2.8, 0, np.log(emission_measure)],
                               atol=1e-6, rtol=0)
    assert correct['max_abs_residual_k'] < 1e-8
    # 25-mK toy noise: omitted-ff residual is sub-noise, while kappa is spurious.
    assert wrong['max_abs_residual_k'] < 0.025 / 5
    assert wrong['parameters'][2] > 0.03
    assert abs(correct['local_fisher_kappa_logem_correlation']) > 0.99


def test_floor_audit_is_a_necessary_bound_not_a_sigma_assignment():
    result = pointset_feasibility({'effective_frequency_mhz': 2427.8,
                                   'temperature_k': [2.329, 2.840],
                                   'ra_deg': [171.5, 287.0],
                                   'common_zero_level_k': 0.284})
    assert result['minimum_deficit_k']['cmb_plus_fixed_isotropic'][0] > 0.16
    assert result['minimum_deficit_k']['cmb_plus_fixed_isotropic'][1] == 0
    assert result['minimum_deficit_k']['cmb_only'][0] > 0
    assert 'hypothetical' in result['sigma_warning']


def test_stockert_reader_converts_mk_and_keeps_finite_missing_sentinel(tmp_path):
    hp = pytest.importorskip('healpy')
    fits = pytest.importorskip('astropy.io.fits')
    data = np.full(12 * 256**2, 3500, dtype=np.float32)
    data[0] = -32768
    table = fits.BinTableHDU.from_columns([fits.Column(name='UNKNOWN1', format='E', array=data)])
    table.header.update({'NSIDE': 256, 'COORDSYS': 'G', 'ORDERING': 'NESTED'})
    path = tmp_path / 'map.fits'
    table.writeto(path)
    temperature, valid, metadata = read_stockert_map(path)
    assert metadata['invalid_pixels'] == 1
    assert metadata['input_unit_header'] is None
    assert np.isnan(temperature[hp.nest2ring(256, 0)])
    assert not valid[hp.nest2ring(256, 0)]
    np.testing.assert_array_equal(temperature[valid], np.full(valid.sum(), 3.5))
    with fits.open(path, mode='update') as hdus:
        hdus[1].header['TUNIT1'] = 'K'
    with pytest.raises(ValueError, match='unit'):
        read_stockert_map(path)
