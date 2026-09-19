"""Physical checks for equal-area TRIS feature integration."""

import numpy as np
import pytest

from examples.inference.tris_quadrature import feature_maps, integrate_features


def test_features_preserve_sky_and_region_partition():
    template = np.arange(12.0) + 10
    region = np.arange(12) % 3
    columns = feature_maps(template, region)
    np.testing.assert_array_equal(columns[:, :3].sum(1), template)
    np.testing.assert_array_equal(columns[:, 3:6].sum(1), 1)
    with pytest.raises(ValueError):
        feature_maps(template, region + 1)


def _fixture():
    hp = pytest.importorskip("healpy")
    tris = pytest.importorskip("limTOD.tris")
    from limTOD.tris.geometry import tris_zenith_geometry

    angle = np.linspace(0, 180, 181)
    cuts = tris.TRISPrincipalPlaneCuts(angle, -np.minimum((angle / 10) ** 2, 60),
                                     -np.minimum((angle / 7) ** 2, 60))
    theta, phi = hp.pix2ang(8, np.arange(hp.nside2npix(8)))
    template = 20 + 10 * np.sin(theta) * np.cos(phi)
    region = np.arange(len(theta)) % 3
    return cuts, tris_zenith_geometry(np.array([7.0, 109.0, 231.0])), template, region


def test_quadrature_oracle_agrees_at_nonzero_roll_and_conserves_monopole():
    cuts, geom, template, region = _fixture()
    vector = integrate_features(cuts, geom, template, region, nside=32)
    oracle = integrate_features(cuts, geom, template, region, nside=32, frame="spherical")
    np.testing.assert_allclose(vector, oracle, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(vector[..., -1], 1, atol=1e-13)
    np.testing.assert_allclose(vector[..., 3:6].sum(-1), 1, atol=1e-13)


def test_subpixel_integration_keeps_constant_sky_on_refinement_and_downgrade():
    cuts, geom, template, region = _fixture()
    for nside in (4, 16):
        columns = integrate_features(cuts, geom, np.full_like(template, 17), region, nside=nside)
        np.testing.assert_allclose(columns[..., :3].sum(-1), 17, atol=1e-12)


def test_width_perturbations_change_anisotropic_sky_prediction():
    cuts, geom, template, region = _fixture()
    columns = integrate_features(cuts, geom, template, region, nside=32,
                                 widths=((0, 0), (0.01, -0.01)))
    assert np.max(np.abs(columns[0, :, :3].sum(-1) - columns[1, :, :3].sum(-1))) > 1e-5
