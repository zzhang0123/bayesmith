import numpy as np
import pytest

from examples.inference.tris_sky_components import (
    component_fields,
    free_free_temperature,
    smooth_weights,
)


def test_freefree_zero_thin_and_opaque_limits():
    te = 7000.0
    assert free_free_temperature(0.408, 0.0, te) == 0
    thin = free_free_temperature(np.array([0.4, 0.8, 10.0]), 1e-10, te)
    assert np.all(thin > 0) and np.all(np.diff(thin) < 0)
    np.testing.assert_allclose(
        free_free_temperature(0.6, 2e-10, te),
        2 * free_free_temperature(0.6, 1e-10, te),
        rtol=1e-12,
    )
    assert free_free_temperature(0.01, 1e20, te) == te
    with pytest.raises(ValueError):
        free_free_temperature(0.4, -1, te)


def test_smooth_partition_positive_periodic_and_continuous_at_equator():
    latitude = np.linspace(-90, 90, 201)
    longitude = np.linspace(-np.pi, np.pi, 201)
    weights = smooth_weights(latitude, longitude)
    assert weights.shape == (201, 6) and weights.min() >= 0
    np.testing.assert_allclose(weights.sum(-1), 1, rtol=0, atol=3e-16)
    np.testing.assert_allclose(
        smooth_weights(latitude, longitude + 2 * np.pi), weights, atol=1e-15
    )
    np.testing.assert_array_equal(smooth_weights(-1e-5, 0), smooth_weights(1e-5, 0))
    for pole in (-90.0, 90.0):
        np.testing.assert_allclose(
            smooth_weights(pole, longitude),
            np.broadcast_to(smooth_weights(pole, 0), weights.shape),
            atol=1e-15,
        )


def test_signed_source_field_is_retained_and_components_have_distinct_columns():
    h = np.arange(12.0) + 20
    w = smooth_weights(np.linspace(-80, 80, 12), np.arange(12.0))
    source = np.linspace(-1, 2, 12)
    f = np.tile(np.array([1.0, 0.4, 0.2])[:, None], (1, 12))
    fields = component_fields(h, w, source, f)
    np.testing.assert_allclose(fields[:, :6].sum(-1), h)
    np.testing.assert_allclose(fields[:, 12:18].sum(-1), f[0])
    np.testing.assert_array_equal(fields[:, 18], source)
