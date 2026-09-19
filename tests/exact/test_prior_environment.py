"""The shared anchor retains model semantics independently of classification."""

import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith.exact._environment import prior_environment
from bayesmith.graph.trace import det, observe, sample, trace
from tests.exact.models import (
    improper_outside_prior,
    overflowing_outside_latent,
    plated_student_t_latent,
)


def test_legacy_anchor_imports_are_the_same_function():
    from bayesmith.dispatch import prior_environment as public
    from bayesmith.dispatch.classify import prior_environment as legacy

    assert public is legacy is prior_environment


def test_dependent_prior_and_observation_replay_use_the_anchored_parent():
    def model():
        parent = sample("parent", lambda: dist.Normal(2.0, 0.5))
        child = sample("child", lambda p: dist.Normal(3 * p, 1.0), parent)
        prediction = det("prediction", lambda p, c: p + c, parent, child)
        observe("data", lambda m: dist.Normal(m, 1.0), prediction, obs=17.0)

    environment = prior_environment(trace(model))
    assert float(environment["parent"]) == 2.0
    assert float(environment["child"]) == 6.0
    assert float(environment["prediction"]) == 8.0
    assert float(environment["data"]) == 17.0


@pytest.mark.parametrize("build", [improper_outside_prior, overflowing_outside_latent])
def test_undefined_prior_mean_still_has_a_finite_anchor(build):
    environment = prior_environment(build())
    assert float(environment["z"]) == 0.0
    assert all(np.all(np.isfinite(value)) for value in environment.values())


def test_non_gaussian_plate_anchor_preserves_shape_and_nonzero_mean():
    environment = prior_environment(plated_student_t_latent(n=5))
    np.testing.assert_allclose(environment["u"], np.full(5, 0.4), rtol=1e-7)
