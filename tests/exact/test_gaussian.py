"""The (loc, scale) extractor, and the log_prob probe that checks it."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from bayesmith import const, det, evaluate, observe, sample, to_numpyro, trace
from bayesmith.errors import NotGaussian, StructureError
from bayesmith.exact.gaussian import (
    check_gaussian,
    gaussian_parts,
    node_shape,
    noise_std_at,
    observation_parts,
)
from tests.exact.models import (
    LyingNormal,
    plated_latent,
    radiometer,
    straight_line,
    two_observations,
)

WEIGHT = 2.5
SIGMA = 0.5


def test_gaussian_parts_reads_loc_and_scale_off_a_plain_normal():
    graph = straight_line(weight=WEIGHT, sigma=SIGMA, prior_std=2.0)
    env = evaluate(graph, {"w": jnp.asarray(WEIGHT)})
    loc, scale = gaussian_parts(graph, graph.node("d"), env)
    assert jnp.allclose(loc, WEIGHT * graph.node("X").value)
    assert scale.shape == loc.shape
    assert jnp.allclose(scale, SIGMA)


def test_gaussian_parts_unwraps_a_to_event_wrapper():
    """`.to_event(1)` only changes how log_prob is reduced, not the density."""

    def model():
        xs = const("X", jnp.ones(4))
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, 0.25).to_event(1), mu, obs=jnp.zeros(4))

    graph = trace(model)
    loc, scale = gaussian_parts(graph, graph.node("d"), evaluate(graph, {"w": 1.0}))
    assert loc.shape == (4,)
    assert jnp.allclose(scale, 0.25)


def test_gaussian_parts_refuses_a_node_that_is_not_gaussian():
    def model():
        sample("k", lambda: dist.Gamma(2.0, 3.0))
        observe("d", lambda: dist.Normal(0.0, 1.0), obs=jnp.zeros(()))

    graph = trace(model)
    env = evaluate(graph, {"k": jnp.asarray(1.0)})
    with pytest.raises(NotGaussian, match="Gamma"):
        gaussian_parts(graph, graph.node("k"), env)


def test_check_gaussian_accepts_a_real_normal():
    graph = straight_line()
    env = evaluate(graph, {"w": jnp.asarray(WEIGHT)})
    errors = check_gaussian(graph, graph.node("d"), env)
    assert set(errors) and all(err < 1e-4 for err in errors.values())


def test_check_gaussian_catches_a_distribution_that_lies_about_its_log_prob():
    """The whole reason the probe exists.

    Introspection passes here -- LyingNormal IS a Normal, its `.loc` and
    `.scale` are exactly what the model meant -- and the density is still
    wrong. Delete the probe and this test is the one that goes red.
    """

    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda w_: LyingNormal(w_, 0.7), w, obs=jnp.zeros(3))

    graph = trace(model)
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    loc, scale = gaussian_parts(graph, graph.node("d"), env)  # introspection is happy
    assert jnp.allclose(loc, 0.3) and jnp.allclose(scale, 0.7)
    with pytest.raises(StructureError, match="log_prob"):
        check_gaussian(graph, graph.node("d"), env)


def test_check_gaussian_treats_nan_as_a_failure_in_isolation():
    """Isolates the `not isfinite` half of check_gaussian's own guard.

    `linearity.py`'s `affinity_errors` carries the identical shape of guard
    (``if not finite or errors[scale] > rtol``, with the identical comment
    "NaN must count as a FAILURE") and has a dedicated test for it,
    `test_affinity_errors_treats_nan_as_a_failure_in_isolation` -- added
    during Task 4 after its implementer diagnosed exactly this "mutation
    does not go red" situation there. The same guard exists in this
    module's `check_gaussian` (line: ``if not np.isfinite(errors[offset])
    or errors[offset] > rtol:``) but had no test of its own: measured,
    deleting the ``not np.isfinite(errors[offset]) or`` half here left the
    entire pre-existing suite green.

    NaN, not +inf, is the case that actually needs its own guard: if
    `log_prob` returned +inf instead of NaN, `departure` would be +inf too
    and `errors[offset] > rtol` alone would already catch it (`inf > rtol`
    is True in IEEE-754) -- a check missing the `not isfinite` half would
    still refuse THAT case. `NaN > rtol` is False, so only a clean NaN
    isolates the branch, exactly as the linearity.py sibling test does for
    its own guard.

    NaN only below `loc` (which covers PROBE_OFFSETS' two negative entries,
    -3.0 and -1.0): the loop raises on the FIRST failing probe and
    PROBE_OFFSETS is visited in order, so -3.0 is guaranteed to be the one
    reached, with nothing upstream of it able to raise first for an
    unrelated reason.
    """

    class NaNBelowTheMean(dist.Normal):
        def log_prob(self, value):
            exact = super().log_prob(value)
            return jnp.where(value < self.loc, jnp.nan, exact)

    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda w_: NaNBelowTheMean(w_, 0.7), w, obs=jnp.zeros(()))

    graph = trace(model)
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    loc, scale = gaussian_parts(graph, graph.node("d"), env)  # introspection is happy
    assert jnp.allclose(loc, 0.3) and jnp.allclose(scale, 0.7)
    with pytest.raises(StructureError, match="log_prob"):
        check_gaussian(graph, graph.node("d"), env)


def test_the_probe_evaluates_log_prob_at_the_shape_the_node_s_value_takes():
    """A dist_fn correct on a scalar and wrong on an array must be refused.

    `gaussian_parts` returns dist_fn's own batch shape, which for a plated
    latent with an unplated prior -- and for an unplated observed node with
    vector data -- is a SCALAR while the node's value is an array. Probing at
    the scalar evaluates log_prob at a shape the consumer never uses.

    Measured before this guard existed: every reported error was exactly 0.0
    against a real discrepancy of 2.0e6 nats over 2000 observations.
    """

    class ShapeSensitiveNormal(dist.Normal):
        def log_prob(self, value):
            true = super().log_prob(value)
            return true if jnp.ndim(value) == 0 else true + 1000.0

    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda w_: ShapeSensitiveNormal(w_, 0.7), w, obs=jnp.zeros(200))

    graph = trace(model)
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    loc, _ = gaussian_parts(graph, graph.node("d"), env)
    assert jnp.shape(loc) == ()  # the gap this test exists for
    assert node_shape(graph, graph.node("d"), env) == (200,)
    with pytest.raises(StructureError, match="log_prob"):
        check_gaussian(graph, graph.node("d"), env)


def test_the_probe_is_not_diluted_by_the_entries_that_are_correct():
    """One wrong element among many must not hide behind the others.

    Measured directly against this module's own check_gaussian: a summed
    comparison dilutes one entry off by 50 nats below the default rtol only
    once there are enough OTHER entries to swamp it -- around 750,000, for
    this fixture's smallest-magnitude probe (offset 0.0, scale 0.7). Below
    that the summed comparison still raises regardless, so it cannot be told
    apart from the elementwise one: at 5,000 entries -- this test's first
    draft -- the summed relative error ranges from 1.975e-3 (offset -3.0) to
    1.78e-2 (offset 0.0), both far above rtol, so THAT mutation went
    undetected here even though it is a real regression. 2,000,000 entries
    give comfortable margin (worst-offset relative error 4.45e-5 against
    rtol 1.19e-4) and still run in a couple hundred milliseconds. Elementwise
    reports 50 regardless of n.
    """

    class OneBadEntryNormal(dist.Normal):
        def log_prob(self, value):
            true = super().log_prob(value)
            return true.at[0].add(50.0) if jnp.ndim(value) > 0 else true

    def model():
        xs = const("X", jnp.ones(2_000_000))
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda u: OneBadEntryNormal(u, 0.7), mu, obs=jnp.zeros(2_000_000))

    graph = trace(model)
    env = evaluate(graph, {"w": jnp.asarray(0.3)})
    with pytest.raises(StructureError, match="log_prob"):
        check_gaussian(graph, graph.node("d"), env)


def test_check_gaussian_refuses_a_scale_that_is_not_strictly_positive():
    def model():
        w = sample("w", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda w_: dist.Normal(w_, 0.0), w, obs=jnp.zeros(()))

    graph = trace(model)
    # Matching the strict-positivity guard's OWN words, not just "scale": with
    # a looser match, mutating `scale > 0` to `scale >= 0` still passes,
    # because sigma=0 then reaches the probe loop, log(0) makes `predicted`
    # non-finite, and that refusal's message also contains the word "scale".
    with pytest.raises(StructureError, match="strictly positive"):
        check_gaussian(graph, graph.node("d"), evaluate(graph, {"w": jnp.asarray(0.0)}))


def test_node_shape_agrees_with_the_numpyro_bridge():
    """An independent reading of the same question.

    The bridge builds the site through numpyro.plate; node_shape derives the
    shape from the distribution and the declared plate size. They must agree,
    or the block's domain is a different space from the one NUTS samples.
    """
    graph = plated_latent(n=6)
    traced = handlers.trace(
        handlers.seed(to_numpyro(graph), jax.random.key(0))
    ).get_trace()
    env = evaluate(graph, {"z": traced["z"]["value"]})
    assert node_shape(graph, graph.node("z"), env) == traced["z"]["value"].shape == (6,)
    assert node_shape(graph, graph.node("d"), env) == traced["d"]["value"].shape == (6,)


def test_observation_parts_covers_every_observed_node():
    """Keys, shapes, ``scale`` -- and, load-bearing on its own, ``data``.

    ``data`` carries a burden no other test in this package's suite shares:
    every exact solve is conditioned on these values, but R1 (the matrix-free
    CG path) and R2 (``tests/exact/oracle.py``'s dense oracle) both read
    ``data`` through this SAME function, ``observation_parts`` -- so a bug
    here shifts both sides of the acceptance gate together and the gate
    cannot see it, structurally, regardless of what else this suite checks.
    Measured directly: mutating ``observation_parts`` to add 5.0 to every
    returned ``data`` value leaves the whole acceptance gate green
    (`tests/exact/test_solve.py::test_wiener_solve_matches_the_dense_oracle`
    and its siblings) either way.

    The last two assertions below did NOT exist until this finding: before
    them, that same mutation left the *entire* 174-test suite green except
    one accidental, unrelated flip in `tests/exact/test_solve.py`
    (`test_the_precision_floor_alone_makes_the_guard_unreachable`, which
    reds through a float32 residual perturbation, not because it checks
    anything about `data`) -- so nothing in this package verified that
    `observation_parts` returns the right `data` values, anywhere. With them
    present, the SAME mutation now reds this test specifically, and nothing
    else in the suite.
    """
    graph = two_observations(n=7, m=5)
    env = evaluate(graph, {"w": jnp.asarray(1.25)})
    data, loc, scale = observation_parts(graph, env)
    assert set(data) == set(loc) == set(scale) == {"d1", "d2"}
    assert data["d1"].shape == loc["d1"].shape == scale["d1"].shape == (7,)
    assert data["d2"].shape == loc["d2"].shape == scale["d2"].shape == (5,)
    assert jnp.allclose(scale["d1"], 0.3) and jnp.allclose(scale["d2"], 0.9)
    assert jnp.allclose(data["d1"], graph.node("d1").observed)
    assert jnp.allclose(data["d2"], graph.node("d2").observed)


def test_noise_std_at_moves_with_the_latent_only_for_a_prediction_dependent_node():
    """The seam that decides Wiener vs GLS, exercised on both sides."""
    constant = straight_line()
    a = noise_std_at(constant, {"w": jnp.asarray(1.0)})["d"]
    b = noise_std_at(constant, {"w": jnp.asarray(9.0)})["d"]
    assert jnp.allclose(a, b)

    tracking = radiometer()
    c = noise_std_at(tracking, {"w": jnp.asarray(1.0)})["d"]
    d = noise_std_at(tracking, {"w": jnp.asarray(9.0)})["d"]
    assert not jnp.allclose(c, d)
    assert jnp.all(d > c)
