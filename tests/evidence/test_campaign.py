"""A campaign compressed straight from a graph, against dense references.

This is the bridge: every input `compress_epoch` needs -- the partition, the
designs, the constant part, the data slice, the covariance slice and the
nuisance priors -- read off the graph rather than assembled by a caller.

The references are dense Gaussians written from the model by hand, so the
comparison is against the model rather than against another of our routines.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest

from bayesmith import const, det, observe, plate, sample, trace
from bayesmith.errors import StructureError
from bayesmith.evidence import SqrtInfo, compress_campaign, epoch_terms, marginalise

N_EPOCH, TAU, SIGMA, GAIN, PRIOR_MEAN = 4, 1.3, 0.55, 2.0, 0.4


def _scalar_campaign(seed=0, n_epoch=N_EPOCH):
    """One survivor, one per-epoch nuisance, one datum per epoch."""
    data = np.random.default_rng(seed).normal(size=n_epoch)

    def model():
        epoch = plate("epoch", n_epoch)
        g = sample("g", lambda: ndist.Normal(0.0, 3.0))
        n = sample("n", lambda: ndist.Normal(PRIOR_MEAN, TAU), plate=epoch)
        mu = det(
            "mu", lambda a, b: GAIN * a + b, g, n, plate=epoch, linear_in=("g", "n")
        )
        observe(
            "d",
            lambda m: ndist.Normal(m, SIGMA),
            mu,
            plate=epoch,
            obs=jnp.asarray(data),
        )

    return trace(model), data


class TestTheBridgeReproducesTheModel:
    @pytest.mark.parametrize("point", [0.0, 1.5, -2.0])
    def test_a_scalar_campaign_matches_the_dense_marginal(self, point):
        """``d ~ N(GAIN g + m_n, sigma^2 + tau^2)`` once the nuisance is gone."""
        with jax.enable_x64(True):
            graph, data = _scalar_campaign()
            term = compress_campaign(graph, "epoch")
            got = float(term.log_prob({"g": jnp.asarray(point)}))
        covariance = (SIGMA**2 + TAU**2) * np.eye(N_EPOCH)
        residual = data - (GAIN * point + PRIOR_MEAN)
        _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
        expected = -0.5 * float(
            residual @ np.linalg.solve(covariance, residual)
        ) - 0.5 * float(logdet)
        assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)

    def test_the_term_carries_the_survivors_own_shapes(self):
        """So a caller's existing value dict ravels without reshaping."""
        with jax.enable_x64(True):
            graph, _ = _scalar_campaign()
            term = compress_campaign(graph, "epoch")
        assert term.names == ("g",)
        assert term.shapes == ((),)

    def test_a_vector_survivor_and_a_covariate_work_too(self):
        """The design is a jacobian, so a vector survivor needs no special case.

        The covariate is a PLATED const. An unplated one inside a plated
        ``det`` broadcasts -- measured while writing this: each epoch's
        prediction came back with the whole covariate's shape rather than its
        own row, which is a modelling error the plate makes loud.
        """
        n_epoch = 5
        rng = np.random.default_rng(3)
        covariate = rng.normal(size=(n_epoch, 2))
        data = rng.normal(size=n_epoch)

        def model():
            epoch = plate("epoch", n_epoch)
            xs = const("X", jnp.asarray(covariate), plate=epoch)
            w = sample("w", lambda: ndist.Normal(jnp.zeros(2), 4.0))
            n = sample("n", lambda: ndist.Normal(0.0, TAU), plate=epoch)
            mu = det(
                "mu",
                lambda weights, cov, nuis: jnp.sum(weights * cov) + nuis,
                w,
                xs,
                n,
                plate=epoch,
                linear_in=("w", "n"),
            )
            observe(
                "d",
                lambda m: ndist.Normal(m, SIGMA),
                mu,
                plate=epoch,
                obs=jnp.asarray(data),
            )

        point = np.asarray([0.7, -1.1])
        with jax.enable_x64(True):
            term = compress_campaign(trace(model), "epoch")
            got = float(term.log_prob({"w": jnp.asarray(point)}))
        covariance = (SIGMA**2 + TAU**2) * np.eye(n_epoch)
        residual = data - covariate @ point
        _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
        expected = -0.5 * float(
            residual @ np.linalg.solve(covariance, residual)
        ) - 0.5 * float(logdet)
        assert term.names == ("w",)
        assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)

    def test_the_campaign_evidence_matches_a_dense_one(self):
        """Marginalise the survivor too: every constant at once, from a graph."""
        prior_std = 3.0
        with jax.enable_x64(True):
            graph, data = _scalar_campaign(seed=5)
            total = SqrtInfo.combine(
                compress_campaign(graph, "epoch"),
                SqrtInfo(
                    factor=jnp.asarray([[1.0 / prior_std]]),
                    target=jnp.zeros(1),
                    offset=jnp.asarray(
                        -math.log(prior_std) - 0.5 * math.log(2.0 * math.pi)
                    ),
                    names=("g",),
                    shapes=((),),
                ),
            )
            evidence = float(marginalise(total, ["g"]).log_prob({}))
        design = np.full((N_EPOCH, 1), GAIN)
        covariance = (
            (SIGMA**2 + TAU**2) * np.eye(N_EPOCH)
            + prior_std**2 * design @ design.T
        )
        residual = data - PRIOR_MEAN
        _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
        expected = -0.5 * float(
            residual @ np.linalg.solve(covariance, residual)
        ) - 0.5 * float(logdet)
        assert evidence == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_the_fold_holds_one_epoch_at_a_time(self):
        """`epoch_terms` yields E separate terms, each over the survivors only.

        That is the property the whole layer exists for: the accumulated term
        does not grow with the campaign.
        """
        with jax.enable_x64(True):
            graph, _ = _scalar_campaign(n_epoch=7)
            terms = epoch_terms(graph, "epoch")
            folded = compress_campaign(graph, "epoch")
        assert len(terms) == 7
        assert all(term.names == ("g",) for term in terms)
        assert all(term.factor.shape == folded.factor.shape for term in terms)


class TestWhatTheBridgeRefuses:
    def test_two_epoch_plated_observations_are_refused(self):
        """"One epoch's data" has to be one node's slice."""
        data = np.random.default_rng(1).normal(size=N_EPOCH)

        def model():
            epoch = plate("epoch", N_EPOCH)
            g = sample("g", lambda: ndist.Normal(0.0, 3.0))
            n = sample("n", lambda: ndist.Normal(0.0, TAU), plate=epoch)
            mu = det("mu", lambda a, b: a + b, g, n, plate=epoch, linear_in=("g", "n"))
            observe(
                "d1",
                lambda m: ndist.Normal(m, SIGMA),
                mu,
                plate=epoch,
                obs=jnp.asarray(data),
            )
            observe(
                "d2",
                lambda m: ndist.Normal(m, SIGMA),
                mu,
                plate=epoch,
                obs=jnp.asarray(data),
            )

        with jax.enable_x64(True), pytest.raises(StructureError, match="2 observed"):
            compress_campaign(trace(model), "epoch")

    def test_an_unplated_observation_is_refused(self):
        def model():
            epoch = plate("epoch", N_EPOCH)
            g = sample("g", lambda: ndist.Normal(0.0, 3.0))
            n = sample("n", lambda: ndist.Normal(0.0, TAU), plate=epoch)
            total = det("total", lambda a, b: a + jnp.sum(b), g, n, linear_in=("g", "n"))
            observe("d", lambda m: ndist.Normal(m, SIGMA), total, obs=jnp.asarray(0.0))

        with jax.enable_x64(True), pytest.raises(StructureError):
            compress_campaign(trace(model), "epoch")

    def test_a_covariance_that_does_not_batch_over_epochs_is_refused(self):
        """A campaign's noise has to be epoch-separable.

        Sliced directly rather than through a graph, because the plate makes a
        batched covariance by construction -- so this is the guard for a
        caller who builds one by hand.
        """
        from bayesmith.evidence.campaign import _slice_precision
        from bayesmith.exact.precision import CirculantPrecision, DiagonalPrecision

        with jax.enable_x64(True):
            lag = np.minimum(np.arange(6), 6 - np.arange(6))
            spanning = CirculantPrecision(
                first_column=jnp.asarray(0.4**lag + 0.5)  # 1-D: one kernel for all
            )
            with pytest.raises(StructureError, match="COUPLES the epochs"):
                _slice_precision(spanning, 0, N_EPOCH)
            wrong_width = DiagonalPrecision(sigma=jnp.ones(N_EPOCH + 1))
            with pytest.raises(StructureError, match="does not batch"):
                _slice_precision(wrong_width, 0, N_EPOCH)

    def test_a_leaky_per_epoch_latent_is_refused_by_the_bridge_too(self):
        """`epoch_terms` derives the factorization, so it inherits the check."""
        lower = np.tril(np.ones((N_EPOCH, N_EPOCH)) * 0.8, -1) + np.eye(N_EPOCH)
        data = np.random.default_rng(2).normal(size=(N_EPOCH, N_EPOCH))

        def model():
            epoch = plate("epoch", N_EPOCH)
            g = sample("g", lambda: ndist.Normal(0.0, 3.0))
            eps = sample("eps", lambda: ndist.Normal(0.0, TAU), plate=epoch)
            chain = det(
                "chain", lambda e: jnp.asarray(lower) @ e, eps, linear_in=("eps",)
            )
            mu = det(
                "mu",
                lambda a, e, c: a + e + c,
                g,
                eps,
                chain,
                plate=epoch,
                linear_in=("g", "eps", "chain"),
            )
            observe(
                "d",
                lambda m: ndist.Normal(m, SIGMA),
                mu,
                plate=epoch,
                obs=jnp.asarray(data),
            )

        with jax.enable_x64(True), pytest.raises(StructureError, match="reaches other"):
            compress_campaign(trace(model), "epoch")


def test_a_campaign_with_a_per_epoch_offset_and_a_per_epoch_sigma():
    """Two things every fixture above happens not to have, and both survived.

    Mutation found the gap, not review: dropping ``offset_prediction`` and
    slicing every epoch's covariance at index 0 both passed the whole file,
    because no model here had a constant part in its prediction or a sigma
    that varied from epoch to epoch. A real campaign has both -- a known
    foreground and a night that got worse -- so this fixture is closer to the
    thing than the ones that were easier to write.
    """
    n_epoch = 5
    rng = np.random.default_rng(11)
    offsets = rng.normal(size=n_epoch) * 2.0
    sigmas = np.linspace(0.3, 1.4, n_epoch)
    data = rng.normal(size=n_epoch)

    def model():
        epoch = plate("epoch", n_epoch)
        known = const("c", jnp.asarray(offsets), plate=epoch)
        scale = const("s", jnp.asarray(sigmas), plate=epoch)
        g = sample("g", lambda: ndist.Normal(0.0, 3.0))
        n = sample("n", lambda: ndist.Normal(PRIOR_MEAN, TAU), plate=epoch)
        mu = det(
            "mu",
            lambda a, b, off: GAIN * a + b + off,
            g,
            n,
            known,
            plate=epoch,
            linear_in=("g", "n"),
        )
        observe(
            "d",
            lambda m, s: ndist.Normal(m, s),
            mu,
            scale,
            plate=epoch,
            depends_on_prediction=False,
            obs=jnp.asarray(data),
        )

    for point in (0.0, 1.25, -0.8):
        with jax.enable_x64(True):
            term = compress_campaign(trace(model), "epoch")
            got = float(term.log_prob({"g": jnp.asarray(point)}))
        covariance = np.diag(sigmas**2 + TAU**2)
        residual = data - (GAIN * point + PRIOR_MEAN + offsets)
        _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
        expected = -0.5 * float(
            residual @ np.linalg.solve(covariance, residual)
        ) - 0.5 * float(logdet)
        assert got == pytest.approx(expected, rel=1e-10, abs=1e-10), point
