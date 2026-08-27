"""Amortized posterior estimation (G5).

The oracle here is **closed form and written in numpy**: a one-dimensional
linear-Gaussian problem whose posterior is a Gaussian with a precision and a
mean that can be written down. Nothing in this file recomputes the estimator's
own arithmetic in order to compare against it.

Three independent instruments, deliberately not one:

* **quadrature** -- ``exp(log_prob)`` integrated on a grid says whether the
  density is a density at all, and it is sensitive to the change-of-variables
  Jacobian that the standardization introduces;
* **the sampler** -- ``sample`` and ``log_prob`` are two different pieces of
  code reading the same mixture, so their first two moments agreeing is a
  statement about both;
* **the exact posterior** -- what the fitted density is supposed to be.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.amortize import MIN_SCALE, NeuralPosterior, train_posterior
from bayesmith.errors import StructureError

# --- the problem, in numpy -------------------------------------------------
#
# theta ~ Normal(M0, S0); x = A theta + Normal(0, SIGMA) on eight points.
# Both the prior centre and its width are far from the standardizing
# transform's fixed point (mean 0, scale 1), which is what makes the Jacobian
# term in `log_prob` load-bearing rather than a no-op -- see
# `test_the_bank_really_does_rescale_theta`.
M0 = 0.5
S0 = 2.0
SIGMA = 0.4
A = np.linspace(0.5, 1.5, 8)


def exact_posterior(x) -> tuple[float, float]:
    """Closed form, in numpy. The oracle."""
    precision = float(A @ A) / SIGMA**2 + 1.0 / S0**2
    mean = (float(A @ np.asarray(x)) / SIGMA**2 + M0 / S0**2) / precision
    return mean, precision**-0.5


def draw_bank(key, n_simulations: int):
    """``(theta, x)`` pairs from the joint. No graph, no simulator object."""
    theta_key, noise_key = jax.random.split(key)
    theta = M0 + S0 * jax.random.normal(theta_key, (n_simulations, 1))
    data = theta * jnp.asarray(A) + SIGMA * jax.random.normal(
        noise_key, (n_simulations, len(A))
    )
    return theta, data


def observation(theta_true: float, seed: int = 4):
    return jnp.asarray(
        np.asarray(A) * theta_true
        + SIGMA * np.random.default_rng(seed).normal(size=len(A))
    )


@pytest.fixture(scope="module")
def untrained():
    theta, data = draw_bank(jax.random.key(0), 512)
    q = NeuralPosterior.create(theta, data, key=jax.random.key(1), n_components=3)
    return q, data


@pytest.fixture(scope="module")
def trained():
    """One training run, reused. That reuse is the point of the method."""
    theta, data = draw_bank(jax.random.key(0), 8192)
    q = NeuralPosterior.create(theta, data, key=jax.random.key(1), n_components=1)
    return train_posterior(
        q, theta, data, key=jax.random.key(2), n_steps=2000, batch_size=256
    )


def eqx_arrays(tree):
    return [leaf for leaf in jax.tree.leaves(tree) if hasattr(leaf, "dtype")]


def quadrature(q: NeuralPosterior, datum, moments: bool = False):
    """Integrate the density on a grid wide enough to hold all of it."""
    grid = jnp.linspace(-40.0, 40.0, 40001)
    density = jnp.exp(
        jax.vmap(lambda t: q.log_prob(jnp.atleast_1d(t), datum))(grid)
    )
    mass = float(jnp.trapezoid(density, grid))
    if not moments:
        return mass
    first = float(jnp.trapezoid(density * grid, grid))
    second = float(jnp.trapezoid(density * grid**2, grid))
    return mass, first, (second - first**2) ** 0.5


class TestTheDensityIsADensity:
    """Quadrature, on an UNTRAINED estimator -- being normalized is a property
    of the parameterization, not of the fit."""

    def test_it_integrates_to_one(self, untrained):
        q, data = untrained
        assert quadrature(q, data[0]) == pytest.approx(1.0, abs=1e-6)

    def test_a_different_observation_is_also_normalized(self, untrained):
        q, data = untrained
        assert quadrature(q, data[17]) == pytest.approx(1.0, abs=1e-6)
        # ... and it is a DIFFERENT density, or the check above is vacuous.
        here = q.log_prob(jnp.asarray([0.3]), data[0])
        there = q.log_prob(jnp.asarray([0.3]), data[17])
        assert abs(float(here) - float(there)) > 1e-3

    def test_the_bank_really_does_rescale_theta(self, untrained):
        """The sibling assertion for the two above.

        ``log_prob`` subtracts ``sum(log(theta_scale))`` because standardizing
        is a change of variables. On a bank whose theta happened to have unit
        width and zero mean that term is zero, and every normalization check
        above would pass with it deleted. It is not zero here, and this test
        is what says so.
        """
        q, _ = untrained
        assert float(q.theta_scale[0]) == pytest.approx(S0, rel=0.1)
        assert abs(float(jnp.sum(jnp.log(q.theta_scale)))) > 0.5

    def test_a_negative_min_scale_would_have_reached_the_density(self, untrained):
        """Why `create` refuses one, stated as a measurement rather than a
        worry: the parameters stay finite and the density does not."""
        q, data = untrained
        broken = NeuralPosterior(
            net=q.net,
            embed=q.embed,
            n_components=q.n_components,
            n_params=q.n_params,
            theta_mean=q.theta_mean,
            theta_scale=q.theta_scale,
            data_mean=q.data_mean,
            data_scale=q.data_scale,
            min_scale=-1.0,
        )
        assert all(
            bool(jnp.all(jnp.isfinite(leaf)))
            for leaf in jax.tree.leaves(eqx_arrays(broken))
        )
        assert not bool(jnp.isfinite(broken.log_prob(jnp.asarray([0.3]), data[0])))


class TestTheSamplerAndTheDensityAgree:
    """Two different pieces of code read the same mixture. Quadrature knows
    nothing about categorical draws; the sampler knows nothing about grids."""

    def test_the_first_two_moments_match(self, untrained):
        q, data = untrained
        _, mean, sd = quadrature(q, data[0], moments=True)
        draws = q.sample(data[0], jax.random.key(9), 200_000)
        assert float(jnp.mean(draws)) == pytest.approx(mean, abs=0.05 * sd)
        assert float(jnp.std(draws)) == pytest.approx(sd, rel=0.05)
        # The moments are not trivially zero and one, which is what would make
        # the two comparisons above agree for the wrong reason.
        assert abs(mean) > 0.1 and sd > 1.0


class TestAgainstTheExactPosterior:
    """One mixture component is EXACT for a Gaussian posterior, which is what
    makes this comparison sharp rather than indicative."""

    OBSERVED = (0.5, 1.6, -0.9)

    def test_training_reduces_the_loss(self, trained):
        _, history = trained
        assert float(history.train[-1]) < float(history.train[0]) - 0.5

    def test_the_posterior_mean_matches(self, trained):
        q, _ = trained
        for theta_true in self.OBSERVED:
            x = observation(theta_true)
            mean, sd = exact_posterior(x)
            draws = q.sample(x, jax.random.key(5), 20_000)
            # Measured worst case over these three: 0.36 posterior sd.
            assert abs(float(jnp.mean(draws)) - mean) < 0.6 * sd

    def test_the_posterior_width_matches(self, trained):
        q, _ = trained
        for theta_true in self.OBSERVED:
            x = observation(theta_true)
            _, sd = exact_posterior(x)
            # Measured range over these three: 0.938 to 0.985.
            assert 0.75 < float(jnp.std(q.sample(x, jax.random.key(5), 20_000))) / sd < 1.35

    def test_the_answer_is_not_just_the_prior(self, trained):
        """The large-effect half of the comparison, and the one no machine's
        rounding can reach: the exact posterior is about fifteen times tighter
        than the prior it came from, so an estimator that had learned nothing
        would miss by a factor, not by a tolerance."""
        q, _ = trained
        _, sd = exact_posterior(observation(0.5))
        assert S0 / sd > 12.0
        widths = [
            float(jnp.std(q.sample(observation(t), jax.random.key(5), 20_000)))
            for t in self.OBSERVED
        ]
        assert max(widths) < S0 / 8.0

    def test_it_is_amortized(self, trained):
        """The property the method exists for: one fit, then every observation
        is a forward pass. The three answers must differ from each other, or
        `q` is returning the prior with extra steps."""
        q, _ = trained
        means = [
            float(jnp.mean(q.sample(observation(t), jax.random.key(5), 20_000)))
            for t in self.OBSERVED
        ]
        for got, theta_true in zip(means, self.OBSERVED, strict=True):
            expected, sd = exact_posterior(observation(theta_true))
            assert abs(got - expected) < 0.6 * sd
        assert max(means) - min(means) > 2.0


class TestOverFittingIsVisibleOnlyInTheHeldOutSplit:
    """The failure that does not look like one, measured in this package.

    Bank of 512, four components, 8000 steps. The upstream implementation's
    docstring blames the STEP COUNT for this; re-measured, it is the bank --
    8192 pairs at 4000 steps comes back at 1.002 of the exact width.
    """

    N_SIMULATIONS = 512
    N_STEPS = 8000

    @pytest.fixture(scope="class")
    @classmethod
    def two_runs(cls):
        theta, data = draw_bank(jax.random.key(0), cls.N_SIMULATIONS)
        start = NeuralPosterior.create(
            theta, data, key=jax.random.key(1), n_components=4
        )
        common = {
            "key": jax.random.key(2),
            "n_steps": cls.N_STEPS,
            "batch_size": 256,
            "learning_rate": 1e-3,
        }
        blind = train_posterior(start, theta, data, validation_fraction=0.0, **common)
        watched = train_posterior(start, theta, data, validation_fraction=0.2, **common)
        return blind, watched

    def _width_ratio(self, q) -> float:
        x = observation(0.5)
        _, sd = exact_posterior(x)
        return float(jnp.std(q.sample(x, jax.random.key(5), 20_000))) / sd

    def test_without_a_split_the_fit_becomes_over_confident(self, two_runs):
        (q, history), _ = two_runs
        # Measured: 0.271. A factor, not a tolerance.
        assert self._width_ratio(q) < 0.5
        # ... while the training loss is still improving, which is the whole
        # point: the only instrument that was watching says everything is fine.
        assert float(history.train[-1]) < float(history.train[0])
        assert history.validation.shape == (0,)

    def test_the_split_catches_it(self, two_runs):
        _, (q, history) = two_runs
        # Measured: 0.844.
        assert 0.6 < self._width_ratio(q) < 1.4
        # Measured: step 423 of 8000.
        assert 1 < int(history.best_step) < self.N_STEPS // 4

    def test_the_validation_curve_turns_around(self, two_runs):
        _, (_, history) = two_runs
        # Measured: minimum -0.605, last +34.08.
        assert float(history.validation[-1]) > float(jnp.min(history.validation)) + 10.0

    def test_the_returned_estimator_is_the_argmin_of_that_curve(self, two_runs):
        """Not the last step, and not an arbitrary one."""
        _, (_, history) = two_runs
        assert int(history.best_step) == int(jnp.argmin(history.validation)) + 1


class TestTheKnobsReachTheAnswer:
    def test_min_scale_widens_the_density(self, untrained):
        """A floor that no arithmetic reads is a floor that can be deleted."""
        q, data = untrained
        wide = NeuralPosterior(
            net=q.net,
            embed=q.embed,
            n_components=q.n_components,
            n_params=q.n_params,
            theta_mean=q.theta_mean,
            theta_scale=q.theta_scale,
            data_mean=q.data_mean,
            data_scale=q.data_scale,
            min_scale=2.0,
        )
        _, _, narrow_sd = quadrature(q, data[0], moments=True)
        _, _, wide_sd = quadrature(wide, data[0], moments=True)
        assert wide_sd > 1.5 * narrow_sd

    def test_a_zero_floor_is_allowed_and_finite(self, untrained):
        """The sibling of the refusal below: zero is NOT refused, because
        softplus never reaches it and there is no failure there to refuse."""
        _, data = untrained
        theta, bank = draw_bank(jax.random.key(0), 512)
        q = NeuralPosterior.create(
            theta, bank, key=jax.random.key(1), n_components=3, min_scale=0.0
        )
        assert MIN_SCALE > 0.0
        assert bool(jnp.isfinite(q.log_prob(jnp.asarray([0.3]), data[0])))
        assert quadrature(q, data[0]) == pytest.approx(1.0, abs=1e-6)


class TestRefusals:
    def test_a_one_dimensional_theta_stack_is_refused(self):
        theta, data = draw_bank(jax.random.key(0), 64)
        with pytest.raises(StructureError, match="n_params"):
            NeuralPosterior.create(
                jnp.ravel(theta), data, key=jax.random.key(1)
            )

    def test_mismatched_bank_halves_are_refused(self):
        theta, data = draw_bank(jax.random.key(0), 64)
        with pytest.raises(StructureError, match="same pairs"):
            NeuralPosterior.create(theta[:32], data, key=jax.random.key(1))

    def test_zero_components_are_refused(self):
        theta, data = draw_bank(jax.random.key(0), 64)
        with pytest.raises(StructureError, match="n_components"):
            NeuralPosterior.create(
                theta, data, key=jax.random.key(1), n_components=0
            )

    def test_a_negative_scale_floor_is_refused(self):
        theta, data = draw_bank(jax.random.key(0), 64)
        with pytest.raises(StructureError, match="min_scale"):
            NeuralPosterior.create(
                theta, data, key=jax.random.key(1), min_scale=-1e-6
            )

    def test_zero_steps_are_refused(self, untrained):
        q, _ = untrained
        theta, bank = draw_bank(jax.random.key(0), 512)
        with pytest.raises(StructureError, match="n_steps"):
            train_posterior(q, theta, bank, key=jax.random.key(2), n_steps=0)

    def test_a_validation_fraction_of_one_is_refused(self, untrained):
        q, _ = untrained
        theta, bank = draw_bank(jax.random.key(0), 512)
        with pytest.raises(StructureError, match=r"\[0, 1\)"):
            train_posterior(
                q, theta, bank, key=jax.random.key(2), n_steps=1,
                validation_fraction=1.0,
            )

    def test_a_fraction_that_holds_out_nothing_is_refused(self, untrained):
        """Silently training on everything while the caller believes a split
        exists is the shape this refusal is for."""
        q, _ = untrained
        theta, bank = draw_bank(jax.random.key(0), 512)
        with pytest.raises(StructureError, match="holds out zero"):
            train_posterior(
                q, theta, bank, key=jax.random.key(2), n_steps=1,
                validation_fraction=1e-4,
            )


class TestADivergedRun:
    """D43: two outcomes, one refused and one reported.

    Measured at a learning rate a thousandfold above the working one.
    """

    @staticmethod
    def _start():
        theta, data = draw_bank(jax.random.key(0), 512)
        q = NeuralPosterior.create(
            theta, data, key=jax.random.key(1), n_components=2
        )
        return q, theta, data

    def test_without_a_split_a_nonfinite_estimator_is_refused(self):
        q, theta, data = self._start()
        with pytest.raises(Exception, match="non-finite estimator"):
            train_posterior(
                q, theta, data, key=jax.random.key(7), n_steps=60,
                batch_size=64, learning_rate=1e3, validation_fraction=0.0,
            )

    def test_the_same_rate_with_a_split_comes_back_finite_and_says_so(self):
        """The other outcome. Nothing here is wrong with the estimator -- the
        record of the divergence is in the history, and nowhere else."""
        q, theta, data = self._start()
        fitted, history = train_posterior(
            q, theta, data, key=jax.random.key(7), n_steps=60,
            batch_size=64, learning_rate=1e3, validation_fraction=0.1,
        )
        assert bool(jnp.isfinite(fitted.log_prob(jnp.asarray([1.0]), data[0])))
        assert int(history.best_step) == 1
        assert not bool(jnp.all(jnp.isfinite(history.validation)))

    def test_the_working_rate_does_not_trip_the_guard(self):
        """The sibling that says the two tests above are about divergence and
        not about this fixture being unrunnable."""
        q, theta, data = self._start()
        fitted, history = train_posterior(
            q, theta, data, key=jax.random.key(7), n_steps=60,
            batch_size=64, learning_rate=1e-3, validation_fraction=0.0,
        )
        assert bool(jnp.isfinite(fitted.log_prob(jnp.asarray([1.0]), data[0])))
        assert bool(jnp.all(jnp.isfinite(history.train)))
