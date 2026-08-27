"""G3 -- the chain recursion, against a DENSE joint density built in numpy.

The module's own warning is the design of this file: six constants reach the
answer, and the recursion's shape, gradient and curvature are all correct
without any of them. Every test that checks a posterior mean, a width or a
derivative passes on a version that has dropped one. **Only a comparison
against a dense joint notices**, so that is what the oracle is -- and the
constants are then deleted one at a time to measure what each was worth, which
is the number a future reader needs when one of them goes missing again.

The dense reference assembles the whole ``(theta, zeta_1..zeta_N)`` Gaussian
in numpy and integrates ``zeta`` out with a log-determinant and a Schur
complement, forming matrices the recursion never forms. It shares the fixture
and nothing else.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.evidence.chain import (
    HyperTransition,
    LinearGaussianTransition,
    chain_log_likelihood,
    chain_marginal,
    ornstein_uhlenbeck,
    smooth,
)

NAMES = ("gain", "offset")
SHAPES = ((), ())
EPOCHS, WIDTH = 6, 1


@pytest.fixture(autouse=True)
def _double_precision():
    """Every test here runs in float64.

    A square-root information recursion over six epochs is exactly where the
    explicit ``(F, b)`` form goes indefinite, and the constants being measured
    are fractions of a nat against a density of order 100. Overriding this
    fixture by NAME is how a test opts out, and none does.
    """
    with jax.enable_x64(True):
        yield


def _blocks(seed=0, epochs=EPOCHS, n_theta=2, n_zeta=WIDTH):
    """Per-epoch square joint forms over ``(theta..., zeta)``, zeta LAST.

    Random rather than structured on purpose: a block with a special shape --
    diagonal, or with a zero theta half -- lets a slice at ``n_theta`` take one
    half for the other and still agree.
    """
    generator = np.random.default_rng(seed)
    width = n_theta + n_zeta
    factors = generator.normal(size=(epochs, width, width))
    targets = generator.normal(size=(epochs, width))
    offsets = generator.normal(size=(epochs,)) * 0.1
    return (jnp.asarray(factors), jnp.asarray(targets), jnp.asarray(offsets))


def _transition(width=WIDTH):
    return LinearGaussianTransition(
        phi=0.8 * np.eye(width),
        process_std=np.full(width, 0.5),
        initial_std=np.full(width, 1.3),
        initial_mean=np.full(width, 0.2),
    )


def dense_log_likelihood(blocks, transition, theta, *, n_theta=2):
    """``log p(d | theta)`` by assembling the whole joint and integrating.

    Nothing here is square-root and nothing is recursive: the ``(N*w, N*w)``
    precision is formed explicitly, the linear term with it, and the Gaussian
    integral over ``zeta`` is done with a log-determinant. That is the form the
    module exists to avoid, which is exactly what makes it an independent
    check.
    """
    factors, targets, offsets = (np.asarray(part) for part in blocks)
    epochs = factors.shape[0]
    n_zeta = int(transition.width)
    size = epochs * n_zeta
    theta = np.asarray(theta, dtype=np.float64)

    precision = np.zeros((size, size))
    linear = np.zeros(size)
    constant = float(np.sum(offsets))

    for epoch in range(epochs):
        rows = factors[epoch]
        chain = rows[:, n_theta:]
        residual = targets[epoch] - rows[:, :n_theta] @ theta
        start = epoch * n_zeta
        precision[start : start + n_zeta, start : start + n_zeta] += chain.T @ chain
        linear[start : start + n_zeta] += chain.T @ residual
        constant += -0.5 * float(residual @ residual)

    initial_precision = np.diag(1.0 / np.asarray(transition.initial_std) ** 2)
    initial_mean = np.asarray(transition.initial_mean)
    precision[:n_zeta, :n_zeta] += initial_precision
    linear[:n_zeta] += initial_precision @ initial_mean
    constant += -0.5 * float(initial_mean @ initial_precision @ initial_mean)
    constant += -0.5 * n_zeta * np.log(2.0 * np.pi)
    constant += -float(np.sum(np.log(np.asarray(transition.initial_std))))

    inverse_q = np.diag(1.0 / np.asarray(transition.process_std) ** 2)
    phi = np.asarray(transition.phi)
    for epoch in range(epochs - 1):
        here, nxt = epoch * n_zeta, (epoch + 1) * n_zeta
        precision[here : here + n_zeta, here : here + n_zeta] += phi.T @ inverse_q @ phi
        precision[nxt : nxt + n_zeta, nxt : nxt + n_zeta] += inverse_q
        precision[here : here + n_zeta, nxt : nxt + n_zeta] += -phi.T @ inverse_q
        precision[nxt : nxt + n_zeta, here : here + n_zeta] += -inverse_q @ phi
        constant += -0.5 * n_zeta * np.log(2.0 * np.pi)
        constant += -float(np.sum(np.log(np.asarray(transition.process_std))))

    # int exp(-0.5 z^T P z + l^T z) dz = (2 pi)^(k/2) |P|^-1/2 exp(0.5 l^T P^-1 l)
    sign, logdet = np.linalg.slogdet(precision)
    assert sign > 0, "the dense joint precision must be positive definite"
    solved = np.linalg.solve(precision, linear)
    return (
        constant
        + 0.5 * size * np.log(2.0 * np.pi)
        - 0.5 * logdet
        + 0.5 * float(linear @ solved)
    )


def _theta(a=0.4, b=-1.1):
    return {"gain": jnp.asarray(a), "offset": jnp.asarray(b)}


class TestAgainstTheDenseJoint:
    """The whole point of the file. Exact, not approximate."""

    @pytest.mark.parametrize(
        ("a", "b"), [(0.4, -1.1), (0.0, 0.0), (-2.3, 1.7), (1e2, -1e2)]
    )
    def test_the_recursion_equals_the_dense_integral(self, a, b):
        blocks, transition = _blocks(), _transition()
        found = float(
            chain_log_likelihood(blocks, transition, _theta(a, b), NAMES, SHAPES)
        )
        expected = dense_log_likelihood(blocks, transition, [a, b])
        assert found == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_it_holds_for_a_WIDE_chain_too(self):
        """A width-1 chain cannot tell ``diag(1/q) @ phi`` from
        ``phi @ diag(1/q)``, and those are different models. Width 3, with a
        phi that genuinely rotates."""
        width = 3
        blocks = _blocks(seed=4, n_zeta=width)
        phi = np.array([[0.7, 0.2, 0.0], [-0.1, 0.6, 0.3], [0.0, 0.1, 0.5]])
        transition = LinearGaussianTransition(
            phi=phi,
            process_std=np.array([0.4, 0.7, 0.55]),
            initial_std=np.array([1.1, 0.9, 1.4]),
            initial_mean=np.array([0.2, -0.3, 0.1]),
        )
        found = float(chain_log_likelihood(blocks, transition, _theta(), NAMES, SHAPES))
        expected = dense_log_likelihood(blocks, transition, [0.4, -1.1])
        assert found == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_a_longer_campaign_still_matches(self):
        blocks, transition = _blocks(seed=7, epochs=20), _transition()
        found = float(chain_log_likelihood(blocks, transition, _theta(), NAMES, SHAPES))
        expected = dense_log_likelihood(blocks, transition, [0.4, -1.1])
        assert found == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_the_marginal_is_a_quadratic_form_in_theta_alone(self):
        """It returns a SqrtInfo over ``names``, and its ``log_prob`` is the
        likelihood -- so downstream can add a prior to it rather than to a
        number."""
        blocks, transition = _blocks(), _transition()
        info = chain_marginal(blocks, transition, _theta(), NAMES, SHAPES)
        assert info.names == NAMES and info.shapes == SHAPES
        assert info.factor.shape[1] == 2
        assert float(info.log_prob(_theta())) == pytest.approx(
            float(chain_log_likelihood(blocks, transition, _theta(), NAMES, SHAPES))
        )


class TestEachConstantIsWorthSomething:
    """The six constants, deleted one at a time and measured.

    This is the class the module docstring is about. Every one of these
    deletions leaves the posterior's mean, width and gradient untouched -- so
    the number recorded here is the only evidence that the term was ever
    there.
    """

    @staticmethod
    def _cost(monkeypatch, name, replacement):
        """``patched - truth``, with the truth measured BEFORE the patch.

        The first version of this helper measured the truth after, inside the
        same monkeypatch scope, so every deletion "cost" exactly 0.0 -- the
        baseline was taken under the very mutation it was the baseline for.
        The same shape as the guards this session has already caught twice,
        and it went green in the most reassuring possible way.
        """
        import bayesmith.evidence.chain as module

        blocks, transition = _blocks(), _transition()
        truth = float(
            module.chain_log_likelihood(blocks, transition, _theta(), NAMES, SHAPES)
        )
        monkeypatch.setattr(module, name, replacement)
        patched = float(
            module.chain_log_likelihood(blocks, transition, _theta(), NAMES, SHAPES)
        )
        assert patched != truth, "the patch did not take"
        return patched - truth

    def test_dropping_the_initial_prior_normalisation_costs_nats(self, monkeypatch):
        cost = self._cost(
            monkeypatch, "_initial_log_norm", lambda transition: jnp.asarray(0.0)
        )
        # `0.5 log(2 pi) + log(1.3)` for this fixture's scalar chain.
        expected = 0.5 * np.log(2 * np.pi) + np.log(1.3)
        assert cost == pytest.approx(expected, rel=1e-9)

    def test_dropping_the_transition_normalisation_costs_five_of_them(
        self, monkeypatch
    ):
        cost = self._cost(
            monkeypatch, "_transition_log_norm", lambda transition: jnp.asarray(0.0)
        )
        one = 0.5 * np.log(2 * np.pi) + np.log(0.5)
        assert cost == pytest.approx((EPOCHS - 1) * one, rel=1e-9)

    def test_keeping_only_the_log_determinant_is_STILL_wrong(self, monkeypatch):
        """The shorthand that reads plausible: ``0.5 logdet Q^-1`` without the
        ``2 pi``. It leaves ``+0.5 log(2 pi)`` per transition -- no effect on
        any mean, width or gradient, and a wrong evidence."""
        cost = self._cost(
            monkeypatch,
            "_transition_log_norm",
            lambda transition: -jnp.sum(jnp.log(transition.process_std)),
        )
        assert cost == pytest.approx(
            (EPOCHS - 1) * 0.5 * np.log(2 * np.pi), rel=1e-9
        )

    def test_dropping_the_fold_corner_is_the_largest_of_them(self, monkeypatch):
        """The residual no choice of the latents can remove -- a chi-square per
        epoch, and it grows with the campaign."""
        import bayesmith.evidence.chain as module

        real = module._fold

        def cornerless(factor, target, offset, block, width):
            new_factor, new_target, _ = real(factor, target, offset, block, width)
            block_offset = block[2]
            # Put back everything except the corner: offset + block_offset.
            return new_factor, new_target, offset + block_offset

        cost = self._cost(monkeypatch, "_fold", cornerless)
        assert cost > 1.0

    def test_every_deletion_leaves_the_MEAN_untouched(self, monkeypatch):
        """Why the tests above have to exist. The posterior over theta is
        unchanged by any of them, so nothing that looks at a result would
        notice."""
        import bayesmith.evidence.chain as module

        blocks, transition = _blocks(), _transition()
        before = chain_marginal(blocks, transition, _theta(), NAMES, SHAPES)
        monkeypatch.setattr(
            module, "_initial_log_norm", lambda transition: jnp.asarray(0.0)
        )
        monkeypatch.setattr(
            module, "_transition_log_norm", lambda transition: jnp.asarray(0.0)
        )
        after = module.chain_marginal(blocks, transition, _theta(), NAMES, SHAPES)
        assert np.allclose(np.asarray(before.factor), np.asarray(after.factor))
        assert np.allclose(np.asarray(before.target), np.asarray(after.target))
        assert not np.isclose(float(before.offset), float(after.offset))


class TestOrnsteinUhlenbeck:
    def test_it_is_stationary_by_arithmetic(self):
        """``var(zeta_{e+1}) = phi^2 var + Q`` returns ``sigma^2`` when it
        starts there -- so stationarity is not an assumption here."""
        tau, sigma = 3.0, 1.7
        transition = ornstein_uhlenbeck(tau, sigma)
        phi = float(transition.phi[0, 0])
        variance = float(transition.initial_std[0]) ** 2
        stepped = phi**2 * variance + float(transition.process_std[0]) ** 2
        assert stepped == pytest.approx(sigma**2, rel=1e-12)

    def test_phi_is_exp_minus_one_over_tau(self):
        assert float(ornstein_uhlenbeck(4.0, 1.0).phi[0, 0]) == pytest.approx(
            np.exp(-0.25), rel=1e-12
        )

    def test_a_wide_ou_is_that_chain_component_wise(self):
        transition = ornstein_uhlenbeck(2.0, 0.9, width=3)
        assert transition.width == 3
        assert np.allclose(
            np.asarray(transition.phi), np.exp(-0.5) * np.eye(3)
        )

    def test_it_matches_the_dense_joint_like_any_other_transition(self):
        blocks = _blocks(seed=9)
        transition = ornstein_uhlenbeck(3.0, 1.2)
        found = float(chain_log_likelihood(blocks, transition, _theta(), NAMES, SHAPES))
        expected = dense_log_likelihood(blocks, transition, [0.4, -1.1])
        assert found == pytest.approx(expected, rel=1e-9, abs=1e-9)


class TestTheHyperTransition:
    """The distinction that is a TYPE rather than a caveat: a transition built
    from theta is resolved inside the likelihood, so it is differentiated."""

    @staticmethod
    def _hyper():
        def build(values):
            # exp(), so positivity is structural -- the class cannot check a
            # traced spread and says so.
            return ornstein_uhlenbeck(
                tau=jnp.exp(values["log_tau"]), sigma=jnp.exp(values["log_sigma"])
            )

        return HyperTransition(build=build, width=1, hyper=("log_tau", "log_sigma"))

    def test_it_agrees_with_the_fixed_one_at_the_same_numbers(self):
        """One code path serves both, and this is what says so: resolving a
        hyper transition at particular values must give the fixed transition's
        answer exactly."""
        names = ("log_tau", "log_sigma")
        blocks = _blocks(seed=11)
        values = {"log_tau": jnp.asarray(np.log(3.0)), "log_sigma": jnp.asarray(0.2)}
        hyper = float(
            chain_log_likelihood(blocks, self._hyper(), values, names, SHAPES)
        )
        fixed_transition = ornstein_uhlenbeck(3.0, float(np.exp(0.2)))
        fixed = float(
            chain_log_likelihood(blocks, fixed_transition, values, names, SHAPES)
        )
        assert hyper == pytest.approx(fixed, rel=1e-12)

    def test_the_likelihood_is_differentiable_in_the_hyper_parameters(self):
        """The reason the type exists. A filter run once at compression time
        would pin ``Q(theta)`` and ``phi(theta)``, and the derivative would be
        exactly zero -- finite, plausible, and the wrong model."""
        names = ("log_tau", "log_sigma")
        blocks = _blocks(seed=11)

        def density(values):
            return chain_log_likelihood(blocks, self._hyper(), values, names, SHAPES)

        values = {"log_tau": jnp.asarray(np.log(3.0)), "log_sigma": jnp.asarray(0.2)}
        gradient = jax.grad(density)(values)
        assert np.isfinite(float(gradient["log_tau"]))
        assert abs(float(gradient["log_tau"])) > 1e-6
        assert abs(float(gradient["log_sigma"])) > 1e-6

    def test_the_gradient_matches_a_finite_difference(self):
        """Anti-vacuity: a non-zero derivative could still be the wrong one."""
        names = ("log_tau", "log_sigma")
        blocks = _blocks(seed=11)

        def at(log_tau):
            values = {
                "log_tau": jnp.asarray(log_tau),
                "log_sigma": jnp.asarray(0.2),
            }
            return float(
                chain_log_likelihood(blocks, self._hyper(), values, names, SHAPES)
            )

        step = 1e-6
        centre = np.log(3.0)
        numeric = (at(centre + step) - at(centre - step)) / (2 * step)
        analytic = float(
            jax.grad(
                lambda v: chain_log_likelihood(
                    blocks, self._hyper(), v, names, SHAPES
                )
            )(
                {"log_tau": jnp.asarray(centre), "log_sigma": jnp.asarray(0.2)}
            )["log_tau"]
        )
        assert analytic == pytest.approx(numeric, rel=1e-5)


class TestSmooth:
    """``p(zeta_e | d, theta)``, against the same dense assembly."""

    def test_the_mean_and_variance_match_the_dense_solve(self):
        blocks, transition = _blocks(), _transition()
        mean, variance = smooth(blocks, transition, _theta(), NAMES, SHAPES)

        # The dense reference, conditioned on theta, formed here.
        factors, targets, _ = (np.asarray(part) for part in blocks)
        theta = np.array([0.4, -1.1])
        n_zeta, n_theta = 1, 2
        size = EPOCHS * n_zeta
        precision = np.zeros((size, size))
        linear = np.zeros(size)
        for epoch in range(EPOCHS):
            rows = factors[epoch]
            chain = rows[:, n_theta:]
            residual = targets[epoch] - rows[:, :n_theta] @ theta
            start = epoch * n_zeta
            precision[start : start + n_zeta, start : start + n_zeta] += chain.T @ chain
            linear[start : start + n_zeta] += chain.T @ residual
        initial = np.diag(1.0 / np.asarray(transition.initial_std) ** 2)
        precision[:n_zeta, :n_zeta] += initial
        linear[:n_zeta] += initial @ np.asarray(transition.initial_mean)
        inverse_q = np.diag(1.0 / np.asarray(transition.process_std) ** 2)
        phi = np.asarray(transition.phi)
        for epoch in range(EPOCHS - 1):
            here, nxt = epoch * n_zeta, (epoch + 1) * n_zeta
            precision[here : here + n_zeta, here : here + n_zeta] += (
                phi.T @ inverse_q @ phi
            )
            precision[nxt : nxt + n_zeta, nxt : nxt + n_zeta] += inverse_q
            precision[here : here + n_zeta, nxt : nxt + n_zeta] += -phi.T @ inverse_q
            precision[nxt : nxt + n_zeta, here : here + n_zeta] += -inverse_q @ phi
        covariance = np.linalg.inv(precision)
        expected_mean = covariance @ linear

        assert np.allclose(np.asarray(mean).ravel(), expected_mean, rtol=1e-9)
        assert np.allclose(
            np.asarray(variance).ravel(), np.diagonal(covariance), rtol=1e-9
        )

    def test_a_tighter_chain_smooths_harder(self):
        """The property a smoother exists for, stated so it can fail: a
        smaller process spread must make the epoch-to-epoch scatter smaller."""
        blocks = _blocks(seed=13)
        loose = LinearGaussianTransition(
            phi=[[0.9]], process_std=[2.0], initial_std=[2.0]
        )
        tight = LinearGaussianTransition(
            phi=[[0.9]], process_std=[0.02], initial_std=[2.0]
        )
        loose_mean, _ = smooth(blocks, loose, _theta(), NAMES, SHAPES)
        tight_mean, _ = smooth(blocks, tight, _theta(), NAMES, SHAPES)
        assert float(np.std(np.diff(np.asarray(tight_mean).ravel()))) < float(
            np.std(np.diff(np.asarray(loose_mean).ravel()))
        )

    def test_the_shapes_are_per_epoch_and_per_component(self):
        blocks = _blocks(seed=4, n_zeta=3)
        transition = _transition(width=3)
        mean, variance = smooth(blocks, transition, _theta(), NAMES, SHAPES)
        assert mean.shape == (EPOCHS, 3) and variance.shape == (EPOCHS, 3)


class TestWhatItRefuses:
    def test_a_phi_that_does_not_match_the_spread_is_refused(self):
        with pytest.raises(StructureError, match="phi is"):
            LinearGaussianTransition(
                phi=np.eye(2), process_std=[0.5], initial_std=[1.0]
            )

    def test_an_initial_std_of_a_different_width_is_refused(self):
        with pytest.raises(StructureError, match="initial_std"):
            LinearGaussianTransition(
                phi=np.eye(2), process_std=[0.5, 0.5], initial_std=[1.0]
            )

    @pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
    def test_a_spread_that_is_not_finite_and_positive_is_refused(self, bad):
        """``not (x > 0)`` rather than ``x <= 0``, so NaN is refused too, and
        ``isfinite`` separately because ``inf > 0`` is True -- an infinite
        spread zeroes the transition rows and sends the whole campaign's
        density to -inf, a thousand epochs after the declaration."""
        with pytest.raises(StructureError, match="finite and strictly positive"):
            LinearGaussianTransition(
                phi=[[0.5]], process_std=[bad], initial_std=[1.0]
            )

    def test_blocks_of_the_wrong_width_are_refused_by_name(self):
        """The slice at ``n_theta`` would take theta's columns for the chain's
        and come back finite, plausible, and a quantity nothing re-derives."""
        blocks = _blocks(n_theta=3)
        with pytest.raises(StructureError, match="columns wide"):
            chain_log_likelihood(blocks, _transition(), _theta(), NAMES, SHAPES)
