"""The log-determinant gap, measured on the package that documents it.

``rheplicant/inference/noise.py`` states a closed form: for the multiplicative
(radiometer) model, dropping ``log 2 pi sigma^2`` from the Gaussian density
gives "a *different estimator*" -- generalized least squares -- which "returns
``sum d^2 / sum d``, biased high by ``(1 + f^2)``", while the full density is
asymptotically unbiased.

That sentence is the whole of defect B1: the same rheplicant model reaches
``nuts`` through ``numpyro_bridge``, whose ``dist.Normal`` carries its own
``-log sigma``, and reaches ``plan.sample``'s gradient block through
``engines.conditional_potential``, which does not. Two estimators, one model,
no guard between them.

**Why this lives here rather than in ``tests/exact/``.** It is a fact about
rheplicant, asserted by running rheplicant, and it is the reference the port
is allowed to disagree with only on purpose. ``bayesmith.exact.gls`` does NOT
have this bias -- it freezes sigma per inner solve (IRLS) rather than
differentiating through it, so its fixed point sits on the unbiased side. The
migration spec's first draft asked for the opposite, that bayesmith's
frozen-sigma path differ from a live-sigma path by ``(1 + f^2)``; measured,
it does not, and building that test would have pulled a correct estimator
toward a bias it does not have. ``tests/exact/test_correct.py`` carries the
bayesmith half of the same claim.

Nothing here samples. Both estimators have a closed form on this model, so
every assertion is deterministic given the data:

* ``include_logdet=False``:  ``mu = sum d^2 / sum d``
* ``include_logdet=True``:   ``n f^2 mu^2 + mu sum d - sum d^2 = 0``, positive
  root -- which is why the full density is unbiased rather than merely less
  biased. Substituting the large-``n`` moments ``sum d / n -> mu0`` and
  ``sum d^2 / n -> mu0^2 (1 + f^2)`` gives
  ``mu0 (-1 + sqrt(1 + 4 f^2 + 4 f^4)) / (2 f^2) = mu0 (-1 + (1 + 2 f^2)) /
  (2 f^2) = mu0`` exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

pytestmark = pytest.mark.crosscheck

#: Constant mean, so both estimators are closed-form. A design matrix would
#: reduce to the same problem in ``u = d / x`` and prove nothing extra here.
TRUE_MEAN = 4.0

#: Large enough that the asymptotic statement is what is being read rather
#: than one draw's scatter. Measured at f=0.5: the ratio lands 0.024% from
#: ``1 + f^2``, against 2.5% at n=2000.
BIG = 200_000


def _noise(fractional: float, **kw):
    """A ``RadiometerNoise`` with a chosen ``f``.

    ``f = 1 / sqrt(channel_width * integration_time)``, so the product is
    ``1 / f**2``. Built through the real constructor rather than by patching
    ``fractional``, because ``__check_init__`` is part of what is being used.
    """
    from rheplicant.inference.noise import RadiometerNoise

    return RadiometerNoise(
        channel_width=1.0 / fractional**2, integration_time=1.0, **kw
    )


def _argmax(like, data, n):
    """The maximiser, found by a route that shares nothing with the closed form.

    scipy on a Python closure over the rheplicant likelihood: no JAX gradient,
    no algebra of ours. A closed form checked against its own rearrangement
    would be checking arithmetic, not the estimator.
    """
    return minimize_scalar(
        lambda m: -float(like(jnp.full((n,), m), data)),
        bracket=(TRUE_MEAN * 0.5, TRUE_MEAN, TRUE_MEAN * 3.0),
        options={"xtol": 1e-12},
    ).x


def _data(noise, n, seed):
    """Drawn by the noise model's OWN generator, so the fixture cannot
    disagree with the likelihood weighting it.

    ``realise`` is multiplicative, ``d (1 + f w)``. On THIS fixture that is
    indistinguishable from the additive ``mu + sigma(mu) w``, because the mean
    is positive and there is no floor, so ``|mu| = mu`` and the two coincide
    -- measured: swapping the generator for the additive form leaves all ten
    tests green. The two part company only where the prediction crosses zero
    (``std`` takes an absolute value, a generator must not) or where a floor
    applies (``std`` imposes it, ``realise`` does not), which is defect B4's
    territory and not what is under test here. Using the package's own
    generator anyway costs nothing and removes the question.
    """
    return noise.realise(jnp.full((n,), TRUE_MEAN), key=jax.random.key(seed))


@pytest.mark.parametrize("fractional", [0.05, 0.2, 0.5])
def test_dropping_the_logdet_gives_the_sum_d2_over_sum_d_estimator(fractional):
    """rheplicant's stated closed form for the GLS variant, run."""
    from rheplicant.inference.noise import NoiseModelLikelihood

    with jax.enable_x64(True):
        noise = _noise(fractional)
        data = _data(noise, 2_000, seed=11)
        like = NoiseModelLikelihood(noise=noise, include_logdet=False)
        found = _argmax(like, data, 2_000)
        closed = float(jnp.sum(data**2) / jnp.sum(data))
    assert found == pytest.approx(closed, rel=1e-8)


@pytest.mark.parametrize("fractional", [0.05, 0.2, 0.5])
def test_keeping_the_logdet_gives_the_root_of_the_quadratic(fractional):
    """The full density's own closed form, which the module does not spell out.

    Derived here because the asymptotic claim rests on it: it is what makes
    the full density exactly unbiased rather than differently biased, and a
    test that asserted only "closer to the truth" would pass on an estimator
    that was merely less wrong.
    """
    from rheplicant.inference.noise import NoiseModelLikelihood

    n = 2_000
    with jax.enable_x64(True):
        noise = _noise(fractional)
        data = _data(noise, n, seed=11)
        like = NoiseModelLikelihood(noise=noise, include_logdet=True)
        found = _argmax(like, data, n)
        s1, s2 = float(jnp.sum(data)), float(jnp.sum(data**2))
        f = noise.fractional
        closed = (-s1 + np.sqrt(s1**2 + 4 * n * f**2 * s2)) / (2 * n * f**2)
    assert found == pytest.approx(closed, rel=1e-8)


@pytest.mark.parametrize("fractional", [0.05, 0.2, 0.5])
def test_the_gap_between_them_is_one_plus_f_squared_in_direction_and_size(
    fractional,
):
    """B1's acceptance: the known discrepancy, as a regression test.

    Both halves are asserted. The RATIO pins the magnitude across an f that
    varies the answer by a factor of 25 between the ends, so a constant
    fudge cannot satisfy it; the SIGN pins the direction, which is the half
    that says which of the two engines is the optimistic one.
    """
    from rheplicant.inference.noise import NoiseModelLikelihood

    with jax.enable_x64(True):
        noise = _noise(fractional)
        data = _data(noise, BIG, seed=11)
        gls = _argmax(
            NoiseModelLikelihood(noise=noise, include_logdet=False), data, BIG
        )
        full = _argmax(
            NoiseModelLikelihood(noise=noise, include_logdet=True), data, BIG
        )
    assert gls > full > 0.0, (gls, full)
    assert gls / full == pytest.approx(1.0 + fractional**2, rel=2e-3)
    assert full == pytest.approx(TRUE_MEAN, rel=5e-3)


def test_the_gap_closes_when_sigma_does_not_depend_on_the_prediction():
    """ANTI-VACUITY, and the reason the gap is attributed to the dependence.

    With a constant sigma the log-determinant is an additive constant and
    dropping it changes nothing -- so the two estimators must coincide. A
    test suite that only ever showed the two differing would be consistent
    with the difference coming from anywhere in the likelihood; this is what
    makes ``depends_on_prediction`` the named cause.
    """
    from rheplicant.inference.noise import HomoscedasticNoise, NoiseModelLikelihood

    n = 2_000
    with jax.enable_x64(True):
        noise = HomoscedasticNoise(sigma=jnp.asarray(0.5))
        data = TRUE_MEAN + 0.5 * jax.random.normal(jax.random.key(11), (n,))
        gls = _argmax(NoiseModelLikelihood(noise=noise, include_logdet=False), data, n)
        full = _argmax(NoiseModelLikelihood(noise=noise, include_logdet=True), data, n)
        # INSIDE the context on purpose. `data` is float64, but `jnp.mean`
        # evaluated outside it truncates to float32 and the comparison below
        # fails at the eighth digit -- which reads as the estimator being
        # wrong. The x64 context manager governs the operation, not the array.
        mean = float(jnp.mean(data))
    assert gls == pytest.approx(full, rel=1e-9)
    assert gls == pytest.approx(mean, rel=1e-8)
