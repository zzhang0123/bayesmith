"""``smooth`` on a stiff chain, where inverting the precision stops working.

This file exists because ``rheplicant`` declined to delegate its smoother to
this one during the Wave D migration, and the reason was measured rather than
stylistic. Every test in ``test_chain.py`` uses ``process_std`` at or above
0.02, and at that stiffness the two implementations agree to ~5e-16. They part
company below about 1e-6, and this one is the side that leaves.

**The two arithmetics.** :func:`~bayesmith.marginal.chain.smooth` assembles the
explicit precision over ``zeta_1:N`` and calls ``jnp.linalg.inv``, so it pays
``kappa(F)``. ``rheplicant`` assembles the joint information **square root** and
does two triangular solves, so it pays ``sqrt(kappa(F))``.

**This module's own docstring already says which of those is right**: the
square-root form is "what keeps a thousand-epoch accumulation inside float64
where the explicit ``(F, b)`` form goes indefinite". ``chain_marginal`` follows
that advice. ``smooth`` does not.

**Deciding it needs no oracle.** Set ``phi = 1`` and let ``process_std -> 0``.
The chain freezes -- ``zeta_e`` is one latent shared by every epoch -- so the
smoothed posterior must converge and its across-epoch spread must vanish.
Measured (x64, ``rheplicant``'s ``chain_bank`` fixture, theta = (0.4, -1.1)):

====================  ==================  ==================
``process_std``       square root         this module
====================  ==================  ==================
1e-6                  0.454968749367      0.454968385497
1e-7                  0.454968749468      0.454928244813
1e-8                  0.454968748764      0.460387792656
1e-9                  0.454968747262      **0.931437422422**
1e-10                 0.454968730633      **nan**
====================  ==================  ==================

The ``0.93`` is the entry that matters. A ``nan`` announces itself; a value that
is exactly twice the right answer does not.

**On this file's own fixture** the symptom is slightly different from the table
above and worth stating exactly, because the table was measured elsewhere: here
the smoothed mean walks from -0.200652 to -0.469638 between ``process_std``
1e-6 and 1e-8 -- while the across-epoch spread reads 7.2e-16, so the answer
*looks* converged -- and every returned variance is ``nan`` at 1e-9. No
negative variance appears on this fixture.

**The markers are the point of the file.** These are ``xfail(strict=True)``, so
they pass today as known failures and go **red the moment the smoother is
fixed** -- at which point the markers come off and the tests become ordinary
guards. Without them a repair here has nothing to turn green, which is the
state this module was in when the defect was found: neither package had a test
that could see it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.marginal.chain import LinearGaussianTransition, smooth


@pytest.fixture(autouse=True)
def _double_precision():
    with jax.enable_x64(True):
        yield


def _blocks(epochs, n_theta=2, n_zeta=1, seed=0):
    generator = np.random.default_rng(seed)
    width = n_theta + n_zeta
    return (
        jnp.asarray(generator.normal(size=(epochs, width, width))),
        jnp.asarray(generator.normal(size=(epochs, width))),
        jnp.asarray(generator.normal(size=(epochs,))) * 0.1,
    )


_NAMES = ("gain", "offset")
_SHAPES = ((), ())
_THETA = {"gain": jnp.asarray(0.4), "offset": jnp.asarray(-1.1)}


@pytest.mark.parametrize(
    "process_std",
    (
        1e-6,
        1e-7,
        1e-8,
        pytest.param(
            1e-9,
            marks=pytest.mark.xfail(
                strict=True,
                reason="the inverted precision goes indefinite here and every "
                "returned variance is nan. Fix by assembling the information "
                "square root, as chain_marginal does; then remove this marker.",
            ),
        ),
    ),
)
@pytest.mark.parametrize("n_epochs", (6, 64))
def test_every_smoothed_variance_is_positive(process_std, n_epochs):
    """A variance is finite and positive. No tolerance, no reference value.

    Marked cell by cell rather than wholesale: 1e-6 through 1e-8 **pass** on this
    fixture and are here to keep the failing cell honest -- a repair that made
    1e-9 finite by breaking 1e-8 would show up as a new failure rather than as a
    marker to delete.
    """
    transition = LinearGaussianTransition(
        phi=jnp.eye(1),
        process_std=jnp.full((1,), process_std),
        initial_std=jnp.ones(1),
    )
    _, variance = smooth(_blocks(n_epochs), transition, _THETA, _NAMES, _SHAPES)
    variance = np.asarray(variance)
    assert np.all(np.isfinite(variance))
    assert np.all(variance > 0.0), (
        f"process_std={process_std:g}, n_epochs={n_epochs}: minimum variance is "
        f"{float(variance.min()):.6e}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="a frozen chain has a limit and this smoother does not reach it: the "
    "smoothed mean walks 0.269 between process_std 1e-6 and 1e-8, then goes nan. "
    "Same cause and same fix as the variance test above.",
)
def test_a_frozen_chain_converges_to_one_latent():
    """``phi = 1``, ``process_std -> 0``: one latent, so the mean must settle.

    **This is the sharper of the two tests, and the reason both are here.** At
    ``process_std = 1e-8`` the across-epoch *spread* is 7.2e-16 -- the answer
    looks perfectly converged -- while the mean itself has moved from -0.200652
    to -0.469638. A smoother that has lost its conditioning can go on returning
    a tight, plausible, wrong distribution; only comparing across stiffness
    catches it.
    """
    blocks = _blocks(6)
    previous = None
    for process_std in (1e-4, 1e-6, 1e-8, 1e-9):
        transition = LinearGaussianTransition(
            phi=jnp.eye(1),
            process_std=jnp.full((1,), process_std),
            initial_std=jnp.ones(1),
        )
        mean, _ = smooth(blocks, transition, _THETA, _NAMES, _SHAPES)
        mean = np.asarray(mean)
        assert np.all(np.isfinite(mean))
        assert float(np.max(np.ptp(mean, axis=0))) < 1e4 * process_std**2 + 1e-7
        if previous is not None:
            assert float(np.max(np.abs(mean - previous))) < 1e-7
        previous = mean
