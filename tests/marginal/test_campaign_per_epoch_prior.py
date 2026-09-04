"""A per-epoch nuisance whose declared prior WIDTH varies across epochs.

``epoch_terms`` read one entry of each per-epoch latent's declared prior and
broadcast it to every epoch. For a campaign declaring ``tau = [0.5, 1, 2, 4]``
that means three of the four epochs were integrated against a prior the model
does not declare, and what came back was a finite, plausible, wrong marginal
likelihood -- measured at 2.75 nats over four epochs, with no warning and no
refusal.

The comment at the reduction site acknowledged the limitation and named no
ledger row, which is the shape of thing that stays true for a release. The
information was never missing: ``gaussian_parts`` returns the whole per-epoch
vector and the code took ``[0]``; ``nuisance_prior`` already broadcasts a
per-component std; and ``one_epoch`` is already ``vmap``ped, so the prior only
had to stop being a closed-over scalar and start being a mapped argument.

**The oracle here shares nothing with the implementation.** Each epoch's
marginal is a one-dimensional Gaussian written out by hand --
``d_e ~ N(2g + m, tau_e^2 + sigma^2)`` -- summed in Python. No QR, no pivots,
no offset arithmetic, and no ``SqrtInfo``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import det, observe, plate, sample, trace
from bayesmith.marginal import compress_campaign

EPOCHS = 4
SIGMA = 0.55
PRIOR_MEAN = 0.4
DATA = (0.3, 1.1, 0.4, 0.8)

#: Four different declared widths. The first is the one the old code used for
#: all of them, which is why it is the one a wrong implementation reproduces.
HETEROGENEOUS = (0.5, 1.0, 2.0, 4.0)
HOMOGENEOUS = (0.5, 0.5, 0.5, 0.5)


def _campaign(widths):
    """`n` is per-epoch and declares its own width in each epoch.

    Both parameters are ``(E,)`` arrays. The spelling with a SCALAR mean and a
    vector width is refused upstream, in ``unchecked_operator``, before this
    layer is reached -- pinned by
    :func:`test_a_scalar_mean_beside_a_vector_width_is_refused_upstream`.
    """
    taus = jnp.asarray(widths)
    data = jnp.asarray(DATA)

    def model():
        epoch = plate("epoch", EPOCHS)
        g = sample("g", lambda: dist.Normal(0.0, 2.0))
        n = sample(
            "n",
            lambda: dist.Normal(jnp.full((EPOCHS,), PRIOR_MEAN), taus),
            plate=epoch,
        )
        mu = det(
            "mu", lambda g_, n_: 2 * g_ + n_, g, n, plate=epoch, linear_in=("g", "n")
        )
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data, plate=epoch)

    return trace(model)


def _dense(g_value, widths):
    """``sum_e log N(d_e | 2g + m, tau_e^2 + sigma^2)``, by hand."""
    total = 0.0
    for observation, tau in zip(DATA, widths, strict=True):
        variance = tau**2 + SIGMA**2
        residual = observation - (2.0 * g_value + PRIOR_MEAN)
        total += -0.5 * (
            residual * residual / variance
            + np.log(variance)
            + np.log(2.0 * np.pi)
        )
    return float(total)


@pytest.mark.parametrize("g_value", [0.0, 1.5])
def test_each_epoch_is_integrated_against_the_width_it_declares(g_value):
    with jax.enable_x64(True):
        term = compress_campaign(_campaign(HETEROGENEOUS), "epoch")
        found = float(term.log_prob({"g": jnp.asarray(g_value)}))
        assert found == pytest.approx(_dense(g_value, HETEROGENEOUS), abs=1e-9)


def test_the_wrong_answer_is_far_enough_away_to_fail_on():
    """The fixture must be able to fail, or the test above proves nothing.

    A campaign that used the first epoch's width everywhere answers
    ``-3.086436`` where the truth is ``-5.838560``: 2.75 nats, on four epochs.
    The gap grows with the campaign, which is what makes it worth a test rather
    than a comment.
    """
    with jax.enable_x64(True):
        truth = _dense(0.0, HETEROGENEOUS)
        first_width_everywhere = _dense(0.0, HOMOGENEOUS)
        assert abs(truth - first_width_everywhere) > 2.0
        assert first_width_everywhere == pytest.approx(-3.086435510, abs=1e-8)
        assert truth == pytest.approx(-5.838560122, abs=1e-8)


@pytest.mark.parametrize("g_value", [0.0, 1.5])
def test_a_homogeneous_campaign_is_unchanged(g_value):
    """The half that must NOT move.

    Every campaign fixture in this package declares one width, so the repair
    has to leave that case exactly where it was -- against the same hand-written
    oracle, not against a recorded number.
    """
    with jax.enable_x64(True):
        term = compress_campaign(_campaign(HOMOGENEOUS), "epoch")
        found = float(term.log_prob({"g": jnp.asarray(g_value)}))
        assert found == pytest.approx(_dense(g_value, HOMOGENEOUS), abs=1e-9)


def test_a_scalar_mean_beside_a_vector_width_is_refused_upstream():
    """The spelling this repair does NOT reach, pinned so it stays visible.

    ``dist.Normal(0.4, taus)`` with a vector ``taus`` fails inside
    ``unchecked_operator``, before ``epoch_terms`` sees it. That is a limitation
    of the block layer rather than of this one, and it FAILS rather than lying,
    which is the acceptable half of the two outcomes. Pinned so that a later
    release lifting it finds this test rather than rediscovering the shape.
    """
    with jax.enable_x64(True):
        taus = jnp.asarray(HETEROGENEOUS)
        data = jnp.asarray(DATA)

        def model():
            epoch = plate("epoch", EPOCHS)
            g = sample("g", lambda: dist.Normal(0.0, 2.0))
            n = sample("n", lambda: dist.Normal(PRIOR_MEAN, taus), plate=epoch)
            mu = det(
                "mu",
                lambda g_, n_: 2 * g_ + n_,
                g,
                n,
                plate=epoch,
                linear_in=("g", "n"),
            )
            observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=data, plate=epoch)

        with pytest.raises(ValueError, match="[Bb]roadcast"):
            compress_campaign(trace(model), "epoch")


def test_the_fold_is_still_flat_in_the_campaign_length():
    """The property the vectorisation exists for, kept.

    A folded term is three leaves whatever the campaign length, and the cost is
    dominated by tracing rather than by the epoch count. Carrying the prior as
    a mapped argument rather than a closed-over scalar must not change either.
    """
    with jax.enable_x64(True):
        for size in (8, 64):
            widths = tuple(np.linspace(0.4, 2.0, size))
            data = tuple(np.linspace(-1.0, 1.0, size))
            taus = jnp.asarray(widths)
            observations = jnp.asarray(data)

            def model(taus=taus, observations=observations, size=size):
                epoch = plate("epoch", size)
                g = sample("g", lambda: dist.Normal(0.0, 2.0))
                n = sample(
                    "n",
                    lambda: dist.Normal(jnp.full((size,), PRIOR_MEAN), taus),
                    plate=epoch,
                )
                mu = det(
                    "mu",
                    lambda g_, n_: 2 * g_ + n_,
                    g,
                    n,
                    plate=epoch,
                    linear_in=("g", "n"),
                )
                observe(
                    "d",
                    lambda m: dist.Normal(m, SIGMA),
                    mu,
                    obs=observations,
                    plate=epoch,
                )

            term = compress_campaign(trace(model), "epoch")
            assert len(jax.tree.leaves(term)) == 3, (
                f"a folded campaign of {size} epochs is three leaves whatever "
                f"its length; that is the archive property the square-root "
                f"form exists for"
            )
