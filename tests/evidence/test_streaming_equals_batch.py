"""The total oracle: a streamed campaign equals the batch it compresses.

The migration spec calls ``streaming == batch to roundoff`` B11's overall
oracle. This is that statement at the level the kernel and the compressor
make it: fold E epochs one at a time, never holding more than one, and
compare against the single dense Gaussian over all the data at once.

**Absolute log-densities and absolute evidences, never shapes.** Every
constant this layer carries is invisible in a posterior and visible only in
the evidence, so a test comparing means or widths would pass for a pipeline
that had dropped the marginalisation constant, the QR fold's corner, or the
mask -- each of which is a separate commit's worth of care upstream.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.evidence import SqrtInfo, compress, marginalise
from bayesmith.exact.precision import DiagonalPrecision

N_EPOCH, N_DATA, N_GLOBAL = 5, 6, 3


def _campaign(seed=0, unobserved=(), sigma_scale=1.0):
    """``[(design, data, sigma), ...]`` over ``N_EPOCH`` epochs.

    ``unobserved`` is ``{(epoch, index)}`` -- samples whose sigma is ``inf``,
    i.e. never taken.
    """
    rng = np.random.default_rng(seed)
    epochs = []
    for e in range(N_EPOCH):
        design = rng.normal(size=(N_DATA, N_GLOBAL))
        sigma = sigma_scale * np.linspace(0.4, 1.3, N_DATA)
        for gone_epoch, gone_index in unobserved:
            if gone_epoch == e:
                sigma[gone_index] = np.inf
        data = rng.normal(size=N_DATA)
        epochs.append((design, data, sigma))
    return epochs


def _batch_log_likelihood(epochs, x):
    """One dense Gaussian over every OBSERVED sample of every epoch."""
    design = np.concatenate([e[0] for e in epochs], axis=0)
    data = np.concatenate([e[1] for e in epochs])
    sigma = np.concatenate([e[2] for e in epochs])
    seen = np.isfinite(sigma)
    residual = data[seen] - design[seen] @ np.asarray(x)
    return float(
        -0.5 * np.sum((residual / sigma[seen]) ** 2)
        - 0.5 * np.sum(np.log(2.0 * math.pi * sigma[seen] ** 2))
    )


def _stream(epochs):
    """Fold the campaign, one epoch at a time, holding one term."""
    shapes = {"x": (N_GLOBAL,)}
    total = SqrtInfo.null(("x",), ((N_GLOBAL,),))
    for design, data, sigma in epochs:
        term = compress(
            {"x": jnp.asarray(design)},
            jnp.asarray(data),
            DiagonalPrecision(sigma=jnp.asarray(sigma)),
            shapes,
        )
        total = SqrtInfo.combine(total, term)
    return total


@pytest.mark.parametrize(
    "unobserved",
    [(), ((0, 2),), ((0, 2), (3, 0), (3, 5), (4, 1))],
    ids=["all observed", "one gap", "four gaps"],
)
def test_streaming_equals_batch(unobserved):
    """THE oracle. Folded one epoch at a time against one dense Gaussian.

    Swept over gaps because the mask is the part that a batch reference can
    disagree with silently: an unmasked normaliser gives ``-inf`` and an
    unmasked WEIGHT gives a NaN, but a mask applied to one half only gives a
    finite, plausible, wrong number.
    """
    with jax.enable_x64(True):
        epochs = _campaign(unobserved=unobserved)
        streamed = _stream(epochs)
        for point in ([0.0, 0.0, 0.0], [1.5, -0.5, 2.0], [-3.0, 4.0, 0.25]):
            x = jnp.asarray(point)
            got = float(streamed.log_prob({"x": x}))
            assert got == pytest.approx(
                _batch_log_likelihood(epochs, point), rel=1e-10, abs=1e-10
            ), point


def test_the_fold_order_does_not_change_the_answer():
    """Order-invariance across a campaign, on the absolute log-density."""
    with jax.enable_x64(True):
        epochs = _campaign(seed=3, unobserved=((1, 4),))
        forward = _stream(epochs)
        backward = _stream(list(reversed(epochs)))
        x = jnp.asarray([0.7, -1.3, 0.2])
        assert float(backward.log_prob({"x": x})) == pytest.approx(
            float(forward.log_prob({"x": x})), rel=1e-10, abs=1e-10
        )


def test_the_campaign_evidence_matches_a_dense_one():
    """Marginalise everything: the streamed evidence against `slogdet`.

    This is where every constant in the layer has to be right at once -- the
    compressor's normalisation, the fold's corner, and the marginalisation
    constant. A posterior would look correct with any of the three missing.
    """
    with jax.enable_x64(True):
        epochs = _campaign(seed=5, unobserved=((2, 3),))
        prior_std = 2.5
        prior = SqrtInfo(
            factor=jnp.eye(N_GLOBAL) / prior_std,
            target=jnp.zeros(N_GLOBAL),
            offset=jnp.asarray(
                -N_GLOBAL * math.log(prior_std)
                - 0.5 * N_GLOBAL * math.log(2.0 * math.pi)
            ),
            names=("x",),
            shapes=((N_GLOBAL,),),
        )
        total = SqrtInfo.combine(_stream(epochs), prior)
        evidence = float(marginalise(total, ["x"]).log_prob({}))

    design = np.concatenate([e[0] for e in epochs], axis=0)
    data = np.concatenate([e[1] for e in epochs])
    sigma = np.concatenate([e[2] for e in epochs])
    seen = np.isfinite(sigma)
    whitened = design[seen] / sigma[seen][:, None]
    target = data[seen] / sigma[seen]
    information = whitened.T @ whitened + np.eye(N_GLOBAL) / prior_std**2
    gradient = whitened.T @ target
    _, logdet = np.linalg.slogdet(information)
    expected = (
        -0.5 * float(target @ target)
        - 0.5 * float(np.sum(np.log(2.0 * math.pi * sigma[seen] ** 2)))
        - N_GLOBAL * math.log(prior_std)
        - 0.5 * N_GLOBAL * math.log(2.0 * math.pi)
        + 0.5 * float(gradient @ np.linalg.solve(information, gradient))
        + 0.5 * N_GLOBAL * math.log(2.0 * math.pi)
        - 0.5 * logdet
    )
    assert evidence == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_a_thousand_epochs_stay_finite_and_stay_right():
    """The condition-number claim the form exists for, at campaign length.

    ``kappa(R) = sqrt(kappa(F))`` is what keeps a long accumulation inside
    float64. A thousand epochs is the scale the spec names; this checks the
    fold neither drifts nor overflows, against the same batch reference.
    """
    with jax.enable_x64(True):
        rng = np.random.default_rng(11)
        shapes = {"x": (N_GLOBAL,)}
        total = SqrtInfo.null(("x",), ((N_GLOBAL,),))
        designs, datas, sigmas = [], [], []
        for _ in range(1000):
            design = rng.normal(size=(2, N_GLOBAL))
            sigma = np.full(2, 0.8)
            data = rng.normal(size=2)
            designs.append(design)
            datas.append(data)
            sigmas.append(sigma)
            total = SqrtInfo.combine(
                total,
                compress(
                    {"x": jnp.asarray(design)},
                    jnp.asarray(data),
                    DiagonalPrecision(sigma=jnp.asarray(sigma)),
                    shapes,
                ),
            )
        x = jnp.asarray([0.3, -0.7, 1.1])
        got = float(total.log_prob({"x": x}))
        condition = float(
            jnp.linalg.cond(total.factor)
        )
    epochs = list(zip(designs, datas, sigmas, strict=True))
    assert np.isfinite(got)
    assert got == pytest.approx(
        _batch_log_likelihood(epochs, [0.3, -0.7, 1.1]), rel=1e-10
    )
    # the working condition number is the SQUARE ROOT of the information's
    assert condition**2 == pytest.approx(
        float(jnp.linalg.cond(total.information())), rel=1e-6
    )
