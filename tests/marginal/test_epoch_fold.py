"""One epoch with its nuisances integrated, against the marginal's own formula.

The oracle here shares nothing with the implementation, not even its shape.
`compress_epoch` whitens, appends prior rows, stacks and re-triangularises;
the reference is the closed form for a linear-Gaussian marginal,

    d | x_g  ~  N(A_g x_g + A_n m_n + c,  N + A_n S_n A_n^T)

evaluated with a materialised covariance and `numpy.linalg`. No QR, no
pivots, no offset arithmetic in common.

**Absolute log-densities.** The epoch's marginalisation constant is one
`-sum log|R_bb,ii|` per epoch, so a campaign of E epochs carries E copies of
it. A test on posterior shape would be blind to every one.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.exact.precision import CirculantPrecision, DiagonalPrecision, dense
from bayesmith.marginal import SqrtInfo, compress_epoch, marginalise

N_DATA, N_GLOBAL, N_NUISANCE = 8, 2, 3


def _epoch(seed=0, sigma=0.6, prior_std=0.7, prior_mean=None, n_data=N_DATA):
    rng = np.random.default_rng(seed)
    return {
        "global_design": rng.normal(size=(n_data, N_GLOBAL)),
        "nuisance_design": rng.normal(size=(n_data, N_NUISANCE)),
        "data": rng.normal(size=n_data),
        "sigma": np.full(n_data, sigma),
        "prior_std": prior_std,
        "prior_mean": (
            np.zeros(N_NUISANCE) if prior_mean is None else np.asarray(prior_mean)
        ),
    }


def _marginal_reference(epoch, x_global, noise_covariance=None, offset=None):
    """``log N(d | A_g x_g + A_n m_n + c, N + A_n S_n A_n^T)``."""
    design = epoch["nuisance_design"]
    noise = (
        np.diag(epoch["sigma"] ** 2)
        if noise_covariance is None
        else np.asarray(noise_covariance)
    )
    covariance = noise + design @ (epoch["prior_std"] ** 2 * np.eye(N_NUISANCE)) @ design.T
    mean = epoch["global_design"] @ np.asarray(x_global) + design @ epoch["prior_mean"]
    if offset is not None:
        mean = mean + np.asarray(offset)
    residual = epoch["data"] - mean
    _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
    return -0.5 * float(residual @ np.linalg.solve(covariance, residual)) - 0.5 * float(
        logdet
    )


def _fold(epoch, precision=None, offset_prediction=None):
    return compress_epoch(
        {"g": jnp.asarray(epoch["global_design"])},
        jnp.asarray(epoch["data"]),
        precision or DiagonalPrecision(sigma=jnp.asarray(epoch["sigma"])),
        {"g": (N_GLOBAL,)},
        nuisance_design={"n": jnp.asarray(epoch["nuisance_design"])},
        nuisance_shapes={"n": (N_NUISANCE,)},
        nuisance_prior_std={"n": epoch["prior_std"]},
        nuisance_prior_mean={"n": jnp.asarray(epoch["prior_mean"])},
        offset_prediction=offset_prediction,
    )


class TestOneEpochMarginal:
    @pytest.mark.parametrize("prior_std", [0.7, 1.0, 3.0])
    def test_it_is_the_marginal_likelihood(self, prior_std):
        """Swept over the prior scale because the term rheplicant once lost is
        exactly zero at ``std = 1``."""
        with jax.enable_x64(True):
            epoch = _epoch(prior_std=prior_std)
            term = _fold(epoch)
            assert term.names == ("g",)
            for point in ([0.0, 0.0], [1.5, -2.0], [-0.75, 0.25]):
                got = float(term.log_prob({"g": jnp.asarray(point)}))
                assert got == pytest.approx(
                    _marginal_reference(epoch, point), rel=1e-9, abs=1e-9
                ), (prior_std, point)

    def test_a_non_zero_prior_mean_moves_it_the_way_the_formula_says(self):
        with jax.enable_x64(True):
            epoch = _epoch(seed=3, prior_mean=[0.5, -1.25, 2.0])
            term = _fold(epoch)
            for point in ([0.0, 0.0], [1.0, 1.0]):
                assert float(term.log_prob({"g": jnp.asarray(point)})) == pytest.approx(
                    _marginal_reference(epoch, point), rel=1e-9, abs=1e-9
                )

    def test_an_offset_prediction_is_carried_through_the_marginalisation(self):
        with jax.enable_x64(True):
            epoch = _epoch(seed=4)
            constant = np.linspace(-1.0, 1.0, N_DATA)
            term = _fold(epoch, offset_prediction=jnp.asarray(constant))
            assert float(term.log_prob({"g": jnp.asarray([0.4, -0.2])})) == pytest.approx(
                _marginal_reference(epoch, [0.4, -0.2], offset=constant),
                rel=1e-9,
                abs=1e-9,
            )

    def test_a_CORRELATED_epoch_marginalises_its_nuisances_too(self):
        """B9's row reaching the part of B11 that integrates.

        The reference's `N` is materialised from the operator by application,
        so nothing on the oracle side is an FFT.
        """
        with jax.enable_x64(True):
            epoch = _epoch(seed=6)
            lag = np.minimum(np.arange(N_DATA), N_DATA - np.arange(N_DATA))
            precision = CirculantPrecision(
                first_column=jnp.asarray(0.5 * 0.45**lag + 0.2)
            )
            term = _fold(epoch, precision=precision)
            covariance = np.linalg.inv(
                np.asarray(dense(precision, N_DATA, jnp.float64))
            )
            got = float(term.log_prob({"g": jnp.asarray([0.9, -0.3])}))
        assert got == pytest.approx(
            _marginal_reference(epoch, [0.9, -0.3], noise_covariance=covariance),
            rel=1e-9,
            abs=1e-9,
        )

    def test_no_nuisances_is_the_plain_compressor(self):
        with jax.enable_x64(True):
            epoch = _epoch(seed=7)
            term = compress_epoch(
                {"g": jnp.asarray(epoch["global_design"])},
                jnp.asarray(epoch["data"]),
                DiagonalPrecision(sigma=jnp.asarray(epoch["sigma"])),
                {"g": (N_GLOBAL,)},
            )
            residual = epoch["data"] - epoch["global_design"] @ np.array([0.5, -1.0])
            expected = -0.5 * float(
                np.sum((residual / epoch["sigma"]) ** 2)
            ) - 0.5 * float(np.sum(np.log(2.0 * math.pi * epoch["sigma"] ** 2)))
            assert float(
                term.log_prob({"g": jnp.asarray([0.5, -1.0])})
            ) == pytest.approx(expected, rel=1e-10)

    def test_nuisances_without_a_prior_are_refused(self):
        """The integral would diverge, and finite arithmetic hides it."""
        with jax.enable_x64(True):
            epoch = _epoch(seed=8)
            with pytest.raises(StructureError, match="optional regulariser"):
                compress_epoch(
                    {"g": jnp.asarray(epoch["global_design"])},
                    jnp.asarray(epoch["data"]),
                    DiagonalPrecision(sigma=jnp.asarray(epoch["sigma"])),
                    {"g": (N_GLOBAL,)},
                    nuisance_design={"n": jnp.asarray(epoch["nuisance_design"])},
                )


class TestACampaignOfNuisanceEpochs:
    """E epochs, each with its OWN nuisance, folded into one term.

    This is the shape the evidence layer exists for: the nuisances never meet,
    only one epoch is ever held, and the campaign carries E separate
    marginalisation constants.
    """

    @staticmethod
    def _campaign(n_epoch=4, prior_std=0.7):
        return [_epoch(seed=20 + i, prior_std=prior_std) for i in range(n_epoch)]

    def test_it_matches_a_dense_joint_with_every_nuisance_integrated_at_once(self):
        """The strongest statement available: the streamed answer against ONE
        dense Gaussian over the whole campaign, nuisances integrated jointly.

        The dense side never folds. It builds the block-diagonal nuisance
        design over all epochs and uses the same marginal formula once, so the
        E per-epoch constants have to sum to the single joint one.
        """
        with jax.enable_x64(True):
            epochs = self._campaign()
            total = SqrtInfo.null(("g",), ((N_GLOBAL,),))
            for epoch in epochs:
                total = SqrtInfo.combine(total, _fold(epoch))
            point = [0.8, -1.4]
            got = float(total.log_prob({"g": jnp.asarray(point)}))

        rows = N_DATA * len(epochs)
        global_design = np.concatenate([e["global_design"] for e in epochs], axis=0)
        data = np.concatenate([e["data"] for e in epochs])
        sigma = np.concatenate([e["sigma"] for e in epochs])
        block = np.zeros((rows, N_NUISANCE * len(epochs)))
        for i, epoch in enumerate(epochs):
            block[
                i * N_DATA : (i + 1) * N_DATA,
                i * N_NUISANCE : (i + 1) * N_NUISANCE,
            ] = epoch["nuisance_design"]
        prior = epochs[0]["prior_std"] ** 2 * np.eye(N_NUISANCE * len(epochs))
        covariance = np.diag(sigma**2) + block @ prior @ block.T
        residual = data - global_design @ np.asarray(point)
        _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
        expected = -0.5 * float(
            residual @ np.linalg.solve(covariance, residual)
        ) - 0.5 * float(logdet)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_the_campaign_evidence_matches_a_dense_one(self):
        """Marginalise the globals too. Every constant in the layer at once."""
        with jax.enable_x64(True):
            epochs = self._campaign(n_epoch=3)
            global_prior_std = 2.0
            total = SqrtInfo.null(("g",), ((N_GLOBAL,),))
            for epoch in epochs:
                total = SqrtInfo.combine(total, _fold(epoch))
            total = SqrtInfo.combine(
                total,
                SqrtInfo(
                    factor=jnp.eye(N_GLOBAL) / global_prior_std,
                    target=jnp.zeros(N_GLOBAL),
                    offset=jnp.asarray(
                        -N_GLOBAL * math.log(global_prior_std)
                        - 0.5 * N_GLOBAL * math.log(2.0 * math.pi)
                    ),
                    names=("g",),
                    shapes=((N_GLOBAL,),),
                ),
            )
            evidence = float(marginalise(total, ["g"]).log_prob({}))

        rows = N_DATA * len(epochs)
        global_design = np.concatenate([e["global_design"] for e in epochs], axis=0)
        data = np.concatenate([e["data"] for e in epochs])
        sigma = np.concatenate([e["sigma"] for e in epochs])
        block = np.zeros((rows, N_NUISANCE * len(epochs)))
        for i, epoch in enumerate(epochs):
            block[
                i * N_DATA : (i + 1) * N_DATA,
                i * N_NUISANCE : (i + 1) * N_NUISANCE,
            ] = epoch["nuisance_design"]
        prior = epochs[0]["prior_std"] ** 2 * np.eye(N_NUISANCE * len(epochs))
        # every latent integrated: d ~ N(0, N + A_n S_n A_n^T + A_g S_g A_g^T)
        covariance = (
            np.diag(sigma**2)
            + block @ prior @ block.T
            + global_design @ (global_prior_std**2 * np.eye(N_GLOBAL)) @ global_design.T
        )
        _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
        expected = -0.5 * float(
            data @ np.linalg.solve(covariance, data)
        ) - 0.5 * float(logdet)
        assert evidence == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_the_fold_order_does_not_change_it(self):
        with jax.enable_x64(True):
            epochs = self._campaign()
            point = jnp.asarray([0.3, 0.9])

            def run(order):
                total = SqrtInfo.null(("g",), ((N_GLOBAL,),))
                for epoch in order:
                    total = SqrtInfo.combine(total, _fold(epoch))
                return float(total.log_prob({"g": point}))

            assert run(list(reversed(epochs))) == pytest.approx(
                run(epochs), rel=1e-10, abs=1e-10
            )


def test_a_unit_prior_fixture_could_not_see_the_missing_term():
    """Why every fixture above sweeps the prior scale, as an executable fact.

    The nuisance prior's own normalisation is
    ``-sum(log std) - (n/2) log 2 pi``, and the first half is **exactly zero
    at std = 1**. rheplicant shipped it missing, and the probe that passed
    used unit priors.

    Measured directly here rather than asserted: the correct offset and the
    defective one are IDENTICAL at ``std = 1`` and differ by exactly
    ``n * log(std)`` elsewhere. So a suite whose fixtures all used unit priors
    would be green on a compressor that had dropped the term -- confirmed by
    mutation: removing it from `nuisance_prior` fails 7 tests here, and NONE
    of them if the sweep is narrowed to ``prior_std = 1.0``.
    """
    with jax.enable_x64(True):
        from bayesmith.marginal import nuisance_prior

        for prior_std in (0.5, 1.0, 2.0, 7.0):
            term = nuisance_prior(
                ("n",),
                {"n": (N_NUISANCE,)},
                {"n": prior_std},
                None,
                ("g",),
                {"g": (N_GLOBAL,)},
            )
            defective = -0.5 * N_NUISANCE * math.log(2.0 * math.pi)
            gap = float(term.offset) - defective
            assert gap == pytest.approx(
                -N_NUISANCE * math.log(prior_std), rel=1e-12, abs=1e-14
            ), prior_std
            if prior_std == 1.0:
                assert gap == pytest.approx(0.0, abs=1e-14), "blind at unit prior"
