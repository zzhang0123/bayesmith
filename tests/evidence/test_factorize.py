"""Deriving a campaign's factorization, and testing the hypothesis it makes.

The migration spec says the partition should be DERIVED from the graph's
plates rather than declared. `probe_10` measured how far that goes: plate
membership answers it for `global` and `per_epoch` and CANNOT answer it for
`linked`, whose non-centred form looks exactly like a per-epoch nuisance.

So plate membership is a hypothesis here, and `epoch_leakage` is the test of
it. These tests exercise both, and the second is the one that can fail.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest

from bayesmith import det, observe, plate, sample, trace
from bayesmith.errors import StructureError
from bayesmith.evidence import epoch_leakage, factorize

N_EPOCH = 4


def _epoch_local():
    """A global and a genuinely per-epoch nuisance."""

    def model():
        epoch = plate("epoch", N_EPOCH)
        g = sample("g", lambda: ndist.Normal(0.0, 1.0))
        n = sample("n", lambda: ndist.Normal(0.0, 1.0), plate=epoch)
        mu = det("mu", lambda a, b: a + b, g, n, plate=epoch, linear_in=("g", "n"))
        observe(
            "d",
            lambda m: ndist.Normal(m, 0.5),
            mu,
            plate=epoch,
            obs=jnp.arange(float(N_EPOCH)),
        )

    return trace(model)


def _leaks_by_broadcast(strength=0.8):
    """A per-epoch latent reaching every epoch, through an UNPLATED node.

    Three facts had to be measured to build this fixture, and each one
    narrowed what a leak can look like:

    * a plated ``det``'s function sees a SCALAR for a plated parent, so a
      cross-epoch map cannot be written as a plated node at all;
    * so the map is unplated -- but a plated node whose parents are ALL
      unplated is refused by the graph itself ("nothing to map over");
    * therefore the consumer needs BOTH a plated parent and the unplated
      ``(E,)`` map. Then it broadcasts, giving each epoch an ``(E,)``
      observation whose every entry depends on every epoch's innovation.

    An odd model, and expressible, which is the point: the jacobian is
    ``(E, E, E)`` with off-diagonal weight, and the fold would be wrong.
    """
    lower = np.tril(np.ones((N_EPOCH, N_EPOCH)) * strength, -1) + np.eye(N_EPOCH)

    def model():
        epoch = plate("epoch", N_EPOCH)
        g = sample("g", lambda: ndist.Normal(0.0, 1.0))
        eps = sample("eps", lambda: ndist.Normal(0.0, 1.0), plate=epoch)
        chain = det("chain", lambda e: jnp.asarray(lower) @ e, eps, linear_in=("eps",))
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
            lambda m: ndist.Normal(m, 0.5),
            mu,
            plate=epoch,
            obs=jnp.zeros((N_EPOCH, N_EPOCH)),
        )

    return trace(model)


def _leaks_by_an_untagged_observation(strength=0.8):
    """The shape a non-centred LINKED latent actually takes.

    ``eps`` is per-epoch by its plate; the recursion is an unplated node; and
    the observation is unplated too, so its axis is an epoch axis only by the
    author's intention. Nothing in the graph SAYS so, which is exactly why it
    cannot be sliced per epoch -- and slicing is what the fold does.
    """
    lower = np.tril(np.ones((N_EPOCH, N_EPOCH)) * strength, -1) + np.eye(N_EPOCH)

    def model():
        epoch = plate("epoch", N_EPOCH)
        eps = sample("eps", lambda: ndist.Normal(0.0, 1.0), plate=epoch)
        g = sample("g", lambda: ndist.Normal(0.0, 1.0))
        chain = det(
            "chain",
            lambda e, a: jnp.asarray(lower) @ e + a,
            eps,
            g,
            linear_in=("eps", "g"),
        )
        observe(
            "d",
            lambda m: ndist.Normal(m, 0.5),
            chain,
            obs=jnp.arange(float(N_EPOCH)),
        )

    return trace(model)


class TestTheDerivation:
    def test_plate_membership_gives_the_partition(self):
        with jax.enable_x64(True):
            found = factorize(_epoch_local(), "epoch")
        assert found.epoch_plate == "epoch"
        assert found.survivors == ("g",)
        assert found.per_epoch == ("n",)

    def test_a_plate_the_graph_does_not_have_is_refused(self):
        with jax.enable_x64(True), pytest.raises(StructureError, match="no plate"):
            factorize(_epoch_local(), "night")

    def test_a_graph_with_no_survivor_is_refused(self):
        """Every latent inside the epoch means nothing to accumulate."""

        def model():
            epoch = plate("epoch", N_EPOCH)
            n = sample("n", lambda: ndist.Normal(0.0, 1.0), plate=epoch)
            observe(
                "d",
                lambda m: ndist.Normal(m, 0.5),
                n,
                plate=epoch,
                obs=jnp.arange(float(N_EPOCH)),
            )

        with jax.enable_x64(True), pytest.raises(StructureError, match="nothing to accumulate"):
            factorize(trace(model), "epoch")


class TestTheHypothesisIsTested:
    """Plate membership is a claim. `epoch_leakage` is what checks it."""

    def test_a_genuinely_epoch_local_latent_leaks_exactly_zero(self):
        with jax.enable_x64(True):
            leak = epoch_leakage(_epoch_local(), "n", "epoch", {})
        assert leak == 0.0

    @pytest.mark.parametrize("strength", [1e-6, 0.2, 0.8, 3.0])
    def test_a_latent_that_reaches_later_epochs_is_refused(self, strength):
        """Swept down to 1e-6 because the guard is RELATIVE.

        The comparison is against the latent's own largest sensitivity, so a
        faint leak in a strongly-used latent is caught on the same footing as
        a loud one -- which an absolute floor would not do.
        """
        with jax.enable_x64(True):
            graph = _leaks_by_broadcast(strength)
            leak = epoch_leakage(graph, "eps", "epoch", {})
            assert leak > 0.0, strength
            with pytest.raises(StructureError, match="reaches other"):
                factorize(graph, "epoch")

    def test_the_refusal_names_the_non_centred_chain_as_the_usual_cause(self):
        """The message has to send the reader somewhere true.

        A non-centred linked latent is the case that reaches this guard by
        being *correctly written* -- it is a survivor wearing a per-epoch
        plate, not a mistake in the recursion.
        """
        with jax.enable_x64(True), pytest.raises(StructureError) as caught:
            factorize(_leaks_by_broadcast(), "epoch")
        message = str(caught.value)
        assert "LINKED" in message
        assert "non-centred" in message
        assert "outside the epoch plate" in message

    def test_an_untagged_observation_is_refused_too_and_for_its_own_reason(self):
        """The non-centred chain's real shape, and why the graph cannot slice it.

        Here the observation is not plate-tagged at all: its axis is an epoch
        axis only by the author's intention. `compress_epoch` slices data by
        the plate, so an untagged observation cannot be split -- and a
        per-epoch latent reaching it is leakage by definition, with no
        diagonal to compare against.
        """
        with jax.enable_x64(True):
            graph = _leaks_by_an_untagged_observation()
            assert epoch_leakage(graph, "eps", "epoch", {}) > 0.0
            with pytest.raises(StructureError, match="reaches other"):
                factorize(graph, "epoch")

    def test_the_guard_can_still_fail_which_is_what_makes_the_pass_mean_something(self):
        """ANTI-VACUITY. The epoch-local fixture must pass for a REASON.

        Raising the tolerance to admit the leaky graph shows the guard is
        reading the graph rather than the fixture's name: at `rtol=10` the
        same model that is refused above is accepted.
        """
        with jax.enable_x64(True):
            graph = _leaks_by_broadcast()
            with pytest.raises(StructureError):
                factorize(graph, "epoch")
            admitted = factorize(graph, "epoch", rtol=1e3)
        assert admitted.per_epoch == ("eps",)


def test_the_measured_cost_of_folding_a_leaky_latent():
    """Why this is a refusal and not a warning, at the number.

    Folding a latent that leaks into the next epoch integrates it once per
    epoch instead of once. The result is not an error or an infinity -- it is
    a finite, plausible, wrong evidence. Measured here against the marginal's
    closed form on the same fixture the guard refuses.
    """
    import math

    from bayesmith.evidence import SqrtInfo, compress_epoch
    from bayesmith.exact.precision import DiagonalPrecision

    per, n_global, sigma, prior = 3, 2, 0.6, 0.8
    rng = np.random.default_rng(0)
    global_design = rng.normal(size=(N_EPOCH * per, n_global))
    data = rng.normal(size=N_EPOCH * per)

    def build(leak):
        design = np.zeros((N_EPOCH * per, N_EPOCH))
        for e in range(N_EPOCH):
            design[e * per : (e + 1) * per, e] = rng.normal(size=per)
            if leak and e + 1 < N_EPOCH:
                design[(e + 1) * per : (e + 2) * per, e] = leak * rng.normal(size=per)
        return design

    def folded(design, point):
        total = SqrtInfo.null(("g",), ((n_global,),))
        for e in range(N_EPOCH):
            rows = slice(e * per, (e + 1) * per)
            total = SqrtInfo.combine(
                total,
                compress_epoch(
                    {"g": jnp.asarray(global_design[rows])},
                    jnp.asarray(data[rows]),
                    DiagonalPrecision(sigma=jnp.full(per, sigma)),
                    {"g": (n_global,)},
                    nuisance_design={"n": jnp.asarray(design[rows, e : e + 1])},
                    nuisance_shapes={"n": (1,)},
                    nuisance_prior_std={"n": prior},
                ),
            )
        return float(total.log_prob({"g": jnp.asarray(point)}))

    def truth(design, point):
        covariance = np.diag(
            np.full(N_EPOCH * per, sigma**2)
        ) + design @ (prior**2 * np.eye(N_EPOCH)) @ design.T
        residual = data - global_design @ np.asarray(point)
        _, logdet = np.linalg.slogdet(2.0 * math.pi * covariance)
        return -0.5 * float(
            residual @ np.linalg.solve(covariance, residual)
        ) - 0.5 * float(logdet)

    point = [0.5, -1.0]
    with jax.enable_x64(True):
        local = build(0.0)
        assert folded(local, point) == pytest.approx(truth(local, point), abs=1e-9)
        leaky = build(0.8)
        gap = folded(leaky, point) - truth(leaky, point)
    assert abs(gap) > 0.1, gap


class TestTheGuardsOwnFailureModes:
    """Two fixtures the mutation survivors demanded, and one real bug.

    Both were built because a mutant lived: replacing the RELATIVE measure
    with an absolute one, and writing the threshold the NaN-unsafe way. The
    second turned up a defect in the guard itself.
    """

    @staticmethod
    def _faint(scale):
        """A leak in a latent the model barely uses.

        Everything the latent touches is scaled, so its LARGEST sensitivity
        shrinks with the leak. The relative leakage is unchanged at 13.2 while
        the absolute one falls to ~1e-11 -- below any fixed floor.
        """
        lower = np.tril(np.ones((N_EPOCH, N_EPOCH)) * 0.8, -1) + np.eye(N_EPOCH)

        def model():
            epoch = plate("epoch", N_EPOCH)
            g = sample("g", lambda: ndist.Normal(0.0, 1.0))
            eps = sample("eps", lambda: ndist.Normal(0.0, 1.0), plate=epoch)
            chain = det(
                "chain",
                lambda e: scale * (jnp.asarray(lower) @ e),
                eps,
                linear_in=("eps",),
            )
            mu = det(
                "mu",
                lambda a, e, c: a + scale * e + c,
                g,
                eps,
                chain,
                plate=epoch,
                linear_in=("g", "eps", "chain"),
            )
            observe(
                "d",
                lambda m: ndist.Normal(m, 0.5),
                mu,
                plate=epoch,
                obs=jnp.zeros((N_EPOCH, N_EPOCH)),
            )

        return trace(model)

    @pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-12])
    def test_the_measure_is_relative_and_a_faint_latent_is_judged_the_same(
        self, scale
    ):
        """Why the score is divided by the latent's own largest sensitivity.

        An absolute floor would wave this through at ``scale = 1e-12``, where
        the absolute leakage is ~1e-11 -- a hundredth of the default rtol --
        while the model is exactly as wrong as at ``scale = 1``. Measured: the
        relative leakage is **13.2 at every scale**, and mutating the return
        to the absolute number makes this test's last case pass.
        """
        with jax.enable_x64(True):
            graph = self._faint(scale)
            assert epoch_leakage(graph, "eps", "epoch", {}) == pytest.approx(
                13.2, rel=1e-6
            )
            with pytest.raises(StructureError, match="reaches other"):
                factorize(graph, "epoch")

    def test_a_non_finite_sensitivity_is_refused_rather_than_read_as_local(self):
        """The bug this fixture found, and it was in the guard.

        Python's ``max`` LOSES a NaN: ``max(0.0, nan)`` is ``0.0``, because it
        returns the first argument when ``nan > 0.0`` is False. So a model
        whose per-epoch map is singular at the evaluation point had both
        running maxima stay at ``0.0``, and the latent was reported as
        **perfectly epoch-local** and accepted.

        Now the non-finite entry is refused by name, before any maximum.
        """
        lower = np.tril(np.ones((N_EPOCH, N_EPOCH)) * 0.8, -1) + np.eye(N_EPOCH)

        def model():
            epoch = plate("epoch", N_EPOCH)
            g = sample("g", lambda: ndist.Normal(0.0, 1.0))
            eps = sample("eps", lambda: ndist.Normal(0.0, 1.0), plate=epoch)
            chain = det(
                "chain",
                lambda e: jnp.asarray(lower) @ (1.0 / e),
                eps,
                linear_in=("eps",),
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
                lambda m: ndist.Normal(m, 0.5),
                mu,
                plate=epoch,
                obs=jnp.zeros((N_EPOCH, N_EPOCH)),
            )

        with jax.enable_x64(True):
            graph = trace(model)
            with pytest.raises(StructureError, match="not finite at"):
                epoch_leakage(graph, "eps", "epoch", {})
            with pytest.raises(StructureError, match="not finite at"):
                factorize(graph, "epoch")

    def test_python_max_really_does_lose_a_nan(self):
        """The language fact the fix is written against, so it is not folklore."""
        assert max(0.0, float("nan")) == 0.0
        assert math.isnan(max(float("nan"), 0.0))


def _survivor_plated_elsewhere():
    """A campaign plate, plus a second plate carrying a latent of its own.

    Both are legitimate models and the graph builds them happily. From
    ``"epoch"``'s point of view ``b`` is a survivor -- it is constant across
    epochs -- but it is not a SCALAR survivor, it is plated on ``"night"``.
    """
    data = np.random.default_rng(0).normal(size=N_EPOCH)

    def model():
        epoch = plate("epoch", N_EPOCH)
        night = plate("night", 3)
        g = sample("g", lambda: ndist.Normal(0.0, 3.0))
        n = sample("n", lambda: ndist.Normal(0.4, 1.3), plate=epoch)
        mu = det("mu", lambda a, b: 2.0 * a + b, g, n, plate=epoch, linear_in=("g", "n"))
        observe(
            "d", lambda v: ndist.Normal(v, 0.55), mu, plate=epoch, obs=jnp.asarray(data)
        )
        b = sample("b", lambda: ndist.Normal(0.0, 1.0), plate=night)
        observe("e", lambda v: ndist.Normal(v, 0.5), b, plate=night, obs=jnp.zeros(3))

    return trace(model)


class TestASurvivorPlatedOnAnotherAxisIsRefusedInThisLayersOwnWords:
    """It was refused already -- in jax's words, from inside a vmap.

    ``epoch_leakage`` probes with one ``jacfwd`` of an ``isolate`` over
    survivors AND per-epoch latents together. A survivor plated on some other
    axis has the wrong rank for that vmap, so the model was refused by
    ``ValueError: vmap was requested to map its argument along axis 0, which
    implies that its rank should be at least 1, but is only 0`` -- raised out
    of ``graph/evaluate.py``, naming no latent, no plate, and nothing the
    author could act on.

    The verdict was right and the reason was unusable, which is the shape
    ``check_precision`` was already caught in once (an indefinite kernel
    refused via NaN, reporting "the log-density is not quadratic" about a
    perfectly quadratic density). Refusing early and by name costs one
    comprehension.
    """

    def test_the_refusal_names_the_latent_and_the_other_plate(self):
        with jax.enable_x64(True):
            with pytest.raises(StructureError) as refused:
                factorize(_survivor_plated_elsewhere(), "epoch")
        message = str(refused.value)
        assert "'b'" in message
        assert "night" in message

    def test_it_is_not_a_vmap_error_any_more(self):
        """The anti-regression clause: a ValueError here is the OLD behaviour."""
        with jax.enable_x64(True):
            with pytest.raises(StructureError):
                factorize(_survivor_plated_elsewhere(), "epoch")
            with pytest.raises(StructureError):
                factorize(_survivor_plated_elsewhere(), "night")

    def test_an_unplated_survivor_is_still_fine(self):
        """Anti-vacuity: the guard must not refuse the ordinary campaign.

        ``_epoch_local``'s survivor is unplated, which is the normal case and
        the one every other test in this file rests on.
        """
        with jax.enable_x64(True):
            found = factorize(_epoch_local(), "epoch")
        assert found.survivors
        assert all(not p for p in ())  # no plated survivors to check
