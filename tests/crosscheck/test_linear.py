"""``linear.py`` against ``exact/{block,linearity,solve}``: the §四 4.1 row.

The row names four things that must agree, and each is asserted at the
strongest form the mathematics allows rather than at a tolerance chosen to
pass:

* the CG solution "逐元素到 float64 roundoff" -- measured **bitwise
  identical**, so that is what is asserted. A tolerance here would let a
  genuinely different iteration through, which is the one thing this file
  exists to notice;
* GCR's mean **and covariance** within MC error, against a dense oracle
  rather than against each other;
* ``condition_estimate`` "同键同数" -- and this is where the row does not
  survive contact: bayesmith has no ``condition_estimate``, deliberately,
  and rheplicant's own docstring for it contradicts its implementation. §5;
* the guards "全部守卫同形": the ``kappa*residual`` criterion, the separate
  ``kappa*eps`` verdict, the 1-D sigma axis ambiguity, and the NoiseModel
  refusal at the conjugate seam. Three of the four map; the fourth cannot be
  stated at this seam, and the reason is measured here rather than asserted.

**Everything is built inside ``with jax.enable_x64(True):``**, arrays
included. The context governs the OPERATION and not the array, so a fixture
built outside arrives as float32 into a float64 graph -- measured while
writing this file: it moved the two packages' ``condition_bound`` apart by
4.096e-09 and made ``check_linearity`` report unresolved departures. Both
went to bitwise agreement once the arrays moved inside.
"""

from __future__ import annotations

# Module scope, and load-bearing for the same reason
# ``test_diagnose_identifiability`` records: ``from __future__ import
# annotations`` stringifies the operator classes' body annotations, and
# dataclasses resolves ``ClassVar`` against THIS module's globals.
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crosscheck

N_TIME, N_FREQ = 8, 8
TONE_CHANNEL = 3
TONE_KELVIN = 5000.0

#: Flat enough that the prior does not resolve the bilinear degeneracy --
#: rheplicant's own ``tests/inference/test_degenerate_partition.py`` value,
#: reused rather than re-guessed, per §二 step 1.
PRIOR_STD = 1e6
NOISE_STD = 1.0

#: Draws for the moment comparison. The keys are FIXED
#: (``jax.random.key(s)`` for ``s`` in ``range(N)``), so every z-score below
#: is deterministic rather than sampled -- a failure is a change in the
#: arithmetic, never a bad roll. Measured at this count: every |z| < 3.
N_DRAWS = 256


def _arrays() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The shared ``gain x T_ant`` fixture. Call INSIDE the x64 block."""

    def poly_basis(n: int, degree: int) -> jax.Array:
        x = jnp.linspace(-1.0, 1.0, n)
        return jnp.stack([x**k for k in range(degree)], axis=1)

    time_basis = poly_basis(N_TIME, 3)
    freq_basis = poly_basis(N_FREQ, 3)
    coeff0 = jnp.array([[3000.0, -180.0, 40.0], [120.0, 25.0, -8.0], [-45.0, 6.0, 2.0]])
    gain0 = 1.5 + 0.05 * jnp.arange(N_TIME, dtype=float)
    t_ant0 = time_basis @ coeff0 @ freq_basis.T
    tone = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(TONE_KELVIN)
    data = gain0[:, None] * (t_ant0 + tone[None, :])
    return gain0, t_ant0, tone, data


def _rheplicant_model(gain0, t_ant0, tone):
    """``(space, pipeline, template, at)`` on rheplicant's own vocabulary.

    Split out so the solve comparison and the linearity comparison build the
    SAME model rather than two hand-copied ones -- a second spelling of a
    fixture is the copy that goes stale.
    """
    from rheplicant import Coordinates, State
    from rheplicant.core.operator import AbstractOperator
    from rheplicant.core.pipeline import Pipeline
    from rheplicant.inference import Bind, Latent, ParameterSpace
    from rheplicant.radio import GainOperator

    class AntennaTemperature(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
        provides: ClassVar[tuple[str, ...]] = ("data",)
        t_ant: jax.Array

        def __call__(self, state):
            return state.with_data(self.t_ant)

    class CalibrationTone(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("data",)
        provides: ClassVar[tuple[str, ...]] = ("data",)
        tone: jax.Array

        def __call__(self, state):
            return state.with_data(state.data + self.tone[None, :])

    pipeline = Pipeline(
        AntennaTemperature(t_ant=t_ant0),
        CalibrationTone(tone=tone),
        GainOperator(gain=gain0),
        names=("t_ant", "tone", "gain"),
    )
    # Both latents declared linear, which is the HONEST declaration: each
    # conditional genuinely is affine and the model is bilinear anyway.
    space = ParameterSpace(
        latents=[
            Latent("gain", init=gain0, linear=True),
            Latent("t_ant", init=t_ant0, linear=True),
        ],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind("t_ant", into=lambda p: p["t_ant"].t_ant),
        ],
    )
    template = State(
        data=jnp.zeros((N_TIME, N_FREQ)),
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 80e6, N_FREQ),
        ),
    )
    return space, pipeline, template, {"gain": gain0, "t_ant": t_ant0}


def _rheplicant_pieces(gain0, t_ant0, tone, block_name):
    """``(block, prior_mean, prior_std)``, through the CHECKING entry point."""
    from rheplicant.inference.linear import linear_operator

    space, pipeline, template, at = _rheplicant_model(gain0, t_ant0, tone)
    block = linear_operator(space, pipeline, template, names=(block_name,), at=at)
    return (
        block,
        {block_name: jnp.zeros_like(at[block_name])},
        {block_name: jnp.full_like(at[block_name], PRIOR_STD)},
    )


def _bayesmith_graph(gain0, t_ant0, tone, data, sigma: Any = NOISE_STD):
    """The same model as a graph.

    The linearity CLAIM lives in a different place in each package, and that
    is the semantic mapping rather than a translation detail: rheplicant
    declares ``linear=True`` on the LATENT, bayesmith declares
    ``linear_in=(...)`` on the DETERMINISTIC node. The graph form scopes the
    claim to the node that makes it, so a model with two predictions can be
    affine in a latent at one of them and not at the other and say so; the
    flat form has one place to put the word.
    """
    import numpyro.distributions as dist

    from bayesmith import det, observe, sample, trace

    def model():
        gain = sample(
            "gain", lambda: dist.Normal(jnp.zeros_like(gain0), PRIOR_STD).to_event(1)
        )
        t_ant = sample(
            "t_ant", lambda: dist.Normal(jnp.zeros_like(t_ant0), PRIOR_STD).to_event(2)
        )
        pred = det(
            "pred",
            lambda g, t: g[:, None] * (t + tone[None, :]),
            gain,
            t_ant,
            linear_in=("gain", "t_ant"),
        )
        observe("d", lambda mu: dist.Normal(mu, sigma).to_event(2), pred, obs=data)

    return trace(model)


def _bayesmith_pieces(gain0, t_ant0, tone, data, block_name):
    """``(block, precision)``, through the CHECKING entry point."""
    from bayesmith.exact.linearity import linear_operator
    from bayesmith.exact.precision import diagonal_from

    graph = _bayesmith_graph(gain0, t_ant0, tone, data)
    outside = {"gain": gain0, "t_ant": t_ant0}
    outside.pop(block_name)
    block = linear_operator(graph, [block_name], outside)
    return block, diagonal_from({"d": jnp.full_like(data, NOISE_STD)})


def _dense_oracle(gain0, t_ant0, tone, data, block_name):
    """``(x*, posterior covariance)`` by dense numpy linear algebra.

    Iron law 4: the two packages agreeing is not evidence. ``A`` and the
    OFFSET are read out of the full model at basis vectors rather than
    hand-written -- the first draft of this oracle wrote ``gain * x`` for the
    ``t_ant`` block, silently dropping the tone (which is the offset), and
    then disagreed with both packages by 64% while looking like a finding.
    """
    if block_name == "gain":
        size = N_TIME

        def mu(x):
            return np.ravel(np.asarray(x[:, None] * (t_ant0 + tone[None, :])))

    else:
        size = N_TIME * N_FREQ

        def mu(x):
            return np.ravel(
                np.asarray(gain0[:, None] * (x.reshape(N_TIME, N_FREQ) + tone[None, :]))
            )

    offset = mu(jnp.zeros(size))
    design = np.stack(
        [mu(jnp.zeros(size).at[i].set(1.0)) - offset for i in range(size)], axis=1
    )
    flat = np.ravel(np.asarray(data))
    inverse_noise = np.eye(flat.size) / NOISE_STD**2
    normal = design.T @ inverse_noise @ design + np.eye(size) / PRIOR_STD**2
    # The prior mean is zero on both sides, so S^-1 m contributes nothing.
    rhs = design.T @ inverse_noise @ (flat - offset)
    return np.linalg.solve(normal, rhs), np.linalg.inv(normal)


def _both_means(block_name):
    """``(rheplicant, bayesmith)`` posterior means and residuals, one fixture."""
    from rheplicant.inference.linear import wiener_solve as theirs

    from bayesmith.exact.solve import wiener_solve as ours

    gain0, t_ant0, tone, data = _arrays()
    block, prior_mean, prior_std = _rheplicant_pieces(gain0, t_ant0, tone, block_name)
    their_value, their_residual = theirs(
        block,
        data,
        noise_std=NOISE_STD,
        prior_mean=prior_mean,
        prior_std=prior_std,
        require_convergence=None,
    )
    our_block, precision = _bayesmith_pieces(gain0, t_ant0, tone, data, block_name)
    our_value, our_residual = ours(
        our_block, precision=precision, require_convergence=None
    )
    # Ravelled: the dense oracle works in the block's flat degrees of
    # freedom, and `t_ant` is (8, 8) on both sides.
    return (
        np.ravel(np.asarray(their_value[block_name])),
        float(their_residual),
        np.ravel(np.asarray(our_value[block_name])),
        float(our_residual),
    )


@pytest.mark.parametrize("block_name", ["gain", "t_ant"])
def test_the_posterior_mean_and_residual_are_bitwise_identical(block_name):
    """§四 4.1's "CG 解逐元素到 float64 roundoff", asserted one step stronger.

    Measured as EQUALITY, not as a tolerance: same operator, same
    right-hand side, same ``jax.scipy.sparse.linalg.cg``, so the two walk
    the same arithmetic in the same order and any difference at all is a
    difference in the mathematics. Writing ``rel=1e-12`` here would admit a
    genuinely different iteration -- and it would have admitted the float32
    fixture bug this module's docstring records, which showed up at 4e-09.
    """
    with jax.enable_x64(True):
        theirs, their_residual, ours, our_residual = _both_means(block_name)
    assert np.array_equal(theirs, ours), np.max(np.abs(theirs - ours))
    assert their_residual == our_residual


@pytest.mark.parametrize("block_name", ["gain", "t_ant"])
def test_both_agree_with_a_dense_oracle_to_the_solver_tolerance(block_name):
    """Iron law 4. Agreeing on a wrong number is still agreement.

    The comparison is to CG's own ``tol`` (1e-6 by default) rather than to
    roundoff, because that is what the solver promises: the residual is
    bounded, the error is not. Measured: 1.24e-06 on ``gain``, 1.50e-07 on
    ``t_ant``.
    """
    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        truth, _ = _dense_oracle(gain0, t_ant0, tone, data, block_name)
        theirs, _, ours, _ = _both_means(block_name)
    for label, got in (("rheplicant", theirs), ("bayesmith", ours)):
        relative = np.max(np.abs(got - truth) / np.maximum(np.abs(truth), 1e-30))
        assert relative < 2e-6, (label, relative)


@pytest.mark.parametrize("block_name", ["gain", "t_ant"])
def test_the_condition_bound_is_bitwise_identical(block_name):
    """``lambda_max * max(prior_variance)``, same power iteration, same key.

    The number a caller divides an accuracy target by, so a difference here
    is a difference in every ``tol`` chosen downstream.
    """
    from rheplicant.inference.linear import condition_bound as theirs

    from bayesmith.exact.solve import condition_bound as ours

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        block, _, prior_std = _rheplicant_pieces(gain0, t_ant0, tone, block_name)
        their_kappa = float(theirs(block, noise_std=NOISE_STD, prior_std=prior_std))
        our_block, precision = _bayesmith_pieces(gain0, t_ant0, tone, data, block_name)
        our_kappa = float(ours(our_block, precision=precision))
    assert their_kappa == our_kappa


def _draws(block_name, count):
    """``(rheplicant, bayesmith)`` GCR draws at the same fixed keys."""
    from rheplicant.inference.linear import gcr_sample as theirs

    from bayesmith.exact.solve import gcr_sample as ours

    gain0, t_ant0, tone, data = _arrays()
    block, prior_mean, prior_std = _rheplicant_pieces(gain0, t_ant0, tone, block_name)
    our_block, precision = _bayesmith_pieces(gain0, t_ant0, tone, data, block_name)
    mine, yours = [], []
    for seed in range(count):
        key = jax.random.key(seed)
        drawn, _ = theirs(
            block,
            data,
            noise_std=NOISE_STD,
            key=key,
            prior_mean=prior_mean,
            prior_std=prior_std,
            require_convergence=None,
        )
        mine.append(np.ravel(np.asarray(drawn[block_name])))
        drawn, _ = ours(
            our_block, precision=precision, key=key, require_convergence=None
        )
        yours.append(np.ravel(np.asarray(drawn[block_name])))
    return np.stack(mine), np.stack(yours)


@pytest.mark.slow
def test_the_gcr_draws_reproduce_the_oracle_mean_and_covariance():
    """§四 4.1's "GCR 的均值**与协方差**在 MC 误差内".

    Against the DENSE ORACLE, not against each other: a constrained
    realization is exact by construction, so the claim worth testing is that
    each package's draws have the posterior's moments -- and two ports of the
    same identity would agree with each other while both getting the
    fluctuation scaling wrong. The covariance half is the half that catches
    that: this package's own mutation testing recorded a prior-term division
    written the wrong way round, which leaves the MEAN untouched and widens
    the drawn covariance 4x.

    The draws are NOT bitwise identical across packages and cannot be -- each
    splits the key over its own pytree -- so this is the one comparison in
    this file stated in MC error rather than in roundoff. Thresholds are the
    design document's |z| < 4; measured at 256 fixed keys, every |z| < 3.
    """
    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        truth, covariance = _dense_oracle(gain0, t_ant0, tone, data, "gain")
        theirs, ours = _draws("gain", N_DRAWS)
    assert not np.array_equal(theirs, ours), (
        "the draws came out bitwise identical, which means the two packages "
        "are sharing a key stream -- this comparison would then be vacuous"
    )
    variance = np.diag(covariance)
    for label, drawn in (("rheplicant", theirs), ("bayesmith", ours)):
        z_mean = np.max(
            np.abs(drawn.mean(axis=0) - truth) / np.sqrt(variance / N_DRAWS)
        )
        assert z_mean < 4.0, (label, "mean", z_mean)
        sample = np.cov(drawn, rowvar=False)
        # SE of a sample variance is sigma^2 sqrt(2/(n-1)); of an off-diagonal
        # entry, sqrt((c_ii c_jj + c_ij^2)/n).
        z_var = np.max(
            np.abs(np.diag(sample) - variance)
            / (variance * np.sqrt(2.0 / (N_DRAWS - 1)))
        )
        assert z_var < 4.0, (label, "variance", z_var)
        se_full = np.sqrt((np.outer(variance, variance) + covariance**2) / N_DRAWS)
        z_full = np.max(np.abs(sample - covariance) / se_full)
        assert z_full < 4.0, (label, "covariance", z_full)


def test_check_linearity_accepts_the_same_blocks_on_both_sides():
    """Each conditional IS affine; the pair is not. Same verdict, both ways.

    The joint refusal is the one that matters: a group holding both latents
    is exactly what an alternating solve silently pretends it has, and it is
    the reason a bilinear model needs more than one block.
    """
    from rheplicant.core.errors import ParameterSpaceError
    from rheplicant.inference.linear import check_linearity as theirs

    from bayesmith.errors import StructureError
    from bayesmith.exact.linearity import check_linearity as ours

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        graph = _bayesmith_graph(gain0, t_ant0, tone, data)
        space, pipeline, template, at = _rheplicant_model(gain0, t_ant0, tone)
        for name in ("gain", "t_ant"):
            outside = {"gain": gain0, "t_ant": t_ant0}
            outside.pop(name)
            ours(graph, [name], outside)
            theirs(space, pipeline, template, names=(name,), at=at)
        with pytest.raises(StructureError):
            ours(graph, ["gain", "t_ant"], {})
        with pytest.raises(ParameterSpaceError):
            theirs(space, pipeline, template, names=("gain", "t_ant"), at=at)


# --------------------------------------------------------------------------
# §二 step 3: refusal agreement. Three of the row's four guards map; the
# fourth cannot be stated at this seam, and that is measured below rather
# than asserted.
# --------------------------------------------------------------------------


def _refusal(fn) -> str:
    """Run ``fn`` and return the refusal text, or ``''`` if it returned."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- the class is what is under test
        return f"{type(exc).__name__}: {exc}"
    return ""


@pytest.mark.parametrize(
    ("block_name", "target", "phrase"),
    [
        # kappa = 1.27e+20 here, so kappa*eps = 2.8e+04 is already above any
        # target: the verdict is "no tol will help".
        ("gain", 1e-3, "cannot reach"),
        # kappa = 3.37e+12, kappa*eps = 7.5e-04 < 1e-3, so the arithmetic CAN
        # represent the answer and the verdict is the other one.
        ("t_ant", 1e-3, "did not converge"),
    ],
)
def test_the_two_convergence_verdicts_are_the_same_verdict_on_both_sides(
    block_name, target, phrase
):
    """§四 4.1's ``kappa*residual`` criterion AND its separate ``kappa*eps``
    branch -- the "tightening tol is useless" judgement, which is a different
    sentence because it has a different remedy.

    Asserted on which BRANCH fires rather than on the wording: the two
    packages phrase the same verdict differently (``"the normal operator's
    condition number"`` against ``"the condition bound"``), and pinning
    wording would pin prose. What has to agree is the decision.
    """
    from rheplicant.inference.linear import wiener_solve as theirs

    from bayesmith.exact.solve import wiener_solve as ours

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        block, prior_mean, prior_std = _rheplicant_pieces(
            gain0, t_ant0, tone, block_name
        )
        our_block, precision = _bayesmith_pieces(gain0, t_ant0, tone, data, block_name)
        their_text = _refusal(
            lambda: theirs(
                block,
                data,
                noise_std=NOISE_STD,
                prior_mean=prior_mean,
                prior_std=prior_std,
                require_convergence=target,
            )
        )
        our_text = _refusal(
            lambda: ours(our_block, precision=precision, require_convergence=target)
        )
    assert phrase in their_text, their_text[:400]
    assert phrase in our_text, our_text[:400]


@pytest.mark.parametrize("target", [1e5, 1e6])
def test_both_flip_from_refusing_to_accepting_at_the_same_target(target):
    """The criterion itself, not just its two branches.

    ``t_ant``'s error bound is between 1e5 and 1e6, so a guard reading a
    DIFFERENT kappa or a different residual would flip somewhere else. Both
    packages refuse at 1e5 and accept at 1e6.
    """
    from rheplicant.inference.linear import wiener_solve as theirs

    from bayesmith.exact.solve import wiener_solve as ours

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        block, prior_mean, prior_std = _rheplicant_pieces(gain0, t_ant0, tone, "t_ant")
        our_block, precision = _bayesmith_pieces(gain0, t_ant0, tone, data, "t_ant")
        their_text = _refusal(
            lambda: theirs(
                block,
                data,
                noise_std=NOISE_STD,
                prior_mean=prior_mean,
                prior_std=prior_std,
                require_convergence=target,
            )
        )
        our_text = _refusal(
            lambda: ours(our_block, precision=precision, require_convergence=target)
        )
    refused = target == 1e5
    assert bool(their_text) is refused, their_text[:300]
    assert bool(our_text) is refused, our_text[:300]


@pytest.mark.parametrize("depends_on_prediction", [False, True])
def test_a_rule_for_sigma_is_refused_at_the_conjugate_seam_on_both_sides(
    depends_on_prediction,
):
    """A conjugate solve needs a DECIDED covariance, not a rule for one.

    rheplicant refuses a ``NoiseModel`` by name, with a longer sentence when
    the model depends on the prediction, because that case is not a packaging
    mistake: the solve has no prediction to evaluate the rule at, the
    prediction being what it solves for.

    bayesmith's ``precision=`` takes a
    :class:`~bayesmith.exact.precision.Precision` -- an OPERATOR -- so a rule
    is refused by the protocol rather than by a keyword check. Both refuse;
    what is compared is that neither silently freezes sigma somewhere.
    """
    from rheplicant.inference.linear import wiener_solve as theirs
    from rheplicant.inference.noise import HomoscedasticNoise, RadiometerNoise

    from bayesmith.exact.solve import wiener_solve as ours

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        block, prior_mean, prior_std = _rheplicant_pieces(gain0, t_ant0, tone, "gain")
        model = (
            RadiometerNoise(channel_width=1e6, integration_time=1.0, floor=1.0)
            if depends_on_prediction
            else HomoscedasticNoise(sigma=NOISE_STD)
        )
        their_text = _refusal(
            lambda: theirs(
                block,
                data,
                noise_std=model,
                prior_mean=prior_mean,
                prior_std=prior_std,
                require_convergence=None,
            )
        )
        our_block, _ = _bayesmith_pieces(gain0, t_ant0, tone, data, "gain")
        our_text = _refusal(
            lambda: ours(
                our_block,
                precision={"d": (lambda mu: 0.05 * mu)},
                require_convergence=None,
            )
        )
    assert "ParameterSpaceError" in their_text, their_text[:300]
    if depends_on_prediction:
        assert "has no prediction to evaluate it at" in their_text
    else:
        assert "takes a plain sigma array" in their_text
    assert our_text, "bayesmith accepted a callable where a Precision is required"


def test_the_ambiguous_1d_sigma_is_resolved_before_bayesmith_can_see_it():
    """The one guard in this row that does NOT map, with the reason measured.

    rheplicant refuses a 1-D ``noise_std`` whose axis the prediction cannot
    settle: against an ``(8, 8)`` grid a length-8 vector reads equally well
    as one sigma per time sample and as one per frequency channel, NumPy
    picks the trailing axis, and every downstream number is finite, correctly
    shaped, and answers a question nobody asked.

    **bayesmith cannot state that refusal at this seam**, and the reason is
    numpyro rather than an omission: ``dist.Normal(loc, scale)`` runs
    ``promote_shapes`` in its own constructor, INSIDE the user's ``dist_fn``,
    so by the time anything in this package reads ``distribution.scale`` a
    bare ``(8,)`` has already become ``(1, 8)`` -- indistinguishable from an
    explicit, unambiguous ``(1, 8)``. A guard written here would either miss
    the ambiguous case or refuse the honest one; the information is gone.

    This test is what makes that a RECORD rather than a claim. If numpyro
    ever stops promoting, the first assertion fails and the decision should
    be revisited -- which is the moment the guard becomes writable.
    """
    import numpyro.distributions as dist

    from bayesmith.exact.gaussian import noise_std_at

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        ambiguous = jnp.linspace(0.01, 1.0, N_TIME)
        # 1. The declarations are already identical at the distribution.
        assert jnp.shape(dist.Normal(data, ambiguous).scale) == (1, N_FREQ)
        assert jnp.shape(dist.Normal(data, ambiguous[None, :]).scale) == (1, N_FREQ)
        assert jnp.shape(dist.Normal(data, ambiguous[:, None]).scale) == (N_TIME, 1)

        # 2. So the graph reads the bare vector as PER-FREQUENCY, silently.
        at = {"gain": gain0, "t_ant": t_ant0}
        bare = noise_std_at(_bayesmith_graph(gain0, t_ant0, tone, data, ambiguous), at)
        per_freq = noise_std_at(
            _bayesmith_graph(gain0, t_ant0, tone, data, ambiguous[None, :]), at
        )
        per_time = noise_std_at(
            _bayesmith_graph(gain0, t_ant0, tone, data, ambiguous[:, None]), at
        )
        assert np.array_equal(np.asarray(bare["d"]), np.asarray(per_freq["d"]))
        assert not np.array_equal(np.asarray(bare["d"]), np.asarray(per_time["d"]))

        # 3. And rheplicant still refuses the same declaration, loudly.
        block, prior_mean, prior_std = _rheplicant_pieces(gain0, t_ant0, tone, "gain")
        from rheplicant.inference.linear import wiener_solve as theirs

        text = _refusal(
            lambda: theirs(
                block,
                data,
                noise_std=ambiguous,
                prior_mean=prior_mean,
                prior_std=prior_std,
                require_convergence=None,
            )
        )
    assert "StateValidationError" in text, text[:300]
    assert "more than one legitimate reading" in text


def test_the_silent_axis_choice_is_over_confident_and_by_how_much():
    """Iron law 5: an unfixed difference is written signed and sized.

    Posterior standard deviations of the eight ``gain`` samples, from the
    dense oracle, under the two readings of the same length-8 vector
    ``linspace(0.01, 1.0, 8)``:

    ======================================  ==========  ==========  =======
    reading                                 min         max         spread
    ======================================  ==========  ==========  =======
    per-time ``(8, 1)`` -- what was meant   9.165e-07   8.690e-05   94.8x
    per-freq ``(1, 8)`` -- what is taken    3.055e-06   3.228e-06    1.1x
    ======================================  ==========  ==========  =======

    The sign is the dangerous one. The sample the data constrains WORST
    comes back **26.9x narrower** than it is, and the ~95x structure the
    sigma vector describes is averaged away without a word: an over-confident
    error bar reads exactly like a well-measured parameter.

    Bounds rather than equalities on the two spreads, so the claim survives
    a change in the fixture's numbers while still failing if the effect
    reverses or disappears.
    """
    with jax.enable_x64(True):
        _, t_ant0, tone, _ = _arrays()
        sigma = np.linspace(0.01, 1.0, N_TIME)

        def mu(x):
            return np.ravel(np.asarray(x[:, None] * (t_ant0 + tone[None, :])))

        offset = mu(jnp.zeros(N_TIME))
        design = np.stack(
            [mu(jnp.zeros(N_TIME).at[i].set(1.0)) - offset for i in range(N_TIME)],
            axis=1,
        )
        spreads = {}
        for label, grid in (
            ("per_time", np.broadcast_to(sigma[:, None], (N_TIME, N_FREQ))),
            ("per_freq", np.broadcast_to(sigma[None, :], (N_TIME, N_FREQ))),
        ):
            inverse_noise = np.diag(1.0 / np.ravel(grid) ** 2)
            normal = design.T @ inverse_noise @ design + np.eye(N_TIME) / PRIOR_STD**2
            spreads[label] = np.sqrt(np.diag(np.linalg.inv(normal)))

    assert spreads["per_time"].max() / spreads["per_time"].min() > 50.0
    assert spreads["per_freq"].max() / spreads["per_freq"].min() < 2.0
    # The direction: what is taken is NARROWER than what was meant, on the
    # worst-constrained sample. An assertion on the sign, with a size.
    assert spreads["per_time"].max() / spreads["per_freq"].max() > 20.0


def test_bayesmith_does_not_carry_condition_estimate():
    """§四 4.1 asks for ``condition_estimate`` "同键同数". There is no such
    function here, deliberately, and this goes red if one is ported.

    The argument is already recorded in ``docs/migration/conditioning.md``:
    rheplicant estimates ``lambda_min`` by a second power iteration on
    ``lambda_max*I - M``, which on a graded spectrum cannot separate the
    eigenvalues crowded against ``lambda_max``, and errs ONE-SIDEDLY toward
    danger -- kappa too small, so a guard built on it is silent exactly when
    it should fire. bayesmith bounds ``lambda_min`` by the prior's own
    curvature instead, which is an UPPER bound on kappa: the direction a
    safety guard needs.
    """
    from bayesmith.exact import solve

    assert not hasattr(solve, "condition_estimate"), (
        "condition_estimate was ported after all; read "
        "docs/migration/conditioning.md before keeping it"
    )


@pytest.mark.parametrize(("block_name", "ceiling"), [("gain", 1e-20), ("t_ant", 1e-12)])
def test_rheplicants_condition_estimate_is_orders_below_its_own_bound(
    block_name, ceiling
):
    """What the estimate actually is, on this fixture, in both blocks.

    ``condition_estimate / condition_bound``: **8.38e-21** on ``gain``,
    **4.43e-13** on ``t_ant``. Bounded rather than pinned, so the claim
    survives an iteration-count retune while still failing if the estimate
    ever becomes the bound.
    """
    from rheplicant.inference.linear import condition_bound, condition_estimate

    with jax.enable_x64(True):
        gain0, t_ant0, tone, _ = _arrays()
        block, _, prior_std = _rheplicant_pieces(gain0, t_ant0, tone, block_name)
        measured = float(
            condition_estimate(block, noise_std=NOISE_STD, prior_std=prior_std)
        )
        bound = float(condition_bound(block, noise_std=NOISE_STD, prior_std=prior_std))
    assert measured / bound < ceiling, (measured, bound)


def test_rheplicants_condition_estimate_no_longer_claims_to_be_the_bound():
    """A finding this row turned up, and the guard that keeps it fixed.

    Measured 2026-08-25, before e-RHINO's ``0c49cae``: the public
    ``condition_estimate`` opened with "An upper bound on the conditioning"
    and stated "The number here is now ``λ_max · max(prior_variance)``" --
    which is what ``condition_bound`` returns. Its implementation called
    ``_condition_estimate``, whose own docstring says "MEASURED κ. A
    diagnostic, and **not a bound** ... Biased low, and therefore **unsafe to
    guard on**". A rename had left the bound paragraph in the wrong function.

    The consequence was one-sided and toward danger: a caller reading it was
    told this is the number to divide an accuracy target by, and on this
    fixture it is 1.2e+20 times too small, so the ``tol`` computed from it is
    too loose by that factor.

    Fixed upstream in e-RHINO's ``0c49cae`` after this row reported it. This
    test is what keeps it fixed: it asserts the docstring now names itself
    MEASURED and warns the reader off, so a future rename cannot quietly put
    the old sentence back.

    **Both sentences are on e-RHINO's ``track-a-tail`` and NOT on its
    ``main``**, measured 2026-08-25 -- the same branch dependency
    ``test_dispatch.py``'s
    ``test_rheplicants_plan_now_attributes_b1_to_the_block_type`` carries,
    and these two are the only guards in this directory that have it. A red
    here means, in decreasing order of likelihood: e-RHINO is checked out on
    ``main``, the branch was dropped in review, or the docstring changed.
    Only the last two are defects, and both make ``linear.md`` §5(a) false.

    **No NUMERIC cross-check depends on the checkout.** Measured
    independently rather than taken on report: ``main...track-a-tail``
    touches exactly two files under ``src/rheplicant/inference/`` --
    ``linear.py`` and ``plan.py`` -- and both are docstring-only, verified
    by comparing ``ast.dump`` with every ``Module``/``ClassDef``/
    ``FunctionDef`` docstring removed. Re-run that comparison rather than
    trusting this sentence; it is a claim about a branch, and branches move.
    """
    import inspect

    from rheplicant.inference.linear import condition_estimate

    text = inspect.getdoc(condition_estimate) or ""
    branch_note = (
        " If e-RHINO is checked out on `main`, this is expected -- the "
        "corrected docstring is on `track-a-tail` and unmerged as of "
        "2026-08-25. Otherwise the branch was dropped or the docstring "
        "changed, and docs/migration/linear.md section 5(a) needs updating."
    )
    assert text.startswith("The MEASURED conditioning"), text[:120] + branch_note
    assert "Do not divide an accuracy target by this number" in text, branch_note


def test_bayesmith_checks_linearity_at_more_at_points_and_says_when_it_cannot():
    """An intended difference, in bayesmith's favour, and a live one.

    rheplicant's ``check_linearity`` probes at ONE at-point -- the ``at``
    handed to it, defaulting to the declared inits -- and sweeps probe
    magnitudes there. bayesmith additionally sweeps ``at_points``: ``at``
    plus draws from the graph's own prior, because a single at-point is
    exactly the moderate-parameter probe ``boundary-validation`` exists to
    prevent.

    On THIS fixture the extra at-points cannot be resolved, and bayesmith
    says so rather than reporting a measured zero: ``PRIOR_STD`` is 1e6 by
    design (flat enough not to resolve the bilinear degeneracy), so a prior
    draw of ``gain`` is ~1e6, the prediction reaches ~1e9, and the departure
    from affinity falls under the per-element roundoff floor even in float64.
    The warning's own advice -- open ``jax.enable_x64`` -- does not apply
    here, because this call already is inside one.

    Asserted as an `Unresolved` in the returned table rather than on the
    warning text: the value is what the API hands a caller, and a caller that
    printed one as a measured zero would be reporting a floor as evidence.
    """
    from bayesmith.exact.linearity import Unresolved, check_linearity

    with jax.enable_x64(True):
        gain0, t_ant0, tone, data = _arrays()
        graph = _bayesmith_graph(gain0, t_ant0, tone, data)
        with pytest.warns(UserWarning, match="only partly evaluated"):
            table = check_linearity(graph, ["t_ant"], {"gain": gain0})
    assert len(table) > 1, ("one at-point only; the difference has gone", table)
    assert any(
        isinstance(value, Unresolved)
        for row in table.values()
        for value in row.values()
    ), table
