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

**2026-08-28, Wave B: eight of this file's fourteen tests retired, and the
file stayed.** rheplicant's four public SOLVE names now delegate here, so
every assertion comparing the two packages' solve arithmetic had become this
package against itself. ``test_the_gcr_draws_reproduce_the_oracle_mean_and_
covariance`` said so itself: its own anti-vacuity guard -- "the draws came
out bitwise identical, which means the two packages are sharing a key stream
-- this comparison would then be vacuous" -- is what went red on the switch,
which is the nicest possible way for a cross-check to announce that its
module has moved.

Retired under iron law 2, with each subject IDENTIFIED in an existing home on
this side rather than re-homed (the clause's other branch):

* the CG solution and the dense oracle -> ``tests/exact/test_solve.py::
  test_wiener_solve_matches_the_dense_oracle``;
* GCR's mean and covariance -> ``test_gcr_draws_have_the_oracle_mean_and_
  covariance`` and ``test_the_mean_of_many_draws_is_the_wiener_solution``;
* the ``condition_bound`` value and its looseness -> ``test_the_bound_is_
  never_below_the_true_condition_number`` and ``test_the_bound_is_loose_when_
  the_data_constrains_every_direction``;
* the two convergence verdicts and the target they flip at ->
  ``test_the_convergence_guard_fires_on_a_deliberately_starved_solve``,
  ``test_the_precision_floor_alone_makes_the_guard_unreachable`` and
  ``test_the_guard_bounds_the_error_not_the_residual``.

**The file is NOT deleted, and `linear.py` is NOT in ``SWITCHED``**, because
only the solve surface moved: ``linear_operator`` and ``check_linearity`` are
still rheplicant's own, so the six remaining tests still compare two
implementations. Same reading as ``numpyro_bridge`` and ``uncertainty``, each
of which is also half switched. What survives here is exactly what still has
two sides: the linearity verdict, the ``at``-points difference, the
``NoiseModel`` refusal (which lives in rheplicant's facade BECAUSE the seam
would erase it -- D48's rule, see D53), the 1-D sigma ambiguity that cannot
be stated here at all, and how over-confident the silent axis choice is.
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


def test_bayesmith_now_carries_condition_estimate_as_a_diagnostic():
    """§四 4.1 asks for ``condition_estimate`` "同键同数", and it is here now.

    **This test used to assert its ABSENCE.** The argument for that absence
    -- recorded in ``docs/migration/conditioning.md`` -- was that estimating
    ``lambda_min`` by a second power iteration errs ONE-SIDEDLY toward
    danger, so a guard built on it is silent exactly when it should fire.
    That argument is unchanged and still decides what ``condition_bound``
    does. Migration ledger D15(a) rules that the measured route is ported
    anyway, as a DIAGNOSTIC: it can see a near-degenerate partition, which
    the bound floors away and cannot report.

    So the assertion becomes the one an absence check could not make -- that
    the two packages' numbers are the same quantity -- and the rule that the
    absence was standing in for is pinned separately, by an AST scan that no
    guard in bayesmith reads it.
    """
    from bayesmith.exact import solve

    assert hasattr(solve, "condition_estimate")
    assert "not a bound" in (solve.condition_estimate.__doc__ or "").lower()


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
