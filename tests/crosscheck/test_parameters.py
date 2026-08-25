"""``parameters.py`` against node declarations: the §四 4.2 row.

The row asks for a **semantic mapping, not a line-by-line port**, and names
a minimum set: the three binding forms giving the same prediction on one toy
pipeline, and an equivalent of ``refuse_stochastic_stages`` -- *"理由改写,
行为不得变"*, the reason rewritten and the behaviour unchanged.

Both halves are here, and the second one found a real gap: bayesmith had no
such refusal at all. A ``det`` node consuming a PRNG key ran happily, which
is the one defect rheplicant's guard exists for and the one nothing
downstream can see.

The third thing this file records is what the graph form makes
**unexpressible**. rheplicant's ``fan=`` exists because a Python container
type -- ``v`` against ``list(v)``, the same data -- selects between tying
one value to every leaf and splitting it element-wise across them, and the
difference is invisible in every value, every shape, and to both structural
diagnostics. On the graph each consumer is a named edge, so the two intents
are two different graphs and there is nothing left to infer.
"""

from __future__ import annotations

import warnings

# Module scope, load-bearing: see ``test_diagnose_identifiability``.
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

pytestmark = pytest.mark.crosscheck

N_TIME, N_FREQ = 4, 5

#: The 2-vector rheplicant's own ``TestFanOut`` uses, and for its reason:
#: a symmetric fixture makes both readings agree and blinds the comparison.
FAN_VALUES = (2.0, 5.0)


def _basis() -> jax.Array:
    """A quadratic basis over the band. Call INSIDE the x64 block."""
    return jnp.stack([jnp.linspace(-1.0, 1.0, N_FREQ) ** k for k in range(3)], axis=1)


COEFF = (2.0, 0.5, -0.25)


# --------------------------------------------------------------------------
# The three binding forms, one toy pipeline, one prediction each.
# --------------------------------------------------------------------------


def _rheplicant_pipeline():
    from rheplicant import Coordinates, State
    from rheplicant.core.operator import AbstractOperator
    from rheplicant.core.pipeline import Pipeline

    class Sky(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
        provides: ClassVar[tuple[str, ...]] = ("data",)
        spectrum: jax.Array

        def __call__(self, state):
            return state.with_data(
                jnp.broadcast_to(self.spectrum, (N_TIME, N_FREQ))
            )

    class Scale(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("data",)
        provides: ClassVar[tuple[str, ...]] = ("data",)
        factor: jax.Array

        def __call__(self, state):
            return state.with_data(state.data * self.factor)

    pipeline = Pipeline(
        Sky(spectrum=jnp.ones(N_FREQ)),
        Scale(factor=jnp.array(1.0)),
        Scale(factor=jnp.array(1.0)),
        names=("sky", "first", "second"),
    )
    template = State(
        data=jnp.zeros((N_TIME, N_FREQ)),
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 80e6, N_FREQ),
        ),
    )
    return pipeline, template


def _rheplicant_prediction(form: str) -> np.ndarray:
    from rheplicant.inference import Bind, Latent, ParameterSpace

    pipeline, template = _rheplicant_pipeline()
    basis, coeff = _basis(), jnp.array(COEFF)
    if form == "direct":
        space = ParameterSpace(
            latents=[Latent("spectrum", init=jnp.ones(N_FREQ))],
            bindings=[Bind("spectrum", into=lambda p: p["sky"].spectrum)],
        )
        values = {"spectrum": basis @ coeff}
    elif form == "derived":
        space = ParameterSpace(
            latents=[Latent("coeff", init=jnp.zeros(3))],
            bindings=[
                Bind(
                    "coeff",
                    into=lambda p: p["sky"].spectrum,
                    fn=lambda c: basis @ c,
                )
            ],
        )
        values = {"coeff": coeff}
    else:
        space = ParameterSpace(
            latents=[Latent("gain", init=jnp.array(1.0))],
            bindings=[
                Bind(
                    "gain",
                    into=(
                        lambda p: p["first"].factor,
                        lambda p: p["second"].factor,
                    ),
                    fan="broadcast",
                )
            ],
        )
        values = {"gain": jnp.array(3.0)}
    forward, _ = space.forward_fn(pipeline, template)
    return np.asarray(forward(values))


def _bayesmith_prediction(form: str) -> np.ndarray:
    from bayesmith import const, det, evaluate, observe, sample, trace

    basis, coeff = _basis(), jnp.array(COEFF)
    zeros = jnp.zeros((N_TIME, N_FREQ))

    if form == "direct":

        def model():
            s = sample(
                "spectrum", lambda: dist.Normal(jnp.ones(N_FREQ), 1.0).to_event(1)
            )
            p = det(
                "pred",
                lambda v: jnp.broadcast_to(v, (N_TIME, N_FREQ)),
                s,
                linear_in=("spectrum",),
            )
            observe("d", lambda m: dist.Normal(m, 1.0).to_event(2), p, obs=zeros)

        values = {"spectrum": basis @ coeff}
    elif form == "derived":

        def model():
            c = sample("coeff", lambda: dist.Normal(jnp.zeros(3), 1.0).to_event(1))
            sp = det("spectrum", lambda v: basis @ v, c, linear_in=("coeff",))
            p = det(
                "pred",
                lambda v: jnp.broadcast_to(v, (N_TIME, N_FREQ)),
                sp,
                linear_in=("spectrum",),
            )
            observe("d", lambda m: dist.Normal(m, 1.0).to_event(2), p, obs=zeros)

        values = {"coeff": coeff}
    else:

        def model():
            ones = const("ones", jnp.ones(N_FREQ))
            gain = sample("gain", lambda: dist.Normal(1.0, 1.0))
            # ONE latent, TWO consuming stages -- two named edges, not one
            # produced value whose container type decides the fan.
            first = det("first", lambda g, v: g * v, gain, ones, linear_in=("gain",))
            second = det("second", lambda g, v: g * v, gain, first)
            p = det("pred", lambda v: jnp.broadcast_to(v, (N_TIME, N_FREQ)), second)
            observe("d", lambda m: dist.Normal(m, 1.0).to_event(2), p, obs=zeros)

        values = {"gain": jnp.array(3.0)}
    return np.asarray(evaluate(trace(model), values)["pred"])


@pytest.mark.parametrize("form", ["direct", "derived", "tied"])
def test_the_three_binding_forms_predict_the_same_thing(form):
    """§四 4.2's minimum set. Measured **bitwise identical** on all three.

    ``direct`` is identity into one leaf; ``derived`` puts a basis
    expansion between latent and leaf; ``tied`` drives two stages from one
    scalar, so the prediction carries it squared -- 3 becomes 9, which is
    what makes the tie observable rather than a relabelling.
    """
    with jax.enable_x64(True):
        theirs = _rheplicant_prediction(form)
        ours = _bayesmith_prediction(form)
    assert np.array_equal(theirs, ours), (
        form,
        float(np.max(np.abs(theirs - ours))),
    )


def test_the_tie_is_observable_and_not_a_relabelling():
    """Anti-vacuity for the row above.

    If ``tied`` merely renamed one multiply, the comparison would pass on a
    model where the second stage did nothing. Measured: the gain enters
    SQUARED, so 3.0 gives 9.0.
    """
    with jax.enable_x64(True):
        ours = _bayesmith_prediction("tied")
    assert ours[0, 0] == pytest.approx(9.0, rel=1e-12), ours[0, 0]


# --------------------------------------------------------------------------
# The FAN question, and why the graph cannot ask it.
# --------------------------------------------------------------------------


def _fan_prediction(fn, fan=None) -> float:
    """rheplicant's own ``TestFanOut`` fixture, reused rather than reinvented.

    ``t_physical = 0`` makes the efficiency a pure multiply, so the two
    leaves enter the prediction as a bare product. Driven through ``bind``
    rather than ``forward_fn`` because ``forward_fn`` validates first, and
    the broadcast reading writes a ``(2,)`` into a scalar leaf.
    """
    from rheplicant import Coordinates, State
    from rheplicant.core.pipeline import Pipeline
    from rheplicant.inference import Bind, Latent, ParameterSpace
    from rheplicant.radio import AntennaLossOperator, GainOperator

    values = jnp.array(FAN_VALUES)
    pipeline = Pipeline(
        AntennaLossOperator(efficiency=jnp.array(1.0), t_physical=jnp.array(0.0)),
        GainOperator(gain=jnp.array(1.0)),
        names=("loss", "gain"),
    )
    template = State(
        coords=Coordinates(
            time=jnp.arange(2, dtype=float), freq=jnp.array([100.0, 110.0])
        ),
        data=jnp.ones((2, 2)),
    )
    space = ParameterSpace(
        latents=[Latent("v", init=values)],
        bindings=[
            Bind(
                "v",
                into=(lambda p: p["loss"].efficiency, lambda p: p["gain"].gain),
                fn=fn,
                fan=fan,
            )
        ],
    )
    return float(space.bind(pipeline, {"v": values})(template).data[0, 0])


def test_a_python_container_type_alone_selects_the_physics_upstream():
    """The defect ``fan=`` exists for, reproduced here as the baseline.

    ``v`` and ``list(v)`` are the SAME DATA. One is a JAX array and one a
    Python list of its elements, and that difference -- invisible in the
    values, in every shape, and to both ``check_linearity`` and
    ``identifiability`` -- chooses between tying one value to both leaves
    and splitting it element-wise. Measured: **4.0** against **10.0**.
    """
    with jax.enable_x64(True), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        broadcast = _fan_prediction(lambda v: v)
        distribute = _fan_prediction(lambda v: list(v))
    assert broadcast == pytest.approx(4.0, rel=1e-12)
    assert distribute == pytest.approx(10.0, rel=1e-12)


def test_the_graph_reaches_both_answers_without_anything_to_infer():
    """The same two numbers, as two graphs that do not resemble each other.

    This is the migration's answer to ``fan=``: not a better inference, but
    a form with nothing left to infer. Each consumer is a named edge, so
    "the same value into both" and "element k into stage k" are written
    differently -- ``x[0]`` twice against ``x[0]`` and ``x[1]`` -- and no
    container type is consulted.

    Asserted against the SAME two numbers the upstream fixture produces, so
    this is a correspondence and not a separate toy.
    """
    from bayesmith import det, evaluate, observe, sample, trace

    values = {"v": jnp.array(FAN_VALUES)}
    zeros = jnp.zeros((2, 2))

    def tie():
        v = sample("v", lambda: dist.Normal(jnp.zeros(2), 1.0).to_event(1))
        a = det("a", lambda x: x[0], v)
        b = det("b", lambda x: x[0], v)
        p = det("pred", lambda x, y: jnp.ones((2, 2)) * x * y, a, b)
        observe("d", lambda m: dist.Normal(m, 1.0).to_event(2), p, obs=zeros)

    def split():
        v = sample("v", lambda: dist.Normal(jnp.zeros(2), 1.0).to_event(1))
        a = det("a", lambda x: x[0], v)
        b = det("b", lambda x: x[1], v)
        p = det("pred", lambda x, y: jnp.ones((2, 2)) * x * y, a, b)
        observe("d", lambda m: dist.Normal(m, 1.0).to_event(2), p, obs=zeros)

    with jax.enable_x64(True):
        tied = float(evaluate(trace(tie), values)["pred"][0, 0])
        splitted = float(evaluate(trace(split), values)["pred"][0, 0])
    assert tied == pytest.approx(4.0, rel=1e-12)
    assert splitted == pytest.approx(10.0, rel=1e-12)


def test_bayesmith_has_no_fan_keyword_to_get_wrong():
    """The absence, asserted, so a port of ``fan=`` has to argue with a test.

    ``Bind`` has no counterpart here; if one appears, the ambiguity it
    manages comes back with it.
    """
    from bayesmith.graph import nodes

    assert not hasattr(nodes, "Bind")
    assert not any("fan" in f for f in nodes.Deterministic.__annotations__)


# --------------------------------------------------------------------------
# refuse_stochastic_stages, and the equivalent that did not exist.
# --------------------------------------------------------------------------


def _rheplicant_stochastic_pipeline():
    from rheplicant.core.contract import RANDOMNESS
    from rheplicant.core.operator import AbstractOperator
    from rheplicant.core.pipeline import Pipeline
    from rheplicant.radio import AntennaLossOperator, GainOperator

    class Noisy(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("data", RANDOMNESS)
        provides: ClassVar[tuple[str, ...]] = ("data",)
        sigma: float = 1.0

        def __call__(self, state):
            return state

    return Pipeline(
        AntennaLossOperator(efficiency=jnp.array(1.0), t_physical=jnp.array(0.0)),
        Noisy(),
        GainOperator(gain=jnp.array(1.0)),
        names=("loss", "noise", "gain"),
    )


def test_both_sides_refuse_a_forward_model_that_draws_its_own_randomness():
    """§四 4.2's ``refuse_stochastic_stages`` equivalent -- *"理由改写,
    行为不得变"*.

    **Measured before this row: bayesmith had no such refusal.** A ``det``
    node whose parent is a PRNG key evaluated happily and returned a field.
    That is precisely the defect rheplicant's guard exists for, and the one
    nothing downstream can see: inference closes the model over ONE
    evaluation, so the draw is made once and the same frozen field rides
    every prediction compared against the data. Adding a constant field is
    exactly affine, so ``check_linearity`` reports a departure of 0 and
    ``identifiability`` reports full rank. Upstream measured **10.6 sigma**
    of bias with BOTH exits reporting the same error bar to every digit.

    The detector is the DECLARATION on each side, which is why the reason
    could be rewritten without the behaviour changing: rheplicant reads
    ``RANDOMNESS`` in an operator's ``requires``; the graph reads a ``Const``
    whose value carries a PRNG key dtype. Same class of case, same signal.
    """
    from rheplicant.core.errors import ParameterSpaceError
    from rheplicant.inference.parameters import refuse_stochastic_stages

    from bayesmith import const, det, observe, trace
    from bayesmith.errors import GraphError

    with jax.enable_x64(True):
        # rheplicant: the clean pipeline passes, the stochastic one does not.
        clean, _ = _rheplicant_pipeline()
        assert refuse_stochastic_stages(clean, "crosscheck") is None
        with pytest.raises(ParameterSpaceError, match="draws randomness"):
            refuse_stochastic_stages(_rheplicant_stochastic_pipeline(), "crosscheck")

        # bayesmith: the same model shape, refused at graph construction.
        def stochastic():
            key = const("key", jax.random.key(0))
            p = det("pred", lambda k: jax.random.normal(k, (N_TIME, N_FREQ)), key)
            observe(
                "d",
                lambda m: dist.Normal(m, 1.0).to_event(2),
                p,
                obs=jnp.zeros((N_TIME, N_FREQ)),
            )

        with pytest.raises(GraphError, match="PRNG key"):
            trace(stochastic)


def test_the_refusal_says_where_randomness_belongs_instead():
    """A refusal that only forbids leaves the modeller with the same model.

    rheplicant's message explains the frozen-field mechanism. This side's
    has to do that AND name the alternative, because on a graph there IS
    one and it is one word away: ``sample`` gives the same field a
    ``log_prob``, so it enters the joint distribution instead of hiding in
    the mean. That is the row's "reason rewritten" -- the rule here is *a
    random node without a density cannot enter the joint*.
    """
    from bayesmith import const, det, observe, trace
    from bayesmith.errors import GraphError

    with jax.enable_x64(True):

        def stochastic():
            key = const("key", jax.random.key(0))
            p = det("pred", lambda k: jax.random.normal(k, (N_TIME, N_FREQ)), key)
            observe(
                "d",
                lambda m: dist.Normal(m, 1.0).to_event(2),
                p,
                obs=jnp.zeros((N_TIME, N_FREQ)),
            )

        try:
            trace(stochastic)
        except GraphError as exc:
            text = str(exc)
        else:
            raise AssertionError("the graph was accepted")
    assert "sample(" in text, text
    assert "log_prob" in text, text
    assert "cannot enter the joint" in text, text


def test_both_guards_are_blind_to_a_closure_and_both_say_so():
    """The limit, asserted so it cannot be mistaken for coverage.

    rheplicant's docstring states it: an operator that draws randomness
    without declaring ``"key"`` is invisible, and so is one hiding a draw in
    a static field. The same hole is here, and it is the SAME hole -- a
    ``fn`` that closes over a frozen draw passes, because there is no
    numerical symptom to find. That is the premise of both guards, not a
    weakness of either implementation.

    Measured rather than argued: the closure graph builds and evaluates.
    """
    from bayesmith import det, evaluate, observe, sample, trace

    with jax.enable_x64(True):
        frozen = jax.random.normal(jax.random.key(0), (N_TIME, N_FREQ))

        def closure():
            x = sample("x", lambda: dist.Normal(0.0, 1.0))
            p = det(
                "pred",
                lambda v: jnp.ones((N_TIME, N_FREQ)) * v + frozen,
                x,
                linear_in=("x",),
            )
            observe(
                "d",
                lambda m: dist.Normal(m, 1.0).to_event(2),
                p,
                obs=jnp.zeros((N_TIME, N_FREQ)),
            )

        prediction = np.asarray(evaluate(trace(closure), {"x": jnp.array(1.0)})["pred"])
    assert np.all(np.isfinite(prediction))
    # And it really is the frozen field riding along, not a coincidence.
    assert prediction == pytest.approx(1.0 + np.asarray(frozen), rel=1e-12)
