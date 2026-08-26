"""D17's adjudication protocol: both partition probes, one fixture set, diffed.

D17 asks where the partition/log-discovery probes live -- rheplicant's, which
read a ``Pipeline`` through a ``ParameterSpace``, or bayesmith's, which read a
``Graph``. The plan does not let that be settled by argument. Its protocol is:
run BOTH over the same fixtures, diff the verdicts case by case, and switch
only where every case agrees; any disagreement is its own decision item.

**The colouring is already shared.** ``rheplicant.inference.partition``
imports ``bayesmith.dispatch.factor.first_fit`` -- one statement of the rule
that turns pairwise verdicts into groups. So what this probe compares is not
the partitioning logic but the PROBES feeding it, which is what D17 is about.

**One numerical function per fixture, wrapped twice.** Each case below defines
its prediction once, as a plain jax function of a dict of latents, and both
sides wrap that same object. Writing the model twice is how a comparison
quietly ends up comparing two different models -- measured earlier in this
programme, where a graph fixture that dropped an additive offset disagreed
with its hand-built twin by 0.16 and read exactly like a solver bug.

The verdict compared is the PARTITION: which latents share a block, and each
block's kind, normalised across the two vocabularies

    rheplicant  conjugate | log_conjugate | gradient
    bayesmith   gcr       | log-gcr       | nuts

into ``exact | log | gradient``. A refusal is a verdict too, and is recorded
as one rather than skipped.

Run it (from either checkout's venv -- both packages are installed in both)::

    .venv/bin/python docs/probes/probe_11_d17_dual_run.py
"""

from __future__ import annotations

import dataclasses
import sys
import traceback
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as ndist

# ----------------------------------------------------------- rheplicant ---
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import Coordinates, State
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace
from rheplicant.inference.partition import auto_blocks
from rheplicant.inference.plan import SamplingPlan

# ------------------------------------------------------------- bayesmith ---
from bayesmith import det, observe, sample, trace
from bayesmith.dispatch.factor import factor_partition

#: The two vocabularies, normalised. Written out rather than defaulted: a
#: lookup with a default arm cannot tell an unknown engine from a known one.
_KIND = {"conjugate": "exact", "log_conjugate": "log", "gradient": "gradient"}
_METHOD = {"gcr": "exact", "log-gcr": "log", "nuts": "gradient"}

N_FREQ = 12
N_TIME = 4

#: Well-conditioned, mutually orthogonal design columns, so no case is
#: measuring conditioning when it means to be measuring affinity.
_X = jnp.linspace(-1.0, 1.0, N_FREQ)
_Q, _ = jnp.linalg.qr(jnp.stack([_X**k for k in range(5)], axis=1))
B1 = _Q[:, :2]
B2 = _Q[:, 2:4]


class Predict(AbstractOperator):
    """Wraps one plain prediction function as a rheplicant operator.

    Generic on purpose: the fixture's mathematics lives in ``fn``, which the
    bayesmith side wraps too, so neither side can drift into a different
    model.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    params: dict[str, jax.Array]
    fn: Callable[[dict[str, jax.Array]], jax.Array] = eqx.field(static=True)

    def __call__(self, state):
        row = self.fn(self.params)
        return state.with_data(jnp.broadcast_to(row, (N_TIME, N_FREQ)))


def _state() -> State:
    return State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        meta={"telescope": "RHINO", "obs_id": "d17-protocol"},
    )


def rheplicant_verdict(case: Case) -> Any:
    """``auto_blocks``' partition, normalised."""
    operator = Predict(params=dict(case.init), fn=case.fn)
    pipeline = Pipeline(operator, names=("predict",))
    space = ParameterSpace(
        latents=[
            Latent(
                name,
                init=case.init[name],
                prior=ndist.Normal(case.prior_mean[name], case.prior_std[name]),
                linear=case.linear[name],
            )
            for name in case.names
        ],
        bindings=[
            Bind(name, into=_selector(name)) for name in case.names
        ],
    )
    # `noise=` since 2026-08-27: a log-conjugate block is a claim about the
    # likelihood, so rheplicant's partition now asks the same question of the
    # noise that bayesmith's always did. Before it, this probe's fourth and
    # sixth cases disagreed for that reason alone.
    blocks = auto_blocks(space, pipeline, _state(), noise=case.noise)
    # The ENGINE is derived by SamplingPlan, not carried on Block -- a derived
    # block has `engine is None`, and reading that field directly maps every
    # one of them to whatever the default arm of a lookup says. Measured: it
    # reported six fixtures as all-gradient while the GROUPING was correct,
    # which is the "cannot tell X from `the lookup did not happen`" shape this
    # programme keeps paying for. So the plan is built and asked.
    plan = SamplingPlan(space, *blocks)
    return normalise(
        (block.names, _KIND[plan.engines[block.names]]) for block in blocks
    )


def _selector(name: str) -> Callable[[Any], Any]:
    return lambda p, _n=name: p["predict"].params[_n]


def bayesmith_verdict(case: Case) -> Any:
    """``factor_partition``'s plan, normalised."""

    def model():
        latents = {
            name: sample(
                name,
                lambda _n=name: ndist.Normal(
                    case.prior_mean[_n], case.prior_std[_n]
                ),
            )
            for name in case.names
        }
        mu = det(
            "mu",
            lambda *values: case.fn(dict(zip(case.names, values, strict=True))),
            *[latents[name] for name in case.names],
            linear_in=tuple(n for n in case.names if case.linear[n]),
        )
        observe(
            "d",
            lambda m, _s=case.sigma: ndist.Normal(
                m, _s(m) if callable(_s) else _s
            ),
            mu,
            obs=case.data,
        )

    plan = factor_partition(trace(model))
    return normalise(
        (block.latents, _METHOD[block.method]) for block in plan.blocks
    )


def normalise(blocks) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """``((kind, sorted names), ...)``, itself sorted. Order is not a verdict."""
    return tuple(sorted((kind, tuple(sorted(names))) for names, kind in blocks))


@dataclasses.dataclass(frozen=True)
class Case:
    """One fixture: a prediction, its latents, and what the data looks like.

    A plain dataclass, not an ``eqx.Module``: it holds jax arrays and is never
    traced, and marking arrays static in a Module warns -- correctly.
    """

    label: str
    why: str
    fn: Callable
    names: tuple[str, ...]
    init: dict[str, Any]
    prior_mean: dict[str, Any]
    prior_std: dict[str, Any]
    linear: dict[str, bool]
    data: Any
    sigma: Any
    noise: Any


def case(label, why, fn, spec, *, sigma=1.0, noise=None, data=None) -> Case:
    """``sigma`` is the bayesmith side's; ``noise`` is rheplicant's twin of it.

    Two spellings of one noise, because the packages take it differently -- a
    ``dist_fn`` on the observed node there, a ``NoiseModel`` object here. They
    are asserted to describe the same thing by construction and by the
    fixtures' own labels; a mismatch would be this probe comparing two models
    again, which its module docstring is about.
    """
    from rheplicant.inference.noise import HomoscedasticNoise
    names = tuple(spec)
    init = {n: spec[n][0] for n in names}
    prior_mean = {n: spec[n][1] for n in names}
    prior_std = {n: spec[n][2] for n in names}
    linear = {n: spec[n][3] for n in names}
    truth = fn(init)
    return Case(
        label=label,
        why=why,
        fn=fn,
        names=names,
        init=init,
        prior_mean=prior_mean,
        prior_std=prior_std,
        linear=linear,
        data=truth if data is None else data,
        sigma=sigma,
        noise=HomoscedasticNoise(sigma=float(sigma)) if noise is None else noise,
    )


def cases() -> list[Case]:
    """The fixture set, covering the three axes the protocol names."""
    u0 = jnp.array([2.0, -0.5])
    v0 = jnp.array([1.0, 0.25])

    def additive(p):
        return B1 @ p["u"] + B2 @ p["v"]

    def multilinear(p):
        return (B1 @ p["u"]) * (B2 @ p["v"] + 3.0)

    def line(p):
        profile = jnp.exp(-0.5 * ((_X - p["centre"]) / 0.4) ** 2)
        return p["amp"] * profile

    def log_gain(p):
        return jnp.exp(B1 @ p["g"]) * (2.0 + 0.5 * _X)

    log_spec = {"g": (jnp.array([0.2, -0.1]), jnp.zeros(2), 0.3, False)}

    def boundary_affine(p):
        signal = B1 @ p["u"]
        return signal + 1e-7 * signal**2

    def bright_and_faint(p):
        bright = 1e8 * (B1 @ p["u"])
        faint = 1e-2 * (B2 @ p["v"]) + 1e-2 * (B2 @ p["v"]) ** 2
        return jnp.concatenate([bright[:6], faint[:6]])

    return [
        case(
            "additive_pair",
            "two linear latents that do not multiply: ONE exact block",
            additive,
            {
                "u": (u0, jnp.zeros(2), 10.0, True),
                "v": (v0, jnp.zeros(2), 10.0, True),
            },
        ),
        case(
            "multilinear_pair",
            "the product couples them: each factor its OWN exact block",
            multilinear,
            {
                "u": (u0, jnp.zeros(2), 1.0, True),
                "v": (v0, jnp.zeros(2), 1.0, True),
            },
        ),
        case(
            "linear_plus_nonlinear",
            "amp is affine, centre is not: one exact block and one gradient",
            line,
            {
                "amp": (jnp.asarray(3.0), jnp.asarray(0.0), 5.0, True),
                "centre": (jnp.asarray(0.1), jnp.asarray(0.0), 0.3, False),
            },
        ),
        case(
            "log_gain_additive_noise",
            "log(prediction) is affine, but the noise is ADDITIVE",
            log_gain,
            log_spec,
        ),
        case(
            "log_gain_multiplicative_f_small",
            "multiplicative noise at f=0.004, well under the 0.06 ceiling",
            log_gain,
            log_spec,
            sigma=lambda m: 0.004 * jnp.abs(m),
            noise=Multiplicative(0.004),
        ),
        case(
            "log_gain_multiplicative_f_large",
            "EXTREME f: 0.3, five times the ceiling first order still holds at",
            log_gain,
            log_spec,
            sigma=lambda m: 0.3 * jnp.abs(m),
            noise=Multiplicative(0.3),
        ),
        case(
            "boundary_affine",
            "affine to within 1e-7 of itself: right at the probe's threshold",
            boundary_affine,
            {"u": (u0, jnp.zeros(2), 10.0, True)},
        ),
        case(
            "bright_and_faint",
            "D16 axis five: a faint quadratic beside a 1e8 linear leaf",
            bright_and_faint,
            {
                "u": (u0, jnp.zeros(2), 1.0, True),
                "v": (v0, jnp.zeros(2), 1.0, True),
            },
        ),
    ]


def run(fn, arg) -> tuple[str, Any]:
    try:
        return "ok", fn(arg)
    except Exception as error:  # noqa: BLE001 - a refusal IS a verdict here
        return "refused", f"{type(error).__name__}: {str(error).splitlines()[0][:90]}"


class Multiplicative:
    """The smallest noise model rheplicant's log transform will read.

    Only ``fractional`` and ``std`` are consulted, so this is the honest
    minimum rather than a stub standing in for something richer.
    """

    def __init__(self, fractional: float):
        self.fractional = fractional

    def std(self, prediction):
        return self.fractional * jnp.abs(prediction)


def solve_time_cross_check() -> None:
    """Where rheplicant applies the noise refusals its PARTITION cannot see.

    The log disagreements below are not two packages holding different
    opinions about the same question. rheplicant asks the same questions and
    refuses on the same constant -- it just asks them later, inside
    ``to_log_space``, which takes the noise model that ``auto_blocks`` never
    receives. This section measures that, so the write-up rests on a run
    rather than on a reading of the source.
    """
    from rheplicant.inference.loglinear import FIRST_ORDER_MAX_FRACTIONAL, to_log_space
    from rheplicant.inference.noise import HomoscedasticNoise

    data = jnp.abs(jnp.linspace(1.0, 3.0, N_FREQ))
    print("\n" + "=" * 78)
    print("Solve-time cross-check: what rheplicant's to_log_space refuses")
    print(f"(rheplicant ceiling {FIRST_ORDER_MAX_FRACTIONAL}, "
          f"bayesmith's the same constant)")
    print("=" * 78)
    trials = [
        ("additive (HomoscedasticNoise)", HomoscedasticNoise(sigma=1.0)),
        ("multiplicative f=0.004", Multiplicative(0.004)),
        ("multiplicative f=0.06 (the ceiling)", Multiplicative(0.06)),
        ("multiplicative f=0.3", Multiplicative(0.3)),
    ]
    for label, noise in trials:
        try:
            to_log_space(data, noise)
            print(f"  {label:38} ACCEPTED")
        except Exception as error:  # noqa: BLE001 - the refusal is the measurement
            head = str(error).splitlines()[0][:60]
            print(f"  {label:38} REFUSED -- {type(error).__name__}: {head}")


def main() -> int:
    rows = []
    for fixture in cases():
        left_status, left = run(rheplicant_verdict, fixture)
        right_status, right = run(bayesmith_verdict, fixture)
        agree = left_status == right_status and left == right
        rows.append((fixture, left_status, left, right_status, right, agree))

    print("=" * 78)
    print("D17 dual-run protocol: rheplicant Pipeline probe vs bayesmith Graph probe")
    print("=" * 78)
    for fixture, ls, left, rs, right, agree in rows:
        print(f"\n### {fixture.label}  --  {fixture.why}")
        print(f"  rheplicant [{ls}]: {left}")
        print(f"  bayesmith  [{rs}]: {right}")
        print(f"  VERDICT: {'AGREE' if agree else '*** DISAGREE ***'}")

    disagreed = [r[0].label for r in rows if not r[5]]
    print("\n" + "=" * 78)
    print(f"{len(rows) - len(disagreed)}/{len(rows)} agree")
    if disagreed:
        print("DISAGREEMENTS (each is its own decision item, per the protocol):")
        for label in disagreed:
            print(f"  - {label}")
    else:
        print("Every case agrees: the protocol's condition for switching is met.")
    solve_time_cross_check()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - a probe reports its own crash and exits 1
        traceback.print_exc()
        sys.exit(1)
