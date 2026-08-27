"""D16, one axis at a time: what each probe contract actually decides.

D16 asks where ``check_linearity``'s contract lives. The plan names five axes
on which the two probes differ. Naming a difference is not measuring one, so
each axis below gets a fixture built to make THAT axis, and only that axis,
decide the verdict -- and, where a knob exists, a demonstration that turning
it alone flips the answer.

The models are wrapped once and read by both sides, reusing
``probe_11_d17_dual_run``'s wrappers: writing each fixture twice is how a
comparison quietly starts comparing two models.

Run it::

    /Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_12_d16_five_axes.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import jax.numpy as jnp
import numpyro.distributions as ndist

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_11_d17_dual_run import (
    B1,
    B2,
    N_FREQ,
    Predict,
    _selector,
    _state,
)
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference.linear import check_linearity as rh_check
from rheplicant.inference.parameters import (
    Bind,
    Latent,
    ParameterSpace,
)

from bayesmith import det, observe, sample, trace
from bayesmith.exact.linearity import (
    DEFAULT_AT_POINTS,
    DEFAULT_SCALES,
    WEIGHTED_RTOL,
)
from bayesmith.exact.linearity import check_linearity as bm_check

SIGMA = 0.5


def rheplicant_side(fn, spec, *, names, at=None, scales=DEFAULT_SCALES, noise=None):
    """``check_linearity`` through a Pipeline. Returns (verdict, detail)."""
    operator = Predict(params={n: spec[n][0] for n in spec}, fn=fn)
    pipeline = Pipeline(operator, names=("predict",))
    space = ParameterSpace(
        latents=[
            Latent(
                n,
                init=spec[n][0],
                prior=ndist.Normal(spec[n][1], spec[n][2]),
                linear=spec[n][3],
            )
            for n in spec
        ],
        bindings=[Bind(n, into=_selector(n)) for n in spec],
    )
    try:
        errors = rh_check(
            space, pipeline, _state(), names=names, at=at, scales=scales,
            noise=noise,
        )
        return "accepted", {float(k): float(v) for k, v in errors.items()}
    except Exception as error:  # noqa: BLE001 - a refusal is the measurement
        return "REFUSED", type(error).__name__


def bayesmith_side(fn, spec, *, names, at=None, at_points=None, sigma=SIGMA):
    """``check_linearity`` on the Graph twin. Returns (verdict, detail)."""
    order = tuple(spec)

    def model():
        latents = {
            n: sample(n, lambda _n=n: ndist.Normal(spec[_n][1], spec[_n][2]))
            for n in order
        }
        mu = det(
            "mu",
            lambda *v: fn(dict(zip(order, v, strict=True))),
            *[latents[n] for n in order],
            linear_in=tuple(n for n in order if spec[n][3]),
        )
        observe(
            "d",
            lambda m, _s=sigma: ndist.Normal(m, _s(m) if callable(_s) else _s),
            mu,
            obs=jnp.broadcast_to(fn({n: spec[n][0] for n in order}), (N_FREQ,)),
        )

    graph = trace(model)
    outside = {n: spec[n][0] for n in order if n not in names}
    try:
        errors = bm_check(
            graph,
            names,
            {**outside, **(at or {})},
            at_points=at_points,
        )
        return "accepted", {k: {float(s): float(e) for s, e in v.items()}
                            for k, v in errors.items()}
    except Exception as error:  # noqa: BLE001 - a refusal is the measurement
        return "REFUSED", type(error).__name__


def head(number: int, title: str, question: str) -> None:
    print("\n" + "=" * 78)
    print(f"AXIS {number} -- {title}")
    print(f"  the question: {question}")
    print("=" * 78)


def axis_1_anchor() -> None:
    head(
        1,
        "the anchor: max|init| (rheplicant) vs the prior width (bayesmith)",
        "at what magnitude is 'one probe' measured?",
    )
    curvature = 1e-7

    def fn(p):
        signal = B1 @ p["u"]
        return signal + curvature * signal**2

    #: init small, prior wide -- so the two anchors are 100x apart and nothing
    #: else about the fixture differs between the sides.
    spec = {"u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 100.0, True)}
    print(f"  model: signal + {curvature:g} * signal**2, max|init| = 1.0, "
          f"prior width = 100.0")
    print(f"  scales {DEFAULT_SCALES} -> rheplicant probes reach "
          f"{max(DEFAULT_SCALES):g}, bayesmith {max(DEFAULT_SCALES) * 100:g}")
    print("  rheplicant:", rheplicant_side(fn, spec, names=("u",)))
    print("  bayesmith :", bayesmith_side(fn, spec, names=("u",)))
    print("\n  THE FLIP (the knob exists): give rheplicant the same reach by")
    print("  multiplying its scales by the ratio of the two anchors --")
    widened = tuple(s * 100.0 for s in DEFAULT_SCALES)
    print(f"  scales={widened} ->",
          rheplicant_side(fn, spec, names=("u",), scales=widened))


def axis_2_at_points() -> None:
    head(
        2,
        f"at-points: 1 (rheplicant) vs {DEFAULT_AT_POINTS} incl. prior draws",
        "affine GIVEN the outside latents -- given which values of them?",
    )
    centre = 2.0

    def fn(p):
        signal = B1 @ p["u"]
        # Exactly affine when w sits at its declared init, curved anywhere else.
        return signal + 1e-3 * (p["w"] - centre) * signal**2

    spec = {
        "u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 1.0, True),
        "w": (jnp.asarray(centre), jnp.asarray(centre), 1.0, False),
    }
    print(f"  model: affine in u exactly when w == {centre}, curved elsewhere;")
    print(f"         w's declared init IS {centre}, and its prior is N({centre}, 1)")
    print("  rheplicant:", rheplicant_side(fn, spec, names=("u",)))
    print("  bayesmith :", bayesmith_side(fn, spec, names=("u",)))
    print("\n  THE FLIP: hand rheplicant one of the outside values bayesmith")
    print("  also looks at, and its verdict changes --")
    print("  at={'w': 3.0} ->",
          rheplicant_side(fn, spec, names=("u",), at={"w": jnp.asarray(3.0)}))


def axis_3_criteria() -> None:
    head(
        3,
        "criteria: one relative (rheplicant) vs relative AND sigma-weighted",
        "small compared to WHAT -- the signal, or the noise?",
    )
    curvature = 1e-7

    def fn(p):
        signal = B1 @ p["u"]
        return signal + curvature * signal**2

    spec = {"u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 1.0, True)}
    print(f"  model: signal + {curvature:g} * signal**2 -- a departure UNDER the")
    print("         relative tolerance at every probe, so the first criterion")
    print("         passes it. The noise is the only thing that can object.")
    from rheplicant.inference.noise import HomoscedasticNoise

    print("\n  rheplicant WITHOUT a noise model (all it could do before D16):")
    for sigma in (1e2, 1.0, 1e-2, 1e-4):
        print(f"    sigma={sigma:<8g} {rheplicant_side(fn, spec, names=('u',))[0]}")
    print("  -- the same verdict four times: with no noise passed there is no")
    print("     second criterion, and that was the whole contract until 2026-08-27.")
    print("\n  rheplicant WITH it (noise= since D16 axis 3):")
    for sigma in (1e2, 1.0, 1e-2, 1e-4):
        verdict = rheplicant_side(
            fn, spec, names=("u",), noise=HomoscedasticNoise(sigma=sigma)
        )[0]
        print(f"    sigma={sigma:<8g} {verdict}")
    print(f"\n  bayesmith, same four (weighted_rtol = {WEIGHTED_RTOL:g}):")
    for sigma in (1e2, 1.0, 1e-2, 1e-4):
        print(f"    sigma={sigma:<8g} {bayesmith_side(fn, spec, names=('u',), sigma=sigma)[0]}")
    print("  -- THE FLIP is inside bayesmith and driven by sigma alone: the")
    print("     model, the probes and the departure never move.")
    _which_criterion_fired(fn, spec, sigma=1.0)


def _which_criterion_fired(fn, spec, *, sigma) -> None:
    """Read the two columns off the refusal, rather than inferring which fired.

    ``AffinityRefused`` carries them since G11 -- before that this line could
    only have quoted the message.
    """
    order = tuple(spec)

    def model():
        latents = {n: sample(n, lambda _n=n: ndist.Normal(spec[_n][1], spec[_n][2]))
                   for n in order}
        mu = det("mu", lambda *v: fn(dict(zip(order, v, strict=True))),
                 *[latents[n] for n in order],
                 linear_in=tuple(n for n in order if spec[n][3]))
        observe("d", lambda m: ndist.Normal(m, sigma), mu,
                obs=jnp.broadcast_to(fn({n: spec[n][0] for n in order}), (N_FREQ,)))

    try:
        bm_check(trace(model), ("u",), {})
    except Exception as error:  # noqa: BLE001 - the payload is the measurement
        print(f"\n  WHICH criterion crossed, read off the payload at sigma={sigma:g}:")
        print(f"    rtol={error.rtol:.3e}  weighted_rtol={error.weighted_rtol:.3e}")
        for scale in sorted(error.errors):
            rel, wei = error.errors[scale], error.weighted[scale]
            marks = []
            if rel > error.rtol:
                marks.append("relative")
            if wei > error.weighted_rtol:
                marks.append("sigma-weighted")
            print(f"    {scale:>8g}x  relative={rel:.3e}  weighted={wei:.3e}"
                  f"   fired: {marks or ['neither']}")


def axis_4_shape() -> None:
    head(
        4,
        "return and exception shape",
        "what does a caller receive, on each branch?",
    )

    def fn(p):
        return B1 @ p["u"]

    spec = {"u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 1.0, True)}
    rh_status, rh_detail = rheplicant_side(fn, spec, names=("u",))
    bm_status, bm_detail = bayesmith_side(fn, spec, names=("u",))
    print(f"  PASSING branch, rheplicant [{rh_status}]: {rh_detail}")
    print("     -> {scale: departure}, one level")
    print(f"  PASSING branch, bayesmith  [{bm_status}]: {bm_detail}")
    print("     -> {at_point: {scale: departure}}, two levels")

    def curved(p):
        signal = B1 @ p["u"]
        return signal + 0.5 * signal**2

    print("\n  FAILING branch -- the exception class and what it carries:")
    _report_payloads(curved, spec)


def _report_payloads(fn, spec) -> None:
    """The payload fields each side's refusal carries, read off a real one."""
    operator = Predict(params={n: spec[n][0] for n in spec}, fn=fn)
    pipeline = Pipeline(operator, names=("predict",))
    space = ParameterSpace(
        latents=[
            Latent(n, init=spec[n][0], prior=ndist.Normal(spec[n][1], spec[n][2]),
                   linear=spec[n][3])
            for n in spec
        ],
        bindings=[Bind(n, into=_selector(n)) for n in spec],
    )
    try:
        rh_check(space, pipeline, _state(), names=("u",))
    except Exception as error:  # noqa: BLE001 - the refusal is the measurement
        fields = sorted(k for k in vars(error) if not k.startswith("_"))
        print(f"    rheplicant: {type(error).__name__}(payload={fields})")

    order = tuple(spec)

    def model():
        latents = {n: sample(n, lambda _n=n: ndist.Normal(spec[_n][1], spec[_n][2]))
                   for n in order}
        mu = det("mu", lambda *v: fn(dict(zip(order, v, strict=True))),
                 *[latents[n] for n in order],
                 linear_in=tuple(n for n in order if spec[n][3]))
        observe("d", lambda m: ndist.Normal(m, SIGMA), mu,
                obs=jnp.broadcast_to(fn({n: spec[n][0] for n in order}), (N_FREQ,)))

    try:
        bm_check(trace(model), ("u",), {})
    except Exception as error:  # noqa: BLE001 - the refusal is the measurement
        fields = sorted(k for k in vars(error) if not k.startswith("_"))
        print(f"    bayesmith : {type(error).__name__}(payload={fields})")


def axis_5_aggregation() -> None:
    head(
        5,
        "aggregation: whole-output max (rheplicant) vs per element",
        "a lie in a faint channel, beside a bright honest one -- seen or diluted?",
    )
    bright = 1e8

    def fn(p):
        loud = bright * (B1 @ p["u"])[:6]
        faint = (B2 @ p["u"])[:6]
        return jnp.concatenate([loud, faint + 1e-2 * faint**2])

    spec = {"u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 1.0, True)}
    print(f"  model: six channels at {bright:g}x, affine; six at 1x with a 1e-2")
    print("         quadratic. The whole-output maximum is set by the bright half.")
    print("  rheplicant:", rheplicant_side(fn, spec, names=("u",)))
    print("  bayesmith :", bayesmith_side(fn, spec, names=("u",)))
    print("\n  THE FLIP: the same lie with no bright channel beside it --")

    def alone(p):
        faint = B2 @ p["u"]
        return faint + 1e-2 * faint**2

    print("  no bright half, rheplicant:",
          rheplicant_side(alone, spec, names=("u",)))
    print("  -- so rheplicant CAN see this curvature. What hid it was the")
    print("     maximum being taken over the whole output, where the bright")
    print("     half sets both the departure and the variation.")


def main() -> int:
    print("=" * 78)
    print("D16, five axes: what each probe contract decides, measured")
    print("=" * 78)
    print(f"scales default {DEFAULT_SCALES}; both sides' rtol default is 1e4*eps")
    for axis in (
        axis_1_anchor,
        axis_2_at_points,
        axis_3_criteria,
        axis_4_shape,
        axis_5_aggregation,
    ):
        axis()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001 - a probe reports its own crash and exits 1
        traceback.print_exc()
        sys.exit(1)
