"""D9, measured: is there ONE float32 rank tolerance that gets every verdict right?

D9's ruling takes option (b) -- "bayesmith relaxes to a dtype-derived rtol" --
with option (a) as a fallback, and writes the fallback's trigger as a
MEASUREMENT rather than a preference. This probe is that measurement.

The question is not whether a float32 tolerance can recover the ONE motivating
model. `DEFAULT_RANK_RTOL`'s own docstring already gives the numbers that say
it can: in float32 that model's null direction surfaces at 3.1e-8 and its
weakest identified direction sits at 4.8e-5, so any tolerance between them is
right for it. The question is whether one tolerance is right across a FAMILY,
because a package ships one default and meets many models.

So the sweep: a two-component power law

    mu = exp(a1) (nu/nu0)^-b  +  exp(a2) (nu/nu0)^-(b + delta)

over four latents. As ``delta -> 0`` the two components become one and the
four parameters collapse to two, so ``delta`` dials the true conditioning
smoothly over decades. For each ``delta`` the probe takes the
column-normalised spectrum in float64 and in float32 and asks, of every
candidate float32 tolerance, whether it reproduces float64's verdict.

Run it::

    /Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_13_d9_precision_policy.py
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

# The repo root, so the fixture the tests use is the fixture this probe reads.
# Writing the model twice is how a comparison quietly starts comparing two
# models -- the defect this repository names most often.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from bayesmith.diagnose.identifiability import DEFAULT_RANK_RTOL, identifiability
from tests.diagnose.models import two_component

#: The MODULE, not the function of the same name that
#: ``bayesmith.diagnose.__init__`` re-exports over it.
identifiability_module = importlib.import_module(
    "bayesmith.diagnose.identifiability"
)

DELTAS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 0.0]
#: Candidate float32 tolerances, spanning the whole plausible range: from
#: float64's own default up through `sqrt(eps)` in float32 (3.45e-4).
CANDIDATES = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, math.sqrt(float(np.finfo(np.float32).eps))]


def spectrum(delta: float, *, x64: bool, free_scale: bool = True):
    """The column-normalised singular values, at the given ambient precision.

    The two precision guards are patched out for the duration, and that is the
    probe's whole subject: they are the thing under test, and what lies behind
    them is what D9 has to decide about. Patching rather than re-deriving the
    SVD here keeps ONE spelling of the arithmetic -- a second one would let the
    probe and the package disagree about what they measured.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", x64)
    kept = (identifiability_module.refuse_ambient_float32,
            identifiability_module.refuse_single_precision)
    identifiability_module.refuse_ambient_float32 = lambda **_: None
    identifiability_module.refuse_single_precision = lambda *_, **__: None
    try:
        dtype = jnp.float64 if x64 else jnp.float32
        graph = two_component(delta, dtype, free_scale=free_scale)
        report = identifiability(graph, rtol=DEFAULT_RANK_RTOL)
        return np.asarray(report.singular_values, dtype=np.float64)
    finally:
        (identifiability_module.refuse_ambient_float32,
         identifiability_module.refuse_single_precision) = kept
        jax.config.update("jax_enable_x64", was)


def verdict(values: np.ndarray, rtol: float) -> int:
    """Nullity at this tolerance: how many directions the data cannot see."""
    return int(np.sum(values <= rtol * values[0]))


def main() -> None:
    print(__doc__.split("Run it::")[0].strip())
    print()
    print(f"float32 eps = {float(np.finfo(np.float32).eps):.3e}, "
          f"sqrt = {math.sqrt(float(np.finfo(np.float32).eps)):.3e}")
    print(f"float64 eps = {float(np.finfo(np.float64).eps):.3e}, "
          f"sqrt = {math.sqrt(float(np.finfo(np.float64).eps)):.3e}")
    print(f"DEFAULT_RANK_RTOL = {DEFAULT_RANK_RTOL:.1e}\n")

    header = f"{'delta':>8} {'f64 s_min/s_max':>16} {'f64 null':>9} " \
             f"{'f32 s_min/s_max':>16} " + " ".join(f"{c:>8.0e}" for c in CANDIDATES)
    print(header)
    print("-" * len(header))

    rows = []
    for free_scale in (True, False):
        label = ("A: with a free overall scale (exactly degenerate with the two "
                 "amplitudes, so nullity >= 1 at every delta)"
                 if free_scale else
                 "B: scale fixed -- the model is IDENTIFIED at large delta, so "
                 "this family exercises the false-ALARM direction too")
        print(f"\nfamily {label}")
        for delta in DELTAS:
            s64 = spectrum(delta, x64=True, free_scale=free_scale)
            s32 = spectrum(delta, x64=False, free_scale=free_scale)
            truth = verdict(s64, DEFAULT_RANK_RTOL)
            got = [verdict(s32, c) for c in CANDIDATES]
            rows.append((delta, truth, got))
            marks = " ".join(
                f"{g:>8}" if g == truth else f"{g:>7}*" for g in got
            )
            print(f"{delta:>8.0e} {s64[-1]/s64[0]:>16.3e} {truth:>9} "
                  f"{s32[-1]/s32[0]:>16.3e} {marks}")

    print("\n(* = this float32 tolerance DISAGREES with float64's verdict)\n")

    print("Which single float32 tolerance reproduces every float64 verdict?")
    winners = [
        c for i, c in enumerate(CANDIDATES)
        if all(got[i] == truth for _, truth, got in rows)
    ]
    if winners:
        print(f"  -> {[f'{c:.0e}' for c in winners]}")
    else:
        print("  -> NONE. Every candidate disagrees with float64 on at least one")
        print("     model in this family, so no dtype-derived rtol is a policy;")
        print("     it is a per-model retune. D9's fallback condition is met.")
        for i, c in enumerate(CANDIDATES):
            bad = [(d, t, g[i]) for d, t, g in rows if g[i] != t]
            print(f"     {c:.0e}: wrong on {len(bad)} of {len(rows)} -- "
                  + ", ".join(f"delta={d:.0e} (f64 {t}, f32 {gv})"
                              for d, t, gv in bad[:3]))


if __name__ == "__main__":
    main()
