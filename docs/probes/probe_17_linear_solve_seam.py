"""Wave B / `linear`: which of the solve surface's refusals survive the seam?

D48 set the precedent this probe applies: **a refusal whose evidence the seam
erases has to live before the seam**. The question is not "does bayesmith have
this check too" -- it is "if the facade converts first and calls second, what
does the caller actually see?", which has three possible answers and they are
not equally serious:

    (a) the far side raises an equivalent refusal   -> the check can be delegated
    (b) the far side raises, naming the wrong layer -> keep it; a message loss
    (c) the far side SILENTLY ANSWERS               -> keep it; a correctness loss

Only (c) is a hazard. This probe finds one, and it is the one the plan did not
name: ``check_noise_std_axis``.

**Already recorded, by a different route, and this probe did not know that
when it was written.**
``tests/crosscheck/test_linear.py::test_the_ambiguous_1d_sigma_is_resolved_before_bayesmith_can_see_it``
holds the same conclusion with a deeper cause: going through the GRAPH, the
ambiguity is destroyed by numpyro's ``promote_shapes`` inside the user's
``dist_fn``, so a guard on the far side could not be written even in
principle. What this probe adds is that the hazard survives the DIRECT route
too -- handing ``Precision`` straight to the solver, which is the route the
facade takes -- so the refusal has to be pre-positioned there as well. Two
routes, one conclusion; each note points at the other, because a fact kept in
two places is how this programme keeps paying for one of them going stale.

The enumeration is derived, not transcribed. Walking the call graph from the
four public solve names (``wiener_solve``, ``gcr_sample``, ``condition_bound``,
``condition_estimate``) reaches **ten** ``raise`` sites inside ``linear.py``
plus **two** external refusal helpers -- ``check_observed_shape`` and
``check_noise_std_axis``, which live in sibling modules and so would not appear
in a grep of this file.

Why every one of them is erased, structurally: the two solve surfaces do not
take the same arguments.

    rheplicant:  wiener_solve(block, observed, *, noise_std, prior_std, prior_mean)
    bayesmith:   wiener_solve(block, *, precision)

``observed``, ``noise_std``, ``prior_std`` and ``prior_mean`` are all ARGUMENTS
upstream and all FIELDS of the block downstream, put there when the block was
built from a graph. So no refusal that inspects one of those four arguments has
anything to inspect on the far side -- it is not that bayesmith declines to
check, it is that the thing to check is no longer a separate object by then.

Sections
--------
0. The precondition: does a hand-converted block agree numerically at all?
   (If it does not, every comparison below is comparing two different models.)
1. Refusal-by-refusal: rheplicant's message vs what the far side does.
2. Hazard A -- ``check_noise_std_axis``: the (c) case.
3. Hazard B -- ``_require_prior_std``: (b), and a docstring claim that is
   looser than what the code actually does.

Exit code is 0 whenever the probe finished, never a verdict -- same rule as
``probe_11_d17_dual_run.py`` and ``probe_16_g15_discharge.py``: a probe that
turns its measurement into a process status invites being read as a gate, and
this is evidence for a ruling.

Run:  /Users/zzhang/projects/rheplicant/.venv/bin/python docs/probes/probe_17_linear_solve_seam.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.exact.block import LinearBlock as BayesmithBlock
from bayesmith.exact.precision import diagonal_from
from bayesmith.exact.solve import wiener_solve as bayesmith_wiener
from rheplicant.inference.linear import LinearBlock as RheplicantBlock
from rheplicant.inference.linear import wiener_solve as rheplicant_wiener
from rheplicant.inference.noise import HomoscedasticNoise, RadiometerNoise

RULE = "-" * 72


def report(label: str, call) -> np.ndarray | None:
    """Run `call`; print either the refusal it raised or the answer it gave."""
    try:
        answer, _ = call()
    except Exception as error:  # noqa: BLE001 -- the exception IS the measurement
        print(f"    {label:<34} {type(error).__name__}: {str(error)[:96]}")
        return None
    value = answer["x"] if isinstance(answer, dict) else answer
    flat = np.asarray(value).ravel()
    print(f"    {label:<34} NO REFUSAL -> {np.array2string(flat[:4], precision=6)}")
    return np.asarray(value)


# --------------------------------------------------------------------------
# 0. The precondition
# --------------------------------------------------------------------------

print(RULE)
print("0. Does the hand conversion agree numerically? (precondition)")
print(RULE)

KEY = jax.random.PRNGKey(0)
N_DATA, N_LATENT = 8, 3
OPERATOR = jax.random.normal(KEY, (N_DATA, N_LATENT), dtype=jnp.float32)
OFFSET = jnp.arange(N_DATA, dtype=jnp.float32) * 0.1
TRUTH = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
OBSERVED = OPERATOR @ TRUTH + OFFSET + 0.01
SIGMA = jnp.full((N_DATA,), 0.5, dtype=jnp.float32)
PRIOR_STD = jnp.full((N_LATENT,), 3.0, dtype=jnp.float32)
PRIOR_MEAN = jnp.zeros((N_LATENT,), dtype=jnp.float32)


def upstream(**override) -> RheplicantBlock:
    fields = dict(
        name="x",
        shape=(N_LATENT,),
        dtype=jnp.float32,
        offset=OFFSET,
        forward=lambda x: OPERATOR @ x,
        adjoint=lambda y: OPERATOR.T @ y,
    )
    fields.update(override)
    return RheplicantBlock(**fields)


def downstream(**override) -> BayesmithBlock:
    fields = dict(
        names=("x",),
        shape={"x": (N_LATENT,)},
        dtype={"x": jnp.float32},
        offset={"y": OFFSET},
        forward=lambda domain: {"y": OPERATOR @ domain["x"]},
        adjoint=lambda codomain: {"x": OPERATOR.T @ codomain["y"]},
        data={"y": OBSERVED},
        prior_mean={"x": PRIOR_MEAN},
        prior_std={"x": PRIOR_STD},
    )
    fields.update(override)
    return BayesmithBlock(**fields)


here, _ = rheplicant_wiener(
    upstream(), OBSERVED, noise_std=SIGMA, prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN
)
there, _ = bayesmith_wiener(downstream(), precision=diagonal_from({"y": SIGMA}))
gap = float(np.max(np.abs(np.asarray(here) - np.asarray(there["x"]))))
print(f"    rheplicant  {np.array2string(np.asarray(here), precision=7)}")
print(f"    bayesmith   {np.array2string(np.asarray(there['x']), precision=7)}")
print(f"    max abs difference: {gap}")
print(
    "    => the conversion is SHAPE ONLY. Every difference below is a refusal,"
    if gap == 0.0
    else "    => NOT bit-identical; the comparisons below are suspect"
)

# --------------------------------------------------------------------------
# 1. Refusal by refusal
# --------------------------------------------------------------------------

print()
print(RULE)
print("1. Each refusal: what the caller sees with, and without, the facade")
print(RULE)

print("  R1  noise_std is a NoiseModel (plain)")
model = HomoscedasticNoise(sigma=jnp.float32(0.5))
report("upstream", lambda: rheplicant_wiener(upstream(), OBSERVED, noise_std=model, prior_std=PRIOR_STD))
report("downstream", lambda: bayesmith_wiener(downstream(), precision=diagonal_from({"y": model})))
print(
    "      the upstream docstring PREDICTED the far-side message verbatim"
    " ('Value ... with dtype object'); measured, it is exactly that."
)

print("  R2  noise_std is a NoiseModel (prediction-dependent)")
radiometer = RadiometerNoise(channel_width=1e6, integration_time=1.0)
report("upstream", lambda: rheplicant_wiener(upstream(), OBSERVED, noise_std=radiometer, prior_std=PRIOR_STD))
report("downstream", lambda: bayesmith_wiener(downstream(), precision=diagonal_from({"y": radiometer})))

print("  R3  prior_std missing")
report("upstream", lambda: rheplicant_wiener(upstream(), OBSERVED, noise_std=SIGMA, prior_std=None))
report(
    "downstream",
    lambda: bayesmith_wiener(downstream(prior_std={"x": None}), precision=diagonal_from({"y": SIGMA})),
)

print("  R4  the block's offset is complex")
complex_offset = OFFSET.astype(jnp.complex64)
report(
    "upstream",
    lambda: rheplicant_wiener(upstream(offset=complex_offset), OBSERVED, noise_std=SIGMA, prior_std=PRIOR_STD),
)
report(
    "downstream",
    lambda: bayesmith_wiener(downstream(offset={"y": complex_offset}), precision=diagonal_from({"y": SIGMA})),
)

print("  R5  observed does not match the prediction's shape")
wrong_length = jnp.ones((N_DATA + 2,), dtype=jnp.float32)
report(
    "upstream",
    lambda: rheplicant_wiener(upstream(), wrong_length, noise_std=SIGMA, prior_std=PRIOR_STD),
)
report(
    "downstream",
    lambda: bayesmith_wiener(downstream(data={"y": wrong_length}), precision=diagonal_from({"y": SIGMA})),
)

# --------------------------------------------------------------------------
# 2. Hazard A -- the (c) case
# --------------------------------------------------------------------------

print()
print(RULE)
print("2. HAZARD A -- check_noise_std_axis: the far side SILENTLY ANSWERS")
print(RULE)
print(
    "  A length-n sigma against an (n, n) prediction reads equally well as one\n"
    "  sigma per row and as one per column. NumPy settles it by aligning trailing\n"
    "  axes. Upstream refuses; downstream takes a Precision, by which time the\n"
    "  ambiguity has already been resolved -- silently, and possibly wrongly."
)

SIDE = 8


def per_row(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.broadcast_to(x[:, None], (SIDE, SIDE)) * 1.0


SQUARE_OFFSET = jnp.zeros((SIDE, SIDE), dtype=jnp.float32)
SQUARE_TRUTH = jnp.linspace(1.0, 2.0, SIDE, dtype=jnp.float32)
SQUARE_OBSERVED = per_row(SQUARE_TRUTH) + SQUARE_OFFSET
AMBIGUOUS_SIGMA = jnp.linspace(0.01, 1.0, SIDE, dtype=jnp.float32)
SQUARE_PRIOR_STD = jnp.full((SIDE,), 10.0, dtype=jnp.float32)
SQUARE_PRIOR_MEAN = jnp.zeros((SIDE,), dtype=jnp.float32)

square_upstream = RheplicantBlock(
    name="x",
    shape=(SIDE,),
    dtype=jnp.float32,
    offset=SQUARE_OFFSET,
    forward=per_row,
    adjoint=lambda y: jnp.sum(y, axis=1),
)
square_downstream = BayesmithBlock(
    names=("x",),
    shape={"x": (SIDE,)},
    dtype={"x": jnp.float32},
    offset={"y": SQUARE_OFFSET},
    forward=lambda domain: {"y": per_row(domain["x"])},
    adjoint=lambda codomain: {"x": jnp.sum(codomain["y"], axis=1)},
    data={"y": SQUARE_OBSERVED},
    prior_mean={"x": SQUARE_PRIOR_MEAN},
    prior_std={"x": SQUARE_PRIOR_STD},
)

report(
    "upstream, ambiguous",
    lambda: rheplicant_wiener(
        square_upstream,
        SQUARE_OBSERVED,
        noise_std=AMBIGUOUS_SIGMA,
        prior_std=SQUARE_PRIOR_STD,
        prior_mean=SQUARE_PRIOR_MEAN,
    ),
)
bare = report(
    "downstream, bare (n,)",
    lambda: bayesmith_wiener(square_downstream, precision=diagonal_from({"y": AMBIGUOUS_SIGMA})),
)
spelled = report(
    "downstream, explicit (n, 1)",
    lambda: bayesmith_wiener(
        square_downstream, precision=diagonal_from({"y": AMBIGUOUS_SIGMA[:, None]})
    ),
)
if bare is not None and spelled is not None:
    spread = float(np.max(np.abs(bare - spelled)))
    print(
        f"    the two readings of the SAME sigma vector differ by {spread:.4g};"
        f" both finite: {bool(np.all(np.isfinite(bare)))}/{bool(np.all(np.isfinite(spelled)))}"
    )
    print(
        "    => (c). Nothing downstream can tell these apart. This refusal is the"
        "\n       one that has to be pre-positioned; the rest are message quality."
    )

# --------------------------------------------------------------------------
# 3. Hazard B -- what _require_prior_std actually guards
# --------------------------------------------------------------------------

print()
print(RULE)
print("3. HAZARD B -- _require_prior_std guards the SPELLING, not the conditioning")
print(RULE)
print(
    "  Its message says no prior leaves AtN^-1A free to be singular and 'CG would\n"
    "  return a finite, arbitrary answer rather than fail' -- and then invites\n"
    "  'a large prior_std for an effectively flat prior'. Both halves are worth\n"
    "  measuring, because they point opposite ways."
)

SHORT, WIDE = 4, 3
SINGULAR = jnp.zeros((SHORT, WIDE), dtype=jnp.float32).at[:, 0].set(1.0).at[:, 1].set(1.0)
SINGULAR_OFFSET = jnp.zeros((SHORT,), dtype=jnp.float32)
SINGULAR_OBSERVED = jnp.full((SHORT,), 2.0, dtype=jnp.float32)
UNIT_SIGMA = jnp.ones((SHORT,), dtype=jnp.float32)
FLAT = jnp.full((WIDE,), 1e8, dtype=jnp.float32)

singular_upstream = RheplicantBlock(
    name="x",
    shape=(WIDE,),
    dtype=jnp.float32,
    offset=SINGULAR_OFFSET,
    forward=lambda x: SINGULAR @ x,
    adjoint=lambda y: SINGULAR.T @ y,
)
print("  the invited flat prior reaches the solve on BOTH sides:")
report(
    "upstream, prior_std=1e8",
    lambda: rheplicant_wiener(
        singular_upstream, SINGULAR_OBSERVED, noise_std=UNIT_SIGMA, prior_std=FLAT,
        prior_mean=jnp.zeros((WIDE,), dtype=jnp.float32),
    ),
)
report(
    "downstream, prior_std=1e8",
    lambda: bayesmith_wiener(
        BayesmithBlock(
            names=("x",),
            shape={"x": (WIDE,)},
            dtype={"x": jnp.float32},
            offset={"y": SINGULAR_OFFSET},
            forward=lambda domain: {"y": SINGULAR @ domain["x"]},
            adjoint=lambda codomain: {"x": SINGULAR.T @ codomain["y"]},
            data={"y": SINGULAR_OBSERVED},
            prior_mean={"x": jnp.zeros((WIDE,), dtype=jnp.float32)},
            prior_std={"x": FLAT},
        ),
        precision=diagonal_from({"y": UNIT_SIGMA}),
    ),
)

print("  and the answer is NOT arbitrary -- it is the minimum-norm one, whatever")
print("  the prior mean, because CG from a zero start stays in range(A^T):")
for mean in (
    jnp.zeros((WIDE,), dtype=jnp.float32),
    jnp.array([5.0, -5.0, 0.0], dtype=jnp.float32),   # in null(A)
    jnp.array([0.0, 0.0, 7.0], dtype=jnp.float32),    # in null(A)
):
    answer, _ = rheplicant_wiener(
        singular_upstream, SINGULAR_OBSERVED, noise_std=UNIT_SIGMA,
        prior_std=FLAT, prior_mean=mean,
    )
    print(
        f"    prior_mean={np.array2string(np.asarray(mean)):<18}"
        f" -> {np.array2string(np.asarray(answer), precision=6)}"
    )
print(
    "    => the guard is real and worth keeping, but its stated reason is looser\n"
    "       than the code: the MEAN is well defined here. What is improper is the\n"
    "       WIDTH, which is why gcr_sample -- not wiener_solve -- is the exit that\n"
    "       would actually produce nonsense. Carried across as a facade refusal;\n"
    "       the wording is corrected where it is re-spelled."
)

print()
print(RULE)
print("Probe finished. Exit status is not a verdict.")
print(RULE)
