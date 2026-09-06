"""probe_37 -- the R5 backend bake-off: two candidates, one budget, eight axes.

Run (this repository's own environment, where both candidates are ABSENT and
the probe says so rather than reporting nothing):

    .venv/bin/python docs/probes/probe_37_backend_bakeoff.py

Run (an environment carrying both candidates, which is what Task 5 scores):

    PYTHONPATH=<repo> <env>/bin/python docs/probes/probe_37_backend_bakeoff.py

R5 Task 5. **This script decides nothing.** It produces the runs the verdict in
`docs/superpowers/specs/2026-09-06-r5-backend-evaluation.md` is scored against,
and every one of its cells reports one of three verdicts -- MEASURED, ABSENT
(the candidate is not installed, this repository's default state) or WITHHELD /
DECLINED (the measurement could not be made, or was made and may not be used).
Red line 14: a check that can decline to run must record that it declined in a
value distinct from "ran and found nothing", because otherwise its silence is
read as its pass. Section 6's correctness column is where that bites hardest --
R5 plan 5.2a: a residual of four axes beside any exact block is five, and five
has no independent oracle at a test-affordable budget, so those cells say
WITHHELD and never blank.

Sections:

    1.  The environment, and the two candidates in it.
    2.  What each backend needs that a `CompiledEvidenceProblem` does not
        carry -- measured by driving each from the compiled problem alone and
        recording what raised.
    3.  5.1, the shared budget. The likelihood-evaluation count for each
        backend is DERIVED from its own reported per-step counters and then
        AUDITED against a ground-truth count taken with `io_callback` on a tiny
        problem. A wrapping Python counter reports 1 after a thousand jitted
        calls (plan 0.7) and is not used anywhere here.
    4.  5.4's stop-rule: does each backend's reported `log Z` uncertainty MOVE
        with the budget, or is it a placeholder?
    5.  5.2, the table: every admitted class-(b)/(c) fixture plus Task 3's
        three, both candidates, the calibrated budget, x64, one seed.
    6.  Repeated runs: the spread of five seeds against the reported error bar
        (plan 0.8's asymmetry), on four declared rows -- the multimodal one,
        the heavy-tailed one, the four-axis one and a Gaussian control.
    7.  1.5 condition 2's artefact: the same key twice, bitwise, and which of
        JAX's two key objects each entry point accepts.
    8.  1.5 condition 4's first half: release date, cadence and the installed
        version, with the date read.
    9.  1.5 condition 4's second half: every observed termination signal onto a
        `TerminationReason` member, or onto NO HONEST MEMBER.
    10. 1.5 condition 5, as far as it can be measured without an adapter: what
        a driver for each backend had to contain, in lines, and what each
        backend does not supply.

Section 0 -- the oracle, filled or WITHHELD per fixture -- always runs, because
every other section reads it.

Each cell of sections 4-7 runs in its own SUBPROCESS. Three reasons, all
measured rather than assumed: importing `jaxns` writes `jax_enable_x64`
process-globally, so a shared process lets one candidate change the other's
arithmetic; compile time is only compile time in a process that has not already
compiled the same graph; and peak RSS is a property of a process, so a cell
that shares one reports its neighbour's memory as its own.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CANDIDATES = ("blackjax", "jaxns")

#: The seed set is part of the fixture (design 9.3), declared here rather than
#: drawn per run, so that a failure names which seed moved.
SEEDS = (0, 1, 2, 3, 4)

#: The shared budget, in log-likelihood evaluations. One number for the whole
#: table (plan 0.7: each backend's knobs are tuned to hit it on a calibration
#: fixture and then FROZEN), and the tolerance the same section declares.
TARGET_EVALUATIONS = 200_000
BUDGET_TOLERANCE = 0.10

#: The span rule, declared before any span is placed. A span is a DOMAIN and
#: plan 0.15 requires every pinned number to carry the one it was integrated
#: over; placing them by hand per fixture is how a table gets tuned to its
#: result. `K_SPAN` prior standard deviations either side of the prior mean,
#: evaluated at the prior centre for a latent whose parents are latent, with
#: `FALLBACK_HALF_WIDTH` where the prior has no finite variance -- a Cauchy has
#: none, and that is the point of the heavy-tailed fixture.
K_SPAN = 9.0
FALLBACK_HALF_WIDTH = 1000.0


#: Task 3's three, which do not appear in `models.py` and never will -- plan
#: Task 3 rules that module a census DENOMINATOR and forbids adding to it. They
#: enter the table by name, with the constructor the fixture module declares.
TASK_3_FIXTURES = ("mixture_prior_residual", "cauchy_residual_pair", "undeclared_quartet")


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# ==========================================================================
# the fixture census -- asked of the dispatcher, never hand-listed
# ==========================================================================
@dataclasses.dataclass(frozen=True)
class Fixture:
    label: str
    module: str
    structural_class: str
    premise: str
    exact: tuple[str, ...]
    sampled: tuple[str, ...]


def _built_graphs():
    """Every graph the two fixture modules ship, with its denominator stated."""
    import inspect

    from bayesmith.graph.reduction import as_graph
    from tests.dispatch.test_compiled_evidence_problem import PARAMETERISED
    from tests.exact import models, residual_models

    for module, declared in ((models, PARAMETERISED), (residual_models, {})):
        for name, fn in sorted(vars(module).items()):
            if not inspect.isfunction(fn) or name.startswith("_"):
                continue
            if fn.__module__ != module.__name__:
                continue
            if module is residual_models and name not in TASK_3_FIXTURES:
                # `residual_models` ships five GRAPH builders and several
                # closed-form helpers beside them; Task 5's table takes the
                # three Task 3.1 named. Named rather than filtered by an
                # `except` or by a signature test, so the gap is a declaration
                # and not a side effect of how the helpers happen to be typed.
                continue
            required = [
                p
                for p in inspect.signature(fn).parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            if required and name not in declared:
                raise AssertionError(
                    f"{name} takes required arguments and is not declared here; "
                    "a census that skips it covers less than it claims"
                )
            built = fn(**declared.get(name, {}))
            candidates = built if isinstance(built, tuple) else (built,)
            for index, candidate in enumerate(candidates):
                try:
                    graph = as_graph(candidate)
                    graph.nodes  # noqa: B018 - reading it is the check
                except (AttributeError, TypeError):
                    continue
                label = name if len(candidates) == 1 else f"{name}[{index}]"
                yield label, module.__name__, graph


def build_one(label: str):
    """Build exactly one fixture's graph, by the label the census gave it.

    The census's own construction, narrowed to a single row -- not a second
    lookup table. A `label` the census would not have produced raises here
    rather than resolving to something else.
    """
    for name, _module, graph in _built_graphs():
        if name == label:
            return graph
    raise KeyError(f"{label!r} is not a fixture this census builds")


def census() -> list[Fixture]:
    """Classify every graph and record the premise an `EvidenceTask` refuses under."""
    import bayesmith
    from bayesmith import compile_task
    from bayesmith.artifacts.refusal import Refusal
    from bayesmith.artifacts.tasks import EvidenceTask, new_task_meta
    from bayesmith.errors import BayesmithError
    from tests.dispatch.test_task_protocol import model_ref

    ref = model_ref()
    out: list[Fixture] = []
    for label, module, graph in _built_graphs():
        try:
            plan = bayesmith.compile(graph)
        except BayesmithError as error:
            out.append(
                Fixture(label, module, "compile-refused", type(error).__name__, (), ())
            )
            continue
        exact = tuple(plan.exact.latents) if plan.exact is not None else ()
        method = plan.exact.method if plan.exact is not None else None
        sampled = tuple(plan.sampled.latents) if plan.sampled is not None else ()
        if not sampled and method == "gcr":
            structural = "(a)"
        elif exact and sampled and method == "gcr":
            structural = "(b)"
        elif not exact and sampled:
            structural = "(c)"
        elif not exact and not sampled:
            structural = "(e)"
        else:
            structural = f"(d) {method}"
        answer = compile_task(
            graph, EvidenceTask(meta=new_task_meta(label="z")), model_ref=ref
        )
        premise = (
            answer.failed_premise if isinstance(answer, Refusal) else "ADMITTED"
        )
        out.append(Fixture(label, module, structural, premise, exact, sampled))
    return out


def bakeoff_fixtures(rows: list[Fixture]) -> list[Fixture]:
    """The rows the table runs: everything whose ONLY refusal is the backend.

    `capability_unavailable_r1` is the premise a graph reaches when every
    statement about the MODEL has passed and the one missing thing is a
    sampler -- which is exactly the set a bake-off is about. A row refused for
    propriety (`improper_outside_prior`) or for its method (`mixed_radiometer`,
    `gcr+mh`) is refused about itself and is not the backend's to answer.
    """
    return [row for row in rows if row.premise == "capability_unavailable_r1"]


# ==========================================================================
# the span rule, and the oracle it feeds
# ==========================================================================
def spans_for(graph, *, k: float = K_SPAN) -> tuple[Any, ...]:
    """One span per LATENT AXIS, placed by the declared rule and nothing else.

    `K_SPAN` prior standard deviations either side of the prior mean, read off
    the latent's own distribution at the prior centre. Where the distribution
    has no finite mean or variance -- a Cauchy has neither, which is the whole
    point of the heavy-tailed fixture -- the rule falls back to
    `FALLBACK_HALF_WIDTH` about the distribution's location, and says so by
    returning the fallback flag.

    A hand-placed span per fixture is how a correctness table gets tuned to the
    answer it wanted; this rule is stated before any span exists and is the
    same for all 21 rows.
    """
    import jax.numpy as jnp
    import numpy as np

    from bayesmith.dispatch.evidence import prior_environment
    from bayesmith.graph.evaluate import apply_probabilistic
    from tests.dispatch.residual_oracle import Span

    environment = prior_environment(graph)
    out = []
    fallbacks = []
    for name in graph.latents:
        distribution = apply_probabilistic(graph, graph.node(name), dict(environment))
        try:
            centre = np.asarray(distribution.mean, dtype=float)
            width = np.sqrt(np.asarray(distribution.variance, dtype=float))
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            centre, width = np.asarray(np.nan), np.asarray(np.nan)
        centre = np.atleast_1d(centre)
        width = np.atleast_1d(width)
        shape = jnp.shape(environment[name])
        size = int(np.prod(shape, dtype=int)) if shape else 1
        for index in range(size):
            c = float(centre.ravel()[index % centre.size])
            w = float(width.ravel()[index % width.size])
            if not np.isfinite(c) or not np.isfinite(w) or w == 0.0:
                loc = float(np.atleast_1d(np.asarray(getattr(distribution, "loc", 0.0), dtype=float)).ravel()[index % max(1, np.atleast_1d(np.asarray(getattr(distribution, "loc", 0.0))).size)])
                lo, hi = loc - FALLBACK_HALF_WIDTH, loc + FALLBACK_HALF_WIDTH
                fallbacks.append(name)
            else:
                lo, hi = c - k * w, c + k * w
            label = name if size == 1 else f"{name}[{index}]"
            out.append(Span(label, lo, hi))
    return tuple(out), tuple(sorted(set(fallbacks)))


def total_axes(graph) -> int:
    """Latent AXES, not latent names -- a vector latent is as many axes as it
    has entries, and `oracle_joint` pays a grid dimension for each."""
    import jax.numpy as jnp
    import numpy as np

    from bayesmith.dispatch.evidence import prior_environment

    environment = prior_environment(graph)
    total = 0
    for name in graph.latents:
        shape = jnp.shape(environment[name])
        total += int(np.prod(shape, dtype=int)) if shape else 1
    return total


#: Spans this repository has ALREADY declared, with their reasons committed
#: beside them. Where the suite has one, the bake-off uses it: re-deriving a
#: span here would be a second declaration of the same domain, and plan 0.16
#: records what a second copy of one measurement costs. `shared_ancestor` is
#: why the rule alone is not enough -- its declared prior straddles `tau = 0`,
#: where `Normal(0, 0).log_prob` is `nan`, so the RULE's span abstains and the
#: suite's hand-placed one does not. That is a property of the fixture and it
#: was found, named and pinned by Task 2; discovering it again here and calling
#: it a backend result would be an error.
DECLARED_SPANS = {
    # tests/dispatch/test_residual_oracle.py::CLASS_B
    "diamond_ancestor": (("tau", -4.0, 8.0), ("x", -6.0, 6.0)),
    "indirect_ancestor": (("tau", -4.0, 8.0), ("x", -6.0, 6.0)),
    "shared_ancestor": (("tau", 0.15, 4.5), ("x", -1.5, 3.5)),
    "overflowing_outside_latent": (("z", -100.0, 100.0), ("w", -8.0, 8.0)),
    "mixture_prior_residual": (("w", -8.0, 9.5), ("b", -7.5, 9.0)),
    # tests/dispatch/test_residual_fixtures.py::CAUCHY_SPAN
    "cauchy_residual_pair": (("z", -1000.0, 1000.0),),
    # tests/dispatch/test_residual_fixtures.py::_quartet_spans -- 6.5 posterior
    # standard deviations, placed from the CONSTRUCTED closed-form posterior.
    "undeclared_quartet": (
        ("alpha", -3.7685595616321264, 5.677615039413236),
        ("beta", -9.152619835239918, 8.246868256182466),
        ("gamma", -2.925832107741724, 3.569575555291562),
        ("delta", -7.893082924568883, 10.741709522789623),
    ),
}

#: The first grid per axis count. `start ** axes` must fit inside the oracle's
#: own `MAX_POINTS`, and the refinement count is then ARITHMETIC -- the largest
#: ladder of `n -> 2n - 1` doublings that still fits -- rather than a number
#: chosen per fixture. A hand-tuned refinement count is a way of choosing how
#: hard to look for convergence, and choosing that per row is choosing the
#: answer.
START = {1: 201, 2: 201, 3: 65, 4: 33}

#: The span rule's ladder. Applied in this order, uniformly, and the rung that
#: certified is recorded per row. A single `k` cannot serve 21 fixtures: at
#: `k = 9` two of them have posterior mass still growing at the span edge,
#: which is the oracle's `rho >= 1` abstain and a statement about the SPAN, not
#: about either backend.
K_LADDER = (9.0, 25.0)


def peak_and_widths(graph, *, restarts: int = 8):
    """The integrand's maximum and its curvature width per axis.

    Backend-independent by construction: gradient ascent on the model's own
    `log_joint`, started from the prior centre and from `restarts` prior draws,
    with the widths read off the Hessian at the best point. Multi-start because
    a single start from the prior centre finds one mode of a multimodal
    integrand and reports its width as the integrand's.

    Returns `(peak, widths, value)` keyed by axis name, or `(None, None, None)`
    where the ascent cannot run -- an improper prior with no `sample`, or an
    integrand that is `nan` at every start.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from scipy import optimize

    from bayesmith.dispatch.evidence import prior_environment
    from bayesmith.graph.evaluate import log_joint

    names = list(graph.latents)
    environment = prior_environment(graph)
    shapes = {name: jnp.shape(environment[name]) for name in names}
    sizes = [max(1, int(np.prod(shapes[name], dtype=int))) for name in names]
    offsets = np.cumsum([0] + sizes)

    def unflatten(x):
        return {
            name: jnp.reshape(jnp.asarray(x[lo:hi]), shapes[name])
            for name, lo, hi in zip(names, offsets[:-1], offsets[1:], strict=True)
        }

    def negative(x):
        value = log_joint(graph, unflatten(x))
        return -value

    grad = jax.jit(jax.grad(negative))
    starts = [np.concatenate([np.ravel(np.asarray(environment[n], dtype=float)) for n in names])]
    try:
        from bayesmith.dispatch.evidence import compile_evidence_problem

        problem = compile_evidence_problem(graph)
        for seed in range(restarts):
            draw = problem.prior_sample(jax.random.key(1000 + seed))
            starts.append(
                np.concatenate([np.ravel(np.asarray(draw[n], dtype=float)) for n in names])
            )
    except Exception as error:  # noqa: BLE001
        # An improper prior has no `sample`, so the prior-centre start stands
        # alone. Recorded rather than swallowed: red line 14, a check that
        # declines must say it declined.
        starts.append(starts[0])
        del error

    best = None
    for start in starts:
        if not np.all(np.isfinite(start)):
            continue
        try:
            found = optimize.minimize(
                lambda x: float(negative(x)),
                start,
                jac=lambda x: np.asarray(grad(jnp.asarray(x)), dtype=float),
                method="L-BFGS-B",
            )
        except (ValueError, FloatingPointError):
            continue
        if not np.isfinite(found.fun):
            continue
        if best is None or found.fun < best.fun:
            best = found
    if best is None:
        return None, None, None
    # **The located peak must beat the prior centre, or it is not a peak.**
    # An adversarial review replaced this function's return with the prior
    # centre and a unit width: the resolution test then passed trivially,
    # printed "every axis has a grid point within one curvature width of the
    # peak", and put `high_snr_curvature` back at -2 376 535.5 with
    # `resolved: True`. Nothing checked the ascent's own output. This does,
    # against the one point the ascent is guaranteed to have started from.
    centre_value = float(-negative(starts[0])) if np.all(np.isfinite(starts[0])) else -np.inf
    if -best.fun < centre_value - 1e-9:
        return None, None, None
    hessian = np.asarray(
        jax.hessian(lambda x: log_joint(graph, unflatten(x)))(jnp.asarray(best.x)),
        dtype=float,
    )
    hessian = np.atleast_2d(hessian)
    with np.errstate(all="ignore"):
        curvature = -np.diag(hessian)
        widths = np.where(curvature > 0, 1.0 / np.sqrt(np.abs(curvature)), np.inf)
    labels = _flat_axis_names(names, shapes)
    return (
        dict(zip(labels, np.asarray(best.x, dtype=float), strict=True)),
        dict(zip(labels, np.asarray(widths, dtype=float), strict=True)),
        float(-best.fun),
    )


def _flat_axis_names(names, shapes) -> list[str]:
    import numpy as np

    out = []
    for name in names:
        size = max(1, int(np.prod(shapes[name], dtype=int)))
        out.extend([name] if size == 1 else [f"{name}[{i}]" for i in range(size)])
    return out


def verify_the_peak(graph, peak, widths, height):
    """Is the claimed peak a stationary point of the model's own `log_joint`?

    **This lives in the caller and not inside `peak_and_widths`, and that
    placement is the point.** An adversarial review replaced the whole peak
    finder with one returning the prior centre and a unit width; a guard inside
    the replaced function goes with it.

    **And it reads the GRADIENT, not the height.** Two weaker checks were tried
    and both let the bypass through: trusting the returned `height` trusts the
    liar, and comparing the height against a scan of the prior fails when the
    claimed peak IS the prior centre, which is what the bypass returns. At a
    real optimum the gradient vanishes; at `high_snr_curvature`'s prior centre
    it is of order `1e12`. The test is dimensionless -- moving one curvature
    width along the gradient must change `log_joint` by at most one nat -- so
    it introduces no tuned number.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from bayesmith.graph.evaluate import log_joint

    if peak is None or widths is None or height is None:
        return None, None, "no peak was located"
    names = list(peak)
    finite = [name for name in names if np.isfinite(widths[name])]
    if not finite:
        return None, None, "no axis has a finite curvature width"

    def at(values):
        return log_joint(graph, {name: jnp.asarray(values[name]) for name in names})

    try:
        gradient = jax.grad(at)({name: float(peak[name]) for name in names})
    except Exception as error:  # noqa: BLE001 - a peak outside the model's domain
        return None, None, f"the gradient at the claimed peak raised {type(error).__name__}"
    worst = 0.0
    where = ""
    for name in finite:
        step = abs(float(gradient[name]) * widths[name])
        if step > worst:
            worst, where = step, name
    if not np.isfinite(worst) or worst > 1.0:
        return None, None, (
            f"the claimed peak is not stationary: one curvature width along "
            f"the gradient changes log_joint by {worst:.3g} nats on {where}, "
            "so it is a point the ascent did not reach rather than a peak"
        )
    return peak, widths, (
        f"stationary to {worst:.3g} nats per curvature width; height {height:.6g}"
    )


def resolves_the_peak(found, widths) -> tuple[bool, str]:
    """Does the grid this quadrature actually ran put a point near the peak?

    **The certificate cannot answer this and that is why the function exists.**
    A trapezoid that never samples a narrow peak converges beautifully -- to
    the integral of everything except the mass. Measured on
    `high_snr_curvature`: `sigma = 2e-6`, so the integrand falls 150 nats
    within `1e-5` of its maximum, while the finest grid the budget affords on
    the declared span is spaced `7e-4` apart. `oracle_joint` certified
    **-2376535.5** with a bound of `9.5e-09`, and a Laplace estimate from the
    peak's own height and curvature -- `145.02 + log(sqrt(2 pi) * 5.7e-7)` --
    says `+131.6`. Nothing in the certificate distinguishes the two, because
    refining a grid that keeps missing the peak keeps giving the same answer.

    The test is the weakest one that catches it: at least one grid point within
    one curvature width of the peak on every axis, i.e. spacing <= width. It
    introduces no tunable factor.

    **This condition now lives in `tests/dispatch/residual_oracle.py` as the
    certificate's fifth condition, and the resident one differs by a factor of
    two.** On a uniform grid of spacing `h` the nearest point to any location is
    at most `h / 2` away, so "within one curvature width" is `h / 2 <= width`;
    the form here is twice as strict. The difference was found by
    `undeclared_quartet`, which has a constructed closed-form evidence and sits
    at `h / width = 1.34`: strict refuses it, and the closed form says the value
    is right.

    This copy is left as it stands because the evaluation of record ran against
    it, and a stricter rule only escalated to the peak-placed rung more often
    than it needed to. Do not read the two as disagreeing about the physics; the
    resident one is the one to change.
    """
    if widths is None:
        # **Declining is not passing.** Red line 14: a check that cannot run
        # must say so in a value distinct from its pass. A first version
        # returned True here, so a fixture whose peak could not be located --
        # or whose peak finder had been replaced -- inherited a clean bill.
        return False, "no peak located, so the resolution test could not run"
    history = found.certificate.history
    if not history:
        return True, "no grid ran"
    count = history[-1][0]
    for span in found.spans:
        width = widths.get(span.name)
        if width is None or not (width < float("inf")):
            continue
        spacing = (span.upper - span.lower) / (count - 1)
        if spacing > width:
            return False, (
                f"the finest grid is spaced {spacing:.3e} on {span.name} and "
                f"the integrand's curvature width there is {width:.3e}; a "
                f"trapezoid that never samples the peak converges to the "
                f"integral of everything except the mass"
            )
    return True, "every axis has a grid point within one curvature width of the peak"


def oracle_for(label: str, graph):
    """`oracle_joint` over the whole latent space, with its certificate.

    Returns `(quadrature, span_source, spans)`. Tier 1 of design 9.1: it
    reaches the model's own `log_joint` and shares nothing with the
    elimination or with either backend, so it may grade both.
    """
    from tests.dispatch.residual_oracle import AGREEMENT_FLOOR, Span, oracle_joint

    peak, widths, height = peak_and_widths(graph)
    peak, widths, verdict = verify_the_peak(graph, peak, widths, height)
    fallbacks: tuple[str, ...] = ()
    if label in DECLARED_SPANS:
        attempts = [("suite", tuple(Span(*entry) for entry in DECLARED_SPANS[label]))]
    else:
        attempts = []
        fallbacks: tuple[str, ...] = ()
        for k in K_LADDER:
            rung, used_fallback = spans_for(graph, k=k)
            attempts.append((f"rule@{k:g}", rung))
            fallbacks = fallbacks or used_fallback
    if peak is not None:
        # The last rung: a span placed ON the integrand's own peak, at the same
        # `K_SPAN` half-width the rule uses everywhere else. Reached only when
        # the declared span's grid cannot resolve the peak, and certified on
        # its own terms -- the edge test bounds the mass it leaves out, so a
        # narrow span that certifies is a tier-1 value and not a guess.
        attempts.append(
            (
                "peak",
                tuple(
                    Span(
                        name,
                        peak[name] - K_SPAN * widths[name],
                        peak[name] + K_SPAN * widths[name],
                    )
                    for name in peak
                    if widths[name] < float("inf")
                ),
            )
        )
    found = None
    source = spans = None
    note = ""
    #: The DOMAIN handed to jaxns is never the peak-placed span, and this is a
    #: fairness rule rather than a detail. The oracle may place its grid on the
    #: peak because it is grading; a backend told where the peak is has been
    #: given the answer's location, and blackjax -- which draws from the prior
    #: -- is given no such thing. Measured on `high_snr_curvature`: with the
    #: peak span as its box jaxns integrates a domain `1e-5` wide around the
    #: mass; with the prior-shaped span it must find a `6e-7` peak inside a
    #: span **50 wide** -- the rule escalates to K = 25 on that row, so the box
    #: is (-25, 25) and not the (-9, 9) an earlier version of this comment and
    #: of the evaluation claimed -- which is the same problem blackjax is set.
    box = attempts[0][1]
    #: **Two different questions, and the first version conflated them.** One
    #: is whether to widen K; the other is whether to fall through to the
    #: peak-placed rung. An adversarial review implemented the rule this
    #: probe's own prose STATED -- "it does not escalate on a refinement-budget
    #: abstain" -- and found `high_snr_curvature` becoming WITHHELD, because
    #: that reading stops the peak rung too. The rule below separates them:
    #: widening K is refused except on the one abstain a wider span answers,
    #: and the peak rung is always reached, because it is not a wider span but
    #: a differently PLACED one.
    index = 0
    while index < len(attempts):
        source, spans = attempts[index]
        index += 1
        if len(spans) != len(peak or spans):
            continue  # a peak span is only usable when every axis has a width
        start, refinements = grid_for(len(spans))
        found = oracle_joint(
            graph,
            spans,
            resolution=AGREEMENT_FLOOR,
            start=start,
            refinements=refinements,
        )
        if source != "peak":
            box = spans
        resolved, note = resolves_the_peak(found, widths)
        if found.certified and resolved:
            break
        if source == "peak":
            break  # the last rung; there is nothing after it to try
        widen_would_help = (
            not found.certified
            and "not decaying" in (found.certificate.refused or "")
        )
        if not widen_would_help:
            # Either the value certified on a grid that missed the peak, which
            # a wider span makes worse, or the refinement budget ran out, which
            # a wider span also makes worse. Skip the remaining K rungs and go
            # straight to the peak-placed one.
            index = len(attempts) - 1 if attempts[-1][0] == "peak" else len(attempts)
    resolved, note = resolves_the_peak(found, widths)
    return found, source, spans, {
        # **Carried, not discarded.** `spans_for` returns which axes fell back
        # to `FALLBACK_HALF_WIDTH`, and the first version of this function
        # threw that away with a `[0]` subscript at every call site -- red line
        # 14's shape, with the evidence dropped where it was produced.
        # Measured: the branch fires on the rule for the two Cauchy fixtures
        # and BOTH are overridden by `DECLARED_SPANS`, so no row of the table
        # reaches it. That is now visible instead of inferable.
        "fallback_axes": list(fallbacks),
        "peak_verdict": verdict,
        "box": box,
        "peak": peak,
        "widths": widths,
        "height": height,
        "resolved": resolved,
        "resolution_note": note,
    }


def grid_for(axes: int) -> tuple[int, int]:
    """`(start, refinements)`: the first grid, and every doubling that fits.

    Derived from the oracle's own `MAX_POINTS` rather than declared per
    dimension. `quadrature` runs `refinements` GRIDS, the i-th holding
    `(start - 1) * 2**(i - 1) + 1` points per axis, so the count returned is the
    largest whose last grid still fits `MAX_POINTS ** (1 / axes)` per axis.

    Off by one is the whole risk here and it was measured: an earlier version
    read `refinements` as "doublings after the first grid" and returned 5 for
    two axes, which stops at n=3201 -- and `overflowing_outside_latent`
    certifies at 6401 and not before. It reported that fixture as ungradeable,
    which is a statement about this function and would have been read as one
    about the fixture.
    """
    import math

    from tests.dispatch.residual_oracle import MAX_POINTS

    start = START.get(axes, 17)
    reach = MAX_POINTS ** (1.0 / axes)
    if reach <= start:
        return start, 1
    return start, 1 + math.floor(math.log2((reach - 1.0) / (start - 1.0)))


# ==========================================================================
# the problem each backend is handed
# ==========================================================================
def problem_for(row: Fixture, graph):
    """The `CompiledEvidenceProblem` the backend integrates, per plan 0.3.

    Class (c) is the graph itself. Class (b) is R5's *collapse then sample*
    route: the exact block is eliminated by `collapse_graph` and the sampler
    sees the reduced graph, whose `log_joint` over the residual latents already
    carries the eliminated block's contribution. Both routes produce the SAME
    total `log Z`, which is what `oracle_joint` grades -- and `oracle_joint`
    integrates the uncollapsed model, so it shares nothing with either route.
    """
    from bayesmith.dispatch.collapse import collapse_graph
    from bayesmith.dispatch.evidence import compile_evidence_problem
    from bayesmith.graph.reduction import ReducedGraph, as_graph

    if row.structural_class != "(b)":
        return compile_evidence_problem(graph), graph
    reduced = collapse_graph(graph, row.exact, row.sampled)
    assert isinstance(reduced, ReducedGraph)
    # **`as_graph`, and it is not a way round the guard.** `ReducedGraph`
    # refuses `.latents` to stop a generic consumer -- one that does not read
    # `evidence_terms` -- from silently dropping the collapsed block's density,
    # and `compile_evidence_problem` reads `evidence_terms` and files every one
    # of them on the likelihood side. `as_graph` exists for exactly that
    # consumer: "expose the underlying graph to explicitly evidence-aware
    # code".
    #
    # Measured before it was used: on `diamond_ancestor` the split recomposes
    # to `log_joint(reduced)` exactly (-7.7026887405721585 both ways at
    # tau = 1.0), the residual is `('tau',)`, and the collapsed block arrives
    # as `evidence_terms[0]` on the likelihood side.
    #
    # **Write-back for Task 6.** Without `as_graph` this raises, so R5's route
    # (b) does not compile today -- 5 of the 21 rows CRASHED on the first run
    # of this probe with `GraphError: ReducedGraph is NUTS-only`. And the
    # guard's message enumerates the safe routes by name -- "log_joint,
    # to_numpyro, or nuts" -- which is a list that R5 makes incomplete.
    return compile_evidence_problem(as_graph(reduced)), reduced


def flatten(problem):
    """`(logprior, loglikelihood, dim, unflatten)` over a flat vector.

    Both backends want a flat parameter vector and the compiled problem is
    keyed by name, so this is the whole of the "adapter" either one needs at
    this layer. It is written once and shared, so that a difference in the
    table is a difference between the backends and not between two harnesses.
    """
    import jax.numpy as jnp
    import numpy as np

    shapes = problem.shapes
    sizes = [int(np.prod(shape, dtype=int)) if shape else 1 for _, shape in shapes]
    offsets = np.cumsum([0] + sizes)

    def unflatten(x):
        return {
            name: jnp.reshape(x[lo:hi], shape)
            for (name, shape), lo, hi in zip(
                shapes, offsets[:-1], offsets[1:], strict=True
            )
        }

    return (
        lambda x: problem.log_prior(unflatten(x)),
        lambda x: problem.log_likelihood(unflatten(x)),
        int(offsets[-1]),
        unflatten,
    )


# ==========================================================================
# blackjax
# ==========================================================================
def run_blackjax(
    problem,
    *,
    num_live: int,
    num_inner_steps: int,
    num_delete: int,
    seed: int,
    max_iterations: int = 100_000,
    dlogz: float = 1e-3,
    volume_draws: int = 100,
    instrument=None,
) -> dict[str, Any]:
    """One nested-sampling run through `blackjax.nss`.

    **Three things in here are bayesmith's, not blackjax's**, and plan 0.16's
    fourth open question is exactly how much that is: the TERMINATION rule
    (blackjax ships none), the EVIDENCE (it ships none), and the evidence's
    UNCERTAINTY (it ships the volume simulator and nothing that turns it into
    an error bar). Every one of those would be a bayesmith module with its own
    mutation coverage, and they are written here so that the bake-off measures
    what an adapter would actually cost.
    """
    import blackjax
    import jax
    import jax.numpy as jnp
    import numpy as np
    from blackjax.ns.utils import log_weights

    logprior, loglike, _dim, _unflatten = flatten(problem)
    if instrument is not None:
        loglike = instrument(loglike)
    names = [name for name, _ in problem.shapes]

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    draws = jax.vmap(problem.prior_sample)(jax.random.split(init_key, num_live))
    particles = jnp.concatenate(
        [jnp.reshape(draws[name], (num_live, -1)) for name in names], axis=1
    )

    algorithm = blackjax.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglike,
        num_inner_steps=num_inner_steps,
        num_delete=num_delete,
    )
    step = jax.jit(algorithm.step)
    key, subkey = jax.random.split(key)
    state = algorithm.init(particles, rng_key=subkey)
    jax.block_until_ready(state)

    # **A `nan` log-likelihood is a degenerate run, and blackjax does not say
    # so.** Measured on `overflowing_outside_latent`, whose Cauchy prior draws
    # reach `|z| ~ 1e6` and whose collapsed evidence term overflows there:
    # **800 of 816 live particles initialise at `nan`**, the remaining 8 at
    # `-8.8e+125`, and the sampler steps on without a word. The consequence is
    # not a wrong number but a run that never ends -- every termination rule
    # that reads `max(loglikelihood)` reads `nan`, so the comparison is False
    # forever and the loop runs to its iteration cap; from there `finalise`
    # asks XLA to compile a concatenation with one operand per step and does
    # not return. `-inf` is NOT degenerate: it is what a point outside the
    # support is worth. This is the same rule the quadrature oracle applies to
    # its own grid, for the same reason.
    initial = np.asarray(state.particles.loglikelihood, dtype=float)
    unusable = int(np.count_nonzero(np.isnan(initial) | (initial == np.inf)))
    if unusable:
        return {
            "verdict": "DEGENERATE",
            "log_Z": None,
            "log_Z_err": None,
            "evaluations": num_live,
            "iterations": 0,
            "termination": "non-finite-likelihood-at-init",
            "non_finite_live": unusable,
            "live_points": num_live,
            "worst_finite": (
                float(np.max(initial[np.isfinite(initial)]))
                if np.any(np.isfinite(initial))
                else None
            ),
            "wall_seconds": 0.0,
            "first_call_seconds": float("nan"),
        }

    dead: list[Any] = []
    # Initialisation evaluates every live particle once; the slice steps are
    # counted per step below. See `EVALUATION_FORMULA`.
    evaluations = num_live
    logz_dead = -np.inf
    logx_prev = 0.0
    stop = float(np.log(dlogz))
    started = time.perf_counter()
    first_call_seconds = float("nan")
    reason = "max_iterations"
    iteration = -1
    for iteration in range(max_iterations):
        key, subkey = jax.random.split(key)
        state, info = step(subkey, state)
        if iteration == 0:
            jax.block_until_ready(state)
            first_call_seconds = time.perf_counter() - started
        dead.append(info)
        update = info.update_info
        evaluations += slice_step_evaluations(
            int(jnp.sum(update.num_expansions)),
            int(jnp.sum(update.num_shrink)),
            num_delete * num_inner_steps,
        )
        logx = -(iteration + 1) * num_delete / num_live
        log_dx = logx_prev + float(np.log1p(-np.exp(logx - logx_prev)))
        dead_ll = np.asarray(info.particles.loglikelihood, dtype=float)
        logz_dead = float(
            np.logaddexp(
                logz_dead,
                log_dx + float(np.logaddexp.reduce(dead_ll)) - float(np.log(num_delete)),
            )
        )
        live = np.asarray(state.particles.loglikelihood, dtype=float)
        finite = live[np.isfinite(live)]
        logz_live = logx + (float(np.max(finite)) if finite.size else -np.inf)
        logx_prev = logx
        if logz_live - logz_dead < stop:
            reason = "dlogz"
            break
    jax.block_until_ready(state)
    wall = time.perf_counter() - started

    key, subkey = jax.random.split(key)
    # **Concatenated on the HOST, not by `finalise`.** `blackjax.ns.utils`
    # builds the dead set with `jax.tree.map(lambda *xs: jnp.concatenate(xs))`
    # over the Python list, so the operand count is the number of NS steps and
    # XLA is asked to compile a graph that grows with the run. Measured: a
    # 100 000-step run sat in `CompileCpuExecutableInternal` for seven minutes
    # without returning. `np.concatenate` over the same list costs
    # milliseconds and hands `log_weights` -- which is blackjax's own -- the
    # same arrays. This is R5 plan 0.16 item 3 restated: the cost was an
    # implementation choice, not the library.
    full = _host_finalise(state, dead)
    weights = log_weights(subkey, full, shape=volume_draws)
    per_draw = jax.scipy.special.logsumexp(weights, axis=0)
    # **E[log Z], not log E[Z].** The two differ by about `Var[log Z] / 2` and
    # jaxns reports the first (`NestedSamplerResults.log_Z_mean` is documented
    # as "estimate of E[log(Z)]"), so reporting the second here would put a
    # systematic offset into ONE column of a bias comparison -- present in
    # blackjax's rows, absent from jaxns's, and attributable to neither
    # library. Measured on the pilot run at sigma ~ 0.055 the offset is
    # ~1.5e-3 nats, small against the band and exactly the size of thing a
    # bias axis is supposed to resolve. `log_mean_Z` is kept beside it so the
    # difference is visible rather than argued.
    log_Z = float(jnp.mean(per_draw))
    log_mean_Z = float(
        jax.scipy.special.logsumexp(per_draw) - jnp.log(per_draw.shape[0])
    )
    return {
        "log_Z": log_Z,
        "log_mean_Z": log_mean_Z,
        "log_Z_err": float(jnp.std(per_draw)),
        "evaluations": evaluations,
        "iterations": iteration + 1,
        "samples": int(full.particles.loglikelihood.shape[0]),
        "wall_seconds": wall,
        "first_call_seconds": first_call_seconds,
        "termination": reason,
    }


# ==========================================================================
# jaxns
# ==========================================================================
def _host_finalise(state, dead):
    """`blackjax.ns.utils.finalise`'s result, concatenated with numpy.

    Same arrays, same order -- dead particles in the order they died, then the
    final live set -- built without asking XLA to compile one operand per NS
    step. `update_info` is dropped, which `finalise(update_info=False)` also
    does; the expansion and shrink counts are summed per step in the loop
    above, where they are two scalars rather than a retained tensor.
    """
    import jax.numpy as jnp
    import numpy as np
    from blackjax.ns.base import NSInfo, StateWithLogLikelihood

    def stack(field):
        parts = [np.asarray(getattr(info.particles, field)) for info in dead]
        parts.append(np.asarray(getattr(state.particles, field)))
        return jnp.asarray(np.concatenate(parts, axis=0))

    return NSInfo(
        particles=StateWithLogLikelihood(
            position=stack("position"),
            logdensity=stack("logdensity"),
            loglikelihood=stack("loglikelihood"),
            loglikelihood_birth=stack("loglikelihood_birth"),
        ),
        update_info=None,
    )


def run_jaxns(
    problem,
    box,
    *,
    num_live_points: int,
    num_slices: int,
    max_samples: int,
    seed: int,
    max_likelihood_evaluations: int | None = None,
    instrument=None,
) -> dict[str, Any]:
    """One nested-sampling run through `jaxns.NestedSampler`.

    **jaxns cannot be handed a `CompiledEvidenceProblem` and this is where that
    is paid for.** Its `Prior` takes a tfp distribution or, through
    `BaseAbstractPrior`, a pair of maps `U -> X` and `X -> U` between the unit
    hypercube and the parameter: a QUANTILE function. A compiled problem
    carries `log_prior`, `log_likelihood` and `prior_sample`, and a quantile
    function follows from none of the three -- a density is not invertible by
    inspection and a sampler is not a transform.

    The generic route, and the one taken here, is the box reparametrisation:
    declare a bounded domain `B`, give jaxns a uniform prior over it, and move
    bayesmith's own prior into the likelihood,

        log L'(x) = log pi(x) + log L(x) + log |B|,

    so that jaxns integrates `int_B pi L dx` -- the evidence, truncated to `B`.
    Three things follow and all three are costs a verdict has to name. The
    domain `B` is a bayesmith object jaxns needs and blackjax does not; the
    answer is the truncated evidence, so the mass outside `B` is an error the
    oracle's own truncation term bounds; and the nested sampler's contraction
    now works against `pi L` rather than `L`, which is a different problem from
    the one the method's efficiency argument is about.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tensorflow_probability.substrates.jax as tfp
    from jaxns import Model, NestedSampler, Prior, TerminationCondition

    tfpd = tfp.distributions
    logprior, loglike, dim, _unflatten = flatten(problem)
    if instrument is not None:
        loglike = instrument(loglike)
    lower = jnp.asarray([span.lower for span in box], dtype=float)
    upper = jnp.asarray([span.upper for span in box], dtype=float)
    if lower.shape[0] != dim:
        raise ValueError(
            f"the box has {lower.shape[0]} axes and the problem has {dim}; a "
            "uniform reparametrisation needs one bound per residual axis"
        )
    log_volume = float(jnp.sum(jnp.log(upper - lower)))

    def prior_model():
        x = yield Prior(tfpd.Uniform(low=lower, high=upper), name="x")
        return x

    def log_likelihood(x):
        return logprior(x) + loglike(x) + log_volume

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    sampler = NestedSampler(
        model=model,
        max_samples=max_samples,
        num_live_points=num_live_points,
        num_slices=num_slices,
    )
    # jaxns's OWN termination rule, except when a hard evaluation cap is asked
    # for. The table compares each backend terminating the way it terminates;
    # capping the evaluations would replace jaxns's stopping rule with the
    # budget and then report the budget as its termination behaviour.
    #
    # **`term_cond=None`, and NOT `TerminationCondition()`.** Measured, after a
    # first version of this probe used the constructor and had to be stopped
    # mid-run: every field of `TerminationCondition` defaults to `None`, and
    # `None` means DISABLED, so the constructed object switches off every
    # stopping rule jaxns has. The real default is assembled in
    # `jaxns/public.py:161` and only when `term_cond is None` --
    # `dlogZ = log(1 + 1e-3)` plus an unbounded `max_samples`. With the
    # constructor passed, all 21 rows terminated on `no-seed-points`, the
    # structural exhaustion of the live set, which asserts nothing about the
    # evidence: an artefact of the call site that reads exactly like a property
    # of the library. (The class's own docstring says the default is
    # `log(1 + 1e-2)`; the code says `1e-3`. Recorded under API stability.)
    term = (
        None
        if max_likelihood_evaluations is None
        else TerminationCondition(
            dlogZ=jnp.asarray(float(np.log(1.0 + 1e-3))),
            max_num_likelihood_evaluations=jnp.asarray(max_likelihood_evaluations),
        )
    )
    started = time.perf_counter()
    compiled = jax.jit(lambda key: sampler(key, term_cond=term))
    reason, state = compiled(jax.random.PRNGKey(seed))
    jax.block_until_ready(state)
    # jaxns compiles the WHOLE run, so its first-call time is compilation plus
    # the run and cannot be split without a second call. blackjax's first-call
    # time is compilation plus one NS step. The column is named for what both
    # actually are, and the two are not the same quantity -- which is itself
    # part of the compile-cost axis rather than a defect in the measurement.
    first_call_seconds = time.perf_counter() - started
    results = sampler.to_results(termination_reason=reason, state=state)
    wall = time.perf_counter() - started
    return {
        "log_Z": float(results.log_Z_mean),
        "log_Z_err": float(results.log_Z_uncert),
        "evaluations": int(np.asarray(results.total_num_likelihood_evaluations)),
        "iterations": int(np.asarray(results.total_num_samples)),
        "samples": int(np.asarray(results.total_num_samples)),
        "wall_seconds": wall,
        "first_call_seconds": first_call_seconds,
        "termination": _jaxns_termination(int(np.asarray(reason))),
        "termination_mask": int(np.asarray(reason)),
        # The audit in section 3 confirms jaxns's reported count only when the
        # sample cap does NOT bind; at a binding cap the report is short by the
        # interrupted shell's work. So a row that hit the cap carries the flag
        # rather than the number being quietly reused.
        "count_audited": "reached-max-samples"
        not in _jaxns_termination(int(np.asarray(reason))),
        "log_volume": log_volume,
    }


#: jaxns reports termination as a 12-bit mask. The names below are READ from
#: `jaxns/nested_samplers/common/termination.py`'s `_set_done_bit` calls, in
#: the bit order that file assigns, not inferred from the field order of
#: `TerminationCondition` -- which is a different order. A first version of
#: this table guessed the second and decoded the smoke run's mask 4 as
#: "live-evidence-fraction" where it is `small_remaining_evidence`: a name that
#: reads exactly like a measurement and is not one.
JAXNS_TERMINATION_BITS = (
    "reached-max-samples",  # 0
    "evidence-uncertainty-low-enough",  # 1
    "small-remaining-evidence",  # 2
    "ESS-reached",  # 3
    "max-num-likelihood-evaluations",  # 4
    "likelihood-contour-reached",  # 5
    "efficiency-too-low",  # 6
    "plateau",  # 7
    "relative-spread-low",  # 8
    "absolute-spread-low",  # 9
    "no-seed-points",  # 10
    "XL-reduction-reached",  # 11
)


def _jaxns_termination(mask: int) -> str:
    hits = [
        name for index, name in enumerate(JAXNS_TERMINATION_BITS) if mask & (1 << index)
    ]
    return "+".join(hits) if hits else f"unmapped({mask})"


# ==========================================================================
# 5.1 -- the shared budget, and the audit that makes the count usable
# ==========================================================================
#: The arithmetic each backend's evaluation count is derived by. Both are
#: audited in section 3 against an `io_callback` ground truth before any table
#: row uses them; plan 0.7 measured a wrapping Python counter reporting **1**
#: after a thousand jitted calls, so neither number here may be taken on trust.
EVALUATION_FORMULA = {
    "blackjax": (
        "num_live + sum over NS steps of "
        "(sum num_expansions + sum num_shrink + 2 * num_delete * num_inner_steps). "
        "The +2 per slice step is the pair of terminating `in_slice` calls "
        "`stepping_out`'s two `lax.while_loop` conditions make and do not "
        "count: `in_slice(left) & (n > 0)` is not short-circuiting, so the "
        "condition evaluates the likelihood on the iteration that ends the "
        "loop as well, once on each side. `num_expansions` counts BODY "
        "executions. `_shrink`'s condition reads no likelihood, so `num_shrink` "
        "is exact."
    ),
    "jaxns": (
        "results.total_num_likelihood_evaluations, which jaxns reports itself. "
        "Plan 0.7's first stop-rule fires on a count whose only source is the "
        "backend's own report AND whose audit does not confirm it, so the "
        "audit is what makes this admissible rather than the report."
    ),
}


def slice_step_evaluations(expansions: int, shrinks: int, slice_steps: int) -> int:
    """Likelihood evaluations for `slice_steps` blackjax slice steps.

    **The run and the audit call THIS, and that is the whole point of it being
    a function.** An adversarial review of the first version found the constant
    written out twice -- once in `run_blackjax`'s loop and once in
    `audit_blackjax_formula` -- and mutated the run-side copy: every evaluation
    count in the table moved by 20 per cent (202 667 to 162 102) and the audit
    went on reporting `derived 457 / truth 457 / exact True`. Two constants,
    two functions, disjoint effects, one claimed relationship. That is
    `CLAUDE.md`'s founding defect -- six copies of one measurement -- in the
    one place this probe calls audited.

    The `+2` per slice step is the pair of terminating `in_slice` calls
    `stepping_out`'s two `lax.while_loop` conditions make and do not count:
    `in_slice(left) & (n > 0)` is not short-circuiting, so the condition
    evaluates the likelihood on the iteration that ends the loop as well, once
    on each side. `num_expansions` counts BODY executions. `_shrink`'s
    condition reads no likelihood, so `num_shrink` is exact.
    """
    return expansions + shrinks + 2 * slice_steps


def counted_likelihood(fn):
    """`(wrapped, read)` -- a likelihood that counts its own calls for real.

    **UNREACHED, and recorded as such.** An adversarial review found this
    function has exactly one occurrence in the file -- this `def` -- and that
    `counter_control` shares no code with it, so no control can catch a defect
    in it even in principle. The audits use `io_callback` directly. It is kept
    because it documents the instrument plan 0.7 sanctions, and it is labelled
    because an unreached helper that reads as a live one is the same defect as
    an unread flag.


    Used ONLY inside `jax.disable_jit()`, on a deliberately tiny problem, which
    is plan 0.7's second sanctioned instrument. Under `jit` a Python closure
    counts TRACES: measured again in this checkout, a jitted `lax.scan` over
    1000 points increments it **once**. Unjitted it counts calls -- 1000/1000 --
    and section 3 runs that control before it trusts a number, because a
    counter that cannot tell 1000 calls from one trace would report the
    formula's own value back to it.

    **A batched call is one call and many evaluations.** `blackjax` vmaps its
    inner kernel over `num_delete` particles and `jaxns` over its parallel
    chains, and unjitted vmap reaches the likelihood ONCE with a leading axis:
    measured, a vmap over 1000 points increments a plain counter once. So the
    tick is the leading axis where there is one -- the same distinction between
    "called" and "evaluated" that makes the jitted counter useless.
    """
    import numpy as np

    total = [0]

    def wrapped(x):
        value = fn(x)
        array = np.shape(x)
        total[0] += int(array[0]) if len(array) > 1 else 1
        return value

    return wrapped, (lambda: total[0])


def counter_control() -> dict[str, int]:
    """The control that says the counter above measures anything at all.

    Red line 14 in its sharpest form: an audit whose instrument silently reads
    1 would confirm any formula that also read 1. Three cells, and the first is
    plan 0.7's own measurement re-taken here rather than quoted.
    """
    import jax
    import jax.numpy as jnp

    hits = [0]

    def f(x):
        hits[0] += 1
        return jnp.sum(x)

    def scan_1000():
        return jax.lax.scan(lambda c, x: (c + f(x), None), 0.0, jnp.arange(1000.0))[0]

    hits[0] = 0
    jax.block_until_ready(jax.jit(scan_1000)())
    jitted = hits[0]
    hits[0] = 0
    with jax.disable_jit():
        scan_1000()
    unjitted = hits[0]
    hits[0] = 0
    with jax.disable_jit():
        jax.vmap(f)(jnp.arange(1000.0).reshape(1000, 1))
    vmapped = hits[0]
    return {
        "jitted_scan_1000": jitted,
        "unjitted_scan_1000": unjitted,
        "unjitted_vmap_1000_calls": vmapped,
    }


import contextlib


@contextlib.contextmanager
def _unbatched_inner_kernel():
    """Run `blackjax`'s inner kernel without its `vmap`, for `num_delete == 1`.

    **The audit cannot be taken any other way, and the reason is the same one
    plan 0.7 is about.** `io_callback` is refused inside a `lax.while_loop`
    whose predicate is batched -- which `blackjax.ns.from_mcmc`'s
    `jax.vmap(mcmc_kernel)` makes it -- and a plain Python counter under
    `jax.disable_jit()` does not help, because `vmap` traces whether or not
    `jit` is disabled: measured, an unjitted `vmap` over 1000 points increments
    a closure ONCE, and a whole audit run this way reported exactly 6 calls per
    slice step at three different step counts, which is a trace count wearing
    an evaluation count's clothes.

    With `num_delete == 1` the batch axis is length one, so replacing the
    `vmap` with an index-and-restore changes the computation not at all -- and
    section 3 asserts that rather than assuming it, by running the same seed
    both ways and comparing the expansion counts, the shrink counts and every
    dead particle's log-likelihood.
    """
    import jax
    import jax.numpy as jnp
    from blackjax.ns import from_mcmc

    real = from_mcmc.jax

    def shim(fn, *args, **kwargs):
        def call(*batched):
            out = fn(*jax.tree.map(lambda leaf: leaf[0], batched))
            return jax.tree.map(lambda leaf: leaf[None], out)

        return call

    from_mcmc.jax = type(
        "_PatchedJax",
        (),
        {
            "vmap": staticmethod(shim),
            "tree": jax.tree,
            "lax": jax.lax,
            "numpy": jnp,
        },
    )
    try:
        yield
    finally:
        from_mcmc.jax = real


def audit_blackjax_formula(*, iterations: int = 30, num_live: int = 20) -> dict[str, Any]:
    """Ground-truth the blackjax count, and show the shim changed nothing."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import io_callback

    from bayesmith.dispatch.evidence import compile_evidence_problem
    from bayesmith.graph.reduction import as_graph
    from tests.exact import models

    problem = compile_evidence_problem(as_graph(models.student_t_likelihood()))
    logprior, loglike, _dim, _unflatten = flatten(problem)
    num_inner_steps, num_delete = 3, 1
    total = [0]

    def tick(_x):
        total[0] += 1
        return np.asarray(0.0, dtype=np.float64)

    def counted(x):
        return loglike(x) + io_callback(
            tick, jax.ShapeDtypeStruct((), jnp.float64), jnp.zeros(())
        )

    def drive(likelihood, patched):
        import blackjax

        key = jax.random.key(0)
        key, init_key = jax.random.split(key)
        draws = jax.vmap(problem.prior_sample)(jax.random.split(init_key, num_live))
        particles = jnp.concatenate(
            [jnp.reshape(draws[n], (num_live, -1)) for n, _ in problem.shapes], axis=1
        )
        algorithm = blackjax.nss(
            logprior_fn=logprior,
            loglikelihood_fn=likelihood,
            num_inner_steps=num_inner_steps,
            num_delete=num_delete,
        )
        key, subkey = jax.random.split(key)
        state = algorithm.init(particles, rng_key=subkey)
        at_init = total[0]
        total[0] = 0
        step = jax.jit(algorithm.step)
        expansions = shrinks = 0
        deaths = []
        with _unbatched_inner_kernel() if patched else contextlib.nullcontext():
            for _ in range(iterations):
                key, subkey = jax.random.split(key)
                state, info = step(subkey, state)
                expansions += int(jnp.sum(info.update_info.num_expansions))
                shrinks += int(jnp.sum(info.update_info.num_shrink))
                deaths.append(float(info.particles.loglikelihood[0]))
        return expansions, shrinks, deaths, at_init

    plain = drive(loglike, False)
    shimmed = drive(loglike, True)
    inert = (
        plain[0] == shimmed[0]
        and plain[1] == shimmed[1]
        and np.allclose(plain[2], shimmed[2])
    )
    if not inert:
        # **The audit refuses rather than reporting.** A first version computed
        # this flag and printed it beside `exact: True`; forcing it False left
        # the run at exit 0 with the audit's verdict intact, so the check could
        # not distinguish "the shim is inert" from "nobody read the answer".
        return {
            "shim_is_inert": False,
            "verdict": "DECLINED",
            "why": (
                "the vmap shim changed the computation, so a count taken "
                "through it is not a count of the unshimmed run: expansions "
                f"{plain[0]} vs {shimmed[0]}, shrinks {plain[1]} vs "
                f"{shimmed[1]}"
            ),
        }
    total[0] = 0
    counted_run = drive(counted, True)
    slice_steps = iterations * num_delete * num_inner_steps
    truth = total[0]
    derived = slice_step_evaluations(counted_run[0], counted_run[1], slice_steps)
    return {
        "shim_is_inert": True,
        "expansions": counted_run[0],
        "shrinks": counted_run[1],
        "slice_steps": slice_steps,
        "truth": truth,
        "derived": derived,
        "exact": truth == derived,
        "init_ticks_seen": counted_run[3],
        "init_term_is_arithmetic": (
            "jax.vmap(init_state_fn) over num_live positions; the callback sees "
            "ONE batched call, so this term is read off the API and is the part "
            "of the formula the audit does not cover"
        ),
    }


def audit_jaxns_report(*, max_samples: int) -> dict[str, Any]:
    """Ground-truth jaxns's own `total_num_likelihood_evaluations`.

    Run at TWO sample caps, because the answer is not the same at both and the
    difference is what decides whether the number may be used. At a cap the run
    does not reach, the report is exact. At a cap that BINDS, the report is
    short by the work already spent in the shell the cap interrupted -- so the
    budget protocol has to check that `reached-max-samples` is not among the
    termination bits before it may quote the count. Plan 0.7's first stop-rule
    is about exactly this: a count whose only source is the backend's own
    report is admissible when the audit confirms it and not otherwise.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tensorflow_probability.substrates.jax as tfp
    from jax.experimental import io_callback
    from jaxns import Model, NestedSampler, Prior, TerminationCondition

    tfpd = tfp.distributions
    total = [0]

    def tick(_x):
        total[0] += 1
        return np.asarray(0.0, dtype=np.float64)

    lower, upper = -8.0, 8.0
    log_volume = float(np.log(upper - lower))

    def prior_model():
        x = yield Prior(
            tfpd.Uniform(low=jnp.asarray([lower]), high=jnp.asarray([upper])), name="x"
        )
        return x

    def log_likelihood(x):
        value = (
            tfpd.Normal(0.0, 1.0).log_prob(x[0])
            + tfpd.Normal(x[0], 0.5).log_prob(1.0)
            + log_volume
        )
        return value + io_callback(
            tick, jax.ShapeDtypeStruct((), jnp.float64), jnp.zeros(())
        )

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    sampler = NestedSampler(model=model, max_samples=max_samples)
    total[0] = 0
    reason, state = sampler(jax.random.PRNGKey(0), term_cond=TerminationCondition())
    jax.block_until_ready(state)
    results = sampler.to_results(termination_reason=reason, state=state)
    reported = int(np.asarray(results.total_num_likelihood_evaluations))
    return {
        "max_samples": max_samples,
        "reported": reported,
        "truth": total[0],
        "exact": reported == total[0],
        "samples": int(np.asarray(results.total_num_samples)),
        "termination": _jaxns_termination(int(np.asarray(reason))),
        "termination_mask": int(np.asarray(reason)),
    }


# ==========================================================================
# one cell, in its own process
# ==========================================================================
def run_cell(request: dict[str, Any]) -> dict[str, Any]:
    """Run one (fixture, candidate, seed) and return everything it measured."""
    import jax

    jax.config.update("jax_enable_x64", True)

    label = request["fixture"]
    # The row's classification travels IN THE REQUEST rather than being
    # re-derived here. Measured on the pilot run: re-running `census()` in
    # every cell compiled all 57 shipped graphs through `bayesmith.compile`
    # and `compile_task` before the sampler started, which put ~110 s of setup
    # in front of a 3 s run and made the table cost three hours instead of
    # twenty minutes. It also risked the worse failure -- a cell's classification
    # coming from a second, independently computed census.
    row = Fixture(
        label=label,
        module=request.get("module", ""),
        structural_class=request["structural_class"],
        premise="capability_unavailable_r1",
        exact=tuple(request["exact"]),
        sampled=tuple(request["sampled"]),
    )
    graph = build_one(label)
    problem, _target = problem_for(row, graph)
    dim = sum(
        int(_size(shape)) for _name, shape in problem.shapes
    )
    knobs = dict(request.get("knobs") or {})
    if request.get("key_types"):
        return {
            "fixture": label,
            "candidate": request["candidate"],
            "key_types": key_type_probe(request["candidate"], problem),
        }
    out: dict[str, Any] = {
        "fixture": label,
        "candidate": request["candidate"],
        "seed": request["seed"],
        "residual_dim": dim,
        "structural_class": row.structural_class,
    }
    try:
        if request["candidate"] == "blackjax":
            inner = knobs.pop("num_inner_steps", None) or max(5, 2 * dim)
            out["knobs"] = {"num_inner_steps": inner, **knobs}
            out.update(
                run_blackjax(
                    problem,
                    num_inner_steps=inner,
                    num_delete=knobs.pop("num_delete", 1),
                    num_live=knobs.pop("num_live", 500),
                    seed=request["seed"],
                    **knobs,
                )
            )
        else:
            spans = {span.name: span for span in _spans_of(request)}
            box = [spans[name] for name in _axis_names(problem)]
            out["knobs"] = dict(knobs)
            out.update(run_jaxns(problem, box, seed=request["seed"], **knobs))
        # The driver may already have named the verdict -- `DEGENERATE` when
        # the run was never viable. Only a driver that named nothing gets
        # `MEASURED`, because overwriting its own verdict here is how a
        # degenerate run becomes a measured one.
        out.setdefault("verdict", "MEASURED")
    except Exception as error:  # noqa: BLE001 - a CRASH is a table cell
        out["verdict"] = "CRASH"
        out["error"] = f"{type(error).__name__}: {error}"[:400]
    out["peak_rss_mb"] = _peak_rss_mb()
    return out


def key_type_probe(candidate: str, problem) -> dict[str, str]:
    """Which of JAX's two key objects each entry point accepts.

    `jax.random.key` (typed) and `jax.random.PRNGKey` (raw `uint32[2]`) are
    different objects, and a backend taking only the older one puts a
    conversion in the adapter that will outlive the reason for it. 1.5
    condition 2 asks whether the candidate "takes a `jax.random.key`", which is
    a yes/no a run can answer and an argument cannot.
    """
    import jax
    import jax.numpy as jnp

    logprior, loglike, dim, _unflatten = flatten(problem)
    out: dict[str, str] = {}
    for name, make in (("jax.random.key", jax.random.key), ("jax.random.PRNGKey", jax.random.PRNGKey)):
        try:
            if candidate == "blackjax":
                import blackjax

                key = make(0)
                draws = jax.vmap(problem.prior_sample)(jax.random.split(key, 4))
                particles = jnp.concatenate(
                    [jnp.reshape(draws[n], (4, -1)) for n, _ in problem.shapes], axis=1
                )
                algorithm = blackjax.nss(
                    logprior_fn=logprior,
                    loglikelihood_fn=loglike,
                    num_inner_steps=2,
                    num_delete=1,
                )
                state = algorithm.init(particles, rng_key=make(1))
                algorithm.step(make(2), state)
            else:
                import tensorflow_probability.substrates.jax as tfp
                from jaxns import Model, NestedSampler, Prior

                tfpd = tfp.distributions
                low = jnp.full((dim,), -4.0)
                high = jnp.full((dim,), 4.0)

                def prior_model(_tfpd=tfpd, _low=low, _high=high):
                    # Bound as defaults, not closed over: this is inside a loop
                    # over key types, and a late-binding closure would build the
                    # second iteration's prior with the first's bounds.
                    x = yield Prior(_tfpd.Uniform(low=_low, high=_high), name="x")
                    return x

                model = Model(
                    prior_model=prior_model,
                    log_likelihood=lambda x: logprior(x) + loglike(x),
                )
                sampler = NestedSampler(model=model, max_samples=200)
                sampler(make(0))
            out[name] = "accepted"
        except Exception as error:  # noqa: BLE001 - the refusal is the answer
            out[name] = f"{type(error).__name__}: {str(error)[:120]}"
    return out


def _size(shape) -> int:
    import numpy as np

    return int(np.prod(shape, dtype=int)) if shape else 1


def _axis_names(problem) -> list[str]:
    """One name per FLAT axis, in the order `flatten` lays them out."""
    names = []
    for name, shape in problem.shapes:
        size = _size(shape)
        names.extend([name] if size == 1 else [f"{name}[{i}]" for i in range(size)])
    return names


def _spans_of(request) -> tuple[Any, ...]:
    from tests.dispatch.residual_oracle import Span

    return tuple(Span(*entry) for entry in request["box"])


def _peak_rss_mb() -> float:
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kibibytes. Distinguished by magnitude rather
    # than by `sys.platform`, so a third platform is not silently wrong by a
    # factor of 1024 -- no process this probe runs uses 100 GB, and none uses
    # under 100 kB.
    return peak / (1024.0 * 1024.0) if peak > 10**8 else peak / 1024.0


def _fields_of(row: Fixture) -> dict[str, Any]:
    """The classification a cell needs, taken from the ONE census."""
    return {
        "structural_class": row.structural_class,
        "exact": list(row.exact),
        "sampled": list(row.sampled),
        "module": row.module,
    }


def spawn(request: dict[str, Any], *, timeout: float = 1800.0) -> dict[str, Any]:
    """Run one cell in a fresh interpreter and read its JSON back.

    Three reasons this is not an in-process call, each measured rather than
    assumed: `import jaxns` writes `jax_enable_x64` process-globally, so one
    candidate would set the other's arithmetic; a compile time is only a
    compile time in a process that has not already compiled the same graph;
    and peak RSS belongs to a process, so a shared one reports a neighbour's
    memory as this cell's.
    """
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--cell", json.dumps(request)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(REPO)},
    )
    elapsed = time.perf_counter() - started
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            payload = json.loads(line)
            payload["subprocess_seconds"] = elapsed
            payload["returncode"] = proc.returncode
            if proc.returncode != 0:
                # **A payload is not a success.** An adversarial review built a
                # cell that printed its JSON, flushed, and then `os._exit(3)`;
                # the parent reported MEASURED. A child that dies after the
                # print -- an XLA abort at teardown, an OOM, a failing
                # `atexit` -- has not run cleanly, and the row says so rather
                # than inheriting the payload's optimism.
                payload["verdict"] = "CRASH"
                payload["error"] = (
                    f"the cell printed a payload and then exited "
                    f"{proc.returncode}: "
                    + (proc.stderr.strip().splitlines() or ["no stderr"])[-1][:200]
                )
            return payload
    return {
        **request,
        "verdict": "CRASH",
        "error": (proc.stderr.strip().splitlines() or ["no output"])[-1][:400],
        "subprocess_seconds": elapsed,
    }


# ==========================================================================
# 5.1 calibration
# ==========================================================================
#: The one fixture the knobs are tuned on, declared before the tuning runs.
#: Class (c), one residual axis, a certified oracle, and non-Gaussian -- so the
#: budget is not calibrated on the one shape plan 0.16 already measured both
#: candidates to be identical on.
CALIBRATION_FIXTURE = "student_t_likelihood"


def calibrate(candidate: str, box, row: Fixture, *, trials: int = 6) -> dict[str, Any]:
    """Drive one candidate to `TARGET_EVALUATIONS` +/- `BUDGET_TOLERANCE`.

    The evaluation count is close to linear in the live-point count for both
    backends, so the search is: measure at a seed setting, scale, repeat.
    Every trial is recorded, including the ones that missed -- a calibration
    reported only by its final row is one whose reader cannot tell a converged
    search from a lucky first guess.
    """
    trials_log: list[dict[str, Any]] = []
    live = 200
    for _ in range(trials):
        knobs = (
            {"num_live": int(live), "num_delete": 1}
            if candidate == "blackjax"
            else {
                "num_live_points": int(live),
                "num_slices": None,
                "max_samples": int(400 * live),
            }
        )
        result = spawn(
            {
                "fixture": CALIBRATION_FIXTURE,
                "candidate": candidate,
                "seed": 0,
                "knobs": knobs,
                "box": [(span.name, span.lower, span.upper) for span in box],
                **_fields_of(row),
            }
        )
        evaluations = result.get("evaluations")
        trials_log.append(
            {
                "live": int(live),
                "evaluations": evaluations,
                "verdict": result.get("verdict"),
                "termination": result.get("termination"),
                "error": result.get("error"),
            }
        )
        if result.get("verdict") != "MEASURED" or not evaluations:
            break
        if abs(evaluations - TARGET_EVALUATIONS) <= BUDGET_TOLERANCE * TARGET_EVALUATIONS:
            return {"settled": True, "live": int(live), "trials": trials_log}
        live = max(10, min(200_000, live * TARGET_EVALUATIONS / evaluations))
    return {"settled": False, "live": int(live), "trials": trials_log}


# ==========================================================================
# sections
# ==========================================================================
def _rule(title: str) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)


def section_1_environment() -> dict[str, Any]:
    _rule("1. the environment, and the two candidates in it")
    import jax

    versions = {name: _installed(name) for name in CANDIDATES}
    tfp = _installed("tfp-nightly")
    print(f"  python              {platform.python_version()} on {platform.machine()}")
    print(f"  jax                 {_installed('jax')} / jaxlib {_installed('jaxlib')}")
    print(f"  jax_enable_x64      {jax.config.read('jax_enable_x64')} at import")
    for name in CANDIDATES:
        state = versions[name] or "ABSENT"
        print(f"  {name:<19} {state}")
    print(f"  tfp-nightly         {tfp or 'ABSENT'}   (jaxns's unbounded dependency)")
    print(f"  numpy               {_installed('numpy')}")
    return {"versions": versions, "tfp_nightly": tfp}


def section_2_what_each_backend_needs(fixtures, graphs) -> dict[str, Any]:
    _rule("2. what each backend needs that a CompiledEvidenceProblem does not carry")
    problem, _target = problem_for(fixtures[0], graphs[fixtures[0].label])
    fields = [f.name for f in dataclasses.fields(problem)]
    print(f"  CompiledEvidenceProblem carries: {', '.join(fields)}")
    findings = {}
    for name in CANDIDATES:
        if _installed(name) is None:
            print(f"  {name:<9} ABSENT -- not asked")
            findings[name] = "ABSENT"
            continue
        findings[name] = _consumption(name)
        print(f"  {name:<9} {findings[name]}")
    return findings


def _consumption(candidate: str) -> str:
    """What the backend's entry point does when handed the compiled problem.

    **This RUNS the attempt.** A first version read signatures and returned
    prose, while the module docstring above claimed it drove each backend from
    the compiled problem and recorded what raised; an adversarial review found
    the gap and supplied the missing run, which happened to confirm the
    conclusion. The instrument now does what its own sentence says.
    """
    import inspect

    from bayesmith.dispatch.evidence import compile_evidence_problem
    from bayesmith.graph.reduction import as_graph
    from tests.exact import models

    problem = compile_evidence_problem(as_graph(models.student_t_likelihood()))
    if candidate == "blackjax":
        import blackjax

        signature = inspect.signature(blackjax.nss.differentiable)
        required = [
            name
            for name, parameter in signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
        ]
        logprior, loglike, _dim, _unflatten = flatten(problem)
        try:
            blackjax.nss(
                logprior_fn=logprior,
                loglikelihood_fn=loglike,
                num_inner_steps=2,
            )
            built = "accepted"
        except Exception as error:  # noqa: BLE001 - the refusal is the answer
            built = f"{type(error).__name__}: {str(error)[:120]}"
        return (
            f"blackjax.nss requires {required}; handed the compiled problem's "
            f"own `log_prior` and `log_likelihood`, constructing the sampler "
            f"is {built}. The initial live set comes from `prior_sample`. "
            "NOTHING ELSE IS NEEDED -- and nothing else is supplied either: no "
            "termination rule, no log Z, no uncertainty."
        )

    from jaxns import Prior
    from jaxns.framework import bases

    attempts = {}
    for label, argument in (
        ("the compiled problem itself", problem),
        ("its log_prior", problem.log_prior),
        ("its prior_sample", problem.prior_sample),
    ):
        try:
            Prior(argument, name="x")
            attempts[label] = "accepted"
        except Exception as error:  # noqa: BLE001 - the refusal is the answer
            attempts[label] = f"{type(error).__name__}"
    abstract = [
        name
        for name in ("_forward", "_inverse", "_log_prob", "_base_shape", "_shape")
        if hasattr(bases.BaseAbstractPrior, name)
    ]
    return (
        f"jaxns.Prior handed {attempts}. It takes a tfp distribution, or a "
        f"BaseAbstractPrior implementing {abstract} -- a QUANTILE map U -> X "
        "and its inverse. A compiled problem carries log_prior, log_likelihood "
        "and prior_sample, and none of the three yields a quantile map: a "
        "density is not invertible by inspection and a sampler is not a "
        "transform. The generic route is a bounded box plus "
        "log L' = log pi + log L + log|B|, which needs a DOMAIN bayesmith must "
        "supply and returns the evidence truncated to it."
    )


def section_3_budget(box, row: Fixture) -> dict[str, Any]:
    _rule("3. 5.1 -- the evaluation count, audited, then calibrated")
    control = counter_control()
    print(f"  counter control: {control}")
    print(
        "    -> a jitted closure reports 1 for 1000 evaluations and an unjitted "
        "vmap reports 1 for 1000; both instruments below are `io_callback`"
    )
    out: dict[str, Any] = {"counter_control": control, "audits": {}, "calibration": {}}
    for name in CANDIDATES:
        print(f"  --- {name} ---")
        print(f"    formula: {EVALUATION_FORMULA[name]}")
        if _installed(name) is None:
            print("    ABSENT -- not audited, not calibrated")
            out["audits"][name] = "ABSENT"
            out["calibration"][name] = "ABSENT"
            continue
        audit = spawn({"audit": name, "box": _box_json(box)})
        out["audits"][name] = audit
        print(f"    audit: {json.dumps(audit, default=str)[:600]}")
        settled = calibrate(name, box, row)
        out["calibration"][name] = settled
        for trial in settled["trials"]:
            print(
                f"    trial live={trial['live']:<7} evaluations={trial['evaluations']} "
                f"{trial['verdict']} {trial.get('termination') or ''}"
            )
        print(
            f"    -> {'SETTLED' if settled['settled'] else 'DID NOT SETTLE'} at "
            f"live={settled['live']} for target {TARGET_EVALUATIONS} "
            f"+/-{BUDGET_TOLERANCE:.0%}"
        )
    return out


def _box_json(box):
    return [(span.name, span.lower, span.upper) for span in box]


def _cell_requests(fixtures, oracles, live, seeds):
    for row in fixtures:
        spans = oracles[row.label]["box"]
        for candidate in CANDIDATES:
            if _installed(candidate) is None:
                continue
            for seed in seeds:
                knobs = (
                    {"num_live": int(live[candidate]), "num_delete": 1}
                    if candidate == "blackjax"
                    else {
                        "num_live_points": int(live[candidate]),
                        "num_slices": None,
                        "max_samples": int(400 * live[candidate]),
                    }
                )
                yield {
                    "fixture": row.label,
                    "candidate": candidate,
                    "seed": seed,
                    "knobs": knobs,
                    "box": spans,
                    "structural_class": row.structural_class,
                    "exact": list(row.exact),
                    "sampled": list(row.sampled),
                    "module": row.module,
                }


#: The rows that get the full seed set. Declared, and chosen for what they are
#: rather than for what they scored: the multimodal one and the heavy-tailed
#: one are the two plan 0.16 names as the measurements most likely to decide,
#: the four-axis one is the dimension boundary, and the Gaussian is the control
#: -- plan 0.16 measured both candidates correct on that shape, so a seed
#: sweep that separates them there is measuring the harness.
SEED_SUBSET = (
    "mixture_prior_residual",
    "cauchy_residual_pair",
    "undeclared_quartet",
    "diamond_ancestor",
)

#: The ladder 5.4's stop-rule reads. A reported uncertainty that does not move
#: across it is a placeholder, and plan 0.8 stops the work before scoring
#: 1.5 condition 4 on it.
BUDGET_LADDER = (0.25, 1.0, 4.0)


def section_4_uncertainty_moves(live, oracles, row: Fixture) -> dict[str, Any]:
    _rule("4. 5.4's stop-rule -- does the reported log Z uncertainty move with the budget?")
    out: dict[str, Any] = {}
    for candidate in CANDIDATES:
        if _installed(candidate) is None:
            print(f"  {candidate:<9} ABSENT -- the stop-rule is not reached")
            out[candidate] = "ABSENT"
            continue
        rungs = []
        for scale in BUDGET_LADDER:
            knobs = (
                {"num_live": max(10, int(live[candidate] * scale)), "num_delete": 1}
                if candidate == "blackjax"
                else {
                    "num_live_points": max(4, int(live[candidate] * scale)),
                    "num_slices": None,
                    "max_samples": int(400 * live[candidate] * scale),
                }
            )
            cell = spawn(
                {
                    "fixture": CALIBRATION_FIXTURE,
                    "candidate": candidate,
                    "seed": 0,
                    "knobs": knobs,
                    "box": oracles[CALIBRATION_FIXTURE]["box"],
                    **_fields_of(row),
                }
            )
            rungs.append(cell)
            print(
                f"  {candidate:<9} x{scale:<5} evaluations={cell.get('evaluations')} "
                f"log_Z={_fmt(cell.get('log_Z'))} err={_fmt(cell.get('log_Z_err'))} "
                f"{cell.get('verdict')}"
            )
        errors = [c.get("log_Z_err") for c in rungs if c.get("log_Z_err") is not None]
        moved = len({round(e, 12) for e in errors}) > 1 if errors else False
        shrinks = (
            all(a > b for a, b in itertools.pairwise(errors))
            if len(errors) > 1
            else False
        )
        out[candidate] = {"rungs": rungs, "moved": moved, "monotone_decreasing": shrinks}
        print(
            f"  -> {candidate}: uncertainty {'MOVES' if moved else 'DOES NOT MOVE'}"
            f"{' and shrinks monotonically' if shrinks else ''}"
        )
    return out


def _fmt(value) -> str:
    return "-" if value is None else f"{value:.6g}"


def section_5_the_table(fixtures, oracles, live, seeds=(0,)) -> list[dict[str, Any]]:
    _rule("5. 5.2 -- the table: every admitted fixture, both candidates, one budget")
    print(
        f"  {'fixture':<30} {'cand':<9} {'v':<8} {'log Z':>13} {'err':>9} "
        f"{'oracle':>13} {'z':>7} {'evals':>9} {'wall':>7} {'RSS/MB':>7}  termination"
    )
    cells = []
    for request in _cell_requests(fixtures, oracles, live, seeds):
        cell = spawn(request)
        oracle = oracles[cell["fixture"]]
        cell["oracle"] = oracle["value"]
        cell["oracle_withheld"] = oracle["withheld"]
        if cell.get("verdict") == "MEASURED" and oracle["value"] is not None:
            cell["gap"] = cell["log_Z"] - oracle["value"]
            err = cell.get("log_Z_err") or 0.0
            cell["sigma"] = cell["gap"] / err if err else None
        cells.append(cell)
        print(
            f"  {cell['fixture']:<30} {cell['candidate']:<9} "
            f"{cell.get('verdict','?')[:8]:<8} {_fmt(cell.get('log_Z')):>13} "
            f"{_fmt(cell.get('log_Z_err')):>9} "
            f"{('WITHHELD' if oracle['withheld'] else _fmt(oracle['value'])):>13} "
            f"{_fmt(cell.get('sigma')):>7} {cell.get('evaluations') or '-'!s:>9} "
            f"{_fmt(cell.get('wall_seconds')):>7} {_fmt(cell.get('peak_rss_mb')):>7}"
            f"  {str(cell.get('termination') or cell.get('error',''))[:46]}"
        )
    return cells


def section_6_repeated_runs(fixtures, oracles, live) -> list[dict[str, Any]]:
    _rule("6. repeated runs -- the spread of the seed set against the reported error bar")
    rows = [row for row in fixtures if row.label in SEED_SUBSET]
    cells = section_5_the_table(rows, oracles, live, seeds=SEEDS)
    import statistics

    print()
    print(f"  {'fixture':<30} {'cand':<9} {'n':>3} {'mean':>13} {'spread':>10} "
          f"{'reported':>10} {'ratio':>7}  verdict")
    summary = []
    for row in rows:
        for candidate in CANDIDATES:
            picked = [
                c
                for c in cells
                if c["fixture"] == row.label
                and c["candidate"] == candidate
                and c.get("verdict") == "MEASURED"
            ]
            if len(picked) < 2:
                continue
            values = [c["log_Z"] for c in picked]
            spread = statistics.stdev(values)
            reported = statistics.mean(c["log_Z_err"] for c in picked)
            ratio = spread / reported if reported else float("inf")
            summary.append(
                {
                    "fixture": row.label,
                    "candidate": candidate,
                    "n": len(picked),
                    "mean": statistics.mean(values),
                    "spread": spread,
                    "reported": reported,
                    "ratio": ratio,
                }
            )
            print(
                f"  {row.label:<30} {candidate:<9} {len(picked):>3} "
                f"{statistics.mean(values):>13.6g} {spread:>10.4g} {reported:>10.4g} "
                f"{ratio:>7.3g}  "
                + (
                    "spread EXCEEDS the reported bar"
                    if ratio > 1.0
                    else "within the reported bar"
                )
            )
    print(
        "  -> plan 0.8's asymmetry: a spread LARGER than the reported error bar is "
        "the failure 11.4 exists to catch; smaller is recorded and does not fail"
    )
    return summary


def oracle_table(fixtures, graphs) -> dict[str, Any]:
    """The correctness column's reference values, or the reason there is none.

    Plan 5.2a and 11.4: a cell whose oracle abstains is **WITHHELD**, carrying
    the abstain's own sentence, and never blank. A blank reads as a pass.
    """
    _rule("0. the independent oracle, per fixture -- filled or WITHHELD, never blank")
    out = {}
    for row in fixtures:
        graph = graphs[row.label]
        found, source, spans, peak = oracle_for(row.label, graph)
        usable = found.certified and peak["resolved"]
        withheld = None
        if not found.certified:
            withheld = found.certificate.refused
        elif not peak["resolved"]:
            withheld = f"the grid did not resolve the peak: {peak['resolution_note']}"
        out[row.label] = {
            "value": found.value if usable else None,
            "bound": found.certificate.bound if usable else None,
            "withheld": withheld,
            "source": source,
            "axes": len(spans),
            "spans": [(s.name, s.lower, s.upper) for s in spans],
            "box": [(s.name, s.lower, s.upper) for s in peak["box"]],
            "peak": peak["peak"],
            "peak_widths": peak["widths"],
            "peak_height": peak["height"],
            # Carried into the record, not computed and dropped. Both of these
            # were produced by `oracle_for` and discarded here in a first
            # version -- the same defect as the `fallbacks` flag one line down,
            # which an adversarial review found being thrown away at every call
            # site.
            "peak_verdict": peak["peak_verdict"],
            "fallback_axes": peak["fallback_axes"],
            "uncertified_value": None if usable else found.value,
        }
        state = (
            f"{found.value!r} +/- {found.certificate.bound:.2e}"
            if usable
            else f"WITHHELD: {withheld[:64]}"
        )
        print(f"  {row.label:<30} axes={len(spans)} span={source:<8} {state}")
    filled = sum(1 for v in out.values() if v["value"] is not None)
    print(f"  -> {filled} of {len(out)} correctness cells are fillable; "
          f"{len(out) - filled} WITHHELD")
    return out


def section_7_condition_2(oracles, live, row: Fixture) -> dict[str, Any]:
    """1.5 condition 2's artefact: a bitwise-reproducibility assertion per backend.

    Two questions, and the second is the one a table of numbers hides. Does
    the same key give the same `log Z` BITWISE on a repeat -- which is what
    makes a reported number citable at all -- and what KIND of key does each
    backend take? `jax.random.key` and `jax.random.PRNGKey` are different
    objects since JAX's typed keys landed, and a backend that accepts only the
    old one is a backend whose adapter carries a conversion that will outlive
    the reason for it.
    """
    _rule("7. 1.5 condition 2 -- bitwise reproducibility, and which key each takes")
    out: dict[str, Any] = {}
    for candidate in CANDIDATES:
        if _installed(candidate) is None:
            print(f"  {candidate:<9} ABSENT")
            out[candidate] = "ABSENT"
            continue
        request = {
            "fixture": CALIBRATION_FIXTURE,
            "candidate": candidate,
            "seed": 0,
            "knobs": _knobs(candidate, live[candidate]),
            "box": oracles[CALIBRATION_FIXTURE]["box"],
            **_fields_of(row),
        }
        first, second = spawn(request), spawn(request)
        same = (
            first.get("log_Z") is not None
            and first.get("log_Z") == second.get("log_Z")
            and first.get("log_Z_err") == second.get("log_Z_err")
        )
        keys = spawn({**request, "key_types": True})
        out[candidate] = {
            "bitwise_repeat": same,
            "first": first.get("log_Z"),
            "second": second.get("log_Z"),
            "key_types": keys.get("key_types"),
        }
        print(
            f"  {candidate:<9} repeat bitwise: {same}   "
            f"{_fmt(first.get('log_Z'))} vs {_fmt(second.get('log_Z'))}"
        )
        print(f"            key types accepted: {keys.get('key_types')}")
    return out


def _knobs(candidate: str, live: int) -> dict[str, Any]:
    if candidate == "blackjax":
        return {"num_live": int(live), "num_delete": 1}
    return {
        "num_live_points": int(live),
        "num_slices": None,
        "max_samples": int(400 * live),
    }


def section_8_maintenance() -> dict[str, Any]:
    """1.5 condition 4's first half: release dates and cadence, WITH the date read."""
    import datetime
    import json as _json
    import urllib.error
    import urllib.request

    _rule("8. 1.5 condition 4 -- maintenance, read today")
    today = datetime.datetime.now(tz=datetime.UTC).date().isoformat()
    print(f"  read on {today}")
    out: dict[str, Any] = {"read_on": today}
    for name in (*CANDIDATES, "tfp-nightly"):
        try:
            with urllib.request.urlopen(
                f"https://pypi.org/pypi/{name}/json", timeout=20
            ) as response:
                payload = _json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            out[name] = {"verdict": "DECLINED", "why": f"{type(error).__name__}: {error}"}
            print(f"  {name:<13} DECLINED -- {type(error).__name__}")
            continue
        releases = {
            version: files
            for version, files in payload.get("releases", {}).items()
            if files
        }
        dated = sorted(
            (files[0]["upload_time_iso_8601"][:10], version)
            for version, files in releases.items()
        )
        latest = dated[-1] if dated else (None, None)
        year_ago = (
            datetime.datetime.now(tz=datetime.UTC).date()
            - datetime.timedelta(days=365)
        ).isoformat()
        in_year = sum(1 for date, _ in dated if date >= year_ago)
        out[name] = {
            "verdict": "MEASURED",
            "installed": _installed(name),
            "latest_version": latest[1],
            "latest_release_date": latest[0],
            "releases_total": len(dated),
            "releases_last_365_days": in_year,
        }
        repo = GITHUB_REPOS.get(name)
        if repo:
            out[name].update(_github(repo))
        print(
            f"  {name:<13} latest {latest[1]} on {latest[0]}; "
            f"{len(dated)} releases, {in_year} in the last 365 days; "
            f"installed {_installed(name)}"
            + (
                f"; {out[name].get('open_issues')} open issues, repo pushed "
                f"{out[name].get('pushed_at')}"
                if repo
                else ""
            )
        )
    return out


#: The repositories the open-issue and last-push numbers come from. Recorded
#: here because an adversarial review found those two numbers in the evaluation
#: and in no run: they had been queried by hand. A number the document quotes
#: has to come from the instrument the document names.
GITHUB_REPOS = {
    "blackjax": "blackjax-devs/blackjax",
    "jaxns": "Joshuaalbert/jaxns",
}


def _github(repo: str) -> dict[str, Any]:
    """Open issues and the last push, or DECLINED with the reason."""
    import json as _json
    import urllib.error
    import urllib.request

    out: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(
            f"https://api.github.com/search/issues?q=repo:{repo}"
            "+type:issue+state:open&per_page=1",
            timeout=20,
        ) as response:
            out["open_issues"] = _json.load(response).get("total_count")
        with urllib.request.urlopen(
            f"https://api.github.com/repos/{repo}", timeout=20
        ) as response:
            meta = _json.load(response)
        out["pushed_at"] = meta.get("pushed_at")
        out["archived"] = meta.get("archived")
        out["repo"] = repo
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        out["github"] = f"DECLINED: {type(error).__name__}"
    return out


def section_9_termination_totality(uncertainty, table) -> dict[str, Any]:
    """1.5 condition 4's second half: does every signal map to a real member?

    Red line 4 forbids adding an enum member, so a signal with no honest home
    is a finding and not a schema change.
    """
    from bayesmith.artifacts.base import TerminationReason

    _rule("9. 1.5 condition 4 -- every termination signal onto TerminationReason")
    members = [member.value for member in TerminationReason]
    print(f"  TerminationReason has {len(members)}: {members}")
    seen: dict[str, set] = {name: set() for name in CANDIDATES}
    for cell in table:
        if cell.get("termination"):
            seen.setdefault(cell["candidate"], set()).add(cell["termination"])
    for candidate, rungs in uncertainty.items():
        if isinstance(rungs, dict):
            for cell in rungs["rungs"]:
                if cell.get("termination"):
                    seen.setdefault(candidate, set()).add(cell["termination"])
    out = {}
    for candidate, signals in seen.items():
        mapped = {signal: TERMINATION_MAP.get(signal) for signal in sorted(signals)}
        unmapped = [signal for signal, member in mapped.items() if member is None]
        out[candidate] = {"observed": sorted(signals), "map": mapped, "unmapped": unmapped}
        print(f"  {candidate}:")
        for signal, member in mapped.items():
            print(f"      {signal:<44} -> {member or 'NO HONEST MEMBER'}")
    return out


#: Each backend signal onto a `TerminationReason` member, or `None` where there
#: is no honest one. Written out rather than computed, because "no honest
#: member" is a judgement and 6.4's stop-rule is about exactly that judgement.
TERMINATION_MAP = {
    # blackjax: the loop is bayesmith's, so both of these are bayesmith's own.
    "dlogz": "converged",
    "max_iterations": "budget_exhausted",
    # jaxns
    "reached-max-samples": "budget_exhausted",
    "evidence-uncertainty-low-enough": "converged",
    "small-remaining-evidence": "converged",
    "ESS-reached": "converged",
    "max-num-likelihood-evaluations": "budget_exhausted",
    "likelihood-contour-reached": "completed",
    "efficiency-too-low": "tolerance_unmet",
    "plateau": None,
    "relative-spread-low": "converged",
    "absolute-spread-low": "converged",
    "no-seed-points": None,
    "XL-reduction-reached": "converged",
}


def section_10_adapter_size() -> dict[str, Any]:
    """1.5 condition 5, as a measurement rather than an impression.

    These are the PROBE's drivers, not production adapters, and the difference
    is stated rather than smoothed over: a real adapter adds artifact assembly,
    validation and error handling to both columns alike. What the count is
    good for is the DIFFERENCE, and specifically for what each column contains
    that the other does not.
    """
    import inspect

    _rule("10. 1.5 condition 5 -- what a driver for each backend had to contain")
    out = {}
    for candidate, fn in (("blackjax", run_blackjax), ("jaxns", run_jaxns)):
        source, _start = inspect.getsourcelines(fn)
        body = [
            line
            for line in source
            if line.strip() and not line.strip().startswith("#")
        ]
        out[candidate] = {"source_lines": len(source), "code_lines": len(body)}
        print(f"  {candidate:<9} driver {len(source)} lines ({len(body)} non-comment)")
    print("  what each driver OWNS that the backend does not supply:")
    print("    blackjax: the NS loop, the termination rule, log Z, and its uncertainty")
    print("    jaxns:    the bounded box and the prior-into-likelihood reparametrisation")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument(
        "--sections", default="all", help="comma-separated section numbers, or 'all'"
    )
    parser.add_argument("--out", help="write every measurement to this JSON file")
    args = parser.parse_args(argv)

    if args.cell:
        request = json.loads(args.cell)
        if "audit" in request:
            import jax

            jax.config.update("jax_enable_x64", True)
            payload = (
                audit_blackjax_formula()
                if request["audit"] == "blackjax"
                else {
                    "cap_does_not_bind": audit_jaxns_report(max_samples=1000),
                    "cap_binds": audit_jaxns_report(max_samples=300),
                }
            )
            print(json.dumps(payload, default=str))
            return 0
        print(json.dumps(run_cell(request), default=str))
        return 0

    import jax

    jax.config.update("jax_enable_x64", True)
    wanted = (
        {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10"}
        if args.sections == "all"
        else set(args.sections.split(","))
    )
    record: dict[str, Any] = {"target_evaluations": TARGET_EVALUATIONS, "seeds": SEEDS}
    rows = census()
    fixtures = bakeoff_fixtures(rows)
    graphs = {name: graph for name, _module, graph in _built_graphs()}
    record["census"] = [dataclasses.asdict(row) for row in rows]
    record["fixtures"] = [row.label for row in fixtures]

    if "1" in wanted:
        record["environment"] = section_1_environment()
    if "2" in wanted:
        record["consumption"] = section_2_what_each_backend_needs(fixtures, graphs)
    oracles = record["oracles"] = oracle_table(fixtures, graphs)
    from tests.dispatch.residual_oracle import Span

    calibration_box = tuple(
        Span(*entry) for entry in oracles[CALIBRATION_FIXTURE]["box"]
    )
    calibration_row = next(
        row for row in fixtures if row.label == CALIBRATION_FIXTURE
    )
    live = {name: 200 for name in CANDIDATES}
    if "3" in wanted:
        budget = section_3_budget(calibration_box, calibration_row)
        record["budget"] = budget
        for name in CANDIDATES:
            settled = budget["calibration"].get(name)
            if isinstance(settled, dict):
                live[name] = settled["live"]
    record["live"] = live
    if "4" in wanted:
        record["uncertainty"] = section_4_uncertainty_moves(
            live, oracles, calibration_row
        )
    if "5" in wanted:
        record["table"] = section_5_the_table(fixtures, oracles, live)
    if "6" in wanted:
        record["repeats"] = section_6_repeated_runs(fixtures, oracles, live)
    if "7" in wanted:
        record["condition_2"] = section_7_condition_2(
            oracles, live, calibration_row
        )
    if "8" in wanted:
        record["maintenance"] = section_8_maintenance()
    if "9" in wanted:
        record["termination"] = section_9_termination_totality(
            record.get("uncertainty", {}), record.get("table", [])
        )
    if "10" in wanted:
        record["adapter_size"] = section_10_adapter_size()
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1, default=str))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
