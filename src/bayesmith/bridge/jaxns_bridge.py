"""The JAXNS nested-sampling adapter: one module, consuming one compiled problem.

R5 Task 6. The backend decision is recorded in the R5 backend
evaluation: jaxns passes
every condition that scores a candidate, blackjax fails condition 1 because it
does not answer ``overflowing_outside_latent``, and §0.6's default drops the
second backend.

**What this module is allowed to see.** A :class:`~bayesmith.compiled.
CompiledEvidenceProblem` and nothing else -- no ``Graph``, asserted by type on
the way in. That class sits in ``bayesmith.compiled`` rather than in
``dispatch/evidence.py`` where it was written, precisely so this import can be
at module scope: ``bridge -> dispatch`` would close a cycle that
``tests/test_layering.py`` raises on. The ruling is in that module.

**What this module does NOT build, and why that departs from plan 6.1.** The
plan says the adapter "returns a populated ``EvidenceResult``". It cannot, and
the reason is a rule this module found by breaking it:
``tests/test_layering.py::test_the_artifact_protocol_is_a_leaf_and_dispatch_is_
what_reaches_it`` asserts that the units reaching ``artifacts`` are exactly
``["dispatch", "evaluation"]``, and says a third "would have to be argued for".
A first draft of this file imported ``EvidenceComponent`` and
``WeightedDrawsPosterior`` at module scope and that assertion failed with
``bridge`` in the list -- which is the test doing precisely what its docstring
promises.

The edge is not argued for, because it is not needed. ``evaluation`` cannot do
its job without artifact types; this module can. It reports NUMBERS, and
``dispatch`` -- which already assembles every other ``EvidenceResult`` in this
package, at ``task.py:3246``, out of facts only the planner has
(``plan_ref``, ``fingerprints``, ``PRODUCER``) -- turns them into artifacts.
That is also what ``bridge/numpyro_bridge.py::nuts`` does: it returns
``dict[str, jax.Array]``, and ``bridge/arviz.py`` imports artifact types inside
a function for the same reason.

**So the split is: jaxns facts here, bayesmith policy in `dispatch`.** The bit
LABELS below are jaxns's own words; the map from a label to a termination
member is this package's judgement, and it is carried here as the enum's plain
string VALUES so that stating it costs no import. The test that it is total --
that every string is a real ``TerminationReason`` and every bit has one -- lives
in ``tests/``, where reaching ``artifacts`` is free.

**The box, and what it costs.** jaxns's ``Prior`` wants a quantile function
``U -> X``, and a compiled problem carries ``log_prior``, ``log_likelihood`` and
``prior_sample`` -- a density is not invertible by inspection and a sampler is
not a transform, so none of the three yields one. The generic route is the
reparametrisation the bake-off measured: declare a bounded box ``B``, give jaxns
a uniform prior over it, and move this package's own prior into the likelihood,

    log L'(x) = log pi(x) + log L(x) + log |B|

so jaxns integrates ``INT_B pi L dx``. Three consequences, each a cost the
verdict already names: ``B`` is an input jaxns needs and the compiled problem
does not carry; the answer is the evidence TRUNCATED to ``B``, so mass outside
it is an error the oracle's own truncation term bounds; and the sampler now
contracts against ``pi L`` rather than ``L``, which is not the problem nested
sampling's efficiency argument is about.

**Two things jaxns does that a caller has to know.** Importing it writes
``jax_enable_x64`` **process-globally** -- reproduced here on 2026-09-07, with
``UserWarning: JAX x64 is not enabled. Setting it now.`` -- against this
package's rule that ``src/`` never touches ``jax.config``. §0.16's owner
decision attaches to this module because the backend it was conditional on was
chosen. And ``TerminationCondition()`` **disables every stopping rule**: each
field defaults to ``None``, ``None`` means "off", and the real defaults live in
``jaxns/public.py``, which builds one only when it is handed none. This module
never constructs one and never passes ``term_cond`` at all.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from bayesmith.compiled import CompiledEvidenceProblem
from bayesmith.errors import BayesmithError

#: The declared, total map from jaxns's termination bits to this package's
#: closed :class:`~bayesmith.artifacts.base.TerminationReason`.
#:
#: The labels are jaxns 2.6.9's own, read from ``jaxns/utils.py`` where it
#: prints them, and the order is the bit order. §0.1 requires a declared table
#: with a test that it is total, and forbids adding an enum member for a signal
#: that does not fit -- so the mapping below is a judgement about each signal,
#: recorded rather than derived.
#:
#: Two are not clean fits and are named here rather than left to be
#: rediscovered. Bit 7 (a single plateau) and bit 10 (no seed points) both stop
#: a run whose own state has become unusable; jaxns's text calls the first
#: "sign of possible precision error" and the second suggests changing a
#: parameter. ``TOLERANCE_UNMET`` is literally true of both -- the run ended and
#: no convergence tolerance was met -- and was rejected because it says a
#: tolerance was tested and missed, which is not what happened. In a closed set
#: that already spells "ran out of budget" and "tolerance missed", ``DIVERGED``
#: is the only member that says the run itself went wrong.
JAXNS_TERMINATION: tuple[tuple[str, str], ...] = (
    ("reached max samples", "budget_exhausted"),
    ("evidence uncertainty low enough", "converged"),
    ("small remaining evidence", "converged"),
    ("reached ESS", "converged"),
    ("used max num likelihood evaluations", "budget_exhausted"),
    ("likelihood contour reached", "completed"),
    ("sampler efficiency too low", "tolerance_unmet"),
    ("all live points on a single plateau", "diverged"),
    ("relative spread of live points < rtol", "converged"),
    ("absolute spread of live points < atol", "converged"),
    ("no seed points left", "diverged"),
    ("XL < max(XL) * peak_XL_frac", "converged"),
)

#: Worst-first. A jaxns mask can carry several bits at once -- it is a bitmask,
#: measured -- so the reported reason is the WORST that fired, not the first.
#:
#: **This ordering is load-bearing and the reason is one line of another
#: module**: ``dispatch/task.py`` reads ``certified = termination.reason is
#: TerminationReason.CONVERGED``. Taking the lowest set bit instead would report
#: a run that both exhausted its budget and collapsed onto a plateau as
#: CONVERGED, because the converging bits (1, 2, 3) sit below the degenerate
#: ones (7, 10) -- and that report is what certifies it. CONVERGED has to be the
#: hardest to earn, so it is last.
_SEVERITY: tuple[str, ...] = (
    "diverged",
    "tolerance_unmet",
    "budget_exhausted",
    "completed",
    "converged",
)


@dataclasses.dataclass(frozen=True, slots=True)
class Termination:
    """One jaxns bitmask, decoded. ``reason`` is a ``TerminationReason`` VALUE.

    A string rather than the enum member so that this module costs no edge to
    ``artifacts``; ``TerminationReason(record.reason)`` is what ``dispatch``
    does with it, and a test asserts every value in the table is a real member.
    """

    reason: str
    message: str
    labels: tuple[str, ...]


def decode_termination(mask: int) -> Termination:
    """One jaxns bitmask, decoded.

    The message names EVERY bit that fired, so collapsing the mask to one
    ``reason`` loses nothing a reader needs.

    A mask of zero raises. §0.1: "An unmapped backend signal is an error, not a
    default -- a nested sampler that stopped for an unrecognised reason must not
    be recorded as ``COMPLETED``." A run that stopped for no declared reason is
    exactly that case, and so is a bit above the twelve this table knows.
    """
    if type(mask) is not int:
        raise TypeError(f"a jaxns termination mask is an int; got {mask!r}")
    if mask <= 0:
        raise BayesmithError(
            f"jaxns reported termination_reason={mask}, which sets no bit this "
            "package knows. A run that stopped for an unrecognised reason must "
            "not be recorded as COMPLETED, so there is no default here. If "
            "jaxns has grown a condition, add it to JAXNS_TERMINATION with the "
            "member that is honest about it."
        )
    width = len(JAXNS_TERMINATION)
    if mask >> width:
        raise BayesmithError(
            f"jaxns reported termination_reason={mask} ({mask:b}), which sets a "
            f"bit above the {width} this package's table declares. That is a "
            "signal with no mapping, which §0.1 makes an error rather than a "
            "default; read jaxns's own printer for the new label and map it."
        )
    fired = [
        (label, reason)
        for index, (label, reason) in enumerate(JAXNS_TERMINATION)
        if (mask >> index) & 1
    ]
    reasons = {reason for _, reason in fired}
    worst = next(reason for reason in _SEVERITY if reason in reasons)
    return Termination(
        reason=worst,
        message="; ".join(label for label, _ in fired),
        labels=tuple(label for label, _ in fired),
    )


@dataclasses.dataclass(frozen=True, slots=True)
class NestedEvidence:
    """What the backend produced, in artifact types and plain Python floats.

    Every number here has been through ``float()`` already. ``_finite`` requires
    ``type(value) in (int, float)``, so a ``np.float64`` or a jax scalar raises
    ``TypeError`` at the artifact boundary (§0.1), and a backend returns none of
    these as Python floats.
    """

    log_evidence: float
    standard_error: float
    draws: tuple[tuple[str, np.ndarray], ...]
    log_weights: np.ndarray
    ess: float
    termination: Termination
    backend_name: str
    backend_version: str | None
    likelihood_evaluations: int
    iterations: int
    truncated_to: tuple[tuple[str, float, float], ...]
    method: str


def _flat_axes(
    problem: CompiledEvidenceProblem,
) -> tuple[tuple[str, ...], dict[str, tuple[int, ...]]]:
    """The residual parameters and their shapes, in the problem's own order."""
    shapes = dict(problem.shapes)
    names = tuple(problem.residual_parameters)
    missing = [name for name in names if name not in shapes]
    if missing:
        raise BayesmithError(
            f"the compiled problem declares residual parameters {missing} with "
            f"no shape; it carries shapes for {sorted(shapes)}"
        )
    return names, shapes


def nested_evidence(
    problem: CompiledEvidenceProblem,
    *,
    box: Mapping[str, tuple[float, float]],
    seed: int = 0,
    num_live_points: int | None = None,
    max_samples: int = 100_000,
    method: str = "jaxns_nested_sampling",
) -> NestedEvidence:
    """Integrate ``problem`` over ``box`` with jaxns, and report what came back.

    ``box`` is per residual parameter, ``(lower, upper)``. It is required
    because jaxns needs a domain and the compiled problem does not carry one;
    see this module's docstring for what the truncation costs.
    """
    if not isinstance(problem, CompiledEvidenceProblem):
        raise TypeError(
            "a nested-sampling adapter consumes a CompiledEvidenceProblem and "
            f"nothing else -- not a Graph, not a plan; got {type(problem).__name__}. "
            "Compile the graph first with dispatch.evidence.compile_evidence_problem."
        )

    # **Everything checkable is checked BEFORE the backend is imported.** The
    # order was put here by a test: with the import first, a caller who passed
    # an empty box got "jaxns is not installed", a true sentence about the wrong
    # thing. It also means these guards run wherever this package is installed
    # rather than only where the optional extra is -- and absence is this
    # repository's default state, so the second would have meant "never in CI".
    #
    # The sentence above was FALSE when first written: the box moved up and the
    # scalar knobs did not, so eight inputs still answered ModuleNotFoundError.
    # A review measured them. Whatever is added below, add its check here.
    names, shapes = _flat_axes(problem)
    unknown = [name for name in box if name not in names]
    if unknown:
        raise BayesmithError(
            f"the box names {sorted(unknown)}, which are not residual "
            f"parameters of this problem; it integrates {list(names)}"
        )
    undeclared = [name for name in names if name not in box]
    if undeclared:
        raise BayesmithError(
            f"the box declares no bounds for {undeclared}. jaxns integrates a "
            "bounded domain, so every residual parameter needs one; there is no "
            "default, because a span chosen here would be a silent truncation."
        )

    sizes = [max(1, int(np.prod(shapes[name], dtype=int))) for name in names]
    lower = np.concatenate(
        [np.full(size, float(box[name][0])) for name, size in zip(names, sizes, strict=True)]
    )
    upper = np.concatenate(
        [np.full(size, float(box[name][1])) for name, size in zip(names, sizes, strict=True)]
    )
    bad = [
        name
        for name in names
        if not (float(box[name][0]) < float(box[name][1]))
        or not np.all(np.isfinite([box[name][0], box[name][1]]))
    ]
    if bad:
        raise BayesmithError(
            f"the box is empty, inverted or unbounded on {bad}; each bound is a "
            "finite (lower, upper). An infinite side is not a wider box, it is "
            "no box: this adapter's whole premise is that jaxns integrates a "
            "BOUNDED domain and the mass outside it is a truncation somebody "
            "can bound."
        )
    # The scalar knobs, checked here rather than at the call sites below.
    # An adversarial review found eight inputs -- a non-integer `seed`, a
    # `max_samples` of 0 or -5 or "x", a `num_live_points` of 0 or "x", a
    # non-string `method` -- surfacing as `ModuleNotFoundError: No module named
    # 'tensorflow_probability'`, because `int(...)` sat after the import. That
    # is the same defect the box checks above were moved up for, left in place
    # one paragraph further down, and the comment claiming otherwise was false
    # when it was written.
    for label, value in (("seed", seed), ("max_samples", max_samples)):
        if type(value) is not int:
            raise TypeError(f"{label} is an int; got {value!r}")
    if max_samples <= 0:
        raise BayesmithError(f"max_samples is a positive count; got {max_samples}")
    if num_live_points is not None:
        if type(num_live_points) is not int:
            raise TypeError(f"num_live_points is an int or None; got {num_live_points!r}")
        if num_live_points <= 0:
            raise BayesmithError(
                f"num_live_points is a positive count; got {num_live_points}"
            )
    if type(method) is not str or not method:
        raise BayesmithError(
            f"method names how this evidence was computed and reaches an "
            f"artifact field as a non-empty string; got {method!r}"
        )

    log_volume = float(np.sum(np.log(upper - lower)))
    offsets = np.cumsum([0, *sizes])

    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from jaxns import Model, NestedSampler, Prior

    def unflatten(vector: Any) -> dict[str, Any]:
        return {
            name: jnp.reshape(vector[start:stop], shapes[name])
            for name, start, stop in zip(names, offsets[:-1], offsets[1:], strict=True)
        }

    tfpd = tfp.distributions

    def prior_model():
        vector = yield Prior(
            tfpd.Uniform(low=jnp.asarray(lower), high=jnp.asarray(upper)), name="theta"
        )
        return vector

    def log_likelihood(vector):
        values = unflatten(vector)
        # The reparametrisation: this package's prior moves into jaxns's
        # likelihood, and `log |B|` cancels the uniform's own normalisation.
        return problem.log_prior(values) + problem.log_likelihood(values) + log_volume

    model = Model(prior_model=prior_model, log_likelihood=log_likelihood)
    sampler = NestedSampler(
        model=model,
        max_samples=int(max_samples),
        **({} if num_live_points is None else {"num_live_points": int(num_live_points)}),
    )
    # **`term_cond` is not passed, and that is the whole of it.** It belongs to
    # `NestedSampler.__call__`, not to its constructor -- measured: passing it
    # to `__init__` raises `TypeError: unexpected keyword argument`. Omitting it
    # IS passing `None`, and `None` is the one value that turns the real
    # defaults on: `jaxns/public.py` builds a `TerminationCondition` with
    # `dlogZ` and `max_samples` exactly when it receives none. Constructing one
    # here would do the opposite of what it looks like -- every field of that
    # class defaults to `None` and `None` means OFF -- so a hand-built
    # condition switches every stopping rule off and the run ends only by
    # structural exhaustion. The R5 bake-off lost a whole 21-row table to that
    # once; this line is here so nobody adds it back.
    mask, state = jax.jit(sampler)(jax.random.key(int(seed)))
    results = sampler.to_results(termination_reason=mask, state=state)

    log_evidence = float(results.log_Z_mean)
    standard_error = float(results.log_Z_uncert)
    samples = results.samples["theta"]
    draws = tuple(
        (
            name,
            np.asarray(
                np.reshape(samples[:, start:stop], (-1, *shapes[name])), dtype=float
            ),
        )
        for name, start, stop in zip(names, offsets[:-1], offsets[1:], strict=True)
    )
    return NestedEvidence(
        log_evidence=log_evidence,
        standard_error=standard_error,
        draws=draws,
        log_weights=np.asarray(results.log_dp_mean, dtype=float),
        ess=float(results.ESS),
        termination=decode_termination(int(np.asarray(mask))),
        backend_name="jaxns",
        backend_version=_jaxns_version(),
        likelihood_evaluations=int(results.total_num_likelihood_evaluations),
        iterations=int(results.total_num_samples),
        truncated_to=tuple(
            (name, float(box[name][0]), float(box[name][1])) for name in names
        ),
        method=method,
    )


def _jaxns_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("jaxns")
    except PackageNotFoundError:  # pragma: no cover - installed by definition here
        return None


__all__ = [
    "JAXNS_TERMINATION",
    "NestedEvidence",
    "Termination",
    "decode_termination",
    "nested_evidence",
]
