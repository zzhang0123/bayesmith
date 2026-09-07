"""R5 Task 6: the jaxns adapter.

**Two populations of test here, and the split is not incidental.** Everything
about the TABLE -- that it is total, that every member it names is real, that a
mask collapses to the worst reason and not the first -- runs everywhere, because
it needs no backend. Everything that RUNS jaxns is skipped wherever jaxns is
absent, which is this repository's default state (§0.9, and condition 6 scores
absence as the normal case).

That means the numerical path is not exercised by CI, and saying so is the
point of this paragraph rather than a caveat buried in one: the guards CI runs
are the ones about mapping, validation and refusal, and the closed-form
comparison runs where jaxns is installed.

**And every test that touches jaxns runs in a SUBPROCESS.** Importing it writes
``jax_enable_x64`` process-globally, and ``dispatch/task.py`` decides
``evidence_requires_x64`` by outcome, so an in-process import turns R4's
precision refusal off for everything after it. A first version of this file did
exactly that: a review measured
``test_a_float32_environment_is_refused_by_name`` and
``test_the_run_record_says_what_actually_ran`` going red once this file had run
-- red only for someone who installed the extra, so CI would never have said
so. ``find_spec`` alone does not help, because it only avoids the import in the
case where importing is harmless. See :func:`_in_a_subprocess`.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from bayesmith.artifacts.base import TerminationReason
from bayesmith.bridge.jaxns_bridge import (
    JAXNS_TERMINATION,
    Termination,
    decode_termination,
    nested_evidence,
)
from bayesmith.compiled import CompiledEvidenceProblem
from bayesmith.errors import BayesmithError

requires_jaxns = pytest.mark.skipif(
    importlib.util.find_spec("jaxns") is None, reason="jaxns is not installed"
)

#: jaxns 2.6.9 prints twelve conditions in `jaxns/utils.py`; the table must have
#: exactly one row per bit.
JAXNS_BIT_COUNT = 12


class TestTheTerminationTableIsTotalAndHonest:
    """§0.1: a declared table, with a test that the table is total.

    The adapter carries the map as the enum's plain string VALUES so that
    stating it costs no edge to ``artifacts`` (see the module docstring there).
    The cost of that choice is that a typo would be a string nobody checks --
    so this is where it gets checked, in a file where reaching ``artifacts`` is
    free.
    """

    def test_the_table_has_one_row_per_jaxns_bit(self):
        assert len(JAXNS_TERMINATION) == JAXNS_BIT_COUNT

    def test_every_reason_the_table_names_is_a_real_termination_member(self):
        for label, reason in JAXNS_TERMINATION:
            assert TerminationReason(reason) is not None, label

    def test_no_bit_is_left_without_a_reason(self):
        for index, (label, reason) in enumerate(JAXNS_TERMINATION):
            assert label.strip(), index
            assert reason.strip(), label

    def test_the_table_uses_no_member_the_enum_does_not_have(self):
        """The complement of the test above: a value outside the enum fails.

        `TerminationReason` is a `StrEnum`, so `TerminationReason("nonsense")`
        raises -- which is what makes the string form checkable at all.
        """
        with pytest.raises(ValueError):
            TerminationReason("stopped_because_reasons")


class TestAMaskCollapsesToTheWorstReasonAndNotTheFirst:
    def test_a_single_bit_maps_through_the_table(self):
        # bit 2, "small remaining evidence" -- the one a real run returned:
        # `termination_reason = 4` on a jaxns 2.6.9 run measured 2026-09-07.
        found = decode_termination(4)
        assert isinstance(found, Termination)
        assert found.reason == "converged"
        assert found.labels == ("small remaining evidence",)

    def test_a_degenerate_bit_beats_a_converging_one_however_they_are_ordered(self):
        """The ordering that matters, and the reason it is not cosmetic.

        `dispatch/task.py` reads `certified = termination.reason is
        TerminationReason.CONVERGED`. jaxns's converging bits (1, 2, 3) sit
        BELOW its degenerate ones (7, 10), so taking the lowest set bit -- the
        obvious implementation -- would report a run that collapsed onto a
        plateau as CONVERGED and certify it.
        """
        both = (1 << 2) | (1 << 7)  # small remaining evidence + single plateau
        found = decode_termination(both)
        assert found.reason == "diverged"
        assert len(found.labels) == 2
        # and nothing is lost: the message still names the converging bit.
        assert "small remaining evidence" in found.message
        assert "all live points on a single plateau" in found.message

    @pytest.mark.parametrize(
        ("mask", "expected"),
        [
            ((1 << 0) | (1 << 1), "budget_exhausted"),
            ((1 << 1) | (1 << 6), "tolerance_unmet"),
            ((1 << 5) | (1 << 8), "completed"),
            ((1 << 1) | (1 << 2) | (1 << 3), "converged"),
            ((1 << 0) | (1 << 6) | (1 << 10), "diverged"),
        ],
    )
    def test_the_worst_reason_wins_across_the_severity_order(self, mask, expected):
        assert decode_termination(mask).reason == expected

    def test_converged_is_reported_only_when_nothing_worse_fired(self):
        """The property the ordering exists for, stated as one assertion."""
        worse = [
            index
            for index, (_, reason) in enumerate(JAXNS_TERMINATION)
            if reason != "converged"
        ]
        for index in worse:
            mask = (1 << 1) | (1 << index)  # a converging bit, plus a worse one
            assert decode_termination(mask).reason != "converged", index


class TestAnUnmappedSignalIsAnErrorAndNotADefault:
    """§0.1: "An unmapped backend signal is an error, not a default -- a nested
    sampler that stopped for an unrecognised reason must not be recorded as
    COMPLETED"."""

    def test_a_mask_of_zero_refuses(self):
        with pytest.raises(BayesmithError, match="sets no bit this package knows"):
            decode_termination(0)

    def test_a_bit_above_the_table_refuses_rather_than_being_ignored(self):
        """A jaxns that grows a thirteenth condition must not pass silently."""
        with pytest.raises(BayesmithError, match="above the 12"):
            decode_termination(1 << JAXNS_BIT_COUNT)

    def test_a_negative_mask_refuses(self):
        with pytest.raises(BayesmithError):
            decode_termination(-1)

    def test_a_mask_that_is_not_an_int_refuses_by_type(self):
        with pytest.raises(TypeError, match="is an int"):
            decode_termination(4.0)


def _gaussian_problem(*, datum: float = 1.3, sigma: float = 0.7):
    """A one-axis problem whose evidence is known in closed form.

    ``x ~ N(0, 1)``, ``d ~ N(x, sigma)``, so

        Z = INT N(x; 0, 1) N(d; x, sigma) dx = N(d; 0, sqrt(1 + sigma^2))

    Built by hand rather than compiled from a graph: this file is about the
    ADAPTER, and a fixture that also exercises `compile_evidence_problem` would
    not say which half moved when it failed.
    """
    import jax.numpy as jnp

    def log_prior(values):
        x = values["x"]
        return -0.5 * x**2 - 0.5 * jnp.log(2 * jnp.pi)

    def log_likelihood(values):
        x = values["x"]
        return -0.5 * ((datum - x) / sigma) ** 2 - jnp.log(
            sigma * jnp.sqrt(2 * jnp.pi)
        )

    problem = CompiledEvidenceProblem(
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        prior_sample=None,
        exact_elimination=(),
        residual_parameters=("x",),
        shapes=(("x", ()),),
        prior_terms=("x",),
        likelihood_terms=("d",),
    )
    spread = float(np.sqrt(1.0 + sigma**2))
    exact = float(
        -0.5 * (datum / spread) ** 2 - np.log(spread * np.sqrt(2 * np.pi))
    )
    return problem, exact


class TestTheAdapterConsumesACompiledProblemAndNothingElse:
    """6.1: no ``Graph``, asserted by type rather than hoped for."""

    @pytest.mark.parametrize(
        "wrong",
        [None, object(), {"log_prior": None}, "a graph", 3],
        ids=["none", "object", "mapping", "string", "int"],
    )
    def test_anything_that_is_not_a_compiled_problem_is_refused(self, wrong):
        with pytest.raises(TypeError, match="consumes a CompiledEvidenceProblem"):
            nested_evidence(wrong, box={"x": (-5.0, 5.0)})

    def test_a_graph_is_refused_by_the_same_assertion(self):
        from tests.exact.models import straight_line

        with pytest.raises(TypeError, match="not a Graph"):
            nested_evidence(straight_line(), box={"x": (-5.0, 5.0)})


class TestTheBoxIsRequiredAndChecked:
    """The domain is an input jaxns needs and the compiled problem does not
    carry, so it cannot have a default: a span chosen inside the adapter would
    be a silent truncation of somebody's evidence."""

    def test_a_missing_bound_refuses_and_names_the_axis(self):
        problem, _ = _gaussian_problem()
        with pytest.raises(BayesmithError, match="no bounds for \\['x'\\]"):
            nested_evidence(problem, box={})

    def test_a_bound_for_something_that_is_not_a_parameter_refuses(self):
        problem, _ = _gaussian_problem()
        with pytest.raises(BayesmithError, match="not residual"):
            nested_evidence(problem, box={"x": (-5.0, 5.0), "nope": (0.0, 1.0)})

    @pytest.mark.parametrize("bounds", [(1.0, 1.0), (2.0, -2.0)], ids=["empty", "inverted"])
    def test_an_empty_or_inverted_box_refuses(self, bounds):
        problem, _ = _gaussian_problem()
        with pytest.raises(BayesmithError, match="empty, inverted or unbounded"):
            nested_evidence(problem, box={"x": bounds})


def _in_a_subprocess(body: str) -> str:
    """Run ``body`` in a fresh interpreter and return its stdout.

    **Every test that touches jaxns runs out of process, and that is the whole
    of R5's answer to §0.16.** Importing jaxns runs
    ``jax.config.update('jax_enable_x64', True)`` at module scope, which is
    process-global; ``dispatch/task.py`` decides ``evidence_requires_x64`` by
    OUTCOME (``jnp.result_type(float)``) deliberately, so that a caller using
    the context manager and one using the environment variable get one answer.
    The two together mean an in-process jaxns import turns R4's precision
    refusal off for everything that runs after it.

    Measured by an adversarial review before this helper existed: with these
    tests importing jaxns in-process,
    ``tests/dispatch/test_evidence_task.py::test_a_float32_environment_is_
    refused_by_name`` and
    ``tests/dispatch/test_task_execution.py::test_the_run_record_says_what_
    actually_ran`` both go red -- the same two the plan's §0.16 predicted, and
    red only for someone who installed the extra, so CI would never have said
    so.

    The gate's own repair is NOT here. §0.16 reserves that for the owner and
    says both of its shapes "are changes to R4's shipped behaviour and neither
    belongs inside an R5 task"; the ruling recorded there is that it stays
    outside R5. What belongs here is not damaging the gate, and a subprocess is
    how this package already isolates a process-global fact -- see
    ``tests/test_public_api.py``, which uses the same shape to prove a bare
    import pulls in no numerical stack.
    """
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", body], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


_PREAMBLE = """
import json, numpy as np
from tests.bridge.test_jaxns_bridge import _gaussian_problem
from bayesmith.bridge.jaxns_bridge import nested_evidence, JAXNS_TERMINATION
problem, exact = _gaussian_problem()
found = nested_evidence(problem, box={"x": (-8.0, 8.0)}, seed=0, max_samples=%d)
"""


class TestEveryInputIsCheckedBeforeTheBackendIsImported:
    """The eight a review found answering ``ModuleNotFoundError``.

    These run where jaxns is ABSENT, which is the point: a caller who passed a
    bad seed should be told about the seed, and CI is the environment that can
    prove it, because there the import would otherwise be the first thing to
    fail.
    """

    @pytest.mark.parametrize(
        ("kwargs", "exc", "phrase"),
        [
            ({"seed": "banana"}, TypeError, "seed is an int"),
            ({"seed": 1.5}, TypeError, "seed is an int"),
            ({"max_samples": "x"}, TypeError, "max_samples is an int"),
            ({"max_samples": 0}, BayesmithError, "positive count"),
            ({"max_samples": -5}, BayesmithError, "positive count"),
            ({"num_live_points": "x"}, TypeError, "num_live_points is an int"),
            ({"num_live_points": 0}, BayesmithError, "positive count"),
            ({"method": 42}, BayesmithError, "non-empty string"),
            ({"method": ""}, BayesmithError, "non-empty string"),
        ],
    )
    def test_a_malformed_knob_is_named_rather_than_the_missing_backend(
        self, kwargs, exc, phrase
    ):
        problem, _ = _gaussian_problem()
        with pytest.raises(exc, match=phrase):
            nested_evidence(problem, box={"x": (-5.0, 5.0)}, **kwargs)

    @pytest.mark.parametrize(
        "bounds",
        [(0.0, float("inf")), (float("-inf"), float("inf")), (0.0, float("nan"))],
        ids=["half-open", "unbounded", "nan"],
    )
    def test_an_unbounded_or_nan_box_refuses_and_names_the_axis(self, bounds):
        """`nan <= 0.0` is False, so a NaN bound used to refuse with an EMPTY
        list of offenders -- a refusal that named nothing."""
        problem, _ = _gaussian_problem()
        with pytest.raises(BayesmithError, match="unbounded on \\['x'\\]"):
            nested_evidence(problem, box={"x": bounds})


@requires_jaxns
class TestARunAgainstAClosedForm:
    """The numerical path, every case out of process. See :func:`_in_a_subprocess`."""

    def test_the_evidence_matches_the_closed_form_within_its_own_uncertainty(self):
        out = _in_a_subprocess(
            (_PREAMBLE % 40_000)
            + """
print(json.dumps({
    "logZ": found.log_evidence, "se": found.standard_error, "exact": exact,
    "finite": bool(np.isfinite(found.log_evidence)),
}))
"""
        )
        got = json.loads(out.strip().splitlines()[-1])
        assert got["finite"]
        assert got["se"] > 0.0
        # Three of its own reported sigma. The truncation to |x| <= 8 is worth
        # far less: the prior puts 1.24e-15 of its mass outside, and the
        # integrand's own truncated tail is 9.1e-36 nats.
        assert abs(got["logZ"] - got["exact"]) <= 3.0 * got["se"], got

    def test_every_number_it_files_is_a_python_float(self):
        """§0.1: ``_finite`` requires ``type(value) in (int, float)``."""
        out = _in_a_subprocess(
            (_PREAMBLE % 20_000)
            + """
print(json.dumps({
    "logZ": type(found.log_evidence).__name__,
    "se": type(found.standard_error).__name__,
    "ess": type(found.ess).__name__,
    "evals": type(found.likelihood_evaluations).__name__,
    "iters": type(found.iterations).__name__,
    "bounds": sorted({type(v).__name__ for _n, lo, hi in found.truncated_to for v in (lo, hi)}),
}))
"""
        )
        got = json.loads(out.strip().splitlines()[-1])
        assert got["logZ"] == "float" and got["se"] == "float"
        assert got["ess"] == "float"
        assert got["evals"] == "int" and got["iters"] == "int"
        assert got["bounds"] == ["float"]

    def test_no_backend_object_reaches_the_result_at_any_depth(self):
        """Condition 5's countable half, read RECURSIVELY.

        A first version read ``type(value).__module__`` of the top-level fields
        only and named two prefixes. A review walked past it twice: ``draws`` is
        a tuple, whose module is ``builtins``, so a raw jax array inside it was
        invisible -- and a jax array's module is ``jaxlib._jax``, which neither
        named prefix covers. Both mutants survived. This walks containers and
        asks the complement question: every leaf must come from ``numpy``,
        ``builtins`` or this package.
        """
        out = _in_a_subprocess(
            (_PREAMBLE % 20_000)
            + """
import dataclasses
def modules(value, seen=None):
    out = {type(value).__module__}
    if isinstance(value, (tuple, list)):
        for item in value:
            out |= modules(item)
    return out
found_modules = set()
for field in dataclasses.fields(found):
    found_modules |= modules(getattr(found, field.name))
print(json.dumps(sorted(found_modules)))
"""
        )
        modules = json.loads(out.strip().splitlines()[-1])
        allowed = {"builtins", "numpy"}
        for module in modules:
            root = module.split(".")[0]
            assert root in allowed or root == "bayesmith", (module, modules)

    def test_the_draws_and_weights_line_up_and_are_named(self):
        out = _in_a_subprocess(
            (_PREAMBLE % 20_000)
            + """
(name, values), = found.draws
print(json.dumps({
    "names": [n for n, _ in found.draws],
    "rows": int(values.shape[0]), "weights": int(found.log_weights.shape[0]),
    "finite": bool(np.all(np.isfinite(values))),
    "backend": found.backend_name, "version": found.backend_version,
}))
"""
        )
        got = json.loads(out.strip().splitlines()[-1])
        assert got["names"] == ["x"]
        assert got["rows"] == got["weights"]
        assert got["finite"]
        assert got["backend"] == "jaxns" and got["version"]

    def test_the_termination_it_reports_is_one_the_table_declares(self):
        out = _in_a_subprocess(
            (_PREAMBLE % 20_000)
            + """
print(json.dumps({"reason": found.termination.reason,
                  "labels": list(found.termination.labels)}))
"""
        )
        got = json.loads(out.strip().splitlines()[-1])
        assert TerminationReason(got["reason"]) is not None
        assert got["labels"]
        for label in got["labels"]:
            assert label in {row for row, _ in JAXNS_TERMINATION}

    def test_the_twelve_labels_are_jaxns_own_and_in_jaxns_own_order(self):
        """The table's LABEL column, checked against jaxns instead of itself.

        Five mutants survived the first version of this file: misspelling a
        label, swapping the rtol/atol rows, swapping the two budget rows,
        inventing text for bit 11, and relabelling bit 3 as bit 1. Every one
        passed, because the only label assertions were membership tests against
        the mutated table -- self-consistent under any relabelling. A permuted
        table maps every bit to the wrong condition and reports it confidently.

        jaxns prints the twelve in ``jaxns/utils.py``, in bit order, and that
        list is what this reads.
        """
        out = _in_a_subprocess(
            """
import ast, inspect, json, jaxns.utils
source = inspect.getsource(jaxns.utils)
start = source.index("termination_bit_mask = ")
opened = source.index("[", source.index("zip(", start))
depth, end = 0, opened
for end in range(opened, len(source)):
    depth += (source[end] == "[") - (source[end] == "]")
    if depth == 0:
        break
print(json.dumps(ast.literal_eval(source[opened : end + 1])))
"""
        )
        upstream = json.loads(out.strip().splitlines()[-1])
        assert len(upstream) == len(JAXNS_TERMINATION), (len(upstream), upstream)
        for index, (theirs, (ours, _reason)) in enumerate(
            zip(upstream, JAXNS_TERMINATION, strict=True)
        ):
            # Ours are normalised -- lower-cased, and two are shortened, which
            # the module docstring says. The test is that they still identify
            # the same condition, so it compares on the words they share.
            theirs_words = set(theirs.lower().replace("-", " ").split())
            ours_words = set(ours.lower().replace("-", " ").split())
            shared = theirs_words & ours_words
            assert len(shared) >= 2, (index, theirs, ours, shared)
