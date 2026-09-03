"""Spec rows that assert something about rheplicant, checked against rheplicant.

§5 of the handover is a list of nine spec claims measurement overturned, and
every one of them was a sentence in a handover, a spec or a proposal. They
were all found the same way -- by someone eventually measuring the thing the
sentence described -- and nothing made them findable any earlier, because
prose has no test.

This module gives the load-bearing ones a test. Each row below pins BOTH
halves:

* that the spec still SAYS what it says. If someone corrects the sentence,
  this fails and the row should be deleted -- a correction ledger that
  outlives its corrections is the next thing to go stale.
* that the code still IS what was measured. If rheplicant changes so that a
  claim marked false becomes true, this fails and the spec may now be right.

Only the second half is a normal regression test. The first is what makes the
pair a ledger rather than a duplicate of the tests in `tests/config/` on the
other side.

**Provenance matters and the spec already records it.** Rows carry `[实测]`
(measured), `[实测确认]` (measured and confirmed), `[互证]` (cross-checked)
or `[R1]`/`[R2]` (a reviewer's assertion, never measured). B12 is `[R1]` and
its claim is false; B11 is `[互证]` and its claim holds. That is the pattern
worth knowing: the untagged-as-measured rows are where the errors were.
"""

from __future__ import annotations

import pathlib

import pytest


def _squash(text: str) -> str:
    """Every run of whitespace REMOVED, not collapsed to a single space.

    The spec wraps CJK mid-sentence, and a newline between two Han characters
    is not a space -- collapsing would insert one the claim strings do not
    have. Applied to both sides so the comparison is symmetric.
    """
    return "".join(text.split())


SPEC = (
    pathlib.Path(__file__).resolve().parents[2]
    / "docs" / "superpowers" / "specs" / "2026-08-24-rheplicant-migration.md"
)

#: B12's claim, verbatim. Wrapped in the source across a line break, so the
#: newline is normalised out before matching rather than spelled here.
B12_CLAIM = _squash("rheplicant 无对应 run kind，只有 Python API。")

#: B11's claim, verbatim -- the control. A ledger with only false claims in it
#: cannot show that its method distinguishes them.
B11_CLAIM = _squash("config 完全够不着的子系统（`campaign:` 保留并拒绝）")


@pytest.fixture(scope="module")
def spec_text() -> str:
    if not SPEC.exists():  # pragma: no cover - the spec is tracked
        pytest.fail(f"the migration spec is missing at {SPEC}")
    # ALL whitespace removed, not collapsed to spaces: the spec wraps CJK
    # mid-sentence and a newline there is not a space, so collapsing would
    # insert one that the claim strings below do not have.
    return _squash(SPEC.read_text(encoding="utf-8"))


class TestB12sPremiseIsFalse:
    """`prior_sensitivity` is NOT "only a Python API" in rheplicant.

    Measured through the real gate rather than read off a constant: all four
    modes are honoured, an invalid one produces a real `Finding` with
    severity `refuse` and check id A1, the default is `off`, and
    `config/postflight/fitting.py` wires it to the package function. What
    rheplicant lacks is a RUN KIND -- which is the true half of B12's
    sentence, and is pinned separately in rheplicant by
    `tests/config/test_evidence_has_no_run_kind.py`.

    The consequence for the plan: B12 is scoped as "build its config face",
    and the config face exists in a different SHAPE (a gated check, not a run
    kind). bayesmith has no config layer at all, so "含其 config 面" cannot
    be executed on this side as written.
    """

    def test_the_spec_still_makes_the_claim(self, spec_text):
        assert B12_CLAIM in spec_text, (
            "B12's claim is no longer in the spec. If it was corrected, delete "
            "this class -- a correction ledger must not outlive its correction."
        )

    # The CODE half of this pair lives in rheplicant, at
    # `tests/config/test_evidence_has_no_run_kind.py::TestB12sPremiseIsFalse`,
    # and NOT here. `rheplicant` is installed into this venv with `--no-deps`
    # (see pyproject.toml's crosscheck group), so `rheplicant.config` cannot
    # import -- it needs yaml. Measured: `ModuleNotFoundError: No module named
    # 'yaml'` on every assertion that touched the config layer.
    #
    # Split rather than skipped, deliberately. A `pytest.importorskip` here
    # would leave the claim unchecked on exactly the machines where nobody
    # notices, which is the failure mode `tests/test_readme_counts.py` on the
    # other side already paid for once. Each half now runs in the environment
    # where it is real, and each names the other.


class TestB11sClaimHolds:
    """The control: a spec row that measurement CONFIRMS.

    Without it, every row in this file would be a correction, and a method
    that only ever finds errors has not been shown to be able to find
    anything else.
    """

    def test_the_spec_still_makes_the_claim(self, spec_text):
        assert B11_CLAIM in spec_text

    # Code half likewise in rheplicant -- same reason as B12's above.


def test_the_spec_marks_b12_as_a_reviewer_assertion_not_a_measurement(spec_text):
    """The provenance tag is the signal, and it was already there.

    B12 is tagged `[R1]` -- reviewer 1's assertion. Every claim in §5 of the
    handover that measurement overturned came from a row like this one, and
    the rows tagged `[实测]` / `[互证]` held. If B12 is ever re-tagged as
    measured, this fails, and whoever re-tagged it should have to say what
    they measured.
    """
    tag = _squash("### B12 — `prior_sensitivity` 的 config 面 `[R1]`")
    assert tag in spec_text
