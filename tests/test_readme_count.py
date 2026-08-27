"""The README's test count is counted, not remembered.

``README.md`` said "Implemented and tested, 1269 tests" while the suite held
1445, and nothing would have said so. That drift is the accumulated residue of
six batches -- G2, G9 in full, G10/G12, G14, G3, G4, G15 -- each of which added
tests and none of which touched the front page. Every one of those batches was
run under a rule (spec 铁律 4(iv)) that says a batch re-measures the numbers its
documentation states, and every one of them satisfied that rule as far as
anyone could tell, because **a number in prose is a claim that no run checks**.
The stale version reads exactly as authoritative as the true one.

This file is the sibling of that rule: it turns the README's count into
something that can fail. It is deliberately the SIMPLEST form of the guard --
one number, pinned by equality, with the true number in the failure message so
the fix is a copy rather than an arithmetic. The e-RHINO checkout carries the
elaborated version (``tests/test_readme_counts.py``), which also has to reason
about modules that stand down behind an ``importorskip``; this package has
none, so a plain collection is the whole measurement.

**Why collection and not the pass count.** Collection is what "the suite has N
tests" means, and it is cheap. A pass count needs the suite to have run, which
this test cannot do from inside it, and it would go soft the first time
something legitimately skips.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

#: How the README spells it. The count is captured; the surrounding words are
#: part of the pattern so that rewording the sentence trips this rather than
#: silently leaving the check with nothing to read.
_CLAIM = re.compile(r"Implemented and tested, ([\d,]+) tests:")

#: pytest's own summary line from a bare collection.
_COLLECTED = re.compile(r"^(\d+) tests collected", re.MULTILINE)


def _collected() -> int:
    """Ask pytest how many tests there are, in a subprocess.

    No extra ``-q``: ``pyproject.toml``'s ``addopts`` already carries one, and
    a second makes it ``-qq``, which removes the very line parsed here.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,  # the return code is asserted on the next line, with context
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout[-2000:])
    match = _COLLECTED.search(proc.stdout)
    assert match, ("pytest printed no collection summary", proc.stdout[-2000:])
    return int(match.group(1))


def test_the_readme_states_the_number_of_tests_there_are():
    text = README.read_text(encoding="utf-8")
    match = _CLAIM.search(text)
    assert match, "README no longer spells its test count in the pinned form"
    stated = int(match.group(1).replace(",", ""))
    counted = _collected()
    assert stated == counted, (
        f"README.md says {stated} tests; a collection finds {counted}. "
        f"Write {counted}."
    )


def test_the_count_is_not_something_the_guard_could_invent():
    """The sibling assertion.

    Every part of the check above has a way of passing while measuring
    nothing: a collection that returned zero, a README whose number is the
    empty string, a regex that matched a sentence somewhere else. This says
    the number is real and that there is exactly ONE claim to pin.
    """
    text = README.read_text(encoding="utf-8")
    assert len(_CLAIM.findall(text)) == 1, "the README states its count twice"
    assert _collected() > 1000
