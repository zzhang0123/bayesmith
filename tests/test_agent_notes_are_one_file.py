"""`AGENTS.md` and `CLAUDE.md` are one file, and this is the price of that.

The two documents exist because two families of tool look for two names.
Keeping both is a second copy, and an unenforced second copy is the defect
this project has spent the most time repairing — its own working notes open by
saying so, and by naming the terms on which the ruling against `AGENTS.md`
could be revisited: *the price of keeping it is the identity test, not a good
intention.*

The failure that reversed the ruling is worth stating because it is the exact
one the ruling was written to prevent. On 2026-08-31 an **untracked**
`AGENTS.md` was found in the checkout, dated four days earlier and no longer
identical: it still carried the pre-layering `1280 passed` recipe and knew
nothing about the fast/full split. An agent session read it as this
repository's instructions. "Delete it rather than commit it" is not a
mechanism — a deleted file is a file nothing checks, so it came back, stale,
and no run could have said so.

So: both tracked, compared byte for byte, and editing either one alone turns
the suite red. That is one file with two names, which is not the same thing as
two files and a hope.

**Bytes, not sentences.** A comparison that normalised whitespace or ignored
trailing newlines would pass on a pair that renders differently, and the
failure mode being guarded is a copy that drifted a little at a time.
"""

from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLAUDE = ROOT / "CLAUDE.md"
AGENTS = ROOT / "AGENTS.md"


def test_both_agent_note_files_are_tracked_and_present():
    """An absent `AGENTS.md` is how the stale one got back in.

    The tools that look for a repository's working notes look for this name
    first; leaving it to be recreated by hand is what produced a four-day-old
    copy nobody could see was wrong.
    """
    assert CLAUDE.is_file(), "CLAUDE.md is missing"
    assert AGENTS.is_file(), (
        "AGENTS.md is missing. It is not optional: it is the name most agent "
        "tools look for, and re-creating it by hand is exactly how a stale "
        "copy last got into this checkout. Copy CLAUDE.md to it."
    )


def test_the_two_agent_note_files_are_byte_identical():
    claude = CLAUDE.read_bytes()
    agents = AGENTS.read_bytes()
    assert agents == claude, (
        "AGENTS.md and CLAUDE.md have drifted. They are one document under "
        "two names, because two families of tool look for two names. Edit one "
        "and copy it over the other:  cp CLAUDE.md AGENTS.md"
    )


def test_the_comparison_could_not_pass_on_two_empty_files():
    """The sibling assertion: an identity check over nothing is not a check."""
    assert len(CLAUDE.read_bytes()) > 4000
    assert "Working notes for coding agents" in CLAUDE.read_text(encoding="utf-8")
