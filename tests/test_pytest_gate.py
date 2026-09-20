"""``tools/pytest_gate.py``: the guard, and the bypasses it must not have.

The habit this gate replaces printed its verdict unconditionally::

    ps aux | grep 'pytest' | grep -v grep | head -3; echo "--- no other run ---"

On 2026-09-05 that printed "no other run" on the line directly below another
session's run. Nothing was wrong with the ``ps``; the label was joined with
``;``, so it fired whatever ``ps`` found. A check whose report cannot
distinguish its two cases is the family ``AGENTS.md`` records under the zsh
glob, the stale ruff cache, and the guard that reads a spelling.

So the verdict is the EXIT CODE, and this file holds it to four things.

**The matcher must not see itself.** ``pytest_gate.py`` contains the substring
``pytest``, so a gate matching that substring reports its own process and
refuses forever -- a guard whose only reliable finding is its reflection. That
is the bypass built and run here rather than reasoned about, in the
``_MUST_NOT_MATCH`` rows.

**The matcher must not see a process that merely QUOTES the command.** The
first bypass was anticipated; this one was not, and it cost thirteen minutes
of a deadlock on 2026-09-20. A ``/bin/zsh -c`` wrapper that had written a
queued-run script via a heredoc kept the script's text in its own argv,
``-m pytest -n 4 -m "not full"`` included. The gate counted it; the ``until
gate; do sleep 30; done`` loop inside that script waited for the gate to
clear; the gate was reporting that loop's own parent. The run directory was
still empty when a second session killed all three processes.

The repair is to key the match on ``argv[0]`` and the interpreter's own
options rather than on a substring anywhere, which is this repository's
standing answer to a guard that reads a SPELLING: assert the consequence, not
the syntax. The bypass is built and run below in ``_WRAPPERS``, whose third
row is that process's actual command line -- 948 characters of it, read off
disk rather than retyped, with its hash and its untruncated tail asserted.

**The repair is not a strict narrowing, and calling it one was wrong.**
Measured over every fixture in this file, nine verdicts change: seven move
True to False, which are the false positives the substring matcher had, and
**two move False to True**, which are real runs it MISSED. ``python -mpytest``
failed the retired spelling because that required whitespace after ``-m``,
and ``.../bin/pytest-3.12`` failed it because that required ``/bin/pytest`` to
be followed by whitespace or the end of the string, and a version suffix is
neither. Every move is an improvement for its row's class
and none contradicts it, which is the property worth having; "it can only lose
false positives" is a different and false claim, and this paragraph exists so
nobody repeats it from the commit message.

**"Could not check" is not "clear".** ``ps`` failing must exit 2, never 0. A
gate that reports a clean machine when it never looked is the failure it exists
to prevent, and it fails in the direction that lets the run start.

**The exit codes are asserted through ``main``**, not by re-deriving the logic
here: the caller spells the gate ``pytest_gate.py && pytest ...``, so what has
to be true is a property of the process's exit status.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_GATE = pathlib.Path(__file__).resolve().parent.parent / "tools" / "pytest_gate.py"


def _module():
    """Load the tool by path; ``tools/`` is not a package on ``sys.path``."""
    spec = importlib.util.spec_from_file_location("pytest_gate", _GATE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: Command lines that ARE a suite run. The first four are the spellings this
#: repository's own recipes produce; the rest are interpreter forms that must
#: not be lost to a scan that stops at the first option it does not know.
#: ``-X dev`` in particular consumes the next token, so a scanner that treated
#: ``dev`` as a script path would miss the run behind it.
_MUST_MATCH = [
    "/Users/z/.venv/bin/python -m pytest -n 4 -m not full",
    "/Users/z/.venv/bin/python -m pytest tests/a.py --junit-xml=runs/x/junit.xml",
    "python -m  pytest --collect-only",
    "/Users/z/.venv/bin/pytest -n 4",
    ".venv/bin/python -m pytest -n 2 tests/marginal -q",
    "python3.12 -m pytest",
    "/opt/py/bin/pythonw -m pytest tests/",
    "python -u -m pytest -p no:cacheprovider",
    "python -X dev -m pytest",
    "python -W ignore -m pytest",
    "python -mpytest",
    "python --version -m pytest",
    "/Users/z/.venv/bin/pytest-3.12 -n 4",
]

#: Command lines that merely CONTAIN "pytest". The first two are the gate
#: itself -- the bypass this guard would have if it matched the substring.
_MUST_NOT_MATCH = [
    "/Users/z/.venv/bin/python tools/pytest_gate.py",
    "/Users/z/.venv/bin/python /abs/path/tools/pytest_gate.py --quiet",
    "python -m pytest_gate",
    "vim tests/test_pytest_helpers.py",
    "grep -rn pytest src/",
    # argv[0] is not pytest, so the old `/bin/pytest` substring was too loose.
    "vim /usr/local/bin/pytest",
    "less /Users/z/.venv/bin/pytest",
    # An xdist worker. Measured on this machine during an `-n 2` run: the
    # payload after `-c` is not argv, and the parent IS matched, so counting
    # these would only inflate the number the gate reports.
    "/Users/z/.venv/bin/python -u -c import sys;exec(eval(sys.stdin.readline()))",
    # A python process whose -c PAYLOAD quotes the command: the same bypass as
    # the shell wrappers below, one layer down.
    "python -c subprocess.run(['python', '-m', 'pytest', '-n', '4'])",
    # The SAFE way to queue a run, and it was never a false positive: a script
    # invoked by PATH puts only the path in argv, so the text inside it is
    # invisible to `ps`. That is the whole difference between the wrapper that
    # deadlocked and the one that did not -- the first was written and invoked
    # in one shell command, which left the writing shell alive with the
    # heredoc's body in its own argv. It is here so the contrast is recorded
    # where someone writing the next queued run will read it.
    "/bin/zsh /tmp/scratch/queued_fast.sh",
]

#: The argv of PID 77621 as ``ps`` printed it on 2026-09-20 while the deadlock
#: was live -- the process this whole repair exists for, captured rather than
#: reconstructed. 948 characters, sha256
#: ``f28ec70013a621efaa452f019a93dc4306672cffa7f608799ac8a61f32662228``,
#: asserted below so a later edit to this literal fails rather than drifts.
#:
#: Captured by the session that found the deadlock, with
#: ``ps -eo pid,etime,pcpu,command | grep "[b]ayesmith"``; etime 13:15, ppid
#: 6994, child 77625 running the script this argv is quoting. Whether ``ps``
#: capped the length cannot be proven, but the string ends with the wrapper's
#: canonical tail, ``&& pwd -P >| /tmp/claude-0fcd-cwd``, and a truncated
#: capture would not -- that is a check, asserted below, not an assumption.
#:
#: The ``\012`` sequences are four literal characters in the argv, not
#: newlines: the wrapper encodes the heredoc body that way, which is part of
#: why the retired matcher saw ``-m pytest -n 4`` as one flat line. Measured:
#: it latched onto the substring ``' -m pytest '`` from inside that body,
#: while the repair reads ``argv[0]`` and finds ``/bin/zsh``.
_DEADLOCKED_77621 = (
    '/bin/zsh -c source /Users/zzhang/.claude/shell-snapshots/snapsho'
    't-zsh-1789860965518-lqluek.sh 2>/dev/null || true && setopt NO_E'
    'XTENDED_GLOB NO_BARE_GLOB_QUAL 2>/dev/null || true && { \\builtin'
    " unalias -- 'unsetenv'; \\builtin unset -f -- 'unsetenv'; } >/dev"
    "/null 2>&1 || true && eval 'SP=/private/tmp/claude-501/-Users-zz"
    'hang-projects-bayesmith/2ed18cf0-338c-4b3a-a330-4c0bdb8451f6/scr'
    'atchpad && cat > "$SP/queued_fast.sh" <<\'"\'"\'SH\'"\'"\'\\012#!/bin/z'
    'sh\\012cd /Users/zzhang/projects/bayesmith\\012D=runs/t004/fastB-$'
    '(date +%Y%m%dT%H%M%S)\\012mkdir -p "$D"\\012echo "$D" > /tmp/t004_'
    'fastB_dir\\012until tools/pytest_gate.py --quiet >/dev/null 2>&1;'
    ' do sleep 30; done\\012.venv/bin/python -m pytest -n 4 -m "not fu'
    'll" --junit-xml="$D/junit.xml" > "$D/log" 2>&1\\012echo "PYTEST_E'
    'XIT=$?" > "$D/exit"\\012SH\\012chmod +x "$SP/queued_fast.sh" && "$'
    'SP/queued_fast.sh"; D=$(cat /tmp/t004_fastB_dir); cat "$D/exit";'
    ' tail -8 "$D/log"\' && pwd -P >| /tmp/claude-0fcd-cwd'
)

#: Its provenance, pinned. A fixture that quietly changed would still be a
#: string some matcher rejects; what would be lost is that it is THE string.
_DEADLOCKED_77621_SHA256 = (
    "f28ec70013a621efaa452f019a93dc4306672cffa7f608799ac8a61f32662228"
)
_DEADLOCKED_77621_TAIL = "&& pwd -P >| /tmp/claude-0fcd-cwd"


#: Wrappers that CARRY a run command as text and run no pytest at all.
#:
#: Provenance, because "measured" and "reconstructed" are worth telling apart
#: in a file whose whole subject is a check that could not tell two things
#: apart. Rows 1 and 2 were captured from `ps` here, from deliberate
#: reproductions of the 2026-09-20 deadlock. Row 3 is the REAL one,
#: :data:`_DEADLOCKED_77621`, captured from `ps` while that process was live
#: and read off disk rather than retyped. Rows 4 and 5 are the same shape
#: written out in the other shells and were never observed anywhere.
#:
#: Row 3 was a reconstruction from another session's prose for one commit, and
#: was labelled as one; it is now the string itself. That the label came
#: first and the string second is the order this file recommends -- a
#: reconstruction that says so is usable, one that reads as a measurement is
#: not.
_WRAPPERS = [
    (
        '/bin/zsh -c echo "would run: .venv/bin/python -m pytest -n 4 -m not'
        ' full"; sleep 20; true'
    ),
    (
        '/bin/zsh -c true && .venv/bin/python -m pytest -n 4 -m "not full"'
        " --junit-xml=x.xml ; sleep 25"
    ),
    _DEADLOCKED_77621,
    (
        "/bin/sh -c until tools/pytest_gate.py; do sleep 30; done;"
        " .venv/bin/python -m pytest -n 4"
    ),
    "bash -c .venv/bin/python -m pytest",
]


@pytest.mark.parametrize("command", _MUST_MATCH)
def test_a_suite_run_is_recognised(command):
    assert _module()._is_a_run(command), command


@pytest.mark.parametrize("command", _MUST_NOT_MATCH)
def test_the_gate_does_not_see_itself_or_its_own_name(command):
    """The reflection bypass, built and run rather than reasoned about."""
    assert not _module()._is_a_run(command), command


@pytest.mark.parametrize("command", _WRAPPERS)
def test_a_shell_that_only_QUOTES_the_command_is_not_a_run(command):
    """The deadlock bypass, built and run rather than reasoned about.

    Every line here contains a complete pytest command and none of these
    processes is running one. Under the substring matcher this file shipped
    with until 2026-09-20, every one of them matched -- which the
    anti-vacuity test below requires rather than assumes.
    """
    assert not _module()._is_a_run(command), command


def test_the_old_substring_matcher_would_have_failed_this_file():
    """Anti-vacuity: the rows above must actually exercise the repair.

    Without this, someone could delete ``_WRAPPERS``' teeth by weakening the
    fixtures and the suite would stay green while claiming to guard the
    deadlock. The retired spelling is reconstructed here and required to be
    WRONG about every wrapper, so the fixtures are pinned to be a real
    discrimination rather than six strings that happen to pass.
    """
    import re

    retired = re.compile(r"(?:^|\s)-m\s+pytest(?:\s|$)|/bin/pytest(?:\s|$)")
    fooled = [command for command in _WRAPPERS if retired.search(command)]
    assert len(fooled) == len(_WRAPPERS), (
        "these wrapper fixtures no longer fool the retired matcher, so they no "
        f"longer demonstrate the repair: {set(_WRAPPERS) - set(fooled)}"
    )
    assert all(_module()._is_a_run(command) for command in _MUST_MATCH)


def test_the_captured_argv_is_still_the_one_that_deadlocked():
    """The fixture's provenance, pinned so an edit fails rather than drifts.

    Three things, and each fails differently. The hash says the literal is
    byte-for-byte what `ps` printed. The length is asserted beside it because
    a hash mismatch alone does not say whether something was added or lost.
    The tail is the truncation check: `ps` capping the command field cannot be
    disproven directly, but a capped capture would not end with the wrapper's
    canonical `pwd -P` redirect, so ending there is evidence it did not.

    None of this makes the fixture a better TEST -- the row below would do its
    job as any string the retired matcher accepts. What it protects is the
    claim that this is THE string, which is the part a reader would otherwise
    have to take on trust.
    """
    import hashlib

    assert len(_DEADLOCKED_77621) == 948
    assert (
        hashlib.sha256(_DEADLOCKED_77621.encode()).hexdigest()
        == _DEADLOCKED_77621_SHA256
    )
    assert _DEADLOCKED_77621.endswith(_DEADLOCKED_77621_TAIL)
    # Literal backslash-zero-one-two, not newlines: it is one flat argv, which
    # is how a heredoc body came to sit inside a command line at all.
    assert "\\012" in _DEADLOCKED_77621 and "\n" not in _DEADLOCKED_77621


def test_the_real_deadlock_is_what_the_retired_matcher_accepted():
    """Not a rephrasing of the parametrized row: that one asserts the repair,
    this one asserts the BUG was real on this exact input.

    Without it the file could hold the true string and still not record that
    the retired matcher was fooled BY IT rather than by a reproduction of it.
    Measured: it latched onto ``' -m pytest '``, from inside the quoted
    heredoc body, while the repair reads argv[0] and finds ``/bin/zsh``.
    """
    import re

    retired = re.compile(r"(?:^|\s)-m\s+pytest(?:\s|$)|/bin/pytest(?:\s|$)")
    found = retired.search(_DEADLOCKED_77621)
    assert found is not None and found.group(0) == " -m pytest "
    assert _DEADLOCKED_77621.split()[0] == "/bin/zsh"
    assert not _module()._is_a_run(_DEADLOCKED_77621)


def test_the_repair_moves_verdicts_in_BOTH_directions():
    """The fixture set must exercise the change both ways, not just one.

    ``test_the_old_substring_matcher_would_have_failed_this_file`` above pins
    the false-POSITIVE direction: every wrapper fooled the retired matcher.
    Nothing pinned the other one, and there is another one -- the repair also
    catches two runs the substring matcher MISSED. Delete those two rows and
    the file would still be green while quietly recording the change as a pure
    narrowing, which is what both the commit message and a reviewer's summary
    called it before this was measured.

    Counts are deliberately NOT asserted: a row added later should not fail a
    test about direction. What is asserted is that neither direction is empty,
    and that every disagreement resolves in favour of the row's own class --
    so a future edit cannot introduce a change where the retired matcher was
    right and the repaired one is wrong.
    """
    import re

    retired = re.compile(r"(?:^|\s)-m\s+pytest(?:\s|$)|/bin/pytest(?:\s|$)")
    module = _module()
    gained, lost = [], []
    for rows, is_a_run in ((_MUST_MATCH, True), (_MUST_NOT_MATCH, False), (_WRAPPERS, False)):
        for command in rows:
            was, now = bool(retired.search(command)), module._is_a_run(command)
            if was == now:
                continue
            assert now is is_a_run, (
                f"the repair disagrees with this row's own class: {command!r} "
                f"is {'a run' if is_a_run else 'not a run'} and the repaired "
                f"matcher says {now}"
            )
            (gained if now else lost).append(command)

    assert lost, "no fixture demonstrates a false positive being dropped"
    assert gained, (
        "no fixture demonstrates a run the retired matcher MISSED. Two did: a "
        "`-m` with no space before the module name, and a console script with "
        "a version suffix. Without one of them this file records the repair as "
        "a strict narrowing, which it is not."
    )


def test_a_clear_machine_exits_zero(monkeypatch, capsys):
    module = _module()
    monkeypatch.setattr(
        module, "_processes", lambda: [(1, "/sbin/launchd"), (2, "vim notes.md")]
    )
    monkeypatch.setattr(sys, "argv", ["pytest_gate.py"])

    assert module.main() == 0
    assert "clear" in capsys.readouterr().out


def test_a_run_in_flight_exits_one(monkeypatch, capsys):
    module = _module()
    monkeypatch.setattr(
        module,
        "_processes",
        lambda: [(1, "/sbin/launchd"), (4242, "/v/bin/python -m pytest -n 4")],
    )
    monkeypatch.setattr(module, "_cwd_of", lambda pid: "/some/worktree")
    monkeypatch.setattr(sys, "argv", ["pytest_gate.py"])

    assert module.main() == 1
    captured = capsys.readouterr().err
    assert "IN FLIGHT" in captured
    # The cwd is what names WHICH session, and is the reason to print anything.
    assert "/some/worktree" in captured and "4242" in captured


def test_a_check_that_could_not_run_exits_two_and_not_zero(monkeypatch, capsys):
    """The direction that matters: unknown must not read as clear.

    Asserted as ``== 2`` rather than ``!= 0`` so that a future refactor cannot
    satisfy it by returning 1, which would read to a caller as "another run is
    in flight" -- a different claim, and one nothing measured.
    """
    module = _module()

    def boom():
        raise RuntimeError("ps exited 1: permission denied")

    monkeypatch.setattr(module, "_processes", boom)
    monkeypatch.setattr(sys, "argv", ["pytest_gate.py"])

    assert module.main() == 2
    assert "COULD NOT CHECK" in capsys.readouterr().err


def test_processes_raises_rather_than_returning_empty_when_ps_fails(monkeypatch):
    """``_processes`` must not fold a failed ``ps`` into "found nothing".

    ``main`` turns an empty list into exit 0, so a ``_processes`` that swallowed
    the failure would put the wrong verdict one layer down where the exit-code
    tests above cannot see it.
    """
    module = _module()

    class _Failed:
        returncode = 1
        stdout = ""
        stderr = "ps: permission denied"

    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: _Failed())
    with pytest.raises(RuntimeError, match="ps exited 1"):
        module._processes()
