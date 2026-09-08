"""``tools/pytest_gate.py``: the guard, and the bypasses it must not have.

The habit this gate replaces printed its verdict unconditionally::

    ps aux | grep 'pytest' | grep -v grep | head -3; echo "--- no other run ---"

On 2026-09-05 that printed "no other run" on the line directly below another
session's run. Nothing was wrong with the ``ps``; the label was joined with
``;``, so it fired whatever ``ps`` found. A check whose report cannot
distinguish its two cases is the family ``AGENTS.md`` records under the zsh
glob, the stale ruff cache, and the guard that reads a spelling.

So the verdict is the EXIT CODE, and this file holds it to three things.

**The matcher must not see itself.** ``pytest_gate.py`` contains the substring
``pytest``, so a gate matching that substring reports its own process and
refuses forever -- a guard whose only reliable finding is its reflection. That
is the bypass built and run here rather than reasoned about, in the
``_MUST_NOT_MATCH`` rows.

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


#: Command lines that ARE a suite run, in both spellings one takes.
_MUST_MATCH = [
    "/Users/z/.venv/bin/python -m pytest -n 4 -m not full",
    "/Users/z/.venv/bin/python -m pytest tests/a.py --junit-xml=runs/x/junit.xml",
    "python -m  pytest --collect-only",
    "/Users/z/.venv/bin/pytest -n 4",
]

#: Command lines that merely CONTAIN "pytest". The first two are the gate
#: itself -- the bypass this guard would have if it matched the substring.
_MUST_NOT_MATCH = [
    "/Users/z/.venv/bin/python tools/pytest_gate.py",
    "/Users/z/.venv/bin/python /abs/path/tools/pytest_gate.py --quiet",
    "python -m pytest_gate",
    "vim tests/test_pytest_helpers.py",
    "grep -rn pytest src/",
]


@pytest.mark.parametrize("command", _MUST_MATCH)
def test_a_suite_run_is_recognised(command):
    assert _module()._PYTEST.search(command), command


@pytest.mark.parametrize("command", _MUST_NOT_MATCH)
def test_the_gate_does_not_see_itself_or_its_own_name(command):
    """The reflection bypass, built and run rather than reasoned about."""
    assert not _module()._PYTEST.search(command), command


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
