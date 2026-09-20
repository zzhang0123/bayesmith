#!/usr/bin/env python3
"""Refuse to start a suite run while another one is in flight.

Three sessions share this checkout. The habit this replaces was::

    ps aux | grep 'pytest' | grep -v grep | head -3; echo "--- no other run ---"

which prints its verdict with ``;`` rather than ``&&``, so the label fires
whatever ``ps`` found -- and on 2026-09-05 it printed "no other run" directly
underneath a line showing another session's run. The label could not
distinguish its two cases, which is the disease ``AGENTS.md`` catalogues under
the zsh glob and the stale ruff cache.

So the verdict is the EXIT CODE, and the caller spells the gate with ``&&``::

    tools/pytest_gate.py && .venv/bin/python -m pytest -n 4 -m "not full"

``0`` no other run, ``1`` at least one in flight, ``2`` the check itself could
not run. The third is why ``ps`` failing is not folded into "found nothing":
a gate that reports a clean tree when it did not look is the failure it exists
to prevent.

Contention between runs produces false FAILURES, not false passes, so a green
result taken during someone else's run is still green. What this protects is
the reverse -- a red result that is really CPU starvation, and the hour spent
chasing it.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

#: ``argv[0]``'s basename for a Python interpreter, as ``ps`` prints it:
#: ``python``, ``python3``, ``python3.12``, ``pythonw``.
_INTERPRETER = re.compile(r"^python[0-9.]*w?$")

#: ``argv[0]``'s basename for a pytest console script.
_CONSOLE = re.compile(r"^pytest(?:-[0-9.]+)?$")

#: Short interpreter options that consume the NEXT token as their value, so a
#: scan for ``-m`` must step over it. ``-c`` and ``-m`` are handled separately
#: because they end the option list rather than continuing it.
_TAKES_VALUE = "WX"


def _is_a_run(command: str) -> bool:
    """Whether ``command`` is a process RUNNING pytest.

    The question is deliberately not "does this command line mention pytest".
    A substring match cannot tell a process running the suite from a process
    that merely carries the command as TEXT, and this gate was wrong in
    exactly that way: measured 2026-09-20, a ``/bin/zsh -c`` wrapper that had
    written a queued-run script via a heredoc kept the script's own text --
    including ``-m pytest -n 4 -m "not full"`` -- in its argv, so the gate
    counted it as a run in flight. The loop inside that script then waited for
    the gate to clear, and the gate was reporting the script's own parent.
    Neither could exit. Thirteen minutes, and the run directory stayed empty.

    That is the fourth member of the family ``AGENTS.md`` catalogues after the
    zsh glob, the stale ruff cache and the guard that reads a spelling: a check
    that cannot distinguish the thing it names from a thing that resembles it.
    The docstring of the old matcher had anticipated the gate seeing ITSELF and
    guarded against that; it had not anticipated a caller quoting the command.

    So the match is keyed on ``argv[0]`` and on the interpreter's own options,
    never on a substring anywhere:

    * ``argv[0]`` is a pytest console script -- ``.../bin/pytest`` -- which is
      a run whatever follows. The old spelling matched ``/bin/pytest``
      anywhere, so ``vim /usr/bin/pytest`` was a run too.
    * ``argv[0]`` is a Python interpreter AND ``-m pytest`` is one of ITS
      options. A ``-c`` reached first ends the scan: everything after it is a
      payload string, not argv, which is the same distinction one layer down.

    A wrapper shell fails the first test on its basename and never reaches the
    second, so a shell carrying the command as text is not a run. Nothing is
    missed by that: a shell that actually starts pytest starts it as a CHILD
    process, which is matched on its own argv, and a shell that ``exec``s it
    is replaced by it. Measured on this machine, both halves -- the parent of
    an ``-n 2`` run is ``.venv/bin/python -m pytest -n 2 tests/marginal -q``
    and matches, and its two xdist workers are
    ``.venv/bin/python -u -c import sys;exec(eval(sys.stdin.readline()))``
    and do not, which is what the old matcher did as well.

    Known limits, stated rather than left to be discovered. ``ps`` joins argv
    with spaces, so an interpreter path containing a space is split and missed;
    a run started through a launcher (``sudo``, ``env FOO=1``, ``nice``) is
    read as that launcher and missed; and ``python -c "import pytest;
    pytest.main()"`` is missed, as it was before. All three fail toward "no run
    found", which costs a contended run -- false FAILURES, never false passes
    -- rather than the deadlock this replaces.
    """
    tokens = command.split()
    if not tokens:
        return False
    head = tokens[0].rsplit("/", 1)[-1]
    if _CONSOLE.match(head):
        return True
    if not _INTERPRETER.match(head):
        return False

    index = 1
    while index < len(tokens):
        token = tokens[index]
        # A bare word here is the script the interpreter runs, so this is
        # `python some_script.py` and anything after belongs to the script.
        # That is what keeps `python tools/pytest_gate.py` from matching.
        if not token.startswith("-") or token == "--":
            return False
        if token.startswith("--"):
            index += 1
            continue
        letters = token[1:]
        where_c, where_m = letters.find("c"), letters.find("m")
        if where_c != -1 and (where_m == -1 or where_c < where_m):
            return False
        if where_m != -1:
            attached = letters[where_m + 1 :]
            if attached:
                return attached == "pytest"
            return tokens[index + 1 : index + 2] == ["pytest"]
        index += 2 if letters and letters[-1] in _TAKES_VALUE else 1
    return False


def _processes() -> list[tuple[int, str]]:
    """``(pid, command)`` for every process on this machine.

    Raises rather than returning ``[]`` when ``ps`` fails: the caller turns an
    empty list into "clear to run", and this function must never say that
    about a question it did not get to ask.
    """
    proc = subprocess.run(
        ["ps", "-Ao", "pid=,command="], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ps exited {proc.returncode}: {proc.stderr.strip()[:200]}")
    found = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid, _, command = stripped.partition(" ")
        if pid.isdigit():
            found.append((int(pid), command.strip()))
    return found


def _cwd_of(pid: int) -> str | None:
    """The working directory of ``pid``, which is what names the session.

    ``lsof`` needs no privilege for one's own processes and is the only way to
    get another process's cwd on macOS. ``None`` when it cannot be read -- the
    command line is then the only identification available, and it is usually
    enough, because a run's ``--junit-xml`` path carries its worktree.
    """
    proc = subprocess.run(
        ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("n/"):
            return line[1:]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print nothing when the tree is clear; the exit code still says so",
    )
    args = parser.parse_args()

    mine = {os.getpid(), os.getppid()}
    try:
        candidates = _processes()
    except (RuntimeError, OSError) as exc:
        print(f"pytest_gate: COULD NOT CHECK -- {exc}", file=sys.stderr)
        print(
            "pytest_gate: this is not 'no other run'. Re-run the gate, or "
            "check by hand before starting a suite.",
            file=sys.stderr,
        )
        return 2

    running = [
        (pid, command)
        for pid, command in candidates
        if pid not in mine and _is_a_run(command)
    ]

    if not running:
        if not args.quiet:
            print(
                f"pytest_gate: clear -- {len(candidates)} processes examined, "
                f"no pytest run in flight"
            )
        return 0

    print(f"pytest_gate: {len(running)} pytest run(s) IN FLIGHT", file=sys.stderr)
    for pid, command in running:
        where = _cwd_of(pid)
        print(f"  pid {pid}  cwd {where or '(unreadable)'}", file=sys.stderr)
        print(f"    {command[:160]}", file=sys.stderr)
    print(
        "pytest_gate: refusing. Wait for these, or run anyway knowing that "
        "contention makes false failures (never false passes).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
