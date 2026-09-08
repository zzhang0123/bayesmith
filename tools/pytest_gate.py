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

#: A pytest invocation, and NOT this script. Matching the bare substring
#: ``pytest`` would match ``pytest_gate.py`` itself, so the gate would report
#: itself as an in-flight run -- a check whose only reliable finding is its own
#: reflection. Both spellings a run actually takes: ``-m pytest``, and a
#: console script at ``.../bin/pytest``.
_PYTEST = re.compile(r"(?:^|\s)-m\s+pytest(?:\s|$)|/bin/pytest(?:\s|$)")


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
        if pid not in mine and _PYTEST.search(command)
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
