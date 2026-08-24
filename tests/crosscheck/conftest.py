"""The one place the cross-check harness decides whether it can run at all.

``rheplicant`` is a local path install (see ``pyproject.toml``'s ``crosscheck``
group), so a clone without an e-RHINO checkout has to skip -- and a skip that
says only "module not found" is the shape this project's upstream has twice
paid for: a guard that stood down quietly while real failures sat behind it.
"""

from __future__ import annotations

import pytest

_WHY = (
    "rheplicant is not installed, so nothing in this directory compared "
    "anything. THIS IS NOT A PASS. These tests are the only mechanical link "
    "between this package and the one it was ported from; without them the "
    "two can diverge for as long as nobody reads both. Install it with\n"
    "    uv pip install --python .venv/bin/python --no-deps -e ../e-RHINO\n"
    "(--no-deps on purpose: see pyproject.toml's crosscheck group)."
)


@pytest.fixture(scope="session", autouse=True)
def _require_rheplicant() -> None:
    pytest.importorskip("rheplicant", reason=_WHY)
