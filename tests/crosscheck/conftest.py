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


@pytest.fixture(autouse=True)
def _heal_the_global_x64_flag():
    """Restore the process-global x64 flag a cross-package call can leak.

    The mechanism, measured under xdist: rheplicant's diagnostics force x64
    with a save/restore built on ``jax.config.read`` + ``update``. Run
    INSIDE this harness's ``with jax.enable_x64(True):`` block, that
    ``read`` returns the EFFECTIVE value -- True, supplied by the context
    manager -- and the ``finally`` then writes True into the GLOBAL config.
    The context manager exits, its thread-local override disappears, and
    the global True remains: every later test in the same worker runs x64
    without asking. The visible symptom was three dispatch tests two
    directories away failing with "output array is read-only", because
    ``np.asarray(jax_float64_array, float)`` is a zero-copy read-only view
    where the float32 path had been a writable copy.

    Setup and teardown both run OUTSIDE any test-body context manager, so
    ``read`` here IS the global value, and putting it back heals the leak
    whichever package caused it.
    """
    import jax

    was = jax.config.read("jax_enable_x64")
    yield
    if jax.config.read("jax_enable_x64") != was:
        jax.config.update("jax_enable_x64", was)
