"""The examples stay runnable, or the suite says so.

Each demo runs as a subprocess at ``--quick`` size: the point is that the
scripts the README and the docs point at execute end to end against the
CURRENT package — imports, partition, sampling, reporting — not that their
short chains mix (the registered chain lengths live in the scripts; the
validation experiment scores mixing, by hand, at its own cost).

Subprocesses rather than imports, deliberately: the scripts patch
``sys.path`` and parse CLI arguments, and running them any other way would
test a different entry point from the one a reader types.
"""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.mark.slow
def test_three_routes_runs_and_prints_its_partition():
    done = _run("three_routes.py", "--quick")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "block 0" in done.stdout and "log-gcr" in done.stdout
    assert "prediction-space residual" in done.stdout


@pytest.mark.slow
def test_hierarchy_runs_and_the_two_parameterisations_agree():
    done = _run("hierarchy.py", "--quick")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "identical, as the ancestry rule requires" in done.stdout
    assert "ancestor of another latent's distribution" in done.stdout


@pytest.mark.slow
def test_the_validation_machinery_smokes():
    done = _run("validate_sampling.py", "--smoke")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "OVERALL: PASS" in done.stdout
    # The registered criteria must be stated before any verdict is printed --
    # the experiment's honesty rests on that ordering being real.
    assert "verdicts not scored" in done.stdout
