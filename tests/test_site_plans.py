"""The plans and tables the documentation shows must still be what the code does.

Pages render ``site/assets/plans/*.json``, recorded by
``tools/record_site_plans.py``. These tests re-measure in a subprocess,
because the recording runs with ``JAX_ENABLE_X64=1`` and precision must be
chosen before JAX starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/record_site_plans.py"
PLANS = ROOT / "site/assets/plans"
README = ROOT / "README.md"


def _check(*args, x64=True):
    env = {**os.environ, "JAX_ENABLE_X64": "1" if x64 else "0"}
    return subprocess.run(
        [sys.executable, str(TOOL), "--check", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=300,
    )


def _tampered(tmp_path, name, change):
    record = json.loads((PLANS / f"{name}.json").read_text(encoding="utf-8"))
    change(record)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_every_recorded_plan_matches_a_fresh_measurement():
    done = _check()
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_check_notices_a_changed_route_and_a_changed_snippet(tmp_path):
    def change(record):
        record["plans"]["mixed"]["blocks"][0]["route"] = "NUTS"
        record["snippet_sha256"] = "0" * 64

    done = _check(
        "--only", "overview", "--record", _tampered(tmp_path, "overview", change)
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert "the snippet changed after it was recorded" in done.stdout
    assert "the measured plan structure differs from the record" in done.stdout


def test_the_check_recomputes_closed_form_columns(tmp_path):
    def change(record):
        record["tables"]["recovery"]["rows"][0]["exact_mean"] += 1e-6

    done = _check(
        "--only", "partial", "--record", _tampered(tmp_path, "partial", change)
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert "recovery population exact_mean recorded" in done.stdout


def test_the_recorder_refuses_to_measure_without_x64():
    done = _check(x64=False)
    assert done.returncode != 0
    assert "JAX_ENABLE_X64=1" in done.stdout + done.stderr


def test_the_records_show_the_routes_the_pages_describe():
    def routes(name, plan):
        record = json.loads((PLANS / f"{name}.json").read_text(encoding="utf-8"))
        return [
            (row["latents"], row["route"]) for row in record["plans"][plan]["blocks"]
        ]

    assert routes("overview", "mixed") == [(["x"], "GCR exact"), (["nu"], "NUTS")]
    assert routes("partial", "hierarchy") == [
        (["groups"], "GCR exact"),
        (["population"], "NUTS"),
    ]
    assert routes("notebook", "model") == [
        (["groups"], "GCR exact"),
        (["population"], "NUTS"),
    ]
    assert routes("decay", "decay") == [(["rate"], "NUTS")]
    assert routes("first", "level") == [(["level"], "GCR exact")]


def test_the_readme_prints_lines_from_the_recorded_plan():
    """The README's opening plan is measured output, not an illustration.

    It carried a hand-written sketch until 2026-09-19, whose route labels
    ("Wiener exact", "enumerate 4 states") the package never prints.
    """
    recorded = json.loads((PLANS / "overview.json").read_text(encoding="utf-8"))
    text = recorded["plans"]["mixed"]["text"].splitlines()
    shown = [
        line
        for line in README.read_text(encoding="utf-8").splitlines()
        if line.startswith(("block ", "execution:"))
    ]
    assert shown, "the README no longer shows a plan"
    assert all(line in text for line in shown), [
        line for line in shown if line not in text
    ]
