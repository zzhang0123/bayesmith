"""Published numerical figures must remain attached to their recorded source."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "site/assets/results"


def check(directory):
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/record_site_results.py"),
            "--check",
            "--record-dir",
            str(directory),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_result_provenance_and_closed_form():
    done = check(RESULTS)
    assert done.returncode == 0, done.stdout + done.stderr


def test_changed_figure_is_not_accepted_as_recorded_output(tmp_path):
    shutil.copytree(RESULTS, tmp_path / "results")
    (tmp_path / "results/first-posterior.svg").write_text("unrelated figure")
    done = check(tmp_path / "results")
    assert done.returncode != 0
    assert "Figure changed or missing: first-posterior.svg" in done.stderr


def test_changed_source_and_false_closed_form_are_detected(tmp_path):
    shutil.copytree(RESULTS, tmp_path / "results")
    path = tmp_path / "results/run.json"
    data = json.loads(path.read_text())
    data["sources"]["first_model.py"] = "0" * 64
    data["tables"]["first"]["rows"][0]["exact"] += 0.01
    path.write_text(json.dumps(data))
    done = check(path.parent)
    assert done.returncode != 0
    assert "Source changed: first_model.py" in done.stderr
    assert "Incorrect closed-form Posterior mean" in done.stderr
