"""Every actual package module has an explicit implementation owner."""

import fnmatch
import re
from pathlib import Path


def test_ownership_inventory_covers_the_actual_package():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/ownership.md").read_text()
    rows = [line.split("|")[1] for line in text.splitlines() if line.startswith("| ")]
    patterns = [
        name for row in rows for name in re.findall(r"`(bayesmith[.\w*]+)`", row)
    ]
    modules = [
        ".".join(path.relative_to(root / "src").with_suffix("").parts)
        for path in (root / "src/bayesmith").rglob("*.py")
    ]
    assert len(modules) > 50, "ownership audit did not find the package source"
    uncovered = [
        name
        for name in modules
        if not any(fnmatch.fnmatchcase(name, p) for p in patterns)
    ]
    assert not uncovered, f"package modules without an owner: {uncovered}"
