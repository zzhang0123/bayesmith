#!/usr/bin/env python3
"""Regenerate `tests/numerical_gates/source_manifest.py` from the scanner.

The manifest is a 9500-line checked-in literal, and until now it had no
generator: `tests/numerical_gates/test_registry.py` asserts ORDERED equality
between `scan_repository(ROOT)` and `EXPECTED_CANDIDATE_IDS`, so adding one
module to `SOURCE_PATHS` meant re-typing every entry the scan produced, by
hand, in scan order.  That is the arrangement in which a wrong row survives.

    PYTHONPATH=. .venv/bin/python tools/sync_source_manifest.py --check
    PYTHONPATH=. .venv/bin/python tools/sync_source_manifest.py --write

**The classification is not generated.** A candidate's
`CandidateClassification` is the human judgement this directory exists to
record, so this script carries an existing candidate's classification forward
by id and REFUSES to write when a new candidate has none.  It prints the new
ids with their source text and exits non-zero; classify them on the command
line and run again:

    ... --write --classify '<candidate id>=GOVERNED_THRESHOLD'

Carrying forward by id is safe because the id contains the normalized-AST
fingerprint: edit the expression and the id changes, so the old classification
is dropped rather than silently applied to different code.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.numerical_gates.source_scan import (
    CandidateClassification,
    scan_repository,
)

TARGET = ROOT / "tests" / "numerical_gates" / "source_manifest.py"

HEADER = '''"""Literal classified snapshot of the numerical-gate source census."""

from tests.numerical_gates.source_scan import (
    CandidateClassification,
    ManifestEntry,
)

EXPECTED_SOURCE_MANIFEST = (
'''

FOOTER = """)

EXPECTED_CANDIDATE_IDS = tuple(entry.candidate_id for entry in EXPECTED_SOURCE_MANIFEST)
"""


def _quote(text: str) -> str:
    """Render a Python string literal, preferring double quotes.

    The checked-in file is double-quoted throughout and many candidates carry
    single quotes inside them (``raise ValueError('...')``), so `repr` alone
    would flip quoting style on exactly those rows and produce a diff that is
    all noise.
    """
    body = (
        text.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace('"', '\\"')
    )
    return f'"{body}"'


#: A comment line sitting above a ``ManifestEntry(``.
_COMMENT = re.compile(r"^\s*#")
#: The candidate id on the first line of a ``ManifestEntry(`` body.
_ENTRY_ID = re.compile(
    r"""^\s+("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')\s*,\s*$"""
)


def _existing_comments() -> dict[str, str]:
    """Hand-written comment blocks, keyed by the candidate id they precede.

    The checked-in manifest carries section comments that say WHY a group of
    rows is classified the way it is -- the R3 batch left one naming D104 and
    D105 by hand.  A generator that re-derives the rows and drops those
    comments would delete the only record of the judgement, so they are
    carried across by the id of the row they sit above.
    """
    if not TARGET.exists():
        return {}
    comments: dict[str, str] = {}
    pending: list[str] = []
    lines = TARGET.read_text().splitlines()
    for index, line in enumerate(lines):
        if _COMMENT.match(line):
            pending.append(line)
            continue
        if line.strip().startswith("ManifestEntry("):
            if pending and index + 1 < len(lines):
                match = _ENTRY_ID.match(lines[index + 1])
                if match:
                    comments[ast.literal_eval(match.group(1))] = "\n".join(pending)
            pending = []
            continue
        if line.strip():
            pending = []
    return comments


def _existing_classifications() -> dict[str, CandidateClassification]:
    """The classification of every candidate the current manifest records."""
    if not TARGET.exists():
        return {}
    import importlib

    module = importlib.import_module("tests.numerical_gates.source_manifest")
    return {
        entry.candidate_id: entry.classification
        for entry in module.EXPECTED_SOURCE_MANIFEST
    }


def render(
    classifications: dict[str, CandidateClassification],
    comments: dict[str, str] | None = None,
) -> str:
    """The manifest file's full text, in scan order.

    Scan order is the contract: `test_current_source_scan_exactly_matches_the_
    checked_in_manifest` compares the two id tuples with `==`, not as sets.
    """
    comments = comments or {}
    parts = [HEADER]
    for candidate in scan_repository(ROOT):
        classification = classifications[candidate.candidate_id]
        comment = comments.get(candidate.candidate_id)
        if comment:
            parts.append(comment + "\n")
        parts.append(
            "    ManifestEntry(\n"
            f"        {_quote(candidate.candidate_id)},\n"
            f"        CandidateClassification.{classification.name},\n"
            f"        {_quote(candidate.syntax)},\n"
            "    ),\n"
        )
    parts.append(FOOTER)
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="rewrite the manifest")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report whether the manifest is already what the scan produces",
    )
    parser.add_argument(
        "--classify",
        action="append",
        default=[],
        metavar="ID=CLASSIFICATION",
        help="classify a candidate the current manifest does not have",
    )
    arguments = parser.parse_args()

    known = _existing_classifications()
    for pair in arguments.classify:
        candidate_id, _, name = pair.partition("=")
        known[candidate_id] = CandidateClassification[name]

    scanned = scan_repository(ROOT)
    unclassified = [c for c in scanned if c.candidate_id not in known]
    if unclassified:
        print(
            f"{len(unclassified)} candidate(s) have no classification. A "
            "classification is a judgement, not a derivation, so this script "
            "will not invent one. Re-run with --classify ID=NAME for each:",
            file=sys.stderr,
        )
        for candidate in unclassified:
            print(
                f"  {candidate.candidate_id}\n"
                f"      line {candidate.lineno}: {candidate.syntax}",
                file=sys.stderr,
            )
        print(
            "  legal names: "
            + ", ".join(item.name for item in CandidateClassification),
            file=sys.stderr,
        )
        return 2

    rendered = render(known, _existing_comments())
    current = TARGET.read_text() if TARGET.exists() else ""
    if arguments.check or not arguments.write:
        identical = rendered == current
        print(
            f"{len(scanned)} candidates scanned; manifest is "
            + ("byte-identical" if identical else "DIFFERENT")
        )
        return 0 if identical else 1

    TARGET.write_text(rendered)
    print(f"{TARGET.relative_to(ROOT)}: {len(scanned)} candidates written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
