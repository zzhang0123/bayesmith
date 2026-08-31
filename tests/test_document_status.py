"""Every document in `docs/` says what authority it has, and the index agrees.

The owner re-established a single normative top-level design on 2026-08-30.
Before that there were sixty-one spec pages and eight plan pages, four of which
carried any status marker at all, so **the authority of a document was
something a reader had to reconstruct from its filename and its date.** That is
the same defect this repository has paid for repeatedly in prose: a claim
nothing checks reads exactly as authoritative as one that is true. A batch page
from 2026-08-27, written as a work log for one day, is indistinguishable at a
glance from the spec that still governs a shipped module.

This module is the guard for the fix. Each page under `docs/` (outside
`docs/migration/`, which has its own regime and its own test) carries ONE
machine-readable status line, and `docs/README.md` indexes them. The two are
checked against each other in BOTH directions, so:

* a new page with no status fails here rather than joining the pile;
* a page that is stamped but missing from the index fails;
* an index row for a page that no longer exists fails;
* a status changed in one place and not the other fails.

**The status lives in the page, not in the index.** The index is derived and
checked; the page is the single home, which is section 10.1 of the top-level
design applied to the record itself. Writing the statuses only in the index
would recreate exactly the two-copies-and-a-hope arrangement the working notes
name as this project's most expensive recurring defect.

**Why `superseded` must name its successor.** "This is old" is not actionable;
"this was replaced by that file" is. The successor is resolved on disk here, so
a rename breaks the pointer loudly instead of leaving a reader to guess.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = DOCS / "README.md"

#: Directories under `docs/` this guard does not govern.
#:
#: `migration/` has its own index and its own bidirectional test
#: (`tests/test_migration_records.py`); giving it a second regime would be the
#: duplication this file exists to prevent. `probes/` and `derivations/` hold
#: executable artefacts rather than pages.
_EXEMPT_DIRS = ("migration", "probes", "derivations")

#: The pinned status line. One per page, in the first few lines, rendered as a
#: blockquote so a human reading the page sees it before anything else.
_STATUS = re.compile(r"^> \*\*文档状态：`(?P<status>[a-z-]+)`\*\*", re.MULTILINE)

#: The successor a `superseded` page must name, on that same line.
_SUPERSEDED_BY = re.compile(
    r"^> \*\*文档状态：`superseded`\*\*.*?`(?P<target>[\w./-]+\.md)`", re.MULTILINE
)

#: What each token claims, and therefore what a reader may rely on.
STATUSES = {
    # The one authoritative document. Its own section 0 says so.
    "normative": "唯一的顶层设计，冲突时以它为准",
    # Current for a shipped module: changing that code invalidates this page.
    "module-spec": "已发布模块/能力的当前设计文档，从属于顶层设计",
    # A living register where decisions are recorded AND resolved.
    "decision-home": "某类决定的唯一登记处，仍在更新",
    # A plan that still governs future work.
    "plan-active": "尚未执行完的计划，仍指导后续工作",
    # Archaeology: true of the day it was written, not authoritative today.
    "record": "已落地批次/审计/测量的历史记录，非当前权威",
    # Replaced, and it must say by what.
    "superseded": "已被指名的后继文档取代",
}

#: The index table's rows: path, status and the page's own title.
#:
#: The title is DERIVED from the page's first H1 rather than written twice, so
#: the third column cannot drift into describing a page that has since been
#: retitled. A page with no H1 at all is the one case this cannot check, and
#: there is exactly one; it is required to carry its file stem instead, which
#: fails loudly if the file is renamed.
_INDEX_ROW = re.compile(
    r"^\| `(?P<path>[\w./-]+\.md)` \| `(?P<status>[a-z-]+)` \| (?P<title>.*?) \|$",
    re.MULTILINE,
)

_H1 = re.compile(r"^# (.+)$", re.MULTILINE)


def _title_of(path: pathlib.Path) -> str:
    match = _H1.search(path.read_text(encoding="utf-8"))
    return match.group(1).strip() if match else path.stem


def _pages() -> list[pathlib.Path]:
    """Every governed page, DISCOVERED rather than listed.

    A hard-coded list would let a new document join the tree without a status,
    which is the exact failure this guard exists to stop -- so the corpus comes
    from the filesystem and only the statuses are written down.
    """
    return sorted(
        path
        for path in DOCS.rglob("*.md")
        if path != INDEX
        and not any(part in _EXEMPT_DIRS for part in path.relative_to(DOCS).parts)
    )


def _status_of(path: pathlib.Path) -> str:
    text = path.read_text(encoding="utf-8")
    found = _STATUS.findall(text)
    assert len(found) == 1, (
        f"{path.relative_to(ROOT)} must carry exactly one status line of the "
        f"form '> **文档状态：`<status>`** · ...'; found {len(found)}. "
        f"Allowed: {sorted(STATUSES)}."
    )
    return found[0]


def test_every_document_declares_one_status_from_the_allowed_set():
    for path in _pages():
        status = _status_of(path)
        assert status in STATUSES, (
            f"{path.relative_to(ROOT)} declares an unknown status {status!r}; "
            f"allowed: {sorted(STATUSES)}"
        )


def test_exactly_one_document_is_normative():
    """Two normative documents is the state this cleanup was called to end.

    The top-level design's section 0 claims sole authority over positioning,
    layering, public concepts and build order. A second page claiming the same
    thing would make that sentence false by existing -- which is precisely how
    the previous top-level design and architecture narrative came to be
    superseded rather than deleted.
    """
    normative = [p for p in _pages() if _status_of(p) == "normative"]
    assert [p.relative_to(ROOT).as_posix() for p in normative] == [
        "docs/superpowers/specs/2026-08-30-bayesmith-top-level-design.md"
    ]


def test_every_superseded_document_names_a_successor_that_exists():
    for path in _pages():
        if _status_of(path) != "superseded":
            continue
        text = path.read_text(encoding="utf-8")
        match = _SUPERSEDED_BY.search(text)
        assert match, (
            f"{path.relative_to(ROOT)} is superseded but its status line names "
            "no successor. 'This is old' is not actionable; name the file."
        )
        target = match.group("target")
        candidates = [DOCS / target, path.parent / target, ROOT / target]
        assert any(c.exists() for c in candidates), (
            f"{path.relative_to(ROOT)} names successor {target!r}, which does "
            "not exist. A renamed successor must break loudly here."
        )


def test_the_index_and_the_pages_agree_in_both_directions():
    """The index is derived and checked; the page is the single home.

    Both directions, because each failure is a different mistake: a page
    missing from the index is a document nobody will find, and an index row
    without a page is a pointer into nothing.
    """
    indexed = {
        match.group("path"): match.group("status")
        for match in _INDEX_ROW.finditer(INDEX.read_text(encoding="utf-8"))
    }
    stamped = {
        path.relative_to(ROOT).as_posix(): _status_of(path) for path in _pages()
    }

    missing = sorted(set(stamped) - set(indexed))
    orphan = sorted(set(indexed) - set(stamped))
    assert not missing, f"pages absent from docs/README.md: {missing}"
    assert not orphan, f"docs/README.md rows with no page: {orphan}"

    disagree = sorted(k for k in stamped if stamped[k] != indexed[k])
    assert not disagree, (
        "docs/README.md disagrees with the page's own status line for: "
        f"{[(k, stamped[k], indexed[k]) for k in disagree]}"
    )


def test_the_index_titles_are_the_pages_own_titles():
    """The third column is derived, so it cannot describe a stale page.

    An index whose prose was written by hand is the same two-copies
    arrangement as the statuses, one column over. This reads each page's own
    first H1 and requires the index to repeat it exactly.
    """
    titled = {
        match.group("path"): match.group("title")
        for match in _INDEX_ROW.finditer(INDEX.read_text(encoding="utf-8"))
    }
    wrong = [
        (relative, titled[relative], _title_of(ROOT / relative))
        for relative in sorted(titled)
        if titled[relative] != _title_of(ROOT / relative)
    ]
    assert not wrong, f"docs/README.md titles differ from the pages': {wrong}"


def test_the_guard_could_not_pass_on_an_empty_corpus():
    """The sibling assertion.

    Every check above passes vacuously against zero pages or an index that
    matched nothing. This says both sides are real, and that the taxonomy is
    actually used rather than being a list of tokens nobody applies.
    """
    pages = _pages()
    assert len(pages) > 50
    statuses = {_status_of(path) for path in pages}
    assert len(statuses) >= 4, (
        f"only {sorted(statuses)} in use; a taxonomy nobody applies is a "
        "taxonomy that is not doing any work"
    )
    assert len(_INDEX_ROW.findall(INDEX.read_text(encoding="utf-8"))) == len(pages)
