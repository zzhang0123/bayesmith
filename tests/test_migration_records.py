"""The §二 step 6 gate, made mechanical.

§二 requires one ``docs/migration/<module>.md`` per migrated module and
gates every part of §六 on them. Prose cannot enforce that -- the directory
was absent for the whole of B9 and B11 and no test noticed, which is
precisely how §5.14 came to be written.

What is derived here rather than re-spelled:

* the module list comes from the SPEC's own §四 tables, parsed;
* the page list comes from the directory;
* the test list comes from ``tests/crosscheck/``.

So a new §四 row, a new page or a new cross-check file changes what this
module asserts without anyone editing it. The one thing written down is the
current STATE -- which rows have a page today -- and it is written as an
exact set, so both directions fail: a page that disappears fails, and a page
that appears without this set being updated fails too, which is the moment
to ask whether §六's gate has moved.

**It lives in ``tests/``, not in ``tests/crosscheck/``, and that is a
measurement rather than a preference.** It compares this repository against
itself and needs no rheplicant, so it must run everywhere -- but
``tests/crosscheck/conftest.py`` carries a session-scoped AUTOUSE
``importorskip("rheplicant")``, which applies to every test in that
directory regardless of marker. Measured by pointing that importorskip at a
name that cannot resolve: all seven tests here reported ``s``. Left there,
this gate would have stood down on exactly the machines where nobody
notices -- the failure ``tests/test_readme_counts.py`` upstream already paid
for once.
"""

from __future__ import annotations

import pathlib
import re

DOCS = pathlib.Path(__file__).resolve().parents[1] / "docs"
MIGRATION = DOCS / "migration"
SPEC = DOCS / "superpowers" / "specs" / "2026-08-24-rheplicant-migration.md"
CROSSCHECK = pathlib.Path(__file__).resolve().parent / "crosscheck"

#: The §四 rows that have a page today, by the module file rheplicant names
#: them by. An exact set, checked in both directions -- see the module
#: docstring.
PAGED_TODAY = {
    "identifiability.py",
    "sensitivity.py",
    "priors.py",
    "conditioning.py",
    "gls.py",
    "uncertainty.py",
    "sqrtinfo.py",
}

#: Rows whose module column names no single rheplicant file (a group, or a
#: destination-side note), excluded from the "one page per module" rule
#: because there is no module to name a page after.
NOT_A_SINGLE_MODULE = {"likelihood.py"}  # paired with noise.py in one row

#: Pages for modules the §四 ledger does not list, with the reason. Kept as
#: an explicit exception rather than by loosening the rule that a page must
#: name something the spec migrates: `sqrtinfo` belongs to the streaming
#: evidence layer, which §四 4.3 marks 不迁移 (rewritten here under iron law
#: 2), but §五 B11 requires its numerical KERNEL preserved exactly and a
#: bitwise cross-check enforces that -- a real comparison, and a record it
#: should have. An entry here must name a cross-check test that exists;
#: `test_every_paged_module_actually_has_a_cross_check_test` covers these
#: too, so the exception buys a page, never a free pass.
#:
#: `conditioning.py` is here for a different and more interesting reason,
#: found by this very check: it appears in §四 4.1 only in a DESTINATION
#: cell (the `linear.py` row's `exact/block.py linearity.py solve.py
#: conditioning.py`), never as a source row of its own -- upstream moved it
#: to `rheplicant.core.conditioning` because `radio` needed it and may not
#: import `inference`, which is also why §0.1a has to tell the reader to
#: re-read every other mention. It has its own cross-check and its own
#: rejected function (`extreme_eigenvalues`), so it earns a page; what it
#: does not have is a ledger row.
OUT_OF_LEDGER = {"sqrtinfo.py", "conditioning.py"}


def _spec_module_names() -> set[str]:
    """Every ``file.py`` named in the FIRST column of a §四 table row.

    The tables are markdown pipes whose first cell is the rheplicant side.
    Parsed rather than listed: a row added to the spec must show up here
    without this test being edited.
    """
    text = SPEC.read_text(encoding="utf-8")
    start = text.index("## 四、迁移台账")
    end = text.index("## 五、新能力")
    names: set[str] = set()
    for line in text[start:end].splitlines():
        if not line.startswith("|"):
            continue
        first = line.split("|")[1]
        names.update(re.findall(r"`?([a-z_]+\.py)`?", first))
    return names


def test_the_spec_tables_are_parseable_and_not_empty():
    """The parser above is the load-bearing part of every test below: if the
    spec's table format changes, everything here silently passes over an
    empty set. Pinned against the rows that must be there."""
    names = _spec_module_names()
    assert len(names) >= 8, names
    for expected in ("linear.py", "gls.py", "identifiability.py",
                     "sensitivity.py", "priors.py", "numpyro_bridge.py"):
        assert expected in names, (expected, sorted(names))


def test_every_page_names_a_module_the_spec_actually_migrates():
    """A page for a module the spec does not list is either a typo or a
    migration nobody recorded in the ledger."""
    spec_names = _spec_module_names()
    pages = {p.stem + ".py" for p in MIGRATION.glob("*.md") if p.stem != "README"}
    assert pages <= spec_names | OUT_OF_LEDGER, sorted(
        pages - spec_names - OUT_OF_LEDGER
    )


def test_the_set_of_paged_modules_is_exactly_what_is_recorded():
    """Both directions. A page that vanishes fails; a page that appears
    without updating PAGED_TODAY fails too -- and that second failure is the
    point at which someone should ask whether §六's gate has moved."""
    pages = {p.stem + ".py" for p in MIGRATION.glob("*.md") if p.stem != "README"}
    assert pages == PAGED_TODAY, {
        "missing": sorted(PAGED_TODAY - pages),
        "unrecorded": sorted(pages - PAGED_TODAY),
    }


def test_the_gate_on_section_six_is_not_open():
    """The claim §六 turns on, asserted rather than assumed.

    While any §四 module lacks a page, no part of §六 may start. When this
    finally fails, that is the migration reaching its next phase -- read
    §六's five steps then, and note step 1's exception list.
    """
    spec_names = _spec_module_names() - NOT_A_SINGLE_MODULE
    unpaged = spec_names - PAGED_TODAY
    assert unpaged, "every module now has a page -- §六 may begin; read its step 1"


def test_each_page_carries_the_five_headings_the_protocol_names():
    """§二's protocol is a sequence of five numbered requirements, and a page
    that records only the easy ones is the shape that decays first. Checked
    on content words rather than exact headings, so a page may phrase its
    sections in its own terms."""
    required = {
        "fixture": ("fixture",),
        "numerical agreement": ("agreement", "agree"),
        "refusal agreement": ("refus",),
        "independent oracle": ("oracle", "independent"),
        "intended differences": ("difference", "intended"),
    }
    for page in MIGRATION.glob("*.md"):
        if page.stem == "README":
            continue
        text = page.read_text(encoding="utf-8").lower()
        for label, words in required.items():
            assert any(word in text for word in words), (page.name, label)


def test_every_paged_module_actually_has_a_cross_check_test():
    """A page without a test is a claim without a guard -- the exact shape
    §二 exists to prevent. Matched on the module's stem appearing in some
    cross-check file's NAME, which is the naming convention the directory
    already follows.
    """
    files = [p.name for p in CROSSCHECK.glob("test_*.py")]
    for module in PAGED_TODAY:
        stem = module[:-3]
        # `priors.py`'s test is named for what it tests, `jeffreys`, so the
        # match is on the page's own subject rather than on the filename
        # alone -- recorded here rather than renaming a good test name.
        aliases = {
            # Named for what each test's subject IS rather than for the
            # module it came from -- recorded here rather than renaming
            # tests whose names are better than the mapping.
            "priors": ("priors", "jeffreys"),
            "gls": ("gls", "noise_logdet"),
            "uncertainty": ("uncertainty", "noise_logdet"),
            "sqrtinfo": ("sqrtinfo",),
        }.get(stem, (stem,))
        assert any(any(a in name for a in aliases) for name in files), (
            module,
            files,
        )


def test_the_readme_points_at_this_test_as_the_authority():
    """The index's own table is prose. It says so, and names this module --
    if it stops doing either, the table becomes a second copy of a fact,
    which is the failure mode this repository's own docstrings keep
    warning about.

    The path is DERIVED from this file's own location rather than spelled,
    and that is not decoration. Written as a bare basename, this assertion
    passed while the README sent the reader to
    ``tests/crosscheck/test_migration_records.py`` -- the one directory the
    docstring above explains this gate must not live in, because the autouse
    ``importorskip`` there would stand it down. A guard matching one shape
    of a claim reads every other shape as absent, and absent read as pass.
    """
    text = (MIGRATION / "README.md").read_text(encoding="utf-8")
    here = pathlib.Path(__file__).resolve()
    root = here.parents[1]
    assert here.relative_to(root).as_posix() in text, here.relative_to(root)
    assert "prose" in text.lower()
