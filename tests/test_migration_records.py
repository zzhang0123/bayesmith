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
#: Modules whose rheplicant side now DELEGATES to this package, so their
#: cross-check has retired -- iron law 2 of the one-implementation plan:
#: "切换即删除,守卫同批退役". A cross-check compares two implementations; once
#: there is one, the file compares this package with itself and is a test that
#: cannot fail, which is the single defect this repository's working notes name
#: most often.
#:
#: Read in BOTH directions by
#: ``test_every_paged_module_actually_has_a_cross_check_test``: a module in
#: here that still HAS a cross-check fails (the retirement was forgotten), and
#: one not in here that has LOST its cross-check fails (the retirement was not
#: recorded). Neither mistake is silent.
#:
#: Each entry must name where the retirement is recorded, and what happened to
#: the independent-oracle assertions that file carried -- iron law 2's second
#: half says they are re-homed here or an existing equivalent is identified.
SWITCHED = {
    # 2026-08-27, Wave A. `docs/superpowers/specs/2026-08-27-wave-A-identifiability.md`.
    # All four assertions in the retired file referenced rheplicant, so all
    # four retired with it. Its one INDEPENDENT oracle -- the four-row table
    # (free/basis x tone on/off, with n_par/rank/nullity) -- already has a home
    # on this side: `tests/diagnose/test_identifiability.py::
    # TestTheMotivatingCase::test_the_four_row_table`. Identified rather than
    # re-homed, which is the clause's other branch.
    "identifiability.py",
    # 2026-08-27, Wave A. `docs/superpowers/specs/2026-08-27-wave-A-sensitivity.md`.
    # Its cross-check compared rheplicant's mode displacement against this
    # package's; rheplicant's now IS this package's. Its independent oracles --
    # the closed form against a finite-difference refit, and the tour table --
    # are already pinned here by `tests/diagnose/test_prior_sensitivity.py`.
    "sensitivity.py",
    # 2026-08-27, Wave A. `docs/superpowers/specs/2026-08-27-wave-A-priors.md`.
    # `JeffreysPrior`'s arithmetic in rheplicant now delegates here, so the
    # cross-check's live comparison had become this package against itself.
    # All three of its subjects already have independent homes on this side and
    # are identified rather than re-homed: the flat constant against a
    # numpy-only closed form (`test_the_flat_constant_equals_a_numpy_closed_
    # form_no_autodiff_touched`), the gradients under both noise models
    # (`test_the_noise_model_chooses_the_priors_shape` and
    # `test_the_radiometer_jeffreys_prior_has_a_zero_gradient`), and the
    # singular block's floor (`test_the_eigh_route_floors_the_singular_block_
    # to_effectively_zero`) -- all in `tests/diagnose/test_jeffreys.py`.
    "priors.py",
}

PAGED_TODAY = {
    "linear.py",
    "noise.py",
    "parameters.py",
    "plan.py",
    "numpyro_bridge.py",
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
NOT_A_SINGLE_MODULE = {
    "likelihood.py",  # paired with noise.py in one §四 4.1 row
    # Paired with `plan.py` in one §四 4.2 row, whose required agreement is a
    # single comparison ("同 partition 同 toy 模型下 plan.estimate 逐值一致").
    # `plan.md` records it. A second page named for the other half of one
    # comparison would be a second copy of one fact, which is the shape this
    # repository's docstrings keep warning about -- so the pairing is declared
    # here rather than paid for with a duplicate page.
    "engines.py",
}

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


def test_the_gate_on_section_six_is_open_and_every_module_has_a_page():
    """The claim §六 turns on -- and it turned over on 2026-08-25.

    This assertion used to read the other way: *"while any §四 module lacks
    a page, no part of §六 may start"*, with a note saying that when it
    finally failed, that was the migration reaching its next phase. It
    failed on 2026-08-25, when `numpyro_bridge.md` landed and the last
    unpaged module got its record.

    Turned around rather than deleted, because the guard is still needed
    and it is the SAME guard: every §四 module must have a page. Before, no
    page could be missing without §六 being blocked; now, no page may go
    missing at all, because §六's steps are being taken against them. A
    page that disappears while `inference/` is being emptied is the one
    failure this file exists to prevent.

    §六 step 1 still governs what may move: nothing in
    ``src/rheplicant/inference/`` except the two exceptions already in
    e-RHINO's Track A Batch 1.
    """
    # The DIRECTORY, not `PAGED_TODAY`. Measured: with the gate reading the
    # recorded set, deleting `numpyro_bridge.md` left this green and only
    # the bookkeeping test fired. A gate on §六 that cannot see a page
    # disappear is guarding a variable, not the repository.
    pages = {p.stem + ".py" for p in MIGRATION.glob("*.md") if p.stem != "README"}
    spec_names = _spec_module_names() - NOT_A_SINGLE_MODULE
    unpaged = spec_names - pages
    assert not unpaged, {
        "modules with no page on disk": sorted(unpaged),
        "note": "§六 is under way; a page may not go missing now",
    }


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
    unknown = SWITCHED - PAGED_TODAY
    assert not unknown, ("SWITCHED names a module with no page", sorted(unknown))
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
            # `plan.py`'s comparison is of the DISPATCH layer, and the test
            # is named for that rather than for the upstream file it came
            # from -- same principle as `priors` -> `jeffreys`.
            "plan": ("plan", "dispatch"),
            # Named for the layer it tests, as `plan` is.
            "numpyro_bridge": ("numpyro_bridge", "bridge"),
        }.get(stem, (stem,))
        found = any(any(a in name for a in aliases) for name in files)
        if module in SWITCHED:
            # The other direction, and it is the one that matters now: a
            # switched module that still has a cross-check is comparing this
            # package with itself, and would go green forever.
            assert not found, (
                (
                    f"{module} is recorded as SWITCHED, so its cross-check "
                    "should have retired with it (iron law 2). A cross-check "
                    "against a facade that delegates here compares this "
                    "package with itself."
                ),
                files,
            )
            continue
        assert found, (module, files)


#: How the README spells small integers. The count in its prose is checked
#: against the derived one, so the word has to be readable as a number.
_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def test_the_readme_agrees_with_the_derived_state_of_the_gate():
    """The one claim in the index that is not a table, derived not trusted.

    It read "five §四 rows" against a real SIX for a whole session -- prose
    counting the bullets under it, with nothing to notice when a row was
    added. Then the list emptied, and a test that only knew how to parse a
    count had nothing to check.

    So it checks both regimes, and can fail in either: while modules are
    unpaged the README must state their number, and once none are it must
    say the gate is open. A page going missing after §六 has started puts
    the README back in the first regime and this back to counting.
    """
    text = (MIGRATION / "README.md").read_text(encoding="utf-8")
    derived = _spec_module_names() - NOT_A_SINGLE_MODULE - PAGED_TODAY
    if not derived:
        assert "**Open, since" in text, (
            "every module has a page, but the README does not say the gate "
            "is open"
        )
        assert "every §四 module has\none" in text or "every §四 module has one" in text
        return
    match = re.search(r"\*\*(\w+) §四 modules? h(?:as|ave) neither", text)
    assert match, ("modules are unpaged and the README does not say so", sorted(derived))
    stated = _NUMBER_WORDS.get(match.group(1).lower())
    assert stated is not None, match.group(1)
    assert stated == len(derived), {"stated": stated, "derived": sorted(derived)}


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
