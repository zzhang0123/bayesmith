"""Which side each comparison still has, asserted per SYMBOL.

``tests/test_migration_records.py`` guards this directory at FILE
granularity: a module recorded as SWITCHED must have no cross-check file at
all, because its far side delegates here and the comparison would be this
package against itself. That guard has a blind spot it cannot close from
where it stands: a module that switches ONE function keeps its file, and the
file-level assertion stays green while one test inside it quietly becomes a
self-comparison.

Measured, and it happened. rheplicant ``b87e44f`` (2026-08-28) delegated
``rheplicant.inference.sqrtinfo.marginalise_arrays``'s Schur complement to
``bayesmith.marginal.sqrtinfo`` -- deliberately, with a bitwise measurement
in the commit message -- and the bitwise cross-check here
(``test_marginalise_arrays_agrees_bitwise``) kept passing for two days with
nothing left to compare. Iron law 2's second half ("守卫同批退役") had no
mechanical enforcement below file granularity. This file is that
enforcement; D90 in the migration spec (§五 B11) records both the
retirement and this guard.

Three exact tables. The first two fail in both directions; the third is what
is left after a requirement is deliberately dropped:

* ``OWN`` -- symbols this directory's comparisons treat as rheplicant's own
  arithmetic. Asserted: the symbol's def reaches NO name its module bound
  from bayesmith, transitively through same-module helpers, except the
  exact allowance listed beside it (shared threshold constants and verdict
  types -- sharing the THRESHOLDS sharpens a comparison of two arithmetics;
  sharing the arithmetic ends it). The allowance is an exact set, both ways:
  a name reached beyond it is delegation creeping in, and an allowance no
  longer reached means the comparison's semantics moved and someone should
  re-measure.

* ``SHARED_KERNEL`` -- symbols whose far side is measured and RULED to be a
  wrapper or facade over this package, where the surviving test's docstring
  explains what route-level failure it can still catch. Asserted: they
  still reach bayesmith. If rheplicant un-delegates one, this fails, and that
  docstring has gone stale -- re-read it and the ruling it cites before
  touching this table; the comparison it guards may have just become real
  again and deserve MORE tests, not fewer.

* ``DELEGATION_PERMITTED`` -- symbols this package no longer requires to be
  on any particular side, mapped to the comparison file that stands on them.
  T-004 put four rows here by taking them OUT of the two tables above, on the
  user's ruling that the Bayesian numerics belong to bayesmith: while a row
  sat in ``OWN`` rheplicant could not delegate the symbol, and while
  ``marginalise`` sat in ``SHARED_KERNEL`` rheplicant could not delete it,
  and those were the only things standing between the two packages and one
  implementation. Asserted: the subject is still a top-level def or class
  over there, OR its comparison file is gone.

  That is the floor the retirement must not go below, and it is the same
  failure D90 was written about -- a comparison whose far side has left,
  passing because nothing noticed. It is strictly weaker than an ``OWN`` row:
  ``OWN`` said "you may never delegate this"; this says "when the definition
  goes, its comparison goes in the same change". A row here is not a
  permanent home. When rheplicant deletes the symbol and this package deletes
  the comparison, the row goes too.

The two directions also guard the CHECKER itself: a walker that stopped
seeing imports would fail every ``SHARED_KERNEL`` row (empty reach where
non-empty is asserted), and one that saw imports everywhere would fail
``OWN``. Neither breakage is silent, so no separate anti-vacuity test is
needed for those two. ``DELEGATION_PERMITTED`` gets one of its own
(``test_every_permitted_row_names_a_comparison_that_exists_today``), because
a row whose file was already gone would satisfy it vacuously for ever.

What this guard structurally cannot see, said here so nobody reads more
into a green run than it holds: reachability stops at the MODULE boundary.
A rheplicant function that only composes another rheplicant module which
itself delegates can show as own here. SamplingPlan used to illustrate this
through partition.auto_blocks; its new direct certificate import now makes
it SHARED_KERNEL. ``test_dispatch.py`` compares the composition by OUTPUT.
Collapsing it whole would be a module-level switch, governed by
``test_migration_records.py``.

Everything below reads source off the installed rheplicant checkout, so
what is asserted is the sibling AS CHECKED OUT -- the same contract as
every other file in this directory (see ``conftest.py``), and the same
reason a failure after an rheplicant pull is a finding, not noise.
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pytest

pytestmark = pytest.mark.crosscheck

#: Comparison subjects whose arithmetic must be rheplicant's own, mapped to
#: the EXACT set of bayesmith-bound names each may reach. Measured
#: 2026-08-30 on rheplicant ``27e621b`` with the AST walk below.
#:
#: The ``linear`` allowances are Wave B's surviving premise made exact:
#: when the four solve names switched (`2026-08-28-wave-B-linear.md`),
#: ``check_linearity`` and ``linear_operator`` stayed upstream, sharing only
#: the dispatch constants and the ``Unresolved`` verdict type -- same
#: thresholds, two arithmetics, which is what keeps ``test_linear.py``'s six
#: surviving tests comparisons.
OWN: dict[tuple[str, str], frozenset[str]] = {
    ("rheplicant.core.conditioning", "tree_norm"): frozenset(),
    ("rheplicant.core.conditioning", "largest_eigenvalue"): frozenset(),
    ("rheplicant.core.conditioning", "extreme_eigenvalues"): frozenset(),
    ("rheplicant.inference.noise", "NoiseModelLikelihood"): frozenset(),
    ("rheplicant.inference.noise", "RadiometerNoise"): frozenset(),
    ("rheplicant.inference.noise", "HomoscedasticNoise"): frozenset(),
    ("rheplicant.inference.likelihood", "GaussianLikelihood"): frozenset(),
    ("rheplicant.inference.numpyro_bridge", "init_to_declared"): frozenset(),
    ("rheplicant.inference.parameters", "refuse_stochastic_stages"): frozenset(),
    ("rheplicant.inference.plan", "Block"): frozenset(),
}

#: Far sides measured and ruled to be wrappers or facades over this
#: package. The ruling and the re-scoped test each row keeps honest:
#:
#: * ``sqrtinfo`` rows -- D90 (migration spec §五 B11);
#:   ``test_sqrtinfo_agrees.py``'s shell-against-shell docstring.
#: * ``linear`` rows -- Wave B (`2026-08-28-wave-B-linear.md`); the module
#:   docstring of ``test_linear.py`` records the eight retirements.
#: * ``fisher_information`` -- `2026-08-27-wave-A-uncertainty-covariance.md`
#:   (mutation U5); ``test_noise_logdet.py``'s two Fisher classes, which
#:   compare two CONSTRUCTION ROUTES to one arithmetic.
#: * ``SamplingPlan`` -- T-004: convergence certification delegates to
#:   optimize.certify; test_dispatch.py compares route outputs and explicitly
#:   tests refusal of a frozen-noise fixed point. It is not an independent
#:   test of the shared certificate arithmetic.
#: * ``auto_blocks`` -- the loop is bayesmith's ``first_fit``, the rule and
#:   the refusal disposition stay upstream (partition.py's own docstring);
#:   ``test_dispatch.py`` reads the disposition.
SHARED_KERNEL: frozenset[tuple[str, str]] = frozenset(
    {
        ("rheplicant.inference.sqrtinfo", "marginalise_arrays"),
        ("rheplicant.inference.linear", "wiener_solve"),
        ("rheplicant.inference.linear", "gcr_sample"),
        ("rheplicant.inference.linear", "condition_bound"),
        ("rheplicant.inference.linear", "condition_estimate"),
        ("rheplicant.inference.uncertainty", "fisher_information"),
        ("rheplicant.inference.partition", "auto_blocks"),
        ("rheplicant.inference.plan", "SamplingPlan"),
    }
)

#: Subjects whose SIDE this package no longer constrains, mapped to the
#: cross-check file whose comparison stands on them. Each row was moved here
#: from ``OWN`` or ``SHARED_KERNEL`` by T-004; see the module docstring and
#: ``docs/ownership.md`` for the ruling.
#:
#: Removing the old rows removed a requirement, which cannot break the far
#: side -- but it would have removed the only mechanical reason a
#: self-comparison fails, which is the whole subject of this file. This is
#: what is kept instead: not "stay on your own side", but "do not leave a
#: comparison behind you".
#:
#: * ``SqrtInfo`` (was ``OWN``) -- ``test_sqrtinfo_agrees.py`` compares
#:   ``combine``, ``null`` and ``log_prob`` as two QR folds. If the class
#:   stops being defined over there, every one of those is this package
#:   against itself.
#: * ``marginalise`` (was ``SHARED_KERNEL``) -- the shell-against-shell
#:   comparison in the same file. It already delegates its kernel, ruled at
#:   D90; what the old row additionally required was that the SHELL keep
#:   existing, and that is the part being released.
#: * ``check_linearity`` and ``linear_operator`` (were ``OWN``) --
#:   ``test_linear.py``'s six surviving tests, the ones Wave B left standing
#:   precisely because these two had not switched.
#:
#: **What the retirement cost, measured rather than argued.** Five mutants in
#: this package's own arithmetic, applied to a scratch copy of ``src/`` and
#: run against ``test_sqrtinfo_agrees.py`` + ``test_linear.py`` + this file
#: (36 passed clean), beside the one-sided oracles in ``tests/marginal/``
#: (42 passed clean). Exit 1 alone counts as a kill (2026-09-20, rheplicant
#: at ``8ecc708``):
#:
#: | mutant | cross-check | one-sided |
#: |---|---|---|
#: | ``log_prob`` quadratic halved | KILLED | KILLED |
#: | ``combine`` drops its corner term | KILLED | KILLED |
#: | ``marginalise_arrays`` drops the ``2 pi`` constant | SURVIVED | KILLED |
#: | ``marginalise_arrays`` drops its corner term | SURVIVED | KILLED |
#: | ``marginalise_arrays`` pivot logs drift 1e-9 | SURVIVED | KILLED |
#:
#: The two kills are the answer to "does removing a requirement make the
#: cross-check toothless": where the two packages own two arithmetics, a real
#: numerical disagreement still turns this directory red. The three survivors
#: are all ``marginalise_arrays``, the kernel BOTH sides execute -- they
#: cannot disagree about it, so the comparison correctly reports nothing and
#: the one-sided oracles are what catch it. That is D90's documented state
#: and it predates this change: none of these three rows asserted anything
#: numerical, so the killing power here is identical before and after.
DELEGATION_PERMITTED: dict[tuple[str, str], str] = {
    ("rheplicant.inference.sqrtinfo", "SqrtInfo"): "test_sqrtinfo_agrees.py",
    ("rheplicant.inference.sqrtinfo", "marginalise"): "test_sqrtinfo_agrees.py",
    ("rheplicant.inference.linear", "check_linearity"): "test_linear.py",
    ("rheplicant.inference.linear", "linear_operator"): "test_linear.py",
}


def _module_tree(module_name: str) -> ast.Module:
    module = importlib.import_module(module_name)
    return ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))


def _bayesmith_bound(tree: ast.Module) -> set[str]:
    """Every local name the module binds to something from bayesmith --
    module scope and function scope alike, aliases included."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "bayesmith":
                bound.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            bound.update(
                (alias.asname or alias.name).split(".")[0]
                for alias in node.names
                if alias.name.split(".")[0] == "bayesmith"
            )
    return bound


def _reach(tree: ast.Module, symbol: str) -> set[str] | None:
    """The bayesmith-bound names ``symbol`` references, transitively through
    same-module top-level defs.

    Prose cannot hit: a docstring is an ``ast.Constant``, never an
    ``ast.Name``, so a comment or docstring mentioning bayesmith is
    invisible here and only code counts. ``None`` means the symbol is no
    longer a top-level def or class -- the caller fails loudly rather than
    skipping, because a subject that moved is a finding.
    """
    top = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if symbol not in top:
        return None
    bound = _bayesmith_bound(tree)
    hits: set[str] = set()
    seen: set[str] = set()
    frontier = {symbol}
    while frontier:
        name = frontier.pop()
        if name in seen or name not in top:
            continue
        seen.add(name)
        for node in ast.walk(top[name]):
            if isinstance(node, ast.Name):
                if node.id in bound:
                    hits.add(node.id)
                elif node.id in top:
                    frontier.add(node.id)
    return hits


_OWN_ROWS = sorted(OWN)
_SHARED_ROWS = sorted(SHARED_KERNEL)
_PERMITTED_ROWS = sorted(DELEGATION_PERMITTED)


def _row_id(row: tuple[str, str]) -> str:
    module_name, symbol = row
    return f"{module_name.rsplit('.', 1)[1]}.{symbol}"


@pytest.mark.parametrize(
    ("module_name", "symbol"), _OWN_ROWS, ids=[_row_id(r) for r in _OWN_ROWS]
)
def test_an_own_subject_reaches_exactly_its_allowance(module_name, symbol):
    reached = _reach(_module_tree(module_name), symbol)
    assert reached is not None, (
        f"{module_name}.{symbol} is no longer a top-level def or class over "
        "there. The subject a cross-check compares has moved, and failing at "
        "the guard is the correct loudness -- find where it went, then "
        "update this row and the test that compares it."
    )
    allowed = OWN[(module_name, symbol)]
    assert reached == allowed, {
        "subject": f"{module_name}.{symbol}",
        "reached beyond the allowance (delegation creeping in?)": sorted(
            reached - allowed
        ),
        "allowance no longer reached (comparison semantics moved?)": sorted(
            allowed - reached
        ),
        "what to do": (
            "re-measure with this file's walker; if the far side now "
            "delegates, retire or re-scope its comparison (iron law 2, D90's "
            "precedent) and move the row to SHARED_KERNEL -- do not widen "
            "the allowance to make this pass"
        ),
    }


@pytest.mark.parametrize(
    ("module_name", "symbol"), _SHARED_ROWS, ids=[_row_id(r) for r in _SHARED_ROWS]
)
def test_a_shared_kernel_subject_still_delegates(module_name, symbol):
    reached = _reach(_module_tree(module_name), symbol)
    assert reached is not None, (
        f"{module_name}.{symbol} is no longer a top-level def or class over "
        "there. The facade a route comparison stands on has moved -- find "
        "where it went before trusting any test that names it."
    )
    assert reached, (
        f"{module_name}.{symbol} no longer reaches bayesmith at all. If "
        "rheplicant un-delegated it, the route-comparison docstring this file's "
        "SHARED_KERNEL comment names for it has gone stale -- re-read it and "
        "its ruling (D90 / Wave B / U5) before touching this table. The "
        "comparison it guards may have just become real again, and a real "
        "comparison deserves more assertions, not a deleted row."
    )


@pytest.mark.parametrize(
    ("module_name", "symbol"),
    _PERMITTED_ROWS,
    ids=[_row_id(r) for r in _PERMITTED_ROWS],
)
def test_a_permitted_subject_has_not_left_a_comparison_behind(module_name, symbol):
    """Either the far side still defines it, or its comparison is gone.

    The one thing dropping the ``OWN``/``SHARED_KERNEL`` rows must not drop.
    A comparison whose subject no longer exists over there does not fail --
    ``from rheplicant.inference.sqrtinfo import SqrtInfo`` keeps resolving
    after rheplicant re-exports this package's class, and every assertion in
    the file then compares bayesmith with bayesmith and passes. That is
    exactly what happened to ``test_marginalise_arrays_agrees_bitwise`` for
    two days in 2026-08, and it is the reason this file exists.

    Note what is NOT asserted any more: nothing here requires the definition
    to be free of bayesmith arithmetic. A far side that delegates its kernel
    while keeping its own shell is a real comparison of the shells -- D90
    ruled that for ``marginalise``, and this row inherits the ruling.
    """
    reached = _reach(_module_tree(module_name), symbol)
    comparison = pathlib.Path(__file__).parent / DELEGATION_PERMITTED[
        (module_name, symbol)
    ]
    if reached is not None:
        return
    assert not comparison.exists(), (
        f"{module_name}.{symbol} is no longer a top-level def or class over "
        f"there, and {comparison.name} still stands on it. Whatever that file "
        "asserts about this subject is now this package compared with itself, "
        "and it will pass. Retire those assertions in the same change that "
        "removed the definition -- identify each subject in a one-sided home "
        "on this side (iron law 2's other branch, and the pattern "
        "test_linear.py's docstring records for its eight retirements) -- "
        "then delete this row. Do not delete the row alone; the row is not "
        "what is broken."
    )


def test_every_permitted_row_names_a_comparison_that_exists_today():
    """The anti-vacuity guard the other two tables get for free.

    ``OWN`` and ``SHARED_KERNEL`` assert opposite things, so a broken walker
    fails one of them. This table's assertion is satisfied by a missing file,
    so a row pointing at a comparison that was already deleted would pass for
    ever while claiming to protect something. Both halves are checked: the
    file exists, and it names the symbol.
    """
    here = pathlib.Path(__file__).parent
    for (module_name, symbol), filename in DELEGATION_PERMITTED.items():
        comparison = here / filename
        assert comparison.exists(), (
            f"{filename} is named by the row for {module_name}.{symbol} and is "
            "not here. Either the comparison was retired and the row should "
            "have gone with it, or the filename is wrong -- in which case the "
            "row has been protecting nothing."
        )
        assert symbol in comparison.read_text(encoding="utf-8"), (
            f"{filename} never mentions {symbol}, so it is not the file that "
            "stands on this subject. Point the row at the one that does."
        )
