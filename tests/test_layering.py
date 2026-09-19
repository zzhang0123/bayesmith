"""Package layering, checked at module scope and across the exact boundary.

Import-time dependencies must be acyclic. Exact arithmetic must never import
its dispatch caller, even lazily; shared prior anchors live below dispatch.
AST checks resolve absolute and relative spellings of the same dependency.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "bayesmith"

#: Subpackages and top-level modules, derived from the tree.
UNITS = sorted(
    p.name if p.is_dir() else p.stem
    for p in SRC.iterdir()
    if (p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith("_"))
    or (p.is_file() and p.suffix == ".py" and not p.name.startswith("_"))
)


def _unit_of(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _absolute_module(path: pathlib.Path, node: ast.ImportFrom) -> str:
    """The dotted name a possibly-relative ``from ... import`` actually names.

    ``from ..dispatch.task import PRODUCER`` in ``evaluation/sbc.py`` and
    ``from bayesmith.dispatch.task import PRODUCER`` in ``evaluation/heldout.py``
    are the same edge, and until 2026-09-04 this file counted only the second.
    Measured then, while merging R3: eight module-scope cross-unit imports were
    invisible to the graph, one of them the ``evaluation -> dispatch`` arrow
    that two of these tests exist to police. Nothing was wrong with the
    assertions; the graph they read was short of the edges.

    That is the failure mode this whole file is about, turned on itself -- a
    guard that cannot fail is worse than no guard, and this one could be walked
    past by a style choice, silently, with the suite green.
    """
    package = ("bayesmith",) + path.relative_to(SRC).parts[:-1]
    if not node.level:
        return node.module or ""
    kept = len(package) - (node.level - 1)
    if kept < 1:
        return ""
    tail = tuple((node.module or "").split(".")) if node.module else ()
    return ".".join(package[:kept] + tail)


def _import_from_targets(path: pathlib.Path, node: ast.ImportFrom) -> list[str]:
    """Every dotted name a ``from ... import`` actually reaches.

    Usually one: ``from bayesmith.dispatch.task import PRODUCER`` names
    ``bayesmith.dispatch.task``. But when the module resolves to the PACKAGE
    itself -- ``from bayesmith import dispatch``, or ``from .. import
    dispatch`` written inside a subpackage -- the unit being imported is the
    imported NAME, not the module, and reading only the module scores the
    arrow as nothing.

    Measured on 2026-09-04: both of those spellings resolved to the bare
    string ``bayesmith``, which the caller's ``len(parts) > 1`` filter then
    dropped, so an ``evaluation -> dispatch`` import written either way was
    invisible to every assertion in this file.
    """
    resolved = _absolute_module(path, node)
    if not resolved:
        return []
    if resolved == "bayesmith":
        return [f"bayesmith.{alias.name}" for alias in node.names]
    return [resolved]


def _is_type_checking(node: ast.expr) -> bool:
    """Whether a test is the ``TYPE_CHECKING`` guard, however it is spelled."""
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    return False


def _module_scope_statements(tree: ast.Module) -> list[ast.stmt]:
    """Every statement that RUNS when the module is imported.

    Not ``tree.body``. A module-level ``try:``/``except ImportError:`` or
    ``if sys.version_info >= ...:`` executes at import exactly as the top level
    does, and its imports are just as real -- but ``tree.body`` contains the
    ``Try`` node, not the ``ImportFrom`` inside it, so a walk of the body alone
    sees nothing.

    Measured on 2026-09-04, after the previous repair to this file: of seven
    spellings of ``bayesmith.dispatch`` written into an ``artifacts`` module,
    five created a real ``sys.modules`` edge and were scored as no edge at all.
    Two of those were ``try:`` and ``if:`` at module scope -- and
    ``try: import x except ImportError:`` is the IDIOMATIC spelling for an
    optional dependency, which is to say the shape this package's own arviz
    rule invites an author to write.

    A function, class or ``if TYPE_CHECKING:`` body is still excluded, because
    those genuinely do not run at import; that distinction is the one this file
    exists to make, and it is the only one.
    """
    statements: list[ast.stmt] = []
    stack: list[ast.stmt] = list(tree.body)
    while stack:
        node = stack.pop()
        statements.append(node)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
            continue
        for field in ("body", "orelse", "finalbody"):
            stack.extend(getattr(node, field, []) or [])
        for handler in getattr(node, "handlers", []) or []:
            stack.extend(handler.body)
    return statements


def _module_scope_imports(path: pathlib.Path) -> set[str]:
    """Which bayesmith units this file imports AT MODULE SCOPE.

    "Module scope" means "runs at import", which is what a layering claim is
    about -- see :func:`_module_scope_statements`, which is where the
    distinction is drawn and why. Relative imports are resolved against the
    file's own package first; see :func:`_absolute_module` for why that is not
    a detail.

    **Dynamic imports are out of scope and this docstring says so rather than
    leaving the promise wider than the check.** ``importlib.import_module
    ("bayesmith.dispatch")`` and ``__import__(...)`` create a real edge that
    nothing here sees. They are not idiomatic in this package -- no module in
    ``src/`` uses either -- but the guard's subject is "nobody may import
    upwards", and a reader is entitled to know which spellings it can actually
    speak for.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in _module_scope_statements(tree):
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom):
            targets.extend(_import_from_targets(path, node))
        elif isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        for name in targets:
            parts = name.split(".")
            if parts[0] == "bayesmith" and len(parts) > 1:
                found.add(parts[1])
    return found


def _graph() -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {unit: set() for unit in UNITS}
    for path in SRC.rglob("*.py"):
        unit = _unit_of(path)
        if unit not in edges:
            continue
        for target in _module_scope_imports(path):
            if target != unit and target in edges:
                edges[unit].add(target)
    return edges


def test_the_producer_this_package_stamps_on_artifacts_is_one_object():
    """One fact, one object -- checked by identity, not by spelling.

    ``ProducerRef(package="bayesmith", version=__version__)`` is who wrote an
    artifact. Three modules built it independently while R3 Wave A was written
    in five parallel branches, and two said in a comment why: importing
    :data:`bayesmith.dispatch.task.PRODUCER` would put an ``evaluation ->
    dispatch`` edge at module scope, which the layering assertions then forbade.
    R3 §0.1 authorises that edge and those assertions now name ``evaluation`` as
    the unit above ``dispatch``, so the obstacle is gone.

    **The first version of this test scanned the AST for calls named
    ``ProducerRef`` and it did not work.** Re-introducing the duplicate as
    ``from ... import ProducerRef as _PR`` followed by ``_PR(package=...)``
    left it green -- measured, that is not a worry. A test that reads the
    spelling can be walked past by a rename, which is the same shape as the
    relative-import hole below. So this one reads the OBJECT: every module that
    exports a ``PRODUCER`` must export the same one, and a second construction
    is a different object however it is spelled.

    ``CLAUDE.md`` opens on the defect this repository has spent the most time
    repairing -- six copies of one measurement, all stale on a day none of them
    was edited -- and the rule it draws is: one file, or a test; not two files
    and a hope.
    """
    import importlib

    exporters: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            names = (
                [t.id for t in node.targets if isinstance(t, ast.Name)]
                if isinstance(node, ast.Assign)
                else [node.target.id]
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                else []
            )
            if "PRODUCER" in names:
                rel = path.relative_to(SRC).with_suffix("")
                exporters.append("bayesmith." + ".".join(rel.parts))

    assert "bayesmith.dispatch.task" in exporters, exporters
    canonical = importlib.import_module("bayesmith.dispatch.task").PRODUCER
    for name in exporters:
        assert importlib.import_module(name).PRODUCER is canonical, name


def test_a_relative_import_is_the_same_edge_as_the_absolute_one():
    """The scanner every other assertion in this file reads through.

    Measured 2026-09-04, while merging R3 Wave A, on the tree as it then was:
    ``evaluation/sbc.py``'s ``from ..dispatch.task import PRODUCER`` contributed
    NOTHING to ``_graph()``, while ``evaluation/heldout.py``'s
    ``from bayesmith.dispatch.task import PRODUCER`` -- the same edge, the same
    line of the same layer -- contributed the arrow. Eight module-scope
    cross-unit imports were invisible, and one of them was the
    ``evaluation -> dispatch`` arrow that two tests below exist to police.

    Nothing in this file was wrong; the graph it read was short of the edges.
    That is the shape it warns about elsewhere, turned on itself: a guard that
    can be walked past by a style choice, silently, with the suite green. The
    branch that wrote the relative import never learned it had to update a
    layering assertion, and the branch that wrote the absolute one did.
    """
    sbc = SRC / "evaluation" / "sbc.py"
    absolute = ast.parse("from bayesmith.dispatch.task import PRODUCER").body[0]
    relative = ast.parse("from ..dispatch.task import PRODUCER").body[0]
    assert _absolute_module(sbc, absolute) == "bayesmith.dispatch.task"
    assert _absolute_module(sbc, relative) == "bayesmith.dispatch.task"

    # One dot stays inside the unit, so it is correctly NOT an edge: the
    # graph is between units, and `_graph` drops `target != unit` itself.
    sibling = ast.parse("from .checks import DRAW_FLOOR").body[0]
    assert _absolute_module(sbc, sibling) == "bayesmith.evaluation.checks"

    # And the file on disk, not just the parser: this is the assertion that
    # goes red if the resolution is ever removed again.
    assert "dispatch" in _module_scope_imports(sbc)


def test_the_module_scope_import_graph_is_acyclic():
    """A cycle here would mean import order decides behaviour."""
    edges = _graph()
    colour: dict[str, int] = dict.fromkeys(edges, 0)
    stack: list[str] = []

    def visit(unit: str) -> None:
        colour[unit] = 1
        stack.append(unit)
        for nxt in sorted(edges[unit]):
            if colour[nxt] == 1:
                raise AssertionError(
                    "module-scope import cycle: "
                    + " -> ".join([*stack[stack.index(nxt):], nxt])
                )
            if colour[nxt] == 0:
                visit(nxt)
        stack.pop()
        colour[unit] = 2

    for unit in sorted(edges):
        if colour[unit] == 0:
            visit(unit)


def test_exact_does_not_import_dispatch_at_module_scope():
    assert "dispatch" not in _graph()["exact"]


def test_exact_does_not_import_dispatch_even_inside_a_function():
    imports: list[str] = []
    for path in sorted((SRC / "exact").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                targets = _import_from_targets(path, node)
            elif isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            else:
                continue
            if any(
                name == "bayesmith.dispatch" or name.startswith("bayesmith.dispatch.")
                for name in targets
            ):
                imports.append(path.name)
    assert imports == [], imports


def test_graph_is_the_foundation_and_evaluation_is_the_top():
    """The narrative in the top-level docstring, measured.

    ``graph`` is depended on by the most units; the top of the ladder by none.

    **The top moved in R3, and the move is the point.** This assertion used to
    read ``in_degree["dispatch"] == 0``, which was true while ``dispatch`` was
    the last rung. R3 §0.1 puts ``evaluation`` above it -- the checks read a
    finished Result through ``dispatch.predictive`` and ``dispatch.task``, and
    §7.2's chain is ``model/graph -> analysis/compiler -> execution adapters ->
    evaluation -> workflow``. So the roof is ``evaluation`` now, and the one
    arrow into ``dispatch`` is the one the design draws. Naming the units
    ABOVE dispatch rather than counting them is what keeps this honest: a
    second, unplanned unit reaching down would show up as a name here rather
    than as a number that someone bumps.
    """
    edges = _graph()
    in_degree = {
        unit: sum(1 for other in edges if unit in edges[other]) for unit in edges
    }
    assert in_degree["evaluation"] == 0, "something now depends on evaluation"
    above_dispatch = sorted(unit for unit in edges if "dispatch" in edges[unit])
    assert above_dispatch == ["evaluation"], above_dispatch
    assert in_degree["graph"] >= 4, in_degree
    assert edges["graph"] <= {"errors", "distributions"}, edges["graph"]


def test_the_artifact_protocol_is_a_leaf_and_dispatch_is_what_reaches_it():
    """§0.1's ladder, in the only direction that can be checked structurally.

    ``artifacts`` imports no other unit of this package: it is data about what
    was asked, planned, produced and judged, and a protocol that reached back
    into the graph layer would put a Graph inside an artifact by the shortest
    available route. ``dispatch`` was for one release the only unit that
    bridged the two, and this assertion said so -- and named itself as the
    place where a second unit's edge would have to be argued for.

    **R3 grew the second, and the argument is that the edge runs the right
    way.** ``evaluation`` reads results and writes ``EvaluationReport``, both
    of which are artifact types, so it cannot do its job without this import;
    R3 §0.1 fixes the direction as ``evaluation -> artifacts`` and
    ``test_nothing_below_the_evaluation_layer_reads_it`` holds the other side
    of it. The direction that stays forbidden is the reverse one, and
    ``edges["artifacts"] == set()`` below is what holds it; the list after it
    is a census of who depends on the leaf, spelled out rather than counted so
    that a THIRD unit growing an edge fails with a name in it.
    """
    edges = _graph()
    assert edges["artifacts"] == set()
    assert "artifacts" in edges["dispatch"]
    reaching = sorted(unit for unit in edges if "artifacts" in edges[unit])
    assert reaching == ["dispatch", "evaluation"], reaching


def test_importing_the_artifact_protocol_pulls_in_no_numerical_stack():
    """The leaf is meant to be CHEAP as well as low: a consumer reading a
    stored artifact should not pay for jax, numpyro or equinox to do it.

    In a subprocess, because by the time this test runs the whole numerical
    stack is in this process's ``sys.modules`` several times over, and an
    in-process check would be a check of the test runner rather than of the
    package. numpy is deliberately not in the set: the codec encodes arrays
    and :class:`~bayesmith.artifacts.base.NamedArray` copies them, so numpy is
    a dependency of the protocol itself rather than of the runtime it avoids.
    """
    import subprocess
    import sys

    code = (
        "import bayesmith.artifacts as a, sys; "
        "assert a.__doc__; "
        "print(sorted({'jax', 'numpyro', 'equinox'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"



def test_nothing_below_the_evaluation_layer_reads_it():
    """R3 §0.1's direction, in the form a module-scope check can hold.

    The evaluation layer reads ``dispatch``, ``graph``, ``artifacts`` and the
    ArviZ bridge, and none of them reads it back. That is not a preference: an
    evaluation that ``dispatch`` could import would let the execution layer
    judge its own output, which is exactly what §2.4 ("Evaluation only
    evaluates Results; it does not modify the posterior and does not choose a
    new algorithm on the execution layer's behalf") exists to forbid. The
    other direction of the same rule is that the reports are DERIVED objects,
    so an artifact that could reach them would put a verdict inside the thing
    the verdict is about.

    The three named units are the ones the layer is built on top of, i.e. the
    ones with something to gain from a shortcut. ``exact``, ``marginal`` and
    the rest are covered by the acyclicity test above, which is what makes a
    back-edge a cycle rather than merely a wrong-way arrow.
    """
    edges = _graph()
    assert "evaluation" in edges, (
        "the evaluation subpackage is missing from the tree; this rule is "
        "about a layer that exists"
    )
    for unit in ("artifacts", "graph", "dispatch"):
        assert "evaluation" not in edges[unit], (
            f"{unit} imports evaluation at module scope: the layer that is "
            "judged now reaches the layer that judges it"
        )


def test_importing_the_evaluation_layer_pulls_in_no_arviz():
    """§0.9 keeps ArviZ OPTIONAL, and an optional dependency is only optional
    while nothing imports it on the way in.

    ``loo.py`` is the one module that will call ``arviz.loo``, and the whole
    of §7.3's "degrade gracefully" contract is that a clone without arviz
    installed gets an UNVERIFIABLE report rather than an ImportError. That
    contract is decided by WHERE the import sits: inside the function that
    needs it, never at module scope, and never re-exported through this
    package's ``__init__``.

    In a subprocess for the same reason as the artifact-protocol check above:
    by the time this test runs, arviz is already in this process's
    ``sys.modules`` because ``tests/bridge`` imported it, so an in-process
    assertion would be a statement about the test runner.
    """
    import subprocess
    import sys

    code = (
        "import bayesmith.evaluation as e, sys; "
        "assert e.__doc__; "
        "print(sorted({'arviz'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_a_package_level_import_of_a_unit_is_an_edge_like_any_other():
    """``from bayesmith import dispatch`` is the same arrow as
    ``from bayesmith.dispatch import task``, and until R4 this file scored the
    first as nothing at all.

    :func:`_absolute_module` resolves both ``from bayesmith import dispatch``
    and ``from .. import dispatch`` to the bare string ``bayesmith``, which
    :func:`_module_scope_imports` then dropped at its ``len(parts) > 1``
    filter. So the two spellings created NO edge and tripped nothing --
    measured on 2026-09-04, with every layering assertion in this file green.

    That is the same defect the docstring of :func:`_absolute_module` records
    being repaired for relative imports one release earlier, and it is the
    reason a guard gets a test of its own rather than a reading: the subject
    here is "nobody may import upwards", and what walks past such a guard is
    the same import written a second way.
    """
    written = {
        "from bayesmith import dispatch": "dispatch",
        "from bayesmith import dispatch as _d": "dispatch",
        "from .. import dispatch": "dispatch",
        "from bayesmith.dispatch import task": "dispatch",
        "import bayesmith.dispatch": "dispatch",
    }
    for source, unit in written.items():
        tree = ast.parse(source)
        node = tree.body[0]
        # Evaluated as if written in evaluation/sbc.py, so the relative form
        # has a package to resolve against.
        path = SRC / "evaluation" / "sbc.py"
        found: set[str] = set()
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom):
            targets.extend(_import_from_targets(path, node))
        else:
            assert isinstance(node, ast.Import)
            targets.extend(alias.name for alias in node.names)
        for name in targets:
            parts = name.split(".")
            if parts[0] == "bayesmith" and len(parts) > 1:
                found.add(parts[1])
        assert found == {unit}, f"{source!r} produced {found}, not {{{unit!r}}}"


def test_no_unit_reaches_a_unit_above_it_however_the_import_is_spelled():
    """The consequence of the test above, asserted on the real tree.

    Kept separate so that a regression in the edge builder and a regression in
    the layering itself do not report as one failure.
    """
    edges = _graph()
    assert "evaluation" not in edges["artifacts"]
    assert "evaluation" not in edges["dispatch"]
    assert "evaluation" not in edges["graph"]
    assert "dispatch" not in edges["artifacts"]
    assert "dispatch" not in edges["graph"]


def test_an_import_that_runs_at_import_time_is_an_edge_however_it_is_nested(
    tmp_path,
):
    """The five spellings an adversarial review walked the guard past.

    Each of these puts ``bayesmith.dispatch`` in ``sys.modules`` when the module
    is imported, and until R4 the graph scored all but the first two as nothing.
    ``try:``/``except ImportError:`` is the one that matters most in practice:
    it is the idiomatic spelling for an optional dependency, which is exactly
    what this package's own arviz rule asks an author to write.

    The last two rows are the boundary the docstring now states rather than
    quietly leaves: a dynamic import is a real edge that this guard does not
    see, and saying so is better than a promise wider than the check.
    """
    caught = {
        "plain": "from bayesmith import dispatch",
        "aliased": "from bayesmith import dispatch as _d",
        "relative": "from .. import dispatch",
        "deep": "from bayesmith.dispatch import task",
        "plain import": "import bayesmith.dispatch",
        "in a try": "try:\n    from bayesmith import dispatch\nexcept ImportError:\n    dispatch = None",
        "in an if": "import os\nif os.name != 'nonesuch':\n    from bayesmith import dispatch",
        "in an else": "import os\nif os.name == 'nonesuch':\n    pass\nelse:\n    from bayesmith import dispatch",
        "in a finally": "try:\n    pass\nfinally:\n    from bayesmith import dispatch",
    }
    excluded = {
        "in a function": "def f():\n    from bayesmith import dispatch",
        "in a class body": "class C:\n    from bayesmith import dispatch",
        "under TYPE_CHECKING": (
            "from typing import TYPE_CHECKING\n"
            "if TYPE_CHECKING:\n    from bayesmith import dispatch"
        ),
        "dynamic": "import importlib\nimportlib.import_module('bayesmith.dispatch')",
    }
    # Written into the real tree's shape so the relative form has a package to
    # resolve against; the file itself is a temporary copy, never imported.
    target = SRC / "evaluation" / "_layering_probe.py"
    for label, source in caught.items():
        probe = tmp_path / "probe.py"
        probe.write_text(source, encoding="utf-8")
        found = _module_scope_imports_from(probe, target)
        assert found == {"dispatch"}, f"{label}: got {found}"
    for label, source in excluded.items():
        probe = tmp_path / "probe.py"
        probe.write_text(source, encoding="utf-8")
        found = _module_scope_imports_from(probe, target)
        assert found == set(), f"{label}: got {found}, expected no edge"


def _module_scope_imports_from(source_path, as_if_at):
    """:func:`_module_scope_imports` on one file, resolved as another path.

    The relative spelling needs a package to resolve against, and a temporary
    file has none. Splitting the read from the resolution is the whole of it.
    """
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in _module_scope_statements(tree):
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom):
            targets.extend(_import_from_targets(as_if_at, node))
        elif isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        for name in targets:
            parts = name.split(".")
            if parts[0] == "bayesmith" and len(parts) > 1:
                found.add(parts[1])
    return found
