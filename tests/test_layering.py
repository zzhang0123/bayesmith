"""The package's layering, asserted rather than described.

Several module docstrings state layering rules -- ``exact/gibbs.py`` says
``dispatch`` reads ``exact`` and not the reverse; the top-level docstring
describes a foundation-to-application order. Until now nothing checked any of
them, and one had drifted into being false: the sentence read "``exact`` never
reads ``dispatch``" while :func:`bayesmith.exact.fisher.push_forward` borrows
``prior_environment`` from ``dispatch.classify`` inside the call.

The correction was to the SENTENCE, not the code, and that is the interesting
part. The borrow cannot be hoisted: ``dispatch.classify`` reaches back into
``exact.gaussian``, which imports ``graph.evaluate`` at module scope, so moving
the borrowed function up would close a cycle rather than open one. What is
true, and what a layering can actually promise, is the MODULE-SCOPE statement.

Parsed with ``ast`` rather than by importing and inspecting: an import that
runs inside a function is invisible to a runtime check of ``sys.modules``, and
a function-scope import is precisely the thing being distinguished here.
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


def _module_scope_imports(path: pathlib.Path) -> set[str]:
    """Which bayesmith units this file imports AT MODULE SCOPE.

    Walks only the top-level body, so an import nested in a function or an
    ``if TYPE_CHECKING`` block is deliberately not counted. Relative imports
    are resolved against the file's own package first -- see
    :func:`_absolute_module` for why that is not a detail.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom):
            resolved = _absolute_module(path, node)
            if resolved:
                targets.append(resolved)
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
    """The rule ``exact/gibbs.py`` states, in the form in which it is true.

    Not "never reads": :func:`bayesmith.exact.fisher.push_forward` reads it
    inside the call, and cannot stop -- see this module's docstring. The
    promise a layering can keep is about module scope, and that is what is
    pinned here.
    """
    assert "dispatch" not in _graph()["exact"]


def test_the_function_scope_borrow_this_rule_tolerates_is_still_exactly_one():
    """The sibling: the rule above is only meaningful while the exception is
    small and named. If a second borrow appears, the docstring in
    ``exact/gibbs.py`` needs rewriting again rather than quietly widening.
    """
    borrows: list[str] = []
    for path in sorted((SRC / "exact").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
                "bayesmith.dispatch"
            ):
                borrows.append(path.name)
    # The FILE and the COUNT, not the line: a line number here would go red on
    # any edit above the import, which trains the next reader to update the
    # number rather than to ask why it moved.
    assert borrows == ["fisher.py"], borrows


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
