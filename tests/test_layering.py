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


def _module_scope_imports(path: pathlib.Path) -> set[str]:
    """Which bayesmith units this file imports AT MODULE SCOPE.

    Walks only the top-level body, so an import nested in a function or an
    ``if TYPE_CHECKING`` block is deliberately not counted.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        targets: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module:
            targets.append(node.module)
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


def test_graph_is_the_foundation_and_dispatch_is_the_top():
    """The narrative in the top-level docstring, measured.

    ``graph`` is depended on by the most units; ``dispatch`` by none. If that
    ever inverts, the package's own description of itself has stopped being
    true.
    """
    edges = _graph()
    in_degree = {
        unit: sum(1 for other in edges if unit in edges[other]) for unit in edges
    }
    assert in_degree["dispatch"] == 0, "something now depends on dispatch"
    assert in_degree["graph"] >= 4, in_degree
    assert edges["graph"] <= {"errors", "distributions"}, edges["graph"]
