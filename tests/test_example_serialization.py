"""The serial-example list must name every module that launches an example.

conftest.SERIAL_EXAMPLE_MODULES exists because four concurrent example
children do not fit in the runner (see that module's docstring for the
measured sizes). A list is only as good as the thing that checks it: add a
module that spawns an example, forget this list, and the nightly starts dying
again nine nights before anyone reads the annotation.

So the list is asserted against the PROPERTY it stands for -- a test module
that hands `sys.executable` a path or a `-m` target under `examples/` -- read
out of the AST rather than out of anybody's memory. A module renamed, copied
or added is caught; so is one removed, because the comparison is an equality
and not a subset.
"""

from __future__ import annotations

import ast
import collections
import importlib.util
import pathlib
import re
import subprocess
import sys
import types

TESTS = pathlib.Path(__file__).resolve().parent


def _conftest():
    """Load the sibling conftest by PATH, not by name.

    `import conftest` is not available here: pytest loads conftest.py as a
    plugin, and whether the name is also importable depends on rootdir and
    import mode. Measured -- it was not, and the first version of this file
    errored on collection instead of checking anything, which is the failure
    this whole file is about arriving in the file itself.
    """
    spec = importlib.util.spec_from_file_location(
        "_bayesmith_tests_conftest", TESTS / "conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

SPAWNERS = {"run", "check_call", "check_output", "Popen", "call"}


def _is_subprocess_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in SPAWNERS
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "subprocess"
    )


def _runs_python(call: ast.Call) -> bool:
    """True when the interpreter being launched is this one."""
    return any(
        isinstance(n, ast.Attribute)
        and n.attr == "executable"
        and isinstance(n.value, ast.Name)
        and n.value.id == "sys"
        for n in ast.walk(call)
    )


def _names_an_example(call: ast.Call) -> bool:
    """True when the thing being run lives under `examples/`.

    Both spellings count: the literal module path (`-m examples.inference`) and
    a path built from a module-level constant (`EXAMPLES / script`). Reading
    only the first is how this check would be walked past by a refactor.
    """
    for n in ast.walk(call):
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and "examples" in n.value.lower()
        ):
            return True
        if isinstance(n, ast.Name) and "examples" in n.id.lower():
            return True
    return False


def _modules_that_launch_examples() -> set[str]:
    found = set()
    for path in sorted(TESTS.rglob("test_*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if _is_subprocess_call(node) and _runs_python(node) and _names_an_example(node):
                found.add(path.stem)
                break
    return found


def test_every_example_launcher_is_serialised():
    assert _modules_that_launch_examples() == set(_conftest().SERIAL_EXAMPLE_MODULES)


class _Item:
    """The two attributes pytest_collection_modifyitems actually reads."""

    def __init__(self, module_name):
        # SimpleNamespace, not `type(...)`: a class built with `__name__` in its
        # namespace still reports the class's own name, so the stub silently
        # described a module called "M" and the hook correctly ignored it.
        self.module = types.SimpleNamespace(__name__=module_name)
        self.markers = []

    def add_marker(self, marker):
        self.markers.append(marker)


def test_the_hook_puts_those_modules_in_one_group_and_others_in_none():
    """The list is inert unless the hook still reads it.

    Asserting the list alone would stay green through a renamed hook, a
    mistyped marker or a group name that differs per module -- the list would
    be right and nothing would act on it. So the CONSEQUENCE is asserted: a
    serial module's item comes out carrying one group, every serial module
    carries the SAME group, and an unrelated module comes out unmarked.
    """
    conftest = _conftest()
    serial = [_Item(name) for name in sorted(conftest.SERIAL_EXAMPLE_MODULES)]
    other = _Item("test_layering")
    conftest.pytest_collection_modifyitems([*serial, other])

    assert other.markers == []
    groups = set()
    for item in serial:
        assert len(item.markers) == 1, item.module.__name__
        marker = item.markers[0]
        assert marker.name == "xdist_group"
        groups.add(marker.args[0] if marker.args else marker.kwargs["name"])
    assert len(groups) == 1, f"serial modules landed in {groups}"


def test_the_detector_can_actually_fail():
    """A guard that cannot fail is worse than no guard.

    The detector must require all three parts, so the check is shown refusing a
    subprocess that is Python but not an example, and one that is an example
    but not Python (the gallery test really does spawn `node`).
    """
    python_not_example = ast.parse(
        "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pytest'])"
    ).body[1].value
    example_not_python = ast.parse(
        "import subprocess\nsubprocess.run(['node', 'examples/inference/gallery.js'])"
    ).body[1].value
    assert _runs_python(python_not_example) and not _names_an_example(python_not_example)
    assert _names_an_example(example_not_python) and not _runs_python(example_not_python)


def test_the_grouping_survives_a_real_xdist_run(tmp_path):
    """The marker must reach the SCHEDULER, which is a separate claim.

    Everything above this test passes with `tryfirst` deleted: the list is
    still right, the hook still runs, the marker is still attached. What breaks
    is invisible from inside the process -- xdist reads `xdist_group` in a hook
    of its own and rewrites the nodeid the controller schedules on, so a marker
    attached after that hook is simply never seen. Measured 2026-09-20 on the
    real modules: without `tryfirst`, `--dist loadgroup -n 2` still split them
    gw0:5 / gw1:20, with the suite green and nothing to read.

    So this one runs the real conftest under a real two-worker loadgroup
    session and asserts the consequence: the grouped module lands on ONE
    worker. It is the only test here that would fail if `tryfirst` were lost.
    """
    (tmp_path / "conftest.py").write_text((TESTS / "conftest.py").read_text())
    # `test_examples` is in the list; the name is what the hook matches on.
    (tmp_path / "test_examples.py").write_text(
        "".join(f"def test_g{i}(): pass\n" for i in range(8))
    )
    (tmp_path / "test_unrelated.py").write_text(
        "".join(f"def test_u{i}(): pass\n" for i in range(8))
    )
    done = subprocess.run(
        [sys.executable, "-m", "pytest", str(tmp_path), "-n", "2",
         "--dist", "loadgroup", "-vv", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=tmp_path, timeout=300, check=False,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
    placed = collections.defaultdict(set)
    for worker, module in re.findall(
        r"^\[(gw\d)\].*?(test_\w+)\.py::", done.stdout, re.MULTILINE
    ):
        placed[module].add(worker)
    assert placed, done.stdout[-3000:]
    assert placed["test_examples"] == {"gw0"} or placed["test_examples"] == {"gw1"}, (
        "the grouped module was split across workers, so the xdist_group marker "
        f"never reached the scheduler: {dict(placed)}"
    )
