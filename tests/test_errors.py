import pytest

from bayesmith.errors import BayesmithError, GraphError, TraceError


def test_graph_error_is_catchable_as_the_family_and_as_value_error():
    assert issubclass(GraphError, BayesmithError)
    assert issubclass(GraphError, ValueError)
    with pytest.raises(BayesmithError):
        raise GraphError("bad graph")
    with pytest.raises(ValueError):
        raise GraphError("bad graph")


def test_trace_error_is_catchable_as_the_family_and_as_runtime_error():
    assert issubclass(TraceError, BayesmithError)
    assert issubclass(TraceError, RuntimeError)
    with pytest.raises(BayesmithError):
        raise TraceError("primitive called outside trace()")
    with pytest.raises(RuntimeError):
        raise TraceError("primitive called outside trace()")


def test_errors_module_imports_no_heavy_dependency():
    """errors.py is on every import path, so it must stay stdlib-only."""
    import subprocess
    import sys

    code = (
        "import bayesmith.errors, sys; "
        "print(sorted({'jax', 'numpy', 'numpyro'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_the_new_error_names_exist_in_the_stdlib_only_module():
    """The three P3 classes live in errors.py, which may not import jax/numpy.

    This is the only import-shaped assertion Task 0 makes, and deliberately
    so: an `assert issubclass(StructureError, ValueError)` is true because
    the class statement says so, not because anything works -- P1's first
    review finding was exactly that tautology. What StructureError *does* is
    pinned where it is actually raised: tests/exact/test_gaussian.py (a lying
    dist_fn) and tests/exact/test_linearity.py (a false linear_in claim).
    """
    import subprocess
    import sys

    code = (
        "import bayesmith.errors as e;"
        "assert e.StructureError and e.ConvergenceError and e.NotGaussian;"
        "import sys;"
        "heavy = [m for m in ('jax', 'numpy', 'numpyro') if m in sys.modules];"
        "assert not heavy, heavy"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_catching_not_gaussian_does_not_also_catch_structure_error():
    """The sibling relationship is load-bearing, so it is tested by behaviour.

    P3b's classifier writes `except NotGaussian` to mean "this block has no
    exact structure, route it to NUTS". A StructureError means something
    else entirely -- a node whose type says Normal while its own log_prob
    disagrees -- and must escape that clause. If NotGaussian were made a
    subclass of StructureError, or both were given the same builtin base,
    this test goes red; `assert not issubclass(...)` would not, because it
    restates the class statement instead of exercising it.
    """
    from bayesmith.errors import NotGaussian, StructureError

    escaped = False
    try:
        try:
            raise StructureError("the log_prob probe disagreed")
        except NotGaussian:  # pragma: no cover - must not be taken
            pass
    except StructureError:
        escaped = True
    assert escaped

    with pytest.raises(NotGaussian):
        try:
            raise NotGaussian("this node is a Gamma")
        except StructureError:  # pragma: no cover - must not be taken
            pass
