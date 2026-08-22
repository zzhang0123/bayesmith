import pytest

from bayesmith.errors import BayesmithError, GraphError, TraceError


def test_graph_error_is_catchable_as_the_family_and_as_value_error():
    assert issubclass(GraphError, BayesmithError)
    assert issubclass(GraphError, ValueError)
    with pytest.raises(BayesmithError):
        raise GraphError("bad graph")


def test_trace_error_is_catchable_as_the_family_and_as_runtime_error():
    assert issubclass(TraceError, BayesmithError)
    assert issubclass(TraceError, RuntimeError)


def test_errors_module_imports_no_heavy_dependency():
    """errors.py is on every import path, so it must stay stdlib-only."""
    import subprocess
    import sys

    code = (
        "import bayesmith.errors, sys; "
        "print(sorted({'jax', 'numpy', 'numpyro'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
