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
    """errors.py is on every import path, so it must stay stdlib-only.

    The name check for P3's three classes rides along in this same
    subprocess rather than getting one of its own: importing the module is
    what proves both, so a second spawn would only re-prove the first half.
    """
    import subprocess
    import sys

    code = (
        "import bayesmith.errors as e, sys; "
        "assert e.StructureError and e.ConvergenceError and e.NotGaussian; "
        "print(sorted({'jax', 'numpy', 'numpyro'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_catching_not_gaussian_does_not_also_catch_structure_error():
    """The sibling relationship is load-bearing, so it is tested by behaviour.

    P3b's classifier writes `except NotGaussian` to mean "this block has no
    exact structure, route it to NUTS". A StructureError means something else
    entirely -- a node whose type says Normal while its own log_prob
    disagrees -- and must escape that clause.

    The two halves cover the two directions a hierarchy could collapse in,
    and each catches exactly one: half (a) goes red if StructureError is made
    a subclass of NotGaussian, half (b) if NotGaussian is made a subclass of
    StructureError. Giving the two the same builtin base changes nothing here
    and this test stays green -- correctly so, because `except` matches on
    the MRO and not on a shared ancestor.

    `assert not issubclass(...)` would restate the class statement instead of
    exercising it, which is the tautology P1's first review finding was about.
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
