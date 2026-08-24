import bayesmith


def test_every_exported_name_resolves():
    for name in bayesmith.__all__:
        assert hasattr(bayesmith, name), name


def test_importing_bayesmith_stays_light():
    """``import bayesmith`` must not pull in jax or numpyro.

    A subprocess, not an in-process ``sys.modules`` check: by the time this
    test runs, pytest's own process has almost certainly already imported
    jax/numpyro via other test modules, which would make an in-process check
    pass regardless of what ``bayesmith/__init__.py`` actually does. Mirrors
    ``test_errors_module_imports_no_heavy_dependency`` in ``test_errors.py``
    -- and pins the more general claim that one relies on: Python always
    runs a package's ``__init__.py`` before any of its submodules, so a bare
    ``import bayesmith`` is the more direct thing to check, and the one that
    was actually broken (an eager ``__init__.py`` drags jax/numpyro in even
    for ``import bayesmith.errors`` alone).
    """
    import subprocess
    import sys

    code = (
        "import bayesmith, sys; "
        "print(sorted({'jax', 'numpy', 'numpyro'} & set(sys.modules)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_the_readme_example_runs():
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist

    def model(data):
        x = bayesmith.sample("x", lambda: dist.Normal(0.0, 2.0))
        bayesmith.observe("d", lambda v: dist.Normal(v, 0.5), x, obs=data)

    graph = bayesmith.trace(model, jnp.array([1.0, 2.0]))
    assert graph.latents == ("x",)
    assert jnp.isfinite(bayesmith.log_joint(graph, {"x": jnp.array(0.0)}))
    draws = bayesmith.nuts(graph, jax.random.key(0), num_warmup=200, num_samples=200)
    assert draws["x"].shape == (200,)


def test_every_exact_name_resolves_and_is_the_same_object_as_its_module_s():
    """Lazy resolution must hand back the real function, not a shim.

    Checked by identity rather than by `hasattr`: a __getattr__ that returned
    the module, or a wrapper, would pass a name check and fail here.
    """
    import bayesmith
    from bayesmith.exact import fisher, gaussian, gls, linearity, solve

    expected = {
        "linear_operator": linearity.linear_operator,
        "check_linearity": linearity.check_linearity,
        "wiener_solve": solve.wiener_solve,
        "gcr_sample": solve.gcr_sample,
        "condition_bound": solve.condition_bound,
        "iterative_gls": gls.iterative_gls,
        "sigma_from_graph": gls.sigma_from_graph,
        "noise_std_at": gaussian.noise_std_at,
        "fisher_information": fisher.fisher_information,
        "parameter_covariance": fisher.parameter_covariance,
    }
    for name, target in expected.items():
        assert getattr(bayesmith, name) is target, name
        assert name in bayesmith.__all__


def test_the_exact_subpackage_s_own_all_reexports_the_right_object():
    """``bayesmith.exact.__all__`` is a SEPARATE contract from the top-level one.

    `bayesmith`'s own lazy attributes (`_LAZY_ATTRS`) point straight at each
    owning submodule (`bayesmith.exact.linearity`, `bayesmith.exact.solve`,
    ...) -- never at `bayesmith.exact` itself -- so
    `test_every_exact_name_resolves_and_is_the_same_object_as_its_module_s`
    above, and the top-level `bayesmith.linear_operator`/`bayesmith.
    wiener_solve`/etc. it checks, cannot see a bug in `exact/__init__.py`'s
    OWN re-export list at all: a wrong or swapped import there is invisible
    to the entire top-level API. `exact/__init__.py`'s own docstring
    promises a second, bare surface -- "the checked name is the one a bare
    `from bayesmith.exact import ...` finds" -- and this is what checks it.

    Measured directly: swapping `linear_operator`/`unchecked_operator`'s
    import sources in `exact/__init__.py` (so `bayesmith.exact.
    linear_operator` silently became the UNCHECKED primitive) left the
    entire pre-existing suite green, `bayesmith`-level tests included --
    this test is what closes that gap.
    """
    import bayesmith.exact as exact_pkg
    from bayesmith.exact import block, fisher, gaussian, gls, linearity, solve

    expected = {
        "LinearBlock": block.LinearBlock,
        "unchecked_operator": block.unchecked_operator,
        "linear_operator": linearity.linear_operator,
        "check_linearity": linearity.check_linearity,
        "gaussian_parts": gaussian.gaussian_parts,
        "check_gaussian": gaussian.check_gaussian,
        "noise_std_at": gaussian.noise_std_at,
        "wiener_solve": solve.wiener_solve,
        "gcr_sample": solve.gcr_sample,
        "condition_bound": solve.condition_bound,
        "iterative_gls": gls.iterative_gls,
        "GLSResult": gls.GLSResult,
        "sigma_from_graph": gls.sigma_from_graph,
        "check_prediction_dependence": gls.check_prediction_dependence,
        "FlatMatrix": fisher.FlatMatrix,
        "dense_operator": fisher.dense_operator,
        "fisher_information": fisher.fisher_information,
        "parameter_covariance": fisher.parameter_covariance,
    }
    # Every name this dict claims to cover really is declared, in both
    # directions -- so a name added to __all__ and forgotten here, or vice
    # versa, is itself a failure rather than a silent gap in the check.
    assert set(expected) == set(exact_pkg.__all__)
    for name, target in expected.items():
        assert getattr(exact_pkg, name) is target, name


def test_importing_bayesmith_still_does_not_import_numpyro():
    """The exact subpackage reaches numpyro, so it must stay lazy.

    Eagerly importing it here would break errors.py's stdlib-only contract,
    because Python runs a package's __init__.py before any submodule of it.
    """
    import subprocess
    import sys

    code = (
        "import bayesmith, sys;"
        "assert 'numpyro' not in sys.modules;"
        "assert 'bayesmith.exact' not in sys.modules;"
        "assert bayesmith.wiener_solve is not None;"
        "assert 'numpyro' in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_compile_is_the_function_not_the_subpackage():
    """``dispatch/``, not ``compile/`` -- and this is the test that would have
    caught the collision the first draft of the spec proposed.

    Measured directly, by building the mutant: the whole of ``dispatch/`` copied
    to ``compile/`` with ``_LAZY_ATTRS`` and ``_LAZY_SUBMODULES`` following it.
    The collision is real and it is **order-dependent**, which is worse than
    unconditional:

    * ``import bayesmith.compile.classify`` (or the ``from ... import`` form
      the spec's first draft used) before anything reads the attribute leaves
      ``bayesmith.compile`` a MODULE -- ``callable`` False. The import machinery
      does ``setattr(parent, "compile", module)`` on first load.
    * Reading ``bayesmith.compile`` FIRST leaves it a function even afterwards,
      because resolving the lazy attribute already put ``bayesmith.compile``
      into ``sys.modules`` and re-caches the function over it; the later import
      finds the parent loaded and never repeats the ``setattr``.

    So the same assertion passes or fails on the order two unrelated test files
    happen to run in -- alone versus in the suite, and under xdist on which
    worker got which file.

    The first assertion is the witness at today's layout; the three after it
    are what actually go red the moment a ``compile`` subpackage exists,
    because the poisoning import above can only name the package that exists
    NOW. A rename would take the whole suite's imports with it, and then the
    witness alone would silently stop witnessing anything.
    """
    import importlib.util
    import pathlib

    import bayesmith
    import bayesmith.dispatch.classify  # the poisoning import

    assert callable(bayesmith.compile)
    assert "compile" not in bayesmith._LAZY_SUBMODULES
    root = pathlib.Path(bayesmith.__file__).parent
    assert not (root / "compile").exists()
    assert importlib.util.find_spec("bayesmith.compile") is None
