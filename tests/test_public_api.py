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
