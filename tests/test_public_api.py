from typing import ClassVar

import pytest

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
    from bayesmith.exact import fisher, gaussian, gls, linearity, loglinear, solve

    expected = {
        "linear_operator": linearity.linear_operator,
        "check_linearity": linearity.check_linearity,
        "wiener_solve": solve.wiener_solve,
        "gcr_sample": solve.gcr_sample,
        "condition_bound": solve.condition_bound,
        "iterative_gls": gls.iterative_gls,
        "sigma_from_graph": gls.sigma_from_graph,
        "noise_std_at": gaussian.noise_std_at,
        "precision_at": gaussian.precision_at,
        "fisher_information": fisher.fisher_information,
        "parameter_covariance": fisher.parameter_covariance,
        "propagate_covariance": fisher.propagate_covariance,
        "push_forward": fisher.push_forward,
        # The log-space family: the four callables are top-level names; the
        # two constants and LogSpace live one level down, on bayesmith.exact,
        # and the subpackage guard below is what pins those.
        "check_log_linearity": loglinear.check_log_linearity,
        "log_linear_operator": loglinear.log_linear_operator,
        "log_space": loglinear.log_space,
        "multiplicative_log_data": loglinear.multiplicative_log_data,
    }
    for name, target in expected.items():
        assert getattr(bayesmith, name) is target, name
        assert name in bayesmith.__all__


def test_the_bridge_exports_are_the_objects_their_own_module_defines():
    """The identity pin, for the three names ``bridge/`` owns.

    Same argument as the dispatch test below, and found the same way: adding
    ``init_to_declared`` to ``_LAZY_ATTRS`` and to ``__all__`` left every guard
    in this file green, because ``test_every_exported_name_resolves`` only asks
    whether the name resolves to SOMETHING. Swapping two bridge entries would
    have gone unnoticed in exactly the way the dispatch docstring describes.
    """
    import bayesmith
    from bayesmith.bridge import numpyro_bridge

    for name in ("nuts", "predict", "init_to_declared"):
        assert getattr(bayesmith, name) is getattr(numpyro_bridge, name), name
        assert name in bayesmith.__all__, name


def test_the_dispatch_exports_are_the_objects_their_own_module_defines():
    """The same identity pin as above, for the two names ``dispatch/`` owns.

    ``Posterior`` and ``Estimate`` are lazy attributes like every name in the
    test above, but they are not ``exact`` names, so that test's map does not
    and should not reach them -- and until this one existed nothing did.
    Measured directly: swapping the two entries in ``_LAZY_ATTRS`` so
    ``bayesmith.Posterior`` resolves to ``execute.Estimate`` and vice versa
    left all seven tests in this file green, ``test_every_exported_name_
    resolves`` included, because both names still resolve to *something*.

    Checked against ``bayesmith.dispatch.execute`` rather than
    ``bayesmith.dispatch.plan``: ``plan`` re-exports both, so a map pointed
    there would be satisfied by the re-export and blind to which object it
    re-exports. ``execute`` is the owning module -- what ``_LAZY_ATTRS``'
    own comment says every entry points at.
    """
    import bayesmith
    from bayesmith.dispatch import execute

    expected = {
        "Posterior": execute.Posterior,
        "Estimate": execute.Estimate,
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
    from bayesmith.exact import (
        block,
        fisher,
        gaussian,
        gls,
        linearity,
        loglinear,
        solve,
    )

    expected = {
        # The affinity check's own vocabulary, advertised when the sibling
        # package adopted these criteria (D16, 2026-08-27): the CONSTANTS and
        # the Unresolved TYPE are imported there rather than respelled, so
        # both sides read one statement of each number and `isinstance` means
        # the same thing across the seam.
        "DEFAULT_SCALES": linearity.DEFAULT_SCALES,
        "DEFAULT_AT_POINTS": linearity.DEFAULT_AT_POINTS,
        "RELATIVE_FLOOR_FACTOR": linearity.RELATIVE_FLOOR_FACTOR,
        "WEIGHTED_FLOOR_FACTOR": linearity.WEIGHTED_FLOOR_FACTOR,
        "WEIGHTED_RTOL": linearity.WEIGHTED_RTOL,
        "Unresolved": linearity.Unresolved,
        "LinearBlock": block.LinearBlock,
        "unchecked_operator": block.unchecked_operator,
        "linear_operator": linearity.linear_operator,
        "check_linearity": linearity.check_linearity,
        "gaussian_parts": gaussian.gaussian_parts,
        "check_gaussian": gaussian.check_gaussian,
        "noise_std_at": gaussian.noise_std_at,
        "precision_parts": gaussian.precision_parts,
        "precision_at": gaussian.precision_at,
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
        "propagate_covariance": fisher.propagate_covariance,
        "push_forward": fisher.push_forward,
        "FIRST_ORDER_MAX_FRACTIONAL": loglinear.FIRST_ORDER_MAX_FRACTIONAL,
        "LOG_DEFAULT_SCALES": loglinear.LOG_DEFAULT_SCALES,
        "LogSpace": loglinear.LogSpace,
        "check_log_linearity": loglinear.check_log_linearity,
        "log_linear_operator": loglinear.log_linear_operator,
        "log_space": loglinear.log_space,
        "multiplicative_log_data": loglinear.multiplicative_log_data,
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


def test_the_evidence_subpackage_is_reachable_from_the_package_root():
    """``bayesmith.marginal`` must resolve, like every other subpackage.

    It did not, for the whole of B11: ``evidence`` was absent from
    ``_LAZY_SUBMODULES``, so ``import bayesmith; bayesmith.marginal`` raised
    ``AttributeError`` and only an explicit ``import bayesmith.marginal``
    reached the layer. The handover recorded the gap as "nothing in
    ``dispatch/`` calls it", which is a design gap; this is the narrower
    reachability one underneath it, and no amount of dispatcher work would
    have fixed it.

    Checked by identity against the real module, not by ``hasattr``: a
    ``__getattr__`` that returned some other module would pass a name check.
    """
    import importlib

    import bayesmith

    assert "evidence" in bayesmith._LAZY_SUBMODULES
    assert bayesmith.marginal is importlib.import_module("bayesmith.marginal")
    assert bayesmith.marginal.compress_campaign is not None


def test_reaching_the_evidence_layer_is_what_imports_jax_not_importing_bayesmith():
    """``evidence`` must be LAZY, not merely present.

    ``evidence/sqrtinfo.py`` imports jax at module scope, so listing it
    eagerly in ``__init__.py`` would break the stdlib-only contract
    ``test_importing_bayesmith_stays_light`` pins -- and it would break it
    for every caller, including one that never touches the evidence layer.
    Same subprocess shape as ``test_importing_bayesmith_still_does_not_
    import_numpyro``, for the same reason: this process has already imported
    jax by now.
    """
    import subprocess
    import sys

    code = (
        "import bayesmith, sys;"
        "assert 'jax' not in sys.modules;"
        "assert 'bayesmith.marginal' not in sys.modules;"
        "assert bayesmith.marginal.compress_campaign is not None;"
        "assert 'jax' in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_the_diagnose_exports_are_the_objects_their_own_modules_define():
    """Identity pins for the three names ``diagnose/`` owns.

    Same rationale as the dispatch pin above: both halves of a swapped
    ``_LAZY_ATTRS`` entry still *resolve*, so only identity can tell the
    right object from a wrong one that answers to the same name.
    """
    import importlib

    import bayesmith

    # importlib rather than `from bayesmith.diagnose import identifiability`:
    # the subpackage re-exports a FUNCTION under that name, so the from-import
    # returns the function and shadows the module this pin needs.
    identifiability_module = importlib.import_module(
        "bayesmith.diagnose.identifiability"
    )
    priors = importlib.import_module("bayesmith.diagnose.priors")
    sensitivity = importlib.import_module("bayesmith.diagnose.sensitivity")

    expected = {
        "identifiability": identifiability_module.identifiability,
        "prior_sensitivity": sensitivity.prior_sensitivity,
        "JeffreysPrior": priors.JeffreysPrior,
    }
    for name, target in expected.items():
        assert getattr(bayesmith, name) is target, name
        assert name in bayesmith.__all__


def test_the_diagnose_subpackage_is_reachable_from_the_package_root():
    """``bayesmith.diagnose`` must resolve, like every other subpackage.

    The evidence layer shipped complete and unreachable once -- absent from
    ``_LAZY_SUBMODULES``, so ``bayesmith.marginal`` raised AttributeError
    for the whole of B11 -- and this package must not repeat that shape.
    Checked by identity against the real module, not by ``hasattr``.
    """
    import importlib

    import bayesmith

    assert "diagnose" in bayesmith._LAZY_SUBMODULES
    assert bayesmith.diagnose is importlib.import_module("bayesmith.diagnose")
    assert bayesmith.diagnose.identifiability is not None
    assert bayesmith.diagnose.prior_sensitivity is not None
    assert bayesmith.diagnose.JeffreysPrior is not None


def test_reaching_the_diagnose_layer_is_what_imports_jax_not_importing_bayesmith():
    """``diagnose`` must be LAZY, not merely present.

    ``diagnose/priors.py`` imports numpyro.distributions at module scope and
    everything under ``diagnose/`` imports jax, so listing the subpackage
    eagerly would break the stdlib-only contract
    ``test_importing_bayesmith_stays_light`` pins. Same subprocess shape as
    the evidence pin above, for the same reason: this process has already
    imported jax by now.
    """
    import subprocess
    import sys

    code = (
        "import bayesmith, sys;"
        "assert 'jax' not in sys.modules;"
        "assert 'bayesmith.diagnose' not in sys.modules;"
        "assert bayesmith.diagnose.identifiability is not None;"
        "assert 'jax' in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_every_submodule_is_reachable_after_a_bare_import():
    """``import bayesmith`` then ``bayesmith.<anything>`` -- DERIVED, not listed.

    The lazy attribute machinery makes a submodule reachable only if its name
    appears in ``_LAZY_SUBMODULES``. Three names were missing at once
    (``optimize``, ``amortize``, ``distributions``) and the suite could not see
    it, because the tests around this one pin the table NAME BY NAME:
    ``assert "evidence" in bayesmith._LAZY_SUBMODULES``. Every such assertion
    is satisfied by a table containing exactly the names someone already
    thought of, so each fix added one line and covered nothing new. Three
    separate guards, and the hole stayed open.

    So this derives the expected set from the filesystem and asserts BOTH
    directions. A module added to the package is covered before anyone
    remembers to be; and a name left in the table after its module is deleted
    is caught too, which a one-way check would not do.

    The failure mode it closes is worse than a plain absence, because it is
    ORDER-DEPENDENT: ``bayesmith.fit`` resolves through ``_LAZY_ATTRS`` and
    imports ``bayesmith.optimize`` as a side effect, so ``bayesmith.optimize``
    raised ``AttributeError`` before that line and succeeded after it.
    """
    import importlib
    import pkgutil

    import bayesmith

    on_disk = {
        name
        for _, name, _ in pkgutil.iter_modules(bayesmith.__path__)
        if not name.startswith("_")
    }
    listed = set(bayesmith._LAZY_SUBMODULES)

    assert on_disk == listed, (
        "the lazy-submodule table and the package's own contents disagree: "
        f"on disk but not listed {sorted(on_disk - listed)}; "
        f"listed but not on disk {sorted(listed - on_disk)}"
    )

    # ... and listing a name is not the same as it resolving. Checked by
    # identity against the real module, the way the older guards do it: a
    # `__getattr__` returning some other object would pass a `hasattr`.
    for name in sorted(on_disk):
        assert getattr(bayesmith, name) is importlib.import_module(
            f"bayesmith.{name}"
        ), name


def test_a_submodule_resolves_without_touching_anything_else_first():
    """The order-dependence itself, pinned.

    In a FRESH interpreter, with nothing else touched: the attribute must be
    there. Run in a subprocess because the check is about import state, and
    this suite has by then imported most of the package.
    """
    import subprocess
    import sys

    for name in ("optimize", "amortize", "distributions"):
        proc = subprocess.run(
            [sys.executable, "-c", f"import bayesmith; bayesmith.{name}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (name, proc.stderr[-800:])


class TestTheDeprecatedEvidencePathStillResolves:
    """`bayesmith.evidence` was the published path through 0.4.0.

    It is kept until 1.0, and the shape of the shim is load-bearing rather
    than incidental. A module-level ``__getattr__`` -- the obvious way to
    write this -- does NOT support deep imports: ``from pkg.old.kernel import
    helper`` raises ``ModuleNotFoundError`` against one. That would have
    broken every published name, because the intersection between this
    subpackage's ``__all__`` at v0.4.0 and bayesmith's top-level ``__all__``
    was EMPTY -- all seventeen were reachable only by the deep path.

    So the shim registers ``sys.modules`` aliases, and each form below is a
    form a 0.4.0 caller could have written.
    """

    FORMS: ClassVar[list[str]] = [
        "from bayesmith.evidence import SqrtInfo; assert SqrtInfo",
        "from bayesmith.evidence.compress import compress; assert compress",
        "import bayesmith.evidence.chain as m; assert m.smooth",
        "import bayesmith; assert bayesmith.evidence.SqrtInfo",
    ]

    @pytest.mark.parametrize("form", FORMS)
    def test_the_form_a_040_caller_could_have_written(self, form):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-W", "ignore::DeprecationWarning", "-c", form],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, (form, proc.stderr[-900:])

    def test_it_is_the_same_object_and_not_a_copy(self):
        """A shim that returned an equal-looking module would pass the forms
        above while giving a caller a second, divergent instance."""
        import bayesmith.evidence.compress

        import bayesmith.evidence
        import bayesmith.marginal
        import bayesmith.marginal.compress

        assert bayesmith.evidence.SqrtInfo is bayesmith.marginal.SqrtInfo
        assert bayesmith.evidence.compress is bayesmith.marginal.compress

    def test_the_old_path_warns_and_the_new_one_does_not(self):
        import subprocess
        import sys

        def warnings_from(statement: str) -> str:
            return subprocess.run(
                [sys.executable, "-W", "always::DeprecationWarning", "-c", statement],
                capture_output=True,
                text=True,
                check=False,
            ).stderr

        assert "DeprecationWarning" in warnings_from("import bayesmith.evidence")
        assert "DeprecationWarning" not in warnings_from("import bayesmith.marginal")

    def test_the_alias_list_is_derived_from_the_new_package(self):
        """Not hand-listed: a module added to `marginal/` must be reachable
        through the deprecated path too, and one deleted must stop being
        aliased rather than leave a dangling entry."""
        import pkgutil

        import bayesmith.evidence
        import bayesmith.marginal

        on_disk = {
            info.name
            for info in pkgutil.iter_modules(bayesmith.marginal.__path__)
            if not info.name.startswith("_")
        }
        assert set(bayesmith.evidence._ALIASED) == on_disk

def test_the_artifacts_subpackage_reexports_each_owning_module_s_protocol():
    """bayesmith.artifacts is the one public inventory for the R1 protocol.

    Checked by IDENTITY against each owning submodule, the way the exact and
    bridge pins above are: a re-export that returned a copy, or a name that
    resolved through the wrong module, would pass a hasattr check and fail
    here. The __all__ equality closes the other direction -- a name re-exported
    but left out of __all__, or a name in __all__ with no import behind it.
    """
    from bayesmith import artifacts
    from bayesmith.artifacts import (
        _codec,
        base,
        gates,
        identity,
        refusal,
        reports,
        results,
        tasks,
    )

    expected = {}
    for module in (identity, base, tasks, results, refusal, reports, gates):
        for name in module.__all__:
            expected[name] = getattr(module, name)
    for name in (
        "ArtifactCodecError",
        "ArtifactFile",
        "UnsupportedSchemaVersion",
        "dump_artifact",
        "load_artifact",
    ):
        expected[name] = getattr(_codec, name)

    assert set(expected) == set(artifacts.__all__), (
        "artifacts.__all__ and the owning modules disagree: "
        f"only in __all__ {sorted(set(artifacts.__all__) - set(expected))}; "
        f"missing from __all__ {sorted(set(expected) - set(artifacts.__all__))}"
    )
    for name, target in expected.items():
        assert getattr(artifacts, name) is target, name


def test_the_artifact_root_exposes_only_compile_and_execute_task():
    """Root adds exactly two R1 names, lazily, and leaks no schema names.

    compile_task/execute_task resolve through _LAZY_ATTRS to dispatch.task --
    importing them pulls in JAX/NumPyro, so they must stay lazy, and the
    dozens of Task/Result/Refusal/Gate names belong to bayesmith.artifacts,
    not to the root namespace.
    """
    import bayesmith
    from bayesmith.dispatch import task as task_module

    assert bayesmith.compile_task is task_module.compile_task
    assert bayesmith.execute_task is task_module.execute_task
    assert "compile_task" in bayesmith.__all__
    assert "execute_task" in bayesmith.__all__
    for name in (
        "PosteriorTask",
        "EvidenceResult",
        "Refusal",
        "GateResult",
        "NamedArray",
        "Fingerprint",
    ):
        assert name not in bayesmith.__all__, name
        assert not hasattr(bayesmith, name), name


class TestArtifactPersistence:
    """dump_artifact / load_artifact: the 8.2 canonical transport envelope.

    The envelope is NOT a bare pickle and not a digest the artifact computes
    for itself: payload_sha256 hashes the decoded payload bytes, and
    load_artifact checks schema, digest and expected type before handing any
    object back. A payload byte, a digest byte, or the type tag, each flipped
    alone, must fail before returning.
    """

    @staticmethod
    def _artifact():
        import numpy as np

        from bayesmith.artifacts import NamedArray

        return NamedArray("x", np.array([1.0, 2.0, 3.0]), ("draw",))

    def test_round_trip_and_no_temp_residue(self, tmp_path):
        from bayesmith.artifacts import NamedArray, dump_artifact, load_artifact

        artifact = self._artifact()
        path = tmp_path / "posterior.json"
        dump_artifact(artifact, path)
        assert [entry.name for entry in tmp_path.iterdir()] == ["posterior.json"]
        loaded = load_artifact(path, expected=NamedArray)
        assert loaded == artifact
        assert loaded is not artifact

    def test_a_flipped_payload_byte_fails_before_returning(self, tmp_path):
        import base64
        import json

        from bayesmith.artifacts import (
            ArtifactCodecError,
            NamedArray,
            dump_artifact,
            load_artifact,
        )

        path = tmp_path / "a.json"
        dump_artifact(self._artifact(), path)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = bytearray(base64.b64decode(envelope["payload_base64"]))
        payload[len(payload) // 2] ^= 0x01
        envelope["payload_base64"] = base64.b64encode(bytes(payload)).decode("ascii")
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(ArtifactCodecError):
            load_artifact(path, expected=NamedArray)

    def test_a_flipped_digest_byte_fails_before_returning(self, tmp_path):
        import json

        from bayesmith.artifacts import (
            ArtifactCodecError,
            NamedArray,
            dump_artifact,
            load_artifact,
        )

        path = tmp_path / "a.json"
        dump_artifact(self._artifact(), path)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        digest = envelope["payload_sha256"]
        envelope["payload_sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(ArtifactCodecError):
            load_artifact(path, expected=NamedArray)

    def test_a_flipped_type_tag_fails_before_returning(self, tmp_path):
        import base64
        import hashlib
        import json

        from bayesmith.artifacts import (
            ArtifactCodecError,
            NamedArray,
            dump_artifact,
            load_artifact,
        )

        path = tmp_path / "a.json"
        dump_artifact(self._artifact(), path)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        payload = json.loads(
            base64.b64decode(envelope["payload_base64"]).decode("utf-8")
        )
        # Flip the qualified type name the decoder resolves THROUGH its
        # registry, and recompute the digest so only the type tag is wrong.
        payload["type"] = payload["type"].replace("NamedArray", "NotNamedArray")
        reencoded = json.dumps(
            payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        envelope["payload_base64"] = base64.b64encode(reencoded).decode("ascii")
        envelope["payload_sha256"] = hashlib.sha256(reencoded).hexdigest()
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(ArtifactCodecError):
            load_artifact(path, expected=NamedArray)

    def test_an_expected_type_mismatch_fails_before_returning(self, tmp_path):
        from bayesmith.artifacts import (
            ArtifactCodecError,
            ComputeBudget,
            dump_artifact,
            load_artifact,
        )

        path = tmp_path / "a.json"
        dump_artifact(self._artifact(), path)
        with pytest.raises(ArtifactCodecError):
            load_artifact(path, expected=ComputeBudget)

    def test_a_newer_codec_version_is_refused_as_unsupported(self, tmp_path):
        import json

        from bayesmith.artifacts import (
            UnsupportedSchemaVersion,
            dump_artifact,
            load_artifact,
        )

        path = tmp_path / "a.json"
        dump_artifact(self._artifact(), path)
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["codec_version"] = 2
        path.write_text(json.dumps(envelope), encoding="utf-8")
        with pytest.raises(UnsupportedSchemaVersion):
            load_artifact(path)


def test_the_artifact_docs_pin_the_five_results_and_grounds():
    """docs/artifacts.md is a published module-spec, so a lightweight guard
    pins the protocol's own names and the frozen Refusal field -- not the prose.

    The tables are a design decision to edit freely; what must not silently
    drift is the five Result names and the field 0 ruling 3 froze as grounds
    (never evidence).
    """
    import pathlib

    text = pathlib.Path("docs/artifacts.md").read_text(encoding="utf-8")
    for name in (
        "PosteriorResult",
        "EvidenceResult",
        "PredictiveResult",
        "PointEstimateResult",
        "SimulationResult",
    ):
        assert name in text, name
    assert "`grounds`" in text


def test_the_readme_workflow_mentions_the_typed_round_trip():
    """README must show the typed chain and say the legacy entry points stay."""
    import pathlib

    text = pathlib.Path("README.md").read_text(encoding="utf-8")
    for token in ("PosteriorTask", "compile_task", "execute_task", "PosteriorResult"):
        assert token in text, token
    assert "legacy" in text.lower()
