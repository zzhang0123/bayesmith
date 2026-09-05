"""R5 Task 4 -- the residual-evidence extra is absent, and that is the default.

Two halves, and the second is what makes the first a boundary rather than a
regression: an ``EvidenceTask`` over a class-(b) or class-(c) graph is refused
by name, and a ``PosteriorTask`` over the **same graph** compiles unchanged.

**The absence path RUNS here.** Nothing below is an ``importorskip`` and nothing
is monkeypatched into place: neither candidate is installed in this checkout, so
"absent" is the state the repository is in and the state most consumers will be
in. R5 plan 0.9.

**And the capability probe is graded as a BYPASS, not read.** The plan's 0.16
measured that a partially installed ``jaxns`` writes ``jax.config`` at import and
*then* fails with an ``AttributeError`` -- which ``except ImportError`` does not
catch -- so the obvious probe

    try:
        import jaxns
    except ImportError:
        pass

flips ``jax_enable_x64`` process-globally while returning a clean-looking
negative. A flipped flag is not a wrong number; it is R4's own precision gate no
longer firing, because ``dispatch/task.py`` decides ``evidence_requires_x64`` by
OUTCOME (``jnp.result_type(float)``) so that the context manager and the
process-global switch give one answer.

``test_the_naive_probe_poisons_a_fresh_process_and_this_one_does_not`` builds
that exact module -- writes the flag, then raises ``AttributeError`` -- installs
it under each candidate's own distribution name, and runs both probes in one
subprocess. The subprocess is not decoration: demonstrating the poisoning
in-process would poison this suite.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import jax
import pytest

from bayesmith import compile_task
from bayesmith.artifacts.refusal import CAPABILITY_UNAVAILABLE_R1, Refusal
from bayesmith.artifacts.tasks import EvidenceTask, PosteriorTask, new_task_meta
from tests.dispatch.test_task_protocol import model_ref as _model_ref

ROOT = Path(__file__).resolve().parents[2]

#: A module that behaves the way plan 0.16 measured a partially installed
#: ``jaxns`` behaving: ``jax.config`` is written at module scope, and the import
#: then fails with the exception a capability probe does NOT catch. The message
#: is the one 0.16 recorded, so a reader can match them up.
_POISONING_MODULE = (
    "import jax\n"
    "jax.config.update('jax_enable_x64', True)\n"
    "raise AttributeError(\n"
    "    \"module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'\"\n"
    ")\n"
)

#: A module that imports cleanly and touches nothing.
_QUIET_MODULE = "STATUS = 'importable'\n"


def _install_fake(root: Path, distribution: str, source: str, version="9.9.9"):
    """A findable distribution whose module has not been imported.

    Both halves matter. The ``.dist-info`` is what ``importlib.metadata`` reads,
    and the ``.py`` is what an import would execute -- so a probe that answers
    from the first and never touches the second is the thing under test, and a
    probe that reaches the second is caught by the module's own side effect.
    """
    (root / f"{distribution}.py").write_text(source, encoding="utf-8")
    info = root / f"{distribution}-{version}.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (info / "RECORD").write_text("", encoding="utf-8")
    return version


class _RaisingFinder:
    """A ``sys.meta_path`` entry whose distribution lookup cannot answer.

    This is the third state red line 14 requires: the probe did not find the
    distribution absent, it failed to look. An unreadable path entry, a
    metadata backend that raises -- the shape does not matter, the DISTINCT
    ANSWER does.
    """

    def __init__(self, names=None):
        #: Which distributions this finder refuses to answer about. ``None``
        #: means all of them; a set lets the refusal path keep working for
        #: everything that is not the subject of the test.
        self.names = names

    def find_distributions(self, context=None):
        wanted = getattr(context, "name", None)
        if self.names is None or wanted in self.names:
            raise OSError("the metadata backend could not be read")
        return ()

    def find_spec(self, fullname, path=None, target=None):
        return None


#: Wave B's five residual fixtures, with the structural class each one compiles
#: to. **Measured on this tree** (2026-09-05, at `42d77db` plus this change),
#: not read off the plan: three are class (b) -- an exact block AND a residual
#: one -- and two are class (c).
#:
#:     mixture_prior_residual     (b)  exact=('b',)   gcr  sampled=('w',)
#:     outside_observation_pair   (b)  exact=('x',)   gcr  sampled=('tau',)
#:     shifted_block_prior        (b)  exact=('x',)   gcr  sampled=('tau',)
#:     cauchy_residual_pair       (c)  exact=()            sampled=('z',)
#:     undeclared_quartet         (c)  exact=()            sampled=4 latents
#:
#: Class (b) is the harder half and §8 R5's headline, so the refusal is graded
#: against it rather than only against the all-residual case. These live in
#: ``tests/exact/residual_models.py``; ``tests/exact/models.py`` is a census
#: denominator pinned in three places and gains nothing here.
#:
#: ``tests/dispatch/test_residual_fixtures.py`` owns the taxonomy census and
#: this list is not a second copy of it: that test asserts where each fixture
#: SITS, and this one asserts what the refusal SAYS. If the routing moves, that
#: census reddens first and its message names the remedy.
GRAPHS = {
    "b_mixture_prior_residual": "mixture_prior_residual",
    "b_outside_observation_pair": "outside_observation_pair",
    "b_shifted_block_prior": "shifted_block_prior",
    "c_cauchy_residual_pair": "cauchy_residual_pair",
    "c_undeclared_quartet": "undeclared_quartet",
}


def _graph(name):
    """One of Wave B's fixtures, built at the precision its closed form assumes.

    Every caller is already inside ``jax.enable_x64(True)``; ``residual_models``
    raises rather than answering about a float32 model, so a call that drifted
    out of the block fails loudly instead of quietly.
    """
    from tests.exact import residual_models

    return getattr(residual_models, GRAPHS[name])()


def _requirement_name(requirement: str) -> str:
    """The distribution a PEP 508 requirement names, without its version.

    ``blackjax>=1.6`` -> ``blackjax``; ``jaxns[plot] ; python_version<'3.13'``
    -> ``jaxns``. Cut at the first character a distribution name cannot hold.
    """
    return re.split(r"[^A-Za-z0-9._-]", requirement.strip(), maxsplit=1)[0]


def _normalise(name: str) -> str:
    """PEP 503: one project, one string. ``Black_Jax`` -> ``black-jax``."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _extra_provides(requirements, distribution) -> bool:
    """Does installing this extra install the distribution the probe looks for?

    Extracted so the exploit cases below can reach it. The guard over the real
    ``pyproject.toml`` calls this, and so does
    ``test_an_extra_that_installs_a_near_miss_is_not_accepted`` -- which is the
    fixture this repository does not contain: with two well-formed rows,
    ``req.startswith(distribution)`` and this function agree on every input the
    package holds, so restoring the loose version leaves the suite green. That
    is red line 13's fault (b), and the remedy for (b) is a fixture.
    """
    return _normalise(distribution) in {
        _normalise(_requirement_name(requirement)) for requirement in requirements
    }


def _extras_in(command: str) -> set[str]:
    """The extras a ``pip install`` command actually asks for.

    Reads the text rather than rebuilding it. ``pip install "bayesmith[a,b]"``
    -> ``{"a", "b"}``; a command with no bracket -> the empty set, which is the
    case that matters: it is what a command that forgot the extra looks like.
    """
    inside = re.findall(r"\[([^\]]*)\]", command)
    return {part.strip() for group in inside for part in group.split(",") if part.strip()}


def _refuse(graph):
    outcome = compile_task(
        graph,
        EvidenceTask(meta=new_task_meta(label="log evidence")),
        model_ref=_model_ref(),
    )
    assert isinstance(outcome, Refusal), (
        f"expected the capability refusal, got {type(outcome).__name__}"
    )
    assert outcome.failed_premise == CAPABILITY_UNAVAILABLE_R1
    return outcome


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_the_refusal_names_the_missing_extra_and_how_to_install_it(name):
    """4.1's first half, at the layer a caller reads.

    Not "a refusal exists": a caller who is told a capability is missing and
    not told which package supplies it has been given a dead end wearing a
    schema. The extra's NAME and the COMMAND are both asserted, and both are
    asserted against the extras table rather than against a literal, so a
    table that grows is covered without this test being edited.

    **The command is parsed, not compared against itself.** An earlier version
    asserted ``install_command(extra) in prose``, which is the refusal and the
    guard reading the same function -- measured: replacing
    ``install_command``'s body with a bare ``"pip install bayesmith"`` left this
    green, because the message it is checked against is built from the same
    call. The assertion now reads the extras OUT of the command text and
    compares them with the table, so the two sides come from different places.
    """
    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS, install_command

    with jax.enable_x64(True):
        refusal = _refuse(_graph(name))

    message = refusal.grounds[0].message
    assert RESIDUAL_EVIDENCE_EXTRAS, "the extras table is empty, so this proves nothing"
    for extra, _distribution in RESIDUAL_EVIDENCE_EXTRAS:
        assert extra in message, f"the refusal never names the {extra!r} extra"
        command = install_command(extra)
        assert _extras_in(command) == {extra}, (
            f"{command!r} does not ask pip for the {extra!r} extra, so a caller "
            f"who runs it installs bayesmith again and gets this refusal twice"
        )
        assert command in message, (
            f"the refusal names {extra!r} without saying how to install it"
        )


def test_the_refusal_carries_a_remedy_a_program_can_act_on():
    """The remedy row, asserted as a field rather than as prose in a heap.

    **The first version of this file never graded the remedy at all.** It
    concatenated ``grounds[0].message`` with every remedy's message and searched
    the heap, so removing the install remedy entirely left the guard green --
    measured, 141 passed, exit 0 -- because the same commands appear in the
    finding's own message. A refusal's ``remedies`` is the field a caller
    branches on, and a heap that contains the right words is not that field.

    So: exactly one remedy offers the extras, it names every install command,
    and it carries the extras as a PARAMETER, which is the half a program reads
    when it is not reading English.
    """
    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS, install_command

    with jax.enable_x64(True):
        refusal = _refuse(_graph("b_mixture_prior_residual"))

    offering = [
        remedy
        for remedy in refusal.remedies
        if all(
            install_command(extra) in remedy.message
            for extra, _distribution in RESIDUAL_EVIDENCE_EXTRAS
        )
    ]
    assert len(offering) == 1, (
        f"expected exactly one remedy offering the extras; the refusal carries "
        f"{[remedy.action for remedy in refusal.remedies]}"
    )
    parameters = dict(offering[0].parameters)
    assert parameters.get("extras") == tuple(
        extra for extra, _distribution in RESIDUAL_EVIDENCE_EXTRAS
    ), parameters


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_a_posterior_task_over_the_same_graph_is_unaffected(name):
    """4.1's second half -- the one that separates a boundary from a regression.

    The same graph object, the same call, a different task kind.
    """
    with jax.enable_x64(True):
        graph = _graph(name)
        _refuse(graph)
        outcome = compile_task(
            graph, PosteriorTask(meta=new_task_meta(label="p")), model_ref=_model_ref()
        )
    assert not isinstance(outcome, Refusal), (
        "the capability refusal broke the neighbouring capability, which makes "
        "it a regression wearing a boundary's name"
    )


def test_the_extras_table_is_exactly_what_pyproject_declares():
    """The enumeration, with its denominator stated (red line 16).

    **What this counts:** the keys of ``[project.optional-dependencies]`` in
    ``pyproject.toml``, all of them.

    **What it excludes, and why each is not an extra.** The four
    ``[project].dependencies`` -- jax, equinox, numpy, numpyro -- are hard
    requirements; the pyproject comment says numpyro is "the last row of the
    dispatch table, not an optional extra". ``[dependency-groups]`` (``dev``,
    ``crosscheck``) are a different table with a different meaning: they are
    never installed by ``pip install bayesmith[...]`` and never reach a wheel.

    **It is an allow-list in both directions.** An extra added to
    ``pyproject.toml`` and not to the table is a capability the refusal cannot
    name; a table entry with no extra is an install command that does not
    work. Equality catches both, and neither defaults to admitted.
    """
    import tomllib

    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = set(declared["project"].get("optional-dependencies", {}))
    assert extras == {extra for extra, _dist in RESIDUAL_EVIDENCE_EXTRAS}, {
        "declared in pyproject.toml": sorted(extras),
        "named by the refusal": sorted(e for e, _ in RESIDUAL_EVIDENCE_EXTRAS),
    }
    assert extras, "pyproject.toml declares no extras at all"
    for extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
        requirements = declared["project"]["optional-dependencies"][extra]
        # The requirement's NAME, parsed and normalised, not a prefix.
        # `startswith` was the first version of this line and it admits
        # `blackjax-nightly`: the extra would install successfully, the probe
        # would still find no `blackjax`, and the refusal would say the extra
        # is absent with nothing anywhere to say why. PEP 503 normalisation
        # because `Black_Jax`, `black.jax` and `black-jax` are one project to
        # pip and three different strings here.
        assert _extra_provides(requirements, distribution), {
            "the extra": extra,
            "the probe looks for": _normalise(distribution),
            "the extra installs": sorted(
                _normalise(_requirement_name(req)) for req in requirements
            ),
        }


@pytest.mark.parametrize(
    ("requirements", "distribution", "provides"),
    [
        # What the table actually declares today.
        (["blackjax>=1.6"], "blackjax", True),
        (["jaxns>=2.6"], "jaxns", True),
        # The near miss `startswith` admits. The extra would install, and the
        # probe would still report the extra absent -- a refusal that a correct
        # install cannot clear, with nothing anywhere saying why.
        (["blackjax-nightly>=1.6"], "blackjax", False),
        (["jaxns2>=2.6"], "jaxns", False),
        # PEP 503: pip reads each of these pairs as ONE project, so this file
        # must too -- case folds, and `-` `_` `.` are one separator.
        (["BlackJAX>=1.6"], "blackjax", True),
        (["black_jax>=1.6"], "black-jax", True),
        (["black.jax"], "black-jax", True),
        # And it does NOT fold a separator into its absence: `Black_Jax` is
        # `black-jax`, which is a different project from `blackjax`. This row
        # was written the other way round and the parametrisation caught it,
        # which is the whole reason it is a table and not an assertion.
        (["Black_Jax>=1.6"], "blackjax", False),
        # Shapes the version parser has to survive rather than mistake for a
        # different name.
        (["jaxns[plot] ; python_version<'3.13'"], "jaxns", True),
        (["blackjax == 1.6.2"], "blackjax", True),
        (["blackjax"], "blackjax", True),
        # An extra that installs nothing the probe looks for.
        ([], "blackjax", False),
        (["optax"], "blackjax", False),
    ],
)
def test_an_extra_that_installs_a_near_miss_is_not_accepted(
    requirements, distribution, provides
):
    """The fixture the repository does not contain, so the guard has one.

    The real table has two well-formed rows, and on those the loose check
    (`req.startswith(distribution)`) and the parsed one agree -- measured, the
    suite stays green either way. These are the inputs on which they differ.
    """
    assert _extra_provides(requirements, distribution) is provides


def test_the_probe_reports_absent_when_the_distribution_is_not_installed():
    """The state this checkout is in, asked at the function the refusal calls."""
    from bayesmith.dispatch.task import EXTRA_ABSENT, optional_extra_status

    status = optional_extra_status("nothing", "bayesmith-no-such-distribution")
    assert status.state == EXTRA_ABSENT
    assert status.version is None
    assert status.detail


def test_the_probe_reports_installed_without_importing_the_distribution(tmp_path):
    """The present state, and the assertion is about what did NOT happen.

    A probe that answers "installed" by importing is correct and unusable: the
    import is the side effect the extra exists to avoid paying for. So the
    check is that ``sys.modules`` did not gain the module and that the version
    came back anyway.
    """
    from bayesmith.dispatch.task import EXTRA_INSTALLED, optional_extra_status

    name = "bayesmith_probe_quiet"
    version = _install_fake(tmp_path, name, _QUIET_MODULE)
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        assert name not in sys.modules
        status = optional_extra_status("quiet", name)
        assert status.state == EXTRA_INSTALLED
        assert status.version == version
        assert name not in sys.modules, (
            "the probe imported the distribution it was asked about"
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()


def test_a_poisoning_distribution_is_reported_installed_and_never_executed(tmp_path):
    """The broken state, in-process, and the flag is SHOWN unchanged.

    The module here writes ``jax_enable_x64`` and then raises. If the probe
    touches it at all, this test flips the process flag and says so.
    """
    from bayesmith.dispatch.task import EXTRA_INSTALLED, optional_extra_status

    name = "bayesmith_probe_poison"
    _install_fake(tmp_path, name, _POISONING_MODULE)
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    before = jax.config.jax_enable_x64
    try:
        status = optional_extra_status("poison", name)
        assert status.state == EXTRA_INSTALLED
        assert name not in sys.modules
        assert jax.config.jax_enable_x64 == before, (
            "the probe executed the distribution and it moved jax.config"
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        jax.config.update("jax_enable_x64", before)


def test_the_refusal_itself_says_the_lookup_declined_rather_than_saying_absent():
    """Red line 15: the same three states, asked at the layer a caller reads.

    **This test exists because its absence was measured.** The sibling below
    calls ``optional_extra_status`` directly, and with only that one,
    ``_extras_sentence``'s UNKNOWN branch was reachable by nothing: a mutant
    rewriting that branch to print the ABSENT text survived the whole file --
    142 passed, exit 0. A report is an intermediate value, and a test on an
    intermediate value grades the calculation rather than the decision.

    So the metadata backend is broken for exactly the residual-evidence
    distributions -- everything else still resolves, so `compile_task` runs
    normally -- and the REFUSAL is read.
    """
    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS, install_command

    subjects = {distribution for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS}
    finder = _RaisingFinder(subjects)
    sys.meta_path.insert(0, finder)
    try:
        with jax.enable_x64(True):
            message = _refuse(_graph("c_cauchy_residual_pair")).grounds[0].message
    finally:
        sys.meta_path.remove(finder)

    for extra, _distribution in RESIDUAL_EVIDENCE_EXTRAS:
        assert f"whether the {extra} extra is installed could not be" in message, (
            f"the refusal reported {extra!r} as though the lookup had happened"
        )
        assert install_command(extra) not in message, (
            f"the refusal told a caller to install {extra!r} on the strength of "
            f"a lookup that never completed"
        )
    assert "OSError" in message, (
        "a declined lookup that does not say why is a second silence"
    )


def test_a_probe_that_could_not_look_says_so_rather_than_saying_absent():
    """Red line 14, as a value rather than as a promise.

    ``absent`` and ``could not tell`` are different facts about the world and a
    consumer branches differently on them: the first is answered by installing
    the extra, the second by finding out why the lookup failed. Reporting the
    second as the first is this repository's founding disease -- a result that
    cannot distinguish the thing it names from a thing resembling it.
    """
    from bayesmith.dispatch.task import (
        EXTRA_ABSENT,
        EXTRA_UNKNOWN,
        optional_extra_status,
    )

    finder = _RaisingFinder()
    sys.meta_path.insert(0, finder)
    try:
        status = optional_extra_status("blocked", "bayesmith-no-such-distribution")
    finally:
        sys.meta_path.remove(finder)

    assert status.state == EXTRA_UNKNOWN
    assert status.state != EXTRA_ABSENT
    assert "OSError" in status.detail, (
        "a declined lookup that does not say why is a second silence"
    )


def test_the_naive_probe_poisons_a_fresh_process_and_this_one_does_not(tmp_path):
    """4.4 -- run in all three states, and the flag SHOWN rather than asserted.

    One subprocess, four measurements in order, so that the poisoning is
    demonstrated on the same interpreter that had just been shown clean:

    1.  ``jax_enable_x64`` on a fresh process.
    2.  after ``residual_evidence_extras()`` -- this package's probe, over the
        real extras table, with a poisoning module installed under every one of
        those distribution names.
    3.  after the whole refusal path (``compile_task`` on a class-(c) graph),
        which is what a consumer actually runs.
    4.  after ``try: import <candidate> except ImportError: pass`` -- the probe
        this package does not write.

    (4) is the control. If it did not flip the flag the fake would not be
    modelling 0.16's hazard and (2) and (3) would prove nothing.
    """
    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS

    for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
        _install_fake(tmp_path, distribution, _POISONING_MODULE)

    script = textwrap.dedent(
        """
        import json, sys
        import jax
        from bayesmith import compile_task
        from bayesmith.artifacts.tasks import EvidenceTask, new_task_meta
        from bayesmith.artifacts.identity import ModelRef
        from bayesmith.dispatch.task import (
            RESIDUAL_EVIDENCE_EXTRAS, residual_evidence_extras,
        )

        out = {"start": jax.config.jax_enable_x64}

        out["probe"] = [s.state for s in residual_evidence_extras()]
        out["after_probe"] = jax.config.jax_enable_x64
        out["imported_after_probe"] = sorted(
            d for _e, d in RESIDUAL_EVIDENCE_EXTRAS if d in sys.modules
        )

        # A class-(b) fixture: an exact block AND a residual one, which is the
        # half a wrong precision would actually corrupt.
        from tests.exact.residual_models import shifted_block_prior

        with jax.enable_x64(True):
            refusal = compile_task(
                shifted_block_prior(),
                EvidenceTask(meta=new_task_meta(label="Z")),
                model_ref=ModelRef(identifier="line", source_digest="a" * 64),
            )
        out["premise"] = getattr(refusal, "failed_premise", None)
        out["after_refusal"] = jax.config.jax_enable_x64

        naive = {}
        for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
            try:
                __import__(distribution)
                naive[distribution] = "imported"
            except ImportError as error:
                naive[distribution] = "ImportError"
            except BaseException as error:
                naive[distribution] = type(error).__name__
        out["naive"] = naive
        out["after_naive"] = jax.config.jax_enable_x64
        print(json.dumps(out))
        """
    )

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        check=False,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr[-3000:])
    out = json.loads(proc.stdout.strip().splitlines()[-1])

    # (4) the control: the fake really is 0.16's hazard, and the exception it
    # raises is not the one the naive probe catches.
    assert set(out["naive"].values()) == {"AttributeError"}, out["naive"]
    assert out["after_naive"] is True, (
        "the fake did not poison the process, so this test grades nothing"
    )

    # (1)-(3) what this package does instead.
    assert out["start"] is False
    assert out["probe"] == ["installed"] * len(out["probe"])
    assert out["imported_after_probe"] == []
    assert out["after_probe"] is False, "the probe moved jax.config"
    assert out["premise"] == CAPABILITY_UNAVAILABLE_R1
    assert out["after_refusal"] is False, "the refusal path moved jax.config"


def test_an_installed_extra_is_not_reported_as_a_missing_one(tmp_path):
    """The refusal's message follows the probe rather than a constant.

    With the extra present and no adapter behind it -- which is exactly where
    R5 stands until Task 6 -- "install the extra" is false advice: the caller
    would install what they already have and get the same refusal. The two
    states have to produce two messages, and the premise stays the same because
    the missing capability is still what is true.
    """
    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS, install_command

    with jax.enable_x64(True):
        absent = _refuse(_graph("c_cauchy_residual_pair")).grounds[0].message

        for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
            _install_fake(tmp_path, distribution, _QUIET_MODULE)
        sys.path.insert(0, str(tmp_path))
        importlib.invalidate_caches()
        try:
            present = _refuse(_graph("c_cauchy_residual_pair")).grounds[0].message
        finally:
            sys.path.remove(str(tmp_path))
            importlib.invalidate_caches()

    assert "9.9.9" in present, (
        "the refusal did not read the probe: it cannot name the version of a "
        "distribution it never looked up"
    )
    for extra, _distribution in RESIDUAL_EVIDENCE_EXTRAS:
        assert install_command(extra) in absent, (
            "the absent state stopped naming the install command"
        )
        assert install_command(extra) not in present, (
            f"the refusal told a caller to install {extra}, which is installed"
        )
    assert present != absent
