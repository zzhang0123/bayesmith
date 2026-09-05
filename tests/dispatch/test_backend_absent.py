"""R5 Task 4 -- the residual-evidence extra is absent, and that is the default.

Two halves, and the second is what makes the first a boundary rather than a
regression: an ``EvidenceTask`` over a class-(b) or class-(c) graph is refused
by name, and a ``PosteriorTask`` over the **same graph** compiles unchanged.

**The absence path RUNS here.** Neither candidate is installed in this checkout,
so "absent" is the state the repository is in and the state most consumers will
be in, and ``test_the_refusal_reports_whatever_state_this_environment_is_in``
reads it unforced. Plan §0.9.

**But the absence path is not the only state, and the first version of this file
assumed it was.** An adversarial review installed `blackjax` -- the thing the
extra exists to install -- and the file went red: every message assertion took
its "absent" baseline from the machine it was running on. Fast layer without
blackjax `1 failed, 3548 passed`; with it, `2 failed, 3547 passed`. Task 4.6's
stop-rule fired, on a test defect rather than a backend defect.

So the states are now BUILT rather than inherited. ``_refuse_with`` forces the
probe's answer and the table below covers all of them, including **mixed** --
one extra installed and one absent, which is exactly what
``pip install "bayesmith[blackjax]"`` produces and which no fixture used to
build.

**And the capability probe is graded as a BYPASS, not read.** Plan §0.16
measured that a partially installed ``jaxns`` writes ``jax.config`` at import and
*then* fails with an ``AttributeError`` -- which ``except ImportError`` does not
catch -- so the obvious probe

    try:
        import jaxns
    except ImportError:
        pass

flips ``jax_enable_x64`` process-globally while returning a clean-looking
negative. That is not a wrong number; it is R4's own precision gate no longer
firing, because ``dispatch/task.py`` decides ``evidence_requires_x64`` by
OUTCOME (``jnp.result_type(float)``) so the context manager and the
process-global switch give one answer.
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

#: The install commands, written out. **Literals on purpose.** The first
#: version of this file asserted ``install_command(extra) in message``, which is
#: the refusal and the guard calling one function: a review replaced that
#: function's body three ways -- dropping the quoting, returning the wrong
#: extra, naming no extra at all -- and every one survived, because whatever it
#: returned appeared on both sides of the ``in``. A literal has no such
#: symmetry: it comes from a person reading ``pyproject.toml``.
BLACKJAX_COMMAND = 'pip install "bayesmith[blackjax]"'
JAXNS_COMMAND = 'pip install "bayesmith[jaxns]"'
COMMANDS = {"blackjax": BLACKJAX_COMMAND, "jaxns": JAXNS_COMMAND}

#: A module that behaves the way plan §0.16 measured a partially installed
#: ``jaxns`` behaving: ``jax.config`` is written at module scope, and the import
#: then fails with the exception a capability probe does NOT catch.
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
    and the ``.py`` is what an import would execute -- so a probe answering from
    the first and never touching the second is the thing under test, and one
    that reaches the second is caught by the module's own side effect.
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

    The third state red line 14 requires: the probe did not find the
    distribution absent, it failed to look.

    ``names`` limits which distributions it refuses to answer about, so the
    refusal path can be reached with everything else still resolving.
    ``error`` varies the failure, because a guard that only ever sees one
    exception type is a guard on that type: a review inserted an
    ``except ImportError`` arm returning ABSENT *before* the generic handler and
    the whole suite stayed green, since the only case exercised was ``OSError``.
    """

    def __init__(self, names=None, error=None):
        self.names = names
        self.error = error or OSError("the metadata backend could not be read")

    def find_distributions(self, context=None):
        wanted = getattr(context, "name", None)
        if self.names is None or wanted in self.names:
            raise self.error
        return ()

    def find_spec(self, fullname, path=None, target=None):
        return None


#: Wave B's five residual fixtures, with the structural class each compiles to.
#: **Measured on this tree**, not read off the plan: three are class (b) -- an
#: exact block AND a residual one -- and two are class (c).
#:
#:     mixture_prior_residual     (b)  exact=('b',)   gcr  sampled=('w',)
#:     outside_observation_pair   (b)  exact=('x',)   gcr  sampled=('tau',)
#:     shifted_block_prior        (b)  exact=('x',)   gcr  sampled=('tau',)
#:     cauchy_residual_pair       (c)  exact=()            sampled=('z',)
#:     undeclared_quartet         (c)  exact=()            sampled=4 latents
#:
#: Class (b) is the harder half and §8 R5's headline, so the refusal is graded
#: against it rather than only against the all-residual case.
#:
#: **What records this taxonomy, stated exactly, because an earlier version of
#: this comment was wrong about it.** It claimed
#: ``tests/dispatch/test_residual_fixtures.py`` owns the census so that "if the
#: routing moves, that census reddens first". That census asserts routing for
#: three of these five; ``outside_observation_pair`` appears in it **zero**
#: times and ``shifted_block_prior`` only in a closed-form test. For those two
#: the assertion below is the only thing that reads their routing, which is why
#: it asserts the class rather than trusting the key's prefix.
GRAPHS = {
    "b_mixture_prior_residual": "mixture_prior_residual",
    "b_outside_observation_pair": "outside_observation_pair",
    "b_shifted_block_prior": "shifted_block_prior",
    "c_cauchy_residual_pair": "cauchy_residual_pair",
    "c_undeclared_quartet": "undeclared_quartet",
}


def _graph(name):
    """One of Wave B's fixtures, built at the precision its closed form assumes.

    Every caller is inside ``jax.enable_x64(True)``; ``residual_models`` raises
    rather than answering about a float32 model, so a call that drifted out of
    the block fails loudly instead of quietly.
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

    Extracted so the near-miss cases below can reach it. ``startswith`` was the
    first version and it admits ``blackjax-nightly``: the extra installs, the
    probe still finds no ``blackjax``, and the refusal reports the extra absent
    forever. A review built exactly that and the suite stayed green.
    """
    return _normalise(distribution) in {
        _normalise(_requirement_name(requirement)) for requirement in requirements
    }


def _status(extra, state, version=None, distribution=None, detail="forced"):
    """One ``ExtraStatus``, built rather than measured.

    ``distribution`` defaults to something DIFFERENT from ``extra``. In the real
    table the two are the same string for both rows, so nothing distinguished
    them: mutants filling ``extra`` with the distribution, filling
    ``distribution`` with the extra, and swapping the pair at the call site all
    survived. Here they differ, so the message has to name the right one.
    """
    from bayesmith.dispatch.task import ExtraStatus

    return ExtraStatus(
        extra=extra,
        distribution=distribution or f"{extra}-dist",
        state=state,
        version=version,
        detail=detail,
    )


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


def _refuse_with(monkeypatch, statuses, name="c_cauchy_residual_pair"):
    """The refusal a caller reads, with the PROBE's answer forced.

    **This is the repair for a stop-rule that fired.** Forcing is the only way
    to reach three of the four states: this checkout has neither candidate, CI
    has neither, and a test whose expected value is a property of the machine it
    runs on is not a test of the code. The unforced ABSENT path still runs in
    ``test_the_refusal_reports_whatever_state_this_environment_is_in``, which is
    what §0.9 requires -- that path is built and run, not mocked.
    """
    import bayesmith.dispatch.task as task_module

    monkeypatch.setattr(
        task_module, "residual_evidence_extras", lambda: tuple(statuses)
    )
    with jax.enable_x64(True):
        return _refuse(_graph(name)).grounds[0].message


def _clause_for(message: str, extra: str) -> str:
    """The part of the extras sentence that talks about ``extra``.

    The sentence is a ``; ``-joined list, one part per extra. Matching the part
    rather than the whole message is what makes "the message says the right
    thing about the WRONG extra" a failure: a mutant that computed one
    collective state for all extras printed a jaxns clause reading "the jaxns
    extra is installed (jaxns None)" and no whole-message assertion saw it.
    """
    parts = [part for part in message.split("; ") if f"the {extra} extra" in part]
    assert len(parts) == 1, (
        f"expected exactly one clause about the {extra!r} extra, found "
        f"{len(parts)} in {message!r}"
    )
    return parts[0]


# ------------------------------------------------------------ the state table

def _states():
    from bayesmith.dispatch.task import EXTRA_ABSENT, EXTRA_INSTALLED, EXTRA_UNKNOWN

    return {
        # The state this checkout is in, and CI's default.
        "both_absent": (
            _status("blackjax", EXTRA_ABSENT),
            _status("jaxns", EXTRA_ABSENT),
        ),
        # What `pip install "bayesmith[blackjax]"` actually produces. **No
        # fixture built this before an adversarial review named it**, and it is
        # the state that turned the first version of this file red.
        "mixed_blackjax_installed": (
            _status("blackjax", EXTRA_INSTALLED, version="1.6.2"),
            _status("jaxns", EXTRA_ABSENT),
        ),
        "mixed_jaxns_installed": (
            _status("blackjax", EXTRA_ABSENT),
            _status("jaxns", EXTRA_INSTALLED, version="2.6.9"),
        ),
        "both_installed": (
            _status("blackjax", EXTRA_INSTALLED, version="1.6.2"),
            _status("jaxns", EXTRA_INSTALLED, version="2.6.9"),
        ),
        "both_unknown": (
            _status("blackjax", EXTRA_UNKNOWN, detail="OSError: unreadable"),
            _status("jaxns", EXTRA_UNKNOWN, detail="OSError: unreadable"),
        ),
        # The third state beside a first: red line 14's whole point is that
        # these do not collapse, so they have to be observed apart.
        "mixed_unknown_and_absent": (
            _status("blackjax", EXTRA_UNKNOWN, detail="RuntimeError: no backend"),
            _status("jaxns", EXTRA_ABSENT),
        ),
    }


@pytest.mark.parametrize("state", sorted(_states()))
def test_the_refusal_message_maps_each_extra_state_to_its_own_sentence(
    monkeypatch, state
):
    """Every state, per extra, against literals.

    Three properties per extra, and each one is a mutant that used to live:

    * ABSENT gets the install command -- the literal, not ``install_command``'s
      own output.
    * INSTALLED gets the version and **not** the command, because telling a
      caller to install what they have hands them the same refusal twice.
    * UNKNOWN gets neither: a lookup that did not complete has not earned an
      install command, and it must not be reported as an absence.
    """
    from bayesmith.dispatch.task import EXTRA_ABSENT, EXTRA_INSTALLED, EXTRA_UNKNOWN

    statuses = _states()[state]
    message = _refuse_with(monkeypatch, statuses)

    for status in statuses:
        clause = _clause_for(message, status.extra)
        command = COMMANDS[status.extra]
        if status.state == EXTRA_ABSENT:
            assert command in clause, (
                f"{state}: {status.extra} is absent and the refusal does not "
                f"say how to install it"
            )
            assert "could not be determined" not in clause
        elif status.state == EXTRA_INSTALLED:
            assert status.version in clause, (
                f"{state}: the refusal cannot name the version of a "
                f"distribution it never looked up"
            )
            assert status.distribution in clause, (
                f"{state}: the refusal names the extra but not the "
                f"distribution that supplies it"
            )
            assert command not in clause, (
                f"{state}: the refusal told a caller to install "
                f"{status.extra}, which is installed"
            )
        else:
            assert status.state == EXTRA_UNKNOWN
            assert "could not be determined" in clause, (
                f"{state}: a lookup that did not complete was reported as "
                f"though it had"
            )
            assert status.detail in clause, (
                f"{state}: a declined lookup that does not say why is a "
                f"second silence"
            )
            assert command not in clause, (
                f"{state}: the refusal earned an install command from a "
                f"lookup that never happened"
            )

    # The other extra's command must never leak into this one's clause.
    for status in statuses:
        clause = _clause_for(message, status.extra)
        for other, command in COMMANDS.items():
            if other != status.extra:
                assert command not in clause, (
                    f"{state}: {status.extra}'s clause carries {other}'s command"
                )


def test_the_refusal_reports_whatever_state_this_environment_is_in():
    """§0.9's requirement: the absence path RUNS, unforced and unmocked.

    Environment-independent by construction -- it reads the live probe and
    asserts the mapping rather than a fixed answer -- so it passes in this
    checkout (both absent), in CI (both absent), and in an environment where
    somebody has installed one or both. **That last part is the repair.** The
    version of this test that hard-coded the absent answer went red the moment
    `blackjax` was installed, which is a stop-rule firing on a test defect.
    """
    from bayesmith.dispatch.task import (
        EXTRA_ABSENT,
        EXTRA_INSTALLED,
        EXTRA_UNKNOWN,
        residual_evidence_extras,
    )

    statuses = residual_evidence_extras()
    assert statuses, "the extras table is empty, so this proves nothing"

    with jax.enable_x64(True):
        message = _refuse(_graph("b_mixture_prior_residual")).grounds[0].message

    for status in statuses:
        clause = _clause_for(message, status.extra)
        assert status.state in (EXTRA_ABSENT, EXTRA_INSTALLED, EXTRA_UNKNOWN)
        if status.state == EXTRA_ABSENT:
            assert COMMANDS[status.extra] in clause
        elif status.state == EXTRA_INSTALLED:
            assert str(status.version) in clause
            assert COMMANDS[status.extra] not in clause
        else:
            assert "could not be determined" in clause


def test_the_refusal_says_what_is_true_of_the_graph_and_claims_nothing_more(
    monkeypatch,
):
    """The fields beside the message, and the sentences it must not contain.

    `observed`, `expected`, `scope` and `summary` were read by nothing, so a
    refusal could swap the exact and residual blocks, or tell the caller the
    posterior task was refused too -- flatly contradicting the test one function
    down that proves it is not -- with the suite green.
    """
    from bayesmith.artifacts.refusal import ScopeKind
    from bayesmith.dispatch.plan import compile as compile_plan

    with jax.enable_x64(True):
        graph = _graph("b_mixture_prior_residual")
        plan = compile_plan(graph)
        refusal = _refuse(graph)

    exact = tuple(plan.exact.latents) if plan.exact is not None else ()
    sampled = tuple(plan.sampled.latents) if plan.sampled is not None else ()
    assert exact and sampled, "this fixture stopped being class (b)"

    ground = refusal.grounds[0]
    assert ground.code == "residual_backend_unavailable"
    assert ground.observed == (exact, sampled), (
        "the refusal reported the exact and residual blocks as something other "
        "than what the plan actually built"
    )
    assert ground.expected == "an installed residual-evidence backend"
    assert refusal.scope.kind is ScopeKind.BACKEND
    assert refusal.scope.name == "residual_evidence"
    assert "no backend" in refusal.meta.summary

    # The message names the residual block as the thing without a runner, and
    # the exact block as the thing that collapses -- not the other way round.
    message = ground.message
    assert f"integral over {list(sampled)}" in message
    assert f"{list(exact)} collapses exactly" in message
    assert "posterior task is unaffected" in message
    assert "posterior task is also refused" not in message


def test_the_refusal_carries_two_remedies_and_only_one_offers_the_extras():
    """The remedy row, asserted as a field rather than as prose in a heap.

    The first version concatenated `grounds[0].message` with every remedy's
    message and searched the heap, so **either** remedy could be deleted, the
    new one's parameters emptied, or its action renamed to the other's, all
    green -- the install commands appear in the finding's own message too.
    """
    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS

    with jax.enable_x64(True):
        refusal = _refuse(_graph("b_mixture_prior_residual"))

    actions = [remedy.action for remedy in refusal.remedies]
    assert len(actions) == len(set(actions)), f"two remedies, one action: {actions}"
    assert len(refusal.remedies) == 2, actions

    offering = [
        remedy
        for remedy in refusal.remedies
        if all(command in remedy.message for command in COMMANDS.values())
    ]
    assert len(offering) == 1, (
        f"expected exactly one remedy offering the extras, got {actions}"
    )
    parameters = dict(offering[0].parameters)
    assert parameters.get("extras") == tuple(
        extra for extra, _distribution in RESIDUAL_EVIDENCE_EXTRAS
    ), parameters

    # The other remedy is the premise's own, and it is still there: a caller
    # whose release cannot answer the task at all needs it.
    generic = [remedy for remedy in refusal.remedies if remedy not in offering]
    assert len(generic) == 1
    assert all(command not in generic[0].message for command in COMMANDS.values())


def test_a_refusal_that_is_not_about_the_residual_integral_offers_no_sampler():
    """The live wrong answer an adversarial review measured.

    `_refusal` reads `_REMEDIES[failed_premise]`, and `capability_unavailable_r1`
    has two consumers: a task kind this release does not answer, and a residual
    integral with no sampler. With the install remedy in the premise's row, a
    **simulation** refusal told the caller to install a nested sampler -- they
    would install something and come back to the identical refusal.

    Asserted twice: the premise's own row carries no install command, so every
    consumer that gets only the table row is safe; and the task-kind refusal
    built directly carries none either.
    """
    import bayesmith.dispatch.task as task_module
    from bayesmith.artifacts.identity import ArtifactKind
    from bayesmith.artifacts.tasks import TaskKind

    row = task_module._REMEDIES[CAPABILITY_UNAVAILABLE_R1]
    assert row, "the premise lost its remedy row"
    for remedy in row:
        for command in COMMANDS.values():
            assert command not in remedy.message, (
                f"the premise's own remedy row offers {command!r}, so every "
                f"refusal naming this premise offers it -- including the ones "
                f"that are not about the residual integral at all"
            )

    # The KIND is what `_capability_refusal` writes into its message and the
    # subject here is which remedies come back, so the task object only has to
    # be a valid one -- a `SimulationTask` needs a `ParameterSource` this test
    # has no opinion about.
    task = EvidenceTask(meta=new_task_meta(label="e"))
    with jax.enable_x64(True):
        graph = _graph("c_cauchy_residual_pair")
        bundle = task_module.input_fingerprints(graph, task, model_ref=_model_ref())
    refusal = task_module._capability_refusal(
        task, TaskKind.SIMULATION, bundle, ArtifactKind.PLAN
    )
    assert "simulation" in refusal.grounds[0].message
    assert refusal.failed_premise == CAPABILITY_UNAVAILABLE_R1
    assert refusal.grounds[0].code == "task_kind_unavailable"
    for remedy in refusal.remedies:
        for command in COMMANDS.values():
            assert command not in remedy.message, (
                "a task-kind refusal told the caller to install a nested sampler"
            )


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_a_posterior_task_over_the_same_graph_is_unaffected(name):
    """4.1's second half -- the one that separates a boundary from a regression.

    The same graph object, the same call, a different task kind. It also
    asserts the structural class each fixture sits in, because for two of the
    five nothing else in the suite reads their routing.
    """
    from bayesmith.dispatch.plan import compile as compile_plan

    with jax.enable_x64(True):
        graph = _graph(name)
        _refuse(graph)
        plan = compile_plan(graph)
        outcome = compile_task(
            graph, PosteriorTask(meta=new_task_meta(label="p")), model_ref=_model_ref()
        )
    assert not isinstance(outcome, Refusal), (
        "the capability refusal broke the neighbouring capability, which makes "
        "it a regression wearing a boundary's name"
    )

    exact = tuple(plan.exact.latents) if plan.exact is not None else ()
    sampled = tuple(plan.sampled.latents) if plan.sampled is not None else ()
    assert sampled, f"{name} has no residual block, so it is not (b) or (c)"
    expected_class = "b" if exact else "c"
    assert name.startswith(f"{expected_class}_"), (
        f"{name} compiles to class ({expected_class}): exact={exact} "
        f"sampled={sampled}. The key records the class, so it moves with it."
    )


def test_the_extras_table_is_exactly_what_pyproject_declares():
    """The enumeration, with its denominator stated (red line 16).

    **What this counts:** the keys of ``[project.optional-dependencies]`` in
    ``pyproject.toml``, all of them.

    **What it excludes.** The four ``[project].dependencies`` -- jax, equinox,
    numpy, numpyro -- are hard requirements; the pyproject comment says numpyro
    is "the last row of the dispatch table, not an optional extra".
    ``[dependency-groups]`` (``dev``, ``crosscheck``) are a different table:
    they never reach a wheel and ``pip install bayesmith[...]`` cannot ask for
    them.

    **Compared as an ordered sequence, not as a set.** A review added a
    duplicate row to the table and the set comparison passed, so the docstring's
    claim to hold the two lists equal "in both directions" was false about
    multiplicity. Duplicating a row makes the refusal name an extra twice.
    """
    import tomllib

    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS

    declared = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = declared["project"].get("optional-dependencies", {})
    named = [extra for extra, _distribution in RESIDUAL_EVIDENCE_EXTRAS]

    assert extras, "pyproject.toml declares no extras at all"
    assert len(named) == len(set(named)), f"the table names an extra twice: {named}"
    assert sorted(named) == sorted(extras), {
        "declared in pyproject.toml": sorted(extras),
        "named by the refusal": sorted(named),
    }
    assert len(named) == len(extras), {
        "the table has this many rows": len(named),
        "pyproject declares this many extras": len(extras),
    }

    distributions = [d for _extra, d in RESIDUAL_EVIDENCE_EXTRAS]
    assert len(distributions) == len(set(distributions)), distributions

    for extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
        assert _extra_provides(extras[extra], distribution), {
            "the extra": extra,
            "the probe looks for": _normalise(distribution),
            "the extra installs": sorted(
                _normalise(_requirement_name(req)) for req in extras[extra]
            ),
        }


@pytest.mark.parametrize(
    ("requirements", "distribution", "provides"),
    [
        # What the table declares today.
        (["blackjax>=1.6"], "blackjax", True),
        (["jaxns>=2.6"], "jaxns", True),
        # The near miss `startswith` admitted. The extra installs, the probe
        # still finds no `blackjax`, and the refusal reports it absent forever.
        (["blackjax-nightly>=1.6"], "blackjax", False),
        (["jaxns-nightly>=2.6"], "jaxns", False),
        (["jaxns2>=2.6"], "jaxns", False),
        # PEP 503: pip reads each pair as ONE project.
        (["BlackJAX>=1.6"], "blackjax", True),
        (["black_jax>=1.6"], "black-jax", True),
        (["black.jax"], "black-jax", True),
        # And it does NOT fold a separator into its absence: `Black_Jax` is
        # `black-jax`, a different project from `blackjax`. This row was
        # written the other way round and the parametrisation caught it.
        (["Black_Jax>=1.6"], "blackjax", False),
        # Shapes the parser must survive rather than mistake for a name.
        (["jaxns[plot] ; python_version<'3.13'"], "jaxns", True),
        (["blackjax == 1.6.2"], "blackjax", True),
        (["blackjax"], "blackjax", True),
        # Nothing the probe looks for.
        ([], "blackjax", False),
        (["optax"], "blackjax", False),
    ],
)
def test_an_extra_that_installs_a_near_miss_is_not_accepted(
    requirements, distribution, provides
):
    """The fixture the repository does not contain, so the guard has one.

    The real table has two well-formed rows, and on those the loose check
    (`req.startswith(distribution)`) and the parsed one agree -- restoring
    `startswith` leaves the suite green. These are the inputs on which they
    differ. Red line 13's fault (b), and (b) needs a fixture.
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
    """The present state, and the assertion is about what did NOT happen."""
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
        assert status.extra == "quiet", "the probe filled `extra` with something else"
        assert status.distribution == name
        assert name not in sys.modules, (
            "the probe imported the distribution it was asked about"
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()


def test_a_poisoning_distribution_is_reported_installed_and_never_executed(tmp_path):
    """The broken state, in-process, and the flag is SHOWN unchanged."""
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


def test_an_absent_distribution_whose_module_is_importable_is_not_imported(tmp_path):
    """The bypass that survived: importing on the way to answering ABSENT.

    A review inserted ``__import__(distribution)`` inside the
    ``PackageNotFoundError`` arm -- the answer stays correct in all three
    states, the versions stay right, and the module body runs anyway. Every
    fixture planted a ``.dist-info`` beside the module, so the absent arm never
    had a module to reach.

    Here the module exists with **no** metadata, so ABSENT is the answer and the
    module is the poisoning one: an import shows up in the flag.
    """
    from bayesmith.dispatch.task import EXTRA_ABSENT, optional_extra_status

    name = "bayesmith_probe_orphan"
    (tmp_path / f"{name}.py").write_text(_POISONING_MODULE, encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    before = jax.config.jax_enable_x64
    try:
        status = optional_extra_status("orphan", name)
        assert status.state == EXTRA_ABSENT
        assert name not in sys.modules, (
            "the probe imported a module on its way to reporting it absent"
        )
        assert jax.config.jax_enable_x64 == before, (
            "the probe executed the module and it moved jax.config"
        )
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop(name, None)
        importlib.invalidate_caches()
        jax.config.update("jax_enable_x64", before)


@pytest.mark.parametrize(
    "error",
    [
        OSError("the metadata backend could not be read"),
        RuntimeError("the metadata backend is not available"),
        ValueError("a dist-info directory could not be parsed"),
    ],
    ids=["OSError", "RuntimeError", "ValueError"],
)
def test_a_probe_that_could_not_look_says_so_rather_than_saying_absent(error):
    """Red line 14, as a value rather than as a promise, on more than one shape.

    ``absent`` and ``could not tell`` are different facts and a consumer acts
    differently on each. **One exception type is not the rule**: a review
    inserted an ``except ImportError`` arm returning ABSENT ahead of the generic
    handler, and because the only case exercised was ``OSError`` the whole suite
    stayed green while every other lookup failure started reporting an absence.
    """
    from bayesmith.dispatch.task import (
        EXTRA_ABSENT,
        EXTRA_UNKNOWN,
        optional_extra_status,
    )

    finder = _RaisingFinder(error=error)
    sys.meta_path.insert(0, finder)
    try:
        status = optional_extra_status("blocked", "bayesmith-no-such-distribution")
    finally:
        sys.meta_path.remove(finder)

    assert status.state == EXTRA_UNKNOWN
    assert status.state != EXTRA_ABSENT
    assert type(error).__name__ in status.detail, (
        "a declined lookup that does not say why is a second silence"
    )


def test_the_refusal_itself_says_the_lookup_declined_rather_than_saying_absent():
    """Red line 15: the same three states, asked at the layer a caller reads.

    **This test exists because its absence was measured.** With only the probe
    asserted directly, ``_extras_sentence``'s UNKNOWN branch was reachable by
    nothing: mutants folding it into the ABSENT text and into the INSTALLED text
    both survived the whole fast layer.

    The metadata backend is broken for exactly the residual-evidence
    distributions -- everything else still resolves, so `compile_task` runs.
    """
    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS

    subjects = {distribution for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS}
    finder = _RaisingFinder(subjects)
    sys.meta_path.insert(0, finder)
    try:
        with jax.enable_x64(True):
            message = _refuse(_graph("c_cauchy_residual_pair")).grounds[0].message
    finally:
        sys.meta_path.remove(finder)

    for extra, _distribution in RESIDUAL_EVIDENCE_EXTRAS:
        clause = _clause_for(message, extra)
        assert "could not be determined" in clause
        assert COMMANDS[extra] not in clause, (
            f"the refusal told a caller to install {extra} on the strength of "
            f"a lookup that never completed"
        )
    assert "OSError" in message


def test_the_naive_probe_poisons_a_fresh_process_and_this_one_does_not(tmp_path):
    """4.4 -- run in all three states, and the flag SHOWN rather than asserted.

    One subprocess, four measurements in order, so the poisoning is demonstrated
    on the same interpreter that had just been shown clean:

    1.  ``jax_enable_x64`` on a fresh process.
    2.  after ``residual_evidence_extras()`` -- this package's probe, over the
        real extras table, with a poisoning module installed under every one of
        those distribution names.
    3.  after the whole refusal path, which is what a consumer runs.
    4.  after ``try: import <candidate> except ImportError: pass`` -- the probe
        this package does not write.

    (4) is the control, and the control was itself checked: a review broke the
    planted module so it no longer wrote the flag, and this test went red on the
    right line.

    **The probe assertions carry a denominator**, computed in the parent. The
    first version compared against ``["installed"] * len(out["probe"])``, whose
    right side is built from its left, and asserted emptiness of a list that is
    empty when nothing was probed at all -- so with the probe returning nothing
    this test passed alone, exit 0.
    """
    from bayesmith.dispatch.task import RESIDUAL_EVIDENCE_EXTRAS

    expected = len(RESIDUAL_EVIDENCE_EXTRAS)
    assert expected >= 2, "the mixed case needs at least two extras"
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
            except ImportError:
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

    # (4) the control: the fake really is §0.16's hazard, and the exception it
    # raises is not the one the naive probe catches.
    assert set(out["naive"].values()) == {"AttributeError"}, out["naive"]
    assert len(out["naive"]) == expected
    assert out["after_naive"] is True, (
        "the fake did not poison the process, so this test grades nothing"
    )

    # (1)-(3) what this package does instead, against a denominator.
    assert out["start"] is False
    assert len(out["probe"]) == expected, (
        f"the probe answered about {len(out['probe'])} extras and the table "
        f"has {expected}"
    )
    assert out["probe"] == ["installed"] * expected
    assert out["imported_after_probe"] == []
    assert out["after_probe"] is False, "the probe moved jax.config"
    assert out["premise"] == CAPABILITY_UNAVAILABLE_R1
    assert out["after_refusal"] is False, "the refusal path moved jax.config"
