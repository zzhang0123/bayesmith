"""A refusal a caller must act on carries its evidence as data (G11).

Three classes changed shape: :class:`AffinityRefused` is new and carries the
probe's numbers, and the two blameless verdicts -- :class:`NotGaussian`,
:class:`NotLogLinear` -- now name their reason in a field instead of only in
prose.

What the payloads are FOR is worth stating, because it is not decoration. The
adapter being built in the sibling repository has to turn one package's
refusal into the other's, with the same numbers attached; the only other way
to get them across is to parse the message, and a message is not an interface.
The rheplicant class it translates into (``LinearityRefused``) already carries
``errors``/``rtol``/``failed`` for exactly this reason, and its docstring
records the defect that produced them: the failing branch rendered the numbers
into a sentence and dropped them while the PASSING branch returned them
structured, so the only path with something to report was the only path with
nothing readable.

The end-to-end assertions -- that a real refusal from a real model carries the
right reason -- live beside the models that produce them, in
``tests/exact/test_linearity.py`` and ``tests/exact/test_loglinear.py``. What
is here is the part that is about the classes themselves.
"""

import ast
import pickle  # in-process round-trip of objects built here; no external data
from pathlib import Path

import pytest

from bayesmith.errors import (
    NOT_GAUSSIAN_REASONS,
    NOT_LOG_LINEAR_REASONS,
    AffinityRefused,
    NotGaussian,
    NotLogLinear,
    StructureError,
)

SRC = Path(__file__).resolve().parents[1] / "src"

#: Which keyword each payload class must be given at every raise site, and the
#: vocabulary it is checked against where it has one. Derived from, not a
#: second spelling of, the classes' own sets.
REQUIRED = {
    "NotGaussian": ("reason", NOT_GAUSSIAN_REASONS),
    "NotLogLinear": ("reason", NOT_LOG_LINEAR_REASONS),
    "AffinityRefused": ("names", None),
}


def _payload_raises() -> list[tuple[Path, int, str, set[str], str | None]]:
    """Every ``raise <payload class>(...)`` under ``src/``, with its keywords.

    An AST walk rather than a grep: a raise spans several lines here (the
    messages are long), so a line-oriented scan would see the class name and
    none of the keywords that follow it.

    Returns the literal value of the required keyword where it IS a literal,
    which is most sites; a site computing it -- ``reason=error.reason``, or a
    conditional -- reports ``None`` and is checked only for presence. Pinning
    a computed reason would mean re-implementing the branch in the test, and a
    test that re-implements the code cannot disagree with it.
    """
    found = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)):
                continue
            func = node.exc.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name not in REQUIRED:
                continue
            keywords = {k.arg for k in node.exc.keywords if k.arg}
            literal = None
            for keyword in node.exc.keywords:
                if keyword.arg == REQUIRED[name][0] and isinstance(
                    keyword.value, ast.Constant
                ):
                    literal = keyword.value.value
            found.append((path, node.lineno, name, keywords, literal))
    return found


def test_the_scan_still_finds_raise_sites():
    """The guard below is only worth its run if it scanned something.

    Without this, moving the classes, renaming the package directory, or an
    `ast.parse` that quietly returned nothing would leave the next test
    iterating an empty list and passing -- the failure mode this repository's
    working notes call a guard that has stopped being able to fail.
    """
    sites = _payload_raises()
    assert len(sites) >= 10, sites
    assert {name for _, _, name, _, _ in sites} == set(REQUIRED)


def test_every_raise_site_names_its_payload():
    """A discriminator that MAY be omitted is one that will be.

    This is what makes ``reason`` required rather than optional: the field
    exists so a dispatcher can branch without reading prose, and one site that
    leaves it off puts the prose fallback back in the consumer for good.
    """
    missing = [
        f"{path.name}:{line} {name} has {sorted(keywords)}"
        for path, line, name, keywords, _ in _payload_raises()
        if REQUIRED[name][0] not in keywords
    ]
    assert not missing, missing


def test_every_literal_reason_is_in_its_vocabulary():
    """A typo'd reason would route as "some other reason", silently.

    The constructor refuses one, so this could only fail on a site that never
    runs in the suite -- which is exactly the site worth checking statically.
    """
    wrong = [
        f"{path.name}:{line} {name} reason={literal!r}"
        for path, line, name, _, literal in _payload_raises()
        if literal is not None and (vocab := REQUIRED[name][1]) and literal not in vocab
    ]
    assert not wrong, wrong


def test_an_unknown_reason_is_refused_where_it_is_written():
    with pytest.raises(ValueError, match="unknown NotGaussian reason"):
        NotGaussian("x", reason="banana")
    with pytest.raises(ValueError, match="unknown NotLogLinear reason"):
        NotLogLinear("x", reason="banana")


def test_the_reason_is_required_and_not_defaulted():
    """No default, because any default would be wrong for most sites."""
    with pytest.raises(TypeError):
        NotGaussian("x")
    with pytest.raises(TypeError):
        NotLogLinear("x")


def test_affinity_refused_still_answers_to_the_structural_family():
    """Every ``except StructureError`` already written keeps working.

    Behaviour, not ``issubclass``: the class statement restated as an
    assertion would pass whatever the catch semantics turned out to be.
    """
    caught = False
    try:
        raise AffinityRefused(
            "not affine",
            names=("w",),
            at="{}",
            errors={1.0: 0.5},
            weighted={1.0: 0.4},
            rtol=1e-6,
            weighted_rtol=1e-3,
            failed=(1.0,),
        )
    except StructureError as refused:
        caught = True
        assert refused.failed == (1.0,)
    assert caught


def test_catching_the_blameless_verdicts_still_misses_affinity_refused():
    """The narrow catch is the point of the new class.

    ``AffinityRefused`` means "you declared it and it is false". A dispatcher
    writing ``except NotGaussian`` to mean "no exact structure here, use
    NUTS" must not swallow it, or a broken model is routed as an ordinary one.
    """
    with pytest.raises(AffinityRefused):
        try:
            raise AffinityRefused(
                "not affine",
                names=("w",),
                at="{}",
                errors={},
                weighted={},
                rtol=1e-6,
                weighted_rtol=1e-3,
                failed=(),
            )
        except (NotGaussian, NotLogLinear):  # pragma: no cover - must not be taken
            pass


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: NotLogLinear(
                "too high", reason="fractional_too_large", node="d", fractional=0.3
            ),
            id="not-log-linear",
        ),
        pytest.param(
            lambda: NotGaussian("a Gamma", reason="not_normal", node="w", found="Gamma"),
            id="not-gaussian",
        ),
        pytest.param(
            lambda: AffinityRefused(
                "not affine",
                names=("w", "b"),
                at="{'w': 1.0}",
                errors={0.1: 1e-9, 1.0: 0.5},
                weighted={0.1: 1e-9, 1.0: 0.4},
                rtol=1e-6,
                weighted_rtol=1e-3,
                failed=(1.0,),
            ),
            id="affinity-refused",
        ),
    ],
)
def test_the_payload_survives_a_round_trip(build):
    """Pickling an exception is not exotic: pytest-xdist does it.

    Python's default reconstruction calls ``cls(*args)``, which cannot work
    once construction takes required keywords -- and the failure surfaces far
    from here, as a TypeError about missing arguments where the real error was
    supposed to be reported. ``__reduce__`` is what closes it, and an untested
    ``__reduce__`` is the kind that is wrong.
    """
    original = build()
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is type(original)
    assert str(restored) == str(original)
    assert restored.__dict__ == original.__dict__


def test_the_payload_does_not_alias_the_callers_mapping():
    """A caught exception sharing mutable state with its raiser is a trap.

    The probe hands over the same dict it returns on the passing branch, so an
    alias here would let a consumer's edit reach back into the prober.
    """
    errors = {1.0: 0.5}
    refused = AffinityRefused(
        "not affine",
        names=("w",),
        at="{}",
        errors=errors,
        weighted={1.0: 0.4},
        rtol=1e-6,
        weighted_rtol=1e-3,
        failed=(1.0,),
    )
    errors[2.0] = 9.9
    assert refused.errors == {1.0: 0.5}
