"""The canonical codec, pinned at its refusals as much as at its round-trips.

A fingerprint is only worth its bytes if two encodings of one value are the
same bytes, so what is checked here are the properties a digest depends on
rather than a sample of values that happen to survive a round trip:

* **the same value twice is the same bytes**, and re-encoding a decoded value
  reproduces them -- a digest taken over an encoding that drifts is a digest
  of the encoder's mood;
* **mapping order does not reach the wire**, because a Python dict remembers
  its insertion order and nothing about a model's identity should;
* **an array's memory layout does not reach the wire either**: the C-order and
  Fortran-order spellings of one matrix are one artifact, and a slice's stride
  is not part of what was computed;
* **floats travel as `float.hex()`**, so the decoder gets back the double that
  was encoded rather than the nearest one to a decimal rendering, and the three
  non-finite values have one spelling each rather than one per NaN payload the
  hardware happened to produce.

The refusals are tested at the same weight as the round trips, because the
whitelist is the property being bought (plan §0 ruling 8): a codec that
constructs whatever a payload names is an ``importlib.import_module()`` with
extra steps. The load-bearing case is ``Unregistered`` -- a frozen dataclass of
exactly ``Registered``'s shape, defined in this very module and therefore
trivially importable by name -- which must be refused in both directions. If
that test ever passes because the decoder found the class, the decoder is
importing.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import struct
from enum import StrEnum

import numpy as np
import pytest

from bayesmith.artifacts._codec import (
    ArtifactCodecError,
    canonical_dumps,
    canonical_loads,
    canonical_payload,
    register_artifact_type,
)

#: The on-wire discriminator. Written out here rather than imported, so that
#: changing the wire format has to be done twice, deliberately, in two files.
TAG = "$"


@register_artifact_type
class Flavour(StrEnum):
    EXACT = "exact"
    APPROXIMATE = "approximate"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class Registered:
    name: str
    weight: float
    tags: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class Unregistered:
    """``Registered``'s twin in everything but the registration."""

    name: str
    weight: float
    tags: tuple[str, ...]


class UnregisteredFlavour(StrEnum):
    """The same twin for the other admissible shape.

    Its own test exists because a mutation run found the hole: the encoder
    checks the registry once per shape, and a test that only ever encoded an
    unregistered DATACLASS left the enum branch's check free to be deleted.
    """

    EXACT = "exact"


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class Envelope:
    label: str
    fidelity: Flavour
    created_at: dt.datetime
    options: tuple[tuple[str, int], ...]
    inner: Registered


def fingerprint(value: object) -> str:
    """What Task 2 will call a fingerprint: SHA-256 over the canonical bytes."""
    return hashlib.sha256(canonical_dumps(value)).hexdigest()


def sample() -> Envelope:
    return Envelope(
        label="pilot",
        fidelity=Flavour.APPROXIMATE,
        created_at=dt.datetime(2026, 8, 30, 12, 34, 56, 789012, tzinfo=dt.UTC),
        options=(("maxiter", 200), ("chains", 4)),
        inner=Registered(name="theta", weight=1.5, tags=("x", "y")),
    )


def _no_constants(token: str) -> object:
    raise AssertionError(f"payload carried the non-JSON constant {token!r}")


def test_registered_dataclass_enum_tuple_and_mapping_round_trip():
    value = sample()
    assert canonical_loads(canonical_dumps(value)) == value

    assert canonical_loads(canonical_dumps(Flavour.EXACT)) is Flavour.EXACT
    assert canonical_loads(canonical_dumps(("a", 1, True, None))) == (
        "a",
        1,
        True,
        None,
    )
    assert canonical_loads(canonical_dumps({"b": 2, "a": (1, 2)})) == {
        "a": (1, 2),
        "b": 2,
    }
    assert canonical_loads(canonical_dumps(b"\x00\xff")) == b"\x00\xff"

    # A tuple is not a list on the way back: the collections this layer freezes
    # are the immutable ones, and a round trip returning a list would hand the
    # caller a mutable view of an artifact.
    assert isinstance(canonical_loads(canonical_dumps(("a",))), tuple)


def test_canonical_dumps_is_byte_stable_and_mapping_order_independent():
    value = sample()
    assert canonical_dumps(value) == canonical_dumps(value)
    assert canonical_dumps(canonical_loads(canonical_dumps(value))) == canonical_dumps(
        value
    )

    first = {"a": 1, "b": (2, 3), "c": "z"}
    second = {"c": "z", "b": (2, 3), "a": 1}
    assert list(first) != list(second), "the fixture must differ in insertion order"
    assert canonical_dumps(first) == canonical_dumps(second)
    assert fingerprint(first) == fingerprint(second)

    # ... and two values that genuinely differ do not collide.
    assert fingerprint(first) != fingerprint({"a": 1, "b": (2, 3), "c": "y"})


def test_canonical_payload_is_json_safe_on_its_own():
    """The seam a fingerprint hashes, JSON-safe before dumps touches it."""
    payload = canonical_payload(sample())
    json.dumps(payload, allow_nan=False)  # must not raise
    assert isinstance(payload, dict)


def test_utc_datetimes_round_trip_and_naive_ones_are_refused():
    moment = dt.datetime(2026, 8, 30, 12, 34, 56, 789012, tzinfo=dt.UTC)
    assert canonical_loads(canonical_dumps(moment)) == moment
    assert b"Z" in canonical_dumps(moment)

    # The same instant written in another zone is the same artifact.
    shifted = moment.astimezone(dt.timezone(dt.timedelta(hours=2)))
    assert shifted.utcoffset() != dt.timedelta(0), "the fixture must be off-UTC"
    assert canonical_dumps(shifted) == canonical_dumps(moment)

    with pytest.raises(ArtifactCodecError):
        canonical_dumps(dt.datetime(2026, 8, 30, 12, 34, 56))  # noqa: DTZ001
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(dt.date(2026, 8, 30))


def test_floats_travel_as_hex_rather_than_as_a_decimal_rendering():
    for value in (0.0, -0.0, 1.5, math.pi, 1e-300, -2.5e300, 0.1 + 0.2):
        back = canonical_loads(canonical_dumps(value))
        assert isinstance(back, float)
        assert back == value
        assert math.copysign(1.0, back) == math.copysign(1.0, value)

    raw = canonical_dumps(0.1 + 0.2)
    assert (0.1 + 0.2).hex().encode() in raw
    assert b"0.30000000000000004" not in raw


def test_nan_and_infinities_are_normalised_to_one_spelling_each():
    quiet = float("nan")
    # NaNs with a different payload and with the sign bit set -- the hardware
    # picks both, so an encoding carrying either would make one value hash
    # several ways.
    others = [
        struct.unpack("<d", struct.pack("<Q", bits))[0]
        for bits in (0x7FF8000000000001, 0xFFF8000000000000, 0xFFF8000000000001)
    ]
    assert all(math.isnan(other) for other in others)
    assert math.copysign(1.0, others[1]) == -1.0, "the fixture must be a signed NaN"
    for other in others:
        assert canonical_dumps(quiet) == canonical_dumps(other)
    assert math.isnan(canonical_loads(canonical_dumps(quiet)))

    assert canonical_loads(canonical_dumps(math.inf)) == math.inf
    assert canonical_loads(canonical_dumps(-math.inf)) == -math.inf

    # And none of the three leaves behind a token only Python's json reads:
    # NaN and Infinity are not JSON, so a strict parser elsewhere would reject
    # a payload carrying them.
    encoded = canonical_dumps((quiet, math.inf, -math.inf))
    assert b"NaN" not in encoded
    assert b"Infinity" not in encoded
    json.loads(encoded.decode("utf-8"), parse_constant=_no_constants)


def test_array_round_trip_preserves_dtype_shape_and_values():
    arrays = (
        np.arange(6, dtype=np.int32).reshape(2, 3),
        np.linspace(0.0, 1.0, 5, dtype=np.float64),
        np.array([[True, False]]),
        np.array([1 + 2j, -3j], dtype=np.complex128),
        np.zeros((0, 3), dtype=np.float32),
        np.array(2.5, dtype=np.float64),  # 0-d
        np.array([1.0, np.nan, np.inf], dtype=np.float64),
    )
    for array in arrays:
        back = canonical_loads(canonical_dumps(array))
        assert isinstance(back, np.ndarray)
        assert back.dtype == array.dtype, array.dtype
        assert back.shape == array.shape, array.shape
        assert np.array_equal(back, array, equal_nan=array.dtype.kind == "f")

    # Arrays compose: one inside a tuple travels the same way.
    (inside,) = canonical_loads(canonical_dumps((arrays[0],)))
    assert np.array_equal(inside, arrays[0])


def test_a_decoded_array_is_read_only():
    """An artifact's array is not a handle on the artifact.

    §0.2 freezes arrays as read-only C-contiguous copies, so that a caller
    cannot edit an artifact after its fingerprint was taken.
    """
    back = canonical_loads(canonical_dumps(np.arange(3, dtype=np.int64)))
    assert not back.flags.writeable
    with pytest.raises(ValueError):
        back[0] = 7


def test_c_and_f_order_arrays_share_one_fingerprint():
    c_order = np.arange(6, dtype=np.float64).reshape(2, 3)
    f_order = np.asfortranarray(c_order)
    assert not f_order.flags.c_contiguous, "the fixture must not already be C-order"
    assert canonical_dumps(c_order) == canonical_dumps(f_order)
    assert fingerprint(c_order) == fingerprint(f_order)

    # A strided view is normalised too, rather than encoding its base buffer.
    view = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    assert not view.flags.c_contiguous
    assert canonical_dumps(view) == canonical_dumps(np.ascontiguousarray(view))

    # Shape is part of identity even where the bytes are not ...
    assert fingerprint(c_order) != fingerprint(c_order.reshape(3, 2))
    # ... and so is dtype.
    assert fingerprint(np.ones(2, dtype=np.float32)) != fingerprint(
        np.ones(2, dtype=np.float64)
    )


def test_decoder_refuses_unregistered_type_and_object_array():
    twin = Unregistered(name="theta", weight=1.5, tags=("x", "y"))
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(twin)
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(UnregisteredFlavour.EXACT)

    # The registered twin of the same shape goes through, so the refusal is
    # about the registration and not about the shape.
    registered = Registered(name="theta", weight=1.5, tags=("x", "y"))
    assert canonical_loads(canonical_dumps(registered)) == registered

    with pytest.raises(ArtifactCodecError):
        canonical_dumps(np.array([object()], dtype=object))
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(np.array([registered], dtype=object))
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(np.array(["a", "b"]))  # dtype '<U1' is not numeric


def test_the_decoder_reads_its_registry_and_never_an_import():
    """The refusal that would be silent if the decoder imported by name.

    Unregistered lives in this module, so importlib.import_module plus getattr
    would find it and construct it happily. Only a decoder consulting its own
    registry can refuse it.
    """
    payload = json.loads(canonical_dumps(Registered("theta", 1.5, ())).decode())
    assert isinstance(payload, dict)
    key = next(k for k, v in payload.items() if v == f"{__name__}.Registered")
    payload[key] = f"{__name__}.Unregistered"

    with pytest.raises(ArtifactCodecError):
        canonical_loads(json.dumps(payload).encode("utf-8"))


def test_the_decoder_refuses_an_unknown_tag():
    for tagged in (
        {TAG: "importable", "module": "os", "name": "system"},
        {TAG: "pickle", "payload": ""},
        {TAG: 17},
        {"no": "tag"},
    ):
        with pytest.raises(ArtifactCodecError):
            canonical_loads(json.dumps(tagged).encode("utf-8"))

    for not_a_value in (b"not json at all", b"[1, 2]", b"1.5"):
        with pytest.raises(ArtifactCodecError):
            canonical_loads(not_a_value)


def test_callables_lists_and_loose_objects_are_refused():
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(len)
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(lambda: None)
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(Registered)  # the class, not an instance
    with pytest.raises(ArtifactCodecError):
        canonical_dumps([1, 2])  # a list is mutable; artifacts use tuples
    with pytest.raises(ArtifactCodecError):
        canonical_dumps({"a", "b"})  # a set has no canonical order
    with pytest.raises(ArtifactCodecError):
        canonical_dumps({1: "a"})  # a mapping key that is not a string
    with pytest.raises(ArtifactCodecError):
        canonical_dumps(object())


def test_expected_is_checked_on_load():
    raw = canonical_dumps(Registered("theta", 1.5, ()))
    assert canonical_loads(raw, expected=Registered) == Registered("theta", 1.5, ())
    with pytest.raises(ArtifactCodecError):
        canonical_loads(raw, expected=Flavour)


def test_registration_refuses_what_it_could_not_reconstruct():
    with pytest.raises(ArtifactCodecError):

        @register_artifact_type
        @dataclasses.dataclass
        class Mutable:
            x: int

    with pytest.raises(ArtifactCodecError):

        @register_artifact_type
        class Plain:
            pass

    with pytest.raises(ArtifactCodecError):
        register_artifact_type(sample)


def test_registering_the_same_class_twice_is_idempotent_and_a_clash_is_not():
    assert register_artifact_type(Registered) is Registered

    # A DIFFERENT class claiming the registered name. Built rather than
    # declared, because a class statement inside this function would take its
    # qualname from the function and so could never collide -- which is
    # exactly how this guard would end up passing while checking nothing.
    clash = dataclasses.make_dataclass(
        "Registered", [("name", str)], frozen=True, slots=True
    )
    clash.__module__ = __name__
    clash.__qualname__ = "Registered"
    with pytest.raises(ArtifactCodecError):
        register_artifact_type(clash)
