"""Canonical, whitelist-only encoding for artifact values.

Every fingerprint in this package is a SHA-256 over the bytes this module
produces, which makes those bytes -- not the Python objects -- the thing that
identity is actually made of. Two obligations follow, and they are the whole
design:

**One value must encode to one byte string.** So a mapping is written as its
key-sorted pairs rather than in insertion order; an array is written from its
C-contiguous bytes, so the C-order and Fortran-order spellings of a matrix and
any strided view of it agree; a float is written as :meth:`float.hex`, which is
exact, rather than as a decimal rendering, which is a choice of formatter; and
the non-finite doubles get one spelling each, because NaN carries a payload
the hardware picks and a digest that varied with it would make one value hash
two ways.

**Decoding must construct nothing it was not told about in advance.** The
decoder resolves a type name in a registry this package fills by explicit
:func:`register_artifact_type` calls. It never calls
``importlib.import_module()``, never evaluates a dotted path, and never
unpickles: a payload is untrusted input, and a decoder that imports what a
payload names is a remote-code-execution surface wearing a schema. The
whitelist is why object arrays, callables, classes, sets and lists are refused
outright rather than encoded on a best-effort basis -- plan §0 ruling 8, and
§0 ruling 4: an artifact is data, not a runtime object dump.

Refusals are :class:`ArtifactCodecError`, which is a ``ValueError`` so a caller
that already handles bad input keeps working, and the message names the type
or tag that was refused, because "cannot encode" without the offending type is
a bug report nobody can act on.

Layering: stdlib and NumPy only. This module is the bottom of the artifacts
ladder (``_codec ← identity ← base ← tasks ← results``) and importing anything
of bayesmith's from here would put a cycle under every artifact.
"""

from __future__ import annotations

import base64
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import tempfile
from enum import StrEnum
from typing import Literal, TypeVar

import numpy as np

__all__ = [
    "ArtifactCodecError",
    "register_artifact_type",
    "canonical_payload",
    "canonical_dumps",
    "canonical_loads",
    "ArtifactFile",
    "UnsupportedSchemaVersion",
    "dump_artifact",
    "load_artifact",
]

T = TypeVar("T")


class ArtifactCodecError(ValueError):
    """A value cannot be encoded, or a payload cannot be decoded, canonically."""


#: The key that carries a tagged value's kind. A single character, because it
#: appears once per encoded value and these payloads are hashed rather than
#: read; ``$`` cannot collide with a dataclass field name.
_TAG = "$"

#: How deep a value may nest before the codec calls it a cycle. Artifact
#: envelopes nest a handful of levels, so anything near this bound is either a
#: mistake or a dict that refers to itself -- and a ``RecursionError`` escaping
#: from here would be an untyped refusal that no caller can catch by contract.
_MAX_DEPTH = 100

#: The array dtypes an artifact may hold: boolean, integer, unsigned, float
#: and complex. Everything else -- object (arbitrary Python), void, strings,
#: datetimes and timedeltas -- is refused, because the first of those is the
#: whole hole the whitelist exists to close and the rest have no single
#: canonical byte form.
_ARRAY_KINDS = frozenset("biufc")

#: name -> class, and class -> name. Filled ONLY by register_artifact_type.
_BY_NAME: dict[str, type] = {}
_BY_CLASS: dict[type, str] = {}


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_frozen_dataclass(cls: type) -> bool:
    return (
        isinstance(cls, type)
        and dataclasses.is_dataclass(cls)
        and cls.__dataclass_params__.frozen
    )


def register_artifact_type(cls: type[T]) -> type[T]:
    """Admit ``cls`` to the codec's whitelist, and return it unchanged.

    Usable as a decorator, in which case it goes OUTSIDE
    ``@dataclass(frozen=True, slots=True)``: ``slots=True`` builds a new class
    object, so the inner decorator must run first or the registry would hold
    the discarded one.

    Only frozen dataclasses whose fields are all constructor arguments, and
    :class:`enum.StrEnum` subclasses, are admissible -- those are the two
    shapes this codec can rebuild from a payload exactly. A mutable dataclass
    is refused because an artifact that can be edited after its fingerprint
    was taken is not an identity; a field with ``init=False`` is refused
    because its value would be silently dropped on the way back.

    Registering the same class twice is a no-op, so a module that is imported
    twice does not fail. A DIFFERENT class claiming a name already registered
    is refused: that name is what a payload resolves through, so allowing the
    later registration to win would silently change what old payloads decode
    to.
    """
    if not isinstance(cls, type):
        raise ArtifactCodecError(
            f"only classes can be registered as artifact types; got {cls!r}"
        )

    if issubclass(cls, StrEnum):
        pass
    elif _is_frozen_dataclass(cls):
        loose = [f.name for f in dataclasses.fields(cls) if not f.init]
        if loose:
            raise ArtifactCodecError(
                f"{_qualified(cls)} has non-init field(s) {loose}, which a "
                "payload could not restore; make them init fields or keep "
                "them out of the artifact"
            )
    else:
        raise ArtifactCodecError(
            f"{_qualified(cls)} is neither a frozen dataclass nor a StrEnum, "
            "so this codec could not rebuild it from a payload"
        )

    name = _qualified(cls)
    existing = _BY_NAME.get(name)
    if existing is cls:
        return cls
    if existing is not None:
        raise ArtifactCodecError(
            f"{name} is already registered as {existing!r}; two classes cannot "
            "share one artifact type name"
        )
    _BY_NAME[name] = cls
    _BY_CLASS[cls] = name
    return cls


def _float_token(value: float) -> str:
    """``float.hex()``, with one spelling for each non-finite double.

    Measured in this checkout, and worth writing down because it makes the
    first two branches look removable: CPython's ``float.hex()`` ALREADY
    renders every NaN as ``'nan'`` -- sign bit and payload included -- and the
    infinities as ``'inf'``/``'-inf'``. So a mutant deleting them survives, and
    that is not a hole in the tests. They stay because the collapse is
    CPython's rendering rather than a documented promise, and this codec's
    guarantee is that one value hashes one way on every interpreter that runs
    it. The line that would be a duplicate is the one that isn't here: nothing
    calls ``ascontiguousarray`` before ``tobytes(order="C")`` either.
    """
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0.0 else "-inf"
    return value.hex()


def _float_from_token(token: object) -> float:
    if not isinstance(token, str):
        raise ArtifactCodecError(f"a float payload must carry a string; got {token!r}")
    if token in ("nan", "inf", "-inf"):
        return float(token)
    # Only the exact spelling float.hex() produces. float.fromhex() would also
    # accept "NaN", "Infinity" and decimal strings, which would give the three
    # non-finite values several encodings again.
    body = token.removeprefix("-")
    if not body.startswith("0x"):
        raise ArtifactCodecError(f"{token!r} is not a canonical float spelling")
    try:
        return float.fromhex(token)
    except ValueError as exc:
        raise ArtifactCodecError(f"{token!r} is not a canonical float spelling") from exc


def _encode_array(value: np.ndarray) -> dict[str, object]:
    if type(value) is not np.ndarray:
        raise ArtifactCodecError(
            f"{_qualified(type(value))} is an ndarray subclass; its extra state "
            "(a masked array's mask, for instance) would be dropped silently"
        )
    dtype = value.dtype
    if dtype.hasobject or dtype.kind not in _ARRAY_KINDS:
        raise ArtifactCodecError(
            f"an array of dtype {dtype.str!r} is not canonical; artifacts hold "
            "boolean, integer, float or complex arrays"
        )
    # C-order bytes: this ONE argument is where an F-order array, a transpose
    # and a strided view all become the same payload. Not preceded by an
    # ascontiguousarray() call: two ways of saying it would each survive a
    # mutation of the other, which is how a normalisation ends up with two
    # implementations and no test.
    return {
        _TAG: "ndarray",
        "dtype": dtype.str,
        "shape": list(value.shape),
        "data": base64.b64encode(value.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(payload: dict) -> np.ndarray:
    spec = payload.get("dtype")
    shape = payload.get("shape")
    data = payload.get("data")
    if not isinstance(spec, str) or not isinstance(data, str):
        raise ArtifactCodecError("an array payload needs a dtype and base64 data")
    if not isinstance(shape, list) or not all(
        isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape
    ):
        raise ArtifactCodecError(f"an array payload needs a shape; got {shape!r}")
    try:
        dtype = np.dtype(spec)
    except TypeError as exc:
        raise ArtifactCodecError(f"{spec!r} is not a dtype") from exc
    if dtype.hasobject or dtype.kind not in _ARRAY_KINDS:
        raise ArtifactCodecError(f"an array of dtype {spec!r} is not canonical")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise ArtifactCodecError("an array payload's data is not base64") from exc
    try:
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    except ValueError as exc:
        raise ArtifactCodecError(
            f"an array payload's {len(raw)} bytes do not fill shape {shape} "
            f"of dtype {spec!r}"
        ) from exc
    # Read-only, as §0.2 requires: a decoded artifact is not a handle on the
    # artifact, and np.frombuffer's own view is read-only anyway.
    array.setflags(write=False)
    return array


def _encode(value: object, depth: int) -> object:
    if depth > _MAX_DEPTH:
        raise ArtifactCodecError(
            f"value nests deeper than {_MAX_DEPTH} levels, or refers to itself"
        )
    kind = type(value)

    # Exact type checks, not isinstance: a subclass carries state this codec
    # cannot see, and a StrEnum member IS a str -- encoding it as one would
    # lose the type that a Result's tagged union is discriminated by.
    if value is None or kind is bool or kind is int or kind is str:
        return value
    if kind is float:
        return {_TAG: "float", "hex": _float_token(value)}
    if kind is bytes:
        return {_TAG: "bytes", "base64": base64.b64encode(value).decode("ascii")}
    if kind is tuple:
        return {_TAG: "tuple", "items": [_encode(item, depth + 1) for item in value]}
    if kind is dict:
        keys = list(value)
        if not all(type(key) is str for key in keys):
            raise ArtifactCodecError(
                "a canonical mapping is keyed by strings; got keys "
                f"{sorted(_qualified(type(k)) for k in keys)}"
            )
        return {
            _TAG: "mapping",
            "items": [[key, _encode(value[key], depth + 1)] for key in sorted(keys)],
        }
    if kind is _dt.datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ArtifactCodecError(
                f"{value!r} is naive; artifact times are UTC RFC 3339, and a "
                "naive time is a different instant on every machine"
            )
        moment = value.astimezone(_dt.UTC)
        return {_TAG: "datetime", "utc": moment.isoformat().replace("+00:00", "Z")}
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, np.generic):
        raise ArtifactCodecError(
            f"a NumPy scalar of dtype {value.dtype.str!r} is not canonical; use "
            "a Python scalar or a 0-d array, so that what was encoded is what "
            "comes back"
        )
    if isinstance(value, StrEnum):
        name = _BY_CLASS.get(kind)
        if name is None:
            raise ArtifactCodecError(
                f"{_qualified(kind)} is not a registered artifact type; "
                "decorate it with register_artifact_type"
            )
        return {_TAG: "enum", "type": name, "value": value.value}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        name = _BY_CLASS.get(kind)
        if name is None:
            raise ArtifactCodecError(
                f"{_qualified(kind)} is not a registered artifact type; "
                "decorate it with register_artifact_type"
            )
        return {
            _TAG: "dataclass",
            "type": name,
            # By field NAME, not declaration order: reordering a class's
            # fields is not a change of value, and a fingerprint that moved
            # when they were reordered would invalidate every artifact.
            "fields": [
                [field.name, _encode(getattr(value, field.name), depth + 1)]
                for field in sorted(dataclasses.fields(value), key=lambda f: f.name)
            ],
        }

    raise ArtifactCodecError(
        f"{_qualified(kind)} has no canonical encoding; artifacts hold None, "
        "bool, int, str, float, bytes, tuple, str-keyed dict, aware datetime, "
        "numeric ndarray, and registered frozen dataclasses and StrEnums"
    )


def _lookup(name: object, expect_enum: bool) -> type:
    """Resolve a type name THROUGH THE REGISTRY, and nowhere else."""
    if not isinstance(name, str):
        raise ArtifactCodecError(f"a type name must be a string; got {name!r}")
    cls = _BY_NAME.get(name)
    if cls is None:
        raise ArtifactCodecError(
            f"{name!r} is not a registered artifact type; this decoder resolves "
            "names in its registry and never imports one"
        )
    if expect_enum is not issubclass(cls, StrEnum):
        raise ArtifactCodecError(f"{name!r} is not encoded as the kind it names")
    return cls


def _decode(payload: object, depth: int) -> object:
    if depth > _MAX_DEPTH:
        raise ArtifactCodecError(f"payload nests deeper than {_MAX_DEPTH} levels")
    if payload is None or isinstance(payload, (bool, str)):
        return payload
    if isinstance(payload, int):
        return payload
    if isinstance(payload, float):
        raise ArtifactCodecError(
            f"{payload!r} is a bare JSON number with a fraction; canonical "
            "floats are tagged and carry float.hex()"
        )
    if not isinstance(payload, dict):
        raise ArtifactCodecError(
            f"{type(payload).__name__} is not a canonical payload; every "
            "compound value is a tagged object"
        )

    tag = payload.get(_TAG)
    if not isinstance(tag, str):
        raise ArtifactCodecError(f"payload carries no {_TAG!r} tag: {payload!r}")

    if tag == "float":
        return _float_from_token(payload.get("hex"))
    if tag == "bytes":
        data = payload.get("base64")
        if not isinstance(data, str):
            raise ArtifactCodecError("a bytes payload needs base64 data")
        try:
            return base64.b64decode(data, validate=True)
        except (ValueError, TypeError) as exc:
            raise ArtifactCodecError("a bytes payload is not base64") from exc
    if tag == "tuple":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ArtifactCodecError("a tuple payload needs a list of items")
        return tuple(_decode(item, depth + 1) for item in items)
    if tag == "mapping":
        items = payload.get("items")
        if not isinstance(items, list):
            raise ArtifactCodecError("a mapping payload needs a list of pairs")
        decoded: dict[str, object] = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ArtifactCodecError(f"{pair!r} is not a (key, value) pair")
            key, encoded = pair
            if not isinstance(key, str):
                raise ArtifactCodecError(f"a mapping key must be a string; got {key!r}")
            decoded[key] = _decode(encoded, depth + 1)
        return decoded
    if tag == "datetime":
        text = payload.get("utc")
        if not isinstance(text, str):
            raise ArtifactCodecError("a datetime payload needs an RFC 3339 string")
        try:
            moment = _dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ArtifactCodecError(f"{text!r} is not RFC 3339") from exc
        if moment.tzinfo is None:
            raise ArtifactCodecError(f"{text!r} carries no offset")
        return moment.astimezone(_dt.UTC)
    if tag == "ndarray":
        return _decode_array(payload)
    if tag == "enum":
        cls = _lookup(payload.get("type"), expect_enum=True)
        member = payload.get("value")
        if not isinstance(member, str):
            raise ArtifactCodecError("an enum payload needs a string value")
        try:
            return cls(member)
        except ValueError as exc:
            raise ArtifactCodecError(
                f"{member!r} is not a member of {payload.get('type')!r}"
            ) from exc
    if tag == "dataclass":
        cls = _lookup(payload.get("type"), expect_enum=False)
        fields = payload.get("fields")
        if not isinstance(fields, list):
            raise ArtifactCodecError("a dataclass payload needs a list of fields")
        arguments: dict[str, object] = {}
        for pair in fields:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ArtifactCodecError(f"{pair!r} is not a (field, value) pair")
            name, encoded = pair
            if not isinstance(name, str):
                raise ArtifactCodecError(f"a field name must be a string; got {name!r}")
            arguments[name] = _decode(encoded, depth + 1)
        expected = {field.name for field in dataclasses.fields(cls)}
        if set(arguments) != expected:
            raise ArtifactCodecError(
                f"payload for {payload.get('type')} names fields "
                f"{sorted(arguments)}, but the registered class has "
                f"{sorted(expected)}"
            )
        try:
            return cls(**arguments)
        except (TypeError, ValueError) as exc:
            raise ArtifactCodecError(
                f"{payload.get('type')} refused the payload's fields: {exc}"
            ) from exc

    raise ArtifactCodecError(
        f"unknown canonical tag {tag!r}; this decoder knows only the tags it "
        "writes, and constructs nothing a payload names on its own"
    )


def canonical_payload(value: object) -> object:
    """The JSON-safe canonical form of ``value``.

    Separate from :func:`canonical_dumps` because a fingerprint is taken over
    the bytes but a Result may want to embed the form itself, and because a
    payload that :func:`json.dumps` would refuse should fail here, where the
    offending type is still in hand and can be named.
    """
    return _encode(value, 0)


def canonical_dumps(value: object) -> bytes:
    """The canonical UTF-8 bytes of ``value``: what a fingerprint hashes.

    ``sort_keys`` and the tight separators are what make the bytes a function
    of the value alone; ``ensure_ascii`` keeps them a function of the value
    rather than of the reader's encoding; ``allow_nan=False`` is a belt on
    braces, since no float ever reaches the JSON writer -- if one did, it would
    emit ``NaN``, which is not JSON and which no strict parser elsewhere would
    read back.
    """
    return json.dumps(
        canonical_payload(value),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_loads(payload: bytes, *, expected: type[T] | None = None) -> T | object:
    """Rebuild the value :func:`canonical_dumps` encoded.

    ``expected`` is checked after decoding and refuses a mismatch, so a caller
    reading a stored artifact does not have to trust the file to hold what its
    name says.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ArtifactCodecError(
            f"canonical_loads reads bytes; got {type(payload).__name__}"
        )
    try:
        text = bytes(payload).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactCodecError("payload is not UTF-8") from exc
    try:
        parsed = json.loads(text, parse_constant=_refuse_constant)
    except json.JSONDecodeError as exc:
        raise ArtifactCodecError(f"payload is not JSON: {exc}") from exc

    value = _decode(parsed, 0)
    if expected is not None and not isinstance(value, expected):
        raise ArtifactCodecError(
            f"payload decoded to {_qualified(type(value))}, not "
            f"{_qualified(expected)}"
        )
    return value


def _refuse_constant(token: str) -> object:
    """Python's json reads ``NaN``/``Infinity``; JSON does not, and neither
    does this codec -- its own floats are tagged."""
    raise ArtifactCodecError(f"{token!r} is not JSON and not a canonical float")


# ---------------------------------------------------------------- persistence

#: The disk format's fixed marker. Read by :func:`load_artifact` to refuse
#: anything that is not a bayesmith artifact envelope rather than decode it
#: and discover the same thing three fields deep.
_ARTIFACT_FORMAT = "bayesmith-artifact"

#: The one codec version this package writes and reads. Bumped only when a
#: stored artifact's shape changes in a way a reader cannot ignore -- never
#: guessed at, because a migration guessed wrong reads as silently dropped
#: fields, and there is no slower way to find that out.
_ARTIFACT_CODEC_VERSION = 1


class UnsupportedSchemaVersion(ArtifactCodecError):
    """A stored artifact was written in a codec version this package cannot read.

    Raised when the envelope names a version NEWER than the one this package
    writes. The caller is expected to upgrade the package rather than guess at
    a migration, so no attempt is made to read the payload.
    """


@dataclasses.dataclass(frozen=True, slots=True)
class ArtifactFile:
    """The canonical JSON transport envelope (§8.2).

    Not the artifact, and not a second copy of its digest the artifact
    computes for itself: ``payload_sha256`` is the SHA-256 of the DECODED
    payload bytes -- exactly what ``canonical_dumps`` produced -- and
    ``payload_base64`` is those bytes under base64. ``load_artifact`` checks
    the digest before it decodes, so a corrupted payload cannot hand back a
    corrupted object with a freshly matching digest.
    """

    format: Literal["bayesmith-artifact"]
    codec_version: Literal[1]
    payload_sha256: str
    payload_base64: str


def _envelope_bytes(artifact: object) -> bytes:
    """The disk bytes of one artifact: a JSON envelope, atomic by construction."""
    payload = canonical_dumps(artifact)
    envelope = ArtifactFile(
        format=_ARTIFACT_FORMAT,
        codec_version=_ARTIFACT_CODEC_VERSION,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_base64=base64.b64encode(payload).decode("ascii"),
    )
    return json.dumps(
        dataclasses.asdict(envelope),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def dump_artifact(artifact: object, path: str | os.PathLike[str]) -> None:
    """Write ``artifact`` to ``path`` as a canonical transport envelope.

    The write is atomic at the filesystem level: the bytes are written to a
    temporary file in the same directory and moved into place with
    ``os.replace``, so a reader -- including one racing this write -- sees
    either the previous complete file or the new complete file, never a
    half-written one. A failure to serialise the artifact raises before any
    file is touched, because ``canonical_dumps`` runs first.
    """
    text = _envelope_bytes(artifact) + b"\n"
    target = os.fspath(path)
    directory = os.path.dirname(target) or "."
    fd, temporary = tempfile.mkstemp(
        dir=directory, prefix=".bayesmith-artifact-", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_artifact(
    path: str | os.PathLike[str], *, expected: type[T] | None = None
) -> T | object:
    """Read an artifact written by :func:`dump_artifact`, verifying it first.

    Three checks, in the order that closes the most holes first: the envelope
    is what it claims to be and a version this package can read; the decoded
    payload hashes to the digest the envelope states; and the payload decodes
    to the ``expected`` type. A payload byte flipped anywhere, a digest byte
    flipped, or a type name the registry does not know all fail here before
    any object is handed back.
    """
    target = os.fspath(path)
    with open(target, "rb") as handle:
        raw = handle.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactCodecError(f"{target} is not UTF-8") from exc
    try:
        parsed = json.loads(text, parse_constant=_refuse_constant)
    except json.JSONDecodeError as exc:
        raise ArtifactCodecError(f"{target} is not a JSON envelope: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ArtifactCodecError(
            f"{target} is not an artifact envelope: {type(parsed).__name__}"
        )
    if parsed.get("format") != _ARTIFACT_FORMAT:
        raise ArtifactCodecError(
            f"{target} is not a {_ARTIFACT_FORMAT!r} envelope; its format is "
            f"{parsed.get('format')!r}"
        )
    version = parsed.get("codec_version")
    if version != _ARTIFACT_CODEC_VERSION:
        if isinstance(version, int) and version > _ARTIFACT_CODEC_VERSION:
            raise UnsupportedSchemaVersion(
                f"{target} was written in codec version {version}; this package "
                f"reads version {_ARTIFACT_CODEC_VERSION} and does not guess a "
                "migration"
            )
        raise ArtifactCodecError(
            f"{target} carries codec_version {version!r}, which is not the "
            f"{_ARTIFACT_CODEC_VERSION} this package reads"
        )

    digest = parsed.get("payload_sha256")
    encoded = parsed.get("payload_base64")
    if not isinstance(digest, str) or not isinstance(encoded, str):
        raise ArtifactCodecError(
            f"{target} is missing its payload_sha256 or payload_base64"
        )
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ArtifactCodecError(f"{target}'s payload_base64 is not base64") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != digest:
        raise ArtifactCodecError(
            f"{target}'s payload hashes to {actual}, not the stated {digest}; "
            "the bytes do not match the envelope that claims them"
        )
    return canonical_loads(payload, expected=expected)
