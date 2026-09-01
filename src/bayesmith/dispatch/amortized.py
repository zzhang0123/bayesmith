"""Encode a trained NeuralPosterior as an ESTIMATOR artifact, and back.

R2 Task 7, and the one place R2 reaches into R1's frozen schema.  A trained
:class:`bayesmith.amortize.NeuralPosterior` is an eqx.Module plus a static
embed callable plus standardization arrays, and none of that has a canonical
artifact form -- a callable has no canonical form (§0 ruling 4), and an
eqx.Module is a runtime object.  It is therefore encoded as the estimator arm
of :class:`bayesmith.artifacts.results.FittedConditionalPosterior`: a
REFERENCE to an artifact of the new :class:`ArtifactKind.ESTIMATOR`, never the
object itself (§0.7).

The estimator artifact is a frozen dataclass holding an opaque
`blob` -- the bytes `equinox.tree_serialise_leaves` writes for the MLP's
parameter leaves -- and a `manifest` -- the static facts a deserializer needs
to rebuild the estimator: the embed callable's importable identity, the mixture
shape, the minimum scale, the MLP's own structure, and the standardization
arrays taken from the training bank.  The blob is `bytes` to the codec, so
the artifacts layer never imports JAX or equinox; this module is where that
stack is allowed, on the dispatch side of the bridge.

Layering: this module imports equinox/JAX (the bridge), reads
`bayesmith.amortize`, and reaches the artifact protocol
(`bayesmith.artifacts.*`).  It is NOT imported by `bayesmith.artifacts`, and
nothing in the artifacts layer sees a single jax array.
"""

from __future__ import annotations

import dataclasses
import importlib
import io
import uuid

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.amortize import NeuralPosterior
from bayesmith.artifacts._codec import register_artifact_type
from bayesmith.artifacts.base import ArtifactRef
from bayesmith.artifacts.identity import ArtifactKind
from bayesmith.artifacts.results import FittedConditionalPosterior

__all__ = [
    "EstimatorManifest",
    "EstimatorArtifact",
    "encode_estimator",
    "decode_estimator",
    "estimator_ref",
    "fitted_conditional_posterior",
]


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class EstimatorManifest:
    """The static half of an estimator artifact: what rebuilds the estimator.

    Everything here is canonical (str / int / float / numpy arrays), so the
    whole manifest round-trips through the codec without the estimator object.
    The embed callable is recorded as its importable identity -- module and
    qualname -- never as the callable itself, and resolved back on deserialize.
    """

    embed_identity: str
    n_components: int
    n_params: int
    min_scale: float
    in_size: int
    out_size: int
    width_size: int
    depth: int
    theta_mean: np.ndarray
    theta_scale: np.ndarray
    data_mean: np.ndarray
    data_scale: np.ndarray

    def __post_init__(self) -> None:
        if type(self.embed_identity) is not str or not self.embed_identity:
            raise ValueError("embed_identity is a non-empty string")
        for name in ("n_components", "n_params"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} is a positive integer")
        if type(self.min_scale) not in (int, float) or self.min_scale < 0.0:
            raise ValueError("min_scale is a non-negative number")
        for name in ("in_size", "out_size", "width_size", "depth"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} is a positive integer")
        for name in (
            "theta_mean",
            "theta_scale",
            "data_mean",
            "data_scale",
        ):
            array = np.asarray(getattr(self, name))
            if type(array) is not np.ndarray or array.dtype.kind not in "biufc":
                raise ValueError(f"{name} is a numeric numpy array")
            object.__setattr__(self, name, array)


@register_artifact_type
@dataclasses.dataclass(frozen=True, slots=True)
class EstimatorArtifact:
    """A fitted estimator, as identity + an opaque blob + a static manifest.

    The `blob` is whatever `equinox.tree_serialise_leaves` wrote for the
    MLP's parameter leaves -- opaque bytes to the artifact codec, decoded only
    by :func:`decode_estimator`.  `artifact_id` and `revision` give it the
    same version identity every artifact carries, so :func:`estimator_ref`
    can point at it.
    """

    artifact_id: str
    revision: int
    blob: bytes
    manifest: EstimatorManifest

    def __post_init__(self) -> None:
        if type(self.artifact_id) is not str:
            raise TypeError(f"artifact_id is a string; got {self.artifact_id!r}")
        try:
            parsed = uuid.UUID(self.artifact_id)
        except ValueError as exc:
            raise ValueError(f"artifact_id is a UUID4 string; got {self.artifact_id!r}") from exc
        if parsed.version != 4 or str(parsed) != self.artifact_id:
            raise ValueError(
                f"artifact_id is a UUID4 in canonical form; got {self.artifact_id!r}"
            )
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("revision is a non-negative integer")
        if type(self.blob) is not bytes:
            raise TypeError(f"blob is bytes; got {type(self.blob).__name__}")
        if not isinstance(self.manifest, EstimatorManifest):
            raise TypeError(f"manifest is an EstimatorManifest; got {self.manifest!r}")


def _embed_identity(embed: object) -> str:
    """The importable identity of the embed callable: module.qualname.

    This is the same identity an operator gets in `dispatch.task`: a callable
    has no canonical form, but its module and qualname pin WHERE it was defined.
    A callable with neither (a lambda, a REPL function) has no identity to
    restore and is refused rather than recorded as an address.
    """
    module = getattr(embed, "__module__", None)
    qualname = getattr(embed, "__qualname__", None) or getattr(embed, "__name__", None)
    if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
        raise ValueError(
            f"the embed callable {embed!r} has no importable identity; a "
            "lambda or REPL-defined function cannot be restored later"
        )
    return f"{module}.{qualname}"


def _resolve_embed(identity: str):
    """Import the callable an :func:`_embed_identity` recorded."""
    module_name, _, qualname = identity.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(f"{identity!r} is not a module.qualname identity")
    try:
        obj = importlib.import_module(module_name)
    except ImportError as exc:
        raise ValueError(f"cannot import the embed's module {module_name!r}") from exc
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ValueError(f"cannot resolve {identity!r}: {module_name!r} has no {part!r}") from exc
    return obj


def encode_estimator(
    posterior: NeuralPosterior, *, artifact_id: str | None = None
) -> EstimatorArtifact:
    """Serialize a trained posterior's estimator into an ESTIMATOR artifact.

    The MLP's parameter leaves are written with `equinox.tree_serialise_leaves`
    into an opaque blob; every static fact (embed identity, mixture shape,
    minimum scale, MLP structure, standardization) goes into the manifest.  The
    result is a frozen dataclass the artifact codec can dump and load, with no
    jax or equinox object in it.
    """
    if not isinstance(posterior, NeuralPosterior):
        raise TypeError(f"posterior is a NeuralPosterior; got {posterior!r}")
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, posterior.net)
    manifest = EstimatorManifest(
        embed_identity=_embed_identity(posterior.embed),
        n_components=int(posterior.n_components),
        n_params=int(posterior.n_params),
        min_scale=float(posterior.min_scale),
        in_size=int(posterior.net.in_size),
        out_size=int(posterior.net.out_size),
        width_size=int(posterior.net.width_size),
        depth=int(posterior.net.depth),
        theta_mean=np.asarray(posterior.theta_mean),
        theta_scale=np.asarray(posterior.theta_scale),
        data_mean=np.asarray(posterior.data_mean),
        data_scale=np.asarray(posterior.data_scale),
    )
    return EstimatorArtifact(
        artifact_id=artifact_id if artifact_id is not None else str(uuid.uuid4()),
        revision=0,
        blob=buffer.getvalue(),
        manifest=manifest,
    )


def decode_estimator(artifact: EstimatorArtifact) -> NeuralPosterior:
    """Rebuild the NeuralPosterior an estimator artifact encodes.

    The manifest's facts rebuild the MLP's structure and the standardization;
    `equinox.tree_deserialise_leaves` fills the weights back into a fresh MLP
    of the same shape; the embed callable is re-imported from its identity.  The
    returned estimator samples exactly as the one that was encoded.
    """
    if not isinstance(artifact, EstimatorArtifact):
        raise TypeError(f"artifact is an EstimatorArtifact; got {artifact!r}")
    manifest = artifact.manifest
    embed = _resolve_embed(manifest.embed_identity)
    net_like = eqx.nn.MLP(
        in_size=manifest.in_size,
        out_size=manifest.out_size,
        width_size=manifest.width_size,
        depth=manifest.depth,
        key=jax.random.key(0),
    )
    net = eqx.tree_deserialise_leaves(io.BytesIO(artifact.blob), net_like)
    return NeuralPosterior(
        net=net,
        embed=embed,
        n_components=manifest.n_components,
        n_params=manifest.n_params,
        theta_mean=jnp.asarray(manifest.theta_mean),
        theta_scale=jnp.asarray(manifest.theta_scale),
        data_mean=jnp.asarray(manifest.data_mean),
        data_scale=jnp.asarray(manifest.data_scale),
        min_scale=manifest.min_scale,
    )


def estimator_ref(artifact: EstimatorArtifact) -> ArtifactRef:
    """The version reference to an estimator artifact: id AND revision (§0.2)."""
    if not isinstance(artifact, EstimatorArtifact):
        raise TypeError(f"artifact is an EstimatorArtifact; got {artifact!r}")
    return ArtifactRef(
        artifact_id=artifact.artifact_id,
        revision=artifact.revision,
        artifact_type=ArtifactKind.ESTIMATOR,
    )


def fitted_conditional_posterior(
    posterior: NeuralPosterior,
    *,
    simulation_bank_ref: ArtifactRef,
    training_run_id: str,
    validation_report_refs: tuple[ArtifactRef, ...] = (),
    artifact_id: str | None = None,
) -> tuple[FittedConditionalPosterior, EstimatorArtifact]:
    """Encode a trained posterior as a FittedConditionalPosterior reference.

    Returns the fitted reference AND the estimator artifact it points at, so the
    caller can persist the latter and keep the former's `estimator_ref` in
    agreement.  The simulation bank, training run and validation reports are
    references the caller already holds; only the estimator is produced here,
    because that is the one thing §0.7 adds to the frozen schema.
    """
    artifact = encode_estimator(posterior, artifact_id=artifact_id)
    fitted = FittedConditionalPosterior(
        estimator_ref=estimator_ref(artifact),
        simulation_bank_ref=simulation_bank_ref,
        training_run_id=training_run_id,
        validation_report_refs=validation_report_refs,
    )
    return fitted, artifact
