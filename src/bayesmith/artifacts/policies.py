"""Serializable controls for posterior preflight, initialization and stopping."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import numpy as np

from ._codec import register_artifact_type
from .base import NamedArray, _count, _tuple_of

__all__ = [
    "DiagnosticPolicy",
    "InitializationPolicy",
    "StoppingPolicy",
    "ProposalBlockPolicy",
    "proposal_options",
]


def _positive_count(name, value, minimum=1):
    _count(name, value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


@register_artifact_type
@dataclass(frozen=True, slots=True)
class ProposalBlockPolicy:
    """An explicit proposal block with original-target MH enabled by default.

    Dense proposal work has an independent budget. The automatic compiler may
    also instantiate policies to enlarge small moving-noise linear blocks;
    fixed-covariance GCR retains priority and failed enlargement keeps its plan.
    Disabling ``mh_correction`` applies finite, supported proposals without
    an MH test. Such state-dependent updates are approximate and need not
    correspond to any known approximate joint target.
    """

    names: tuple[str, ...]
    method: str
    steps: int = 1
    iterations: int = 5
    scale: float = 1.0
    damping: float = 1e-6
    max_parameters: int = 64
    max_matrix_elements: int = 200_000
    mh_correction: bool = True

    def __post_init__(self):
        if not isinstance(self.mh_correction, bool):
            raise TypeError("mh_correction must be bool")
        _tuple_of("proposal names", self.names, str)
        if not self.names or any(not name.strip() for name in self.names):
            raise ValueError("proposal names must be nonempty")
        if len(set(self.names)) != len(self.names):
            raise ValueError("proposal names contain duplicates")
        if self.method not in {
            "iterative_gls",
            "bias_corrected_log_linear",
            "gauss_newton",
        }:
            raise ValueError(f"unsupported proposal method {self.method!r}")
        for name in ("steps", "iterations", "max_parameters", "max_matrix_elements"):
            _positive_count(name, getattr(self, name))
        for name in ("scale", "damping"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a positive finite number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")

    def as_options(self):
        """Canonical scalar/tuple representation for existing task storage."""
        return tuple((f.name, getattr(self, f.name)) for f in fields(self))


def proposal_options(*blocks: ProposalBlockPolicy):
    """Return backend options while preserving the existing task schema.

    Use ``PosteriorTask(..., backend_options=proposal_options(block1, block2))``.
    Other backend options may be concatenated to the returned tuple.
    """
    _tuple_of("proposal blocks", blocks, ProposalBlockPolicy)
    return (("proposals", tuple(block.as_options() for block in blocks)),)


def _proposal_policies(options):
    records = dict(options).get("proposals", ())
    _tuple_of("proposals", records, tuple)
    blocks = []
    seen = set()
    for record in records:
        if any(not isinstance(item, tuple) or len(item) != 2 for item in record):
            raise ValueError("a proposal must contain (setting, value) pairs")
        keys = [item[0] for item in record]
        if any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("proposal settings must have unique string keys")
        block = ProposalBlockPolicy(**dict(record))
        if seen.intersection(block.names):
            raise ValueError("proposal blocks overlap; each site belongs to one block")
        seen.update(block.names)
        blocks.append(block)
    return tuple(blocks)


@register_artifact_type
@dataclass(frozen=True, slots=True)
class DiagnosticPolicy:
    """Bound dense preflight work; an unperformed check is never a pass.

    Bounds count matrix entries and parameter coordinates, not sites.
    Required finding codes block execution unless passed. A resolved Jeffreys
    assessment (flat or nonflat) also satisfies that check without changing priors.
    """

    enabled: bool = True
    max_parameters: int = 64
    max_matrix_elements: int = 200_000
    max_prior_parameters: int = 12
    required: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool")
        for name in ("max_parameters", "max_matrix_elements", "max_prior_parameters"):
            _count(name, getattr(self, name))
        _tuple_of("required", self.required, str)
        if len(set(self.required)) != len(self.required):
            raise ValueError("required contains duplicate finding codes")


@register_artifact_type
@dataclass(frozen=True, slots=True)
class InitializationPolicy:
    """Complete constrained values independently per chain, before warmup.

    A leading ``chain`` dimension means per-chain values; otherwise values
    are shared. Boolean masks mark supplied entries. Missing entries are
    starting points, not draws claimed to follow a conditional posterior.
    """

    values: tuple[NamedArray, ...] = ()
    masks: tuple[NamedArray, ...] = ()
    max_attempts: int = 32
    radius: float = 2.0

    def __post_init__(self):
        _tuple_of("values", self.values, NamedArray)
        _tuple_of("masks", self.masks, NamedArray)
        _positive_count("max_attempts", self.max_attempts)
        if (
            isinstance(self.radius, bool)
            or not math.isfinite(self.radius)
            or self.radius <= 0
        ):
            raise ValueError("radius must be finite and positive")
        for name in ("values", "masks"):
            items = getattr(self, name)
            if len({x.name for x in items}) != len(items):
                raise ValueError(f"duplicate {name} names")
        values = {x.name: x for x in self.values}
        masks = {x.name: x for x in self.masks}
        for name, mask in masks.items():
            value = values.get(name)
            if (
                value is None
                or mask.value.dtype != np.bool_
                or mask.value.shape != value.value.shape
                or mask.dims != value.dims
            ):
                raise ValueError(
                    f"mask for {name} must be boolean and match its value's shape/dims"
                )
        for name, value in values.items():
            mask = (
                masks[name].value
                if name in masks
                else np.ones(value.value.shape, dtype=bool)
            )
            if not np.all(np.isfinite(value.value[mask])):
                raise ValueError(
                    f"supplied values for {name} must be finite; use an explicit mask"
                )


@register_artifact_type
@dataclass(frozen=True, slots=True)
class StoppingPolicy:
    """Fixed budget, or consecutive checks on a continuing warmed chain.

    Existing chain diagnostic thresholds always apply. Optional ESS/R-hat
    limits may make the criterion stricter. MCSE is an absolute standard
    error bound on every scalar posterior mean. Time is a soft batch cap.
    """

    mode: str = "fixed"
    min_draws: int = 200
    batch_size: int = 200
    consecutive: int = 2
    ess_min: float | None = None
    rhat_max: float | None = None
    mcse_mean: float | None = None
    max_divergences: int = 0

    def __post_init__(self):
        if self.mode not in ("fixed", "checkpoints"):
            raise ValueError("stopping mode must be fixed or checkpoints")
        _positive_count("min_draws", self.min_draws, 4)
        _positive_count("batch_size", self.batch_size)
        _positive_count("consecutive", self.consecutive)
        _count("max_divergences", self.max_divergences)
        for name in ("ess_min", "rhat_max", "mcse_mean"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.rhat_max is not None and self.rhat_max < 1:
            raise ValueError("rhat_max must be at least 1")
