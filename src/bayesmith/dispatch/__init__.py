"""Structural dispatch: deriving what to run from what the graph declares.

:mod:`bayesmith.dispatch.classify` answers one question -- which latents an
exact linear-Gaussian method applies to, how they group, and which method the
group needs. It reads the three structural axes P1 recorded (``linear_in``,
``support``, ``depends_on_prediction``) and the guards P3a built, and
produces no samples of its own.
"""

from bayesmith.dispatch.classify import (
    SIGMA_RTOL,
    Classification,
    block_at,
    partition,
    prior_environment,
)

__all__ = [
    "partition",
    "Classification",
    "block_at",
    "prior_environment",
    "SIGMA_RTOL",
]
