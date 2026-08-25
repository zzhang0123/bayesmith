"""identifiability, both packages, one fixture: the migration's §四 4.2 row.

The acceptance is exact where exactness is possible and honest where it is
not: rank and nullity are integers and must be EQUAL; singular values come
from the same float64 arithmetic on Jacobians assembled in different
evaluation orders, so they agree to roundoff-scaled tolerance; a null
DIRECTION is unique only up to sign (nullity 1) or up to rotation within
the null space (nullity 8), so the comparison is sign-fixed elementwise in
the first case and projector-based in the second -- comparing raw rows
would fail on a mathematically identical answer.

The fixture arrays are built ONCE and handed to both packages, per the
harness rule §0.1 records: the two sides have different precision regimes,
so "the same construction code" is not "the same numbers" -- but the same
ARRAYS are.
"""

from __future__ import annotations

# Module scope, not function scope, and load-bearing: `from __future__ import
# annotations` turns the operator classes' body annotations into strings, and
# dataclasses resolves the `ClassVar` marker against THIS module's globals --
# imported only inside the builder, `requires`/`provides` silently become
# ordinary defaulted fields and every subclass with a non-default field fails
# to build.
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crosscheck

N_TIME, N_FREQ = 8, 8
TONE_CHANNEL = 3


def _arrays():
    def poly_basis(n, degree):
        x = jnp.linspace(-1.0, 1.0, n)
        return jnp.stack([x**k for k in range(degree)], axis=1)

    time_basis = poly_basis(N_TIME, 3)
    freq_basis = poly_basis(N_FREQ, 3)
    coeff0 = jnp.array(
        [[3000.0, -180.0, 40.0], [120.0, 25.0, -8.0], [-45.0, 6.0, 2.0]]
    )
    gain0 = 1.5 + 0.05 * jnp.arange(N_TIME, dtype=float)
    return time_basis, freq_basis, coeff0, gain0


def _rheplicant_report(tone_kelvin, kind):
    """rheplicant's identifiability on its own vocabulary, same arrays."""
    from rheplicant import Coordinates, State
    from rheplicant.core.operator import AbstractOperator
    from rheplicant.core.pipeline import Pipeline
    from rheplicant.inference import Bind, Latent, ParameterSpace
    from rheplicant.inference.identifiability import identifiability
    from rheplicant.radio import GainOperator

    time_basis, freq_basis, coeff0, gain0 = _arrays()
    t_ant0 = time_basis @ coeff0 @ freq_basis.T
    tone = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(tone_kelvin)

    class AntennaTemperature(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
        provides: ClassVar[tuple[str, ...]] = ("data",)
        t_ant: jax.Array

        def __call__(self, state):
            return state.with_data(self.t_ant)

    class CalibrationTone(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("data",)
        provides: ClassVar[tuple[str, ...]] = ("data",)
        tone: jax.Array

        def __call__(self, state):
            return state.with_data(state.data + self.tone[None, :])

    pipeline = Pipeline(
        AntennaTemperature(t_ant=t_ant0),
        CalibrationTone(tone=tone),
        GainOperator(gain=gain0),
        names=("t_ant", "tone", "gain"),
    )
    if kind == "basis":
        space = ParameterSpace(
            latents=[Latent("gain", init=gain0), Latent("t_coeff", init=coeff0)],
            bindings=[
                Bind("gain", into=lambda p: p["gain"].gain),
                Bind(
                    "t_coeff",
                    into=lambda p: p["t_ant"].t_ant,
                    fn=lambda c: time_basis @ c @ freq_basis.T,
                ),
            ],
        )
    else:
        space = ParameterSpace(
            latents=[Latent("gain", init=gain0), Latent("t_ant", init=t_ant0)],
            bindings=[
                Bind("gain", into=lambda p: p["gain"].gain),
                Bind("t_ant", into=lambda p: p["t_ant"].t_ant),
            ],
        )
    state = State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        meta={"telescope": "RHINO", "obs_id": "crosscheck-ident"},
    )
    return identifiability(space, pipeline, state)


def _bayesmith_report(tone_kelvin, kind):
    """bayesmith's identifiability on the graph port, same arrays."""
    import numpyro.distributions as dist

    from bayesmith import det, observe, sample, trace
    from bayesmith.diagnose.identifiability import identifiability

    time_basis, freq_basis, coeff0, gain0 = _arrays()
    t_ant0 = time_basis @ coeff0 @ freq_basis.T
    tone = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(tone_kelvin)
    data = gain0[:, None] * (t_ant0 + tone[None, :])

    def model():
        gain = sample("gain", lambda: dist.Normal(gain0, 0.1).to_event(1))
        if kind == "basis":
            second = sample(
                "t_coeff", lambda: dist.Normal(coeff0, 10.0).to_event(2)
            )
            pred = det(
                "pred",
                lambda g, c: g[:, None]
                * (time_basis @ c @ freq_basis.T + tone[None, :]),
                gain,
                second,
            )
        else:
            second = sample("t_ant", lambda: dist.Normal(t_ant0, 100.0).to_event(2))
            pred = det(
                "pred", lambda g, t: g[:, None] * (t + tone[None, :]), gain, second
            )
        observe("d", lambda mu: dist.Normal(mu, 1.0).to_event(2), pred, obs=data)

    return identifiability(trace(model))


def test_rank_and_nullity_agree_on_all_four_rows():
    """The integer half of the acceptance: same table, cell for cell."""
    with jax.enable_x64(True):
        for kind in ("free", "basis"):
            for tone in (5000.0, 0.0):
                theirs = _rheplicant_report(tone, kind)
                ours = _bayesmith_report(tone, kind)
                assert (theirs.n_par, theirs.rank, theirs.nullity) == (
                    ours.n_par,
                    ours.rank,
                    ours.nullity,
                ), (kind, tone)


def test_the_spectra_agree_to_arithmetic_reordering():
    """Same matrix up to evaluation order, same SVD: the spectra agree to a
    roundoff-scaled tolerance, and the landmark ratios to their printed
    digits. NOT bitwise, deliberately -- the two packages walk the model in
    different orders, and demanding bitwise here would pin the walk, not
    the mathematics."""
    with jax.enable_x64(True):
        theirs = _rheplicant_report(0.0, "basis")
        ours = _bayesmith_report(0.0, "basis")
    assert ours.singular_values == pytest.approx(
        theirs.singular_values, rel=1e-10, abs=1e-12
    )
    assert ours.weakest_identified == pytest.approx(
        theirs.weakest_identified, rel=1e-9
    )


def test_the_null_direction_agrees_elementwise_up_to_sign():
    """§四 4.2's "direction 逐分量一致", stated the only way it is
    mathematically meaningful for nullity 1: a null direction is a ray, so
    the rows may differ by an overall sign, fixed here on the largest
    entry."""
    with jax.enable_x64(True):
        theirs = _rheplicant_report(0.0, "basis")
        ours = _bayesmith_report(0.0, "basis")
    assert theirs.nullity == ours.nullity == 1

    def fixed(direction):
        flat = np.concatenate([np.ravel(direction[k]) for k in sorted(direction)])
        return flat * np.sign(flat[np.argmax(np.abs(flat))])

    ours_direction = fixed(ours.direction(0))
    theirs_direction = fixed(theirs.direction(0))
    assert ours_direction == pytest.approx(theirs_direction, abs=1e-9)
    # The shares agree too, in the coordinates the naming happens in.
    ours_share = ours.participation(0)
    theirs_share = theirs.participation(0)
    assert ours_share["gain"] == pytest.approx(theirs_share["gain"], abs=1e-9)
    assert ours_share["t_coeff"] == pytest.approx(theirs_share["t_coeff"], abs=1e-9)


def test_the_eight_dimensional_null_spaces_are_the_same_subspace():
    """For nullity 8 individual rows are basis-dependent; the SUBSPACE is
    not. Compared as projectors ``V^T V``, which are unique."""
    with jax.enable_x64(True):
        theirs = _rheplicant_report(5000.0, "free")
        ours = _bayesmith_report(5000.0, "free")
    assert theirs.nullity == ours.nullity == 8
    projector_theirs = theirs.null_space.T @ theirs.null_space
    projector_ours = ours.null_space.T @ ours.null_space
    assert projector_ours == pytest.approx(projector_theirs, abs=1e-9)
