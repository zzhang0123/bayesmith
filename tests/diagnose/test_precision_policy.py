"""D9 -- why the diagnose family refuses float32 instead of retuning for it.

The migration plan's D9 ruled option (b): let bayesmith derive the rank
tolerance from the ambient dtype, so a float32 caller gets a verdict instead of
a refusal. It wrote the fallback's trigger as a MEASUREMENT rather than a
preference, and the measurement -- ``docs/probes/probe_13_d9_precision_policy``
-- says (b) does not exist to be implemented.

**The reason is not that the cut is hard to place. It is that the quantity the
cut is applied to is gone.** Over a two-component power law whose conditioning
is dialled by ``delta`` across ten decades, float64's smallest singular value
tracks ``delta`` faithfully (5.1e-6 down to 5.2e-16) while float32's sits at
its own roundoff floor, ~1e-7, from ``delta = 1e-2`` downward -- non-monotonic,
because it is noise. Two models that float64 separates by two decades come back
indistinguishable, so no tolerance can tell them apart: not a derived one, not
a tuned one, not one chosen per model.

That is rheplicant's own conclusion, reached independently and written in
``inference/identifiability.py``: *"A per-precision retune of rtol would
therefore recover this model. It would not recover one a few decades worse
conditioned... Forcing float64 is what lets one default be right for both."*

So the refusal stays, and this file is what stops it from being read as
timidity. The tests below are the measurement, kept runnable.
"""

from __future__ import annotations

import importlib
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.diagnose.identifiability import DEFAULT_RANK_RTOL, identifiability
from bayesmith.diagnose.priors import JeffreysPrior
from bayesmith.errors import GraphError
from tests.diagnose.models import doubled_graph, power_law_graph, two_component

#: The pair either side of float64's verdict change in family B: at 1e-2 the
#: model is identified, at 1e-3 it is not.
SEPARATED, MERGED = 1e-2, 1e-3
CANDIDATES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, math.sqrt(float(np.finfo(np.float32).eps)))


def _spectrum(delta: float, *, x64: bool, free_scale: bool = False) -> np.ndarray:
    """The column-normalised spectrum at the given ambient precision.

    The two precision guards are patched out, which is the point: they are what
    is under test, and what lies behind them is what D9 has to decide about.
    Patching rather than re-deriving the SVD keeps ONE spelling of the
    arithmetic -- the probe does the same, for the same reason.
    """
    # The MODULE, not the function of the same name that
    # ``bayesmith.diagnose.__init__`` re-exports over it.
    module = importlib.import_module("bayesmith.diagnose.identifiability")

    was = jax.config.read("jax_enable_x64")
    kept = (module.refuse_ambient_float32, module.refuse_single_precision)
    # Every mutation of global state happens INSIDE the try. Getting this wrong
    # once was instructive: an exception raised between the config update and
    # the `try` left x64 on for the rest of the session, and the next four
    # tests -- which assert that a float32 session is REFUSED -- passed their
    # graphs through a float64 one and reported "DID NOT RAISE". A leaked
    # global turns unrelated guards into tests of nothing.
    try:
        jax.config.update("jax_enable_x64", x64)
        module.refuse_ambient_float32 = lambda **_: None
        module.refuse_single_precision = lambda *_, **__: None
        graph = two_component(
            delta, jnp.float64 if x64 else jnp.float32, free_scale=free_scale
        )
        report = identifiability(graph, rtol=DEFAULT_RANK_RTOL)
        return np.asarray(report.singular_values, dtype=np.float64)
    finally:
        module.refuse_ambient_float32, module.refuse_single_precision = kept
        jax.config.update("jax_enable_x64", was)


def _ratio(values: np.ndarray) -> float:
    return float(values[-1] / values[0])


# --------------------------------------------------------------------------
# Why option (b) does not exist
# --------------------------------------------------------------------------


def test_float64_separates_the_pair_and_float32_does_not():
    """The whole argument, as one comparison.

    Two models, one identified and one not. float64 puts their weakest
    directions two decades apart; float32 puts them within a small factor of
    each other, because both are sitting on its roundoff floor rather than on
    anything about the model.

    Ratios rather than the raw values, and generous factors rather than pinned
    numbers: the claim is about ORDERS OF MAGNITUDE, and pinning digits here
    would make this test about the JAX version.
    """
    wide_64, tight_64 = _ratio(_spectrum(SEPARATED, x64=True)), _ratio(
        _spectrum(MERGED, x64=True)
    )
    wide_32, tight_32 = _ratio(_spectrum(SEPARATED, x64=False)), _ratio(
        _spectrum(MERGED, x64=False)
    )
    assert wide_64 / tight_64 > 50.0, (wide_64, tight_64)
    assert 1 / 5.0 < wide_32 / tight_32 < 5.0, (wide_32, tight_32)


#: Four models either side of a float64 verdict change, in both families.
#: **A pair is not enough** and that was measured: over the pair alone,
#: ``rtol = 1e-7`` reproduces both float64 verdicts, and a test built on it
#: would have concluded that option (b) works. It takes models whose true
#: conditioning differs by decades to show that no single cut follows.
FAMILY = ((False, 1e-2), (False, 1e-3), (True, 1e-2), (True, 1e-3))


def test_no_float32_tolerance_gets_every_verdict_right():
    """The direct form: every candidate is wrong about at least one model.

    This is the assertion that would have to be deleted, not merely adjusted,
    for option (b) to be implementable -- which is what makes the refusal a
    measured position rather than a cautious one. The candidates span from
    float64's own default up through ``sqrt(eps)`` in float32.
    """
    truth = [
        int(np.sum(s <= DEFAULT_RANK_RTOL * s[0]))
        for s in (_spectrum(d, x64=True, free_scale=f) for f, d in FAMILY)
    ]
    assert len(set(truth)) > 1, (
        "the family no longer contains a float64 verdict CHANGE, so this "
        "file's subject has gone: every tolerance would agree trivially"
    )
    single = [_spectrum(d, x64=False, free_scale=f) for f, d in FAMILY]
    for rtol in CANDIDATES:
        got = [int(np.sum(s <= rtol * s[0])) for s in single]
        assert got != truth, f"rtol={rtol:.0e} reproduced every verdict"


# --------------------------------------------------------------------------
# Option (a) fails loudly, which is what makes it usable
# --------------------------------------------------------------------------


def test_wrapping_only_the_call_is_refused_by_name():
    """The fallback's failure mode, and it is not silent.

    D9 calls the naive spelling of option (a) a no-op: opening x64 around the
    CALL leaves a graph whose constants were traced at float32, so the verdict
    would be the float32 verdict wearing a float64 label. It is refused, and
    the message names the remedy -- build the graph inside the block.
    """
    outside = two_component(MERGED, jnp.float32, free_scale=False)
    with jax.enable_x64(True), pytest.raises(GraphError, match="came back float32"):
        identifiability(outside)


def test_building_the_graph_inside_the_block_gets_the_verdict_right():
    """The sibling: the correct spelling reaches float64's answer.

    Without this, the refusal above is consistent with a package that simply
    cannot do it at all.
    """
    with jax.enable_x64(True):
        inside = two_component(MERGED, jnp.float64, free_scale=False)
        report = identifiability(inside)
        assert report.nullity == 1
        assert report.singular_values[-1] / report.singular_values[0] < 1e-9


# --------------------------------------------------------------------------
# The guard D9's registry line did not know it needed
# --------------------------------------------------------------------------


def test_a_jeffreys_information_from_a_truncating_graph_is_refused():
    """The third caller, which had the ambient guard and not the graph one.

    Measured before the guard, on the exactly-degenerate block: the same model
    gives a half-log-determinant of **-338.05** when the graph is built inside
    the x64 block and **-27.52** when built outside it. A 310-nat difference in
    a log-prior, silent, in a term NUTS exponentiates -- and precisely the
    "a determinant that came back finite is not a guard" failure the eigenvalue
    floor exists to prevent, arriving by a route the floor cannot see.
    """
    prior = JeffreysPrior(over=("a", "b", "fg_beta"))
    outside = doubled_graph()
    with jax.enable_x64(True):
        at = {"a": jnp.array(3.9), "b": jnp.array(3.9), "fg_beta": jnp.array(2.55)}
        with pytest.raises(GraphError, match="came back float32"):
            prior.information(outside, at)
        # And the correct spelling still reaches the honest answer.
        inside = doubled_graph()
        floored = float(prior.half_log_determinant(prior.information(inside, at)))
        assert floored < -300.0


def test_the_floor_is_read_off_the_dtype_which_is_what_that_guard_protects():
    """Why 310 nats: the floor substitutes the DTYPE's smallest positive number.

    ``log(tiny)`` is -708 in float64 and -87 in float32, so the same singular
    matrix scores about 310 nats apart depending only on the arithmetic it
    arrived in. Asserted on matrices directly, so it pins the mechanism rather
    than the graph path that exposed it.
    """
    prior = JeffreysPrior(over=("a", "b"))
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    # The float64 half needs an x64 session to exist at all: outside one,
    # `jnp.asarray(..., float64)` truncates back to float32 with a warning,
    # and the test would compare a number with itself.
    with jax.enable_x64(True):
        wide = float(prior.half_log_determinant(jnp.asarray(singular, jnp.float64)))
    narrow = float(prior.half_log_determinant(jnp.asarray(singular, jnp.float32)))
    assert wide < narrow - 250.0, (wide, narrow)


def test_every_diagnostic_that_refuses_float32_refuses_a_truncating_graph_too():
    """A census, not a list -- which is how the missing one would have shown.

    Five entry points call ``refuse_ambient_float32``.  Enumerating only the
    older three is how P1 and P2 escaped the registry, so this asks every one
    the same question instead: given a graph that truncates, does it refuse?
    """
    from bayesmith.diagnose.coupling import block_coupling
    from bayesmith.diagnose.map import Refused as MapRefused
    from bayesmith.diagnose.map import map_estimate
    from bayesmith.diagnose.sensitivity import prior_sensitivity

    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    at = {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}
    # Two graphs, because the entry points disagree about what a prior may be:
    # `prior_sensitivity` needs proper Gaussian priors to have a mode to
    # displace, and `JeffreysPrior` REFUSES a covered latent that has one --
    # that is two priors on one quantity. The shared thing under test is not
    # the graph, it is that each was built OUTSIDE the x64 block.
    flat = power_law_graph(noise="homo")
    proper = power_law_graph(noise="homo", flat_latents=False)

    def run_identifiability():
        identifiability(flat, names=("fg_log_amp", "fg_beta"), at=at)

    def run_prior_sensitivity():
        prior_sensitivity(proper, names=("fg_log_amp", "fg_beta"), at=at)

    def run_jeffreys():
        prior.information(flat, at)

    def run_coupling():
        block_coupling(
            proper,
            ("fg_log_amp",),
            ("fg_beta",),
            at=at,
        )

    def run_map():
        return map_estimate(proper)

    with jax.enable_x64(True):
        for name, call, structured in (
            ("identifiability", run_identifiability, False),
            ("prior_sensitivity", run_prior_sensitivity, False),
            ("JeffreysPrior.information", run_jeffreys, False),
            ("block_coupling", run_coupling, False),
            ("map_estimate", run_map, True),
        ):
            if structured:
                result = call()
                assert isinstance(result, MapRefused)
                reason = result.reason
            else:
                with pytest.raises(
                    GraphError,
                    match="came back float32",
                ) as caught:
                    call()
                reason = str(caught.value)
            assert name  # the loop variable is what the failure names
            assert "came back float32" in reason


def test_all_five_refuse_an_ambient_float32_session():
    """The other half of the pair, and the claim D9's line makes."""
    from bayesmith.diagnose.coupling import block_coupling
    from bayesmith.diagnose.map import Refused as MapRefused
    from bayesmith.diagnose.map import map_estimate
    from bayesmith.diagnose.sensitivity import prior_sensitivity

    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    flat = power_law_graph(noise="homo")
    proper = power_law_graph(noise="homo", flat_latents=False)
    at = {"fg_log_amp": jnp.array(7.8), "fg_beta": jnp.array(2.55)}

    for call, structured in (
        (
            lambda: identifiability(
                flat, names=("fg_log_amp", "fg_beta"), at=at
            ),
            False,
        ),
        (
            lambda: prior_sensitivity(
                proper, names=("fg_log_amp", "fg_beta"), at=at
            ),
            False,
        ),
        (lambda: prior.information(flat, at), False),
        (
            lambda: block_coupling(
                proper,
                ("fg_log_amp",),
                ("fg_beta",),
                at=at,
            ),
            False,
        ),
        (lambda: map_estimate(proper), True),
    ):
        if structured:
            result = call()
            assert isinstance(result, MapRefused)
            reason = result.reason
        else:
            with pytest.raises(
                GraphError, match="float32 as the ambient precision"
            ) as caught:
                call()
            reason = str(caught.value)
        assert "float32 as the ambient precision" in reason
