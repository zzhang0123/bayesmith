"""probe_28's measurements, held where a drift in them turns the suite red.

The R3 plan quotes twenty-odd numbers from
``docs/probes/probe_28_model_checking_seams.py`` and builds every threshold in
§0 on top of them.  A probe is a script: it is run once, its output is pasted
into a plan, and nothing afterwards re-runs it.  So the numbers a plan rests on
decay exactly the way six copies of one measurement decayed in this repository
before -- silently, on a day nobody edited them.

This file re-runs the probe's own functions and pins what §0 depends on.  It
imports the probe rather than restating it: a pin that reimplemented the
measurement would agree with itself by construction and say nothing about the
seam the plan was reading.

**What is pinned, and how tightly, is not one decision.**  Three shapes appear
below and they are deliberately different:

* A **saturated** measurement -- `p == 0.0` for a model that is wrong in the
  direction the discrepancy looks at, `p == 1.0` for a prior a thousand times
  too vague.  No draw of the several thousand crosses; nothing short of a
  changed seam moves it.
* A **counted** measurement -- a weighted p over an iid posterior is a multiple
  of ``1/N_draws``, so 0.9120 is 1824/2000 exactly and the tolerance says how
  many draws may cross before the pin speaks.
* A **property** -- §0.6 does not rest on "the KS p of a 2x-too-wide SBC at
  N=20 is 0.0332", it rests on that value being BELOW α while the correct
  model's is above.  Pinning the property is what the plan asked for and it is
  also the honest pin: the value is one draw of a random experiment, the side
  of α it falls on is the claim.

**Measured on two platforms before any tolerance below was chosen.**  The
sixteen fixtures that burned four release tags in this repository were all
numbers one machine's arithmetic happened to produce, so the probe was re-run
under `linux/amd64` with `OPENBLAS_CORETYPE=ZEN` (the container recipe in
CLAUDE.md) and compared against the development laptop's Accelerate build:
sections 1, 4 and 8 agreed in **every printed digit** -- 0.9120 / 0.3270,
0.3228 / 0.7463, 0.0000 / 0.0000, 0.4300 / 0.3410; N=20 correct 0.4141, wide
0.0332, narrow 0.0193; 0.210 and 1.0000.  So the tolerances here are chosen
for what a change to the SEAM would do, not to absorb a BLAS difference that
was never observed.

Section numbers below are probe_28's own (§1 PPC, §4 SBC power, §8 prior
predictive).
"""

from __future__ import annotations

import functools
import importlib.util
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.evaluation import ALPHA
from tests.exact.models import straight_line

_PROBE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "docs"
    / "probes"
    / "probe_28_model_checking_seams.py"
)


@functools.cache
def probe():
    """probe_28, imported as a module rather than run as a script.

    Loaded by path because ``docs/probes/`` is not a package -- it is a shelf
    of scripts, and making it importable would invite the rest of the suite to
    lean on it.  One file leans on it, this one, and it says so here.

    ``sys.argv`` is swapped for the duration of the import because the probe
    reads ``int(sys.argv[1])`` at module scope as its replicate count.  Under
    pytest that argument is a node id, so importing it unguarded raises
    ``ValueError`` before a single line of the module body runs.  The probe is
    a committed record of a measurement and this task does not edit it; the
    guard belongs on the side that is doing something unusual.
    """
    spec = importlib.util.spec_from_file_location(_PROBE.stem, _PROBE)
    assert spec is not None and spec.loader is not None, _PROBE
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [str(_PROBE)]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


# ------------------------------------------------------- the layer's own α


def test_the_evaluation_layer_declares_one_false_positive_rate():
    """D104 (§0.4), and the reason it is a module constant rather than a
    default argument in five places.

    §9.3 requires every random acceptance test to declare its tolerated
    false-positive rate in advance.  Five checks each picking their own would
    be five magic numbers wearing a statistical hat; one α, read by all of
    them, is a number a close-out can multiply by the number of tests and
    write down.  Its provenance is BORROWED -- statistics' conventional
    default -- not derived from anything in this repository, and §0.4 says so.
    """
    assert ALPHA == 0.05


# ---------------------------------------------------- §1: PPC on the R2 seam


@pytest.fixture(scope="module")
def ppc_straight_line():
    """probe_28 §1's first row, recomputed: 2000 gcr draws through
    ``replicated_draws``, two discrepancies, weighted p for each."""
    return probe().ppc(straight_line(), "straight_line", jax.random.key(1))


def test_probe28_section1_pins_the_correct_models_two_p_values(ppc_straight_line):
    """§0.3 quotes 0.9120 / 0.3270 as what a CORRECT model scores, and §0.4's
    band ``[α/2, 1 - α/2]`` has to contain them or R3's first check fails its
    own fixture on day one.

    The tolerance is the plan's ``rel_tol=1e-3`` and it is worth reading as a
    count: an iid posterior gives every draw weight ``1/2000``, so p is a
    multiple of 0.0005 and 0.9120 is 1824 draws of 2000.  A relative 1e-3
    around 0.912 is +/-0.0009 -- one draw may cross the observed statistic and
    this stays green, two may not.
    """
    assert ppc_straight_line["curvature"] == pytest.approx(0.9120, rel=1e-3)
    assert ppc_straight_line["scale"] == pytest.approx(0.3270, rel=1e-3)
    for name, value in ppc_straight_line.items():
        assert ALPHA / 2 <= value <= 1 - ALPHA / 2, name


def test_probe28_section1_pins_the_misspecified_model_at_zero():
    """The G2 half: ``curved_line(0.6)`` scores 0.0000 on BOTH statistics.

    Saturated, and that is the point -- not one replicated dataset out of 2000
    reaches the observed curvature, so the pin does not depend on where a
    tolerance is drawn.  It is also the sharpest statement §0.3 makes about
    the check's power, and the sentence it sits next to matters: the same
    discrepancies score 0.4300 / 0.3410 on ``curvature=0.15``, so a PASS here
    will never mean "the model is right".
    """
    scores = probe().ppc(
        probe().curved_line(curvature=0.6), "curved_line", jax.random.key(1)
    )
    assert scores["curvature"] == 0.0
    assert scores["scale"] == 0.0
    assert all(value < ALPHA / 2 for value in scores.values())


# ------------------------------------------ §4: the SBC replicate-count sweep


def test_probe28_section4_separates_a_2x_width_error_at_twenty_replicates():
    """§0.6 reads its D106 floor off this sweep, so the sweep's SHAPE is what
    has to hold: at N=20 a correctly calibrated sampler sits above α and a
    2x-too-wide one sits below it.

    The property, not the two p-values (0.4141 and 0.0332 when measured).  A
    KS p-value is one draw of a random experiment; which side of α it lands on
    is the claim §0.6 makes, and pinning 0.4141 to three digits would be
    pinning the draw.  The seeds are the probe's own -- 31 for the correct
    sampler, 23 for the distorted one -- because §9.3 requires a fixed seed
    and because a red test here is a reason to look at the sampler, never a
    reason to try another seed.

    N=20 is also BELOW the floor §0.6 sets: the plan's own table has the
    too-wide case at 0.0332, less than 1.5x from α, which is why the floor is
    50 rather than 20.  This test pins the direction, not sufficiency.
    """
    _ranks, correct, _seconds, _method = probe().sbc_exact(
        20, jax.random.key(31), distort=1.0
    )
    _ranks, wide, _seconds, _method = probe().sbc_exact(
        20, jax.random.key(23), distort=2.0
    )
    assert correct.pvalue > ALPHA > wide.pvalue, (correct.pvalue, wide.pvalue)


# ------------------------------------------------- §8: prior predictive draws


def _vague_line():
    """``straight_line``'s data under a prior a million times too wide.

    The probe builds this inline inside ``run_prior_predictive`` rather than as
    a reusable function, so it is rebuilt here.  Everything except ``w``'s
    prior standard deviation is read off the probe's own module constants, so
    the two cannot drift apart on the parts that are shared.
    """
    p = probe()
    y = np.asarray(straight_line().node("d").observed)

    def model():
        xs = const("X", p.X)
        w = sample("w", lambda: dist.Normal(0.0, 1e6))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, p.SIGMA), mu, obs=jnp.asarray(y))

    return trace(model)


def _prior_tail_mass(graph, key, n=4000):
    """``P(max|d_prior| >= max|d_obs|)`` -- §8's prior-scale statistic."""
    drawn = np.asarray(probe().prior_draws(graph, key, n)["d"])
    observed = np.asarray(graph.node("d").observed)
    return float(np.mean(np.abs(drawn).max(axis=1) >= np.abs(observed).max()))


def test_probe28_section8_pins_both_ends_of_the_prior_scale_check():
    """§0.2's ``prior_predictive_check`` row rests on these two numbers being
    on opposite sides of §0.4's band, and on the wide one being saturated.

    0.210 is 840 draws of 4000, and unlike the PPC pins above it involves no
    linear solve at all: a prior draw is threefry bits, a scalar-vector
    multiply and an elementwise Normal sample, so the count is reproducible
    rather than merely stable.  It is pinned to the draw.

    1.0000 is every one of 4000 draws.  A prior with sd 1e6 puts the observed
    data so far inside its own bulk that the check cannot fail to notice, and
    a check that could not see THAT would not be worth running.
    """
    tight = _prior_tail_mass(straight_line(), jax.random.key(9))
    vague = _prior_tail_mass(_vague_line(), jax.random.key(9))

    assert tight == pytest.approx(0.210, abs=1.0 / 4000)
    assert ALPHA / 2 <= tight <= 1 - ALPHA / 2
    assert vague == 1.0


def test_probe28_section8_pins_the_plated_refusal_task_2_has_to_answer():
    """The probe's ``prior_draws`` REFUSES a plated node, and §0.7 is the
    ruling that answers it.

    ``apply_probabilistic`` returns an unmapped distribution for a plated node
    with no plated parent -- deliberately, so ``log_joint`` and the numpyro
    bridge can broadcast one shared prior across a plate -- and its ``.sample``
    is therefore ONE value, not a plate of them.  §0.7 expands it with
    ``graph.plate_size``.  Pinned here so that Task 2's primitive is answering
    a refusal somebody measured rather than one somebody remembered.
    """
    from tests.exact.models import plated_latent

    with pytest.raises(NotImplementedError, match="plated node 'z'"):
        probe().prior_draws(plated_latent(), jax.random.key(1), 2)
