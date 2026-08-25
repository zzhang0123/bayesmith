"""``likelihood.py`` / ``noise.py`` against ``exact/gaussian``: §四 4.1's
fourth row, and §四 4.2's probabilistic-node row, on one page.

Both rows name ``noise.py``, so both live here. 4.1 asks for
``-½Σ[r²/σ² + log 2πσ²]`` and for a flagged ``σ=∞`` to contribute a clean
zero **by masking rather than by letting inf propagate**; 4.2 asks for the
three noise models × with and without flags, in **log-density and in
sampling distribution**.

Three things this file establishes that a reading of either package could
not:

* the density agrees to the last digit across FIVE spellings, one of them
  an independent NumPy closed form (iron law 4 -- two implementations
  agreeing is not evidence);
* the flagged number agrees too, but the mask lives in a **different layer**
  here, deliberately and with a written argument. `precision.log_normalizer`
  returning ``+inf`` looks exactly like the defect 4.1 warns about and is
  not one;
* ``realise`` and the node's own draw are bitwise identical for constant
  sigma and **diverge exactly on the negative predictions** for the
  radiometer -- same distribution, mirrored draw.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

pytestmark = pytest.mark.crosscheck

#: ``1 / sqrt(delta_nu * tau)`` at ``delta_nu * tau = 400``. Chosen so the
#: fractional level is an exact binary fraction (0.05 is not, but 400 is,
#: and the sqrt of 400 is): the two packages compute it by different
#: expressions and a non-representable intermediate would put a ULP between
#: them for no reason of substance.
CHANNEL_WIDTH, INTEGRATION_TIME = 400.0, 1.0
FRACTIONAL = 0.05


def _fixture():
    """Prediction, data and a per-sample sigma. Call INSIDE the x64 block.

    The prediction **crosses zero** on purpose: it is the only regime where
    the radiometer's multiplicative generator and an additive one differ,
    and where ``std``'s ``abs`` is load-bearing.
    """
    prediction = jnp.array([3.0, -2.0, 5.0, 1.0, -4.0, 2.5])
    residual = jnp.array([0.1, -0.2, 0.05, 0.3, 0.15, -0.1])
    sigma = jnp.array([1.0, 0.5, 2.0, 1.0, 0.8, 1.5])
    return prediction, prediction + residual, sigma


#: Which samples the flagged fixtures drop.
FLAGS = (False, False, True, False, True, False)


def _numpy_gaussian(prediction, observed, sigma, keep=None) -> float:
    """The density in NumPy, written out. The independent oracle.

    ``-log(sigma) - 0.5 log 2pi`` rather than ``-0.5 log(2 pi sigma**2)``:
    a different grouping of the same expression, so a shared algebraic slip
    in the two packages does not survive here too.
    """
    prediction = np.asarray(prediction, dtype=float)
    observed = np.asarray(observed, dtype=float)
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), prediction.shape)
    if keep is not None:
        mask = np.asarray(keep, dtype=bool)
        prediction, observed, sigma = prediction[mask], observed[mask], sigma[mask]
    return float(
        np.sum(
            -0.5 * ((observed - prediction) / sigma) ** 2
            - np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
        )
    )


def test_the_constant_sigma_density_agrees_across_five_spellings():
    """§四 4.1's ``-½Σ[r²/σ² + log 2πσ²]``.

    rheplicant has two spellings of it (``GaussianLikelihood`` and
    ``NoiseModelLikelihood`` under ``HomoscedasticNoise``, which its own
    docstring says must agree); bayesmith has two more (``log_density`` on a
    ``Precision``, and ``log_joint`` walking the graph). The fifth is NumPy,
    and it is the only one of the five that is evidence.
    """
    from rheplicant.inference.likelihood import GaussianLikelihood
    from rheplicant.inference.noise import HomoscedasticNoise, NoiseModelLikelihood

    from bayesmith import log_joint, observe, trace
    from bayesmith.exact.precision import DiagonalPrecision, log_density

    with jax.enable_x64(True):
        prediction, observed, sigma = _fixture()

        def model():
            observe(
                "d", lambda: dist.Normal(prediction, sigma).to_event(1), obs=observed
            )

        values = [
            float(GaussianLikelihood(noise_std=sigma)(prediction, observed)),
            float(
                NoiseModelLikelihood(noise=HomoscedasticNoise(sigma=sigma))(
                    prediction, observed
                )
            ),
            float(log_density(DiagonalPrecision(sigma=sigma), observed - prediction)),
            float(log_joint(trace(model))),
            _numpy_gaussian(prediction, observed, sigma),
        ]
    assert len(set(values)) == 1, values


def test_the_prediction_dependent_density_agrees_to_one_ulp():
    """The radiometer, ``sigma = |prediction| * f``, floor off.

    Not bitwise: the two sides reach the same sigma by different
    expressions (``abs(x) * f`` against the node's own ``f * abs(m)``) and
    then sum in different orders. One ULP is the honest tolerance, and
    ``rel=1e-15`` says so without admitting a real difference.
    """
    from rheplicant.inference.noise import NoiseModelLikelihood, RadiometerNoise

    from bayesmith import log_joint, observe, trace
    from bayesmith.exact.precision import DiagonalPrecision, log_density

    with jax.enable_x64(True):
        prediction, observed, _ = _fixture()
        noise = RadiometerNoise(
            channel_width=CHANNEL_WIDTH, integration_time=INTEGRATION_TIME
        )
        assert noise.fractional == FRACTIONAL
        sigma = noise.std(prediction)

        def model():
            observe(
                "d",
                lambda: dist.Normal(
                    prediction, FRACTIONAL * jnp.abs(prediction)
                ).to_event(1),
                depends_on_prediction=True,
                obs=observed,
            )

        theirs = float(NoiseModelLikelihood(noise=noise)(prediction, observed))
        ours_operator = float(
            log_density(DiagonalPrecision(sigma=sigma), observed - prediction)
        )
        ours_graph = float(log_joint(trace(model)))
        truth = _numpy_gaussian(prediction, observed, sigma)
    for label, got in (
        ("rheplicant", theirs),
        ("log_density", ours_operator),
        ("log_joint", ours_graph),
    ):
        assert got == pytest.approx(truth, rel=1e-15), (label, got, truth)


def test_dropping_the_logdet_is_a_different_number_and_stays_one():
    """``include_logdet=False`` is generalized least squares, not a
    normalisation choice. Guarded against a reading that would make the row
    above vacuous: if the two coincided, the density comparison would not be
    testing the log-determinant at all.

    The size is the whole of B1, and the sign is fixed: dropping
    ``Σ log 2πσ²`` for a strictly positive sigma REMOVES a term, so on this
    fixture (every sigma below 1, so every log negative) it makes the
    objective smaller.
    """
    from rheplicant.inference.noise import NoiseModelLikelihood, RadiometerNoise

    with jax.enable_x64(True):
        prediction, observed, _ = _fixture()
        noise = RadiometerNoise(
            channel_width=CHANNEL_WIDTH, integration_time=INTEGRATION_TIME
        )
        full = float(NoiseModelLikelihood(noise=noise)(prediction, observed))
        gls = float(
            NoiseModelLikelihood(noise=noise, include_logdet=False)(
                prediction, observed
            )
        )
        gap = float(-0.5 * jnp.sum(jnp.log(2.0 * jnp.pi * noise.std(prediction) ** 2)))
    assert gls != full
    assert full - gls == pytest.approx(gap, rel=1e-12)
    assert gls < full


# --------------------------------------------------------------------------
# The flagged half: §四 4.1's "flagged σ=∞ 的干净零贡献，且以 MASK 实现而非
# 让 inf 传播". The number agrees; the mask lives one layer away, and that
# is a decision with an argument rather than an omission.
# --------------------------------------------------------------------------


def _masked_term(prediction, observed, sigma, flags):
    """bayesmith's masked Gaussian, through the layer that owns the concept.

    ``compress`` needs a design, so it gets a zero one: with ``R = 0`` the
    term is ``-0.5||0·x - z||² + offset`` at every ``x``, which is exactly
    the observed-node density and nothing else.
    """
    from bayesmith.evidence.compress import compress
    from bayesmith.exact.precision import DiagonalPrecision

    precision = DiagonalPrecision(sigma=jnp.where(flags, jnp.inf, sigma))
    term = compress(
        design={"w": jnp.zeros((jnp.size(prediction), 1))},
        data=observed - prediction,
        precision=precision,
        shapes={"w": ()},
    )
    return float(-0.5 * jnp.sum(term.target**2) + term.offset), precision


def test_a_flagged_sample_contributes_exactly_zero_on_both_sides():
    """The number, four ways, and the fourth is the oracle.

    Measured: **-3.5202942825891324**, identical across rheplicant's two
    spellings, bayesmith's masked term, and a NumPy density over the four
    unflagged samples alone. That last comparison is the one that says the
    contribution is exactly zero rather than merely small: it never sees the
    flagged samples at all.
    """
    from rheplicant.inference.likelihood import MaskedGaussianLikelihood
    from rheplicant.inference.noise import (
        FlaggedNoise,
        HomoscedasticNoise,
        NoiseModelLikelihood,
    )

    with jax.enable_x64(True):
        prediction, observed, sigma = _fixture()
        flags = jnp.array(FLAGS)
        flagged = FlaggedNoise(base=HomoscedasticNoise(sigma=sigma), flags=flags)
        values = [
            float(NoiseModelLikelihood(noise=flagged)(prediction, observed)),
            float(
                MaskedGaussianLikelihood(noise_std=sigma, flags=flags)(
                    prediction, observed
                )
            ),
            _masked_term(prediction, observed, sigma, flags)[0],
            _numpy_gaussian(prediction, observed, sigma, keep=~np.asarray(FLAGS)),
        ]
    assert len(set(values)) == 1, values


def test_the_quadratic_half_masks_and_the_normaliser_deliberately_does_not():
    """Why the previous test had to go through ``compress``.

    ``DiagonalPrecision.apply`` is ``r / sigma**2``, so an infinite sigma
    gives weight zero with no special case and ``quadratic`` is already the
    four-sample sum. ``log_normalizer`` is ``+inf``, so ``log_density`` is
    ``-inf`` -- which LOOKS exactly like the "let inf propagate" defect §四
    4.1 warns against, and is not.

    ``evidence/compress.py``'s module docstring is where the decision is
    argued: *"a sample with infinite variance has no density, which is the
    honest answer to the question a Precision is asked. Reading it as 0 is a
    statement that the sample is UNOBSERVED, which is a modelling concept
    this layer has and the interface does not."* So the mask lives in the
    layer that owns "unobserved", and ``precision.py`` keeps a normaliser
    that is never silently wrong.

    Pinned in **both** directions, because a reader arriving at ``-inf``
    with 4.1's sentence in hand will otherwise fix it: the quadratic must
    keep masking, and the normaliser must keep refusing to.
    """
    from bayesmith.evidence.compress import observed_mask
    from bayesmith.exact.precision import log_density, quadratic

    with jax.enable_x64(True):
        prediction, observed, sigma = _fixture()
        flags = jnp.array(FLAGS)
        _, precision = _masked_term(prediction, observed, sigma, flags)
        residual = observed - prediction
        kept = ~np.asarray(FLAGS)
        expected = float(
            np.sum((np.asarray(residual)[kept] / np.asarray(sigma)[kept]) ** 2)
        )
        measured = float(quadratic(precision, residual))
        density = float(log_density(precision, residual))
        mask = np.asarray(observed_mask(precision))
    assert measured == pytest.approx(expected, rel=1e-14)
    assert density == float("-inf")
    assert np.array_equal(mask, kept)


def test_rheplicants_inverse_variance_and_the_precision_operator_agree():
    """The weights themselves, not just the density they sum to.

    ``inverse_variance`` is the quantity every weighted solve in rheplicant
    reads; ``Precision.apply`` is the same quantity here, expressed as an
    operator so a correlated covariance costs nothing structural. On a
    diagonal they must coincide, flagged samples included -- an exact 0
    rather than an underflowed denominator.
    """
    from rheplicant.inference.noise import (
        FlaggedNoise,
        HomoscedasticNoise,
        inverse_variance,
    )

    with jax.enable_x64(True):
        prediction, _, sigma = _fixture()
        flags = jnp.array(FLAGS)
        flagged = FlaggedNoise(base=HomoscedasticNoise(sigma=sigma), flags=flags)
        theirs = np.asarray(inverse_variance(flagged, prediction))
        from bayesmith.exact.precision import DiagonalPrecision

        precision = DiagonalPrecision(sigma=jnp.where(flags, jnp.inf, sigma))
        ours = np.asarray(precision.apply(jnp.ones_like(prediction)))
    assert np.array_equal(theirs, ours)
    assert np.array_equal(theirs == 0.0, np.asarray(FLAGS))


# --------------------------------------------------------------------------
# §四 4.2's other half: the SAMPLING distribution, not only the density.
# --------------------------------------------------------------------------


def test_constant_sigma_realises_bitwise_the_same_draw_as_the_node():
    """``d + sigma w`` both ways, same key, same numbers.

    Bitwise rather than distributional, because it is available: both are
    ``prediction + sigma * jax.random.normal(key, shape)``.
    """
    from rheplicant.inference.noise import HomoscedasticNoise

    with jax.enable_x64(True):
        prediction, _, sigma = _fixture()
        key = jax.random.key(3)
        theirs = np.asarray(
            HomoscedasticNoise(sigma=sigma).realise(prediction, key=key)
        )
        ours = np.asarray(
            prediction + sigma * jax.random.normal(key, jnp.shape(prediction))
        )
    assert np.array_equal(theirs, ours)


def test_the_radiometer_generator_differs_exactly_where_the_prediction_is_negative():
    """An intended difference, measured, with its sign and its size.

    rheplicant's ``RadiometerNoise.realise`` is MULTIPLICATIVE --
    ``d(1 + f w)`` -- and says why in its own docstring: ``sigma =
    |prediction| * f`` uses an absolute value that a *generator* must not,
    "and the two forms differ in sign wherever the prediction does". A node
    declaring ``Normal(mu, f|mu|)`` draws the additive form ``mu + f|mu| w``.

    So the two agree wherever the prediction is positive and the deviation
    is exactly **negated** wherever it is negative. Same distribution (``w``
    is symmetric, so ``-f|mu|w`` and ``+f|mu|w`` are equal in law); different
    realisation at a fixed key.

    Asserted as the exact reflection rather than as "they differ", so a
    change that broke the generator some OTHER way would not pass.
    """
    from rheplicant.inference.noise import RadiometerNoise

    with jax.enable_x64(True):
        prediction, _, _ = _fixture()
        key = jax.random.key(3)
        noise = RadiometerNoise(
            channel_width=CHANNEL_WIDTH, integration_time=INTEGRATION_TIME
        )
        draw = jax.random.normal(key, jnp.shape(prediction))
        theirs = np.asarray(noise.realise(prediction, key=key))
        ours = np.asarray(prediction + FRACTIONAL * jnp.abs(prediction) * draw)
        centre = np.asarray(prediction)
    negative = centre < 0.0
    assert negative.any() and (~negative).any(), "the fixture must cross zero"
    assert np.array_equal(theirs[~negative], ours[~negative])
    # Mirrored about the prediction, exactly.
    assert theirs[negative] == pytest.approx(
        2.0 * centre[negative] - ours[negative], rel=1e-14
    )


def test_flagging_changes_the_covariance_and_not_the_generator():
    """``FlaggedNoise.realise`` is the base model's draw, unchanged.

    Its docstring's reason is the one worth cross-checking: flags say a
    sample was not OBSERVED, not that it had no true value. Drawing at
    ``sigma=inf`` would produce a data set no instrument could record, and
    every consumer that turns ``inf`` into a clean zero weight expects the
    datum underneath to be finite. Here that expectation is asserted rather
    than assumed.
    """
    from rheplicant.inference.noise import FlaggedNoise, HomoscedasticNoise

    with jax.enable_x64(True):
        prediction, _, sigma = _fixture()
        key = jax.random.key(3)
        base = HomoscedasticNoise(sigma=sigma)
        flagged = FlaggedNoise(base=base, flags=jnp.array(FLAGS))
        plain = np.asarray(base.realise(prediction, key=key))
        drawn = np.asarray(flagged.realise(prediction, key=key))
        std = np.asarray(flagged.std(prediction))
    assert np.array_equal(plain, drawn)
    assert np.all(np.isfinite(drawn))
    assert np.array_equal(np.isinf(std), np.asarray(FLAGS))


# --------------------------------------------------------------------------
# §二 step 3: refusal agreement for this row.
# --------------------------------------------------------------------------


def _refusal(fn) -> str:
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 -- the class is what is under test
        return f"{type(exc).__name__}: {exc}"
    return ""


@pytest.mark.parametrize("bad", [0.0, -2.0, float("nan")])
def test_a_sigma_that_is_not_strictly_positive_is_refused_on_both_sides(bad):
    """A zero, negative or NaN sigma is an infinite, negative or unusable
    weight -- never a tight constraint.

    rheplicant refuses it where the model is CONSTRUCTED
    (``RadiometerNoise.__check_init__``, and ``check_gaussian``'s equivalent
    on this side runs where the block is built). Both refuse; what is
    compared is that neither produces a number.

    NaN is in the sweep because it is the value that defeats a
    comparison-based guard: ``nan > 0`` and ``nan < 0`` are both False, so a
    guard written as a single inequality lets it through.
    """
    from rheplicant.core.errors import StateValidationError
    from rheplicant.inference.noise import RadiometerNoise

    from bayesmith.errors import StructureError
    from bayesmith.exact.gaussian import check_gaussian
    from bayesmith.graph.trace import observe, trace

    with jax.enable_x64(True):
        prediction, observed, sigma = _fixture()
        broken = sigma.at[2].set(bad)

        def model():
            observe(
                "d",
                lambda: dist.Normal(prediction, broken).to_event(1),
                obs=observed,
            )

        graph = trace(model)
        with pytest.raises(StructureError):
            check_gaussian(graph, graph.nodes[0], {})
        # rheplicant's own constructor-time guard, on the same three values.
        with pytest.raises(StateValidationError):
            RadiometerNoise(channel_width=bad, integration_time=1.0)


def test_the_scale_refusal_names_the_entry_that_offends():
    """The evidence a refusal offers has to point at the fault.

    Measured while writing this row: on ``[1, 1, inf, 1, inf, 1]`` the
    message read *"a scale that is not strictly positive and finite (min
    1)"* -- and ``1`` is strictly positive and finite. The sentence was
    right and the number beside it contradicted the sentence, which is worse
    than no number: a reader checks the evidence, sees it exonerate the
    scale, and concludes the guard is broken.

    An infinite sigma reaching this guard is not hypothetical. It is exactly
    how a flagged sample is spelled one layer down
    (``evidence/compress.py``'s ``observed_mask``), so a modeller who tries
    to declare flagging ON THE GRAPH lands here, and this message is the
    only thing that tells them where the concept actually lives.
    """
    from bayesmith.exact.gaussian import check_gaussian
    from bayesmith.graph.trace import observe, trace

    with jax.enable_x64(True):
        prediction, observed, sigma = _fixture()
        with_inf = jnp.where(jnp.array(FLAGS), jnp.inf, sigma)

        def model():
            observe(
                "d",
                lambda: dist.Normal(prediction, with_inf).to_event(1),
                obs=observed,
            )

        graph = trace(model)
        text = _refusal(lambda: check_gaussian(graph, graph.nodes[0], {}))
    assert "StructureError" in text, text
    # The COUNT and the OFFENDER's position, both derived from the fixture:
    # FLAGS drops samples 2 and 4 of 6, so the first offender is at flat
    # index 2 and there are two of them.
    assert f"{sum(FLAGS)} of {len(FLAGS)} entries" in text, text
    assert f"at flat index {FLAGS.index(True)}" in text, text
    assert "inf at flat index" in text, text
    # No minimum, in any form. Asserted as the absence of the WORD rather
    # than of a particular value: the first draft of this test asserted
    # `"min 1" not in text`, a literal carried over from a probe whose
    # sigma was all ones. On this fixture the reverted message reads
    # "min 0.5", so the mutation SURVIVED a test written to kill it -- the
    # assertion pinned a number nothing here produces.
    assert "min " not in text, text
    # And it must send the reader to the layer that owns "unobserved".
    assert "unobserved" in text.lower(), text


def test_a_mask_that_cannot_line_up_with_the_data_is_refused_on_both_sides():
    """rheplicant's own shape guard, and where this side's refusal comes from.

    On the graph there is no separate ``flags=`` argument to mis-shape: the
    sigma expression IS where a sample is declared unobserved. So the two
    shape faults arrive at different layers here, and both refuse -- but
    only one of them names the node:

    ======================  ==============================================
    fault                   what the modeller reads
    ======================  ==============================================
    sigma length != node    a raw ``ValueError`` from JAX broadcasting,
                            inside ``dist_fn``: *"Incompatible shapes for
                            broadcasting: shapes=[(), (6,), (7,)]"*. Names
                            both shapes; names no node.
    data length != loc      ``node_shape``'s own ``StructureError``, which
                            names the node and all three sources.
    ======================  ==============================================

    Recorded as a **gap, not a difference**: on a one-node graph the raw
    error is perfectly usable, and on a twenty-node graph it is not.
    Closing it means wrapping ``apply_probabilistic``, which is the hot path
    ``log_joint`` and the bridge both run under trace, so it is not this
    row's to do. Asserted as it stands so the next reader sees the state
    rather than assuming the named message covers both.

    ``trace()`` does NOT evaluate ``dist_fn`` -- it records the node -- so
    neither refusal happens at declaration. Measured, after a first draft of
    this test asserted it did.
    """
    from rheplicant.inference.noise import FlaggedNoise, HomoscedasticNoise

    from bayesmith.exact.gaussian import node_shape
    from bayesmith.graph.trace import observe, trace

    with jax.enable_x64(True):
        prediction, observed, sigma = _fixture()
        wrong = jnp.zeros(len(FLAGS) + 1, dtype=bool)
        flagged = FlaggedNoise(base=HomoscedasticNoise(sigma=sigma), flags=wrong)
        text = _refusal(lambda: flagged.std(prediction))
        assert "StateValidationError" in text, text
        assert "does not match the prediction shape" in text

        def mis_shaped_sigma():
            observe(
                "d",
                lambda: dist.Normal(prediction, jnp.ones(len(FLAGS) + 1)).to_event(1),
                obs=observed,
            )

        graph = trace(mis_shaped_sigma)  # the declaration itself is accepted
        raw = _refusal(lambda: node_shape(graph, graph.nodes[0], {}))

        def mis_shaped_data():
            observe(
                "d",
                lambda: dist.Normal(prediction, sigma).to_event(1),
                obs=jnp.zeros(len(FLAGS) + 1),
            )

        graph = trace(mis_shaped_data)
        named = _refusal(lambda: node_shape(graph, graph.nodes[0], {}))

    # Both refuse. Only one names the node -- pinned in both directions, so
    # closing the gap shows up here as a red test rather than silently.
    assert "ValueError" in raw and "StructureError" not in raw, raw
    assert "(6,)" in raw and "(7,)" in raw, raw
    assert "'d'" not in raw, raw
    assert "StructureError" in named, named
    assert "'d'" in named, named
