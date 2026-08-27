"""G1 -- an unobserved sample informs nothing, on every exact route.

The gap this closes, in one sentence: rheplicant spells a flagged sample
``sigma = inf`` and every one of its consumers turns that into a clean zero
weight, while a graph node could not, because ``Normal(mu, inf)`` has
log-density ``-inf`` everywhere and takes the whole joint with it.

**What is asserted here is an EQUIVALENCE, not a table of numbers.** A masked
model and the model over the samples that were actually taken are the same
posterior, so every route is checked against its own sub-problem rather than
against a pinned constant. A pinned constant moves when the fixture moves and
says nothing about whether the mask is being honoured; the equivalence says
exactly that and nothing else.

Every equivalence test here has a sibling that shows the **unmasked** answer
is far away, because an equivalence between two things that are equal for a
different reason is not evidence. The flagged entries of
``tests.exact.models.flagged_line`` carry 1e3 for that purpose.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.gaussian import noise_std_at, precision_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.precision import (
    DiagonalPrecision,
    MaskedPrecision,
    masked,
    per_sample_sigma,
)
from bayesmith.exact.solve import gcr_sample, wiener_solve
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.nodes import Probabilistic
from tests.exact.models import flagged_line

AT = {"a": 0.0, "b": 0.0}


def _unmasked(graph):
    """The same graph with its mask struck out -- the wrong answer, on purpose.

    Built by ``eqx.tree_at``-style replacement of the one node rather than by
    re-tracing the model, so it differs from ``masked`` in exactly the field
    under test and in nothing else.
    """
    nodes = tuple(
        Probabilistic(
            name=node.name,
            parents=node.parents,
            plate=node.plate,
            dist_fn=node.dist_fn,
            observed=node.observed,
            support=node.support,
            depends_on_prediction=node.depends_on_prediction,
            observed_mask=None,
        )
        if isinstance(node, Probabilistic) and node.observed_mask is not None
        else node
        for node in graph.nodes
    )
    return type(graph)(nodes=nodes, plates=graph.plates)


def _mean(graph, at=None):
    values = AT if at is None else at
    block = linear_operator(graph, ("a", "b"))
    return wiener_solve(block, precision=precision_at(graph, values))[0]


def _flat(value):
    return np.array([float(value["a"]), float(value["b"])])


# --------------------------------------------------------------------------
# The solve
# --------------------------------------------------------------------------


def test_the_masked_posterior_mean_is_the_sub_problems():
    """The independent oracle: masking a sample is deleting it.

    Nothing in the two routes is shared below ``wiener_solve`` -- one solves
    an 8-row system with two rows given zero weight, the other a 6-row system
    that never had them -- so agreement is a statement about the mask and not
    about a shared helper.
    """
    graph, kept, _ = flagged_line()
    difference = np.max(np.abs(_flat(_mean(graph)) - _flat(_mean(kept))))
    assert difference < 1e-12, difference


def test_ignoring_the_mask_would_move_the_answer():
    """The sibling that makes the test above able to fail.

    Without it, a ``MaskedPrecision`` that silently degenerated to its base
    would still pass every equivalence here IF the flagged data happened to
    lie on the line. It does not: the flagged entries are 1e3.
    """
    graph, kept, _ = flagged_line()
    honoured = _flat(_mean(graph))
    ignored = _flat(_mean(_unmasked(graph)))
    assert np.max(np.abs(honoured - ignored)) > 100.0
    assert np.max(np.abs(_flat(_mean(kept)) - ignored)) > 100.0


def test_a_nan_at_a_masked_sample_does_not_reach_the_solution():
    """A flagged datum is routinely ``nan``; ``0 * nan`` is ``nan``.

    So the residual is zeroed BEFORE the covariance is applied, not only
    after. Masking only the output leaves one flagged channel able to poison
    the whole solve while every weight looks right.
    """
    graph, kept, mask = flagged_line()
    node = graph.node("d")
    poisoned = jnp.where(mask, node.observed, jnp.nan).at[5].set(jnp.inf)
    nodes = tuple(
        Probabilistic(
            name=n.name,
            parents=n.parents,
            plate=n.plate,
            dist_fn=n.dist_fn,
            observed=poisoned,
            support=n.support,
            depends_on_prediction=n.depends_on_prediction,
            observed_mask=n.observed_mask,
        )
        if n.name == "d"
        else n
        for n in graph.nodes
    )
    sick = type(graph)(nodes=nodes, plates=graph.plates)
    answer = _flat(_mean(sick))
    assert np.all(np.isfinite(answer))
    assert np.max(np.abs(answer - _flat(_mean(kept)))) < 1e-12


def test_the_sigma_at_a_masked_sample_does_not_matter():
    """Nothing about an unobserved sample can move the answer -- not even its width.

    This is what "contributes nothing" means, said as a property rather than
    as a number, and it is the assertion a mask that reached only THREE of the
    four operations would fail.
    """
    graph, _, mask = flagged_line()
    at = AT
    block = linear_operator(graph, ("a", "b"))
    precision = precision_at(graph, at)
    assert isinstance(precision["d"], MaskedPrecision)
    base = precision["d"].base.sigma
    wilder = {
        "d": MaskedPrecision(
            base=DiagonalPrecision(sigma=jnp.where(mask, base, 1e-9)),
            seen=precision["d"].seen,
        )
    }
    first = wiener_solve(block, precision=precision)[0]
    second = wiener_solve(block, precision=wilder)[0]
    assert np.max(np.abs(_flat(first) - _flat(second))) < 1e-12


def test_the_quadratic_form_ignores_a_non_finite_datum_at_a_masked_sample():
    """The multiplication lives in ``quadratic``, not in ``apply``.

    ``apply`` returns a clean zero at a masked sample -- ``jnp.where`` selects
    -- but ``quadratic`` forms ``residual * apply(residual)``, and ``nan * 0``
    is ``nan``. Measured before the guard: ``apply`` gave ``[4, 8, 0, 12]`` and
    ``quadratic`` gave ``nan`` on the same input, so a masked model with a
    clean solve still handed back a ``nan`` log-density.
    """
    from bayesmith.exact.precision import log_density, quadratic

    seen = jnp.array([True, True, False, True])
    precision = MaskedPrecision(
        base=DiagonalPrecision(sigma=jnp.full((4,), 0.5)), seen=seen
    )
    poisoned = jnp.array([1.0, 2.0, jnp.nan, 3.0])
    sub = DiagonalPrecision(sigma=jnp.full((3,), 0.5))
    assert float(quadratic(precision, poisoned)) == float(
        quadratic(sub, jnp.array([1.0, 2.0, 3.0]))
    )
    assert np.isfinite(float(log_density(precision, poisoned)))


def test_the_gradient_through_the_guard_is_finite_and_right():
    """A finite value with a ``nan`` gradient is this codebase's recurring defect.

    ``normal_operator`` differentiates ``quadratic`` -- that IS the curvature --
    so a guard that fixed the value and left the derivative poisoned would look
    fixed from every forward assertion and break only the solve. ``jnp.where``
    is also the classic place for it: the branch it does not take still runs
    its own VJP, and ``0 * nan`` there is ``nan``.

    Checked against the closed form rather than against "is finite" alone. At
    ``x = 1`` the residual is ``[0, -1, nan, -2]``, the masked entry drops, and
    ``d/dx sum r**2/sigma**2 = 2 (0 - 1 - 2) / 0.25 = -24``.
    """
    from bayesmith.exact.precision import quadratic

    precision = MaskedPrecision(
        base=DiagonalPrecision(sigma=jnp.full((4,), 0.5)),
        seen=jnp.array([True, True, False, True]),
    )
    data = jnp.array([1.0, 2.0, jnp.nan, 3.0])

    def chi2(x):
        return quadratic(precision, x * jnp.ones(4) - data)

    assert float(chi2(1.0)) == 20.0
    assert float(jax.grad(chi2)(1.0)) == -24.0


def test_the_quadratic_form_still_propagates_a_nan_that_WAS_observed():
    """The guard is on the weight, not on the residual, and that is the point.

    Mapping a ``nan`` at a sample that was taken to zero would MEAN
    "unobserved", which is a claim only the model may make. A poisoned datum
    the model believes it recorded has to stay loud -- the same reason the
    sibling package keeps two weight formulas that disagree on ``nan``.
    """
    from bayesmith.exact.precision import quadratic

    precision = MaskedPrecision(
        base=DiagonalPrecision(sigma=jnp.full((4,), 0.5)),
        seen=jnp.array([True, True, False, True]),
    )
    assert not np.isfinite(
        float(quadratic(precision, jnp.array([1.0, jnp.nan, 0.0, 3.0])))
    )


# --------------------------------------------------------------------------
# The draw
# --------------------------------------------------------------------------


def test_the_masked_gcr_draws_have_the_sub_problems_moments():
    """``whiten`` masks too, so the fluctuation term has no flagged noise in it.

    Moments rather than values: two correct implementations draw different
    numbers from the same key when the system they solve has a different
    number of rows.
    """
    graph, kept, _ = flagged_line()

    def draws(g, count=600):
        block = linear_operator(g, ("a", "b"))
        precision = precision_at(g, AT)
        keys = jax.random.split(jax.random.key(11), count)
        return np.stack(
            [_flat(gcr_sample(block, precision=precision, key=k)[0]) for k in keys]
        )

    here, there = draws(graph), draws(kept)
    spread = there.std(axis=0)
    assert np.all(np.abs(here.mean(axis=0) - there.mean(axis=0)) < 0.25 * spread)
    assert np.all(np.abs(here.std(axis=0) / spread - 1.0) < 0.15)


# --------------------------------------------------------------------------
# The density
# --------------------------------------------------------------------------


def test_the_masked_log_joint_drops_the_flagged_terms():
    """``log_joint`` is the joint over the samples that were taken.

    The two graphs' PRIOR terms are identical, so the difference between the
    joints is the likelihood's, which is what is being checked.
    """
    graph, kept, _ = flagged_line()
    values = {"a": 1.1, "b": -2.2}
    assert float(log_joint(graph, values)) == pytest.approx(
        float(log_joint(kept, values)), rel=1e-6
    )


def test_the_numpyro_model_masks_the_same_terms_as_the_joint():
    """``to_numpyro`` and ``log_joint`` are two scans of one graph.

    NUTS is the oracle every exact path is checked against, so a mask honoured
    by one and not the other would make every masked comparison in this file a
    comparison of two different posteriors -- silently, and in the direction
    that looks like a solver defect.

    The density rather than a chain: a handler that dropped the mask changes
    the potential at EVERY point, so one evaluation settles it and no sampler
    variance is involved.
    """
    from numpyro.infer.util import log_density

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    graph, kept, _ = flagged_line()
    values = {"a": 1.1, "b": -2.2}
    here, _ = log_density(to_numpyro(graph), (), {}, values)
    there, _ = log_density(to_numpyro(kept), (), {}, values)
    assert float(here) == pytest.approx(float(there), rel=1e-6)
    assert float(here) == pytest.approx(float(log_joint(graph, values)), rel=1e-6)


def test_an_unmasked_joint_is_dominated_by_the_flagged_samples():
    """The sibling: the flagged entries are 1e3, so leaving them in is loud."""
    graph, _, _ = flagged_line()
    values = {"a": 1.1, "b": -2.2}
    assert float(log_joint(_unmasked(graph), values)) < float(
        log_joint(graph, values)
    ) - 1e6


# --------------------------------------------------------------------------
# The two spellings of "not observed" agree
# --------------------------------------------------------------------------


def test_per_sample_sigma_reports_inf_where_masked():
    """The encoding survives at exactly one seam, and this is it.

    ``GLSResult.noise_std`` and ``Estimate.noise_std`` are both
    ``per_sample_sigma``, so a caller who spelled the mask ``sigma = inf``
    upstream reads the same word back.
    """
    graph, _, mask = flagged_line()
    sigma = per_sample_sigma(precision_at(graph, AT))["d"]
    assert np.array_equal(np.isfinite(np.asarray(sigma)), np.asarray(mask))


def test_the_masked_normaliser_is_the_kept_terms():
    """Bitwise, not to a tolerance: it is the same expression with terms dropped."""
    graph, _, mask = flagged_line()
    precision = precision_at(graph, AT)["d"]
    terms = precision.base.log_normalizer_terms()
    assert float(precision.log_normalizer()) == float(
        jnp.sum(jnp.where(mask, terms, 0.0))
    )


def test_compress_and_the_precision_agree_on_the_masked_normaliser():
    """Two layers computed this independently; they must not have drifted.

    ``evidence.compress`` masks by reading ``inf`` out of ``per_sample_sigma``
    and knows nothing about :class:`MaskedPrecision`. That it lands on the
    same number is the check that the encoding and the declaration describe
    one covariance rather than two.
    """
    from bayesmith.evidence.compress import observed_mask

    graph, _, mask = flagged_line()
    precision = precision_at(graph, AT)["d"]
    sigma = per_sample_sigma({"_": precision})["_"]
    seen = observed_mask(precision)
    assert np.array_equal(np.asarray(seen), np.asarray(mask))
    safe = jnp.where(seen, sigma, 1.0)
    compressed = jnp.sum(jnp.where(seen, jnp.log(2.0 * jnp.pi * safe**2), 0.0))
    assert float(compressed) == float(precision.log_normalizer())


def test_the_spectrum_of_a_masked_sample_is_a_constant():
    """``log lambda = 0`` there, so the variance-information term gets nothing.

    Not zero-by-accident: the Fisher matrix's second term is built from
    ``d log lambda_k``, and a masked sample must contribute no DERIVATIVE, not
    merely a small one. A constant is the only value that guarantees it,
    whatever the sigma expression does.
    """
    graph, _, mask = flagged_line()
    spectrum = precision_at(graph, AT)["d"].log_spectrum()
    assert np.all(np.asarray(spectrum)[~np.asarray(mask)] == 0.0)


# --------------------------------------------------------------------------
# What a mask may not be
# --------------------------------------------------------------------------


def test_a_latent_may_not_declare_a_mask():
    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe("d", lambda m: dist.Normal(m, 0.5), x, obs=jnp.zeros(3))

    graph = trace(model)
    latent = graph.node("x")
    with pytest.raises(GraphError, match="LATENT and declares an observed_mask"):
        type(graph)(
            nodes=(
                Probabilistic(
                    name=latent.name,
                    parents=latent.parents,
                    plate=latent.plate,
                    dist_fn=latent.dist_fn,
                    observed=None,
                    observed_mask=jnp.array([True]),
                ),
            )
            + graph.nodes[1:],
            plates=graph.plates,
        )


@pytest.mark.parametrize(
    ("mask", "pattern"),
    [
        (jnp.array([1.0, 0.0, 1.0]), "must be boolean"),
        (jnp.array([True, False]), "shape"),
    ],
)
def test_a_mask_that_is_not_a_selection_over_this_data_is_refused(mask, pattern):
    """A float mask multiplies; a broadcasting one masks other samples.

    Both come back finite and correctly shaped, which is why neither may be
    inferred from what happens to work.
    """

    def model():
        x = sample("x", lambda: dist.Normal(0.0, 1.0))
        observe(
            "d", lambda m: dist.Normal(m, 0.5), x, obs=jnp.zeros(3), mask=mask
        )

    with pytest.raises(GraphError, match=pattern):
        trace(model)


def test_masking_a_correlated_covariance_is_refused():
    """An unobserved sample inside a stationary covariance has no exact meaning.

    Measured in ``evidence/compress.py``: on a 6-point kernel with one sample
    dropped, the observed submatrix's log-determinant is -0.7084 while the
    closest subset sum of log-eigenvalues is 0.47 nats away. So the refusal is
    a statement about arithmetic, not a missing feature.
    """
    circulant = getattr(dist, "CirculantNormal", None)
    if circulant is None:  # pragma: no cover - numpyro always ships it here
        pytest.skip("numpyro has no CirculantNormal")
    kernel = jnp.array([1.0, 0.3, 0.1, 0.05, 0.1, 0.3])
    mask = jnp.array([True, True, False, True, True, True])

    def model():
        x = sample("x", lambda: dist.Normal(jnp.zeros(6), 3.0))
        observe(
            "d",
            lambda m: circulant(m, kernel),
            x,
            obs=jnp.zeros(6),
            mask=mask,
        )

    graph = trace(model)
    with pytest.raises(StructureError, match="no per-sample sigma"):
        precision_at(graph, {"x": jnp.zeros(6)})


def test_masked_refuses_a_mask_that_is_not_a_selection():
    """``masked`` guards its own arguments, not only ``precision_parts``' callers."""
    precision = DiagonalPrecision(sigma=jnp.full((4,), 0.5))
    with pytest.raises(StructureError, match="must be boolean"):
        masked(precision, jnp.ones((4,)))
    with pytest.raises(StructureError, match="shape"):
        masked(precision, jnp.ones((3,), dtype=bool))


# --------------------------------------------------------------------------
# Carried, not re-derived
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["multiplicative", "lognormal"])
def test_the_log_transform_carries_the_mask(kind):
    """``log_space`` rewrites the data and the scale; it must not drop the mask.

    A node that lost its mask here would solve with the flagged channels back
    in -- finitely, and wrongly, with the transform being the last place
    anyone would look.

    **Both scenarios, because they are two different rebuild sites.**
    ``log_space`` has one ``Probabilistic(...)`` reconstruction per scenario,
    and the first version of this test exercised only the multiplicative one:
    mutating the mask out of the LOGNORMAL site left the whole suite green.
    That survivor is the reason for the parametrisation, not tidiness.
    """
    from bayesmith.exact.loglinear import log_space

    x = jnp.linspace(1.0, 2.0, 6)
    mask = jnp.array([True, True, False, True, True, False])

    def model():
        coordinate = const("X", x)
        lam = sample("log_gain", lambda: dist.Normal(0.0, 1.0))
        if kind == "multiplicative":
            mu = det("mu", lambda l, x_: jnp.exp(l) * x_, lam, coordinate)
            observe(
                "d", lambda m: dist.Normal(m, 0.004 * m), mu, obs=2.0 * x, mask=mask
            )
        else:
            ell = det(
                "ell",
                lambda l, x_: l + jnp.log(x_),
                lam,
                coordinate,
                linear_in=("log_gain",),
            )
            observe(
                "d", lambda e: dist.LogNormal(e, 0.004), ell, obs=2.0 * x, mask=mask
            )

    transformed = log_space(trace(model))
    # The transform actually fired, and as the scenario this case is about. A
    # SKIPPED node keeps its original form, so a mask surviving an
    # untransformed node proves nothing about the rewrite -- this assertion is
    # what makes the next one about the rewrite.
    assert transformed.kind["d"] == kind
    assert np.array_equal(
        np.asarray(transformed.graph.node("d").observed_mask), np.asarray(mask)
    )


def test_noise_std_at_is_not_the_mask_seam():
    """Stated because the absence is deliberate, and absence says nothing.

    ``noise_std_at`` reports the scale the node's own distribution declares,
    which is finite at a masked sample -- the mask is not in the distribution.
    ``per_sample_sigma`` is the seam that reports ``inf``, because it reads the
    COVARIANCE the solve was given. A reader who assumed the two agreed would
    conclude the mask was being dropped.
    """
    graph, _, mask = flagged_line()
    declared = np.asarray(noise_std_at(graph, AT)["d"])
    assert np.all(np.isfinite(declared))
    operator = np.asarray(per_sample_sigma(precision_at(graph, AT))["d"])
    assert np.array_equal(np.isfinite(operator), np.asarray(mask))
