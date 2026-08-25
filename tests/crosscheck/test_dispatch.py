"""``plan.py`` + ``engines.py`` against the dispatch layer: the §四 4.2 row.

The row asks for ``plan.estimate`` value-for-value on one partition and one
toy model, and for ``plan.sample`` compared by posterior MOMENTS -- a chi²
trace is not comparable across NUTS implementations. It also carries the
ordering constraint **"先落 B1"**, without which "the comparison fixes the
GLS-type target as the reference".

That constraint turned out not to be reachable here, and the measurement is
this row's main result. B1 is a property of the BLOCK TYPE, not of the exit:

* on a **conjugate** block both packages land on the unbiased estimator and
  agree to 9e-12, because frozen-sigma reweighting's fixed point is the
  unbiased one on both sides;
* on a **gradient** block rheplicant lands on the GLS-type target -- 6.2483
  against the unbiased 5.1046 -- which is B1, live, at the dispatch layer;
* and bayesmith has no gradient-block point estimate to compare it against.
  A non-linear graph goes to NUTS, whose ``Normal(mu, sigma)`` carries its
  own ``-log sigma``. So the trap the row warns about cannot be sprung: it
  needs a second place that could drop the log-determinant, and there is
  not one.
"""

from __future__ import annotations

# Module scope, load-bearing: see ``test_diagnose_identifiability``.
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

pytestmark = pytest.mark.crosscheck

N_TIME, N_FREQ = 6, 4
NOISE_STD = 0.5
PRIOR_STD = 50.0

#: Draws for the moment comparison, warmup excluded. Fixed key on both
#: sides, so every z-score below is deterministic.
N_DRAWS = 800


def _arrays():
    """A quadratic spectrum, replicated over time, plus noise. Inside x64."""
    x = jnp.linspace(-1.0, 1.0, N_FREQ)
    basis = jnp.stack([x**k for k in range(3)], axis=1)
    coeff = jnp.array([10.0, -2.0, 0.5])
    truth = jnp.broadcast_to(basis @ coeff, (N_TIME, N_FREQ))
    data = truth + NOISE_STD * jax.random.normal(jax.random.key(11), (N_TIME, N_FREQ))
    return basis, data


def _rheplicant_plan(basis, *, degenerate: bool = False):
    """``(plan, pipeline, template)`` on rheplicant's own vocabulary."""
    from rheplicant import Coordinates, State
    from rheplicant.core.operator import AbstractOperator
    from rheplicant.core.pipeline import Pipeline
    from rheplicant.inference import Bind, Latent, ParameterSpace
    from rheplicant.inference.plan import Block, SamplingPlan

    class Sky(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
        provides: ClassVar[tuple[str, ...]] = ("data",)
        spectrum: jax.Array

        def __call__(self, state):
            return state.with_data(jnp.broadcast_to(self.spectrum, (N_TIME, N_FREQ)))

    pipeline = Pipeline(Sky(spectrum=jnp.zeros(N_FREQ)), names=("sky",))
    template = State(
        data=jnp.zeros((N_TIME, N_FREQ)),
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 80e6, N_FREQ),
        ),
    )
    if degenerate:
        # a + b reaches the prediction only as a sum: nullity 4 of 8.
        space = ParameterSpace(
            latents=[
                Latent(
                    name,
                    init=jnp.zeros(N_FREQ),
                    linear=True,
                    prior=dist.Normal(jnp.zeros(N_FREQ), PRIOR_STD),
                )
                for name in ("a", "b")
            ],
            bindings=[
                Bind(
                    ("a", "b"),
                    into=lambda p: p["sky"].spectrum,
                    fn=lambda x, y: x + y,
                )
            ],
        )
        return SamplingPlan(space, Block("a", "b")), pipeline, template

    space = ParameterSpace(
        latents=[
            Latent(
                "coeff",
                init=jnp.zeros(3),
                linear=True,
                prior=dist.Normal(jnp.zeros(3), PRIOR_STD),
            )
        ],
        bindings=[
            Bind("coeff", into=lambda p: p["sky"].spectrum, fn=lambda c: basis @ c)
        ],
    )
    return SamplingPlan(space, Block("coeff")), pipeline, template


def _bayesmith_plan(basis, data, *, degenerate: bool = False):
    from bayesmith import compile as compile_graph
    from bayesmith import det, observe, sample, trace

    if degenerate:

        def model():
            a = sample(
                "a", lambda: dist.Normal(jnp.zeros(N_FREQ), PRIOR_STD).to_event(1)
            )
            b = sample(
                "b", lambda: dist.Normal(jnp.zeros(N_FREQ), PRIOR_STD).to_event(1)
            )
            pred = det(
                "pred",
                lambda x, y: jnp.broadcast_to(x + y, (N_TIME, N_FREQ)),
                a,
                b,
                linear_in=("a", "b"),
            )
            observe(
                "d",
                lambda m: dist.Normal(m, NOISE_STD).to_event(2),
                pred,
                depends_on_prediction=False,
                obs=data,
            )

    else:

        def model():
            c = sample(
                "coeff", lambda: dist.Normal(jnp.zeros(3), PRIOR_STD).to_event(1)
            )
            pred = det(
                "pred",
                lambda v: jnp.broadcast_to(basis @ v, (N_TIME, N_FREQ)),
                c,
                linear_in=("coeff",),
            )
            observe(
                "d",
                lambda m: dist.Normal(m, NOISE_STD).to_event(2),
                pred,
                depends_on_prediction=False,
                obs=data,
            )

    return compile_graph(trace(model))


def _dense_oracle(basis, data, *, degenerate: bool = False):
    """``(posterior mean, posterior covariance)`` in dense NumPy."""
    size = 2 * N_FREQ if degenerate else 3

    def mu(v):
        if degenerate:
            spectrum = v[:N_FREQ] + v[N_FREQ:]
        else:
            spectrum = np.asarray(basis) @ v
        return np.ravel(np.broadcast_to(spectrum, (N_TIME, N_FREQ)))

    offset = mu(np.zeros(size))
    design = np.stack([mu(np.eye(size)[i]) - offset for i in range(size)], axis=1)
    flat = np.ravel(np.asarray(data))
    inverse_noise = np.eye(flat.size) / NOISE_STD**2
    normal = design.T @ inverse_noise @ design + np.eye(size) / PRIOR_STD**2
    return (
        np.linalg.solve(normal, design.T @ inverse_noise @ (flat - offset)),
        np.linalg.inv(normal),
    )


def test_plan_estimate_agrees_value_for_value_on_the_same_partition():
    """§四 4.2's "``plan.estimate`` 逐值一致".

    Not bitwise, and honestly so: the two run different iteration schemes
    to the same fixed point (block-coordinate descent against a single
    reweighted solve), so they agree to float64 roundoff rather than to the
    bit. Measured max absolute difference **8.9e-15** on a quantity of
    order 10, i.e. ~1e-15 relative.
    """
    from rheplicant.inference.noise import HomoscedasticNoise

    with jax.enable_x64(True):
        basis, data = _arrays()
        plan, pipeline, template = _rheplicant_plan(basis)
        theirs = np.asarray(
            plan.estimate(
                pipeline, template, data, noise=HomoscedasticNoise(sigma=NOISE_STD)
            ).values["coeff"]
        )
        ours = np.asarray(_bayesmith_plan(basis, data).estimate().values["coeff"])
    assert ours == pytest.approx(theirs, rel=1e-12, abs=1e-13), (theirs, ours)


def test_both_estimates_match_a_dense_oracle():
    """Iron law 4. Both to ~1e-14 relative, which is the conjugate solve's
    own accuracy on this well-conditioned block, not a chosen tolerance."""
    from rheplicant.inference.noise import HomoscedasticNoise

    with jax.enable_x64(True):
        basis, data = _arrays()
        truth, _ = _dense_oracle(basis, data)
        plan, pipeline, template = _rheplicant_plan(basis)
        theirs = np.asarray(
            plan.estimate(
                pipeline, template, data, noise=HomoscedasticNoise(sigma=NOISE_STD)
            ).values["coeff"]
        )
        ours = np.asarray(_bayesmith_plan(basis, data).estimate().values["coeff"])
    for label, got in (("rheplicant", theirs), ("bayesmith", ours)):
        assert got == pytest.approx(truth, rel=1e-13), (label, got, truth)


@pytest.mark.slow
def test_plan_sample_reproduces_the_posterior_moments_on_both_sides():
    """§四 4.2's "``plan.sample`` 比后验矩".

    Compared to the DENSE ORACLE rather than to each other, and by moments
    rather than by trace: a chi² trace is not comparable across NUTS
    implementations, and two Gibbs sweeps over the same partition visit
    different states in a different order even at the same key.

    Thresholds are the design document's |z| < 4. Measured at 800
    post-warmup draws with fixed keys on both sides: mean |z| 0.89 and
    0.72, variance within 1.6% and 6.3% of the oracle against a sampling
    error of 5.0%.
    """
    from rheplicant.inference.noise import HomoscedasticNoise

    with jax.enable_x64(True):
        basis, data = _arrays()
        truth, covariance = _dense_oracle(basis, data)
        plan, pipeline, template = _rheplicant_plan(basis)
        theirs = np.asarray(
            plan.sample(
                pipeline,
                template,
                data,
                noise=HomoscedasticNoise(sigma=NOISE_STD),
                key=jax.random.key(5),
                n_sweeps=2 * N_DRAWS,
                warmup=N_DRAWS,
            ).samples["coeff"]
        )
        ours = np.asarray(
            _bayesmith_plan(basis, data)
            .sample(jax.random.key(5), num_samples=N_DRAWS, num_warmup=N_DRAWS // 2)
            .samples["coeff"]
        )
    variance = np.diag(covariance)
    for label, drawn in (("rheplicant", theirs), ("bayesmith", ours)):
        assert drawn.shape[0] == N_DRAWS, (label, drawn.shape)
        z_mean = np.max(
            np.abs(drawn.mean(axis=0) - truth) / np.sqrt(variance / N_DRAWS)
        )
        assert z_mean < 4.0, (label, "mean", z_mean)
        sample_var = np.diag(np.cov(drawn, rowvar=False))
        z_var = np.max(
            np.abs(sample_var - variance) / (variance * np.sqrt(2.0 / (N_DRAWS - 1)))
        )
        assert z_var < 4.0, (label, "variance", z_var)


# --------------------------------------------------------------------------
# B1 at the dispatch layer: which block type actually drops the
# log-determinant, and why the row's ordering warning cannot bite here.
# --------------------------------------------------------------------------

B1_N = 40
B1_KAPPA = 0.5
B1_W = 5.0
B1_PRIOR_STD = 100.0


def _b1_arrays():
    """``mu = w x`` with ``sigma = kappa |mu|`` -- the model §三 B1 works in."""
    x = jnp.linspace(1.0, 3.0, B1_N)
    truth = B1_W * x
    sigma = B1_KAPPA * jnp.abs(truth)
    data = truth + sigma * jax.random.normal(jax.random.key(3), (B1_N,))
    return x, data


def _b1_closed_forms(x, data):
    """The two estimators §三 B1 names, in closed form.

    ``sum d^2/x^2 / sum d/x`` is the argmax with the log-determinant
    DROPPED -- the GLS-type target -- and ``mean(d/x)`` is the argmax with
    it KEPT, which is exactly unbiased. Written in NumPy from the algebra,
    so neither package supplies the reference.
    """
    xs, ds = np.asarray(x, dtype=float), np.asarray(data, dtype=float)
    dropped = float(np.sum(ds**2 / xs**2) / np.sum(ds / xs))
    kept = float(np.mean(ds / xs))
    return dropped, kept


def _b1_rheplicant(x, data, *, nonlinear: bool):
    from rheplicant import Coordinates, State
    from rheplicant.core.operator import AbstractOperator
    from rheplicant.core.pipeline import Pipeline
    from rheplicant.inference import Bind, Latent, ParameterSpace
    from rheplicant.inference.noise import RadiometerNoise
    from rheplicant.inference.plan import Block, SamplingPlan

    class Line(AbstractOperator):
        requires: ClassVar[tuple[str, ...]] = ("coords.time",)
        provides: ClassVar[tuple[str, ...]] = ("data",)
        w: jax.Array

        def __call__(self, state):
            value = jnp.exp(self.w) if nonlinear else self.w
            return state.with_data(value * x[None, :])

    pipeline = Pipeline(Line(w=jnp.array(1.0)), names=("line",))
    space = ParameterSpace(
        latents=[
            Latent(
                "w",
                init=jnp.array(1.0),
                linear=not nonlinear,
                prior=dist.Normal(0.0, B1_PRIOR_STD),
            )
        ],
        bindings=[Bind("w", into=lambda p: p["line"].w)],
    )
    template = State(
        data=jnp.zeros((1, B1_N)),
        coords=Coordinates(time=jnp.arange(1.0), freq=jnp.linspace(60e6, 80e6, B1_N)),
    )
    plan = SamplingPlan(space, Block("w"))
    estimate = plan.estimate(
        pipeline,
        template,
        data[None, :],
        noise=RadiometerNoise(channel_width=1.0 / B1_KAPPA**2, integration_time=1.0),
        max_iter=60,
    )
    value = float(np.asarray(estimate.values["w"]))
    return (float(np.exp(value)) if nonlinear else value), estimate.diagnostics


def _b1_bayesmith(x, data, *, nonlinear: bool):
    from bayesmith import compile as compile_graph
    from bayesmith import const, det, observe, sample, trace

    def model():
        xs = const("X", x)
        w = sample("w", lambda: dist.Normal(0.0, B1_PRIOR_STD))
        mu = det(
            "mu",
            (lambda w_, x_: jnp.exp(w_) * x_)
            if nonlinear
            else (lambda w_, x_: w_ * x_),
            w,
            xs,
            **({} if nonlinear else {"linear_in": ("w",)}),
        )
        observe(
            "d",
            lambda m: dist.Normal(m, B1_KAPPA * jnp.abs(m) + 1e-9),
            mu,
            depends_on_prediction=True,
            obs=data,
        )

    return compile_graph(trace(model))


def test_a_conjugate_block_lands_on_the_unbiased_side_on_both__sides():
    """B1 does NOT bite a conjugate block, and this is the measurement.

    §三 B1's own analysis says frozen-sigma reweighting's fixed point is the
    unbiased estimator -- it says it about bayesmith's ``iterative_gls``.
    Measured here, it is equally true of rheplicant's CONJUGATE block, on a
    prediction-dependent sigma where the two estimators are 22% apart:

    ==============================  ==========
    closed form, log-det kept       5.104641
    closed form, log-det dropped    6.258841
    rheplicant, conjugate block     5.104558
    bayesmith, same model           5.104558
    ==============================  ==========

    The two packages agree to 9e-12 and BOTH sit on the unbiased side. So
    the row's ordering warning -- "先落 B1, or the comparison fixes the
    GLS-type target as the reference" -- is not reachable through this
    door. The anti-vacuity clause is the distance to the other estimator:
    without it, agreeing on 6.2588 would look identical to agreeing on
    5.1046.
    """
    with jax.enable_x64(True):
        x, data = _b1_arrays()
        dropped, kept = _b1_closed_forms(x, data)
        theirs, _ = _b1_rheplicant(x, data, nonlinear=False)
        ours = float(
            np.asarray(_b1_bayesmith(x, data, nonlinear=False).estimate().values["w"])
        )
    # The two estimators must be far apart, or nothing below discriminates.
    assert dropped / kept == pytest.approx(1.0 + B1_KAPPA**2, rel=0.03), (
        dropped,
        kept,
    )
    assert theirs == pytest.approx(ours, rel=1e-10), (theirs, ours)
    for label, got in (("rheplicant", theirs), ("bayesmith", ours)):
        assert got == pytest.approx(kept, rel=1e-4), (label, got, kept)
        assert abs(got - dropped) > 0.15 * kept, (label, "landed on the GLS target")


def test_a_gradient_block_lands_on_the_gls_target_and_here_there_is_no_door():
    """B1, live, at the dispatch layer -- and why this row cannot fix it in.

    Same noise law, latent made non-linear (``mu = exp(w) x``) so
    rheplicant takes its GRADIENT block. Measured: **6.2483**, against the
    unbiased 5.1046 and the GLS-type closed form 6.2588. That is §三 B1's
    "``plan.sample``'s gradient block targets a GLS-type objective",
    confirmed on the ESTIMATE exit too -- so the property belongs to the
    BLOCK TYPE, not to the exit, which is a sharper statement than the
    spec's.

    bayesmith has nothing to compare: a non-linear graph has no exact
    subgraph, so ``estimate()`` refuses by name and points at ``sample()``,
    which goes through NumPyro, whose ``Normal(mu, sigma)`` carries its own
    ``-log sigma``. There is no second place here that could drop the
    log-determinant, so the reference cannot be fixed to the wrong target
    by this comparison -- not because the comparison was careful, but
    because one side of it does not exist.
    """
    with jax.enable_x64(True):
        x, data = _b1_arrays()
        dropped, kept = _b1_closed_forms(x, data)
        theirs, diagnostics = _b1_rheplicant(x, data, nonlinear=True)
        plan = _b1_bayesmith(x, data, nonlinear=True)
        with pytest.raises(NotImplementedError, match="no point estimate"):
            plan.estimate()
    assert diagnostics.engines == {("w",): "gradient"}, diagnostics.engines
    # On the GLS-type side, and nowhere near the unbiased one.
    assert theirs == pytest.approx(dropped, rel=0.01), (theirs, dropped)
    assert abs(theirs - kept) > 0.15 * kept, (theirs, kept)
    # And the plan says NUTS, which is the exit that carries the log-det.
    assert "NUTS" in str(plan)


# --------------------------------------------------------------------------
# §二 step 3: refusal agreement.
# --------------------------------------------------------------------------


def test_a_rank_deficient_partition_is_refused_there_and_answered_here():
    """The row's intended difference, with an oracle behind it.

    rheplicant refuses: *"its joint Jacobian has nullity 4 of 8 parameters,
    so that many independent directions leave the prediction unchanged and
    any answer along them is arbitrary"*. On the graph the answer is **not**
    arbitrary -- a proper prior makes the posterior proper along the null
    direction, and what comes back is its mean. Measured against dense
    NumPy: identical to every digit, with ``a`` and ``b`` split evenly as
    equal priors require. This is §5.19 recurring one layer up: a
    rank-deficient OBSERVED Jacobian is not an undefined posterior.

    The conditioning is reported rather than hidden -- the plan prints
    ``kappa=120001`` against a true condition number of 120001.0000 -- so
    the modeller is told, in the number they are told to divide a tolerance
    by, exactly how badly the data constrains the split.
    """
    from rheplicant.core.errors import ParameterSpaceError
    from rheplicant.inference.noise import HomoscedasticNoise

    with jax.enable_x64(True):
        basis, data = _arrays()
        truth, _ = _dense_oracle(basis, data, degenerate=True)
        plan, pipeline, template = _rheplicant_plan(basis, degenerate=True)
        with pytest.raises(ParameterSpaceError, match="nullity"):
            plan.estimate(
                pipeline, template, data, noise=HomoscedasticNoise(sigma=NOISE_STD)
            )
        ours = _bayesmith_plan(basis, data, degenerate=True)
        estimate = ours.estimate()
        got = np.concatenate(
            [np.asarray(estimate.values["a"]), np.asarray(estimate.values["b"])]
        )
        text = str(ours)
    assert got == pytest.approx(truth, rel=1e-10), (got, truth)
    # Equal priors, so the sum is split evenly -- the property, not the value.
    assert got[:N_FREQ] == pytest.approx(got[N_FREQ:], rel=1e-12)
    # And the conditioning is on the face of the plan.
    assert "kappa=" in text, text


def test_rheplicants_plan_now_attributes_b1_to_the_block_type():
    """The other half of a cross-repository record, so neither can go stale
    alone.

    This row measured that B1 attaches to which ENGINE ran rather than to
    which exit was called, and e-RHINO's ``7f03af1`` rewrote
    ``inference/plan.py``'s module docstring around that, carrying the two
    numbers and crediting this file for them.

    Asserted here rather than trusted, because a docstring is the one kind
    of claim nothing else executes -- which is exactly how
    ``condition_estimate`` came to open with a paragraph describing a
    different function.

    **The paragraph is on e-RHINO's ``track-a-tail`` branch and NOT on its
    ``main``**, measured 2026-08-25. So this guard reads whatever the
    editable install has checked out, and a red here means one of three
    things, in decreasing order of likelihood: that branch is not the one
    checked out, the branch was dropped in review, or the docstring was
    edited. The first is not a defect in either package; the second and
    third mean ``docs/migration/plan.md`` §5(a)'s "Carried upstream"
    paragraph has become false and is what needs changing.

    Recorded rather than softened into a skip: a guard that cannot fail is
    worse than one that fails for a reason the message explains.
    """
    import rheplicant.inference.plan as upstream

    text = upstream.__doc__ or ""
    assert "It is the BLOCK TYPE that decides, not the exit" in text, (
        "rheplicant's plan.py does not attribute B1 to the block type. If "
        "e-RHINO is checked out on `main`, that is expected -- the paragraph "
        "is on `track-a-tail` and unmerged as of 2026-08-25. Otherwise the "
        "branch was dropped or the docstring changed, and "
        "docs/migration/plan.md section 5(a) is what needs updating."
    )
    # The numbers this row supplied, still the ones it is arguing from.
    assert "5.104558" in text and "6.248269" in text, text[:200]
