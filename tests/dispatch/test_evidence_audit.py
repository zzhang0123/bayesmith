"""R4 Task 4 -- is this prior proper, and is it normalised, and are those two.

``Z = INT p(d | theta) p(theta) d theta`` is a number only if ``p(theta)`` has
finite mass, and it is THE number only if that mass is one. Those are two
questions and the audit answers them separately, because a prior can fail
either alone: ``dist.Uniform(-2, 5)`` is proper and normalised; an
``ImproperUniform`` over the same interval is proper in mass and unnormalised
in density; an ``ImproperUniform`` over the line is neither.

**Nothing here may read a type name.** ``isinstance(d, dist.ImproperUniform)``
is a spelling guard, and this repository has been walked past by two of those --
a duplicate re-introduced under an import alias, and a threshold spelled
``int("40")``. The audit integrates the declared density and reads the answer
off the mass, so a user ``Distribution`` subclass whose ``log_prob`` returns a
constant is caught by being what it is rather than by being called what it is.
Every bypass below was built and run.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.diagnose.priors import JeffreysPrior
from bayesmith.dispatch.evidence import (
    PriorAudit,
    PriorVerdict,
    audit_graph_priors,
    audit_prior,
    conditional_prior_range_report,
    conditional_prior_verdicts,
)
from bayesmith.graph.evaluate import apply_probabilistic
from bayesmith.graph.graph import Graph


class ConstantDensity(dist.Distribution):
    """A user distribution whose ``log_prob`` is flat. Not named 'improper'.

    The bypass: a guard reading ``isinstance(d, dist.ImproperUniform)`` sees an
    ordinary subclass here and passes it. ``unwrap`` strips only ``Independent``,
    so nothing else in the package looks through it either.
    """

    support = dist.constraints.real
    has_rsample = False

    def __init__(self, value=0.0):
        self._value = value
        super().__init__(batch_shape=(), event_shape=())

    def log_prob(self, value):
        return jnp.zeros_like(jnp.asarray(value, dtype=float)) + self._value

    def sample(self, key, sample_shape=()):  # pragma: no cover - never drawn
        raise NotImplementedError


class TestOnePriorAtATime:
    def test_a_normal_is_proper_and_normalised(self):
        with jax.enable_x64(True):
            audit = audit_prior(dist.Normal(0.35, 1.7))
            assert isinstance(audit, PriorAudit)
            assert audit.verdict is PriorVerdict.PROPER
            assert audit.normalised is True
            assert audit.mass == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("scale", [0.05, 0.25, 1.0, 4.0, 60.0])
    def test_a_normal_is_proper_at_every_scale_the_sweep_visits(self, scale):
        """The audit must not be a statement about one width.

        The quadrature is centred and scaled on the distribution's own moments,
        so a prior 1200 times wider than another gets the same verdict; a fixed
        integration window would silently mis-handle both ends of this range.
        """
        with jax.enable_x64(True):
            audit = audit_prior(dist.Normal(0.0, scale))
            assert audit.verdict is PriorVerdict.PROPER
            assert audit.normalised is True
            assert audit.mass == pytest.approx(1.0, abs=1e-6)

    def test_a_bounded_uniform_is_proper_and_normalised(self):
        with jax.enable_x64(True):
            audit = audit_prior(dist.Uniform(-2.0, 5.0))
            assert audit.verdict is PriorVerdict.PROPER
            assert audit.normalised is True

    def test_an_unbounded_flat_density_is_improper(self):
        """The mass grows without bound, so there is no Z to normalise."""
        with jax.enable_x64(True):
            audit = audit_prior(dist.ImproperUniform(dist.constraints.real, (), ()))
            assert audit.verdict is PriorVerdict.IMPROPER
            assert audit.normalised is None

    def test_a_flat_density_under_a_user_class_is_improper_too(self):
        """The bypass, built and run: a guard reading the TYPE passes this.

        ``ConstantDensity`` is not ``ImproperUniform`` and is not wrapped in
        ``Independent``, so no spelling check reaches it. The audit integrates
        and finds the mass diverging, which is what being improper IS.
        """
        with jax.enable_x64(True):
            audit = audit_prior(ConstantDensity())
            assert audit.verdict is PriorVerdict.IMPROPER

    def test_a_bounded_flat_density_is_proper_but_unnormalised(self):
        """Propriety and normalisation are two verdicts and this needs both.

        ``ImproperUniform`` over ``[-2, 5]`` has mass 7: finite, so ``Z`` is
        defined up to a constant -- and 7 != 1, so the ``Z`` it defines is not
        the one a Bayes factor compares. A single boolean cannot carry that,
        which is why the audit reports two fields.
        """
        with jax.enable_x64(True):
            audit = audit_prior(
                dist.ImproperUniform(dist.constraints.interval(-2.0, 5.0), (), ())
            )
            assert audit.verdict is PriorVerdict.PROPER
            assert audit.normalised is False
            assert audit.mass == pytest.approx(7.0, rel=1e-6)

    @pytest.mark.parametrize(
        "scale, why",
        [
            (float("inf"), "an infinite width is a flat density"),
            (0.0, "a zero width has no density to integrate"),
            (-3.0, "a negative width is not a width"),
            (float("nan"), "nan is not a width"),
        ],
    )
    def test_a_degenerate_scale_never_returns_proper(self, scale, why):
        """Measured on ``nuisance_prior``: these return -inf, +inf, nan and nan
        for the offset, and none of them raises. An audit that answered PROPER
        here would hand that straight to an evidence assembly.
        """
        with jax.enable_x64(True):
            audit = audit_prior(dist.Normal(0.0, scale))
            assert audit.verdict is not PriorVerdict.PROPER, why

    def test_an_undecidable_density_is_unverifiable_and_not_a_guess(self):
        """A density the quadrature cannot resolve gets its own answer.

        UNVERIFIABLE is not IMPROPER: one says the prior has no finite mass,
        the other says this audit could not tell. §2.4's rule that a method not
        applying must not be dressed as a failure, one layer down.
        """
        with jax.enable_x64(True):
            audit = audit_prior(ConstantDensity(value=float("nan")))
            assert audit.verdict is PriorVerdict.UNVERIFIABLE


class TestAWholeGraphAtOnce:
    def test_every_latent_of_a_proper_graph_passes(self):
        with jax.enable_x64(True):
            audits = audit_graph_priors(_straight_line())
            assert {a.latent for a in audits} == {"w"}
            assert all(a.verdict is PriorVerdict.PROPER for a in audits)
            assert all(a.normalised for a in audits)

    def test_a_latent_covered_by_a_joint_prior_is_undeclared_not_improper(self):
        """§0.6. ``diagnose/priors.py`` REQUIRES ``ImproperUniform`` on a
        latent a ``JeffreysPrior`` covers -- the node-level declaration is
        deliberately absent because the graph-level prior is the declaration.

        Routing that through the improper-prior verdict would report this
        package's own mandated configuration as a user error and hand back the
        wrong remedy. The two states need different names because they need
        different fixes.
        """
        with jax.enable_x64(True):
            audits = audit_graph_priors(_jeffreys_covered())
            covered = [a for a in audits if a.latent == "w"]
            assert len(covered) == 1
            assert covered[0].verdict is PriorVerdict.UNDECLARED
            assert "joint_prior" in covered[0].reason


def _straight_line():
    basis = jnp.linspace(-1.0, 1.0, 6) + 0.3
    data = 1.2 * (basis * 0.9)

    def model():
        w = sample("w", lambda: dist.Normal(0.35, 1.7))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, 0.5).to_event(1), mu, obs=data)

    return trace(model)


def _jeffreys_covered():
    """`w`'s node-level prior is ImproperUniform BY REQUIREMENT.

    `diagnose/priors.py` refuses a JeffreysPrior over a latent that also
    declares its own proper prior: the graph-level term IS the declaration, so
    the node must not carry a second one. That is the configuration this
    package mandates, and an audit must not report it as a user error.
    """
    basis = jnp.linspace(-1.0, 1.0, 6) + 0.3
    data = 1.2 * (basis * 0.9)

    def model():
        w = sample("w", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, 0.5).to_event(1), mu, obs=data)

    bare = trace(model)
    return Graph(
        nodes=bare.nodes,
        plates=bare.plates,
        joint_prior=JeffreysPrior(over=("w",)),
    )


def test_numpyro_does_not_mask_its_own_support_and_that_is_why_we_read_it():
    """The premise the audit's support handling rests on, pinned.

    A reader would reasonably assume ``log_prob`` is ``-inf`` outside the
    declared support, and that assumption would make ``_support_bounds``
    look like dead weight -- integrate wide enough and the density would fall
    to zero on its own.  It does not.  Measured here: both a proper
    ``Uniform(-2, 5)`` and an ``ImproperUniform`` over the same interval return
    their inside-support value at ``x = 100``.

    So a bounded prior integrated over a window without clipping reports as
    IMPROPER, and every bounded prior in the package would be refused for an
    evidence task.  If numpyro ever changes this, this test goes red and the
    audit's support handling can be simplified rather than silently kept.
    """
    with jax.enable_x64(True):
        far = jnp.asarray([-100.0, 100.0])
        proper = dist.Uniform(-2.0, 5.0)
        assert float(proper.log_prob(far)[0]) == pytest.approx(-jnp.log(7.0), abs=1e-9)
        flat = dist.ImproperUniform(dist.constraints.interval(-2.0, 5.0), (), ())
        assert float(flat.log_prob(far)[0]) == 0.0


def test_a_density_too_large_to_integrate_is_unverifiable_not_improper():
    """The other UNVERIFIABLE arm, and it was untested until a mutation said so.

    A flat density at ``log_prob = 709`` overflows the quadrature sum before
    the window sequence can say anything about growth.  That is a limit of this
    check, not a property of the prior, and the two must not read the same --
    §2.4's rule again.

    Found because rewriting this arm's verdict to ``IMPROPER`` left the suite
    green: the ``nan`` fixture reaches the zero-mass arm instead, so nothing
    exercised this one.
    """
    with jax.enable_x64(True):
        audit = audit_prior(ConstantDensity(value=709.0))
        assert audit.verdict is PriorVerdict.UNVERIFIABLE
        assert audit.mass is None
        assert "not a finite number" in audit.reason

    with jax.enable_x64(True):
        # And the neighbouring value IS decidable, so the boundary is a real
        # one rather than this test naming whatever the code happens to do.
        assert audit_prior(ConstantDensity(value=700.0)).verdict is (
            PriorVerdict.IMPROPER
        )


class TestTheVerdictsAnAdversarialReviewFoundWrong:
    """Thirty-one stock priors were misclassified and these pin the repairs.

    The first version of the audit read the DIFFERENCE between successive
    window masses and asked whether it had fallen below ``1e-6 * max(mass, 1)``.
    That is wrong in both directions, and both were measured:

    * too tight for a polynomial tail, so every heavy-tailed prior in numpyro
      came back IMPROPER -- including ``HalfCauchy``, which is the standard
      weakly-informative scale prior and appears in half the models anyone
      writes;
    * floored by the ``max(..., 1.0)``, so the verdict depended on the UNITS.
      The same flat improper prior on a length was IMPROPER in metres and
      PROPER in nanometres.

    What replaced it reads the RATIO of successive increments, which is a
    property of the density carrying no units: a flat density doubles its mass
    with its window (ratio 2), a ``1/x`` tail adds a constant (ratio 1), and
    every convergent density falls below 1.
    """

    @pytest.mark.parametrize(
        "name, factory",
        [
            ("HalfCauchy(1)", lambda: dist.HalfCauchy(1.0)),
            ("HalfCauchy(5)", lambda: dist.HalfCauchy(5.0)),
            ("Cauchy(0,1)", lambda: dist.Cauchy(0.0, 1.0)),
            ("Cauchy(3,2)", lambda: dist.Cauchy(3.0, 2.0)),
            ("StudentT(1.5)", lambda: dist.StudentT(1.5, 0.0, 1.0)),
            ("StudentT(3)", lambda: dist.StudentT(3.0, 0.0, 1.0)),
            ("Pareto(1,3)", lambda: dist.Pareto(1.0, 3.0)),
            ("Pareto(1,1.5)", lambda: dist.Pareto(1.0, 1.5)),
            ("InverseGamma(1,1)", lambda: dist.InverseGamma(1.0, 1.0)),
            ("HalfNormal(2)", lambda: dist.HalfNormal(2.0)),
            ("Exponential(1)", lambda: dist.Exponential(1.0)),
            ("Laplace(0,1)", lambda: dist.Laplace(0.0, 1.0)),
        ],
    )
    def test_a_heavy_tail_is_proper_and_normalised(self, name, factory):
        with jax.enable_x64(True):
            audit = audit_prior(factory())
            assert audit.verdict is PriorVerdict.PROPER, f"{name}: {audit.reason}"
            assert audit.normalised is True, f"{name}: mass {audit.mass}"

    def test_a_window_far_from_the_density_does_not_report_a_mass(self):
        """``Cauchy(1e5, 1)``'s ``mean`` is ``nan``.

        Before the fallback chain the centre defaulted to zero, the window
        integrated empty space eight orders of magnitude from the peak, and the
        audit reported PROPER with ``mass = 8.15e-9`` -- maximum confidence,
        obtained by looking in the wrong place.
        """
        with jax.enable_x64(True):
            audit = audit_prior(dist.Cauchy(1e5, 1.0))
            assert audit.verdict is PriorVerdict.PROPER
            assert audit.normalised is True
            assert audit.mass == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.parametrize("level", [0.0, -20.0, -50.0])
    def test_a_flat_density_is_improper_at_every_level(self, level):
        """The units bug, pinned at the level where it used to flip.

        ``max(|settled|, 1.0)`` is an absolute floor, so a diverging density
        whose mass is small in absolute terms could never trip the growth test.
        Measured: the boundary sat at ``log_prob = -18.6675``, and the same
        flat improper prior was IMPROPER in metres and PROPER in nanometres.
        The mass DOUBLES between the last two windows in every row here.
        """
        with jax.enable_x64(True):
            audit = audit_prior(ConstantDensity(value=level))
            assert audit.verdict is PriorVerdict.IMPROPER, audit.reason

    @pytest.mark.parametrize(
        "name, factory",
        [
            ("Bernoulli", lambda: dist.Bernoulli(probs=0.3)),
            ("Poisson(3)", lambda: dist.Poisson(3.0)),
            ("Poisson(50)", lambda: dist.Poisson(50.0)),
            ("Binomial", lambda: dist.Binomial(10, 0.3)),
            ("Geometric", lambda: dist.Geometric(0.3)),
        ],
    )
    def test_a_discrete_prior_is_unverifiable_not_a_number(self, name, factory):
        """The mass of a PMF is a SUM, and this rule is an integral.

        ``Bernoulli`` came back IMPROPER and ``Poisson(50)`` came back PROPER
        and normalised -- the second right by coincidence, because the
        Euler-Maclaurin error happens to vanish at large lambda, and nothing in
        the output distinguished it from ``Poisson(3)``, which was wrong.
        """
        with jax.enable_x64(True):
            audit = audit_prior(factory())
            assert audit.verdict is PriorVerdict.UNVERIFIABLE, name
            assert "discrete" in audit.reason

    @pytest.mark.parametrize(
        "name, factory",
        [
            ("MultivariateNormal", lambda: dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2))),
            ("batched Normal", lambda: dist.Normal(jnp.zeros(3), 1.0)),
            ("Independent", lambda: dist.Independent(dist.Normal(jnp.zeros(2), 1.0), 1)),
            ("Dirichlet", lambda: dist.Dirichlet(jnp.ones(3))),
        ],
    )
    def test_a_joint_prior_is_unverifiable_rather_than_an_exception(
        self, name, factory
    ):
        """Six stock distributions raised ``TypeError`` out of ``log_prob`` on
        a one-dimensional grid. An audit that cannot run is not a verdict that
        the prior is improper, and an uncaught exception is neither."""
        with jax.enable_x64(True):
            audit = audit_prior(factory())
            assert audit.verdict is PriorVerdict.UNVERIFIABLE, name
            assert "one dimension" in audit.reason

    def test_a_singular_density_the_rule_cannot_resolve_abstains(self):
        """``Gamma(0.5, 1)`` has an integrable ``x^-0.5`` singularity at zero.

        Its window masses creep upward -- 0.987483, 0.988001, 0.988039 -- and
        the creep is the rule's own discretisation error rather than tail mass,
        so the old growth test called a perfectly proper prior IMPROPER. The
        quadrature-convergence check catches it: halving the panel width at the
        widest window moves the answer, so the audit abstains instead of
        naming either verdict.
        """
        with jax.enable_x64(True):
            for factory in (
                lambda: dist.Gamma(0.5, 1.0),
                lambda: dist.Beta(0.5, 0.5),
                lambda: dist.LogNormal(0.0, 3.0),
            ):
                audit = audit_prior(factory())
                assert audit.verdict is PriorVerdict.UNVERIFIABLE, audit.reason
                assert "converged" in audit.reason or "window" in audit.reason


class TestTheParametersThatNothingUsedToPin:
    """Twenty of thirty-five mutations survived, and these are why.

    Nothing pinned the quadrature resolution, the window count, or either
    tolerance: ``_NODES`` 24 -> 2, ``_PANELS_PER_SCALE`` 2 -> 1, the panel floor
    8 -> 1, five windows -> two, and both thresholds moved by three orders of
    magnitude, all with the suite green. A constant nothing measures is a
    constant nobody chose.
    """

    def test_the_window_sequence_has_enough_members_to_read_a_ratio(self):
        """The docstring says the point is the SEQUENCE. Two members give one
        increment and no ratio, so the convergence test degenerates."""
        from bayesmith.dispatch.evidence import _WINDOWS

        assert len(_WINDOWS) >= 4, (
            "the extrapolation reads two increments and its uncertainty needs "
            "a third, so four windows is the floor"
        )
        assert all(
            b == pytest.approx(2.0 * a) for a, b in itertools.pairwise(_WINDOWS)
        ), "each window doubles the last; the ratios below assume it"

    def test_the_convergence_ratio_sits_between_the_two_families(self):
        """A flat density gives exactly 2 and a 1/x tail exactly 1; every
        convergent density measured falls below 1. The threshold has to
        separate those, and the derivation names 1."""
        from bayesmith.dispatch.evidence import _CONVERGENT_INCREMENT_RATIO

        assert 0.9 < _CONVERGENT_INCREMENT_RATIO < 1.0

    def test_halving_the_resolution_changes_a_measured_mass(self):
        """``_NODES`` and ``_PANELS_PER_SCALE`` are load-bearing, asserted by
        their consequence: a coarser rule must actually move an answer, or the
        values were never chosen."""
        import bayesmith.dispatch.evidence as module

        with jax.enable_x64(True):
            fine = module._mass_on(dist.Gamma(0.5, 1.0), 1e-9, 40.0, 64)
            coarse = module._mass_on(dist.Gamma(0.5, 1.0), 1e-9, 40.0, 2)
            assert abs(fine - coarse) > 1e-6, (
                "if the panel count does not change this integral, nothing in "
                "the suite is measuring the quadrature at all"
            )


def test_the_audit_answers_for_every_fixture_this_package_ships():
    """The plan's own step 4.3, as a standing test rather than a measurement.

    §4.3 of the R4 plan reads: *record how many of ``tests/exact/models.py``'s
    shipped fixtures the predicate refuses; if it refuses a fixture this
    package ships, stop and rule before widening.* It was not run before the
    audit was committed. An adversarial review ran it and found **six of the
    fixtures crashing** with ``KeyError`` -- every hierarchical prior, because
    ``apply_probabilistic(graph, node, {})`` reads ``env[parent]`` and a latent
    with parents has none in an empty environment.

    A measurement performed once is a measurement that goes stale, so it lives
    here. What it asserts is not a distribution of verdicts -- those may
    legitimately move -- but the two things that must stay true: the audit
    RETURNS for every shipped graph, and every answer is a member of the
    vocabulary. A crash is not a verdict.
    """
    from tests.exact import models

    with jax.enable_x64(True):
        seen = 0
        verdicts: dict[str, int] = {}
        for name in sorted(dir(models)):
            if name.startswith("_"):
                continue
            factory = getattr(models, name)
            if not callable(factory):
                continue
            try:
                graph = factory()
            except Exception:  # noqa: BLE001, S112 - not every name is one
                continue
            if not hasattr(graph, "latents"):
                continue
            seen += 1
            for audit in audit_graph_priors(graph):
                assert isinstance(audit.verdict, PriorVerdict), name
                assert audit.reason, f"{name}: a verdict with no grounds"
                verdicts[audit.verdict.value] = (
                    verdicts.get(audit.verdict.value, 0) + 1
                )
        assert seen >= 20, f"only {seen} fixtures reached; the walk did not run"
        assert verdicts, "no latent was audited"


def test_a_hierarchical_prior_is_audited_through_its_conditional():
    """The six crashes, reduced to the shape they all share -- and R5's answer.

    ``s ~ HalfNormal(1); w ~ Normal(0, s)``. R4 answered ``UNVERIFIABLE`` for
    ``w`` because ``p(w)`` is not a fixed density: it is defined only once ``s``
    is integrated out. That remains true of the MARGINAL, and R5 stopped asking
    it, because nothing integrates against ``p(w)``. The joint prior factorises
    as ``p(s) p(w | s)`` and both factors are densities the one-dimensional rule
    can weigh, so both are weighed.

    **What must not come back is the crash.** The reason that arm existed is
    that ``apply_probabilistic(graph, node, {})`` reads ``env[parent]`` and
    raised ``KeyError`` on six shipped fixtures. The conditional is built at the
    prior centre instead, which supplies the parent -- and a graph with no
    evaluable centre still gets a verdict rather than an exception.
    """
    with jax.enable_x64(True):
        basis = jnp.linspace(-1.0, 1.0, 6) + 0.3
        data = 1.2 * basis

        def model():
            s = sample("s", lambda: dist.HalfNormal(1.0))
            w = sample("w", lambda s_: dist.Normal(0.0, s_), s)
            b = const("basis", basis)
            mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
            observe("d", lambda m: dist.Normal(m, 0.5).to_event(1), mu, obs=data)

        graph = trace(model)
        audits = {a.latent: a for a in audit_graph_priors(graph)}
        assert audits["s"].verdict is PriorVerdict.PROPER
        assert audits["w"].verdict is PriorVerdict.PROPER, audits["w"].reason
        # And the conditional is REALLY what was weighed: at s pinned to a
        # degenerate width the same latent stops being proper. A rule that
        # answered PROPER for everything with a parent would not move here.
        assert (
            conditional_prior_verdicts(graph, {"s": jnp.asarray(0.0)})["w"]
            is not PriorVerdict.PROPER
        )


class SlightlyUnnormalised(dist.Distribution):
    """Mass 1.01 on ``[0, 1]``: wrong by more than 1e-6 and less than 1e-1."""

    support = dist.constraints.interval(0.0, 1.0)
    has_rsample = False

    def __init__(self, mass=1.01):
        self._mass = mass
        super().__init__(batch_shape=(), event_shape=())

    def log_prob(self, value):
        return jnp.full_like(jnp.asarray(value, dtype=float), jnp.log(self._mass))

    def sample(self, key, sample_shape=()):  # pragma: no cover - never drawn
        raise NotImplementedError


def test_the_normalisation_tolerance_is_not_a_tenth():
    """A prior 1% away from normalised is not normalised.

    Every fixture above is either exactly one or off by a factor of seven, so
    widening the tolerance from 1e-6 to 1e-1 left the whole suite green -- the
    threshold had no fixture in the band it governs. Measured by an adversarial
    review; this is the fixture that band was missing.

    The tolerance is ``max(1e-6, uncertainty)`` where the second term is the
    extrapolation's own measured error, so a heavy tail is not refused for
    being 3.6e-06 from one. This density has no tail at all, so the
    uncertainty is zero and 1e-6 is what governs.
    """
    with jax.enable_x64(True):
        audit = audit_prior(SlightlyUnnormalised(1.01))
        assert audit.verdict is PriorVerdict.PROPER
        assert audit.normalised is False, audit.reason
        assert audit.mass == pytest.approx(1.01, rel=1e-9)

        assert audit_prior(SlightlyUnnormalised(1.0)).normalised is True


def test_the_base_resolution_does_not_decide_a_verdict():
    """Why ``_NODES`` and ``_PANELS_PER_SCALE`` are NOT pinned to a number.

    An adversarial review found both surviving mutation and called it a gap.
    Measured, it is a property: halving ``_PANELS_PER_SCALE`` leaves every
    verdict in this file unchanged, because the convergence check compares the
    rule against a refinement OF ITSELF. A coarser base makes the audit abstain
    sooner, never answer differently -- which is what a self-validating rule is
    supposed to do.

    So the honest assertion is that property, not a number. Pinning 24 and 2
    would be a spelling guard: it would fail on any change to constants that
    the design says are free, and pass on the changes that matter.
    """
    import bayesmith.dispatch.evidence as module

    priors = [
        dist.Normal(0.0, 1.0),
        dist.HalfCauchy(1.0),
        dist.Cauchy(0.0, 1.0),
        dist.Gamma(0.5, 1.0),
        dist.Uniform(-2.0, 5.0),
        dist.ImproperUniform(dist.constraints.real, (), ()),
    ]
    original = module._PANELS_PER_SCALE
    try:
        with jax.enable_x64(True):
            fine = [audit_prior(p).verdict for p in priors]
            # An absolute coarser value, not a fraction of the current one:
            # deriving it would make this test's own subject move with the
            # constant it is asserting is free.
            module._PANELS_PER_SCALE = 1
            coarse = [audit_prior(p).verdict for p in priors]
    finally:
        module._PANELS_PER_SCALE = original
    assert fine == coarse, (
        "a coarser rule changed a verdict, which means the convergence check "
        "is not doing the work the design says it does"
    )


# ------------------------------------------------------------------ R5 Task 7
#
# The propriety restatement, and the range check the collapse guard does not
# perform. Every claim below was measured before it was written, and where a
# measurement contradicted the R5 plan the contradiction is named here and
# written back onto the plan's own line, per that plan's red line 11.


def _shipped_graphs():
    """Every shipped GRAPH, with the counting convention stated.

    **Graphs, not fixture functions.** ``flagged_line`` returns three objects,
    two of which are graphs, so it contributes 2. Counting fixture functions
    gives a different number, and counting them while letting a tuple return
    fall into an ``except: continue`` gives a third; all three describe the
    same set, and the R5 plan records that two independent censuses of it
    differed because neither said which it meant.
    """
    import inspect

    from bayesmith.graph.reduction import as_graph
    from tests.exact import models

    parameterised = {
        "cancelling_sum": {"cancel": 1e2},
        "many_observations": {"count": 3},
        "roundoff_stress": {"big": 1e6, "sigma": 1e-3},
        "sigma_functional_block": {"weights": (1.0, 0.0, -1.0)},
        "wide_plate": {"size": 4},
    }
    for name, fn in sorted(vars(models).items()):
        if not inspect.isfunction(fn) or name.startswith("_"):
            continue
        if fn.__module__ != models.__name__:
            continue
        built = fn(**parameterised.get(name, {}))
        for index, candidate in enumerate(
            built if isinstance(built, tuple) else (built,)
        ):
            try:
                graph = as_graph(candidate)
                graph.nodes  # noqa: B018 - reading it is the check
            except (AttributeError, TypeError):
                continue
            yield (
                name if not isinstance(built, tuple) else f"{name}[{index}]"
            ), graph


def _fixture(name, **kw):
    """A shipped fixture by name, from either module that ships one."""
    from bayesmith.graph.reduction import as_graph
    from tests.exact import models, residual_models

    module = models if hasattr(models, name) else residual_models
    return as_graph(getattr(module, name)(**kw))


def _verdicts(graph):
    return {audit.latent: audit.verdict for audit in audit_graph_priors(graph)}


#: The four HIERARCHICAL exact-plus-residual fixtures -- the ones whose
#: admission the propriety restatement is responsible for. The other three of
#: the seven shipped exact-plus-residual graphs each have their own reason not
#: to be here: ``mixed_radiometer`` is ``gcr+mh`` and is refused by the method
#: row, ``improper_outside_prior`` stays refused on ``z``, and
#: ``overflowing_outside_latent`` was never gated by propriety at all.
HIERARCHICAL_TARGETS = (
    "diamond_ancestor",
    "indirect_ancestor",
    "shared_ancestor",
    "three_latent_chain",
)

#: Class (b) has TWO shapes and only one of them is hierarchical, which is a
#: dimension the family held constant until Wave B shipped a second.
#: ``overflowing_outside_latent`` and ``mixture_prior_residual`` carry only
#: ROOT latents, so their priors go through R4's one-dimensional rule
#: unchanged and the conditional arm never runs for them. Both must still be
#: admitted, and for a different reason from the four above -- which is why
#: they are named separately rather than folded into one list.
ROOT_ONLY_CLASS_B = ("overflowing_outside_latent", "mixture_prior_residual")


class TestTheJointPriorIsAuditedFactorisedAlongTheGraph:
    """A latent with parents has a CONDITIONAL prior, and that is the density
    anything ever integrates against.

    R4 answered UNVERIFIABLE for every latent with parents, on the ground that
    ``p(w)`` is not a fixed density until its parents are integrated out. That
    is true of the MARGINAL and it is the wrong question: the joint prior
    factorises as ``prod p(theta_i | parents(theta_i))``, and every factor in
    that product is a conditional density that either is or is not proper.

    **The R5 plan said this differently and the difference is measurable.**
    Its section 0.15 restates propriety as a property of *the residual block's*
    joint prior. That cannot be evaluated where the audit runs -- the prior
    audit is a PRE-compile refusal and there is no block partition yet -- and
    it is also weaker than it needs to be: measured, three of the four target
    fixtures carry the offending latent in the EXACT block, not the residual
    one, so a residual-only audit would leave them unexamined rather than
    admitted for a reason. Auditing the graph's own factorisation needs no
    partition, covers both blocks, and admits the same four.
    """

    @pytest.mark.parametrize("name", HIERARCHICAL_TARGETS)
    def test_a_hierarchical_fixture_is_proper_on_every_latent(self, name):
        with jax.enable_x64(True):
            verdicts = _verdicts(_fixture(name))
        assert set(verdicts.values()) == {PriorVerdict.PROPER}, (
            f"{name} is R5's headline class and every one of its latents "
            f"declares a proper conditional density; got {verdicts}"
        )

    @pytest.mark.parametrize("name", ROOT_ONLY_CLASS_B)
    def test_a_root_only_class_b_fixture_goes_through_the_unchanged_rule(self, name):
        """Class (b) is not all hierarchical, and asserting only the four
        hierarchical fixtures would leave that dimension untested.

        ``mixture_prior_residual`` is the case worth naming: its ``w`` carries
        a two-component mixture prior and its residual posterior is bimodal,
        so it is the first multimodal graph R5's headline class admits -- and
        the restatement has nothing to do with its admission. Both of its
        latents are roots, so R4's one-dimensional rule answers, unchanged,
        and what was blocking it is the structure premise alone.
        """
        with jax.enable_x64(True):
            graph = _fixture(name)
            assert all(not tuple(graph.node(n).parents) for n in graph.latents), (
                f"{name} is in this list because it has no hierarchical prior; "
                "if it grows one the list is what is wrong"
            )
            verdicts = _verdicts(graph)
        assert set(verdicts.values()) == {PriorVerdict.PROPER}

    def test_the_conditional_is_evaluated_and_not_assumed(self):
        """The verdict has to come from integrating the realised conditional.

        A rule that answered PROPER for every latent with parents would pass
        the four fixtures above and be worthless. ``improper_outside_prior``
        separates them: its ``z`` is a root with a genuinely improper prior and
        must still be refused.
        """
        with jax.enable_x64(True):
            verdicts = _verdicts(_fixture("improper_outside_prior"))
        assert verdicts["z"] is PriorVerdict.IMPROPER
        assert verdicts["w"] is PriorVerdict.PROPER

    def test_a_conditional_that_is_improper_at_the_centre_is_refused(self):
        """Built as a bypass rather than found: a hierarchical prior whose
        conditional is improper AT the point the audit evaluates it.

        If the new arm answered PROPER for anything with parents, this graph
        would pass. It is the same shape as ``shared_ancestor`` with the width
        replaced by one that is infinite at the prior centre.
        """
        with jax.enable_x64(True):
            def model():
                tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
                x = sample("x", lambda t: dist.Normal(0.0, t * jnp.inf), tau)
                b = const("basis", jnp.linspace(-1.0, 1.0, 5))
                mu = det("mu", lambda b_, x_: b_ * x_, b, x, linear_in=("x",))
                observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(5))

            verdicts = _verdicts(trace(model))
        assert verdicts["tau"] is PriorVerdict.PROPER
        assert verdicts["x"] is not PriorVerdict.PROPER, (
            "an infinite conditional width is not a proper density, and a rule "
            "that answers PROPER for any latent with parents would say it is"
        )

    def test_the_shipped_census_moves_by_exactly_the_hierarchical_fixtures(self):
        """The whole-graph-exact class must be untouched by this widening.

        R4 closed on class (a) being admitted 15/15 -- 19/19 at the denominator
        Task 1 corrected it to -- and the prior audit refuses nothing R4
        admits. Measured over every shipped graph, the only latents whose
        verdict this restatement moves are the ones with parents, and no
        whole-graph-exact fixture has one.
        """
        moved = {}
        with jax.enable_x64(True):
            for label, graph in _shipped_graphs():
                for audit in audit_graph_priors(graph):
                    node = graph.node(audit.latent)
                    if not tuple(node.parents):
                        continue
                    moved.setdefault(label, set()).add(audit.latent)
                    assert audit.verdict is not PriorVerdict.UNVERIFIABLE, (
                        f"{label}.{audit.latent} has parents and is still "
                        f"UNVERIFIABLE, so the restatement did not reach it"
                    )
        assert set(moved) == {
            "diamond_ancestor",
            "indirect_ancestor",
            "mixed_radiometer",
            "orphaned_child_latent",
            "shared_ancestor",
            "three_latent_chain",
        }, (
            "the set of shipped graphs carrying a latent with parents is the "
            f"set this widening can move, and it is not what was measured: "
            f"{sorted(moved)}"
        )

    def test_a_latent_with_no_parents_still_goes_through_R4s_rule_unchanged(self):
        """Section 0.20's protection, asserted where it actually lives.

        **The plan's own form of this assertion is false and was measured to
        be.** Section 0.20 rules that *every exact-block latent with no
        residual-block parent still passes R4's one-dimensional rule*, on the
        ground that ``_is_gaussian`` makes such a latent's prior a diagonal
        Gaussian. Measured: ``diamond_ancestor`` and ``indirect_ancestor`` each
        carry an exact-block ``x`` whose only parent is a DETERMINISTIC node,
        so it has no residual-block parent and R4's rule answered UNVERIFIABLE
        for it -- because the rule short-circuits on ``node.parents`` and never
        reaches the density at all. The reasoning described a density family;
        the rule never looked at one.

        What is true, and is what protects the restatement: a latent with NO
        parents is audited by R4's rule exactly as before. Everything with
        parents is covered by its conditional instead, which is a strictly
        stronger statement than the one section 0.20 asked for.
        """
        with jax.enable_x64(True):
            for name in ("improper_outside_prior", "overflowing_outside_latent"):
                graph = _fixture(name)
                for audit in audit_graph_priors(graph):
                    if tuple(graph.node(audit.latent).parents):
                        continue
                    direct = audit_prior(
                        apply_probabilistic(graph, graph.node(audit.latent), {}),
                        latent=audit.latent,
                    )
                    assert audit.verdict is direct.verdict
                    assert audit.normalised == direct.normalised


class TestConditionalProprietyOverADeclaredRange:
    """``p(x | tau)`` proper at the centre does not make it proper everywhere,
    and the guard that was supposed to catch that does not fire.

    The identity the restatement rests on is
    ``Z = INT p(tau) [ INT p(x|tau) p(d|x,tau) dx ] dtau``. The inner integral
    is ``marginal_log_density``, and it needs ``p(x|tau)`` proper AT EVERY tau
    the outer integral visits, not on average.
    """

    def test_tau_alone_separates_nothing_which_is_why_the_check_reads_the_density(
        self,
    ):
        """The finding that decided the FORM of this check.

        A range check needs a coordinate, and the obvious candidate -- tau
        itself -- separates nothing. Measured at ``tau = 0``: ``shared_ancestor``
        is degenerate there and ``three_latent_chain`` and ``mixed_radiometer``
        are perfectly proper, because each declares a different function of tau
        as its conditional width. The quantity that separates them is the
        conditional's own realised scale, which is a DIFFERENT function of tau
        in every fixture -- so the check evaluates the conditional density
        rather than gating on any coordinate of tau.
        """
        with jax.enable_x64(True):
            at_zero = {
                name: conditional_prior_verdicts(
                    _fixture(name), {"tau": jnp.asarray(0.0)}
                )
                for name in ("shared_ancestor", "three_latent_chain")
            }
        assert at_zero["shared_ancestor"]["x"] is not PriorVerdict.PROPER
        assert at_zero["three_latent_chain"]["x"] is PriorVerdict.PROPER, (
            "tau = 0 is degenerate for one fixture and ordinary for the other; "
            "any threshold on tau would have to admit both or refuse both"
        )

    @pytest.mark.parametrize("name", HIERARCHICAL_TARGETS)
    def test_the_targets_are_proper_across_the_declared_probe_range(self, name):
        with jax.enable_x64(True):
            graph = _fixture(name)
            report = conditional_prior_range_report(graph)
        assert report.degenerate == (), (
            f"{name} is admitted by 7.2 and must stay proper across the range "
            f"the check declares; got {report.degenerate}"
        )
        assert report.at_points, "a verdict must carry the points it was taken at"

    def test_a_conditional_that_diverges_inside_the_range_is_reported(self):
        """The bypass, built and run.

        ``x``'s width is 1.0 above the cut and infinite below it, and the cut
        sits inside the range the check declares. Nothing about the graph at
        its prior centre says so: the centre is ``tau = 2.0``, where the model
        is an ordinary straight line.
        """
        with jax.enable_x64(True):
            graph = trace(_corner_divergent(cut=1.0))
            assert _verdicts(graph) == {
                "tau": PriorVerdict.PROPER,
                "x": PriorVerdict.PROPER,
            }, "at the centre this graph is proper, which is the point of it"
            report = conditional_prior_range_report(graph)
        assert report.degenerate, (
            "the conditional is infinite over a positive-measure corner of the "
            "declared range and the report says it is proper"
        )
        assert report.degenerate[0][0] == "x"

    def test_the_report_carries_the_points_it_did_not_reach(self):
        """A sample is not a proof, and the honest form is to say so.

        The same bypass with the cut moved below the range's own lower end is
        NOT reported, because no point the check evaluated is degenerate. That
        is a limit of the check and it is recorded rather than gated: closing
        it needs a claim about the MEASURE of the degenerate set, which needs a
        threshold, and R5 pre-authorises none for this.
        """
        with jax.enable_x64(True):
            graph = trace(_corner_divergent(cut=0.4))
            report = conditional_prior_range_report(graph)
        assert report.degenerate == ()
        low = min(float(v) for values in report.at_points.values() for v in values)
        assert low > 0.4, (
            "this test's whole subject is a corner BELOW the range; if the "
            "range reached it the test would be asserting nothing"
        )

    def test_today_the_collapse_guard_does_not_fire_on_this_at_all(self):
        """Why the check has to exist, measured rather than argued.

        The R5 plan's section 0.15 says an improper conditional surfaces as an
        ``eqx.error_if`` abort from ``collapse.py``'s pivot guard, and that the
        defect is its being an exception where a Refusal is required. Measured:
        **it is not an exception either.** ``marginal_log_density`` returns
        ``-inf`` and the guard never fires, because ``pivots_constrain_block``
        tests a RELATIVE floor over the joint prior-and-data information, and
        the data still constrains the block however improper the prior is.

        ``-inf`` is the value the R5 plan's section 0.1 records as
        unrepresentable in ``EvidenceResult.log_evidence``, so left alone this
        surfaces as a validator error a long way from its cause, or not at all.
        """
        from bayesmith.dispatch.collapse import marginal_log_density

        with jax.enable_x64(True):
            graph = trace(_corner_divergent(cut=1.0))
            inside = float(
                jnp.asarray(marginal_log_density(graph, ("x",), {"tau": jnp.asarray(2.0)}))
            )
            corner = float(
                jnp.asarray(marginal_log_density(graph, ("x",), {"tau": jnp.asarray(0.5)}))
            )
        assert jnp.isfinite(inside)
        assert corner == float("-inf"), (
            "the corner returns a value rather than aborting, which is what "
            "makes a separate check necessary rather than merely tidier"
        )


def _corner_divergent(*, cut):
    """``x``'s prior width is 1.0 above ``cut`` and infinite at or below it.

    ``tau ~ Normal(2.0, 0.5)``, so the corner always carries positive prior
    mass. ``compile`` probes each outside latent at +/-1 and +/-3 prior widths
    about its centre -- ``{0.5, 1.5, 2.5, 3.5}`` here -- so a cut at 1.0 lies
    inside that range and a cut at 0.4 lies below all of it.
    """

    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 6))
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        width = det("width", lambda t: jnp.where(t > cut, 1.0, jnp.inf), tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

    return model
