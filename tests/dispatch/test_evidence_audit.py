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

from bayesmith import compile as bayesmith_compile
from bayesmith import const, det, observe, plate, sample, trace
from bayesmith.diagnose.priors import JeffreysPrior
from bayesmith.dispatch.evidence import (
    PriorAudit,
    PriorVerdict,
    _prior_centre,
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
    """Every shipped GRAPH, from BOTH fixture modules, convention stated.

    **Graphs, not fixture functions.** ``flagged_line`` returns three objects,
    two of which are graphs, so it contributes 2. Counting fixture functions
    gives a different number, and counting them while letting a tuple return
    fall into an ``except: continue`` gives a third; all three describe the
    same set, and the R5 plan records that two independent censuses of it
    differed because neither said which it meant.

    **Both modules, and that is not cosmetic.** This walked only
    ``tests/exact/models.py`` while its own docstring said "every shipped
    graph". ``tests/exact/residual_models.py`` ships class-(b) graphs too, and
    two of them -- ``outside_observation_pair`` and ``shifted_block_prior`` --
    carry a latent with a parent, which is exactly the population the census
    below is about. A denominator that excludes part of its own subject is the
    hazard the plan names twice: a count whose denominator is unstated is one
    the next reader re-derives differently.
    """
    import inspect

    from bayesmith.graph.reduction import as_graph
    from tests.exact import models, residual_models

    parameterised = {
        "cancelling_sum": {"cancel": 1e2},
        "many_observations": {"count": 3},
        "roundoff_stress": {"big": 1e6, "sigma": 1e-3},
        "sigma_functional_block": {"weights": (1.0, 0.0, -1.0)},
        "wide_plate": {"size": 4},
    }
    for module in (models, residual_models):
        for name, fn in sorted(vars(module).items()):
            if not inspect.isfunction(fn) or name.startswith("_"):
                continue
            if fn.__module__ != module.__name__:
                continue
            try:
                built = fn(**parameterised.get(name, {}))
            except TypeError:
                # A helper that needs arguments this census does not declare.
                # NAMED by being skipped here rather than swallowed as a graph
                # that failed to build -- the two are different silences.
                continue
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
            # tests/exact/models.py
            "diamond_ancestor",
            "indirect_ancestor",
            "mixed_radiometer",
            "orphaned_child_latent",
            "shared_ancestor",
            "three_latent_chain",
            # tests/exact/residual_models.py -- both from Wave B's review
            "outside_observation_pair",
            "shifted_block_prior",
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
        assert report.unresolved == (), (
            "the parents ARE probeable here; the corner is simply below the "
            "grid, and conflating that with an unswept parent is the exact "
            "confusion the third state exists to prevent"
        )
        low = min(
            component
            for _parent, points in report.at_points
            for point in points
            for component in point
        )
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

        **And ``-inf`` is not the only shape it takes.** An adversarial review
        measured the LOCATION-driven version of the same degeneracy -- an
        infinite conditional mean rather than an infinite width -- and the
        collapse returns ``nan`` there, not ``-inf``. Both are silent, and an
        account naming only the first would send the next reader looking for a
        sign.
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
        # The LOCATION-driven spelling of the same degeneracy returns a
        # different silent value. Measured rather than described, because the
        # first version of this account named only `-inf`, and a reader chasing
        # a sign would have looked past half the failure mode.
        with jax.enable_x64(True):
            displaced = trace(_collapsible_mean_corner())
            outside = float(
                jnp.asarray(
                    marginal_log_density(displaced, ("x",), {"tau": jnp.asarray(2.0)})
                )
            )
            at_corner = float(
                jnp.asarray(
                    marginal_log_density(displaced, ("x",), {"tau": jnp.asarray(0.5)})
                )
            )
        assert jnp.isfinite(jnp.asarray(outside))
        assert jnp.isnan(jnp.asarray(at_corner)), (
            f"an infinite conditional MEAN gives nan where an infinite width "
            f"gives -inf; got {at_corner!r}. Both are silent and neither "
            f"aborts, which is the point"
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


def _collapsible_mean_corner():
    """``x``'s prior MEAN is infinite below the cut, and ``x`` IS collapsible.

    Linear in the prediction on purpose: this fixture exists to be handed to
    ``marginal_log_density`` directly, which is what a sampler loop would do.
    """

    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 6))
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        centre = det("centre", lambda t: jnp.where(t > 1.0, 0.0, jnp.inf), tau)
        x = sample("x", lambda c: dist.Normal(c, 1.0), centre)
        mu = det("mu", lambda x_, g_: x_ * g_, x, xs, linear_in=("x",))
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

    return model


def _mean_corner(*, cut):
    """``x``'s prior MEAN, not its width, is infinite at or below ``cut``.

    The mean is a SEPARATE axis and the fixture family holds it constant: in
    every hierarchical fixture -- ``diamond_ancestor``, ``indirect_ancestor``,
    ``shared_ancestor``, ``three_latent_chain``, ``mixed_radiometer`` -- the
    parent drives the conditional's SCALE and the mean is a literal ``0.0``.
    A check developed against those alone could be blind on this axis and
    nothing in the family would say so.

    ``mu`` is quadratic in ``x`` so the latent stays in the residual block; made
    linear it joins the exact block and ``check_gaussian`` answers first.
    """

    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 6))
        tau = sample("tau", lambda: dist.Normal(2.2, 0.45))
        centre = det("centre", lambda t: jnp.where(t > cut, t, jnp.inf), tau)
        x = sample("x", lambda c: dist.Normal(c, 0.3), centre)
        mu = det("mu", lambda x_, g_: x_ * x_ * g_, x, xs)
        observe("d", lambda m: dist.Normal(m, 0.4), mu, obs=jnp.zeros(6))

    return model


class TestTheDimensionsTheFixtureFamilyHoldsConstant:
    """Coverage chosen by asking which dimensions take the SAME value in every
    class-(b) fixture, rather than by asking what this change did not vary.

    The second question is answerable only by whoever wrote the change, and
    their decomposition of the problem is what produced the blind spot, so the
    list they write omits the same axes again. The first is answerable by grep.
    Wave B's review measured four such dimensions over its five fixtures --
    one observed node in 5 of 5, the observation a descendant of the block in
    5 of 5, the eliminated block's prior mean exactly 0.0 in 4 of 5, and
    ``|m|/s`` constant across the span in 5 of 5 -- and found a survivor in
    each.
    """

    @pytest.mark.parametrize(
        "name", ["outside_observation_pair", "shifted_block_prior"]
    )
    def test_the_two_fixtures_that_break_the_constant_rows_are_admitted(self, name):
        """Both are class (b) with a hierarchical prior, so both run the
        conditional arm, and each varies a row the rest of the family fixes:
        ``outside_observation_pair`` has TWO observed nodes with one outside
        the block's descendants, and ``shifted_block_prior`` puts the residual
        latent in the block's prior MEAN.
        """
        with jax.enable_x64(True):
            graph = _fixture(name)
            verdicts = _verdicts(graph)
            report = conditional_prior_range_report(graph)
        assert set(verdicts.values()) == {PriorVerdict.PROPER}, verdicts
        assert report.degenerate == ()
        assert report.at_points, "a verdict must carry the points it was taken at"

    def test_an_impropriety_on_the_MEAN_axis_is_found_too(self):
        """The bypass for the axis the family holds constant, built and run.

        A rule that only ever inspected the conditional's SCALE would pass
        every shipped fixture and every test above it, and would wave this
        through. It does not: the verdict comes from integrating the realised
        density, which is degenerate whichever parameter made it so.
        """
        with jax.enable_x64(True):
            inside = conditional_prior_range_report(trace(_mean_corner(cut=1.75)))
            below = conditional_prior_range_report(trace(_mean_corner(cut=0.0)))
        assert inside.degenerate, (
            "the conditional's mean is infinite over a corner inside the "
            "declared range and the report calls it proper"
        )
        assert inside.degenerate[0][0] == "x"
        # Same graph, corner moved below the range: not found, and that limit
        # is recorded rather than gated -- see the class above.
        assert below.degenerate == ()

    def test_the_mean_axis_alone_does_not_make_a_prior_improper(self):
        """The other half, so the test above cannot pass by over-refusing.

        A Normal with any FINITE mean is proper however far from zero it sits,
        and ``shifted_block_prior`` sweeps its block's mean across an order of
        magnitude. A check that refused on a large mean would look like it had
        found the axis while actually being broken on it.
        """
        with jax.enable_x64(True):
            for centre in (0.0, 1.0, 25.0, -400.0):
                audit = audit_prior(dist.Normal(centre, 0.3), latent="x")
                assert audit.verdict is PriorVerdict.PROPER, (centre, audit.reason)


class TestWhatTheAdversarialReviewFoundSurviving:
    """One test per survivor cluster of the Wave C review, which BLOCKED with
    17 of 33 mutants alive.

    Every one lived in a dimension the class-(b)/(c) fixture family holds
    constant, and the two that mattered were the same fault: ``degenerate ==
    ()`` meant both "swept, found nothing" and "never swept", and the refusal
    built on it asserted a coverage it did not have. That is the failure family
    ``CLAUDE.md`` opens with, written into a guard meant to enforce it.
    """

    def test_a_corner_that_needs_BOTH_parents_low_is_found(self):
        """K1: every shipped conditioned latent has exactly one parent.

        The review built the two-parent spelling of the same fault and it
        sailed through: both ``t1 = 0.5`` and ``t2 = 0.5`` appeared in
        ``at_points``, and the cell where they are low TOGETHER was never
        asked about. It carries 5.18e-04 of the prior and
        ``marginal_log_density`` returns ``-inf`` on it.
        """
        with jax.enable_x64(True):
            both = conditional_prior_range_report(trace(_two_parent_corner(both=True)))
            one = conditional_prior_range_report(trace(_two_parent_corner(both=False)))
        assert both.degenerate, (
            "the conditional is improper only where both parents are low, and "
            "a sweep that pins one parent at a time cannot see it"
        )
        assert both.degenerate[0][1] == ("t1", "t2"), (
            "the cell has to name both parents; naming one would report a "
            "corner of the grid as though it were a point on an axis"
        )
        assert one.degenerate, "the one-parent spelling must still be caught"

    @pytest.mark.parametrize(
        "name, factory",
        [
            ("StudentT", lambda: dist.StudentT(4.0, 2.0, 0.5)),
            ("Laplace", lambda: dist.Laplace(2.0, 0.5)),
            ("Uniform", lambda: dist.Uniform(0.0, 4.0)),
            ("LogNormal", lambda: dist.LogNormal(0.7, 0.3)),
        ],
    )
    def test_a_non_gaussian_hyperprior_is_swept_rather_than_skipped(
        self, name, factory
    ):
        """K11: every swept parent in the corpus is a root ``Normal``.

        ``_probe_values`` returns ``None`` for all four of these, and the
        report used to ``continue`` past them without recording anything: zero
        points evaluated, ``degenerate == ()``, admitted, and a refusal
        downstream asserting the conditionals were densities "across the range
        the integral covers". For the StudentT the corner carried 0.058 of the
        prior -- a sixteenth, not a corner.
        """
        with jax.enable_x64(True):
            report = conditional_prior_range_report(trace(_hyperprior_corner(factory)))
        assert report.points_for("tau"), f"{name} was not probed at all"
        assert report.degenerate, f"{name}'s corner was not found"
        assert report.unresolved == ()

    def test_an_unsweepable_parent_is_recorded_and_refused_not_passed(self):
        """The third state, which is the whole repair.

        A conditional the sweep could not cover is not a conditional the sweep
        found proper. It goes to ``unresolved``, ``covers()`` answers False,
        and the task is refused -- for the same reason R4 refuses a prior whose
        mass it cannot resolve. An audit that cannot run is not a pass.
        """
        with jax.enable_x64(True):
            graph = trace(_hyperprior_corner(lambda: dist.Poisson(3.0)))
            report = conditional_prior_range_report(graph)
        assert report.unresolved, "a discrete parent cannot be probed off-lattice"
        assert not report.covers("x")
        assert report.degenerate == (), (
            "nothing was evaluated, so nothing may be reported as degenerate "
            "either -- the two absences are different and stay different"
        )

    def test_an_IMPROPER_conditional_is_caught_and_not_only_an_UNVERIFIABLE_one(
        self,
    ):
        """K9: every degeneracy any test builds is an infinite width, which
        audits UNVERIFIABLE. Narrowing the arm to that member left the suite
        green while a genuinely divergent conditional walked through.
        """
        with jax.enable_x64(True):

            def model():
                xs = const("X", jnp.linspace(1.0, 2.0, 6))
                tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
                x = sample(
                    "x",
                    lambda t: dist.Normal(0.0, 1.0)
                    if t > 1.0
                    else dist.ImproperUniform(dist.constraints.real, (), ()),
                    tau,
                )
                mu = det("mu", lambda x_, g_: x_ * x_ * g_, x, xs)
                observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

            report = conditional_prior_range_report(trace(model))
        verdicts = {cell[3] for cell in report.degenerate}
        assert "improper" in verdicts, (
            f"a flat conditional below the cut audits IMPROPER, and the arm "
            f"has to catch that member too; got {verdicts}"
        )

    def test_every_degenerate_cell_is_reported_not_just_the_first(self):
        """K7: every positive report in the suite had exactly one cell, so a
        report that kept one could not be told from one that kept them all."""
        with jax.enable_x64(True):
            report = conditional_prior_range_report(
                trace(_two_parent_corner(both=False))
            )
        assert len(report.degenerate) > 1, (
            "this graph is degenerate at every cell where t1 is low, which is "
            "more than one; a report capped at one cell passes a weaker test"
        )

    def test_every_probe_point_is_reported_not_just_the_first(self):
        """K10/M31: the one test that read ``at_points`` took a ``min`` and
        would have passed against a stub that kept a single number."""
        with jax.enable_x64(True):
            report = conditional_prior_range_report(_fixture("three_latent_chain"))
        points = dict(report.at_points)
        assert set(points) == {"tau", "x"}, (
            f"three_latent_chain is the only fixture with two swept parents, "
            f"and both have to be swept; got {sorted(points)}"
        )
        for parent, values in points.items():
            assert len(values) == 4, (
                f"{parent} was probed at {len(values)} points, not the four "
                f"the declared grid has"
            )

    def test_a_degenerate_conditional_is_caught_with_an_exact_block_too(self):
        """K8: the only degenerate conditional in the suite was class (c), so
        guarding the check with ``if runtime.exact is None`` left it green.

        Here the residual latent ``x`` is quadratic (so it stays residual) and
        a SECOND latent ``y`` is linear (so an exact block exists beside it).
        """
        from bayesmith import compile_task
        from bayesmith.artifacts.refusal import Refusal
        from bayesmith.artifacts.tasks import EvidenceTask, new_task_meta
        from tests.dispatch.test_task_protocol import model_ref

        with jax.enable_x64(True):
            graph = trace(_class_b_corner())
            plan = bayesmith_compile(graph)
            assert plan.exact is not None and plan.sampled is not None, (
                "this fixture exists to have BOTH blocks; if it stops having "
                "them the test is asserting nothing"
            )
            report = conditional_prior_range_report(graph)
            # Through the TASK, not only through the report. The first version
            # of this test called the report directly and a mutant guarding the
            # gate with `if runtime.exact is None` survived it -- the report was
            # right and nothing asked whether the route consulted it.
            outcome = compile_task(
                graph,
                EvidenceTask(meta=new_task_meta(label="z")),
                model_ref=model_ref(),
            )
        assert report.degenerate
        assert isinstance(outcome, Refusal)
        assert outcome.failed_premise == "evidence_conditional_prior_proper"

    def test_a_non_normal_conditional_is_read_rather_than_waved_through(self):
        """K3: every conditional in the corpus is a ``Normal``, so a rule that
        answered PROPER for anything else was invisible."""
        with jax.enable_x64(True):
            verdicts = conditional_prior_verdicts(
                _fixture("orphaned_child_latent"), {"w": jnp.asarray(1.0)}
            )
        assert "v" in verdicts, "orphaned_child_latent.v is the corpus's StudentT"
        assert verdicts["v"] is PriorVerdict.PROPER

    def test_a_plated_hierarchical_prior_is_unverifiable_and_that_is_recorded(self):
        """Not a mutant -- a scope statement the widening did not make.

        ``audit_prior`` integrates ONE dimension, so a conditional with an
        event shape is UNVERIFIABLE however proper it is, and every plated
        hierarchical model is therefore refused at ``evidence_prior_proper``.
        The restatement did not change that and does not claim to. Asserted
        here so the limit is a measured fact rather than a gap.
        """
        with jax.enable_x64(True):
            audit = audit_prior(dist.Normal(jnp.zeros(3), 1.0), latent="w")
        assert audit.verdict is PriorVerdict.UNVERIFIABLE
        assert "one dimension" in audit.reason

    def test_a_plated_parents_whole_probe_is_recorded_not_its_first_element(self):
        """K2: every conditioned latent and every swept parent in the corpus is
        scalar, so ``ravel()[0]`` and ``ravel()[-1]`` name the same number
        everywhere and a mutant swapping them survived the whole suite.

        A plated parent whose declared centres vary across the plate separates
        them: the probe is ``(0.5, 4.5, 8.5, 12.5)`` and quoting ``0.5`` names
        one element of the cell as though it were the cell.
        """
        with jax.enable_x64(True):
            report = conditional_prior_range_report(trace(_plated_parent_corner()))
        assert report.degenerate, "the plated parent drives an infinite width"
        _latent, parents, values, _verdict = report.degenerate[0]
        assert parents == ("tau",)
        assert len(values) == 1
        assert len(values[0]) == 4, (
            f"the cell has to carry the whole probe, not one component of it; "
            f"got {values[0]}"
        )
        assert values[0][0] != values[0][-1], (
            "this fixture exists because its probe components differ; if they "
            "stop differing it separates nothing"
        )

    def test_a_graph_with_no_evaluable_centre_reports_unresolved(self):
        """M16: ``_prior_centre`` returns ``None`` for 0 of the 60 shipped
        graphs, so the arm that handles it was never taken.

        A density whose ``mean`` raises is enough. What matters is that the
        conditionals come back as ``unresolved`` rather than as an empty
        ``degenerate``, which would read as "checked, and fine".
        """
        with jax.enable_x64(True):
            graph = trace(_no_centre())
            centre, reason = _prior_centre(graph)
            report = conditional_prior_range_report(graph)
        assert centre is None and reason
        assert report.unresolved, (
            "no centre means no sweep, and a report that said nothing about "
            "that would be the exact confusion this repair removed"
        )
        assert not report.covers("x")
        assert report.degenerate == ()

    def test_the_refusal_may_name_any_degenerate_cell_but_reports_them_all(self):
        """M26, measured rather than argued: naming the LAST cell instead of
        the first is an EQUIVALENT mutation.

        Both are true cells, both identify a real degeneracy, and the count is
        reported separately from the example. So the property worth asserting
        is not which cell is quoted -- it is that the quoted cell is a member
        of the report and that the total is not lost.
        """
        from bayesmith import compile_task
        from bayesmith.artifacts.refusal import Refusal
        from bayesmith.artifacts.tasks import EvidenceTask, new_task_meta
        from tests.dispatch.test_task_protocol import model_ref

        with jax.enable_x64(True):
            graph = trace(_two_parent_corner(both=False))
            report = conditional_prior_range_report(graph)
            outcome = compile_task(
                graph,
                EvidenceTask(meta=new_task_meta(label="z")),
                model_ref=model_ref(),
            )
        assert isinstance(outcome, Refusal)
        assert len(report.degenerate) > 1
        named = outcome.grounds[0].observed
        assert tuple(named) == report.degenerate, (
            "the finding carries every cell, so a reader is never left with "
            "one example and no idea how many there were"
        )
        assert f"{len(report.degenerate)} such cell" in outcome.grounds[0].message


def _plated_parent_corner():
    """A PLATED parent whose declared centres vary across the plate."""
    centres = jnp.array([2.0, 6.0, 10.0, 14.0])

    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 4))
        i = plate("i", 4)
        tau = sample("tau", lambda: dist.Normal(centres, 0.5), plate=i)
        width = det("width", lambda t: jnp.where(jnp.min(t) < 1.0, jnp.inf, 1.0), tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * x_ * g_, x, xs)
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(4))

    return model


class _NoCentre(dist.Distribution):
    """A density that declares no centre this package can evaluate."""

    support = dist.constraints.real

    def __init__(self):
        super().__init__(batch_shape=(), event_shape=())

    @property
    def mean(self):
        raise RuntimeError("this density declares no centre")

    def log_prob(self, value):
        return -0.5 * value**2

    def sample(self, key, sample_shape=()):  # pragma: no cover - never drawn
        raise NotImplementedError


def _no_centre():
    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 6))
        tau = sample("tau", _NoCentre)
        x = sample("x", lambda t: dist.Normal(0.0, jnp.abs(t) + 0.1), tau)
        mu = det("mu", lambda x_, g_: x_ * x_ * g_, x, xs)
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

    return model



def _two_parent_corner(*, both):
    """``x``'s width is infinite where both parents are low, or just the first."""

    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 6))
        t1 = sample("t1", lambda: dist.Normal(2.0, 0.5))
        t2 = sample("t2", lambda: dist.Normal(2.0, 0.5))
        if both:
            width = det(
                "width",
                lambda a, b: jnp.where((a < 1.0) & (b < 1.0), jnp.inf, 1.0),
                t1,
                t2,
            )
        else:
            width = det(
                "width",
                lambda a, b: jnp.where(a < 1.0, jnp.inf, 1.0) + 0.0 * b,
                t1,
                t2,
            )
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * x_ * g_, x, xs)
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

    return model


def _hyperprior_corner(factory):
    """The same infinite-width conditional under an arbitrary hyperprior."""

    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 6))
        tau = sample("tau", factory)
        width = det("width", lambda t: jnp.where(t > 1.0, 1.0, jnp.inf), tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        mu = det("mu", lambda x_, g_: x_ * x_ * g_, x, xs)
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

    return model


def _class_b_corner():
    """A degenerate conditional on the residual side WITH an exact block."""

    def model():
        xs = const("X", jnp.linspace(1.0, 2.0, 6))
        tau = sample("tau", lambda: dist.Normal(2.0, 0.5))
        width = det("width", lambda t: jnp.where(t > 1.0, 1.0, jnp.inf), tau)
        x = sample("x", lambda w: dist.Normal(0.0, w), width)
        y = sample("y", lambda: dist.Normal(0.0, 1.0))
        mu = det("mu", lambda x_, y_, g_: (x_ * x_ + y_) * g_, x, y, xs,
                 linear_in=("y",))
        observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=jnp.zeros(6))

    return model
