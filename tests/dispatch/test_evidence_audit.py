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
)
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
