"""G13 -- a joint prior declared ON the graph, so both scans of it agree.

``JeffreysPrior`` could always evaluate itself; what it could not do was be
part of a model. A caller wrote ``numpyro.factor("joint_prior", ...)`` beside
the model by hand, which means the graph's own
:func:`~bayesmith.graph.evaluate.log_joint` did not know about it and a model
that simply forgot the line sampled a different posterior with every
diagnostic healthy.

**The load-bearing test in this file is the agreement one.** NUTS is the
oracle every exact path is checked against, and ``log_joint`` is what the
exact paths are compared with; a prior honoured by one and not the other makes
every such comparison a comparison of two different posteriors. So the
declaration has exactly one home and both readers go to it.

Everything here runs under ``jax.enable_x64(True)``: the information matrix
refuses float32 by name, because the null direction it has to see sits many
decades below single precision's roundoff.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.infer.util import log_density

from bayesmith import Graph, joint_prior, sample, trace
from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.diagnose.priors import JeffreysPrior
from bayesmith.errors import GraphError
from bayesmith.graph.evaluate import log_joint
from tests.diagnose.models import power_law_graph

BLOCK = ("fg_log_amp", "fg_beta")


def at(log_amp=7.8):
    """The evaluation point, built INSIDE whatever precision is ambient.

    A module-level dict would be built at import, i.e. in float32, and then
    handed to a graph whose constants were traced under ``enable_x64`` -- which
    is not a silent wrong answer but a bare "tangent aval float64 for primal
    aval float32" from inside JAX, naming neither the cause nor the remedy.
    ``refuse_ambient_float32``'s docstring names this exact shape; it costs a
    function to stay on the right side of it.
    """
    return {"fg_log_amp": jnp.array(log_amp), "fg_beta": jnp.array(2.55)}


def _pair(**kwargs):
    """The same model with and without the declaration, from one function."""
    prior = JeffreysPrior(over=BLOCK)
    return (
        power_law_graph(declare_prior=prior, **kwargs),
        power_law_graph(**kwargs),
        prior,
    )


# --------------------------------------------------------------------------
# The declaration is read, and read by both
# --------------------------------------------------------------------------


def test_log_joint_adds_the_declared_prior_and_nothing_else():
    """The difference between the two graphs is exactly the prior's value.

    An equality rather than a pinned constant: the prior's own value is
    already measured in ``test_jeffreys.py`` against a numpy closed form, and
    re-pinning it here would test that file's fixture twice while saying
    nothing about whether ``log_joint`` reads the declaration.
    """
    with jax.enable_x64(True):
        declared, bare, prior = _pair(noise="homo")
        difference = float(log_joint(declared, at())) - float(log_joint(bare, at()))
        assert difference == pytest.approx(float(prior.log_density(bare, at())), rel=1e-12)


def test_the_prior_is_not_a_no_op_on_this_fixture():
    """The sibling that makes the test above able to fail.

    Under a CONSTANT sigma the same block is ``p(log A) ~ A^2`` -- improper
    upward and steeply ``log_amp``-dependent -- so a `log_joint` that silently
    dropped the term would be caught. Under the radiometer declaration the very
    same prior is exactly flat, which is the module's headline measurement and
    would make this file's equality hold for the wrong reason.
    """
    with jax.enable_x64(True):
        _, bare, prior = _pair(noise="homo")
        here = float(prior.log_density(bare, at()))
        there = float(
            prior.log_density(bare, at(9.8))
        )
        assert abs(here - there) > 1.0


def test_the_bridge_and_log_joint_agree_with_a_joint_prior_declared():
    """The whole point of the declaration living on the graph.

    Two scans of one graph. Before this, one of them could not see the prior
    at all, so the exact paths were checked against a NUTS potential that was
    missing a term -- silently, since both answers are finite, smooth and
    plausible.
    """
    with jax.enable_x64(True):
        declared, _, _ = _pair(noise="homo")
        ours = float(log_joint(declared, at()))
        theirs, _ = log_density(to_numpyro(declared), (), {}, at())
        assert float(theirs) == pytest.approx(ours, rel=1e-12)


def test_a_model_that_forgets_the_factor_targets_a_different_potential():
    """The sibling for the agreement test: the two potentials really differ.

    Without this, a bridge that dropped the factor AND a ``log_joint`` that
    dropped it would agree with each other perfectly.
    """
    with jax.enable_x64(True):
        declared, bare, prior = _pair(noise="homo")
        with_it, _ = log_density(to_numpyro(declared), (), {}, at())
        without, _ = log_density(to_numpyro(bare), (), {}, at())
        assert float(with_it) - float(without) == pytest.approx(
            float(prior.log_density(bare, at())), rel=1e-12
        )
        assert abs(float(with_it) - float(without)) > 1.0


def test_the_prior_term_is_differentiable_through_the_graph():
    """NUTS differentiates the potential at every leapfrog step.

    A term that is right in value and dead in gradient leaves the sampler
    exploring the posterior WITHOUT the prior while the reported log-density
    includes it -- finite, plausible, and wrong in the one direction nothing
    prints.
    """
    with jax.enable_x64(True):
        declared, bare, prior = _pair(noise="homo")

        def potential(log_amp):
            return log_joint(declared, {**at(), "fg_log_amp": log_amp})

        def just_the_prior(log_amp):
            return prior.log_density(bare, {**at(), "fg_log_amp": log_amp})

        def without(log_amp):
            return log_joint(bare, {**at(), "fg_log_amp": log_amp})

        whole = float(jax.grad(potential)(at()["fg_log_amp"]))
        assert np.isfinite(whole)
        assert whole == pytest.approx(
            float(jax.grad(without)(at()["fg_log_amp"]))
            + float(jax.grad(just_the_prior)(at()["fg_log_amp"])),
            rel=1e-10,
        )
        # And the prior's own slope is the +2 the module docstring measures,
        # so the assertion above is not two zeros agreeing.
        assert float(jax.grad(just_the_prior)(at()["fg_log_amp"])) == pytest.approx(
            2.0, rel=1e-6
        )


def test_the_declaration_may_come_before_the_latents_it_covers():
    """It is a term of the joint, not a node, so it has no place in the order.

    Declared first here (the fixture calls ``joint_prior`` before ``sample``),
    which is the harder direction: the names it is over do not exist yet when
    it is recorded, and the structural check runs at ``Graph`` construction
    where they do.
    """
    with jax.enable_x64(True):
        declared, _, _ = _pair(noise="homo")
        assert declared.joint_prior is not None
        assert declared.joint_prior.over == BLOCK
        assert np.isfinite(float(log_joint(declared, at())))


# --------------------------------------------------------------------------
# What a declaration may not be
# --------------------------------------------------------------------------


def test_a_second_joint_prior_is_refused():
    """Two conditional Jeffreys blocks are not two independent factors.

    Each is the prior of its block GIVEN the other latents, and a product of
    conditionals is in general the joint density of nothing. Refused where it
    is declared, because at evaluation both terms are finite and the potential
    looks like a model.
    """

    def model():
        joint_prior(JeffreysPrior(over=("a",)))
        joint_prior(JeffreysPrior(over=("b",)))

    with pytest.raises(GraphError, match="already declares a joint_prior"):
        trace(model)


def test_a_block_naming_a_non_latent_is_refused_at_construction():
    """Earlier than the prior's own check, which cannot run without values.

    A typo in a block name should not survive until the first leapfrog step,
    and the graph knows its own latents at construction.
    """

    def model():
        joint_prior(JeffreysPrior(over=("fg_log_amp", "fg_gamma")))
        sample("fg_log_amp", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))

    with pytest.raises(GraphError, match=r"\['fg_gamma'\]"):
        trace(model)


def test_a_joint_prior_that_cannot_evaluate_itself_is_refused():
    """The graph layer checks structure, and says which two names it needs.

    ``graph.py`` is the core and imports nothing from ``diagnose``, so the
    check here is duck-typed on purpose -- and a duck-typed check that fails
    silently is worse than none, which is why it names the missing attribute.
    """

    class NotAPrior:
        over = ("x",)

    def model():
        joint_prior(NotAPrior())
        sample("x", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))

    with pytest.raises(GraphError, match="no 'log_density'"):
        trace(model)


def test_a_covered_latent_carrying_its_own_density_is_still_refused():
    """The graph route reaches the double-prior refusal, rather than bypassing it.

    Two priors on one quantity: the posterior is multiplied by both, each one
    correct on its own, and no diagnostic reports a prior counted twice. The
    refusal belongs to ``JeffreysPrior`` and this pins that declaring on the
    graph does not route around it.
    """
    with jax.enable_x64(True):
        graph = power_law_graph(
            noise="homo", flat_latents=False, declare_prior=JeffreysPrior(over=BLOCK)
        )
        with pytest.raises(GraphError, match="declare a proper"):
            log_joint(graph, at())


def test_a_graph_with_no_joint_prior_is_unchanged():
    """The field defaults to None and nothing else moves.

    Stated as a test because every graph in the package predates the field,
    and 'it defaults, so it is fine' is the kind of claim that is true until a
    reader is added that treats None as a value.
    """
    with jax.enable_x64(True):
        bare = power_law_graph(noise="homo")
        assert bare.joint_prior is None
        ours = float(log_joint(bare, at()))
        theirs, _ = log_density(to_numpyro(bare), (), {}, at())
        assert float(theirs) == pytest.approx(ours, rel=1e-12)


def test_a_hand_built_graph_may_carry_one_too():
    """``trace`` is sugar; the field is on ``Graph`` and is checked there."""
    with jax.enable_x64(True):
        bare = power_law_graph(noise="homo")
        rebuilt = Graph(
            nodes=bare.nodes,
            plates=bare.plates,
            joint_prior=JeffreysPrior(over=BLOCK),
        )
        assert float(log_joint(rebuilt, at())) != float(log_joint(bare, at()))
