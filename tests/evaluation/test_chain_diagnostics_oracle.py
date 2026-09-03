"""ArviZ as an INDEPENDENT oracle for this package's chain diagnostics.

R3 plan §0.12, last row, and Task 7's second half.  ``SiteDiagnostic`` takes
split-R̂ and judges it against an ESS-dependent ceiling
(:func:`~bayesmith.dispatch.execute.r_hat_ceiling`); arviz takes
**rank-normalized** split-R̂ and the community judges it against a flat 1.01.
Those are two different statistics and two different rules, so they are worth
cross-checking -- and the cross-check has to be on the VERDICT, because the
values genuinely differ and pinning them to each other would be pinning a
coincidence.

**This file does not propose replacing ``SiteDiagnostic``.**  §1.5 sends R̂ and
ESS upstream as statistics; the thing that is bayesmith's own here is the
CEILING, which is derived from ESS rather than fixed, and the measurement in
:class:`TestTheyAreNotTheSameStatistic` is what stops a later reader deciding
the two are interchangeable.  The frozen-coordinate case says the same thing
from the other side: arviz reports ``ess_bulk = 2000.0`` for a bitwise
constant site -- the full nominal draw count -- so an ESS floor alone would
call the most unconverged parameter it is possible to have well sampled, and
only the ``nan`` R̂ saves the verdict.

**The two numbers in the oracle rule are ARVIZ-side convention, not a
bayesmith threshold**, and they are written here rather than registered for
that reason: 1.01 is the ceiling Vehtari et al. recommend for rank-normalized
split-R̂ and 100 the ESS below which its own estimate is not trusted.  No
D-number is claimed for them; nothing in ``src/`` reads them.  What they exist
to catch is named at each assertion.

Measured in this worktree before the assertions were written (arviz 1.3.0):

===================================  ===============  ================
fixture                              bayesmith        arviz
===================================  ===============  ================
bilinear_pair (2 x 1000) gain        1.0094 / 168.6   1.0093 / 149.9
bilinear_pair (2 x 1000) t_ant       1.0084 / 125.0   1.0087 / 149.3
bilinear_pair (2 x 400)  gain        1.0083 / 118.5   1.0173 / 110.9
bilinear_pair (2 x 400)  t_ant       1.0199 / 111.2   1.0262 / 109.3
t_ant frozen bitwise                 inf    / 0.0     nan    / 2000.0
===================================  ===============  ================

**The two rules DISAGREE at the plan's own S7.1 budget, and the converged
fixture is 2 x 1000 for that reason.**  S7.1 asks for verdict agreement on
``bilinear_pair (2, 400)``.  Measured there (2 chains, 400 draws, 400 warmup,
seed 4)::

    gain   ours r_hat 1.0083  ess 118.5  ceiling 1.1070 -> converged True
           az   rhat   1.0173  ess_bulk 110.9           -> verdict   False
    t_ant  ours r_hat 1.0199  ess 111.2  ceiling 1.1139 -> converged True
           az   rhat   1.0262  ess_bulk 109.3           -> verdict   False

Both sites disagree, and neither side is wrong: an ESS-derived ceiling of
1.107 is simply looser than a flat 1.01, and at 400 draws the two statistics
are 6.3e-3 and 8.9e-3 apart -- comparable to the whole distance either one has
to travel.  So the agreement asserted in :class:`TestTheVerdictsAgree` is a
statement about a WELL-SEPARATED regime and not a general property of the two
rules; reading it as the latter is exactly the conclusion
:class:`TestTheyAreNotTheSameStatistic` exists to deny, and this paragraph is
here so that the denial does not have to be reconstructed from the table.

The 2 x 400 disagreement is documented rather than asserted.  Turning it into
a test would pin a cell where ``gain`` sits 7e-4 from arviz's ceiling, and
CLAUDE.md's rule about ``ubuntu-latest`` drawing three different CPUs applies
with full force to a marginal cell of a NUTS run.  What IS asserted from that
budget is the thing that is robust there: the two R-hats differ by more than
1e-3.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from bayesmith.artifacts.base import ComputeBudget
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.tasks import PosteriorTask, new_task_meta
from bayesmith.dispatch.execute import chain_diagnostics
from bayesmith.dispatch.task import compile_task, execute_task
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import bilinear_pair

try:
    import arviz as _arviz
except ImportError:  # the wheel venv omits the dev-only arviz extra
    _arviz = None

#: Skipped at RUN time, not at collection -- ``tests/bridge`` states the
#: reason: a module-level ``importorskip`` shrinks the collection that
#: ``tests/test_readme_count.py`` pins, and a partial view reads as a shrunken
#: suite.
requires_arviz = pytest.mark.skipif(_arviz is None, reason="arviz is not installed")

#: arviz's own rule, borrowed verbatim and NOT registered as a threshold of
#: this package (see the module docstring).
ARVIZ_RHAT_CEILING = 1.01
ARVIZ_ESS_FLOOR = 100.0

CHAINS = 2
SITES = ("gain", "t_ant")


def posterior(*, draws, warmup, seed):
    graph = bilinear_pair()
    task = PosteriorTask(
        meta=new_task_meta(label="t7-oracle"),
        budget=ComputeBudget(draws=draws, warmup=warmup, chains=CHAINS),
        nuts_on_collapse=False,
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    result = execute_task(planned, key=jax.random.key(seed))
    assert not isinstance(result, Refusal), result
    return graph, result


def samples_of(result):
    """``{site: draws}`` with chains concatenated on the leading axis."""
    return {a.name: np.asarray(a.value) for a in result.representation.draws}


def arviz_verdicts(samples):
    """``{site: (rhat, ess_bulk, converged)}`` under arviz's own rule.

    Built with ``az.from_dict`` from the same numbers ``chain_diagnostics``
    reads, so the comparison is between the two STATISTICS and not between two
    export paths. That the bridge agrees with ``from_dict`` to six digits is
    asserted separately below.
    """
    stacked = {
        name: value.reshape((CHAINS, -1, *value.shape[1:]))
        for name, value in samples.items()
    }
    idata = _arviz.from_dict({"posterior": stacked})
    rhat, ess = _arviz.rhat(idata), _arviz.ess(idata)
    out = {}
    for name in samples:
        r = float(rhat[name].values)
        e = float(ess[name].values)
        out[name] = (r, e, bool(r < ARVIZ_RHAT_CEILING and e >= ARVIZ_ESS_FLOOR))
    return out


@pytest.fixture(scope="module")
def long_run():
    """(2 chains, 1000 draws, 1000 warmup), seed 4 -- converged on both sides."""
    return posterior(draws=1000, warmup=1000, seed=4)


@pytest.fixture(scope="module")
def short_run():
    """(2 chains, 400 draws, 400 warmup), seed 4 -- probe_28 §9's own budget."""
    return posterior(draws=400, warmup=400, seed=4)


@requires_arviz
class TestTheVerdictsAgree:
    """One fixture per side of the verdict, as §0.12's last paragraph asks."""

    def test_a_converged_pair_is_converged_on_both_sides(self, long_run):
        """What this catches: a ceiling so loose, or an ESS so mis-scaled,
        that ``converged`` stops tracking the upstream statistic at all. Both
        sites here sit clear of both rules -- R̂ within 0.01 of 1 and ESS
        above 100 -- so agreement is a statement about the implementations
        rather than about a marginal cell."""
        _graph, result = long_run
        samples = samples_of(result)
        ours = chain_diagnostics(samples, num_chains=CHAINS)
        theirs = arviz_verdicts(samples)
        for name in SITES:
            rhat, ess, verdict = theirs[name]
            assert verdict is True, (name, rhat, ess)
            assert ours[name].converged is True, (name, ours[name])

    # arviz divides the between-chain variance by the within-chain one, and a
    # constant site makes the latter exactly zero. The RuntimeWarning is the
    # nan these tests are ABOUT, so it is filtered rather than left to read as
    # a defect in the run.
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar divide:RuntimeWarning"
    )
    def test_a_frozen_coordinate_is_unconverged_on_both_sides(self, long_run):
        """A site held bitwise constant -- the most unconverged parameter it
        is possible to have.

        bayesmith maps numpyro's ``nan`` R̂ to ``inf`` deliberately (``nan >
        ceiling`` is False, so a naive comparison would pass it silently);
        arviz leaves the ``nan``, and ``nan < 1.01`` is also False. The two
        agree on the verdict by two different mechanisms, which is worth
        knowing rather than assuming."""
        _graph, result = long_run
        samples = samples_of(result)
        frozen = dict(samples)
        frozen["t_ant"] = np.full_like(samples["t_ant"], 2.0)
        ours = chain_diagnostics(frozen, num_chains=CHAINS)
        theirs = arviz_verdicts(frozen)

        assert ours["t_ant"].converged is False
        assert theirs["t_ant"][2] is False
        assert ours["gain"].converged is True
        assert theirs["gain"][2] is True

    # arviz divides the between-chain variance by the within-chain one, and a
    # constant site makes the latter exactly zero. The RuntimeWarning is the
    # nan these tests are ABOUT, so it is filtered rather than left to read as
    # a defect in the run.
    @pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar divide:RuntimeWarning"
    )
    def test_arvizs_ess_alone_would_have_passed_the_frozen_site(self, long_run):
        """Measured: ``ess_bulk = 2000.0`` for a constant array -- the full
        nominal draw count. The reason ``SiteDiagnostic`` reports the PAIR and
        judges R̂ against a ceiling derived from ESS, rather than filing an ESS
        floor of its own."""
        _graph, result = long_run
        samples = samples_of(result)
        frozen = dict(samples)
        frozen["t_ant"] = np.full_like(samples["t_ant"], 2.0)
        rhat, ess, _verdict = arviz_verdicts(frozen)["t_ant"]
        assert np.isnan(rhat)
        assert ess >= ARVIZ_ESS_FLOOR
        assert chain_diagnostics(frozen, num_chains=CHAINS)["t_ant"].ess == 0.0


@requires_arviz
class TestTheyAreNotTheSameStatistic:
    """Why the agreement above is on the VERDICT and not on the value."""

    def test_the_two_r_hats_differ_at_the_probes_own_budget(self, short_run):
        """probe_28 §9: 1.0083 vs 1.0173 (``gain``), 1.0199 vs 1.0262
        (``t_ant``) -- rank normalization is not a rounding difference.

        What this catches: somebody deciding ``SiteDiagnostic`` is a
        re-spelling of ``az.rhat`` and replacing it, which §0.12 forbids and
        which this suite would otherwise not notice. The gap is asserted as a
        magnitude rather than a value: 6.3e-3 and 8.9e-3 measured here. The
        1e-3 it is pinned at is a FITTED constant with no derived form -- what
        is measured about it is only that it sits an order of magnitude below
        the smaller of the two gaps, which is the headroom a rank
        normalization has over a rounding difference. It carries no claim
        about how far a BLAS change could move these numbers; nothing here has
        measured that.
        """
        _graph, result = short_run
        samples = samples_of(result)
        ours = chain_diagnostics(samples, num_chains=CHAINS)
        theirs = arviz_verdicts(samples)
        for name in SITES:
            gap = abs(ours[name].r_hat - theirs[name][0])
            assert gap > 1e-3, (name, ours[name].r_hat, theirs[name][0])

    def test_the_ceiling_is_this_packages_own_and_it_moves_with_ess(
        self, long_run, short_run
    ):
        """The flat 1.01 is what ``r_hat_ceiling`` exists NOT to be: measured
        here, the 400-draw run's lower ESS buys it a LOOSER ceiling than the
        1000-draw run's, in the direction a fixed constant cannot express."""
        ours_long = chain_diagnostics(samples_of(long_run[1]), num_chains=CHAINS)
        ours_short = chain_diagnostics(samples_of(short_run[1]), num_chains=CHAINS)
        for name in SITES:
            assert ours_short[name].ess < ours_long[name].ess
            assert ours_short[name].ceiling > ours_long[name].ceiling
            assert ours_short[name].ceiling > ARVIZ_RHAT_CEILING


@requires_arviz
def test_the_bridge_and_a_raw_dict_hand_arviz_the_same_chains(long_run):
    """The comparison above feeds arviz through ``az.from_dict``; the export
    R2 shipped has to reach the same numbers, or the oracle would be an oracle
    about the test helper. Measured equal to six decimals on both sites."""
    from bayesmith.bridge.arviz import to_inference_data

    graph, result = long_run
    samples = samples_of(result)
    through_bridge = _arviz.rhat(to_inference_data(result, graph=graph))
    through_dict = arviz_verdicts(samples)
    for name in SITES:
        assert float(through_bridge[name].values) == pytest.approx(
            through_dict[name][0], abs=1e-6
        )
