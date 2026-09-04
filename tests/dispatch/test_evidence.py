"""R4 Task 2 -- the exact log evidence as five separately derived terms.

The whole difference between a posterior and an evidence is that every
theta-independent constant is invisible in the first and load-bearing in the
second.  So a log evidence reported as one scalar is a number nobody can audit:
a dropped constant moves it, and the value it moves to is finite and plausible.

R4 §0.3 therefore requires the exact block's contribution to be carried as five
named components, each RE-DERIVED from the block's own inputs, and the
production sqrt-information total asserted equal to their sum.  Two derivations,
not one entry point checked against itself -- §9.1 names normalisation
constants, Jacobians and log determinants as exactly where that matters.

The decomposition is the determinant lemma, written out:

    log p(d) = -1/2 [ r^T C^-1 r + logdet C + n log 2pi ],   C = A S A^T + N
    logdet C = logdet N + logdet S + logdet F,               F = A^T N^-1 A + S^-1

so

    data_log_normaliser   = -1/2 sum_i log(2 pi sigma_i^2)     = -1/2 logdet(2 pi N)
    residual_quadratic    = -1/2 r^T C^-1 r
    prior_log_normaliser  = -sum_j log s_j - k/2 log 2pi       = -1/2 logdet(2 pi S)
    integral_log_two_pi   = +k/2 log 2pi
    block_log_determinant = -1/2 logdet F

``prior_log_normaliser`` and ``integral_log_two_pi`` cancel exactly -- the prior
contributes one row per block degree of freedom, so ``k`` is the same ``k``.
They are kept apart on purpose: they are produced in different places, and a
change to either alone is a real change that a pre-cancelled pair would hide.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts import EvidenceComponent
from bayesmith.dispatch.collapse import marginal_log_density
from bayesmith.dispatch.evidence import (
    EVIDENCE_COMPONENT_NAMES,
    ExactAssembly,
    assemble_exact,
)

SIGMA_D = 0.5
PRIOR_MEAN = 0.35
TAU_AT = 0.7


def _graph(*, prior_std=1.7, n=6, offset=0.0):
    basis = jnp.linspace(-1.0, 1.0, n) + 0.3
    data = 1.2 * (basis * 0.9) + SIGMA_D * jax.random.normal(jax.random.key(3), (n,))

    def model():
        w = sample("w", lambda: dist.Normal(PRIOR_MEAN, prior_std))
        b = const("basis", basis)
        mu = det("mu", lambda b_, w_: b_ * w_ + offset, b, w, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, SIGMA_D).to_event(1), mu, obs=data)

    return trace(model), np.asarray(basis, float), np.asarray(data, float)


def _by_hand(basis, data, prior_std, offset=0.0):
    """The five terms, from (A, N, S, m, c, d) and numpy alone.

    No QR, no pivots, no offset arithmetic in common with the implementation:
    a covariance is materialised, a residual is solved, slogdet is taken.
    """
    a = basis[:, None]
    n = basis.size
    noise = SIGMA_D**2 * np.eye(n)
    prior_covariance = np.asarray([[prior_std**2]])
    covariance = a @ prior_covariance @ a.T + noise
    residual = data - (a @ np.asarray([PRIOR_MEAN]) + offset)
    fisher = a.T @ np.linalg.solve(noise, a) + np.linalg.inv(prior_covariance)

    _, logdet_noise = np.linalg.slogdet(noise)
    _, logdet_prior = np.linalg.slogdet(prior_covariance)
    _, logdet_fisher = np.linalg.slogdet(fisher)
    k = prior_covariance.shape[0]
    return {
        "data_log_normaliser": -0.5 * (logdet_noise + n * np.log(2.0 * np.pi)),
        "residual_quadratic": -0.5
        * float(residual @ np.linalg.solve(covariance, residual)),
        "prior_log_normaliser": -0.5 * (logdet_prior + k * np.log(2.0 * np.pi)),
        "integral_log_two_pi": +0.5 * k * np.log(2.0 * np.pi),
        "block_log_determinant": -0.5 * logdet_fisher,
    }


class TestTheFiveTerms:
    def test_every_component_is_named_from_the_closed_set(self):
        """A provenance guard that routes on a free-text name is a spelling
        guard, and this repository has been walked past by two of those.  The
        set is closed and the assembler refuses a name outside it, so the guard
        below can assert MEMBERSHIP rather than match a string it hopes about.
        """
        with jax.enable_x64(True):
            graph, _, _ = _graph()
            assembly = assemble_exact(graph, ("w",), {})
            assert isinstance(assembly, ExactAssembly)
            names = [component.name for component in assembly.components]
            assert set(names) == set(EVIDENCE_COMPONENT_NAMES)
            assert len(names) == len(set(names)), "a term is reported twice"

    def test_every_component_carries_a_method_it_could_not_default_to(self):
        """``EvidenceComponent.method`` has the default ``""``, which is
        unconstructible -- ``_text`` refuses an empty string, so the signature
        advertises a value the dataclass rejects.  R4 supplies the value rather
        than repairing the default, which would be a schema change.
        """
        with jax.enable_x64(True):
            graph, _, _ = _graph()
            for component in assemble_exact(graph, ("w",), {}).components:
                assert isinstance(component, EvidenceComponent)
                assert component.method
                assert component.standard_error is None, (
                    "an exactly assembled term has no error to report, and "
                    "0.0 would claim a measured zero"
                )

    @pytest.mark.parametrize("prior_std", [0.05, 0.25, 1.0, 4.0, 60.0])
    def test_each_term_matches_a_hand_derivation_at_five_prior_scales(
        self, prior_std
    ):
        """§0.9: a sweep that does not leave ``std = 1`` cannot see
        ``-sum(log std)`` go missing, because the term is exactly zero there.
        That is how it shipped missing once already.
        """
        with jax.enable_x64(True):
            assert jnp.zeros(1).dtype == jnp.float64
            graph, basis, data = _graph(prior_std=prior_std)
            assembly = assemble_exact(graph, ("w",), {})
            found = {c.name: c.log_value for c in assembly.components}
            expected = _by_hand(basis, data, prior_std)
            for name, value in expected.items():
                assert found[name] == pytest.approx(float(value), abs=1e-9), name

    @pytest.mark.parametrize("prior_std", [0.05, 0.25, 1.0, 4.0, 60.0])
    def test_the_five_terms_sum_to_the_square_root_information_total(
        self, prior_std
    ):
        """The second derivation.  The components come from a dense route and
        the total comes from the QR route, so the two agree only if both are
        right; a constant dropped from either side moves one and not the other.
        """
        with jax.enable_x64(True):
            graph, _, _ = _graph(prior_std=prior_std)
            assembly = assemble_exact(graph, ("w",), {})
            production = float(marginal_log_density(graph, ("w",), {}))
            summed = sum(c.log_value for c in assembly.components)
            assert summed == pytest.approx(production, abs=1e-9)
            assert float(assembly.log_evidence) == pytest.approx(
                production, abs=1e-9
            )

    def test_a_constant_part_of_the_prediction_reaches_the_residual(self):
        """``offset_prediction`` is the term a mutation once removed without
        failing a single test in this package, because no fixture had a
        constant part in its prediction.  One does now.
        """
        with jax.enable_x64(True):
            graph, basis, data = _graph(offset=0.4)
            assembly = assemble_exact(graph, ("w",), {})
            found = {c.name: c.log_value for c in assembly.components}
            expected = _by_hand(basis, data, 1.7, offset=0.4)
            assert found["residual_quadratic"] == pytest.approx(
                float(expected["residual_quadratic"]), abs=1e-9
            )
            without_offset = _by_hand(basis, data, 1.7, offset=0.0)
            assert abs(
                float(expected["residual_quadratic"])
                - float(without_offset["residual_quadratic"])
            ) > 1.0, "the fixture must actually distinguish the two"

    def test_the_prior_normaliser_is_blind_at_unit_width(self):
        """Why the sweep has five cells and not one, asserted rather than said.

        ``prior_log_normaliser`` is ``-sum(log s) - k/2 log 2pi``, and the first
        half is exactly zero at ``s = 1``.  Measured on this module: deleting
        that half kills ``test_each_term_matches_a_hand_derivation`` at
        0.05, 0.25, 4.0 and 60.0 and **leaves 1.0 passing**.  A fixture that
        swept only unit priors would therefore report a missing constant as
        green, which is how rheplicant shipped one missing.
        """
        with jax.enable_x64(True):
            graph_unit, _, _ = _graph(prior_std=1.0)
            unit = {
                c.name: c.log_value
                for c in assemble_exact(graph_unit, ("w",), {}).components
            }
            blind = -0.5 * 1 * np.log(2.0 * np.pi)
            assert unit["prior_log_normaliser"] == pytest.approx(blind, abs=1e-12), (
                "at unit width the declared-scale half of the prior normaliser "
                "is exactly zero, so this cell cannot see it go missing"
            )

            graph_wide, _, _ = _graph(prior_std=60.0)
            wide = {
                c.name: c.log_value
                for c in assemble_exact(graph_wide, ("w",), {}).components
            }
            assert abs(wide["prior_log_normaliser"] - blind) > 1.0, (
                "a non-unit cell must be able to see it"
            )

    def test_an_observation_the_block_does_not_reach_is_not_in_the_evidence(self):
        """The Task 1 defect, one layer up.

        ``_dense_block`` reads the same seam ``marginal_log_density`` does and
        must apply the same descendant filter.  Without it, a second observed
        node that depends only on a sampled latent lands in the design with
        zero columns over the block, and its density is added to log Z --
        a number that is not p(d | block) at all.

        Measured: dropping the filter leaves every other test in this file
        green, because the rest of the fixtures have one observed node.
        """
        with jax.enable_x64(True):
            n_e = 3
            sigma_e = 0.8
            basis = jnp.linspace(-1.0, 1.0, 6) + 0.3
            d_obs = 1.2 * (basis * 0.9) + SIGMA_D * jax.random.normal(
                jax.random.key(3), (6,)
            )
            e_obs = jnp.asarray([0.4, -0.2, 1.1])

            def model():
                w = sample("w", lambda: dist.Normal(PRIOR_MEAN, 1.7))
                tau = sample("tau", lambda: dist.Normal(-0.2, 1.1))
                b = const("basis", basis)
                mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
                observe(
                    "d", lambda m: dist.Normal(m, SIGMA_D).to_event(1), mu, obs=d_obs
                )
                nu = det("nu", lambda t_: t_ * jnp.ones(n_e), tau)
                observe(
                    "e", lambda m: dist.Normal(m, sigma_e).to_event(1), nu, obs=e_obs
                )

            graph = trace(model)
            assembly = assemble_exact(graph, ("w",), {"tau": jnp.asarray(TAU_AT)})

            expected = _by_hand(
                np.asarray(basis, float), np.asarray(d_obs, float), 1.7
            )
            found = {c.name: c.log_value for c in assembly.components}
            for name, value in expected.items():
                assert found[name] == pytest.approx(float(value), abs=1e-9), name

            log_p_e = float(
                np.sum(
                    -0.5 * ((np.asarray(e_obs, float) - TAU_AT) / sigma_e) ** 2
                    - np.log(sigma_e)
                    - 0.5 * np.log(2.0 * np.pi)
                )
            )
            assert abs(log_p_e) > 1.0, "the fixture must distinguish the two"
            assert float(assembly.log_evidence) == pytest.approx(
                sum(expected.values()), abs=1e-9
            )
