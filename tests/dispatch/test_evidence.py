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


class TestTheShapesOneScalarSigmaHides:
    """Fixtures for the seven mutants an adversarial review left alive.

    Every test above this class uses ONE latent, ONE observation and ONE scalar
    sigma, and that made six wrong implementations of ``_dense_block`` green:
    a noise matrix built from ``variance[0]``, a ``_width_of`` returning 1, a
    prior mean truncated to its first component, an unsorted walk over
    ``block.data``, a skip branch that forgets its row cursor, and a
    ``_variance_of`` that accepts anything. None is exotic; each is what the
    code would look like if written slightly wrong.
    """

    def _oracle(self, design, data, prior_std, prior_mean, variance):
        design = np.asarray(design, float)
        n, _k = design.shape
        noise = np.diag(np.asarray(variance, float))
        prior_covariance = np.diag(np.asarray(prior_std, float) ** 2)
        covariance = design @ prior_covariance @ design.T + noise
        residual = np.asarray(data, float) - design @ np.asarray(prior_mean, float)
        _, logdet = np.linalg.slogdet(covariance)
        return float(
            -0.5
            * (
                residual @ np.linalg.solve(covariance, residual)
                + logdet
                + n * np.log(2.0 * np.pi)
            )
        )

    def test_per_sample_noise_is_not_the_first_samples_noise(self):
        """Kills ``np.diag(variance)`` -> ``np.eye(n) * variance[0]``.

        Every other fixture in this file has one scalar sigma, where the two
        are the same matrix.
        """
        with jax.enable_x64(True):
            n = 6
            sigma = jnp.asarray([0.2, 0.35, 0.5, 0.8, 1.2, 1.6])
            basis = jnp.linspace(-1.0, 1.0, n) + 0.3
            data = 1.2 * (basis * 0.9)

            def model():
                w = sample("w", lambda: dist.Normal(PRIOR_MEAN, 1.7))
                b = const("basis", basis)
                s = const("sigma", sigma)
                mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
                observe(
                    "d", lambda m, s_: dist.Normal(m, s_).to_event(1), mu, s, obs=data
                )

            assembly = assemble_exact(trace(model), ("w",), {})
            oracle = self._oracle(
                np.asarray(basis, float)[:, None],
                data,
                [1.7],
                [PRIOR_MEAN],
                np.asarray(sigma, float) ** 2,
            )
            assert float(assembly.log_evidence) == pytest.approx(oracle, abs=1e-9)
            assert float(sigma[0]) != float(sigma[-1]), "the fixture must vary"

    def test_a_two_latent_block_with_distinct_prior_widths(self):
        """Kills ``_width_of -> 1`` and the prior-mean truncation.

        With k = 1 every ``logdet`` is a scalar log and every broadcast is the
        identity, so a transposed or truncated vector is invisible.
        """
        with jax.enable_x64(True):
            n = 8
            x = jnp.linspace(-1.0, 1.0, n)
            design = jnp.stack([jnp.ones(n), x], axis=1)
            data = design @ jnp.asarray([0.4, 1.1])
            stds = (0.6, 3.2)
            means = (0.35, -0.8)

            def model():
                a = sample("wa", lambda: dist.Normal(means[0], stds[0]))
                bb = const("design", design)
                c = sample("wb", lambda: dist.Normal(means[1], stds[1]))
                mu = det(
                    "mu",
                    lambda d_, a_, c_: d_ @ jnp.stack([a_, c_]),
                    bb,
                    a,
                    c,
                    linear_in=("wa", "wb"),
                )
                observe(
                    "d", lambda m: dist.Normal(m, SIGMA_D).to_event(1), mu, obs=data
                )

            assembly = assemble_exact(trace(model), ("wa", "wb"), {})
            oracle = self._oracle(
                design, data, stds, means, np.full(n, SIGMA_D**2)
            )
            assert float(assembly.log_evidence) == pytest.approx(oracle, abs=1e-9)
            assert stds[0] != stds[1] and means[0] != means[1]

    def test_the_design_rows_follow_sorted_order_not_declaration_order(self):
        """Kills ``sorted(block.data)`` -> ``list(block.data)``.

        ``dense_operator`` lays its rows out in sorted-name order. A walk in
        declaration order slices the design at the wrong offsets whenever the
        two differ, and until this fixture nothing made them differ.
        """
        with jax.enable_x64(True):
            n_z, n_a = 4, 3
            bz = jnp.linspace(-1.0, 1.0, n_z) + 0.3
            ba = jnp.linspace(0.2, 0.9, n_a)
            dz = 1.2 * bz
            da = 0.7 * ba

            def model():
                w = sample("w", lambda: dist.Normal(PRIOR_MEAN, 1.7))
                cz = const("bz", bz)
                ca = const("ba", ba)
                muz = det("muz", lambda b_, w_: b_ * w_, cz, w, linear_in=("w",))
                # "z" is declared FIRST and sorts LAST.
                observe(
                    "z", lambda m: dist.Normal(m, SIGMA_D).to_event(1), muz, obs=dz
                )
                mua = det("mua", lambda b_, w_: b_ * w_, ca, w, linear_in=("w",))
                observe(
                    "a", lambda m: dist.Normal(m, SIGMA_D).to_event(1), mua, obs=da
                )

            graph = trace(model)
            assert tuple(graph.observed) == ("z", "a"), (
                "the fixture only tests the ordering if declaration order and "
                "sorted order differ"
            )
            assembly = assemble_exact(graph, ("w",), {})
            design = np.concatenate(
                [np.asarray(ba, float)[:, None], np.asarray(bz, float)[:, None]]
            )
            data = np.concatenate([np.asarray(da, float), np.asarray(dz, float)])
            oracle = self._oracle(
                design, data, [1.7], [PRIOR_MEAN], np.full(n_z + n_a, SIGMA_D**2)
            )
            assert float(assembly.log_evidence) == pytest.approx(oracle, abs=1e-9)

    def test_a_correlated_observation_is_refused_by_name(self):
        """The refusal fires, and nothing proved it until now.

        A correlated covariance HAS an exact evidence -- ``compress`` reads it
        through ``Precision`` with no special case -- and this module refuses
        it anyway, because the dense decomposition has no independent oracle
        for one and §9.1 does not admit a number whose only check is the route
        that produced it. A refusal nothing exercises is a refusal nobody knows
        still works.
        """
        with jax.enable_x64(True):
            n = 8
            kernel = jnp.asarray([1.0, 0.4, 0.1, 0.0, 0.0, 0.0, 0.1, 0.4]) * 0.25
            basis = jnp.linspace(-1.0, 1.0, n) + 0.3
            data = 1.2 * basis

            def model():
                w = sample("w", lambda: dist.Normal(PRIOR_MEAN, 1.7))
                b = const("basis", basis)
                mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
                observe("d", lambda m: dist.CirculantNormal(m, kernel), mu, obs=data)

            with pytest.raises(NotImplementedError, match="correlated"):
                assemble_exact(trace(model), ("w",), {})

    def test_a_masked_observation_is_refused_rather_than_crashing(self):
        """The guard that read a spelling, in the function written to avoid one.

        ``per_sample_sigma(...) is None`` looks like a consequence check and is
        a type check: a ``MaskedPrecision`` ANSWERS it, reporting ``inf`` for a
        sample never taken. So a masked observation walked past, handed
        ``slogdet`` an infinite variance, and died two modules away in
        ``EvidenceComponent``'s validator with ``log_value must be finite; got
        -inf`` -- naming neither the cause nor the fix.

        Found by an adversarial review. The finiteness test is the consequence
        and it holds whatever the class is called.
        """
        with jax.enable_x64(True):
            n = 6
            basis = jnp.linspace(-1.0, 1.0, n) + 0.3
            data = 1.2 * basis
            mask = jnp.asarray([True, True, False, True, True, True])

            def model():
                w = sample("w", lambda: dist.Normal(PRIOR_MEAN, 1.7))
                b = const("basis", basis)
                mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
                observe(
                    "d",
                    lambda m: dist.Normal(m, SIGMA_D),
                    mu,
                    obs=data,
                    mask=mask,
                )

            with pytest.raises(NotImplementedError, match="never taken"):
                assemble_exact(trace(model), ("w",), {})

    def test_a_block_no_observation_reaches_is_refused_rather_than_crashing(self):
        """``np.concatenate([])`` is not an error message.

        ``marginal_log_density`` answers 0.0 here -- the integral is over the
        prior alone. An evidence DECOMPOSITION has no data normaliser and no
        residual to report, so the honest answer is a refusal that says which
        of the two the caller is standing in, not a numpy exception about
        array counts.
        """
        with jax.enable_x64(True):
            observed = jnp.asarray([0.4, -0.2, 1.1])

            def model():
                z = sample("z", lambda: dist.Normal(1.5, 0.4))
                tau = sample("tau", lambda: dist.Normal(-0.2, 1.1))
                nu = det("nu", lambda t_: t_ * jnp.ones(3), tau)
                observe(
                    "e", lambda m: dist.Normal(m, 0.8).to_event(1), nu, obs=observed
                )
                det("unused", lambda z_: z_ * 2.0, z, linear_in=("z",))

            with pytest.raises(NotImplementedError, match="no observation reaches"):
                assemble_exact(trace(model), ("z",), {"tau": jnp.asarray(0.7)})


def test_every_component_reports_the_method_that_produced_it():
    """``assert component.method`` is truthiness, and truthiness is decoration.

    An adversarial review set every component's ``method`` to one string and
    the whole suite stayed green: four distinct methods over five terms, and a
    single value satisfied every assertion in the file. A provenance field
    nothing distinguishes is a field nothing carries.
    """
    with jax.enable_x64(True):
        graph, _, _ = _graph()
        found = {c.name: c.method for c in assemble_exact(graph, ("w",), {}).components}
        assert found == {
            "data_log_normaliser": "exact_gaussian_normaliser",
            "residual_quadratic": "exact_dense_solve",
            "prior_log_normaliser": "exact_gaussian_normaliser",
            "integral_log_two_pi": "exact_gaussian_integral",
            "block_log_determinant": "exact_dense_slogdet",
        }
        assert len(set(found.values())) == 4, (
            "the five terms are produced four ways; collapsing them to one "
            "string is what this test exists to notice"
        )


def test_an_assembly_missing_a_term_is_refused_at_construction():
    """The subset hole, built and run by an adversarial review.

    ``set(names) - set(EVIDENCE_COMPONENT_NAMES)`` is empty for any subset, so
    an ``ExactAssembly`` reporting four components -- or none at all -- used to
    construct cleanly. Only one test, over one producer, noticed; any other
    producer of the type was unguarded. A dropped term is the whole defect this
    class exists to make visible.
    """
    with jax.enable_x64(True):
        graph, _, _ = _graph()
        whole = assemble_exact(graph, ("w",), {})
        with pytest.raises(ValueError, match="missing"):
            ExactAssembly(
                log_evidence=whole.log_evidence, components=whole.components[:-1]
            )
        with pytest.raises(ValueError, match="missing"):
            ExactAssembly(log_evidence=whole.log_evidence, components=())


class TestAVectorLatentAndAnUnreachedObservationThatSortsFirst:
    """Two more shapes, for two more mutants that a scalar block cannot see.

    ``test_a_two_latent_block_with_distinct_prior_widths`` above uses two
    SCALAR latents, and every scalar has width one -- so ``_width_of`` returning
    a hard-coded 1 and a prior mean truncated to ``[:1]`` are both still
    correct there. One latent of shape ``(2,)`` is what distinguishes them.
    """

    def test_a_single_latent_of_width_two(self):
        with jax.enable_x64(True):
            n = 8
            x = jnp.linspace(-1.0, 1.0, n)
            design = jnp.stack([jnp.ones(n), x], axis=1)
            data = design @ jnp.asarray([0.4, 1.1])
            means = jnp.asarray([0.35, -0.8])
            stds = jnp.asarray([0.6, 3.2])

            def model():
                w = sample("w", lambda: dist.Normal(means, stds).to_event(1))
                d = const("design", design)
                mu = det("mu", lambda d_, w_: d_ @ w_, d, w, linear_in=("w",))
                observe(
                    "d", lambda m: dist.Normal(m, SIGMA_D).to_event(1), mu, obs=data
                )

            graph = trace(model)
            assembly = assemble_exact(graph, ("w",), {})

            a = np.asarray(design, float)
            noise = SIGMA_D**2 * np.eye(n)
            prior_covariance = np.diag(np.asarray(stds, float) ** 2)
            covariance = a @ prior_covariance @ a.T + noise
            residual = np.asarray(data, float) - a @ np.asarray(means, float)
            _, logdet = np.linalg.slogdet(covariance)
            oracle = -0.5 * (
                residual @ np.linalg.solve(covariance, residual)
                + logdet
                + n * np.log(2.0 * np.pi)
            )
            assert float(assembly.log_evidence) == pytest.approx(
                float(oracle), abs=1e-9
            )
            assert float(stds[0]) != float(stds[1]), (
                "per-component widths are what make the truncation visible"
            )

    def test_an_unreached_observation_that_sorts_before_the_absorbed_one(self):
        """``_dense_block`` walks the full sorted list and skips row groups.

        With the unreached observation sorting LAST there is nothing after it,
        so dropping the cursor advance in the skip branch is dead code. Named
        ``a`` it sorts first, and the design is then sliced at the wrong offset
        for every later group. The same fixture exists one layer down in
        ``test_collapse.py``; this one is for the decomposition.
        """
        with jax.enable_x64(True):
            n_d, n_a = 6, 3
            basis = jnp.linspace(-1.0, 1.0, n_d) + 0.3
            d_obs = 1.2 * basis
            a_obs = jnp.asarray([0.4, -0.2, 1.1])

            def model():
                w = sample("w", lambda: dist.Normal(PRIOR_MEAN, 1.7))
                tau = sample("tau", lambda: dist.Normal(-0.2, 1.1))
                b = const("basis", basis)
                mu = det("mu", lambda b_, w_: b_ * w_, b, w, linear_in=("w",))
                observe(
                    "d", lambda m: dist.Normal(m, SIGMA_D).to_event(1), mu, obs=d_obs
                )
                nu = det("nu", lambda t_: t_ * jnp.ones(n_a), tau)
                observe(
                    "a", lambda m: dist.Normal(m, 0.8).to_event(1), nu, obs=a_obs
                )

            graph = trace(model)
            assembly = assemble_exact(graph, ("w",), {"tau": jnp.asarray(0.7)})
            expected = _by_hand(np.asarray(basis, float), np.asarray(d_obs, float), 1.7)
            found = {c.name: c.log_value for c in assembly.components}
            for name, value in expected.items():
                assert found[name] == pytest.approx(float(value), abs=1e-9), name
