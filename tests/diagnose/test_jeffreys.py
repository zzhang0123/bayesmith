"""``JeffreysPrior`` -- available, explicit, and honest about what it is.

The headline is a measurement rather than a warning: under a
radiometer-style noise declaration (``Normal(mu, f |mu|)``) the Jeffreys
prior of a bare power law over ``(log A, beta)`` **is the flat prior** --
nine grid points spanning two decades in amplitude and a full unit in
spectral index return the same half-log-determinant to the last printed
digit, and it equals a numpy-only closed form no autodiff touched. Under a
constant-sigma declaration the same block is ``p(log A) ~ A^2``, improper
upward. Same declaration, two noise models, opposite ``d/d beta`` signs on
the floored power law: the noise model chooses the prior's shape, and the
prior reads the noise FROM THE GRAPH so the choice cannot be made twice.

The other half of the file is the refusals, and one of them is the reason
the determinant is taken by ``eigvalsh``: on a block that is degenerate by
construction, ``slogdet`` and ``cholesky`` both return plausible finite
answers (measured below, to the same digits rheplicant measured) for a
density that does not exist.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from bayesmith.diagnose.identifiability import DEFAULT_RANK_RTOL
from bayesmith.diagnose.priors import JeffreysPrior
from bayesmith.errors import GraphError
from tests.diagnose.models import (
    N_FREQ,
    N_TIME,
    NU0,
    RADIOMETER_F,
    doubled_graph,
    power_law_graph,
)

#: The flat value, measured here and identical to rheplicant's on the same
#: fixture (8x8 grid, f = 1e-3). Every one of the nine grid points below
#: returns it.
RADIOMETER_FLAT_HALF_LOGDET = 15.80169853

GRID = [(la, be) for la in (6.8, 7.8, 8.8) for be in (2.05, 2.55, 3.05)]


def _values(log_amp, beta):
    return {"fg_log_amp": jnp.array(log_amp), "fg_beta": jnp.array(beta)}


# ------------------------------------------------------ the measured headline --


@pytest.mark.parametrize(("log_amp", "beta"), GRID)
def test_the_radiometer_jeffreys_prior_of_a_bare_power_law_is_exactly_flat(
    log_amp, beta
):
    """Under ``sigma = f |mu|`` every ``mu`` cancels: ``I_ij = (1 + 2 f^2) /
    f^2 sum_k g_i g_j``, a constant matrix. Two decades of amplitude and a
    unit of spectral index move it by nothing."""
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        graph = power_law_graph()
        value = prior.log_density(graph, _values(log_amp, beta))
    assert float(value) == pytest.approx(RADIOMETER_FLAT_HALF_LOGDET, abs=1e-8)


def test_the_flat_constant_equals_a_numpy_closed_form_no_autodiff_touched():
    """The independent oracle the iron law asks for: ``(1 + 2 f^2)/f^2 *
    G^T G`` with ``G`` the mu-normalised design, assembled by numpy alone.
    Two implementations agreeing is not evidence; an implementation
    agreeing with the algebra is."""
    freq = np.linspace(60e6, 85e6, N_FREQ)
    g_matrix = np.stack([np.ones(N_FREQ), -np.log(freq / NU0)], axis=1)
    g_full = np.tile(g_matrix, (N_TIME, 1))
    closed = (1.0 + 2.0 * RADIOMETER_F**2) / RADIOMETER_F**2 * (g_full.T @ g_full)
    _, logdet = np.linalg.slogdet(closed)
    assert 0.5 * logdet == pytest.approx(RADIOMETER_FLAT_HALF_LOGDET, abs=1e-8)


def test_the_radiometer_jeffreys_prior_has_a_zero_gradient():
    """Flat to the last digit is a claim about the value; this is the
    derivative, which NUTS is the consumer of."""
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        graph = power_law_graph()
        grad = jax.grad(lambda v: prior.log_density(graph, v))(_values(7.8, 2.55))
    assert abs(float(grad["fg_log_amp"])) < 1e-14
    assert abs(float(grad["fg_beta"])) < 1e-14


def test_under_constant_sigma_the_amplitude_prior_is_A_squared():
    """``p(log A) ~ A^2``: slope +2.000000 in ``log A`` over six decades,
    with the midpoint on the line -- linear, not merely that average slope.
    Improper upward, so emphatically not a neutral default, and a different
    prior from the one the identical declaration gives under the
    radiometer."""
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        graph = power_law_graph(noise="homo")

        def logp(log_amp):
            return float(prior.log_density(graph, _values(log_amp, 2.55)))

        low, high, mid = logp(-3.0), logp(3.0), logp(0.0)
    assert (high - low) / 6.0 == pytest.approx(2.000000, abs=5e-7)
    assert mid == pytest.approx(0.5 * (low + high), abs=5e-7)


def test_the_noise_model_chooses_the_priors_shape():
    """One model, one block, two noise declarations, opposite signs in
    beta. On the power law with a fixed 300 K floor -- where the radiometer
    variance no longer factorises and the prior stops being flat --
    ``d/d beta`` is -1.366854e-2 under the radiometer declaration and
    +8.052944e-3 under the constant one (measured here, and identical to
    rheplicant's own numbers on the same fixture). The prior carries no
    noise model of its own precisely so that this choice cannot be made
    twice, differently, in one run."""
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))

        def slope(graph):
            return float(
                jax.grad(lambda v: prior.log_density(graph, v))(_values(7.8, 2.55))[
                    "fg_beta"
                ]
            )

        radiometer = slope(power_law_graph(floor=300.0))
        constant = slope(power_law_graph(floor=300.0, noise="homo"))
    assert radiometer == pytest.approx(-1.366854e-02, rel=1e-5)
    assert constant == pytest.approx(+8.052944e-03, rel=1e-5)
    assert radiometer * constant < 0.0


def test_skipping_a_false_dependence_claim_drops_exactly_the_variance_term():
    """The ``depends_on_prediction`` gate, pinned by its arithmetic: on the
    bare power law the variance term multiplies ``I`` by ``(1 + 2 f^2)``,
    so declaring the radiometer node ``depends_on_prediction=False``
    (falsely) must come back exactly ``log(1 + 2 f^2)`` lower -- a 2x2
    determinant scales by the factor squared, and half of that is one
    factor. A gate that skipped nothing, or skipped the wrong half, cannot
    produce this number."""
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        honest = prior.log_density(power_law_graph(), _values(7.8, 2.55))
        skipped = prior.log_density(
            power_law_graph(depends_on_prediction=False), _values(7.8, 2.55)
        )
    assert float(honest - skipped) == pytest.approx(
        np.log(1.0 + 2.0 * RADIOMETER_F**2), rel=1e-6
    )


# ------------------------------------------------ reparameterisation invariance --


def _u_space_graph(floor=300.0):
    """The floored power law in ``u``, with ``beta = 2 + exp(u)``."""
    from bayesmith import det, observe, sample, trace

    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    data = jnp.broadcast_to(
        jnp.exp(7.8) * (freq / NU0) ** (-2.55) + floor, (N_TIME, N_FREQ)
    )

    def model():
        la = sample(
            "fg_log_amp", lambda: dist.ImproperUniform(dist.constraints.real, (), ())
        )
        u = sample("u", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
        pred = det(
            "pred",
            lambda a, u_: jnp.broadcast_to(
                jnp.exp(a) * (freq / NU0) ** (-(2.0 + jnp.exp(u_))) + floor,
                (N_TIME, N_FREQ),
            ),
            la,
            u,
        )
        observe(
            "d",
            lambda mu: dist.Normal(mu, RADIOMETER_F * jnp.abs(mu)).to_event(2),
            pred,
            obs=data,
            depends_on_prediction=True,
        )

    return trace(model)


@pytest.mark.parametrize("beta", [2.20, 2.55, 3.30])
def test_it_is_reparameterisation_invariant(beta):
    """The defining property, and the reason to want this prior at all.

    ``p(u) = p(beta) |d beta/d u|``, so in log terms the u-space density
    must equal the beta-space one plus ``log|d beta/d u|`` -- exactly, not
    approximately. Measured agreement is at the last bit of numbers of
    order 15. A prior that failed this would be a different prior in every
    coordinate system a user might reach for, which is exactly what
    ``Normal(2.3, 0.3)`` on beta is and what this is not.
    """
    with jax.enable_x64(True):
        prior_beta = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        prior_u = JeffreysPrior(over=("fg_log_amp", "u"))
        in_beta = prior_beta.log_density(
            power_law_graph(floor=300.0), _values(7.8, beta)
        )
        u_value = jnp.log(jnp.array(beta - 2.0))
        in_u = prior_u.log_density(
            _u_space_graph(), {"fg_log_amp": jnp.array(7.8), "u": u_value}
        )
        log_jacobian = float(u_value)  # d beta / d u = exp(u) = beta - 2
    assert float(in_u) == pytest.approx(float(in_beta) + log_jacobian, abs=1e-11)


# ------------------------------------------------------ the singular block --


def _singular_information():
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("a", "b", "fg_beta"))
        return prior.information(
            doubled_graph(),
            {"a": jnp.array(0.0), "b": jnp.array(0.0), "fg_beta": jnp.array(2.55)},
        )


def test_slogdet_and_cholesky_both_return_plausible_numbers_on_the_singular_block():
    """The reason the determinant is not taken either obvious way.

    ``a`` and ``b`` enter only as ``a + b``, so this matrix is exactly rank
    2 of 3 and its determinant is exactly zero. Neither routine says so:
    the null eigenvalue lands at ``-2.117e-09`` against a largest of
    ``1.281e+08`` -- 1.7e-17 relative -- and the SIGN of that roundoff
    decides whether ``slogdet`` reports ``-inf`` or a finite number and
    whether ``cholesky`` returns NaN or a factor. Here it lands positive
    and both succeed, to the same digits rheplicant measured on its own
    spelling of the fixture. If this ever fails on another platform's BLAS,
    the roundoff sign flipped and the claim is unchanged: neither routine
    RAISES, so neither is a guard.
    """
    with jax.enable_x64(True):
        matrix = _singular_information()
        sign, logabsdet = jnp.linalg.slogdet(matrix)
        assert float(sign) == pytest.approx(1.0), (
            "slogdet reported a non-positive sign; the roundoff sign flipped "
            "on this platform. The claim under test -- that slogdet does not "
            "RAISE on a singular matrix -- still holds."
        )
        assert float(0.5 * logabsdet) == pytest.approx(6.420496, abs=5e-6)

        factor = jnp.linalg.cholesky(matrix)
        assert bool(jnp.all(jnp.isfinite(factor))), (
            "cholesky returned NaN; the roundoff sign flipped on this "
            "platform. It still did not raise."
        )
        pivots = jnp.diag(factor)
        assert float(jnp.min(pivots)) == pytest.approx(9.755e-05, rel=1e-3)
        assert float(jnp.sum(jnp.log(pivots))) == pytest.approx(6.566517, abs=5e-6)


def test_the_eigh_route_floors_the_singular_block_to_effectively_zero():
    """What this prior returns where the other two returned ``+6.42``.

    Not ``-inf``: an infinite potential is a NaN gradient rather than a
    rejected proposal. The floor is the smallest positive float64, so the
    answer is about ``-338`` -- a density of ``e^-338``, which is zero for
    every purpose a sampler has.
    """
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("a", "b", "fg_beta"))
        value = float(prior.half_log_determinant(_singular_information()))
    assert np.isfinite(value)
    assert value < -300.0
    assert value == pytest.approx(-338.05, abs=0.05)


def test_the_rank_floor_is_identifiabilitys_own_tolerance_by_default():
    """One cut, justified in one place, read from there rather than
    restated."""
    assert JeffreysPrior(over="a").rank_tolerance == DEFAULT_RANK_RTOL
    assert JeffreysPrior(over="a", rank_rtol=1e-6).rank_tolerance == 1e-6


def test_a_well_conditioned_block_agrees_with_slogdet():
    """The floor must not be doing anything where there is nothing to
    floor: on the non-degenerate floored power-law block the eigh route and
    ``slogdet`` agree to 1e-9. The disagreement above is the singular
    matrix, not the method."""
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        matrix = prior.information(power_law_graph(floor=300.0), _values(7.8, 2.55))
        sign, logabsdet = jnp.linalg.slogdet(matrix)
        assert float(sign) == 1.0
        assert float(prior.half_log_determinant(matrix)) == pytest.approx(
            float(0.5 * logabsdet), abs=1e-9
        )


def test_information_rows_are_in_over_order_not_sorted_order():
    """The rheplicant wart that does NOT port: its fisher machinery
    flattened by sorted key, so ``over=`` order and row order disagreed and
    the docstring had to warn. The graph machinery preserves the caller's
    order, so reversing ``over`` transposes the matrix -- and row 0 really
    is the first name asked for, pinned by the two diagonal entries, which
    differ by construction (``sum g_1^2 = n`` for log-amp against
    ``sum log(nu/nu0)^2`` for beta)."""
    with jax.enable_x64(True):
        graph = power_law_graph()
        declared = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        reversed_ = JeffreysPrior(over=("fg_beta", "fg_log_amp"))
        forward_matrix = declared.information(graph, _values(7.8, 2.55))
        reverse_matrix = reversed_.information(graph, _values(7.8, 2.55))

        # Row order follows over=: the two spellings are transposes of one
        # permutation, not the same matrix.
        assert np.allclose(
            np.asarray(forward_matrix),
            np.asarray(reverse_matrix)[::-1, ::-1],
            rtol=1e-14,
        )
        # Row 0 is the FIRST name passed. log-amp's diagonal is the
        # mu-normalised design's constant column: (1 + 2 f^2)/f^2 * n_data.
        n_data = N_TIME * N_FREQ
        expected = (1.0 + 2.0 * RADIOMETER_F**2) / RADIOMETER_F**2 * n_data
        assert float(forward_matrix[0, 0]) == pytest.approx(expected, rel=1e-10)
        assert float(reverse_matrix[1, 1]) == pytest.approx(expected, rel=1e-10)
        # The determinant does not care, so the prior itself is unaffected.
        assert float(
            declared.half_log_determinant(forward_matrix)
        ) == pytest.approx(
            float(reversed_.half_log_determinant(reverse_matrix)), rel=1e-14
        )


# ----------------------------------------------------------------- refusals --


def test_over_is_mandatory_and_a_bare_string_is_one_name():
    with pytest.raises(TypeError):
        JeffreysPrior()
    assert JeffreysPrior(over="fg_beta").over == ("fg_beta",)


def test_an_empty_block_is_refused():
    with pytest.raises(GraphError, match="over no latents"):
        JeffreysPrior(over=())


def test_a_repeated_latent_in_over_is_refused():
    with pytest.raises(GraphError, match="more than once"):
        JeffreysPrior(over=("a", "a"))


def test_a_non_string_name_in_over_is_refused():
    with pytest.raises(GraphError, match="takes latent NAMES"):
        JeffreysPrior(over=(1, 2))


def test_a_non_positive_rank_tolerance_is_refused():
    with pytest.raises(GraphError, match="positive relative cut"):
        JeffreysPrior(over="a", rank_rtol=0.0)


def test_over_naming_a_latent_the_graph_does_not_have_is_refused():
    """The block would silently shrink to the names that matched, which is
    a different prior from the one written down."""
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_index"))
        with pytest.raises(GraphError, match=r"\['fg_index'\]"):
            prior.information(
                power_law_graph(), {**_values(7.8, 2.55), "fg_index": jnp.array(0.0)}
            )


def test_a_covered_latent_with_a_proper_density_is_refused():
    """Two priors on one quantity, multiplied, with no symptom: each on its
    own is correct, and no diagnostic reports a prior counted twice. In
    rheplicant this refusal read ``Latent(prior=...)``; here every latent
    carries a density, so the flat spelling is ``ImproperUniform`` and
    anything else is the double-count."""
    with jax.enable_x64(True):
        graph = power_law_graph(flat_latents=False)  # Normal priors declared
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        with pytest.raises(GraphError, match="two priors on one quantity"):
            prior.log_density(graph, _values(7.8, 2.55))


def test_evaluating_at_a_values_dict_missing_a_block_member_is_refused():
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        with pytest.raises(GraphError, match=r"no entry for \['fg_beta'\]"):
            prior.information(power_law_graph(), {"fg_log_amp": jnp.array(7.8)})


def test_a_rank_deficient_block_is_refused_by_name():
    """Refusal delegated to identifiability so the direction is named."""
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("a", "b", "fg_beta"))
        with pytest.raises(GraphError) as excinfo:
            prior.check_identified(doubled_graph())
    message = str(excinfo.value)
    assert "nullity 1 of 3" in message
    assert "sqrt(det I) is not a density" in message
    assert "slogdet" in message and "cholesky" in message
    # The direction is named as a combination of latents, 0.50/0.50.
    assert "direction 0" in message
    assert "a 0.50" in message and "b 0.50" in message


def test_an_identified_block_passes_the_check_and_returns_its_report():
    with jax.enable_x64(True):
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        report = prior.check_identified(power_law_graph())
    assert (report.rank, report.nullity) == (2, 0)


# ------------------------------------------------------------- the NUTS exit --


def test_a_numpyro_factor_site_carries_exactly_the_half_log_determinant():
    """The consumption pattern the docstring promises, exercised: a NumPyro
    model with flat sites for the covered latents and a ``numpyro.factor``
    carrying ``log_density``. Differenced against the same model without
    the factor, so the likelihood cancels and what is left is the prior
    term alone -- at two very different parameter points, because a factor
    that added a CONSTANT would pass a single-point test."""
    import numpyro
    from numpyro.infer.util import log_density as numpyro_log_density

    with jax.enable_x64(True):
        graph = power_law_graph(floor=300.0)
        prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
        freq = jnp.linspace(60e6, 85e6, N_FREQ)
        data = jnp.broadcast_to(
            jnp.exp(7.8) * (freq / NU0) ** (-2.55) + 300.0, (N_TIME, N_FREQ)
        )

        def base_model():
            la = numpyro.sample(
                "fg_log_amp",
                dist.ImproperUniform(dist.constraints.real, (), ()),
            )
            be = numpyro.sample(
                "fg_beta", dist.ImproperUniform(dist.constraints.real, (), ())
            )
            mu = jnp.broadcast_to(
                jnp.exp(la) * (freq / NU0) ** (-be) + 300.0, (N_TIME, N_FREQ)
            )
            numpyro.sample(
                "d",
                dist.Normal(mu, RADIOMETER_F * jnp.abs(mu)).to_event(2),
                obs=data,
            )
            return la, be

        def with_prior():
            la, be = base_model()
            numpyro.factor(
                "joint_prior",
                prior.log_density(graph, {"fg_log_amp": la, "fg_beta": be}),
            )

        for log_amp, beta in ((7.8, 2.55), (8.3, 2.20)):
            params = _values(log_amp, beta)
            with_, _ = numpyro_log_density(with_prior, (), {}, params)
            without, _ = numpyro_log_density(base_model, (), {}, params)
            expected = float(prior.log_density(graph, params))
            assert float(with_ - without) == pytest.approx(expected, abs=1e-6)
