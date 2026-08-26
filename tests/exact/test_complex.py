"""A complex latent solves in its real degrees of freedom (G9, minimal surface).

Sky ``alm`` coefficients are complex and the visibilities they predict are
real, so the map from coefficients to data is **R-linear but not C-linear**.
That is the whole reason this is not bookkeeping: run CG over C and it
minimises a different objective, because the objective is not holomorphic --
it has no complex derivative to descend. Splitting each complex member into
``(re, im)`` puts the solve on the vector space the posterior actually lives
on, which is R^(2n).

The second reason is the adjoint convention, and it is the one that would
have rotted silently. ``normal_operator`` and ``_conjugate_solve``'s
``pair_with`` both take ``jax.grad`` of a real pairing, and both docstrings
argue that in a real domain the gradient and the VJP pullback are the same
map -- no conjugate transpose for them to disagree about. JAX returns the
CONJUGATE gradient for a complex input, so the day a complex latent arrived
that argument would have become false while every existing test stayed green.
Splitting first is what keeps it true: the gradient is always taken over real
leaves.

The reference here is dense linear algebra built from the block's own
``forward`` by pushing real basis vectors through it -- it shares no code with
the solver, forms the 4x4 normal equations explicitly, and inverts them.

The declaration route is here too, since the owner ruled it INTO this minimal
surface: `ComplexNormal` is what a graph states a complex prior with, because
every numpyro distribution samples real and the block reads its dtype off the
prior's `loc`. The hand-built blocks below stay -- they exercise the solver
without a graph in the way -- and `TestTheDeclarationRoute` walks the whole
chain a caller actually uses.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as ndist
import pytest

from bayesmith import det, observe, sample, trace
from bayesmith.distributions import ComplexNormal
from bayesmith.errors import StructureError
from bayesmith.exact.block import (
    LinearBlock,
    domain_zero,
    real_parts,
    variance_parts,
)
from bayesmith.exact.gaussian import check_gaussian, precision_at
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.precision import diagonal_from
from bayesmith.exact.solve import gcr_sample, wiener_solve

#: A complex design: four real data points from two complex coefficients.
#: Nothing physical -- chosen only to be well conditioned at float32, so that
#: a disagreement with the dense reference is the solver's and not the
#: arithmetic's.
DESIGN = jnp.array(
    [
        [1.0 + 0.3j, 0.2 - 0.5j],
        [0.4 - 0.2j, 1.0 + 0.1j],
        [0.7 + 0.6j, -0.3 + 0.4j],
        [-0.2 + 0.1j, 0.8 - 0.7j],
    ],
    dtype=jnp.complex64,
)
NOISE_STD = 0.5
PRIOR_STD = 1.0
#: Complex, because a prior mean is a value of the latent and this one is not
#: zero -- a zero-centred prior would let a sign error in the mean path pass.
PRIOR_MEAN = jnp.array([0.5 - 0.25j, -0.75 + 0.5j], dtype=jnp.complex64)
DATA = jnp.array([1.2, -0.4, 0.9, 0.1], dtype=jnp.float32)
OFFSET = jnp.array([0.1, 0.05, -0.2, 0.0], dtype=jnp.float32)


def _forward(x):
    """``{a: complex} -> {d: real}``. R-linear, and deliberately not C-linear."""
    return {"d": jnp.real(DESIGN @ x["a"])}


def complex_block() -> LinearBlock:
    primal = {"a": jnp.zeros((2,), dtype=jnp.complex64)}
    _, pullback = jax.vjp(_forward, primal)
    return LinearBlock(
        names=("a",),
        shape={"a": (2,)},
        dtype={"a": jnp.complex64},
        offset={"d": OFFSET},
        forward=_forward,
        adjoint=lambda y: pullback(y)[0],
        data={"d": DATA},
        prior_mean={"a": PRIOR_MEAN},
        prior_std={"a": jnp.asarray(PRIOR_STD, dtype=jnp.float32)},
    )


def mixed_block() -> LinearBlock:
    """One complex member beside a real one -- the asymmetry the split keeps."""
    gain = jnp.array([0.3, -0.6, 0.2, 0.9], dtype=jnp.float32)

    def forward(x):
        return {"d": jnp.real(DESIGN @ x["a"]) + gain * x["b"]}

    primal = {
        "a": jnp.zeros((2,), dtype=jnp.complex64),
        "b": jnp.zeros((), dtype=jnp.float32),
    }
    _, pullback = jax.vjp(forward, primal)
    return LinearBlock(
        names=("a", "b"),
        shape={"a": (2,), "b": ()},
        dtype={"a": jnp.complex64, "b": jnp.float32},
        offset={"d": OFFSET},
        forward=forward,
        adjoint=lambda y: pullback(y)[0],
        data={"d": DATA},
        prior_mean={"a": PRIOR_MEAN, "b": jnp.asarray(0.25, dtype=jnp.float32)},
        prior_std={
            "a": jnp.asarray(PRIOR_STD, dtype=jnp.float32),
            "b": jnp.asarray(2.0, dtype=jnp.float32),
        },
    )


def dense_posterior():
    """``(mean, covariance)`` over ``[re a0, re a1, im a0, im a1]``, densely.

    Built by pushing the four real basis vectors through the block's own
    ``forward``, which is the only thing shared with the solver -- the normal
    equations are then formed and inverted here, in numpy, with no CG, no
    tree map and no knowledge of how the solver represents anything.
    """
    columns = []
    for index in range(4):
        parts = np.zeros(4)
        parts[index] = 1.0
        latent = jnp.asarray(parts[:2] + 1j * parts[2:], dtype=jnp.complex64)
        columns.append(np.asarray(_forward({"a": latent})["d"], dtype=np.float64))
    design = np.stack(columns, axis=1)  # (4 data, 4 real dof)

    noise_precision = np.eye(4) / NOISE_STD**2
    # Each PART carries prior_std**2 -- the convention `variance_parts`
    # documents, and the one rheplicant's `gcr_sample` states for a complex
    # latent. Getting it wrong here would show up as a factor of sqrt(2).
    prior_precision = np.eye(4) / PRIOR_STD**2
    fisher = design.T @ noise_precision @ design + prior_precision

    residual = np.asarray(DATA - OFFSET, dtype=np.float64)
    centre = np.concatenate(
        [np.real(np.asarray(PRIOR_MEAN)), np.imag(np.asarray(PRIOR_MEAN))]
    ).astype(np.float64)
    rhs = design.T @ noise_precision @ residual + prior_precision @ centre
    covariance = np.linalg.inv(fisher)
    return covariance @ rhs, covariance


def _as_parts(latent, name: str = "a") -> np.ndarray:
    value = np.asarray(latent[name])
    return np.concatenate([np.real(value), np.imag(value)]).astype(np.float64)


class TestTheSplit:
    def test_it_is_exactly_invertible(self):
        """``join(split(x)) == x``. The treedef every tree map aligns against."""
        block = mixed_block()
        split, join = real_parts(block)
        x = {
            "a": jnp.array([1.5 - 0.25j, -0.75 + 2.0j], dtype=jnp.complex64),
            "b": jnp.asarray(3.25, dtype=jnp.float32),
        }
        back = join(split(x))
        assert set(back) == set(x)
        for name, value in x.items():
            np.testing.assert_array_equal(np.asarray(back[name]), np.asarray(value))

    def test_a_real_member_is_not_wrapped(self):
        """No one-element tuple: there would be nothing to unwrap it back from,
        and a uniform wrapper only moves the asymmetry somewhere less visible."""
        split, _ = real_parts(mixed_block())
        parts = split({"a": jnp.zeros((2,), jnp.complex64), "b": jnp.asarray(1.0)})
        assert isinstance(parts["a"], tuple) and len(parts["a"]) == 2
        assert not isinstance(parts["b"], tuple)

    def test_an_all_real_block_sees_the_identity(self):
        """Every caller predating complex support must read unchanged."""
        block = LinearBlock(
            names=("b",),
            shape={"b": ()},
            dtype={"b": jnp.float32},
            offset={"d": OFFSET},
            forward=lambda x: {"d": x["b"] * jnp.ones(4)},
            adjoint=lambda y: {"b": jnp.sum(y["d"])},
            data={"d": DATA},
            prior_mean={"b": jnp.asarray(0.0)},
            prior_std={"b": jnp.asarray(1.0)},
        )
        split, join = real_parts(block)
        x = {"b": jnp.asarray(2.5)}
        assert split(x) == x
        assert join(x) == x

    def test_the_zero_and_the_variance_live_in_parts_space(self):
        block = complex_block()
        zero = domain_zero(block)
        variance = variance_parts(block)
        assert isinstance(zero["a"], tuple) and len(zero["a"]) == 2
        # Real dtype, not complex: these seed the solver's working vector, and
        # a complex zero would put the solve back on the wrong vector space.
        assert all(not jnp.iscomplexobj(half) for half in zero["a"])
        assert isinstance(variance["a"], tuple)
        # Duplicated, not halved. A complex latent's TOTAL prior variance is
        # therefore 2 * prior_std**2, which is what the seam's other side
        # documents; halving instead would report a sqrt(2) as physics.
        assert float(variance["a"][0]) == pytest.approx(PRIOR_STD**2)
        assert float(variance["a"][1]) == pytest.approx(PRIOR_STD**2)


class TestTheAdjointConvention:
    """Both halves of the identity, so the distinction cannot rot into a
    silent factor -- one half alone would pass under either convention."""

    def test_it_transposes_under_the_REAL_inner_product(self):
        block = complex_block()
        x = {"a": jnp.array([0.8 - 0.4j, -0.3 + 1.1j], dtype=jnp.complex64)}
        y = {"d": jnp.array([0.5, -1.2, 0.7, 0.25], dtype=jnp.float32)}
        left = float(jnp.real(jnp.sum(x["a"] * block.adjoint(y)["a"])))
        right = float(jnp.sum(block.forward(x)["d"] * y["d"]))
        assert left == pytest.approx(right, rel=1e-5)

    def test_the_sesquilinear_pairing_is_NOT_the_one_that_holds(self):
        """The half that makes the first half mean something.

        `sum(conj(x) * adjoint(y))` is the pairing a reader expects on a
        complex space, and it is the wrong one here: the likelihood forms a
        real pairing, so that is the convention `adjoint` answers to. Pinning
        only the identity that holds would leave a package free to switch to
        the other one and stay green wherever x happens to be real.
        """
        block = complex_block()
        x = {"a": jnp.array([0.8 - 0.4j, -0.3 + 1.1j], dtype=jnp.complex64)}
        y = {"d": jnp.array([0.5, -1.2, 0.7, 0.25], dtype=jnp.float32)}
        sesquilinear = complex(jnp.sum(jnp.conj(x["a"]) * block.adjoint(y)["a"]))
        right = float(jnp.sum(block.forward(x)["d"] * y["d"]))
        assert abs(sesquilinear.real - right) > 1e-3


class TestTheMeanPath:
    def test_wiener_solve_matches_the_dense_reference(self):
        block = complex_block()
        precision = diagonal_from({"d": jnp.full((4,), NOISE_STD)})
        solution, residual = wiener_solve(
            block, precision=precision, tol=1e-10, maxiter=200
        )
        expected, _ = dense_posterior()
        assert float(residual) < 1e-5
        np.testing.assert_allclose(_as_parts(solution), expected, rtol=2e-4, atol=2e-6)

    def test_the_solution_comes_back_in_the_domain(self):
        """Parts space stops at the solver's boundary: a complex member in,
        a complex member out, so a caller never learns the representation."""
        block = complex_block()
        precision = diagonal_from({"d": jnp.full((4,), NOISE_STD)})
        solution, _ = wiener_solve(block, precision=precision, tol=1e-10)
        assert jnp.iscomplexobj(solution["a"])
        assert solution["a"].shape == (2,)

    def test_a_mixed_block_solves_too(self):
        """A real member beside a complex one, which is the shape a gain and a
        sky share in practice. Compared against its own dense reference over
        five real degrees of freedom."""
        block = mixed_block()
        precision = diagonal_from({"d": jnp.full((4,), NOISE_STD)})
        solution, _ = wiener_solve(block, precision=precision, tol=1e-10, maxiter=200)

        columns = []
        basis = [
            {"a": jnp.array([1, 0], jnp.complex64), "b": jnp.asarray(0.0)},
            {"a": jnp.array([0, 1], jnp.complex64), "b": jnp.asarray(0.0)},
            {"a": jnp.array([1j, 0], jnp.complex64), "b": jnp.asarray(0.0)},
            {"a": jnp.array([0, 1j], jnp.complex64), "b": jnp.asarray(0.0)},
            {"a": jnp.zeros((2,), jnp.complex64), "b": jnp.asarray(1.0)},
        ]
        for unit in basis:
            columns.append(np.asarray(block.forward(unit)["d"], dtype=np.float64))
        design = np.stack(columns, axis=1)
        noise_precision = np.eye(4) / NOISE_STD**2
        prior_precision = np.diag([1.0, 1.0, 1.0, 1.0, 1.0 / 4.0])
        fisher = design.T @ noise_precision @ design + prior_precision
        centre = np.array([0.5, -0.75, -0.25, 0.5, 0.25])
        residual = np.asarray(DATA - OFFSET, dtype=np.float64)
        rhs = design.T @ noise_precision @ residual + prior_precision @ centre
        expected = np.linalg.solve(fisher, rhs)

        got = np.concatenate(
            [
                np.real(np.asarray(solution["a"])),
                np.imag(np.asarray(solution["a"])),
                [float(solution["b"])],
            ]
        )
        np.testing.assert_allclose(got, expected, rtol=5e-4, atol=5e-6)


class TestTheDrawPath:
    @pytest.mark.slow
    def test_gcr_sample_has_the_posterior_moments(self):
        """Mean AND covariance, over the real degrees of freedom.

        The mean alone would pass for a draw with no scatter at all, and the
        covariance is where the duplicated prior variance shows up: halve it
        and the drawn scatter moves by sqrt(2) while the mean does not budge.
        """
        block = complex_block()
        precision = diagonal_from({"d": jnp.full((4,), NOISE_STD)})
        keys = jax.random.split(jax.random.key(0), 512)
        draws = np.stack(
            [
                _as_parts(gcr_sample(block, precision=precision, key=k, tol=1e-10)[0])
                for k in keys
            ]
        )
        expected_mean, expected_covariance = dense_posterior()

        error = draws.mean(axis=0) - expected_mean
        standard_error = np.sqrt(np.diag(expected_covariance) / len(draws))
        np.testing.assert_array_less(np.abs(error / standard_error), 4.0)

        got = np.cov(draws, rowvar=False)
        np.testing.assert_allclose(got, expected_covariance, rtol=0.2, atol=0.02)


def sky_model():
    """The chain a caller writes: a complex prior, a real prediction, real data.

    Deliberately the SAME model the hand-built block encodes, `OFFSET` and
    all -- so one dense reference serves both routes and the comparison says
    the two agree, rather than that each agrees with a reference of its own.
    Measured while writing this: dropping `OFFSET` here left the graph solving
    a different problem and the two answers differed by 0.16, which reads
    exactly like a solver bug.
    """

    def model():
        alm = sample("alm", lambda: ComplexNormal(PRIOR_MEAN, PRIOR_STD))
        mu = det(
            "mu", lambda a: jnp.real(DESIGN @ a) + OFFSET, alm, linear_in=("alm",)
        )
        observe("d", lambda m: ndist.Normal(m, NOISE_STD), mu, obs=DATA)

    return trace(model)


class TestTheDistribution:
    def test_the_two_parts_are_independent_gaussians_of_equal_width(self):
        """The density, against two `Normal`s written out separately.

        The class computes it from the parts rather than from
        `abs(value - loc)**2`; this compares against the thing that convention
        is supposed to mean, which the magnitude form would not distinguish
        from a circularly-symmetric reading with a different normalisation.
        """
        prior = ComplexNormal(PRIOR_MEAN, PRIOR_STD)
        value = jnp.array([1.25 - 0.5j, -0.25 + 1.75j], dtype=jnp.complex64)
        expected = ndist.Normal(
            jnp.real(PRIOR_MEAN), PRIOR_STD
        ).log_prob(jnp.real(value)) + ndist.Normal(
            jnp.imag(PRIOR_MEAN), PRIOR_STD
        ).log_prob(jnp.imag(value))
        np.testing.assert_allclose(
            np.asarray(prior.log_prob(value)), np.asarray(expected), rtol=1e-6
        )

    def test_the_variance_property_is_the_total_of_both_parts(self):
        """`2 * scale**2`, and the docstring says which. A reader taking it for
        one part's variance is the sqrt(2) this whole convention is about."""
        prior = ComplexNormal(PRIOR_MEAN, PRIOR_STD)
        np.testing.assert_allclose(
            np.asarray(prior.variance), np.full((2,), 2.0 * PRIOR_STD**2), rtol=1e-6
        )

    def test_draws_are_complex_and_carry_scale_in_each_part(self):
        prior = ComplexNormal(PRIOR_MEAN, PRIOR_STD)
        draws = prior.sample(jax.random.key(1), (8192,))
        assert jnp.iscomplexobj(draws) and draws.shape == (8192, 2)
        real = np.asarray(jnp.real(draws))
        imag = np.asarray(jnp.imag(draws))
        # Each half, separately, at `scale` -- not the magnitude at `scale`,
        # which is what a halved convention would produce.
        np.testing.assert_allclose(real.std(axis=0), PRIOR_STD, rtol=0.05)
        np.testing.assert_allclose(imag.std(axis=0), PRIOR_STD, rtol=0.05)
        np.testing.assert_allclose(
            real.mean(axis=0), np.real(np.asarray(PRIOR_MEAN)), atol=0.05
        )
        np.testing.assert_allclose(
            imag.mean(axis=0), np.imag(np.asarray(PRIOR_MEAN)), atol=0.05
        )


class TestTheDeclarationRoute:
    def test_the_block_reads_the_complex_prior_off_the_graph(self):
        block = linear_operator(sky_model(), ("alm",))
        assert block.dtype["alm"] == jnp.complex64
        assert block.shape["alm"] == (2,)
        np.testing.assert_allclose(
            np.asarray(block.prior_mean["alm"]), np.asarray(PRIOR_MEAN), rtol=1e-6
        )
        # Real, and the width of EACH part -- so `variance_parts` duplicating
        # it is the same statement the distribution makes.
        assert not jnp.iscomplexobj(block.prior_std["alm"])
        np.testing.assert_allclose(np.asarray(block.prior_std["alm"]), PRIOR_STD)

    def test_the_graph_route_reaches_the_dense_reference(self):
        """The whole chain: declaration, linearity check, block, solve.

        `linear_operator`, not `unchecked_operator` -- the `linear_in` claim is
        probed on the complex latent too, so this also pins that the linearity
        machinery runs in the complex domain rather than being skipped there.
        """
        graph = sky_model()
        block = linear_operator(graph, ("alm",))
        solution, residual = wiener_solve(
            block,
            precision=precision_at(graph, {"alm": PRIOR_MEAN}),
            tol=1e-10,
            maxiter=200,
        )
        expected, _ = dense_posterior()
        assert float(residual) < 1e-5
        np.testing.assert_allclose(
            _as_parts(solution, "alm"), expected, rtol=2e-4, atol=2e-6
        )

    def test_a_lying_log_prob_is_refused_even_for_a_complex_node(self):
        """The Gaussianity probe has to reach the imaginary half.

        A subclass that keeps `loc` and `scale` and changes the density is
        exactly what `check_gaussian` exists for, and the complex branch would
        be worthless if it only displaced the real part: this liar is correct
        on the real half and wrong on the imaginary one, so a probe that never
        moves the imaginary part accepts it.
        """

        class HalfHonest(ComplexNormal):
            def log_prob(self, value):
                real = (jnp.real(value) - jnp.real(self.loc)) / self.scale
                imag = (jnp.imag(value) - jnp.imag(self.loc)) / self.scale
                return (
                    -0.5 * (real**2 + 7.0 * imag**2)
                    - 2.0 * jnp.log(self.scale)
                    - float(np.log(2.0 * np.pi))
                )

        def model():
            alm = sample("alm", lambda: HalfHonest(PRIOR_MEAN, PRIOR_STD))
            mu = det(
                "mu", lambda a: jnp.real(DESIGN @ a) + OFFSET, alm, linear_in=("alm",)
            )
            observe("d", lambda m: ndist.Normal(m, NOISE_STD), mu, obs=DATA)

        graph = trace(model)
        with pytest.raises(StructureError, match="log_prob"):
            check_gaussian(graph, graph.node("alm"), {})

    @pytest.mark.slow
    def test_nuts_reaches_the_same_posterior_through_the_reparameterisation(self):
        """The package's standing rule, kept rather than exempted.

        `to_numpyro` emits a complex latent as two real sites plus the
        deterministic that recombines them, because HMC steps in real
        unconstrained space. Without that, this graph would qualify for an
        exact method and NOT for NUTS -- and NUTS is what the exact paths are
        checked against, so the exception would have removed the oracle
        exactly where a new solve path most needed one.
        """
        from bayesmith.bridge.numpyro_bridge import nuts

        graph = sky_model()
        draws = nuts(graph, jax.random.key(0), num_warmup=500, num_samples=2000)
        assert {"alm", "alm__re", "alm__im"} <= set(draws)
        assert jnp.iscomplexobj(draws["alm"])

        expected, covariance = dense_posterior()
        got = np.asarray(draws["alm"])
        mean = np.concatenate([np.real(got).mean(axis=0), np.imag(got).mean(axis=0)])

        # The standard error must come from the EFFECTIVE sample size, not the
        # draw count. Measured while writing this: dividing by 2000 puts two
        # of the four components at ~10 sigma from a posterior the exact solve
        # and NUTS actually agree on to three decimals -- NUTS draws are
        # autocorrelated, so the naive error bar is too small and the test
        # would have failed for being wrong about statistics rather than about
        # the code.
        from numpyro.diagnostics import effective_sample_size

        chains = np.stack(
            [np.asarray(draws["alm__re"]), np.asarray(draws["alm__im"])], axis=-1
        )[None]
        ess = effective_sample_size(chains).reshape(-1, order="F")
        assert np.all(ess > 100), ess
        standard_error = np.sqrt(np.diag(covariance) / ess)
        np.testing.assert_array_less(np.abs((mean - expected) / standard_error), 4.0)
