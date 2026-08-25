"""The square-root information kernel, against dense references.

The oracle throughout is a DENSE Gaussian computed in NumPy: a log-density
evaluated as ``-1/2 (x-m)^T F (x-m) + const``, or an integral done by
``slogdet``. Nothing here compares one of our own routines against another.

**Absolute log-densities, never shapes.** Every constant this module carries
is invisible in a posterior's shape and visible only in the evidence, which
is precisely how rheplicant shipped the marginalisation constant with a term
missing: the probe that passed used unit priors, where that term is exactly
zero.
"""


import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.evidence import SqrtInfo, marginalise, marginalise_arrays


def _term(rows, width, seed, names=("x",), shapes=None, offset=0.0):
    """A random ``[R | z]`` over one flat latent."""
    rng = np.random.default_rng(seed)
    return SqrtInfo(
        factor=jnp.asarray(rng.normal(size=(rows, width))),
        target=jnp.asarray(rng.normal(size=rows)),
        offset=jnp.asarray(offset),
        names=names,
        shapes=shapes or ((width,),),
    )


def _dense_log_prob(term, x):
    """``-1/2 ||R x - z||^2 + offset`` in NumPy, from the arrays."""
    residual = np.asarray(term.factor) @ np.asarray(x) - np.asarray(term.target)
    return float(term.offset) - 0.5 * float(residual @ residual)


class TestTheFormItself:
    def test_log_prob_is_the_dense_quadratic(self):
        with jax.enable_x64(True):
            term = _term(5, 3, seed=0, offset=1.25)
            x = jnp.asarray([0.3, -1.1, 2.0])
            assert float(term.log_prob({"x": x})) == pytest.approx(
                _dense_log_prob(term, x), rel=1e-12
            )

    def test_information_is_positive_semi_definite_even_when_rank_deficient(self):
        """The property the form exists for: it cannot go indefinite.

        Two rows over three columns is rank-deficient on purpose -- one epoch
        constraining a subspace, which is the normal case and not an error.
        """
        # Both reads INSIDE the context. `information()` taken outside it
        # recomputes `R^T R` at float32, where the rank-deficient direction's
        # singular value comes back at ~1e-7 instead of ~5e-18 and
        # `matrix_rank` counts it -- reporting 3 for a matrix that cannot have
        # rank above 2. A dtype problem presenting as a linear-algebra one.
        with jax.enable_x64(True):
            term = _term(2, 3, seed=1)
            information = np.asarray(term.information())
            eigenvalues = np.linalg.eigvalsh(information)
            assert eigenvalues.min() > -1e-12, eigenvalues
            assert np.linalg.matrix_rank(information) == 2
            assert np.linalg.svd(information, compute_uv=False)[-1] < 1e-15

    def test_a_term_whose_columns_do_not_match_its_names_is_refused(self):
        with jax.enable_x64(True), pytest.raises(StructureError, match="ravel to"):
            SqrtInfo(
                factor=jnp.zeros((2, 3)),
                target=jnp.zeros(2),
                offset=jnp.zeros(()),
                names=("x",),
                shapes=((4,),),
            )


class TestAccumulationIsAQR:
    """`combine` must be the SUM of the two densities, constant included."""

    def test_combining_two_terms_adds_their_log_densities(self):
        """The `-1/2 rho^2` corner is what makes this true, and it is a
        constant -- so this compares an ABSOLUTE log-density, at three points,
        against the two dense quadratics added.
        """
        with jax.enable_x64(True):
            first, second = _term(4, 3, seed=2), _term(5, 3, seed=3, offset=-0.75)
            merged = SqrtInfo.combine(first, second)
            for point in ([0.0, 0.0, 0.0], [1.5, -0.5, 2.0], [-3.0, 4.0, 0.25]):
                x = jnp.asarray(point)
                expected = _dense_log_prob(first, x) + _dense_log_prob(second, x)
                assert float(merged.log_prob({"x": x})) == pytest.approx(
                    expected, rel=1e-10, abs=1e-10
                ), point

    def test_dropping_the_corner_is_wrong_and_this_says_by_how_much(self):
        """ANTI-VACUITY. The corner is the whole reason `combine` is not a
        concatenation, and its absence is invisible in the posterior's shape.

        The gap is asserted against the corner's OWN value rather than
        against a magnitude, so this says why it is what it is: on this
        fixture the corner is a single entry `1.01934956`, and the gap is
        exactly `-1/2 * 1.01934956**2 = -0.5195367591`, identical at every
        point. A test comparing shapes, gradients or argmaxes would see none
        of it.
        """
        with jax.enable_x64(True):
            first, second = _term(4, 3, seed=2), _term(5, 3, seed=3)
            merged = SqrtInfo.combine(first, second)
            without = SqrtInfo(
                factor=merged.factor,
                target=merged.target,
                offset=first.offset + second.offset,  # corner dropped
                names=merged.names,
                shapes=merged.shapes,
            )
            gaps = [
                float(merged.log_prob({"x": jnp.asarray(p)}))
                - float(without.log_prob({"x": jnp.asarray(p)}))
                for p in ([0.0, 0.0, 0.0], [1.5, -0.5, 2.0], [-3.0, 4.0, 0.25])
            ]
            # the corner, computed here from the stacked matrix rather than
            # read out of `combine`, so the two are not the same arithmetic
            stacked = np.concatenate(
                [
                    np.concatenate(
                        [np.asarray(t.factor), np.asarray(t.target)[:, None]], axis=1
                    )
                    for t in (first, second)
                ],
                axis=0,
            )
            upper = np.linalg.qr(stacked, mode="r")
            corner = upper[3:, 3]
        expected = -0.5 * float(corner @ corner)
        assert expected == pytest.approx(-0.5195367591, rel=1e-9)
        assert gaps[0] == pytest.approx(expected, rel=1e-9)
        assert max(gaps) - min(gaps) < 1e-9, "a constant, at every point"

    @pytest.mark.parametrize("order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
    def test_accumulation_is_order_invariant_and_associative(self, order):
        """Order-invariance holds by construction, and this measures it.

        Not to `approx` on the shape -- on the absolute log-density, because
        an offset that failed to accumulate associatively would leave every
        ordering agreeing about the posterior and disagreeing about the
        evidence.
        """
        with jax.enable_x64(True):
            terms = [_term(3, 3, seed=10 + i, offset=0.5 * i) for i in range(3)]
            reference = SqrtInfo.combine(
                SqrtInfo.combine(terms[0], terms[1]), terms[2]
            )
            shuffled = SqrtInfo.combine(
                terms[order[0]],
                SqrtInfo.combine(terms[order[1]], terms[order[2]]),
            )
            x = jnp.asarray([0.7, -1.3, 0.2])
            assert float(shuffled.log_prob({"x": x})) == pytest.approx(
                float(reference.log_prob({"x": x})), rel=1e-10, abs=1e-10
            )

    def test_streaming_equals_batch(self):
        """THE oracle the spec names, on this kernel.

        Folding ten per-epoch terms one at a time must give the same density
        as stacking all ten and factorising once. That is the statement the
        whole streaming layer rests on, and it is checked here at the level
        where it is a property of the arithmetic rather than of a pipeline.
        """
        with jax.enable_x64(True):
            terms = [_term(2, 4, seed=100 + i, offset=0.1 * i) for i in range(10)]
            streamed = terms[0]
            for term in terms[1:]:
                streamed = SqrtInfo.combine(streamed, term)
            batch = SqrtInfo(
                factor=jnp.concatenate([t.factor for t in terms], axis=0),
                target=jnp.concatenate([t.target for t in terms]),
                offset=jnp.asarray(sum(float(t.offset) for t in terms)),
                names=terms[0].names,
                shapes=terms[0].shapes,
            )
            x = jnp.asarray([0.4, -0.9, 1.7, 0.05])
            assert float(streamed.log_prob({"x": x})) == pytest.approx(
                float(batch.log_prob({"x": x})), rel=1e-10, abs=1e-10
            )

    def test_null_is_the_identity_of_combine(self):
        with jax.enable_x64(True):
            term = _term(3, 3, seed=7, offset=2.5)
            merged = SqrtInfo.combine(term, SqrtInfo.null(term.names, term.shapes))
            x = jnp.asarray([1.0, 2.0, -0.5])
            assert float(merged.log_prob({"x": x})) == pytest.approx(
                float(term.log_prob({"x": x})), rel=1e-10, abs=1e-10
            )

    def test_terms_over_different_latents_are_refused(self):
        with jax.enable_x64(True):
            a, b = _term(2, 3, seed=1), _term(2, 3, seed=2, names=("y",))
            with pytest.raises(StructureError, match="different latents"):
                SqrtInfo.combine(a, b)


def _joint_with_prior(n_block, n_keep, prior_std, seed=41, rows=9):
    """A joint term over ``(b, k)`` with the block's prior rows appended.

    The prior is NOT unit. rheplicant shipped the marginalisation constant
    missing ``-sum(log(std))``, which is exactly zero at ``std = 1`` -- the
    probe that passed used unit priors. Every fixture here uses a prior that
    would expose it.
    """
    rng = np.random.default_rng(seed)
    width = n_block + n_keep
    data_rows = jnp.asarray(rng.normal(size=(rows, width)))
    prior_rows = jnp.concatenate(
        [jnp.eye(n_block) / prior_std, jnp.zeros((n_block, n_keep))], axis=1
    )
    return SqrtInfo(
        factor=jnp.concatenate([data_rows, prior_rows], axis=0),
        target=jnp.concatenate(
            [jnp.asarray(rng.normal(size=rows)), jnp.zeros(n_block)]
        ),
        # the prior rows' own normalisation, which the caller owns
        offset=jnp.asarray(
            -n_block * math.log(prior_std) - 0.5 * n_block * math.log(2.0 * math.pi)
        ),
        names=("b", "k"),
        shapes=((n_block,), (n_keep,)),
    )


def _dense_marginal(term, n_block, keep_value):
    """``log int exp(term) db`` at fixed ``k``, by NumPy `slogdet`.

    Splits the quadratic by hand: ``F = R^T R`` and ``g = R^T z`` over the
    joint, then completes the square in the block. Shares no arithmetic with
    `marginalise` -- no QR, no pivots.
    """
    factor = np.asarray(term.factor)
    target = np.asarray(term.target)
    information = factor.T @ factor
    gradient = factor.T @ target
    keep = np.asarray(keep_value)

    block_block = information[:n_block, :n_block]
    block_keep = information[:n_block, n_block:]
    rhs = gradient[:n_block] - block_keep @ keep
    solved = np.linalg.solve(block_block, rhs)
    sign, logdet = np.linalg.slogdet(block_block)
    assert sign > 0, "the block must be positive definite for this reference"

    # log int exp(-1/2 b^T A b + b^T rhs) db, plus the k-only part
    full = np.concatenate([solved, keep])
    at_optimum = float(term.offset) - 0.5 * float(
        (factor @ full - target) @ (factor @ full - target)
    )
    return at_optimum + 0.5 * n_block * math.log(2.0 * math.pi) - 0.5 * logdet


class TestTheMarginalisationConstant:
    """The Gaussian integral's constant, against `slogdet`, at a NON-unit prior."""

    @pytest.mark.parametrize("prior_std", [0.7, 1.0, 3.0])
    @pytest.mark.parametrize("n_block, n_keep", [(2, 3), (3, 2), (1, 4)])
    def test_marginalising_matches_a_dense_gaussian_integral(
        self, prior_std, n_block, n_keep
    ):
        with jax.enable_x64(True):
            term = _joint_with_prior(n_block, n_keep, prior_std)
            reduced = marginalise(term, ["b"])
            keep = jnp.asarray(np.linspace(-1.0, 1.5, n_keep))
            got = float(reduced.log_prob({"k": keep}))
            expected = _dense_marginal(term, n_block, keep)
        assert reduced.names == ("k",)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_marginalising_nothing_is_the_identity_on_the_density(self):
        with jax.enable_x64(True):
            term = _joint_with_prior(2, 3, 0.7)
            same = marginalise(term, [])
            values = {"b": jnp.asarray([0.3, -1.0]), "k": jnp.asarray([1.0, 0.0, 2.0])}
            assert float(same.log_prob(values)) == pytest.approx(
                float(term.log_prob(values)), rel=1e-10, abs=1e-10
            )

    def test_marginalising_everything_gives_the_marginal_likelihood(self):
        with jax.enable_x64(True):
            term = _joint_with_prior(2, 3, 0.7)
            everything = marginalise(term, ["b", "k"])
            assert everything.names == ()
            assert everything.factor.shape == (0, 0)
            evidence = float(everything.log_prob({}))
            # against a dense evidence: -1/2 z^T z + 1/2 g^T F^-1 g
            #                            + n/2 log 2pi - 1/2 logdet F, + offset
            factor = np.asarray(term.factor)
            target = np.asarray(term.target)
            information, gradient = factor.T @ factor, factor.T @ target
            _, logdet = np.linalg.slogdet(information)
            width = information.shape[0]
            expected = (
                float(term.offset)
                - 0.5 * float(target @ target)
                + 0.5 * float(gradient @ np.linalg.solve(information, gradient))
                + 0.5 * width * math.log(2.0 * math.pi)
                - 0.5 * logdet
            )
        assert evidence == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_the_prior_normalisation_term_is_not_this_functions_to_add(self):
        """The defect rheplicant shipped, pinned by its own signature.

        The block's own `-sum(log(std))` belongs to whoever appended the prior
        rows. Dropping it is invisible at `std = 1` -- exactly zero -- and
        grows with the prior's width and scale everywhere else. Measured here
        at three widths against the dense integral: the gap IS
        `n_block * log(std)`, so a fixture at unit prior could not see it.
        """
        with jax.enable_x64(True):
            keep = jnp.asarray([0.5, -0.25, 1.0])
            for prior_std in (0.7, 1.0, 3.0):
                for n_block in (1, 3, 5):
                    term = _joint_with_prior(n_block, 3, prior_std)
                    honest = float(marginalise(term, ["b"]).log_prob({"k": keep}))
                    missing = SqrtInfo(
                        factor=term.factor,
                        target=term.target,
                        offset=term.offset + n_block * math.log(prior_std),
                        names=term.names,
                        shapes=term.shapes,
                    )
                    dropped = float(marginalise(missing, ["b"]).log_prob({"k": keep}))
                    gap = dropped - honest
                    assert gap == pytest.approx(
                        n_block * math.log(prior_std), rel=1e-9, abs=1e-12
                    ), (prior_std, n_block)
                    if prior_std == 1.0:
                        assert gap == pytest.approx(0.0, abs=1e-12)


class TestTheRefusalsAreReal:
    def test_an_unconstrained_block_is_refused_rather_than_integrated(self):
        """An unconstrained direction makes the integral divergent, and finite
        arithmetic returns a large plausible number for it instead."""
        with jax.enable_x64(True):
            term = _joint_with_prior(2, 3, 0.7)
            crippled = SqrtInfo(
                factor=term.factor.at[:, 0].set(0.0),  # b[0] constrains nothing
                target=term.target,
                offset=term.offset,
                names=term.names,
                shapes=term.shapes,
            )
            with pytest.raises(StructureError, match="does not constrain"):
                marginalise(crippled, ["b"])

    @pytest.mark.parametrize("poison", [np.nan, np.inf])
    def test_a_poisoned_term_is_refused_rather_than_marginalised_to_nan(self, poison):
        """BOTH ends, because the degeneracy threshold is relative.

        The pivot test compares against `max(pivots)`, so one `nan` makes the
        threshold `nan` and every comparison against it False, while one `inf`
        makes it `inf` and admits every pivot there is. rheplicant measured
        both being ACCEPTED, with `offset` nan, past a `__check_init__` that
        validates shapes only.
        """
        with jax.enable_x64(True):
            term = _joint_with_prior(2, 3, 0.7)
            poisoned = SqrtInfo(
                factor=term.factor.at[0, 0].set(poison),
                target=term.target,
                offset=term.offset,
                names=term.names,
                shapes=term.shapes,
            )
            with pytest.raises(StructureError, match="not finite"):
                marginalise(poisoned, ["b"])

    def test_a_repeated_or_unknown_name_is_refused(self):
        with jax.enable_x64(True):
            term = _joint_with_prior(2, 3, 0.7)
            with pytest.raises(StructureError, match="names a latent twice"):
                marginalise(term, ["b", "b"])
            with pytest.raises(StructureError, match="not over"):
                marginalise(term, ["nope"])


class TestTheKernelAndTheCheckedPathAgree:
    def test_they_return_the_same_numbers(self):
        """`marginalise_arrays` is what a traced caller reaches for; it must be
        the same arithmetic, not a second implementation of it."""
        with jax.enable_x64(True):
            term = _joint_with_prior(2, 3, 0.7)
            checked = marginalise(term, ["b"])
            factor, target, offset, pivots = marginalise_arrays(
                term.factor, term.target, term.offset, 2
            )
            assert jnp.allclose(factor, checked.factor)
            assert jnp.allclose(target, checked.target)
            assert float(offset) == pytest.approx(float(checked.offset), rel=1e-12)
            assert bool(jnp.all(jnp.isfinite(pivots)))

    def test_the_kernel_traces_where_the_checked_path_cannot(self):
        """The whole reason the two exist: `grad` is the half that matters.

        A correlation time inferred rather than pinned is differentiated on
        every leapfrog step, so a `marginalise` that could be jitted but not
        differentiated would still be unusable inside a filter.

        The term itself has to be traced for this to mean anything. Closing
        over a concrete `SqrtInfo` and jitting only the multiplication runs
        `marginalise` eagerly and raises nothing -- which is how a test like
        this passes while checking that a lambda can multiply.

        Both errors are pinned and they are DIFFERENT, which rheplicant's own
        docstring does not say: `jit` trips the `bool(jnp.all(...))` in the
        pivot guard and gives `TracerBoolConversionError`, while `grad` trips
        the `float(jnp.max(pivots))` and gives `ConcretizationTypeError`.
        """
        with jax.enable_x64(True):
            term = _joint_with_prior(2, 3, 0.7)

            def kernel(scale):
                _, _, offset, _ = marginalise_arrays(
                    term.factor * scale, term.target, term.offset, 2
                )
                return offset

            def checked(scale):
                scaled = SqrtInfo(
                    factor=term.factor * scale,
                    target=term.target,
                    offset=term.offset,
                    names=term.names,
                    shapes=term.shapes,
                )
                return marginalise(scaled, ["b"]).offset

            assert bool(jnp.isfinite(jax.jit(kernel)(1.0)))
            assert bool(jnp.isfinite(jax.grad(kernel)(1.0)))
            with pytest.raises(jax.errors.TracerBoolConversionError):
                jax.jit(checked)(1.0)
            with pytest.raises(jax.errors.ConcretizationTypeError):
                jax.grad(checked)(1.0)
