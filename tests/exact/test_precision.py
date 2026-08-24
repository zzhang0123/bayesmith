"""B9's acceptance: the diagonal case must come back numerically."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from bayesmith.exact.precision import (
    CirculantPrecision,
    DiagonalPrecision,
    Precision,
    dense,
    log_density,
    quadratic,
)


def _kernel(size: int = 8, decay: float = 0.45, amplitude: float = 4.0):
    """A symmetric, periodic autocovariance -- correlated but positive definite.

    Symmetric under ``k -> n - k`` by construction rather than by luck, which
    is what makes the matrix symmetric and its FFT real.
    """
    lag = np.minimum(np.arange(size), size - np.arange(size))
    return jnp.asarray(amplitude * decay**lag)


class TestTheDiagonalCaseIsTheOldNumbers:
    """The cross-check B9 names: degeneracy has to be numerical, not moral.

    Compared against the literal expressions the package used before -- not
    against another implementation of the same idea -- because two spellings
    of one formula agree even when both are wrong.
    """

    @pytest.mark.parametrize("size", [1, 5, 32])
    def test_apply_is_division_by_sigma_squared(self, size):
        with jax.enable_x64(True):
            sigma = jnp.linspace(0.2, 1.7, size)
            residual = jnp.linspace(-3.0, 2.0, size)
            found = DiagonalPrecision(sigma=sigma).apply(residual)
        assert np.allclose(np.asarray(found), np.asarray(residual / sigma**2), rtol=0)

    @pytest.mark.parametrize("size", [1, 5, 32])
    def test_log_normalizer_is_the_old_sum(self, size):
        with jax.enable_x64(True):
            sigma = jnp.linspace(0.2, 1.7, size)
            found = DiagonalPrecision(sigma=sigma).log_normalizer()
            expected = jnp.sum(jnp.log(2.0 * jnp.pi * sigma**2))
        assert float(found) == float(expected)

    def test_the_density_is_the_gaussian_the_package_already_writes(self):
        """``-1/2 sum[r^2/sigma^2 + log 2 pi sigma^2]``, spelled out."""
        with jax.enable_x64(True):
            sigma = jnp.linspace(0.3, 1.1, 6)
            residual = jnp.linspace(-2.0, 1.5, 6)
            found = log_density(DiagonalPrecision(sigma=sigma), residual)
            expected = -0.5 * jnp.sum(
                residual**2 / sigma**2 + jnp.log(2.0 * jnp.pi * sigma**2)
            )
        assert float(found) == pytest.approx(float(expected), rel=1e-14)

    def test_a_diagonal_circulant_is_the_diagonal_case(self):
        """The two implementations must meet where they overlap.

        A circulant whose kernel is ``[s, 0, 0, ...]`` IS ``s I``. If the FFT
        route and the division route disagree there, one of them is wrong,
        and this is the only fixture where both are defined on the same
        matrix.
        """
        size, variance = 16, 0.75
        with jax.enable_x64(True):
            kernel = jnp.zeros(size).at[0].set(variance)
            circulant = CirculantPrecision(first_column=kernel)
            diagonal = DiagonalPrecision(sigma=jnp.full((size,), math.sqrt(variance)))
            residual = jnp.linspace(-1.0, 1.0, size)
            assert np.allclose(
                np.asarray(circulant.apply(residual)),
                np.asarray(diagonal.apply(residual)),
                rtol=1e-12,
            )
            assert float(circulant.log_normalizer()) == pytest.approx(
                float(diagonal.log_normalizer()), rel=1e-12
            )


class TestTheCirculantOracleIsDirectMatrixInversion:
    """B9's second acceptance clause, against a dense inverse rather than a
    second FFT.

    The oracle shares nothing with the implementation: scipy builds the matrix
    from the same kernel, numpy inverts it densely, numpy takes its
    log-determinant by LU. If the FFT route were wrong in a way that happened
    to be self-consistent, this is what would say so.
    """

    @pytest.mark.parametrize("size", [4, 8, 15])
    def test_apply_reproduces_the_dense_inverse(self, size):
        with jax.enable_x64(True):
            kernel = _kernel(size)
            found = np.asarray(dense(CirculantPrecision(first_column=kernel), size))
            matrix = sla.circulant(np.asarray(kernel))
        assert np.allclose(found, np.linalg.inv(matrix), rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("size", [4, 8, 15])
    def test_the_normalizer_reproduces_the_dense_log_determinant(self, size):
        with jax.enable_x64(True):
            kernel = _kernel(size)
            found = float(CirculantPrecision(first_column=kernel).log_normalizer())
            matrix = sla.circulant(np.asarray(kernel))
        _, expected = np.linalg.slogdet(2.0 * np.pi * matrix)
        assert found == pytest.approx(expected, rel=1e-10)

    def test_the_quadratic_form_matches_the_dense_one(self):
        with jax.enable_x64(True):
            kernel = _kernel(12)
            residual = jnp.asarray(np.linspace(-2.0, 3.0, 12))
            found = float(quadratic(CirculantPrecision(first_column=kernel), residual))
            matrix = sla.circulant(np.asarray(kernel))
            r = np.asarray(residual)
        expected = float(r @ np.linalg.solve(matrix, r))
        assert found == pytest.approx(expected, rel=1e-10)

    def test_correlation_is_actually_present_in_the_fixture(self):
        """ANTI-VACUITY. Every test above would also pass on a diagonal.

        Without this, a kernel that had quietly become ``[a, 0, 0, ...]`` --
        a decay of 0, a broken construction -- would leave the whole class
        green while testing nothing the diagonal case does not already cover.
        """
        with jax.enable_x64(True):
            matrix = np.asarray(dense(CirculantPrecision(first_column=_kernel(8)), 8))
        off = matrix - np.diag(np.diag(matrix))
        assert np.max(np.abs(off)) > 0.05 * np.max(np.abs(np.diag(matrix)))


class TestTheContractRefusesWhatItCannotDescribe:
    def test_a_kernel_that_is_not_positive_definite_is_refused(self):
        """The FFT answers finitely for an indefinite covariance, so something
        has to be the thing that does not.

        A negative eigenvalue gives ``log`` a NaN and ``apply`` a sign-flipped
        weight in that mode -- a residual pushed the WRONG WAY, which no
        finiteness check downstream would catch.
        """
        from bayesmith.exact.precision import check_positive_definite

        with jax.enable_x64(True):
            # alternating kernel: its highest-frequency eigenvalue goes negative
            kernel = jnp.asarray([1.0, -0.9, 0.8, -0.9])
            with pytest.raises(ValueError, match="positive definite"):
                check_positive_definite(CirculantPrecision(first_column=kernel))

    def test_an_all_positive_kernel_can_still_be_indefinite(self):
        """The check must read the SPECTRUM, not the kernel's own values.

        Found by mutation: replacing ``min(fft(first_column))`` with
        ``min(first_column)`` survived the whole file, because the fixture
        above happens to have negative entries and so is refused for the
        wrong reason. The two are not the same test.

        ``[1, 1.5, 1, 1.5]`` is every-entry-positive and indefinite: for a
        symmetric 4-kernel ``[a, b, c, b]`` the eigenvalues are
        ``a+2b+c, a-c, a-2b+c, a-c``, so ``2b > a + c`` drives one negative --
        here to exactly ``-1``. An autocovariance that stays positive at every
        lag while describing no realisable process is not exotic; it is what
        writing a kernel down by hand tends to produce, which is the case this
        guard exists for.
        """
        from bayesmith.exact.precision import check_positive_definite

        with jax.enable_x64(True):
            kernel = jnp.asarray([1.0, 1.5, 1.0, 1.5])
            assert bool(np.all(np.asarray(kernel) > 0.0))
            with pytest.raises(ValueError, match="positive definite"):
                check_positive_definite(CirculantPrecision(first_column=kernel))

    def test_construction_itself_does_not_validate_and_says_so(self):
        """Nobody should read the class and assume building one is safe.

        The check moved OUT of ``__check_init__`` (see the next test for why),
        so an indefinite kernel now constructs happily. That is a real hazard
        and the reason ``check_positive_definite`` exists as a named, callable
        thing rather than an implicit one -- this pins the hazard so the
        docstring cannot quietly stop being true.
        """
        with jax.enable_x64(True):
            built = CirculantPrecision(first_column=jnp.asarray([1.0, -0.9, 0.8, -0.9]))
        assert built is not None

    @pytest.mark.parametrize("transform", ["jit", "linearize", "grad"])
    def test_a_circulant_can_be_built_under_a_trace(self, transform):
        """Why the check cannot live in the constructor.

        Validating there means reading the smallest eigenvalue as a Python
        float, which CONCRETISES. Measured before this split: building one
        inside ``jit``, ``linearize`` or ``grad`` raised
        ``ConcretizationTypeError`` -- so the class could not be used anywhere
        its kernel was traced, which is the whole solve path, and the failure
        appeared at the call site rather than as anything about validation.

        ``gaussian.py`` already splits ``gaussian_parts`` from
        ``check_gaussian`` for this exact reason. All three transforms are
        checked because they fail independently: ``jit`` alone would not have
        caught the ``grad`` path that ``gcr_sample`` takes.
        """
        lag = np.minimum(np.arange(8), 8 - np.arange(8))

        def build(kernel):
            return CirculantPrecision(first_column=kernel).apply(jnp.ones(8))

        with jax.enable_x64(True):
            kernel = jnp.asarray(4.0 * 0.4**lag)
            if transform == "jit":
                found = jax.jit(build)(kernel)
            elif transform == "linearize":
                found = jax.linearize(build, kernel)[0]
            else:
                found = jax.grad(lambda k: jnp.sum(build(k)))(kernel)
        assert np.all(np.isfinite(np.asarray(found)))

    def test_both_implementations_satisfy_the_protocol(self):
        """``Precision`` is runtime-checkable, so this is a real check."""
        with jax.enable_x64(True):
            assert isinstance(DiagonalPrecision(sigma=jnp.ones(3)), Precision)
            assert isinstance(CirculantPrecision(first_column=_kernel(4)), Precision)

    def test_the_quadratic_form_cannot_disagree_with_the_operator(self):
        """``quadratic`` is derived, so this holds for any implementation.

        Stated as a test anyway because it is the property the module is for:
        a consumer taking ``r^T N^-1 r`` and a consumer taking ``N^-1 r`` are
        reading one object.
        """
        with jax.enable_x64(True):
            precision = CirculantPrecision(first_column=_kernel(8))
            residual = jnp.linspace(-1.0, 2.0, 8)
            assert float(quadratic(precision, residual)) == pytest.approx(
                float(jnp.sum(residual * precision.apply(residual))), rel=1e-15
            )


class TestWhiteningIsTheThirdOperationTheInterfaceNeeded:
    """``N^-1/2``, which ``apply`` and ``log_normalizer`` cannot build.

    ``gcr_sample`` forms ``sqrt(1/sigma**2) * omega`` to turn a standard
    normal draw into one with covariance ``N^-1``. B9's interface named two
    operations and that draw is a third -- the spec mentions the operation for
    the evidence layer ("the whitening row becomes ``L^-1 r``") without
    carrying it into the interface. Found by reading what solve.py actually
    does with its weights rather than what the interface said it would need.
    """

    @pytest.mark.parametrize("size", [4, 8, 15])
    def test_whitening_twice_is_applying_once(self, size):
        """The defining property, and deliberately not "it is the Cholesky".

        ``N^-1/2 N^-1/2 = N^-1`` holds for ANY square root, so this is the
        strongest claim that does not over-specify the implementation. It also
        cannot be satisfied by an implementation that returned its input.
        """
        with jax.enable_x64(True):
            for precision in (
                CirculantPrecision(first_column=_kernel(size)),
                DiagonalPrecision(sigma=jnp.linspace(0.4, 1.3, size)),
            ):
                omega = jnp.asarray(np.linspace(-2.0, 3.0, size))
                twice = precision.whiten(precision.whiten(omega))
                once = precision.apply(omega)
                assert np.allclose(
                    np.asarray(twice), np.asarray(once), rtol=1e-10, atol=1e-12
                ), type(precision).__name__

    def test_the_diagonal_whitening_is_the_old_sqrt_of_the_weight(self):
        """``sqrt(w) * omega`` with ``w = 1/sigma**2`` -- solve.py's literal line."""
        with jax.enable_x64(True):
            sigma = jnp.linspace(0.3, 1.4, 9)
            omega = jnp.linspace(-1.0, 1.0, 9)
            found = DiagonalPrecision(sigma=sigma).whiten(omega)
            expected = jnp.sqrt(1.0 / sigma**2) * omega
        assert np.allclose(np.asarray(found), np.asarray(expected), rtol=1e-14)

    def test_the_drawn_covariance_is_the_inverse(self):
        """What whitening is FOR, measured on draws rather than on algebra.

        ``omega ~ N(0, I)`` whitened must have covariance ``N^-1``. The
        algebraic identity above would still hold for a square root that was
        right up to an orthogonal factor and wrong in some other way; this
        reads the covariance that actually comes out.
        """
        size, count = 8, 200_000
        with jax.enable_x64(True):
            precision = CirculantPrecision(first_column=_kernel(size))
            omega = jax.random.normal(jax.random.key(3), (count, size))
            drawn = jax.vmap(precision.whiten)(omega)
            empirical = np.cov(np.asarray(drawn), rowvar=False)
            target = np.asarray(dense(precision, size))
        assert np.max(np.abs(empirical - target)) < 0.02 * np.max(np.abs(target))


def test_diagonal_from_wraps_a_decided_sigma_dict():
    """The bridge, and that it goes through the protocol rather than around it."""
    from bayesmith.exact.precision import diagonal_from

    with jax.enable_x64(True):
        made = diagonal_from({"d": jnp.linspace(0.5, 1.5, 4)})
        assert set(made) == {"d"}
        assert isinstance(made["d"], Precision)
        residual = jnp.ones(4)
        assert np.allclose(
            np.asarray(made["d"].apply(residual)),
            np.asarray(residual / jnp.linspace(0.5, 1.5, 4) ** 2),
        )


class TestTheMotivatingPhysicsIsActuallyExpressible:
    """1/f drift, as a measurement rather than as a claim in a docstring.

    ``precision.py`` says circulant is the right model for the physics that
    motivated B9 -- 1/f gain drift and atmospheric correlation, both
    stationary. That was asserted, not shown, and an unmeasured claim in a
    docstring is what this project treats as a defect. So it is shown here on
    the EXISTING api: no constructor is added, because nothing can yet declare
    a correlated noise on a graph node, and building a constructor ahead of
    the caller that would use it is the mistake of shipping machinery with no
    reader.

    The construction inverts the obvious approach, and that is the point. A
    circulant's eigenvalues ARE its power spectrum, so noise specified the way
    instruments specify it -- a knee frequency and a slope -- is declared by
    writing the SPECTRUM and taking one inverse FFT. Writing an autocovariance
    down directly and hoping it comes out positive definite is the harder
    route, not the easier one.
    """

    @staticmethod
    def _flicker(size: int, alpha: float, knee: float):
        """``S(f) = 1 + (knee/f)**alpha`` -- white noise plus a 1/f^alpha tail.

        Returns NUMPY arrays; the caller converts inside its own x64 context.
        Returning a jax array from here would build it outside that context,
        where float64 is unavailable, and the kernel would arrive as float32 --
        surfacing not as a dtype complaint but as the recovered spectrum
        missing its target at the eighth digit. That trap has now been hit
        twice in this work; the rule is that ``jax.enable_x64`` governs the
        OPERATION, never an array that already exists.
        """
        freq = np.abs(np.fft.fftfreq(size, d=1.0))
        spectrum = 1.0 + (knee / np.maximum(freq, 1.0 / size)) ** alpha
        return spectrum, np.real(np.fft.ifft(spectrum))

    @pytest.mark.parametrize(
        ("size", "alpha", "knee"),
        [(64, 1.0, 0.05), (256, 1.0, 0.05), (256, 2.0, 0.05), (256, 2.0, 0.005)],
    )
    def test_a_flicker_spectrum_gives_a_valid_covariance(self, size, alpha, knee):
        """Positive definite, and its spectrum comes back out unchanged.

        The second half is what makes "declare it by its spectrum" a workflow
        rather than a coincidence: the eigenvalues must BE the PSD that was
        asked for, not merely resemble it.
        """
        spectrum, kernel = self._flicker(size, alpha, knee)
        with jax.enable_x64(True):
            precision = CirculantPrecision(first_column=jnp.asarray(kernel))
            eigenvalues = np.asarray(precision.eigenvalues)
        assert eigenvalues.min() > 0.0
        assert np.allclose(eigenvalues, spectrum, rtol=1e-10, atol=1e-12)

    def test_the_kernel_really_is_correlated_and_decays(self):
        """ANTI-VACUITY: a flat spectrum would pass everything above.

        ``1 + (knee/f)**alpha`` degenerates to white noise as its 1/f term
        vanishes, and white noise is the DIAGONAL case -- every other
        assertion in this class would still hold while testing nothing the
        diagonal tests do not already cover. Measured: neutralising that term
        in the fixture turns this red and nothing else.
        """
        _, values = self._flicker(256, alpha=2.0, knee=0.05)
        assert abs(values[1]) > 0.05 * abs(values[0]), "no correlation at lag 1"
        near = float(np.mean(np.abs(values[1:9])))
        far = float(np.mean(np.abs(values[100:129])))
        assert near > 5.0 * far, (near, far)

    def test_flicker_noise_does_not_strain_the_fisher_condition_ceiling(self):
        """A worry that measurement did NOT bear out, recorded so it is not
        raised again.

        1/f has a large dynamic range between DC and Nyquist, so the natural
        fear is that a realistic flicker model gives an ill-conditioned ``N``,
        that ``F = J^T N^-1 J`` squares it, and that
        ``fisher.parameter_covariance`` then refuses on its own ceiling of
        ``1/sqrt(eps) = 6.71e+07``.

        Measured, it does not come close at instrument-plausible parameters.
        The worst of the cases here is n=256, alpha=2, knee=0.05:
        ``kappa(N) = 1.6e+02``, so ``kappa(F)`` of order ``2.7e+04`` -- three
        orders inside the ceiling. The dynamic range stays bounded because the
        white floor of the ``1 +`` term sets ``lambda_min``, and the 1/f tail
        is cut off at the lowest resolvable frequency ``1/n`` rather than
        running to zero.

        This asserts the HEADROOM rather than the exact number, so it fails if
        a future kernel quietly loses its white floor -- which is the change
        that would make the fear real.
        """
        ceiling = 1.0 / math.sqrt(float(np.finfo(np.float64).eps))
        worst = 0.0
        for size, alpha, knee in ((64, 1.0, 0.05), (256, 2.0, 0.05), (256, 2.0, 0.005)):
            _, kernel = self._flicker(size, alpha, knee)
            with jax.enable_x64(True):
                eigenvalues = np.asarray(
                    CirculantPrecision(first_column=jnp.asarray(kernel)).eigenvalues
                )
            worst = max(worst, float(eigenvalues.max() / eigenvalues.min()))
        assert worst < 1.0e3, worst
        assert worst**2 < ceiling / 1.0e3, (worst**2, ceiling)
