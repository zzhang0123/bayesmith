"""B9's acceptance: the diagonal case must come back numerically."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from bayesmith import evaluate
from bayesmith.errors import NotGaussian
from bayesmith.exact.precision import (
    CirculantPrecision,
    DiagonalPrecision,
    Precision,
    PrecisionMismatch,
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
        so an indefinite kernel still constructs happily. That is a real
        hazard and the reason ``check_positive_definite`` exists as a named,
        callable thing rather than an implicit one -- this pins the hazard so
        the docstring cannot quietly stop being true.

        It now HAS a caller, which is a different claim and a different test:
        ``TestThePositiveDefiniteCheckIsWiredIntoTheBuildPath`` below covers
        the build path. Construction staying permissive is not a gap that
        wiring closes -- the class must remain traceable, so this stays true
        for as long as the split does.
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


class TestCheckPrecisionSeparatesWhatTheScalarProbeCannot:
    """The guard the correlated path needs, and the measurement of its power.

    ``check_gaussian`` probes ``log_prob`` at five multiples of
    ``sqrt(diag N)``. For a STATIONARY covariance that is constant, so the
    five offsets are five points along one direction and the family pins two
    scalars of the covariance. The A/B pair below is 0.4060 apart in relative
    Frobenius norm and agrees at every one of those offsets to 1.851e-16.

    So this class asserts the SEPARATION, not merely the verdict. A guard that
    accepts the right pairing tells you nothing on its own; what matters is
    the ratio between what it reports for a matched pairing and for a
    mismatched one, and that ratio is what a future change could quietly
    erode.
    """

    #: Scaling one conjugate mode pair by ``t`` and another by ``1/t``.
    #: The product over modes is unchanged, so ``sum log lambda`` -- the
    #: log-determinant, hence the normaliser -- is EXACTLY equal; and mode 0
    #: is untouched, so ``lambda_0`` is too. Those are the two scalars the
    #: scalar-offset family constrains. Conjugate PAIRS because a real
    #: symmetric kernel needs ``lambda_k == lambda_{n-k}``; scaling one mode
    #: alone would give a complex kernel and prove nothing.
    MODE_SCALE: float = 2.0

    @staticmethod
    def _pair(size: int = 8):
        """Two circulant covariances the scalar-offset family cannot tell apart.

        Measured on this construction: ``lambda_0`` identical, ``sum log
        lambda`` different by exactly 0.0, both positive definite, and 0.4212
        apart in relative Frobenius norm.
        """
        base = np.array([4.0, 2.0, 1.5, 1.0, 0.8, 1.0, 1.5, 2.0])[:size]
        spectrum_a = np.real(np.fft.fft(base))
        spectrum_b = spectrum_a.copy()
        scale = TestCheckPrecisionSeparatesWhatTheScalarProbeCannot.MODE_SCALE
        spectrum_b[1] *= scale
        spectrum_b[size - 1] *= scale
        spectrum_b[2] /= scale
        spectrum_b[size - 2] /= scale
        return (
            np.real(np.fft.ifft(spectrum_a)),
            np.real(np.fft.ifft(spectrum_b)),
        )

    def test_the_fixture_is_the_blind_spot_it_claims_to_be(self):
        """ANTI-VACUITY for the whole class, and it comes first deliberately.

        Every test below is about a guard separating A from B. If A and B
        differed in the two scalars the OLD family already sees, none of them
        would be demonstrating anything the old family could not do. So:
        identical ``lambda_0``, identical ``sum log lambda``, both positive
        definite, and genuinely far apart.
        """
        kernel_a, kernel_b = self._pair()
        spectrum_a = np.real(np.fft.fft(kernel_a))
        spectrum_b = np.real(np.fft.fft(kernel_b))
        assert spectrum_a[0] == pytest.approx(spectrum_b[0], rel=1e-14)
        assert float(np.sum(np.log(spectrum_a))) == pytest.approx(
            float(np.sum(np.log(spectrum_b))), abs=1e-12
        )
        assert spectrum_b.min() > 0.0
        apart = float(np.linalg.norm(kernel_a - kernel_b) / np.linalg.norm(kernel_a))
        assert apart > 0.2, apart

    def test_the_matched_pairing_passes_and_the_mismatched_one_is_refused(self):
        """Both directions, because either alone is satisfiable by a stub."""
        import numpyro.distributions as ndist

        from bayesmith.exact.precision import PrecisionMismatch, check_precision

        with jax.enable_x64(True):
            kernel_a, kernel_b = self._pair()
            loc = jnp.zeros(8, dtype=jnp.float64)
            node_a = ndist.CirculantNormal(
                loc=loc, covariance_row=jnp.asarray(kernel_a)
            )
            precision_a = CirculantPrecision(first_column=jnp.asarray(kernel_a))
            precision_b = CirculantPrecision(first_column=jnp.asarray(kernel_b))

            matched = check_precision(node_a, precision_a, loc)
            with pytest.raises(PrecisionMismatch, match="operator"):
                check_precision(node_a, precision_b, loc)
        assert matched["operator"] <= 1e-12, matched
        assert matched["linearity"] <= 1e-12, matched

    def test_the_separation_is_pinned_not_only_the_verdict(self):
        """How much power the guard has, as a number.

        Measured: the matched pairing reports ~1e-16 on the operator and the
        mismatched one ~1e-1, a separation of order 1e+14. Asserting only
        "one passes and one raises" would still hold if the guard's power
        collapsed to a hair above the tolerance.
        """
        from bayesmith.exact.precision import check_precision

        with jax.enable_x64(True):
            kernel_a, kernel_b = self._pair()
            loc = jnp.zeros(8, dtype=jnp.float64)
            import numpyro.distributions as ndist

            node_a = ndist.CirculantNormal(
                loc=loc, covariance_row=jnp.asarray(kernel_a)
            )
            matched = check_precision(
                node_a, CirculantPrecision(first_column=jnp.asarray(kernel_a)), loc
            )
            mismatched = check_precision(
                node_a,
                CirculantPrecision(first_column=jnp.asarray(kernel_b)),
                loc,
                # Read the numbers without raising. `inf` rather than a large
                # finite value, so this cannot start raising the day the
                # separation grows -- and NaN is still refused, since
                # `not (nan <= inf)` is True.
                rtol=float("inf"),
            )
        assert mismatched["operator"] >= 1e-2, mismatched
        assert matched["operator"] <= 1e-12, matched
        assert mismatched["operator"] / max(matched["operator"], 1e-18) >= 1e10

    def test_the_normalizer_alone_cannot_separate_the_pair(self):
        """Which is why the guard returns two numbers and checks both.

        A and B share a normaliser by construction, so a guard that checked
        only ``log_prob`` at the mode would pass the mismatched pairing. This
        is the clause that makes the operator check load-bearing rather than
        redundant.
        """
        from bayesmith.exact.precision import check_precision

        with jax.enable_x64(True):
            kernel_a, kernel_b = self._pair()
            loc = jnp.zeros(8, dtype=jnp.float64)
            import numpyro.distributions as ndist

            node_a = ndist.CirculantNormal(
                loc=loc, covariance_row=jnp.asarray(kernel_a)
            )
            mismatched = check_precision(
                node_a,
                CirculantPrecision(first_column=jnp.asarray(kernel_b)),
                loc,
                rtol=float("inf"),
            )
        assert mismatched["normalizer"] <= 1e-12, (
            "the fixture no longer shares a normaliser, so this test has "
            f"stopped making its point: {mismatched}"
        )
        assert mismatched["operator"] >= 1e-2, mismatched

    def test_the_diagonal_case_degenerates_to_the_old_answer(self):
        """A ``Normal`` node must pass against its own DiagonalPrecision.

        The acceptance clause B9 sets for every part of this work: the
        existing behaviour has to come back, numerically. Here it also
        strengthens the diagonal path, because the gradient probe checks
        ``1/sigma**2`` at EVERY entry where the scalar family checks the
        density at five offsets.
        """
        import numpyro.distributions as ndist

        from bayesmith.exact.precision import check_precision

        with jax.enable_x64(True):
            sigma = jnp.linspace(0.4, 1.6, 9, dtype=jnp.float64)
            loc = jnp.linspace(-1.0, 1.0, 9, dtype=jnp.float64)
            node = ndist.Normal(loc, sigma)
            errors = check_precision(node, DiagonalPrecision(sigma=sigma), loc)
        assert max(errors.values()) <= 1e-12, errors

    def test_a_non_quadratic_density_is_refused_as_such(self):
        """``linearity`` names the case where the other two are meaningless.

        A density that is not quadratic has no covariance to extract, so
        reporting "the covariance is wrong" would send the reader to fix the
        wrong thing.
        """
        from bayesmith.exact.precision import PrecisionMismatch, check_precision

        class Cauchyish:
            """Not Gaussian: a heavy-tailed log-density, so grad is not linear."""

            def log_prob(self, value):
                return -jnp.log1p(value**2)

        with jax.enable_x64(True):
            loc = jnp.zeros(6, dtype=jnp.float64)
            # `match="linearity="` -- with the equals sign -- because the
            # message EXPLAINS all three names in prose, so a bare
            # "linearity" matches the explanation rather than the finding.
            # Found by mutation: forcing the linearity term to 0.0 left this
            # test green, since `operator` also fires here and the word still
            # appeared in the text. The same self-matching trap that turned
            # the docs guard from a regex into an AST walk.
            with pytest.raises(PrecisionMismatch, match=r"linearity=[0-9]"):
                check_precision(Cauchyish(), DiagonalPrecision(sigma=jnp.ones(6)), loc)

    def test_a_wrong_normalizer_is_caught_when_the_operator_is_right(self):
        """The half the A/B pair cannot exercise, because it shares a normaliser.

        Found by mutation: forcing the `normalizer` term to 0.0 left the whole
        file green. Every fixture here either agrees on everything or differs
        in the OPERATOR, so nothing was asking whether a wrong log-determinant
        is noticed. A covariance scaled by a constant is exactly that case in
        the wild -- same correlation structure, wrong overall scale -- and it
        is invisible to the gradient, which the normaliser does not enter.
        """
        from bayesmith.exact.precision import PrecisionMismatch, check_precision

        class RightOperatorWrongNormalizer(eqx.Module):
            """Delegates `apply`, and lies about `log det 2 pi N` alone."""

            inner: DiagonalPrecision

            def apply(self, residual):
                return self.inner.apply(residual)

            def whiten(self, omega):
                return self.inner.whiten(omega)

            def log_normalizer(self):
                return self.inner.log_normalizer() + 5.0

        import numpyro.distributions as ndist

        with jax.enable_x64(True):
            sigma = jnp.linspace(0.5, 1.2, 7, dtype=jnp.float64)
            loc = jnp.zeros(7, dtype=jnp.float64)
            node = ndist.Normal(loc, sigma)
            honest = DiagonalPrecision(sigma=sigma)
            liar = RightOperatorWrongNormalizer(inner=honest)
            # the operator half is untouched, which is what makes this the
            # normaliser's own test rather than a second operator test
            errors = check_precision(node, liar, loc, rtol=float("inf"))
            assert errors["operator"] <= 1e-12, errors
            assert errors["normalizer"] >= 1e-2, errors
            with pytest.raises(PrecisionMismatch, match=r"normalizer=[0-9]"):
                check_precision(node, liar, loc)

    def test_a_non_finite_error_is_refused_rather_than_compared(self):
        """`not value <= tolerance`, which `value > tolerance` would not give.

        Found by mutation: rewriting the gate as `value > tolerance` left the
        file green, because nothing produced a NaN. NaN fails every
        comparison, so `>` is False and a Precision that has gone non-finite
        would be ACCEPTED -- reported as agreeing with a distribution it
        cannot even be compared against.
        """
        import numpyro.distributions as ndist

        from bayesmith.exact.precision import PrecisionMismatch, check_precision

        with jax.enable_x64(True):
            loc = jnp.zeros(5, dtype=jnp.float64)
            node = ndist.Normal(loc, jnp.ones(5, dtype=jnp.float64))
            broken = DiagonalPrecision(
                sigma=jnp.asarray([1.0, jnp.nan, 1.0, 1.0, 1.0], dtype=jnp.float64)
            )
            with pytest.raises(PrecisionMismatch):
                check_precision(node, broken, loc)


class TestThePositiveDefiniteCheckIsWiredIntoTheBuildPath:
    """B9 step 4's explicit obligation: the check has a caller now.

    Split out of ``__check_init__`` in ``296d911`` so the class could be
    traced, which left it correct, mutation-checked and called by nothing.
    """

    @staticmethod
    def _graph(kernel, n=4):
        import numpyro.distributions as ndist

        from bayesmith import const, det, observe, sample, trace

        grid = jnp.linspace(1.0, 4.0, n)

        def model():
            xs = const("X", grid)
            w = sample("w", lambda: ndist.Normal(0.0, 5.0))
            mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
            observe(
                "d",
                lambda m: ndist.CirculantNormal(m, kernel),
                mu,
                depends_on_prediction=False,
                obs=2.0 * grid,
            )

        return trace(model)

    @pytest.mark.parametrize(
        "label, kernel",
        [
            ("all entries positive", [1.0, 1.5, 1.0, 1.5]),
            ("alternating", [1.0, -0.9, 0.8, -0.9]),
        ],
    )
    def test_an_indefinite_kernel_is_refused_when_the_block_is_built(
        self, label, kernel
    ):
        """The hazard the docstring names, now caught where a kernel enters.

        `unchecked_operator` probes every observed node before it linearises
        anything, and that probe is `check_observed`, which runs
        `check_positive_definite` for a correlated node. Both fixtures are
        refused: the all-positive one matters most, since reading the kernel's
        own entries rather than its SPECTRUM would let it through.
        """
        from bayesmith.exact.block import unchecked_operator

        with jax.enable_x64(True):
            graph = self._graph(jnp.asarray(kernel))
            with pytest.raises(ValueError, match="positive definite"):
                unchecked_operator(graph, ("w",), {})

    def test_the_positive_definite_check_runs_before_the_density_comparison(self):
        """Order, and it is about the DIAGNOSIS rather than the verdict.

        `check_precision` also refuses an indefinite kernel -- but only
        through NaN propagation, and it names the wrong cause. Measured on
        three indefinite kernels: numpyro's `CirculantNormal.log_prob` returns
        `nan`, so `check_precision` reports `linearity=nan, normalizer=nan`
        and its message explains `linearity` as "the log-density is not
        quadratic, so it has no covariance to extract". The log-density IS
        quadratic. A user sent to look at their `det` nodes and `linear_in`
        declarations would be looking in the wrong place.

        So this asserts which message comes out, not merely that one does.
        """
        from bayesmith.exact.gaussian import check_observed
        from bayesmith.exact.precision import check_precision

        with jax.enable_x64(True):
            kernel = jnp.asarray([1.0, 1.5, 1.0, 1.5])
            graph = self._graph(kernel)
            env = evaluate(graph, {"w": jnp.asarray(2.0)})
            with pytest.raises(ValueError, match="autocovariance kernel") as caught:
                check_observed(graph, graph.node("d"), env)
            assert "positive definite" in str(caught.value)
            assert "not quadratic" not in str(caught.value)

            # ...and the guard it precedes really would have misdiagnosed it,
            # so the ordering is load-bearing rather than tidy.
            import numpyro.distributions as ndist

            with pytest.raises(PrecisionMismatch, match="linearity=nan") as second:
                check_precision(
                    ndist.CirculantNormal(jnp.zeros(4), kernel),
                    CirculantPrecision(first_column=kernel),
                    jnp.zeros(4),
                )
            assert "not quadratic" in str(second.value)

    def test_a_well_formed_correlated_node_passes_the_probe(self):
        """The anti-vacuity clause: this refuses kernels, not correlation.

        A guard that refused every correlated node would pass both tests
        above while testing nothing. What stops the block being built here is
        the block builder's own diagonal-only data and loc walks, which is a
        LATER stage and a different message -- so the probe itself accepted.
        """
        from bayesmith.exact.block import unchecked_operator
        from bayesmith.exact.gaussian import check_observed

        size = 8
        lag = np.minimum(np.arange(size), size - np.arange(size))
        with jax.enable_x64(True):
            graph = self._graph(jnp.asarray(1.0 * 0.4**lag + 0.5), n=size)
            env = evaluate(graph, {"w": jnp.asarray(2.0)})
            errors = check_observed(graph, graph.node("d"), env)
            assert errors["operator"] < 1e-10
            with pytest.raises(NotGaussian, match="CirculantNormal"):
                unchecked_operator(graph, ("w",), {})


class TestTheLogSpectrumIsTheFourthOperation:
    """``log lambda_k``, which the Fisher matrix needs and the other three cannot give."""

    @pytest.mark.parametrize("size", [1, 4, 8])
    def test_log_normalizer_is_the_log_spectrums_own_sum(self, size):
        """The two are ONE rule kept in two places, so this renders them side by side.

        `log_normalizer` was NOT rewritten to call `log_spectrum`, and that is
        a deliberate choice with a measured reason: `log(2 pi sigma**2)`
        summed is not bitwise `n log 2 pi + sum(2 log sigma)` -- measured, a
        `sigma = 1e-8` fixture differs by one ULP (1.6e-16 relative). Deriving
        would have been the tidier code and would have moved a shipped number.

        Two copies of a rule is how one goes stale, and the remedy this
        codebase uses is to make the drift loud. That is this test: it is the
        only thing rendering the two definitions against each other.
        """
        with jax.enable_x64(True):
            for precision in (
                DiagonalPrecision(sigma=jnp.linspace(0.3, 2.1, size)),
                CirculantPrecision(first_column=_kernel(size) if size > 1 else jnp.asarray([2.0])),
            ):
                spectrum = precision.log_spectrum()
                assert spectrum.shape == (size,)
                assert float(precision.log_normalizer()) == pytest.approx(
                    size * math.log(2.0 * math.pi) + float(jnp.sum(spectrum)),
                    rel=1e-12,
                )

    def test_half_the_diagonal_log_spectrum_is_log_sigma_BITWISE(self):
        """What keeps the diagonal Fisher answer the number it always was.

        `_log_spectrum_curvature` jacobians `0.5 * log_spectrum()`. For a
        diagonal that must be exactly the `log(sigma)` the old
        `_log_sigma_curvature` used -- not approximately, or the generalisation
        would move every existing forecast in its last digits.

        `log_spectrum` therefore returns `2 log sigma` rather than
        `log(sigma**2)`: multiplying a float by two and halving it is exact in
        binary, and `log(sigma**2)` is not `2 log sigma`.
        """
        with jax.enable_x64(True):
            sigma = jnp.asarray([1e-8, 0.3, 1.0, 7.5, 1e6])
            precision = DiagonalPrecision(sigma=sigma)
            assert bool(jnp.all(0.5 * precision.log_spectrum() == jnp.log(sigma)))
            # ...and the tidier spelling would NOT have been bitwise, which is
            # the whole reason for the one above.
            assert not bool(jnp.all(jnp.log(sigma**2) == 2.0 * jnp.log(sigma)))

    def test_the_circulant_log_spectrum_is_the_kernels_own_fft(self):
        """Against `scipy.linalg`'s eigenvalues, not against our FFT again."""
        with jax.enable_x64(True):
            kernel = _kernel(8)
            precision = CirculantPrecision(first_column=kernel)
            found = np.sort(np.exp(np.asarray(precision.log_spectrum())))
        dense_matrix = sla.circulant(np.asarray(kernel))
        expected = np.sort(np.real(np.linalg.eigvals(dense_matrix)))
        assert np.allclose(found, expected, rtol=1e-12)

    def test_the_protocol_admits_only_implementations_that_have_it(self):
        """`Precision` is `runtime_checkable`, so a missing method must show.

        A four-operation protocol that still accepted a three-operation object
        would let `fisher_information` fail at the call site instead of at the
        boundary.
        """

        class ThreeOperations(eqx.Module):
            def apply(self, residual):
                return residual

            def log_normalizer(self):
                return jnp.asarray(0.0)

            def whiten(self, omega):
                return omega

        assert not isinstance(ThreeOperations(), Precision)
        assert isinstance(DiagonalPrecision(sigma=jnp.ones(3)), Precision)


class TestWhatTheEvidenceLayerWillFindHere:
    """B11's whitening row against this interface, before it is written.

    The migration spec says the evidence layer was "waiting on this
    interface's shape" and that the whitening row becomes ``L^-1 r``, which
    :meth:`Precision.whiten` supplies. That is true of the ROW. It is not true
    of the layer, and the difference is worth pinning before someone builds on
    the shorter claim.

    B11's must-preserve list names "the masked normalisation of ``sigma=inf``
    samples" among its numerical kernels. rheplicant does it with
    ``weight = where(seen, 1/sigma, 0)`` in ``inference/compress.py:111`` and
    a normaliser summed over finite-sigma samples only
    (``compress.py:422``). Measured below: this interface gets the first half
    exactly right and does not express the second at all.
    """

    @staticmethod
    def _with_unobserved():
        return DiagonalPrecision(sigma=jnp.asarray([0.5, 1.0, jnp.inf, 2.0]))

    def test_whitening_and_weighting_already_mask_an_unobserved_sample(self):
        """The half the spec is right about, and it is exactly right.

        ``sigma = inf`` gives ``1/sigma**2 = 0`` and ``1/sigma = 0`` with no
        special case and no NaN, so the quadratic form stays finite and the
        unseen sample contributes nothing -- which is what
        ``compress.py``'s explicit `where(seen, ...)` mask is for.
        """
        with jax.enable_x64(True):
            precision = self._with_unobserved()
            residual = jnp.asarray([1.0, -2.0, 7.0, 0.5])
            assert float(precision.apply(residual)[2]) == 0.0
            assert float(precision.whiten(residual)[2]) == 0.0
            assert float(quadratic(precision, residual)) == pytest.approx(8.0625)
            # ...and applying the whitening twice is applying the weight, at
            # the masked entry as much as anywhere else.
            twice = precision.whiten(precision.whiten(residual))
            assert jnp.allclose(twice, precision.apply(residual))

    def test_the_normaliser_does_NOT_mask_it_and_that_is_B11s_first_decision(self):
        """The half the spec is silent about. Measured, not assumed.

        ``log_normalizer`` sums ``log(2 pi sigma**2)`` over every sample, so an
        unobserved one takes it to ``+inf``; ``log_spectrum`` carries the same
        ``inf`` in its own entry. rheplicant's masked normaliser on this
        fixture is ``5.513631199``.

        **This is not a defect to fix here.** A ``sigma = inf`` sample has no
        density, so ``inf`` is the honest answer to the question ``Precision``
        is asked. Treating it as ``0`` is a statement that the sample is
        UNOBSERVED, which is a modelling concept this interface does not have
        and the evidence layer does. Masking it here would put a
        silently-wrong normaliser one import away from every consumer --
        exactly the defect B1 shape ``precision.py`` exists to prevent.

        So this pins the gap rather than closing it: whoever writes B11 owns
        the decision, and this test is what stops them assuming
        ``log_normalizer`` already made it.
        """
        with jax.enable_x64(True):
            precision = self._with_unobserved()
            assert not bool(jnp.isfinite(precision.log_normalizer()))
            spectrum = precision.log_spectrum()
            assert not bool(jnp.isfinite(spectrum[2]))
            assert bool(jnp.all(jnp.isfinite(spectrum[jnp.asarray([0, 1, 3])])))

            sigma = precision.sigma
            seen = jnp.isfinite(sigma)
            masked = float(
                jnp.sum(
                    jnp.where(
                        seen, jnp.log(2.0 * jnp.pi * jnp.where(seen, sigma, 1.0) ** 2), 0.0
                    )
                )
            )
        assert masked == pytest.approx(5.513631199, rel=1e-9)
