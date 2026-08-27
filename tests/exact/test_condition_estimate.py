"""G14 -- the MEASURED kappa, and the reason it is a diagnostic and not a bound.

``condition_bound`` measures only the top of the spectrum and replaces
``lambda_min`` with the prior's own curvature, which makes it an UPPER bound --
the direction a guard needs. ``condition_estimate`` measures both ends. It is
biased LOW, which is the direction that certifies an answer it should have
refused, so it is never a guard; what it can do is SEE a degeneracy, which a
bound that floors ``lambda_min`` structurally cannot.

D15(a) is the ruling that brings it here, and this file's job is to make the
bias measurable rather than merely asserted -- so an implementation that
quietly "fixed" it would go red.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bayesmith.errors import GraphError
from bayesmith.exact.conditioning import extreme_eigenvalues, largest_eigenvalue


class TestExtremeEigenvaluesOnAKnownSpectrum:
    """A diagonal operator whose spectrum is written down, so both ends have
    an answer that owes nothing to this module."""

    @staticmethod
    def _diagonal(spectrum):
        values = jnp.asarray(spectrum)

        def operator(parts):
            return {"x": parts["x"] * values}

        return operator, {"x": jnp.zeros(values.shape)}

    def test_both_ends_are_right_on_a_flat_spectrum(self):
        """The easy case, and the one that would catch the shift being
        applied backwards: every eigenvalue equal, so lambda_max and
        lambda_min are the same number and the shifted operator is zero."""
        operator, template = self._diagonal([4.0] * 6)
        largest, smallest = extreme_eigenvalues(
            operator, template, jax.random.key(0), 40
        )
        assert float(largest) == pytest.approx(4.0, rel=1e-5)
        assert float(smallest) == pytest.approx(4.0, rel=1e-4)

    def test_both_ends_are_right_on_a_well_separated_spectrum(self):
        """Two clusters far apart: the shifted power iteration has a real gap
        to work with and finds the bottom."""
        operator, template = self._diagonal([100.0, 100.0, 1.0, 1.0])
        largest, smallest = extreme_eigenvalues(
            operator, template, jax.random.key(1), 200
        )
        assert float(largest) == pytest.approx(100.0, rel=1e-4)
        assert float(smallest) == pytest.approx(1.0, rel=1e-2)

    def test_it_is_biased_LOW_on_a_graded_spectrum_and_that_is_the_point(self):
        """The measurement the whole design rests on, kept as an assertion.

        On a geometric spectrum the shifted operator's leading eigenvalues
        crowd against ``lambda_max`` with vanishing gaps, so the iteration
        cannot separate them however long it runs. ``lambda_min`` comes back
        too LARGE and kappa too SMALL -- the direction that certifies an
        answer it should have refused.

        Measured here, on ``geomspace(1, 1e7, 50)`` whose true
        ``lambda_min`` is 1.0:

        ========  =================  ==================
        iters     lambda_min         kappa (true 1e7)
        ========  =================  ==================
        50        10210.8            979
        200       2351.3             4.25e3
        800       805.9              1.24e4
        2000      501.2              2.00e4
        ========  =================  ==================

        Asserted as inequalities rather than pinned values: the exact factor
        depends on the starting vector, and pinning it would make this test
        about the PRNG. What must not drift is the DIRECTION and that it is
        enormous.
        """
        with jax.enable_x64(True):
            spectrum = jnp.geomspace(1.0, 1e7, 50)
            operator, template = self._diagonal(spectrum)
            largest, smallest = extreme_eigenvalues(
                operator, template, jax.random.key(2), 200
            )
        # The true bottom is 1.0; this is three orders above it.
        assert float(smallest) > 100.0, float(smallest)
        # ... so the kappa it yields is three orders BELOW the true 1e7.
        assert float(largest / smallest) < 1e5, float(largest / smallest)

    def test_more_iterations_do_not_close_it(self):
        """The other half of "fails in principle": this is not a budget
        problem, so a caller cannot buy their way out of it.

        Measured: 40x the work (50 -> 2000 steps) takes ``lambda_min`` from
        10210.8 to 501.2, still 500x above a true 1.0.
        """
        with jax.enable_x64(True):
            spectrum = jnp.geomspace(1.0, 1e7, 50)
            operator, template = self._diagonal(spectrum)
            _, few = extreme_eigenvalues(operator, template, jax.random.key(2), 50)
            _, many = extreme_eigenvalues(operator, template, jax.random.key(2), 2000)
        assert float(many) < float(few)          # it does improve ...
        assert float(many) > 100.0, float(many)  # ... and nowhere near enough

    def test_the_top_is_the_same_number_largest_eigenvalue_gives(self):
        """One implementation of the top, not two: `extreme_eigenvalues`
        calls `largest_eigenvalue` rather than repeating it."""
        operator, template = self._diagonal([9.0, 3.0, 1.0])
        key = jax.random.key(4)
        alone = largest_eigenvalue(operator, template, key, 60)
        both, _ = extreme_eigenvalues(operator, template, key, 60)
        assert float(both) == float(alone)

    def test_a_non_positive_iteration_count_is_refused(self):
        operator, template = self._diagonal([1.0, 2.0])
        with pytest.raises(GraphError, match="iteration"):
            extreme_eigenvalues(operator, template, jax.random.key(5), 0)


class TestConditionEstimateAgainstConditionBound:
    """The two numbers side by side on real blocks, which is the only way to
    show they are different quantities rather than two spellings."""

    @staticmethod
    def _block_and_precision(graph, names):
        from bayesmith.dispatch.classify import prior_environment
        from bayesmith.exact.gaussian import precision_at
        from bayesmith.exact.linearity import linear_operator

        centres = prior_environment(graph)
        # A block is affine GIVEN the latents outside it, so they have to be
        # somewhere -- `linear_operator` refuses an empty `at` when there are
        # any, which is how a one-latent block of a two-latent model differs
        # from the joint one.
        outside = {
            name: centres[name] for name in graph.latents if name not in set(names)
        }
        block = linear_operator(graph, names, at=outside)
        precision = precision_at(
            graph, {n: centres[n] for n in graph.latents}
        )
        return block, precision

    def test_the_estimate_is_below_the_bound(self):
        """The bound is an upper bound on the same quantity, so this holds by
        construction -- and would fail if either were computing the other's
        formula by accident."""
        from bayesmith.exact.solve import condition_bound, condition_estimate
        from tests.exact.models import two_linear_latents

        with jax.enable_x64(True):
            graph = two_linear_latents()
            block, precision = self._block_and_precision(graph, ("a", "b"))
            bound = float(condition_bound(block, precision=precision))
            measured = float(condition_estimate(block, precision=precision))
        assert measured <= bound * (1.0 + 1e-9), (measured, bound)
        assert measured > 0.0

    def test_it_sees_a_degeneracy_the_bound_structurally_cannot(self):
        """The reason this exists at all (D15(a)).

        A near-degenerate JOINT block shows up entirely in ``lambda_min``,
        and the bound replaces ``lambda_min`` with the prior's floor -- so
        the bound cannot report it however tight the spectrum gets.
        `collinear_pair` is the degenerate case: the data fixes ``a + b`` and
        the prior alone fixes ``a - b``.

        Measured as a RATIO of ratios so the assertion does not depend on
        either absolute number: the joint block's measured kappa must exceed
        a single member's by far more than its bound does.
        """
        from bayesmith.exact.solve import condition_bound, condition_estimate
        from tests.exact.models import collinear_pair

        with jax.enable_x64(True):
            graph = collinear_pair()
            joint, precision = self._block_and_precision(graph, ("a", "b"))
            alone, precision_a = self._block_and_precision(graph, ("a",))
            joint_measured = float(condition_estimate(joint, precision=precision))
            alone_measured = float(condition_estimate(alone, precision=precision_a))
            joint_bound = float(condition_bound(joint, precision=precision))
            alone_bound = float(condition_bound(alone, precision=precision_a))
        measured_ratio = joint_measured / alone_measured
        bound_ratio = joint_bound / alone_bound
        assert measured_ratio > 10.0 * bound_ratio, (measured_ratio, bound_ratio)

    def test_the_estimate_is_reproducible(self):
        """A fixed default key, so a diagnostic printed twice is one number.
        Without it the "is this partition badly conditioned?" question gets a
        different answer every time it is asked."""
        from bayesmith.exact.solve import condition_estimate
        from tests.exact.models import two_linear_latents

        graph = two_linear_latents()
        block, precision = self._block_and_precision(graph, ("a", "b"))
        one = float(condition_estimate(block, precision=precision))
        two = float(condition_estimate(block, precision=precision))
        assert one == two


class TestItSaysItIsNotAGuard:
    """D15(a) asks for the label in the CODE rather than in someone's memory,
    so the label is asserted."""

    def test_the_docstring_says_it_is_not_a_bound(self):
        from bayesmith.exact.solve import condition_estimate

        text = condition_estimate.__doc__ or ""
        assert "not a bound" in text.lower() or "never a guard" in text.lower()
        assert "condition_bound" in text

    def test_the_docstring_names_the_direction_of_the_bias(self):
        """Not just "approximate": a reader has to be able to tell WHICH way
        it is wrong, because only one of the two directions is dangerous."""
        from bayesmith.exact.solve import condition_estimate

        text = condition_estimate.__doc__ or ""
        assert "too small" in text.lower() or "biased low" in text.lower()

    def test_the_conditioning_module_no_longer_claims_it_was_not_ported(self):
        """The claim this batch makes false, pinned so it cannot come back.

        The module docstring used to say `extreme_eigenvalues` is
        "deliberately not ported". It is ported now, as a diagnostic, and the
        argument for why it must never be a guard is unchanged -- but the
        sentence that says it was not ported has to go, or the file contains
        a statement its own contents refute.
        """
        import bayesmith.exact.conditioning as module

        text = module.__doc__ or ""
        # Asserted POSITIVELY. An absence check would have been satisfied by
        # deleting the paragraph, and it was also tripped by the docstring's
        # own account of what it used to say -- an absence check cannot tell
        # a claim from a quotation of a retracted one.
        assert "condition_estimate" in text
        assert "G14" in text or "D15" in text
        # ... and the reasoning the old paragraph carried survives, because
        # it is still true of the GUARD.
        assert "one-sided" in text or "too large" in text
        assert "condition_bound" in text


class TestNoGuardReadsTheMeasuredRoute:
    """The rule the old absence assertions were standing in for, pinned
    directly.

    Two cross-check tests used to assert that this package did not carry
    ``extreme_eigenvalues`` or ``condition_estimate`` at all. That was a proxy
    for the thing that actually matters -- **nothing that GUARDS may read
    them** -- and it stopped being available the moment D15(a) ruled the
    diagnostic in. A proxy is worth replacing with the thing it stood for
    rather than deleting.

    An AST scan, not a text scan, so a name in a docstring or a comment is not
    a finding; and two-directional, so the allowlist cannot rot into a list of
    names nobody checks.
    """

    @staticmethod
    def _call_sites(name):
        """``{enclosing function: count}`` for every call to ``name`` under
        ``src/bayesmith``."""
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2] / "src" / "bayesmith"
        found: dict[str, int] = {}
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                for inner in ast.walk(node):
                    if not isinstance(inner, ast.Call):
                        continue
                    fn = inner.func
                    called = (
                        fn.id if isinstance(fn, ast.Name)
                        else fn.attr if isinstance(fn, ast.Attribute)
                        else None
                    )
                    if called == name:
                        key = f"{path.name}::{node.name}"
                        found[key] = found.get(key, 0) + 1
        return found

    def test_only_condition_estimate_calls_extreme_eigenvalues(self):
        sites = self._call_sites("extreme_eigenvalues")
        assert set(sites) == {"solve.py::condition_estimate"}, sites

    def test_nothing_inside_this_package_calls_condition_estimate(self):
        """Nothing at all: it is a diagnostic for a CALLER to read, and the
        moment something here consumes it, it has become an input to a
        decision."""
        assert self._call_sites("condition_estimate") == {}

    def test_the_scan_can_still_find_something(self):
        """The self-check this repository's notes keep asking for: a scan that
        matches nothing goes green on an empty codebase, a moved directory, or
        a renamed AST node. `largest_eigenvalue` has real call sites, so this
        fails if the walker has stopped walking."""
        sites = self._call_sites("largest_eigenvalue")
        assert len(sites) >= 2, sites
        assert "conditioning.py::extreme_eigenvalues" in sites
