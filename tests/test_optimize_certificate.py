"""``minimize`` and ``fit`` reporting a PROVEN distance to the minimum.

:mod:`bayesmith.optimize.certify` is tested on plain callables in
``tests/test_certify.py``. This file is the seam: what the two graph-facing
entry points do with it, and -- the part worth a file of its own -- which
answers they refuse to give.

The one property every test here defends is that ``converged`` is a PROOF and
never an impression. Three separate things can take it away, and each has a
test below that makes it fail for that reason alone: an unproven curvature
floor, a solve that did not converge, and a bound outside the caller's limit.
A fourth case is the one a reader is most likely to get wrong -- not asking
for a certificate at all is also not converged, and `refusal` is what tells
the two apart.

**What this file kills, and what it does not.** Six mutants aimed at turning a
refusal into an APPROVAL -- the only direction that matters for a safety claim
-- applied to a scratch copy of ``src/`` and run against this file (21 passed
clean) beside ``tests/test_certify.py`` (50 passed clean):

===================================  ==========  ==========
mutant                               this file   unit tests
===================================  ==========  ==========
a probed floor counts as proven       KILLED      KILLED
``certifies`` ignores the floor       KILLED      KILLED
the verdict reads the ESTIMATE        SURVIVED    KILLED
the verdict ignores the status        SURVIVED    KILLED
``Fit.converged`` returns True        KILLED      SURVIVED
the seam drops the caller's floor     KILLED      SURVIVED
===================================  ==========  ==========

Every mutant dies to the union and neither half is sufficient alone, which is
the division of labour the two files are for: the unit tests own the
CONDITIONS of the bound, this file owns the WIRING. The two survivors here
need a point whose solve fails while its floor is proven -- an inexact solve
that stops short, or a conjugate gradient at its iteration cap -- which this
seam cannot force cheaply and which ``test_certify.py`` constructs directly.

The objectives are quadratics whose Newton decrement is known in closed form.
For ``f(p) = 0.5 |(p - t) / s|**2`` the gradient is ``(p - t) / s**2`` and the
Hessian ``1 / s**2``, so ``sqrt(g^T H^-1 g) = |p - t| / s`` exactly: the
distance in units of the curvature's own standard deviation, which for a
negative log posterior is the posterior sigma. That closed form is the oracle,
computed in numpy, differentiating nothing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import StructureError
from bayesmith.optimize import Fit, certify, fit, minimize
from tests.exact.models import straight_line

#: Distance, in posterior sigma, the tests ask to be proven within.
LIMIT = 0.1


@pytest.fixture(autouse=True)
def _double_precision():
    """float64 throughout: the certificate's rounding allowance refuses any
    Hessian whose condition number approaches ``1 / eps``, and a float32
    epsilon is where that starts to bite on ordinary models."""
    with jax.enable_x64(True):
        yield


def _well(target=(1.5, -0.75), sigma=(0.4, 0.9)):
    """``0.5 |(p - t)/s|^2`` -- decrement ``|p - t| / s``, known exactly."""
    t, s = jnp.asarray(target), jnp.asarray(sigma)

    def objective(values):
        return 0.5 * jnp.sum(((values["p"] - t) / s) ** 2)

    return objective


def _oracle_decrement(point, target=(1.5, -0.75), sigma=(0.4, 0.9)):
    return float(np.linalg.norm((np.asarray(point) - np.asarray(target)) / np.asarray(sigma)))


def _wide(n=1200, low=0.4, high=1.2):
    """The same shape above :data:`~bayesmith.optimize.certify.DENSE_MAX`, so
    the dense path is unavailable and the floor has to come from somewhere.

    Returns the objective and the TRUE smallest eigenvalue of its Hessian,
    ``diag(1 / scale**2)``, which is ``1 / high**2``.

    The scales are spread rather than equal, and that is load-bearing. On an
    isotropic Hessian the Lanczos recursion terminates at the first step --
    ``H v = lambda v`` leaves nothing to orthogonalise against -- so the
    tridiagonal matrix is ``diag(lambda, 0, 0, ...)``, its smallest Ritz value
    is 0, and the probe reports no usable floor at all (``floor_source
    "none"``) rather than the probed one this fixture is for. Measured while
    writing this file, on ``scale = 0.5`` throughout.
    """
    scale = jnp.linspace(low, high, n)

    def objective(values):
        return 0.5 * jnp.sum((values["p"] / scale) ** 2)

    return objective, float(1.0 / high**2)


class TestWhatIsReportedWhenNothingIsAsked:
    def test_a_fit_without_certify_carries_no_certificate(self):
        found = minimize(_well(), {"p": jnp.asarray([0.0, 0.0])}, steps=50)
        assert found.certificate is None
        assert found.limit is None

    def test_not_asking_is_not_converged_and_the_refusal_says_so(self):
        """The trap this property exists against: reading ``converged is
        False`` as "it was measured and it failed"."""
        found = minimize(_well(), {"p": jnp.asarray([0.0, 0.0])}, steps=50)
        assert found.converged is False
        assert "no certificate was requested" in found.refusal

    def test_the_old_three_field_construction_still_works(self):
        """Every existing caller builds a ``Fit`` by these three keywords."""
        made = Fit(
            values={"p": jnp.asarray(0.0)},
            objective=jnp.asarray(1.0),
            history=jnp.zeros(3),
        )
        assert made.certificate is None and made.converged is False


class TestACertificateThatHolds:
    def test_a_well_conditioned_descent_is_proven_within_the_limit(self):
        found = minimize(
            _well(), {"p": jnp.asarray([0.0, 0.0])}, steps=2000, certify=LIMIT
        )
        assert found.converged is True
        assert found.refusal is None
        assert found.certificate.proven
        assert found.certificate.floor_source == certify.FLOOR_DENSE
        assert found.certificate.distance <= LIMIT

    def test_the_reported_distance_is_the_closed_form_decrement(self):
        """The oracle: ``|p - t| / s`` in numpy, against the certificate's own
        estimate at the same point, with the descent stopped short on purpose
        so the number is not zero."""
        start = {"p": jnp.asarray([0.0, 0.0])}
        found = minimize(
            _well(), start, steps=1, learning_rate=1e-9, certify=LIMIT, polish=False
        )
        expected = _oracle_decrement(found.values["p"])
        assert found.certificate.estimate == pytest.approx(expected, rel=1e-8)
        assert expected > LIMIT, "the fixture must not accidentally be converged"


class TestTheThreeWaysToBeRefused:
    def test_a_point_outside_the_limit_refuses_and_names_the_distance(self):
        found = minimize(
            _well(),
            {"p": jnp.asarray([0.0, 0.0])},
            steps=1,
            learning_rate=1e-9,
            certify=LIMIT,
            polish=False,
        )
        assert found.converged is False
        assert found.certificate.proven, "the floor is fine; the DISTANCE is not"
        assert "outside the limit" in found.refusal

    def test_a_probed_floor_never_certifies_however_small_the_distance(self):
        """The property the upstream module was five times reviewed for.

        Above ``DENSE_MAX`` with no supplied floor the curvature floor is a
        Lanczos probe, which bounds SOME eigenvalue rather than the smallest
        -- measured upstream to sit above the true one in 5 of 240 float32
        spectra, the worst by a factor 7.4. So it may refuse and must never
        approve. Here the point IS the minimum, the distance IS ~0, and the
        answer is still not converged.
        """
        objective, _ = _wide()
        found = minimize(
            objective,
            {"p": jnp.full((1200,), 1e-3)},
            steps=1,
            learning_rate=1e-9,
            certify=LIMIT,
            polish=False,
        )
        assert found.certificate.floor_source == certify.FLOOR_PROBE
        assert found.certificate.distance <= LIMIT, "the distance is not the problem"
        assert found.converged is False
        assert not found.certificate.proven
        assert "estimate" in found.refusal and "probe" in found.refusal

    def test_a_point_that_is_not_near_a_minimum_refuses_on_the_SOLVE(self):
        """The third condition, on its own: proven floor, distance inside the
        limit, and still refused because the point is not near a minimum.

        ``0.5 p0**2 - 0.5 p1**2`` has Hessian ``diag(1, -1)``: a saddle. The
        decrement of a quadratic model with a descending direction is not a
        distance to anything, so the solve reports ``NONCONVEX`` and the
        verdict must read it. Measured while writing this file: a mutant that
        dropped the status check from ``certifies`` passed every other test in
        this file.
        """

        def saddle(values):
            return 0.5 * values["p"][0] ** 2 - 0.5 * values["p"][1] ** 2

        found = minimize(
            saddle,
            {"p": jnp.asarray([1e-4, 1e-4])},
            steps=1,
            learning_rate=1e-9,
            certify=LIMIT,
            polish=False,
        )
        assert found.certificate.status == certify.NONCONVEX
        assert found.certificate.floor_source == certify.FLOOR_NONE
        assert found.converged is False
        # Both the floor and the solve fail here, and `certifies` reads the
        # floor first, so the sentence has to name the cause and not only the
        # condition it tripped over.
        assert "no usable curvature floor" in found.refusal
        assert "not near a minimum" in found.refusal

    def test_the_same_point_with_a_supplied_floor_does_certify(self):
        """The contrast that makes the test above mean something: identical
        objective and identical point, one provable number added."""
        objective, floor = _wide()
        found = minimize(
            objective,
            {"p": jnp.full((1200,), 1e-3)},
            steps=1,
            learning_rate=1e-9,
            certify=LIMIT,
            floor=floor,
            polish=False,
        )
        assert found.certificate.floor_source == certify.FLOOR_SUPPLIED
        assert found.converged is True


class TestThePolish:
    def test_polish_is_what_makes_a_first_order_descent_certifiable(self):
        """Adam's step-size floor, as the difference between two verdicts.

        A first-order step of ``rate * sign(gradient)`` does not shrink with
        the gradient, so the iterate settles a fraction of a step size from the
        optimum -- far more than a tenth of a posterior sigma. The Newton
        polish is what removes it, which is why `certify=` turns it on.
        """
        start = {"p": jnp.asarray([0.0, 0.0])}
        settings = {"steps": 60, "learning_rate": 0.05, "certify": LIMIT}
        raw = minimize(_well(), start, polish=False, **settings)
        polished = minimize(_well(), start, polish=True, **settings)

        assert raw.converged is False
        assert polished.converged is True
        assert polished.polished > 0
        assert raw.polished == 0
        assert float(polished.objective) <= float(raw.objective)


class TestFit:
    def test_a_graph_map_reports_a_certificate_over_its_own_latents(self):
        graph = straight_line(n=8, weight=2.5, sigma=0.5, prior_std=2.0, prior_mean=0.0)
        found = fit(graph, steps=2000, learning_rate=0.02, certify=LIMIT)
        assert found.converged is True
        assert found.certificate.floor_source == certify.FLOOR_DENSE
        assert set(found.values) == set(graph.latents)

    def test_a_held_block_certifies_the_block_and_not_the_graph(self):
        """``names=`` moves one latent; the decrement is over that one."""
        graph = straight_line(n=8, weight=2.5, sigma=0.5, prior_std=2.0, prior_mean=0.0)
        found = fit(
            graph, names=("w",), steps=2000, learning_rate=0.02, certify=LIMIT
        )
        assert found.converged is True
        assert found.certificate.dense


class TestTheVerdictAndTheReasonCannotDisagree:
    """One object, two readings, and they were written as two chains once.

    Measured while writing this file: ``refusal``'s last branch returned "the
    point is within 0 ... which is outside the limit 0.1" for a fit whose
    ``converged`` was ``True`` and whose ``distance`` was ``0.0``, because the
    chain fell through without re-checking the limit. ``converged`` alone was
    green. The repair was to make ``refusal`` delegate rather than re-derive,
    and this is the assertion that says so for every case the file reaches.
    """

    @staticmethod
    def _cases():
        objective, floor = _wide()
        small = {"p": jnp.asarray([0.0, 0.0])}
        wide = {"p": jnp.full((1200,), 1e-3)}
        starved = {"steps": 1, "learning_rate": 1e-9, "polish": False}
        return [
            ("not asked", minimize(_well(), small, steps=50)),
            ("proven", minimize(_well(), small, steps=2000, certify=LIMIT)),
            ("too far", minimize(_well(), small, certify=LIMIT, **starved)),
            ("probed", minimize(objective, wide, certify=LIMIT, **starved)),
            (
                "supplied",
                minimize(objective, wide, certify=LIMIT, floor=floor, **starved),
            ),
        ]

    def test_refusal_is_none_exactly_when_converged(self):
        for label, found in self._cases():
            assert (found.refusal is None) == found.converged, label

    def test_every_case_this_file_builds_is_actually_distinct(self):
        """Anti-vacuity: the table above must not be five spellings of one
        state, which would make the check above pass on almost anything."""
        seen = {
            label: (
                found.converged,
                None if found.certificate is None else found.certificate.floor_source,
            )
            for label, found in self._cases()
        }
        assert len(set(seen.values())) == len(seen), seen
        assert sum(state for state, _ in seen.values()) == 2, seen


class TestWhatIsRefusedBeforeAnythingIsSpent:
    def test_a_floor_without_a_certificate_is_refused(self):
        with pytest.raises(StructureError, match="no certificate was asked for"):
            minimize(_well(), {"p": jnp.asarray([0.0, 0.0])}, steps=10, floor=1.0)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_a_limit_that_can_never_be_met_is_refused(self, bad):
        with pytest.raises(StructureError, match="certify must be > 0"):
            minimize(_well(), {"p": jnp.asarray([0.0, 0.0])}, steps=10, certify=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_a_floor_at_or_below_zero_is_refused(self, bad):
        with pytest.raises(StructureError, match="floor must be > 0"):
            minimize(
                _well(),
                {"p": jnp.asarray([0.0, 0.0])},
                steps=10,
                certify=LIMIT,
                floor=bad,
            )
