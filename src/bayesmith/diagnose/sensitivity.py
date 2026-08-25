"""Prior sensitivity: how far each latent's own prior moved the answer.

Every latent in a graph carries a density, and every exit reads it. None of
them says what the prior *did*. That is a different question from "is the
posterior right", and it is the one a referee asks: the mode you report sits
somewhere between where the data put it and where the prior wanted it, and
the only honest way to quote it is with the distance stated.

**A chain cannot answer this about itself.** On the tour's nonlinear pair
(the cross-check fixture shared with rheplicant) the declared
``fg_beta ~ Normal(2.3, 0.3)`` moves the mode by **0.0069 sigma**. The Monte
Carlo standard error of a posterior mean from ``n_eff`` draws is
``1/sqrt(n_eff)`` sigma, so seeing 0.0069 sigma at all needs ``n_eff`` of
order 2 x 10^4, and *measuring* it needs a second chain with the prior
removed to difference against -- two chains whose noise adds in quadrature.
Running longer is not a small ask, it is a 10^4-fold one, and it answers a
question two Newton solves answer exactly.

Two routes, and they are both deterministic.

**The closed form.** Write the negative log-posterior as ``l(x) + 0.5 *
sum(((x - m) / s)^2)``, where ``l`` is every probabilistic term EXCEPT the
selected latents' own priors (the observed nodes, and any unselected latent
whose density the selection parameterises), and ``(m, s)`` are the selected
priors' location and scale. The direction convention, stated once and
unambiguously because its ancestor's docstring was read both ways:

    ``theta_hat - theta_L  =  H^{-1} P (m - theta_hat)``,   ``P = diag(s^-2)``

where ``theta_hat`` is the MAP, ``theta_L`` the mode of ``l`` alone, and
``H`` is ``l``'s curvature at ``theta_hat``. The left side is **MAP minus
likelihood-only mode**: a positive entry means the prior pulled that latent
UP relative to where the data alone would put it, and that is the sign
:attr:`PriorSensitivityReport.shift_sigma` reports. ``H`` is the
LIKELIHOOD's curvature, not the posterior's ``H + P`` already in hand for
``sigma_post``: putting that here is wrong by ``diag((H + P)^-1 P)``, the
prior's share of the posterior precision -- invisible where the prior is
weak and unbounded as it tightens, which is the regime the report exists to
describe.

Per latent the diagonal law is ``sigma_post * |m - theta_hat| / s^2`` --
what :attr:`PriorSensitivityReport.criterion_std` inverts -- plus a cross
term from every OTHER latent's prior. The cross term is not decoration: on
the tour's pair it is ten times the amplitude's own pull and of the opposite
sign, so a per-latent scalar formula would report the direction of the bias
wrongly, which is worse than reporting its size wrongly.

**The refit.** Newton to the mode with the selected priors in the
objective, Newton again with them removed, difference the two. No
expansion, no linearisation -- the answer the closed form is approximating.
Which of the two is the approximate one depends on the model, which is the
reason for shipping both rather than picking: on a latent the prediction is
affine in, the log-posterior is exactly quadratic, the closed form is exact
and the refit loses digits to cancellation; on a nonlinear pair the roles
reverse. :attr:`PriorSensitivityReport.verified` says whether they agreed.

Three things this does NOT do, stated so they are not assumed.

It is **local**. Both routes expand about one mode; a multimodal posterior
has more than one, and the prior's job there may be to select between them,
which is not a displacement and is not measured here.

It reads each selected latent's **own declared density** only. A selected
latent whose ``dist_fn`` is not a Gaussian with parameters fixed under the
selection is refused by name rather than approximated -- see
:func:`prior_sensitivity`'s ``Raises``. Two refusal classes its rheplicant
ancestor needed have no bayesmith shape at all: a latent with *no* prior
cannot be declared here (every ``sample`` node carries a density), and a
prior living at a solver call-site cannot either (the graph is the single
statement of the model).

It needs the **likelihood's** mode to exist. Along a direction the rest
term does not hold there is no such mode, only a ray, and the displacement
from a ray is not a number -- so a selection whose rest-term curvature is
singular or digit-starved at the mode is refused. The verdict is taken on
that curvature rather than on the observed Jacobian's rank, because in a
graph the two can disagree where they never could in rheplicant: a latent
that reaches no observed node can still be held by a downstream latent's
density, which is part of the rest term. The refusal delegates its NAMING
to :func:`~bayesmith.diagnose.identifiability.identifiability` where the
two verdicts agree, since that is the tool that attributes a null
direction to latents in scale-free coordinates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.diagnose.identifiability import identifiability
from bayesmith.diagnose.local import (
    check_differentiable,
    check_observed_have_locs,
    flat_view,
    latent_values,
    refuse_ambient_float32,
    refuse_single_precision,
    resolve_names,
    unflatten,
)
from bayesmith.errors import ConvergenceError, GraphError, NotGaussian
from bayesmith.exact.gaussian import gaussian_parts, node_shape
from bayesmith.graph.evaluate import apply_probabilistic, evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic

#: The shift :attr:`PriorSensitivityReport.criterion_std` solves for, in
#: posterior sigmas.
#:
#: Chosen against what a chain can see rather than against a convention. The
#: MCSE of a posterior mean from ``n_eff`` effective draws is
#: ``1/sqrt(n_eff)`` sigma; a well-run 4 x 1000 NUTS fit lands at ``n_eff``
#: of a few hundred, so its own noise is 0.04-0.06 sigma. A shift of 0.1
#: sigma is therefore the smallest bias such a run could distinguish from its
#: own scatter at about 2 sigma of separation -- the point below which "is
#: the prior moving this?" stops being answerable by sampling and starts
#: needing this function.
#:
#: It is also small against the thing being biased: 0.1 sigma moves a 68%
#: interval's endpoints by 10% of the interval's half-width, which shifts the
#: reported central value without visibly changing the error bar. That is
#: the regime where a bias is easiest to publish by accident.
CRITERION_SHIFT: float = 0.1

#: Relative tolerance at which the closed form and the refit are called agreed.
#:
#: What it has to cover is the model's NONLINEARITY over the displacement,
#: not an error in the derivation: on an affine model the two routes agree at
#: the refit's own cancellation floor. Measured on the tour's nonlinear pair
#: (cross-checked against rheplicant's implementation, which pins the same
#: numbers): 9.6e-6 and 2.1e-6 relative at the declared priors. 3e-3 sits
#: two and a half decades above that, deliberately -- a posterior a prior
#: moves by a whole sigma curves over that distance far more than one moved
#: by 0.007 -- and a factor of 3 below the 1e-2 scale at which a "0.1 sigma
#: or not" verdict could flip.
VERIFY_RTOL: float = 3e-3

#: Absolute floor under :data:`VERIFY_RTOL`, in posterior sigmas.
#:
#: A relative comparison of two numbers that are both 1e-9 sigma reports the
#: Newton solver's own convergence floor and nothing else. One millionth of
#: a sigma is four decades below :data:`CRITERION_SHIFT` and below any MCSE
#: a chain could reach, so a disagreement smaller than this is not a
#: disagreement about anything.
VERIFY_ATOL: float = 1e-6

#: Newton steps allowed before the solve is called failed.
#:
#: The tour's MAP takes 7 from the prior centre and its likelihood-only
#: refit 3 from the MAP, so this is 14x the measured need. It is a ceiling
#: on a loop that either converges quadratically or is not going to.
MAX_NEWTON_STEPS: int = 100

#: Convergence test: ``max(|dx| / (1 + |x|)) < NEWTON_TOL``.
#:
#: Mixed relative/absolute so a latent at 1e-8 and one at 1e8 are held to
#: the same standard. 1e-13 is two decades above float64 eps, which is where
#: a quadratically-converging step size stops shrinking and starts
#: jittering.
NEWTON_TOL: float = 1e-13

#: Halvings of a Newton step allowed before the line search gives up.
#:
#: 2^-40 is 9e-13 of the full step: past that the step is doing nothing and
#: the problem is the direction, not its length.
MAX_BACKTRACKS: int = 40

#: Escalations of the Cholesky jitter before an indefinite Hessian is given up.
#:
#: The shift starts just past ``|lambda_min|`` and multiplies by 10, so 30
#: rounds cover 30 decades above the measured deficiency -- far past the
#: point where the modified matrix is dominated by its shift and the step
#: has degraded to (scaled) gradient descent, which still descends.
MAX_JITTER_ESCALATIONS: int = 30


def _descent_direction(matrix: jax.Array, gradient: jax.Array) -> jax.Array:
    """A descent direction from a possibly-indefinite curvature matrix.

    The B3 repair, and the reason it is a repair: a plain
    ``solve(H, gradient)`` at an indefinite ``H`` steps TOWARD a saddle --
    the solve is exact and the direction points uphill along the negative
    eigenvector -- and the line search then shortens a wrong direction 40
    times and reports failure, or worse, accepts a step that happens to
    lower the objective while walking out of the basin. The likelihood-only
    refit is exactly where this bites: the priors were the term keeping the
    Hessian comfortably positive.

    ``eigvalsh`` decides; positive-definite keeps the exact Newton step.
    Otherwise the spectrum is shifted just past positive and factored --
    Cholesky-with-jitter -- which only ever ROTATES the step toward the
    gradient, never reverses it, so a converged result is a stationary point
    of the true objective and not of some modified problem. ``cholesky``
    returns NaN rather than raising on a matrix that roundoff keeps
    indefinite after the shift, so finiteness of the factor is the retry
    test, and a shift that escalates past
    :data:`MAX_JITTER_ESCALATIONS` returns NaN for the caller's own
    finiteness check to fail the solve on.
    """
    eigenvalues = jnp.linalg.eigvalsh(matrix)
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    if np.isfinite(smallest) and smallest > 0.0:
        return jnp.linalg.solve(matrix, gradient)
    scale = max(abs(largest), abs(smallest)) if np.isfinite(largest) else 0.0
    if not np.isfinite(smallest) or scale == 0.0:
        return jnp.full_like(gradient, jnp.nan)
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    # 2|lambda_min| reflects the most negative eigenvalue to +|lambda_min|
    # rather than parking it just above zero: a shift of |lambda_min| + eps
    # leaves the modified matrix nearly singular along that eigenvector, and
    # the "descent direction" it returns is a near-infinite lunge the line
    # search then spends its whole budget taming (measured on the 2x2 pin
    # below: 2e7 against 0.5 for the reflected shift). The 1e-8 * scale term
    # keeps the shift nonzero for an exactly-semidefinite matrix.
    shift = 2.0 * abs(smallest) + 1e-8 * scale
    for _ in range(MAX_JITTER_ESCALATIONS):
        factor = jnp.linalg.cholesky(matrix + shift * identity)
        if bool(jnp.all(jnp.isfinite(factor))):
            return jax.scipy.linalg.cho_solve((factor, True), gradient)
        shift *= 10.0
    return jnp.full_like(gradient, jnp.nan)


def _newton(objective: Callable[[jax.Array], jax.Array], x0: jax.Array) -> tuple[jax.Array, int, bool]:
    """Damped Newton on a scalar objective. Returns ``(x, steps, converged)``.

    Damped rather than plain because the second solve here runs with the
    priors REMOVED, which is exactly the configuration where a full Newton
    step can leave the basin. The line search only ever shortens a step; the
    direction itself is :func:`_descent_direction`'s, which handles the
    indefinite case the same solve used to walk into (B3).

    A solve that fails comes back with ``converged=False`` and the last
    iterate rather than raising, because the caller has to decide whether a
    failed MAP (fatal) or a failed verification refit (reportable) is the
    situation.
    """
    value_and_grad = jax.value_and_grad(objective)
    hessian = jax.hessian(objective)
    x = x0
    for step in range(1, MAX_NEWTON_STEPS + 1):
        value, gradient = value_and_grad(x)
        direction = _descent_direction(hessian(x), gradient)
        if not bool(jnp.all(jnp.isfinite(direction))) or not bool(jnp.isfinite(value)):
            return x, step, False
        # `+ eps * |value|` rather than a strict decrease: at the mode the
        # two objective values differ only in their last bits, and a strict
        # test would backtrack 40 times on a converged solve and report
        # failure.
        ceiling = float(value) + 1e-12 * abs(float(value))
        length = 1.0
        for _ in range(MAX_BACKTRACKS):
            trial = x - length * direction
            if float(objective(trial)) <= ceiling:
                break
            length *= 0.5
        else:
            return x, step, False
        moved = float(jnp.max(jnp.abs(length * direction) / (1.0 + jnp.abs(trial))))
        x = trial
        if moved < NEWTON_TOL:
            return x, step, True
    return x, MAX_NEWTON_STEPS, False


def _refuse_entangled_selection(graph: Graph, names: Sequence[str]) -> None:
    """A selected latent's prior must be a FIXED Gaussian under the selection.

    The closed form's ``P`` is a constant diagonal; a selected latent whose
    ``dist_fn`` takes another selected latent (or anything derived from one)
    as a parent has a prior that MOVES with ``x``, and the identity would be
    differentiating a matrix it treats as constant. Held-fixed dependence on
    an UNSELECTED latent is fine -- those parents are constants at ``at``.
    """
    # One function is borrowed from the block machinery, for the same walk.
    from bayesmith.exact.block import _ancestors

    selected = set(names)
    for name in names:
        tangled = sorted(_ancestors(graph, name) & selected)
        if tangled:
            raise GraphError(
                f"prior_sensitivity was asked about latent {name!r}, whose own "
                f"density is parameterised (through its parents) by {tangled} "
                "-- also selected. The closed form treats each selected "
                "prior's (m, s) as constants, and here they move with the "
                "very parameters being analysed. Analyse the parent and the "
                "child in separate calls (names=), holding the other fixed "
                "with at=."
            )


def _prior_moments(
    graph: Graph,
    names: Sequence[str],
    shapes: Sequence[tuple[int, ...]],
    env: dict[str, Any],
) -> tuple[jax.Array, jax.Array]:
    """``(loc, scale)`` of every selected latent's density, flattened.

    Refuses, loudly, the one thing that would make the report a confident
    wrong number: a selected latent whose density has no quadratic form to
    put into ``P``. Both of its ancestor's other refusals -- no prior at
    all, and a prior living at a solver call-site -- are unbuildable in this
    package and appear here only as this sentence.
    """
    locations: list[jax.Array] = []
    scales: list[jax.Array] = []
    for name, shape in zip(names, shapes, strict=True):
        node = graph.node(name)
        try:
            loc, scale = gaussian_parts(graph, node, env)
        except NotGaussian as error:
            raise NotGaussian(
                f"prior_sensitivity was asked about latent {name!r}, whose "
                "density is not a diagonal Gaussian. The shift this reports "
                "is the displacement a Gaussian's quadratic pull puts on the "
                "mode; a Uniform exerts none at all inside its support and an "
                "unbounded one outside it, an ImproperUniform is the absence "
                "of a prior to be sensitive TO, and a LogNormal is Gaussian "
                "in log x, not in x. Reading any of them as (m, s) would "
                "report a smooth pull the declared density does not apply. "
                "Sample the graph with NUTS, which honours the density as "
                f"written, or exclude {name!r} with names=."
            ) from error
        shape_full = node_shape(graph, node, env)
        locations.append(jnp.ravel(jnp.broadcast_to(loc, shape_full)))
        scales.append(jnp.ravel(jnp.broadcast_to(scale, shape_full)))
    return jnp.concatenate(locations), jnp.concatenate(scales)


def _refuse_unanchored_selection(
    graph: Graph,
    names: Sequence[str],
    values: dict[str, jax.Array],
    likelihood_precision: jax.Array,
) -> None:
    """Refuse a selection whose likelihood-only mode does not exist.

    The verdict is taken on the REST term's own curvature ``H`` at the mode,
    not on the observed Jacobian's rank, and the difference is live in a
    graph the way it never was in this function's rheplicant ancestor: a
    selected latent that reaches no observed node at all can still be held
    by a DOWNSTREAM latent's density -- ``child ~ Normal(parent, s)`` with
    ``child`` outside the selection -- and that density is part of the rest
    term, so the likelihood-only mode is perfectly well defined. The
    Jacobian rank test would refuse that legitimate question.

    ``H`` must be invertible with digits to spare: the closed form solves
    against it and the refit walks on it. The ceiling is
    :func:`~bayesmith.exact.fisher.condition_ceiling`'s -- past
    ``1/sqrt(eps)`` the inverse has spent half the arithmetic's digits, the
    same line the covariance path draws, read from the same place rather
    than re-decided here.

    When the refusal fires, the NAMING is delegated to
    :func:`~bayesmith.diagnose.identifiability.identifiability` where its
    verdict agrees -- it is the tool that knows how to attribute a null
    direction to latents in scale-free coordinates. When the two disagree
    (an observed-Jacobian rank deficiency that some downstream density
    almost, but not quite, holds), the message says what was measured
    instead.
    """
    from bayesmith.exact.fisher import condition_ceiling

    eigenvalues = jnp.linalg.eigvalsh(likelihood_precision)
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])
    ceiling = condition_ceiling(likelihood_precision.dtype)
    # `not <` rather than `>=` so a NaN curvature is refused too.
    healthy = smallest > 0.0 and not largest / smallest >= ceiling
    if healthy:
        return

    report = identifiability(graph, names=names, at=values)
    if report.nullity:
        participation = report.participation(0)
        mixed = ", ".join(
            f"{name} {share:.2f}"
            for name, share in sorted(participation.items(), key=lambda item: -item[1])
        )
        named = (
            f"identifiability() reports rank {report.rank} of {report.n_par}, "
            f"and the first null direction mixes {mixed} (participation, in "
            "column-normalised coordinates)."
        )
    else:
        named = (
            "identifiability() reports the observed Jacobian as full-rank, so "
            "the deficiency comes through a density the selection "
            "parameterises rather than through the observed nodes; the "
            f"curvature's spectrum runs from {smallest:.3e} to {largest:.3e}."
        )
    raise GraphError(
        f"prior_sensitivity cannot report a prior shift for {list(names)}: at "
        "the mode, the likelihood's own curvature is singular or "
        f"digit-starved (smallest eigenvalue {smallest:.3e} against largest "
        f"{largest:.3e}, ceiling {ceiling:.1e}). The shift is defined as the "
        "displacement from the mode the LIKELIHOOD alone would choose, and "
        "along a direction the likelihood does not hold there is no such "
        "mode -- it is a ray, and the distance from a ray is not a number. "
        "What would come back is nevertheless finite, because the declared "
        "priors make the posterior proper: it would be the prior reporting "
        "on itself, with a well-formed sigma and a plausible magnitude. "
        + named
        + " Fix the parameterization, or restrict names= to a subset the "
        "likelihood determines."
    )


@dataclasses.dataclass(frozen=True)
class PriorSensitivityReport:
    """What the selected priors did to the mode, per latent, in posterior sigmas.

    A plain frozen dataclass holding **numpy**, for the same reasons
    :class:`~bayesmith.diagnose.identifiability.IdentifiabilityReport` is
    one: a derived verdict rather than a differentiable model, whose float64
    contents would silently truncate the moment a default-precision JAX
    caller touched them, and whose ``verified`` is a decision no traced
    program can branch on.

    Every array is flat over the SELECTED latents, in the order they were
    asked for -- :attr:`names` and :attr:`spans` are the only coordinate
    system in the object, and it is the selection's own order, not sorted
    order.

    Attributes:
        names: the latents analysed, in the order the caller asked for.
        shapes: their shapes, in the same order.
        spans: ``(start, stop)`` of each latent in the flat vector.
        n_par: total number of real parameters.
        mode: ``theta_hat``, the MAP found by Newton on the exact
            log-posterior.
        prior_loc: the declared ``m``, broadcast per element.
        prior_std: the declared ``s``, broadcast per element.
        mean_offset: ``|m - theta_hat|``. A magnitude -- the direction of the
            pull lives in :attr:`shift_sigma`'s sign, where a reader will
            look for it.
        sigma_post: ``sqrt(diag(Sigma))`` at the mode, from the exact Hessian
            of the negative log-posterior -- the observed curvature the
            Newton refit walks on, which is the one the two routes have to
            share if their agreement is to mean anything.
        shift_sigma: the closed form, signed, in units of :attr:`sigma_post`.
            Positive means the prior pulled the latent UP relative to the
            likelihood-only mode; negative, DOWN.
        shift_sigma_refit: the same displacement from an actual second Newton
            solve with the selected priors removed. All-NaN if that solve did
            not converge, in which case :attr:`refit_converged` is ``False``.
        verified: per element, whether the two routes agreed to
            :data:`VERIFY_RTOL` (with :data:`VERIFY_ATOL` underneath).
        criterion_std: the prior width at which this latent's shift would
            reach :data:`CRITERION_SHIFT`, from the diagonal law
            ``sqrt(sigma_post * mean_offset / 0.1)``. Compare it with
            :attr:`prior_std`: a declared width comfortably ABOVE it is the
            statement that the prior is not driving the fit. ``0.0`` when the
            prior mean sits exactly on the mode, which means no tightening of
            it moves anything.
        precision: the ``(n_par, n_par)`` posterior precision at the mode.
        newton_steps, refit_steps: what the two solves cost.
        refit_converged: whether the likelihood-only solve reached a mode.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    spans: tuple[tuple[int, int], ...]
    n_par: int
    mode: np.ndarray
    prior_loc: np.ndarray
    prior_std: np.ndarray
    mean_offset: np.ndarray
    sigma_post: np.ndarray
    shift_sigma: np.ndarray
    shift_sigma_refit: np.ndarray
    verified: np.ndarray
    criterion_std: np.ndarray
    precision: np.ndarray
    newton_steps: int
    refit_steps: int
    refit_converged: bool

    #: Which flat arrays :meth:`for_latent` splits. A ``ClassVar`` and not a
    #: field: annotated without it, a dataclass would make this the tenth
    #: positional argument of the constructor and put the list of field names
    #: inside the report's own repr.
    _PER_ELEMENT: ClassVar[tuple[str, ...]] = (
        "mode",
        "prior_loc",
        "prior_std",
        "mean_offset",
        "sigma_post",
        "shift_sigma",
        "shift_sigma_refit",
        "verified",
        "criterion_std",
    )

    def _index(self, name: str) -> int:
        if name not in self.names:
            raise GraphError(
                f"this report says nothing about {name!r}; it covers "
                f"{list(self.names)}. Pass it in names= to have it analysed."
            )
        return self.names.index(name)

    def for_latent(self, name: str) -> dict[str, np.ndarray]:
        """Every per-element quantity for one latent, reshaped like the latent.

        The form a caller acts in:
        ``report.for_latent("fg_beta")["shift_sigma"]`` is a number about
        beta, not an offset into a vector whose layout the caller has to
        have got right.
        """
        index = self._index(name)
        start, stop = self.spans[index]
        shape = self.shapes[index]
        return {
            field: np.asarray(getattr(self, field))[start:stop].reshape(shape)
            for field in self._PER_ELEMENT
        }

    def mode_of(self, name: str) -> np.ndarray:
        """``theta_hat`` for one latent, shaped like it."""
        return self.for_latent(name)["mode"]

    @property
    def worst(self) -> tuple[str, int, float]:
        """``(latent, index within it, signed shift)``, by largest magnitude.

        What to print first. An anonymous argmax over the flat vector would
        name a position in a layout the caller did not choose.
        """
        flat = int(np.argmax(np.abs(self.shift_sigma)))
        for name, (start, stop) in zip(self.names, self.spans, strict=True):
            if start <= flat < stop:
                return name, flat - start, float(self.shift_sigma[flat])
        raise GraphError(  # pragma: no cover - spans tile [0, n_par)
            f"inconsistent report: flat index {flat} falls in no latent's "
            f"span {list(self.spans)} over {self.n_par} parameters."
        )

    def shift_at(self, name: str, prior_std: Any) -> np.ndarray:
        """The shift this latent would suffer under a DIFFERENT prior width.

        The counterfactual :attr:`criterion_std` inverts, evaluated exactly
        rather than through the diagonal law: only ``name``'s entries of
        ``P`` are replaced, and
        ``(H + P_s)^-1 P_s (I + H^-1 P_d)(m - theta_hat)`` is solved whole.
        Every other latent's prior stays as declared, cross terms included
        -- which is why this can return a shift of the opposite sign to
        ``name``'s own pull once ``name``'s prior is loose enough for a
        neighbour's to dominate.

        **Anchoring.** That is not the expression :attr:`shift_sigma` uses,
        and the comment on the solve below derives why: the two are the same
        displacement written about different modes, and a counterfactual can
        only stand on the likelihood's. The ``(I + H^-1 P_d)`` factor is how
        it gets there without a second fit, and it makes the whole thing
        exact on a quadratic. Two consequences worth stating: at
        ``P_s = P_d`` it collapses to :attr:`shift_sigma` algebraically, so
        ``shift_at(name, declared_width)`` returns the reported shift to the
        last bit -- if it did not, one of the two would be wrong. And what
        is left at tight widths is the model, not the method.

        Reported in the sigma the caller actually HAS -- :attr:`sigma_post`,
        at the declared priors -- not in the sigma the counterfactual prior
        would produce. Dividing each row of a ladder by its own width would
        fold the prior's shrinking of the error bar into a number meant to
        report only the movement of the mode, and 'a 0.1 sigma shift' would
        then mean a different displacement in every row.

        Args:
            name: the latent to re-prior.
            prior_std: its hypothetical width -- a scalar, or anything
                broadcastable to the latent's shape.

        Returns:
            The signed shift in posterior sigmas, shaped like the latent.

        Raises:
            GraphError: if ``name`` is not in this report, or the width is
                not positive and finite, or does not broadcast.
        """
        index = self._index(name)
        start, stop = self.spans[index]
        shape = self.shapes[index]
        try:
            replacement = np.broadcast_to(
                np.asarray(prior_std, dtype=np.float64), shape
            ).ravel()
        except (TypeError, ValueError) as error:
            raise GraphError(
                f"prior_std={prior_std!r} does not broadcast to {name!r}'s "
                f"shape {shape}, so there is no width to put on "
                f"{stop - start} of its elements."
            ) from error
        if not np.all(np.isfinite(replacement)) or np.any(replacement <= 0.0):
            raise GraphError(
                f"shift_at({name!r}, {prior_std!r}) needs a positive, finite "
                "prior width: the shift goes as 1/s^2, so a zero or negative "
                "s is a division by zero or a NEGATIVE prior precision -- an "
                "anti-prior that pushes the mode away from its own mean, and "
                "would come back as a finite shift with the sign reversed. "
                "For 'no prior at all', pass a width large against the "
                "posterior's, which is the limit s -> inf."
            )
        scales = self.prior_std.copy()
        scales[start:stop] = replacement
        # (H + P_s)^-1 P_s (I + H^-1 P_d) (m - theta_hat), and the shape of
        # it is the anchor rather than the algebra. Two identities hold
        # exactly on a quadratic:
        #
        #   theta_s - theta_L = H^-1 P_s (m - theta_s)         [at theta_s]
        #                     = (H + P_s)^-1 P_s (m - theta_L) [at theta_L]
        #
        # `shift_sigma` reports the DECLARED prior, where theta_hat IS
        # theta_s, so it uses the first and is exact. A counterfactual has no
        # theta_s -- finding it is the refit this method exists to avoid --
        # so it must use the second, whose anchor theta_L it does not hold
        # either. But theta_L is not unknown: the declared solve already
        # displaced the mode by a measured amount, and running that
        # displacement backwards is exact --
        #
        #     m - theta_L = (I + H^-1 P_d) (m - theta_hat)
        #
        # -- which is the (I + H^-1 P_d) here. With it the counterfactual is
        # exact on a quadratic, and at s = s_declared the two P's coincide
        # and the whole thing collapses to H^-1 P_d (m - theta_hat): the
        # ladder's declared row IS `shift_sigma` to roundoff.
        diagonal = np.arange(self.n_par)
        likelihood_precision = self.precision.copy()
        likelihood_precision[diagonal, diagonal] -= 1.0 / self.prior_std**2
        offset = self.prior_loc - self.mode
        to_likelihood_mode = offset + np.linalg.solve(
            likelihood_precision, offset / self.prior_std**2
        )
        precision = likelihood_precision.copy()
        precision[diagonal, diagonal] += 1.0 / scales**2
        shift = np.linalg.solve(precision, to_likelihood_mode / scales**2)
        return (shift / self.sigma_post)[start:stop].reshape(shape)


def prior_sensitivity(
    graph: Graph,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
) -> PriorSensitivityReport:
    """How far the selected priors moved the mode, in posterior sigmas.

    See the module docstring for why a NUTS run cannot be asked this and for
    the two routes taken instead. Both are deterministic; nothing here
    samples. The data and the noise are the graph's own -- its observed
    nodes -- so there is nothing separate to pass and nothing that can
    disagree with what a sampler would read.

    The work is two Newton solves -- one on the exact log-posterior, one
    with the selected priors removed -- plus one dense Jacobian for the rank
    check. That is a design-time cost for tens to a few thousand parameters,
    the same envelope :func:`~bayesmith.diagnose.identifiability.
    identifiability` states, and for the same reason.

    Args:
        graph: the model. Every SELECTED latent must carry a diagonal
            Gaussian density whose parameters are fixed under the selection.
        names: which latents to analyse -- a sequence, or a bare string for
            one. ``None`` means all of them, in declaration order. A subset
            asks the CONDITIONAL question, holding the rest at ``at``.
        at: values for the latents NOT selected, and the Newton starting
            point for those that are. Defaults to each latent's prior
            centre.

    Returns:
        A :class:`PriorSensitivityReport`. The two numbers to read first are
        ``shift_sigma`` and ``criterion_std``::

            report = prior_sensitivity(graph)
            name, index, shift = report.worst
            criterion = report.for_latent(name)["criterion_std"].ravel()[index]
            print(f"{name}[{index}] moved {shift:+.4f} sigma by its prior; "
                  f"0.1 sigma would need s = {criterion:.3g}")

    Raises:
        GraphError: if ``names``/``at`` name an undeclared latent or a value
            does not broadcast; if a selected latent is complex or
            non-floating; if a selected latent's density is parameterised by
            the selection itself; if the likelihood's own curvature at the
            mode is singular or digit-starved, so the likelihood-only mode
            the shift is measured from does not exist; or if the arithmetic
            comes back float32 -- run the call (graph construction included)
            inside ``with jax.enable_x64(True):``.
        NotGaussian: if a selected latent's density has no quadratic form (a
            Uniform, an ImproperUniform, a LogNormal, ...).
        ConvergenceError: if the Newton solve for the MAP does not converge.
            (A failed VERIFICATION refit is reported, not raised:
            ``refit_converged=False`` and an all-NaN refit column.)
    """
    refuse_ambient_float32(doing="prior_sensitivity's mode displacement")
    selected = resolve_names(graph, names)
    values0 = latent_values(graph, at)
    check_differentiable(graph, selected, values0)
    check_observed_have_locs(graph, values0)
    _refuse_entangled_selection(graph, selected)

    x0, shapes, spans = flat_view(values0, selected)
    env0 = evaluate(graph, values0)
    loc, scale = _prior_moments(graph, selected, shapes, env0)

    def neg_log_rest(x: jax.Array) -> jax.Array:
        # Every probabilistic term EXCEPT the selected latents' own
        # densities: the observed nodes, and any unselected latent whose
        # density the selection parameterises. This is the graph-native "the
        # likelihood alone" -- what remains of log_joint once the priors
        # under study are removed.
        env = evaluate(graph, {**values0, **unflatten(x, selected, shapes, spans)})
        total = jnp.zeros((), dtype=x.dtype)
        for node in graph.nodes:
            if isinstance(node, Probabilistic) and node.name not in selected:
                distribution = apply_probabilistic(graph, node, env)
                total = total + jnp.sum(distribution.log_prob(env[node.name]))
        return -total

    def neg_log_posterior(x: jax.Array) -> jax.Array:
        return neg_log_rest(x) + 0.5 * jnp.sum(((x - loc) / scale) ** 2)

    refuse_single_precision(neg_log_posterior(x0), doing="the log-posterior")

    mode, newton_steps, converged = _newton(neg_log_posterior, x0)
    if not converged:
        raise ConvergenceError(
            f"prior_sensitivity could not find the mode: {newton_steps} damped "
            "Newton steps on the exact log-posterior did not converge to "
            f"max|dx|/(1+|x|) < {NEWTON_TOL:g}. Every number this reports is "
            "a displacement FROM that mode, so there is nothing to report -- "
            "the closed form would expand about a point that is not "
            "stationary and come back finite. Three things do this: a "
            "starting point in another basin (pass at=), a model whose "
            "arithmetic is float32 (the Hessian is then noise at the 1e-7 "
            "level), and a genuinely non-quadratic posterior, for which NUTS "
            "is the right tool and this one is not."
        )

    precision = jax.hessian(neg_log_posterior)(mode)
    covariance = jnp.linalg.inv(precision)
    sigma_post = jnp.sqrt(jnp.diag(covariance))

    # theta_hat - theta_L = H^-1 P (m - theta_hat), with H the LIKELIHOOD
    # curvature -- not the posterior's. Using (H + P)^-1 here is wrong by
    # exactly diag((H + P)^-1 P), the prior's share of the posterior
    # precision, which grows without bound as the prior tightens -- the
    # regime the report exists to describe.
    likelihood_precision = jax.hessian(neg_log_rest)(mode)
    values_at_mode = {**values0, **unflatten(mode, selected, shapes, spans)}
    _refuse_unanchored_selection(graph, selected, values_at_mode, likelihood_precision)

    offset = loc - mode
    shift_sigma = (
        jnp.linalg.solve(likelihood_precision, offset / scale**2) / sigma_post
    )

    likelihood_mode, refit_steps, refit_converged = _newton(neg_log_rest, mode)
    if refit_converged:
        shift_refit = (mode - likelihood_mode) / sigma_post
        agreed = jnp.abs(shift_sigma - shift_refit) <= (
            VERIFY_RTOL * jnp.abs(shift_refit) + VERIFY_ATOL
        )
    else:
        shift_refit = jnp.full_like(shift_sigma, jnp.nan)
        agreed = jnp.zeros_like(shift_sigma, dtype=bool)

    mean_offset = jnp.abs(offset)
    criterion = jnp.sqrt(sigma_post * mean_offset / CRITERION_SHIFT)

    # float64 explicitly rather than by inheritance: these arrays leave the
    # caller's x64 context, and an array that arrived here as float32 would
    # be a report whose digits stop before the effect it is measuring does.
    def as_numpy(array: jax.Array) -> np.ndarray:
        return np.asarray(array, dtype=np.float64)

    return PriorSensitivityReport(
        names=selected,
        shapes=shapes,
        spans=spans,
        n_par=int(x0.size),
        mode=as_numpy(mode),
        prior_loc=as_numpy(loc),
        prior_std=as_numpy(scale),
        mean_offset=as_numpy(mean_offset),
        sigma_post=as_numpy(sigma_post),
        shift_sigma=as_numpy(shift_sigma),
        shift_sigma_refit=as_numpy(shift_refit),
        verified=np.asarray(agreed, dtype=bool),
        criterion_std=as_numpy(criterion),
        precision=as_numpy(precision),
        newton_steps=int(newton_steps),
        refit_steps=int(refit_steps),
        refit_converged=bool(refit_converged),
    )


__all__ = [
    "CRITERION_SHIFT",
    "MAX_BACKTRACKS",
    "MAX_JITTER_ESCALATIONS",
    "MAX_NEWTON_STEPS",
    "NEWTON_TOL",
    "VERIFY_ATOL",
    "VERIFY_RTOL",
    "PriorSensitivityReport",
    "prior_sensitivity",
]
