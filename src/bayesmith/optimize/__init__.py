"""Gradient MAP -- the exit an exact solve does not have.

``wiener_solve`` and ``iterative_gls`` answer a model that is affine in its
latents; everything else has only a gradient. This module is that other exit:
descend the NEGATIVE joint log-density and report where you got to.

**The objective is the FULL density, and that is a decision rather than an
implementation detail** (D7 in the migration ledger). For a
prediction-dependent sigma the joint carries a ``sum log sigma`` term that a
GLS-flavoured potential drops, and dropping it moves the optimum -- so a point
estimate and a draw from the same declaration would target different
distributions. :func:`fit` reads :func:`~bayesmith.graph.evaluate.log_joint`,
which is also what ``to_numpyro`` samples, so there is one density and not
two.

Three things are public, and the split is the one the callers need:

* :func:`minimize` -- the optimiser on any scalar objective. A calibrator
  scoring a PREDICTION against DATA is not maximising a joint, and forcing it
  through a graph would mean inventing a model for a least-squares fit.
* :func:`fit` -- the graph entry. Joint MAP over every latent, or block
  coordinate over ``names=`` with the rest held, which is what a gradient
  block inside a Gibbs sweep is.
* :func:`check_loss_sense` -- the direction guard. A log-density has an
  error's signature and the opposite optimum, so a minimiser handed one
  descends a function unbounded below while the loss history looks like
  textbook convergence.

**There is a convergence verdict, and only when it is asked for.** Until
0.10.0 there was none: ``steps`` steps were taken and the objective reached
was reported, and a caller who needed a verdict compared it against something
of their own. That is still what a call without ``certify=`` does, and it is
still what every existing caller gets.

Given ``certify=<limit>``, :func:`minimize` and :func:`fit` return a
:class:`Fit` whose :attr:`~Fit.converged` is a PROOF: the Newton decrement
``sqrt(g^T H^-1 g)`` at the returned point is bounded above by ``limit``, in
units of the curvature's own standard deviation -- for a negative log
posterior, the posterior sigma. The Hessian is never formed from the model,
every product being a ``jvp`` of a ``grad``, and the inexactness of the solve
behind ``H^-1 g`` is bounded from the recomputed residual, so a truncated
iteration can only make the verdict REFUSE and never approve. Where the
arithmetic cannot support the claim -- an unproven curvature floor, a solve
that did not converge, a precision in which the decrement's digits are
rounding -- the answer is a refusal with its reason (:attr:`Fit.refusal`),
not a certificate. :mod:`bayesmith.optimize.certify` is where that lives and
says at length what each piece rests on.

The verdict bounds the Newton decrement of the local quadratic model at
this point. For a quadratic objective this is a distance to its minimum;
for a nonlinear objective it is a local statement, not proof of a global
MAP. It says nothing about whether the objective is the right one, nor --
with ``names=`` -- about anything beyond that block's conditional objective.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from bayesmith.errors import StructureError
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.graph import Graph

# The certificate. Imported as a module so `bayesmith.optimize.certify` is an
# attribute of this package, and by name for what this file uses -- `certify`
# is also the KEYWORD below, and a parameter that shadowed the module would
# make the two unusable in the same scope.
#
# The MODULE is deliberately not in `__all__`, and `bayesmith.exact` sets the
# precedent: it exports `precision_at` and never `precision`. `__all__` here
# has three readers and a submodule satisfies only two of them -- `import *`
# and the identity guard in `tests/test_public_api.py` take it, while
# `tools/build_docs.py` resolves every name to a documented API object and
# raises "Unresolved public API exports" on a module. Measured: listing it
# broke four tests in `test_documentation_site.py` and `test_campbell_
# scaling.py`, none of which names `optimize`. The import below is what makes
# `from bayesmith.optimize import certify` work; `__all__` never was.
from bayesmith.optimize import certify
from bayesmith.optimize.certify import (
    CONVERGED,
    STATUS_SAID,
    Decrement,
    decrement,
)
from bayesmith.optimize.certify import polish as newton_polish

__all__ = [
    "MAXIMIZE",
    "MINIMIZE",
    "Decrement",
    "Fit",
    "check_loss_sense",
    "fit",
    "minimize",
    "sense_of",
]

#: A scoring function whose optimum is its LOWEST value -- an error.
MINIMIZE: str = "minimize"

#: A scoring function whose optimum is its HIGHEST value -- a log-density.
MAXIMIZE: str = "maximize"

#: Methods :func:`minimize` knows. Named rather than duck-typed so a typo is a
#: refusal instead of a silent fall-through to whichever branch is last.
_METHODS: tuple[str, ...] = ("adam", "gradient")


class Fit(NamedTuple):
    """Where a descent got to, and how it got there.

    Attributes:
        values: the fitted point. For :func:`fit` this is EVERY latent, with
            the ones outside ``names`` at the values they were held at -- a
            caller stepping one block of a sweep gets back a complete
            environment rather than a fragment to merge itself.
        objective: the objective AT :attr:`values`, evaluated after the last
            step. Deliberately not ``history[-1]``, which is the value one
            step earlier: reporting that one beside this point is an
            off-by-one that reads as a converged fit.
        history: the objective at the start of each step, shape ``(steps,)``.
            So ``history[0]`` is the objective the caller started from, and
            ``objective < history[0]`` is what "it improved" means.
        certificate: the :class:`~bayesmith.optimize.certify.Decrement`
            measured at :attr:`values`, or ``None`` when no ``certify=`` was
            asked for. Read :attr:`converged` rather than this, unless the
            question is WHY.
        limit: the distance, in units of the curvature's own standard
            deviation, that the caller asked to be proven. ``None`` with no
            certificate.
        polished: Newton steps kept after the descent, when ``certify=``
            turned the polish on. ``0`` means the point came back as the
            descent left it.
    """

    values: dict[str, Any]
    objective: jax.Array
    history: jax.Array
    certificate: Decrement | None = None
    limit: float | None = None
    polished: int = 0

    @property
    def converged(self) -> bool:
        """Whether this point is PROVEN within :attr:`limit` of the minimum.

        Three conditions, all required: the curvature floor the bound divides
        by is a proof rather than an estimate, the decrement's own solve
        converged, and the upper bound it allows is inside the limit. See
        :meth:`~bayesmith.optimize.certify.Decrement.certifies`.

        **``False`` is not "it was measured and it failed".** It is also what
        a fit that was never asked for a certificate reports, which is most of
        them. The two are worth telling apart and :attr:`refusal` is what
        tells them apart, in a sentence. This attribute is deliberately not
        tri-state: a caller writing ``if fit.converged:`` gets the safe
        reading of every case, and one who needs more asks for more.
        """
        if self.certificate is None or self.limit is None:
            return False
        return self.certificate.certifies(self.limit)

    @property
    def refusal(self) -> str | None:
        """Why :attr:`converged` is ``False``, or ``None`` when it is ``True``.

        Reports the FIRST unmet condition in the order
        :meth:`~bayesmith.optimize.certify.Decrement.certifies` reads them, so
        the sentence names the thing to fix rather than the last thing
        checked.

        **The verdict is read from :attr:`converged`, not re-derived here.**
        Two chains of conditions over one object can disagree, and this pair
        did: written as independent branches, the last one returned "outside
        the limit" for a point whose distance was 0.0 and whose
        :attr:`converged` was ``True``. Delegating makes that
        unrepresentable rather than merely tested for.
        """
        if self.converged:
            return None
        if self.certificate is None or self.limit is None:
            return (
                "no certificate was requested, so nothing about this point's "
                "distance to the minimum has been measured. Pass "
                "`certify=<limit>` to `minimize` or `fit` to measure it; the "
                "limit is a distance in units of the curvature's own standard "
                "deviation, which for a negative log posterior is the "
                "posterior sigma."
            )
        note = self.certificate
        if note.floor_source == certify.FLOOR_NONE:
            # Distinct from a probed floor, and it reads differently: "none"
            # is not an estimate that failed to be a proof, it is no usable
            # curvature at all -- the dense path's smallest eigenvalue was not
            # positive, or the probe's interval did not clear zero. On a
            # saddle both this and the solve's status fail, and `certifies`
            # reads this one first, so the sentence has to carry both or it
            # names the symptom and hides the cause.
            return (
                "there is no usable curvature floor at this point: the "
                f"smallest eigenvalue of the Hessian was not positive, and "
                f"the decrement's solve {STATUS_SAID[note.status]}. A "
                "distance to the minimum presumes there is one to measure "
                "to; at a saddle or on a flat direction there is not. This "
                "is a statement about the point, not about the arithmetic -- "
                "descend further, or start somewhere the objective curves "
                "upwards in every direction."
            )
        if not note.proven:
            return (
                f"the curvature floor came from {note.floor_source!r}, which "
                "is an estimate and not a proof, so no distance here can be "
                "certified. A Lanczos probe bounds the distance from SOME "
                "eigenvalue rather than from the smallest, so its floor can "
                "sit above the true one and the bound built on it would be "
                "too small. Supply `floor=` -- a verified lower bound on the "
                "smallest eigenvalue of the Hessian -- or reduce the problem "
                f"to at most {certify.DENSE_MAX} real parameters, where the "
                "Hessian is formed and its smallest eigenvalue computed."
            )
        if note.status != CONVERGED:
            return (
                f"the decrement's solve {STATUS_SAID[note.status]}, so the "
                "number it would report is not bounded and nothing is "
                "certified."
            )
        return (
            f"the upper bound {note.distance:.4g} on the local Newton "
            f"decrement is outside the limit {self.limit:.4g} that was "
            "asked for. An upper bound above the limit does not prove "
            "that the true decrement exceeds it; this certificate is "
            "insufficient. Refine the solve or take more optimisation steps."
        )


def sense_of(scoring: Any) -> str:
    """:data:`MAXIMIZE` if ``scoring`` declares it, :data:`MINIMIZE` otherwise.

    A declaration is cheap and exact and gives the best message, but on its
    own it is a whitelist -- and a whitelist is wrong about precisely the code
    it has not met. :func:`check_loss_sense` therefore uses this as its first
    half and measures the second.
    """
    return MAXIMIZE if getattr(scoring, "sense", MINIMIZE) == MAXIMIZE else MINIMIZE


def check_loss_sense(
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
    predicted: jax.Array,
    observed: jax.Array,
) -> None:
    """Refuse a scoring function whose optimum lies the way a minimiser walks.

    Measured on a one-parameter gain fit with truth ``g = 1.0``, before this
    guard existed::

        mean_squared_error       ->  g = +0.9999    loss  2499  ->  0.002617
        GaussianLikelihood(0.05) ->  g = -30.7349   loss -3.2e7 -> -1.3e11

    Both runs report a monotonically improving loss, which is the only
    evidence a caller has.

    The measured half evaluates the score where its direction is unambiguous:
    at the PERFECT prediction, ``loss_fn(observed, observed)``. An error
    attains its minimum there and a log-density its maximum, so a perfect
    prediction scoring ABOVE the current one means the function increases
    toward the truth. That holds for any callable, declaration or not.

    Args:
        loss_fn: ``f(predicted, observed) -> scalar``.
        predicted: the prediction at the starting point.
        observed: the data.

    Raises:
        StructureError: if the sense is wrong by either test, or if either
            score is not finite. The last is not scope creep: NaN compares
            False against everything, so treating a non-finite score as
            "cannot tell, proceed" would wave through exactly the case this
            guard exists for whenever it arrives with a NaN attached.
    """
    if sense_of(loss_fn) == MAXIMIZE:
        raise StructureError(
            f"{type(loss_fn).__name__} declares sense={MAXIMIZE!r}: it is a "
            "log-density, and this optimiser minimises. Minimising a "
            "log-density walks away from the truth while reporting an "
            "improving loss. Pass `lambda p, o: -score(p, o)`, or use a "
            "density-aware route (`fit`, `nuts`)."
        )
    at_start = jnp.asarray(loss_fn(predicted, observed))
    at_truth = jnp.asarray(loss_fn(observed, observed))
    if not jnp.isfinite(at_start) or not jnp.isfinite(at_truth):
        raise StructureError(
            f"the loss is not finite at entry (start={at_start}, perfect-fit="
            f"{at_truth}). A fit cannot begin from here, and the sense of the "
            "scoring function cannot be established either -- a non-finite "
            "score compares False against everything."
        )
    if at_truth > at_start:
        raise StructureError(
            f"{getattr(loss_fn, '__name__', type(loss_fn).__name__)} scores a "
            f"PERFECT prediction ({at_truth}) higher than the starting one "
            f"({at_start}), so it increases toward the truth and must be "
            "maximised -- but this optimiser minimises, and will walk away "
            "from the answer while the loss history improves. Negate it: "
            "`lambda p, o: -score(p, o)`."
        )


def _checked_certificate(certify: float | None, floor: float | None) -> None:
    """The two numbers a certificate rests on, refused before anything is spent.

    ``floor`` without ``certify`` is refused rather than ignored, because the
    two readings of that call are "I wanted a certificate and forgot to ask"
    and "I passed a number that does nothing", and both deserve to be told.
    That is the opposite disposition to ``polish``, which IS ignored without
    ``certify`` -- the difference is that a wrong ``floor`` is the one input
    here that can produce a wrong ANSWER, so silence about it is not cheap.
    """
    if floor is not None and certify is None:
        raise StructureError(
            "floor= is the curvature floor a certificate's bound divides by, "
            "and no certificate was asked for. Pass certify=<limit> as well, "
            "or drop floor=."
        )
    if certify is None:
        return
    if not certify > 0:
        raise StructureError(
            f"certify must be > 0, got {certify!r}. It is the distance to the "
            "minimum that the fit must be proven to be within, in units of "
            "the curvature's own standard deviation; a non-positive one can "
            "never be met. Same `not >` spelling as learning_rate above, so a "
            "NaN limit is refused rather than quietly certifying nothing."
        )
    if floor is not None and not floor > 0:
        raise StructureError(
            f"floor must be > 0, got {floor!r}. It is a lower bound on the "
            "smallest eigenvalue of the Hessian, and the bound divides by its "
            "square root; at or below zero the objective has no minimum to "
            "measure a distance to."
        )


def _checked_settings(
    method: str, steps: int, learning_rate: float, beta1: float, beta2: float
) -> None:
    if method not in _METHODS:
        raise StructureError(
            f"method={method!r} is not one this optimiser has; it knows "
            f"{list(_METHODS)}. Named rather than guessed, because a typo that "
            "fell through to a default would change the algorithm silently."
        )
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise StructureError(f"steps must be a positive int, got {steps!r}.")
    # `not >` rather than `<=`, so a NaN rate is refused too: NaN compares
    # False against everything and would sail through the obvious spelling,
    # then turn every parameter into NaN on the first step.
    if not learning_rate > 0:
        raise StructureError(f"learning_rate must be > 0, got {learning_rate}.")
    # Adam's betas are exponential decay rates, so `[0, 1)` is what they mean
    # rather than a taste. Outside it the bias corrections `1 - beta**t` change
    # sign, and the result is not always loud: measured on `(x - 3)**2` from
    # x=0, 200 steps at rate 0.1, `beta1=1.5` returns **15.38** -- finite,
    # confidently, five times the true minimum. `beta1=1.0` and `beta2` outside
    # the range happen to trip the divergence guard below on that fixture, but
    # that is the fixture's luck and not a second guard; `beta2=1.5` returns
    # 2.99994 and looks perfect. Same `not (...)` spelling as the rate above,
    # so a NaN beta is refused rather than sailing through.
    # Adam only, because `beta1, beta2, eps: Adam's, ignored by "gradient"` is
    # this function's documented contract. Refusing an out-of-range beta under
    # `method="gradient"` would reject a call whose answer is correct, for a
    # value that provably did not enter it. The rule the two halves share:
    # refuse where it changes the answer, honour "ignored" where the contract
    # says ignored.
    for name, beta in (("beta1", beta1), ("beta2", beta2)) if method == "adam" else ():
        if not 0.0 <= beta < 1.0:
            raise StructureError(
                f"{name} must be in [0, 1), got {beta}. Adam's betas are "
                "exponential decay rates on the gradient's first and second "
                "moments; outside that range the bias correction "
                f"`1 - {name}**t` changes sign. This is refused rather than "
                "left to the caller because it does not announce itself: "
                "measured, beta1=1.5 minimises (x-3)^2 to 15.38 without a "
                "warning, a finite and wrong answer."
            )


def _refuse_complex(at: Any) -> None:
    """A complex leaf is refused rather than stepped.

    ``jax.grad`` of a real-valued objective at a complex point returns the
    CONJUGATE gradient, so subtracting it walks the wrong way -- without
    raising, without a NaN, and with a loss history that looks ordinary. The
    exact route splits a complex latent into its real parts and this one does
    not, so it says so.
    """
    complex_leaves = [
        jax.tree_util.keystr(path)
        for path, leaf in jax.tree_util.tree_flatten_with_path(at)[0]
        if jnp.issubdtype(jnp.result_type(leaf), jnp.complexfloating)
    ]
    if complex_leaves:
        raise StructureError(
            f"complex starting values ({', '.join(complex_leaves)}) cannot be "
            "stepped by this optimiser. `jax.grad` of a real objective at a "
            "complex point gives the CONJUGATE gradient, so a descent using it "
            "moves the wrong way without erroring. Split the latent into real "
            "and imaginary parts, or use the exact route "
            "(`linear_operator` / `wiener_solve` / `gcr_sample`), which does "
            "that split internally."
        )


def _rates(at: Any, learning_rate: float, step_sizes: Mapping[str, float] | None) -> Any:
    """A tree of per-leaf step sizes, shaped like ``at``.

    ``step_sizes`` needs names, and only a mapping has them; a general pytree
    gets one rate. That is not a limitation dressed as a rule -- a per-latent
    rate is per-latent because the latents carry different UNITS, and a pytree
    with no names has nothing for the caller to attach a unit to.
    """
    if step_sizes is None:
        return jax.tree.map(lambda _: learning_rate, at)
    if not isinstance(at, Mapping):
        raise StructureError(
            "step_sizes= names its rates, so the starting point has to be a "
            f"mapping of name to value; this one is a {type(at).__name__}. "
            "Pass learning_rate= for a single rate over an unnamed pytree."
        )
    unknown = sorted(set(step_sizes) - set(at))
    if unknown:
        raise StructureError(
            f"step_sizes names {unknown}, which this fit is not moving; it "
            f"moves {sorted(at)}. A rate for a held parameter has nothing to "
            "scale, and the most likely cause is that `names=` and "
            "`step_sizes=` were written against different blocks."
        )
    for name, rate in step_sizes.items():
        if not rate > 0:
            raise StructureError(
                f"step_sizes[{name!r}] must be > 0, got {rate}."
            )
    return {name: float(step_sizes.get(name, learning_rate)) for name in at}


def minimize(
    objective: Callable[[Any], jax.Array],
    at: Any,
    *,
    method: str = "adam",
    steps: int = 200,
    learning_rate: float = 1e-2,
    step_sizes: Mapping[str, float] | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    certify: float | None = None,
    floor: float | None = None,
    polish: bool = True,
) -> Fit:
    """Descend ``objective`` from ``at`` for ``steps`` steps.

    Args:
        objective: ``f(values) -> scalar``, MINIMISED. Handing it something
            that should be maximised is the failure :func:`check_loss_sense`
            exists for; this function cannot detect it, because a bare
            objective has no perfect-prediction point to evaluate at.
        at: the starting point -- any pytree, or a ``{name: value}`` mapping
            if ``step_sizes`` is used.
        method: ``"adam"`` or ``"gradient"``.
        steps: how many to take. There is no early stop, so this is also
            exactly how many are taken.
        learning_rate: the step size for every leaf without one of its own.
        step_sizes: per-name step sizes, in each parameter's own units.
        beta1, beta2, eps: Adam's, ignored by ``"gradient"``.
        certify: a distance to PROVE the returned point is within, in units of
            the curvature's own standard deviation -- for a negative log
            posterior, the posterior sigma. Given one, the fit carries a
            :class:`~bayesmith.optimize.certify.Decrement` and
            :attr:`Fit.converged` answers. Left out, nothing is measured and
            :attr:`Fit.converged` is ``False`` for that reason; see
            :attr:`Fit.refusal`.
        floor: a verified lower bound on the smallest eigenvalue of the
            objective's Hessian. Only consulted above
            :data:`~bayesmith.optimize.certify.DENSE_MAX` real parameters,
            where the Hessian is not formed; below it the floor is computed
            and this is not needed. It is a CLAIM the caller must be able to
            prove -- a model affine in every parameter jointly, with proper
            Gaussian priors and a noise scale that does not depend on the
            prediction, is bounded below by its prior precision. An unsound
            floor makes an unsound certificate, which is the one way to get a
            wrong answer here. Refused without ``certify=``.
        polish: whether to take Newton steps after the descent before the
            certificate is measured. Only consulted with ``certify=``, and on
            by default there because without it almost nothing certifies: a
            first-order step of ``rate * sign(gradient)`` does not shrink with
            the gradient, so Adam settles a fraction of a step size from the
            optimum -- measured upstream at 2881 curvature standard deviations
            on a two-parameter power law, at every step count. Each Newton
            step is kept only if it lowers the objective, so this cannot make
            the answer worse. Ignored without ``certify=``, on the same rule
            as ``beta1``/``beta2`` under ``method="gradient"``: refuse where
            it changes the answer, honour "ignored" where the contract says
            ignored.

    Returns:
        A :class:`Fit`.

    Raises:
        StructureError: for an unknown method, a non-positive step count or
            rate, a NaN rate, a complex starting value, or a ``step_sizes``
            naming something not being moved.
        Exception: from ``eqx.error_if`` if the objective at the point
            reached is not finite -- a diverged descent, which otherwise
            comes back looking like any other answer.

    Note:
        **The two methods do not accept the same rates**, and that is
        arithmetic rather than taste: plain gradient descent diverges for any
        rate above ``2/L``, where ``L`` is the objective's curvature, and
        Adam's step is bounded by its rate regardless. Measured on a
        one-latent Gaussian with ``L = 231`` (limit 0.00865): ``"gradient"``
        converges at 0.006 and returns NaN at 0.02, while ``"adam"``
        converges at both. Switching ``method=`` without revisiting
        ``learning_rate=`` is therefore a real hazard, which is why the
        non-finite result is refused rather than returned.
    """
    _checked_settings(method, steps, learning_rate, beta1, beta2)
    _checked_certificate(certify, floor)
    _refuse_complex(at)
    rates = _rates(at, learning_rate, step_sizes)

    if method == "gradient":

        def step(carry: Any, _: Any) -> tuple[Any, jax.Array]:
            value, grads = jax.value_and_grad(objective)(carry)
            moved = jax.tree.map(lambda p, g, r: p - r * g, carry, grads, rates)
            return moved, value

        found, history = lax.scan(step, at, None, length=steps)
    else:
        zeros = jax.tree.map(jnp.zeros_like, at)

        def adam_step(carry: Any, index: jax.Array) -> tuple[Any, jax.Array]:
            point, first, second = carry
            value, grads = jax.value_and_grad(objective)(point)
            first = jax.tree.map(lambda a, g: beta1 * a + (1 - beta1) * g, first, grads)
            second = jax.tree.map(
                lambda a, g: beta2 * a + (1 - beta2) * g**2, second, grads
            )
            count = index + 1
            point = jax.tree.map(
                lambda p, m, v, r: p
                - r
                * (m / (1 - beta1**count))
                / (jnp.sqrt(v / (1 - beta2**count)) + eps),
                point, first, second, rates,
            )
            return (point, first, second), value

        (found, _, _), history = lax.scan(
            adam_step, (at, zeros, zeros), jnp.arange(steps)
        )
    reached = jnp.asarray(objective(found))
    # `eqx.error_if` rather than a Python `if`, so the guard also fires under
    # jit -- the same mechanism, and the same reason, as `wiener_solve`'s.
    # Attached to the POINT, because that is what a caller reads first and an
    # unused check can be optimised away.
    found = eqx.error_if(
        found,
        ~jnp.isfinite(reached),
        "the objective is not finite at the point this fit reached, so there "
        "is no answer here to return. The usual cause is a step size above "
        "the stability limit: plain gradient descent diverges for any rate "
        "above 2/L, where L is the objective's curvature, and Adam does not "
        "-- so a `method=` changed without changing `learning_rate=` lands "
        "here. Measured on a one-latent Gaussian with L = 231: rate 0.006 "
        "converges and 0.02 gives NaN, while Adam converges at either. Lower "
        "the rate, use step_sizes= if the latents differ in units, or start "
        "somewhere the objective is finite.",
    )
    if certify is None:
        return Fit(values=found, objective=reached, history=history)

    if polish:
        found, kept = newton_polish(objective, found)
        # Re-read the objective AT the returned point, for `Fit.objective`'s
        # own contract -- "the objective at `values`, evaluated after the last
        # step" -- which a polish that moved the point would otherwise break.
        reached = jnp.asarray(objective(found))
        steps_kept = int(kept)
    else:
        steps_kept = 0
    return Fit(
        values=found,
        objective=reached,
        history=history,
        certificate=decrement(objective, found, floor=floor, limit=certify),
        limit=float(certify),
        polished=steps_kept,
    )


def fit(
    graph: Graph,
    at: Mapping[str, Any] | None = None,
    *,
    names: tuple[str, ...] | None = None,
    method: str = "adam",
    steps: int = 200,
    learning_rate: float = 1e-2,
    step_sizes: Mapping[str, float] | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    certify: float | None = None,
    floor: float | None = None,
    polish: bool = True,
) -> Fit:
    """Gradient MAP: maximise the graph's joint log-density over its latents.

    Args:
        graph: the model. Its ``log_joint`` is the objective, negated -- the
            FULL density, ``sum log sigma`` and any declared ``joint_prior``
            included, which is the same density ``to_numpyro`` samples (D7).
        at: where to start, ``{latent: value}``. Any latent left out starts at
            its prior centre, which is where
            :func:`~bayesmith.dispatch.classify.block_at` anchors a block, so
            the two entries agree about "before seeing anything". Defaults to
            every latent there.
        names: the latents to MOVE; the rest are held at their value in
            ``at``. ``None`` moves all of them (joint MAP). A subset is block
            coordinate, which is what a gradient block inside a sweep is.
        method, steps, learning_rate, step_sizes, beta1, beta2, eps: see
            :func:`minimize`. ``step_sizes`` is per latent, in the latent's own
            units -- a caller who knows the declared scales should use it,
            because one global rate cannot serve two latents whose units are
            orders of magnitude apart.

    Returns:
        A :class:`Fit` whose ``values`` covers EVERY latent, held ones
        included.

    Raises:
        StructureError: for a graph with no latents, a ``names`` entry the
            graph does not declare, an empty ``names``, or anything
            :func:`minimize` refuses.
    """
    from bayesmith.exact._environment import prior_environment

    latents = tuple(graph.latents)
    if not latents:
        raise StructureError(
            "fit() has nothing to move: this graph declares no latents. Every "
            "probabilistic node in it is observed, so its joint density is a "
            "constant."
        )
    if names is not None and not tuple(names):
        raise StructureError(
            "fit(names=()) moves nothing. Name at least one latent, or leave "
            "names= out for the joint MAP over all of them."
        )
    moving = latents if names is None else tuple(names)
    unknown = [name for name in moving if name not in latents]
    if unknown:
        raise StructureError(
            f"fit() was asked to move {unknown}, which this graph does not "
            f"declare as latents; its latents are {list(latents)}."
        )

    centres = prior_environment(graph)
    given = {} if at is None else dict(at)
    environment = {
        name: jnp.asarray(given[name] if name in given else centres[name])
        for name in latents
    }
    held = {name: environment[name] for name in latents if name not in set(moving)}
    start = {name: environment[name] for name in moving}

    def objective(values: dict[str, Any]) -> jax.Array:
        return -log_joint(graph, {**held, **values})

    found = minimize(
        objective,
        start,
        method=method,
        steps=steps,
        learning_rate=learning_rate,
        step_sizes=step_sizes,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        certify=certify,
        floor=floor,
        polish=polish,
    )
    return Fit(
        values={**environment, **found.values},
        objective=found.objective,
        history=found.history,
        # The certificate is over the latents that MOVED, which is the block
        # this call optimised and the only thing a decrement here can mean.
        # `values` carries the held ones too, so a reader must not take the
        # two to be over the same set: with `names=`, `converged` says this
        # BLOCK is at its conditional minimum given the rest, which is what a
        # gradient block inside a sweep needs and is not a statement about
        # the graph's joint MAP.
        certificate=found.certificate,
        limit=found.limit,
        polished=found.polished,
    )
