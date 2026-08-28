"""The ``gibbs_fn`` numpyro's ``HMCGibbs`` calls, and the assembly around it.

``HMCGibbs``'s own docstring says "it is the user's responsibility to provide a
correct implementation of ``gibbs_fn`` that samples from the corresponding
posterior conditional". **That sentence is this package's product.** NumPyro
supplies the scaffolding and cannot know which conditionals are exactly
solvable, nor solve them; the graph is what says so, and
:mod:`bayesmith.exact.block` and :mod:`bayesmith.exact.solve` are what do it.

**Two sweeps, one factory.** ``method="gcr"`` draws the conditional exactly:
sigma does not move with the block, so the frozen-sigma Gaussian *is* the
conditional. ``method="gcr+mh"`` is for the case sigma does move -- then the
frozen-sigma draw is only a **proposal**, and the sweep is an
independence-proposal Metropolis step whose correctness argument is
:func:`_mh_step`'s docstring and is the single largest correction in P3b's
spec.

**Why this module takes a graph and a name list rather than a plan object.**
``InferencePlan`` lives in :mod:`bayesmith.dispatch`, which is the layer
*above* this one: ``dispatch`` reads ``exact``'s checkers, and **no module in
``exact`` imports ``dispatch`` at module scope**. Importing the plan here
would invert that. So :func:`gibbs_factory` and :func:`assemble` take the
primitives they actually need and the plan layer adapts to them -- a layering
decision, not an oversight.

The sentence used to read "``exact`` never reads ``dispatch``", which is not
true and had no guard: :func:`bayesmith.exact.fisher.push_forward` borrows
``prior_environment`` from :mod:`bayesmith.dispatch.classify` inside the call.
That one is a function-scope import and cannot be hoisted -- ``classify``
reaches back into ``exact.gaussian`` for ``gaussian_parts``, and
``exact/gaussian.py`` imports ``graph.evaluate`` at module scope, so moving
the borrowed function up would close a real cycle. What is actually true, and
what ``tests/test_layering.py`` now asserts, is the module-scope statement.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS, HMCGibbs

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.errors import NotGaussian
from bayesmith.exact.block import (
    LinearBlock,
    _env_before,
    domain_centre,
    unchecked_operator,
)
from bayesmith.exact.correct import log_weight
from bayesmith.exact.gaussian import precision_at
from bayesmith.exact.gls import iterative_gls, precision_from_graph
from bayesmith.exact.solve import gcr_sample, wiener_solve
from bayesmith.graph.graph import Graph

#: The two corrections this module implements, and there is no third.
#: ``"nuts"`` and ``"gcr+snis"`` are also rows of the dispatch table, but
#: neither is a Gibbs sweep: ``"nuts"`` has no exact block and ``"gcr+snis"``
#: spans every latent, so it produces iid draws and reweights them once, with
#: no chain to be a kernel of.
GIBBS_METHODS: tuple[str, ...] = ("gcr", "gcr+mh")


def _prior_centre(graph: Graph) -> dict[str, jax.Array]:
    """Every latent at the centre of its own prior.

    Obtained by running :func:`~bayesmith.exact.block._env_before` with EVERY
    latent named as a member and an empty ``at``: that scan centres each
    member at its own prior mean as it reaches it, in topological order, so a
    latent whose prior depends on an earlier latent -- ``mixed_radiometer``'s
    ``w ~ N(0, |tau| + 0.2)`` -- gets its width from ``tau``'s centre rather
    than from a placeholder. Reusing that scan rather than writing a third one
    is deliberate: :func:`~bayesmith.graph.evaluate.evaluate` and
    ``_env_before`` already repeat one isinstance ladder between them, and a
    third copy is a third thing to drift.

    Raises:
        NotGaussian: if any latent is not a diagonal Gaussian, so it has no
            ``loc`` to centre at. Re-raised with what to do instead, because
            the caller has two ways out that the underlying message does not
            mention.
    """
    try:
        env, _ = _env_before(graph, tuple(graph.latents), {}, probe_gaussian=False)
    except NotGaussian as exc:
        raise NotGaussian(
            f"sigma cannot be hoisted out of the sweep on this graph: {exc} "
            "Hoisting evaluates sigma once with every latent at the centre of "
            "its own prior, and a latent with no loc has no centre. Either pass "
            "sigma_rebuild=True, so sigma is rebuilt from `at` inside the sweep "
            "and the non-Gaussian latent's value comes from NUTS, or pass an "
            "explicit precision=.",
            reason="no_centre",
            node=exc.node,
        ) from exc
    return {name: env[name] for name in graph.latents}


def _precision_at(
    graph: Graph,
    block: LinearBlock,
    at: dict[str, Any],
    method: str,
    tol: float,
    maxiter: int | None,
) -> dict[str, Any]:
    """The frozen ``{observed: N^-1}``, as a function of ``at`` ALONE.

    The OPERATOR side of the seam throughout: nothing in this module reads a
    per-sample sigma, it only ever hands the noise to ``gcr_sample``,
    ``wiener_solve`` and ``log_weight``. So the conversion happens once, here,
    where the frozen covariance is decided -- not at each of those four call
    sites, which is where two vocabularies would meet.

    That it does not see the block's current value is the whole correctness
    argument of :func:`_mh_step`, so it is enforced by this function's
    signature: there is no ``x`` to pass.

    ``method="gcr"`` reads sigma at the block's own prior mean. What licenses
    the choice of point is the method itself -- ``"gcr"`` means sigma does not
    move with the block, and then every point of the block gives the same
    sigma. ``method="gcr+mh"`` means it DOES move, so the point matters, and
    the GLS fixed point is the one that makes the proposal resemble the
    target. Any x-independent point would be **correct**; this one is the one
    that gets **accepted**.
    """
    if method == "gcr":
        return precision_at(graph, {**at, **domain_centre(block)})
    return iterative_gls(
        block,
        precision_of=precision_from_graph(graph, at),
        depends_on_prediction=True,
        tol=tol,
        maxiter=maxiter,
        require_convergence=None,
    ).precision


def _hoisted_precision(
    graph: Graph,
    names: tuple[str, ...],
    method: str,
    tol: float,
    maxiter: int | None,
) -> dict[str, Any]:
    """The covariance evaluated ONCE, at the graph's prior centre, outside the sweep.

    What ``sigma_rebuild=False`` buys. The block is rebuilt here too, at the
    outside latents' prior centres, and with ``probe_gaussian`` left ON --
    this runs at compile time, which is exactly where
    :func:`~bayesmith.exact.block.unchecked_operator` says the Gaussianity
    probe has to run once before the sweep may disable it.
    """
    centre = _prior_centre(graph)
    at = {name: value for name, value in centre.items() if name not in set(names)}
    block = unchecked_operator(graph, names, at=at)
    return _precision_at(graph, block, at, method, tol, maxiter)


def gibbs_factory(
    graph: Graph,
    names: Iterable[str],
    *,
    tol: float,
    method: str = "gcr",
    sigma_rebuild: bool = False,
    maxiter: int | None = None,
    precision: Mapping[str, Any] | None = None,
) -> Callable[..., dict[str, jax.Array]]:
    """Build the callable ``HMCGibbs`` invokes once per sweep.

    Args:
        graph: the model. ``log p`` inside the sweep is its own ``log_joint``,
            so the Gibbs step and NUTS cannot target different posteriors.
        names: the block's members -- the ``gibbs_sites``.
        tol: CG tolerance, chosen at compile time as
            ``CONVERGENCE_TARGET / condition_bound(...)``. The in-sweep
            convergence guard is **off**: it costs ``POWER_ITERATIONS`` extra
            operator applications per solve and ``eqx.error_if`` cannot help a
            sweep it cannot interrupt. So this number is the only thing
            standing between the sweep and a silently over-narrow posterior,
            since an unconverged CG leaves the prior-dominated directions at
            their starting value. Never leave it at a default with the guard
            off.
        method: one of :data:`GIBBS_METHODS`. ``"gcr"`` for a genuine
            conditional draw; ``"gcr+mh"`` when sigma depends on the block
            itself and the frozen-sigma draw is only a PROPOSAL.
        sigma_rebuild: recompute the covariance inside the sweep, at the
            current ``at``. Required whenever any observed node's scale has a
            latent ancestor -- including one OUTSIDE the block, which
            :func:`~bayesmith.exact.gls.check_prediction_dependence` cannot
            see because it only ever moves block members. Measured on
            ``contrast_sigma_pair`` split as block ``{a}`` with ``b`` outside
            (sigma ``base * exp(a - b)``): the movement probe reads
            ``1.71828183e+00`` -- bitwise identical at ``b = 0.5`` and
            ``b = 2.0`` -- while hoisting sigma gives a posterior 1.64x and
            7.35x too WIDE at those two. The criterion is therefore
            structural, not numeric; a movement probe is precisely what
            cannot see it.
        maxiter: CG iteration cap, passed to every solve in the sweep.
        precision: an x-independent ``{observed: N^-1}`` of the caller's own
            choosing, used verbatim in place of everything above.
            **Correctness does not depend on this being any good** -- see
            :func:`_mh_step`; a worse covariance costs acceptance rate, not
            validity. Useful when the GLS fixed point is too expensive to
            recompute every sweep, or when a deliberately loose covariance is
            wanted. Build one with
            :func:`~bayesmith.exact.precision.diagonal_from` from a sigma
            dict, or take :attr:`~bayesmith.exact.gls.GLSResult.precision`.

    Returns:
        A function taking ``rng_key``, ``gibbs_sites`` and ``hmc_sites``
        **by keyword** -- that is numpyro's actual calling convention, see
        ``HMCGibbs.sample`` -- and returning exactly ``names``, in constrained
        space. Returning a subset raises an ``AssertionError`` upstream whose
        message is empty; returning an extra key naming a NUTS latent is
        silently ignored. Both measured on numpyro 0.21.0.

    Raises:
        ValueError: if ``method`` is not in :data:`GIBBS_METHODS`.
        NotGaussian: if the covariance must be hoisted
            (``sigma_rebuild=False``, no ``precision``) and some latent has no
            prior centre to hoist at.
    """
    names = tuple(names)
    if method not in GIBBS_METHODS:
        raise ValueError(
            f"unknown method {method!r}; this module implements "
            f"{list(GIBBS_METHODS)}. 'gcr' draws the conditional exactly and "
            "'gcr+mh' corrects a frozen-sigma proposal -- the two differ by a "
            "Metropolis accept step, so a typo silently buys or skips one."
        )
    frozen: dict[str, Any] | None = None
    if precision is not None:
        frozen = dict(precision)
    elif not sigma_rebuild:
        frozen = _hoisted_precision(graph, names, method, tol, maxiter)

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # `hmc_sites` arrives POST-processed, so it carries the graph's
        # deterministic nodes as well as its NUTS latents; `at` is only ever
        # the latents, which is also what `_validated_at` will accept.
        at = {k: v for k, v in hmc_sites.items() if k in graph.latents}
        block = unchecked_operator(graph, names, at=at, probe_gaussian=False)
        noise = (
            frozen
            if frozen is not None
            else _precision_at(graph, block, at, method, tol, maxiter)
        )
        if method == "gcr":
            draw, _ = gcr_sample(
                block,
                precision=noise,
                key=rng_key,
                tol=tol,
                maxiter=maxiter,
                require_convergence=None,
            )
            return {name: draw[name] for name in names}
        return _mh_step(
            graph, block, gibbs_sites, at, noise, rng_key, tol, maxiter, names
        )

    return gibbs_fn


def _mh_step(
    graph: Graph,
    block: LinearBlock,
    current: Mapping[str, Any],
    at: dict[str, Any],
    precision: dict[str, Any],
    key: jax.Array,
    tol: float,
    maxiter: int | None,
    names: tuple[str, ...],
) -> dict[str, jax.Array]:
    """Independence-proposal Metropolis, so that ``log det M`` genuinely cancels.

    The spec's first draft froze sigma at ``sigma(m(x))`` -- the CURRENT state
    -- and rebuilt the proposal at ``x'`` to get a reverse density. That is
    where it went wrong. Rebuilding makes ``M' != M``, so the ratio carries
    ``1/2 (log det M' - log det M)``, which is nonzero exactly when sigma
    moved, i.e. in every case this path exists for -- and ``log det`` of an
    implicit operator is the ONE quantity a matrix-free method cannot produce.

    Measured against 1-D quadrature on ``steep_radiometer``, one step applied
    to a quadrature-exact sample of the target: the correct step moves the
    sample's mean by +0.0 to +0.9 standard errors and its sd by -0.15% to
    -0.34%, while the draft's adaptive sigma moves it by **-16.1 to -17.0
    standard errors** and narrows the sd by 13.0-17.3% -- at an acceptance
    rate of 0.591-0.614 against the correct step's 0.456-0.477, i.e. one that
    looks HEALTHIER. That is the silent-wrong-answer shape.

    Freezing sigma at a function of the OUTER state alone fixes it. The
    proposal is then genuinely independent of ``x``, ``M' = M`` exactly, the
    constant cancels for real, and the cost drops from 3 CG solves to 2 --
    the forward mean and the draw, with both densities being quadratic forms
    costing one operator application each. :func:`_precision_at` has no ``x``
    parameter for that reason.

    A consequence worth stating: **correctness does not depend on sigma-hat
    being any good.** Any x-independent choice gives a valid chain; the
    quality of sigma-hat sets only the acceptance rate. That converts a
    correctness risk into a performance knob, and it is why ``precision=``
    can be handed in from outside without a caveat.

    Sigma must also NOT be taken from the previously accepted ``x``: that
    depends on the chain's history, making this an adaptive chain valid only
    under diminishing adaptation. ``at`` is the outer state and nothing else
    reaches this function.

    The reverse-density term does NOT vanish -- ``q(x)/q(x')`` is not 1 for an
    independence proposal, only ``det M`` is. Dropping ``log_weight(now)``
    leaves ``log alpha = log w(x')``, which on ``steep_radiometer`` is so
    negative that not one proposal in 96,000 was accepted: the kernel becomes
    the identity, which passes an invariance test trivially and is why
    ``test_the_mh_step_leaves_the_exact_conditional_invariant`` asserts an
    acceptance band as well as invariance.
    """
    propose_key, accept_key = jax.random.split(key)
    mu, _ = wiener_solve(
        block, precision=precision, tol=tol, maxiter=maxiter, require_convergence=None
    )
    draw, _ = gcr_sample(
        block,
        precision=precision,
        key=propose_key,
        tol=tol,
        maxiter=maxiter,
        require_convergence=None,
    )
    proposed = {name: draw[name] for name in names}
    now = {name: current[name] for name in names}
    log_alpha = log_weight(
        graph, block, proposed, at=at, precision=precision, mu=mu
    ) - log_weight(graph, block, now, at=at, precision=precision, mu=mu)
    take = jnp.log(jax.random.uniform(accept_key)) < log_alpha
    # `jnp.where` rather than a Python branch: `take` is traced. A rejection
    # returns `now` BITWISE, which is what lets a test read the acceptance
    # rate off `out != current` without the kernel reporting it.
    return {name: jnp.where(take, proposed[name], now[name]) for name in names}


def assemble(
    graph: Graph,
    names: Iterable[str],
    *,
    tol: float,
    method: str = "gcr",
    sigma_rebuild: bool = False,
    maxiter: int | None = None,
    precision: Mapping[str, Any] | None = None,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 1,
    chain_method: str = "sequential",
    progress_bar: bool = False,
    nuts_options: Mapping[str, Any] | None = None,
) -> MCMC:
    """Wire the block's sweep into ``HMCGibbs`` and hand back an unrun ``MCMC``.

    Unrun on purpose: ``mcmc.run(key)`` is the caller's, so the plan layer
    keeps control of the key and of whatever it wants to do with
    ``mcmc.get_extra_fields()`` afterwards.

    Args:
        graph, names, tol, method, sigma_rebuild, maxiter, precision: passed
            straight to :func:`gibbs_factory`.
        num_warmup, num_samples, num_chains, progress_bar: ``MCMC``'s own.
        chain_method: ``"sequential"`` or ``"parallel"``. Both were run
            end-to-end on this package's own fixtures.
        nuts_options: keywords for the inner ``NUTS`` kernel
            (``target_accept_prob``, ``max_tree_depth``, ``dense_mass``, ...).

    Raises:
        NotImplementedError: for ``chain_method="vectorized"``. Measured on
            numpyro 0.21.0: ``HMC.init`` branches on ``rng_key.ndim`` but
            ``HMCGibbs.init`` does not, so it calls ``random.split`` on the
            batched key MCMC hands it and raises ``ValueError: split accepts a
            single key, but was given a key array of shape (2,) != ()`` --
            before ``gibbs_fn`` is ever called. Refusing it here turns that
            into a sentence, and pins the limitation so an upstream fix
            surfaces as a decision rather than a behaviour change.
        ValueError: if ``names`` covers every latent, leaving the inner NUTS
            with nothing to sample.
    """
    names = tuple(names)
    if chain_method == "vectorized":
        raise NotImplementedError(
            "chain_method='vectorized' is not supported for an HMC-within-Gibbs "
            "sweep on numpyro 0.21.0: HMCGibbs.init splits rng_key "
            "unconditionally, so a batched key raises before gibbs_fn is ever "
            "reached. Use chain_method='sequential' or 'parallel'."
        )
    if not set(graph.latents) - set(names):
        raise ValueError(
            f"block {list(names)} covers every latent of this graph, so the "
            "inner NUTS kernel would have no site to sample and the sweep would "
            "be the Gibbs step alone -- with a frozen-sigma approximation that "
            "nothing then corrects. A whole-graph block's row in the dispatch "
            "table is 'gcr+snis': iid draws reweighted once, with no chain."
        )
    kernel = HMCGibbs(
        NUTS(to_numpyro(graph), **dict(nuts_options or {})),
        gibbs_fn=gibbs_factory(
            graph,
            names,
            tol=tol,
            method=method,
            sigma_rebuild=sigma_rebuild,
            maxiter=maxiter,
            precision=precision,
        ),
        gibbs_sites=list(names),
    )
    return MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
