"""The dense route: a materialised design matrix, and what it buys.

``jax.jacfwd`` forms ``A`` explicitly, where
:mod:`bayesmith.exact.solve` only ever applies it. For a block small enough
to hold as a matrix that gives a posterior covariance in one ``inv`` instead
of one CG per direction -- forecasts, error bars, and the Gaussian a Laplace
approximation samples from.

**It is a different implementation, not an independent one.** Both routes go
through JAX's autodiff, so a bug in how the block's ``forward`` was built
shows up identically in both. The independent reference is
``tests/exact/oracle.py``, which probes ``g`` on a basis of the domain and
differentiates nothing.

``F = J^T N^-1 J`` is the LIKELIHOOD's information, which is a different
quantity from the posterior precision every other exit here targets.
``include_prior=`` chooses, and :attr:`FlatMatrix.kind` records the answer so
one cannot quietly be used as the other.

Ported from ``rheplicant.inference.uncertainty``; ``propagate_covariance`` and
``push_forward`` are P5.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.exact.block import LinearBlock


class FlatMatrix(eqx.Module):
    """A matrix over a block's domain, plus where each latent sits in it.

    Attributes:
        values: the matrix, ``(n, n)`` over the flattened domain.
        names: the latents, in the block's own order.
        spans: ``(start, stop)`` per latent, derived from the actual
            flattening rather than assumed -- a plated latent occupies as many
            rows as it has entries.
        kind: ``"fisher"`` (likelihood information alone),
            ``"posterior_precision"`` (with the prior's curvature) or
            ``"covariance"``. Carried rather than inferred, because the three
            are the same shape and confusing them is silent.
    """

    values: jax.Array
    names: tuple[str, ...] = eqx.field(static=True)
    spans: tuple[tuple[int, int], ...] = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def block(self, name: str) -> jax.Array:
        """The diagonal sub-matrix belonging to one latent.

        Raises:
            KeyError: if ``name`` is not in this matrix.
        """
        for latent, (start, stop) in zip(self.names, self.spans, strict=True):
            if latent == name:
                return self.values[start:stop, start:stop]
        raise KeyError(f"{name!r} is not in this matrix; it covers {list(self.names)}.")

    def std(self) -> dict[str, jax.Array]:
        """``{name: sqrt(diagonal)}``. Meaningful only for ``kind='covariance'``.

        Raises:
            ValueError: for any other kind -- the square root of a precision's
                diagonal is not an error bar, and returning it would be a
                confident wrong number rather than a mistake.
        """
        if self.kind != "covariance":
            raise ValueError(
                f"std() needs a covariance, and this is a {self.kind!r}. The "
                "square root of a precision's diagonal is not an error bar; it "
                "is the error bar of a parameter with every other one held "
                "fixed. Invert it first with parameter_covariance()."
            )
        diagonal = jnp.diagonal(self.values)
        return {
            name: jnp.sqrt(diagonal[start:stop])
            for name, (start, stop) in zip(self.names, self.spans, strict=True)
        }


def _spans(block: LinearBlock) -> tuple[tuple[tuple[int, int], ...], int]:
    spans: list[tuple[int, int]] = []
    start = 0
    for name in block.names:
        size = int(np.prod(block.shape[name], dtype=int))
        spans.append((start, start + size))
        start += size
    return tuple(spans), start


def _unravel(flat: jax.Array, block: LinearBlock, spans) -> dict[str, jax.Array]:
    return {
        name: jnp.reshape(flat[start:stop], block.shape[name])
        for name, (start, stop) in zip(block.names, spans, strict=True)
    }


def _domain_dtype(block: LinearBlock):
    return jnp.result_type(*[block.dtype[name] for name in block.names])


def dense_operator(block: LinearBlock) -> jax.Array:
    """``A`` materialised, ``(n_data, n_parameters)``.

    Rows are the observed nodes concatenated in **sorted name order**, columns
    are the latents in the block's own order -- the same layout
    ``tests/exact/oracle.py`` uses, so the two are comparable element for
    element.
    """
    spans, size = _spans(block)

    def flat_forward(flat: jax.Array) -> jax.Array:
        pushed = block.forward(_unravel(flat, block, spans))
        return jnp.concatenate(
            [jnp.reshape(pushed[name], (-1,)) for name in sorted(pushed)]
        )

    return jax.jacfwd(flat_forward)(jnp.zeros(size, dtype=_domain_dtype(block)))


def _log_sigma_curvature(
    block: LinearBlock,
    sigma_of: Any,
    centre: dict[str, Any],
    spans: Any,
) -> jax.Array:
    """``(dlog sigma/dx)^T (dlog sigma/dx)`` over the observed samples.

    The information carried by the VARIANCE rather than the mean. Flattened
    the same way :func:`dense_operator` flattens ``A`` -- observed nodes in
    sorted name order, latents in the block's own order -- because the two are
    added together and a different row order would be a silent transpose.
    """

    def log_sigma(flat: jax.Array) -> jax.Array:
        sigma = sigma_of(_unravel(flat, block, spans))
        return jnp.concatenate(
            [
                jnp.reshape(jnp.log(jnp.asarray(sigma[name])), (-1,))
                for name in sorted(sigma)
            ]
        )

    flat_centre = jnp.concatenate(
        [jnp.reshape(jnp.asarray(centre[name]), (-1,)) for name in block.names]
    )
    jac = jax.jacfwd(log_sigma)(flat_centre)
    return jac.T @ jac


def fisher_information(
    block: LinearBlock,
    *,
    noise_std: dict[str, Any],
    include_prior: bool = True,
    depends_on_prediction: bool = True,
    sigma_of: Any = None,
    centre: dict[str, Any] | None = None,
) -> FlatMatrix:
    """``J^T N^-1 J``, plus the variance's own information and the priors'.

    **When sigma depends on the prediction, ``J^T N^-1 J`` is not the Fisher
    matrix.** For ``d ~ N(mu(x), Sigma(x))`` the information has a second term
    from the covariance's own parameter dependence::

        F = J^T Sigma^-1 J  +  1/2 tr(Sigma^-1 dSigma Sigma^-1 dSigma)

    which for a diagonal Sigma is ``2 (dlog sigma/dx)^T (dlog sigma/dx)``.
    Under a radiometer it is a clean factor: ``F = (1 + 2 f^2) J^T N^-1 J``.

    Omitting it is the forgiving-looking error, which is why it survives
    review: a factor above 1 dropped from ``F`` makes ``F^-1`` -- the error
    bar -- too WIDE, by ``sqrt(1 + 2 f^2)``. That is 0.25% at ``f = 0.05`` but
    22% at ``f = 0.5`` and 73% at ``f = 1``, and a forecast that is too
    conservative reads as safe.

    Args:
        block: from :func:`bayesmith.exact.linearity.linear_operator`.
        noise_std: ``{observed: sigma}``, from
            :func:`bayesmith.exact.gaussian.noise_std_at`.
        include_prior: add ``S^-1``, making the result the posterior precision
            rather than the likelihood's information. Default ``True``,
            because that is the quantity every other exit in this package
            targets and a forecast that silently answered a different question
            would agree with none of them.
        depends_on_prediction: the node's own claim, and it governs only
            whether ``sigma_of`` is REQUIRED -- not whether the term is added.
            **Check it first** with
            :func:`~bayesmith.exact.gls.check_prediction_dependence`; this
            function cannot, having been handed a decided dict. It defaults
            ``True``, the safe side: a caller who has not thought about it is
            stopped rather than handed a matrix quietly missing a term.
        sigma_of: the ``{name: x} -> {observed: sigma}`` seam, from
            :func:`~bayesmith.exact.gls.sigma_from_graph` -- the same one
            :func:`~bayesmith.exact.gls.iterative_gls` iterates. The decided
            ``noise_std`` cannot supply this: a dict has no derivative.
            Passing it for a genuinely constant sigma is harmless and costs
            one ``jacfwd``, because the term is then exactly ``0.0``.
        centre: the domain point ``noise_std`` was read at, i.e. the point the
            curvature is taken at. Checked against ``sigma_of(centre)`` rather
            than trusted, because the two are redundant by construction and an
            unchecked redundancy is how a covariance ends up weighted at one
            point and curved at another.

    Raises:
        ValueError: if ``depends_on_prediction`` is True and no rule is given.
        ValueError: if ``noise_std`` is not what ``sigma_of(centre)`` produces.
    """
    spans, _ = _spans(block)
    design = dense_operator(block)
    # Same 1/sigma**2 weighting as solve.py::_weights, over a flat
    # concatenation instead of a per-observed dict -- solve.py::_weights is
    # the reference, so a future floor or clamp on this weighting needs to
    # land in both places.
    weight = jnp.concatenate(
        [
            jnp.reshape(1.0 / jnp.asarray(noise_std[name]) ** 2, (-1,))
            for name in sorted(noise_std)
        ]
    )
    values = design.T @ (weight[:, None] * design)
    if depends_on_prediction and (sigma_of is None or centre is None):
        raise ValueError(
            "fisher_information() was told the noise depends on the prediction "
            "but given no rule to differentiate: J^T N^-1 J is then missing the "
            "variance's own information, 2 (dlog sigma/dx)^T (dlog sigma/dx), "
            "and the error bar it implies is too WIDE rather than too narrow, "
            "which reads as safe. Pass sigma_of=sigma_from_graph(graph, at) "
            "and centre= the point noise_std was read at; or, if the sigma "
            "really is constant, pass depends_on_prediction=False. "
            "check_prediction_dependence() settles which, and this function "
            "cannot -- a decided noise_std dict has no derivative."
        )
    if sigma_of is not None and centre is not None:
        implied = sigma_of(centre)
        for name in sorted(noise_std):
            if not np.allclose(
                np.asarray(noise_std[name]), np.asarray(implied[name]), rtol=1e-6
            ):
                raise ValueError(
                    f"noise_std[{name!r}] is not what sigma_of produces at "
                    "centre, so the weighting and the curvature would be taken "
                    "at different points. Pass the centre noise_std was "
                    "actually read at."
                )
        values = values + 2.0 * _log_sigma_curvature(block, sigma_of, centre, spans)
    if include_prior:
        curvature = jnp.concatenate(
            [
                jnp.reshape(1.0 / jnp.asarray(block.prior_std[name]) ** 2, (-1,))
                for name in block.names
            ]
        )
        values = values + jnp.diag(curvature)
    return FlatMatrix(
        values=values,
        names=block.names,
        spans=spans,
        kind="posterior_precision" if include_prior else "fisher",
    )


def condition_ceiling(dtype: Any) -> float:
    """``1 / sqrt(eps)`` -- where an inverse has spent half its digits.

    Inverting a matrix of condition ``kappa`` costs about ``log10(kappa)`` of
    the ``log10(1 / eps)`` decimal digits the arithmetic carries, so this is
    the point at which half are gone. float32: 2.90e+03. float64: 6.71e+07.

    **Read from the dtype in hand rather than hard-wired**, because that is
    the entire content of the defect this gates. ``F = J^T N^-1 J`` SQUARES
    the design's condition number, so an ordinary model lands here easily:
    measured on ``kappa(J) = 1e3``, float32 returns a covariance 2.4% wrong
    while float64 returns one 1.08e-12 wrong. One rule, two answers, because
    the digits available differ -- a fixed ceiling would either wave the
    float32 case through or refuse the float64 case that is fine.
    """
    return 1.0 / math.sqrt(float(jnp.finfo(dtype).eps))


def parameter_covariance(
    fisher: FlatMatrix,
    jitter: float = 0.0,
    *,
    max_condition: float | None | Literal["auto"] = "auto",
) -> FlatMatrix:
    """Invert a precision. ``jitter`` adds ``jitter * I`` first.

    Args:
        fisher: the precision to invert.
        jitter: added to the diagonal before inverting, and before the
            condition is measured -- jitter is the one remedy this function
            offers for ill-conditioning, so measuring the matrix it was
            already applied to is the only reading that does not refuse a
            caller who has fixed the problem.
        max_condition: the ceiling. ``"auto"`` derives it from the values'
            own dtype via :func:`condition_ceiling`; a float sets it
            explicitly; ``None`` removes it, for a caller who has already
            decided a degenerate cell is expected and wants the number
            anyway. The three spellings each mean the obvious thing, which
            is why this does not follow ``iterative_gls``'s ``None``-is-off
            convention alone: both "derive it" and "no ceiling" are wanted
            here and one token cannot say both.

    Raises:
        ValueError: if handed a covariance -- inverting one gives a precision,
            which is a legitimate operation but not what this function's name
            promises, and the ``kind`` field would then be a lie.
        ValueError: if the condition number exceeds the ceiling. This is a
            Cramer-Rao bound; returning one that is silently wrong is worse
            than returning none, and the arithmetic that formed the matrix is
            what decides. **The remedy named in that message is to widen the
            arithmetic around building the GRAPH, not around this inverse**:
            the digits are already gone by the time ``F`` exists, and
            ``jax.enable_x64`` does not widen an array that was traced
            outside it. ``tests/exact/test_fisher.py::
            test_widening_only_the_inverse_does_not_recover_the_bound``
            measures both halves of that.
    """
    if fisher.kind == "covariance":
        raise ValueError(
            "parameter_covariance() was handed a covariance. Inverting it would "
            "give a precision back, which is not what the name says and would "
            "leave kind='covariance' on a matrix that is not one."
        )
    values = fisher.values + jitter * jnp.eye(
        fisher.values.shape[0], dtype=fisher.values.dtype
    )
    ceiling = (
        condition_ceiling(values.dtype) if max_condition == "auto" else max_condition
    )
    if ceiling is not None:
        measured = float(jnp.linalg.cond(values))
        if not measured <= ceiling:  # `not <=` so a NaN condition is refused too
            name = jnp.dtype(values.dtype).name
            raise ValueError(
                f"condition number {measured:.1e} exceeds the {name} ceiling "
                f"{ceiling:.1e}, so inverting this precision spends more than "
                f"half the digits {name} carries and the Cramer-Rao bound it "
                "produces would be wrong without saying so. F = J^T N^-1 J "
                "SQUARES the design's condition number, so this is reached by "
                "ordinary models rather than pathological ones."
                + (
                    " This is float32: build the graph inside `with "
                    "jax.enable_x64(True):` -- around the CONSTRUCTION, not "
                    "around this call, because an array traced outside that "
                    "context stays float32 and widening only the inverse "
                    "recovers nothing (measured: 2.45e-02 relative error "
                    "against 2.41e-02 for doing nothing)."
                    if name == "float32"
                    else " Add `jitter=` if the degeneracy is expected, or "
                    "pass `max_condition=None` to take the number as it is."
                )
            )
    return FlatMatrix(
        values=jnp.linalg.inv(values),
        names=fisher.names,
        spans=fisher.spans,
        kind="covariance",
    )
