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

Ported from ``rheplicant.inference.uncertainty``, including
:func:`propagate_covariance` and :func:`push_forward` -- the two ways to carry
a parameter uncertainty onto a prediction, kept side by side because choosing
between them is the decision a reader arrives with. (An earlier line here said
those two "are P5"; that was a scheduling note that went stale, not a
criterion. They are the migration plan's G7.)
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.errors import StructureError
from bayesmith.exact.block import LinearBlock, is_complex, variance_parts
from bayesmith.exact.gaussian import gaussian_parts, node_shape
from bayesmith.exact.precision import diagonal_from
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.graph import Graph
from bayesmith.graph.nodes import Probabilistic


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
    """``(start, stop)`` per latent over the block's REAL degrees of freedom.

    A complex latent with ``n`` entries occupies ``2n`` rows -- its real half
    first, then its imaginary half, contiguously. That is not a formatting
    choice: every prediction here is real, so the map from complex
    coefficients to data is R-linear and not C-linear, and the matrix a
    caller wants is over ``R^(2n)``. The solver has said so since G9's
    minimal surface (:func:`~bayesmith.exact.block.real_parts`); this is the
    same statement for the DENSE route.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    for name in block.names:
        size = int(np.prod(block.shape[name], dtype=int))
        if is_complex(block.dtype[name]):
            size *= 2
        spans.append((start, start + size))
        start += size
    return tuple(spans), start


def _unravel(flat: jax.Array, block: LinearBlock, spans) -> dict[str, jax.Array]:
    """The inverse of :func:`_spans`' layout, rebuilding complex members.

    ``join(split(x)) is x`` element-wise is the property
    :func:`~bayesmith.exact.block.real_parts` is pinned on, and this is the
    flat-vector spelling of the same map -- so a complex latent's two halves
    are read back in the order they were laid out and nowhere else.
    """
    out: dict[str, jax.Array] = {}
    for name, (start, stop) in zip(block.names, spans, strict=True):
        if is_complex(block.dtype[name]):
            middle = start + (stop - start) // 2
            out[name] = jnp.reshape(
                flat[start:middle] + 1j * flat[middle:stop], block.shape[name]
            )
        else:
            out[name] = jnp.reshape(flat[start:stop], block.shape[name])
    return out


def _domain_dtype(block: LinearBlock):
    """The dtype of the flat REAL domain :func:`_spans` lays out.

    A complex latent contributes its own real part's dtype, because the flat
    vector holds ``(re, im)`` separately: ``jacfwd`` refuses a complex input
    outright (``jacfwd requires real-valued inputs``), and asking it to
    differentiate holomorphically would be asking the wrong question -- the
    objective is not holomorphic.
    """
    return jnp.result_type(
        *[
            jnp.zeros((), block.dtype[name]).real.dtype
            if is_complex(block.dtype[name])
            else block.dtype[name]
            for name in block.names
        ]
    )


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


def _log_spectrum_curvature(
    block: LinearBlock,
    precision_of: Any,
    centre: dict[str, Any],
    spans: Any,
) -> jax.Array:
    """``1/2 sum_k d_a log lambda_k d_b log lambda_k`` over the observed samples.

    The information carried by the COVARIANCE rather than the mean. Written
    on the SPECTRUM rather than on per-sample sigmas, which is what makes it
    cover a correlated node:

        ``1/2 tr(N^-1 d_a N N^-1 d_b N) = 1/2 sum_k d_a log lam_k d_b log lam_k``

    holds whenever ``N``'s eigenbasis does not move with the parameters, and
    both rows the gate accepts have a fixed basis -- ``I`` for a ``Normal``,
    the DFT for a ``CirculantNormal``. Derived symbolically in
    ``docs/derivations/variance_information_spectral.wls``; measured against a
    dense finite-difference Fisher matrix in
    ``docs/probes/probe_9_correlated_variance_information.py``, agreeing to
    the difference floor (~1e-10) on a circulant whose kernel changes SHAPE
    with the parameters.

    **This is not a new rule for the diagonal case, it is the same one.** The
    jacobian is taken of ``1/2 log lambda_k``, and for a ``DiagonalPrecision``
    ``lambda_i = sigma_i**2``, so that is ``log sigma_i`` -- bitwise, because
    ``log_spectrum`` returns ``2 log sigma`` and halving it is exact. The
    caller still multiplies by ``2.0``, so the diagonal answer is the number
    it always was.

    What the old per-sample form got WRONG on a correlated node is worth
    stating, because it is not a small error. ``sqrt(diag N)`` is CONSTANT
    across samples for a stationary covariance, so a kernel whose shape moves
    while its diagonal does not registers as no information at all. Measured
    on such a fixture: the shape parameter's entry is exactly ``0.0`` against
    a true ``3.44`` -- not inaccurate, blind.

    Flattened the same way :func:`dense_operator` flattens ``A`` -- observed
    nodes in sorted name order, latents in the block's own order -- because
    the two are added together and a different row order would be a silent
    transpose.
    """

    def half_log_spectrum(flat: jax.Array) -> jax.Array:
        precision = precision_of(_unravel(flat, block, spans))
        return jnp.concatenate(
            [
                jnp.reshape(0.5 * precision[name].log_spectrum(), (-1,))
                for name in sorted(precision)
            ]
        )

    flat_centre = jnp.concatenate(
        [jnp.reshape(jnp.asarray(centre[name]), (-1,)) for name in block.names]
    )
    jac = jax.jacfwd(half_log_spectrum)(flat_centre)
    return jac.T @ jac


def _weighted_design(
    block: LinearBlock, design: jax.Array, precision: dict[str, Any]
) -> jax.Array:
    """``N^-1 A``, applied column by column through the noise's own operator.

    This used to build ``1 / sigma**2`` inline, with a comment conceding that
    ``solve.py::_weights`` was "the reference" and that a future change to the
    weighting "needs to land in both places". Two places is how one of them
    goes stale, and the operator is now an interface rather than an array, so
    the second copy can simply go: the same
    :class:`~bayesmith.exact.precision.Precision` the solver uses is what
    weights the design here.

    Rows are split per observed node because a precision belongs to ONE node
    -- a correlated covariance couples samples within a node, never across
    two independent observations -- and the sizes come from ``block.data``,
    which :func:`dense_operator` also flattens in ``sorted`` order, so the two
    cannot disagree about the layout.

    ``vmap`` over columns rather than one call on the matrix: ``apply`` is
    written for a residual, and a diagonal implementation broadcasting
    ``(n,)`` against ``(n, k)`` would be right by accident while a circulant
    one -- which FFTs along the last axis -- would be wrong.

    Each column is reshaped to the NODE's own shape before ``apply`` and
    flattened again after, because a residual is node-shaped and a design
    column is not: ``dense_operator`` flattens the prediction, while the
    ``Precision`` read off the graph keeps ``node_shape`` -- ``(8, 8)`` for a
    waterfall, against a ``(64,)`` column. Handing the flat column straight to
    ``apply`` broadcast-failed on every observed node with more than one axis,
    through ``linear_operator`` and all -- measured before this reshape
    existed, so the pairing is not hypothetical.
    """
    pieces, start = [], 0
    for name in sorted(precision):
        shape = jnp.shape(block.data[name])
        size = int(np.prod(shape, dtype=int))
        rows = jax.lax.dynamic_slice(design, (start, 0), (size, design.shape[1]))

        def weighted(column, name=name, shape=shape):
            return jnp.reshape(precision[name].apply(jnp.reshape(column, shape)), (-1,))

        pieces.append(jax.vmap(weighted, in_axes=1, out_axes=1)(rows))
        start += size
    return jnp.concatenate(pieces, axis=0)


def fisher_information(
    block: LinearBlock,
    *,
    precision: dict[str, Any],
    include_prior: bool = True,
    depends_on_prediction: bool = True,
    sigma_of: Any = None,
    precision_of: Any = None,
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
        precision: ``{observed: N^-1}``, from
            :func:`bayesmith.exact.gaussian.precision_at`, or
            :func:`~bayesmith.exact.precision.diagonal_from` applied to
            :func:`bayesmith.exact.gaussian.noise_std_at`. **This function
            reads the noise in BOTH vocabularies and they are not the same
            argument**: the operator weights the design, while ``sigma_of``
            supplies the per-sample VALUES whose log-derivative is the
            variance's own information. A dict of ``Precision`` objects has no
            per-sample sigma to differentiate, and a dict of sigmas is not an
            operator a correlated covariance can satisfy -- which is why B9's
            step 4 splits this seam rather than renaming it.
        include_prior: add ``S^-1``, making the result the posterior precision
            rather than the likelihood's information. Default ``True``,
            because that is the quantity every other exit in this package
            targets and a forecast that silently answered a different question
            would agree with none of them.
        depends_on_prediction: the node's own claim, and it governs only
            whether a RULE is REQUIRED -- not whether the term is added.
            **Check it first** with
            :func:`~bayesmith.exact.gls.check_prediction_dependence`; this
            function cannot, having been handed a decided dict. It defaults
            ``True``, the safe side: a caller who has not thought about it is
            stopped rather than handed a matrix quietly missing a term.
        sigma_of: the ``{name: x} -> {observed: sigma}`` seam, from
            :func:`~bayesmith.exact.gls.sigma_from_graph` -- the same one
            :func:`~bayesmith.exact.gls.iterative_gls` iterates. The DIAGONAL
            spelling of the rule, kept because that is what the reweighting
            loop produces; it is wrapped in
            :func:`~bayesmith.exact.precision.diagonal_from` and takes the
            same path as ``precision_of``. The decided ``precision`` cannot
            supply either: an operator has no derivative.
        precision_of: the general form of the same rule,
            ``{name: x} -> {observed: Precision}`` -- from
            :func:`~bayesmith.exact.gaussian.precision_at` curried on the
            graph. **Required for a correlated node**, which has no
            per-sample sigma for ``sigma_of`` to return. Give one or the
            other, not both. Passing a rule for a genuinely constant
            covariance is harmless and costs one ``jacfwd``, because the term
            is then exactly ``0.0``.
        centre: the domain point ``precision`` was read at, i.e. the point the
            curvature is taken at. Checked against the rule at ``centre``
            rather than trusted, because the two are redundant by
            construction and an unchecked redundancy is how a covariance ends
            up weighted at one point and curved at another. The check
            compares the two as OPERATORS -- both applied to one fixed probe
            -- rather than comparing sigma arrays, because ``precision`` need
            not have sigma arrays to compare.

    Raises:
        ValueError: if ``depends_on_prediction`` is True and no rule is given.
        ValueError: if both ``sigma_of`` and ``precision_of`` are given.
        ValueError: if ``precision`` is not the operator the rule implies at
            ``centre``.
    """
    if sigma_of is not None and precision_of is not None:
        raise ValueError(
            "fisher_information() was given both sigma_of= and precision_of=, "
            "which are two spellings of one rule -- the diagonal one and the "
            "general one. Two rules is two chances to describe a different "
            "covariance than the one `precision` weights by, which is the "
            "redundancy the centre check exists to catch. Pass whichever "
            "matches how the noise is produced."
        )
    if sigma_of is not None:
        # One curvature implementation, not two: the diagonal rule is wrapped
        # rather than special-cased, so the degenerate case keeps going
        # through the same code the correlated one does.
        def rule(x):
            return diagonal_from(sigma_of(x))
    else:
        rule = precision_of

    spans, _ = _spans(block)
    design = dense_operator(block)
    values = design.T @ _weighted_design(block, design, precision)
    if depends_on_prediction and (rule is None or centre is None):
        raise ValueError(
            "fisher_information() was told the noise depends on the prediction "
            "but given no rule to differentiate: J^T N^-1 J is then missing the "
            "variance's own information, 2 (dlog sigma/dx)^T (dlog sigma/dx), "
            "and the error bar it implies is too WIDE rather than too narrow, "
            "which reads as safe. Pass sigma_of=sigma_from_graph(graph, at) "
            "and centre= the point the precision was read at (or precision_of= "
            "for a correlated node); or, if the sigma "
            "really is constant, pass depends_on_prediction=False. "
            "check_prediction_dependence() settles which, and this function "
            "cannot -- a decided precision has no derivative."
        )
    if rule is not None and centre is not None:
        implied = rule(centre)
        for name in sorted(precision):
            # Compared as OPERATORS, on one fixed probe, rather than as sigma
            # arrays: `precision` is an interface and a correlated
            # implementation has no per-sample sigma to line up. For a
            # diagonal pair this is not weaker than the elementwise
            # comparison it replaces -- `apply` divides the probe by sigma**2
            # elementwise, so the same n numbers are still being compared, and
            # a probe drawn rather than constant is what keeps that true for
            # an implementation that mixes samples.
            probe = jax.random.normal(
                jax.random.key(0),
                jnp.shape(block.data[name]),
                jnp.asarray(block.data[name]).dtype,
            )
            if not np.allclose(
                np.asarray(precision[name].apply(probe)),
                np.asarray(implied[name].apply(probe)),
                rtol=1e-6,
            ):
                raise ValueError(
                    f"precision[{name!r}] is not the operator sigma_of implies "
                    "at centre, so the weighting and the curvature would be "
                    "taken at different points. Pass the centre the precision "
                    "was actually read at. (A CORRELATED precision reaches "
                    "this too: sigma_of produces per-sample sigmas, whose "
                    "diagonal operator no off-diagonal covariance matches. "
                    "The correlated form of the variance-information term is "
                    "not derived yet, and precision_parts refuses that "
                    "combination upstream.)"
                )
        values = values + 2.0 * _log_spectrum_curvature(block, rule, centre, spans)
    if include_prior:
        # From `variance_parts`, which is the ONE spelling of "the prior's
        # curvature per real degree of freedom" -- the same one
        # `normal_operator` reads, so the dense route and the iterative one
        # cannot disagree about it. A complex member's variance is duplicated
        # across its two halves there, which is what `prior_std` means for
        # one, and it is why this cannot be read off `prior_std` directly.
        #
        # Broadcast per span rather than concatenated as-is. The previous
        # spelling reshaped `prior_std` to `(-1,)` and called `jnp.diag` on
        # it, so a SCALAR `prior_std` against a vector latent produced a 1x1
        # matrix -- which then broadcast into the whole `values` array and
        # added the curvature to every OFF-DIAGONAL entry as well. Measured
        # on a hand-built (3,) block at prior_std=2.0: every off-diagonal was
        # 0.25 too large, and the diagonal was right, so the matrix stayed
        # symmetric, finite and plausible. The graph route never reached it
        # (numpyro broadcasts a Normal's scale to the batch shape, so
        # `linear_operator` hands over a full-shaped array), which is why it
        # survived a release; a complex block reaches it always, because
        # there the real degrees of freedom outnumber the declared entries
        # two to one.
        parts = variance_parts(block)
        pieces = []
        for name, (start, stop) in zip(block.names, spans, strict=True):
            leaves = jax.tree.leaves(parts[name])
            width = (stop - start) // len(leaves)
            for leaf in leaves:
                pieces.append(
                    jnp.broadcast_to(
                        jnp.reshape(1.0 / jnp.asarray(leaf), (-1,)), (width,)
                    )
                )
        values = values + jnp.diag(jnp.concatenate(pieces))
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


def _flat_domain(
    graph: Graph, matrix: FlatMatrix, at: dict[str, Any], caller: str
) -> tuple[jax.Array, tuple[tuple[int, int], ...]]:
    """``at``'s covered latents raveled in ``matrix``'s OWN order, checked.

    The order is the matrix's, never the graph's or the dict's, because the
    matrix is the thing whose rows have to line up. Its ``names`` and ``spans``
    say what it was built over; this refuses anything that does not match, by
    SHAPE and not only by name.

    rheplicant's ancestor could only compare pytree structures, and its own
    docstring records what that missed: for a dict-keyed space a treedef
    encodes the KEY NAMES alone, so two spaces with the same latent names and
    different per-latent shapes pass the check and produce finite, wrong error
    bars. A :class:`FlatMatrix` carries the spans, so here the sizes are
    checked as well and that gap does not port.
    """
    for name in matrix.names:
        if name not in graph.latents:
            raise StructureError(
                f"{caller} was given a {matrix.kind} over {list(matrix.names)}, "
                f"and {name!r} is not a latent of this graph; its latents are "
                f"{list(graph.latents)}. The Jacobian's columns would be taken "
                "against a different set of parameters from the ones the "
                "matrix describes -- finite, correctly shaped, and wrong."
            )
        if name not in at:
            raise StructureError(
                f"{caller} needs a value for {name!r}: it is one of the "
                f"{matrix.kind}'s own parameters. Given "
                f"{sorted(at)}."
            )
    parts = []
    for name, (start, stop) in zip(matrix.names, matrix.spans, strict=True):
        flat = jnp.reshape(jnp.asarray(at[name]), (-1,))
        if flat.size != stop - start:
            raise StructureError(
                f"{caller}: the {matrix.kind} gives {name!r} {stop - start} "
                f"entries but `at` has {flat.size} "
                f"({jnp.shape(at[name])}). Two spaces with the same latent "
                "NAMES and different per-latent shapes are exactly the case a "
                "name check passes and the numbers come back finite and wrong."
            )
        parts.append(flat)
    vector = jnp.concatenate(parts) if parts else jnp.zeros((0,))
    size = matrix.spans[-1][1] if matrix.spans else 0
    if jnp.shape(matrix.values) != (size, size):
        raise StructureError(
            f"{caller}: the {matrix.kind} is {jnp.shape(matrix.values)} but its "
            f"spans cover {size} entries. The matrix and its own index do not "
            "agree, so nothing downstream can be trusted to line up."
        )
    return vector, matrix.spans


def _node_of(graph: Graph, node: str | None, caller: str) -> str:
    """The node to propagate onto, or a refusal that names the candidates.

    Defaulted only when the graph leaves no choice -- one observed node, whose
    PREDICTION is what a caller means by "the error bar". Guessing among
    several would pick one silently.
    """
    if node is not None:
        if node not in graph.names:
            raise StructureError(
                f"{caller} was asked for node {node!r}, which this graph does "
                f"not declare. Its nodes are {list(graph.names)}."
            )
        target = graph.node(node)
        if isinstance(target, Probabilistic) and target.is_latent:
            raise StructureError(
                f"{caller} was asked for latent {node!r}. A latent's own "
                "uncertainty is what the covariance already reports -- "
                "`FlatMatrix.block(name)` is that sub-matrix -- and pushing it "
                "through the identity would return the same numbers by a much "
                "longer route while reading as a prediction. Name a "
                "deterministic node, or the observed node whose prediction is "
                "wanted."
            )
        return node
    if len(graph.observed) == 1:
        return graph.observed[0]
    raise StructureError(
        f"{caller} needs `node=`: this graph has {len(graph.observed)} observed "
        f"nodes ({list(graph.observed)}), so there is no single prediction to "
        "default to. Name the deterministic node whose uncertainty is wanted -- "
        f"the graph's nodes are {list(graph.names)}."
    )


def _predicted(graph: Graph, target: str, values: dict[str, Any]) -> jax.Array:
    """The node's value, except that an OBSERVED node gives its ``loc``.

    The distinction is the whole of both functions' correctness and it is not
    a nicety. ``evaluate`` gives an observed node its DATA -- that is what
    conditioning means -- and the data does not depend on the parameters at
    all. So propagating the value would report a Jacobian of exactly zero and
    an error bar of exactly zero, on every entry, for any model: finite,
    correctly shaped, and the most confident possible wrong answer. Measured
    while writing this: the delta method returned ``0.0`` everywhere for the
    single-observed-node default, and the test comparing the default against
    the deterministic node passed, because it was comparing two zeros.

    An observed node's ``loc`` IS the prediction, so that is what is carried.
    """
    env = evaluate(graph, values)
    node = graph.node(target)
    if isinstance(node, Probabilistic) and not node.is_latent:
        loc, _ = gaussian_parts(graph, node, env)
        return jnp.broadcast_to(loc, node_shape(graph, node, env))
    return env[target]


def propagate_covariance(
    graph: Graph,
    covariance: FlatMatrix,
    at: dict[str, Any],
    *,
    node: str | None = None,
) -> jax.Array:
    """Delta-method prediction uncertainty: ``sqrt(diag(J Sigma J^T))``.

    The linearised answer, and the cheap one: one ``jacfwd`` of the node's
    value with respect to the covered latents, at ``at``. Its Monte-Carlo
    counterpart is :func:`push_forward`, and choosing between them is a
    modelling decision rather than a performance one -- see that function.

    Args:
        graph: the model. The prediction, the parameters and their point all
            come from it, so there is no second statement of any of them to
            disagree with the first.
        covariance: from :func:`parameter_covariance`. Its ``kind`` must be
            ``"covariance"``: a Fisher matrix and a covariance are the same
            SHAPE, and putting one where the other belongs gives an error bar
            that is wrong by the square of everything.
        at: where to linearise -- every latent the graph declares, since the
            node's value generally depends on latents outside the covariance
            too. Only the covered ones enter the Jacobian; the rest are held.
        node: whose value to propagate -- and for an OBSERVED node that is its
            ``loc``, the prediction, because ``evaluate`` gives such a node its
            data and the data does not depend on the parameters at all. See
            :func:`_predicted`. Defaults to the single observed node when the
            graph has exactly one, and is refused otherwise. A latent is
            refused: its own uncertainty is what the covariance already holds.

    Returns:
        Per-entry standard deviation, shaped like the node's value.

    Raises:
        StructureError: for a matrix that is not a covariance, a covered name
            the graph does not declare, a per-latent shape that disagrees with
            the matrix's own spans, or an ambiguous ``node``.
    """
    if covariance.kind != "covariance":
        raise StructureError(
            f"propagate_covariance was given a {covariance.kind!r}, not a "
            "covariance. The delta method wants Sigma, and a precision is its "
            "INVERSE -- the same shape, so nothing downstream would notice: "
            "the error bar comes back finite and wrong by the square of "
            "everything. Pass parameter_covariance(fisher) rather than the "
            "fisher."
        )
    target = _node_of(graph, node, "propagate_covariance")
    vector, spans = _flat_domain(graph, covariance, at, "propagate_covariance")

    def value_of(flat: jax.Array) -> jax.Array:
        moved = dict(at)
        for name, (start, stop) in zip(covariance.names, spans, strict=True):
            moved[name] = jnp.reshape(flat[start:stop], jnp.shape(at[name]))
        return jnp.reshape(_predicted(graph, target, moved), (-1,))

    jacobian = jax.jacfwd(value_of)(vector)
    variance = jnp.einsum("ip,pq,iq->i", jacobian, covariance.values, jacobian)
    shape = jnp.shape(_predicted(graph, target, at))
    return jnp.reshape(jnp.sqrt(variance), shape)


def push_forward(
    graph: Graph,
    samples: Mapping[str, Any],
    *,
    node: str | None = None,
) -> jax.Array:
    """Monte-Carlo pushforward: a node's value over a stack of parameter draws.

    The counterpart of :func:`propagate_covariance`, and the difference is a
    modelling one rather than a matter of cost. The delta method reports the
    spread of a LINEARISATION of the node about one point; this reports the
    spread of the node itself. They agree exactly when the map is affine over
    the posterior's width and diverge when it is not -- a log-amplitude
    latent, an exponent, a beam width -- where the linear answer is symmetric
    about the point and the true one is not.

    Noiseless, which is what makes it different from
    :func:`~bayesmith.bridge.numpyro_bridge.predict`: that one runs NumPyro's
    ``Predictive`` and therefore DRAWS the observation noise, giving the
    posterior predictive. This propagates the parameters alone, so the spread
    it reports is the model's uncertainty about its own prediction with no
    measurement error added. Summarising the wrong one of those two as "the
    error bar" is an ordinary and invisible mistake, which is why both exist
    under names that say which they are.

    Args:
        graph: the model.
        samples: ``{latent: (n_draws, *latent shape)}`` -- what
            :func:`~bayesmith.bridge.numpyro_bridge.nuts` returns, unchanged.
        node: whose value to push, with an observed node again giving its
            ``loc``. Defaults as :func:`propagate_covariance`'s.

    Returns:
        ``(n_draws, *node shape)``. Summarise with ``mean(0)`` / ``std(0)`` /
        quantiles.

    Raises:
        StructureError: if a latent is missing from ``samples``, if a stack's
            per-draw shape is not the latent's own, or if the stacks disagree
            about how many draws there are. The leading axis IS the draw axis,
            and a transposed square stack broadcasts into a finite, correctly
            shaped, wrong answer -- the same guard, and the same measurement,
            as ``predict``'s.
    """
    # Function scope, deliberately: `exact` does not import `dispatch` at
    # module scope (see `exact/gibbs.py`'s note on the layering, and
    # `tests/test_layering.py`, which asserts it). Hoisting this would close a
    # cycle rather than open one -- `classify` reads `exact.gaussian`, which
    # imports `graph.evaluate` at module scope.
    from bayesmith.dispatch.classify import prior_environment

    target = _node_of(graph, node, "push_forward")
    declared = prior_environment(graph)
    draws: set[int] = set()
    for name in graph.latents:
        if name not in samples:
            raise StructureError(
                f"push_forward is missing latent {name!r}; it has "
                f"{sorted(samples)}. Every latent the graph declares has to be "
                "there, because the deterministic nodes read them."
            )
        expected = jnp.shape(declared[name])
        got = jnp.shape(samples[name])
        if got[1:] != expected:
            raise StructureError(
                f"push_forward: samples[{name!r}] has per-draw shape {got[1:]} "
                f"but the latent is {expected} (its full stack is {got}). The "
                "LEADING axis must be the draw axis; a transposed square stack "
                "broadcasts into a finite, correctly shaped, wrong answer."
            )
        draws.add(got[0])
    if len(draws) > 1:
        raise StructureError(
            f"push_forward: the sample stacks disagree about the number of "
            f"draws: {sorted(draws)}. They must all come from one run."
        )

    def one(values: dict[str, Any]) -> jax.Array:
        return _predicted(graph, target, values)

    return jax.vmap(one)({name: samples[name] for name in graph.latents})


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
