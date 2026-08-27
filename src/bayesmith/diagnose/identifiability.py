"""Cross-block identifiability: the rank test a per-block guard cannot perform.

Every convergence guard in :mod:`bayesmith.exact` is computed **from one
block**. That is not an oversight, it is arithmetic: a residual ``|M x - b|``
and a condition bound ``kappa(A^T N^-1 A + S^-1)`` are both properties of the
operator for the block being solved, so neither can see a degeneracy whose
two halves live in *different* blocks. The linearity check cannot see it
either, and it is right not to: each conditional of a bilinear model
genuinely is affine.

The failure that follows is silent and large, and it is the measured case
this module was built against (ported from
``rheplicant.inference.identifiability``, re-measured on this package's own
graph vocabulary): an alternating solve over ``gain x T_ant`` with a free
antenna temperature per (time, frequency) cell reports a clean per-block
kappa and residual while sitting thousands of kelvin from the truth. Nothing
in the sweep is wrong; the *partition* is, and no per-block number is
entitled to say so.

What can say so is the rank of the Jacobian of the prediction with respect
to **all** the selected latents at once. Re-measured on the graph port of
that fixture::

    free-per-cell T_ant,  tone ON  (5000 K)   n_par=72 rank=64 nullity=8
    free-per-cell T_ant,  tone OFF            n_par=72 rank=64 nullity=8
    (3,3)-basis  T_coeff, tone ON  (5000 K)   n_par=17 rank=17 nullity=0
    (3,3)-basis  T_coeff, tone OFF            n_par=17 rank=16 nullity=1

Read that as: a known calibration tone buys **exactly nothing** against a
free-per-cell antenna temperature -- the free cell at the tone's channel
absorbs the gain sample by sample, so the nullity stays at ``n_time`` either
way -- and **everything** against a frequency-smooth one, where a delta at
one channel is not in the span of three smooth basis functions and cannot be
reabsorbed.

Three things about the method are not decoration.

**The Jacobian's columns are normalised.** A latent whose natural scale is
1e3 and one whose scale is 1e-3 produce columns differing by 1e6 in norm, and
a rank verdict taken on those reports the choice of units rather than the
identifiability of the model. Column normalisation measures each parameter in
units of its own effect on the prediction, which is the only scale-free
question there is to ask.

**It refuses single precision rather than forcing double.** rheplicant runs
this diagnostic inside a process-global x64 switch of its own; this package's
rule is that ``src/`` never touches ``jax.config``, so the caller opens
``with jax.enable_x64(True):`` -- graph construction included -- and a
float32 Jacobian is refused by name (see
:func:`~bayesmith.diagnose.local.refuse_single_precision`). The refusal is
load-bearing, not ceremony, and the boundary was re-measured under THIS
regime rather than ported: in float64 the motivating model's null direction
sits at **7.479e-17** of the largest singular value; computed in float32 the
same direction surfaces at **3.117e-8** -- above the default tolerance -- and
the degenerate model is reported as fully identified.

**The result is named.** An anonymous index into a flattened vector tells a
user they have a problem and nothing about which;
:meth:`IdentifiabilityReport.direction` hands back ``{"gain": ..., "t_ant":
...}``, shaped like the latents, so "the degenerate direction is *this*
combination" is something you can read and act on.

One thing about it is a limit rather than a feature. **Cost:** a dense
Jacobian and a dense SVD, ``O(n_data * n_par * min(...))`` time and
``n_data * n_par`` float64 words of memory. This is a design-time diagnostic
for tens to a few thousand parameters -- the size a Gibbs partition is
*chosen* at -- not something to run inside a sweep over a 10^6-coefficient
block. For one block's conditioning without forming anything, use
:func:`bayesmith.exact.solve.condition_bound`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.diagnose.local import (
    check_differentiable,
    check_observed_have_locs,
    flat_view,
    latent_values,
    local_block,
    refuse_ambient_float32,
    refuse_single_precision,
    resolve_names,
)
from bayesmith.errors import StructureError
from bayesmith.exact.fisher import dense_operator
from bayesmith.graph.graph import Graph

#: Singular values at or below ``rtol * s_max`` are called null.
#:
#: Justified against the spectrum of the motivating model, RE-MEASURED on this
#: package's graph port under the caller-opens-x64 regime rather than carried
#: over from rheplicant -- the two packages put the dtype boundary in
#: different places (a process-global switch there, a context manager plus a
#: refusal here), and a constant justified under one regime is not
#: automatically justified under the other. Measured here, basis model:
#:
#: * its null direction sits at **7.479266e-17** of the largest singular
#:   value (rheplicant's own arithmetic puts the same direction at 6.6e-17;
#:   the spectrum moved with the evaluation order, the verdict did not);
#: * its weakest genuinely IDENTIFIED direction at **4.822138e-05** -- the
#:   same value to every printed digit, because that direction is resolved
#:   far above either package's roundoff;
#: * the SVD's own noise floor is a few times ``n * eps`` ~ **1e-14**, below
#:   which no tolerance is measuring anything but roundoff.
#:
#: Every tolerance in (1e-13, 4.8e-5) therefore returns the same verdict --
#: an 8.7-decade window whose geometric centre is 2e-9. 1e-8 sits within a
#: decade of that centre, and it is also ``sqrt(eps)`` in float64: the scale
#: at which a direction stops being constrained by the data and starts being
#: constrained by the arithmetic.
#:
#: The upper end is model-dependent and the lower end is not: a genuinely
#: weakly-constrained direction can sit at 1e-6, and then this default is only
#: two decades from mis-classifying it. Read
#: :attr:`IdentifiabilityReport.weakest_identified` before trusting the
#: verdict, and pass ``rtol=`` when it comes back close to the cut.
#:
#: **The test suite pins this value into 2.4 decades, not 8.7.** Two
#: counterfactuals are stated against this default rather than against a
#: literal, and each end is a measured number a real claim turns on:
#:
#: * **lower, 1.0e-10** -- the mixed-scale fixture's RAW spectrum ratio
#:   (measured here: 1.000000e-10, the fixture's 1e10 scale gap exactly).
#:   The no-normalisation counterfactual only demonstrates anything while
#:   that ratio sits *below* this default.
#: * **upper, 3.116759e-8** -- where the basis model's null direction
#:   surfaces in float32. A default *above* it would call that direction
#:   null, single precision would get the verdict right, and the refusal of
#:   float32 would have nothing to protect. The margin is 3.1x, not a decade.
#:
#: ``test_the_suite_pins_this_constant_more_tightly_than_the_physics`` states
#: that window in one place; read it first if a retune starts failing tests.
#:
#: **This constant is float64's, and there is no float32 counterpart to write
#: -- measured, not assumed.** The migration's D9 proposed deriving the cut
#: from the ambient dtype, on the reasonable ground that the fisher ceiling is
#: already derived that way. It does not carry over. A ceiling on a CONDITION
#: NUMBER is a statement about how many digits the arithmetic has; a rank cut
#: is a statement about where a spectrum stops describing the model, and in
#: float32 the spectrum stops describing it well ABOVE any cut one might pick.
#: ``docs/probes/probe_13_d9_precision_policy.py`` sweeps a family over ten
#: decades of true conditioning: float64's smallest singular value follows it,
#: float32's sits at ~1e-7 and wanders non-monotonically, and no candidate cut
#: reproduces every float64 verdict. So the diagnose family REFUSES float32
#: rather than retuning for it -- the same conclusion rheplicant reached
#: independently and records in its own ``identifiability`` module.
DEFAULT_RANK_RTOL: float = 1e-8


@dataclasses.dataclass(frozen=True)
class IdentifiabilityReport:
    """What the joint Jacobian's rank says about a set of latents.

    Deliberately a plain frozen dataclass rather than an ``eqx.Module``, for
    the same reason :class:`~bayesmith.exact.block.LinearBlock` is: this is a
    derived linear-algebra verdict, not a differentiable model. It holds
    **numpy** arrays, not JAX ones -- a float64 JAX array that escapes the
    caller's x64 context truncates, with a warning, the moment a
    default-precision caller touches it, throwing away exactly the precision
    the diagnostic exists to obtain. And ``rank``/``nullity`` are Python
    ints, which is why this function cannot be jitted: a rank is a decision,
    and a traced decision is one you cannot branch on.

    Attributes:
        names: the latents analysed, in the order the caller asked for.
        shapes: their shapes, in the same order.
        spans: ``(start, stop)`` of each latent within the flat parameter
            vector, in the same order.
        n_par: total number of real parameters.
        n_data: size of the flattened prediction over every observed node.
            ``nullity`` can never be below ``n_par - n_data``: more
            parameters than data points is a null space by counting alone.
        rank: number of singular values of the COLUMN-NORMALISED Jacobian
            strictly above :attr:`threshold`.
        nullity: ``n_par - rank`` -- the dimension of the space of parameter
            perturbations the prediction is blind to.
        singular_values: ``(n_par,)`` descending. When ``n_data < n_par`` the
            SVD returns only ``n_data`` values and the rest are exact zeros;
            they are included rather than dropped, so ``rank`` is always the
            count of entries above the threshold and never needs a caveat.
        null_space: ``(nullity, n_par)`` orthonormal rows, in the
            column-normalised coordinates the rank verdict is taken in. Use
            :meth:`direction` for raw latent coordinates.
        jacobian: ``(n_data, n_par)`` column-normalised, as analysed. Rows
            are the observed nodes in sorted name order, each flattened --
            :func:`~bayesmith.exact.fisher.dense_operator`'s own layout.
        column_norms: the SAFE norms the normalisation divided by -- the real
            norm where a column is live, 1.0 where it is exactly zero.
        rtol: the relative tolerance used.
        threshold: ``rtol * singular_values[0]``, the absolute cutoff.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    spans: tuple[tuple[int, int], ...]
    n_par: int
    n_data: int
    rank: int
    nullity: int
    singular_values: np.ndarray
    null_space: np.ndarray
    jacobian: np.ndarray
    column_norms: np.ndarray
    rtol: float
    threshold: float

    @property
    def weakest_identified(self) -> float:
        """``s[rank-1] / s[0]`` -- how well the worst identified direction is seen.

        The headline number: how much less the data says about the direction
        it constrains least than about the one it constrains most. ``0.0``
        when nothing at all is identified, which is the only case where the
        ratio has no meaning.
        """
        if self.rank == 0 or float(self.singular_values[0]) == 0.0:
            return 0.0
        return float(self.singular_values[self.rank - 1] / self.singular_values[0])

    def _row(self, index: int) -> np.ndarray:
        # jnp/np arrays index out of range by CLAMPING rather than raising, so
        # direction(5) on a 1-dimensional null space would silently hand back
        # direction 0 again -- a wrong answer with no symptom.
        if not 0 <= index < self.nullity:
            raise StructureError(
                f"there is no null direction {index}: this model has nullity "
                f"{self.nullity}, so the valid indices are "
                f"{list(range(self.nullity)) or 'none -- it is fully identified'}."
            )
        # `nullity` and `null_space` are two records of one fact, and the
        # bounds check above trusts the first while the lookup below uses the
        # second. An SVD taken with full_matrices=False truncates the second
        # whenever there are fewer data points than parameters -- the
        # motivating case -- and the two then disagree silently: this lookup
        # runs off the end and numpy raises a bare IndexError naming neither
        # cause. Name it.
        if self.null_space.shape != (self.nullity, self.n_par):
            raise StructureError(
                f"inconsistent report: nullity is {self.nullity} over "
                f"{self.n_par} parameters, so null_space should have shape "
                f"{(self.nullity, self.n_par)}, but it has "
                f"{self.null_space.shape}. The SVD behind it must be taken "
                "with full_matrices=True whenever n_data < n_par, or the null "
                "space is truncated there."
            )
        return self.null_space[index]

    def direction(self, index: int) -> dict[str, np.ndarray]:
        """One null direction in RAW latent coordinates, split by name.

        Add a small multiple of this to the latents and the prediction does
        not move, to first order -- that is the whole content of the report,
        and it is the form a caller acts in. Note that it is NOT the raw SVD
        row: the SVD is taken of the column-normalised Jacobian, so a null
        vector there has to be divided by the column norms again to become a
        perturbation of the parameters themselves. Returned with unit 2-norm
        over the flat vector; the scale is arbitrary, only the direction
        means anything.

        For per-latent weights comparable across quantities in different
        units, use :meth:`participation` instead -- in raw kelvin and
        dimensionless gain, a null direction's two halves are not comparable
        numbers.
        """
        raw = self._row(index) / self.column_norms
        norm = float(np.linalg.norm(raw))
        # `norm` cannot reach 0 while `column_norms` holds the SAFE norms:
        # `raw` is then an orthonormal row divided by finite positive numbers.
        # The fallback is a floor under that invariant, not live code. It
        # becomes reachable only if `column_norms` is ever changed to store
        # the raw norms, and then it makes things worse rather than better:
        # `norm` is NaN, `NaN > 0.0` is False, and dividing by 1.0 preserves
        # the NaN.
        raw = raw / (norm if norm > 0.0 else 1.0)
        return {
            name: raw[start:stop].reshape(shape)
            for name, shape, (start, stop) in zip(
                self.names, self.shapes, self.spans, strict=True
            )
        }

    def participation(self, index: int) -> dict[str, float]:
        """Fraction of a null direction carried by each latent, summing to 1.

        Measured in the COLUMN-NORMALISED coordinates, not raw ones: a 3000 K
        antenna temperature and a gain near 1 cannot be compared in their own
        units, and a raw-unit share would report which quantity is
        numerically larger rather than which one the degeneracy involves. In
        normalised coordinates the bilinear ``gain x T_ant`` degeneracy comes
        out at 0.50/0.50, which is the true statement about it.
        """
        row = self._row(index)
        return {
            name: float(np.sum(row[start:stop] ** 2))
            for name, (start, stop) in zip(self.names, self.spans, strict=True)
        }


def identifiability(
    graph: Graph,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
    rtol: float = DEFAULT_RANK_RTOL,
) -> IdentifiabilityReport:
    """Rank of the joint Jacobian: what the data cannot tell apart.

    The diagnostic that sees ACROSS blocks. See the module docstring for why
    no per-block guard can, and for the measured case that motivates it.

    Args:
        graph: the model.
        names: which latents to differentiate with respect to -- a sequence,
            or a bare string for one. ``None`` (the default) means all of
            them, in declaration order. A subset asks the **conditional**
            question a Gibbs block faces -- "is this block identified, with
            the others held fixed?" -- and the answer is routinely *yes* for
            every block of a partition whose joint model is degenerate. That
            is the whole reason this function takes the joint by default.
        at: values for the latents NOT selected (and evaluation values for
            those that are). Identifiability is a LOCAL property of a
            nonlinear model, so a sweep has to ask it where the sampler
            currently is; defaults to each latent's prior centre, which is
            right exactly once.
        rtol: singular values at or below ``rtol * s_max`` are called null.
            See :data:`DEFAULT_RANK_RTOL` for how the default is chosen and
            when to override it.

    Returns:
        An :class:`IdentifiabilityReport`. The three numbers to read first
        are ``n_par``, ``nullity`` and, when the nullity is non-zero,
        ``report.participation(0)`` -- which names the latents the degenerate
        direction mixes::

            report = identifiability(graph)
            if report.nullity:
                print(report.nullity, "blind directions;", report.participation(0))

    Raises:
        GraphError: if ``names`` is empty, repeats a latent, or names one the
            graph does not declare; if ``at`` names one, or a value does not
            broadcast; if a selected latent is complex or non-floating; if
            the graph has no observed node or an observed node's distribution
            carries no ``loc``; or if the Jacobian comes back float32 -- run
            the call (graph construction included) inside
            ``with jax.enable_x64(True):``.
    """
    refuse_ambient_float32(doing="identifiability's rank verdict")
    selected = resolve_names(graph, names)
    values = latent_values(graph, at)
    check_differentiable(graph, selected, values)
    check_observed_have_locs(graph, values)

    block = local_block(graph, selected, values)
    jacobian = dense_operator(block)  # (n_data, n_par), tangent at `values`
    refuse_single_precision(jacobian, doing="the joint Jacobian")

    _, shapes, spans = flat_view(values, selected)
    norms = jnp.linalg.norm(jacobian, axis=0)
    # A zero column is an exact null direction; dividing it by its own zero
    # norm makes the whole spectrum NaN, and a NaN spectrum reports rank 0
    # for every model. Leaving it at zero is both finite and correct.
    safe_norms = jnp.where(norms > 0, norms, 1.0)
    normalised = jacobian / safe_norms

    n_par = int(jacobian.shape[1])
    n_data = int(jacobian.shape[0])
    # The full_matrices flag is load-bearing in exactly ONE regime, and it is
    # the headline case: the free-per-cell model has 64 data points against
    # 72 parameters. Turned off there, `right` comes back (n_data, n_par)
    # with no rows past index n_data, so `right[rank:]` below would be EMPTY
    # while `nullity` still reported n_par - rank -- a report whose two
    # halves disagree, and whose direction() passes its bounds check and then
    # indexes off the end. Everywhere else the flag is pure waste: `U` is
    # discarded, and for n_data >= n_par both spellings return the same
    # spectrum and the same (n_par, n_par) `right` -- while the full (n_data,
    # n_data) left factor is the dominant cost of the whole diagnostic on a
    # realistic grid (measured upstream at 8.59 GB for U alone on a (32768,
    # n_par) float64 Jacobian).
    _, spectrum, right = jnp.linalg.svd(normalised, full_matrices=n_data < n_par)
    spectrum = np.asarray(spectrum, dtype=np.float64)
    # The SVD returns min(n_data, n_par) values; the remaining directions of
    # parameter space are exactly null, so pad rather than drop them. Then
    # rank is simply "how many are above the cutoff", with no caveat.
    spectrum = np.concatenate([spectrum, np.zeros(n_par - spectrum.size)])
    threshold = float(rtol * spectrum[0])
    rank = int(np.sum(spectrum > threshold))
    return IdentifiabilityReport(
        names=selected,
        shapes=shapes,
        spans=spans,
        n_par=n_par,
        n_data=n_data,
        rank=rank,
        nullity=n_par - rank,
        singular_values=spectrum,
        null_space=np.asarray(right[rank:], dtype=np.float64),
        jacobian=np.asarray(normalised, dtype=np.float64),
        column_norms=np.asarray(safe_norms, dtype=np.float64),
        rtol=float(rtol),
        threshold=threshold,
    )


__all__ = ["DEFAULT_RANK_RTOL", "IdentifiabilityReport", "identifiability"]
