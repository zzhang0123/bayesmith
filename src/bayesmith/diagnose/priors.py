"""Joint priors: a Jeffreys prior over a named block of a graph's latents.

A latent's own ``dist_fn`` says what one quantity is a priori, and it is the
only per-latent prior the rest of the package needs. A Jeffreys prior is not
of that shape: it is a single density over several latents at once, it is a
function of the forward model and the noise rather than of the latents
alone, and it moves when the model does. So it is a separate object, declared
over a named block::

    prior = JeffreysPrior(over=("fg_log_amp", "fg_beta"))
    prior.check_identified(graph)                  # once, at build time
    value = prior.log_density(graph, values)       # inside the potential

**Read this first: under a radiometer-style noise declaration this prior is
the flat prior.** For an observed node declaring ``Normal(mu, f * |mu|)``
over a bare power law ``mu = A (nu/nu0)^-beta`` with the block
``(log A, beta)``, the half-log-determinant measured on this package's own
8x8 fixture is **+15.80169853 at every one of nine grid points** spanning
two decades of amplitude and a unit of spectral index -- identical to the
last printed digit, gradient ~1e-17 -- and it equals both rheplicant's
evaluation of the same block and a numpy-only closed form, because the
algebra cancels exactly: ``sigma = |mu| f`` gives ``N^-1 = 1/(mu^2 f^2)``
while ``J_{k,i} = mu_k g_i(nu_k)``, so every ``mu`` cancels and ``I_ij =
(1 + 2 f^2) / f^2 * sum_k g_i g_j`` -- a constant matrix. Switching this
prior on for that model is the same thing as declaring the latents flat, at
the cost of a Jacobian per leapfrog step.

**Under a constant-sigma declaration the same block gives** ``p(log A)
proportional to A^2``: the half-log-determinant is exactly linear in
``log A`` with slope +2.000000 over six decades -- improper *upward*, so it
is a prior that needs the likelihood to be doing the work, and it is not a
neutral choice. Those two sentences are the same declaration under two noise
models, which is the thing to take from them: **the noise model chooses the
prior's shape** (on the same power law with a fixed 300 K floor,
``d(half-logdet)/d beta`` comes out with opposite signs under the two).
This is also why the prior reads the noise **from the graph** rather than
carrying one of its own: the graph is the single statement of the model, so
a likelihood/prior noise mismatch is not something this API can express --
where its rheplicant ancestor had to document "the prior inherits the
exit's noise", here there is no exit-supplied noise to inherit.

**Why the determinant comes from ``eigvalsh`` and not from the two obvious
routes.** On an exactly degenerate block -- an amplitude ``exp(a + b)``, the
same parameter twice -- the null eigenvalue lands at roundoff distance from
zero with a coin-flip sign, and both obvious routes return a plausible,
finite answer for a matrix that is singular by construction: ``slogdet``
reports a positive sign and a finite half-log-determinant, ``cholesky``
factors with a small positive pivot (measured upstream: +6.42 and +6.57
against a truth of -infinity). **A determinant that came back finite is not
a guard.** So the eigenvalues are taken explicitly, everything at or below
``rank_rtol * max`` is floored to the smallest positive number the dtype
has, and the result on the degenerate block is ~-338: an honest zero
density rather than a plausible lie. Do not replace this with ``slogdet``
or ``cholesky``; on ill-conditioned blocks they are precisely the routines
that cannot say no.

That floor is the *arithmetic*; the *refusal* is :meth:`check_identified`,
which a consumer calls once at build time and which delegates its verdict to
:func:`~bayesmith.diagnose.identifiability.identifiability` -- an SVD of the
column-normalised Jacobian, which does not square the condition number the
way ``J^T N^-1 J`` does, and which already knows how to name a degeneracy as
a combination of latents.

**Who calls it.** :meth:`log_density` is jit-safe and differentiable, so its
consumer is any potential that wants the prior -- today that means a NumPyro
model adding ``numpyro.factor("joint_prior", prior.log_density(graph,
values))`` beside flat sites for the covered latents (the bridge does not
yet declare joint priors on a graph; that integration is the
``numpyro_bridge`` row of the migration, recorded in
``docs/migration/priors.md``). The covered latents themselves must be
declared FLAT (``ImproperUniform``), and :meth:`information` refuses a
covered latent carrying a proper density: that would be two priors on one
quantity, multiplied, with no symptom -- each one on its own is correct, and
no diagnostic reports a prior counted twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith.diagnose.identifiability import (
    DEFAULT_RANK_RTOL,
    IdentifiabilityReport,
    identifiability,
)
from bayesmith.diagnose.local import local_block, refuse_ambient_float32
from bayesmith.errors import GraphError
from bayesmith.exact.fisher import (
    _log_spectrum_curvature,
    _spans,
    _weighted_design,
    dense_operator,
)
from bayesmith.exact.gaussian import precision_at, unwrap
from bayesmith.exact.gls import precision_from_graph
from bayesmith.graph.evaluate import apply_probabilistic, evaluate
from bayesmith.graph.graph import Graph

#: How many null directions a refusal names before it says "and N more".
_DIRECTIONS_SHOWN: int = 3


def _as_names(names: Any) -> tuple[str, ...]:
    """"One or many" for latent names -- careful not to explode a string.

    ``over="fg_beta"`` must mean one latent, not seven characters. The same
    convention :func:`~bayesmith.diagnose.local.resolve_names` applies to
    ``names=``.
    """
    if isinstance(names, str):
        return (names,)
    if isinstance(names, Sequence):
        return tuple(names)
    raise GraphError(
        f"JeffreysPrior(over={names!r}) -- `over` takes a latent name or a "
        "sequence of them, and it is mandatory: there is no default block, "
        "because the conditional Jeffreys prior of one block and the "
        "full-space one are different priors."
    )


class JeffreysPrior(eqx.Module):
    r"""``p(theta) = sqrt(det I(theta))`` over a named block, conditional on the rest.

    See the module docstring for the measured headlines and for who calls
    this. Attributes:

    Attributes:
        over: the latents the prior is over -- a name, or a sequence of
            them. Mandatory and explicit. **It is the CONDITIONAL Jeffreys
            prior of that block**, with every other latent held at whatever
            value the potential currently has, and the conditional and the
            full-space priors are different priors. They are not
            interchangeable and the full-space one often does not exist: on
            a many-latent graph the joint Jacobian is routinely
            rank-deficient, so ``det I`` is zero over the full space and
            ``sqrt(det I)`` is not a density at all. Naming the block is
            therefore not boilerplate -- it is the part of the declaration
            that decides whether the object exists.
        rank_rtol: the relative cut separating a null eigenvalue from a
            small one. ``None`` (the default) means
            :data:`~bayesmith.diagnose.identifiability.DEFAULT_RANK_RTOL`,
            read from the one place in this package where that number is
            justified against a measured spectrum.
    """

    over: tuple[str, ...] = eqx.field(static=True, converter=_as_names)
    rank_rtol: float | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if not self.over:
            raise GraphError(
                "JeffreysPrior(over=()) is over no latents, so its "
                "information matrix is 0x0, its determinant is the empty "
                "product 1, and it would contribute a flat zero to every "
                "posterior while reading as a declared prior. Name the "
                "block: over=('fg_log_amp', 'fg_beta')."
            )
        wrong = [name for name in self.over if not isinstance(name, str)]
        if wrong:
            raise GraphError(
                f"JeffreysPrior(over=...) takes latent NAMES, got {wrong}. "
                "The block is declared over the names a graph uses, not over "
                "node objects."
            )
        repeated = sorted({name for name in self.over if self.over.count(name) > 1})
        if repeated:
            raise GraphError(
                f"JeffreysPrior(over={list(self.over)}) lists {repeated} more "
                "than once. Two copies of one latent are exactly degenerate "
                "with each other, so the information matrix would be singular "
                "by construction and this prior would not exist -- for a "
                "reason that says nothing about the model."
            )
        if self.rank_rtol is not None and not self.rank_rtol > 0.0:
            raise GraphError(
                f"JeffreysPrior(rank_rtol={self.rank_rtol!r}) -- the rank "
                "tolerance is a positive relative cut. rank_rtol=0 keeps "
                "every eigenvalue including the roundoff ones, which is "
                "exactly the slogdet behaviour this prior does not use: on an "
                "exactly degenerate block it returns a plausible finite "
                "number for a density that does not exist."
            )

    # ------------------------------------------------------------- reading --

    @property
    def rank_tolerance(self) -> float:
        """:attr:`rank_rtol`, or identifiability's default when ``None``."""
        if self.rank_rtol is not None:
            return float(self.rank_rtol)
        return float(DEFAULT_RANK_RTOL)

    def covers(self, name: str) -> bool:
        """Whether this prior is the prior on latent ``name``."""
        return name in self.over

    @property
    def label(self) -> str:
        """How a message names this block."""
        return "(" + ", ".join(repr(name) for name in self.over) + ")"

    # ------------------------------------------------------------ refusing --

    def _check_against(self, graph: Graph, values: dict[str, Any]) -> None:
        """The declaration against the graph, and the double-prior refusal.

        Distribution *types* are static under tracing, so every check here is
        safe inside the jitted potential :meth:`log_density` serves.
        """
        unknown = [name for name in self.over if name not in graph.latents]
        if unknown:
            raise GraphError(
                f"JeffreysPrior(over={list(self.over)}) names {unknown}, "
                f"which this graph does not declare as latents; its latents "
                f"are {list(graph.latents)}. The block would be assembled "
                "from the names that DO match and the prior would be a "
                "different prior from the one written down -- over a smaller "
                "block, which is a different density and not a subset of the "
                "same one."
            )
        missing = [name for name in self.over if name not in values]
        if missing:
            raise GraphError(
                f"JeffreysPrior over {self.label} was evaluated at a values "
                f"dict with no entry for {missing}; it has {sorted(values)}. "
                "The block would be built from the names that are present, "
                "which is a prior over a different block."
            )
        env = evaluate(graph, values)
        doubled = [
            name
            for name in self.over
            if not isinstance(
                unwrap(apply_probabilistic(graph, graph.node(name), env)),
                dist.ImproperUniform,
            )
        ]
        if doubled:
            raise GraphError(
                f"latent(s) {doubled} are covered by "
                f"JeffreysPrior(over={list(self.over)}) AND declare a proper "
                "density of their own. That is two priors on one quantity: "
                "the posterior would be multiplied by both, which is a "
                "proper density and a plausible chain and not the model "
                "either declaration describes -- and no diagnostic reports a "
                "prior counted twice, because each one on its own is "
                "correct. Declare the covered latents flat "
                "(dist.ImproperUniform(dist.constraints.real, (), ())), or "
                "take them out of `over`."
            )

    def check_identified(
        self,
        graph: Graph,
        *,
        at: dict[str, jax.Array] | None = None,
        caller: str = "This JeffreysPrior",
    ) -> IdentifiabilityReport:
        """Refuse a block whose information matrix is rank-deficient.

        ``sqrt(det I)`` with ``det I = 0`` is not a density: it is zero
        everywhere the rank is deficient, which is everywhere, so there is
        no prior to normalise and nothing for a sampler to explore. The
        verdict is delegated to
        :func:`~bayesmith.diagnose.identifiability.identifiability`, which
        takes the rank of the column-normalised Jacobian rather than of
        ``J^T N^-1 J`` -- the Jacobian's own condition number, not its
        square -- and which reports the degenerate direction as a share of
        each latent.

        Args:
            graph: the model this prior is declared against.
            at: where to ask. Identifiability is a LOCAL property of a
                nonlinear model; ``None`` means each latent's prior centre,
                which is where a build-time check can ask.
            caller: what to name in the message.

        Returns:
            The report, when the block is identified.

        Raises:
            GraphError: naming the nullity and the latents each null
                direction mixes.
        """
        report = identifiability(
            graph, names=self.over, at=at, rtol=self.rank_tolerance
        )
        if report.nullity == 0:
            return report

        lines = []
        for index in range(min(report.nullity, _DIRECTIONS_SHOWN)):
            share = report.participation(index)
            carried = sorted(share.items(), key=lambda item: -item[1])
            lines.append(
                f"  direction {index}: "
                + ", ".join(
                    f"{name} {value:.2f}" for name, value in carried if value > 1e-3
                )
            )
        more = report.nullity - len(lines)
        if more > 0:
            lines.append(f"  ... and {more} more")

        raise GraphError(
            f"{caller} is declared over {self.label}, whose Jacobian has "
            f"nullity {report.nullity} of {report.n_par} parameters -- so "
            "det I is zero and sqrt(det I) is not a density: it is zero "
            "everywhere, there is nothing to normalise, and a sampler would "
            "be exploring a potential with an arbitrary additive constant "
            "along the null directions. Nothing downstream would say so: on "
            "a block degenerate by construction, slogdet returns a positive "
            "sign with a finite half-log-determinant and cholesky succeeds "
            "with a small positive pivot. The degenerate directions, as "
            "shares of each latent:\n"
            + "\n".join(lines)
            + "\nRe-parameterize the block, drop one of its latents from "
            "over=, or give the degenerate combination a declared density of "
            f"its own. identifiability(graph, names={list(self.over)}) "
            "reports the same thing in full."
        )

    # ---------------------------------------------------------- evaluating --

    def information(self, graph: Graph, values: dict[str, Any]) -> jax.Array:
        """The CONDITIONAL information matrix over ``over``, at ``values``.

        Every latent not in ``over`` is held at its entry in ``values`` --
        that is what makes this the conditional prior and not the full-space
        one. Rows and columns are in ``over``'s own order (its rheplicant
        ancestor returned them in sorted order and had to document the
        wart; the graph machinery preserves the caller's order, so the wart
        does not port).

        The matrix is assembled from the same pieces
        :func:`~bayesmith.exact.fisher.fisher_information` uses -- the
        weighted design and the variance's own information, ``2 (dlog
        sigma/dx)^T (dlog sigma/dx)`` for a diagonal noise, spelled on the
        spectrum so a fixed-basis correlated node is covered too -- but
        assembled here rather than through that function, deliberately: its
        centre check compares an already-decided precision against the rule
        at a claimed centre, redundancy that exists because its callers hold
        the noise in two places. Here the graph is the single statement of
        the noise and BOTH the weighting and the curvature are read from it
        at the same ``values``, so there is no second spelling to reconcile
        -- and the check's concrete-value comparison would break the jit
        path NUTS differentiates through.

        The variance-information term is included whenever any observed node
        claims ``depends_on_prediction`` (the default). It is not
        decoration: it is half of why the radiometer prior for a bare power
        law is exactly flat, the ``(1 + 2 f^2)`` in the module docstring's
        algebra. For a node declaring ``depends_on_prediction=False`` the
        term is skipped -- on such a node it is exactly zero, and the claim
        is separately checkable with
        :func:`~bayesmith.exact.gls.check_prediction_dependence`.

        Raises:
            GraphError: if ``over`` names something the graph does not
                declare, ``values`` misses a block member, or a covered
                latent declares a proper density of its own (two priors on
                one quantity).
        """
        refuse_ambient_float32(doing="a Jeffreys information matrix")
        self._check_against(graph, values)
        block = local_block(graph, self.over, values)
        design = dense_operator(block)
        matrix = design.T @ _weighted_design(block, design, precision_at(graph, values))
        if any(graph.node(name).depends_on_prediction for name in graph.observed):
            # The rule and the centre are the graph itself at `values`.
            spans, _ = _spans(block)
            centre = {name: jnp.asarray(values[name]) for name in self.over}
            matrix = matrix + 2.0 * _log_spectrum_curvature(
                block, precision_from_graph(graph, values), centre, spans
            )
        # Symmetric by construction and not quite symmetric in floating
        # point; eigvalsh reads one triangle, so which one it reads would
        # otherwise be a choice nobody made.
        return 0.5 * (matrix + matrix.T)

    def log_density(self, graph: Graph, values: dict[str, Any]) -> jax.Array:
        """``0.5 * log det I`` over the block -- the log prior, up to a constant.

        Jit-safe and differentiable, which is the whole requirement: NUTS
        differentiates it at every leapfrog step. It therefore cannot refuse
        a rank-deficient block, because a rank is a decision and a traced
        decision is one you cannot branch on -- :meth:`check_identified` is
        that refusal and a consumer calls it once, before sampling. What
        this returns on a degenerate block is the floored value (~-338 on
        the measured fixture): an honest zero rather than the plausible
        finite number ``slogdet`` gives.

        Arguments are :meth:`information`'s.
        """
        return self.half_log_determinant(self.information(graph, values))

    def half_log_determinant(self, matrix: jax.Array) -> jax.Array:
        """``0.5 * log det`` by eigendecomposition, with the rank floor applied.

        Separate from :meth:`log_density` so a caller holding an information
        matrix already -- or a test pinning this against ``slogdet`` and
        ``cholesky`` on the same array -- can reach the arithmetic without
        re-differentiating the model.
        """
        eigenvalues = jnp.linalg.eigvalsh(matrix)  # ascending, real
        floor = self.rank_tolerance * eigenvalues[-1]
        # The smallest positive number the dtype holds, not zero: log(0) is
        # -inf, and an infinite potential is a NaN gradient rather than a
        # rejected proposal.
        tiny = jnp.finfo(eigenvalues.dtype).tiny
        kept = jnp.where(eigenvalues > floor, eigenvalues, tiny)
        return 0.5 * jnp.sum(jnp.log(kept))


__all__ = ["JeffreysPrior"]
