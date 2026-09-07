"""The compiled residual problem, at the leaf so both layers may import it.

**Why this is a module of its own and not in ``dispatch/evidence.py``, where it
was written.** R5 Task 6's adapter lives in ``bridge/`` and must assert what it
was handed BY TYPE. Measured, the declared direction is ``dispatch -> bridge``:
``bridge/`` imports only ``graph/`` and ``distributions`` at module scope and
never ``dispatch/``, while ``dispatch/`` imports ``bridge/`` at
``evidence.py`` and at ``execute.py``. A module-scope import of this class from
``bridge/`` would therefore close a cycle that
``tests/test_layering.py::test_the_module_scope_import_graph_is_acyclic``
raises on.

Plan 6.2 offers two ways out and says only one keeps "dependency direction is
unchanged" true. The other -- importing it inside the adapter's function --
leaves the module-scope graph untouched, so the layering test passes while the
runtime dependency exists. That test's SUBJECT is which layer depends on which;
its IMPLEMENTATION is which import statements sit at module scope. Taking the
function-scope route would be choosing to satisfy the instrument rather than
the property, knowingly, and three separate guards in this repository have
already been walked past exactly that way.

So the class moves to a leaf: ``dispatch -> compiled`` and ``bridge ->
compiled``, with no new edge between them. It was cheap to move -- a plain
frozen dataclass, **not** a registered artifact type, and nothing under ``src/``
imported it. ``dispatch.evidence`` re-exports it, so existing imports keep
working.

Not ``bayesmith/evidence/``: that is a deprecated alias for
:mod:`bayesmith.marginal`, retiring at 1.0, whose own docstring argues the word
was wrong there. Not ``artifacts/``: this is not an artifact.

〔Design 0.17: the parent ``CompiledProblem`` this is nominally a variant of has
never been written, and R5 does not write it. This module's name leaves room
for it without claiming it exists.〕
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class CompiledEvidenceProblem:
    """A residual evidence problem, with its prior and likelihood separated.

    Nested sampling needs ``log L(theta)`` apart from ``log pi(theta)``, and
    nothing else in this package produces that separation:
    :func:`~bayesmith.graph.evaluate.log_joint` sums every ``Probabilistic``
    node into ONE running total and then adds ``joint_prior`` and every
    ``evidence_terms`` entry in the same loop. Building the split is a compiler
    pass and bayesmith owns it -- design section 1.5's first clause is graph
    semantics, structure discovery, premise validation and task-aware
    compilation.

    **No FIELD is a Graph, and the densities close over one.** Both halves are
    asserted, because stating only the first is how a guard that reads a
    spelling gets written: an earlier test here walked the field values for a
    ``.nodes`` attribute, and they are functions, so it passed while the graph
    was reachable through every one of the three closures.

    Design line 192 says a compiled problem may contain the residual log
    density, transforms, constant terms and a reconstruction map, "but it may
    not re-interpret the Graph". What that forbids is a BACKEND doing the
    re-interpreting. bayesmith compiling the densities itself, by closing over
    the graph, is the contract kept rather than broken -- the adapter receives
    callables and cannot reach a graph without walking closure internals. The
    half that binds an adapter is asserted where an adapter exists: its module
    imports no ``Graph`` and its signature takes only a compiled problem.

    The parent type design line 482 names, ``CompiledProblem``, does not exist
    in this package -- the name appears six times in the design and nowhere in
    the source -- so this is built standalone and to line 192's contract, in
    order that the variant relation is a refactor rather than a redesign when
    the parent is written.

    Attributes:
        log_prior: ``theta -> log pi(theta)``, the latent nodes' own densities
            plus the graph-level ``joint_prior``.
        log_likelihood: ``theta -> log L(theta)``, the observed nodes' densities
            (honouring ``observed_mask``) plus every graph-level
            ``evidence_terms`` entry. Those hold graph-level LIKELIHOOD factors
            despite the field's name, which the R4 PLAN records and declines to
            rename (`2026-09-04-r4-evidence.md:650`; the close-out does not
            mention the field at all, and an earlier version of this line cited
            it). The assignment is asserted by consequence -- the prior side
            integrates to one -- and never by the name.
        prior_sample: ``key -> theta``, a draw from the prior. Strictly
            stronger than a prior that integrates to one, and it is what a
            nested sampler actually needs: ``improper_outside_prior`` raises
            here rather than returning a number.
        exact_elimination: latents already integrated in closed form.
        residual_parameters: latents the backend must integrate. Disjoint from
            ``exact_elimination`` -- ``InferencePlanRecord`` already refuses a
            name in both, and the two would otherwise disagree about which
            parameters a backend was handed.
        shapes: the parameter layout, ``(name, shape)`` per residual parameter.
        prior_terms, likelihood_terms: which term went to which side. Carried
            as data so that a test can assert every term is filed exactly once
            without re-deriving the partition it is grading.
    """

    log_prior: Any
    log_likelihood: Any
    prior_sample: Any
    exact_elimination: tuple[str, ...]
    residual_parameters: tuple[str, ...]
    shapes: tuple[tuple[str, tuple[int, ...]], ...]
    prior_terms: tuple[str, ...]
    likelihood_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        both = sorted(set(self.exact_elimination) & set(self.residual_parameters))
        if both:
            raise ValueError(
                f"{both} are named as both eliminated and residual; an "
                "eliminated parameter is precisely one the problem does not "
                "carry"
            )
        shared = sorted(set(self.prior_terms) & set(self.likelihood_terms))
        if shared:
            raise ValueError(
                f"{shared} are filed on both the prior and the likelihood "
                "side; a term counted twice is the failure an evidence layer "
                "exists to prevent"
            )


__all__ = ["CompiledEvidenceProblem"]
