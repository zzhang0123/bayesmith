"""Explicit proposal schedules, separate from the legacy exact/NUTS partition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from bayesmith.artifacts import InitializationPolicy, PosteriorTask, TargetFidelity
from bayesmith.dispatch.initialization import complete_initial_values
from bayesmith.dispatch.plan import Block
from bayesmith.graph.graph import Graph


@dataclass(frozen=True)
class ProposalRuntimePlan:
    """A proposal chain with per-block correction and an optional NUTS remainder."""

    graph: Graph
    blocks: tuple[Block, ...]
    compiled: Any
    sigma_needs_rebuild: bool = True
    selection: str = "explicit"
    selection_decisions: tuple = ()

    @property
    def method(self):
        return "proposal_mh" if self.compiled.mh_corrected else "proposal_approximate"

    @property
    def target_fidelity(self):
        return (
            TargetFidelity.EXACT
            if self.compiled.mh_corrected
            else TargetFidelity.APPROXIMATE
        )

    @property
    def exact(self):
        # Proposal construction never certifies a Gaussian conditional.
        return None

    @property
    def sampled(self):
        return next((block for block in self.blocks if block.method == "nuts"), None)

    def _execution(self):
        return " → ".join(
            f"{block.method}({', '.join(block.latents)})" for block in self.blocks
        ) + (
            "; original-target Metropolis-within-Gibbs"
            if self.compiled.mh_corrected
            else "; approximate unadjusted updates: MH is OFF for one or more blocks"
        )

    def __str__(self):
        lines = [
            f"block {i} {{{', '.join(block.latents)}}} {block.method}\n  {block.reason}"
            for i, block in enumerate(self.blocks)
        ]
        return "\n".join((*lines, "execution: " + self._execution()))

    def sample(
        self,
        key,
        *,
        num_samples=2000,
        num_warmup=1000,
        num_chains=1,
        chain_method="sequential",
        progress_bar=False,
        nuts_options=None,
        nuts_on_collapse=False,
        collapse=False,
        _control=None,
    ):
        from bayesmith.dispatch.proposal_sampling import run_proposal_mcmc
        from bayesmith.dispatch.sampling import SamplingControl

        if collapse:
            raise ValueError("proposal schedules do not support collapse=True")
        return run_proposal_mcmc(
            self.graph,
            self.compiled,
            key,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            nuts_options=nuts_options,
            progress_bar=progress_bar,
            control=SamplingControl() if _control is None else _control,
        )


def compile_proposal_plan(
    graph: Graph, task: PosteriorTask, *, policies=None, initial_values=None,
    selection="explicit", decisions=(),
) -> ProposalRuntimePlan:
    """Validate static capabilities before any proposal chain is run."""
    from bayesmith.exact.proposals import compile_proposals

    if dict(task.backend_options).get("collapse", False):
        raise ValueError("proposal schedules do not support collapse=True")
    if task.chain_method not in (None, "sequential", "vectorized"):
        raise ValueError("proposal schedules support sequential or vectorized chains")
    if any(
        value is not None
        for value in (
            task.solver_tolerance,
            task.solver_maxiter,
            task.ess_floor,
        )
    ):
        raise ValueError(
            "proposal schedules use their policy's iterations/damping/scale; "
            "legacy CG and SNIS settings do not apply"
        )
    policies = task.proposals if policies is None else policies
    if initial_values is None:
        points, _ = complete_initial_values(
            graph, jax.random.key(193), task.initialization or InitializationPolicy(),
            task.budget.chains or 1,
        )
        initial_values = points[0]
    compiled = compile_proposals(graph, policies, initial_values)
    blocks = tuple(
        Block(
            latents=policy.names,
            method=policy.method + ("+mh" if policy.mh_correction else "+uncorrected"),
            reason=(
                f"{'Automatically selected' if selection == 'automatic' else 'Explicit'} {policy.method} proposal; {policy.steps} MH step(s) per sweep. "
                "Normalized forward/reverse proposal densities correct to the original "
                "joint model, including all declared priors and support constraints."
                if policy.mh_correction
                else f"Explicit {policy.method} proposal; MH OFF, {policy.steps} approximate update(s) per sweep. "
                "Finite, supported draws are applied without an acceptance ratio; "
                "the unadjusted schedule need not have a known joint target."
            ),
        )
        for policy in policies
    )
    if compiled.remainder:
        blocks += (
            Block(
                latents=compiled.remainder,
                method="nuts",
                reason="Unselected parameters form the NUTS remainder after proposal updates.",
            ),
        )
    return ProposalRuntimePlan(graph, blocks, compiled, selection=selection,
                               selection_decisions=tuple(decisions))
