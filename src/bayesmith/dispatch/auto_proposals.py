"""Bounded proposal discovery and enlargement of moving-noise linear blocks.

This eager compiler never changes priors or the user task. Existing matrix-free
routes are retained unless a small moving-noise block can be enlarged. Local
applicability probes select proposals, not exact Gaussian conditionals.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.artifacts import ProposalBlockPolicy
from bayesmith.diagnose.structure import (
    _gaussian_parameters,
    gaussian_flatness_certificate,
)
from bayesmith.dispatch.initialization import complete_initial_values
from bayesmith.dispatch.proposal_plan import compile_proposal_plan
from bayesmith.errors import GraphError, NotGaussian, StructureError
from bayesmith.exact.proposals import compile_proposals
from bayesmith.graph.evaluate import evaluate


def _observed_mean_moves(block, values):
    flat = block.pack(values)
    for term in block.terms:
        if term.prior:
            continue
        jacobian = jax.jacfwd(
            lambda x, term=term: block._residual(term, values, x).reshape(-1)
        )(flat)
        mask = block.graph.node(term.name).observed_mask
        if mask is not None:
            shape = block._residual(term, values, flat).shape
            mask = jnp.broadcast_to(mask, shape).reshape(-1, 1)
            jacobian = jnp.where(mask, jacobian, 0.0)
        if bool(jnp.all(jnp.isfinite(jacobian))) and bool(jnp.any(jacobian != 0)):
            return True
    return False


def select_proposals(graph, task, legacy):
    """Return the legacy plan or a validated automatic proposal schedule."""
    if (
        not graph.latents
        or not all(block.method in {"nuts", "gcr+mh"} for block in legacy.blocks)
        or task.initialization is None
        or task.block_order is not None
        or dict(task.backend_options).get("collapse", False)
        or task.chain_method not in (None, "sequential", "vectorized")
        or any(x is not None for x in (task.solver_tolerance, task.solver_maxiter, task.ess_floor))
    ):
        return legacy, ()
    decisions = []
    try:
        points, _ = complete_initial_values(
            graph, jax.random.key(193), task.initialization, task.budget.chains or 1,
        )
    except (GraphError, StructureError):
        raise
    except (ValueError, TypeError, NotImplementedError) as error:
        return legacy, (("initialization", "legacy_plan", str(error)),)
    values, policies = points[0], []
    certificate = gaussian_flatness_certificate(graph, graph.latents, values)
    if not certificate["numerical_derivatives_trusted"] or not any(
        not row["covariance_independent"] for row in certificate.get("observations", ())
    ):
        return legacy, (("model", "legacy_plan", "automatic_extension_requires_supported_parameter_dependent_gaussian_noise"),)
    def covariance_descriptors(point):
        env = evaluate(graph, point)
        return tuple(_gaussian_parameters(graph, n, env)[1] for n in graph.observed)

    direction = {n: jnp.ones_like(v) * (1 + jnp.abs(v)) for n, v in values.items()}
    _, covariance_change = jax.jvp(covariance_descriptors, (values,), (direction,))
    if not any(bool(jnp.all(jnp.isfinite(v))) and bool(jnp.any(v != 0)) for v in covariance_change):
        return legacy, (("model", "legacy_plan", "no_covariance_dependence_witness"),)
    # Keep an established moving-noise block indivisible. Only replace it if
    # one validated joint GLS block strictly enlarges it within dense budgets.
    seed = tuple(n for block in legacy.blocks if block.method == "gcr+mh" for n in block.latents)
    if seed:
        policy = ProposalBlockPolicy(seed, "iterative_gls")
        try:
            compile_proposals(graph, (policy,), values)
        except (GraphError, StructureError):
            raise
        except (ValueError, TypeError, NotImplementedError, NotGaussian) as error:
            return legacy, ((",".join(seed), "legacy_plan", str(error)),)
        policies.append(policy)
    # Start with vector-valued sites, then merge only validated joint proposals.
    for name in graph.latents:
        if name in seed:
            continue
        for method in ("bias_corrected_log_linear", "iterative_gls"):
            policy = ProposalBlockPolicy((name,), method)
            if np.size(values[name]) > policy.max_parameters:
                decisions.append((name, method, "skipped_parameter_budget"))
                continue
            try:
                candidate = compile_proposals(graph, (policy,), values).blocks[0]
                if not _observed_mean_moves(candidate, values):
                    decisions.append((name, method, "no_observed_mean_dependence"))
                    continue
            except (GraphError, StructureError):
                raise
            except (ValueError, TypeError, NotImplementedError, NotGaussian) as error:
                decisions.append((name, method, str(error)))
                continue
            policies.append(policy)
            decisions.append((name, method, "selected_with_original_target_mh"))
            break
        else:
            decisions.append((name, "nuts", "no_supported_structured_proposal"))
    if not policies:
        return legacy, tuple(decisions)
    groups = []
    for policy in policies:
        for index, group in enumerate(groups):
            if group.method != policy.method:
                continue
            combined = replace(group, names=group.names + policy.names)
            try:
                compile_proposals(graph, (combined,), values)
            except (GraphError, StructureError):
                raise
            except (ValueError, TypeError, NotImplementedError, NotGaussian) as error:
                decisions.append((",".join(combined.names), "merge_refused", str(error)))
                continue
            groups[index] = combined
            decisions.append((",".join(combined.names), policy.method, "joint_affinity_validated_merged"))
            break
        else:
            groups.append(policy)
    if seed and not any(
        group.method == "iterative_gls" and set(seed) < set(group.names)
        for group in groups
    ):
        decisions.append((",".join(seed), "legacy_plan", "no_valid_enlargement_of_existing_block"))
        return legacy, tuple(decisions)
    plan = compile_proposal_plan(
        graph, task, policies=tuple(groups), initial_values=values,
        selection="automatic", decisions=decisions,
    )
    return plan, tuple(decisions)
