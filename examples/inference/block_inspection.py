"""Record actual automatic partition decisions, without inventing a route.

The runtime plan comes from PosteriorTask; whether it was executed is recorded.
factor_partition is queried
on the SAME full graph, for comparison only. Its result or refusal is recorded;
no variables are fixed, no priors are replaced, and no alternative is sampled.
"""

import warnings

import numpy as np
import numpyro.distributions as dist

from bayesmith import factor_partition
from bayesmith.dispatch.classify import prior_environment
from bayesmith.errors import NotGaussian, NotLogLinear
from bayesmith.exact.gaussian import unwrap
from bayesmith.graph.evaluate import apply_probabilistic


def block_rows(plan, coordinates):
    names = tuple(name for block in plan.blocks for name in block.latents)
    return [
        {
            "index": index,
            "latents": list(block.latents),
            "coordinates": sum(coordinates[name] for name in block.latents),
            "conditional_on": [name for name in names if name not in block.latents],
            "method": block.method,
            "reason": block.reason,
            "linearity_evidence": repr(block.linearity) if block.linearity else None,
            "kappa": block.kappa,
            "solver_tolerance": block.tol,
            "tolerance_attainable": block.tol_attainable
            if block.kappa is not None
            else None,
        }
        for index, block in enumerate(plan.blocks)
    ]


def inspect_blocks(graph, runtime_plan, *, executed=True):
    env = prior_environment(graph)
    coordinates = {name: int(np.size(env[name])) for name in graph.latents}
    priors = []
    for name in graph.latents:
        distribution = unwrap(apply_probabilistic(graph, graph.node(name), env))
        item = {
            "latent": name,
            "coordinates": coordinates[name],
            "family": type(distribution).__name__,
            "parents": list(graph.node(name).parents),
        }
        if isinstance(distribution, dist.Uniform):
            item["bounds"] = {
                "lower": np.asarray(distribution.low).tolist(),
                "upper": np.asarray(distribution.high).tolist(),
            }
        priors.append(item)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            factor_plan = factor_partition(graph)
            factor = {
                "status": "planned",
                "blocks": block_rows(factor_plan, coordinates),
                "plan_text": str(factor_plan),
                "log_space_kind": factor_plan.log_space.kind
                if factor_plan.log_space
                else None,
            }
        except (NotGaussian, NotLogLinear) as refusal:
            factor = {
                "status": "refused",
                "blocks": [],
                "exception": type(refusal).__name__,
                "reason": str(refusal),
            }
    factor.update(
        {
            "strategy": "factor_partition",
            "executed": False,
            "probe_key_seed": 0,
            "warnings": [str(item.message) for item in captured],
        }
    )
    explicit = getattr(runtime_plan, "method", None) in {"proposal_mh", "proposal_approximate"}
    automatic = explicit and getattr(runtime_plan, "selection", None) == "automatic"
    approximate = getattr(runtime_plan, "method", None) == "proposal_approximate"
    return {
        "priors": priors,
        "executed": {
            "strategy": "PosteriorTask → automatic structured proposals → MH sweeps" if automatic else "PosteriorTask → explicit proposal policies → approximate sweeps (MH off)" if approximate else "PosteriorTask → explicit proposal policies → MH sweeps" if explicit else "PosteriorTask → compile_task → compile(strategy='declared') → partition",
            "description": "Automatic bounded proposals for parameter-dependent Gaussian noise: validate conditional log-linear/linear sites, then merge compatible sites only after a joint affinity and budget check. Small moving-noise GCR blocks may be enlarged; otherwise their existing route is retained. All priors and support enter MH; other sites remain in NUTS." if automatic else "Explicit proposal schedule with MH disabled in at least one block. Unadjusted approximate updates retain support checks; the schedule need not preserve the original posterior." if approximate else "Explicitly selected proposal methods and block order, validated by the compiler; original-target MH with an optional NUTS remainder." if explicit else "Automatic default compiler: read Gaussian-prior eligibility, declared linear_in paths, measured affinity and prediction-dependent noise; eligible coefficients share one block, with a NUTS remainder.",
            "selection_decisions": getattr(runtime_plan, "selection_decisions", ()),
            "executed": executed,
            "status": "planned",
            "blocks": block_rows(runtime_plan, coordinates),
            "plan_text": str(runtime_plan),
        },
        "factor_comparison": {
            **factor,
            "description": "Separate automatic factor strategy on the same full graph: test linear candidates individually, group compatible pairs, then probe log-space candidates. This comparison compiles only; it does not generate posterior draws.",
        },
    }
