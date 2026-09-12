"""Shared task plumbing and checks; the scientific models live in the demos.

The recovery criteria are declared before running: a marginal 99% equal-tail
interval must cover each generating value, and each posterior SD must be less
than half its marginal prior SD. The latter catches a sampler that returns
the prior. This is a fixed-truth recovery experiment, NOT an SBC claim about
99% repeated-experiment coverage. No seed retries or tolerance adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any

import jax
import numpy as np

from bayesmith import compile_task, execute_task
from bayesmith.artifacts import (
    ComputeBudget,
    DrawsPosterior,
    NamedArray,
    ParameterSource,
    PosteriorResult,
    PosteriorTask,
    Refusal,
    SimulationResult,
    SimulationTask,
    model_ref_from_callable,
    new_task_meta,
    proposal_options,
)
from bayesmith.dispatch.execute import chain_diagnostics
from bayesmith.graph.nodes import Const, Deterministic

INTERVAL_MASS = 0.99
MAX_SD_RATIO = 0.5  # Demo informativeness requirement, not a library quality gate.


@dataclass
class Demo:
    title: str
    graph: Any
    plan: Any
    posterior: PosteriorResult
    simulation: SimulationResult | None
    truths: dict[str, Any]
    prior_sds: dict[str, Any]
    x: Any
    data: Any
    true_signal: Any
    signal_draws: Any
    simulation_method: str = "SimulationTask(FIXED)"
    note: str = ""
    blocking_probes: tuple[dict[str, Any], ...] = ()
    view: dict[str, Any] = field(default_factory=dict)


def require_result(value, expected):
    if isinstance(value, Refusal):
        raise RuntimeError(f"Task refused: {value}")  # noqa: TRY004 -- a task refusal, not a bad input type
    if not isinstance(value, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
    return value


def simulate_fixed(graph, model, truths, key):
    """Generate ONE dataset through the existing Gaussian simulation task."""
    values = tuple(
        NamedArray(
            name, np.asarray(value), tuple(f"axis{i}" for i in range(np.ndim(value)))
        )
        for name, value in truths.items()
    )
    task = SimulationTask(
        meta=new_task_meta(label="Generate a dataset at known parameters"),
        parameter_source=ParameterSource.fixed(values),
        observed_sites=("obs",),
        budget=ComputeBudget(draws=1),
    )
    reference = model_ref_from_callable(model, identifier=model.__module__)
    planned = compile_task(graph, task, model_ref=reference, key=key)
    if isinstance(planned, Refusal):
        raise RuntimeError(f"Simulation compilation refused: {planned}")  # noqa: TRY004 -- capability refusal
    result = require_result(execute_task(planned, key=key), SimulationResult)
    return result.observation_draws[0].value[0], result


def infer(graph, model, key, draws, warmup, *, initialization=None, proposals=()):
    """Compile an inspectable plan and run two sequential chains when needed."""
    task = PosteriorTask(
        meta=new_task_meta(label="Recover simulation parameters"),
        budget=ComputeBudget(draws=draws, warmup=warmup, chains=2),
        chain_method="sequential",
        nuts_on_collapse=False,
        backend_options=(("progress_bar", False),) + (proposal_options(*proposals) if proposals else ()),
        **({"initialization": initialization} if initialization is not None else {}),
    )
    compile_key, sample_key = jax.random.split(key)
    planned = compile_task(
        graph,
        task,
        model_ref=model_ref_from_callable(model, identifier=model.__module__),
        key=compile_key,
    )
    if isinstance(planned, Refusal):
        raise RuntimeError(f"Posterior compilation refused: {planned}")  # noqa: TRY004 -- capability refusal
    posterior = require_result(execute_task(planned, key=sample_key), PosteriorResult)
    if not isinstance(posterior.representation, DrawsPosterior):
        raise TypeError("These demos require unweighted draws; never discard weights.")
    return planned, posterior


def samples(posterior):
    return {
        item.name: np.asarray(item.value) for item in posterior.representation.draws
    }


def recovery_checks(draws, truths, prior_sds):
    """Check every scalar coordinate, including vector-valued latent nodes."""
    if not truths or set(draws) != set(truths) or set(prior_sds) != set(truths):
        raise ValueError(
            "Draws, truths and marginal prior SDs must name the same latents."
        )
    rows = []
    tail = (1.0 - INTERVAL_MASS) / 2.0
    for name, truth in truths.items():
        values = np.asarray(draws[name])
        truth = np.asarray(truth)
        prior = np.broadcast_to(np.asarray(prior_sds[name]), truth.shape)
        if values.ndim != truth.ndim + 1 or values.shape[1:] != truth.shape:
            raise ValueError(
                f"{name}: draws do not match the generating parameter shape"
            )
        if values.shape[0] < 4 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name}: too few or nonfinite draws")
        if not np.all(np.isfinite(truth)) or not np.all(
            np.isfinite(prior) & (prior > 0)
        ):
            raise ValueError(f"{name}: invalid truth or marginal prior scale")
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        lower, upper = np.quantile(values, [tail, 1 - tail], axis=0)
        for index in np.ndindex(truth.shape):
            row = {
                "name": name + (str(list(index)) if index else ""),
                "truth": float(truth[index]),
                "mean": float(mean[index]),
                "posterior_sd": float(sd[index]),
                "lower": float(lower[index]),
                "upper": float(upper[index]),
                "covered": bool(lower[index] <= truth[index] <= upper[index]),
                "sd_ratio": float(sd[index] / prior[index]),
                "informative": bool(0 < sd[index] < MAX_SD_RATIO * prior[index]),
            }
            rows.append(row)
    return {
        "passed": all(row["covered"] and row["informative"] for row in rows),
        "interval_mass": INTERVAL_MASS,
        "max_sd_ratio": MAX_SD_RATIO,
        "parameters": rows,
    }


def diagnostic_checks(posterior):
    """Reuse bayesmith's existing ESS/r-hat verdict, with true chain grouping."""
    shape = posterior.representation.chain_shape
    if shape is None:
        return {"passed": True, "kind": "iid_exact_draws", "sites": {}}
    diagnostics = chain_diagnostics(samples(posterior), num_chains=shape[0])
    sites = {name: site._asdict() for name, site in diagnostics.items()}
    policy = dict(posterior.run.sampling_details)
    policy_passed = policy.get("diagnostics_passed")
    return {
        "passed": all(site.converged for site in diagnostics.values())
        and policy_passed is True,
        "kind": "bayesmith_chain_diagnostics",
        "sites": sites,
        "sampling_policy_passed": policy_passed,
        "divergences": policy.get("divergences"),
    }


def operator_name(fn):
    if isinstance(fn, partial):
        arguments = [repr(value) for value in fn.args]
        arguments.extend(f"{key}={value!r}" for key, value in fn.keywords.items())
        return f"{operator_name(fn.func)}({', '.join(arguments)})"
    return getattr(fn, "__name__", type(fn).__name__)


def graph_rows(graph):
    """Export topology from the actual traced Graph, never a hand-drawn twin."""
    rows = []
    for node in graph.nodes:
        if isinstance(node, Const):
            kind, operator = "constant", "const"
        elif isinstance(node, Deterministic):
            kind, operator = "deterministic", operator_name(node.fn)
        else:
            kind = "latent" if node.observed is None else "observed"
            operator = operator_name(node.dist_fn)
        rows.append(
            {
                "name": node.name,
                "parents": list(node.parents),
                "kind": kind,
                "operator": operator,
                "plate": list(node.plate),
                "linear_in": list(node.linear_in)
                if isinstance(node, Deterministic)
                else None,
                "depends_on_prediction": getattr(node, "depends_on_prediction", None),
            }
        )
    return rows


def mermaid(rows):
    ids = {row["name"]: f"n{i}" for i, row in enumerate(rows)}
    lines = ["flowchart LR"]
    for row in rows:
        label = f"{row['name']} · {row['operator']} · {row['kind']}"
        lines.append(f'  {ids[row["name"]]}["{label}"]')
        for parent in row["parents"]:
            lines.append(f"  {ids[parent]} --> {ids[row['name']]}")
    return "\n".join(lines) + "\n"
