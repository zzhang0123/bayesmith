"""probe_34 — the seams R5 lands on, measured rather than read.

Run:  .venv/bin/python docs/probes/probe_34_residual_seams.py

Four measurements, each of which contradicted something a draft of the R5 plan
asserted. They are here so the next reader can re-run them rather than quote
this file.

1.  The prior/likelihood split reproduces every term exactly once, and the
    RECOMPOSITION is not bitwise. The obvious assertion --
    ``log_prior + log_likelihood == log_joint`` -- is a property of the
    summation ORDER, because ``log_joint`` accumulates one running total and
    splitting it into two reorders that sum.

2.  Which premise each structural class actually refuses under. Six of the
    seven exact-plus-residual graphs are stopped by the PRIOR AUDIT, not by the
    structure gate, because "a latent with a latent parent" is precisely what
    makes a graph leave a residual behind.

3.  The residual dimension of every shipped graph, sized from the compiled
    LAYOUT. Sizing it from a prior draw silently drops the graph whose prior
    cannot be drawn from; sizing it from `getattr(graph, "shape", {})` -- which
    this probe did until a review caught it -- silently counts latents instead,
    because `Graph` has no `.shape` attribute.

4.  ``p(x | tau)`` is not proper for every tau. The conditional in
    ``shared_ancestor`` is degenerate at tau=0, which is interior to the
    declared support.
"""

from __future__ import annotations

import inspect
import math
import sys
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import scipy.stats as st

import bayesmith
from bayesmith.dispatch.classify import prior_environment
from bayesmith.dispatch.evidence import _latent_shape, compile_evidence_problem
from bayesmith.errors import BayesmithError
from bayesmith.graph.evaluate import log_joint
from bayesmith.graph.reduction import as_graph

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))

from exact import models

PARAMETERISED = {
    "cancelling_sum": {"cancel": 1e2},
    "many_observations": {"count": 3},
    "roundoff_stress": {"big": 1e6, "sigma": 1e-3},
    "sigma_functional_block": {"weights": (1.0, 0.0, -1.0)},
    "wide_plate": {"size": 4},
}


def graphs():
    """Every shipped GRAPH. ``flagged_line`` returns three objects, two of
    which are graphs, so the denominator is graphs and not fixture functions --
    two independent censuses of this differed because neither said which."""
    for name, fn in sorted(vars(models).items()):
        if not inspect.isfunction(fn) or name.startswith("_"):
            continue
        if fn.__module__ != models.__name__:
            continue
        built = fn(**PARAMETERISED.get(name, {}))
        for index, candidate in enumerate(
            built if isinstance(built, tuple) else (built,)
        ):
            try:
                graph = as_graph(candidate)
                graph.nodes  # noqa: B018 - reading it is the check
            except (AttributeError, TypeError):
                continue  # flagged_line's third return is an array
            yield (name if not isinstance(built, tuple) else f"{name}[{index}]"), graph


def section_1_the_split():
    print("=" * 78)
    print("1. the split: per-term bitwise, recomposition NOT bitwise")
    print("=" * 78)
    from numpyro import handlers

    from bayesmith.bridge.numpyro_bridge import to_numpyro

    def draw(g, seed):
        tr = handlers.trace(
            handlers.seed(to_numpyro(g), jax.random.key(seed))
        ).get_trace(observed={})
        return {k: v["value"] for k, v in tr.items() if k in set(g.latents)}

    watch = (
        "straight_line",
        "two_observations",
        "observation_reused_downstream",
        "diamond_ancestor",
        "three_latent_chain",
    )
    with jax.enable_x64(True):
        for name in watch:
            g = as_graph(getattr(models, name)())
            p = compile_evidence_problem(g)
            hits = 0
            worst = 0.0
            for seed in range(200):
                v = draw(g, seed)
                total = float(log_joint(g, v))
                rec = float(p.log_prior(v)) + float(p.log_likelihood(v))
                if rec == total:
                    hits += 1
                else:
                    worst = max(worst, abs(rec - total))
            print(f"  {name:<32} recomposition bitwise {hits:>3}/200  worst {worst:.3e}")
    print("  -> the identity is float associativity, not the partition's property")


def section_2_which_premise_refuses():
    print("=" * 78)
    print("2. which premise each structural class actually refuses under")
    print("=" * 78)
    from bayesmith import compile_task
    from bayesmith.artifacts.refusal import Refusal
    from bayesmith.artifacts.tasks import EvidenceTask, new_task_meta
    from tests.dispatch.test_task_protocol import model_ref

    ref = model_ref()
    table: dict[str, Counter] = {}
    refused_to_compile: list[tuple[str, str]] = []
    with jax.enable_x64(True):
        for label, g in graphs():
            try:
                plan = bayesmith.compile(g)
            except BayesmithError as error:
                # NAMED, not swallowed. "a fixture that legitimately refuses to
                # compile" and "a row the census never reached" are different
                # silences, and an `except: continue` reports them as one.
                refused_to_compile.append((label, type(error).__name__))
                continue
            ex = tuple(plan.exact.latents) if plan.exact is not None else ()
            method = plan.exact.method if plan.exact is not None else None
            sampled = tuple(plan.sampled.latents) if plan.sampled is not None else ()
            if not sampled and method == "gcr":
                cls = "(a) whole-exact"
            elif ex and sampled:
                cls = "(b) exact+residual"
            elif not ex and sampled:
                cls = "(c) all-residual"
            elif not ex and not sampled:
                cls = "(e) no latents"
            else:
                cls = f"(d) {method}"
            out = compile_task(
                g, EvidenceTask(meta=new_task_meta(label="z")), model_ref=ref
            )
            premise = (
                out.failed_premise if isinstance(out, Refusal) else "(ADMITTED)"
            )
            table.setdefault(cls, Counter())[premise] += 1
    for cls in sorted(table):
        total = sum(table[cls].values())
        print(f"  {cls}  n={total}")
        for premise, count in table[cls].most_common():
            print(f"      {count:>3}  {premise}")
    classified = sum(sum(c.values()) for c in table.values())
    print(f"  compile refuses: {len(refused_to_compile)}")
    for label, kind in refused_to_compile:
        print(f"      {label:<34} {kind}")
    print(f"  denominator check: {classified} classified + "
          f"{len(refused_to_compile)} refused = {classified + len(refused_to_compile)}")


def section_3_residual_dimension():
    print("=" * 78)
    print("3. residual dimension, sized from the compiled LAYOUT")
    print("=" * 78)
    dims: Counter = Counter()
    total = 0
    with jax.enable_x64(True):
        for _label, g in graphs():
            try:
                plan = bayesmith.compile(g)
            except BayesmithError:
                continue  # counted and NAMED in section 2
            sampled = tuple(plan.sampled.latents) if plan.sampled is not None else ()
            if not sampled:
                continue
            total += 1
            # `_latent_shape`, not `getattr(g, "shape", {})`. This probe shipped
            # with the second spelling in the same batch that repaired it in
            # `dispatch/evidence.py`: `Graph` has no `.shape`, so the lookup was
            # always `{}` and `size` was a LATENT COUNT under a heading reading
            # "sized from the DECLARED shape". It happened to print the right
            # histogram, because no graph with a plated latent currently reaches
            # a sampled block -- a number right for a reason the file did not
            # have, which is this repository's named disease and not an
            # exemption from it.
            env = prior_environment(g)
            size = 0
            for name in sampled:
                shape = _latent_shape(g, name, env)
                size += int(jnp.prod(jnp.array(shape))) if shape else 1
            dims[size] += 1
    print(f"  graphs with a sampled block: {total}")
    print(f"  histogram: {dict(sorted(dims.items()))}")
    print("  -> sizing from a prior DRAW instead drops improper_outside_prior")
    print("     out of the histogram without it appearing anywhere as a gap;")
    print("     sizing from getattr(graph, \'shape\', {}) counts LATENTS, because")
    print("     Graph has no .shape -- this probe did that until a review caught")
    print("     it, and printed this same histogram for the wrong reason")


def section_4_the_degenerate_conditional():
    print("=" * 78)
    print("4. p(x | tau) is NOT proper for every tau")
    print("=" * 78)
    print("  shared_ancestor: tau ~ N(2.0, 0.5),  x ~ N(0, |tau|)")
    with jax.enable_x64(True):
        for tau in (0.1, 1e-3, 1e-8, 0.0):
            d = dist.Normal(0.0, jnp.abs(jnp.array(tau)))
            at0 = float(d.log_prob(0.0))
            at1 = float(d.log_prob(1.0))
            flag = "  <- degenerate" if not math.isfinite(at1) else ""
            print(
                f"    tau={tau:<8} log p(x=0)={at0:+10.4f}  "
                f"log p(x=1)={at1:+.4g}{flag}"
            )
    print(f"    P(tau <= 0) under the declared prior = {st.norm.cdf(0.0, 2.0, 0.5):.3e}")
    print("  -> tau=0 is INTERIOR to the support, so the declared integration")
    print("     span is a domain of the ORACLE and not a propriety requirement")


if __name__ == "__main__":
    section_1_the_split()
    section_2_which_premise_refuses()
    section_3_residual_dimension()
    section_4_the_degenerate_conditional()
