#!/usr/bin/env python3
"""Record, or re-check, the inference plans and tables the documentation shows.

Each record runs one snippet under ``site/snippets/`` and writes
``site/assets/plans/<name>.json``: the printed plans, a per-block summary the
pages render as tables (``{{plan:plans/<name>.json#<plan>}}``), optional result
tables (``{{table:...}}``) and the environment they were measured in.

    JAX_ENABLE_X64=1 .venv/bin/python tools/record_site_plans.py
    JAX_ENABLE_X64=1 .venv/bin/python tools/record_site_plans.py --check

``--check`` re-measures every plan and compares STRUCTURE only: block
members, route labels, the execution line and which declarations were
refused. Condition numbers and tolerances in the printed text depend on the
BLAS, and sampled columns in tables depend on the platform's arithmetic, so
the pages label both with their platform and they are not compared. Table
columns derived in closed form are recomputed and compared.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import runpy
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SNIPPETS = ROOT / "site/snippets"
OUTPUT = ROOT / "site/assets/plans"
REFUSED_CLAIM = "is declared linear, but the prediction is not affine in it"
EXACT_RTOL = 1e-9


@dataclass(frozen=True)
class Record:
    """One snippet and what to measure from it.

    ``plans`` maps a plan name to a function of the snippet's namespace that
    returns ``(model, trace_args)``. ``tables`` maps a table name to a function
    returning ``(columns, rows, settings)``; it may sample and runs only when
    recording. ``exact`` maps a table name to a function returning, per row,
    the closed-form columns ``--check`` recomputes. ``invariants`` returns
    problems with the measured structure beyond "matches the record".
    """

    snippet: str
    plans: dict[str, Callable[[dict], tuple[Any, tuple]]]
    tables: dict[str, Callable[[dict], tuple[list, list, dict]]] = field(
        default_factory=dict
    )
    exact: dict[str, Callable[[dict], list[dict]]] = field(default_factory=dict)
    invariants: Callable[[dict], list[str]] = lambda measured: []


def _overview_declared(ns):
    """``mixed`` with the affinity of ``x`` declared; the Overview says it
    prints the same plan, and ``--check`` holds that sentence to the
    measurement."""
    import jax.numpy as jnp
    import numpyro.distributions as dist

    import bayesmith as bs

    A = ns["A"]

    def mixed_declared(data):
        x = bs.sample("x", lambda: dist.Normal(jnp.zeros(2), 2.0).to_event(1))
        nu = bs.sample("nu", lambda: dist.LogNormal(0.0, 0.5))
        mu = bs.det(
            "mu",
            lambda x_, nu_: (A @ x_) * jnp.exp(0.1 * nu_),
            x,
            nu,
            linear_in=("x",),
        )
        bs.observe("d", lambda m: dist.Normal(m, 0.5), mu, obs=data)

    return mixed_declared, (ns["DATA"],)


def _overview_invariants(measured):
    problems = []
    if measured["mixed"] != measured["mixed_declared"]:
        problems.append("declaring linear_in=('x',) no longer prints the same plan")
    if not any(row[2] for row in measured["curved"]["blocks"]):
        problems.append("the false declaration in `curved` was not refused")
    return problems


def _partial_data(ns):
    import jax.numpy as jnp

    return ns["hierarchy"], (ns["INDEX"], jnp.asarray(ns["simulate"]()[1]))


RECOVERY_COLUMNS = [
    {"key": "name", "label": "Parameter"},
    {"key": "truth", "label": "Truth", "format": ".3f"},
    {"key": "exact_mean", "label": "Exact mean", "format": ".4f"},
    {"key": "sampled_mean", "label": "Sampled mean", "format": ".4f"},
    {"key": "exact_sd", "label": "Exact sd", "format": ".4f"},
    {"key": "sampled_sd", "label": "Sampled sd", "format": ".4f"},
]
RECOVERY_SETTINGS = {"seed": 1, "num_warmup": 1000, "num_samples": 4000}


def _partial_recovery(ns):
    rows = ns["compare"](**RECOVERY_SETTINGS)
    return RECOVERY_COLUMNS, rows, RECOVERY_SETTINGS


def _partial_exact(ns):
    truth, data = ns["simulate"]()
    mean, sd = ns["exact_posterior"](data)
    truths = [truth["population"], *truth["groups"]]
    return [
        {
            "truth": float(truths[i]),
            "exact_mean": float(mean[i]),
            "exact_sd": float(sd[i]),
        }
        for i in range(len(mean))
    ]


def _partial_invariants(measured):
    routes = [row[:2] for row in measured["hierarchy"]["blocks"]]
    if routes != [[["groups"], "GCR exact"], [["population"], "NUTS"]]:
        return ["the hierarchy no longer puts the group means in an exact block"]
    return []


def _notebook_invariants(measured):
    routes = [row[:2] for row in measured["model"]["blocks"]]
    if routes != [[["groups"], "GCR exact"], [["population"], "NUTS"]]:
        return [
            "the notebook model no longer has exact groups inside NUTS; revise its page"
        ]
    return []


RECORDS: dict[str, Record] = {
    "overview": Record(
        snippet="overview_plan.py",
        plans={
            "mixed": lambda ns: (ns["mixed"], (ns["DATA"],)),
            "curved": lambda ns: (ns["curved"], (ns["DATA"],)),
            "mixed_declared": _overview_declared,
        },
        invariants=_overview_invariants,
    ),
    "partial": Record(
        snippet="partial_structure.py",
        plans={"hierarchy": _partial_data},
        tables={"recovery": _partial_recovery},
        exact={"recovery": _partial_exact},
        invariants=_partial_invariants,
    ),
    "first": Record(
        snippet="first_model.py",
        plans={"level": lambda ns: (ns["model"], (ns["DATA"],))},
        invariants=lambda measured: (
            []
            if measured["level"]["execution"] == "iid draws, no chain"
            else ["First model is no longer fully exact; revise its page"]
        ),
    ),
    "decay": Record(
        snippet="nonlinear.py",
        plans={"decay": lambda ns: (ns["model"], (ns["DATA"],))},
    ),
    "notebook": Record(
        snippet="notebook_model.py",
        plans={"model": lambda ns: (ns["model"], (ns["INDICES"], ns["DATA"]))},
        invariants=_notebook_invariants,
    ),
}


def _require_x64():
    if os.environ.get("JAX_ENABLE_X64") != "1":
        raise SystemExit(
            "Set JAX_ENABLE_X64=1 before running: the snippets' docstrings run "
            "them that way, and precision must be chosen before JAX starts."
        )


def _first_sentence(text):
    """The first sentence of a block's printed evidence, on one line."""
    flat = " ".join(text.split())
    end = flat.find(". ")
    return flat if end < 0 else flat[: end + 1]


def _block_rows(plan, text, labels):
    lines = text.splitlines()
    heads = [i for i, line in enumerate(lines) if line.startswith("block ")]
    rows = []
    for index, (block, start) in enumerate(zip(plan.blocks, heads, strict=True)):
        label = labels.get(block.method, block.method)
        members = "{" + ", ".join(block.latents) + "}"
        head = lines[start]
        evidence = head[head.index(members) + len(members) :].strip()
        evidence = evidence[len(label) :].strip()
        stop = heads[index + 1] if index + 1 < len(heads) else len(lines) - 1
        if not evidence:
            evidence = _first_sentence(" ".join(lines[start + 1 : stop]))
        rows.append(
            {
                "latents": list(block.latents),
                "method": block.method,
                "route": label,
                "evidence": evidence,
                "refused_claim": REFUSED_CLAIM in block.reason,
            }
        )
    return rows


def _namespace(record):
    return runpy.run_path(str(SNIPPETS / record.snippet))


def measure_plans(record, ns):
    import bayesmith as bs
    from bayesmith.dispatch import plan as plan_module

    plans = {}
    for name, build in record.plans.items():
        model, args = build(ns)
        plan = bs.compile(bs.trace(model, *args))
        text = str(plan)
        plans[name] = {
            "text": text,
            "blocks": _block_rows(plan, text, plan_module._LABELS),
            "execution": text.splitlines()[-1].removeprefix("execution: "),
        }
    return plans


def structure(plans):
    return {
        name: {
            "blocks": [
                [row["latents"], row["route"], row["refused_claim"]]
                for row in plan["blocks"]
            ],
            "execution": plan["execution"],
        }
        for name, plan in plans.items()
    }


def environment():
    import jax
    import numpy

    blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "recorded": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "src_differs_from_head": bool(
            git("status", "--porcelain", "--", "src/bayesmith")
        ),
        "src_sha256": _source_digest(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "bayesmith": metadata.version("bayesmith"),
        "jax": jax.__version__,
        "numpy": numpy.__version__,
        "numpyro": metadata.version("numpyro"),
        "blas": " ".join(
            str(part)
            for part in (blas.get("name"), blas.get("version"))
            if part and part != "unknown"
        ),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
    }


def _source_digest():
    digest = hashlib.sha256()
    for path in sorted((ROOT / "src/bayesmith").rglob("*.py")):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def snippet_digest(record):
    return hashlib.sha256((SNIPPETS / record.snippet).read_bytes()).hexdigest()


def check(name, record_path):
    record = RECORDS[name]
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    problems = []
    if stored.get("snippet_sha256") != snippet_digest(record):
        problems.append(f"{name}: the snippet changed after it was recorded")
    ns = _namespace(record)
    measured = structure(measure_plans(record, ns))
    if structure(stored["plans"]) != measured:
        problems.append(
            f"{name}: the measured plan structure differs from the record:\n"
            + json.dumps(measured, indent=1)
        )
    problems.extend(f"{name}: {problem}" for problem in record.invariants(measured))
    for table, exact in record.exact.items():
        rows = stored["tables"][table]["rows"]
        for stored_row, fresh in zip(rows, exact(ns), strict=True):
            for key, value in fresh.items():
                if abs(stored_row[key] - value) > EXACT_RTOL * max(abs(value), 1.0):
                    problems.append(
                        f"{name}: {table} {stored_row['name']} {key} recorded "
                        f"{stored_row[key]!r}, closed form now gives {value!r}"
                    )
    return problems


def record_one(name, output):
    record = RECORDS[name]
    ns = _namespace(record)
    tables = {}
    for table, build in record.tables.items():
        columns, rows, settings = build(ns)
        tables[table] = {"columns": columns, "rows": rows, "settings": settings}
    stored = {
        "snippet": record.snippet,
        "snippet_sha256": snippet_digest(record),
        "environment": environment(),
        "plans": measure_plans(record, ns),
        "tables": tables,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
    print(f"Recorded {name}: {', '.join(stored['plans'])} -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--only", choices=sorted(RECORDS), action="append")
    parser.add_argument(
        "--record",
        type=Path,
        help="check this file instead of the committed one (requires one --only)",
    )
    args = parser.parse_args()
    _require_x64()
    names = args.only or sorted(RECORDS)
    if args.record and len(names) != 1:
        parser.error("--record needs exactly one --only")
    if not args.check:
        for name in names:
            record_one(name, args.record or OUTPUT / f"{name}.json")
        return 0
    problems = []
    for name in names:
        problems.extend(check(name, args.record or OUTPUT / f"{name}.json"))
    if problems:
        print("Re-record with this script:\n- " + "\n- ".join(problems))
        return 1
    print(f"Site plans: {', '.join(names)} match a fresh measurement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
