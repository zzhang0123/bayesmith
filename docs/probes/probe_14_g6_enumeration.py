"""G6's enumeration: the evidence layer's public surface, one verdict a row.

The migration plan's G6 is one sentence -- "the gaps D12's wrapper needs,
REGISTERED ONE BY ONE" -- and a prerequisite nobody has sized. This counts it.

Run it::

    cd /Users/zzhang/projects/bayesmith
    /Users/zzhang/projects/e-RHINO/.venv/bin/python \
        docs/probes/probe_14_g6_enumeration.py

Exit code 0 means it FINISHED, not that the list is unchanged -- the numbers
are in the table it prints. Same reason as ``probe_11``'s: a probe that turned
"the list moved" into a non-zero exit invites the next person to make it green.

**The verdicts are data here, and the counts are derived from them.** Written
the other way round -- a prose summary with hand-added totals -- the first
draft of the page this feeds had two of its three totals wrong, which is the
defect this programme pays for most often and the reason the table below is
the source rather than the transcript.
"""

from __future__ import annotations

import importlib
import inspect

#: The evidence layer, as `tests/test_migration_records.py` and the Wave D row
#: name it.
MODULES = (
    "sqrtinfo", "factorize", "compress", "compressed",
    "diagnostics", "memory", "archive", "chain", "reduced_basis",
)

#: Where a counterpart could live on the far side.
FAR = (
    "bayesmith.evidence.sqrtinfo", "bayesmith.evidence.factorize",
    "bayesmith.evidence.compress", "bayesmith.evidence.campaign",
    "bayesmith.evidence.diagnostics", "bayesmith.exact.correct",
    "bayesmith.exact.fisher", "bayesmith.exact.solve",
    "bayesmith.exact.gaussian", "bayesmith.dispatch.factor",
)

#: The verdict per public name. Five values, and every one of them is a
#: decision recorded in `2026-08-27-g6-enumeration.md`:
#:
#:   HAVE   a counterpart exists on the far side (a LEAD -- same name is not
#:          same question, and only SqrtInfo has a cross-check today)
#:   STAY   a container, a format or a declaration; D12 and the plan's §0
#:   G3     belongs to `exact.chain`'s own ledger entry, not to G6
#:   G4     belongs to `exact.reduced_basis`'s own entry, not to G6
#:   G6     a numerical gap D12's wrapper would call
#:   OPEN   not decided here, and the reason is written on the page
VERDICTS: dict[str, str] = {
    # --- already there -------------------------------------------------
    "SqrtInfo": "HAVE", "marginalise": "HAVE", "marginalise_arrays": "HAVE",
    "Factorization": "HAVE", "compress": "HAVE", "coherent_mode": "HAVE",
    # --- containers, formats, declarations ------------------------------
    "CompressedLikelihood": "STAY", "QuadraticLikelihood": "STAY",
    "RawLikelihood": "STAY", "ReducedBasisLikelihood": "STAY",
    "EpochResidual": "STAY", "HeldOut": "STAY",
    "FidelityReport": "STAY", "ReducedBasis": "STAY",
    "BayesMemory": "STAY", "ChainMemory": "STAY",
    "save_memory": "STAY", "load_memory": "STAY",
    # --- G3's own entry --------------------------------------------------
    "HyperTransition": "G3", "LinearGaussianTransition": "G3",
    "chain_log_likelihood": "G3", "chain_marginal": "G3",
    "ornstein_uhlenbeck": "G3", "smooth": "G3",
    # --- G4's own entry --------------------------------------------------
    "basis_fidelity": "G4", "build_reduced_basis": "G4",
    "numerical_rank": "G4", "orthonormal_transform": "G4",
    "orthonormalise": "G4", "score_directions": "G4",
    "select_greedy": "G4", "select_svd": "G4",
    # --- G6 proper -------------------------------------------------------
    "compress_linear": "G6", "compress_reduced_basis": "G6",
    "epoch_residuals": "G6", "held_out_z": "G6",
    "shrinkage_power": "G6", "shrinkage_report": "G6",
    "systematic_floor": "G6",
    # --- undecided -------------------------------------------------------
    "reject_bad_term": "OPEN",
}


def far_side() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for path in FAR:
        try:
            module = importlib.import_module(path)
        except Exception as error:  # pragma: no cover - environment report
            print(f"  (could not import {path}: {error})")
            continue
        for name, obj in vars(module).items():
            if name.startswith("_"):
                continue
            if not (inspect.isfunction(obj) or inspect.isclass(obj)):
                continue
            if getattr(obj, "__module__", "").startswith("bayesmith"):
                found.setdefault(name, []).append(path.split(".")[-1])
    return found


def public(module_name: str) -> list[str]:
    module = importlib.import_module(f"rheplicant.inference.{module_name}")
    return sorted(
        name
        for name, obj in vars(module).items()
        if not name.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and getattr(obj, "__module__", "") == f"rheplicant.inference.{module_name}"
    )


def main() -> int:
    counterparts = far_side()
    rows: list[tuple[str, str, str, str]] = []
    for module_name in MODULES:
        for name in public(module_name):
            rows.append(
                (
                    module_name,
                    name,
                    VERDICTS.get(name, "!! UNCLASSIFIED !!"),
                    ",".join(counterparts.get(name, [])) or "-",
                )
            )

    width = max(len(name) for _, name, _, _ in rows)
    print(f"{'module':<14} {'public name':<{width}} {'verdict':<8} far-side name match")
    print("-" * (34 + width))
    for module_name, name, verdict, where in rows:
        print(f"{module_name:<14} {name:<{width}} {verdict:<8} {where}")
    print("-" * (34 + width))

    tally: dict[str, int] = {}
    for _, _, verdict, _ in rows:
        tally[verdict] = tally.get(verdict, 0) + 1
    total = sum(tally.values())
    print(f"{total} public names")
    for verdict in ("HAVE", "STAY", "G3", "G4", "G6", "OPEN"):
        print(f"  {verdict:<6} {tally.get(verdict, 0)}")
    stray = {v: n for v, n in tally.items() if v.startswith("!!")}
    if stray:
        # Not an exit code -- see the module docstring -- but it must be loud:
        # a name added upstream and not classified is exactly the drift this
        # page exists to make visible.
        print(f"  UNCLASSIFIED: {stray}  <-- add a verdict and update the page")
    print(f"\nG6 proper: {tally.get('G6', 0)} settled + {tally.get('OPEN', 0)} open")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
