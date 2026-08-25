# Cross-check: `identifiability`

`rheplicant.inference.identifiability` → `bayesmith.diagnose.identifiability`
(migration spec §八 step 5, acceptance §四 4.2). Executed 2026-08-25; every
number below was measured in this repo's venv on that date.

## 1. Fixtures

Per §二 step 1, rheplicant's own pinned fixtures, re-expressed on the graph
(`tests/diagnose/models.py`):

- **Healthy + degenerate pair**: `data[t,f] = gain[t] * (T_ant[t,f] +
  tone[f])`, 8×8, antenna temperature free-per-cell (72 parameters, the
  under-determined case) or through a (3,3) polynomial basis (17). Same
  coefficient matrix, deliberately asymmetric.
- **Must-refuse fixtures**: ambient float32; a graph pinning its own
  arithmetic to float32; complex/integer selected latents; empty/unknown/
  repeated names; a graph with no observed node; a Poisson observation
  (no `loc`).
- Auxiliary: mixed-scale (1e10 unit gap), zero-column (a dead latent),
  sort-trap (declaration order ≠ sorted order).

## 2. Numerical agreement

`tests/crosscheck/test_diagnose_identifiability.py`, same arrays handed to
both packages (never the same construction recipe — §0.1's PRNG trap):

| quantity | agreement |
|---|---|
| (n_par, rank, nullity), all four rows | equal, cell for cell: (72,64,8)×2, (17,17,0), (17,16,1) |
| singular values (basis/off) | rel 1e-10 |
| `weakest_identified` | rel 1e-9, both 4.822138e-05 |
| null direction, nullity 1 | elementwise abs 1e-9, sign-fixed on the largest entry |
| null space, nullity 8 | as projectors `VᵀV`, abs 1e-9 — rows are basis-dependent, the subspace is not |

## 3. Refusal agreement

Every loud rheplicant refusal has a same-shape refusal here (empty/unknown/
repeated names, complex → the R-linear explanation, integer, out-of-range
direction index, inconsistent report). Exception classes map
`ParameterSpaceError → GraphError`, `StateValidationError → StructureError`
— bayesmith's own taxonomy; the shared-identity concern of the design doc's
appendix applies to rheplicant's importers, not to new code here.

**One refusal changed mechanism, deliberately.** rheplicant forces
process-global x64 *inside* the diagnostic and refuses only a model that
pins its output to float32. This package's rule is that `src/` never touches
`jax.config`, so there are TWO refusals: ambient float32 (before any
tracing — a graph built in x64 and called outside it otherwise dies inside
`jax.linearize` with a bare dtype inconsistency) and result float32 (the
graph itself casts). Both are pinned, and the float32 counterfactual is
computed by hand in the suite to show what they protect.

## 4. Independent oracles (iron law 4)

- "Moving along the reported direction does not move the model": the
  end-to-end statement, evaluated through the graph itself, against a
  random direction of the same size (< 1e-3 of the random movement), on
  both the over- and under-determined fixtures.
- The free model's `weakest_identified` = 1/√2 exactly — fixed by the
  fixture's geometry, not by either implementation.
- The no-normalisation counterfactual: the mixed-scale fixture's raw
  spectrum ratio is measured (1.000000e-10) and shown to sit below the
  default tolerance.
- 5 of the 17 hand-applied mutations in the port's mutation pass targeted
  this module; all killed.

## 5. Intended differences

1. **The x64 mechanism** (above). Consequence: `DEFAULT_RANK_RTOL = 1e-8`
   was **re-measured, not ported**, per the spec's recorded trap. Under
   this regime: null direction 7.479266e-17 (upstream arithmetic 6.6e-17 —
   the spectrum moved with evaluation order, the verdict did not), weakest
   identified 4.822138e-05 (identical to every printed digit), float32
   null direction 3.116759e-08 (upstream pinned 3.1168e-8). The window and
   the constant survive; the justification now cites this side's numbers.
2. **Signature**: `(space, pipeline, state_template)` → `graph`. `at`
   defaults to prior centres (`prior_environment`, the dispatch layer's own
   anchoring rule) rather than declared inits — the same point, one
   spelling.
3. **The Jacobian's row layout** is `dense_operator`'s (observed nodes in
   sorted name order, flattened); rheplicant's is a single prediction
   array. Irrelevant to the rank and the null space; recorded because the
   `jacobian` field is public.
4. rheplicant's raw-`Bind`-space and x64-subprocess tests do not port
   (no such concepts here); the refusal-mechanism family replaces them.
