# What B11 will find here

The migration spec's step 6 and its ledger row B11 describe the streaming
evidence layer as **rewritten from the graph** rather than transplanted, and
say it was "waiting on this interface's shape". B9 is finished, so this
records what the interface actually offers — measured, on this tree, with
re-runnable probes.

Nothing here is a plan. It is the set of facts a plan would otherwise assume.

---

> **Sections 1 and 4 have moved since this was written.** The masked
> normalisation is no longer an open decision — it was made, in
> `bayesmith.evidence.compress`, and §1 records what the decision was and why.
> §5 is what remains.

## 1. The whitening row is ready. The layer is not.

Step 6 reads: *"the whitening row becomes `L^-1 r`; `Precision.whiten` is the
operation it needs and already exists."* True of the row, false of the layer.

B11's must-preserve list names "the masked normalisation of `sigma = inf`
samples" among its numerical kernels. rheplicant does it with
`weight = where(seen, 1/sigma, 0)` (`inference/compress.py:111`) and a
normaliser summed over finite-sigma samples only (`compress.py:422`).
Measured on a fixture with one unobserved sample:

| operation | result |
|---|---|
| `apply(r)` at the `inf` entry | `0.0` — exactly the mask, no special case |
| `whiten(r)` at that entry | `0.0` — likewise |
| `quadratic(precision, r)` | finite; the unseen sample contributes nothing |
| `log_normalizer()` | `+inf` |
| `log_spectrum()[2]` | `+inf` |
| rheplicant's masked figure | `5.513631199` |

**Not a defect to fix in `precision.py`.** A `sigma = inf` sample has no
density, so `+inf` is the honest answer to the question a `Precision` is
asked. Reading it as `0` is a statement that the sample is UNOBSERVED — a
modelling concept the evidence layer has and this interface does not. Masking
it here would put a silently-wrong normaliser one import away from every
consumer, which is defect B1's shape.

So it was B11's first design decision. **It is made:**
`bayesmith.evidence.compress` owns the mask, because it is the layer that has
the concept, and `precision.py` keeps a normaliser that is never silently
wrong. `TestWhatTheEvidenceLayerWillFindHere` in
`tests/exact/test_precision.py` still pins both halves of the interface's
behaviour.

**And masking turned out to be a DIAGONAL concept**, measured rather than
assumed: for a stationary covariance the observed submatrix is neither
circulant nor a subset of the spectrum. On a 6-point kernel with one sample
dropped, its log-determinant is `-0.7084` while the closest subset sum of
log-eigenvalues is **0.47 nats away**. So an unobserved sample inside a
correlated epoch is *refused*, not approximated — any "mask the spectrum"
rule would be an approximation wearing an exact result's name, and the
constant is this layer's whole job.

---

## 2. Two of the three scopes are derivable. The third is a convention.

The spec's argument for rewriting rather than transplanting is that
rheplicant declares each latent's extent by hand (`Factorization`,
`scope="global" | "per_epoch" | "linked"`) while bayesmith's graph carries
plates, so the factorization can be DERIVED — "which kills the whole error
class `factorize.py` exists for (the same space declared twice)".

`docs/probes/probe_10_b11_scope_derivation.py` measures that claim.

| scope | expressible? | reaches the exact path? |
|---|---|---|
| `global` | yes | yes |
| `per_epoch` | yes | yes |
| `linked`, non-centred | yes | **yes — `gcr`** |
| `linked`, centred | yes | **no — `nuts`** |

**`global` and `per_epoch` really are a plate-membership question.**
`node.plate` is empty for one and names the epoch plate for the other. That
half of the spec's argument holds.

**`linked` is expressible and exact — non-centred.** An AR(1) chain written as
iid innovations through a deterministic linear recursion is affine in the
innovations, so the block stays linear. Measured against a dense posterior
mean: `2.5e-16` at `tol=1e-14`. At the plan's own default tolerance the error
is `1.2e-05`, which is CG tolerance and not disagreement — worth knowing
before someone reads the default as a discrepancy.

**But it is not a scope tag the graph can read off.** In that spelling the
innovations are per-epoch and the chain lives in a `Deterministic` node, so
`node.plate` alone cannot tell a linked latent from an ordinary per-epoch one.
The factorization has to know the modelling convention. **The spec's
derivation claim is two-thirds true**, and the remaining third is a naming
question rather than a structural one.

**The centred spelling needs a third `Precision` row**, and its cost is not
the QR. `fisher._log_spectrum_curvature` rests on the identity

    1/2 tr(N^-1 d_a N N^-1 d_b N) = 1/2 sum_k d_a log lam_k d_b log lam_k

which holds only where the covariance's eigenBASIS does not move with the
parameters — true of `I` and of the DFT, not of a general band. **A third row
must bring its own variance-information term.** Both spellings are pinned in
`tests/dispatch/test_classify.py` so the routing changing reads as a decision.

---

## 3. The kernel is in. `bayesmith.evidence.sqrtinfo`

Three of the five numerical kernels the ledger row names are ported and
cross-checked **bitwise** against `rheplicant.inference.sqrtinfo`
(`tests/crosscheck/test_sqrtinfo_agrees.py`, `rtol=0 atol=0`): the `[R|z]`
form, the QR fold's `-1/2 sum rho^2` corner, and the marginalisation constant
`+1/2 n log 2pi - sum log|R_ii| - 1/2 rho^2`.

The oracle throughout is dense NumPy — a log-density as
`-1/2 (x-m)^T F (x-m)`, an integral by `slogdet` — never one of our own
routines against another. Every comparison is on the **absolute**
log-density, because every constant here is invisible in a posterior's shape
and visible only in the evidence. That is how rheplicant shipped the
marginalisation constant with `-sum(log std)` missing: the probe that passed
used unit priors, where the term is exactly zero. Every fixture sweeps the
prior scale, and one test asserts the gap **is** `n_block * log(std)` at three
widths and three scales.

Two things measured that rheplicant's own docstring does not say:

* **`marginalise` raises different errors by transform.** `jit` trips the
  `bool(...)` in the pivot guard and gives `TracerBoolConversionError`;
  `grad` trips `float(max(pivots))` and gives `ConcretizationTypeError`.
  Both pinned. `marginalise_arrays` does both cleanly, which is the whole
  reason the two exist.
* **The NaN-safe spelling of the pivot guard is a SHAPE, not a live guard.**
  Rewriting `not all(> floor)` as `any(<= floor)` survives every test,
  because the finiteness check above refuses every `nan` first. Recorded in
  the comment with the measurement. The other two constants are convicted:
  dropping the corner fails 4 tests, dropping the log-pivot term fails 16.

## 4. The compressor and the campaign oracle are in

`bayesmith.evidence.compress` turns one epoch of a linear-Gaussian model into
a `[R | z]` term: `R = N^-1/2 A`, `z = N^-1/2 (d - c)`, offset
`-1/2 log det (2 pi N)`. It reads the noise through B9's `Precision`, so a
CORRELATED epoch compresses with no special case — checked against a dense
Gaussian whose covariance is materialised by `precision.dense` and inverted
in NumPy, so the oracle shares no FFT with the implementation.

**The spec's total oracle now runs**:
`tests/evidence/test_streaming_equals_batch.py` folds a campaign one epoch at
a time, holding one term, and compares against one dense Gaussian over every
observed sample — at three points, over three gap patterns, and at a thousand
epochs. It also checks the campaign EVIDENCE against `slogdet`, which is
where the compressor's normalisation, the fold's corner and the
marginalisation constant all have to be right at once. Mutating any one of
the three is caught by the campaign tests alone (3, 5 and 1 failures
respectively).

`kappa(R) = sqrt(kappa(F))` is asserted at a thousand epochs, which is the
property the square-root form exists for.

## 5. The per-epoch fold integrates nuisances

`compress_epoch` is the streaming analysis in one call: compress the epoch
over BOTH the survivors and its own nuisances, append the nuisances' prior
rows, and marginalise them. What comes back is a term over the survivors
alone, and folding those across a campaign is `SqrtInfo.combine`.

The prior rows go in **before** the marginalisation, and that is not a
convenience: a per-epoch nuisance is integrated exactly once, so a prior
arriving later has nowhere to be applied — and without one the block need not
constrain itself, which makes the integral divergent. `marginalise` refuses
that rather than returning the large plausible number finite arithmetic gives.

The oracle shares nothing with the implementation, not even its shape. The
marginal of a linear-Gaussian epoch has a closed form,

    d | x_g  ~  N(A_g x_g + A_n m_n + c,  N + A_n S_n A_n^T)

evaluated with a materialised covariance and `numpy.linalg` — no QR, no
pivots, no offset arithmetic in common. A whole campaign of per-epoch
nuisances is checked against **one** dense Gaussian with every nuisance
integrated jointly, so the E separate per-epoch constants have to sum to the
single joint one.

**And the reason every fixture sweeps the prior scale is now an executable
fact.** The nuisance prior's normalisation is `-sum(log std) - (n/2) log 2pi`
and the first half is exactly zero at `std = 1`. Mutation: removing it fails
7 tests — and **none** of them when the sweep is narrowed to `prior_std =
1.0`. That is rheplicant's historical blind spot reproduced on demand.

## 6. The factorization derives the partition and TESTS it

`bayesmith.evidence.factorize` splits a campaign's latents by plate
membership — `global` outside the epoch plate, `per_epoch` inside it — which
is the part of §2's derivation claim that holds. But plate membership is a
*hypothesis*, and `epoch_leakage` is the test of it: one forward-mode
jacobian of the observed predictions asks whether the latent touches any epoch
but its own, which is the question the fold actually depends on.

**Why it is a refusal.** A per-epoch nuisance is integrated exactly once. One
that also touches its neighbours is integrated E times instead, and the result
is a finite, plausible, WRONG evidence — measured at 0.15 nats over four
epochs and tens of nats over thirty-two, with no consistent sign. At zero
leakage the fold is exact to the bit.

**What a leak can look like** took three measurements to establish, each from
a fixture that did not work:

* a plated `Deterministic`'s function sees a **scalar** for a plated parent,
  so a cross-epoch map cannot be a plated node;
* a plated node whose parents are all unplated is refused by the graph itself
  ("nothing to map over");
* so a leak needs a plated consumer with **both** a plated parent and an
  unplated `(E,)` map.

The other reachable shape is an observation with no plate tag: its axis is an
epoch axis only by the author's intention, so it cannot be sliced. Both are
refused, and there is one fixture for each.

**The guard had a bug, found by mutation.** Python's `max` LOSES a NaN —
`max(0.0, nan)` is `0.0`, because it returns the first argument when
`nan > 0.0` is False. A model whose per-epoch map is singular at the
evaluation point had both running maxima stay at `0.0` and was reported as
**perfectly epoch-local**. A non-finite sensitivity is now refused by name,
before any maximum.

## 7. A campaign compresses straight from the graph

`bayesmith.evidence.campaign` is the bridge. Every input `compress_epoch`
needs is read off the graph:

| what | where it comes from |
|---|---|
| the partition | `factorize` — plate membership, then `epoch_leakage` |
| the designs | one `jacfwd` of the graph's own prediction map |
| the constant part | that map at zero |
| the data | the observed node's `obs`, sliced by epoch |
| the covariance | `precision_at`, sliced by epoch |
| the nuisance priors | the per-epoch latents' own `dist_fn` |

The last row is the one rheplicant cannot have: its `Factorization` carries
per-epoch priors separately from the `ParameterSpace` that declares them, and
"the same space declared twice" is the error class this kills.

`compress_campaign` folds the epochs holding one term at a time, so the
accumulated statistic does not grow with the campaign. Checked against dense
Gaussians written from each model by hand — a scalar campaign, a vector
survivor with a plated covariate, the campaign **evidence** with the survivor
marginalised too, and a campaign with both a per-epoch offset and a per-epoch
sigma.

**That last fixture exists because mutation found the gap, not review.**
Dropping `offset_prediction` and slicing every epoch's covariance at index 0
both passed the whole file: no earlier model had a constant part in its
prediction or a sigma that varied between epochs. A real campaign has both.

Two facts about plates were measured on the way: a latent OUTSIDE the epoch
plate can still have a shape of its own, so domain shapes come from a block
rather than from plate membership; and an unplated `const` inside a plated
`det` broadcasts, so a covariate has to be plated too.

## 8. The fold is vectorised, and the cost is flat in E

The archive property the ledger row names — rheplicant's "12,007 leaves → 8"
— **already held by construction**: a folded `SqrtInfo` is 3 leaves and 3
floats whatever the campaign length. Measured at E = 10, 100 and 1000.

What did not hold was the COST. The Python loop ran an eager QR per epoch,
and an unrolled `jit` of it compiled in 5.4 s at E = 1000 for a fold whose
warm run is 2 ms. Now the per-epoch build is one `vmap` and the fold is one
`lax.scan`:

| epochs | Python loop | vectorised | ms/epoch |
|---|---|---|---|
| 10 | 1.442 s | 1.413 s | 141 |
| 200 | 1.288 s | 1.325 s | 6.6 |
| 1000 | 2.847 s | 1.407 s | 1.41 |
| 5000 | — | 1.513 s | 0.30 |

**The number that matters is the slope, not the total.** About 1.4 s is fixed
— tracing and compiling — and the marginal cost per epoch fell from
**2.82 ms to 0.027 ms**, read off the 1000 → 5000 difference. Below roughly
500 epochs the fixed cost dominates and the loop was just as good; above it
the campaign is essentially free.

Two things made it possible, and both are the same split this codebase uses
everywhere. `compress` now always uses the masked normaliser for a
per-sample sigma — **bitwise** `log_normalizer()` when nothing is masked, so
nothing moved, and it removes the one concretisation that stopped the
function tracing. And `epoch_joint` was split out of `compress_epoch` so the
assembly exists once while `marginalise` (checked) and `marginalise_arrays`
(traced) can both use it.

**The refusal is moved, not weakened**, and that is now tested rather than
asserted: `_refuse_unconstrained_epochs` judges every epoch's pivots in one
comparison instead of E. Mutation found it untested at first — both branches
survived — so it has a graph-level fixture (a nuisance the data does not see,
with a `1e12` prior, since a wide prior alone is accepted when the data
constrains it) and direct tests for the ordering.

## 9. The diagnostics, including the refusal

`bayesmith.evidence.diagnostics` reads only the stored terms — no graph, no
prediction, no raw data. `coherent_mode` reports the **detectable** half of a
common-mode error as a z-score, because what such an error moves is a MEAN,
and a mean over N epochs is resolved at `sqrt(N)`.

**The load-bearing test is not the detection.** It is that an error of the
same size, placed inside the design's column space, is invisible — measured on
the same construction, not on two draws:

| injection | `chi2_z` | the answer |
|---|---|---|
| out-of-span, `s = 1.2` | matches `s²/√(2·dof/N)` | unbiased |
| in-span, `A·bias` | **bitwise the clean value** | displaced by exactly `bias` |

And it gets **worse with more data**: the bias does not shrink while the error
bar does, so the displacement in units of the posterior width grows as
`sqrt(N)` — measured at 4× over a 16× longer campaign.

That is why `refuse_undeclared_coherent_error` returns nothing and raises
instead. A campaign that has not declared this class is not clean, it is
unexamined, and the two must not read the same. There is no statistic to
improve.

## 10. What is still untouched

The thousand-epoch opaque-leaf archive. That is the whole list.

This section also named `diagnostics.py`'s declarative refusal of in-span
coherent errors, which section 9 — four paragraphs above — describes as
shipped and explains the reasoning for. It is shipped:
`refuse_undeclared_coherent_error` lives in `evidence/diagnostics.py`, is
exported from the package, and is covered by
`tests/evidence/test_diagnostics.py`. Corrected 2026-08-26.

A page that contradicts itself across two adjacent sections is worse than
either half would be alone. Whichever one a reader believes, the other is
evidence that they have misread the document, so the usual repair — read more
carefully — makes it worse rather than better.

Nothing is blocked. The numerics are not the reason, and neither is the
graph.
