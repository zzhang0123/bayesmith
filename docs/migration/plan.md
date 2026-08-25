# Cross-check: `plan` (with `engines`)

`rheplicant.inference.plan` (`SamplingPlan`, `Block`, `estimate`, `sample`,
`PlanDiagnostics`) and `rheplicant.inference.engines` → `bayesmith.dispatch`
(`compile`, `InferencePlan.estimate`, `InferencePlan.sample`).

Test: `tests/crosscheck/test_dispatch.py` (6 cases, ~9 s). Measured
2026-08-25.

§四 4.2 asks for `plan.estimate` value-for-value on one partition and one
toy model, and for `plan.sample` compared by posterior **moments** — a χ²
trace is not comparable across NUTS implementations. It carries the
ordering constraint **"先落 B1"**, without which *"the comparison fixes the
GLS-type target as the reference"*.

**That constraint turned out not to be reachable here, and finding out why
is this row's main result.** See §5(a).

## 1. Fixtures

A quadratic spectrum replicated over a 6×4 grid, one conjugate block over
three coefficients, σ = 0.5 constant, prior width 50. Healthy, and
well-conditioned enough that the estimate comparison is about the fixed
point rather than about the solver's tolerance.

Must be refused: the **degenerate** partition — two length-4 latents that
reach the prediction only as `a + b`, nullity 4 of 8.

For B1, `μ = w·x` with `σ = 0.5·|μ|` over 40 points, which is the model
§三 B1 works in, run twice: once with the latent declared linear (conjugate
block) and once with `μ = exp(w)·x` (gradient block).

## 2. Numerical agreement

| quantity | agreement |
|---|---|
| `plan.estimate`, three coefficients | **8.9e-15 absolute**, ~1e-15 relative |
| both against the dense oracle | ~1e-14 relative |
| `plan.sample` mean, 800 draws | \|z\| 0.89 (theirs) and 0.72 (ours) |
| `plan.sample` variance | within 1.6% and 6.3% of the oracle, against a 5.0% sampling error |
| conjugate block on a prediction-dependent σ | **9e-12** between packages |

Not bitwise, and honestly so: the two run different iteration schemes to
the same fixed point — block-coordinate descent against a single reweighted
solve — so float64 roundoff is the right claim and the bit is not
available.

The sampling comparison is to the **dense oracle**, not between packages.
Two Gibbs sweeps over the same partition visit different states in a
different order even at the same key, so there is nothing pairwise to
compare; what must agree is the distribution they leave invariant.

## 3. Refusal agreement

| rheplicant | here | mapping |
|---|---|---|
| `observed` not shaped like the prediction | the data lives on the node, so `node_shape` reconciles it | `ParameterSpaceError` → `StructureError`; recorded in `noise.md` §3 |
| rank-deficient joint Jacobian | **no counterpart, deliberately** | §5(b) |
| — | `estimate()` on a graph with no exact subgraph | `NotImplementedError`, naming `sample()`; §5(a) |

## 4. Independent oracle

**Dense linear algebra** for the point estimate and for the posterior
covariance the draws are measured against, built from the full model at
basis vectors.

**Closed forms** for B1, written in NumPy from the algebra §三 B1 states —
`Σd²/x² ÷ Σd/x` with the log-determinant dropped, `mean(d/x)` with it kept
— so neither package supplies the reference. The two are 22% apart on this
fixture, and the test asserts that distance before asserting where anything
landed: without it, agreeing on 6.2588 would look identical to agreeing on
5.1046.

**Mutation.** Two, both killed by the test named against them:
`estimate()` no longer refusing a non-exact graph, and the dispatch
estimate dropping its reweighting.

## 5. Intended differences

**(a) B1 belongs to the BLOCK TYPE, not to the exit — and this side has no
second door.** This is sharper than the spec's own statement, and measured:

| exit | lands on |
|---|---|
| closed form, log-det **kept** (unbiased) | **5.104641** |
| closed form, log-det **dropped** (GLS-type) | **6.258841** |
| rheplicant `plan.estimate`, **conjugate** block | 5.104558 |
| bayesmith `plan.estimate()`, same model | 5.104558 |
| rheplicant `plan.estimate`, **gradient** block | **6.248269** |
| bayesmith, same non-linear model | **refuses** |

So §三 B1's analysis of frozen-sigma reweighting — made about bayesmith's
`iterative_gls` — is equally true of rheplicant's **conjugate** block: its
fixed point is the unbiased estimator. The gap is the gradient block, on
the estimate exit as well as the sampling one, which is why the property is
better stated of the block type.

**Carried upstream — on a branch, not yet on `main`.** e-RHINO's
`7f03af1` rewrote `inference/plan.py`'s module docstring around this, with
both numbers and an attribution back to this test. Measured 2026-08-25:
that commit is on `track-a-tail`, **unmerged and unpushed**, so a checkout
of e-RHINO's `main` does not carry it — which is also why the guard below
can go red for a reason that is nobody's defect — its own words: *"reading it as a
property of one exit is what left the estimate path unexamined."*
`test_rheplicants_plan_now_attributes_b1_to_the_block_type` asserts that
framing is still there, because a docstring is the one kind of claim
nothing else executes, and that is precisely how `condition_estimate` came
to open with a paragraph about a different function (§5(a) of
`linear.md`). The two records now hold each other up.

And the row's ordering warning cannot bite here. A non-linear graph has no
exact subgraph, so `estimate()` refuses by name and points at `sample()`,
which goes through NumPyro, whose `Normal(μ, σ)` carries its own `−log σ`.
There is no second place in this package that could drop the
log-determinant, so the comparison cannot fix the reference to the wrong
target — **not because the comparison was careful, but because one side of
it does not exist.**

**(b) A rank-deficient partition is refused there and answered here.**
rheplicant refuses: *"its joint Jacobian has nullity 4 of 8 parameters, so
that many independent directions leave the prediction unchanged and any
answer along them is arbitrary."*

On the graph the answer is **not** arbitrary. A proper prior makes the
posterior proper along the null direction, and what comes back is its mean
— verified against dense NumPy, identical to every digit, with `a` and `b`
split evenly as equal priors require. This is §5.19 recurring one layer up:
a rank-deficient **observed** Jacobian is not an undefined posterior, and
`sensitivity.md` records the same discovery on the diagnostics.

The conditioning is reported rather than hidden. The plan prints
`kappa=120001` against a true condition number of **120001.0000** — so the
modeller is told, in the very number they are told to divide a tolerance
by, how badly the data constrains the split. A refusal is not the only way
to be loud.

## 6. What this row does NOT cover

`PlanDiagnostics.rhat` against `chain_ess` — the two summarise different
quantities (a joint χ² trace against per-parameter ESS), and B7 already
records that the per-parameter version needs its own threshold argument
rather than rheplicant's 1.05. Multi-block partitions, the
Metropolis-within-Gibbs path (`steps=` on a gradient block), streaming
routes, and `nuts_on_collapse` are all untested here: one conjugate block
is the shape §四 4.2 names, and each of the others deserves its own
fixture rather than a mention.
