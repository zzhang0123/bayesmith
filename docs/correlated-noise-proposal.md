# Declaring a correlated noise on a graph node

An investigation, not an implementation. Nothing in `src/` was changed. Every
number below comes from a probe in `docs/probes/`, re-runnable as described in
§0.

`exact/precision.py` gave the package an `N^-1` that need not be diagonal, and
`solve.py` and `fisher.py` now consume it. The question is what it would take
for a NODE to declare one, such that `log_joint`, `wiener_solve` and
`fisher_information` all read ONE object.

**Short answer.** The gap is one `isinstance` at `exact/gaussian.py:100`, and
three things behind it that are less obvious than the gate:

1. numpyro **already ships the distribution we need** — `dist.CirculantNormal`
   — and its density agrees with `CirculantPrecision` to 1.5e-16 (§2). The
   density half of a correlated noise therefore already works today:
   `log_joint` and the numpyro bridge both return -12.422781 on the same
   correlated graph, and NUTS runs on it. It is only the EXACT path that
   refuses (§1.1).
2. `check_gaussian`'s probe cannot be widened in place. Its five scalar
   offsets are, for a stationary covariance, five points along **one
   direction** of an n-dimensional space, and two genuinely different
   circulant covariances agree at every one of them to 1.9e-16 (§3.2).
3. `CirculantPrecision` **cannot be constructed under a trace** — its
   `__check_init__` concretises — and the noise is built under a trace on the
   solve path (§3.4). That is a blocker for the fast path, not a detail.

Recommendation: **extract a `Precision` from the node's own distribution**
(option A, §4.1), and build the gradient-based guard FIRST (§5).

---

## 0. Re-running the probes

```bash
cd <this worktree>
for p in docs/probes/probe_*.py; do
    PYTHONPATH=$PWD/src /Users/zzhang/projects/bayesmith/.venv/bin/python "$p"
done
```

`docs/probes/OUTPUT.txt` is the captured output of exactly that loop. All eight
probes exit 0 and none raises.

| probe | question |
|---|---|
| `probe_1_what_the_chain_does.py` | what today's chain does with a correlated node, end to end |
| `probe_2_extraction_and_the_guard.py` | can a `Precision` be recovered from a numpyro distribution |
| `probe_3_what_a_correlated_guard_must_probe.py` | what the current probe family can and cannot see |
| `probe_4_traceability_and_cost.py` | can a `Precision` be built under trace; what the routes cost |
| `probe_5_the_one_object_seam.py` | how much of the one-object discipline is structural |
| `probe_6_sigma_census.py` | mechanical census of the per-sample-sigma vocabulary |
| `probe_7_what_the_feature_buys.py` | what ignoring correlation costs, in posterior units |
| `probe_8_what_the_gate_refuses.py` | what the type gate refuses that it arguably should accept |

Versions the numbers were measured against: numpyro 0.21.0, jax 0.11.1,
equinox 0.13.8. `dist.CirculantNormal` is the load-bearing one — re-measure §2
if numpyro moves.

---

## 1. Where the sigma vocabulary is enforced

### 1.1 One choke point, not twelve

`probe_1` runs the same two-node model three ways, changing only the observed
node's distribution:

| spelling | `log_joint` | numpyro `log_density` | `gaussian_parts` | `noise_std_at` | `linear_operator` |
|---|---|---|---|---|---|
| `Normal` | -10.631340 | -10.631340 | OK | OK | OK |
| `MultivariateNormal` | -12.422781 | -12.422781 | `NotGaussian` | `NotGaussian` | `NotGaussian` |
| `CirculantNormal` | -12.422780 | -12.422780 | `NotGaussian` | `NotGaussian` | `NotGaussian` |

Two things follow, and the second is the one that matters.

**The density layer is already correlated-capable.** `log_joint` and the bridge
agree to the digit on both correlated spellings, because both call the node's
own `log_prob` through `apply_probabilistic`. NUTS runs: `w = -0.068 +/- 0.707`
(MVN) and `w = -0.090 +/- 0.690` (circulant), 400 draws each. So "nothing can
DECLARE a correlated noise" is true of the exact path only — see §6.1, which I
am flagging rather than assuming was already known.

**Every exact refusal is the same refusal.** `noise_std_at`, `unchecked_
operator` and `linear_operator` all fail with the identical `NotGaussian`,
raised at one line:

```python
# src/bayesmith/exact/gaussian.py:100
if not isinstance(distribution, dist.Normal):
```

`probe_6` scans every `src/bayesmith/**/*.py` for the pattern and finds exactly
one such gate (plus the `dist.Independent` unwrap loop at `:77`). Widening the
vocabulary is not twelve edits at the gate; it is one.

### 1.2 The census: 23 signatures, 3 arithmetic sites

`probe_6` finds **23 functions** whose signature carries a sigma-shaped noise
parameter (`noise_std`, `sigma`, `sigma_of`, `scale`, `scales`), spread over 8
modules. But almost all of them only PASS a value through. Classified by what
they do with it:

**Load-bearing — a guard or a computation with a stated reason:**

| site | what it assumes | why it is load-bearing |
|---|---|---|
| `gaussian.py:100` | the distribution is a `Normal` | the classification gate; `NotGaussian` is an outcome, routing the block to NUTS |
| `gaussian.py:226` | `-0.5((x-loc)/scale)^2 - log scale - 0.5 log 2pi` | `check_gaussian`'s predicted density — the probe's whole content |
| `precision.py:121,124,127` | `r/sigma^2`, `sum log 2pi sigma^2`, `omega/sigma` | `DiagonalPrecision` — the intended home of the arithmetic, and the degenerate case everything else is checked against |
| `linearity.py:211,287,549` | departure from affinity measured in units of sigma | B1's fix: the second criterion the relative one cannot see. `_refuse_unusable_scale` names the node whose scale is unreadable |
| `fisher.py:131-158` | `(dlog sigma/dx)^T(dlog sigma/dx)` | the variance's own information; for a non-diagonal `N` this term is `1/2 tr(N^-1 dN N^-1 dN)` and the diagonal reduction stops applying |
| `fisher.py:267-278` | `noise_std` equals `sigma_of(centre)` | the ONE place two noise statements are reconciled rather than trusted |

**Incidental — a dict threaded through, no per-sample assumption:**
`solve.py:158,210,401,465` (`condition_bound`, `_conjugate_solve`,
`wiener_solve`, `gcr_sample`), `gibbs.py:146,249,327`, `gls.py:372,456`,
`correct.py:62`, `execute.py:361`, `classify.py:406`. These would change TYPE
and nothing else — they hand the value to `_weights`.

`diagonal_from` — the sigma-dict -> Precision-dict shim — is reached from
exactly **four** places (grep, confirmed): directly at `fisher.py:254`, and
through `solve.py:72`'s `_weights` wrapper from `solve.py:197`, `solve.py:225`
and `correct.py:124`. Those four are where a `{observed: Precision}` dict would
arrive ready-made instead of being manufactured.

**Two sites are in the PUBLIC return surface** and would change type visibly:
`GLSResult.noise_std` (`gls.py:206`) and `Estimate.noise_std`
(`execute.py:161`). Both are documented as "the covariance ... feed it back to
`gcr_sample`", which is precisely the role a `Precision` would take over.

**Out of scope but worth naming:** the PRIOR is diagonal too —
`block.py:110` and `fisher.py:283` build `S^-1` as `1/prior_std**2`. Nothing
in this proposal touches it, and a correlated prior is a separate exercise.

### 1.3 The one-object discipline today is call convention, not structure

`probe_5` builds a straight-line model whose node declares `sigma = 0.5`, then
hands `wiener_solve` and `fisher_information` a sigma that contradicts it:

| `noise_std` | `wiener_solve` w | reported posterior sd | `log_joint` at that w |
|---|---|---|---|
| x1.0 (honest) | +1.975278 | 0.106551 | -53.5719 |
| x2.0 | +1.972591 | 0.212956 | -53.5722 |
| x10.0 | +1.890294 | 1.042332 | -53.8900 |

Nothing refuses. The error bar scales exactly with the lie while the density
barely moves — the silent-wrong-answer shape. The only thing keeping the two
honest is that callers pass `noise_std_at(graph, ...)`.

There is exactly one exception, and it is narrow: `fisher_information` refuses
a `noise_std` that disagrees with `sigma_of(centre)` — measured, it raises —
but only when `sigma_of=` and `centre=` are supplied, which
`depends_on_prediction=False` lets a caller omit entirely.

The sharpest instance is `correct.py::log_weight`, which computes
`log_joint(graph, ...)` (density, from the node's distribution) and
`normal_operator(block, _weights(noise_std), ...)` (covariance, from the dict)
**in one expression**. Its own docstring already records `at` as
"**Unverifiable here**". If a node declared a correlated noise while the dict
stayed diagonal, that function would combine a correlated `log p` with a
diagonal `q` and return a confident wrong importance weight. This is the
defect B1 shape the one-object clause exists to prevent, and it is already
latent in the code — not created by this feature, but exposed by it.

### 1.4 What the feature buys

`probe_7` is pure numpy — no bayesmith code — so it is a statement about the
statistics. Two parameters (offset, slope) fit to n=256 samples under an
exponential circulant kernel, 200 realisations per row:

| corr. length | mean bias, offset (true post. sd) | mean bias, slope | sd_diag/sd_true, offset | sd_diag/sd_true, slope |
|---|---|---|---|---|
| 1 | 0.001 | 0.122 | 0.6432 | 0.6946 |
| 4 | 0.002 | 0.641 | 0.3474 | 0.4659 |
| 16 | 0.003 | 1.960 | 0.1763 | 0.6364 |
| 64 | 0.006 | 4.932 | 0.0952 | 1.2370 |

And the coverage that implies, at correlation length 64, 2000 realisations:

* nominal 95% interval, correlation-aware: **94.5%** coverage
* nominal 95% interval, diagonal only: **14.9%** coverage

A 10.5x too-narrow error bar and a 95% interval that covers 15% of the time.
That is the size of the gap the migration spec calls "the most conspicuous gap
in the physics".

---

## 2. Can numpyro express it? Yes — and better than expected

**numpyro 0.21.0 ships `dist.CirculantNormal`.** Signature:

```python
CirculantNormal(loc, covariance_row=None, covariance_rfft=None, *, validate_args=None)
```

It is the exact counterpart of `CirculantPrecision`. `probe_2` measures the
correspondence:

* `CirculantNormal(loc, covariance_row=k).log_prob(x)` vs
  `precision.log_density(CirculantPrecision(first_column=k), x - loc)`:
  **worst relative disagreement 1.547e-16** over 5 random draws (4 of the 5
  are bitwise equal).
* `.covariance_rfft` -> `[4.25, 2.798528, 1.45, 1.101472, 1.05]` is exactly
  `np.fft.rfft(kernel).real`, and `CirculantPrecision.eigenvalues` is its
  full-spectrum twin `[4.25, 2.798528, 1.45, 1.101472, 1.05, 1.101472, 1.45,
  2.798528]`.
* `.covariance_row` returns the kernel unchanged.

So **extraction is one attribute read**, exactly parallel to `Normal.loc` /
`Normal.scale`:

```
Normal            ->  (loc, scale)          ->  DiagonalPrecision(sigma=scale)
CirculantNormal   ->  (loc, covariance_row) ->  CirculantPrecision(first_column=covariance_row)
```

The class names even agree about what is and is not exact: numpyro named it
`CirculantNormal`, for the same reason `precision.py` refused to call its class
`Toeplitz`. There is no naming reconciliation to do.

`dist.MultivariateNormal` also works and agrees with the circulant to
3.27e-11 at n=4096 (`probe_4`), but it is not a viable carrier at scale — see
§3.5.

**Would the probe guard accept it?** No, twice over, and for two independent
reasons. It never reaches the probe (§1.1: the type gate refuses first), and
if the gate were relaxed the probe as written would refuse it anyway (§3.1).

---

## 3. What breaks the `log_prob` probe

`check_gaussian` verifies that the extracted `(loc, scale)` reproduce the
node's own density at five offsets, elementwise. Four separate things break
when the node is correlated. Only the first is obvious.

### 3.1 The comparison is elementwise; a correlated log_prob is a scalar

`CirculantNormal` has `event_shape (8,)`, `batch_shape ()`, and
`log_prob` returns shape `()` — one joint number, not one per sample.
`check_gaussian` does `jnp.broadcast_to(distribution.log_prob(probe), shape)`,
which silently replicates that scalar across all n entries and compares it
against a per-element diagonal formula.

`probe_2(b)` measures what that produces, taking the natural generalisation of
`scale` as `sqrt(diag N)`:

| offset | joint `log_prob` | per-element predicted | worst departure |
|---|---|---|---|
| -3.0 | -26.53784 | -5.76551 | 3.6029e+00 |
| -1.0 | -11.47902 | -1.76551 | 5.5018e+00 |
| +0.0 | -9.59667 | -1.26551 | 6.5832e+00 |
| +0.5 | -10.06726 | -1.39051 | 6.2400e+00 |
| +2.0 | -17.12608 | -3.26551 | 4.2445e+00 |

against a default rtol of 2.220e-13 at float64. So the guard would refuse —
loudly and for the wrong reason. It is comparing a joint density to a marginal
one. The comparison has to become scalar-to-scalar:
`log_prob(probe)` vs `precision.log_density(precision, probe - loc)`.

### 3.2 The probe DIRECTIONS are blind to a stationary covariance

This is the finding that decides the design.

For a stationary covariance the diagonal is constant, so `sqrt(diag N)` is a
constant vector and every probe `loc + offset * scale` is a displacement along
the SAME direction — the all-ones vector. Five points on one line.

`probe_3` constructs two circulant covariances that are identical along that
line by construction: same `lambda_0` (so the quadratic form at a constant
displacement matches) and same `sum log lambda` (so `log_normalizer` matches),
achieved by scaling `lambda_1` up by 1.6 and `lambda_3` down by 1.6, each of
which appears twice in the symmetric spectrum. Both remain positive definite.

```
eigenvalues A = [4.25, 2.798528, 1.45, 1.101472, 1.05, 1.101472, 1.45, 2.798528]
eigenvalues B = [4.25, 4.477645, 1.45, 0.688420, 1.05, 0.688420, 1.45, 4.477645]
lambda_0    A 4.250000000000   B 4.250000000000
sum log eig A 4.490318172083   B 4.490318172083
```

These are not close covariances: `||N_A^-1 - N_B^-1||_F / ||N_A^-1||_F =
0.4060`. What the probe family reports:

| offset | A | B | rel |
|---|---|---|---|
| -3.0 | -26.5378438223 | -26.5378438223 | 0.000e+00 |
| -1.0 | -11.4790202929 | -11.4790202929 | 0.000e+00 |
| +0.0 | -9.5966673517 | -9.5966673517 | 1.851e-16 |
| +0.5 | -10.0672555870 | -10.0672555870 | 0.000e+00 |
| +2.0 | -17.1260791164 | -17.1260791164 | 0.000e+00 |

**Worst over the whole family: 1.851e-16, against rtol 2.220e-13.** The guard
cannot tell them apart. A wrong circulant kernel would pass every probe.

Note that this is not fixed by adding offsets. More points on the same line
still measure the same two scalars. It is fixed only by changing the
DIRECTIONS.

### 3.3 The affordable replacement: one gradient, n equations

For any Gaussian, `grad log_prob(x) = -N^-1 (x - loc)` exactly. So a single
reverse-mode pass yields the full vector `N^-1 r`, which can be compared
against `precision.apply(r)` **elementwise** — n equations from one AD
evaluation, where a scalar probe gives one.

`probe_3(e)`, at a random displacement `r`:

* `-grad log_prob_A(loc+r)` vs `CirculantPrecision_A.apply(r)`:
  worst elementwise relative error **2.220e-16**
* `-grad log_prob_B(loc+r)` vs `CirculantPrecision_A.apply(r)`:
  worst elementwise relative error **1.882e-01**

A separation of **8.48e+14x** between the matching and the mismatched pairing —
on precisely the pair that the existing probe family reports as identical.

The other half, the normaliser, comes from `log_prob` AT the mode, which
`PROBE_OFFSETS` already includes as offset `0.0`:

| | `log_prob(loc)` | `-0.5 * log_normalizer()` | rel |
|---|---|---|---|
| A | -9.596667351679 | -9.596667351679 | 1.851e-16 |
| B | -9.596667351679 | -9.596667351679 | 3.702e-16 |

Both halves are needed and neither subsumes the other: A and B have the SAME
normaliser by construction, so that check alone cannot separate them either;
and the gradient is blind to the normaliser, which does not appear in it.

Together they pin the Gaussian exactly (given that the log-density is
quadratic, which linearity of the gradient checks for free: `grad` at `2r`
must be twice `grad` at `r`).

Cost, per observed node of size n:

| verification style | cost | numbers compared |
|---|---|---|
| scalar-offset family (today) | 5 x `log_prob` | 5 |
| random displacements | k x `log_prob` | k |
| **gradient probe** | 1 x `log_prob` + 1 x `grad log_prob` | n+1 |
| dense oracle | `dense(precision, n)` = n applies | n^2 |

A random-displacement family (`probe_3(d)`: 1.9e-3 to 7.8e-2 separation over 8
draws) also works and is cheaper per probe, but it is probabilistic and gives
one equation each. The gradient probe is deterministic, complete for a
kernel with at most n free parameters, and costs about 2x a `log_prob`.

### 3.4 `CirculantPrecision` cannot be built under a trace

`gaussian_parts` is documented as the traceable fast path — "fully traceable,
and it is what runs inside `jax.linearize` on every solve". Its correlated
analogue would have to produce a `Precision`. `probe_4(a)`:

| | under `jax.jit` | under `jax.linearize` |
|---|---|---|
| `DiagonalPrecision` | OK (16.000000) | OK |
| `CirculantPrecision` | **`ConcretizationTypeError`** | **`ConcretizationTypeError`** |

The cause is `__check_init__`, which does `float(jnp.min(eigenvalues))` to
refuse an indefinite kernel — a concretisation. The arithmetic itself is
perfectly traceable: the same FFT quadratic form without the check jits fine
(1.577428) and has a finite gradient.

And the noise really is built under a trace on the solve path. `probe_4(d)`
instruments an observed node's `dist_fn` and calls `jax.linearize(isolate(...))`
as `block.py` does — the types seen are `['LinearizeTracer']`.

So the constructor's guard would have to be split out, exactly the way
`gaussian_parts` / `check_gaussian` are already split, and for exactly the
reason that module's docstring gives: *"a guard that cannot run where the fast
path runs must run before it, on the values the fast path will see."* This is a
change to library code and I am describing it rather than making it (§5.2).

### 3.5 The dense route does not scale, so the carrier matters

`probe_4(b,c)`, best of 3 timed runs after warmup:

| n | circ `log_prob` | circ `grad` | dense `log_prob` | dense RAM |
|---|---|---|---|---|
| 256 | 0.037 ms | 0.040 ms | 0.029 ms | 0.0005 GB |
| 1024 | 0.027 ms | 0.031 ms | 0.232 ms | 0.008 GB |
| 4096 | 0.050 ms | 0.052 ms | 3.792 ms | 0.125 GB |
| 16384 | 0.137 ms | 0.191 ms | not built | 2.0 GB |
| 65536 | 0.424 ms | 0.765 ms | not built | 32 GB |
| 262144 | 1.980 ms | 3.629 ms | not built | 512 GB |
| 1048576 | **7.858 ms** | **15.813 ms** | not built | 8192 GB |

numpy Cholesky, which has no compile step to hide the scaling: 0.34 ms at
n=256, 8.60 ms at n=1024, **229.67 ms at n=4096**.

So the gradient guard costs ~16 ms once per block build at n=10^6 — clearly
affordable — while a `MultivariateNormal` carrier is unusable past a few
thousand samples. **The carrier must be `CirculantNormal`, not
`MultivariateNormal`.** MVN remains fine as a small-n oracle, where it agrees
with the circulant to 3.27e-11 at n=4096.

---

## 4. The honest options

All three satisfy the *arithmetic*. They differ in where the source of truth
lives, and therefore in whether the one-object clause holds by construction or
by guard.

### 4.1 Option A — extract the Precision from the node's own distribution

The node says what it already says; only the vocabulary widens:

```python
observe("d", lambda m: dist.CirculantNormal(m, covariance_row=k), mu, obs=data)
```

`gaussian_parts` returns a `Precision` instead of `(loc, scale)`;
`observation_parts` returns `{obs: Precision}`; `noise_std_at` becomes
`precision_at`. `Normal` continues to yield `DiagonalPrecision`, so the
diagonal path stays exercised by every existing test.

**One-object clause: satisfied by construction, in the strongest available
sense.** `log_joint` reads `distribution.log_prob`; the `Precision` is DERIVED
from that same distribution object and VERIFIED against its density (§3.3).
There is no second declaration to reconcile, because there is no second
declaration. This is the property the package already has for the diagonal
case, extended rather than replaced.

**Costs.**
* Rewrite `check_gaussian` around the gradient probe (§3.3). Real work, and
  the reason to do it first.
* Split `CirculantPrecision.__check_init__` (§3.4). Small, but library code.
* the 4 `diagonal_from` reach points become "the dict you were handed"; the
  remaining ~19 signatures change type only.
* `GLSResult.noise_std` and `Estimate.noise_std` change type in public API.
* `fisher.py`'s `2 (dlog sigma/dx)^T(dlog sigma/dx)` term has no correlated
  form yet — the general expression is `1/2 tr(N^-1 dN N^-1 dN)`, which for a
  circulant is a sum over eigenvalues but is NOT the diagonal reduction. Until
  it is derived and measured, a correlated node must be refused unless it
  declares `depends_on_prediction=False`. That refusal is cheap and honest;
  silently applying the diagonal formula is not.
* Ties the vocabulary to numpyro's class list. A covariance for which numpyro
  has no distribution cannot be declared. Today that means: diagonal and
  circulant, which is exactly what `precision.py` implements.

**Guarantee bought.** The density and the covariance are the same object, and
the guard is a complete elementwise check of `apply` plus an exact scalar check
of `log_normalizer`, not a five-point sample along one direction.

### 4.2 Option B — an explicit side-channel on the node

`Probabilistic` grows a field: `precision_fn: Callable[[prediction], Precision]`,
the migration spec's `at(prediction) -> Precision`. `dist_fn` still produces
the density.

**One-object clause: violated unless a guard restores it.** Two independent
declarations of the covariance now exist, and §1.3 measured what happens when
two noise statements are not compared: `wiener_solve` returns a posterior sd of
1.042332 instead of 0.106551 and nothing objects. The spec's argument that
`at(prediction) -> Precision` makes the property structural is about the shape
of the noise object's own interface — it does not, on its own, bind that object
to `dist_fn`, and in bayesmith `log_joint` reads `dist_fn`.

Restoring the clause requires a guard that checks `precision_fn`'s output
against `dist_fn`'s density — which is **exactly the gradient probe option A
needs anyway**. So B costs strictly more than A: the same guard, plus a new
node field, plus a new failure mode (a node whose two halves disagree) that A
cannot have.

**What B buys that A does not:** freedom from numpyro's class list. A user
could supply any object satisfying the `Precision` protocol. `probe_8` confirms
the protocol is structurally checkable — `isinstance(DiagonalPrecision(...),
Precision)` and `isinstance(CirculantPrecision(...), Precision)` are both
`True`, `isinstance(dist.Normal(...), Precision)` is `False`. Caveat:
`runtime_checkable` checks method NAMES only, not signatures or behaviour, so
that check is a spelling check, not a guarantee.

### 4.3 Option C — generate the distribution FROM the Precision

Invert A: the node declares a `Precision`, and `dist_fn` is generated from it
(`CirculantPrecision(k)` -> `CirculantNormal(m, covariance_row=k)`).

**One-object clause: satisfied by construction, trivially** — there is only one
object and the density is a view of it.

**Costs.** It closes the `Precision` protocol: only implementations with a
matching numpyro distribution could ever be declared, so the open Protocol in
`precision.py` becomes a closed enum in practice. It also inverts the
declaration direction the package uses everywhere else — `dist_fn` is the one
statement of the prior as well as the noise, and `solve.py`'s docstring rests
on that ("there is no keyword to override it, and therefore no way for the
exact exit and NUTS to target different posteriors"). Introducing a second way
for a node to acquire a density is a larger change to the package's shape than
the feature warrants.

### 4.4 Comparison

| | A: extract | B: side-channel | C: generate |
|---|---|---|---|
| one-object clause | by construction | by guard only | by construction |
| needs the gradient guard | yes | yes | no (nothing to check) |
| new node field | no | yes | yes |
| new disagreement failure mode | no | yes | no |
| covariances expressible | numpyro's list | anything | numpyro's list |
| changes how a node declares a density | no | no | yes |

---

## 5. Recommendation, and what to build first

**Take option A.** It is the only one that satisfies the load-bearing clause by
construction without changing what a node is. B pays for openness with the
exact failure mode the clause exists to prevent, and the openness is
speculative — `precision.py` ships two implementations and numpyro carries a
distribution for both.

### 5.1 Build the guard first, not the plumbing

The first increment should be **`check_precision`** — the gradient probe of
§3.3 — and nothing else.

Reasons, in order:

1. **It decides whether A is sound at all.** A rests on the claim that a
   `Precision` extracted from a distribution can be verified against that
   distribution's density completely and affordably. §3.3 measured that claim
   (8.48e+14x separation, ~16 ms at n=10^6), but a measurement in a probe
   script is not a guard in the suite.
2. **It needs no change to the solve path.** It takes a distribution and a
   `Precision` and returns errors. It can be written, tested and mutation-
   checked before `gaussian_parts` moves an inch.
3. **It is the piece that would otherwise be written last and weakest.** The
   existing probe family is the cautionary case: it is a genuinely careful
   guard that nonetheless constrains only TWO scalars of the covariance
   (`lambda_0` and `sum log lambda`, §3.2). For the n=8 fixture that leaves 3
   of the 5 independent spectral parameters unconstrained; the family grows
   as `n/2 - 1`. Nothing in the suite would reveal that, because no fixture is
   correlated.
4. **It retroactively strengthens the diagonal path.** A gradient probe applied
   to a `Normal` node checks `1/sigma^2` elementwise at every entry, where the
   present probe checks the density at five offsets. Worth measuring whether it
   subsumes `PROBE_OFFSETS` or complements it — I did not test that, and it
   should not be assumed.

Acceptance for that increment, in this package's idiom:

* the diagonal case degenerates numerically — `check_precision` on a `Normal`
  node must agree with `check_gaussian` on the same node;
* the A/B fixture of §3.2 is the anti-vacuity clause: a test that PASSES a
  mismatched circulant is a test that is not testing;
* pin the separation, not just the pass — assert the mismatched pairing reports
  >= 1e-2 while the matched one reports <= 1e-12, so the guard's power is
  measured and not merely its verdict;
* the normaliser check and the operator check must each fail alone on the A/B
  pair (both were measured to be individually blind to it, §3.3).

### 5.1a Step 4's cost, attempted and measured (2026-08-25)

Step 4 below reads as a rename. It is not, and the attempt was reverted rather
than half-landed. Measured before starting: **111 call sites** pass
`noise_std=` (19 in `src/`, 92 in `tests/`), 77 of them the single expression
`noise_std=sigma`.

What makes it more than a rename is that **tests use a producer's output two
ways**, and both are legitimate:

* as an OPERATOR, fed to a solver;
* as sigma VALUES -- arithmetic (`sigma ** 2`), elementwise comparison against
  an oracle, assertions about how sigma moves with a latent.

A blanket conversion of the producers therefore breaks the second use. The
attempt did exactly that and the failures name it precisely: 51 x
`TypeError: unsupported operand ** on DiagonalPrecision`, 27 x
`AttributeError: ArrayImpl has no attribute 'apply'`, 15 x a stale
`noise_std` reference, spread over eight test files.

Two things are worth keeping from the attempt:

1. **The failure mode is loud, which makes the migration safe to do
   mechanically.** A raw sigma dict has no `.apply` and a `Precision` has no
   `** 2`; nothing fails silently. That is not luck -- it is what having two
   incompatible types buys, and it is the argument for step 4 rather than for
   type-punning `noise_std=` into meaning both.
2. **`GLSResult.precision`** is the right shape for the GLS boundary: the loop
   iterates sigma values honestly, and the conversion belongs once where its
   answer is handed on, not at each call site.

**The correct order is per-site, not a sweep**: for each of the 111, decide
whether that expression is an operator or a value. A regex cannot make that
distinction, which is the same reason the docs guard had to become an AST walk
-- and the irony of using one here, in a refactor whose entire purpose is to
stop two vocabularies being confused, is worth recording.

### 5.1b Step 4 landed, per-site (2026-08-25)

Done the way 5.1a prescribes. What the second attempt found that the first
did not:

* **The census was incomplete and it did not matter.** 104 of the 111 are
  `noise_std=` KEYWORD arguments; the rest are prose. But
  `_iid_draws(block, sigma, ...)` in `dispatch/execute.py` passes the noise
  POSITIONALLY, so no keyword census could see it. It surfaced as
  `AttributeError: ArrayImpl has no attribute 'apply'` on the first run. That
  is the argument for step 4 restated: an incomplete plan is survivable
  precisely because the failure is loud.

* **Only SIX call sites are genuinely mixed** -- far fewer than the first
  attempt's 93 failures suggested. The difference is that the first attempt
  converted the PRODUCERS (`noise_std_at` -> `precision_at`), which breaks
  every value use downstream; keeping `noise_std_at` and adding
  `precision_at` beside it leaves the value uses untouched by construction.
  The six, each converted at the call rather than at the variable:

  ==================================  ================================
  site                                the two uses, together
  ==================================  ================================
  `dispatch/classify.py:492`          `_data_informed_point` (operator)
                                      and `_relative_movement` (values),
                                      three lines apart
  `exact/fisher.py`                   weights the design (operator) and
                                      is checked against `sigma_of`
                                      (values) -- in one function body
  `exact/gls.py::iterative_gls`       iterates values, hands on an
                                      operator: `GLSResult.precision`
  `tests/exact/test_correct.py:592`   `wrong["d"] / right["d"]`
                                      elementwise, then both solve
  `tests/exact/test_correct.py:631`   `{k: v * factor}` -- the wide
                                      variant is built by SCALING sigma
  `tests/dispatch/test_acceptance.py` `_dense_at` ravels sigma into the
                                      analytic posterior, then solves
  ==================================  ================================

* **`node_shape` was an unlisted blocker, and the spec's claim that the seam
  is "the only thing standing between a correlated node and the solver" is
  false.** Measured: `linear_operator` -> `check_linearity` -> `noise_std_at`
  -> `observation_parts` -> `node_shape` -> `gaussian_parts` -> `NotGaussian`.
  SHAPE refused a correlated node, before any covariance was read. `loc` is
  all `node_shape` needs, so it now reads it through `_loc_of`; the gate stays
  where the covariance is.

* **A live defect in `precision_parts`, found on the way.** It broadcast the
  diagonal sigma to `loc`'s shape rather than to `node_shape`. For the shape
  `node_shape`'s own docstring names -- a plated node whose `dist_fn` takes no
  plated parent, so `loc` is scalar and the value is plated -- `apply` stayed
  right by broadcasting while `log_normalizer` summed ONE term instead of `n`.
  Measured at n=6: 0.4515827298 against 2.70949626, short by exactly the plate
  size. That is defect B1's shape arriving through a broadcast rather than
  through two objects, and no fixture could see it because every other one's
  loc is already plate-shaped.

* **The `fisher_information` redundancy check is now operator-vs-operator**,
  both applied to one fixed probe, because `precision` need not have sigma
  arrays to compare elementwise. Swept: it is if anything SHARPER than the
  elementwise check it replaces, refusing a 1e-6 perturbation the old one
  accepted (`apply` divides by sigma**2, roughly doubling the relative gap).

**What step 4 bought, measured.** A `CirculantPrecision` handed to
`wiener_solve` now solves to the dense Wiener filter at **1.4e-16** relative,
against a reference built by `precision.dense` (materialised by application)
and inverted in NumPy -- sharing no arithmetic with CG. `gcr_sample` at the
same covariance reproduces the exact posterior variance within Monte-Carlo
error. `probes/probe_5_the_one_object_seam.py` section (c) recorded that
NEITHER `Precision` could be passed to a solver at all; re-run it, and both
are accepted. Section (b) is unchanged and deliberately so: widening the TYPE
did not make the one-object discipline structural, and was never claimed to.

**What it did NOT buy: a correlated node still cannot be DECLARED on a graph
and solved.** Three blockers remain, all on the value side, all measured:

1. `linearity.py:735` -- `check_linearity` reads `noise_std_at` for its
   per-sample UNIT ("departure from affinity in units of scale"). A circulant
   has one: `sqrt(diag N)`, constant because it is stationary. The `Precision`
   protocol does not expose it, and adding a fourth operation is a design
   decision, not a fix.
2. ~~`block.py:427` -- `unchecked_operator` probes every observed node with
   `check_gaussian`, the DIAGONAL guard.~~ **Done.** That probe is now
   `check_observed`, which routes: a `Normal` to `check_gaussian` unchanged,
   a `CirculantNormal` to `check_positive_definite` and then
   `check_precision`. See 5.1c.
3. `block.py:345,434` -- `isolate` and the data walk both go through
   `observation_parts`, which computes a `scale` neither of them uses.

None is large. All three are the value side of the same seam, and each wants
its own measurement, so they are increment 5 rather than the tail of step 4.

### 5.1c `check_positive_definite` has a caller (2026-08-25)

Split out of `__check_init__` in `296d911` so the class could be traced,
which left it correct, mutation-checked and called by nothing. Wired into
`unchecked_operator`'s observed-node probe through a new
`gaussian.check_observed`, which routes by distribution rather than
duplicating either guard.

**The ORDER inside it is load-bearing, and it is about the diagnosis rather
than the verdict.** `check_precision` does refuse an indefinite kernel on its
own -- but only through NaN propagation. Measured on three indefinite
kernels, including one whose entries are all positive: numpyro's
`CirculantNormal.log_prob` returns `nan`, so `check_precision` reports
`linearity=nan, normalizer=nan`, and its message explains `linearity` as
"the log-density is not quadratic, so it has no covariance to extract". The
log-density IS quadratic; the kernel describes no realisable process. A user
sent to look at their `det` nodes and `linear_in` declarations would be
looking in the wrong place. `check_positive_definite` runs first and says
what is actually wrong. Refusing through a NaN is also fragile in a way that
saying so is not.

**Wiring it at the CLASSIFIER too was a regression, and the suite could not
see it.** `classify._is_gaussian` asks a different question -- "can the exact
path solve a block containing this node?" -- and for a correlated node the
answer is still no, because the block builder's data and loc walks are
diagonal-only. With `check_observed` there, `compile()` on a well-formed
`CirculantNormal` graph stopped returning a NUTS plan and raised
`NotGaussian` from deeper in the builder. **All 748 tests stayed green
through it**, because nothing in the suite compiles a correlated graph.
`test_a_correlated_graph_still_compiles_to_nuts_rather_than_raising` is that
missing test, and `_is_gaussian`'s docstring now says why the two call sites
are not the same call and must not be tidied into one.

Construction itself still does not validate, and that is not a gap this
closes: the class has to stay traceable, so
`test_construction_itself_does_not_validate_and_says_so` stays true for as
long as the split does.

### 5.1d The correlated variance-information term (2026-08-25)

Done symbolically first (`docs/derivations/variance_information_*.wls`, run
with `wolframscript`), then measured against a dense finite-difference Fisher
matrix (`docs/probes/probe_9_correlated_variance_information.py`) — an
independent route sharing no algebra, no FFT and no autodiff with the
implementation.

**The partial hand-derivation in the handover is exactly right**, and
Mathematica confirms every step of it at general symbolic `n=3` and at
explicit `n = 2,3,4`:

* `N^-1 d_a N = D^-1 C^-1 G_a C D + G_a` with `G_a = D^-1 d_a D` — identity
  holds identically;
* for a single scaling parameter (`mu = theta x`), `N^-1 d_th N = 2I/theta`
  exactly, for ANY `C`;
* the factor is `1 + 2 f^2 n / (1^T C^-1 1)`, reducing to `1 + 2 f^2` at
  `C = I`, and `1^T C^-1 1 = n/lambda0` for a circulant.

Measured against the dense route: `2.3096380799` predicted vs
`2.3096380799` found, `6e-12` relative.

**But that factor belongs to a model this package cannot express.** It comes
from `N = D C D` with a PER-SAMPLE `D = diag(f mu_i)`, which is not
stationary unless every `mu_i` is equal — so it is not a `CirculantNormal`.
What a `CirculantNormal` *can* express is a covariance that stays circulant
while its kernel depends on the prediction, and that has no universal scalar
factor at all (measured: `1 + 2 f^2 n/(x^T C^-1 x)` for a common scale, which
does not reduce to anything kernel-only). **A "correlated factor" is the
wrong shape for the answer.**

**The right shape covers both accepted rows at once.** Whenever `N`'s
eigenBASIS does not move with the parameters,

    1/2 tr(N^-1 d_a N N^-1 d_b N) = 1/2 sum_k d_a log lam_k d_b log lam_k

and both rows qualify: `I` for a `Normal`, the DFT for a `CirculantNormal`
whatever its kernel does. Verified symbolically for a circulant whose kernel
changes SHAPE with `theta` at `n = 3,4,5` (`exact - spectral` simplifies to
`0`), and the shipped diagonal rule
`2 (dlog sigma/dx)^T (dlog sigma/dx)` is that identity at
`lam_i = sigma_i^2` — not a different rule.

The identity also predicts its own failure, and the prediction was checked: a
covariance whose eigenbasis DOES move (`D(theta) C D(theta)`, per-sample `D`)
is not covered — measured, exact `2.7847` against spectral `2.7595`.

**How wrong the shipped per-sample form is on a correlated node**, measured
against the dense reference at `n=12`:

  ====================================  ========  ==============  =======
  case                                  truth     per-sample form  ratio
  ====================================  ========  ==============  =======
  parameter moves amplitude AND shape   10.6646   6.0000          0.563
  parameter moves shape only            3.4445    0.0000          0.000
  ====================================  ========  ==============  =======

`sqrt(diag N)` is CONSTANT across samples for a stationary covariance, so
that form can only see how the diagonal moves — one number, `n` times over.
When the kernel's shape moves while its diagonal does not, it reports
**exactly zero**. Both errors are too SMALL, making `F^-1` too WIDE, which is
the direction that reads as safe.

**What landed.** `Precision.log_spectrum()` is a fourth operation returning
`log lam_k` as a vector; `fisher._log_sigma_curvature` became
`_log_spectrum_curvature`, jacobian-ing `1/2 log lam_k`; and
`fisher_information` gained `precision_of=`, the general form of the rule,
with `sigma_of=` wrapped in `diagonal_from` so there is ONE curvature
implementation rather than two. The diagonal answer is unchanged **bitwise**:
`log_spectrum` returns `2 log sigma` rather than `log(sigma**2)` precisely so
that halving it is exact.

Measured end to end: `fisher_information` with `precision_of=` on a circulant
whose kernel changes shape agrees with the dense finite-difference Fisher to
`1.4e-10` — the difference floor. On that fixture the variance term is 71% of
the total, so omitting it would have widened the error bar by 1.86x.

`log_normalizer` was deliberately NOT rewritten to call `log_spectrum`:
measured, `sum(log(2 pi sigma^2))` is not bitwise
`n log 2 pi + sum(2 log sigma)` (one ULP apart at `sigma = 1e-8`), and moving
a shipped number to remove a duplicate is the wrong trade. The duplicate is
guarded instead, by `test_log_normalizer_is_the_log_spectrums_own_sum`, which
is the only thing rendering the two definitions side by side.

### 5.2 Then, in order

2. **Split `CirculantPrecision`'s constructor check** into a `check_circulant`
   that runs outside the trace, leaving `__check_init__` trace-safe. Measured
   blocker (§3.4). This is a library change; I am describing it, not making it.
3. **Widen `gaussian_parts`** to return a `Precision`, with the gate accepting
   `Normal` -> `DiagonalPrecision` and `CirculantNormal` -> `CirculantPrecision`
   and refusing everything else with the same `NotGaussian` (still a
   classification outcome, still routed to NUTS).
4. ~~**Rename the seam**: `noise_std_at` -> `precision_at`,
   `observation_parts` returning `{obs: Precision}`.~~ **Done, but as a SPLIT
   rather than a rename** -- see 5.1b. `noise_std_at` stays: `iterative_gls`
   iterates sigma VALUES and an operator has none, so `precision_at` is added
   beside it. The 4 `diagonal_from` reach points (`solve.py::condition_bound`,
   `solve.py::_conjugate_solve`, `correct.py::log_weight`,
   `fisher.py::fisher_information`) do stop manufacturing and start receiving.
5. ~~**Refuse a correlated node with `depends_on_prediction=True`**~~
   **Done, and the refusal is deleted** -- see 5.1d. The correlated form is
   derived (Mathematica), measured (dense finite-difference Fisher) and
   implemented, and it turned out to cover BOTH accepted rows rather than
   being a correlated special case.
6. Only then the evidence layer's whitening row, which the spec says was
   waiting on this interface's shape.

---

## 6. Flagged findings

Three things that are findings in their own right rather than steps in the
plan. Two of them meet the "stop and ask" conditions in my brief.

### 6.1 A correlated noise is ALREADY expressible for the density

I was told "nothing can yet DECLARE a correlated noise on a graph node". That
is true of the exact path and NOT true of the density. `probe_1`, working
example, no library change:

```python
observe("d", lambda m: dist.CirculantNormal(m, covariance_row=kernel), mu, obs=data)
```

`log_joint` returns -12.422780; the numpyro bridge returns -12.422780; NUTS
samples it (`w = -0.090 +/- 0.690`, 400 draws). `dist.MultivariateNormal`
works identically (-12.422781, agreeing with the circulant to 1e-6 on the same
kernel).

I do not think this changes the plan — the exact path is the point of the
exercise, and it refuses — but it changes the framing: this is **widening an
existing capability to the exact path**, not adding a capability. It also means
a user can today write a correlated model, get a correct NUTS answer, and never
learn that the exact path silently declined to consider it. Which leads to:

### 6.2 The gate refuses six spellings of one density — four of them wrongly

This is the "refuses something it arguably should accept" case, and I am
flagging rather than working around it, as instructed.

`probe_8` builds six spellings of the IDENTICAL density `N(m, diag(0.5^2))`.
They agree with each other to at most 1.776e-15. Four are refused:

| spelling | `log_prob` | vs reference | exact path |
|---|---|---|---|
| `Normal` | -9.814748115868 | 0.000e+00 | **ACCEPTED** (probe error 1.879e-16) |
| `Normal.to_event(1)` | -9.814748115868 | 0.000e+00 | **ACCEPTED** (probe error 1.879e-16) |
| `MultivariateNormal(diag)` | -9.814748115868 | 1.776e-15 | REFUSED `NotGaussian` |
| `MVN(scale_tril=diag)` | -9.814748115868 | 1.776e-15 | REFUSED `NotGaussian` |
| `CirculantNormal(s*I)` | -9.814748115868 | 0.000e+00 | REFUSED `NotGaussian` |
| `TransformedDistribution` | -9.814748115868 | 1.776e-15 | REFUSED `NotGaussian` |

And the consequence is visible in the plan:

```
Normal                     ->  block 0  {w}   GCR exact   linear_in ✓ 3 scales x 3 a...
MultivariateNormal(diag)   ->  block 0  {w}   NUTS
```

The same model, the same density to 1.8e-15, loses the exact path.

The asymmetry is worth stating precisely, because `gaussian.py`'s docstring
argues the opposite direction convincingly: *"Reading `.loc`/`.scale` off a
`Normal` trusts the type ... the type is evidence, not proof. The probe is what
raises the bar."* True for REFUSAL. For ACCEPTANCE the type is dispositive —
`gaussian_parts` refuses on `isinstance` at `:100`, before `check_gaussian`
evaluates any density at all. **The probe can only ever refuse a `Normal` whose
`log_prob` disagrees; it can never accept a non-`Normal` whose `log_prob`
agrees exactly.**

Note the third row especially: `CirculantNormal` with kernel `[s^2, 0, 0, ...]`
IS `s*I`, and `precision.py`'s own test suite already relies on that identity
("a circulant with kernel `[s, 0, 0, ...]` IS `sI`, the one fixture where both
are defined"). The package asserts the equivalence in its tests and refuses it
at its gate.

I did not work around this and I am not proposing to fix it as part of the
feature. But it is the same line the feature has to touch, and the gradient
probe of §3.3 is precisely the mechanism that would let acceptance be decided
by density rather than by class. Whether it SHOULD be is a judgement about how
much rope to give a user, and that is yours to make: accepting by density means
accepting arbitrary `Distribution` subclasses on the strength of a probe, which
is a real widening of the trust surface even with a complete elementwise check.

### 6.3 `correct.py::log_weight` mixes the two vocabularies in one expression

Already described in §1.3. Not created by this feature — it is latent today —
but it is the one function where `log_joint`'s density and `_weights`' dict
meet, so it is where a half-migrated correlated node would produce a wrong
answer quietly. It should be on the checklist for step 4 of §5.2 regardless of
which option is taken.
