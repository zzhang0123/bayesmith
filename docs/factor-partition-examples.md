# From a model to an auto-partitioned sampler: two worked examples

Two models, worked end to end: written down, traced, handed to
`factor_partition`, and sampled by `sample_factors`. Every partition printout
and every posterior number on this page was produced by running the code shown
— nothing is sketched. The two are graded: the first exercises the three
routing outcomes (`gcr`, `log-gcr`, `nuts`) on a flat model; the second adds a
hierarchy, which brings in the one routing rule the first cannot show.

Both carry multiplicative radiometer noise at `F = 4.05e-3` — a 61 kHz channel
at 1 s — which is what makes the log-space route live: see
`bayesmith.exact.loglinear` for the transform and its measured first-order
caveat.

---

## Example 1 — three factors, three routes

$$d = e^{Ax}\,\bigl[B y + e^{Cz}\bigr]\,(1 + F\,w)$$

`A`, `B`, `C` are known matrices; `x`, `y`, `z` are parameter vectors; `w` is
unit white noise. One data vector, three latents, and each latent has a
different conditional structure — which is the whole point of a factor
partition.

### The model, as a graph

```python
import jax, jax.numpy as jnp
import numpyro.distributions as dist
from bayesmith import trace, sample, det, observe
from bayesmith.dispatch.factor import factor_partition, sample_factors

N  = 32
xi = jnp.linspace(-1.0, 1.0, N)
A  = jnp.stack([xi, xi**2 - jnp.mean(xi**2)], axis=1)   # no constant column
B  = jnp.stack([jnp.ones(N), xi, xi**2], axis=1)
C  = jnp.stack([xi, xi**3], axis=1)
F  = 4.05e-3

def model(data):
    x = sample("x", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
    y = sample("y", lambda: dist.Normal(jnp.array([5.0, 1.0, 0.8]), 0.5).to_event(1))
    z = sample("z", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
    s  = det("s",  lambda y_, z_: B @ y_ + jnp.exp(C @ z_), y, z, linear_in=("y",))
    mu = det("mu", lambda x_, s_: jnp.exp(A @ x_) * s_,     x, s, linear_in=("s",))
    observe("d", lambda m: dist.Normal(m, F * m), mu, obs=data)

graph = trace(model, data)
plan  = factor_partition(graph)
print(plan)
```

Note what is and is not declared. `linear_in=("y",)` on `s` and
`linear_in=("s",)` on `mu` are the two claims that ARE true — and they are
checked, not trusted. Nothing is declared about `x` at all: there is no
`log_linear_in`, deliberately, because that property is discovered by probe.

And note `y`'s prior width, 0.5, because it is load-bearing rather than
cosmetic. At width 2.0 this prior puts real mass on a NEGATIVE summed sky
$By + e^{Cz}$ — and where the sky is negative, $\log\mu$ does not exist, so
the log route for `x` genuinely fails on part of the prior's support. The
probes then answer according to which prior draws they happen to take, and a
renaming of the latents (which reseeds the draws) can flip the partition.
That is not probe fragility; it is the probe faithfully reporting that
"log-linear over this prior" was only half true.
`tests/dispatch/test_factor.py::TestGenerality` pins the repaired version:
the partition is invariant under renaming and reordering, the log route
follows the exponential when the model is rearranged, and the routing tracks
the noise structure — which is what "derived by general rules" has to mean.
The modelling rule: **a latent you want the log route for needs a prior that
keeps the prediction positive**, and a prior that does not is a statement
worth hearing.

### What the probes conclude, latent by latent

Write $s = By + e^{Cz}$, so $\mu = e^{Ax}\,s$ and
$\log\mu = Ax + \log s$.

| latent | original space | log space | verdict |
|---|---|---|---|
| `x` | $e^{Ax}s$ — not affine | $Ax + \text{const}$ — **affine** | **`log-gcr`**, discovered |
| `y` | $e^{Ax}By + \text{const}$ — **affine**, but $\sigma = F\mu$ moves with it (measured movement 0.293) | $\log(By + \dots)$ — not affine | **`nuts`**, with the movement number and the remedy in its reason |
| `z` | $e^{Cz}$ inside a sum — not affine | $\log(\text{sum})$ — not affine | **`nuts`**, with both probes' departures quoted |

The printed plan, verbatim:

```text
block 0  {x}  log-gcr
block 1  {y, z}  nuts  ('y': sigma moves with block ('y',) (relative movement
  0.293 > 1e-08); the frozen-sigma draw is only a proposal there, and this
  module does not sweep 'gcr+mh' -- see its docstring. Use
  bayesmith.exact.gibbs.assemble for that block alone, or accept NUTS.;
  'z': log(prediction) not affine alone: ...departure... 1x -> 1.35e-01 |
  sigma-weighted 1.62e+00 ...)
```

Two of this plan's lines carry a lesson each. `y`'s: being affine is not
enough for an exact draw when the noise level itself moves with the block —
the frozen-sigma Gaussian is then only a *proposal*, and this module refuses
to run it without the Metropolis correction rather than run an unargued
sampler. `z`'s: the sigma-weighted criterion (departure `1.62` in units of the
noise) is what makes the refusal decisive — at `F = 4e-3` a curvature that
looks small *relatively* is enormous in units of what the likelihood can
resolve.

### Sampling it, and the caveat that matters more than the partition

```python
draws = sample_factors(graph, plan, jax.random.key(1),
                       num_warmup=2000, num_samples=3000)
```

`x` is updated by an exact log-space GCR draw every sweep; `y, z` advance
jointly under the inner NUTS kernel between sweeps. Measured, at the truths
$x=(0.3,-0.2)$, $y=(5,1,0.8)$, $z=(0.5,-0.3)$:

```text
x: [ 0.352 -0.189] +- [0.019 0.063]   |pull| max 2.69
y: [ 5.034  0.957  0.677] +- [0.135 0.216 0.368]   |pull| max 0.33
z: [ 0.253 -0.386] +- [0.261 0.064]   |pull| max 1.35
prediction-space residual rms / (F mu) = 0.49
```

The prediction-space residual — the fitted model against the truth, in units
of the noise — is well under 1: the model is fit. The parameter-space pulls of
2–3 at these chain lengths are the signature of this model's own geometry:
`identifiability` reports nullity 0 of 7 but `weakest_identified = 5.4e-4` —
no exact degeneracy, and one direction constrained thousands of times more
weakly than the rest, because the shape of $e^{Ax}$ can be partly traded
against the shape of $s$. Two consequences to internalise:

* **A chi-squared-based diagnostic cannot see motion along that valley** —
  chi-squared is flat there by construction. Run a second chain from a
  different start; two tight "posteriors" sitting far apart is the cheap,
  decisive tell. (Measured on a three-single-site Gibbs variant of this model:
  two 400-sweep chains each reported sub-2% widths while sitting sixteen of
  those widths apart.)
* **Keep the strongly-coupled pair in ONE block.** `sample_factors` already
  does the right thing here — `y, z` advance jointly under NUTS — which is
  why the numbers above are honest. Splitting a correlated pair into
  single-site exact blocks is the arrangement that mixes worst, however exact
  each conditional draw is.

---

## Example 2 — two kinds of "noise", and a hierarchy

$$d = e^{Ax}\,\bigl[B\,w_1 + e^{Cz}\bigr]\,(1 + F\,w_2),
  \qquad w_1 \sim p(w_1 \mid y)$$

Superficially the same model with $y \to w_1$ — but $w_1$ is now a *stochastic
field* whose statistics are set by a hyperparameter $y$, and that changes what
kind of object each "noise" is:

* $w_2$ — noise you only ever **weight by**. It never appears as a latent: it
  IS the observation's scatter, encoded once in the `Normal`'s scale, and
  inference integrates over it analytically.
* $w_1$ — noise you **resolve**. It multiplies into the prediction sample by
  sample, so it must be a latent with a prior, and inference marginalises it
  by sampling it — alongside $y$, the parameter of that prior.

The modelling rule of thumb: *scatter you only weight by goes in the
observation's scale; structure you need to know goes in a latent.*

### Both parameterisations of $w_1$'s statistics

```python
def model(data):
    y = sample("y", lambda: dist.Normal(jnp.array([5.0, 1.0, 0.8]), 0.5).to_event(1))

    # linear: y sets the MEAN of the field
    w1 = sample("w1", lambda y_: dist.Normal(D @ y_, 0.3).to_event(1), y)
    # -- or nonlinear: y sets the SCALE, through an exponential --
    # w1 = sample("w1", lambda y_: dist.Normal(m0, 0.3 * jnp.exp(0.2 * (y_ - y0))).to_event(1), y)

    x = sample("x", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
    z = sample("z", lambda: dist.Normal(jnp.zeros(2), 0.5).to_event(1))
    s  = det("s",  lambda w_, z_: B @ w_ + jnp.exp(C @ z_), w1, z, linear_in=("w1",))
    mu = det("mu", lambda x_, s_: jnp.exp(A @ x_) * s_,     x, s, linear_in=("s",))
    observe("d", lambda m: dist.Normal(m, F * m), mu, obs=data)
```

### The partition — identical for both parameterisations, and why

```text
block 0  {x}  log-gcr
block 1  {y, w1, z}  nuts  ('y': 'y' is an ancestor of another latent's
  distribution; 'w1': sigma moves with block ('w1',) ...; 'z': ...)
```

`x` and `z` route exactly as in Example 1. The two new verdicts:

* **`w1`** is affine in the prediction and Gaussian given `y` — a textbook
  conjugate block *shape* — and is refused for the same reason `y` was in
  Example 1: the multiplicative noise's sigma moves with it. Declare the
  observation `LogNormal` instead of multiplicative-`Normal` and this
  obstruction changes character (the original-space route closes, the
  log-space one stays closed because $\log(Bw_1+\dots)$ is not affine);
  either way, `nuts`.

* **`y` is ejected before any probe runs**, in BOTH parameterisations, by the
  ancestry rule: *a latent that is an ancestor of another latent's
  distribution may not join an exact block.* The reason is not caution but
  arithmetic — `y`'s full conditional is
  $p(y)\,p(w_1\mid y)$; an exact block solves only against **observed**
  nodes, so it would drop the $p(w_1\mid y)$ factor *silently* and converge,
  quietly, to the wrong posterior. The linear and the nonlinear case differ
  only in what is lost: nonlinearly, `nuts` is simply correct; linearly,
  $y \mid w_1$ is itself Gaussian — a conjugate update with the *latent*
  $w_1$ as its data — which the dispatcher does not yet cut. That is a real,
  bounded piece of future work (hierarchy-internal blocks), and until it is
  written the ejection is the honest verdict: a slower right answer over a
  fast wrong one.

### Sampling it

```python
draws = sample_factors(graph, plan, jax.random.key(1),
                       num_warmup=1500, num_samples=2000)
```

Measured, linear parameterisation, against the truths (with $w_1$'s truth one
realisation of its own prior at $y=(5,1,0.8)$):

```text
x:  [ 0.239 -0.101] +- [0.023 0.05 ]                    |pull| max 2.60
w1: [4.459 0.639 0.328] +- [0.097 0.176 0.294]          |pull| max 1.98
y:  [4.61  0.738 0.463] +- [0.27  0.284 0.34 ]          |pull| max 1.45
z:  [ 0.624 -0.377] +- [0.129 0.06 ]                    |pull| max 1.27
prediction-space residual rms / (F mu) = 0.31
```

Worth noticing against Example 1: the hierarchy *helps*. `y`'s prior anchors
the field's scale, which tames the very valley Example 1 warns about — the
posterior is wider where it should be (`y` at ±0.3–0.5, honestly) and the
pulls are healthy at shorter chains. A hierarchy is not only a modelling
statement; it is a regulariser the sampler feels.

---

## The validation experiment, and the finding it refused to soften

`examples/validate_sampling.py` replicates both models at fresh noise (and,
for the hierarchy, a fresh field realisation from its own prior), with
acceptance criteria registered in its docstring before any run: pooled
central-interval coverage, normalised-error sanity bands, partition
stability across replications, and a two-chain agreement check. It runs two
arms per replication — the factor sweep under test, and a **pure-NUTS
control** on the same graph at the same budget, which shares the joint
density by construction and therefore isolates *which* claim fails.

The first registered run failed, and the control arm said why. On the worst
replication of the hierarchy, at an equal budget and again at four times it:

| sampler | budget | max \|pull\| |
|---|---|---|
| factor sweep | 600 + 900 | 8.46 |
| factor sweep | 2400 + 3600 | 4.10 |
| pure NUTS | 600 + 900 | **1.05** |
| pure NUTS | 2400 + 3600 | **0.97** |

The posterior is right and NUTS reaches it easily; the SWEEP is what
struggles. (The experiment itself was amended twice after this finding, and
its docstring carries the history rather than absorbing it: fixed truths at
the priors' centres were replaced by prior-drawn truths after the registered
z-band caught the over-coverage they cause, and the coverage bands were
re-derived from the replication count after a run failed on band arithmetic
-- one run's scalars share its noise and its chain, so the replication, not
the scalar, is the statistical unit.) The mechanism is the valley again, wearing its third face: the
exact block `x` couples strongly to the remainder through the data, and
alternation — however exact each conditional draw — diffuses along that
coupling, while NUTS's adapted mass matrix glides along it. Exactness per
block and mixing across blocks are different properties, and this experiment
measured them apart.

**So when does the factor sweep earn its keep?** Not on a seven-parameter
model, and these pages say so rather than hint at it. Its regime is the one
a NUTS control cannot be run in at all: an exact block too LARGE for
gradient sampling — the 10^5-coefficient sky map, the per-channel gain
field — where the choice is not "sweep vs NUTS" but "sweep vs nothing".
There, one exact CG draw per sweep replaces a NUTS trajectory that cannot
take a single step, and the coupling cost this experiment measures is the
price of admission rather than a reason to abstain. On a model small enough
to NUTS whole, NUTS it whole; the plan's own printout tells you the block
sizes, which is the number that decides.

## What to take away

1. **Write the model; declare only what is true.** `linear_in` where the map
   really is affine; nothing for exp-affine latents — the log route is
   probed, not declared.
2. **Read the plan before sampling.** Every `nuts` line names its reason with
   the measured number that decided it; a reason you disagree with is a
   conversation with the model, not with the dispatcher.
3. **The partition is not a convergence proof.** Check `identifiability`,
   and on any model with a weak direction, run a second chain from a
   different start. Exactness per block and mixing across blocks are
   different properties.
4. **Two noises, two homes.** Weight-only scatter lives in the observation's
   scale; resolvable structure lives in a latent — and its hyperparameters
   ride with it into the NUTS remainder, by the ancestry rule.
