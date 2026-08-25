# Cross-check: `numpyro_bridge`

`rheplicant.inference.numpyro_bridge` (`to_numpyro_model`,
`init_to_declared`, `predict_from_samples`) → `bayesmith.bridge`
(`to_numpyro`, `nuts`, `predict`).

Test: `tests/crosscheck/test_bridge.py`. Case count and runtime are not written
here: `test_dispatch.py`'s said 6 within hours of being 7, and nothing in
this repository reads a count out of a page. Ask pytest —
`pytest tests/crosscheck/test_bridge.py --collect-only -q`. Measured
2026-08-25.

**This is the last of §四's rows, and closing it opens §六's gate.**

§四 4.2 does not ask for a numerical comparison here — both sides hand the
same joint to the same sampler — but for **three rheplicant-specific things**
to be carried. Two of the three turned out to be reachable failures on this
side, and were fixed in this pass.

## 1. Fixtures

**The needle.** A power law `A(ν/70)^β` over 24 channels, `A = 5000` with a
prior width of 1e4 and a posterior width of ~0.4 — so the posterior is ~1e4
times narrower than the prior, which is the geometry `init_to_declared`
exists for. Two chains of 400 after 400 warmup.

**The square stack.** A length-3 latent with three draws, so a sample stack
is square and its transpose has the same shape. Chosen because that is the
one case a shape guard cannot see — see §5.

**The covered latents.** A radiometer power law whose two latents are
covered by a `JeffreysPrior`, built twice: once declared
`ImproperUniform` (legal) and once with their own proper `Normal`s (the
double count).

## 2. Numerical agreement

Not the point of this row, and saying so is part of the record: both
packages construct a NumPyro model and hand it to the same NUTS, so a
value-for-value comparison would be comparing NumPyro against itself. What
is compared is **behaviour under the three failure modes upstream names**.

The one number that is compared is the Jeffreys factor's contribution: the
joint's log-density moves by the prior's own term and by nothing else,
measured to `rel=1e-12` against `JeffreysPrior.log_density` evaluated
independently.

## 3. Refusal agreement

| rheplicant | here | mapping |
|---|---|---|
| `predict_from_samples`: missing site | `predict` | `StateValidationError` → `GraphError` |
| `predict_from_samples`: wrong per-sample shape | same | same — **ported in this pass**; §5(b) |
| `predict_from_samples`: differing draw counts | same | same |
| `_refuse_sampled_noise_std_under_a_joint_prior` | `JeffreysPrior._check_against` | `ParameterSpaceError` → `GraphError`; §5(c) |

## 4. Independent oracle

The predictive's correct answer is written out by hand
(`[[0, 2, 6], [3, 8, 15], [6, 14, 24]]` for `c ⊙ x` over three draws), so
the guard is checked against arithmetic rather than against a second run of
the thing it guards. The transposed answer is written out too — it is the
evidence that the silent case is silent.

The Jeffreys term is measured through `JeffreysPrior.log_density` and then
subtracted from the NumPyro joint, so "added once" is a subtraction rather
than an inspection of the trace.

**Mutation.** Three, all killed by the test named against them: `predict`
checking only the name, `predict` not counting draws, and `nuts_options`
not reaching the kernel.

## 5. Intended differences, and what this row changed

**(a) `init_to_declared` has no code to port, and the lesson is real
here.** A graph latent has a prior, not a declared `init`, so there is
nothing to wrap. The question the row actually poses is whether the failure
transfers. It does:

| init | r_hat (amp) | ESS |
|---|---|---|
| default `init_to_uniform` — what `nuts()` ships | **1609** | **1.0** |
| `init_to_value` at the declared values | **1.006** | **138.6** |

Upstream's own numbers on its ring toy are r_hat 840 / ESS 2 against 1.002
/ 1327. Same shape, different model, this side.

**The remedy needed no new code**: `nuts()`'s `nuts_options` forwards
straight to the `NUTS` kernel, and `init_strategy` is one of its keywords.
What was missing was the sentence saying so, which is now in `nuts`'s
docstring **with these numbers** — which is what *"带过去的是教训不是代码"*
asks for. The test bounds rather than pins them, so a NumPyro release that
closes the gap fails it, and that is the moment to delete the paragraph.

**(b) `predict_from_samples`' shape guard did not exist here, and now
does.** Before this pass there was no posterior-predictive entry point at
all, so callers reached NumPyro's `Predictive` directly. Measured there: a
non-square transposition raises a broadcast `TypeError` from three layers
down that names neither the site nor the axis, and a **square** one raises
nothing.

`bayesmith.predict` now carries the check, reading each latent's declared
shape off its own `dist_fn` via `prior_environment` — the graph's
equivalent of rheplicant reading it off `ParameterSpace`.

**(c) The square transposition is invisible to BOTH guards, and that is
stated rather than implied.** A shape check cannot separate a stack from
its own transpose when the draw count equals the latent's size, and
rheplicant's compares `shape[1:]`, which transposing a square preserves
exactly. Measured, both accepted, both finite, both correctly shaped:

    correct     [[0, 2, 6], [3, 8, 15], [6, 14, 24]]
    transposed  [[0, 6, 18], [1, 8, 21], [2, 10, 24]]

The remedy is not a better shape check — there is no shape left to look at
— it is not building the stack by hand. `predict` takes what `nuts` returns,
unchanged, for that reason.

**(d) The "density added once" guard is stronger here, and was already
present.** rheplicant refuses a *sampled* `noise_std` under a joint prior.
bayesmith refuses any latent that is both covered by a `JeffreysPrior` and
declares a proper density of its own, and its message says why the failure
is invisible: *"each one on its own is correct"*, so their product is a
proper density and a plausible chain and no diagnostic reports a prior
counted twice. The fix it names — declare the covered latents
`ImproperUniform` — is the one this row's legal fixture uses.

## 6. What this row does NOT cover

`to_numpyro_model`'s own construction is not compared site-by-site against
`to_numpyro`: the two build models over different vocabularies (a
`ParameterSpace` over pipeline leaves against graph nodes), and the joint
they declare is already cross-checked at the density level in `noise.md`.

Declaring a joint prior **on the graph** — so `to_numpyro` honours it
without a hand-written `numpyro.factor`, and `check_identified` becomes its
build-time gate — is the work this row unlocks rather than the work it
does. `priors.md` §5 records the same boundary from the other side.
