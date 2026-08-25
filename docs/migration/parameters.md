# Cross-check: `parameters`

`rheplicant.inference.parameters` (`Latent`, `Bind`, `ParameterSpace`,
`validate`, `forward_fn`, `refuse_stochastic_stages`) → node declarations
on the graph (`sample`, `det`, `const`, `linear_in`).

Test: `tests/crosscheck/test_parameters.py`. Case count and runtime are not written
here: `test_dispatch.py`'s said 6 within hours of being 7, and nothing in
this repository reads a count out of a page. Ask pytest —
`pytest tests/crosscheck/test_parameters.py --collect-only -q`. Measured
2026-08-25.

§四 4.2 asks for a **semantic mapping, not a line-by-line port**, and names
a minimum set: the three binding forms giving the same prediction on one toy
pipeline, and an equivalent of `refuse_stochastic_stages` — *"理由改写,
行为不得变"*, the reason rewritten and the behaviour unchanged. That second
clause is where this row did real work: **the behaviour did not exist here
yet.**

## 1. Fixtures

One toy pipeline — a spectrum into two multiplicative stages, on a 4×5
grid — reached three ways:

| form | rheplicant | here |
|---|---|---|
| **direct** | `Bind("spectrum", into=…)`, identity | `sample("spectrum", …)` used by one `det` |
| **derived** | `Bind("coeff", into=…, fn=lambda c: basis @ c)` | a `det` node between latent and prediction |
| **tied** | one `Bind`, two `into` selectors, `fan="broadcast"` | one latent, two named consuming edges |

Must be refused: a forward model that draws its own randomness.

The FAN fixture is rheplicant's own `TestFanOut` — `AntennaLossOperator`
with `t_physical = 0` so the efficiency is a pure multiply, and the
deliberately asymmetric `[2, 5]`, because a symmetric vector makes both
readings agree and blinds every comparison.

## 2. Numerical agreement

| quantity | agreement |
|---|---|
| prediction, all three binding forms | **bitwise identical** |
| the tie's observable consequence | gain enters squared: 3 → 9 |
| the FAN fixture, both readings | 4.0 and 10.0, reproduced on both sides |

The squared-gain check is the anti-vacuity clause. If `tied` merely renamed
one multiply, the comparison would pass on a model whose second stage did
nothing.

## 3. Refusal agreement

| rheplicant | here | mapping |
|---|---|---|
| `refuse_stochastic_stages` | `Graph.__check_init__` refuses a `Const` holding a PRNG key | `ParameterSpaceError` → `GraphError` — **written for this row; see §5** |
| `linear_in` naming a non-parent | same, at graph construction | no rheplicant counterpart: `linear=True` on a latent has no parent list to be checked against |

## 4. Independent oracle

The three forms are compared **against each other across packages**, which
by iron law 4 is not evidence on its own — so the tie carries an
independent arithmetic check (3² = 9, computed nowhere in either package)
and the FAN pair is pinned to the two numbers rheplicant's own test
measured, so this file is a correspondence rather than a second toy.

**Mutation.** Three, `__pycache__` cleared before each, judged on exit code
1 **and** on the failing test's name:

| mutation | caught by |
|---|---|
| the stochastic-stage guard removed | the refusal comparison |
| the guard stops naming the alternative | the "says where randomness belongs" test |
| the tied fixture's second stage made an identity | the three-forms comparison |

## 5. Intended differences

**(a) The stochastic-stage refusal did not exist here, and now does.** This
is the row's finding. Measured before the change: a `det` node whose parent
was a PRNG key evaluated happily and returned a field. That is exactly the
defect rheplicant's guard exists for, and the one nothing downstream can
see — inference closes the model over **one** evaluation, so the draw is
made once and the same frozen field rides every prediction compared against
the data. Adding a constant field is exactly affine, so `check_linearity`
reports a departure of 0 and `identifiability` reports full rank. Upstream
measured **10.6 σ** of bias with *both* exits reporting the same error bar
to every digit.

The reason is rewritten, as the row asks, and the behaviour is not: the
detector is the **declaration** on each side. rheplicant reads `RANDOMNESS`
in an operator's `requires`; the graph reads a `Const` whose value carries
a PRNG key dtype. Same class of case, same signal, same blind spot.

What the graph adds is an alternative to name. rheplicant can only forbid;
here the fix is one word — `sample` gives the same field a `log_prob`, so
it enters the joint distribution instead of hiding in the mean. The rule
this side states is **a random node without a density cannot enter the
joint**, which is the wording §四 4.2 asked for. A test asserts the message
carries it, because a refusal that only forbids leaves the modeller holding
the same model.

**(b) Both guards are blind to a closure, and it is the same blind spot.**
rheplicant's docstring states it: an operator that draws without declaring
`"key"`, or one hiding a draw in a static field, is invisible. A `fn` that
closes over a frozen draw passes here too — measured, not argued — as does
a legacy raw-`uint32` key, which is indistinguishable from an ordinary
array. There is no numerical symptom to find; the declaration is the whole
signal. Asserted so it cannot be mistaken for coverage.

**(c) `fan=` has no counterpart, because the question cannot be asked.**
rheplicant's `fan=` exists because a Python **container type** selects the
physics: `fn = lambda v: v` ties one produced value to every selector,
`fn = lambda v: list(v)` splits it element-wise, and `v` and `list(v)` are
the same data. Measured on the upstream fixture: **4.0 against 10.0**,
with the difference invisible in every value, every shape, and to both
`check_linearity` and `identifiability`.

On the graph each consumer is a **named edge**. "The same value into both"
is `x[0]` twice; "element k into stage k" is `x[0]` and `x[1]`. They are
two different graphs, not one graph plus an inference, and both numbers are
reproduced here to confirm the correspondence is real and not an evasion.
`test_bayesmith_has_no_fan_keyword_to_get_wrong` asserts the absence, so a
port of `Bind` has to argue with a test — the ambiguity would come back
with it.

**(d) `linear_in` is scoped to a node's own parents; `linear=True` is not
scoped at all.** rheplicant declares linearity on the **latent**, so it is
a claim about the whole model. bayesmith declares it on the **deterministic
node**, so a chain `coeff → spectrum → pred` states two claims rather than
one, and a node naming a non-parent is refused at construction — measured
while writing this row, when `linear_in=("coeff",)` on `pred` was refused
with `coeff` correctly identified as not a parent.

## 6. What this row does NOT cover

`ParameterSpace.validate`'s aliased-target and ignored-latent refusals,
`Latent`'s `scope`/`prior` declarations, and the `AmbiguousFanWarning`
single-selector case have no comparison here — the first two have no
graph-side counterpart to compare against (there are no selectors and no
scopes), and the third is a sub-case of §5(c). `forward_fn`'s ghost-pipeline
sibling `build_forward_fn` is the D7 seam and belongs to the
`numpyro_bridge` row.
