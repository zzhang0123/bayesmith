# Automatic affine discovery and conditional blocks

> **文档状态：`module-spec`** · Automatic linear discovery, structural evidence and default conditional-block execution; subordinate to the top-level design.

The compiler inspects the model computation. `linear_in` remains accepted for
compatibility, but omitting it does not prevent discovery and supplying it does
not prove a claim.

```python
import jax
import numpyro.distributions as dist
from bayesmith import compile, det, observe, sample, trace

def model():
    a = sample("a", lambda: dist.Normal(0., 1.))
    b = sample("b", lambda: dist.Normal(0., 1.))
    mu = det("mu", lambda a, b: 2*a + 3*b + 0.4, a, b)
    observe("y", lambda m: dist.Normal(m, 0.5), mu, obs=0.7)

plan = compile(trace(model))
assert [(b.latents, b.method) for b in plan.blocks] == [(("a", "b"), "gcr")]
posterior = plan.sample(jax.random.key(0), num_samples=1000)
```

Replacing the prediction by `a*b` produces two conditional `gcr` blocks. The
default plan's `sample()` runs their ordered Gibbs sweep, using the latest
complement after each update. Draws from that plan form a chain, with method
`factor_gibbs`; they are not independent joint Gaussian draws. Array-valued
latent sites remain intact.

`factor_partition()` and `sample_factors()` remain available for explicit
factor-plan inspection and execution, including their existing log-space path.
The default compiler's new factor runtime uses certified original-coordinate
Gaussian blocks and an optional NUTS remainder. It does not introduce new
automatic log-space transformations.

## What is proved

`diagnose.structure.conditional_affinity_certificate(graph, names, values)`
traces **all** latent inputs. Block inputs have affine degree one; complementary
latent inputs have degree zero with respect to that block, while remaining
symbolic. The supplied values establish input shapes and dtypes, not fixed
external coefficients. The certificate is about real arithmetic, for that pure
program, shape and static configuration, wherever its distributions are valid.
It does not prove domain validity, absence of overflow, or floating-point accuracy.

Known operations propagate sufficient structural facts. Addition, array layout
operations and sums preserve affinity. Multiplication and matrix products are
affine when at most one operand depends on the block. A divisor must be block
independent; validity still requires it to be nonzero. Literal and closed
constant zeros can eliminate dependence; an outside latent whose supplied value
happens to be zero cannot.

Examples:

| Prediction | Structural result |
|---|---|
| `A @ a + B @ b + c` | Jointly affine in `(a, b)` |
| `a*b` | Affine separately; not jointly certified |
| `a*sin(b)` | Affine in `a`, with a symbolic nonlinear coefficient |
| `a + b*a**2`, initialized at `b=0` | Not certified affine in `a` |
| `where(b >= 0, a, a**2)` | Not certified affine in `a` across complementary values |

Unknown primitives, unsupported control flow and custom derivative semantics
remain unknown where a block-dependent value reaches them; on inputs that are
constant in the block they are functions of the symbolic complement and
contribute degree 0, so an emulator on the sampled branch does not withhold
the exact block's certificate. A failed sufficient proof does not establish nonlinearity;
algebraic cancellation may simply exceed the analyzer. The interpreter examines
primal operations, so custom JVP/VJP rules or `stop_gradient` cannot manufacture
a proof. The final proposed group is certified as a whole.

## Structure, numerical checks and method eligibility

Each runtime block carries `structure` evidence. A certificate reports members,
scope, symbolic complement, input signature, mean affinity, covariance
independence and Gaussian-prior eligibility. Discovery adds numerical status and
probe details. Typed task analysis serializes the certificate, or compact
candidate summaries for the NUTS remainder, independently of the method label.

Finite probes compare actual predictions with their linearization at multiple
scales and complementary conditions. Passing them does not upgrade an unknown
structure into a proof. Contradictions or unusable probes prevent an automatic
exact update. Numerical roundoff qualifications are retained in the block's
evidence instead of being emitted as compile-time warnings.

A linear prediction can coexist with a non-Gaussian prior. Its structure is
still recognized; Gaussian solver eligibility is assessed separately. The
ancestry guard excludes updates that would drop another latent's distribution.
Arbitrary graph-level joint priors are conservatively excluded from automatic
Gaussian eligibility because their density is not included in the Gaussian
solver. Evidence-term boundaries remain enforced.

An affine mean with covariance not certified block independent needs an
original-target correction or a general sampler. The single-block compiler
retains its corrected moving-noise methods. A multiple exact-block sweep admits
only block-independent covariance; its remaining variables use NUTS. Existing
bounded automatic proposal selection can still consider eligible models with
an original-target MH correction.

## Execution limits

Multiple conditional Gaussian blocks do not provide a joint analytic posterior
mean or evidence integral. The default factor plan therefore refuses `.estimate()`
and `EvidenceTask`; estimate posterior quantities from the chain instead. An
explicit `sample(collapse=True)` request selects one certified constant-covariance
block to integrate out and samples the remaining variables through the existing
collapse route. The cost ladder also remains a single-block facility.

The factor runtime supports ordinary chain counts, initialization and checkpoint
stopping through the shared sampling controller. In-sweep concrete convergence
guards are unavailable under JIT; per-block compiled tolerances retain the
existing solver discipline. SNIS-specific `ess_floor` does not apply to this
Gibbs chain.
