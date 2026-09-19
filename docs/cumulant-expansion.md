# Cumulant likelihoods for array-valued fields

> **文档状态：`module-spec`** · 手动截断的场级 Edgeworth likelihood；空间表示留在数值 operator 与 utilities 内。

## Field-level model

The requested model is p(field | parents), including connected correlations
between different pixels/voxels/modes. A product of scalar Edgeworth factors
cannot supply those cross-cumulants. The scalar expansion remains a reference case. `FieldEdgeworth` implements
the joint field likelihood through interchangeable contraction operators.

For a vectorized real field with residual r and conditional covariance C,
define H_(i1...in) = (-1)^n G_C(r)^(-1) partial_(i1)...partial_(in) G_C(r).
The multivariate Edgeworth expression through weight 2 is

    p(field | parents) ~ G_C(r) [
        1 + K3_ijk H_ijk / 6
          + K4_ijkl H_ijkl / 24
          + K3_ijk K3_lmn H_ijklmn / 72
    ]

with repeated indices contracted. K3 and K4 are conditional connected
cumulant tensors; the final term is present at `order=4` under the same
hierarchy as the scalar formula. Every mean, covariance and cumulant operator
may depend on parent values.

The field implementation uses multivariate **Edgeworth**
ordering. Gram–Charlier A is a different truncation policy over a related
Hermite expansion; supporting both does not supply missing spatial
correlations. If added later, it should share contraction primitives and
declare its order convention separately. Neither policy is universally
correct for a strongly non-Gaussian field. No automatic selection is planned
for this first implementation.

The field is one NumPyro event: `event_shape=map.shape` or `cube.shape`, with
one joint log density per field. Independent realizations belong in batch or
plate dimensions. Merely applying `.to_event` to independent scalar factors
does not create connected spatial correlations.

Large fields require operators/structured contractions rather than dense
N^3/N^4 tensors. **User decision, refined: support both real-space cumulant
operators and Fourier/polyspectrum constructions, while keeping the core
workflow independent of representation.** The earlier phrase "separate
backends" refers only to numerical implementations behind the same interface;
it must not introduce real/Fourier workflow branches, node types or a required
`space` flag. Optional utilities construct/transform operators and arrays.
This is independent of the choice of Edgeworth versus Gram–Charlier ordering.

## Representation-independent architecture

Keep the existing `observe(name, dist_fn, *parents, obs=array)` boundary.
The field distribution composes a Gaussian reference distribution,
a cumulant-contraction provider and a user-selected static order. It is one
distribution type regardless of how any component computes its answer.
`FieldEdgeworth` is exported by `bayesmith.cumulants` and
`bayesmith.distributions`.

| Layer | Responsibility | Representation knowledge |
|---|---|---|
| Graph / workflow | Dependencies, events/batches, conditioning, log density, inference capabilities | No spatial/Fourier dispatch; respects shape, support and real degrees of freedom |
| Field expansion | Truncation rule, coefficients and cross terms, approximate log density | Calls common operations; does not choose a coordinate system |
| Numerical operators | Gaussian quadratic form/normalizer, cumulant–Hermite contractions, derivatives | May use dense, sparse, low-rank, FFT or other appropriate calculations |
| Optional utilities | Array transforms, kernel/polyspectrum adapters, coordinate conventions | Own basis, FFT normalization, axis and reality conventions |

The contraction boundary must handle products, not only single cumulants:
the expansion needs K3:H3, K4:H4 and (K3 tensor-product K3):H6 at order 4.
The implemented protocol is `prepared = operator.prepare(value, reference)`
followed by `prepared.contract(orders)`, with `orders=(3,)`, `(4,)` or `(3, 3)`,
returning exactly one scalar per sample/batch event. Preparation shares the
projected Gaussian geometry across terms.
The expansion owns the combinatorial coefficients and requested terms; an
operator owns how to compute each contraction with the **same Gaussian
reference**. This avoids both materializing dense high-rank tensors and
duplicating truncation logic in a Fourier adapter. The exported `CumulantOperator` also declares `event_shape`, `batch_shape`
and explicitly supplied `orders`. Custom providers should be JAX pytrees
(e.g. Equinox modules) so fitted parameter arrays stay differentiable.

The existing private `Precision` machinery in `exact/precision.py` is a
precedent: consumers ask for `apply` and `log_normalizer`, while
`CirculantPrecision` uses FFT internally. Reuse the responsibility boundary;
do not promote that entire private protocol or its Gaussian-only consumers
into a new public requirement without an implementation need. General
inference still consumes `log_prob`; optional optimizations depend on proven
capabilities/structure, not a coordinate label.

Two distinct operations must stay explicit:

1. **Internal computation in another basis.** The distribution accepts the
   original array, may evaluate contractions with FFT internally, and returns
   the density in its declared input coordinates. The workflow is unchanged.
2. **Changing the random variable's coordinates.** If the user replaces y by
   A y, the corresponding distribution must transform the mean, covariance
   and every cumulant consistently, and account for the density Jacobian.
   For an invertible real linear map, K_r transforms with A on every index
   and log p_(Ay)(Ay) = log p_y(y) - log|det A|. An appropriate distribution
   wrapper/adapter owns this accounting; an array-only FFT utility does not.

A `det` node can express a forward-model transform, but does not by itself
change a random variable's probability measure or add a Jacobian. Packed
real FFT coordinates require explicit handling of independent real degrees
of freedom and conjugate symmetry. Dropping modes is a projection/marginal
model, not an invertible change of coordinates. These are local distribution
or utility responsibilities, not reasons to teach the core workflow Fourier
semantics.

Covariance solve and log determinant also require an appropriate
representation. Fourier acceleration must not silently assume homogeneity,
periodicity or independence of a real field's conjugate modes. A generic
fallback need not be computationally affordable.

`tests/test_field_cumulants.py` compares small-field direct contractions with
structured implementations, verifies coordinate/Jacobian consistency, parent
and operator gradients, and cross-pixel non-Gaussian dependence. It also
exercises maps/cubes, sample/batch dimensions, graph plates, mixed precision,
FFT versus dense references, and the existing compile/sample workflow. Automatic order selection and
global validity assessment remain deferred.

Reference: [Sellentin, Jaffe & Heavens (2017)](https://arxiv.org/abs/1709.03452)
derives Edgeworth expansions for Fourier modes of non-Gaussian fields.
The built-in dense and low-rank contractions are implemented. General
polyspectrum specialization can use the same protocol.

## Field API and executable example

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from bayesmith import observe, sample, trace
from bayesmith.cumulants import FieldEdgeworth, LowRankCumulants

directions = jnp.array([[[0.5, 0.2], [0.1, 0.7]]])  # rank=1, event=(2, 2)
data = jnp.array([[0.2, 0.1], [0.4, -0.1]])

def likelihood(theta):
    base = dist.Normal(jnp.ones((2, 2)) * theta, 1.0).to_event(2)
    cumulants = LowRankCumulants(
        directions, {3: jnp.reshape(theta / 10, (1,)), 4: jnp.array([0.02])},
    )
    return FieldEdgeworth(base, cumulants, order=4)

def model():
    theta = sample("theta", lambda: dist.Uniform(-0.2, 0.2))
    observe("map", likelihood, theta, obs=data)

graph = trace(model)
```

`log_joint`, `to_numpyro` and `compile(graph).sample(...)` consume this graph.
The runnable [cumulant field example](../examples/inference/cumulant_field.py)
uses a periodic Gaussian reference, with FFT confined to that distribution.
Parameter inference on an observed field is supported; generating a draw from
an order > 2 field distribution raises `NotImplementedError`. A smoke test
of inference does not establish posterior convergence or global validity.

## Built-in numerical operators

`LowRankCumulants(directions, weights, event_shape=...)` represents

    K_r = sum_a weights[r][a] * directions[a]**(tensor r).

Directions have shape `batch + (rank,) + event_shape`; by default the first
axis is rank and all remaining axes are the field. Weights have shape
`batch + (rank,)`; scalars broadcast. They are **raw connected cumulants**,
not standardized moments or factorial-divided expansion coefficients.
All orders through the chosen N must be explicit; use zeros when needed.
The same directions can carry different weights at each order.

This is a caller-supplied structured model, not an automatic decomposition
of arbitrary cumulant tensors. Renormalizing the directions changes the model
unless the coefficients change with them. Nonzero shared directions express
cross-pixel cumulants even with an independent Gaussian reference.

Projected score and precision are derived from the reference's own log
density using JAX gradients and Hessian-vector products. No field-size dense
Hessian or high-rank cumulant tensor is built. Geometry storage is
O(rank*N + rank^2); a product of k cumulants takes rank^k combinations.
Reverse-mode differentiation also has loop-storage cost. Rank and order are
therefore material performance choices, not free accuracy controls.

`DenseCumulants({r: tensor, ...}, event_shape=shape)` accepts raw tensors
with trailing shape `(N,)*r`, using flattened C-order field indices.
It independently differentiates the Hermite generating function and can
allocate N**(3*(order-2)) elements. Use it for small-field/reference work,
not large maps. Only the symmetric part of a tensor contributes.

Gaussian references currently admitted: exact NumPyro `Normal`,
`MultivariateNormal`, `LowRankMultivariateNormal`, independent/expanded
wrappers and supported affine/reshape transforms, plus `PeriodicNormal`.
Arbitrary distribution subclasses are not assumed Gaussian. The field
expansion claims no eligibility for Gaussian exact elimination, including at
order 2; declare a normal directly when that structural route is needed.

## Optional Fourier utilities

`PeriodicNormal(loc, power, event_ndim=...)` evaluates and samples a real
Gaussian field with a periodic covariance. Power contains positive, finite
covariance eigenvalues on the **full orthonormal FFT grid**, exactly even
under simultaneous mode reversal. It is not an rFFT half-grid or an
unconverted continuum power spectrum. Default event rank is `loc.ndim`;
explicit `event_ndim` leaves preceding axes as batch dimensions.

`low_rank_from_fourier(coefficients, weights, event_shape=...)` constructs
real-coordinate low-rank cumulants from direction factors on that full FFT
grid, using an orthonormal inverse transform. Reality is checked, including
under JIT; only numerical imaginary residue is discarded. Each cumulant index
transforms through its direction factors, so coefficients stay unchanged.
This supports factored Fourier cumulants. It does not factorize an arbitrary
bispectrum/trispectrum or automatically impose translation invariance.
Specialized polyspectrum contractions can implement the same operator protocol.

The observed variable stays in real coordinates in these utilities. Actual
changes of coordinates require transforming the reference, every cumulant,
and the density Jacobian, as described above. Complex packed coordinates
and projections require their own support/marginalization treatment.

Whole-field observations are supported. Do not apply graph pixel masks to
this joint scalar density: removing correlated event components requires a
marginal distribution over the retained components, which is not supplied.

## Existing conditional distribution interface

Bayesmith records a callable in `Probabilistic.dist_fn` and the ordered names
of its parents. `sample(name, dist_fn, *parents)` declares a latent node;
`observe(name, dist_fn, *parents, obs=data)` declares a conditional likelihood.
At evaluation time the callable receives the parents' **values** and returns
a NumPyro distribution. `log_joint` reads that object's `log_prob`; the
NumPyro bridge places the same object at a `numpyro.sample` site. Gaussian,
Student-t, mixture and other existing distributions come from NumPyro.
Bayesmith's own `ComplexNormal` supplies its complex-coordinate convention.

`EdgeworthExpansion` adds a JAX-compatible, manually truncated likelihood at
this existing boundary. It does not change the graph or implement an inference
engine. It is a `Distribution` subclass to use the density interface, but an
arbitrary truncation is **not a certified probability distribution**.

## Scalar reference usage

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from bayesmith import log_joint, observe, sample, trace
from bayesmith.distributions import EdgeworthExpansion

def model():
    mu = sample("mu", lambda: dist.Normal(0.0, 2.0))
    k3 = sample("k3", lambda: dist.Normal(0.0, 0.1))
    observe(
        "y",
        lambda m, third: EdgeworthExpansion(
            loc=m, scale=1.0, cumulants=(third, 0.1), order=4,
        ),
        mu, k3,
        obs=jnp.array([0.2, 0.5, 0.9]),
    )

graph = trace(model)
value = log_joint(graph, {"mu": jnp.array(0.1), "k3": jnp.array(0.05)})
```

Every cumulant can depend on parents; the example fixes the fourth only for
brevity. This is a scalar-event likelihood with batch broadcasting and graph
plates; it does not represent multivariate cross-cumulant tensors. NumPyro's
`.to_event(...)` can group independent components, without adding correlation.

`loc` is kappa_1, `scale` is sqrt(kappa_2), and `cumulants` is the sequence
`(kappa_3, kappa_4, ...)`. These are unstandardized cumulants, not central
moments: kappa_4 is the fourth central moment minus three times the variance
squared. The implementation divides kappa_r by `scale**r` internally.

`order=N` is a static integer >= 2 chosen by the user. Supply all cumulants
through kappa_N, using explicit zeros when needed. Later entries are ignored,
so the same longer sequence can be used to compare user-selected orders.
Order 2 requires no higher cumulants and gives `Normal(loc, scale)`.

## What the order retains

Let z=(y-loc)/scale, a_r=kappa_r/(scale**r r!), and H_n be the probabilists'
Hermite polynomials. The Gaussian multiplier P_N is

    P_N(z) = sum H_(sum r*m_r)(z) product a_r**m_r / m_r!

over nonnegative integer powers m_3,...,m_N satisfying
`sum((r-2)*m_r) <= N-2`, including the constant term. Thus:

| order | Terms added to the previous order |
|---|---|
| 2 | Gaussian baseline, P_2=1 |
| 3 | a_3 H_3 |
| 4 | a_4 H_4 + a_3^2 H_6 / 2 |
| 5 | a_5 H_5 + a_3 a_4 H_7 + a_3^3 H_9 / 6 |

This is Edgeworth ordering under the usual weak-non-Gaussian hierarchy, not
truncation at Hermite degree N. The largest Hermite degree is 3(N-2).
Higher selected orders are generated from the same finite partition rule.
No convergence or error guarantee follows from choosing a larger N.
The expansion convention follows the arbitrary-order Edgeworth discussion in
[Blinnikov & Moessner (1998)](https://arxiv.org/abs/astro-ph/9711239).

## Current execution boundary

- `correction(y)` returns P_N, including negative values.
- `log_prob(y)` evaluates `Normal(loc, scale).log_prob(y) + log1p(P_N-1)`.
  Negative corrections yield NaN; zero yields negative infinity. There is no
  clipping, absolute value, positive-part renormalization, change of order, or
  Gaussian fallback. A finite value at observed data is not a global check.
- JAX differentiation, JIT and graph plate evaluation are supported. An
  observed likelihood can reach NumPyro's density-based inference interface;
  successful inference still depends on its evaluated target being usable.
- Order > 2 `sample` raises `NotImplementedError`. Using the expansion as a
  latent prior, prior predictive, posterior predictive, simulation bank or SBC
  generator is not supported. Order 2 samples its Gaussian baseline.
- The class is not a `Normal` subclass and does not claim eligibility for
  Gaussian exact elimination. A Gaussian-only consumer may refuse it even at
  order 2; use `Normal` when that structural shortcut is required.
- NumPyro's `validate_args` checks the declared parameter constraints; it
  does not prove validity of a cumulant sequence or global positivity.

The signed expression has unit integral formally, by Hermite orthogonality.
This is not a nonnegativity guarantee, and does not qualify it for a Bayesian
evidence calculation as a normalized probability model.

## TODO — explicitly deferred

- Automatic order selection and truncation-error diagnostics.
- Global nonnegativity/validity assessment and any explicitly chosen repair
  or replacement family, including its parameter-dependent normalization.
- A matching non-Gaussian sampler, conditional on an appropriate valid model.
- Non-Gaussian field generation, specialized arbitrary-polyspectrum
  contractions, and additional coordinate/marginalization adapters.

None of these is a prerequisite for the current manual likelihood evaluator.
