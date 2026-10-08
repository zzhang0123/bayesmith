"""Conservative primal certificates for conditional Gaussian flatness.

This is a compile-time check, never part of a sampled update. It certifies an
affine mean and a block-independent covariance in model coordinates, holding
the complement at the supplied values. Fisher rank is assessed separately.
Neither user annotations nor vanishing automatic derivatives are proofs.

The supported algebra is interpreted over real arithmetic; the preflight
report retains its numerical rank, roundoff and contradictory-probe checks.
Unknown operations, control flow and custom derivative rules are refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax import lax
from jax.extend import core

from bayesmith.diagnose.local import check_differentiable, resolve_names
from bayesmith.distributions import ComplexNormal
from bayesmith.errors import BayesmithError
from bayesmith.exact.block import _ancestors
from bayesmith.graph.evaluate import apply_probabilistic, evaluate


def _primitives(*names):
    # Identity, not primitive.name: an extension can reuse a built-in's name
    # while implementing completely different primal or derivative semantics.
    return frozenset(getattr(lax, name + "_p") for name in names)


_AFFINE = _primitives(
    "add",
    "sub",
    "neg",
    "reshape",
    "squeeze",
    "transpose",
    "broadcast_in_dim",
    "concatenate",
    "slice",
    "rev",
    "pad",
    "reduce_sum",
    "cumsum",
    "copy",
    "real",
    "imag",
    "conj",
    "complex",
    "fft",
) | frozenset(
    primitive for name in ("stack_p", "unstack_p")
    if (primitive := getattr(lax, name, None)) is not None
)  # Newer JAX uses dedicated array packing primitives; both are linear.
_BILINEAR = _primitives("mul", "dot_general", "conv_general_dilated")
_NONLINEAR = _primitives(
    "abs",
    "sign",
    "exp",
    "expm1",
    "log",
    "log1p",
    "sqrt",
    "rsqrt",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "atan2",
    "pow",
    "max",
    "min",
    "floor",
    "ceil",
    "round",
    "clamp",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and",
    "or",
    "xor",
    "not",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_or",
    "reduce_and",
    "is_finite",
) | frozenset((lax.linalg.cholesky_p,))
# Obtain the actual JIT primitive without relying on a private JAX import or
# trusting a primitive merely because it carries a nested jaxpr parameter.
_JIT = jax.make_jaxpr(jax.jit(lambda x: x))(0.0).jaxpr.eqns[0].primitive


@dataclass(frozen=True)
class _Dependence:
    degree: int = 0  # 0: block-constant; 1: affine; 2: not certified affine.
    unsupported: tuple[str, ...] = ()
    zero: bool = False  # Proven literal/closed constant zero, never an input anchor.


def _combined(states, *, degree=None, unsupported=()):
    return _Dependence(
        max((state.degree for state in states), default=0)
        if degree is None
        else degree,
        tuple(
            sorted(
                {reason for state in states for reason in state.unsupported}
                | set(unsupported)
            )
        ),
    )


def _walk(jaxpr, inputs, constants=None):
    """Follow only the primal ancestors of outputs, including inside JIT.

    A primitive outside the tables is refused only where a block-dependent
    value reaches it; on a block-constant input it is a function of the
    complement and contributes degree 0 (see the ``else`` branch).
    """
    env = dict(zip(jaxpr.invars, inputs, strict=True))
    needed = {var for var in jaxpr.outvars if isinstance(var, core.Var)}
    live = []
    for equation in reversed(jaxpr.eqns):
        if any(var in needed for var in equation.outvars):
            live.append(equation)
            needed.update(var for var in equation.invars if isinstance(var, core.Var))

    if constants is None:
        env.update((var, _Dependence()) for var in jaxpr.constvars if var in needed)
    else:
        # Dead constants are not part of the proof. In particular, instrument
        # state can retain an unused typed PRNG key, which cannot be converted
        # to NumPy and says nothing about the live prediction's affinity.
        env.update((var, _Dependence(zero=bool(np.all(np.asarray(value) == 0))))
                   for var, value in zip(jaxpr.constvars, constants, strict=True)
                   if var in needed)

    def read(var):
        return (_Dependence(zero=bool(np.all(np.asarray(var.val) == 0)))
                if isinstance(var, core.Literal) else env[var])

    for equation in reversed(live):
        states = [read(var) for var in equation.invars]
        primitive = equation.primitive
        combined = _combined(states)
        if primitive is _JIT:
            nested = equation.params["jaxpr"]
            outputs = _walk(nested.jaxpr, states, nested.consts)
        else:
            if equation.effects:
                state = _combined(states, unsupported=("effectful_operation",))
            elif primitive in _AFFINE:
                state = _Dependence(combined.degree, combined.unsupported,
                                    all(s.zero for s in states))
            elif primitive in _BILINEAR:
                zero = any(s.zero for s in states)
                state = _Dependence(0 if zero else min(2, sum(s.degree for s in states)),
                                    combined.unsupported, zero)
            elif primitive is lax.div_p:
                state = _combined(
                    states, degree=2 if states[1].degree else states[0].degree
                )
            elif primitive is lax.integer_pow_p:
                exponent = equation.params["y"]
                degree = 0 if exponent == 0 else combined.degree if exponent == 1 else 2
                state = _combined(states, degree=degree if combined.degree else 0)
            elif primitive is lax.convert_element_type_p:
                old = equation.invars[0].aval.dtype
                new = equation.outvars[0].aval.dtype
                preserving = (
                    np.issubdtype(old, np.inexact)
                    and np.issubdtype(new, np.inexact)
                    and np.can_cast(old, new, casting="safe")
                )
                state = _combined(states, degree=combined.degree if preserving else 2)
                if not combined.degree:
                    state = combined
            elif primitive in (lax.gather_p, lax.dynamic_slice_p):
                state = _combined(
                    states,
                    degree=2 if any(s.degree for s in states[1:]) else states[0].degree,
                )
            elif primitive is lax.dynamic_update_slice_p:
                state = _combined(
                    states,
                    degree=2
                    if any(s.degree for s in states[2:])
                    else max(states[0].degree, states[1].degree),
                )
            elif primitive is lax.select_n_p:
                state = _combined(
                    states, degree=2 if states[0].degree else combined.degree
                )
            elif primitive is lax.stop_gradient_p:
                # Its primal value still depends on the block. In addition,
                # AD may report false full rank for x - stop_gradient(x), so
                # even an affine stopped value cannot support the rank proof.
                state = _combined(
                    states, unsupported=("stop_gradient",) if combined.degree else ()
                )
            elif primitive in _NONLINEAR:
                state = _combined(states, degree=2 if combined.degree else 0)
            elif primitive is lax.iota_p:
                state = combined
            else:
                # An unknown primitive whose inputs are all block-constant
                # yields a block-constant output: the proof quantifies over
                # the complement's values, so whatever the primitive computes
                # from them is some function of the complement alone, of
                # degree 0 in the block. Only a block-dependent input makes
                # its primal or derivative semantics matter -- the same gate
                # the stop_gradient branch applies. Measured before this gate
                # (2026-10-08): an emulator's erf/custom_jvp_call/scan on the
                # NUTS branch withheld the exact block's certificate on a
                # prediction that is affine in that block by inspection.
                state = _combined(
                    states, unsupported=(primitive.name,) if combined.degree else ()
                )
            outputs = [state] * len(equation.outvars)
        env.update(zip(equation.outvars, outputs, strict=True))
    return [read(var) for var in jaxpr.outvars]


def _gaussian_parameters(graph, name, env):
    distribution = apply_probabilistic(graph, graph.node(name), env)
    while type(distribution) in (dist.Independent, dist.ExpandedDistribution):
        if "log_prob" in vars(distribution):
            raise ValueError(f"custom_gaussian_density:{name}")
        distribution = distribution.base_dist
    # Exact types protect density semantics; an arbitrary Normal subclass is
    # not a canonical Gaussian even if it exports loc and scale attributes.
    if type(distribution) in (dist.Normal, ComplexNormal):
        covariance = distribution.scale
    elif type(distribution) is dist.MultivariateNormal:
        # The complete factor retains eigenvector/rotation dependence too.
        covariance = distribution.scale_tril
    elif type(distribution) is getattr(dist, "CirculantNormal", None):
        # A fixed Fourier basis makes the row a complete covariance descriptor.
        covariance = distribution.covariance_row
    else:
        raise ValueError(
            f"unsupported_gaussian_density:{name}:{type(distribution).__name__}"
        )
    if "log_prob" in vars(distribution):
        raise ValueError(f"custom_gaussian_density:{name}")
    return jnp.asarray(distribution.loc), jnp.asarray(covariance)


def derivative_semantics_evidence(function, *arguments):
    """Check primal derivative semantics without certifying any distribution.

    Numerical non-Gaussian information can use this same conservative walk.
    Known nonlinear primitives remain eligible for ordinary automatic
    differentiation; custom/unknown operations and control flow do not.
    """
    evidence = {"method": "primal_jaxpr", "trusted": False}
    try:
        primal = jax.make_jaxpr(function)(*arguments)
        states = _walk(primal.jaxpr, [_Dependence(1)] * len(primal.jaxpr.invars))
        unsupported = _combined(states).unsupported
        evidence.update(trusted=not unsupported, unsupported_primitives=unsupported,
                        output_degrees=tuple(state.degree for state in states))
    except (
        BayesmithError,
        ValueError,
        TypeError,
        NotImplementedError,
        AttributeError,
    ) as error:
        evidence["detail"] = str(error)
    return evidence


def conditional_affinity_certificate(graph, names, values):
    """Certify affine predictions with *symbolic* complementary latents.

    Unlike ``gaussian_flatness_certificate``, every latent is an abstract
    Jaxpr input. Degree zero means independent of this block, NOT frozen at
    an initial value. Thus an outside zero, branch or nonlinear coefficient
    cannot specialize the program used to justify a persistent partition.

    This is a sufficient real-arithmetic proof for the traced shapes/static
    configuration, wherever the pure program defines valid distributions.
    It neither proves domain validity nor finite-precision accuracy. Failure
    to certify is unknown, not proof of nonlinearity. Priors are assessed
    separately and do not change the prediction's structural verdict.
    """
    names = resolve_names(graph, names)
    evidence = {
        "certified": False,
        "method": "primal_jaxpr",
        "members": names,
        "conditioned_on": tuple(n for n in graph.latents if n not in names),
        "complement": "symbolic",
        "coordinates": "model",
        "scope": "all_latent_values_on_valid_domain",
        "arithmetic": "real",
        "covariance_independent": False,
        "gaussian_priors": False,
        "numerical_derivatives_trusted": False,
    }
    try:
        if not graph.observed:
            raise ValueError("no_observed_likelihood")
        arguments = tuple(jnp.asarray(values[n], dtype=jnp.asarray(values[n]).dtype)
                          for n in graph.latents)
        evidence["input_signature"] = tuple(
            (name, tuple(value.shape), str(value.dtype))
            for name, value in zip(graph.latents, arguments, strict=True)
        )

        def parameters(*latent_values):
            env = evaluate(graph, dict(zip(graph.latents, latent_values, strict=True)))
            return tuple(part for name in graph.observed
                         for part in _gaussian_parameters(graph, name, env))

        inputs = [_Dependence(int(n in names)) for n in graph.latents]
        primal = jax.make_jaxpr(parameters)(*arguments)
        states = _walk(primal.jaxpr, inputs, primal.consts)
        observations = tuple(
            {"name": name,
             "mean_affine": states[2 * i].degree <= 1,
             "covariance_independent": states[2 * i + 1].degree == 0,
             "unsupported_primitives": _combined(states[2 * i:2 * i + 2]).unsupported}
            for i, name in enumerate(graph.observed)
        )
        trusted = not any(row["unsupported_primitives"] for row in observations)
        certified = trusted and all(row["mean_affine"] for row in observations)
        evidence.update(
            observations=observations,
            numerical_derivatives_trusted=trusted,
            certified=certified,
            covariance_independent=trusted and all(
                row["covariance_independent"] for row in observations),
            reason=("affine_prediction" if certified else
                    "mean_not_certified_affine" if trusted else
                    "unsupported_primal_or_derivative_semantics"),
        )
        evidence["gaussian_priors"] = _conditional_gaussian_priors(
            graph, names, arguments, inputs)
    except (BayesmithError, ValueError, TypeError, NotImplementedError,
            AttributeError, KeyError) as error:
        evidence.update(reason="unsupported_structure_trace", detail=str(error))
    return evidence


def _conditional_gaussian_priors(graph, names, arguments, inputs):
    """Check prior family and block independence without filtering discovery."""
    # A joint density can depend on the whole environment, not only its
    # declared `over`. Node priors alone cannot justify an update omitting it.
    if graph.joint_prior is not None:
        return False
    def parameters(*latent_values):
        env = evaluate(graph, dict(zip(graph.latents, latent_values, strict=True)))
        return tuple(part for name in names
                     for part in _gaussian_parameters(graph, name, env))

    try:
        primal = jax.make_jaxpr(parameters)(*arguments)
        states = _walk(primal.jaxpr, inputs, primal.consts)
    except (BayesmithError, ValueError, TypeError, NotImplementedError, AttributeError):
        return False
    return all(state.degree == 0 and not state.unsupported for state in states)


def gaussian_flatness_certificate(graph, names, values):
    """Return serializable structural evidence, with full rank left to the caller.

    A positive result concerns every block value for which this traced, pure
    program defines a valid Gaussian likelihood, conditional on the supplied
    complement. It says nothing about a prior, posterior propriety or marginal
    information. Unsupported programs return explicit negative evidence.
    """
    names = resolve_names(graph, names)
    evidence = {
        "certified": False,
        "method": "primal_jaxpr",
        "members": names,
        "conditioned_on": tuple(n for n in graph.latents if n not in names),
        "coordinates": "model",
        "scope": "direct_observed_gaussian_likelihood",
        "rank": "assessed_separately",
        "numerical_derivatives_trusted": False,
    }
    try:
        check_differentiable(graph, names, values)
        if not graph.observed:
            raise ValueError("no_observed_likelihood")
        for name in graph.observed:
            if _ancestors(graph, name).intersection(graph.observed):
                raise ValueError(f"observation_dependent_parameters:{name}")

        def parameters(*block_values):
            point = dict(values)
            point.update(zip(names, block_values, strict=True))
            env = evaluate(graph, point)
            return tuple(
                part
                for name in graph.observed
                for part in _gaussian_parameters(graph, name, env)
            )

        primal = jax.make_jaxpr(parameters)(*(jnp.asarray(values[n]) for n in names))
        states = _walk(primal.jaxpr, [_Dependence(1)] * len(primal.jaxpr.invars))
        observations = tuple(
            {
                "name": name,
                "mean_affine": states[2 * index].degree <= 1,
                "covariance_independent": states[2 * index + 1].degree == 0,
                "unsupported_primitives": _combined(
                    states[2 * index : 2 * index + 2]
                ).unsupported,
            }
            for index, name in enumerate(graph.observed)
        )
        evidence["observations"] = observations
        evidence["numerical_derivatives_trusted"] = not any(
            row["unsupported_primitives"] for row in observations
        )
        if any(row["unsupported_primitives"] for row in observations):
            reason = "unsupported_primal_or_derivative_semantics"
        elif not all(row["covariance_independent"] for row in observations):
            reason = "parameter_dependent_covariance"
        elif not all(row["mean_affine"] for row in observations):
            reason = "mean_not_certified_affine"
        else:
            evidence["certified"] = True
            reason = "affine_mean_block_independent_covariance"
        evidence["reason"] = reason
    except (
        BayesmithError,
        ValueError,
        TypeError,
        NotImplementedError,
        AttributeError,
    ) as error:
        evidence["reason"] = "unsupported_structure_trace"
        evidence["detail"] = str(error)
    return evidence
