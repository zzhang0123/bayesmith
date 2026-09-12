"""Small dense Gaussian proposals with optional model-coordinate MH correction.

These are explicitly budgeted proposal builders. They do not replace the
matrix-free GCR path. With MH enabled, approximation affects efficiency, while the original
full graph density and both normalized proposal directions define the target.
Explicitly disabling correction instead applies unadjusted approximate updates,
while retaining support and numerical-validity rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_solve, solve_triangular

from bayesmith.distributions import ComplexNormal
from bayesmith.exact.gaussian import precision_parts, unwrap
from bayesmith.exact.loglinear import (
    FIRST_ORDER_MAX_FRACTIONAL,
    multiplicative_log_data,
)
from bayesmith.graph.evaluate import apply_probabilistic, evaluate, log_joint
from bayesmith.graph.nodes import Probabilistic


class DenseGaussianProposal(NamedTuple):
    """A normalized law with precision ``L @ L.T`` in flat model coordinates."""

    mean: jax.Array
    precision_cholesky: jax.Array
    valid: Any = True

    def sample(self, key):
        noise = jax.random.normal(key, self.mean.shape, dtype=self.mean.dtype)
        return self.mean + solve_triangular(
            self.precision_cholesky.T, noise, lower=False
        )

    def log_prob(self, value):
        residual = self.precision_cholesky.T @ (value - self.mean)
        return (
            jnp.sum(jnp.log(jnp.diag(self.precision_cholesky)))
            - 0.5 * jnp.sum(residual**2)
            - 0.5 * self.mean.size * jnp.log(2 * jnp.pi)
        )

    def is_valid(self):
        return (
            jnp.asarray(self.valid)
            & jnp.all(jnp.isfinite(self.mean))
            & jnp.all(jnp.isfinite(self.precision_cholesky))
            & jnp.all(jnp.diag(self.precision_cholesky) > 0)
        )


class MHInfo(NamedTuple):
    attempted: jax.Array
    accepted: jax.Array
    numerical_failure: jax.Array
    support_rejected: jax.Array
    log_accept_ratio: jax.Array


class ProposalDiagnostics(NamedTuple):
    """One counter per block; ``accepted`` counts applied proposals in either mode."""

    attempted: jax.Array
    accepted: jax.Array
    numerical_failure: jax.Array
    support_rejected: jax.Array


def mh_transition(key, current, build, log_target, support=None, *, mh_correction=True):
    """One proposal attempt, with original-target MH enabled by default.

    ``build`` and ``log_target`` accept flat vectors. A support rejection is
    distinct from a numerical failure. No retry or clipping is performed.
    With ``mh_correction=False`` no reverse law or MH ratio is evaluated:
    a supported draw with finite target density is applied, and the absent
    log acceptance ratio is represented by NaN. This is an unadjusted kernel,
    not a claim to sample the original target or a known approximate joint.
    """
    proposal_key, accept_key = jax.random.split(key)
    forward = build(current)
    candidate = forward.sample(proposal_key)
    supported = jnp.asarray(True) if support is None else support(candidate)
    finite_candidate = jnp.all(jnp.isfinite(candidate))
    eligible = forward.is_valid() & finite_candidate & supported

    def ratio(_):
        reverse = build(candidate)
        old_density, new_density = log_target(current), log_target(candidate)
        forward_density, reverse_density = (
            forward.log_prob(candidate),
            reverse.log_prob(current),
        )
        delta = new_density - old_density + reverse_density - forward_density
        valid = (
            reverse.is_valid()
            & jnp.isfinite(old_density)
            & jnp.isfinite(new_density)
            & jnp.isfinite(forward_density)
            & jnp.isfinite(reverse_density)
            & jnp.isfinite(delta)
        )
        return delta, valid

    if mh_correction:
        delta, valid = jax.lax.cond(
            eligible,
            ratio,
            lambda _: (jnp.asarray(-jnp.inf, current.dtype), jnp.asarray(False)),
            operand=None,
        )
        accepted = (
            eligible
            & valid
            & (
                jnp.log(jax.random.uniform(accept_key, dtype=current.dtype))
                < jnp.minimum(0.0, delta)
            )
        )
    else:
        valid = jax.lax.cond(
            eligible,
            lambda _: jnp.isfinite(log_target(candidate)),
            lambda _: jnp.asarray(False),
            operand=None,
        )
        delta = jnp.asarray(jnp.nan, current.dtype)
        accepted = eligible & valid
    result = jnp.where(accepted, candidate, current)
    numerical_failure = ~forward.is_valid() | ~finite_candidate | (eligible & ~valid)
    return result, MHInfo(
        jnp.asarray(1, jnp.int32),
        accepted.astype(jnp.int32),
        numerical_failure.astype(jnp.int32),
        (finite_candidate & ~supported).astype(jnp.int32),
        delta,
    )


def full_support(graph, state):
    """Check every latent against its distribution in the candidate environment."""
    env = evaluate(graph, state)
    valid = jnp.asarray(True)
    for name in graph.latents:
        distribution = apply_probabilistic(graph, graph.node(name), env)
        valid = (
            valid
            & jnp.all(jnp.isfinite(state[name]))
            & jnp.all(distribution.support(state[name]))
        )
    return valid


@dataclass(frozen=True)
class _Term:
    name: str
    prior: bool
    log_kind: str | None = None
    fractional: Any = None


def _value_shape(graph, node, env):
    """Full distribution batch/event shape, including broadcast observations."""
    distribution = apply_probabilistic(graph, node, env)
    batch = distribution.batch_shape
    if node.plate:
        batch = jnp.broadcast_shapes(batch, (graph.plate_size(node.plate[0]),))
    return jnp.broadcast_shapes(
        batch + distribution.event_shape, jnp.shape(env[node.name])
    )


def _check_target_layout(graph, env):
    """Refuse layouts for which graph and NumPyro repeat different densities."""
    for node in graph.nodes:
        if not isinstance(node, Probabilistic):
            continue
        distribution = apply_probabilistic(graph, node, env)
        if distribution.event_shape and (node.plate or node.observed_mask is not None):
            raise ValueError(
                f"proposal target does not support plated or masked event-reduced site {node.name}"
            )
        if node.plate:
            shape = jnp.shape(distribution.log_prob(env[node.name]))
            if node.observed_mask is not None:
                shape = jnp.broadcast_shapes(shape, jnp.shape(node.observed_mask))
            repeated = jnp.broadcast_shapes(shape, (graph.plate_size(node.plate[0]),))
            if repeated != shape:
                raise ValueError(
                    f"plate target repetition differs between graph and NumPyro at {node.name}; supply explicitly expanded values"
                )


@dataclass(frozen=True)
class ProposalBlock:
    """Static flattening metadata and a traceable conditional proposal builder."""

    graph: Any
    policy: Any
    names: tuple[str, ...]
    unpack: Any
    terms: tuple[_Term, ...]

    def pack(self, state):
        return ravel_pytree({name: state[name] for name in self.names})[0]

    def state_at(self, state, flat):
        return {**state, **self.unpack(flat)}

    def _residual(self, term, state, flat):
        env = evaluate(self.graph, self.state_at(state, flat))
        node = self.graph.node(term.name)
        distribution = unwrap(apply_probabilistic(self.graph, node, env))
        if term.prior:
            return jnp.asarray(env[term.name] - distribution.loc)
        data = jnp.broadcast_to(
            jnp.asarray(node.observed), _value_shape(self.graph, node, env)
        )
        if term.log_kind == "multiplicative":
            data, _ = multiplicative_log_data(data, term.fractional)
            return jnp.broadcast_to(jnp.log(distribution.loc), data.shape) - data
        if term.log_kind == "lognormal":
            return jnp.broadcast_to(distribution.loc, data.shape) - jnp.log(data)
        return jnp.broadcast_to(distribution.loc, data.shape) - data

    def _normal_equations(self, state, flat, damping_center):
        env = evaluate(self.graph, self.state_at(state, flat))
        size = flat.size
        metric = self.policy.damping * jnp.eye(size, dtype=flat.dtype)
        rhs = self.policy.damping * damping_center
        valid = jnp.asarray(True)
        for term in self.terms:
            node = self.graph.node(term.name)
            distribution = unwrap(apply_probabilistic(self.graph, node, env))
            residual_fn = lambda x, term=term: self._residual(term, state, x).reshape(
                -1
            )
            residual = residual_fn(flat)
            jacobian = jax.jacfwd(residual_fn)(flat)
            offset = residual - jacobian @ flat
            if (
                term.prior
                or term.log_kind is not None
                or isinstance(distribution, dist.Normal)
            ):
                sigma = (
                    term.fractional
                    if term.log_kind == "multiplicative"
                    else distribution.scale
                )
                weights = (
                    jnp.broadcast_to(
                        jnp.asarray(sigma), self._residual(term, state, flat).shape
                    ).reshape(-1)
                    ** -2
                )
                if node.observed_mask is not None:
                    weights = jnp.where(
                        jnp.broadcast_to(
                            jnp.asarray(node.observed_mask),
                            self._residual(term, state, flat).shape,
                        ).reshape(-1),
                        weights,
                        0.0,
                    )
                pushed_j = weights[:, None] * jacobian
                pushed_offset = weights * offset
                valid = (
                    valid
                    & jnp.all(jnp.isfinite(sigma))
                    & jnp.all(jnp.asarray(sigma) > 0)
                )
            else:
                _, precision = precision_parts(self.graph, node, env)
                shape = _value_shape(self.graph, node, env)
                apply = lambda x, precision=precision, shape=shape: precision.apply(
                    x.reshape(shape)
                ).reshape(-1)
                pushed_j = jax.vmap(apply, in_axes=1, out_axes=1)(jacobian)
                pushed_offset = apply(offset)
                if isinstance(distribution, dist.Normal):
                    valid = (
                        valid
                        & jnp.all(jnp.isfinite(distribution.scale))
                        & jnp.all(distribution.scale > 0)
                    )
            metric = metric + jacobian.T @ pushed_j
            rhs = rhs - jacobian.T @ pushed_offset
        metric = (metric + metric.T) / 2
        cholesky = jnp.linalg.cholesky(metric)
        mean = cho_solve((cholesky, True), rhs)
        return DenseGaussianProposal(mean, cholesky / self.policy.scale, valid)

    def build(self, state):
        current = self.pack(state)
        if self.policy.method == "iterative_gls":
            # Each solve freezes noise at the previous iterate. One FINAL
            # solve at the final frozen noise keeps centre and metric paired.
            point = jax.lax.fori_loop(
                0,
                self.policy.iterations,
                lambda _, x: self._normal_equations(state, x, current).mean,
                current,
            )
            return self._normal_equations(state, point, current)
        # GN uses the tangent at the current state; its offset is mu-Jx.
        # A log-linear builder uses that same affine form in log data space.
        return self._normal_equations(state, current, current)

    def step(self, key, state):
        current = self.pack(state)
        build = lambda x: self.build(self.state_at(state, x))
        target = lambda x: log_joint(self.graph, self.state_at(state, x))
        support = lambda x: full_support(self.graph, self.state_at(state, x))
        result, info = mh_transition(
            key,
            current,
            build,
            target,
            support,
            mh_correction=self.policy.mh_correction,
        )
        # Selection on each original leaf keeps rejected leaves byte-exact,
        # even when flattening promotes heterogeneous floating-point dtypes.
        proposed = self.unpack(result)
        result_state = dict(state)
        for name in self.names:
            result_state[name] = jnp.where(
                info.accepted.astype(bool), proposed[name], state[name]
            )
        return result_state, info


@dataclass(frozen=True)
class CompiledProposals:
    names: tuple[str, ...]
    remainder: tuple[str, ...]
    blocks: tuple[ProposalBlock, ...]
    initial_values: dict[str, Any]

    @property
    def mh_corrected(self):
        """Whether every scheduled proposal preserves the original target by MH."""
        return all(block.policy.mh_correction for block in self.blocks)

    def sweep(self, key, full_state):
        state, records = full_state, []
        for block, block_key in zip(
            self.blocks, jax.random.split(key, len(self.blocks)), strict=True
        ):

            def one(carry, draw_key, block=block):
                return block.step(draw_key, carry)

            state, info = jax.lax.scan(
                one, state, jax.random.split(block_key, block.policy.steps)
            )
            records.append(
                tuple(
                    jnp.sum(getattr(info, field), dtype=jnp.int32)
                    for field in ProposalDiagnostics._fields
                )
            )
        return state, ProposalDiagnostics(
            *(jnp.stack([r[i] for r in records]) for i in range(4))
        )


def _reachable(graph, names):
    dependent = set(names)
    for node in graph.nodes:
        if any(parent in dependent for parent in node.parents):
            dependent.add(node.name)
    return dependent


def _log_term(graph, node, initial_values):
    env = evaluate(graph, initial_values)
    distribution = unwrap(apply_probabilistic(graph, node, env))
    data = np.asarray(node.observed)
    if not np.all(np.isfinite(data) & (data > 0)):
        raise ValueError(f"log proposal requires positive finite data at {node.name}")
    if type(distribution) is dist.LogNormal:
        return _Term(node.name, False, "lognormal")
    if type(distribution) is not dist.Normal:
        raise ValueError(
            f"log proposal needs Normal or LogNormal observations at {node.name}"
        )
    fractional = distribution.scale / distribution.loc
    if not bool(
        jnp.all(
            jnp.isfinite(fractional)
            & (fractional > 0)
            & (fractional <= FIRST_ORDER_MAX_FRACTIONAL)
        )
    ):
        raise ValueError(
            f"log proposal fractional noise must be positive and <= {FIRST_ORDER_MAX_FRACTIONAL}"
        )
    flat, unpack = ravel_pytree(initial_values)

    def ratio(x):
        found = unwrap(apply_probabilistic(graph, node, evaluate(graph, unpack(x))))
        return jnp.asarray(found.scale / found.loc)

    # Eager applicability checks; the original target is still MH-corrected.
    # Never claim this finite probe is a global structural certificate.
    tol = 256 * np.finfo(np.asarray(flat).dtype).eps
    for index, displacement in enumerate((0.0, 0.071, -0.053)):
        direction = (1 + jnp.abs(flat)) * jnp.cos(jnp.arange(flat.size) + 0.3 + index)
        point = flat + displacement * direction
        value, derivative = jax.jvp(ratio, (point,), (direction,))
        if not (
            bool(jnp.allclose(value, fractional, rtol=tol, atol=0.0))
            and bool(jnp.all(jnp.abs(derivative) <= tol * jnp.max(fractional)))
        ):
            raise ValueError(
                f"log proposal needs constant fractional noise at {node.name}"
            )
    return _Term(node.name, False, "multiplicative", fractional)


def _check_affinity(block, initial_values):
    """Finite primal applicability checks, not a global linearity certificate.

    These mirror the package's measured linearity checks. The full MH target
    remains unchanged even if an unprobed region departs from the local fit.
    """
    if block.policy.method == "gauss_newton":
        return
    flat = block.pack(initial_values)
    tol = 256 * np.finfo(np.asarray(flat).dtype).eps
    for term in block.terms:
        if term.prior:
            continue
        residual = lambda x, term=term: block._residual(
            term, initial_values, x
        ).reshape(-1)
        value, jacobian = residual(flat), jax.jacfwd(residual)(flat)
        data = jnp.asarray(block.graph.node(term.name).observed)
        if term.log_kind is not None:
            data = jnp.log(data)
        for index, displacement in enumerate((0.173, -0.231, 0.097)):
            direction = (1 + jnp.abs(flat)) * jnp.cos(
                jnp.arange(flat.size) + 0.3 + index
            )
            point = flat + displacement * direction
            actual, expected = residual(point), value + jacobian @ (point - flat)
            scale = jnp.maximum(
                1.0,
                jnp.maximum(
                    jnp.abs(actual),
                    jnp.maximum(jnp.abs(expected), jnp.max(jnp.abs(data))),
                ),
            )
            if not bool(
                jnp.all(
                    jnp.isfinite(actual) & (jnp.abs(actual - expected) <= tol * scale)
                )
            ):
                raise ValueError(
                    f"{block.policy.method} needs an affine {'log-' if term.log_kind else ''}mean at {term.name}; use gauss_newton for nonlinear means"
                )


def compile_proposals(graph, policies, initial_values):
    """Validate explicit blocks and dense budgets before tracing transitions.

    Block priors may be Normal or Uniform; other factors remain in the target.
    Observations used for a proposal must have the existing Gaussian precision
    interface. Log proposals additionally accept canonical LogNormal data.
    Complex/discrete blocks are left to the existing inference routes.
    """
    policies = tuple(policies)
    names = tuple(name for policy in policies for name in policy.names)
    if (
        not policies
        or len(names) != len(set(names))
        or not set(names) <= set(graph.latents)
    ):
        raise ValueError("proposal blocks must name distinct, nonempty latent sets")
    if set(initial_values) != set(graph.latents):
        raise ValueError(
            "proposal compilation requires a complete initial latent state"
        )
    initial_values = {
        name: jnp.asarray(value) for name, value in initial_values.items()
    }
    env = evaluate(graph, initial_values)
    for name in graph.latents:
        node = graph.node(name)
        distribution = apply_probabilistic(graph, node, env)
        batch = distribution.batch_shape
        if node.plate:
            batch = jnp.broadcast_shapes(batch, (graph.plate_size(node.plate[0]),))
        expected = batch + distribution.event_shape
        if initial_values[name].shape != expected:
            raise ValueError(
                f"initial value {name} has shape {initial_values[name].shape}, expected declared shape {expected}"
            )
        if name not in names and (
            distribution.support.is_discrete
            or isinstance(unwrap(distribution), ComplexNormal)
            or not jnp.issubdtype(initial_values[name].dtype, jnp.floating)
        ):
            raise ValueError(
                f"NUTS remainder {name} requires continuous real floating coordinates; discrete and complex remainders are unsupported"
            )
    if not bool(full_support(graph, initial_values)) or not bool(
        jnp.isfinite(log_joint(graph, initial_values))
    ):
        raise ValueError(
            "proposal initial state violates support or has nonfinite target"
        )
    env = evaluate(graph, initial_values)
    _check_target_layout(graph, env)
    blocks = []
    for policy in policies:
        if not isinstance(policy.mh_correction, bool):
            raise TypeError("mh_correction must be bool")
        if not policy.names or policy.method not in (
            "iterative_gls",
            "bias_corrected_log_linear",
            "gauss_newton",
        ):
            raise ValueError("unknown or empty proposal block")
        if (
            policy.steps < 1
            or policy.iterations < 1
            or not np.isfinite(policy.scale)
            or policy.scale <= 0
            or not np.isfinite(policy.damping)
            or policy.damping < 0
        ):
            raise ValueError(
                "proposal budgets must be positive; scale finite positive; damping finite nonnegative"
            )
        for name in policy.names:
            value = initial_values[name]
            if not jnp.issubdtype(value.dtype, jnp.floating):
                raise ValueError(
                    "proposal blocks require real floating model coordinates"
                )
            found = unwrap(apply_probabilistic(graph, graph.node(name), env))
            if type(found) not in (dist.Normal, dist.Uniform):
                raise ValueError(
                    f"proposal member {name} needs a Normal or Uniform prior"
                )
        flat, unpack = ravel_pytree(
            {name: initial_values[name] for name in policy.names}
        )
        dependent = _reachable(graph, policy.names)
        selected = []
        elements = flat.size**2
        for node in graph.nodes:
            if not isinstance(node, Probabilistic) or node.name not in dependent:
                continue
            found = unwrap(apply_probabilistic(graph, node, env))
            if node.is_latent:
                if type(found) is not dist.Normal:
                    continue
                selected.append((node, True))
                elements += int(jnp.size(initial_values[node.name])) * flat.size
            else:
                selected.append((node, False))
                elements += int(np.prod(_value_shape(graph, node, env))) * flat.size
        if flat.size > policy.max_parameters or elements > policy.max_matrix_elements:
            raise ValueError(
                f"proposal dense budget exceeded: {flat.size} parameters, {elements} matrix elements"
            )
        terms = []
        for node, prior in selected:
            if prior:
                terms.append(_Term(node.name, True))
            elif policy.method == "bias_corrected_log_linear":
                terms.append(_log_term(graph, node, initial_values))
            else:
                distribution = unwrap(apply_probabilistic(graph, node, env))
                if not isinstance(distribution, dist.Normal):
                    precision_parts(graph, node, env)
                terms.append(_Term(node.name, False))
        block = ProposalBlock(graph, policy, tuple(policy.names), unpack, tuple(terms))
        _check_affinity(block, initial_values)
        if not bool(block.build(initial_values).is_valid()):
            raise ValueError(
                "proposal initial mean or precision is nonfinite; increase damping or provide a better initial state"
            )
        blocks.append(block)
    return CompiledProposals(
        names,
        tuple(n for n in graph.latents if n not in names),
        tuple(blocks),
        initial_values,
    )
