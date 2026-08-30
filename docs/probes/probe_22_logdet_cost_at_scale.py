"""M8 repeated, with a deterministic trace--log collapse beside the QR path.

The model is the one used for M8::

    y | x, theta ~ Normal(theta * B @ x, noise_std)
    x            ~ Normal(0, prior_std)

with deliberately non-unit ``prior_std=2``.  The QR column measures the
existing square-root marginalisation route.  The trace--log column uses the
same collapsed Gaussian objective but replaces the QR by

* an order-``m`` trace--log series for ``logdet(Sigma)``; and
* the matching order-``m`` Neumann series for ``y.T @ inv(Sigma) @ y``.

For this particular M8 family, ``A(theta) = theta B``, the eigenvalues of the
fixed ``B.T @ B`` are computed once, outside both compilation and timing.
They make every ``Tr(X**r)`` exact without materialising the ``n x n``
covariance.  This is a real specialisation: a generic matrix-vector oracle
does *not* reveal an exact trace in one application.  Such an oracle would
need a trace provider or frozen-probe Hutchinson/SLQ (ladder level 7), and the
latter must not be reported as deterministic exact-trace level 6.

The scalar preconditioner is

    Lambda(theta) = c(theta) I,
    c = (lambda_min(Sigma) + lambda_max(Sigma)) / 2.

Thus ``rho(X) = (kappa(Sigma)-1)/(kappa(Sigma)+1) < 1`` by construction for
finite positive ``noise_std``.  ``m`` is the production ladder's certified
whole-trace order: the first integer for which
``n*rho**(m+1) / ((m+1)*(1-rho)) <= TRACE_TOL``.  The nullspace eigenvalue of
``X`` is nonzero for this centred scalar preconditioner, so the required
multiplicity is ``n``, not ``k``.

All timings are post-JIT medians of amortised batches and synchronise the CPU
device.  Setup, compilation, and the one-time spectrum are excluded.  Exit 0
means the measurement completed, never that one route won.

Run:
    .venv/bin/python docs/probes/probe_22_logdet_cost_at_scale.py
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: I001  # x64 configuration must precede this import.
import numpy as np

from bayesmith.marginal.logdet import choose_trace_order
from bayesmith.marginal.sqrtinfo import marginalise_arrays


SIZES = ((100, 8), (100, 64), (400, 256), (1000, 512))
NOISE_STD = 0.5
PRIOR_STD = 2.0  # deliberately non-unit
THETA = 0.7
TRACE_TOL = 1.0e-6


def _time_us(function: Callable[..., jax.Array], *args: jax.Array) -> float:
    """Median microseconds per synchronised call, excluding JIT compilation."""
    compiled = jax.jit(function)
    jax.block_until_ready(compiled(*args))

    # Keep each sample long enough that timer quantisation and dispatch are a
    # small part of it, without making the k=512 QR batch needlessly large.
    trial_start = time.perf_counter()
    jax.block_until_ready(compiled(*args))
    one_call = time.perf_counter() - trial_start
    batch = max(1, min(1000, math.ceil(0.05 / max(one_call, 1.0e-9))))

    samples = []
    for _ in range(9):
        start = time.perf_counter()
        value = None
        for _ in range(batch):
            value = compiled(*args)
        jax.block_until_ready(value)
        samples.append((time.perf_counter() - start) * 1.0e6 / batch)
    return float(np.median(samples))


def _qr_collapsed(B: jax.Array, y: jax.Array) -> Callable[[jax.Array], jax.Array]:
    prior_rows = jnp.eye(B.shape[1], dtype=B.dtype) / PRIOR_STD
    zero_prior_target = jnp.zeros(B.shape[1], dtype=B.dtype)

    def objective(theta: jax.Array) -> jax.Array:
        factor = jnp.concatenate((theta * B / NOISE_STD, prior_rows), axis=0)
        target = jnp.concatenate((y / NOISE_STD, zero_prior_target), axis=0)
        _, _, log_marginal, _ = marginalise_arrays(
            factor, target, jnp.zeros((), dtype=B.dtype), B.shape[1]
        )
        return -log_marginal

    return objective


def _conditional(B: jax.Array, y: jax.Array, x: jax.Array) -> Callable:
    prediction_basis = B @ x

    def objective(theta: jax.Array) -> jax.Array:
        residual = y - theta * prediction_basis
        return 0.5 * jnp.vdot(residual, residual) / NOISE_STD**2

    return objective


def _trace_collapsed(
    B: jax.Array,
    y: jax.Array,
    gram_eigenvalues: jax.Array,
    order: int,
) -> Callable[[jax.Array], jax.Array]:
    n, k = B.shape
    lambda_max = gram_eigenvalues[-1]

    def covariance_action(theta: jax.Array, vector: jax.Array) -> jax.Array:
        scale = (PRIOR_STD * theta) ** 2
        return NOISE_STD**2 * vector + scale * (B @ (B.T @ vector))

    def objective(theta: jax.Array) -> jax.Array:
        scale = (PRIOR_STD * theta) ** 2
        lambda_min_sigma = NOISE_STD**2
        lambda_max_sigma = NOISE_STD**2 + scale * lambda_max
        centre = 0.5 * (lambda_min_sigma + lambda_max_sigma)

        # Exact traces for the fixed-spectrum M8 family.  There are n-k
        # eigenvalues in the nullspace of B and k represented by B.T @ B.
        null_eigenvalue = lambda_min_sigma / centre - 1.0
        represented = (
            lambda_min_sigma + scale * gram_eigenvalues
        ) / centre - 1.0
        null_power = null_eigenvalue
        represented_power = represented
        logdet_correction = jnp.zeros((), dtype=B.dtype)

        # The inverse series really does use `order` covariance-operator
        # actions.  It is included so this timing is a full collapsed
        # objective, rather than timing only the cheap half of a likelihood.
        inverse_term = y
        inverse_sum = y
        for degree in range(1, order + 1):
            trace_power = (n - k) * null_power + jnp.sum(represented_power)
            coefficient = (1.0 if degree % 2 else -1.0) / degree
            logdet_correction = logdet_correction + coefficient * trace_power

            inverse_term = inverse_term - covariance_action(theta, inverse_term) / centre
            inverse_sum = inverse_sum + inverse_term
            null_power = null_power * null_eigenvalue
            represented_power = represented_power * represented

        logdet = n * jnp.log(centre) + logdet_correction
        quadratic = jnp.vdot(y, inverse_sum) / centre
        return 0.5 * (logdet + quadratic)

    return objective


def main() -> None:
    print(f"device={jax.devices()[0]} dtype=float64 theta={THETA}")
    print(
        f"prior_std={PRIOR_STD} noise_std={NOISE_STD} "
        f"trace_whole_logdet_tol={TRACE_TOL:g}"
    )
    print(
        " n    k      c_gc QR      c_gtheta       c_A op    "
        "r_QR    rho    m    c_gc trace    r_trace   grad rel_err"
    )

    for n, k in SIZES:
        rng = np.random.default_rng(22_000 + n + k)
        B_np = rng.normal(size=(n, k)) / math.sqrt(n)
        y_np = rng.normal(size=n)
        x_np = PRIOR_STD * rng.normal(size=k)
        gram_eigenvalues_np = np.linalg.eigvalsh(B_np.T @ B_np)

        B = jnp.asarray(B_np)
        y = jnp.asarray(y_np)
        x = jnp.asarray(x_np)
        gram_eigenvalues = jnp.asarray(gram_eigenvalues_np)
        theta = jnp.asarray(THETA)

        scale = (PRIOR_STD * THETA) ** 2
        minimum = NOISE_STD**2
        maximum = minimum + scale * float(gram_eigenvalues_np[-1])
        rho = (maximum - minimum) / (maximum + minimum)
        order = choose_trace_order(rho, TRACE_TOL, multiplicity=n)

        qr_gradient = jax.value_and_grad(_qr_collapsed(B, y))
        conditional_gradient = jax.value_and_grad(_conditional(B, y, x))
        trace_gradient = jax.value_and_grad(
            _trace_collapsed(B, y, gram_eigenvalues, order)
        )

        def covariance_operator(
            vector: jax.Array, matrix: jax.Array = B
        ) -> jax.Array:
            return matrix @ (matrix.T @ vector)

        qr_us = _time_us(qr_gradient, theta)
        conditional_us = _time_us(conditional_gradient, theta)
        action_us = _time_us(covariance_operator, y)
        trace_us = _time_us(trace_gradient, theta)
        _, qr_derivative = qr_gradient(theta)
        _, trace_derivative = trace_gradient(theta)
        gradient_relative_error = float(
            jnp.abs(trace_derivative - qr_derivative)
            / jnp.maximum(jnp.abs(qr_derivative), jnp.finfo(B.dtype).tiny)
        )
        print(
            f"{n:4d} {k:4d}  {qr_us:10.1f} us  {conditional_us:10.1f} us  "
            f"{action_us:9.1f} us  {qr_us / conditional_us:6.1f}  "
            f"{rho:5.3f} {order:4d}  {trace_us:10.1f} us  "
            f"{trace_us / conditional_us:7.1f}   {gradient_relative_error:11.3e}"
        )


if __name__ == "__main__":
    main()
