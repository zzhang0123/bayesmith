"""Probe 1 -- what today's chain does with a node that declares a correlated noise.

Runs the SAME graph three ways:
  * `log_joint`             -- reads the node's own `log_prob`
  * `to_numpyro` / NUTS     -- reads the node's own `log_prob`
  * the exact chain          -- reads `(loc, scale)` extracted by `gaussian.py`

and reports, for three spellings of the observed node's distribution:
  Normal            (the vocabulary that exists today)
  MultivariateNormal(a dense correlated covariance)
  CirculantNormal   (numpyro's own stationary-circulant Gaussian)

Run:
    cd <worktree> && PYTHONPATH=$PWD/src \
        /Users/zzhang/projects/bayesmith/.venv/bin/python probes/probe_1_what_the_chain_does.py
"""

import traceback

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from numpyro.infer.util import log_density as numpyro_log_density

from bayesmith.bridge.numpyro_bridge import to_numpyro
from bayesmith.exact.block import unchecked_operator
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.gaussian import check_gaussian, gaussian_parts, noise_std_at
from bayesmith.graph.evaluate import evaluate, log_joint
from bayesmith.graph.trace import const, det, observe, sample, trace

N = 8
KERNEL = np.array([2.0, 0.7, 0.3, 0.1, 0.05, 0.1, 0.3, 0.7])  # symmetric k -> n-k
X = np.linspace(-1.0, 1.0, N)
DATA = np.array([0.3, -0.2, 0.9, 0.1, -0.4, 0.6, 0.0, 0.25])


def dense_circulant(row):
    return np.array([np.roll(row, i) for i in range(len(row))])


def model_for(kind):
    def model():
        x = const("x", jnp.asarray(X))
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, x, linear_in=("w",))
        if kind == "normal":
            observe("d", lambda m: dist.Normal(m, 1.0), mu, obs=jnp.asarray(DATA))
        elif kind == "mvn":
            cov = jnp.asarray(dense_circulant(KERNEL))
            observe(
                "d",
                lambda m: dist.MultivariateNormal(m, covariance_matrix=cov),
                mu,
                obs=jnp.asarray(DATA),
            )
        elif kind == "circulant":
            observe(
                "d",
                lambda m: dist.CirculantNormal(m, covariance_row=jnp.asarray(KERNEL)),
                mu,
                obs=jnp.asarray(DATA),
            )
        return None

    return model


def attempt(label, thunk):
    try:
        value = thunk()
        print(f"    {label:<28} OK       {value}")
        return True
    except Exception as exc:  # noqa: BLE001 - the point is to report the type
        lines = str(exc).strip().splitlines()
        first = lines[0] if lines else repr(exc)
        print(f"    {label:<28} REFUSED  {type(exc).__name__}: {first[:96]}")
        return False


print(f"n = {N}, kernel = {KERNEL.tolist()}")
print(f"circulant eigenvalues = {np.real(np.fft.fft(KERNEL)).round(4).tolist()}")
print()

for kind in ("normal", "mvn", "circulant"):
    print(f"[{kind}]")
    graph = trace(model_for(kind))
    at = {"w": jnp.array(0.5)}

    attempt("log_joint", lambda g=graph: f"{float(log_joint(g, at)):.6f}")
    attempt(
        "numpyro log_density",
        lambda g=graph: f"{float(numpyro_log_density(to_numpyro(g), (), {}, at)[0]):.6f}",
    )

    env = evaluate(graph, at)
    node = graph.node("d")
    attempt("gaussian_parts(d)", lambda n=node, g=graph, e=env: str(
        tuple(np.shape(v) for v in gaussian_parts(g, n, e))))
    attempt("check_gaussian(d)", lambda n=node, g=graph, e=env: str(
        {k: round(v, 12) for k, v in check_gaussian(g, n, e).items()}))
    attempt("noise_std_at", lambda g=graph: str(
        {k: np.asarray(v).round(4).tolist() for k, v in noise_std_at(g, at).items()}))
    attempt(
        "unchecked_operator({w})",
        lambda g=graph: f"offset shape {np.shape(unchecked_operator(g, ('w',)).offset['d'])}",
    )
    attempt(
        "linear_operator({w})",
        lambda g=graph: f"offset shape {np.shape(linear_operator(g, ('w',)).offset['d'])}",
    )
    print()

# does numpyro's own NUTS run on the correlated graph?
from bayesmith.bridge.numpyro_bridge import nuts  # noqa: E402

for kind in ("mvn", "circulant"):
    graph = trace(model_for(kind))
    draws = nuts(graph, jax.random.key(0), num_warmup=200, num_samples=400)
    w = np.asarray(draws["w"])
    print(f"NUTS on [{kind}]: w = {w.mean():+.6f} +/- {w.std():.6f}  (n={w.size})")
