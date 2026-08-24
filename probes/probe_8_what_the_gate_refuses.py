"""Probe 8 -- the gate is a TYPE check; the probe can only tighten it.

`gaussian.py`'s docstring argues that reading `.loc`/`.scale` off a `Normal`
"trusts the type ... the type is evidence, not proof", and that the log_prob
probe "is what raises the bar". True for REFUSAL. But acceptance runs the
other way: `gaussian_parts` refuses on `isinstance` BEFORE `check_gaussian`
ever evaluates a density, so a node whose log_prob is *exactly* a diagonal
Normal's is still refused if it is spelled as another class.

This measures, for each spelling: (i) is it accepted, (ii) does its log_prob
in fact equal the diagonal Normal's, to what precision.

Run:
    cd <worktree> && PYTHONPATH=$PWD/src \
        /Users/zzhang/projects/bayesmith/.venv/bin/python probes/probe_8_what_the_gate_refuses.py
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import transforms

from bayesmith.exact.gaussian import check_gaussian, gaussian_parts
from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.trace import const, det, observe, sample, trace

jax.config.update("jax_enable_x64", True)

N = 6
X = np.linspace(-1.0, 1.0, N)
DATA = np.array([0.3, -0.2, 0.9, 0.1, -0.4, 0.6])
SIGMA = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])


def spellings(m):
    """Every one of these declares the SAME density: N(m, diag(SIGMA**2))."""
    s = jnp.asarray(SIGMA)
    return {
        "Normal": dist.Normal(m, s),
        "Normal.to_event(1)": dist.Normal(m, s).to_event(1),
        "MultivariateNormal(diag)": dist.MultivariateNormal(
            m, covariance_matrix=jnp.diag(s**2)
        ),
        "MVN(scale_tril=diag)": dist.MultivariateNormal(m, scale_tril=jnp.diag(s)),
        "CirculantNormal(s*I)": dist.CirculantNormal(
            m, covariance_row=jnp.asarray(np.r_[SIGMA[0] ** 2, np.zeros(N - 1)])
        ),
        "TransformedDistribution": dist.TransformedDistribution(
            dist.Normal(jnp.zeros(N), jnp.ones(N)).to_event(1),
            transforms.AffineTransform(m, s),
        ),
    }


reference = dist.Normal(jnp.asarray(X), jnp.asarray(SIGMA)).to_event(1)
value = jnp.asarray(DATA)
print("do these spellings really declare the same density?")
ref_lp = float(reference.log_prob(value))
for label, d in spellings(jnp.asarray(X)).items():
    lp = float(jnp.sum(d.log_prob(value)))
    print(f"  {label:<26} log_prob {lp:+.12f}   |diff| {abs(lp - ref_lp):.3e}")

print()
print("and what does the exact path do with each?")
for label in spellings(jnp.asarray(X)):

    def model(_label=label):
        x = const("x", jnp.asarray(X))
        w = sample("w", lambda: dist.Normal(0.0, 3.0))
        mu = det("mu", lambda w_, x_: w_ * x_, w, x, linear_in=("w",))
        observe(
            "d",
            lambda m, _l=_label: spellings(m)[_l],
            mu,
            obs=jnp.asarray(DATA),
        )
        return None

    graph = trace(model)
    env = evaluate(graph, {"w": jnp.array(0.5)})
    node = graph.node("d")
    try:
        gaussian_parts(graph, node, env)
        errors = check_gaussian(graph, node, env)
        print(f"  {label:<26} ACCEPTED  worst probe error {max(errors.values()):.3e}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label:<26} REFUSED   {type(exc).__name__}")

print()
print("the asymmetry, stated precisely:")
print("  gaussian_parts refuses on `isinstance(distribution, dist.Normal)` at")
print("  gaussian.py:100, BEFORE check_gaussian evaluates any density. So the")
print("  probe can only ever REFUSE a Normal whose log_prob disagrees; it can")
print("  never ACCEPT a non-Normal whose log_prob agrees exactly.")

print()
print("what the dispatcher then does with the refused spelling:")
from bayesmith.dispatch.plan import compile as bayes_compile  # noqa: E402


def mvn_model():
    x = const("x", jnp.asarray(X))
    w = sample("w", lambda: dist.Normal(0.0, 3.0))
    mu = det("mu", lambda w_, x_: w_ * x_, w, x, linear_in=("w",))
    observe(
        "d",
        lambda m: dist.MultivariateNormal(m, covariance_matrix=jnp.diag(jnp.asarray(SIGMA) ** 2)),
        mu,
        obs=jnp.asarray(DATA),
    )
    return None


def normal_model():
    x = const("x", jnp.asarray(X))
    w = sample("w", lambda: dist.Normal(0.0, 3.0))
    mu = det("mu", lambda w_, x_: w_ * x_, w, x, linear_in=("w",))
    observe("d", lambda m: dist.Normal(m, jnp.asarray(SIGMA)), mu, obs=jnp.asarray(DATA))
    return None


for label, m in (("Normal", normal_model), ("MultivariateNormal(diag)", mvn_model)):
    plan = bayes_compile(trace(m))
    print(f"  {label:<26} plan -> {str(plan).splitlines()[0][:70]}")

print()
print("is the Precision protocol structurally checkable (for a side-channel)?")
from bayesmith.exact.precision import (  # noqa: E402
    CirculantPrecision,
    DiagonalPrecision,
    Precision,
)

for obj in (
    DiagonalPrecision(sigma=jnp.asarray(SIGMA)),
    CirculantPrecision(first_column=jnp.asarray(np.r_[1.0, np.zeros(N - 1)])),
    dist.Normal(0.0, 1.0),
):
    print(f"  isinstance({type(obj).__name__:<20}, Precision) = {isinstance(obj, Precision)}")
