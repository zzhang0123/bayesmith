"""probe_28 -- the model-checking seams R3 builds on, measured on this checkout.

Written while planning R3 (docs/superpowers/plans/2026-09-02-r3-model-checking.md)
so that every number the plan quotes -- a p-value, a rank-uniformity
statistic, a cost per replicate, an ArviZ verdict -- is a measurement of THIS
tree rather than a description of one.  Run from the repository root with the
test package importable:

    PYTHONPATH=. .venv/bin/python docs/probes/probe_28_model_checking_seams.py

Nine sections, each independent (a failure in one prints its traceback and
the rest still run):

1. PPC discrepancy on the R2 seam: replicated draws vs observed data, a
   curvature-sensitive discrepancy, one correct model and one misspecified.
2. LOO through the ArviZ export: what `az.loo` needs of the exported
   InferenceData (chain axis or not), and what it returns.
3. SBC on the exact route: prior-drawn truths, `gcr` posteriors, continuous
   ranks, KS uniformity; a 2x-too-wide and a 2x-too-narrow control.
4. The replicate count at which a 2x width error becomes detectable.
5. SBC on the NUTS route: cost per replicate at a small budget.
6. The reference NPE's calibration on the amortize test problem.
7. Held-out prediction through `observe(mask=...)`: elpd on masked points.
8. Prior predictive draws: a prior-scale check, and what a plated node needs.
9. Chain diagnostics against ArviZ's r-hat / ESS as an independent oracle;
   identifiability and prior-sensitivity numbers the R3 reports will carry.
"""

from __future__ import annotations

import sys
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from scipy import stats

from bayesmith import compile as compile_graph
from bayesmith import const, det, observe, sample, trace
from bayesmith.artifacts.base import ArtifactKind, ArtifactRef, ComputeBudget
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.tasks import PosteriorTask, PredictiveTask, new_task_meta
from bayesmith.dispatch.execute import chain_diagnostics
from bayesmith.dispatch.predictive import pointwise_log_likelihood, replicated_draws
from bayesmith.dispatch.task import compile_task, execute_task
from bayesmith.graph.evaluate import apply_deterministic, apply_probabilistic, evaluate
from bayesmith.graph.nodes import Const, Deterministic, Probabilistic
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import bilinear_pair, collinear_pair, radiometer, straight_line

SBC_REPLICATES = int(sys.argv[1]) if len(sys.argv) > 1 else 100  # argv[2:] selects sections
X = jnp.linspace(1.0, 4.0, 8)
SIGMA = 0.5
PRIOR_STD = 2.0


def section(title):
    print(f"\n=== {title} ===")


def line_with(data, *, mask=None, sigma=SIGMA):
    """`straight_line`'s model at caller-supplied data (and optionally a mask)."""

    def model():
        xs = const("X", X)
        w = sample("w", lambda: dist.Normal(0.0, PRIOR_STD))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data, mask=mask)

    return trace(model)


def curved_line(*, weight=2.5, curvature=0.6, seed=0, mask=None):
    """Data with a quadratic term the linear model cannot express."""
    noise = SIGMA * jax.random.normal(jax.random.key(seed), X.shape)
    return line_with(weight * X + curvature * X**2 + noise, mask=mask)


def bilinear_with(data, *, sigma=0.3):
    x = jnp.linspace(0.5, 3.0, 10)

    def model():
        xs = const("X", x)
        g = sample("gain", lambda: dist.Normal(1.0, 1.0))
        t = sample("t_ant", lambda: dist.Normal(2.0, 3.0))
        mu = det("mu", lambda g_, t_, x_: g_ * t_ * x_, g, t, xs, linear_in=("gain", "t_ant"))
        observe("d", lambda m: dist.Normal(m, sigma), mu, obs=data)

    return trace(model)


def posterior_of(graph, key, *, draws=2000, warmup=1000, chains=1):
    plan = compile_graph(graph)
    return plan.sample(
        key, num_samples=draws, num_warmup=warmup, num_chains=chains, nuts_on_collapse=False
    )


def weights_of(posterior):
    if posterior.log_weights is None:
        n = next(iter(posterior.samples.values())).shape[0]
        return np.full(n, 1.0 / n)
    w = np.exp(np.asarray(posterior.log_weights) - np.max(np.asarray(posterior.log_weights)))
    return w / w.sum()


def prior_draws(graph, key, n):
    """Every node drawn from the prior, observed nodes included (prior predictive).

    Refuses a plated node: `apply_probabilistic` returns an UNMAPPED
    distribution for a plated node with no plated parent, whose `.sample`
    is one shared draw rather than a plate of them -- a prior sampler has to
    expand it to the plate size, which is an R3 task decision, not a probe's.
    """
    for node in graph.nodes:
        if isinstance(node, Probabilistic) and node.plate:
            raise NotImplementedError(f"plated node {node.name!r}: see section 8")

    def one(k):
        env = {}
        keys = jax.random.split(k, len(graph.nodes))
        for node, kk in zip(graph.nodes, keys, strict=True):
            if isinstance(node, Const):
                env[node.name] = node.value
            elif isinstance(node, Deterministic):
                env[node.name] = apply_deterministic(graph, node, env)
            else:
                env[node.name] = apply_probabilistic(graph, node, env).sample(kk)
        return env

    return jax.vmap(one)(jax.random.split(key, n))


# --------------------------------------------------------------- 1. PPC


def ppc(graph, label, key):
    posterior = posterior_of(graph, key)
    samples = posterior.samples
    weights = weights_of(posterior)
    rep = replicated_draws(graph, samples, jax.random.fold_in(key, 7))["d"]
    mu = jax.vmap(lambda v: evaluate(graph, v)["mu"])(samples)
    y = jnp.asarray(graph.node("d").observed)
    x = jnp.asarray(graph.node("X").value)  # the fixture's own grid, not X
    curvature_basis = x**2 - jnp.mean(x**2)

    def curvature(yy):
        return jnp.sum((yy - mu) * curvature_basis, axis=-1)

    def scale(yy):
        return jnp.std(yy - mu, axis=-1)

    out = {}
    for name, stat in (("curvature", curvature), ("scale", scale)):
        t_obs = np.asarray(stat(y))
        t_rep = np.asarray(stat(rep))
        out[name] = float(np.sum(weights * (t_rep >= t_obs)))
    print(f"{label:<22} method={posterior.method:<9} "
          f"p_curvature={out['curvature']:.4f} p_scale={out['scale']:.4f}")
    return out


def run_ppc():
    section("1. PPC discrepancy on the R2 replicated draws")
    ppc(straight_line(), "straight_line (ok)", jax.random.key(1))
    ppc(radiometer(), "radiometer (ok, snis)", jax.random.key(1))
    ppc(curved_line(curvature=0.6), "curved_line 0.6 (bad)", jax.random.key(1))
    ppc(curved_line(curvature=0.15), "curved_line 0.15 (mild)", jax.random.key(1))


# --------------------------------------------------------------- 2. LOO


def typed_posterior(graph, key, *, draws, warmup, chains):
    task = PosteriorTask(
        meta=new_task_meta(label="probe28"),
        budget=ComputeBudget(draws=draws, warmup=warmup, chains=chains),
        nuts_on_collapse=False,
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    result = execute_task(planned, key=key)
    assert not isinstance(result, Refusal), result
    return result


def typed_predictive(graph, posterior, key, *, latent_sites):
    task = PredictiveTask(
        meta=new_task_meta(label="probe28-ppc"),
        source_posterior_ref=ArtifactRef(
            artifact_id=posterior.meta.artifact_id,
            revision=posterior.meta.revision,
            artifact_type=ArtifactKind.RESULT,
        ),
        conditioned_sites=("d",),
        replicated_sites=("d",),
        latent_sites=latent_sites,
    )
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    result = execute_task(planned, key=key, source_posterior=posterior)
    assert not isinstance(result, Refusal), result
    return result


def describe_elpd(elpd):
    fields = {}
    for name in ("elpd", "se", "p", "n_samples", "n_data_points", "warning", "scale"):
        if hasattr(elpd, name):
            fields[name] = getattr(elpd, name)
    k = getattr(elpd, "pareto_k", None)
    if k is not None:
        try:
            fields["max_pareto_k"] = float(np.max(np.asarray(k)))
        except Exception:  # noqa: BLE001 -- a probe reports, it does not judge
            fields["pareto_k"] = repr(k)[:80]
    return fields


def run_loo():
    section("2. LOO through the ArviZ export")
    import arviz as az

    from bayesmith.bridge.arviz import to_inference_data

    graph = straight_line()
    posterior = typed_posterior(graph, jax.random.key(2), draws=2000, warmup=1000, chains=1)
    predictive = typed_predictive(graph, posterior, jax.random.key(3), latent_sites=("w",))
    print("PosteriorResult method:", posterior.representation.method,
          "chain_shape:", posterior.representation.chain_shape)
    flat = to_inference_data(predictive, graph=graph)
    print("flat export log_likelihood dims:", flat["log_likelihood"]["log_likelihood"].dims)
    try:
        print("az.loo on the FLAT draw axis:", describe_elpd(az.loo(flat)))
    except Exception as exc:  # noqa: BLE001
        print("az.loo on the FLAT draw axis raised:", type(exc).__name__, str(exc)[:160])
    n = predictive.pointwise_log_density.value.shape[0]
    chained = to_inference_data(predictive, graph=graph, chain_shape=(1, n))
    print("chain_shape=(1, n) export dims:", chained["log_likelihood"]["log_likelihood"].dims)
    print("az.loo with chain_shape=(1, n):", describe_elpd(az.loo(chained)))
    print("waic available at top level:", hasattr(az, "waic"))

    graph2 = bilinear_pair()
    nuts = typed_posterior(graph2, jax.random.key(4), draws=400, warmup=400, chains=2)
    idata = to_inference_data(nuts, graph=graph2)
    print("bilinear_pair NUTS chain_shape:", nuts.representation.chain_shape,
          "loo:", describe_elpd(az.loo(idata)))


# --------------------------------------------------------------- 3/4. SBC


def sbc_exact(replicates, key, *, draws=100, distort=1.0):
    ranks = []
    cost = []
    for index in range(replicates):
        k_truth, k_noise, k_post = jax.random.split(jax.random.fold_in(key, index), 3)
        w_true = PRIOR_STD * jax.random.normal(k_truth)
        data = w_true * X + SIGMA * jax.random.normal(k_noise, X.shape)
        graph = line_with(data)
        started = time.perf_counter()
        posterior = posterior_of(graph, k_post, draws=draws, warmup=1)
        cost.append(time.perf_counter() - started)
        w = np.asarray(posterior.samples["w"])
        if distort != 1.0:
            w = w.mean() + distort * (w - w.mean())
        ranks.append(float(np.sum(weights_of(posterior) * (w < float(w_true)))))
    ranks = np.asarray(ranks)
    ks = stats.kstest(ranks, "uniform")
    return ranks, ks, float(np.mean(cost)), posterior.method


def run_sbc_exact():
    section(f"3. SBC on the exact route (straight_line, N={SBC_REPLICATES})")
    for distort, label in ((1.0, "correct"), (2.0, "2x too wide"), (0.5, "2x too narrow")):
        ranks, ks, seconds, method = sbc_exact(SBC_REPLICATES, jax.random.key(11), distort=distort)
        hist = np.histogram(ranks, bins=5, range=(0.0, 1.0))[0]
        print(f"{label:<14} method={method:<8} KS D={ks.statistic:.4f} p={ks.pvalue:.4f} "
              f"bins5={hist.tolist()} cost/replicate={seconds:.3f}s")


def run_sbc_power():
    section("4. Replicate count at which a 2x width error is detectable (KS p < 0.05)")
    for n in (10, 20, 50, 100, 200):
        if n > max(SBC_REPLICATES, 20):
            print(f"N={n:<4} skipped (raise argv[1] to run)")
            continue
        _, wide, _, _ = sbc_exact(n, jax.random.key(23), distort=2.0)
        _, narrow, _, _ = sbc_exact(n, jax.random.key(29), distort=0.5)
        _, ok, _, _ = sbc_exact(n, jax.random.key(31), distort=1.0)
        print(f"N={n:<4} KS p: correct={ok.pvalue:.4f} wide={wide.pvalue:.4f} narrow={narrow.pvalue:.4f}")


def run_sbc_nuts():
    section("5. SBC on the NUTS route (bilinear_pair, cost per replicate)")
    replicates = 10
    key = jax.random.key(41)
    ranks = {"gain": [], "t_ant": []}
    cost = []
    x = jnp.linspace(0.5, 3.0, 10)
    for index in range(replicates):
        k_g, k_t, k_noise, k_post = jax.random.split(jax.random.fold_in(key, index), 4)
        g_true = 1.0 + 1.0 * jax.random.normal(k_g)
        t_true = 2.0 + 3.0 * jax.random.normal(k_t)
        data = g_true * t_true * x + 0.3 * jax.random.normal(k_noise, x.shape)
        graph = bilinear_with(data)
        started = time.perf_counter()
        posterior = posterior_of(graph, k_post, draws=100, warmup=100)
        cost.append(time.perf_counter() - started)
        for name, truth in (("gain", g_true), ("t_ant", t_true)):
            draws = np.asarray(posterior.samples[name])
            ranks[name].append(float(np.mean(draws < float(truth))))
    for name, values in ranks.items():
        ks = stats.kstest(np.asarray(values), "uniform")
        print(f"{name:<6} KS D={ks.statistic:.3f} p={ks.pvalue:.3f} (N={replicates}, low power by design)")
    print(f"method={posterior.method} cost/replicate first={cost[0]:.2f}s "
          f"mean(rest)={np.mean(cost[1:]):.2f}s (100 warmup + 100 draws)")


# --------------------------------------------------------------- 6. NPE


def run_npe():
    section("6. Reference NPE calibration (tests/test_amortize.py's linear-Gaussian problem)")
    from bayesmith.amortize import NeuralPosterior, train_posterior
    from tests.test_amortize import M0, S0, A, draw_bank
    from tests.test_amortize import SIGMA as SIG

    theta, data = draw_bank(jax.random.key(0), 2048)
    q = NeuralPosterior.create(theta, data, key=jax.random.key(1), n_components=1)
    started = time.perf_counter()
    q, history = train_posterior(q, theta, data, key=jax.random.key(2), n_steps=1500)
    print(f"train: 2048 bank, 1500 steps, {time.perf_counter() - started:.1f}s, "
          f"best_step={int(history.best_step)}")
    n = 300
    truths = M0 + S0 * jax.random.normal(jax.random.key(5), (n, 1))
    obs = truths * jnp.asarray(A) + SIG * jax.random.normal(jax.random.key(6), (n, len(A)))
    ranks = []
    covered = []
    for index in range(n):
        s = np.asarray(q.sample(obs[index], jax.random.fold_in(jax.random.key(8), index), 200))[:, 0]
        t = float(truths[index, 0])
        ranks.append(float(np.mean(s < t)))
        lo, hi = np.quantile(s, [0.05, 0.95])
        covered.append(lo <= t <= hi)
    ks = stats.kstest(np.asarray(ranks), "uniform")
    print(f"SBC ranks over {n} prior draws: KS D={ks.statistic:.4f} p={ks.pvalue:.4f}; "
          f"90% interval coverage={np.mean(covered):.3f}")


# --------------------------------------------------------------- 7. held-out


def heldout_elpd(graph, key, mask):
    posterior = posterior_of(graph, key)
    samples = posterior.samples
    weights = weights_of(posterior)
    from bayesmith.exact.gaussian import observation_parts

    def parts(v):
        data, loc, scale = observation_parts(graph, evaluate(graph, v))
        return data["d"], loc["d"], scale["d"]

    data, loc, scale = jax.vmap(parts)(samples)
    logp = np.asarray(dist.Normal(loc, scale).log_prob(data))  # (draws, n)
    held = ~np.asarray(mask)
    per_point = np.log(np.sum(weights[:, None] * np.exp(logp[:, held]), axis=0))
    return float(np.sum(per_point)), posterior.method, int(held.sum())


def run_heldout():
    section("7. Held-out prediction through observe(mask=...)")
    mask = np.array([True] * 6 + [False] * 2)
    noise = SIGMA * jax.random.normal(jax.random.key(0), X.shape)
    for label, data in (
        ("straight (ok)", 2.5 * X + noise),
        ("curved 0.6 (bad)", 2.5 * X + 0.6 * X**2 + noise),
    ):
        graph = line_with(data, mask=jnp.asarray(mask))
        pw = pointwise_log_likelihood(graph, posterior_of(graph, jax.random.key(3), draws=50, warmup=1).samples)
        elpd, method, count = heldout_elpd(graph, jax.random.key(3), mask)
        masked_zero = bool(np.all(np.asarray(pw.value)[:, ~mask] == 0.0))
        print(f"{label:<18} method={method:<5} held-out points={count} "
              f"elpd_heldout={elpd:.3f} pointwise LL zero at masked positions={masked_zero}")


# --------------------------------------------------------------- 8. prior predictive


def run_prior_predictive():
    section("8. Prior predictive draws")
    graph = straight_line()
    env = prior_draws(graph, jax.random.key(9), 4000)
    d = np.asarray(env["d"])
    y = np.asarray(graph.node("d").observed)
    print(f"straight_line: prior predictive sd(d[-1])={d[:, -1].std():.3f} "
          f"(w~N(0,{PRIOR_STD}), x={float(X[-1])}); observed d[-1]={y[-1]:.3f}; "
          f"P(max|d_prior| >= max|d_obs|)={np.mean(np.abs(d).max(axis=1) >= np.abs(y).max()):.3f}")
    def vague():
        xs = const("X", X)
        w = sample("w", lambda: dist.Normal(0.0, 1e6))
        mu = det("mu", lambda w_, x_: w_ * x_, w, xs, linear_in=("w",))
        observe("d", lambda m: dist.Normal(m, SIGMA), mu, obs=y)

    env = prior_draws(trace(vague), jax.random.key(9), 4000)
    d = np.asarray(env["d"])
    print(f"vague prior (w~N(0,1e6)): P(max|d_prior| >= max|d_obs|)="
          f"{np.mean(np.abs(d).max(axis=1) >= np.abs(y).max()):.4f}; "
          f"prior predictive sd(d[-1])={d[:, -1].std():.3e}")
    from tests.exact.models import plated_latent

    try:
        prior_draws(plated_latent(), jax.random.key(1), 2)
    except NotImplementedError as exc:
        print("plated node:", exc)


# --------------------------------------------------------------- 9. oracles


def run_oracles():
    section("9. ArviZ as an independent oracle for chain diagnostics; report numbers")
    import arviz as az

    from bayesmith.bridge.arviz import to_inference_data
    from bayesmith.diagnose.identifiability import identifiability
    from bayesmith.diagnose.sensitivity import prior_sensitivity

    graph = bilinear_pair()
    nuts = typed_posterior(graph, jax.random.key(4), draws=400, warmup=400, chains=2)
    samples = {a.name: a.value for a in nuts.representation.draws}
    ours = chain_diagnostics(samples, num_chains=2)
    idata = to_inference_data(nuts, graph=graph)
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    for name in ("gain", "t_ant"):
        print(f"{name:<6} bayesmith r_hat={ours[name].r_hat:.4f} ess={ours[name].ess:.1f} "
              f"| arviz rhat={float(rhat[name].values):.4f} ess_bulk={float(ess[name].values):.1f}")

    # Both diagnostics refuse float32 as the ambient precision (measured: the
    # first run of this probe was refused with `refuse_ambient_float32`), and
    # the graph has to be BUILT inside the x64 block so its constants and data
    # are traced at the wider dtype -- the R3 report projection inherits that.
    with jax.enable_x64(True):
        for label, build in (("straight_line", straight_line), ("collinear_pair", collinear_pair)):
            report = identifiability(build())
            extra = f" participation(0)={report.participation(0)}" if report.nullity else ""
            print(f"identifiability {label:<15} n_par={report.n_par} rank={report.rank} "
                  f"nullity={report.nullity}{extra}")
        for label, build in (("straight_line", straight_line), ("radiometer", radiometer)):
            try:
                report = prior_sensitivity(build())
                name, index, shift = report.worst
                crit = float(report.for_latent(name)["criterion_std"].ravel()[index])
                prior = float(report.for_latent(name)["prior_std"].ravel()[index])
                print(f"prior_sensitivity {label:<14} worst={name}[{index}] shift={shift:+.4f} sigma "
                      f"criterion_std={crit:.4g} prior_std={prior:.4g}")
            except Exception as exc:  # noqa: BLE001
                print(f"prior_sensitivity {label:<14} raised {type(exc).__name__}: {str(exc)[:120]}")


SECTIONS = {
    "1": run_ppc,
    "2": run_loo,
    "3": run_sbc_exact,
    "4": run_sbc_power,
    "5": run_sbc_nuts,
    "6": run_npe,
    "7": run_heldout,
    "8": run_prior_predictive,
    "9": run_oracles,
}


def main() -> None:
    """``probe_28.py [replicates] [section ...]`` -- every section by default."""
    started = time.perf_counter()
    wanted = sys.argv[2:] or list(SECTIONS)
    for number in wanted:
        try:
            SECTIONS[number]()
        except Exception:  # noqa: BLE001 -- keep measuring; the traceback is the record
            traceback.print_exc()
    print(f"\ntotal {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
