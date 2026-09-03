"""probe_30 -- the SBC replicate floor D106, re-measured over ten seeds.

The R3 plan's §0.6 set D106 = 50 from probe_28 §4, which swept ONE seed per
distortion (`key(23)` wide, `key(29)` narrow, `key(31)` correct).  A single
draw of a p-value is a sample from a distribution, not a property of the
estimator, so Task 6.1 requires the sweep be repeated across ten seeds and the
WORST case -- not the median, not the one that was measured first -- be the
number the floor is registered against.

Ten seeds, `jax.random.key(23 + k)` for k in 0..9, exactly as the plan names
them.  Run from the repository root with the test package importable:

    PYTHONPATH=. .venv/bin/python docs/probes/probe_30_sbc_replicate_floor.py

**One posterior fit serves all three distortions and all three replicate
counts.**  probe_28's `sbc_exact` applies `distort` by rescaling the posterior
draws about their own mean AFTER the fit, and `fold_in(key, index)` makes
replicate i of an N=20 sweep bit-identical to replicate i of an N=100 sweep.
So 100 fits per seed cover the whole grid, and the numbers printed here are
the numbers probe_28's own loop would print -- the helpers below are imported
from it rather than re-typed, so "the same sweep" is checked by the import
rather than asserted in a comment.

What the verdict line reports: the largest (worst, least detectable) KS p over
the ten seeds for each misspecification, at each N.  The plan's rule is that
if the worst 2x-wide or 2x-narrow p at N = 50 crosses alpha/3 = 0.0166667,
the floor rises to 100.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import jax
import numpy as np
from scipy import stats

_HERE = Path(__file__).resolve().parent


def _load_probe_28():
    """Import probe_28 as a module without letting it read our argv.

    Its `SBC_REPLICATES` is computed from `sys.argv[1]` at import time, so a
    probe_30 invocation carrying arguments would be parsed by probe_28's
    module body.  Blanking argv for the duration is the whole trick.
    """
    spec = importlib.util.spec_from_file_location(
        "probe_28", _HERE / "probe_28_model_checking_seams.py"
    )
    module = importlib.util.module_from_spec(spec)
    saved = sys.argv
    sys.argv = [saved[0]]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


P28 = _load_probe_28()

SEEDS = tuple(23 + k for k in range(10))
COUNTS = (20, 50, 100)
DISTORTIONS = (("correct", 1.0), ("wide", 2.0), ("narrow", 0.5))
ALPHA = 0.05


def one_seed(seed: int, replicates: int):
    """`(w_true, draws, weights)` for each replicate of probe_28's exact sweep."""
    key = jax.random.key(seed)
    records = []
    for index in range(replicates):
        k_truth, k_noise, k_post = jax.random.split(jax.random.fold_in(key, index), 3)
        w_true = P28.PRIOR_STD * jax.random.normal(k_truth)
        data = w_true * P28.X + P28.SIGMA * jax.random.normal(k_noise, P28.X.shape)
        posterior = P28.posterior_of(P28.line_with(data), k_post, draws=100, warmup=1)
        records.append(
            (
                float(w_true),
                np.asarray(posterior.samples["w"]),
                P28.weights_of(posterior),
            )
        )
    return records


def ks_p(records, count: int, distort: float) -> float:
    ranks = []
    for w_true, draws, weights in records[:count]:
        w = draws if distort == 1.0 else draws.mean() + distort * (draws - draws.mean())
        ranks.append(float(np.sum(weights * (w < w_true))))
    return float(stats.kstest(np.asarray(ranks), "uniform").pvalue)


def main() -> None:
    started = time.perf_counter()
    print(f"seeds={list(SEEDS)} counts={list(COUNTS)} alpha={ALPHA} alpha/3={ALPHA / 3:.7f}")
    table: dict[tuple[str, int], list[float]] = {
        (label, count): [] for label, _ in DISTORTIONS for count in COUNTS
    }
    for seed in SEEDS:
        clock = time.perf_counter()
        records = one_seed(seed, max(COUNTS))
        row = []
        for label, distort in DISTORTIONS:
            for count in COUNTS:
                p = ks_p(records, count, distort)
                table[(label, count)].append(p)
                row.append(f"{label[0]}{count}={p:.4f}")
        print(f"key({seed}) {' '.join(row)}  [{time.perf_counter() - clock:.1f}s]")

    print("\nworst (largest, least detectable) KS p over the ten seeds")
    for label, _ in DISTORTIONS:
        for count in COUNTS:
            values = table[(label, count)]
            print(
                f"  {label:<8} N={count:<4} worst={max(values):.4f} "
                f"median={float(np.median(values)):.4f} best={min(values):.4f}"
            )

    worst_50 = max(max(table[("wide", 50)]), max(table[("narrow", 50)]))
    worst_100 = max(max(table[("wide", 100)]), max(table[("narrow", 100)]))
    print(
        f"\nD106 verdict: worst misspecified p at N=50 is {worst_50:.4f}; "
        f"alpha/3 = {ALPHA / 3:.7f}; "
        f"{'CROSSES -- raise the floor to 100' if worst_50 > ALPHA / 3 else 'below -- 50 stands'}"
    )
    print(f"             worst misspecified p at N=100 is {worst_100:.4f}")
    print(f"\ntotal {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
