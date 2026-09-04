"""probe_29 -- the amortized backends, all judged by ONE calibration harness.

R3 Task 9.  §0.11 of the R3 plan says the SBC harness "accepts any
``sampler(datum, key, n) -> draws``", and that Task 9 is where that claim gets
spent: the local reference ``NeuralPosterior`` and every candidate upstream go
through :func:`bayesmith.evaluation.sbc.sbc_ranks`' sampler arm, on the same
bank, the same replicates and the same draw budget, and the verdict comes from
:func:`~bayesmith.evaluation.sbc.sbc_report` rather than from a rank loop
written here.  probe_28 §6 measured the reference with its OWN loop; this probe
does not reuse that arithmetic, because a candidate comparison in which each
candidate is scored by its own code is not a comparison.

Run from the repository root with the test package importable::

    PYTHONPATH=. .venv/bin/python docs/probes/probe_29_amortized_candidates.py
    PYTHONPATH=. .venv/bin/python docs/probes/probe_29_amortized_candidates.py 300 2

Three sections, each independent (a failure in one prints its traceback and the
rest still run):

1. **The control.** The amortize problem's EXACT posterior -- closed form, in
   numpy, from ``tests/test_amortize.py`` -- through the harness on the graph
   this probe builds.  Everything else here rests on that graph's forward law
   being the same joint ``draw_bank`` samples; if it were not, the exact
   posterior of the ONE would be miscalibrated against replicates of the OTHER
   and this section would fail.  It is the boundary check for the whole probe,
   and it is cheap.
2. **The local reference.** ``NeuralPosterior``, bank 2048, 1500 Adam steps at
   batch 256, seeds fixed, through the same harness -- live, using the key the
   harness hands each replicate.
3. **The candidates.** BayesFlow and sbiJAX, each in its own THROWAWAY venv
   that has no bayesmith in it, trained on the same bank at the same budget and
   asked for draws on the same replicate data; the draws come back and are
   judged by the same ``sbc_report``.

Section 3 needs an interpreter per candidate, and refuses to guess::

    PROBE29_BAYESFLOW_PYTHON=/path/to/bf_venv/bin/python \
    PROBE29_SBIJAX_PYTHON=/path/to/sj_venv/bin/python \
    PYTHONPATH=. .venv/bin/python docs/probes/probe_29_amortized_candidates.py 300 3

With neither set it prints what it did not attempt, which is a measurement of
this run rather than a claim about the packages.

**The candidate half of this file runs under the CANDIDATE's interpreter**,
where bayesmith is not installed::

    <candidate-python> probe_29_amortized_candidates.py --candidate sbijax --workdir DIR

It reads ``bank.npz`` and ``tape.npz`` from ``DIR``, trains, writes
``draws_<name>.npz``, and imports nothing from this repository -- which is what
makes "the same bank, the same replicates" checkable rather than asserted.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

# ------------------------------------------------------------------ the budget
#
# One place, because "the same budget" is the claim section 3 makes and a
# second spelling of any of these numbers would be the way it stops being true.

#: The simulation bank every arm trains on: ``draw_bank(key(0), BANK)``.
BANK = 2048
#: Adam gradient steps at ``BATCH``, learning rate ``LEARNING_RATE``, with
#: ``VALIDATION_FRACTION`` of the bank held out and the best-validation
#: parameters returned.  These are ``train_posterior``'s own settings for the
#: reference; each candidate is asked for the same and what it actually spent
#: is recorded, because epoch semantics and early stopping differ per package.
STEPS = 1500
BATCH = 256
LEARNING_RATE = 1e-3
VALIDATION_FRACTION = 0.1

#: Calibration replicates, and posterior draws per replicate.  300 matches
#: probe_28 §6 so the reference arm is comparable to it; 200 likewise.  Both
#: are above :data:`~bayesmith.evaluation.sbc.REPLICATE_FLOOR` (D106 = 100).
REPLICATES = 300
DRAWS = 200

#: Seeds, all fixed (§9.3).  ``KEY_BANK`` draws the bank, ``KEY_INIT`` builds
#: the estimator, ``KEY_TRAIN`` trains it, ``KEY_HARNESS`` is the one key
#: ``sbc_ranks`` splits into a simulation key and per-replicate keys, and
#: ``KEY_DRAW`` seeds a candidate's own sampling in its own venv.
KEY_BANK, KEY_INIT, KEY_TRAIN, KEY_HARNESS, KEY_DRAW = 0, 1, 2, 11, 8

#: The central posterior interval whose coverage is reported beside the KS
#: verdict.  0.90 matches probe_28 §6.  Coverage is read off the harness's own
#: ranks: with uniform weights the rank IS the share of draws below the truth,
#: so the truth lies inside the central ``level`` interval exactly when its
#: rank lies in ``[(1 - level) / 2, 1 - (1 - level) / 2]``.
COVERAGE_LEVEL = 0.90


def section(title):
    print(f"\n=== {title} ===")


def central_coverage(ranks, level: float = COVERAGE_LEVEL) -> float:
    """Share of truths inside the central ``level`` posterior interval."""
    lo = (1.0 - level) / 2.0
    values = np.asarray(ranks)
    return float(np.mean((values >= lo) & (values <= 1.0 - lo)))


# ============================================================================
#  The candidate half: runs under the CANDIDATE's interpreter, imports no
#  bayesmith, and is the only code in this file that may not.
# ============================================================================


def _split(n: int, fraction: float, seed: int = 0):
    """A shuffled train/validation split, mirroring ``train_posterior``'s."""
    order = np.random.default_rng(seed).permutation(n)
    held = round(fraction * n)
    return order[held:], order[:held]


def run_bayesflow(bank, tape, out) -> dict:
    """BayesFlow, offline-trained on the bank, sampled on the tape."""
    os.environ.setdefault("KERAS_BACKEND", "jax")
    import bayesflow as bf
    import keras

    # Keras' PROCESS-GLOBAL seed, because neither `BasicWorkflow` nor
    # `fit_offline` takes one: measured on this checkout, two runs of the
    # identical unseeded command gave final losses 0.452678 and 0.456457 and
    # sample sds 1.9166 and 1.6419, while two seeded runs were identical to
    # every printed digit.  §9.3 asks for a fixed seed; this is where this
    # package keeps it.
    keras.utils.set_random_seed(KEY_TRAIN)

    theta = np.asarray(bank["theta"], dtype="float32")
    data = np.asarray(bank["data"], dtype="float32")
    train, validation = _split(len(theta), VALIDATION_FRACTION)
    # Steps per epoch is what Keras does with a batch size, so the epoch count
    # is derived from the step budget rather than chosen.
    per_epoch = int(np.ceil(len(train) / BATCH))
    epochs = max(1, round(STEPS / per_epoch))

    notes = {}
    # `CouplingFlow` is `BasicWorkflow`'s DEFAULT inference network and it does
    # not build on this pairing; the attempt is recorded rather than skipped,
    # because "the default configuration does not run" is the finding.
    try:
        probe = bf.BasicWorkflow(
            inference_network=bf.networks.CouplingFlow(),
            inference_variables=["theta"],
            inference_conditions=["x"],
            initial_learning_rate=LEARNING_RATE,
        )
        probe.fit_offline(
            {"theta": theta[train][:BATCH], "x": data[train][:BATCH]},
            epochs=1,
            batch_size=BATCH,
            verbose=0,
        )
        notes["coupling_flow"] = "builds"
    except Exception as exc:  # noqa: BLE001 -- the message IS the measurement
        notes["coupling_flow"] = f"{type(exc).__name__}: {str(exc)[:200]}"

    workflow = bf.BasicWorkflow(
        inference_network=bf.networks.FlowMatching(),
        inference_variables=["theta"],
        inference_conditions=["x"],
        initial_learning_rate=LEARNING_RATE,
    )
    started = time.perf_counter()
    history = workflow.fit_offline(
        {"theta": theta[train], "x": data[train]},
        epochs=epochs,
        batch_size=BATCH,
        validation_data={"theta": theta[validation], "x": data[validation]},
        verbose=0,
    )
    trained = time.perf_counter() - started

    observations = np.asarray(tape["observations"], dtype="float32")
    started = time.perf_counter()
    drawn = workflow.approximator.sample(
        num_samples=DRAWS,
        conditions={"x": observations},
        seed=keras.random.SeedGenerator(KEY_DRAW),
    )
    sampled = time.perf_counter() - started
    draws = np.asarray(drawn["theta"])[:, :, 0]

    np.savez(out, draws=draws)
    return {
        "package": "bayesflow",
        "version": bf.__version__,
        "keras": keras.__version__,
        "backend": keras.backend.backend(),
        "network": "FlowMatching",
        "epochs_requested": epochs,
        "epochs_run": len(history.history["loss"]),
        "steps_per_epoch": per_epoch,
        "steps_run": len(history.history["loss"]) * per_epoch,
        "n_train": len(train),
        "n_validation": len(validation),
        "train_seconds": round(trained, 1),
        "sample_seconds": round(sampled, 1),
        "draws_shape": list(draws.shape),
        "draws_dtype": str(draws.dtype),
        "notes": notes,
        "seeding": "keras.utils.set_random_seed (process-global; no seed argument)",
    }


def run_sbijax(bank, tape, out) -> dict:
    """sbiJAX's NPE with a one-component MDN -- the reference's own family."""
    import jax
    import jax.numpy as jnp
    import optax
    import sbijax
    import sbijax.nn as snn

    theta = jnp.asarray(bank["theta"])
    data = jnp.asarray(bank["data"])
    # `n_iter` is EPOCHS here and sbijax makes its own validation split, so the
    # step budget is converted the same way BayesFlow's is.
    per_epoch = int(np.ceil((1.0 - VALIDATION_FRACTION) * len(theta) / BATCH))
    epochs = max(1, round(STEPS / per_epoch))

    objective = sbijax.npe(snn.make_mdn(n_dimension=1, n_components=1))
    started = time.perf_counter()
    params, info = sbijax.train(
        jax.random.key(KEY_TRAIN),
        objective,
        {"y": data, "theta": theta},
        optimizer=optax.adam(LEARNING_RATE),
        n_iter=epochs,
        batch_size=BATCH,
        percentage_data_as_validation_set=VALIDATION_FRACTION,
    )
    trained = time.perf_counter() - started

    observations = jnp.asarray(tape["observations"])
    started = time.perf_counter()
    # sbijax's `sample` conditions on ONE observable: handing it the stack
    # returns 300x200 draws for a single condition, so the loop is required
    # rather than a missed optimisation.
    draws = np.stack(
        [
            np.asarray(
                sbijax.sample(
                    jax.random.fold_in(jax.random.key(KEY_DRAW), index),
                    objective,
                    params,
                    observations[index],
                    n_samples=DRAWS,
                )[0]["theta"]
            ).reshape(-1)[:DRAWS]
            for index in range(observations.shape[0])
        ]
    )
    sampled = time.perf_counter() - started

    np.savez(out, draws=draws)
    return {
        "package": "sbijax",
        "version": sbijax.__version__,
        "jax": jax.__version__,
        "network": "make_mdn(n_dimension=1, n_components=1)",
        "epochs_requested": epochs,
        "epochs_run": int(np.asarray(info.losses).shape[0]),
        "steps_per_epoch": per_epoch,
        "steps_run": int(np.asarray(info.losses).shape[0]) * per_epoch,
        "train_seconds": round(trained, 1),
        "sample_seconds": round(sampled, 1),
        "draws_shape": list(draws.shape),
        "draws_dtype": str(draws.dtype),
        "notes": {"early_stopping": "sbijax.train's default patience is on"},
    }


CANDIDATES = {"bayesflow": run_bayesflow, "sbijax": run_sbijax}


def candidate_main(name: str, workdir: str) -> None:
    """The inner half.  Writes ``draws_<name>.npz`` and a JSON report line."""
    directory = Path(workdir)
    bank = np.load(directory / "bank.npz")
    tape = np.load(directory / "tape.npz")
    record = CANDIDATES[name](bank, tape, directory / f"draws_{name}.npz")
    (directory / f"report_{name}.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


# ============================================================================
#  The orchestrating half: bayesmith's interpreter from here down.
# ============================================================================


@functools.cache
def bayesmith() -> SimpleNamespace:
    """Everything this probe needs from the repository, imported ONCE and late.

    Late because the candidate half of this file runs under an interpreter
    that has no bayesmith in it, and a module-scope import would make that
    half unrunnable -- which is exactly the separation section 3 relies on.
    Cached because ``tests/evaluation/test_sbc.py`` imports this module to
    reuse the fixture below rather than write a second spelling of it.
    """
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from scipy import stats

    from bayesmith import const, det, observe, sample, trace
    from bayesmith.amortize import NeuralPosterior, train_posterior
    from bayesmith.artifacts.base import ArtifactRef
    from bayesmith.artifacts.identity import ArtifactKind
    from bayesmith.evaluation import ALPHA
    from bayesmith.evaluation.sbc import REPLICATE_FLOOR, sbc_ranks, sbc_report
    from tests.dispatch.test_task_protocol import model_ref
    from tests.test_amortize import M0, S0, A, draw_bank, exact_posterior, observation
    from tests.test_amortize import SIGMA as NOISE

    # Spelled out rather than ``SimpleNamespace(**locals())``: the sweep is
    # shorter, but every name above then reads as unused to a linter, and a
    # genuinely dead import in this block would be invisible.
    return SimpleNamespace(
        jax=jax,
        jnp=jnp,
        dist=dist,
        stats=stats,
        const=const,
        det=det,
        observe=observe,
        sample=sample,
        trace=trace,
        NeuralPosterior=NeuralPosterior,
        train_posterior=train_posterior,
        ArtifactRef=ArtifactRef,
        ArtifactKind=ArtifactKind,
        ALPHA=ALPHA,
        REPLICATE_FLOOR=REPLICATE_FLOOR,
        sbc_ranks=sbc_ranks,
        sbc_report=sbc_report,
        model_ref=model_ref,
        M0=M0,
        S0=S0,
        A=A,
        NOISE=NOISE,
        draw_bank=draw_bank,
        exact_posterior=exact_posterior,
        observation=observation,
    )


def amortize_graph(data) -> Any:
    """``tests/test_amortize.py``'s problem as a bayesmith Graph.

    ``theta ~ N(M0, S0)``; ``x = theta * A + N(0, SIGMA)`` on eight points --
    the joint ``draw_bank`` samples, written as a graph so that the
    ``SimulationTask(PRIOR)`` inside :func:`sbc_ranks` draws replicates from
    the SAME joint the bank was drawn from.  Section 1 is what checks that
    claim rather than asserting it: the exact posterior of one problem is not
    calibrated against replicates of another.

    The numbers are imported from ``tests/test_amortize.py`` rather than
    retyped, so the graph and the bank cannot drift apart.
    """
    b = bayesmith()

    def model():
        design = b.const("A", b.jnp.asarray(b.A))
        theta = b.sample("theta", lambda: b.dist.Normal(b.M0, b.S0))
        mu = b.det("mu", lambda t_, a_: t_ * a_, theta, design, linear_in=("theta",))
        b.observe("x", lambda m: b.dist.Normal(m, b.NOISE), mu, obs=data)

    return b.trace(model)


def a_ref():
    """A fresh subject_ref: the sampler arm produces no Result to point at."""
    b = bayesmith()
    return b.ArtifactRef(
        artifact_id=str(uuid.uuid4()),
        revision=0,
        artifact_type=b.ArtifactKind.RESULT,
    )


class Recorder:
    """Wraps a sampler and keeps every ``datum`` it was handed, in call order."""

    def __init__(self, inner):
        self.inner = inner
        self.data: list[dict] = []

    def __call__(self, datum, key, n):
        self.data.append({k: np.asarray(v) for k, v in datum.items()})
        return self.inner(datum, key, n)

    def observations(self) -> np.ndarray:
        return np.stack([item["x"] for item in self.data])


class Replay:
    """Returns pre-computed draws, and REFUSES if they are not this replicate's.

    A candidate's draws are produced in another process, in call order, and the
    only thing pairing replicate i's draws with replicate i's truth is that
    order.  If the harness ever changed how it derives or orders its
    replicates, a silent misalignment would still produce ranks, a KS test and
    a verdict -- all about nothing.  So every call checks the datum it is
    handed against the taped one bit for bit, and raises rather than ranking.
    """

    def __init__(self, tape: np.ndarray, draws: np.ndarray):
        self.tape = tape
        self.draws = draws
        self.index = 0

    def __call__(self, datum, key, n):
        index = self.index
        self.index += 1
        taped = self.tape[index]
        got = np.asarray(datum["x"])
        if got.shape != taped.shape or not np.array_equal(got, taped):
            raise AssertionError(
                f"replicate {index}: the harness handed a datum the tape does "
                "not hold at that position, so these draws belong to a "
                f"different replicate (taped {taped[:3]}..., got {got[:3]}...)"
            )
        return {"theta": self.draws[index][:n]}


def judged(graph, sampler, *, replicates: int | None = None, draws: int | None = None):
    """One arm: the harness's sampler arm, then its verdict.  Returns both.

    ``replicates`` defaults to the module's :data:`REPLICATES` READ AT CALL
    TIME rather than through a default argument, because ``main`` rebinds it
    from ``argv`` -- a default argument would have frozen 300 at import and
    made the command-line count silently ineffective.
    """
    b = bayesmith()
    ranks = b.sbc_ranks(
        graph,
        key=b.jax.random.key(KEY_HARNESS),
        replicates=REPLICATES if replicates is None else replicates,
        model_ref=b.model_ref(),
        sampler=sampler,
        sampler_draws=DRAWS if draws is None else draws,
        subject_ref=a_ref(),
    )
    return ranks, b.sbc_report(ranks)


def describe(label: str, ranks, report) -> None:
    values = np.asarray(ranks.ranks[0])
    test = bayesmith().stats.kstest(values, "uniform")
    print(
        f"{label:<26} {report.conclusion.name:<8} "
        f"KS D={test.statistic:.4f} p={test.pvalue:.4f}  "
        f"{int(COVERAGE_LEVEL * 100)}% coverage={central_coverage(values):.3f}  "
        f"usable={ranks.usable}/{ranks.requested} "
        f"(refused {ranks.refused}, unconverged {ranks.unconverged}, "
        f"undrawn {ranks.undrawn})"
    )


def exact_sampler():
    """The amortize problem's closed-form posterior as a §0.11 sampler."""
    b = bayesmith()

    def sampler(datum, key, n):
        mean, sd = b.exact_posterior(np.asarray(datum["x"]))
        return {"theta": np.asarray(mean + sd * b.jax.random.normal(key, (n,)))}

    return sampler


def train_reference():
    """The local reference at the declared budget: ``(q, history, seconds, bank)``."""
    b = bayesmith()
    theta, data = b.draw_bank(b.jax.random.key(KEY_BANK), BANK)
    start = b.NeuralPosterior.create(
        theta, data, key=b.jax.random.key(KEY_INIT), n_components=1
    )
    started = time.perf_counter()
    fitted, history = b.train_posterior(
        start,
        theta,
        data,
        key=b.jax.random.key(KEY_TRAIN),
        n_steps=STEPS,
        batch_size=BATCH,
        learning_rate=LEARNING_RATE,
        validation_fraction=VALIDATION_FRACTION,
    )
    return fitted, history, time.perf_counter() - started, (theta, data)


def reference_sampler(q):
    """``NeuralPosterior`` as a §0.11 sampler: ``(n, 1)`` draws flattened to ``(n,)``."""
    jnp = bayesmith().jnp

    def sampler(datum, key, n):
        return {"theta": np.asarray(q.sample(jnp.asarray(datum["x"]), key, n))[:, 0]}

    return sampler


def width_against_exact(q, theta_true: float, draws: int = 20_000):
    """``(width / exact_width, (mean - exact_mean) / exact_width)`` at one datum.

    The one instrument here whose target is EXACT: ``exact_posterior`` is
    closed form in numpy, so this ratio says how good the estimator is without
    reference to any measurement of this machine's.
    """
    b = bayesmith()
    x = b.observation(theta_true)
    mean, sd = b.exact_posterior(x)
    drawn = q.sample(x, b.jax.random.key(5), draws)
    return float(b.jnp.std(drawn)) / sd, (float(b.jnp.mean(drawn)) - mean) / sd


#: The three observations ``tests/test_amortize.py`` scores its own estimator on.
WIDTH_OBSERVATIONS = (0.5, 1.6, -0.9)


# ---------------------------------------------------------------- 1. control


def run_control():
    section("1. Control: the amortize problem's EXACT posterior through the harness")
    b = bayesmith()
    _, data = b.draw_bank(b.jax.random.key(KEY_BANK), BANK)
    graph = amortize_graph(np.asarray(data[0]))
    print(f"graph latents={graph.latents} observed={graph.observed}")
    ranks, report = judged(graph, exact_sampler())
    describe("exact posterior", ranks, report)
    print(
        "  This PASS is what says the graph's forward law IS the bank's joint: "
        "the exact posterior of one problem would not be calibrated against "
        "replicates of another."
    )


# -------------------------------------------------------------- 2. reference


def run_reference():
    section("2. The local reference NeuralPosterior through the SAME harness")
    q, history, seconds, (_, data) = train_reference()
    print(
        f"train: bank {BANK}, {STEPS} steps, batch {BATCH}, lr {LEARNING_RATE}, "
        f"{seconds:.1f}s, best_step={int(history.best_step)}"
    )
    ranks, report = judged(amortize_graph(np.asarray(data[0])), reference_sampler(q))
    describe("reference NeuralPosterior", ranks, report)
    for theta_true in WIDTH_OBSERVATIONS:
        ratio, bias = width_against_exact(q, theta_true)
        print(
            f"  theta_true={theta_true:+.1f}  width/exact={ratio:.4f}  "
            f"(mean-exact)/exact_sd={bias:+.4f}"
        )


# ------------------------------------------------------------- 3. candidates


def run_candidates():
    section("3. Candidate upstreams: same bank, same replicates, same harness")
    b = bayesmith()
    theta, data = b.draw_bank(b.jax.random.key(KEY_BANK), BANK)
    graph = amortize_graph(np.asarray(data[0]))

    # One recording pass fixes the replicate data every candidate is scored on.
    # The sampler it wraps is irrelevant -- only the data handed to it is kept
    # -- so the cheapest possible one is used.
    recorder = Recorder(lambda datum, key, n: {"theta": np.zeros(n)})
    judged(graph, recorder)
    observations = recorder.observations()
    print(f"tape: {observations.shape} observations, from key({KEY_HARNESS})")

    with tempfile.TemporaryDirectory(prefix="probe29-") as workdir:
        directory = Path(workdir)
        np.savez(
            directory / "bank.npz", theta=np.asarray(theta), data=np.asarray(data)
        )
        np.savez(directory / "tape.npz", observations=observations)

        for name in CANDIDATES:
            python = os.environ.get(f"PROBE29_{name.upper()}_PYTHON")
            if python is None:
                print(
                    f"{name:<26} NOT ATTEMPTED -- set "
                    f"PROBE29_{name.upper()}_PYTHON to a venv's interpreter"
                )
                continue
            started = time.perf_counter()
            finished = subprocess.run(
                [
                    python,
                    str(Path(__file__).resolve()),
                    "--candidate",
                    name,
                    "--workdir",
                    str(directory),
                ],
                capture_output=True,
                text=True,
                check=False,
                # Emptied so the candidate cannot reach this repository's
                # source: "it has no bayesmith in it" is the claim.
                env={**os.environ, "PYTHONPATH": ""},
            )
            elapsed = time.perf_counter() - started
            if finished.returncode != 0:
                print(f"{name:<26} EXIT {finished.returncode} after {elapsed:.1f}s")
                print(finished.stderr[-2000:])
                continue
            record = json.loads((directory / f"report_{name}.json").read_text())
            print(f"{name:<26} {json.dumps(record)}")
            drawn = np.load(directory / f"draws_{name}.npz")["draws"]
            ranks, report = judged(graph, Replay(observations, drawn))
            describe(name, ranks, report)


SECTIONS = {"1": run_control, "2": run_reference, "3": run_candidates}


def main() -> None:
    """``probe_29.py [replicates] [section ...]`` -- every section by default."""
    global REPLICATES
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        REPLICATES = int(sys.argv[1])
    b = bayesmith()
    print(
        f"bank={BANK} steps={STEPS} batch={BATCH} lr={LEARNING_RATE} "
        f"replicates={REPLICATES} draws={DRAWS} "
        f"alpha={b.ALPHA} floor={b.REPLICATE_FLOOR}"
    )
    started = time.perf_counter()
    for number in sys.argv[2:] or list(SECTIONS):
        try:
            SECTIONS[number]()
        except Exception:  # noqa: BLE001 -- keep measuring; the traceback is the record
            traceback.print_exc()
    print(f"\ntotal {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    if "--candidate" in sys.argv:
        arguments = sys.argv[1:]
        candidate_main(
            arguments[arguments.index("--candidate") + 1],
            arguments[arguments.index("--workdir") + 1],
        )
    else:
        main()
