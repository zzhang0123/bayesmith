"""probe_36 -- the residual-evidence backend survey, and the probe that is a bypass.

Run:  .venv/bin/python docs/probes/probe_36_backend_survey.py

R5 Task 4.3 and 4.4. Six sections, and the point of writing them as a script is
that the R5 plan's 0.16 carries a planning-time version of most of it which this
re-measures rather than quotes -- three of its numbers had already moved by the
time this ran.

**Every section reports one of three verdicts, and the third is why the script
exists.** MEASURED, ABSENT (the candidate is not installed, which is this
repository's default state), or DECLINED (the measurement could not be made --
no network, a metadata backend that raised). Red line 14: a check that can
decline to run must record that it declined, in a value distinct from "ran and
found nothing", because otherwise its silence reads as its pass.

Run it twice to see all of that: once in this repository's own environment,
where sections 2, 5 and 6 report ABSENT and section 3 still runs in full, and
once in an environment carrying both candidates.

    1.  This package's own capability probe, in the environment you ran it in.
    2.  The install survey: version, declared `jax` bound, **whether that bound
        actually admits the `jax` installed here** (evaluated, not printed side
        by side and left to the reader -- an earlier version of this docstring
        claimed the comparison and the code only printed the two numbers), and
        whether the two candidates coexist in one environment.
    2b. The nested-sampling entry point's real signature, which Task 4.3 asks
        for and the first version of this script omitted.
    3.  The three states -- absent, broken, present -- each in a fresh
        subprocess, with `jax_enable_x64` printed before and after. The BROKEN
        state is built here rather than waited for: a module that writes the
        flag at import and then raises `AttributeError`, which is what plan 0.16
        measured a partially installed `jaxns` doing and is NOT the exception
        `except ImportError` catches. Section 3 runs the naive probe against
        that module as a CONTROL: if it did not poison the process, this
        section's clean results would be measuring nothing.
    4.  Whether importing the real candidate writes `jax.config`.
    5.  Release date and open-issue count, with the date read.
    6.  The smallest end-to-end run against a closed-form 1-D Gaussian: the
        error, the reported uncertainty, and where the wall clock went.
"""

from __future__ import annotations

import datetime
import importlib
import importlib.metadata
import inspect
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bayesmith.dispatch.task import (
    RESIDUAL_EVIDENCE_EXTRAS,
    install_command,
    residual_evidence_extras,
)

MEASURED, ABSENT, DECLINED = "MEASURED", "ABSENT", "DECLINED"

#: Plan 0.16's measured failure shape, reproduced deterministically: the flag is
#: written at module scope and the import then dies with the exception a
#: capability probe does not catch.
POISONING_MODULE = (
    "import jax\n"
    "jax.config.update('jax_enable_x64', True)\n"
    "raise AttributeError(\n"
    "    \"module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'\"\n"
    ")\n"
)
QUIET_MODULE = "STATUS = 'importable'\n"


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def write_distribution(root: Path, name: str, source: str, version="9.9.9") -> None:
    """A findable distribution whose module has not been imported."""
    (root / f"{name}.py").write_text(source, encoding="utf-8")
    info = root / f"{name}-{version}.dist-info"
    info.mkdir(exist_ok=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )
    (info / "RECORD").write_text("", encoding="utf-8")


# ---------------------------------------------------------------- 1. the probe

rule("1. this package's capability probe, in the environment this ran in")
for status in residual_evidence_extras():
    print(
        f"  {status.extra:>10}: {status.state:<9} version={status.version!r:<10} "
        f"{status.detail}"
    )
print(f"  install commands: {[install_command(e) for e, _d in RESIDUAL_EVIDENCE_EXTRAS]}")
print(f"  jax_enable_x64 after running the probe: {jax.config.jax_enable_x64}")

# --------------------------------------------------------------- 2. the survey

rule("2. install survey: declared jax bound against the jax installed here")
installed_jax = importlib.metadata.version("jax")
print(f"  jax installed here: {installed_jax}")
print(
    "  NOTE: pyproject declares `jax>=0.5`, an open lower bound. 0.11.1 is what is "
    "INSTALLED, not what is pinned -- 1.5 condition 2 is a question about this "
    "stack, and a future jax is a different question."
)
def _admits(requirement: str, version: str) -> str:
    """Does a PEP 508 requirement admit this version? Answered, not displayed.

    Compares release segments as integers, so `0.11.1` sorts above `0.9.0` --
    a string comparison says the opposite and would have reported blackjax's
    `jax>=0.9.0` as excluding the installed 0.11.1.
    """
    match = re.search(r"(>=|>|==|<=|<|~=)\s*([0-9][0-9.]*)", requirement)
    if match is None:
        return f"{requirement}: no version bound, so anything is admitted"
    operator, bound = match.group(1), match.group(2)

    def parts(text):
        return tuple(int(piece) for piece in text.split(".") if piece.isdigit())

    have, want = parts(version), parts(bound)
    verdicts = {
        ">=": have >= want,
        ">": have > want,
        "==": have == want,
        "<=": have <= want,
        "<": have < want,
        "~=": have >= want,
    }
    return f"{requirement}: {'ADMITS' if verdicts[operator] else 'EXCLUDES'} {version}"


present: list[str] = []

for extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        print(f"  {distribution:>10}: {ABSENT} -- not installed in this environment")
        continue
    requires = importlib.metadata.requires(distribution) or []
    # A requirement is "jax>=0.9.0", not "jax >= 0.9.0", so splitting on
    # whitespace returns the whole string and the filter below silently matches
    # nothing. It did, in this script's first run: blackjax reported `[]` where
    # it declares `jax>=0.9.0`. Cut at the first character that is not part of a
    # distribution name.
    jax_bounds = [
        requirement
        for requirement in requires
        if re.split(r"[^A-Za-z0-9._-]", requirement, maxsplit=1)[0].lower()
        in ("jax", "jaxlib")
    ]
    admits = [_admits(bound, installed_jax) for bound in jax_bounds]
    print(
        f"  {distribution:>10}: {MEASURED} {version}, requires {jax_bounds}, "
        f"{len(requires)} requirements in all"
    )
    print(f"  {'':>10}  does that admit jax {installed_jax}? {admits}")
    present.append(distribution)

if len(present) == len([d for _e, d in RESIDUAL_EVIDENCE_EXTRAS]):
    print(f"  COEXIST: {MEASURED} -- all of {present} resolve in ONE environment, ")
    print(f"           and jax is {installed_jax} with all of them installed")
elif present:
    print(f"  COEXIST: {ABSENT} -- only {present} installed here; run this script ")
    print("           again in an environment carrying every candidate")
else:
    print(f"  COEXIST: {ABSENT} -- no candidate installed in this environment")

rule("2b. the nested-sampling entry point's real signature")
ENTRY_POINTS = {
    "blackjax": [("blackjax", "nss"), ("blackjax.ns.nss", "as_top_level_api")],
    "jaxns": [("jaxns", "NestedSampler"), ("jaxns", "Model"), ("jaxns", "Prior")],
}
for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
    try:
        importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        print(f"  {distribution:>10}: {ABSENT} -- not installed in this environment")
        continue
    for module_name, attribute in ENTRY_POINTS.get(distribution, []):
        # Imported deliberately, in THIS process, and only in the section whose
        # subject is the API. Everything above answers without importing; that
        # is the probe's contract, not this script's.
        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, attribute)
            target = obj.__init__ if isinstance(obj, type) else obj
            print(f"  {distribution:>10}: {MEASURED} {module_name}.{attribute}"
                  f"{inspect.signature(target)}")
        except BaseException as error:  # noqa: BLE001 -- the failure IS the finding
            print(f"  {distribution:>10}: {DECLINED} {module_name}.{attribute} -- "
                  f"{type(error).__name__}: {error}")

# ------------------------------------------------- 3. the three states, shown

rule("3. absent / broken / present -- one subprocess each, the flag printed")

#: Names nothing has ever published, so the ABSENT row is absent in every
#: environment this script can be run in. The first version of this section
#: swept the real candidate names, and in an environment carrying both of them
#: the row labelled "absent" was measuring the present state and saying
#: otherwise -- the disease the whole script is about, in the script.
SYNTHETIC = ("bayesmith_probe36_alpha", "bayesmith_probe36_beta")

STATE_SCRIPT = textwrap.dedent(
    """
    import json, sys
    sys.path.insert(0, {src!r})
    import jax
    from bayesmith.dispatch.task import optional_extra_status

    names = json.loads(sys.argv[1])
    out = {{"start": jax.config.jax_enable_x64}}
    out["probe"] = [
        (s.distribution, s.state, s.version)
        for s in (optional_extra_status(n, n) for n in names)
    ]
    out["after_probe"] = jax.config.jax_enable_x64
    out["imported"] = sorted(n for n in names if n in sys.modules)

    naive = {{}}
    for name in names:
        try:
            __import__(name)
            naive[name] = "imported"
        except ImportError:
            naive[name] = "ImportError"
        except BaseException as error:
            naive[name] = type(error).__name__
    out["naive"] = naive
    out["after_naive"] = jax.config.jax_enable_x64
    print(json.dumps(out))
    """
).format(src=str(Path(__file__).resolve().parents[2] / "src"))


def run_state(label: str, names: tuple[str, ...], module_source: str | None) -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        if module_source is not None:
            for name in names:
                write_distribution(root, name, module_source)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONPATH": str(root)}
        proc = subprocess.run(
            [sys.executable, "-c", STATE_SCRIPT, json.dumps(list(names))],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            print(f"  {label}: {DECLINED} -- subprocess exit {proc.returncode}")
            print(textwrap.indent(proc.stderr[-1500:], "      "))
            return
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(f"  {label}:")
    print(f"      x64 at start                     {out['start']}")
    print(f"      probe result                     {out['probe']}")
    print(f"      x64 after this package's probe   {out['after_probe']}")
    print(f"      candidate modules imported by it {out['imported']}")
    print(f"      naive `except ImportError` saw   {out['naive']}")
    print(f"      x64 after the naive probe        {out['after_naive']}")


run_state("absent  (published under no name)", SYNTHETIC, None)
run_state("present (a module that imports cleanly)", SYNTHETIC, QUIET_MODULE)
run_state(
    "broken  (writes jax.config, then raises AttributeError)", SYNTHETIC, POISONING_MODULE
)
run_state(
    "the real candidate names, in THIS environment",
    tuple(distribution for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS),
    None,
)
print(
    "  The broken row is the control: `after_naive` True there is what makes the\n"
    "  False in `after_probe` a measurement rather than a coincidence."
)

# ------------------------------------------- 4. the real candidates on import

rule("4. does importing the REAL candidate write jax.config?")
IMPORT_SCRIPT = textwrap.dedent(
    """
    import json, sys
    import jax
    out = {"before": jax.config.jax_enable_x64}
    try:
        __import__(sys.argv[1])
        out["import"] = "ok"
    except BaseException as error:
        out["import"] = f"{type(error).__name__}: {error}"
    out["after"] = jax.config.jax_enable_x64
    print(json.dumps(out))
    """
)
for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
    try:
        importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        print(f"  {distribution:>10}: {ABSENT} -- not installed in this environment")
        continue
    proc = subprocess.run(
        [sys.executable, "-c", IMPORT_SCRIPT, distribution],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(f"  {distribution:>10}: {DECLINED} -- subprocess exit {proc.returncode}")
        continue
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    print(
        f"  {distribution:>10}: {MEASURED} import={out['import']!r} "
        f"x64 {out['before']} -> {out['after']}"
    )

# ------------------------------------------------------ 5. release provenance

rule("5. release date and open issues, with the date read")
READ_AT = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
print(f"  read at {READ_AT}")

REPOSITORIES = {"blackjax": "blackjax-devs/blackjax", "jaxns": "Joshuaalbert/jaxns"}


def fetch(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": "bayesmith-probe-36"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
    try:
        data = fetch(f"https://pypi.org/pypi/{distribution}/json")
        version = data["info"]["version"]
        files = data["releases"][version]
        stamp = min(f["upload_time_iso_8601"] for f in files) if files else None
        print(
            f"  {distribution:>10}: {MEASURED} latest {version} uploaded {stamp}, "
            f"{len(data['releases'])} releases on the index"
        )
    except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
        print(f"  {distribution:>10}: {DECLINED} pypi -- {type(error).__name__}: {error}")
    repository = REPOSITORIES.get(distribution)
    if repository is None:
        print(f"  {distribution:>10}: {DECLINED} github -- no repository recorded")
        continue
    try:
        info = fetch(f"https://api.github.com/repos/{repository}")
        search = fetch(
            f"https://api.github.com/search/issues?q=repo:{repository}"
            f"+type:issue+state:open"
        )
        print(
            f"  {distribution:>10}: {MEASURED} {search['total_count']} open issues "
            f"({info['open_issues_count']} including PRs), pushed {info['pushed_at']}"
        )
    except (urllib.error.URLError, OSError, KeyError, ValueError) as error:
        print(
            f"  {distribution:>10}: {DECLINED} github -- {type(error).__name__}: {error}"
        )

# ------------------------------------------------------- 6. the smallest run

rule("6. smallest end-to-end run against a closed-form 1-D Gaussian")

N, SIGMA, S0 = 8, 0.5, 2.0
DATA = np.asarray([0.9, 1.4, 0.7, 1.1, 1.6, 0.8, 1.2, 1.0], dtype=np.float64)


def closed_form_log_evidence() -> float:
    """``d_i ~ N(w, SIGMA)``, ``w ~ N(0, S0)``: the determinant-lemma answer.

    Built in numpy float64 from exact decimal inputs, so it is the same number
    on every machine this could run on (red line 9): no sampler, no quadrature
    and no jax touches it.
    """
    design = np.ones((N, 1))
    covariance = S0**2 * (design @ design.T) + SIGMA**2 * np.eye(N)
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    quadratic = DATA @ np.linalg.solve(covariance, DATA)
    return float(-0.5 * (quadratic + logdet + N * np.log(2 * np.pi)))


TRUE = closed_form_log_evidence()
print(f"  closed form log Z = {TRUE!r}")

# The runs below are at x64, which is the premise R4's evidence gate states and
# the one jaxns refuses to run without. Written here, in a probe, and never in
# `src/`: this package's rule is that the CALLER opens the context. It is set
# rather than entered because both candidates capture the dtype inside a jit
# whose lifetime outlives a `with` block.
jax.config.update("jax_enable_x64", True)
_data = jnp.asarray(DATA)


def loglikelihood(w):
    return jnp.sum(jax.scipy.stats.norm.logpdf(_data, jnp.asarray(w).reshape(()), SIGMA))


def logprior(w):
    return jax.scipy.stats.norm.logpdf(jnp.asarray(w).reshape(()), 0.0, S0)


def run_blackjax(seed: int) -> dict:
    import blackjax
    import blackjax.ns.utils as nsutils

    live, delete, inner = 200, 10, 5  # inner = max(5, 2 * dim), dim = 1
    key = jax.random.key(seed)
    key, sub = jax.random.split(key)
    algorithm = blackjax.nss(
        logprior_fn=logprior,
        loglikelihood_fn=loglikelihood,
        num_inner_steps=inner,
        num_delete=delete,
    )
    state = algorithm.init(jax.random.normal(sub, (live, 1)) * S0)
    step = jax.jit(algorithm.step)

    dead, steps = [], 0
    start = time.perf_counter()
    for _ in range(400):
        key, sub = jax.random.split(key)
        state, info = step(sub, state)
        dead.append(info)
        steps += 1
        # blackjax supplies NO termination condition; this rule is the caller's,
        # which is 1.5 condition 5 scope landing on bayesmith rather than on the
        # library.
        if bool(state.integrator.logZ_live - state.integrator.logZ < -3.0):
            break
    loop = time.perf_counter() - start

    start = time.perf_counter()
    final = nsutils.finalise(state, dead)
    key, sub = jax.random.split(key)
    realisations = jax.scipy.special.logsumexp(
        nsutils.log_weights(sub, final, shape=100), axis=0
    )
    post = time.perf_counter() - start
    return {
        "log_Z": float(jnp.mean(realisations)),
        "uncert": float(jnp.std(realisations)),
        "work": f"{steps} steps x {delete} deleted",
        "loop": loop,
        "post": post,
        "termination": "the caller's rule (logZ_live - logZ < -3); blackjax has none",
    }


def run_jaxns(seed: int) -> dict:
    import tensorflow_probability.substrates.jax as tfp
    from jaxns import Model, NestedSampler, Prior

    def prior_model():
        w = yield Prior(tfp.distributions.Normal(loc=0.0, scale=S0), name="w")
        return w

    model = Model(prior_model=prior_model, log_likelihood=loglikelihood)
    sampler = NestedSampler(model=model, max_samples=20000, num_live_points=200)
    start = time.perf_counter()
    reason, state = jax.jit(sampler)(jax.random.PRNGKey(seed))
    results = sampler.to_results(termination_reason=reason, state=state)
    loop = time.perf_counter() - start
    return {
        "log_Z": float(results.log_Z_mean),
        "uncert": float(results.log_Z_uncert),
        "work": f"{int(results.total_num_samples)} samples",
        "loop": loop,
        "post": 0.0,
        "termination": f"jaxns termination_reason={int(reason)}",
    }


RUNNERS = {"blackjax": run_blackjax, "jaxns": run_jaxns}

for _extra, distribution in RESIDUAL_EVIDENCE_EXTRAS:
    runner = RUNNERS.get(distribution)
    if runner is None:
        print(f"  {distribution:>10}: {DECLINED} -- no run recorded for this candidate")
        continue
    try:
        importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        print(f"  {distribution:>10}: {ABSENT} -- not installed in this environment")
        continue
    try:
        out = runner(0)
    except BaseException as error:  # noqa: BLE001 -- the failure IS the finding
        print(f"  {distribution:>10}: {DECLINED} -- {type(error).__name__}: {error}")
        continue
    error_nats = out["log_Z"] - TRUE
    print(
        f"  {distribution:>10}: {MEASURED} log_Z={out['log_Z']:+.6f} "
        f"err={error_nats:+.6f} reported_sigma={out['uncert']:.6f} "
        f"err/sigma={error_nats / out['uncert']:+.2f}"
    )
    print(
        f"  {'':>10}  work={out['work']} loop={out['loop']:.2f}s "
        f"post={out['post']:.2f}s"
    )
    print(f"  {'':>10}  termination: {out['termination']}")

print()
print(
    "  ONE seed each. That is enough to say both reach the closed form and not\n"
    "  nearly enough to say anything about bias -- plan 0.16 records that four\n"
    "  seeds gave mixed signs, and separating a bias from scatter needs ~100.\n"
    "  1.5 condition 3 is Task 5's, on Task 3's fixtures, not this script's."
)

