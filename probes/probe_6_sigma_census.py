"""Probe 6 -- a mechanical census of the per-sample-sigma vocabulary.

Counts, over `src/bayesmith/**/*.py`:
  * every function whose SIGNATURE carries a sigma-shaped noise parameter
  * every site that CONSUMES a sigma arithmetically (1/sigma**2, log sigma, ...)
  * every site that only PASSES one through

so "which functions assume a per-sample scale" is answered by count, not by
impression.

Run:
    cd <worktree> && PYTHONPATH=$PWD/src \
        /Users/zzhang/projects/bayesmith/.venv/bin/python probes/probe_6_sigma_census.py
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent / "src" / "bayesmith"
NOISE_PARAMS = {"noise_std", "sigma", "sigma_of", "scale", "scales", "noise"}

signatures = []
for path in sorted(ROOT.rglob("*.py")):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        args = node.args
        names = [a.arg for a in args.args + args.kwonlyargs]
        hit = [n for n in names if n in NOISE_PARAMS]
        if hit:
            signatures.append(
                (str(path.relative_to(ROOT.parent.parent)), node.name, hit, node.lineno)
            )

print("=" * 78)
print(f"(a) functions whose signature carries a sigma-shaped parameter: {len(signatures)}")
print("=" * 78)
by_file = {}
for file, fn, hit, line in signatures:
    by_file.setdefault(file, []).append((fn, hit, line))
for file in sorted(by_file):
    print(f"  {file}")
    for fn, hit, line in by_file[file]:
        print(f"      L{line:<5} {fn}({', '.join(hit)})")

print()
print("=" * 78)
print("(b) sites that do sigma ARITHMETIC (not just pass it along)")
print("=" * 78)
patterns = (
    ("1/sigma**2 weight", ("/ self.sigma**2", "1.0 / jnp.asarray", "/ scale**2")),
    ("log-normaliser", ("jnp.log(2.0 * jnp.pi", "log(2 * np.pi", "_LOG_2PI")),
    ("divide by scale", ("/ scale", "/ self.sigma", "/ sigma")),
    ("log sigma", ("jnp.log(scale)", "jnp.log(jnp.asarray(sigma", "log_sigma")),
    ("prior 1/std**2", ("prior_std[name]) ** 2", "prior_std[n]) ** 2")),
)
for label, needles in patterns:
    hits = []
    for path in sorted(ROOT.rglob("*.py")):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if any(nd in line for nd in needles):
                hits.append((str(path.relative_to(ROOT.parent.parent)), i, line.strip()))
    print(f"  [{label}] {len(hits)} site(s)")
    for file, i, line in hits:
        print(f"      {file}:{i}  {line[:76]}")

print()
print("=" * 78)
print("(c) the single choke point: where a non-Normal is refused")
print("=" * 78)
for path in sorted(ROOT.rglob("*.py")):
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if "dist.Normal)" in line or "isinstance(distribution" in line:
            print(f"  {path.relative_to(ROOT.parent.parent)}:{i}  {line.strip()}")
