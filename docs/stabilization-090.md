# 0.9.0 stabilization review — T-002

> **文档状态：`record`** · 2026-09-19 的本地收尾与验收记录；本地未发布候选验收已完成。公共契约见 `stability.md`，架构裁决见唯一顶层设计。

## Release decision

The user selected a 0.9.0 stable baseline. This freezes the supported architecture
and API contracts while preserving explicit experimental and unavailable routes.
It is not a 1.0 declaration or a new scientific validation of the TRIS/Campbell
applications. The starting checkout was `f53e10a`, with substantial pre-existing
uncommitted research work. T-001 and its unresolved recovery evidence are retained.

This record covers a local candidate. No Git commit, tag, push, PyPI upload or
GitHub Pages deployment has been performed by T-002. At the user's request, the
0.9.0 candidate remains unpublished and the documentation is English-only. Every
served HTML page requests `noindex, nofollow, noarchive`; Pages publication requires
manual opt-in and no longer runs on push. These crawler directives do not provide
access control. New CI workflows define
future checks; they are not evidence of a remote run.

## Architecture and correctness

The design retains one authoritative top-level specification and an exhaustive
module ownership map. Two internal boundaries were corrected without changing
supported import identities: prior-environment construction now lives below
`dispatch` in `exact._environment`; task identity projection is isolated in
`dispatch._task_identity`. Typed tasks/results/refusals and the existing artifact
codec remain the public boundary.

Numerical review concentrated on the experimental reweighting interface and the
shared graph evaluator. Graph-derived density ratios now check coordinate shapes
and base measures, enforce support, and refuse unsupported implicit plate
broadcasting. Target samples outside support have zero weight; invalid reference
samples are rejected. Missing observed values use distribution masking before
log-probability evaluation, preventing invalid placeholders from poisoning
otherwise valid gradients. Masks on event-valued observations remain explicitly
refused where the existing graph contract cannot represent them correctly.

Stable, experimental and unavailable capabilities are documented separately.
Whole-graph analytic evidence is available. The low-level JAXNS adapter does not
make the public residual-evidence route executable. Posterior recycling still
requires overlap and convergence evidence; a finite estimate is not validation.

## Independent review and improvement rounds

| Perspective | Finding and resulting change | Re-review |
|---|---|---|
| Architecture | Removed the lower-layer dependency on dispatch; separated identity projection while retaining aliases. | Architect and independent code reviewer approved. |
| Numerical semantics | Added measure/support checks, masked-gradient safety, broadcast-aware observation signatures and explicit plate-axis refusal. | Python reviewer constructed negative controls and confirmed the final fixes. |
| Distribution | Made the portable suite self-contained; retained two original probe oracles; rejected noncanonical archive paths and unexpected wheel installation paths. | Independent release reviewer approved after fixes. |
| Documentation | Resolved facade exports and joined `__all__`; validated generated page membership, local links, anchors and supported CSS resource syntax. | Python review confirmed the current site's coverage and stated parser limits. |
| Browser/accessibility | Fixed narrow-screen API headings and named the search dialog; checked keyboard focus and navigation. | Independent initial browser review plus final Chrome regression. |
| JavaScript | Reviewed search rendering, relative links, theme and mobile menu state. | Independent JavaScript reviewer approved. |

The review overview is in `runs/t002/review-matrix.json`. The reviews covered
implemented behavior and demonstrated bypasses rather than counting approvals.

## Documentation as a user journey

The site has six navigation groups: Start, Tutorials, Guides, Architecture,
Reference and Project. Nineteen curated pages and ninety generated API pages cover
901 documented entries. Five executable examples cover a hierarchical Graph, a first posterior, typed
tasks, nonlinear sampling and cumulant-field scoring. The short nonlinear
example reports its unconverged diagnostics honestly rather than treating a
small demonstration budget as a scientific acceptance test.

The standalone architecture diagram passed Archify's 9/9 showcase validation
with zero errors and warnings. Four browser sizes were checked, followed by
manual review of light/dark screenshots. The final documentation regression
checked a 390-pixel mobile viewport and desktop navigation/search. Local receipts
are `runs/t002/archify-delivery.json`, `runs/t002/architecture/receipt.json` and
`runs/t002/browser-final.json`.

The static builder has a deliberately bounded contract: it resolves the current
package's supported export patterns and checks HTML links plus CSS `url()` and
`@import` resources. It is not an arbitrary Python interpreter or complete CSS
network auditor. Pages deploys generated output, excluding source fragments and
local review receipts.

## Distribution and validation evidence

The wheel contains all 100 package modules. The sdist contains the declared 127
portable test modules and their two original probe fixtures; repository-only
website/governance and application tests remain source-CI responsibilities.
Archive verification compares file bytes and expected membership, checks metadata
and canonical paths, then rebuilds and tests an installed wheel outside the source
checkout. Artifact hashes live in `runs/t002/release-manifest.json`.

### Findings closed during final validation

The full source run collected 7,085 cases: 7,070 passed, 13 skipped and two
failed. Both failures were test-fixture defects. The new plate regression used a
context-manager spelling that the public `PlateRef` API does not support. The SBC
sampled-route fixture assumed omitting `linear_in` would force NUTS, but automatic
affinity recognition now correctly selects the exact route. The former now passes
the plate reference to `observe`; the latter expresses the same Normal prior as
an affine transform of a standard Normal, which the current exact-prior contract
does not unwrap. It therefore exercises real Task dispatch to NUTS. Seeds,
budgets, rank definitions, thresholds and route assertions remain unchanged.
Independent reviewers checked both repairs; the source repair/governance run
passed all 93 cases, including the sampled SBC and the deliberately stretched
posterior negative control.

The full run's 13 skips are recorded individually in
`runs/t002/source-full/summary.json`: six require optional JAXNS, three are
explicit improper-prior premises, one is a session-local shape-sweep premise,
and three application tests require unavailable `limTOD.tris`. None is counted
as a pass. After that full run, package Python changes consist only of two
English docstring translations; executable ASTs are identical, as checked in
`runs/t002/final-source-delta.json`. This is full-run evidence plus an explicit
repair regression, not a claim that the initial full invocation was green.

The first clean Python 3.12 installation resolved NumPyro 0.22 and exposed
23 failures before deliberate interruption (exit 2, not a completed suite):
19 changed argument-validation cases, one changed Uniform support behavior,
one incompatible Gibbs dynamic-support path, and two changed moving-sigma route
classifications. Independent review confirmed that the Gibbs change affects the
production adapter. The candidate now declares `numpyro>=0.15,<0.22`; this
baseline is verified with 0.21, while migration to 0.22 is future work. The two
classification cases passed in the fresh 0.21 installed-wheel suite. This is a
verified compatibility outcome, without attributing every 0.22 failure to one
mechanism.

### Measured checks

| Check | Result | Local evidence |
|---|---|---|
| Full source suite | 7,070 passed; 13 skipped; two fixture failures subsequently repaired | `runs/t002/source-full/` |
| Repair, negative-control and documentation/distribution governance regression | 93 passed, exit 0 | `runs/t002/source-repairs/` |
| rheplicant inference consumer | 1,115 passed, exit 0 | `runs/t002/consumer-inference/` |
| rheplicant x64 seam consumer | 31 passed, one expected failure, exit 0 | `runs/t002/consumer-seam/` |
| Original installed wheel, Python 3.11 | 466 passed, exit 0 | `runs/t002/candidate-py311/` |
| Rebuilt-sdist installed wheel, Python 3.12 portable fast suite | 3,922 passed, 46 skipped, exit 0 | `runs/t002/candidate-portable/` |
| Linux installed rebuilt wheel | 191 passed before the final plate-axis guard; final 45-case reweight regression passed, exit 0 | `runs/t002/linux-rebuilt-wheel/`, `runs/t002/linux-final/` |
| Documentation build and API/link/anchor coverage | 109 generated pages, 901 entries; passed | `tools/build_docs.py --check` |
| English and crawler directives | All 110 served HTML pages checked | `runs/t002/unpublished-docs-check.json` |
| Five executable tutorials | Original four exited 0; new Graph example passed in source and both final installed wheels | `runs/t002/snippet-tool-output.json`, `runs/t002/hierarchical-example/` |
| Final document/governance regression | 46 passed, exit 0 | `runs/t002/final-docs-junit.xml` |
| Project Ruff, including tutorial sources | Passed with the project binary and `--no-cache`, exit 0 | `runs/t002/final-ruff.log` |
| Original wheel vs. sdist-rebuilt wheel | Byte-identical | `runs/t002/release-manifest.json` |

The consumer checkout is a dirty editable workspace at recorded HEAD
`8a07bffd91b072f0269255fd16620b6c6b92e421`; the source fingerprint is retained in
`runs/t002/consumer-source.json`. Its HEAD alone is not a claim of reproducibility.
The source and isolated-runtime versions are recorded in `runs/t002/stacks.json`.

## Final local baseline verdict

**Accepted as an unpublished 0.9.0 baseline.** The architecture, supported API
boundary, dependency cap, documentation and local artifacts are reviewable and
verified within the checks above. No unresolved test failure remains in the
covered final source/installed checks. The initial full invocation remains
recorded as failed; its two failed cases passed the repair regression. Optional
and premise-dependent skips retain their reasons in the JUnit artifacts.

The last Linux run emitted one JAX garbage-collection `KeyboardInterrupt` warning
during worker teardown; all 45 test cases passed and pytest exited 0. This is
recorded separately from numerical assertions. Remote CI and public publication
were not run. A future public release should run the configured remote Linux
workflows against its committed snapshot before any tag or upload.

The final package directory is `runs/t002/unpublished-0.9.0/`. Its wheel differs
from the fully tested candidate only in README description metadata and the
corresponding RECORD checksum: all package Python bytes and all metadata headers
(including dependencies) are identical. The final sdist rebuild is byte-identical
to that final wheel, and both final installed wheels passed the new hierarchical
Graph example. These relations are recorded in `runs/t002/release-manifest.json`.

The offline documentation snapshot is `runs/t002/documentation/`, with a portable
archive at `runs/t002/bayesmith-0.9.0-docs.zip`. All 110 served HTML pages are
English and request `noindex, nofollow, noarchive`. Neither crawler directives
nor local acceptance constitute publication or access control.

### Graph narrative confirmed with the user

“Flowchart as a hierarchical model” is now the entry point in the README, home
page and Graph chapter. A shared population parameter leads to unit-level latent
variables, deterministic signals and conditional observations. The tutorial
checks the same graph as a joint density against an independent Gaussian
calculation. It distinguishes the generative DAG from an execution lifecycle,
conditional from marginal independence, graph-level additive prior factors from
site priors, and a joint density from a normalized posterior. Desktop and mobile
browser review confirmed the dependency arrows and responsive reading order.


## Documentation revision after reader feedback

The subsequent documentation-only revision replaces the cream/green theme with
white/slate surfaces and blue accents, including the standalone architecture map.
Headings now use the same sans-serif family as the text. Short navigation labels,
plain group breadcrumbs and consolidated API-reference conventions reduce repeated
introductory prose. The homepage shows the shared population outside a repeated-unit
boundary; the software lifecycle remains in the architecture chapter.

Twelve formulas across five pages now use locally bundled KaTeX 0.18.7 with MathML
output. The strict Node check validates the same expressions and rendering options
used by the browser; the Python build records the expressions and checks local
fonts, links and generated output. No CDN is required.

The new hierarchical notebook follows model/equations, simulated observations,
posterior signal, parameter recovery and diagnostics. It reuses five original SVGs
from the historical inference notebook snapshot associated with `6c644b1`, with
source hashes and figure checksums in `site/assets/notebook/provenance.json`.
Captions distinguish posterior medians of group means, 99% credible intervals and
future-observation uncertainty. This is historical evidence, not a rerun of the
0.9.0 candidate or an SBC result. Plot tabs support keyboard navigation; all plots
remain visible without JavaScript, and full-size figures can be opened directly.

The refreshed site contains twenty curated pages and ninety generated API pages
(901 entries), plus the standalone architecture HTML. Independent code, Python,
JavaScript, scientific and static accessibility reviews closed their findings.
The browser reviewer could not attach because of a shared-profile conflict; root
performed the actual Chrome verification. Twenty curated pages and two API pages
were checked at 360 pixels with no document overflow or math errors. Desktop/dark
plots, both viewers' click/arrow/Home/End controls, mobile menu/Escape, nested API
search, local assets and no-JavaScript fallback were checked separately.

The documentation/governance regression passed 28 tests (exit 0). Strict rebuild,
source-output comparison, formula validation and project Ruff passed. Receipts are
under `runs/t002/docs-refresh/`. The offline documentation archive and release
manifest were refreshed; the wheel and sdist hashes remain unchanged. No publication
or package-runtime change occurred in this revision.


## Seminar-inspired visual explanation

A further reader request references seminar_talk_sep2026 slides 35–38. The new
`reweighting.html` tutorial follows their sequence: population question, fixed
sample positions with changing weights, marginal hyperparameter fitting, and
sample-bank reuse. Three original two-cell teaching figures retain their shared
positions, marked draw and color scale. Supporting mathematics is expandable;
wide comparisons retain readable labels through local scrolling on narrow screens.
The tutorial links to the experimental API guide instead of repeating its contract.

Scientific review verified the full proposal-density ratio, fixed evidence scale,
variance convention and peak-normalized plotted score. The broadened Gaussian
proposal is explicitly distinguished from a reference posterior. Original SVGs,
validation JSON and checksums are bundled locally. Neither the historical teaching
results nor the seminar draw-only timing are presented as current package benchmarks.
Code/accessibility review approved the implementation. Root inspected all four
source slides and the new page in desktop/light and mobile/light+dark Chrome.

The site now contains 21 curated pages and 90 API pages (901 entries), plus the
standalone diagram. Strict build/check and all 17 formulas across six pages pass.
The 18 documentation regression tests passed (exit 0). The process gate reported a
concurrent rheplicant suite; the documented advisory exception was used for this
small documentation-only run with xdist disabled. No numerical suite was rerun.
Evidence is in `runs/t002/seminar-docs/`. Offline docs and their manifest were
refreshed; package artifacts remain byte-identical and unpublished.
