# Inference composition and diagnostics implementation plan

> **文档状态：`record`** · Implemented bounded Uniform demos, Jeffreys diagnostics and the composed-process notebook; dated execution and validation are recorded below, not a normative specification.

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent diagnostic task; the controller integrates the demo and its presentation. Review all changes before completing.

**Goal:** Explain Jeffreys applicability accurately, support an executable hierarchical process composition, and place proposal methods last.

**Architecture:** Retain current graph/compiler/task APIs. Extend the preflight information adapter only for supported Bernoulli observations; distinguish conditional information from marginal hierarchical information. Generate the new demo using declared operators and save actual compiler, policy and recovery records.

**Tech Stack:** JAX, NumPyro, NumPy, Python, existing offline bilingual HTML/Matplotlib renderer.

**Spec:** Current user request and `docs/superpowers/specs/2026-09-08-block-methodology-design.md`.

## Constraints

- Preserve existing user edits and original live notebook address. Work in the explicitly shared checkout; do not commit, restore, merge or publish.
- No unified proposal/MH implementation in this change.
- No seed retries or widened recovery intervals to obtain passing results.
- Unknown checks must retain their mathematical scope and reason.
- Fix f=1; infer sigma_w, since only their product is identifiable.
- Keep Gaussian process conditional laws even if a top-level prior becomes flat.

## Task 1: Jeffreys diagnostic correctness

Files: `src/bayesmith/dispatch/preflight.py`, a focused `tests/dispatch/test_preflight_information.py`.

- [x] Reproduce incorrect unresolved status for a GCR block with an outside mean parameter, and unsupported Bernoulli Fisher.
- [x] Implement Bernoulli expected Fisher using differentiated logits and analytic weights: `J.T @ ((p*(1-p))[:,None] * J)` with correct broadcasting, family/mask checks and coordinate flattening.
- [x] Judge GCR flatness with respect to the current block, not the unrelated across-sweep covariance cache flag.
- [x] Keep hierarchical rank findings scoped to the direct observed likelihood; report latent-density factors and marginalization limitations explicitly.
- [x] Compare Fisher to independent analytic matrices and distinguish dependent covariance and unsupported families.

## Task 2: Composed stochastic process demo

Files: `examples/inference/composed_process.py`, runner/common and `tests/test_inference_examples.py`.

- [x] Build a small Fourier Gaussian process: `s_k ~ Normal(0, sqrt(P_k(h)))`, one or two spectral hyperparameters.
- [x] Apply known linear response; add nonlinear shape and linear background; multiply by `exp(U @ gain)`; observe `Normal(mu, abs(mu)*sigma_w)`.
- [x] Draw one true process instance from its conditional law; pass only data/model to inference.
- [x] Save all parameter draws and actual diagnostics. Test conditional simulation, factor topology, noise likelihood and posterior recovery without assuming a compiler route.
- [x] Check hyperparameter uncertainty honestly: one process instance supplies finite spectral information.

## Task 3: Bilingual presentation and prior audit

Files: guide/content/panels, plots, examples README, design spec.

- [x] Order methods as fixed GLS, log-GCR, NUTS, collapse, iterative GLS, corrected log-linear, Gauss–Newton, correction. Update all displayed numbers together.
- [x] Describe autodiff versus Fisher expectation/global-flatness proof, hierarchical likelihood scope and per-example flat-prior conditions.
- [x] Add the sixth example to every stage; include component plots and posterior power/noise uncertainty.
- [x] Respect the user's answer on flat-prior changes; otherwise retain current priors and explain the choice.

## Task 4: Validation and delivery

- [x] Run gated focused tests and applicable core fast checks, exact `.venv/bin/ruff check --no-cache`, JavaScript syntax checks and code review.
- [x] Run all affected demos with fixed registered seed/budget; do not label a failed or unexecuted result verified.
- [x] Render, inspect browser controls/plots, preserve previous live output, and update port 8766.

## Execution record

- User selected bounded Uniform for all root parameters; conditional Gaussian laws remain unchanged. Bounds are explicit model assumptions.
- Task 1: implemented and reviewed; 34 focused tests passed in 7.01s. Plate broadcasting, numerical probability clipping and budget counts are covered.
- New process: fixed seed-0 random initialization failed mixing. An explicit data-only L-BFGS initialization restored sampler diagnostics without changing automatic partition or truth.
- Recovery remains FAIL for the process: 19/20 intervals cover, noise sigma_w misses; no seed, truth or acceptance threshold changed. The first five cases pass under Uniform priors.
- Broad reviewer corrections: fixed stale joint-NUTS prose and separated mean/variance ESS for the linear oracle.
- Fast suite: 3,843 passed, 6 failed, 10 skipped (runs/inference-uniform-fast). One obsolete Bernoulli-unsupported assertion was updated to Poisson; the other five failures came from this plan's missing status declaration. After those fixes, all 41 focused diagnostic/document checks passed (runs/inference-uniform-recheck). The full suite was not repeated after these test/document-only fixes.
- All six CLI integration tests passed in the fast suite. Their fixed seed-0, 2,000-draw/1,000-warmup outputs were reused, with every example/library source hash checked against the current source; runs/inference-demo-uniform/validation.json records their origin. A failing recovery report retains CLI exit 1 even when the test correctly verifies that result.
- Code and final artifacts received independent review approval. Ruff, both JavaScript syntax checks and diff whitespace checks passed. All 256 local page asset/link targets exist; renderer and input hashes match.
- Browser checks verified bilingual navigation, proposal method order, the actual 01/03/05/08 method mapping, process plots, prior bounds, starting values and diagnostics. Fixed a remaining Chinese navigation label to the count-neutral “模型例子”.
- Delivered at http://127.0.0.1:8766/ from runs/inference-demo-verified; previous notebook preserved in runs/inference-demo-before-uniform. Initial-value recommendations are in the sampling chapter.
