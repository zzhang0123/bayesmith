# Structural Jeffreys diagnostics and unified proposals

> **文档状态：`record`** · Completed implementation and validation record for the approved Jeffreys/proposal design; measured on 2026-09-09.

> **For agentic workers:** Use superpowers:subagent-driven-development for independent implementation and review tasks. The controller integrates public tasks, artifacts and the notebook.

**Goal:** Diagnose Jeffreys flatness independently of the sampler, and provide composable, target-correct proposal/MH updates.

**Architecture:** A conservative compile-time Jaxpr structure certificate supplements numerical Fisher diagnostics. A common JAX proposal/correction layer separates proposal construction from target evaluation; explicit task configuration composes proposal blocks and an optional NUTS remainder. Existing matrix-free GCR retains its cost and behavior.

**Tech Stack:** JAX, Equinox, NumPyro, existing task artifacts and bilingual notebook.

**Spec:** User requests of 2026-09-08 and docs/superpowers/specs/2026-09-08-block-methodology-design.md. The user explicitly authorized implementing the previously planned unified proposal/MH layer.

## Constraints and rulings

- Work in the user's existing shared checkout. Preserve all other edits; no commits, restores, mutation runs, releases or dependency upgrades.
- No host conversion, SciPy optimizer, Python value-dependent branch or dense conversion may be added to existing matrix-free sampled updates.
- Numerical probes alone cannot prove global flatness. Certificates inspect primal operations and canonical Gaussian density semantics, not user annotations or AD zeros alone.
- Flatness is in declared model coordinates, conditional on the stated complement. User priors remain unchanged.
- Every proposal must pair its actual draw law with the matching evaluable density. State-dependent proposals retain both forward/reverse normalizers and coordinate Jacobians.
- Reject invalid proposals without clipping, retain current state bitwise, count rejection as a step, and do not retry until accepted.
- Keep current routes as defaults. The new composable route is explicit and records its methods, order, initial state, stopping and correction diagnostics.
- In user-facing text the existing moving-noise route is named **GCR + iterative GLS + MH**; compatibility identifiers can remain unchanged.
- CPU JIT/vmap/scan and transfer guards are required checks. Report actual GPU device availability instead of claiming a GPU run on CPU-only hardware.
- Run gated pytest with per-run log/JUnit/exit artifacts; do not add a second -q. No seed retries or relaxed statistical criteria to force demo PASS.

## Task 1: Conservative structural flatness

Own: new src/bayesmith/diagnose/structure.py, src/bayesmith/dispatch/preflight.py, new tests/dispatch/test_preflight_structure.py.

- [x] Add a pure compile-time `gaussian_flatness_certificate(graph, names, values)` helper returning a serializable evidence dictionary. It identifies canonical observed Gaussian mean dependence and covariance independence using a conservative Jaxpr walk.
- [x] Propagate block-constant and block-linear dependence through supported algebra, shape/index operations and nested JIT calls. Unsupported primitives/control flow/custom derivative semantics return unsupported evidence, never a positive proof.
- [x] Use a full-rank certificate to classify a fixed-noise linear likelihood flat regardless of Uniform/Normal prior or NUTS/GCR method. Retain measured non-flatness, rank and numerical uncertainty with explicit reasons; remove the sampler-name shortcut.
- [x] Tests: fixed-X Normal likelihood with bounded Uniform is flat; different priors/methods agree; translated and multidimensional models; parameter-dependent covariance and dishonest `linear_in`/custom gradients do not pass; outer-parameter dependence is conditional; unsupported programs fall back; rank deficiency remains visible.
- [x] Run focused tests and independent code review before integration is declared complete.

## Task 2: Unified proposal and MH kernels

Own: new proposal modules under src/bayesmith, exact/gibbs.py integration if appropriate, focused proposal tests. Interface details are finalized from the existing executor seams before implementation.

- [x] Define one proposal draw/log-density contract with normalized state-dependent Gaussian proposals. Preserve the existing fixed-law matrix-free GCR adapter rather than replacing its determinant/cost contract.
- [x] Implement one MH acceptance kernel evaluating the original full target and both proposal directions, preserving support and handling invalid proposals.
- [x] Provide iterative-GLS, bias-corrected log-linear and Gauss–Newton builders with traceable linear algebra, fixed shape/budgets and explicit applicability limits.
- [x] Implement composition across parameter blocks; support an optional NUTS remainder and whole-proposal sweeps without fabricating a dummy latent.
- [x] Independent numerical tests: asymmetric forward/reverse ratio, one-step invariance plus nonzero movement, Gaussian moments, constrained target, varying covariance/normalizer, rejection semantics, sequential conditioning, reproducible RNG and JIT/vmap/scan.

## Task 3: Task integration and notebook

Own: task/artifact integration, bilingual examples/docs, renderer. Avoid overlapping Task 1/2 files without coordination.

- [x] Expose explicit serializable proposal selections through PosteriorTask; validate block membership, order, support and incompatible options before running.
- [x] Reuse initialization/checkpoint policies and retain actual proposal/acceptance records in artifacts. Existing defaults and fingerprints remain compatible where settings are absent.
- [x] Use the full method label GCR + iterative GLS + MH and explain the roles of all three components; pure GCR remains a distinct conditional draw.
- [x] Add/run compact numerical demos for the unified route; display actual route choices and diagnostics without replacing existing automatic runs with hand-selected claims.
- [x] Refresh original port-8766 notebook from verified numerical results, with prior output retained.

## Task 4: Validation and delivery

- [x] Focused RED/GREEN tests, relevant integration tests and gated fast suite.
- [x] Ruff with project binary/no cache, JS syntax and document index checks.
- [x] Independent final code/artifact review; both-language HTML content, local assets, nonempty anchor IDs and generated JS syntax checked. Browser automation of the local file preview was blocked by its URL policy; no workaround was attempted.
- [x] Record remaining supported boundaries, exact test results and device coverage below.

## Execution record

- Initial source inspection confirms the current flat shortcut is `block.method == "gcr"`, and legacy MH freezes its Gaussian proposal using iterative GLS as a function of the complement only.
- Compatibility ruling: serialize explicit `ProposalBlockPolicy` records in existing `PosteriorTask.backend_options`, with a typed `.proposals` property. Adding a dataclass field would invalidate strict-codec historical payloads. The default task fingerprint was measured before/after and is unchanged.
- Scope ruling: the new composable builders are small, dense Gaussian laws with explicit limits. General automatic candidate selection and a normalized generic matrix-free proposal contract remain future work; existing GCR routes retain their existing specialized correction and matrix-free cost.
- Independent review found and prompted regression tests for custom derivative trust, Gaussian/Bernoulli broadcast budgets, graph/NumPyro plate target parity, dense-budget ordering, method applicability, dynamic-support mixed updates, all-chain initialization validity, unsupported remainders and x64 counter dtypes.
- Demo protocol: preserve all six automatic runs. Three separate explicit comparisons use the same fixed seed, simulated data, priors and recovery thresholds. The first zero/one-start pilot is retained in `runs/inference-jeffreys-mh-verified/proposals`: linear GLS passed, while nonlinear GN and multiplicative log-linear/GLS barely moved and failed the unchanged checks. Final nonlinear comparisons explicitly use the existing bounded data-only L-BFGS initializer with zero perturbation, independent of truth. `--initialization simple` retains the reproducible original setup; this is a declared initialization improvement, not a seed or tolerance retry.

- Final numerical runs: all three explicit comparisons pass at seed 0, 2,000 draws × two chains and 1,000 warmup steps. Production MH acceptance: GLS 4,000/4,000; GN 3,424/4,000; multiplicative gain and signal 4,000/4,000 each. Data and truth exactly match the automatic runs.
- All six automatic posterior arrays are bitwise identical to the prior notebook. Five recovery runs pass; composed-process chain diagnostics pass but the pre-existing sigma_w recovery miss remains a FAIL.
- Validation hardware: CPU only. JIT/vmap/scan, transfer guards and FP64 sampler-state tests passed; no GPU execution was claimed.
- Original port-8766 output refreshed from runs/inference-jeffreys-mh-final; previous output retained in runs/inference-demo-before-jeffreys-mh.
- Final gated fast suite: **3,945 passed, 10 skipped, exit 0** in 381.08 seconds; log, JUnit and exit file are in `runs/jeffreys-proposals-fast-final/`. The earlier full fast invocation exposed the no-latent-parameter graph regression; its fix also passed 114 focused evidence/diagnostic/task tests in `runs/jeffreys-empty-graph-fix/` and independent review before this final run. The nightly `full` grids were not run.
- All nine numerical reports were regenerated after that final source fix. Their library/example source hashes match the final checkout, their posterior arrays match the preceding verified run bit for bit, and all six automatic arrays also match the original notebook. Both galleries were rerendered from these final reports; local assets, unique nonempty anchors and generated/source JavaScript passed validation. `runs/inference-jeffreys-mh-final/validation.json` records the checks.
- Final lint: `.venv/bin/ruff check --no-cache src/ tests/ examples/inference/` exits 0. The document index was synchronized after closing this plan.
