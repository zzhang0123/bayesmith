# Automatic affinity discovery implementation plan

> **文档状态：`plan-active`** · Implements the user-approved automatic discovery design in the current session; subordinate to the normative top-level design.

> **For agentic workers:** Execute the approved tasks below in order, with regression checks and an independent code review.

**Goal:** Discover conditional affine blocks without `linear_in`, with scoped structural evidence and conservative execution.

**Architecture:** Reuse the primal Jaxpr interpreter in `diagnose/structure.py`; trace all latent inputs while measuring dependence on a candidate group. Share discovery between the single-block and factor compilers. Preserve numerical probes as independent evidence and Gaussian eligibility as a separate decision.

**Tech Stack:** Python, JAX, NumPyro, pytest.

**Spec:** The implementation prompt approved in this session, and `docs/superpowers/specs/2026-08-30-bayesmith-top-level-design.md` §11.2.

## Constraints

- Preserve the shared checkout's existing edits and untracked implementation.
- Certify real-arithmetic structure on the valid domain, for the traced shapes and static configuration; do not freeze complementary latent values.
- Unknown operations and custom derivative semantics do not acquire a proof through numerical probes.
- Structure and Gaussian solver eligibility are separate. Preserve all joint-density factors.
- Gate pytest, save log/JUnit/exit together, use project Ruff with `--no-cache`.

## Tasks

- [x] Add `tests/dispatch/test_automatic_affinity.py`: missing declarations, additive and multiplicative blocks, non-Gaussian priors, zero outside coefficient, branch dependence, unknown/custom primitives, covariance dependence, public compile/report behavior, and analytic posterior checks. Observe missing-feature failures.
- [x] Extend `diagnose/structure.py` with `conditional_affinity_certificate(graph, names, values)`, reusing `_walk` with degree one for block inputs and degree zero for symbolic complementary inputs. Keep existing fixed-complement flatness diagnostics unchanged. Return scope and eligibility evidence separately.
- [x] Add shared discovery in `dispatch/affinity.py`; integrate it into `classify.py` and `factor.py`. Retain numeric checks, certify each final group, and retain reasons for conservative fallbacks. Carry evidence through `Block`, plan rendering and task analysis.
- [x] Connect supported multiple exact conditional blocks to the default execution path using existing Gibbs machinery; keep unsupported execution policies explicit. Verify no joint-mean/evidence path mistakes multiple conditional blocks for one joint Gaussian.
- [x] Update examples and documentation, run focused tests and the fast layer, obtain independent review, and resolve actionable findings.

## Acceptance examples

```python
# No declarations are needed:
mu = det("mu", lambda a, b: A @ a + B @ b + c, a, b)
# a*b must have separate conditional blocks; a+b*a**2 must not be
# certified affine in a merely because b's initial value is zero.
```

Focused command (after the gate):

```sh
.venv/bin/python -m pytest tests/dispatch/test_automatic_affinity.py tests/dispatch/test_preflight_structure.py
```

Full verification uses the repository fast-layer command and a per-run artifact directory. No automatic commits of unrelated shared changes.
