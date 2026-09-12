# TRIS / Haslam real-data inference implementation plan

> **文档状态：`record`** — Real TRIS/Haslam inference implementation.

**Goal:** Fit a Haslam-based spectral sky to maps reconstructed from the public
TRIS data using limTOD, and present the actual run in the inference notebook.

**Architecture:** Offline preparation creates maps, their full uncertainty and
the mapmaking response. Bayesmith samples a low-dimensional specialization of
BellaNasirudin/bayesian_skymap's `amps * (nu / ref_freq)**beta` model. A dedicated
real-data presentation shares the notebook's navigation and bilingual controls.

**Tech stack:** limTOD, healpy/astropy for preparation; NumPy/SciPy and
JAX/bayesmith for inference; matplotlib and the existing offline HTML renderer.

## Scientific contract

- Read the real local LAMBDA archive; preserve RA labels, effective frequencies
  and file hashes. No simulated observations or known-truth recovery claims.
- Fit the 600.5 and 817.8 MHz rings. Show the six 2427.8 MHz points separately:
  their third column is a common zero-level error, with no statistical errors
  supplied, so they cannot define the requested Gaussian likelihood.
- Use the measured beam cuts and limTOD pointing, HEALPix RING/equatorial maps,
  and the frequency-dependent Rayleigh–Jeans CMB monopole.
- A Wiener map is `m = b + W d`, where `b = m0 - W A m0`. Its sampling noise is
  `Cnoise = W N W.T`, distinct from its prior-conditional posterior covariance.
  Fit `b + W (A sky(theta) + offset)` to `m` with `Cnoise`; whiten its supported
  modes using an SVD of `W sqrt(N)`. Record any rank truncation. Never use the
  posterior diagonal as independent observational noise or fit the unsmoothed
  sky directly to the regularized map. Check likelihood differences against
  the original rings and invariance to the mapmaking prior.
- Use three explicit Galactic latitude regions for amplitudes and spectral
  indices; fix Haslam calibration as the reference. A second free gain would
  be degenerate with amplitude. State this reduction from the upstream model.
- Carry one shared offset per frequency; retain the asymmetric 820-MHz scales.
  Keep reported chain convergence separate from model adequacy.
- No edits to the existing core inference work or the sibling limTOD checkout.

## Task 1 — Offline data and map contract

Files: `examples/inference/tris_prepare.py`, `tris_maps.py`,
`tests/test_tris_case.py`, optional requirements and source manifest.

- [x] Write independent Gaussian likelihood and map-prior invariance tests.
- [x] Implement `compress_map(map_k, prior_k, operator, sigma_k, covariance)`
  returning whitened map data/sky response/offset response and rank evidence.
- [x] Export limTOD maps, posterior and propagated noise uncertainties, measured
  beam operators, Haslam template and actual archive data to one offline NPZ.

## Task 2 — Parameter inference and findings

Files: `examples/inference/tris_sky.py`, `tris_case.py`.

- [x] Declare Haslam amplitude / spectral operators and proper parameter priors.
- [x] Run two chains via `PosteriorTask`; save task, analysis, posterior, DAG,
  source/input hashes, diagnostics, residual and posterior predictive summaries.
- [x] Independently validate the forward equation, graph likelihood, and
  likelihood equivalence. Report prior restrictions and lack of 2.5-GHz fit.

## Task 3 — Notebook case study

Files: `examples/inference/tris_presentation.py`, `presentation.py`,
`plot_results.py`, `gallery.js`, `README.md`, targeted navigation tests.

- [x] Add a separate Real observations / 真实观测 section, outside six demos.
- [x] Present data/coverage, map response and model, priors, actual sampling,
  and scientific findings with bilingual text and saved-source links.
- [x] Render maps with masks and units, frequency-faceted residuals with real
  error bars, posterior intervals/correlations/traces, and map uncertainty.
- [x] Execute numerical tests behind `tools/pytest_gate.py`, lint changed
  sources with `.venv/bin/ruff check --no-cache`, inspect the rendered notebook
  in the browser, and have a code reviewer examine the scientific seam.

## Recorded outcome

Implemented the three-region specialization, real-data preparation, map-space likelihood, posterior artifacts, and a separate bilingual notebook case. The six 2427.8-MHz points are contextual only because statistical errors are not supplied.

The two-chain run retained 2,000 draws per chain after 1,500 warmup steps; seed 0, float64. Declared R-hat <=1.01 / ESS >=400 and zero-divergence checks passed. Each frequency retained all 120 map-noise modes.

Measured chi-square per observation: 600.5 MHz: 2419.261, 817.8 MHz: 440.030. Both posterior predictive comparisons had 0/4000 exceedances. This is model mismatch, not physical spectral-index precision.

Scientific-seam and final code reviews found and corrected the asymmetric zero-level sign, saved-source provenance, and failed-diagnostic rendering. Targeted mathematical, graph, navigation and document tests accompany the change. Generated results are at `runs/inference-demo-verified/tris_haslam/`; the notebook is `runs/inference-demo-verified/index.html`.

Final validation: 20 targeted tests passed (JUnit: `runs/tris-validation/junit.xml`); `.venv/bin/ruff check --no-cache src/ tests/` and the changed example files were clean. Browser checks verified bilingual chapter navigation, keyboard arrows, both map frequencies, KaTeX, comparison-page return links, and absence of browser warnings/errors.

## Follow-up — posterior sky maps and shared setup presentation

User requested maps of a, beta and recalibrated Haslam, and the same setup detail
as the simulated demos. Implemented posterior mean, sample SD and 95% equal-tail
arrays on the prepared grid. At 408 MHz the sky is `a * (Haslam - CMB) + CMB`;
neither the spectral slope nor the TRIS instrumental correction changes this
reference-frequency sky. Full-sky maps are explicitly regional-model
extrapolations. Haslam input and recalibrated output share a temperature scale.

The real case now uses the common method/block, prior/preflight and sampling
panels. Reports save actual block inspection, diagnostic scope and task policy;
the notebook shows actual initial states and checkpoints. Scientific residuals
and assumptions accompany the posterior maps in the findings chapter. No
alternative-prior experiment is implied by a Jeffreys diagnostic.

For this saved run, the original map bundle was extended only with the reference
CMB scalar computed by its hash-matched limTOD source. The extension script and
original input hash are recorded under `runs/tris-skyfields-validation/` and the
new input manifest. Every pre-existing numerical input array was checked for
exact equality. Future preparation records the scalar directly.

Follow-up validation: 25 targeted tests passed with zero skips/failures/errors
(`runs/tris-skyfields-validation/junit.xml`, exit 0). Original and refreshed
posterior arrays are exactly equal. Ruff was clean for `src/`, `tests/` and the
changed Python presentation files. Browser checks confirmed the new maps load,
shared setup shows real block/initialization/checkpoint data, both languages
work, formulas have no KaTeX errors, and the page has no horizontal overflow or
console warnings/errors. Review found and fixed a zero-SD LogNorm failure;
an isolated failed-chain reproduction then rendered all 10 PNG/SVG figures.
