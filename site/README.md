# Documentation source and build

`content/` contains reviewed HTML fragments; `navigation.json` defines the task-oriented hierarchy. `snippets/` holds downloadable Python examples embedded by the builder. `diagrams/` holds separately delivered diagram HTML/JSON and is copied byte for byte into temporary builds so manual diagram links and anchors are validated. `assets/` contains offline CSS and JavaScript; `assets/search-index.js`, root HTML, `api/` and `manifest.json` are generated.

From the repository root:

```sh
python3 tools/build_docs.py
python3 tools/build_docs.py --check
node tools/check_docs_math.cjs
```

Two tokens embed measured output. `{{plan:plans/<record>.json#<name>}}` renders a recorded inference plan as a block/route table with the printed plan folded underneath, and `{{table:plans/<record>.json#<name>}}` renders a recorded result table. Both read `site/assets/plans/*.json`, written by `tools/record_site_plans.py`, which runs the snippets under `snippets/` and records the plans, the tables and the environment they were measured in. Re-record with `JAX_ENABLE_X64=1 .venv/bin/python tools/record_site_plans.py`; `--check` re-measures and compares structure only, because condition numbers and sampled columns depend on the platform's BLAS. `tests/test_site_plans.py` runs that check, and the pages state the platform beside every recorded number.

The stdlib-only builder reads Python ASTs without importing numerical libraries, resolves public exports and lazy facade aliases, generates signatures/docstrings, checks all internal links and anchors, and verifies every API entry has an anchor. `--check` also refuses stale output. A release build uses the actual source version in `pyproject.toml`; a candidate number is not substituted before the package has adopted it.

Use `--source-root /path/to/checkout` only when preparing documentation against a different checkout. The committed build must ultimately be regenerated against its own checkout. All assets and search operate offline, including under `file://`. No font, analytics or script CDN is used. The static checker covers HTML resource attributes and supported CSS `@import` / `url()` references; it is not a complete CSS parser or a network audit. API extraction covers the package's straight-line static declarations and supported facade forwarding, not arbitrary conditional Python execution. Pages builds into `_site/` and deploys that generated directory; source fragments and build metadata are not uploaded.

Tutorial snippets are syntax checked by the build, not automatically executed. Numerical execution and browser validation are separate checks and must be recorded separately. `five_tasks.py` needs `JAX_ENABLE_X64=1`. Research records and internal specifications remain in the repository, linked only as clearly labeled supporting material.

The candidate site is English-only. Every served HTML page, including the standalone architecture diagram, requests `noindex, nofollow, noarchive`. This is a crawler directive, not access control. Pages does not deploy on push; its manual workflow defaults to no publication and requires explicit opt-in. T-002 currently keeps the candidate local and unpublished.

The Graph chapter treats a flowchart as a hierarchical probability model. `hierarchical_model.py` checks its joint density against an independent Gaussian formula; it complements the four inference/artifact examples.

Mathematics uses `data-math` for display equations and `data-tex` for inline expressions. The builder records expressions in the manifest; the Node check validates them with the same strict KaTeX options as the browser. KaTeX and its WOFF2 fonts are vendored with their license.

The hierarchical notebook embeds a historical run with curated provenance under `assets/notebook/`; builds never read ignored `runs/` files. Plot tabs enhance ordinary figures, which remain visible without JavaScript. Snapshot figures are not evidence of rerunning the current candidate. The plan shown beside them is compiled from the same declaration today, so the recorded run and the current plan are labelled separately.
