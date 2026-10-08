# Demo C: the physical moment foreground through the RHINO twin

The third foreground strategy of the rheplicant example
`examples/global21cm` (the RHINO global 21-cm drift scan with Demos A and B,
an oracle and a figure of merit). Demo C keeps Demo A's treatment, one free
coefficient set per LST on a fixed spectral basis with a flat prior that is
integrated out exactly, and replaces the ad hoc log-polynomial basis by
channel-integrated responses of the synchrotron moment expansion (SyncMoments,
Zhang & Chluba). Three things come out:

1. `plan_<scenario>*.txt|svg|png`: what `bayesmith.compile` says about the
   model written as a `sample` / `det` / `observe` graph (section 6 of the
   report explains the three printouts);
2. `signal_recovery_moments*.svg`, `trough_bias_vs_order.svg`: the 21-cm
   posterior at each Taylor order, scored with the twin's figure of merit;
3. `identifiable_combinations.svg`: which coefficient combinations one LST's
   channels constrain, after the exact scaling identity and at the noise.

`REPORT.md` under `runs/rhino-moment-foreground/<full|quick>/` carries the
tables and the honesty record. Nothing here is committed to the twin; the
twin's saved products are read and hashed.

## Where things run

Two interpreters, because SyncMoments 0.4.0 pins jax 0.10.x and the twin's
venv holds jax 0.11.0:

| step | interpreter | writes |
|---|---|---|
| `basis.py` | a scratch venv from `make_basis_env.sh` | `basis_<scenario>.npz|json` |
| `recover.py`, `identify.py`, `model.py`, `figures.py` | `<rheplicant>/.venv/bin/python` with `PYTHONPATH=<rheplicant>/examples` | `full/` or `quick/`, and `plan_*` |
| `report.py` | any Python 3.12 | `REPORT.md` |

The twin is found at `$RHEPLICANT_GLOBAL21CM` (default
`/Users/zzhang/projects/rheplicant/examples/global21cm`); its
`results/sim*/` arrays (kept in its git) and, when present,
`results/analysis/{fom.json,posteriors.npz}` are read.

```bash
sh make_basis_env.sh /tmp/smvenv
sh run.sh /tmp/smvenv/bin/python --quick     # about 3 minutes
sh run.sh /tmp/smvenv/bin/python             # the full path, about 10 minutes
```

`--quick` runs one SMC seed of 4000 particles per posterior; the full path
four seeds of 20000. The grid, the basis and the plans are the same on both.

## Files

| file | what |
|---|---|
| `common.py` | constants (scenarios, reference population, palette), paths, provenance helpers |
| `basis.py` | the SyncMoments columns per cell and order, the scaling-identity and scipy checks |
| `recover.py` | the order grid (Demo A's rule), the collapse + SMC sweeps, the twin's scores |
| `identify.py` | the noise-whitened SVD of the per-LST response |
| `model.py` | the bayesmith graph, `compile`, `declared_partition`, the plan renderings, a sampler smoke |
| `figures.py` | the figures, from saved products only |
| `report.py` | `REPORT.md` |
| `run.sh`, `make_basis_env.sh` | the pipeline and the scratch venv |
