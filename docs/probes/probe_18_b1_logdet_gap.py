"""B1, sized: how far apart are the two targets, in rheplicant's own terms?

The defect the migration ledger calls B1 is that rheplicant's
``engines.conditional_potential`` is ``0.5*chi2 - log_prior``, and ``chi2``
evaluates sigma at the current prediction **without** adding ``sum(log
sigma)``, while the numpyro bridge's ``dist.Normal(prediction, sigma)`` carries
``-log sigma`` inside ``log_prob``. So on one model with a
prediction-dependent sigma, the GRADIENT block and the ``nuts`` exit minimise
two different functions -- with no guard and no note between them.

The ledger states the size as ``(1+f**2)`` and pins it from the estimator side
in ``tests/crosscheck/test_noise_logdet.py``. This probe states it from the
side that matters for the fix: **which optimum each exit actually walks to**,
found by scipy on plain numpy closures so neither package's gradients are
involved.

Measured 2026-08-28, n = 20000, constant-mean model ``d = mu (1 + w)``:

    f      argmin no-logdet   with-logdet   ratio     (1+f^2)
    0.05   5.010654           4.998159      1.002500  1.0025
    0.2    5.193557           4.993752      1.040011  1.0400
    0.5    6.241192           4.990950      1.250502  1.2500

Two things to read off it. The ratio is ``(1+f**2)`` to four digits at every
f, so this is the ledger's quantity and not something else. And the
with-logdet optimum recovers the truth (5.0) while the no-logdet one does not
-- so the exit that is wrong is the GRADIENT block, and the amount it is wrong
by is 25 percent at f = 0.5, which is not a rounding matter.

What this does NOT decide is the remedy. Adding the term to
``conditional_potential`` fixes the gradient block; the conjugate blocks drop
the same term by freezing sigma, and for those the designed remedy is
``bayesmith.exact.correct``'s importance weight, which needs a ``Graph`` and
changes what ``plan.sample`` returns. Both belong to the ``plan``/``engines``
row, and the ledger says B1 lands before that row's cross-check, or that
comparison fixes the GLS-type target as the reference.

Exit code is 0 whenever the probe finished, never a verdict.

Run:  /Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_18_b1_logdet_gap.py
"""
import jax, jax.numpy as jnp, numpy as np
from scipy.optimize import minimize_scalar
from rheplicant.inference.noise import RadiometerNoise

# constant-mean model, the same shape the crosscheck uses: d = mu*(1+w)
N = 20000
for f in (0.05, 0.2, 0.5):
    tau = 1.0 / f**2
    noise = RadiometerNoise(1.0, tau)
    mu_true = 5.0
    key = jax.random.key(11)
    d = mu_true * (1.0 + f * jax.random.normal(key, (N,), dtype=jnp.float64
                                               if jax.config.read("jax_enable_x64") else jnp.float32))
    d = np.asarray(d, dtype=np.float64)

    def without_logdet(mu):           # what conditional_potential minimises
        s = f * abs(mu)
        return 0.5 * np.sum((d - mu) ** 2 / s**2)

    def with_logdet(mu):              # what dist.Normal's log_prob minimises
        s = f * abs(mu)
        return 0.5 * np.sum((d - mu) ** 2 / s**2) + N * np.log(s)

    a = minimize_scalar(without_logdet, bracket=(1.0, 5.0, 20.0)).x
    b = minimize_scalar(with_logdet, bracket=(1.0, 5.0, 20.0)).x
    closed = np.sum(d**2) / np.sum(d)          # the dropped-logdet closed form
    print(f"  f={f:<5} argmin no-logdet={a:.6f} (closed {closed:.6f})  "
          f"with-logdet={b:.6f}  ratio={a/b:.6f}  (1+f^2)={1+f*f:.4f}")
