"""Probe 7 -- what a correlated noise actually buys, in posterior units.

If the answer is "nothing measurable" the feature is not worth its guards. So:
generate data under a correlated N, then compare the exact linear-Gaussian
posterior computed with the TRUE N against the one computed with only
sqrt(diag N) -- which is all today's vocabulary can express.

Dense oracle throughout: numpy builds N, numpy inverts it. Nothing here shares
code with `bayesmith.exact`, so it is a statement about the statistics, not
about this package.

Run:
    cd <worktree> && PYTHONPATH=$PWD/src \
        /Users/zzhang/projects/bayesmith/.venv/bin/python probes/probe_7_what_the_feature_buys.py
"""

import numpy as np

rng = np.random.default_rng(11)
N = 256


def circulant(row):
    return np.array([np.roll(row, i) for i in range(len(row))])


def kernel(corr, variance=1.0, floor=1e-3):
    lag = np.minimum(np.arange(N), N - np.arange(N))
    row = variance * np.exp(-lag / corr) + floor
    assert np.real(np.fft.fft(row)).min() > 0
    return row


# A two-parameter design: an offset and a slope. The offset is the direction a
# smooth (1/f-like) noise contaminates most, which is the whole physical point.
t = np.linspace(-1.0, 1.0, N)
design = np.stack([np.ones(N), t], axis=1)
prior_std = np.array([10.0, 10.0])
truth = np.array([1.0, 2.0])


def posterior(data, noise_cov):
    inv = np.linalg.inv(noise_cov)
    precision = design.T @ inv @ design + np.diag(1.0 / prior_std**2)
    cov = np.linalg.inv(precision)
    return cov @ (design.T @ inv @ data), cov


print(f"n = {N}, two parameters (offset, slope), {len(prior_std)} priors of sd 10")
print(f"{'corr len':>9} {'bias/sd_true (offset)':>22} {'bias/sd_true (slope)':>21} "
      f"{'sd_diag/sd_true off':>20} {'sd_diag/sd_true slope':>22}")

for corr in (1.0, 4.0, 16.0, 64.0):
    row = kernel(corr)
    cov_true = circulant(row)
    cov_diag = np.diag(np.diag(cov_true))
    chol = np.linalg.cholesky(cov_true)

    trials = 200
    biases, ratios = [], []
    for _ in range(trials):
        data = design @ truth + chol @ rng.normal(size=N)
        m_true, c_true = posterior(data, cov_true)
        m_diag, c_diag = posterior(data, cov_diag)
        sd_true = np.sqrt(np.diag(c_true))
        biases.append((m_diag - m_true) / sd_true)
        ratios.append(np.sqrt(np.diag(c_diag)) / sd_true)
    biases = np.array(biases)
    ratios = np.array(ratios)
    print(
        f"{corr:>9.1f} {np.abs(biases[:, 0]).mean():>22.3f} "
        f"{np.abs(biases[:, 1]).mean():>21.3f} "
        f"{ratios[:, 0].mean():>20.4f} {ratios[:, 1].mean():>22.4f}"
    )

print()
print("  bias/sd_true : how far the diagonal-only posterior MEAN sits from the")
print("                 correlation-aware one, in units of the TRUE posterior sd,")
print("                 averaged over 200 realisations.")
print("  sd_diag/sd_true : ratio of the reported error bars. Below 1 means the")
print("                 diagonal answer is OVER-CONFIDENT.")

print()
print("  and the coverage that implies for the offset, at corr = 64:")
row = kernel(64.0)
cov_true, chol = circulant(row), np.linalg.cholesky(circulant(row))
cov_diag = np.diag(np.diag(cov_true))
inside_true = inside_diag = 0
trials = 2000
for _ in range(trials):
    data = design @ truth + chol @ rng.normal(size=N)
    m_true, c_true = posterior(data, cov_true)
    m_diag, c_diag = posterior(data, cov_diag)
    inside_true += abs(m_true[0] - truth[0]) <= 1.96 * np.sqrt(c_true[0, 0])
    inside_diag += abs(m_diag[0] - truth[0]) <= 1.96 * np.sqrt(c_diag[0, 0])
print(f"    nominal 95% interval, correlation-aware : {100*inside_true/trials:.1f}% coverage")
print(f"    nominal 95% interval, diagonal only     : {100*inside_diag/trials:.1f}% coverage")
