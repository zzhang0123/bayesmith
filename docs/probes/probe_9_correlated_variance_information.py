"""Probe 9 -- the correlated variance-information term, against a dense route.

`exact/fisher.py` applies the variance's own information as
`2 (dlog sigma/dx)^T (dlog sigma/dx)`, derived for a DIAGONAL N, and
`precision_parts` refuses a correlated node that also claims
`depends_on_prediction=True` until the correlated form is derived AND
measured.

`docs/derivations/variance_information_*.wls` does the algebra in Mathematica.
Its result:

    1/2 tr(N^-1 d_a N N^-1 d_b N) = 1/2 Sum_k d_a log lam_k d_b log lam_k

whenever N's EIGENBASIS does not move with the parameters -- true for both
rows the gate accepts (a diagonal Normal, basis I; a CirculantNormal, basis
DFT). The shipped diagonal formula is that at lam_i = sigma_i^2.

This is the independent route: a dense Fisher matrix built by finite
differences of the exact Gaussian log-likelihood, sharing no algebra with
either.

Run:
    cd <worktree> && PYTHONPATH=$PWD/src .venv/bin/python \
        docs/probes/probe_9_correlated_variance_information.py
"""

import numpy as np

RNG = np.random.default_rng(11)


def circulant(column):
    n = column.size
    return np.array([[column[(j - i) % n] for j in range(n)] for i in range(n)])


def fisher_dense(mean_of, cov_of, theta, step=1e-5):
    """F_ab by finite differences of the EXACT Gaussian Fisher formula.

    F_ab = (d_a mu)^T N^-1 (d_b mu) + 1/2 tr(N^-1 d_a N N^-1 d_b N),
    with every derivative central-differenced. No spectral identity, no
    log-sigma jacobian, no FFT.
    """
    theta = np.asarray(theta, dtype=float)
    k = theta.size
    inverse = np.linalg.inv(cov_of(theta))

    def d(fn, a):
        up, down = theta.copy(), theta.copy()
        up[a] += step
        down[a] -= step
        return (fn(up) - fn(down)) / (2.0 * step)

    dmu = [d(mean_of, a) for a in range(k)]
    dcov = [d(cov_of, a) for a in range(k)]
    out = np.zeros((k, k))
    for a in range(k):
        for b in range(k):
            out[a, b] = dmu[a] @ inverse @ dmu[b] + 0.5 * np.trace(
                inverse @ dcov[a] @ inverse @ dcov[b]
            )
    return out


def spectral_variance_term(cov_of, theta, eigen_of, step=1e-5):
    """1/2 Sum_k d_a log lam_k d_b log lam_k -- the derived form."""
    theta = np.asarray(theta, dtype=float)
    k = theta.size

    def dlog(a):
        up, down = theta.copy(), theta.copy()
        up[a] += step
        down[a] -= step
        return (np.log(eigen_of(cov_of(up))) - np.log(eigen_of(cov_of(down)))) / (
            2.0 * step
        )

    rows = [dlog(a) for a in range(k)]
    return 0.5 * np.array([[rows[a] @ rows[b] for b in range(k)] for a in range(k)])


N = 12
X = np.linspace(0.7, 3.1, N)
Y = np.linspace(-1.0, 2.0, N) ** 2
LAG = np.minimum(np.arange(N), N - np.arange(N))
F = 0.35

print("=" * 78)
print("(a) DIAGONAL, one parameter -- reproducing the (1 + 2 f^2) factor")
print("=" * 78)


def diag_mean(t):
    return t[0] * X


def diag_cov(t):
    return np.diag((F * t[0] * X) ** 2)


for theta in ([1.0], [2.5], [7.0]):
    dense = fisher_dense(diag_mean, diag_cov, theta)
    mean_only = float(X @ np.linalg.inv(diag_cov(theta)) @ X)
    print(
        f"  theta={theta[0]:<5} F/F_mean = {dense.item() / mean_only:.10f}"
        f"   (1 + 2 f^2) = {1 + 2 * F**2:.10f}"
    )

print()
print("=" * 78)
print("(b) N = D C D with a PER-SAMPLE D -- the 1 + 2 f^2 n/(1^T C^-1 1) factor")
print("=" * 78)
print("  This is the model the Mathematica derivation is written for:")
print("  multiplicative noise, D = diag(f mu_i), C a fixed correlation.")
print("  Note it is NOT stationary unless every mu_i is equal, so it is a")
print("  model `CirculantNormal` cannot express -- see (b2).")
KERNEL = 1.0 * 0.55**LAG + 0.30
CORRELATION = circulant(KERNEL / KERNEL[0])
ONE = np.ones(N)
QUADRATIC = float(ONE @ np.linalg.inv(CORRELATION) @ ONE)


def dcd_mean(t):
    return t[0] * X


def dcd_cov(t):
    scale = np.diag(F * t[0] * X)  # per-sample D
    return scale @ CORRELATION @ scale


for theta in ([1.0], [2.5], [7.0]):
    dense = fisher_dense(dcd_mean, dcd_cov, theta)
    mean_only = float(X @ np.linalg.inv(dcd_cov(theta)) @ X)
    predicted = 1 + 2 * F**2 * N / QUADRATIC
    found = dense.item() / mean_only
    print(
        f"  theta={theta[0]:<5} F/F_mean = {found:.10f}"
        f"   1 + 2 f^2 n/(1^T C^-1 1) = {predicted:.10f}"
        f"   rel {abs(found / predicted - 1):.2e}"
    )
print(f"  1^T C^-1 1 = {QUADRATIC:.10f}   n/lambda0 = "
      f"{N / float(np.sum(KERNEL / KERNEL[0])):.10f}   (equal, as derived)")

print()
print("=" * 78)
print("(b2) the same covariance made STATIONARY -- a common scale times C")
print("=" * 78)
print("  What a CirculantNormal CAN express. Its factor is NOT 1 + 2 f^2")
print("  lambda0: that came from D = diag(f mu_i), whose per-sample variation")
print("  is exactly what makes 1^T C^-1 1 appear. With a common scale the")
print("  mean term keeps x, so no universal factor exists.")


def scaled_mean(t):
    return t[0] * X


def scaled_cov(t):
    return (F * t[0]) ** 2 * CORRELATION


for theta in ([1.0], [2.5]):
    dense = fisher_dense(scaled_mean, scaled_cov, theta)
    mean_only = float(X @ np.linalg.inv(scaled_cov(theta)) @ X)
    predicted = 1 + 2 * F**2 * N / float(X @ np.linalg.inv(CORRELATION) @ X)
    found = dense.item() / mean_only
    print(
        f"  theta={theta[0]:<5} F/F_mean = {found:.10f}"
        f"   1 + 2 f^2 n/(x^T C^-1 x) = {predicted:.10f}"
        f"   rel {abs(found / predicted - 1):.2e}"
    )

print()
print("=" * 78)
print("(c) CIRCULANT whose kernel changes SHAPE with the parameters")
print("=" * 78)
print("  the case no scalar factor covers, and the one the spectral form is for")


def shape_cov(t):
    column = t[0] ** 2 * (0.55 ** (t[1] * LAG)) + 0.05 * t[0] ** 2
    return circulant(column)


def shape_mean(t):
    return t[0] * X + t[1] * Y


def circulant_eigenvalues(matrix):
    return np.real(np.fft.fft(matrix[0, :]))


for theta in ([1.0, 1.0], [2.0, 0.6], [0.8, 1.7]):
    dense = fisher_dense(shape_mean, shape_cov, theta)
    inverse = np.linalg.inv(shape_cov(theta))
    step = 1e-5

    def dmu(a, theta=theta):
        up, down = np.array(theta), np.array(theta)
        up[a] += step
        down[a] -= step
        return (shape_mean(up) - shape_mean(down)) / (2 * step)

    mean_part = np.array([[dmu(a) @ inverse @ dmu(b) for b in range(2)] for a in range(2)])
    derived = spectral_variance_term(shape_cov, theta, circulant_eigenvalues)
    residual = dense - mean_part - derived
    worst = np.max(np.abs(residual)) / max(np.max(np.abs(dense)), 1e-300)
    print(f"  theta={theta}  worst relative residual of (dense - mean - spectral) = {worst:.3e}")

print()
print("=" * 78)
print("(d) the SHIPPED diagonal formula on case (c) -- how wrong, and which way")
print("=" * 78)
print("  For a circulant, sqrt(diag N) is CONSTANT across i, so the shipped")
print("  2 (dlog sigma/dx)^T (dlog sigma/dx) sees the DC mode's motion n times")
print("  over and is blind to every other mode. The whole matrix is printed:")
print("  the [0,0] entry agrees by construction -- theta_0 is a pure scale, so")
print("  every eigenvalue moves together -- and it is the SHAPE parameter that")
print("  separates them.")
for theta in ([1.0, 1.0], [2.0, 0.6], [0.8, 1.7]):
    derived = spectral_variance_term(shape_cov, theta, circulant_eigenvalues)
    shipped = spectral_variance_term(shape_cov, theta, lambda m: np.full(N, m[0, 0]))
    print(f"  theta={theta}")
    for label, matrix in (("derived", derived), ("shipped", shipped)):
        print(f"    {label} = [[{matrix[0, 0]:12.6f}, {matrix[0, 1]:12.6f}],"
              f" [{matrix[1, 0]:12.6f}, {matrix[1, 1]:12.6f}]]")
    ratio = shipped[1, 1] / derived[1, 1]
    print(f"    shape-parameter entry [1,1]: shipped/derived = {ratio:.4f}"
          f"   ({'too LARGE' if ratio > 1 else 'too SMALL'}, so the error bar is"
          f" too {'narrow' if ratio > 1 else 'wide'})")
