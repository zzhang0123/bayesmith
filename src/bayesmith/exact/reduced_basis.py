"""Reduced-basis selection and orthonormalisation (G4).

A reduced basis is a dictionary of directions in the DATA space such that a
model's prediction is well approximated by a combination of them:
``mu(theta) ~ sum_k c_k(theta) s_k``. The scheme is Field, Galley, Hesthaven,
Kaye & Tiglio (2014, PRX 4, 031006); the special case ``s_k = d mu / d
theta_k`` at a fiducial point is MOPED (Heavens, Jimenez & Lahav 2000) and
score compression (Alsing & Wandelt 2018).

**What is here is the array-level linear algebra and nothing else.** Choosing
candidate directions from a bank, and turning candidates into a basis. What is
NOT here, deliberately:

* the **containers** -- a basis object carrying its raw and whitened rows, and
  a fidelity report. The migration ledger's D12 keeps those upstream, because
  their constructor-time refusals are part of a published exception identity;
* the **declaration layer** that produces a bank from a parameter space and a
  pipeline. That is a question about a model's own vocabulary, and this
  package has no opinion about pipelines.

**Selection and basis are different things, and that separation is
load-bearing.** :func:`select_svd` and :func:`select_greedy` choose
CANDIDATES; :func:`orthonormalise` turns candidates into a basis. Storing raw
candidates gives a Gram matrix that no float64 quadratic form survives --
``c^T G c`` then returns a finite and occasionally negative number rather than
raising. That is also why :func:`numerical_rank` cuts at ``sqrt(eps)`` and not
at ``eps``: the quadratic form squares the conditioning, so a set that is
merely invertible is not usable.

**The metric is the caller's, applied before anything here runs.** Every
routine takes rows already whitened -- multiplied by the likelihood's own
``N^-1/2`` -- so "orthonormal in the metric the likelihood uses" is just
"orthonormal", the Gram matrix is the identity and the projector is a matrix
product. With any other convention the projector is not self-adjoint in the
likelihood's inner product, the truncation residual stops being orthogonal to
the span, and the score at the truth acquires a term that does not vanish: a
BIAS, not a loss of sensitivity.

Ported from ``rheplicant.inference.reduced_basis``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from bayesmith.errors import StructureError

__all__ = [
    "numerical_rank",
    "orthonormal_transform",
    "orthonormalise",
    "select_greedy",
    "select_svd",
]


def _refuse_single_precision(doing: str) -> None:
    """Refuse before any work when the ambient precision is float32.

    The package rule -- ``src/`` never opens an x64 context; it refuses and
    names the remedy -- and judged by OUTCOME rather than by reading the
    config flag, so it holds whether the caller used the context manager, the
    process-global switch or neither.

    **The argument here is not the one
    :func:`~bayesmith.diagnose.local.refuse_ambient_float32` makes**, and the
    two are separate guards for that reason rather than one repeated. That
    one is about a rank verdict living decades below float32's roundoff. This
    one is about the GRAM MATRIX: everything in this module exists to produce
    rows a caller will build a projector from, ``c^T G c`` squares the
    conditioning, and the retention cut is ``sqrt(eps)`` of the arithmetic in
    hand. In float32 that cut is 3.4e-04 -- so a bank whose smallest useful
    direction sits anywhere below a ten-thousandth of its largest is silently
    discarded, and the science direction such a basis exists to keep is
    exactly the small one.

    Widening around the CALL does not help and is not what this asks for: a
    bank computed in float32 has already spent the digits, and
    ``jax.enable_x64`` does not widen an array traced outside it. The remedy
    is to build the bank inside the block.
    """
    dtype = jnp.result_type(float)
    if jnp.finfo(dtype).bits >= 64:
        return
    raise StructureError(
        f"{doing} was asked for with {jnp.dtype(dtype).name} as the ambient "
        "precision. This module produces rows a projector is built from, and "
        "`c^T G c` squares the conditioning: the retention cut is sqrt(eps) "
        "of the arithmetic in hand, which is 3.4e-04 in float32 against "
        "1.5e-08 in float64 -- so every direction below a ten-thousandth of "
        "the largest is dropped, and on a foreground-dominated bank that is "
        "precisely the direction the basis exists to keep. Run inside `with "
        "jax.enable_x64(True):`, building the BANK inside the block: an array "
        "traced outside it stays float32 and widening only this call recovers "
        "nothing."
    )


def _rows(candidates: Any, caller: str) -> np.ndarray:
    """``(k, n)`` as float64 numpy, or a refusal that names the shape.

    A single row passed as ``(n,)`` would otherwise orthonormalise into one
    direction of length one -- finite, plausible, and not what the caller
    meant.
    """
    array = np.asarray(candidates, dtype=np.float64)
    if array.ndim != 2:
        raise StructureError(
            f"{caller} takes a two-dimensional (k, n_data) array of rows and "
            f"this one has shape {array.shape}. A single direction passed as "
            "(n,) would come back as one row of length one -- finite, "
            "correctly typed, and not the question. Pass `row[None, :]`."
        )
    return array


def orthonormal_transform(
    candidates: jax.Array, rtol: float | None = None
) -> tuple[jax.Array, tuple[int, ...]]:
    """``(M, kept)`` with ``M @ candidates`` orthonormal in the rows' metric.

    Modified Gram-Schmidt with **one reorthogonalisation pass**, in numpy,
    because the TRANSFORM is what a caller needs and a QR only hands back the
    result. ``M`` is what lets the same combination be applied to the RAW
    rows: a basis has to be usable by an epoch whose flag pattern differs from
    the reference's, and the whitened copy is infinite at exactly the samples
    the reference could not see -- there is no inverse back to unwhitened data
    there, and that is what a zero weight MEANS rather than a limitation.

    The second pass is not defensive. On a near-dependent set one pass loses
    orthogonality and two recover it, measured in
    ``tests/exact/test_reduced_basis.py`` against a single-pass reference
    computed in that file.

    **Order is preserved**, so ``span(row_1..row_j)`` is nested. That is the
    property seeding depends on: a direction placed first survives whatever
    the later candidates do, which is how a science direction three orders
    below the foregrounds is retained by construction rather than by choosing
    the basis size large enough.

    Args:
        candidates: ``(k, n_data)`` rows, **already whitened**.
        rtol: drop a direction whose residual norm falls below
            ``rtol * max_row_norm``. Defaults to ``sqrt(eps)``, for the reason
            :func:`numerical_rank` gives.

    Returns:
        ``(M, kept)``. ``M`` is ``(r, k)`` with ``r <= k``, and ``kept`` names
        which candidate indices survived. ``r < k`` means the candidate set
        was rank-deficient -- a fact about the model, which the caller decides
        what to do about, so it is reported rather than raised.

    Raises:
        StructureError: if ``candidates`` is not two-dimensional.
    """
    _refuse_single_precision("orthonormal_transform")
    whitened = _rows(candidates, "orthonormal_transform")
    count = whitened.shape[0]
    if rtol is None:
        rtol = float(np.sqrt(np.finfo(np.float64).eps))
    scale = float(np.max(np.linalg.norm(whitened, axis=1))) if count else 0.0
    cut = rtol * scale
    vectors = np.zeros((0, whitened.shape[1]))
    transform = np.zeros((0, count))
    kept: list[int] = []
    for index in range(count):
        vector, row = whitened[index].copy(), np.eye(count)[index].copy()
        for _ in range(2):  # twice is enough; once is not -- see the docstring
            for position in range(len(kept)):
                overlap = float(vectors[position] @ vector)
                vector = vector - overlap * vectors[position]
                row = row - overlap * transform[position]
        norm = float(np.linalg.norm(vector))
        if norm <= cut:
            continue
        vectors = np.vstack([vectors, vector / norm])
        transform = np.vstack([transform, row / norm])
        kept.append(index)
    return jnp.asarray(transform), tuple(kept)


def orthonormalise(candidates: jax.Array, rtol: float | None = None) -> jax.Array:
    """The orthonormal rows themselves, when the transform is not needed."""
    transform, _ = orthonormal_transform(candidates, rtol)
    return transform @ jnp.asarray(candidates)


def numerical_rank(whitened_bank: jax.Array) -> int:
    """Largest ``k`` with ``s_k / s_0 > sqrt(eps)``.

    **Not a tuning knob.** Beyond this cut the Gram matrix of the retained set
    is numerically singular in float64, and ``c^T G c`` returns a finite,
    occasionally negative number rather than raising.

    ``sqrt(eps)`` rather than ``eps`` because the quadratic form SQUARES the
    conditioning: a set that is merely invertible is not usable in one. The
    two answers differ on real inputs -- a direction at ``1e-10`` of the
    leading one is above ``eps`` and below this cut -- and
    ``test_the_cut_is_sqrt_eps_and_not_eps`` is constructed to separate them,
    because every other test in that class passes under either.

    An empty bank, or one whose leading singular value is zero, is rank zero:
    answered before the ratio rather than by it, since neither has a ratio to
    take.
    """
    _refuse_single_precision("numerical_rank")
    singular = np.asarray(
        jnp.linalg.svd(jnp.asarray(whitened_bank), compute_uv=False)
    )
    if singular.size == 0 or singular[0] == 0.0:
        return 0
    cut = float(np.sqrt(np.finfo(singular.dtype).eps))
    return int(np.sum(singular / singular[0] > cut))


def select_svd(whitened_bank: jax.Array, count: int) -> jax.Array:
    """The ``count`` leading right singular directions of the bank.

    **Candidates, not a basis.** They happen to be orthonormal here, which is
    a property of the SVD rather than of the pipeline; :func:`orthonormalise`
    is still what makes that a claim the code depends on.

    Singular values of a bank of prior draws order modes by prior-induced
    AMPLITUDE, which is why a caller with a small signal of interest seeds the
    candidate set with its score direction first and relies on the nesting
    :func:`orthonormal_transform` guarantees.
    """
    _refuse_single_precision("select_svd")
    bank = jnp.asarray(whitened_bank)
    if count <= 0:
        return jnp.zeros((0, bank.shape[1]))
    return jnp.linalg.svd(bank, full_matrices=False)[2][:count]


def select_greedy(whitened_bank: jax.Array, count: int) -> jax.Array:
    """Greedy EIM-style selection: the worst-represented row, repeatedly.

    Returns rows OF THE BANK, in the order chosen -- the selection step of
    Field/Galley/Puerrer and nothing more. Orthonormalising them is a separate
    call, because storing the raw picks is what makes the Gram matrix
    unusable.

    Not the same as sorting by norm, and the difference is the point: after
    the first pick, a second row nearly parallel to it is almost entirely
    explained, so greedy takes a smaller ORTHOGONAL row instead. Pinned in
    ``test_it_picks_a_SPANNING_set_rather_than_the_largest_rows``.

    Stops early when the residual is exhausted -- a rank-``r`` bank yields
    ``r`` picks however many are asked for, rather than repeating one.
    """
    _refuse_single_precision("select_greedy")
    bank = _rows(whitened_bank, "select_greedy")
    if count <= 0:
        return jnp.zeros((0, bank.shape[1]))
    chosen: list[int] = []
    residual = bank.copy()
    for _ in range(min(count, bank.shape[0])):
        index = int(np.argmax(np.linalg.norm(residual, axis=1)))
        direction = residual[index]
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            break
        chosen.append(index)
        direction = direction / norm
        residual = residual - np.outer(residual @ direction, direction)
    return jnp.asarray(bank[chosen])
