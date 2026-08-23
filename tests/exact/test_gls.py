"""Finding the covariance a prediction-dependent sigma implies."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.errors import GraphError, StructureError
from bayesmith.exact.gls import (
    check_prediction_dependence,
    iterative_gls,
    sigma_from_graph,
)
from bayesmith.exact.linearity import linear_operator
from bayesmith.exact.solve import wiener_solve
from tests.exact.models import radiometer, straight_line
from tests.exact.oracle import flat_domain, graph_oracle

KAPPA = 0.05
FLOOR = 1e-3


def test_a_constant_sigma_converges_in_one_step():
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), depends_on_prediction=False, tol=1e-14
        )
        direct, _ = wiener_solve(
            block,
            noise_std=sigma_from_graph(graph, {})({"w": jnp.asarray(0.0)}),
            tol=1e-14,
        )
    assert int(result.iterations) == 1
    assert bool(result.converged)
    assert float(result.solution["w"]) == pytest.approx(float(direct["w"]), rel=1e-10)


def test_iterative_gls_finds_the_fixed_point_a_dense_iteration_finds():
    """A NumPy fixed-point loop, sharing nothing with the JAX while_loop."""
    with jax.enable_x64(True):
        graph = radiometer(kappa=KAPPA, floor=FLOOR)
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), tol=1e-14, reweight_tol=1e-10
        )
        oracle = graph_oracle(graph, ("w",), at={})

    design, offset, data = oracle.design, oracle.offset, oracle.data
    prior_precision = np.diag(1.0 / oracle.prior_std**2)
    x = oracle.prior_mean.copy()
    for _ in range(400):
        sigma = KAPPA * np.abs(design @ x + offset) + FLOOR
        noise_precision = np.diag(1.0 / sigma**2)
        x = np.linalg.solve(
            design.T @ noise_precision @ design + prior_precision,
            design.T @ noise_precision @ (data - offset)
            + prior_precision @ oracle.prior_mean,
        )

    assert bool(result.converged)
    assert np.allclose(flat_domain(result.solution, block.names), x, rtol=1e-6)
    # Mutation guard (Task 8 mutation 1): a `delta` denominator that reads the
    # PRE-update iterate rather than the post-update one has nothing left to
    # shrink it once the sequence is near the fixed point, so a broken
    # normalisation that never triggers convergence runs out the clock at
    # max_reweights (100) instead of stopping early -- this fixture converges
    # in single digits.
    assert int(result.iterations) < 50


def test_the_returned_sigma_really_is_a_fixed_point():
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        sigma_of = sigma_from_graph(graph, {})
        result = iterative_gls(block, sigma_of, tol=1e-14, reweight_tol=1e-10)
        recomputed = sigma_of(result.solution)
    assert np.allclose(
        np.asarray(recomputed["d"]), np.asarray(result.noise_std["d"]), rtol=1e-6
    )


def test_check_prediction_dependence_catches_a_false_declaration():
    """`depends_on_prediction=False` on a radiometer node is a claim, and false.

    Declared False, a dispatcher skips the reweighting loop entirely and
    solves at whatever sigma the prior mean implies -- a confident answer at
    the wrong covariance, with nothing to notice.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(StructureError, match="depends_on_prediction"):
            check_prediction_dependence(
                block, sigma_from_graph(graph, {}), declared=False
            )


def test_check_prediction_dependence_accepts_a_true_declaration():
    with jax.enable_x64(True):
        graph = straight_line()
        block = linear_operator(graph, ("w",), at={})
        movement = check_prediction_dependence(
            block, sigma_from_graph(graph, {}), declared=False
        )
    assert movement == pytest.approx(0.0, abs=1e-12)


def test_a_capped_run_reports_converged_false_rather_than_pretending():
    """converged=False means the returned covariance is NOT a fixed point.

    Everything conditioned on it inherits that, so it is returned as a flag
    rather than raised -- but it must never read True.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        result = iterative_gls(
            block,
            sigma_from_graph(graph, {}),
            tol=1e-14,
            min_reweights=1,
            max_reweights=1,
        )
    assert not bool(result.converged)


def test_a_one_sided_probe_would_miss_a_clipped_sigma():
    from tests.exact.models import one_sided_sigma

    with jax.enable_x64(True):
        graph = one_sided_sigma()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(StructureError, match="depends_on_prediction"):
            check_prediction_dependence(
                block, sigma_from_graph(graph, {}), declared=False
            )


# Structural-dimension audit (not in the plan): every fixture above is a
# single SCALAR latent constrained by a single observed node. `sigma_of`,
# `check_prediction_dependence`'s probe construction, and the reweighting
# loop's `tree_norm`/`jax.tree.map` plumbing are generic dict/pytree code
# with no explicit branch on leaf count or element count -- and Task 7 found
# two real bugs (`_split_like` sharing a key across leaves, `omega_prior`
# collapsing a plate to one draw) hiding behind exactly that genericity on a
# test set that never varied those dimensions. The two fixtures below give
# "more than one latent leaf", "more than one observed leaf", and "one leaf
# with more than one element" each at least one exercise through the actual
# reweighting loop, checked against the same independent dense oracle the
# plan's own test uses.

N1_GROUP, M2_GROUP = 9, 6
KAPPA_GROUP, FLOOR_GROUP, S2_GROUP = 0.04, 2e-3, 0.25

N_PLATE, TAU_PLATE, KAPPA_PLATE, FLOOR_PLATE = 6, 3.0, 0.06, 2e-3


def test_iterative_gls_handles_a_multi_leaf_multi_observed_block():
    """Two latent leaves solved JOINTLY; two observed leaves, one
    prediction-dependent and one not.

    `radiometer_group`'s ``a``, ``b`` are solved together (like
    `two_linear_latents`), and its ``d1``/``d2`` are two observed nodes
    (like `two_observations`) -- but combined with a genuinely
    prediction-dependent sigma, which neither of those two constant-sigma
    fixtures has, so this is the first test to run the reweighting loop
    itself on more than one leaf on either side.
    """
    from tests.exact.models import radiometer_group

    with jax.enable_x64(True):
        graph = radiometer_group(
            n=N1_GROUP, m=M2_GROUP, kappa=KAPPA_GROUP, floor=FLOOR_GROUP, s2=S2_GROUP
        )
        block = linear_operator(graph, ("a", "b"), at={})
        seam = sigma_from_graph(graph, {})
        result = iterative_gls(block, seam, tol=1e-14, reweight_tol=1e-10)
        oracle = graph_oracle(graph, ("a", "b"), at={})
        # The probe scans every observed leaf, not just the first -- d1
        # genuinely moves and d2 by construction does not.
        with pytest.raises(StructureError, match="depends_on_prediction"):
            check_prediction_dependence(block, seam, declared=False)

    design, offset, data = oracle.design, oracle.offset, oracle.data
    prior_precision = np.diag(1.0 / oracle.prior_std**2)
    x = oracle.prior_mean.copy()
    for _ in range(400):
        pred = design @ x + offset
        sigma = np.concatenate(
            [
                KAPPA_GROUP * np.abs(pred[:N1_GROUP]) + FLOOR_GROUP,
                np.full(pred.size - N1_GROUP, S2_GROUP),
            ]
        )
        noise_precision = np.diag(1.0 / sigma**2)
        x = np.linalg.solve(
            design.T @ noise_precision @ design + prior_precision,
            design.T @ noise_precision @ (data - offset)
            + prior_precision @ oracle.prior_mean,
        )

    assert bool(result.converged)
    assert np.allclose(flat_domain(result.solution, block.names), x, rtol=1e-6)


def test_iterative_gls_handles_a_plated_member():
    """One latent leaf with several elements, rather than several scalar
    leaves.

    `radiometer`'s ``w`` is a scalar; `plated_radiometer`'s ``z`` is a plate
    of six, each element's own sigma tracking that SAME element's
    prediction -- so `tree_norm`'s reduction and the probe in
    `check_prediction_dependence` run on one array leaf here, never
    exercised by a fixture whose domain is a dict of scalars.
    """
    from tests.exact.models import plated_radiometer

    with jax.enable_x64(True):
        graph = plated_radiometer(
            n=N_PLATE, tau=TAU_PLATE, kappa=KAPPA_PLATE, floor=FLOOR_PLATE
        )
        block = linear_operator(graph, ("z",), at={})
        result = iterative_gls(
            block, sigma_from_graph(graph, {}), tol=1e-14, reweight_tol=1e-10
        )
        oracle = graph_oracle(graph, ("z",), at={})

    design, offset, data = oracle.design, oracle.offset, oracle.data
    prior_precision = np.diag(1.0 / oracle.prior_std**2)
    x = oracle.prior_mean.copy()
    for _ in range(400):
        sigma = KAPPA_PLATE * np.abs(design @ x + offset) + FLOOR_PLATE
        noise_precision = np.diag(1.0 / sigma**2)
        x = np.linalg.solve(
            design.T @ noise_precision @ design + prior_precision,
            design.T @ noise_precision @ (data - offset)
            + prior_precision @ oracle.prior_mean,
        )

    assert bool(result.converged)
    assert np.allclose(flat_domain(result.solution, block.names), x, rtol=1e-6)


def test_iterative_gls_refuses_min_reweights_above_max():
    """The loop caps at max_reweights either way, so min > max would
    silently deliver fewer steps than asked for -- refused up front rather
    than discovered from a suspiciously low `iterations`.
    """
    with jax.enable_x64(True):
        graph = radiometer()
        block = linear_operator(graph, ("w",), at={})
        with pytest.raises(GraphError, match="min_reweights"):
            iterative_gls(
                block,
                sigma_from_graph(graph, {}),
                min_reweights=5,
                max_reweights=2,
            )
