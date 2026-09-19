"""Executable demos must recover simulated parameters, not merely run.

The CLI test catches a changed operator, a lost likelihood, a wrong route or
a runner that reports failure while exiting successfully. The guard tests
catch permissive recovery checks without touching the inference implementation.
"""

import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
@pytest.mark.parametrize(
    ("case", "method"),
    [
        ("linear_gaussian", "nuts"),
        ("power_law", "nuts"),
        ("hierarchical", "gcr+nuts"),
        ("bernoulli", "nuts"),
        ("multiplicative_noise", "bias_corrected_log_linear+mh+iterative_gls+mh"),
        ("composed_process", "iterative_gls+mh+nuts"),
    ],
)
def test_demo_executes_and_reports_actual_recovery(tmp_path, case, method):
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.inference",
            "--case",
            case,
            "--output",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode in (0, 1), done.stdout[-4000:] + done.stderr[-4000:]
    report = json.loads((tmp_path / case / "result.json").read_text())
    assert done.returncode == int(not report["passed"])
    if case != "composed_process":
        assert report["passed"], report["checks"]
    else:
        # A fixed truth can legitimately miss a 99% interval. The CLI must
        # preserve that failure, while the sampler itself still passes checks.
        assert report["checks"]["chain_diagnostics"]["passed"]
        assert all(row["informative"] for row in report["parameters"])
        assert report["passed"] == all(row["covered"] for row in report["parameters"])
    assert report["method"] == method
    expected_dimensions = {
        "linear_gaussian": 4,
        "power_law": 4,
        "hierarchical": 9,
        "bernoulli": 4,
        "multiplicative_noise": 4,
        "composed_process": 20,
    }
    assert len(report["parameters"]) == expected_dimensions[case]
    # A visually labelled diagnostic must come from the executed task's artifact.
    from bayesmith.artifacts import load_artifact

    analysis = load_artifact(tmp_path / case / "analysis.artifact.json")
    assert [(f["code"], f["conclusion"]) for f in report["preflight"]] == [
        (f.code, f.conclusion) for f in analysis.findings
    ]
    posterior = load_artifact(tmp_path / case / "posterior.artifact.json")
    assert report["execution"]["termination"] == posterior.run.termination.reason.value
    assert report["execution"]["initial_values"] == {
        item.name: item.value.tolist() for item in posterior.run.initial_values
    }
    assert report["execution"]["sampling"] == json.loads(
        json.dumps(dict(posterior.run.sampling_details))
    )
    if case == "bernoulli":
        # A policy failure (e.g. divergences) must not become a green demo just
        # because the same draws still satisfy the older ESS/R-hat check.
        from dataclasses import replace

        from examples.inference.common import diagnostic_checks

        details = dict(posterior.run.sampling_details)
        details.update(diagnostics_passed=False, divergences=1)
        failed_run = replace(posterior.run, sampling_details=tuple(details.items()))
        assert not diagnostic_checks(replace(posterior, run=failed_run))["passed"]
        assert report["signal"]["view"]["kind"] == "surface"
        assert len(report["signal"]["view"]["x2"]) == len(report["signal"]["data"])
    if case == "power_law":
        assert set(report["signal"]["view"]["channel"]) == {0, 1}
        assert report["checks"]["quadrature_oracle"]["passed"]
        assert report["checks"]["repeated_reference"]["passed"]
        assert report["signal"]["view"]["model_kind"] == "power_law_regression"
        assert any(
            f["code"] == "block_0_jeffreys" and f["conclusion"] == "nonflat"
            for f in report["preflight"]
        )
    blocking = report["blocking"]
    assert blocking["executed"]["executed"] is True
    assert {
        name for block in blocking["executed"]["blocks"] for name in block["latents"]
    } == {p["latent"] for p in blocking["priors"]}
    assert all(block["reason"] for block in blocking["executed"]["blocks"])
    assert blocking["factor_comparison"]["executed"] is False
    if case == "multiplicative_noise":
        assert [b["latents"] for b in blocking["executed"]["blocks"]] == [
            ["p_g"],
            ["p_n"],
        ]
        assert [b["coordinates"] for b in blocking["executed"]["blocks"]] == [2, 2]
        assert report["proposal_selection"] == "automatic"
        assert blocking["factor_comparison"]["status"] == "refused"
        assert blocking["factor_comparison"]["exception"] == "NotGaussian"
        probe = report["blocking_probes"][0]
        assert probe["observations"] == 1500
        assert probe["posterior_executed"] is False
        large = probe["blocking"]
        assert {p["latent"]: p["coordinates"] for p in large["priors"]} == {
            "p_g": 2,
            "p_n": 500,
        }
        assert large["executed"]["executed"] is False
        assert [b["coordinates"] for b in large["executed"]["blocks"]] == [2, 500]
        assert [b["method"] for b in large["executed"]["blocks"]] == [
            "bias_corrected_log_linear+mh",
            "nuts",
        ]
        assert large["factor_comparison"]["status"] == "refused"
        assert large["factor_comparison"]["exception"] == "NotGaussian"
    if case == "composed_process":
        assert blocking["executed"]["blocks"][0]["latents"] == [
            "instance",
            "background",
        ]
        assert blocking["executed"]["blocks"][0]["coordinates"] == 14
        assert blocking["executed"]["blocks"][1]["coordinates"] == 6
    assert report["parameters"]
    if case != "composed_process":
        assert all(row["covered"] for row in report["parameters"])
    assert all(row["posterior_sd"] > 0 for row in report["parameters"])
    assert any(node["kind"] == "deterministic" for node in report["dag"])
    assert any(node["kind"] == "observed" for node in report["dag"])
    roots = {
        node["name"]
        for node in report["dag"]
        if node["kind"] == "latent" and not node["parents"]
    }
    assert all(
        p["family"] == "Uniform" for p in blocking["priors"] if p["latent"] in roots
    )
    if case == "composed_process":
        nodes = {node["name"]: node for node in report["dag"]}
        assert nodes["instance"]["parents"] == ["power"]
        assert nodes["power"]["parents"] == ["frequencies", "power_amplitude"]
        assert report["signal"]["view"]["noise_factor"] == 1.0
        assert len(report["signal"]["view"]["spectrum"]["truth"]) == 12
        assert (
            next(p for p in blocking["priors"] if p["latent"] == "instance")["family"]
            == "Normal"
        )
    assert (tmp_path / case / "dag.mmd").is_file()
    assert (tmp_path / case / "posterior.npz").is_file()


def test_recovery_guard_rejects_a_biased_or_uninformative_posterior():
    common = importlib.import_module("examples.inference.common")
    quantiles = np.linspace(-2.0, 2.0, 2000)
    for draws, covered, informative in (
        (quantiles / 10.0 + 10.0, False, True),
        (quantiles * 100.0, True, False),
        (quantiles / 10.0, True, True),
    ):
        checked = common.recovery_checks(
            {"theta": draws},
            {"theta": np.array(0.0)},
            {"theta": np.array(2.0)},
        )
        row = checked["parameters"][0]
        assert row["covered"] is covered
        assert row["informative"] is informative
        assert checked["passed"] is (covered and informative)


def test_saved_block_record_fields_render_as_text_not_html():
    """A saved dimension/index must not inject active markup into the notebook."""
    from html.parser import HTMLParser

    from examples.inference.presentation import blocking_panel

    payload = '<img src=x onerror="alert(1)">'
    block = {
        "index": payload,
        "coordinates": payload,
        "latents": ["beta"],
        "conditional_on": [],
        "method": "gcr",
        "reason": "linear",
        "linearity_evidence": None,
    }
    report = {
        "blocking": {
            "executed": {
                "executed": True,
                "strategy": "declared",
                "description": "test",
                "status": "planned",
                "blocks": [block],
                "plan_text": "plan",
            },
            "factor_comparison": {
                "executed": False,
                "strategy": "factor_partition",
                "description": "test",
                "status": "refused",
                "exception": "NotGaussian",
                "reason": "test",
            },
            "priors": [{"latent": "beta", "family": "Normal", "coordinates": payload}],
        }
    }

    class Reader(HTMLParser):
        def __init__(self):
            super().__init__()
            self.active = []
            self.text = []

        def handle_starttag(self, tag, attrs):
            if tag in {"img", "script"} or any(k.startswith("on") for k, _ in attrs):
                self.active.append((tag, attrs))

        def handle_data(self, data):
            self.text.append(data)

    reader = Reader()
    reader.feed(blocking_panel(report))
    assert not reader.active
    assert payload in "".join(reader.text)


def test_recovery_guard_rejects_nonfinite_or_missing_draws():
    common = importlib.import_module("examples.inference.common")
    for draws in ({}, {"theta": np.full(20, np.nan)}):
        with pytest.raises(ValueError):
            common.recovery_checks(
                draws, {"theta": np.array(0.0)}, {"theta": np.array(2.0)}
            )


def test_declared_operators_match_independent_values():
    linear = importlib.import_module("examples.inference.linear_gaussian")
    decay = importlib.import_module("examples.inference.exponential_decay")
    logistic = importlib.import_module("examples.inference.bernoulli")
    np.testing.assert_allclose(
        linear.linear_mean(
            np.array([[1.0, -1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
            np.array([2.0, 3.0, 4.0, 5.0]),
        ),
        [4.0, 6.0],
    )
    np.testing.assert_allclose(
        decay.decay(
            np.array([0.0, 1.0, 1.0]),
            np.array([2.0, 4.0]),
            np.log(np.array([2.0, 4.0])),
            np.array([1.0, -1.0]),
            np.array([0, 0, 1]),
        ),
        [3.0, 2.0, 0.0],
    )
    np.testing.assert_allclose(logistic.binary_observation(np.array([0.0])).mean, [0.5])


def test_multiplicative_operators_and_parameter_dependent_noise():
    import jax

    from bayesmith import trace
    from bayesmith.graph.evaluate import apply_probabilistic, evaluate

    demo = importlib.import_module("examples.inference.multiplicative_noise")
    with jax.enable_x64(True):
        U = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
        A = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        p_g, p_n = np.log([2.0, 3.0]), np.array([4.0, 5.0])
        expected = np.array([8.0, 15.0, 6.0])
        mu = demo.mean_signal(U, A, p_g, p_n)
        np.testing.assert_allclose(mu, expected, rtol=1e-12)
        distribution = demo.relative_gaussian(mu)
        np.testing.assert_allclose(distribution.loc, expected, rtol=1e-12)
        np.testing.assert_allclose(
            distribution.scale, [0.008, 0.015, 0.006], rtol=1e-12
        )
        # Sigma must move with the inferred mean; a frozen/noise-floor twin fails.
        doubled = demo.relative_gaussian(demo.mean_signal(U, A, p_g, 2 * p_n))
        np.testing.assert_allclose(doubled.scale, 2 * distribution.scale, rtol=1e-12)
        # Check the actual graph wiring and the likelihood normalisation too.
        data = expected + np.array([-1.0, 0.5, 2.0]) * np.array([0.008, 0.015, 0.006])
        graph = trace(demo.model, U, A, data)
        for factor in (1.0, 2.0):
            env = evaluate(graph, {"p_g": p_g, "p_n": factor * p_n})
            actual = apply_probabilistic(graph, graph.node("obs"), env)
            center, sigma = factor * expected, factor * np.array([0.008, 0.015, 0.006])
            loglike = np.sum(
                -0.5 * ((data - center) / sigma) ** 2
                - np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
            )
            np.testing.assert_allclose(actual.log_prob(data).sum(), loglike, rtol=1e-12)


def test_multiplicative_design_identifies_all_four_coordinates():
    import jax

    demo = importlib.import_module("examples.inference.multiplicative_noise")
    with jax.enable_x64(True):
        U, A = map(np.asarray, demo.design_matrices(np.linspace(0.0, 1.0, 128)))
        assert U.shape == A.shape == (128, 2)
        assert np.all(A >= 0)
        assert np.all(A[:, 0] > 0)
        p_g, p_n = np.array([0.12, -0.08]), np.array([1.2, 0.7])
        gain = np.exp(U @ p_g)
        mu = gain * (A @ p_n)
        # Independent analytic Jacobian, including both parameter branches.
        jacobian = np.column_stack([mu[:, None] * U, gain[:, None] * A])
        assert np.linalg.matrix_rank(jacobian) == 4
        # A gain intercept would make an exact scaling degeneracy with p_n.
        assert np.linalg.matrix_rank(np.column_stack([U, np.ones(128)])) == 3


def test_composed_process_conditional_power_and_white_noise_density():
    import jax
    import jax.numpy as jnp

    from bayesmith import trace
    from bayesmith.graph.evaluate import apply_probabilistic, evaluate
    from examples.inference import composed_process as demo

    with jax.enable_x64(True):
        x = np.linspace(0.0, 1.0, 24, endpoint=False)
        k, _, R, B, U = demo.design_matrices(jnp.asarray(x))
        instance = np.linspace(-0.1, 0.1, 12)
        theta = np.array([0.45, 0.09])
        background = np.array([3.0, 0.2])
        gain = np.array([0.08, -0.05])
        expected = (
            np.asarray(R) @ instance
            + 0.7 * np.exp(-0.5 * ((x - theta[0]) / theta[1]) ** 2)
            + np.asarray(B) @ background
        ) * np.exp(np.asarray(U) @ gain)
        data = expected * (1 + 0.015 * np.linspace(-1.0, 1.0, 24))
        graph = trace(demo.model, x, k, R, B, U, data)
        for amplitude, sigma_w in [(0.5, 0.015), (1.0, 0.03)]:
            values = {
                "power_amplitude": jnp.asarray(amplitude),
                "instance": jnp.asarray(instance),
                "nonlinear": jnp.asarray(theta),
                "background": jnp.asarray(background),
                "gain": jnp.asarray(gain),
                "sigma_w": jnp.asarray(sigma_w),
            }
            env = evaluate(graph, values)
            np.testing.assert_allclose(env["mu"], expected, rtol=1e-12)
            process = apply_probabilistic(graph, graph.node("instance"), env)
            np.testing.assert_allclose(
                process.variance, (amplitude / np.asarray(k)) ** 2, rtol=1e-12
            )
            law = apply_probabilistic(graph, graph.node("obs"), env)
            sigma = sigma_w * np.abs(expected)
            reference = np.sum(
                -0.5 * ((data - expected) / sigma) ** 2
                - np.log(sigma)
                - 0.5 * np.log(2 * np.pi)
            )
            np.testing.assert_allclose(law.log_prob(data).sum(), reference, rtol=1e-12)
        # The process remains stochastic; it is not replaced by a deterministic zero.
        a = process.sample(jax.random.key(0))
        b = process.sample(jax.random.key(1))
        assert not np.array_equal(a, b)
