"""Optional ArviZ export: observation unit and chain semantics survive (§0.8).

ArviZ is an optional, export-only dependency, and the export is a projection,
not a re-judgement: the NamedArrays a result already holds are reshaped into
arviz's groups, and reading the groups back must recover the observation unit
(plate / grouping) and the chain structure (draw / chains axes).
"""

from __future__ import annotations

import subprocess
import sys

import jax
import numpy as np
import pytest

from bayesmith.artifacts.base import ArtifactKind, ArtifactRef, ComputeBudget
from bayesmith.artifacts.refusal import Refusal
from bayesmith.artifacts.results import PosteriorResult, PredictiveResult
from bayesmith.artifacts.tasks import PosteriorTask, PredictiveTask, new_task_meta
from bayesmith.bridge.arviz import to_inference_data
from bayesmith.dispatch.task import compile_task, execute_task
from tests.dispatch.test_task_protocol import model_ref
from tests.exact.models import bilinear_pair, plated_latent, straight_line

try:
    import arviz as _arviz
except ImportError:  # the wheel venv omits the dev-only arviz extra
    _arviz = None

#: Skip the export tests at RUN time, not at collection. A module-level
#: ``pytest.importorskip`` would drop these seven tests from collection when
#: arviz is absent, shrinking the collection that ``tests/test_readme_count.py``
#: pins to the README's count -- a partial view read as a shrunken suite. A
#: class-level ``skipif`` keeps them collected and skips them when they run.
requires_arviz = pytest.mark.skipif(_arviz is None, reason="arviz is not installed")

BUDGET = ComputeBudget(draws=8, warmup=8, chains=1)


def posterior_task(**overrides) -> PosteriorTask:
    fields = {
        "meta": new_task_meta(label="run"),
        "budget": BUDGET,
        "nuts_on_collapse": False,
    }
    fields.update(overrides)
    return PosteriorTask(**fields)


def planned_for(graph, task):
    planned = compile_task(graph, task, model_ref=model_ref())
    assert not isinstance(planned, Refusal), planned
    return planned


def source_ref(source) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=source.meta.artifact_id,
        revision=source.meta.revision,
        artifact_type=ArtifactKind.RESULT,
    )


def predictive_task(source, **overrides) -> PredictiveTask:
    fields = {
        "meta": new_task_meta(label="ppc"),
        "source_posterior_ref": source_ref(source),
        "conditioned_sites": ("d",),
        "replicated_sites": ("d",),
        "latent_sites": ("w",),
    }
    fields.update(overrides)
    return PredictiveTask(**fields)


@requires_arviz
class TestAPosteriorExportsItsLatentsAndLikelihood:
    def test_a_chained_run_splits_the_draw_axis_into_chain_and_draw(self):
        """A NUTS posterior came from a chain, so its draw axis becomes arviz's
        ``chain`` and ``draw`` axes and the pointwise likelihood keeps them."""
        graph = bilinear_pair()
        result = execute_task(
            planned_for(graph, posterior_task()), key=jax.random.key(3)
        )
        assert result.representation.method == "nuts"
        assert result.representation.chain_shape == (1, 8)

        idata = to_inference_data(result, graph=graph)

        assert set(idata.children) == {"posterior", "observed_data", "log_likelihood"}
        posterior = idata["posterior"]
        assert set(posterior.dims) == {"chain", "draw"}
        assert posterior["gain"].dims == ("chain", "draw")
        assert posterior["gain"].shape == (1, 8)
        log_likelihood = idata["log_likelihood"]["log_likelihood"]
        assert log_likelihood.dims == ("chain", "draw", "d_dim0")
        assert log_likelihood.shape == (1, 8, 10)

    def test_an_iid_run_keeps_one_flat_draw_axis(self):
        """``gcr`` draws independently, so there is no chain axis to invent."""
        graph = straight_line()
        result = execute_task(
            planned_for(graph, posterior_task()), key=jax.random.key(2)
        )
        assert result.representation.method == "gcr"
        assert result.representation.chain_shape is None

        idata = to_inference_data(result, graph=graph)

        posterior = idata["posterior"]
        assert posterior["w"].dims == ("draw",)
        assert "chain" not in posterior.dims
        assert idata["log_likelihood"]["log_likelihood"].dims == ("draw", "d_dim0")

    def test_a_plated_run_keeps_its_plate_axis(self):
        """The observation unit is the plate the model declared, so the plate
        name must survive in the likelihood and the observed data."""
        graph = plated_latent()
        result = execute_task(
            planned_for(graph, posterior_task()), key=jax.random.key(2)
        )

        idata = to_inference_data(result, graph=graph)

        assert idata["log_likelihood"]["log_likelihood"].dims == ("draw", "obs")
        assert idata["observed_data"]["d"].dims == ("obs",)
        assert "obs" in idata["observed_data"].dims

    def test_observed_data_is_the_observed_node_data(self):
        """The observed_data group holds the node's declared data -- the
        ``data`` half of ``observation_parts`` -- not a resampled value."""
        graph = straight_line()
        result = execute_task(
            planned_for(graph, posterior_task()), key=jax.random.key(2)
        )
        idata = to_inference_data(result, graph=graph)
        np.testing.assert_allclose(
            idata["observed_data"]["d"].values,
            np.asarray(graph.node("d").observed),
            rtol=0.0,
            atol=0.0,
        )


@requires_arviz
class TestAPredictiveExportsItsReplicatedDraws:
    def test_a_predictive_result_exports_posterior_predictive_and_log_likelihood(self):
        graph = straight_line()
        posterior = execute_task(
            planned_for(graph, posterior_task()), key=jax.random.key(2)
        )
        predictive = execute_task(
            planned_for(graph, predictive_task(posterior)),
            key=jax.random.key(3),
            source_posterior=posterior,
        )
        assert isinstance(posterior, PosteriorResult)
        assert isinstance(predictive, PredictiveResult)

        idata = to_inference_data(predictive, graph=graph)

        assert set(idata.children) == {
            "posterior",
            "posterior_predictive",
            "observed_data",
            "log_likelihood",
        }
        assert idata["posterior"]["w"].dims == ("draw",)
        assert idata["posterior_predictive"]["d"].dims == ("draw", "d_dim0")
        assert idata["log_likelihood"]["log_likelihood"].dims == ("draw", "d_dim0")
        assert idata["observed_data"]["d"].dims == ("d_dim0",)

    def test_a_chained_predictive_splits_the_shared_draw_axis(self):
        """A predictive result has no chain_shape field of its own, so the
        source posterior's is handed in; latent and replicated draws then share
        the same ``chain``/``draw`` axes."""
        graph = bilinear_pair()
        posterior = execute_task(
            planned_for(graph, posterior_task()), key=jax.random.key(3)
        )
        predictive = execute_task(
            planned_for(
                graph, predictive_task(posterior, latent_sites=("gain", "t_ant"))
            ),
            key=jax.random.key(3),
            source_posterior=posterior,
        )

        idata = to_inference_data(
            predictive, graph=graph, chain_shape=posterior.representation.chain_shape
        )

        assert idata["posterior"]["gain"].dims == ("chain", "draw")
        assert idata["posterior_predictive"]["d"].dims == ("chain", "draw", "d_dim0")
        assert idata["log_likelihood"]["log_likelihood"].dims == ("chain", "draw", "d_dim0")


class TestTheModuleStaysCheapWithoutArviZ:
    def test_importing_the_export_module_does_not_import_arviz(self):
        """ArviZ is read inside the function, so importing the module alone --
        the cheap path a consumer pays to discover the export exists -- must
        not pull arviz or xarray in."""
        code = (
            "import sys; import bayesmith.bridge.arviz; "
            "print(sorted({'arviz', 'xarray'} & set(sys.modules)))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "[]"

