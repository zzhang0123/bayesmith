"""Optional ArviZ export: a posterior or predictive result, as an InferenceData.

ArviZ is an OPTIONAL dependency and this is an export-only seam (§0.8): nothing
in the core package imports it, and this module reads it only inside the
function, so ``import bayesmith.bridge.arviz`` stays cheap and succeeds even
where arviz is not installed. The export is a projection, not a re-judgement:
the NamedArrays a result already holds are reshaped and renamed into arviz's
groups, and no number is recomputed.

Group mapping:

    posterior               the latent draws
    posterior_predictive    the replicated draws (a PredictiveResult only)
    log_likelihood          the pointwise log density
    observed_data           the observed node data, the ``data`` half of
                            ``observation_parts``

The draw axis a NamedArray carries is split into arviz's ``chain``/``draw``
axes when the result came from a chain, and stays a single ``draw`` axis when
it did not. A plated node keeps its plate name as an axis and a flat node
keeps its ``{name}_dim{i}`` axis, so the observation unit survives the
round-trip.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bayesmith.graph.evaluate import evaluate
from bayesmith.graph.graph import Graph

__all__ = ["to_inference_data"]


def _reshape_leading(value: Any, dims: tuple[str, ...], chain_shape: Any) -> tuple[Any, list[str]]:
    """A draw-leading array, split for the chain axis; its event dims returned.

    Every named array a result holds is draw-indexed, so ``dims[0]`` is always
    ``"draw"``. The returned array keeps that many axes under arviz's sample
    dims, and the event dims (``dims[1:]``) are handed back for the caller to
    pass arviz beside them.
    """
    if not dims or dims[0] != "draw":
        raise ValueError(
            f"a result's named arrays are draw-indexed (dims[0] == 'draw'); got {dims}"
        )
    array = np.array(value, copy=True)
    event_dims = list(dims[1:])
    if chain_shape is None:
        return array, event_dims
    num_chains, num_draws = chain_shape
    if array.shape[0] != num_chains * num_draws:
        raise ValueError(
            f"the draw axis has {array.shape[0]} entries, which is not "
            f"{num_chains} chains x {num_draws} draws"
        )
    return array.reshape((num_chains, num_draws, *array.shape[1:])), event_dims


def _group(named_arrays: Any, chain_shape: Any) -> tuple[dict, dict]:
    """One arviz group's ``(data, dims)`` from a tuple of NamedArrays."""
    data: dict = {}
    dims: dict = {}
    for array in named_arrays:
        value, event_dims = _reshape_leading(array.value, array.dims, chain_shape)
        data[array.name] = value
        dims[array.name] = event_dims
    return data, dims


def _observed_data(graph: Graph, values: dict) -> tuple[dict, dict]:
    """The observed nodes' data and dims, read off ``observation_parts``.

    The data half of ``observation_parts`` does not depend on the latent
    values -- it is the node's declared data broadcast to its shape -- so any
    complete set of latent values serves to evaluate the graph. The first draw
    of the result is used.
    """
    from bayesmith.dispatch.predictive import _dims
    from bayesmith.exact.gaussian import observation_parts

    env = evaluate(graph, values)
    data, _loc, _scale = observation_parts(graph, env)
    arrays: dict = {}
    dims: dict = {}
    for name in graph.observed:
        value = np.array(data[name], copy=True)
        arrays[name] = value
        dims[name] = list(_dims(graph, name, value, draw=False))
    return arrays, dims


def to_inference_data(
    result: Any,
    *,
    graph: Graph,
    chain_shape: tuple[int, int] | None = None,
) -> Any:
    """Export a posterior or predictive result into arviz's InferenceData.

    ArviZ 1.x returns a ``DataTree``, which IS its InferenceData; the groups
    are its children. The export is a thin projection: ``posterior`` holds the
    latent draws, ``posterior_predictive`` the replicated draws,
    ``log_likelihood`` the pointwise log density, and ``observed_data`` the
    observed node data.

    Args:
        result: a PosteriorResult or PredictiveResult.
        graph: the model the result was produced from, read for the observed
            data.
        chain_shape: ``(num_chains, num_draws)`` to split the draw axis into
            arviz's ``chain``/``draw`` axes. Defaults to the
            ``chain_shape`` a DrawsPosterior already carries, and to ``None``
            (one flat draw axis) for a PredictiveResult unless the caller
            supplies the source posterior's chain shape.
    """
    import arviz as az
    from xarray import DataTree

    from bayesmith.artifacts.results import PosteriorResult, PredictiveResult

    if not isinstance(graph, Graph):
        raise TypeError(f"to_inference_data's graph is a Graph; got {graph!r}")

    if isinstance(result, PosteriorResult):
        latent = list(result.representation.draws)
        replicated: list = []
        log_likelihood = (
            [result.pointwise_log_likelihood]
            if result.pointwise_log_likelihood is not None
            else []
        )
        if chain_shape is None:
            chain_shape = getattr(result.representation, "chain_shape", None)
    elif isinstance(result, PredictiveResult):
        latent = list(result.latent_draws)
        replicated = list(result.replicated_draws)
        log_likelihood = (
            [result.pointwise_log_density]
            if result.pointwise_log_density is not None
            else []
        )
    else:
        raise TypeError(
            f"to_inference_data exports a PosteriorResult or a PredictiveResult; "
            f"got {type(result).__name__}"
        )

    sample_dims = ("chain", "draw") if chain_shape else ("draw",)

    groups: dict = {}
    if latent:
        data, dims = _group(latent, chain_shape)
        groups["posterior"] = az.dict_to_dataset(
            data, dims=dims, sample_dims=sample_dims
        )
    if replicated:
        data, dims = _group(replicated, chain_shape)
        groups["posterior_predictive"] = az.dict_to_dataset(
            data, dims=dims, sample_dims=sample_dims
        )
    if log_likelihood:
        data, dims = _group(log_likelihood, chain_shape)
        groups["log_likelihood"] = az.dict_to_dataset(
            data, dims=dims, sample_dims=sample_dims
        )
    if graph.observed:
        if not latent:
            raise ValueError(
                "observed data needs the result's latent draws to evaluate the "
                "graph; a result with none cannot be exported with observed_data"
            )
        values = {array.name: array.value[0] for array in latent}
        data, dims = _observed_data(graph, values)
        groups["observed_data"] = az.dict_to_dataset(
            data, dims=dims, sample_dims=()
        )

    return DataTree.from_dict(groups)

