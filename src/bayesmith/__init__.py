"""bayesmith: a graph of operators is a Bayesian model.

Deterministic operators propagate dependence; probabilistic ones contribute a
conditional density. The graph's structure is what selects the inference
method -- exact where a subgraph permits one, NUTS where it does not.
"""

from bayesmith.bridge.numpyro_bridge import nuts, to_numpyro
from bayesmith.errors import BayesmithError, GraphError, TraceError
from bayesmith.graph.evaluate import evaluate, log_joint
from bayesmith.graph.graph import Graph, Plate
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic
from bayesmith.graph.trace import NodeRef, const, det, observe, plate, sample, trace

__all__ = [
    # tracing
    "trace",
    "const",
    "det",
    "sample",
    "observe",
    "plate",
    "NodeRef",
    # graph
    "Graph",
    "Plate",
    "Node",
    "Const",
    "Deterministic",
    "Probabilistic",
    # evaluation
    "evaluate",
    "log_joint",
    # inference
    "to_numpyro",
    "nuts",
    # errors
    "BayesmithError",
    "GraphError",
    "TraceError",
]
