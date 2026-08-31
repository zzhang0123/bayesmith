"""Serialisable, invalidatable, evaluable protocol objects for Bayesian tasks.

The leaf layer: nothing here imports :mod:`bayesmith.graph`,
:mod:`bayesmith.dispatch`, JAX, Equinox or NumPyro at module scope, and
nothing here holds a runtime object. An artifact is data -- a Graph, a
callable, a compiled executable or a backend handle is represented by a
reference, a fingerprint or a runtime attachment, never pickled into the
artifact itself.

R1 builds this package one task at a time and this module stays the single
public inventory, so it is deliberately empty of names until the protocol
types it would export actually exist: a name re-exported before its module
lands is a name that resolves to a promise. The canonical codec every artifact
is encoded through is :mod:`bayesmith.artifacts._codec`, private because the
wire format is the package's own business rather than a caller's.
"""
