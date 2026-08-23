"""Exception family for bayesmith.

Deliberately stdlib-only: this module is imported by everything, including
from contexts that must not pay for jax, so it may never import numpy or jax.

Every concrete class also derives from the closest builtin, so generic
handlers keep working.
"""


class BayesmithError(Exception):
    """Base class for every error bayesmith raises."""


class GraphError(BayesmithError, ValueError):
    """A graph was declared or evaluated inconsistently.

    Covers: a node naming a parent that was not declared before it, a
    duplicate node name, a latent node left without a value, and a plate
    used by a node whose parents are all outside it.
    """


class TraceError(BayesmithError, RuntimeError):
    """A tracing primitive was called outside ``trace(...)``."""


class StructureError(BayesmithError, ValueError):
    """A declared structural claim was checked and found false.

    Raised where a declaration and the model contradict each other: a
    ``Deterministic`` declaring ``linear_in`` whose prediction is not affine
    in that parent, a ``Probabilistic`` declaring
    ``depends_on_prediction=False`` whose scale does in fact move with the
    block, a ``dist_fn`` whose *type* says Normal but whose ``log_prob`` does
    not match the ``loc``/``scale`` read off it.

    **Not** raised when a graph merely fails to qualify for an exact method.
    "You did not declare it" is a dispatch outcome -- the block falls through
    to NUTS and the plan records why. "You declared it, and it is false" is
    this. Conflating the two would make an ordinary model that simply has no
    exact structure look like a broken one.
    """


class ConvergenceError(BayesmithError, RuntimeError):
    """An iterative procedure did not reach the accuracy it was asked for.

    For guards *inside* a jitted solve the mechanism is ``equinox.error_if``
    instead, because a Python ``if`` cannot branch on a traced value. This
    class is for the checks that run on concrete values, outside any trace --
    ``iterative_gls``'s ``converged=False`` promoted to a hard failure, say.
    """


class NotGaussian(BayesmithError, TypeError):
    """A node's distribution is not a diagonal Gaussian.

    Purely descriptive, and deliberately blameless: most models contain
    perfectly good non-Gaussian nodes. P3b's classifier catches this and
    routes the block to NUTS.

    **A sibling of** :class:`StructureError`. A dispatcher writing
    ``except NotGaussian`` must NOT also swallow a :class:`StructureError`: that one means a node's *type*
    says Normal while its own ``log_prob`` says otherwise, and silently
    downgrading it to NUTS would hide a broken model behind an
    ordinary-looking fallback.

    A subclass relationship, in either direction, is the ONE thing that would
    break this -- and it is worth being precise about which, because two
    classes that merely *share* a base are unaffected. ``except`` matches on
    the raised exception's MRO, not on a common ancestor: these two already
    share :class:`BayesmithError`, and neither catches the other. Measured,
    because an earlier draft of this docstring claimed otherwise.

    The differing builtin bases -- ``TypeError`` here against
    :class:`StructureError`'s ``ValueError`` -- therefore buy something
    narrower, and something real: a *generic* handler can tell the two apart,
    so an ``except ValueError`` around a modelling call sees the broken-model
    case and not the ordinary not-conjugate one.
    """
