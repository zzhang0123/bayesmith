"""Exception family for bayesmith.

Deliberately stdlib-only: this module is imported by everything, including
from contexts that must not pay for jax, so it may never import numpy or jax.

Every concrete class also derives from the closest builtin, so generic
handlers keep working.

**A refusal that a caller must act on carries its evidence as ATTRIBUTES, not
only in its message.** The three that do -- :class:`AffinityRefused`,
:class:`NotGaussian`, :class:`NotLogLinear` -- were message-only until the
payload was added, and the defect that shape produces is worth stating: the
branch with something to report was the only branch with nothing readable,
because the passing branch returned its numbers structured while the failing
one rendered them into a sentence and dropped them. A consumer that wanted
them had to parse prose.

Their ``reason``/payload keywords are REQUIRED, not optional. An optional
discriminator is one that eventually gets omitted, and the caller then needs
a prose fallback -- which is the thing being removed. ``test_errors.py`` scans
the source for raise sites that skip it.

Payload-carrying exceptions define ``__reduce__`` through :func:`_rebuild`.
Python's default reconstructs an exception by calling ``cls(*args)``, which
cannot work once construction requires keywords -- the failure shows up far
from here, when something pickles an exception across a process boundary
(pytest-xdist does exactly that), as a TypeError about missing arguments
rather than as the error being reported.
"""


class BayesmithError(Exception):
    """Base class for every error bayesmith raises."""


def _rebuild(cls: type, args: tuple, state: dict) -> BaseException:
    """Reconstruct a payload-carrying exception without calling ``__init__``.

    The target of every ``__reduce__`` below. Bypassing ``__init__`` is the
    point: these classes validate their keywords, and an unpickle is a
    restoration rather than a fresh assertion -- re-running the validation
    there would turn a transport step into a second opportunity to fail.
    """
    obj = cls.__new__(cls)
    Exception.__init__(obj, *args)
    obj.__dict__.update(state)
    return obj


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


class AffinityRefused(StructureError):
    """A declared ``linear_in`` was probed and the prediction is not affine.

    A SUBCLASS of :class:`StructureError`, so every ``except StructureError``
    already written keeps catching it and the verdict is unchanged. What the
    subclass buys is the same two things rheplicant's ``LinearityRefused``
    buys on its side of the seam: a NARROW catch -- "the affinity claim is
    false", told apart from the rest of the structural family -- and the
    probe's numbers as data rather than as a rendered sentence.

    The numbers matter because the TREND across scales is the diagnostic:
    "departs at 1x and 1000x but not at 0.001x" is a different fault from
    "departs everywhere", and only the second is usually a wrong declaration.
    Passing probes are included for that reason -- a payload holding only the
    failures cannot show a trend.

    Two criteria are reported because the guard is a disjunction, and one
    number against one threshold is unreadable half the time: a reader sees a
    value under its tolerance printed beside the refusal and concludes the
    guard is broken, when the other criterion is what fired.

    Attributes:
        names: the latents the affinity claim was made about, in declaration
            order. More than one means the claim was JOINT -- each latent may
            well be affine alone, which is why it is not checked one at a
            time.
        at: where the probe was evaluated, as the caller spelled it.
        errors: ``{scale: relative departure}`` at every probed scale,
            passing probes included.
        weighted: ``{scale: departure in units of the noise sigma}``, same
            keys. bayesmith's second criterion; rheplicant's probe has no
            counterpart, so a translator reads ``errors`` and leaves this.
        rtol: the relative tolerance actually used -- derived from the
            prediction's dtype when the caller passed none, so it is a
            measurement of the run and not an echo of the argument.
        weighted_rtol: the threshold the sigma-weighted criterion used.
        failed: the scales that exceeded either tolerance, ascending; a
            subset of ``errors``' keys.

    A value in ``errors`` may be an ``Unresolved`` float -- a departure the
    per-element roundoff floor declined to judge, NOT one measured as zero.
    It is a ``float`` subclass, so arithmetic on the payload works either way
    and a consumer that cares can ask ``isinstance``. Nothing is coerced here,
    because coercing would destroy the distinction silently.
    """

    def __init__(
        self,
        *args: object,
        names: "tuple[str, ...]",
        at: str,
        errors: "dict[float, float]",
        weighted: "dict[float, float]",
        rtol: float,
        weighted_rtol: float,
        failed: "tuple[float, ...]",
    ) -> None:
        super().__init__(*args)
        self.names = tuple(names)
        self.at = str(at)
        # Copied rather than aliased: the caller's mapping is the same object
        # the probe returns on the passing branch, and an exception sharing
        # mutable state with its raiser is a trap for whoever catches it.
        self.errors = dict(errors)
        self.weighted = dict(weighted)
        self.rtol = float(rtol)
        self.weighted_rtol = float(weighted_rtol)
        self.failed = tuple(failed)

    def __reduce__(self):
        return (_rebuild, (type(self), self.args, self.__dict__))


class ConvergenceError(BayesmithError, RuntimeError):
    """An iterative procedure did not reach the accuracy it was asked for.

    For guards *inside* a jitted solve the mechanism is ``equinox.error_if``
    instead, because a Python ``if`` cannot branch on a traced value. This
    class is for the checks that run on concrete values, outside any trace --
    ``iterative_gls``'s ``converged=False`` promoted to a hard failure, say.

    No P3a call site actually raises this: ``iterative_gls`` returns
    ``converged`` as a field instead, by design, and leaves the promotion to
    its caller. P3b's ``estimate()`` is the expected first one.
    """


#: The reasons a node can fail to be a diagonal Gaussian, as data.
#:
#: A closed set, checked at construction. Written here rather than at the
#: raise sites so that the classifier, the tests and any consumer read ONE
#: spelling -- a second copy is what goes stale.
NOT_GAUSSIAN_REASONS: frozenset[str] = frozenset(
    {
        # the node's distribution is not a diagonal Normal at all
        "not_normal",
        # block members are ancestors of one another, so however Normal each
        # is alone, the pair is not JOINTLY Gaussian
        "jointly_dependent",
        # a latent has no loc, so "every latent at the centre of its prior"
        # -- what hoisting sigma out of the sweep needs -- has no value here
        "no_centre",
    }
)

#: The reasons no log-linear route exists, as data. Closed, checked at
#: construction, and the same one-spelling argument as above.
NOT_LOG_LINEAR_REASONS: frozenset[str] = frozenset(
    {
        # the DATA has non-positive or non-finite values: no log to take
        "data_not_positive",
        # the observed node returns neither a Normal nor a LogNormal
        "not_gaussian_family",
        # the prediction does not move with the latents, so whether the scale
        # tracks it cannot be measured
        "prediction_static",
        # the PREDICTION is not strictly positive where inference would run
        "prediction_not_positive",
        # the scale is constant while the prediction moves: additive noise,
        # which log space would restate rather than simplify
        "noise_additive",
        # the scale moves, but not proportionally to the prediction: neither
        # additive nor multiplicative
        "noise_neither",
        # multiplicative, but above the fractional level at which the
        # first-order log-space equivalence still holds
        "fractional_too_large",
        # graph-level: no observed node had a log-Gaussian reading
        "no_node_qualifies",
        # log(prediction) is not affine in the latents asked about
        "log_not_affine",
    }
)


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

    Attributes:
        reason: which of :data:`NOT_GAUSSIAN_REASONS` applies. Required,
            because a dispatcher's next move differs by reason and reading it
            out of the message means parsing prose.
        node: the node or latent the verdict is about, or ``None`` where the
            verdict is about the graph rather than one node.
        found: the distribution type actually returned, where one was --
            ``"Gamma"``, ``"MultivariateNormal"``. ``None`` for the reasons
            that are not about a type at all.
    """

    def __init__(
        self,
        *args: object,
        reason: str,
        node: "str | None" = None,
        found: "str | None" = None,
    ) -> None:
        super().__init__(*args)
        if reason not in NOT_GAUSSIAN_REASONS:
            raise ValueError(
                f"unknown NotGaussian reason {reason!r}; the vocabulary is "
                f"{sorted(NOT_GAUSSIAN_REASONS)}. A reason that is not in the "
                "set is a typo the consumer would silently route as 'some "
                "other reason', so it is refused where it is written."
            )
        self.reason = reason
        self.node = node
        self.found = found

    def __reduce__(self):
        return (_rebuild, (type(self), self.args, self.__dict__))


class NotLogLinear(BayesmithError, TypeError):
    """No log-linear route exists here -- for one of a NAMED set of reasons.

    Purely descriptive and deliberately blameless, exactly as
    :class:`NotGaussian` is: most models have no log-linear structure, and a
    dispatcher that asked and heard "no" routes the block to NUTS and moves
    on. The message always says WHICH reason, because they call for different
    responses:

    * the observation's noise is additive, so log space would not simplify
      anything -- it would just state a different likelihood;
    * the noise is multiplicative but its fractional level exceeds
      :data:`~bayesmith.exact.loglinear.FIRST_ORDER_MAX_FRACTIONAL`, so the
      first-order log-space equivalence is no longer good to the tolerance
      this package promises;
    * the prediction, or the data, is not strictly positive, so there is no
      log to take -- and ``log`` of a non-positive value is a NaN that would
      READ AS PASSING every ``departure > rtol`` comparison downstream, which
      is why this refuses eagerly instead;
    * ``log`` of the prediction is not affine in the latents asked about --
      the log-space counterpart of an ordinary linearity refusal.

    The same MRO argument as :class:`NotGaussian`'s applies against making
    this a subclass of :class:`StructureError`: a dispatcher probing for
    log-linearity must be able to catch "no" without also swallowing "your
    graph is broken".

    The named set above is :data:`NOT_LOG_LINEAR_REASONS`, and ``reason``
    carries it as data. The message still says which, for a human; the field
    is what a dispatcher reads, because "the fractional level is too high"
    invites a different response (loosen the model, or accept NUTS) from "the
    data is not positive" (the model is wrong about its own observable).

    Attributes:
        reason: which of :data:`NOT_LOG_LINEAR_REASONS` applies. Required.
        node: the observed node this is about, or ``None`` where no node in
            the graph qualified and the verdict is about the graph.
        found: the distribution type actually returned, where the reason is
            about a type; ``None`` otherwise.
        fractional: the measured fractional noise level, present only for
            ``"fractional_too_large"`` -- the number a caller needs to decide
            whether the model or the ceiling is the thing to revisit.
        per_node: ``{node: reason}`` for ``"no_node_qualifies"`` -- each
            value is itself from this vocabulary, so a graph-level refusal
            can be read node by node WITHOUT parsing the sentences, which are
            in the message. A caller deciding "fix one node, or give up on
            log space" needs exactly this.
    """

    def __init__(
        self,
        *args: object,
        reason: str,
        node: "str | None" = None,
        found: "str | None" = None,
        fractional: "float | None" = None,
        per_node: "dict[str, str] | None" = None,
    ) -> None:
        super().__init__(*args)
        if reason not in NOT_LOG_LINEAR_REASONS:
            raise ValueError(
                f"unknown NotLogLinear reason {reason!r}; the vocabulary is "
                f"{sorted(NOT_LOG_LINEAR_REASONS)}. A reason that is not in "
                "the set is a typo the consumer would silently route as "
                "'some other reason', so it is refused where it is written."
            )
        self.reason = reason
        self.node = node
        self.found = found
        self.fractional = None if fractional is None else float(fractional)
        self.per_node = None if per_node is None else dict(per_node)

    def __reduce__(self):
        return (_rebuild, (type(self), self.args, self.__dict__))
