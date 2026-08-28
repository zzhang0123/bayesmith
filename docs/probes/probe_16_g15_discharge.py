"""G15's discharge: is it the one line the plan says it is?

The plan (``2026-08-26-one-implementation.md`` G15) and the deferral's own
docstring both promised the discharge was ONE line -- flip
``include_prior=space is not None`` and delete ``uncertainty._prior_precision``,
because "the delegation is already written that way". This probe asks whether
that is true, and answers four questions the ruling turns on:

    (1) Does the plan's LITERAL one-liner work? (No: two other call sites each
        independently decided "no prior here", and both were right in
        isolation.)
    (2) When the delegation IS wired correctly, does the far side agree with
        the arithmetic being deleted -- and to what?
    (3) What happens to the five refusals the local spelling carried, three of
        which are pinned by name?
    (4) Do the two packages agree on WHICH SPELLINGS of a Gaussian prior are
        admissible?

(3) and (4) are the ones that matter. Both turn on the same fact about the
seam, and it is not a bug in either package: ``graph_bridge.translate`` files
bayesmith's ``NotGaussian`` as a BLAMELESS VERDICT -- caught and not
re-raised, left on the yielded ``Seam`` -- because a caller asking "is there an
exact route here?" in order to branch should not have to write ``except``
around a question. ``fisher_information(space=...)`` is not that caller.

Exit code is 0 whenever the probe finished, never a verdict -- same rule as
``probe_11_d17_dual_run.py``: a probe that turns its measurement into a process
status invites being read as a gate, and this is evidence for a ruling.

Run:  /Users/zzhang/projects/e-RHINO/.venv/bin/python docs/probes/probe_16_g15_discharge.py
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.graph_bridge import graph_for_information, translate
from rheplicant.inference.uncertainty import (
    _flat_forward,
    _named_spans,
    as_noise_model,
    fisher_information,
)
from bayesmith.diagnose.local import local_block
from bayesmith.exact.fisher import fisher_information as bm_fisher
from bayesmith.exact.gaussian import check_gaussian, precision_at

import bayesmith

DESIGN = jnp.array(np.linspace(1.0, 2.0, 12).reshape(4, 3), dtype=jnp.float32)
NOISE = 0.5
VALUES = {"a_vec": jnp.array([2.0, 3.0]), "z_scalar": jnp.array(1.0)}


def forward(params):
    return DESIGN @ jnp.concatenate(
        [params["a_vec"], jnp.atleast_1d(params["z_scalar"])]
    )


def space_with(scalar_prior, vec_prior=None):
    if vec_prior is None:
        vec_prior = dist.Normal(jnp.zeros(2), jnp.array([0.5, 0.25]))
    return ParameterSpace(
        latents=[
            Latent("a_vec", init=jnp.array([2.0, 3.0]), prior=vec_prior),
            Latent("z_scalar", init=jnp.array(1.0), prior=scalar_prior),
        ],
        bindings=[
            Bind("a_vec", into=lambda p: p["x"], fn=lambda v: v),
            Bind("z_scalar", into=lambda p: p["y"], fn=lambda v: v),
        ],
    )


def delegated(space, *, block_priors, graph_priors, include_prior):
    """The delegation with all THREE knobs the discharge turns, exposed.

    ``found`` starts as a sentinel so that a ``with`` block ended early by a
    swallowed verdict is visible as itself rather than as an
    ``UnboundLocalError`` from the line after.
    """
    _, _, prediction = _flat_forward(forward, VALUES)
    noise = as_noise_model(
        NOISE, None, prediction_shape=jnp.shape(prediction), caller="probe"
    )
    names, _, shapes = _named_spans(VALUES)
    block_names = tuple(sorted(names))
    values = {name: jnp.asarray(VALUES[name]) for name in block_names}
    declared = None
    if graph_priors:
        declared = {name: space.latent(name).prior for name in block_names}
    graph = graph_for_information(
        forward, values, noise, priors=declared, caller="probe"
    )
    found = "<<the `with` block ended early -- nothing was assigned>>"
    with translate("fisher_information") as seam:
        found = bm_fisher(
            local_block(graph, block_names, values, priors=block_priors),
            precision=precision_at(graph, values),
            include_prior=include_prior,
            depends_on_prediction=bool(noise.depends_on_prediction),
            precision_of=lambda moving: precision_at(graph, {**values, **moving}),
            centre={name: values[name] for name in block_names},
        )
    return seam, found


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------- (1)
rule("(1) The plan's LITERAL one-liner: flip include_prior, change nothing else")

good = space_with(dist.Normal(0.0, 2.0))
for label, block_priors, graph_priors in [
    ("block priors=False (today's local_block call), graph flat", False, False),
    ("block priors=True, graph still flat (declared=None)", True, False),
]:
    try:
        _, found = delegated(
            good, block_priors=block_priors, graph_priors=graph_priors,
            include_prior=True,
        )
        print(f"  {label}\n      -> {found if isinstance(found, str) else 'returned'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {label}\n      -> {type(exc).__name__}: {str(exc)[:120]}")
print(
    "\n  Both fail. `_bayesmith_fisher` builds the block WITHOUT priors and\n"
    "  `fisher_information` hardcodes `declared = None` in both branches, each\n"
    "  for its own good reason. Three sites, not one."
)

# ----------------------------------------------------------------- (2)
rule("(2) Wired correctly: does the far side reproduce the deleted arithmetic?")


def deleted_spelling(space):
    """The arithmetic ``_prior_precision`` used to do, written out here.

    **The oracle must not be the code under test.** Calling
    ``fisher_information(space=...)`` for the reference was how this section
    first read -- and once the delegation landed, both sides of the comparison
    were the same call, so the difference was 0.0 for a reason that had
    nothing to do with the two implementations agreeing. Kept as an
    independent spelling so this stays a comparison after the local one is
    gone: numpy, no jax, and the flat layout written out rather than derived.
    """
    names, spans, shapes = _named_spans(VALUES)
    size = sum(stop - start for start, stop in spans)
    precision = np.zeros(size)
    for name, (start, stop), shape in zip(names, spans, shapes, strict=True):
        prior = space.latent(name).prior
        base = getattr(prior, "base_dist", prior)
        scale = np.broadcast_to(np.asarray(base.scale, dtype=np.float64), shape)
        precision[start:stop] = 1.0 / np.ravel(scale) ** 2
    likelihood = np.asarray(
        fisher_information(forward, VALUES, noise_std=NOISE).matrix, dtype=np.float64
    )
    return likelihood + np.diag(precision)


here = deleted_spelling(good)
_, found = delegated(good, block_priors=True, graph_priors=True, include_prior=True)
there = np.asarray(found.values, dtype=np.float64)
diff = float(np.max(np.abs(here - there)))
scale = max(float(np.max(np.abs(here))), 1e-30)
print(f"  the deleted spelling, in numpy   diag = {np.round(np.diag(here), 5)}")
print(f"  the delegation                   diag = {np.round(np.diag(there), 5)}")
print(f"  max |difference| = {diff:.6e}    relative = {diff / scale:.6e}")
print(f"  kind: local spelling='posterior_precision'  far side={found.kind!r}")
print(
    "\n  Both reduce to diag(1/sigma^2) over the same spans, so this is the\n"
    "  arithmetic that can be deleted. The residue is float32 rounding in the\n"
    "  likelihood half, which BOTH sides take from the same call; the prior\n"
    "  half agrees exactly."
)

# ----------------------------------------------------------------- (3)
rule("(3) The five refusals, asked of the far side instead")

for label, space in [
    ("Uniform    (pinned: ParameterSpaceError naming the latent)",
     space_with(dist.Uniform(0.0, 3.0))),
    ("LogNormal  (pinned: ParameterSpaceError naming LogNormal)",
     space_with(dist.LogNormal(0.0, 1.0))),
    ("prior=None (pinned: ParameterSpaceError naming the latent)",
     space_with(None)),
]:
    seam, found = delegated(
        space, block_priors=True, graph_priors=True, include_prior=True
    )
    ended_early = isinstance(found, str)
    verdict = type(seam.blameless).__name__ if seam.blameless else "none"
    print(f"  {label}")
    print(f"      seam.blameless = {verdict};  block ended early = {ended_early}")
print(
    "\n  No exception leaves the seam. `translate` files NotGaussian as a\n"
    "  blameless verdict by design, so the caller's next line reads `.values`\n"
    "  off a name never assigned: UnboundLocalError where a named refusal was\n"
    "  promised. The remaining two -- a declared `joint_prior` and an unnamed\n"
    "  params pytree -- have no counterpart on the far side AT ALL."
)

# ----------------------------------------------------------------- (4)
rule("(4) Do the two packages admit the same spellings of a Gaussian?")


def bayesmith_admits(prior, shape):
    def model(data):
        ref = bayesmith.sample("a", lambda: prior)
        mu = bayesmith.det("mu", lambda v: v * 1.0, ref)
        bayesmith.observe("obs", lambda m: dist.Normal(m, 0.5), mu, obs=data)

    graph = bayesmith.trace(model, jnp.zeros(shape))
    node = next(n for n in graph.nodes if n.name == "a")
    try:
        check_gaussian(graph, node, {})
        return "accept"
    except Exception as exc:  # noqa: BLE001
        return f"refuse ({type(exc).__name__})"


def rheplicant_admits(prior):
    from rheplicant.inference.linear import _gaussian_parameters

    return "accept" if _gaussian_parameters(prior) is not None else "refuse"


print(f"  {'spelling':<34} {'rheplicant':<12} bayesmith")
for label, prior in [
    ("Normal(zeros(2), full(2, .5))", dist.Normal(jnp.zeros(2), jnp.full(2, 0.5))),
    ("Normal(0, .5).expand([2])", dist.Normal(0.0, 0.5).expand([2])),
    ("Normal(zeros(2), .5).to_event(1)",
     dist.Normal(jnp.zeros(2), jnp.full(2, 0.5)).to_event(1)),
    ("LogNormal(0, 1)", dist.LogNormal(0.0, 1.0)),
]:
    print(f"  {label:<34} {rheplicant_admits(prior):<12} "
          f"{bayesmith_admits(prior, (2,))}")
print(
    "\n  They differ on exactly one: `.expand([2])`. rheplicant unwraps it\n"
    "  (an ExpandedDistribution only re-shapes a base distribution);\n"
    "  bayesmith refuses it. Neither is wrong -- but routed through a seam\n"
    "  that swallows the 'no', the disagreement reaches a user as an\n"
    "  UnboundLocalError. Hence the canonicalisation: what crosses is\n"
    "  Normal(loc, scale) broadcast to the latent's shape, one form for all.\n"
    "\n  A scalar Normal(0, .5) on a (2,) latent -- the shape that reached\n"
    "  bayesmith's G9 broadcast defect -- cannot arrive from a ParameterSpace\n"
    "  at all: Latent.__check_init__ refuses the shape mismatch by name at\n"
    "  construction. Measured, and it is why that defect stays out of reach\n"
    "  even now that the facade's prior path is live."
)
