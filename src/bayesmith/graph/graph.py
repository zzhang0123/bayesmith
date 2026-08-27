"""The graph container, and the structural rules it refuses to violate.

Declaration order **is** topological order. A node may only name parents
already declared, which is automatic when the graph comes from ``trace`` --
you cannot pass a ``NodeRef`` you have not created -- and is checked here so
that a hand-built graph obeys the same rule.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.errors import GraphError
from bayesmith.graph.nodes import Const, Deterministic, Node, Probabilistic


class Plate(eqx.Module):
    """A named repetition axis. ``size`` is fixed at construction."""

    name: str = eqx.field(static=True)
    size: int = eqx.field(static=True)


class Graph(eqx.Module):
    """A static DAG of nodes in topological order, plus its plates.

    Attributes:
        nodes: every node, in topological order.
        plates: the repetition axes the nodes may live inside.
        joint_prior: a density over SEVERAL latents at once, or ``None``. A
            latent's own ``dist_fn`` says what one quantity is a priori and is
            the only per-latent prior the rest of the package needs; a Jeffreys
            prior is not of that shape -- it is one density over a named block,
            a function of the forward model and the noise rather than of the
            latents alone. Declaring it HERE is what lets
            :func:`~bayesmith.graph.evaluate.log_joint` and
            :func:`~bayesmith.bridge.numpyro_bridge.to_numpyro` both find it,
            so the two scans of one graph cannot target different posteriors.

            Structural only at this layer: this module is the core and imports
            nothing from ``diagnose``, so what is checked here is that the
            object answers ``over`` and ``log_density`` and that its block
            names latents this graph declares. Everything else --
            identifiability, the double-prior refusal -- belongs to
            :class:`~bayesmith.diagnose.priors.JeffreysPrior` and fires where
            it is evaluated.
    """

    nodes: tuple[Node, ...]
    plates: tuple[Plate, ...]
    joint_prior: Any = None

    def __check_init__(self) -> None:
        plate_names: set[str] = set()
        for p in self.plates:
            if p.name in plate_names:
                raise GraphError(
                    f"duplicate plate name {p.name!r}: plate names must be "
                    "unique within a graph."
                )
            plate_names.add(p.name)

        seen: set[str] = set()
        for node in self.nodes:
            if node.name in seen:
                raise GraphError(
                    f"duplicate node name {node.name!r}: node names must be "
                    "unique within a graph."
                )
            for parent in node.parents:
                if parent not in seen:
                    raise GraphError(
                        f"node {node.name!r} names parent {parent!r}, which is not "
                        "declared before it. Declaration order is topological "
                        "order: declare the parent first."
                    )
            if isinstance(node, Deterministic):
                extra = set(node.linear_in) - set(node.parents)
                if extra:
                    raise GraphError(
                        f"node {node.name!r} declares linear_in={node.linear_in!r}, "
                        f"which names {sorted(extra)} -- not a parent of this node. "
                        f"Parents are {node.parents!r}. linear_in is a claim about "
                        "the model, not a hint -- it decides whether an exact "
                        "conjugate solve may be used, so it must at least name "
                        "declared parents before anything checks whether the claim "
                        "is true."
                    )
            if isinstance(node, Const) and jnp.issubdtype(
                jnp.asarray(node.value).dtype, jax.dtypes.prng_key
            ):
                raise GraphError(
                    f"node {node.name!r} is a Const holding a PRNG key. A "
                    "deterministic node that consumes one draws randomness "
                    "inside the forward model, and inference closes the model "
                    "over ONE evaluation: the draw is made once and the same "
                    "frozen field rides every prediction compared against the "
                    "data. Nothing downstream can tell -- adding a constant "
                    "field is exactly affine, so check_linearity sees a "
                    "departure of 0 and identifiability reports full rank. "
                    "Upstream measured 10.6 sigma of bias with BOTH exits "
                    "reporting the same error bar to every digit.\n\n"
                    "Randomness in a graph belongs to a probabilistic node, "
                    "which carries a density: `sample(...)` gives the same "
                    "field a log_prob, so it enters the joint distribution "
                    "instead of hiding in the mean. That is the rule this "
                    "refusal enforces -- a random node without a density "
                    "cannot enter the joint.\n\n"
                    "Blind spot, stated rather than implied: a `fn` that "
                    "closes over a draw, or a legacy raw-uint32 key, is "
                    "invisible here, exactly as an operator that draws "
                    "without declaring it is invisible to rheplicant's "
                    "`refuse_stochastic_stages`. There is no numerical "
                    "symptom to find; the declaration is the whole signal."
                )
            if isinstance(node, Probabilistic) and node.observed_mask is not None:
                mask = jnp.asarray(node.observed_mask)
                if node.observed is None:
                    raise GraphError(
                        f"node {node.name!r} is LATENT and declares an "
                        "observed_mask. A mask says which of this node's data "
                        "were actually taken, and a latent node has no data -- "
                        "it is what the graph is solving for. Masking a latent "
                        "would have to mean dropping degrees of freedom, which "
                        "is a different model, declared by not sampling them."
                    )
                if mask.dtype != jnp.bool_:
                    raise GraphError(
                        f"node {node.name!r} declares an observed_mask of "
                        f"dtype {mask.dtype}; it must be boolean. A float mask "
                        "multiplies where a boolean one selects, so a 0.5 "
                        "would halve a sample's weight and call it unobserved."
                    )
                if mask.shape != jnp.shape(node.observed):
                    raise GraphError(
                        f"node {node.name!r} declares an observed_mask of "
                        f"shape {mask.shape} but its data is "
                        f"{jnp.shape(node.observed)}. Broadcasting these would "
                        "mask a different set of samples than the caller "
                        "named, and every shape downstream would still be "
                        "right."
                    )
            if len(node.plate) > 1:
                raise GraphError(
                    f"node {node.name!r} is in plates {node.plate}: nested plates "
                    "are not supported yet. Flatten the two axes into one plate."
                )
            for plate in node.plate:
                if plate not in plate_names:
                    raise GraphError(
                        f"node {node.name!r} is in plate {plate!r}, which the graph "
                        f"does not declare. Known plates: {sorted(plate_names)}."
                    )
            seen.add(node.name)

        if self.joint_prior is not None:
            self._check_joint_prior(seen)

    def _check_joint_prior(self, latents: set[str]) -> None:
        """Structural checks on a declared joint prior, at construction.

        Earlier than the prior's own ``_check_against``, which needs values and
        so cannot run until the potential does. A block naming a node that does
        not exist is a typo, and a typo should not survive until the first
        leapfrog step.
        """
        prior = self.joint_prior
        for attribute in ("over", "log_density"):
            if not hasattr(prior, attribute):
                raise GraphError(
                    f"joint_prior is a {type(prior).__name__}, which has no "
                    f"{attribute!r}. A graph-level prior has to say which "
                    "latents it is over and be able to evaluate itself at a "
                    "values dict -- those two are what log_joint and "
                    "to_numpyro call. JeffreysPrior is the one this package "
                    "ships."
                )
        declared = {
            name
            for name in latents
            if any(
                n.name == name and isinstance(n, Probabilistic) and n.is_latent
                for n in self.nodes
            )
        }
        unknown = [name for name in prior.over if name not in declared]
        if unknown:
            raise GraphError(
                f"joint_prior is over {list(prior.over)}, and {unknown} "
                f"are not latents of this graph; its latents are "
                f"{list(self.latents)}. The block would be assembled from the "
                "names that DO match, which is a prior over a smaller block -- "
                "a different density, and not a marginal of the declared one."
            )

    @property
    def names(self) -> tuple[str, ...]:
        """Every node name, in topological order."""
        return tuple(node.name for node in self.nodes)

    @property
    def latents(self) -> tuple[str, ...]:
        """Names of the probabilistic nodes that are inferred."""
        return tuple(
            n.name for n in self.nodes if isinstance(n, Probabilistic) and n.is_latent
        )

    @property
    def observed(self) -> tuple[str, ...]:
        """Names of the probabilistic nodes that are conditioned on."""
        return tuple(
            n.name
            for n in self.nodes
            if isinstance(n, Probabilistic) and not n.is_latent
        )

    def node(self, name: str) -> Node:
        """The node called ``name``.

        Raises:
            GraphError: if no node has that name.
        """
        for node in self.nodes:
            if node.name == name:
                return node
        raise GraphError(f"no node named {name!r}. Known: {list(self.names)}.")

    def plate_size(self, name: str) -> int:
        """The declared size of plate ``name``.

        Raises:
            GraphError: if no plate has that name.
        """
        for plate in self.plates:
            if plate.name == name:
                return plate.size
        raise GraphError(
            f"no plate named {name!r}. Known: {[p.name for p in self.plates]}."
        )
