"""Amortized posterior estimation -- inference that never writes a likelihood.

Every other route in this package evaluates a density. NUTS does it once per
leapfrog step; :func:`~bayesmith.exact.solve.wiener_solve` exploits its
conjugate form; :func:`~bayesmith.optimize.fit` descends it. Simulation-based
inference evaluates none at all. It is handed pairs ``(theta, x)`` drawn from
the joint, fits a conditional density ``q(theta | x)`` to them, and reads the
posterior off ``q`` at the observation actually in hand.

Two properties follow, and they are why this lives beside the exact solvers
rather than instead of them:

* **Amortized.** The training cost is paid once. A new observation costs a
  forward pass -- no chain, no burn-in, no re-solve. Over many observations
  that is the difference between hours and milliseconds.
* **Likelihood-free.** Nothing here needs the noise to be Gaussian, the
  forward model to be differentiable, or a normalization to be tractable.

The price is that ``q`` has **no internal notion of being wrong**. A
badly-trained estimator returns a smooth, confident, incorrect distribution
and reports nothing amiss. So this module is built to be checkable: on a
linear-Gaussian problem the exact posterior is available in closed form, and
this package's tests hold the estimator to it. Validate on a case you can
solve before trusting one you cannot.

**The simulator is deliberately NOT here, and that is a ruling rather than an
omission** (D10(2) and D42 in the migration ledger). Drawing ``(theta, x)``
means generating data, and a generative law is not the density this package's
graphs carry: for a multiplicative instrument model the two differ by an
absolute value and a floor, so a node's ``dist_fn`` pressed into service as a
simulator would silently swap one law for the other. Generation belongs to
whoever owns the noise physics. This module's contract is a bank of arrays,
which is also why every entry point here is array-level and takes no
:class:`~bayesmith.graph.graph.Graph`.

**Single precision is accepted, and that too is deliberate** -- unlike
:mod:`bayesmith.exact.reduced_basis` (D41) or
:mod:`bayesmith.diagnose.local`, which refuse ambient float32 because a
rank verdict or a Gram matrix sits underneath the rounding. Nothing here
returns a verdict. An approximate density fitted by a network is limited by
the fit long before it is limited by the arithmetic, and float32 is the
precision such a network is normally trained in.

The density is a **conditional Gaussian mixture** -- an MLP mapping a summary
of the data to the weights, means and scales of a mixture over the latent
vector. A normalizing flow is more expressive; a mixture is a few dozen lines,
is exact for a Gaussian posterior at one component, and keeps the failure
modes legible. Adam is hand-rolled here for the same reason it is in
:mod:`bayesmith.optimize`: no optax dependency.

Usage::

    q = NeuralPosterior.create(thetas, bank, key=jax.random.key(1))
    q, losses = train_posterior(q, thetas, bank, key=jax.random.key(2))
    draws = q.sample(observed, key=jax.random.key(3), n_samples=4000)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp

from bayesmith.errors import StructureError

__all__ = [
    "MIN_SCALE",
    "NeuralPosterior",
    "TrainingHistory",
    "train_posterior",
]

#: Floor on a mixture component's scale, as a fraction of the standardized
#: latent's unit width. Without it a component can collapse onto a single
#: training point and take the log-density to infinity.
#:
#: Measured: the floor guards a LIMIT rather than a reachable value. With
#: ``min_scale=0.0`` the scale is still ``softplus(raw)``, which is strictly
#: positive, and a deliberately collapsible bank (eight distinct thetas, eight
#: components, 1500 steps) trained to a finite density either way. So a zero
#: floor is not refused -- there is no failure here to refuse. A NEGATIVE
#: floor is a different matter and is refused; see :meth:`NeuralPosterior.create`.
MIN_SCALE: float = 1e-3


def _standardize(values: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Per-feature mean and a strictly positive scale."""
    mean = jnp.mean(values, axis=0)
    scale = jnp.std(values, axis=0)
    return mean, jnp.where(scale > 0.0, scale, 1.0)


def _all_finite(tree: Any) -> jax.Array:
    """One boolean over every inexact leaf of ``tree``."""
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(True)
    return jnp.all(
        jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
    )


class NeuralPosterior(eqx.Module):
    """``q(theta | x)``: a conditional Gaussian mixture over the latent vector.

    An ``eqx.nn.MLP`` maps ``embed(x)`` to the mixture's log-weights, means and
    log-scales. Both ``theta`` and the embedded data are standardized using
    statistics taken from the training bank, which is why :meth:`create` needs
    the bank rather than only its shapes: an unstandardized network trained on
    instrument-scale data and unit-scale parameters does not converge, and
    reports that as a mediocre fit rather than as an error.

    Attributes:
        net: the MLP.
        embed: ``x -> feature vector`` for ONE datum (static). Defaults to
            ``jnp.ravel``. This is where a large observation is summarized and
            where uninformative samples are dropped.
        n_components: mixture components (static). One is exact for a Gaussian
            posterior, which is what makes the linear-Gaussian check sharp.
        n_params: length of the latent vector (static).
        theta_mean, theta_scale, data_mean, data_scale: standardization,
            derived from the bank.
        min_scale: floor on a component's standardized scale (static).
    """

    net: eqx.nn.MLP
    embed: Callable[[jax.Array], jax.Array] = eqx.field(static=True)
    n_components: int = eqx.field(static=True)
    n_params: int = eqx.field(static=True)
    theta_mean: jax.Array
    theta_scale: jax.Array
    data_mean: jax.Array
    data_scale: jax.Array
    min_scale: float = eqx.field(static=True, default=MIN_SCALE)

    @classmethod
    def create(
        cls,
        thetas: jax.Array,
        data: jax.Array,
        *,
        key: jax.Array,
        embed: Callable[[jax.Array], jax.Array] = jnp.ravel,
        n_components: int = 4,
        width: int = 64,
        depth: int = 3,
        min_scale: float = MIN_SCALE,
    ) -> NeuralPosterior:
        """Build an untrained estimator sized and standardized to a bank.

        Args:
            thetas: ``(n_simulations, n_params)``, the latent half of the bank.
            data: ``(n_simulations, *datum_shape)``, the observed half. The
                two halves must be the same pairs, in the same order.
            key: PRNG key for the network's initialization.
            embed: summary applied to ONE datum before the network sees it.
            n_components: mixture components.
            width, depth: the MLP's hidden size and hidden-layer count.
            min_scale: floor on a component's standardized scale.

        Raises:
            StructureError: if ``thetas`` is not a two-dimensional stack, if
                the two halves disagree on how many simulations there are, if
                ``n_components`` is not positive, or if ``min_scale`` is
                negative.
        """
        if thetas.ndim != 2:
            raise StructureError(
                f"thetas must be (n_simulations, n_params), got shape {thetas.shape}."
            )
        if thetas.shape[0] != data.shape[0]:
            raise StructureError(
                f"thetas has {thetas.shape[0]} simulations but data has "
                f"{data.shape[0]}; they must be the same pairs."
            )
        if not n_components >= 1:  # `not >=` so a NaN count is refused too
            raise StructureError(f"n_components must be positive, got {n_components}.")
        if not min_scale >= 0.0:  # likewise, and this one is measured below
            raise StructureError(
                f"min_scale must be non-negative, got {min_scale}. A negative "
                "floor is subtracted from a strictly positive softplus, so a "
                "component whose raw output is sufficiently negative gets a "
                "NEGATIVE scale -- and `log(scale)` in the density is then NaN "
                "for every query, while the network's own parameters stay "
                "perfectly finite. Measured: min_scale=-1.0 returns NaN from "
                "log_prob on an untrained estimator, and no guard downstream "
                "sees it, because there is nothing wrong with the estimator. "
                "Zero IS allowed: softplus never reaches it."
            )
        features = jax.vmap(embed)(data)
        n_params = thetas.shape[1]
        theta_mean, theta_scale = _standardize(thetas)
        data_mean, data_scale = _standardize(features)
        net = eqx.nn.MLP(
            in_size=features.shape[1],
            out_size=n_components * (1 + 2 * n_params),
            width_size=width,
            depth=depth,
            key=key,
        )
        return cls(
            net=net,
            embed=embed,
            n_components=n_components,
            n_params=n_params,
            theta_mean=theta_mean,
            theta_scale=theta_scale,
            data_mean=data_mean,
            data_scale=data_scale,
            min_scale=min_scale,
        )

    def _mixture(self, datum: jax.Array):
        """``(log_weights, means, scales)`` in STANDARDIZED latent space."""
        features = (self.embed(datum) - self.data_mean) / self.data_scale
        raw = self.net(features)
        k, d = self.n_components, self.n_params
        logits = raw[:k]
        means = raw[k : k + k * d].reshape(k, d)
        # softplus, not exp: a large positive output cannot overflow the scale,
        # and the floor keeps a component from collapsing onto one training
        # point and taking the log-density to infinity.
        scales = jax.nn.softplus(raw[k + k * d :].reshape(k, d)) + self.min_scale
        return jax.nn.log_softmax(logits), means, scales

    def log_prob(self, theta: jax.Array, datum: jax.Array) -> jax.Array:
        """``log q(theta | x)`` for one pair, in the latent's own units."""
        log_weights, means, scales = self._mixture(datum)
        z = (jnp.ravel(theta) - self.theta_mean) / self.theta_scale
        per_component = jnp.sum(
            -0.5 * ((z - means) / scales) ** 2
            - jnp.log(scales)
            - 0.5 * jnp.log(2.0 * jnp.pi),
            axis=-1,
        )
        # The standardization is a change of variables, so its Jacobian belongs
        # in the density: without it log_prob is off by a constant that depends
        # on the training bank, which is exactly the kind of error that trains
        # away invisibly and then breaks any comparison against a real density.
        return jax.nn.logsumexp(log_weights + per_component) - jnp.sum(
            jnp.log(self.theta_scale)
        )

    def sample(self, datum: jax.Array, key: jax.Array, n_samples: int) -> jax.Array:
        """Draw ``(n_samples, n_params)`` from ``q(theta | x)``."""
        log_weights, means, scales = self._mixture(datum)

        def one(subkey):
            pick_key, draw_key = jax.random.split(subkey)
            component = jax.random.categorical(pick_key, log_weights)
            z = means[component] + scales[component] * jax.random.normal(
                draw_key, (self.n_params,)
            )
            return self.theta_mean + self.theta_scale * z

        return jax.vmap(one)(jax.random.split(key, n_samples))


class TrainingHistory(NamedTuple):
    """Per-step losses from :func:`train_posterior`.

    * ``train`` -- mean negative log-density on the minibatch, ``(n_steps,)``.
    * ``validation`` -- the same on the held-out split, ``(n_steps,)``, or an
      empty array when ``validation_fraction`` is zero.
    * ``best_step`` -- the step whose validation loss was lowest, and the
      parameters that were returned. **Zero means no step ever improved on the
      estimator that was passed in**, which is what a run that went non-finite
      immediately looks like; the returned estimator is then the input.

    **The validation curve is the only instrument that reports over-fitting.**
    The training loss falls monotonically past the point where the fit stops
    being a posterior; what over-fitting does to ``q`` is make it too NARROW,
    so the failure presents as an unusually confident answer rather than a
    visibly bad one.

    Non-finite entries are left in place rather than scrubbed. A diverged run
    is a fact about the run, and the only record of it is here -- the returned
    estimator, having been selected on the held-out split, is usually finite
    and says nothing.
    """

    train: jax.Array
    validation: jax.Array
    best_step: jax.Array


def train_posterior(
    posterior: NeuralPosterior,
    thetas: jax.Array,
    data: jax.Array,
    *,
    key: jax.Array,
    n_steps: int = 3000,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[NeuralPosterior, TrainingHistory]:
    """Maximize the mean ``log q(theta | x)`` over the simulation bank.

    That objective is what makes ``q`` a posterior rather than a fit to
    anything else: its population optimum is the true ``p(theta | x)``, because
    the pairs are drawn from the joint.

    Adam, hand-rolled, matching :func:`~bayesmith.optimize.minimize` -- this
    package does not depend on optax.

    Args:
        posterior: from :meth:`NeuralPosterior.create` (or a partly trained one;
            training resumes from wherever it is).
        thetas, data: the bank.
        key: PRNG key for the split and the minibatching.
        n_steps: Adam steps.
        batch_size: simulations per step; capped at the training split.
        learning_rate, beta1, beta2, eps: Adam settings.
        validation_fraction: share of the bank held out. **Defaults to 0.1, and
            the returned estimator is the one from the best validation step,
            not the last one** -- see the note below. Set to ``0.0`` to train on
            everything and return the final parameters, which is the faster and
            more dangerous option.

    Returns:
        ``(posterior, history)`` -- the estimator at its best validation step and
        a :class:`TrainingHistory`.

    Raises:
        StructureError: for a non-positive step count, a ``validation_fraction``
            outside ``[0, 1)``, or a fraction that holds out zero simulations.
        Exception: from ``eqx.error_if`` if the returned estimator is not
            finite. See the second note.

    Note:
        **Over-fitting an NPE makes it over-confident, which is the failure that
        does not look like one.** Measured here, on the linear-Gaussian problem
        this module's tests use (exact posterior width 0.134): a bank of 512
        with four components, run 8000 steps with ``validation_fraction=0.0``,
        returns a posterior **0.271** of the exact width while its training loss
        is still improving. Nothing about that density looks wrong -- it is
        smooth, it integrates to one, and it is centred correctly. The same run
        with a 20% split comes back at **0.844** of the exact width, returned
        from step **423** of 8000, and its validation loss goes from a minimum
        of -0.605 to +34.1 by the last step. Holding out a split and returning
        the best step is what turns the failure into something visible, and it
        is the default for that reason.

        **Steps alone do not cause it, and that correction is worth stating**
        because the upstream implementation this was migrated from carries the
        claim that 4000 steps costs a factor 0.60 in width. Re-measured here,
        that number is a property of the *bank*, not of the step count: with
        8192 simulations and four components, 4000 steps returns **1.002** of
        the exact width. Over-fitting arrives when capacity outruns the bank.

        Relatedly, prefer few components. A Gaussian posterior is exact at
        ``n_components=1``, and extra components mostly buy capacity to memorize
        the bank.

    Note:
        **A diverged run has two outcomes here, and only one of them is
        refused** (D43). Measured, at a rate three orders of magnitude above
        the working one:

        * with a held-out split, the best-validation selection keeps the last
          finite parameters, so the estimator that comes back is finite and
          usable. It is reported, not refused: ``best_step`` collapses to 1
          and ``history.validation`` carries the NaNs.
        * with ``validation_fraction=0.0`` there is no such selection, and the
          final parameters themselves are NaN. Every subsequent ``log_prob``
          and ``sample`` then returns NaN, shaped correctly, from an object
          that looks like any other estimator. **That is refused**, on the
          same reasoning as :func:`~bayesmith.optimize.minimize`: a NaN answer
          that arrives looking like an answer is the failure this package
          spends the most effort making impossible.

        Divergence does not always reach NaN -- at a rate a further thousandfold
        higher the same run came back finite with a hopeless loss. Nothing here
        refuses a bad fit, only an absent one.
    """
    if n_steps < 1:
        raise StructureError(f"n_steps must be positive, got {n_steps}.")
    if not 0.0 <= validation_fraction < 1.0:
        raise StructureError(
            f"validation_fraction must be in [0, 1), got {validation_fraction}."
        )
    n_bank = thetas.shape[0]
    n_validation = round(validation_fraction * n_bank)
    if validation_fraction > 0.0 and n_validation < 1:
        raise StructureError(
            f"validation_fraction={validation_fraction} holds out zero of "
            f"{n_bank} simulations. Enlarge the bank or pass 0.0 explicitly to "
            "train without a held-out split."
        )

    split_key, train_key = jax.random.split(key)
    shuffled = jax.random.permutation(split_key, n_bank)
    validation_index = shuffled[:n_validation]
    train_index = shuffled[n_validation:]
    batch = min(batch_size, int(train_index.shape[0]))

    params, static = eqx.partition(posterior, eqx.is_inexact_array)

    def loss(free: Any, index: jax.Array) -> jax.Array:
        model = eqx.combine(free, static)
        return -jnp.mean(jax.vmap(model.log_prob)(thetas[index], data[index]))

    zeros = jax.tree.map(jnp.zeros_like, params)

    def step(carry, step_key):
        current, m, v, count, best, best_loss, best_step = carry
        picked = train_index[
            jax.random.choice(step_key, train_index.shape[0], (batch,), replace=False)
        ]
        value, grads = jax.value_and_grad(loss)(current, picked)
        m = jax.tree.map(lambda a, g: beta1 * a + (1 - beta1) * g, m, grads)
        v = jax.tree.map(lambda a, g: beta2 * a + (1 - beta2) * g**2, v, grads)
        t = count + 1
        current = jax.tree.map(
            lambda p, mm, vv: p
            - learning_rate
            * (mm / (1 - beta1**t))
            / (jnp.sqrt(vv / (1 - beta2**t)) + eps),
            current,
            m,
            v,
        )
        if n_validation:
            held_out = loss(current, validation_index)
            improved = held_out < best_loss
            best = jax.tree.map(
                lambda old, new: jnp.where(improved, new, old), best, current
            )
            best_loss = jnp.where(improved, held_out, best_loss)
            best_step = jnp.where(improved, t, best_step)
        else:
            held_out = jnp.asarray(jnp.nan)
            best, best_step = current, t
        return (current, m, v, t, best, best_loss, best_step), (value, held_out)

    init = (params, zeros, zeros, 0, params, jnp.asarray(jnp.inf), jnp.asarray(0))
    (_, _, _, _, best, _, best_step), (train_losses, validation_losses) = jax.lax.scan(
        step, init, jax.random.split(train_key, n_steps)
    )
    # `eqx.error_if` rather than a Python `if`, so the guard also fires under
    # jit -- the same mechanism, and the same reason, as `minimize`'s (D33).
    # Attached to the PARAMETERS, because they are what is returned and what
    # every later query reads; an unused check can be optimised away.
    best = eqx.error_if(
        best,
        ~_all_finite(best),
        "this training run returned a non-finite estimator, so every log_prob "
        "and every sample it produces is NaN -- shaped correctly, from an "
        "object that looks like any other posterior. The usual cause is a "
        "learning rate above what the objective tolerates. Lower it, or pass "
        "validation_fraction > 0, whose best-step selection keeps the last "
        "finite parameters and records the divergence in history.validation "
        "instead of returning it.",
    )
    history = TrainingHistory(
        train=train_losses,
        validation=validation_losses if n_validation else jnp.zeros((0,)),
        best_step=best_step,
    )
    return eqx.combine(best, static), history
