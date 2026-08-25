"""Graphs the diagnose tests share.

Everything here is a BUILDER, not a module constant, and that is load-bearing:
the diagnostics refuse float32, so every array has to be created inside the
caller's ``with jax.enable_x64(True):`` block. A module-level constant would
be built once, at import time, in whatever precision the importing process
happens to have -- which is exactly the "array created outside the context"
trap the upstream handovers keep paying for.

The motivating pair (``free_graph``/``basis_graph``) is the graph port of
rheplicant's ``tests/inference/test_identifiability.py`` fixture:
``data[t, f] = gain[t] * (T_ant[t, f] + tone[f])`` on an 8x8 grid, with the
antenna temperature either free per cell or through a (3, 3) polynomial
basis. The coefficient matrix is deliberately not symmetric in i<->j, so a
family of transposition mistakes stays visible.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist

from bayesmith import det, observe, sample, trace

N_TIME, N_FREQ = 8, 8
TONE_CHANNEL, TONE_KELVIN = 3, 5000.0
NU0 = 70e6

#: sigma = RADIOMETER_F * |mu|: rheplicant's RadiometerNoise(channel_width=1e6,
#: integration_time=1.0) has fractional level 1/sqrt(1e6 * 1.0).
RADIOMETER_F = 1e-3


def poly_basis(n: int, degree: int) -> jnp.ndarray:
    x = jnp.linspace(-1.0, 1.0, n)
    return jnp.stack([x**k for k in range(degree)], axis=1)


def fixture_arrays():
    """The motivating pair's constants, built HERE so the dtype is the caller's."""
    time_basis = poly_basis(N_TIME, 3)
    freq_basis = poly_basis(N_FREQ, 3)
    coeff0 = jnp.array(
        [[3000.0, -180.0, 40.0], [120.0, 25.0, -8.0], [-45.0, 6.0, 2.0]]
    )
    t_ant0 = time_basis @ coeff0 @ freq_basis.T
    gain0 = 1.5 + 0.05 * jnp.arange(N_TIME, dtype=float)
    return time_basis, freq_basis, coeff0, t_ant0, gain0


def free_graph(tone_kelvin: float):
    """One free antenna temperature per (time, frequency) cell.

    72 parameters against 64 data points: the case whose null space only a
    ``full_matrices=True`` SVD returns whole.
    """
    _, _, _, t_ant0, gain0 = fixture_arrays()
    tone = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(tone_kelvin)
    data = gain0[:, None] * (t_ant0 + tone[None, :])

    def model():
        gain = sample("gain", lambda: dist.Normal(gain0, 0.1).to_event(1))
        t_ant = sample("t_ant", lambda: dist.Normal(t_ant0, 100.0).to_event(2))
        pred = det(
            "pred", lambda g, t: g[:, None] * (t + tone[None, :]), gain, t_ant
        )
        observe("d", lambda mu: dist.Normal(mu, 1.0).to_event(2), pred, obs=data)

    return trace(model)


def basis_graph(tone_kelvin: float):
    """The same antenna temperature through a (3, 3) time x frequency basis."""
    time_basis, freq_basis, coeff0, t_ant0, gain0 = fixture_arrays()
    tone = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(tone_kelvin)
    data = gain0[:, None] * (t_ant0 + tone[None, :])

    def model():
        gain = sample("gain", lambda: dist.Normal(gain0, 0.1).to_event(1))
        t_coeff = sample("t_coeff", lambda: dist.Normal(coeff0, 10.0).to_event(2))
        pred = det(
            "pred",
            lambda g, c: g[:, None] * (time_basis @ c @ freq_basis.T + tone[None, :]),
            gain,
            t_coeff,
        )
        observe("d", lambda mu: dist.Normal(mu, 1.0).to_event(2), pred, obs=data)

    return trace(model)


def mixed_scale_graph():
    """Two perfectly identified latents whose natural scales differ by 1e10.

    ``big`` drives a 1e5 K signal, ``small`` a 1e-5 K one, into DISJOINT
    frequency channels -- so the two are exactly orthogonal and the model is
    unambiguously of rank 2. What varies between them is only the unit.
    """
    data = jnp.zeros((N_TIME, N_FREQ)).at[:, 1].set(1e5).at[:, 6].set(1e-5)

    def model():
        big = sample("big", lambda: dist.Normal(1.0, 1.0))
        small = sample("small", lambda: dist.Normal(1.0, 1.0))
        pred = det(
            "pred",
            lambda b, s: jnp.broadcast_to(
                jnp.zeros(N_FREQ).at[1].set(1e5 * b).at[6].set(1e-5 * s),
                (N_TIME, N_FREQ),
            ),
            big,
            small,
        )
        observe("d", lambda mu: dist.Normal(mu, 1.0).to_event(2), pred, obs=data)

    return trace(model)


def zero_column_graph():
    """A live latent and a DEAD one, deliberately hard to confuse.

    ``live`` is ``(2,)`` and drives two channels at scales 1e5 apart;
    ``flat`` is ``(3,)`` and enters through ``b**2`` at ``b = 0``, so all
    three of its Jacobian columns are exactly zero. Different shapes AND
    different scales AND different sizes, because symmetric fixtures have
    blinded this test family before: a direction that carried the wrong
    latent, or the right one mis-scaled, must not pass by coincidence.
    """
    data = jnp.zeros((N_TIME, N_FREQ)).at[:, 1].set(1e3).at[:, 4].set(1e-2)

    def model():
        live = sample("live", lambda: dist.Normal(jnp.ones(2), 1.0).to_event(1))
        flat = sample("flat", lambda: dist.Normal(jnp.zeros(3), 1.0).to_event(1))
        pred = det(
            "pred",
            lambda a, b: jnp.broadcast_to(
                jnp.zeros(N_FREQ)
                .at[1]
                .set(1e3 * a[0])
                .at[4]
                .set(1e-2 * a[1])
                .at[2]
                .set(b[0] ** 2)
                .at[5]
                .set(b[1] ** 2)
                .at[7]
                .set(b[2] ** 2),
                (N_TIME, N_FREQ),
            ),
            live,
            flat,
        )
        observe("d", lambda mu: dist.Normal(mu, 1.0).to_event(2), pred, obs=data)

    return trace(model)


def sort_trap_graph():
    """Declaration order that differs from sorted order, with the degeneracy
    between the two ENDS of the declaration and different ends of the sorted
    order, while the middle latent carries none of it.

    ``(sky_scale + load_scale)`` multiplies the whole map, so those two are
    one parameter twice; ``tone_amps`` drives two channels independently.
    """
    _, _, _, t_ant0, _ = fixture_arrays()
    data = t_ant0

    def model():
        sky = sample("sky_scale", lambda: dist.Normal(1.0, 1.0))
        amps = sample(
            "tone_amps", lambda: dist.Normal(jnp.array([10.0, -4.0]), 1.0).to_event(1)
        )
        load = sample("load_scale", lambda: dist.Normal(0.0, 1.0))
        pred = det(
            "pred",
            lambda s, a, load_: (s + load_) * t_ant0
            + jnp.zeros(N_FREQ).at[2].set(a[0]).at[5].set(a[1])[None, :],
            sky,
            amps,
            load,
        )
        observe("d", lambda mu: dist.Normal(mu, 1.0).to_event(2), pred, obs=data)

    return trace(model)


def power_law_graph(
    *,
    floor: float = 0.0,
    noise: str = "radiometer",
    sigma0: float = 0.5,
    flat_latents: bool = True,
    prior_widths: tuple[float, float] = (0.5, 0.3),
    depends_on_prediction: bool | None = None,
):
    """``mu = A (nu/nu0)^-beta + floor`` over (log A, beta), 8x8, data = truth.

    ``flat_latents=True`` declares the two latents ``ImproperUniform`` -- the
    Jeffreys configuration. ``False`` gives them the Gaussian priors
    ``Normal(7.8, w1)`` / ``Normal(2.3, w2)`` the sensitivity tests read.
    ``noise="radiometer"`` declares ``Normal(mu, F |mu|)``; ``"homo"`` a
    constant ``sigma0``, with the ``depends_on_prediction`` claim following
    unless overridden.
    """
    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    truth = jnp.exp(7.8) * (freq / NU0) ** (-2.55) + floor
    data = jnp.broadcast_to(truth, (N_TIME, N_FREQ))
    claim = (noise == "radiometer") if depends_on_prediction is None else depends_on_prediction

    def model():
        if flat_latents:
            la = sample(
                "fg_log_amp",
                lambda: dist.ImproperUniform(dist.constraints.real, (), ()),
            )
            be = sample(
                "fg_beta",
                lambda: dist.ImproperUniform(dist.constraints.real, (), ()),
            )
        else:
            la = sample("fg_log_amp", lambda: dist.Normal(7.8, prior_widths[0]))
            be = sample("fg_beta", lambda: dist.Normal(2.3, prior_widths[1]))
        pred = det(
            "pred",
            lambda a, b: jnp.broadcast_to(
                jnp.exp(a) * (freq / NU0) ** (-b) + floor, (N_TIME, N_FREQ)
            ),
            la,
            be,
        )
        if noise == "radiometer":
            observe(
                "d",
                lambda mu: dist.Normal(mu, RADIOMETER_F * jnp.abs(mu)).to_event(2),
                pred,
                obs=data,
                depends_on_prediction=claim,
            )
        else:
            observe(
                "d",
                lambda mu: dist.Normal(mu, sigma0).to_event(2),
                pred,
                obs=data,
                depends_on_prediction=claim,
            )

    return trace(model)


def doubled_graph():
    """``mu = exp(a + b) (nu/nu0)^-beta``: ``a`` and ``b`` are one parameter twice.

    The exactly singular block the eigh-with-floor route exists for.
    """
    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    truth = (freq / NU0) ** (-2.55)
    data = jnp.broadcast_to(truth, (N_TIME, N_FREQ))

    def model():
        a = sample("a", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
        b = sample("b", lambda: dist.ImproperUniform(dist.constraints.real, (), ()))
        be = sample(
            "fg_beta", lambda: dist.ImproperUniform(dist.constraints.real, (), ())
        )
        pred = det(
            "pred",
            lambda a_, b_, be_: jnp.broadcast_to(
                jnp.exp(a_ + b_) * (freq / NU0) ** (-be_), (N_TIME, N_FREQ)
            ),
            a,
            b,
            be,
        )
        observe(
            "d",
            lambda mu: dist.Normal(mu, RADIOMETER_F * jnp.abs(mu)).to_event(2),
            pred,
            obs=data,
            depends_on_prediction=True,
        )

    return trace(model)


def noisy_power_law_graph(*, prior_widths=(0.5, 0.3), sigma=5.0, seed=3):
    """The sensitivity fixture: a power law with data OFF the truth.

    The data are drawn once, deterministically from ``seed``, INSIDE the
    caller's precision context -- so the mode sits away from both the truth
    and the prior centres and every shift is non-trivial.
    """
    import jax

    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    truth = jnp.exp(7.8) * (freq / NU0) ** (-2.55)
    data = truth + sigma * jax.random.normal(jax.random.key(seed), (N_FREQ,))

    def model():
        la = sample("fg_log_amp", lambda: dist.Normal(7.8, prior_widths[0]))
        be = sample("fg_beta", lambda: dist.Normal(2.3, prior_widths[1]))
        pred = det(
            "pred", lambda a, b: jnp.exp(a) * (freq / NU0) ** (-b), la, be
        )
        observe("d", lambda mu: dist.Normal(mu, sigma), pred, obs=data)

    return trace(model), data


def affine_graph(*, prior_loc=1.4, prior_std=0.2, sigma=1.0):
    """``pred = w * x`` -- exactly quadratic log-posterior, one latent.

    Deliberately starved (unit amplitude, unit noise, few samples) so the
    likelihood curvature H is comparable to the prior's P and the
    ``H^{-1}``-versus-``(H+P)^{-1}`` distinction is measurable: the natural
    wrong matrix is off by ``diag((H+P)^{-1} P)``, which a strong-data
    fixture makes invisibly small.
    """
    x = jnp.linspace(1.0, 2.0, 8)
    data = 1.1 * x

    def model():
        w = sample("w", lambda: dist.Normal(prior_loc, prior_std))
        pred = det("pred", lambda v: v * x, w)
        observe("d", lambda mu: dist.Normal(mu, sigma), pred, obs=data)

    return trace(model)


def saddle_graph(*, target=1.0, prior_std=10.0):
    """``pred = theta^2``: the likelihood Hessian is INDEFINITE near zero.

    ``-log p(d | theta) = (theta^2 - target)^2 / 2`` has curvature
    ``2 (3 theta^2 - target)``, negative for ``|theta| < sqrt(target/3)`` --
    so a Newton solve started at 0.1 with a plain ``solve(H, g)`` steps
    TOWARD the saddle at 0. The B3 fixture: only the eigvalsh check plus the
    Cholesky-with-jitter fallback walks out to the mode near
    ``sqrt(target)``.
    """

    def model():
        theta = sample("theta", lambda: dist.Normal(0.1, prior_std))
        pred = det("pred", lambda t: t**2 * jnp.ones(1), theta)
        observe(
            "d",
            lambda mu: dist.Normal(mu, 1.0).to_event(1),
            pred,
            obs=jnp.full((1,), target),
        )

    return trace(model)


def hierarchical_graph():
    """A latent whose own density is parameterised by another latent.

    ``child ~ Normal(parent, 0.5)``: selecting both entangles the selected
    prior's (m, s) with the selection, which sensitivity must refuse;
    selecting either alone is fine.
    """
    x = jnp.linspace(1.0, 2.0, 8)
    data = 2.0 * x

    def model():
        parent = sample("parent", lambda: dist.Normal(2.0, 1.0))
        child = sample("child", lambda p: dist.Normal(p, 0.5), parent)
        pred = det("pred", lambda c: c * x, child)
        observe("d", lambda mu: dist.Normal(mu, 0.3), pred, obs=data)

    return trace(model)
