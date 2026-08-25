"""prior_sensitivity on the tour, both packages: §四 4.2's sensitivity row.

The tour is rheplicant's own design-phase fixture -- the full radio twin
(noise waves, receiver bandpass, time-varying gain, a 12-bit ADC) with a
nonlinear two-latent foreground -- and its numbers are the measured table
the migration must reproduce: mode ``(7.824320989, 2.553069844)``, shift
``(+0.0024711, -0.0069239)`` sigma, ``criterion_std = 0.0795`` for beta,
and the seven-row s-ladder. Requires ``rhino-cal-jax`` beside
``rheplicant`` (both local editable installs -- see pyproject's crosscheck
group); skips loudly without them, and a skip is not a pass.

The bayesmith side wraps rheplicant's OWN forward function in a det node
-- the two packages then share the model and the data ARRAY bit for bit,
so every disagreement is the diagnostic's and none is the fixture's. The
data is generated once, inside x64, per §0.1: the two precision regimes
draw different numbers from one seed, so the array is what is shared,
never the construction recipe.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.crosscheck

N_TIME, N_FREQ = 64, 8
ADC_SCALE, SIGMA_POST_GAIN = 0.25, 2.0
NOISE_STD = ADC_SCALE * SIGMA_POST_GAIN

#: rheplicant's design-phase s-ladder for ``fg_beta``: prior width -> shift
#: in DECLARED posterior sigmas, quoted to six decimals; QUOTED_ATOL is half
#: a unit in the last place.
LADDER: tuple[tuple[float, float], ...] = (
    (3.0, +0.000033),
    (1.5, -0.000177),
    (0.3, -0.006924),
    (0.1, -0.063106),
    (0.05, -0.252247),
    (0.025, -1.001566),
    (0.01, -5.942289),
)
QUOTED_ATOL = 5e-7


@pytest.fixture(scope="module")
def tour():
    """The twin, its data, rheplicant's space and forward fn -- built once."""
    rcj = pytest.importorskip(
        "rhino_cal_jax",
        reason=(
            "rhino-cal-jax is not installed, so the tour twin cannot be "
            "built and the sensitivity row of the migration went UNCHECKED. "
            "THIS IS NOT A PASS. Install it like rheplicant:\n"
            "    uv pip install --python .venv/bin/python --no-deps -e "
            "../rhino-cal\n"
            "    uv pip install --python .venv/bin/python editables"
        ),
    )
    import numpyro.distributions as dist
    from rheplicant import Coordinates, Environment, State
    from rheplicant.inference import Bind, Latent, ParameterSpace
    from rheplicant.radio import (
        ADCOperator,
        AntennaLossOperator,
        BeamSpillOperator,
        CalLoadOperator,
        ForegroundOperator,
        GainOperator,
        GlobalSignalOperator,
        NoiseOperator,
        NoiseWaveOperator,
        ReceiverOperator,
        assemble,
    )

    with jax.enable_x64(True):
        freq = jnp.linspace(60e6, 85e6, N_FREQ)
        time_s = jnp.arange(float(N_TIME)) * 2.0
        state = State(
            coords=Coordinates(
                time=time_s, freq=freq, extra={"receiver_input": jnp.arange(N_TIME) % 4}
            ),
            env=Environment(temperature=jnp.array(280.0)),
            key=jax.random.key(20260806),
            meta={"telescope": "RHINO", "obs_id": "tour-001"},
        )
        gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)
        gamma_src = jnp.stack(
            [
                rcj.cable_gamma(
                    rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92
                ),
                rcj.termination_gamma("resistive", N_FREQ, impedance=10.0),
                rcj.cable_gamma(
                    rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98
                ),
                rcj.cable_gamma(
                    rcj.termination_gamma("resistive", N_FREQ, impedance=150.0),
                    freq,
                    length=1.1,
                    loss=0.95,
                ),
            ]
        )
        bandpass = 1.0 + 0.10 * jnp.cos(
            2 * jnp.pi * (freq - freq[0]) / (freq[-1] - freq[0])
        )
        twin = assemble(
            GlobalSignalOperator(
                depth=jnp.array(0.5), centre=jnp.array(75e6), width=jnp.array(5e6)
            ),
            ForegroundOperator(
                amplitude=jnp.array(2500.0),
                spectral_index=jnp.array(2.55),
                ref_freq=70e6,
            ),
            BeamSpillOperator(sky_fraction=jnp.array(0.97), t_ground=jnp.array(290.0)),
            AntennaLossOperator(
                efficiency=jnp.array(0.97), t_physical=jnp.array(293.0)
            ),
            CalLoadOperator(t_load=jnp.array(300.0)),
            CalLoadOperator(t_load=jnp.array(400.0)),
            CalLoadOperator(t_load=jnp.array(1200.0)),
            NoiseWaveOperator(
                t_unc=250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),
                t_cos=30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),
                t_sin=-40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,
                t_rx=290.0 + 5.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 3,
                gamma_src_re=gamma_src.real,
                gamma_src_im=gamma_src.imag,
                gamma_rec_re=gamma_rec.real,
                gamma_rec_im=gamma_rec.imag,
            ),
            ReceiverOperator(bandpass=bandpass / jnp.mean(bandpass)),
            GainOperator(gain=1.0 + 0.02 * jnp.sin(2 * jnp.pi * time_s / 60.0)),
            NoiseOperator(sigma=jnp.array(SIGMA_POST_GAIN)),
            ADCOperator(scale=jnp.array(ADC_SCALE), n_bits=12),
        )
        observed = twin(state).data
        space = ParameterSpace(
            latents=[
                Latent(
                    "fg_log_amp",
                    init=jnp.log(jnp.array(2000.0)),
                    prior=dist.Normal(jnp.log(2000.0), 0.5),
                ),
                Latent("fg_beta", init=jnp.array(2.30), prior=dist.Normal(2.3, 0.3)),
            ],
            bindings=[
                Bind("fg_log_amp", into=lambda p: p["foregrounds"].amplitude, fn=jnp.exp),
                Bind("fg_beta", into=lambda p: p["foregrounds"].spectral_index),
            ],
        )
        forward, _ = space.forward_fn(twin.without("noise"), state)
    return {"space": space, "forward": forward, "observed": observed,
            "fit": twin.without("noise"), "state": state}


@pytest.fixture(scope="module")
def bayesmith_report(tour):
    """bayesmith's prior_sensitivity on a graph wrapping rheplicant's own
    forward -- the same model, the same data array, none of the machinery."""
    import numpyro.distributions as dist

    from bayesmith import det, observe, sample, trace
    from bayesmith.diagnose.sensitivity import prior_sensitivity

    forward = tour["forward"]
    observed = tour["observed"]

    with jax.enable_x64(True):

        def model():
            la = sample(
                "fg_log_amp", lambda: dist.Normal(jnp.log(2000.0), 0.5)
            )
            be = sample("fg_beta", lambda: dist.Normal(2.3, 0.3))
            pred = det(
                "pred",
                lambda a, b: forward({"fg_log_amp": a, "fg_beta": b}),
                la,
                be,
            )
            observe(
                "d",
                lambda mu: dist.Normal(mu, NOISE_STD).to_event(2),
                pred,
                obs=observed,
                depends_on_prediction=False,
            )

        graph = trace(model)
        return prior_sensitivity(graph)


@pytest.fixture(scope="module")
def rheplicant_report(tour):
    from rheplicant.inference.sensitivity import prior_sensitivity

    return prior_sensitivity(
        tour["space"], tour["fit"], tour["state"], tour["observed"], NOISE_STD
    )


class TestTheTourTable:
    """The design-phase numbers, pinned as constants AND against the other
    package's live report -- the constant catches both drifting together,
    the live comparison catches either drifting alone."""

    def test_the_mode_is_the_design_phase_pair(self, bayesmith_report):
        assert float(bayesmith_report.mode_of("fg_log_amp")) == pytest.approx(
            7.824320989, rel=1e-8
        )
        assert float(bayesmith_report.mode_of("fg_beta")) == pytest.approx(
            2.553069844, rel=1e-8
        )

    def test_the_shift_is_the_seven_thousandths_headline(self, bayesmith_report):
        assert float(
            bayesmith_report.for_latent("fg_beta")["shift_sigma"]
        ) == pytest.approx(-0.0069239167, rel=1e-5)
        assert float(
            bayesmith_report.for_latent("fg_log_amp")["shift_sigma"]
        ) == pytest.approx(+0.0024711038, rel=1e-5)
        assert bool(np.all(bayesmith_report.verified))

    def test_the_widths_and_the_criterion(self, bayesmith_report):
        assert float(
            bayesmith_report.for_latent("fg_log_amp")["sigma_post"]
        ) == pytest.approx(2.9775575e-04, rel=1e-6)
        assert float(
            bayesmith_report.for_latent("fg_beta")["sigma_post"]
        ) == pytest.approx(2.4990616e-03, rel=1e-6)
        assert float(
            bayesmith_report.for_latent("fg_beta")["criterion_std"]
        ) == pytest.approx(0.0795, rel=1e-3)

    def test_both_packages_agree_field_for_field(
        self, bayesmith_report, rheplicant_report
    ):
        """The live half. Same model, same data array, two diagnostics; the
        flat layouts also agree because both order by declaration
        (log-amp first), which the spans assert before any comparison."""
        assert bayesmith_report.names == rheplicant_report.names
        assert bayesmith_report.spans == rheplicant_report.spans
        for field in ("mode", "sigma_post", "shift_sigma", "criterion_std",
                      "mean_offset", "prior_loc", "prior_std"):
            assert np.asarray(getattr(bayesmith_report, field)) == pytest.approx(
                np.asarray(getattr(rheplicant_report, field)), rel=1e-6
            ), field

    @pytest.mark.parametrize(("prior_std", "expected"), LADDER[:-1])
    def test_the_ladder_down_to_a_prior_of_0_025(
        self, bayesmith_report, prior_std, expected
    ):
        assert float(
            bayesmith_report.shift_at("fg_beta", prior_std)
        ) == pytest.approx(expected, rel=1e-3, abs=QUOTED_ATOL)

    def test_the_last_rung_carries_the_known_nonlinearity(self, bayesmith_report):
        """At s = 0.01 the shift is six sigma and the closed form has
        drifted from the refit truth by the tour's own nonlinearity --
        1.8e-3, measured upstream -- so the quoted -5.942289 is matched at
        that looser tolerance, deliberately."""
        assert float(bayesmith_report.shift_at("fg_beta", 0.01)) == pytest.approx(
            -5.942289, rel=3e-3
        )
