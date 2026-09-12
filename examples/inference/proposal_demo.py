"""Explicit proposal schedules on the same registered forward-model demos.

Run ``python -m examples.inference.proposal_demo --output runs/proposal-demos``.
The automatic six-case gallery is separate; these choices are user policies.
Seeds, data, priors and recovery criteria are unchanged. Explicit data-only
initialization avoids starting narrow nonlinear proposals far from the data.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
from functools import partial
from pathlib import Path

import jax
import numpy as np

from bayesmith.artifacts import InitializationPolicy, NamedArray, ProposalBlockPolicy

from .__main__ import positive, save_demo
from .composed_process import data_initialization

SCHEDULES = {
    "linear_gaussian": (ProposalBlockPolicy(("beta",), "iterative_gls"),),
    "exponential_decay": (
        ProposalBlockPolicy(("amplitude", "offset", "rate"), "gauss_newton"),
    ),
    "multiplicative_noise": (
        ProposalBlockPolicy(("p_g",), "bias_corrected_log_linear"),
        ProposalBlockPolicy(("p_n",), "iterative_gls"),
    ),
}

STARTS = {
    "linear_gaussian": {"beta": np.zeros(4)},
    "exponential_decay": {
        "amplitude": np.ones(2),
        "rate": np.ones(2),
        "offset": np.zeros(2),
    },
    "multiplicative_noise": {"p_g": np.zeros(2), "p_n": np.ones(2)},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/proposal-demos"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--draws", type=positive, default=2000)
    parser.add_argument("--warmup", type=positive, default=1000)
    parser.add_argument("--mh", choices=("on", "off"), default="on", help="MH correction (default on); off produces unadjusted approximate updates.")
    parser.add_argument(
        "--initialization", choices=("data-fit", "simple"), default="data-fit"
    )
    args = parser.parse_args()
    passed = True
    with jax.enable_x64(True):
        for name, policies in SCHEDULES.items():
            policies = tuple(dataclasses.replace(p, mh_correction=args.mh == "on") for p in policies)
            module = importlib.import_module(f"examples.inference.{name}")
            initialization = InitializationPolicy(
                values=tuple(
                    NamedArray(name, value, ("coordinate",))
                    for name, value in STARTS[name].items()
                )
            )
            fit_options = (
                {
                    "initialization_factory": partial(
                        data_initialization, perturbation_scale=0.0
                    )
                }
                if name != "linear_gaussian" and args.initialization == "data-fit"
                else {}
            )
            demo = module.run(
                args.seed,
                args.draws,
                args.warmup,
                proposals=policies,
                initialization=None if fit_options else initialization,
                **fit_options,
            )
            demo.note = (
                "Explicit proposal comparison on the registered model and dataset: "
                + " → ".join(f"{p.method}({', '.join(p.names)})" + (" + MH" if p.mh_correction else " (MH off; approximate)") for p in policies)
                + (". Full original likelihood, prior and support enter every MH ratio. " if args.mh == "on" else ". MH is disabled: unadjusted state-dependent updates do not in general preserve the original posterior. Support and finite-density checks still apply. ")
                +
                "The schedule and initialization are explicitly selected; "
                "initialization uses no generating parameters."
            )
            accepted = save_demo(
                name, demo, args.output, args.seed, args.draws, args.warmup
            )
            passed = passed and accepted
            del demo
            jax.clear_caches()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
