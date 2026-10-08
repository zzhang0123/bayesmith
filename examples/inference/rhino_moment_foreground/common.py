"""Shared constants, paths and provenance helpers of Demo C.

Demo C is the RHINO global 21-cm separation of the rheplicant example
``examples/global21cm`` (Demos A and B, the oracle, the figure of merit)
with a third foreground strategy: a per-LST basis whose columns come from
the physical moment expansion of synchrotron emission (SyncMoments), not
from the ad hoc log-polynomial of Demo A. Everything that both venvs need
lives here; nothing here imports jax, syncmoments or the twin.

Two interpreters run this directory (``README.md``):

* ``basis.py`` runs in a scratch venv holding SyncMoments 0.4.0, whose jax
  pin differs from the twin's;
* every other step runs in the rheplicant checkout's ``.venv`` with
  ``PYTHONPATH`` naming its ``examples`` directory, so that ``global21cm``
  (the twin's simulation products, collapse, SMC and scores) imports.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
RUNS = REPO_ROOT / "runs" / "rhino-moment-foreground"
#: The rheplicant example directory (the twin); override with the variable.
TWIN_ENV = "RHEPLICANT_GLOBAL21CM"
TWIN = Path(os.environ.get(TWIN_ENV, "/Users/zzhang/projects/rheplicant/examples/global21cm"))

# -------------------------------------------------------------- scenarios --


@dataclass(frozen=True)
class Scenario:
    name: str
    channel_width_mhz: float
    beam: str

    @property
    def sim_dir(self) -> Path:
        return TWIN / "results" / ("sim" if self.name == "main" else f"sim_{self.name}")


MAIN = Scenario("main", 2.0, "gaussian")
STRESS = Scenario("stress", 1.0, "hornwet")
SCENARIOS = {s.name: s for s in (MAIN, STRESS)}
N_TIME = 96
NOISE_SIGMA_K = 0.01  # the twin's 10 mK per sample
TROUGH_TRUE_MK = -152.07  # the twin's injected trough on the 0.25 MHz grid

# ------------------------------------------------- the reference population --
#: Ordered field strength of the reference population, Gauss (5 microgauss,
#: the manuscript's Galactic value).
B0_GAUSS = 5e-6
#: Cell counts of the log-energy split, and the largest Taylor order built.
CELL_COUNTS = (1, 2, 3, 4, 6)
N_MAX = 3
#: The pivots' critical frequencies span the band padded by this factor on
#: each side, so that each channel sees kernels peaking below, at and above it.
PIVOT_PAD = 2.0

# cgs
E_ESU = 4.80320425e-10
M_E_G = 9.1093837015e-28
C_CGS = 2.99792458e10
K_B_CGS = 1.380649e-16


def a_b_hz(b_perp_gauss: float) -> float:
    """``a_B = 3 e B_perp / (4 pi m_e c)``: the critical frequency is ``a_B gamma^2``."""
    return 3.0 * E_ESU * b_perp_gauss / (4.0 * 3.141592653589793 * M_E_G * C_CGS)


# ------------------------------------------------------------- talk palette --
PALETTE = {
    "ink": "#172c40",
    "blue": "#2867a3",
    "orange": "#b66117",
    "purple": "#8051a0",
    "green": "#277b63",
    "muted": "#566675",
}
#: Posterior colours by column of the recovery figure.
ORDER_COLOURS = {"oracle": PALETTE["muted"], 0: PALETTE["orange"], 1: PALETTE["purple"],
                 2: PALETTE["blue"], 3: PALETTE["green"]}  # fmt: skip

# -------------------------------------------------------------- provenance --


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions(names: tuple[str, ...]) -> dict[str, str | None]:
    out = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_head(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()  # fmt: skip
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=_jsonable) + "\n")


def _jsonable(value):
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def results_dir(out: Path, quick: bool) -> Path:
    """Where a path's products go: ``<out>/quick`` or ``<out>/full``; the basis and plans stay in ``<out>``."""
    return Path(out) / ("quick" if quick else "full")


def load_freqs_mhz(scenario: Scenario):
    import numpy as np

    return np.load(scenario.sim_dir / "freqs_mhz.npy")
