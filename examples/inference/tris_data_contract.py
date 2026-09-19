"""Data contract and correction ledger for the public TRIS archive products.

P0a of docs/superpowers/plans/2026-09-12-tris-forward-beam-handoff.md asks for a
table that carries every quantity from *raw column* through *applied
correction* and *forward prediction* to *random error* and *common systematic*,
with each row tagged as one of

    archived-confirmed / paper-confirmed / implementation-convention / unconfirmable

and with an explicit sensitivity branch for anything that is not confirmable.

The reader in this module is deliberately independent of limTOD.tris.archive:
it is a second implementation used to check the first, not a wrapper around it.
Disagreement between the two is a finding.  This module depends only on numpy
and the standard library, so it runs (and is tested) in the bayesmith
environment, which has no limTOD install.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "ARCHIVE_FILES",
    "RING_FILES",
    "RING_FREQUENCY_METADATA",
    "LedgerEntry",
    "archive_facts",
    "build_ledger",
    "ledger_document",
    "parse_ra_deg",
    "read_table",
    "render_markdown",
    "sha256_file",
]

RING_FILES = ("TRIS_absolute_600.txt", "TRIS_absolute_820.txt")
ARCHIVE_FILES = RING_FILES + (
    "TRIS_absolute_2500MHz.txt",
    "TRIS_Beam_Profile.txt",
)

_RA_TOKEN = re.compile(r"(\d{1,2})h(\d{2})m(?:(\d{2})s)?\Z")
_FREQUENCY_HEADER = re.compile(r"Frequency\s*=\s*(\S+)\s*GHz", re.IGNORECASE)
_ZERO_HEADER = re.compile(
    r"Systematic Zero Level Uncertainty\s*=\s*(.+)", re.IGNORECASE
)

#: Effective frequencies and bandwidths are not in the archive text files; they
#: are read off Zannoni et al. (2008) and repeated in the prepared manifest.
#: ghz header -> (nominal MHz, effective MHz, bandwidth MHz).
RING_FREQUENCY_METADATA = {
    0.6: (600.0, 600.5, 0.3),
    0.82: (820.0, 817.8, 0.3),
}
POINT_SET_METADATA = (2500.0, 2427.8, 3.0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_ra_deg(token: str) -> float:
    """Convert an archive token such as 0h00m or 11h26m04s to degrees."""
    match = _RA_TOKEN.fullmatch(token.strip())
    if match is None:
        raise ValueError(f"invalid TRIS right-ascension token: {token!r}")
    hour, minute, second = (int(v or 0) for v in match.groups())
    if hour >= 24 or minute >= 60 or second >= 60:
        raise ValueError(f"TRIS right ascension out of range: {token!r}")
    return 15.0 * (hour + minute / 60.0 + second / 3600.0)


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Return (header_lines, fields_per_row) for one archive text file.

    The public products are inconsistent about record separators: the 600/820
    rings use LF, while TRIS_Beam_Profile.txt and TRIS_absolute_2500MHz.txt
    contain bare CRs.  Decoding in universal newline mode and then normalising
    is what makes this reader agree with the first one.
    """
    raw = Path(path).read_bytes().decode("ascii")
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header: list[str] = []
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header.append(stripped)
            continue
        rows.append(stripped.split())
    return header, rows


def _ring_metadata(
    header: Sequence[str], path: Path
) -> tuple[float, float, float, object]:
    frequency_match = next(
        (m for line in header if (m := _FREQUENCY_HEADER.search(line))), None
    )
    if frequency_match is None:
        raise ValueError(f"{path}: ring file is missing its frequency header")
    ghz = float(frequency_match.group(1))
    try:
        nominal, effective, bandwidth = RING_FREQUENCY_METADATA[ghz]
    except KeyError:
        raise ValueError(f"{path}: unsupported ring frequency {ghz} GHz") from None
    zero_match = next((m for line in header if (m := _ZERO_HEADER.search(line))), None)
    if zero_match is None:
        raise ValueError(f"{path}: ring file is missing its zero-level header")
    zero_text = zero_match.group(1).strip()
    asymmetric = re.fullmatch(r"\+?(\S+)K\s*/\s*-(\S+)K(?:\s*.*)?", zero_text)
    if asymmetric is not None:
        zero_level: object = {
            "positive_k": float(asymmetric.group(1)),
            "negative_k": float(asymmetric.group(2)),
        }
    else:
        single = re.match(r"(\S+)K(?:\s*.*)?\Z", zero_text)
        if single is None:
            raise ValueError(f"{path}: invalid zero-level header {zero_text!r}")
        zero_level = float(single.group(1))
    return nominal, effective, bandwidth, zero_level


def archive_facts(archive_dir: Path) -> dict:
    """Parse all four products independently and return the ledger facts."""
    archive = Path(archive_dir)
    facts: dict = {
        "directory": str(archive),
        "files": {},
        "rings": [],
        "point_set": {},
        "beam": {},
    }

    for name in ARCHIVE_FILES:
        path = archive / name
        if not path.is_file():
            raise FileNotFoundError(path)
        facts["files"][name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    for name in RING_FILES:
        path = archive / name
        header, rows = read_table(path)
        nominal, effective, bandwidth, zero_level = _ring_metadata(header, path)
        ra_text = [row[0] for row in rows]
        ra_deg = np.array([parse_ra_deg(token) for token in ra_text])
        temperature = np.array([float(row[1]) for row in rows])
        sigma = np.array([float(row[2]) for row in rows])
        spacing = np.unique(np.round(np.diff(ra_deg), 6)).tolist()
        zero_rows = np.flatnonzero(sigma == 0.0)
        facts["rings"].append(
            {
                "file": name,
                "nominal_frequency_mhz": nominal,
                "effective_frequency_mhz": effective,
                "bandwidth_mhz": bandwidth,
                "rows": len(rows),
                "ra_deg_first": float(ra_deg[0]),
                "ra_deg_last": float(ra_deg[-1]),
                "ra_spacing_deg": spacing,
                "temperature_min_k": float(temperature.min()),
                "temperature_max_k": float(temperature.max()),
                "statistical_sigma_min_k": float(sigma.min()),
                "statistical_sigma_max_k": float(sigma.max()),
                "zero_sigma_rows": int(zero_rows.size),
                "zero_sigma_row_index": int(zero_rows[0]) if zero_rows.size else None,
                "zero_level": zero_level,
            }
        )

    path = archive / "TRIS_absolute_2500MHz.txt"
    header, rows = read_table(path)
    if not any("2.5" in line and "GHz" in line for line in header):
        raise ValueError(f"{path}: point-set file is missing its 2.5-GHz header")
    point_zero = [float(row[2]) for row in rows]
    if len(set(point_zero)) != 1:
        raise ValueError(f"{path}: 2.5-GHz zero level is not common across rows")
    facts["point_set"] = {
        "file": path.name,
        "nominal_frequency_mhz": POINT_SET_METADATA[0],
        "effective_frequency_mhz": POINT_SET_METADATA[1],
        "bandwidth_mhz": POINT_SET_METADATA[2],
        "rows": len(rows),
        "ra_deg": [parse_ra_deg(row[0]) for row in rows],
        "temperature_k": [float(row[1]) for row in rows],
        "common_zero_level_k": point_zero[0],
        "per_row_statistical_sigma": None,
    }

    path = archive / "TRIS_Beam_Profile.txt"
    header, rows = read_table(path)
    angle = np.array([float(row[0]) for row in rows])
    h_cut = np.array([float(row[1]) for row in rows])
    e_cut = np.array([float(row[2]) for row in rows])
    if np.any(np.diff(angle) <= 0):
        raise ValueError(f"{path}: beam cut angles are not strictly increasing")
    half_power = -10.0 * np.log10(2.0)
    facts["beam"] = {
        "file": path.name,
        "rows": len(rows),
        "angle_min_deg": float(angle.min()),
        "angle_max_deg": float(angle.max()),
        "h_plane_min_db": float(h_cut.min()),
        "e_plane_min_db": float(e_cut.min()),
        "h_plane_anti_boresight_db": float(h_cut[-1]),
        "e_plane_anti_boresight_db": float(e_cut[-1]),
        "e_fwhm_deg": 2.0 * float(np.interp(half_power, e_cut[::-1], angle[::-1])),
        "h_fwhm_deg": 2.0 * float(np.interp(half_power, h_cut[::-1], angle[::-1])),
        "achromatic_statement": any(
            "same at the three frequencies" in line for line in header
        ),
    }
    return facts


def _zero_row_text(ring600: dict, ring820: dict) -> str:
    indices = (ring600["zero_sigma_row_index"], ring820["zero_sigma_row_index"])
    if all(index is not None for index in indices):
        return (
            f"600 MHz row {indices[0]}, 820 MHz row {indices[1]} carry 0.000 K; "
            "a 0.004 K floor replaces them"
        )
    return (
        "the published rings carry one 0.000 K row each; this copy does not, so the "
        "floor is the only variance and its value is a modelling convention"
    )


@dataclass(frozen=True)
class LedgerEntry:
    """One row of the correction ledger.

    stage is the point in the chain the row describes: raw (archive column),
    correction (already applied upstream), forward (what the model does with
    it), random_error or common_systematic.
    """

    item: str
    stage: str
    value: str
    unit: str
    provenance: str
    basis: str
    sensitivity: str = ""


def build_ledger(archive_dir: Path) -> tuple[list[LedgerEntry], dict]:
    """Return (entries, facts) for one archive directory."""
    facts = archive_facts(archive_dir)
    ring600, ring820 = facts["rings"]
    points = facts["point_set"]
    beam = facts["beam"]
    entries: list[LedgerEntry] = [
        LedgerEntry(
            item="product level",
            stage="raw",
            value="absolute sky-brightness temperature drift-scan profiles",
            unit="-",
            provenance="archived-confirmed",
            basis="each ring header: Absolute Sky Brightness Temperature Profile",
            sensitivity="treated as corrected profiles, never as raw TOD",
        ),
        LedgerEntry(
            item="product level",
            stage="correction",
            value="ground, atmosphere, loss, gain and polarization corrections already applied",
            unit="-",
            provenance="paper-confirmed",
            basis="Zannoni et al. 2008 (arXiv:0806.1415) sections IV-V; do not re-apply",
            sensitivity="re-applying the receiver chain would double-count",
        ),
        LedgerEntry(
            item="effective frequency",
            stage="raw",
            value=(
                "600.5 / 817.8 MHz rings, "
                f"{points['effective_frequency_mhz']} MHz point set"
            ),
            unit="MHz",
            provenance="paper-confirmed",
            basis="Zannoni et al. 2008 section 2, Table 1; absent from every archive file",
            sensitivity="2 MHz at 600 MHz shifts a synchrotron index by about 0.03",
        ),
        LedgerEntry(
            item="bandwidth",
            stage="raw",
            value="0.3 / 0.3 MHz (rings), 3.0 MHz (point set)",
            unit="MHz",
            provenance="paper-confirmed",
            basis="Zannoni et al. 2008 Table 1",
            sensitivity="used only for chromaticity budgets",
        ),
        LedgerEntry(
            item="right ascension",
            stage="raw",
            value=(
                f"{ring600['rows']} samples, {ring600['ra_deg_first']:.0f}-"
                f"{ring600['ra_deg_last']:.0f} deg, actual spacing "
                f"{ring600['ra_spacing_deg']} deg"
            ),
            unit="deg",
            provenance="archived-confirmed",
            basis="column 1 of each ring, parsed independently here",
            sensitivity="never re-sort onto a uniform grid; the non-uniform labels are the scan",
        ),
        LedgerEntry(
            item="sampling vs resolution",
            stage="forward",
            value=(
                "3 deg sampling is a scan stride, not resolution (E/H FWHM "
                f"{beam['e_fwhm_deg']:.3f}/{beam['h_fwhm_deg']:.3f} deg)"
            ),
            unit="deg",
            provenance="paper-confirmed",
            basis="cut widths measured from the archive cuts; ring header NOTE1 calls 18 deg rounded",
            sensitivity="noise correlation cannot be inferred from beam width alone",
        ),
        LedgerEntry(
            item="coordinates and epoch",
            stage="raw",
            value="equatorial; archive epoch not stated",
            unit="-",
            provenance="unconfirmable",
            basis="no epoch field exists in any product",
            sensitivity="report the epoch-dependent part as unknown; do not equate two epochs rotations",
        ),
        LedgerEntry(
            item="declination label vs site latitude",
            stage="raw",
            value="header +42 deg vs Campo Imperatore 42 deg 26 arcmin",
            unit="deg",
            provenance="archived-confirmed",
            basis="ring header vs Zannoni et al. 2008 site description",
            sensitivity="0.433 deg difference needs an explicit geometry branch, not a silent default",
        ),
        LedgerEntry(
            item="E-plane roll",
            stage="correction",
            value="E plane tilted 7 deg east of the meridian",
            unit="deg",
            provenance="archived-confirmed",
            basis="ring NOTE2 and TRIS_Beam_Profile header",
            sensitivity="belongs to geometry (selfrot), never to a pre-rotated beam map",
        ),
        LedgerEntry(
            item="temperature convention",
            stage="raw",
            value="Rayleigh-Jeans antenna temperature including the CMB monopole",
            unit="K",
            provenance="paper-confirmed",
            basis="TRIS I; limTOD.tris.cmb_monopole_rj_k documents the same reading",
            sensitivity="thermodynamic vs RJ mixing is a unit bug, not a model choice",
        ),
        LedgerEntry(
            item="common zero level, 600 MHz",
            stage="common_systematic",
            value=f"{float(ring600['zero_level']):.3f} K symmetric",
            unit="K",
            provenance="archived-confirmed",
            basis="ring header Systematic Zero Level Uncertainty = 0.066K",
            sensitivity="one correlated number, never copied into 120 independent errors",
        ),
        LedgerEntry(
            item="common zero level, 820 MHz",
            stage="common_systematic",
            value=(
                f"+{ring820['zero_level']['positive_k']:.3f} / "
                f"-{ring820['zero_level']['negative_k']:.3f} K"
            ),
            unit="K",
            provenance="archived-confirmed",
            basis="ring header; raw systematic 0.660 K tightened by astrophysical constraints",
            sensitivity="two-piece half-normal equal-side-mass is implementation-convention, not archive",
        ),
        LedgerEntry(
            item="per-row statistical uncertainty",
            stage="random_error",
            value=(
                f"{ring600['statistical_sigma_min_k']:.3f}-"
                f"{ring600['statistical_sigma_max_k']:.3f} K (600 MHz), "
                f"{ring820['statistical_sigma_min_k']:.3f}-"
                f"{ring820['statistical_sigma_max_k']:.3f} K (820 MHz)"
            ),
            unit="K",
            provenance="archived-confirmed",
            basis="column 3 of each ring",
            sensitivity="treated as independent; the sample covariance is not published",
        ),
        LedgerEntry(
            item="zero-sigma row and the 0.004 K floor",
            stage="random_error",
            value=_zero_row_text(ring600, ring820),
            unit="K",
            provenance="implementation-convention",
            basis="floored row is archived-confirmed; the floor value is a modelling convention",
            sensitivity="compare retaining vs removing the floored row as its own branch",
        ),
        LedgerEntry(
            item="sample covariance",
            stage="random_error",
            value="not published",
            unit="-",
            provenance="unconfirmable",
            basis="no product carries per-sample covariance",
            sensitivity="keep the common zero level separate from per-sample noise",
        ),
        LedgerEntry(
            item="2.5 GHz third frequency",
            stage="random_error",
            value=(
                f"{points['rows']} points, column 3 is a common zero level "
                f"{points['common_zero_level_k']:.3f} K"
            ),
            unit="K",
            provenance="archived-confirmed",
            basis="file header Column 3 = Zero Level uncertainty in K; no per-row sigma exists",
            sensitivity=(
                "display-only in the historical likelihood; D11 permits explicitly labelled "
                "error-envelope diagnostics, not an invented measured per-row sigma"
            ),
        ),
        LedgerEntry(
            item="2.5 GHz paper statistical summaries",
            stage="random_error",
            value="Paper I Table 12: sigma=25 mK, sigma_mean=10 mK; Paper II Table 7: DeltaTstat=103 mK",
            unit="mK",
            provenance="paper-confirmed",
            basis="arXiv:0806.1415v1 Table 12; arXiv:0807.4750 section 3.2 and Table 7",
            sensitivity=(
                "mapping to six archive rows remains unconfirmed; 10/25/103/sqrt(6)*103 mK "
                "per-row assignments are sensitivity assumptions only; retain common zero separately"
            ),
        ),
        LedgerEntry(
            item="820 MHz laboratory calibration alternative",
            stage="common_systematic",
            value="laboratory bounds approximately +/-0.660 K (Paper I) or +/-0.659 K (Paper III)",
            unit="K",
            provenance="paper-confirmed",
            basis="arXiv:0806.1415v1 section 5 and arXiv:0806.4306v1 section 3.3",
            sensitivity=(
                "D11 declares a Uniform bound sensitivity; narrow astrophysical bounds depend "
                "on the original scan and spectral assumptions; neither is automatically Gaussian 1sigma"
            ),
        ),
        LedgerEntry(
            item="beam model",
            stage="forward",
            value=(
                f"principal-plane cuts 0-{beam['angle_max_deg']:.0f} deg; H/E floor "
                f"{beam['h_plane_min_db']:.1f}/{beam['e_plane_min_db']:.1f} dB; "
                f"anti-boresight {beam['h_plane_anti_boresight_db']:.1f}/"
                f"{beam['e_plane_anti_boresight_db']:.1f} dB"
            ),
            unit="dB",
            provenance="archived-confirmed",
            basis="TRIS_Beam_Profile.txt; achromatic per its own header",
            sensitivity="dB vs power principal-plane blend is the 2D-structure sensitivity band",
        ),
        LedgerEntry(
            item="beam measurement error",
            stage="random_error",
            value="no cut error bars and no measured 2D pattern",
            unit="-",
            provenance="unconfirmable",
            basis="the archive publishes two 1D cuts only",
            sensitivity="width priors are assumptions, not a likelihood; report the scale explicitly",
        ),
        LedgerEntry(
            item="Haslam template",
            stage="forward",
            value="Remazeilles 2014 408 MHz, minus RJ CMB; fixed conditional template",
            unit="-",
            provenance="paper-confirmed",
            basis="prepared manifest haslam_convention; template carries its own native smoothing",
            sensitivity="composite with the target beam in the forward model; never deconvolve",
        ),
        LedgerEntry(
            item="three-region sky parameterisation",
            stage="forward",
            value="piecewise amplitude/beta in |b| < 10 deg, 10-30 deg, >= 30 deg",
            unit="-",
            provenance="implementation-convention",
            basis="examples/inference/tris_sky.py",
            sensitivity="no curvature or separate free-free component; boundaries are a modelling choice",
        ),
        LedgerEntry(
            item="map-making prior",
            stage="forward",
            value="Haslam extrapolation beta=-2.8, SD=hypot(0.5*mean, 3 K)",
            unit="K",
            provenance="implementation-convention",
            basis="examples/inference/tris_prepare.py",
            sensitivity="strength/shape change must not move data-constrained posteriors (P0b)",
        ),
        LedgerEntry(
            item="sky and beam discretisation",
            stage="forward",
            value=(
                "sky nside 8; beam built at nside 8 then upsampled to 64 before "
                "the harmonic transform"
            ),
            unit="-",
            provenance="implementation-convention",
            basis="build_tris_mapmaking_inputs(nside=8, nside_hires=64)",
            sensitivity="quantified as P0c; upsampling cannot recover the beam shape nside 8 never sampled",
        ),
    ]
    return entries, facts


def ledger_document(archive_dir: Path) -> dict:
    entries, facts = build_ledger(archive_dir)
    fact_lines = [
        f"{entry.item} [{entry.stage}] = {entry.value} {entry.unit} :: {entry.provenance}"
        for entry in entries
    ]
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.provenance] = counts.get(entry.provenance, 0) + 1
    return {
        "schema": "bayesmith.tris.data-contract.v1",
        "facts": facts,
        "provenance_counts": counts,
        "entries": [asdict(entry) for entry in entries],
        "ledger": fact_lines,
    }


def render_markdown(document: dict) -> str:
    facts = document["facts"]
    ring600, ring820 = facts["rings"]
    lines = [
        "# TRIS data contract and correction ledger (P0a)",
        "",
        "Generated by examples/inference/tris_data_contract.py; the reader is an",
        "independent second implementation of the archive parser, not a wrapper.",
        "",
        "## Parsed facts",
        "",
        f"- archive: {facts['directory']}",
        (
            f"- 600 MHz: {ring600['rows']} rows, effective "
            f"{ring600['effective_frequency_mhz']} MHz, spacing "
            f"{ring600['ra_spacing_deg']} deg, zero-sigma row "
            f"{ring600['zero_sigma_row_index']}"
        ),
        (
            f"- 820 MHz: {ring820['rows']} rows, effective "
            f"{ring820['effective_frequency_mhz']} MHz, spacing "
            f"{ring820['ra_spacing_deg']} deg, zero-sigma row "
            f"{ring820['zero_sigma_row_index']}"
        ),
        (
            f"- 2.5 GHz: {facts['point_set']['rows']} points, common zero level "
            f"{facts['point_set']['common_zero_level_k']:.3f} K"
        ),
        (
            f"- beam: {facts['beam']['rows']} angles to "
            f"{facts['beam']['angle_max_deg']:.0f} deg, E/H FWHM "
            f"{facts['beam']['e_fwhm_deg']:.3f}/{facts['beam']['h_fwhm_deg']:.3f} deg"
        ),
        f"- provenance counts: {json.dumps(document['provenance_counts'], sort_keys=True)}",
        "",
        "## Ledger",
        "",
        "| item | stage | value | unit | provenance | basis | sensitivity |",
        "|---|---|---|---|---|---|---|",
    ]
    for entry in document["entries"]:
        lines.append(
            "| {item} | {stage} | {value} | {unit} | {provenance} | {basis} | "
            "{sensitivity} |".format(**entry)
        )
    lines.append("")
    return "\n".join(lines)


def _main(argv: Iterable[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    document = ledger_document(args.archive)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "data_contract.json").write_text(
        json.dumps(document, indent=2) + "\n"
    )
    (args.output / "data_contract.md").write_text(render_markdown(document))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
