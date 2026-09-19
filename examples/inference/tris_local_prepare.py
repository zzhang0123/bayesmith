"""D15 local-field and conditional Cas A prototype, using frozen D13 data."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.inference.tris_bright_sources import (
    cas_a_flux_jy,
    rj_integral,
    source_operators,
)
from examples.inference.tris_local_fields import (
    field_basis,
    interpolation_entries,
    node_loading,
)
from examples.inference.tris_physical_prepare import CYGNUS_A_RA_DEC


def prepare(args):
    import healpy as hp
    from astropy.coordinates import SkyCoord

    if args.output.exists():
        raise FileExistsError(args.output)
    if not 1965 <= args.haslam_epoch < 1979 or args.haslam_segment not in (1, 2):
        raise ValueError("Haslam must declare epoch1965..1978 and segment1 or2")
    if not 1998 <= args.tris_epoch <= 2001:
        raise ValueError("TRIS drift-profile scenario must be within1998..2001")
    with np.load(args.input / "data.npz", allow_pickle=False) as f:
        arrays = {k: np.asarray(f[k]) for k in f.files}
    if "galactic_scan_template" in arrays:
        raise ValueError("input already contains a Galactic source layer")
    nquad = int(arrays["quadrature_nside"])
    if nquad < 2 * max(args.a_nodes, args.em_nodes):
        raise ValueError("quadrature must have at least twice the node NSIDE")
    theta, phi = hp.pix2ang(nquad, np.arange(hp.nside2npix(nquad)))
    cyg = SkyCoord(*CYGNUS_A_RA_DEC, unit="deg", frame="icrs").galactic
    prior = {}
    for name, nodes, scales, center, mono_sd in (
        ("a", args.a_nodes, [(1.0, 30.0), (0.5, 7.0)], np.log(30.0), 1.5),
        ("em", args.em_nodes, [(1.5, 30.0), (1.0, 10.0)], np.log(10.0), 2.0),
    ):
        loading, _ = node_loading(nodes, scales)
        indices, weights = interpolation_entries(nodes, theta, phi)
        arrays.pop(f"{name}_basis", None)
        arrays[f"{name}_interp_index"] = indices
        arrays[f"{name}_interp_weight"] = weights
        arrays[f"{name}_mean"] = np.r_[center, np.zeros(loading.shape[1])]
        arrays[f"{name}_sd"] = np.r_[mono_sd, np.ones(loading.shape[1])]
        arrays[f"{name}_node_loading"] = loading
        arrays[f"{name}_node_nside"] = np.array(nodes)
        prior[name] = {
            "node_nside": nodes,
            "scales_log_sd_angle_deg": scales,
            "monopole_mean": float(center),
            "monopole_sd": mono_sd,
        }
        if name == "em":
            arrays["source_em_basis"] = field_basis(
                nodes, np.pi / 2 - cyg.b.rad, cyg.l.rad, loading
            )[0]
    operator_sha = hashlib.sha256(
        Path(__file__).with_name("tris_bright_sources.py").read_bytes()
    ).hexdigest()
    if args.operators:
        with np.load(args.operators, allow_pickle=False) as f:
            if not np.array_equal(f["all_ra_deg"], arrays["all_ra_deg"]) or int(
                f["nside"]
            ) != int(arrays["nside"]):
                raise ValueError("cached source observation geometry differs")
            expected = hashlib.sha256(
                (args.archive / "TRIS_Beam_Profile.txt").read_bytes()
            ).hexdigest()
            if str(f["beam_sha256"]) != expected:
                raise ValueError("cached source beam differs")
            if str(f["operator_sha256"]) != operator_sha or not np.array_equal(
                f["cyg_response"], arrays["source_response"]
            ):
                raise ValueError("cached source operator code/normalization differs")
            haslam, scan = f["haslam"], f["scan"]
    else:
        haslam, scan = source_operators(arrays, args.archive)
    # Validate dependencies/cache/beam before claiming a new output directory.
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        args.output / "source_operators.npz",
        haslam=haslam,
        scan=scan,
        all_ra_deg=arrays["all_ra_deg"],
        nside=arrays["nside"],
        operator_sha256=operator_sha,
        cyg_response=arrays["source_response"],
        beam_sha256=hashlib.sha256(
            (args.archive / "TRIS_Beam_Profile.txt").read_bytes()
        ).hexdigest(),
    )
    qh = rj_integral(
        408.0, cas_a_flux_jy(408.0, args.haslam_epoch, segment=args.haslam_segment)
    )
    qs = rj_integral(
        arrays["frequency_mhz"],
        cas_a_flux_jy(arrays["frequency_mhz"], args.tris_epoch, segment=3),
    )
    arrays["galactic_haslam_template"] = haslam * qh
    arrays["galactic_scan_template"] = scan[:, None] * qs
    arrays["galactic_haslam_epoch"] = np.array(args.haslam_epoch)
    arrays["galactic_haslam_segment"] = np.array(args.haslam_segment)
    arrays["galactic_tris_epoch"] = np.array(args.tris_epoch)
    np.savez_compressed(args.output / "data.npz", **arrays)
    inputs = [args.input / "data.npz", args.archive / "TRIS_Beam_Profile.txt"]
    if args.operators:
        inputs.append(args.operators)
    record = {
        "schema": "tris.local_fields.v1",
        "kind": "real",
        "scientific_certification": False,
        "epoch_admissible_for_real_sampling": bool(
            1978 + 7 / 12 <= args.haslam_epoch <= 1978 + 9 / 12
            and args.haslam_segment == 2
        ),
        "haslam_region_epoch_source": "Haslam1981 A&A100 p212 section3b: north polar region observed Aug-Sep1978, declination>=45deg; CasA atdec58.8. Earlier epoch scenarios are controls only.",
        "prior": prior,
        "quadrature_nside": nquad,
        "observation_nside": int(arrays["nside"]),
        "cas_a": {
            "haslam_epoch": args.haslam_epoch,
            "haslam_segment": args.haslam_segment,
            "tris_epoch": args.tris_epoch,
            "tris_segment": 3,
            "source": "https://arxiv.org/abs/1704.00002 Table5",
            "haslam_flux_jy": float(
                cas_a_flux_jy(408.0, args.haslam_epoch, segment=args.haslam_segment)
            ),
            "tris_flux_jy": cas_a_flux_jy(
                arrays["frequency_mhz"], args.tris_epoch, segment=3
            ).tolist(),
            "maximum_haslam_contribution_k": float(np.max(haslam * qh)),
            "maximum_tris_contribution_k": np.max(scan[:, None] * qs, axis=0).tolist(),
        },
        "source_policy": "CasA Galactic, no isotropic RSB subtraction; empirical emergent flux added without a second absorption screen, no inferred3D placement; shared CygA source_scale plus independent CasA relative LogNormal(0,.2); fixed conditional spectral shape/epochs, not measured epoch posterior; CygA pre-screen normalization remains a separate conditional assumption",
        "limitations": "other Galactic/EG sources and source foreground absorption unmodelled; survey diffuse PSF mixing remains approximate; Stockert and2428 held-out interpretation disabled until source epochs contracted",
        "input_sha256": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
        },
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("tris_local_fields.py"),
                Path(__file__).with_name("tris_bright_sources.py"),
            )
        },
        "data_sha256": hashlib.sha256(
            (args.output / "data.npz").read_bytes()
        ).hexdigest(),
    }
    (args.output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "archive", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--operators", type=Path)
    p.add_argument("--a-nodes", type=int, default=4)
    p.add_argument("--em-nodes", type=int, default=2)
    p.add_argument("--haslam-epoch", type=float, required=True)
    p.add_argument("--haslam-segment", type=int, required=True)
    p.add_argument("--tris-epoch", type=float, required=True)
    print(json.dumps(prepare(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
