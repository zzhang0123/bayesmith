"""Attach a saved Campbell experiment to the existing inference notebook.

python -m examples.inference.campbell_notebook --input runs/campbell-large-128 \
    --notebook runs/inference-demo-verified
Only presentation files and the new Campbell chapter are written. Existing
numerical results, sample banks and figures are reused without rerunning fits.
"""

import argparse
import json
import shutil
from pathlib import Path

from .campbell_plot import render_figures
from .presentation import CASES, LEGACY_CASES, write_gallery


def install(source, notebook):
    source, notebook = Path(source), Path(notebook)
    # Validate the complete source bundle before mutating the target directory.
    filenames = ("result.json", "maps.npz", "sky.png", "scaling.png", "scaling.pdf")
    for name in filenames:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    result = json.loads((source / "result.json").read_text())
    if result.get("case") != "campbell_sky":
        raise ValueError("expected a saved Campbell scaling experiment")
    destination = notebook / "campbell_sky"
    destination.mkdir(parents=True, exist_ok=True)
    for name in filenames:
        if (source / name).resolve() != (destination / name).resolve():
            shutil.copyfile(source / name, destination / name)
    if (source / "source").is_dir() and (source / "source").resolve() != (
        destination / "source"
    ).resolve():
        shutil.copytree(source / "source", destination / "source", dirs_exist_ok=True)
    render_figures(result, destination)
    reports = []
    order = list(CASES) + ["campbell_sky", "tris_haslam", "tris_haslam_rsb_comparison"]
    for path in sorted(
        notebook.glob("*/result.json"),
        key=lambda path: (
            order.index(path.parent.name) if path.parent.name in order else len(order)
        ),
    ):
        report = json.loads(path.read_text())
        if report.get("case") in {*order, *LEGACY_CASES}:
            reports.append((report, path.parent.name))
    write_gallery(reports, notebook)
    return notebook / "index.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("runs/campbell-large-128"))
    parser.add_argument(
        "--notebook", type=Path, default=Path("runs/inference-demo-verified")
    )
    args = parser.parse_args()
    print(install(args.input, args.notebook))


if __name__ == "__main__":
    main()
