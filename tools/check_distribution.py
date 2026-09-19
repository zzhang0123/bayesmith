#!/usr/bin/env python3
"""Verify release contents against source, then optionally extract the sdist.

Run after ``python -m build``. This checks bytes and test membership, not merely
that an archive exists. The extracted suite must still be executed against an
installed wheel; the release workflow performs that separate check.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def excluded(path: str, patterns: list[str]) -> bool:
    """Match the anchored file/directory exclusions used by this project."""
    return any(
        fnmatch.fnmatchcase(path, pattern.lstrip("/"))
        or path.startswith(pattern.strip("/") + "/")
        for pattern in patterns
    )


def verify(root: Path, directory: Path, extract_to: Path | None = None) -> dict:
    config = tomllib.loads((root / "pyproject.toml").read_text())
    version = config["project"]["version"]
    wheels = list(directory.glob("*.whl"))
    archives = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise ValueError("release directory must contain exactly one wheel and sdist")
    wheel, archive = wheels[0], archives[0]
    if not wheel.name.startswith(f"bayesmith-{version}-"):
        raise ValueError("wheel filename does not match the declared version")
    if archive.name != f"bayesmith-{version}.tar.gz":
        raise ValueError("sdist filename does not match the declared version")

    source = {
        p.relative_to(root / "src").as_posix(): p.read_bytes()
        for p in (root / "src/bayesmith").rglob("*.py")
    }
    with zipfile.ZipFile(wheel) as handle:
        names = handle.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate members")
        metadata_root = f"bayesmith-{version}.dist-info/"
        for name in names:
            path = PurePosixPath(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != name.rstrip("/")
            ):
                raise ValueError(f"unsafe wheel member: {name}")
            if (
                not name.endswith("/")
                and name not in source
                and not name.startswith(metadata_root)
            ):
                raise ValueError(f"unexpected wheel installation path: {name}")
        for required in ("METADATA", "WHEEL", "RECORD"):
            if metadata_root + required not in names:
                raise ValueError(f"wheel lacks installation metadata: {required}")
        metadata = BytesParser().parsebytes(handle.read(metadata_root + "METADATA"))
        if metadata.get("Name") != "bayesmith" or metadata.get("Version") != version:
            raise ValueError("wheel metadata name/version does not match the project")
        shipped = {
            name: handle.read(name)
            for name in handle.namelist()
            if name.startswith("bayesmith/") and name.endswith(".py")
        }
    if shipped != source:
        raise ValueError("wheel Python files differ from the current package source")

    prefix = f"bayesmith-{version}/"
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        files = {}
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.as_posix() != member.name.rstrip("/")
                or not member.name.startswith(prefix)
            ):
                raise ValueError(f"unsafe or unexpected sdist member: {member.name}")
            if (
                member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"sdist contains a nonregular member: {member.name}")
            if member.isfile():
                stream = handle.extractfile(member)
                assert stream is not None
                name = member.name.removeprefix(prefix)
                if name in files:
                    raise ValueError(f"duplicate sdist member: {name}")
                files[name] = stream.read()
        shipped_source = {
            p.removeprefix("src/"): b
            for p, b in files.items()
            if p.startswith("src/bayesmith/") and p.endswith(".py")
        }
        if shipped_source != source:
            raise ValueError(
                "sdist Python files differ from the current package source"
            )
        for name in ("pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md"):
            if files.get(name) != (root / name).read_bytes():
                raise ValueError(f"sdist is missing or changed project file: {name}")
        patterns = config["tool"]["hatch"]["build"]["targets"]["sdist"]["exclude"]
        portable_tests = {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in (root / "tests").rglob("*.py")
            if not excluded(p.relative_to(root).as_posix(), patterns)
        }
        shipped_tests = {
            p: b
            for p, b in files.items()
            if p.startswith("tests/") and p.endswith(".py")
        }
        if shipped_tests != portable_tests:
            raise ValueError("sdist portable test files differ from the declared suite")
        support_files = {
            name.lstrip("/")
            for name in config["tool"]["hatch"]["build"]["targets"]["sdist"].get(
                "include", []
            )
            if name.startswith("/docs/probes/") and name.endswith(".py")
        }
        for name in support_files:
            if files.get(name) != (root / name).read_bytes():
                raise ValueError(
                    f"sdist is missing or changed test support file: {name}"
                )
        forbidden = (
            ".agents/",
            ".claude/",
            ".agent-state/",
            "runs/",
            "docs/",
            "site/",
            "examples/",
        )
        if any(
            name.startswith(forbidden) and name not in support_files for name in files
        ):
            raise ValueError("sdist includes repository-only material")
        if extract_to is not None:
            if extract_to.exists() and any(extract_to.iterdir()):
                raise ValueError("extraction destination must be empty")
            extract_to.mkdir(parents=True, exist_ok=True)
            # Members have been checked above; filter also rejects unsafe paths
            # on Python versions that provide the modern extraction API.
            handle.extractall(extract_to, filter="data")
    return {
        "version": version,
        "package_modules": len(source),
        "portable_test_modules": sum(
            Path(p).name.startswith("test_") for p in portable_tests
        ),
        "artifacts": {
            p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in (wheel, archive)
        },
        "extracted": str(extract_to / prefix.rstrip("/")) if extract_to else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.root.resolve(), args.directory.resolve(), args.extract_to)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
