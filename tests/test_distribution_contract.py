"""Malformed release archives must fail before an installation is attempted."""

import io
import tarfile
import zipfile

import pytest

from tools.check_distribution import verify


@pytest.fixture
def release(tmp_path):
    root, output = tmp_path / "project", tmp_path / "dist"
    (root / "src/bayesmith").mkdir(parents=True)
    (root / "tests").mkdir()
    output.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname="bayesmith"\nversion="0.9.0"\n'
        "[tool.hatch.build.targets.sdist]\nexclude=[]\n"
    )
    for name in ("README.md", "LICENSE", "CHANGELOG.md"):
        (root / name).write_text(name + "\n")
    (root / "src/bayesmith/__init__.py").write_text('"""Minimal build fixture."""\n')
    (root / "tests/test_core.py").write_text("def test_core(): assert True\n")
    files = {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }

    def build(*, omit=(), extra=None, metadata=True, wheel_extra=None):
        with zipfile.ZipFile(output / "bayesmith-0.9.0-py3-none-any.whl", "w") as wheel:
            wheel.writestr("bayesmith/__init__.py", files["src/bayesmith/__init__.py"])
            for name, content in (wheel_extra or {}).items():
                wheel.writestr(name, content)
            if metadata:
                prefix = "bayesmith-0.9.0.dist-info/"
                wheel.writestr(
                    prefix + "METADATA",
                    "Metadata-Version: 2.1\nName: bayesmith\nVersion: 0.9.0\n",
                )
                wheel.writestr(prefix + "WHEEL", "Wheel-Version: 1.0\n")
                wheel.writestr(prefix + "RECORD", "")
        with tarfile.open(output / "bayesmith-0.9.0.tar.gz", "w:gz") as archive:
            for name, content in (files | (extra or {})).items():
                if name in omit:
                    continue
                member = tarfile.TarInfo(
                    name if name.startswith("/") else "bayesmith-0.9.0/" + name
                )
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        return root, output

    return build


def test_source_and_portable_suite_are_checked_by_bytes(release):
    root, output = release()
    report = verify(root, output)
    assert report["package_modules"] == 1
    assert report["portable_test_modules"] == 1
    (root / "tests/test_core.py").write_text("def test_changed(): assert False\n")
    with pytest.raises(ValueError, match="portable test"):
        verify(root, output)


@pytest.mark.parametrize(
    "missing", ["pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md"]
)
def test_missing_build_or_project_inputs_are_refused(release, missing):
    with pytest.raises(ValueError, match="project file"):
        verify(*release(omit=(missing,)))


def test_unexpected_source_is_not_accepted_as_the_reviewed_package(release):
    with pytest.raises(ValueError, match="sdist Python files"):
        verify(*release(extra={"src/bayesmith/surprise.py": b"raise RuntimeError()\n"}))


def test_noninstallable_wheel_without_metadata_is_refused(release):
    with pytest.raises(ValueError, match="installation metadata"):
        verify(*release(metadata=False))


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "runs/private.json",
        "./src/bayesmith/__init__.py",
        "./runs/private.json",
        "src//bayesmith/__init__.py",
    ],
)
def test_unsafe_or_private_archive_members_are_refused(release, name):
    with pytest.raises(ValueError, match="unsafe|repository-only"):
        verify(*release(extra={name: b"not distributable"}))


@pytest.mark.parametrize(
    "name",
    [
        "injected.pth",
        "bayesmith-0.9.0.data/purelib/bayesmith/__init__.py",
        "bayesmith/./__init__.py",
    ],
)
def test_wheel_cannot_install_unchecked_runtime_paths(release, name):
    with pytest.raises(ValueError, match="unsafe wheel|installation path"):
        verify(*release(wheel_extra={name: b"print('unexpected')"}))
