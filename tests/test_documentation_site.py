"""Public documentation must remain a reproducible, navigable source projection."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location(
        "build_docs", ROOT / "tools/build_docs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_documentation_matches_current_source():
    done = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_docs.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_plan_token_renders_routes_refusal_and_platform():
    module = builder()
    mixed = module.plan_record(ROOT / "site", "plans/overview.json#mixed")
    curved = module.plan_record(ROOT / "site", "plans/overview.json#curved")
    assert 'class="route route-exact">GCR exact' in mixed
    assert "route-refused" not in mixed
    assert "route-refused" in curved and "route-sampled" in curved
    assert "depend on the BLAS" in curved


def test_plan_token_rejects_an_unknown_plan_or_record():
    module = builder()
    with pytest.raises(ValueError, match="no plan named 'absent'"):
        module.plan_record(ROOT / "site", "plans/overview.json#absent")
    with pytest.raises(ValueError, match="does not exist"):
        module.plan_record(ROOT / "site", "plans/missing.json#mixed")


def test_table_token_formats_columns_and_rejects_an_unknown_table():
    module = builder()
    table = module.table_record(ROOT / "site", "plans/partial.json#recovery")
    assert "<th>Exact mean</th>" in table and "groups[5]" in table
    assert "num_samples 4000" in table
    with pytest.raises(ValueError, match="no table named 'absent'"):
        module.table_record(ROOT / "site", "plans/partial.json#absent")


def test_link_guard_rejects_missing_anchor_and_does_not_accept_only_a_file(tmp_path):
    (tmp_path / "index.html").write_text('<a href="other.html#wrong">Next</a>')
    (tmp_path / "other.html").write_text('<h1 id="actual">Title</h1>')
    with pytest.raises(ValueError, match="missing anchor"):
        builder().validate_site(tmp_path)


def test_offline_contract_refuses_a_remote_script(tmp_path):
    (tmp_path / "index.html").write_text(
        '<script src="https://example.invalid/app.js"></script>'
    )
    with pytest.raises(ValueError, match="remote asset"):
        builder().validate_site(tmp_path)


def test_new_export_cannot_disappear_from_reference_silently(tmp_path):
    package = tmp_path / "src/bayesmith"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__all__ = ["missing_contract"]\n')
    with pytest.raises(
        ValueError, match="Unresolved public API exports.*missing_contract"
    ):
        builder().source_inventory(tmp_path)


def test_lazy_facade_alias_preserves_the_actual_owner_and_contract(tmp_path):
    package = tmp_path / "src/bayesmith"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '__all__ = ["public_name"]\n'
        '_LAZY_ATTRS = {"public_name": ("bayesmith.implementation", "operation")}\n'
    )
    (package / "implementation.py").write_text(
        "def operation(value: float, *, strict: bool = True) -> float:\n"
        '    """An actual contract."""\n'
        "    return value\n"
    )
    module = builder()
    alias, owner, node = module.source_inventory(tmp_path)["bayesmith"]["symbols"][0]
    assert alias == "public_name"
    assert owner == "bayesmith.implementation"
    declaration = ast.parse(
        "def " + module.signature(alias, node) + ":\n    pass"
    ).body[0]
    assert declaration.name == "public_name"
    assert [arg.arg for arg in declaration.args.args] == ["value"]
    assert ast.unparse(declaration.args.args[0].annotation) == "float"
    assert [arg.arg for arg in declaration.args.kwonlyargs] == ["strict"]
    assert ast.unparse(declaration.args.kwonlyargs[0].annotation) == "bool"
    assert ast.literal_eval(declaration.args.kw_defaults[0]) is True
    assert ast.unparse(declaration.returns) == "float"


@pytest.mark.parametrize("expression", ['["ok"] + ["missing"]', "choose_exports()"])
def test_nonliteral_exports_cannot_silently_fall_back_to_definitions(
    tmp_path, expression
):
    package = tmp_path / "src/bayesmith"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(f"__all__ = {expression}\ndef ok(): pass\n")
    with pytest.raises(
        ValueError, match="Unresolved public API exports|Unsupported dynamic __all__"
    ):
        builder().source_inventory(tmp_path)


def test_deprecated_evidence_facade_covers_the_marginal_exports():
    inventory = builder().source_inventory(ROOT)
    evidence, marginal = (
        inventory["bayesmith.evidence"],
        inventory["bayesmith.marginal"],
    )
    assert evidence["declared"] is True
    assert evidence["exports"] == marginal["exports"]
    assert evidence["exports"]
    assert [(name, owner) for name, owner, _ in evidence["symbols"]] == [
        (name, owner) for name, owner, _ in marginal["symbols"]
    ]


def test_link_guard_rejects_an_existing_file_outside_the_site(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (tmp_path / "outside.html").write_text("<h1>Outside</h1>")
    (site / "index.html").write_text('<a href="../outside.html#absent">Outside</a>')
    with pytest.raises(ValueError, match="leaves site root"):
        builder().validate_site(site)


def test_check_rejects_an_orphaned_generated_guide(tmp_path):
    module = builder()
    module.build(ROOT, tmp_path, ROOT / "site")
    (tmp_path / "obsolete-guide.html").write_text("<h1>Stale guide</h1>")
    done = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build_docs.py"),
            "--check",
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode != 0
    assert "obsolete-guide.html" in done.stderr


@pytest.mark.parametrize(
    "markup",
    [
        '<style>@import"https://example.invalid/theme.css";</style>',
        "<style>@font-face {src: url(//example.invalid/font.woff2)}</style>",
        '<div style="background:url(https://example.invalid/image.png)">Text</div>',
        r"<style>body { background: u\72l(h\74tps://example.invalid/image.png); }</style>",
    ],
)
def test_inline_css_cannot_hide_a_remote_resource(tmp_path, markup):
    (tmp_path / "index.html").write_text(markup)
    with pytest.raises(ValueError, match="remote CSS asset"):
        builder().validate_site(tmp_path)


@pytest.mark.parametrize(
    "css",
    [
        '@import url("https://example.invalid/theme.css");',
        '@import "//example.invalid/theme.css";',
        "body {background-image: url(https://example.invalid/image.png)}",
    ],
)
def test_external_stylesheet_cannot_hide_a_remote_resource(tmp_path, css):
    (tmp_path / "index.html").write_text('<link rel="stylesheet" href="local.css">')
    (tmp_path / "local.css").write_text(css)
    with pytest.raises(ValueError, match="remote CSS asset"):
        builder().validate_site(tmp_path)


def test_local_and_embedded_css_resources_remain_offline(tmp_path):
    (tmp_path / "index.html").write_text('<link rel="stylesheet" href="local.css">')
    (tmp_path / "local.css").write_text(
        '@import "theme.css"; '
        "body {background:url(image.png)} "
        "@font-face {src:url(data:font/woff2;base64,AAAA)}"
    )
    (tmp_path / "theme.css").write_text("body {color: black}")
    (tmp_path / "image.png").write_bytes(b"local image fixture")
    assert builder().validate_site(tmp_path) == 1
