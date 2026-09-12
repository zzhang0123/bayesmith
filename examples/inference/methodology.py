"""Render the design notebook from its canonical Markdown and a visual guide."""

from __future__ import annotations

import hashlib
import html
import re
import shutil
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/superpowers/specs/2026-09-08-block-methodology-design.md"
ASSETS = Path(__file__).parent


def _escape(value):
    return html.escape(str(value), quote=True)


def render_methodology(directory, reports):
    """Return HTML and input hashes; copy offline assets alongside the notebook."""
    try:
        from markdown_it import MarkdownIt
    except ImportError as exc:
        raise RuntimeError(
            "Rendering the design guide needs markdown-it-py; install "
            "examples/inference/requirements-presentation.txt"
        ) from exc

    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    source = SOURCE.read_text()
    dependencies = {SOURCE}

    def copy_link(href):
        if urlsplit(href).scheme or href.startswith("#"):
            return href
        path_text, _, fragment = href.partition("#")
        path = (SOURCE.parent / path_text).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Unresolvable local design link: {href}")
        relative = path.relative_to(ROOT)
        output = directory / "design-sources" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output)
        dependencies.add(path)
        return "design-sources/" + quote(relative.as_posix()) + (
            "#" + fragment if fragment else ""
        )

    def render(text, *, inline=False):
        # Keep display math out of Markdown's backslash/underscore handling.
        text = re.sub(
            r"^\\\[\s*\n(.*?)\n\\\]\s*$",
            lambda m: "\n```math\n" + m.group(1) + "\n```\n",
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
        tokens = parser.parseInline(text) if inline else parser.parse(text)

        def links(items):
            for token in items:
                if token.type == "link_open":
                    token.attrSet("href", copy_link(token.attrGet("href")))
                if token.children:
                    links(token.children)

        links(tokens)
        return parser.renderer.render(tokens, parser.options, {})

    def fence(tokens, index, options, env):
        token = tokens[index]
        if token.info == "math":
            return f'<div class="design-math" data-math="{_escape(token.content.strip())}">{_escape(token.content.strip())}</div>'
        if token.info == "mermaid":
            nodes = re.findall(r'\b[A-Z]\[(?:"([^"]+)"|([^\]]+))\]', token.content)
            labels = list(dict.fromkeys(a or b for a, b in nodes))
            flow = "".join(f"<li>{_escape(label)}</li>" for label in labels)
            return f'<figure class="document-flow"><ol>{flow}</ol><figcaption>Flow in reading order. Any feedback paths are preserved in the diagram source.</figcaption><details><summary>Diagram source</summary><pre>{_escape(token.content)}</pre></details></figure>'
        return f'<pre><code>{_escape(token.content)}</code></pre>'

    parser.renderer.rules["fence"] = fence
    parts = re.split(r"(?=^## \d+\.)", source, flags=re.MULTILINE)
    document = render(parts[0].replace("文档状态：", "Document status: "))
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        number = re.match(r"## (\d+)\.", heading).group(1)
        document += (
            f'<details class="source-section" id="design-source-{number}">'
            f'<summary>{_escape(heading.removeprefix("## "))}</summary>'
            f'<div class="document-prose">{render(body)}</div></details>'
        )

    if __package__:
        from .methodology_guide import render_guide
    else:
        from methodology_guide import render_guide

    replacements = {
        "@@GUIDE@@": render_guide(reports),
        "@@SOURCE_HASH@@": hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12],
    }
    reference_style = (ASSETS / "gallery.css").read_text() + (ASSETS / "methodology.css").read_text()
    reference_script = r"""
    function openSource() {
      const id = location.hash.slice(1);
      const section = document.getElementById(id);
      if (section && section.matches('details.source-section')) {
        section.open = true;
        section.scrollIntoView();
      }
    }
    document.querySelectorAll('[data-expand-reference]').forEach(button => {
      button.addEventListener('click', () => {
        document.querySelectorAll('details.source-section').forEach(section => {
          section.open = button.dataset.expandReference === 'true';
        });
      });
    });
    if (typeof katex !== 'undefined') document.querySelectorAll('[data-math]').forEach(node => {
      katex.render(node.dataset.math, node, {displayMode:true, throwOnError:false, trust:false});
    });
    openSource();
    window.addEventListener('hashchange', openSource);
    let printDetails = [];
    window.addEventListener('beforeprint', () => {
      printDetails = [...document.querySelectorAll('details')].map(node => [node, node.open]);
      printDetails.forEach(([node]) => { node.open = true; });
    });
    window.addEventListener('afterprint', () => {
      printDetails.forEach(([node, wasOpen]) => { node.open = wasOpen; });
      printDetails = [];
    });
    """
    (directory / "design-reference.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>bayesmith · Complete methodology reference</title><link rel="icon" href="data:,">'
        '<link rel="stylesheet" href="vendor/katex/katex.min.css"><style>' + reference_style + '</style>'
        '<body class="reference-page"><main class="methodology"><a href="index.html#design/proofs">Back to the bilingual guide</a>'
        '<div class="document-tools"><button type="button" data-expand-reference="true">Expand all sections</button>'
        '<button type="button" data-expand-reference="false">Collapse all sections</button><a href="design.md" download>Download Markdown</a></div>'
        '<div class="document-reader">' + document + '</div></main><script src="vendor/katex/katex.min.js"></script>'
        '<script>' + reference_script + '</script></body></html>'
    )
    template = (ASSETS / "methodology.html").read_text()
    for marker, value in replacements.items():
        template = template.replace(marker, value)
    if re.search(r"@@[A-Z_]+@@", template):
        raise ValueError("Unfilled design guide template marker")
    shutil.copyfile(SOURCE, directory / "design.md")
    shutil.copytree(ASSETS / "vendor/katex", directory / "vendor/katex", dirs_exist_ok=True)
    return template, {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(dependencies)
    }
