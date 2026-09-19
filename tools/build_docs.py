#!/usr/bin/env python3
"""Build the offline documentation site from reviewed HTML and Python source.

No package imports or third-party dependencies are needed. Run from any directory:
    python tools/build_docs.py --check
Use --source-root to document a different checkout while preparing a release.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import html
import json
import re
import tempfile
import tomllib
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def esc(value):
    return html.escape(str(value), quote=True)


def slug(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def literal(node, default=None):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return default


def source_inventory(source_root):
    """Resolve declared exports, direct definitions, constants and facade aliases."""
    modules = {}
    for path in sorted((source_root / "src/bayesmith").rglob("*.py")):
        relative = path.relative_to(source_root / "src").with_suffix("")
        parts = list(relative.parts)
        package = parts[-1] == "__init__"
        if package:
            parts.pop()
        name = ".".join(parts)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        definitions, imports, exports, lazy = {}, {}, None, {}
        declared = False
        module_aliases = {}
        star_imports = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if isinstance(target, ast.Name):
                        definitions[target.id] = node
                        if target.id == "__all__":
                            if declared:
                                raise ValueError(
                                    f"Multiple __all__ assignments in {name}"
                                )
                            declared = True
                            exports = node.value
                        elif target.id == "_LAZY_ATTRS":
                            lazy = literal(node.value, {})
            elif (
                isinstance(node, ast.AugAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "__all__"
            ):
                raise ValueError(f"Unsupported dynamic __all__ in {name}")
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "__all__"
            ):
                raise ValueError(f"Unsupported mutation of __all__ in {name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    module_aliases[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                parent = parts if package else parts[:-1]
                base = parent[: len(parent) - node.level + 1] if node.level else []
                destination = ".".join(base + ([node.module] if node.module else []))
                for alias in node.names:
                    if alias.name == "*":
                        star_imports.append(destination)
                    else:
                        imports[alias.asname or alias.name] = (destination, alias.name)
        imports.update(lazy)
        modules[name] = {
            "path": path,
            "doc": ast.get_docstring(tree) or "",
            "definitions": definitions,
            "imports": imports,
            "export_node": exports,
            "declared": declared,
            "module_aliases": module_aliases,
            "star_imports": star_imports,
        }

    def module_reference(current, node):
        item = modules[current]
        if isinstance(node, ast.Name):
            imported = item["imports"].get(node.id)
            candidate = (
                ".".join(imported) if imported else item["module_aliases"].get(node.id)
            )
        elif isinstance(node, ast.Attribute):
            parent = module_reference(current, node.value)
            candidate = parent + "." + node.attr if parent else None
        else:
            return None
        return candidate if candidate in modules else None

    def public_exports(current, visited=()):
        item = modules[current]
        if "exports" in item:
            return item["exports"]
        if current in visited:
            raise ValueError(f"Circular __all__ reference in {current}")

        def evaluate(node, local_names=()):
            if isinstance(node, (ast.List, ast.Tuple)):
                values = [literal(entry) for entry in node.elts]
            elif isinstance(node, ast.Name):
                imported = item["imports"].get(node.id)
                definition = item["definitions"].get(node.id)
                if imported and imported[1] == "__all__" and imported[0] in modules:
                    values = list(public_exports(imported[0], visited + (current,)))
                elif (
                    isinstance(definition, (ast.Assign, ast.AnnAssign))
                    and node.id not in local_names
                ):
                    values = evaluate(definition.value, local_names + (node.id,))
                else:
                    raise ValueError(
                        f"Unsupported __all__ name in {current}: {node.id}"
                    )
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                values = evaluate(node.left, local_names) + evaluate(
                    node.right, local_names
                )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("list", "tuple")
                and len(node.args) == 1
                and not node.keywords
            ):
                values = evaluate(node.args[0], local_names)
            elif isinstance(node, ast.Attribute) and node.attr == "__all__":
                owner = module_reference(current, node.value)
                if owner is None:
                    raise ValueError(
                        f"Unresolved __all__ module reference in {current}"
                    )
                values = list(public_exports(owner, visited + (current,)))
            else:
                raise ValueError(
                    f"Unsupported dynamic __all__ in {current}: {ast.unparse(node)}"
                )
            if not all(isinstance(value, str) and value for value in values):
                raise ValueError(f"Invalid __all__ members in {current}")
            return values

        item["exports"] = (
            evaluate(item["export_node"])
            if item["declared"]
            else [n for n in item["definitions"] if not n.startswith("_")]
        )
        if len(item["exports"]) != len(set(item["exports"])):
            raise ValueError(f"Duplicate __all__ members in {current}")
        return item["exports"]

    for name, module in modules.items():
        public_exports(name)
        getter = module["definitions"].get("__getattr__")
        if isinstance(getter, ast.FunctionDef) and getter.args.args:
            parameter = getter.args.args[0].arg
            destinations = set()
            for node in ast.walk(getter):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Call):
                    call = node.value
                    if (
                        isinstance(call.func, ast.Name)
                        and call.func.id == "getattr"
                        and len(call.args) == 2
                        and isinstance(call.args[1], ast.Name)
                        and call.args[1].id == parameter
                    ):
                        owner = module_reference(name, call.args[0])
                        if owner:
                            destinations.add(owner)
            if len(destinations) == 1:
                module["forward_module"] = destinations.pop()

    def resolve(module, name, visited=()):
        key = (module, name)
        if key in visited or module not in modules:
            return None
        item = modules[module]
        if name in item["definitions"]:
            return module, item["definitions"][name]
        target = item["imports"].get(name)
        if target:
            return resolve(*target, visited + (key,))
        for owner in reversed(item["star_imports"]):
            if owner in modules and name in modules[owner]["exports"]:
                return resolve(owner, name, visited + (key,))
        forward = item.get("forward_module")
        return resolve(forward, name, visited + (key,)) if forward else None

    public_modules = {
        n: m
        for n, m in modules.items()
        if not any(p.startswith("_") for p in n.split("."))
    }
    unresolved = []
    for name, module in public_modules.items():
        module["symbols"] = []
        for symbol in module["exports"]:
            resolved = resolve(name, symbol)
            if resolved is None:
                unresolved.append(f"{name}.{symbol}")
            else:
                owner, node = resolved
                module["symbols"].append((symbol, owner, node))
    if unresolved:
        raise ValueError("Unresolved public API exports: " + ", ".join(unresolved))
    return public_modules


def signature(name, node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        result = f"{name}({ast.unparse(node.args)})"
        if node.returns:
            result += " -> " + ast.unparse(node.returns)
        return result
    if isinstance(node, ast.ClassDef):
        fields = [ast.unparse(n) for n in node.body if isinstance(n, ast.AnnAssign)]
        init = next(
            (
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        if init:
            constructor = copy.deepcopy(init)
            if constructor.args.args and constructor.args.args[0].arg == "self":
                constructor.args.args.pop(0)
            return signature(name, constructor)
        bases = ", ".join(ast.unparse(n) for n in node.bases)
        return (
            f"class {name}"
            + (f"({bases})" if bases else "")
            + ("\n" + "\n".join(fields) if fields else "")
        )
    return ast.unparse(node)


def api_filename(module):
    return "api/" + module + ".html"


def api_body(name, module, source_root):
    body = '<p class="lede">Source-derived signatures and contracts. See the <a href="../reference.html">reference index</a> for coverage and conventions.</p>'
    if name.startswith(("bayesmith.reweight", "bayesmith.cumulants")):
        body += '<div class="warning"><p>Experimental composition surface. These primitives do not certify evidence, convergence, global density validity or reference-target overlap. Read the <a href="../experimental.html">experimental guide</a> before use.</p></div>'
    elif name.startswith("bayesmith.bridge.jaxns"):
        body += '<div class="warning"><p>Residual-evidence adapter. Public residual EvidenceTask execution is not enabled by this adapter or by installing its optional dependency. See <a href="../evidence.html#residual">the evidence boundary</a>.</p></div>'
    elif name == "bayesmith.evidence" or name.startswith("bayesmith.evidence."):
        body += '<div class="warning"><p>Deprecated compatibility namespace. Use bayesmith.marginal. This namespace must not be reused for graph-level EvidenceTask semantics.</p></div>'
    if module["doc"]:
        body += (
            '<details><summary>Module contract</summary><pre class="docstring">'
            + esc(module["doc"])
            + "</pre></details>"
        )
    body += (
        '<div class="symbol-index">'
        + " ".join(f'<a href="#{esc(s)}">{esc(s)}</a>' for s, _, _ in module["symbols"])
        + "</div>"
    )
    for symbol, owner, node in module["symbols"]:
        body += f'<section class="api-symbol"><h2 id="{esc(symbol)}">{esc(symbol)}</h2>'
        body += f'<p class="source-note">Defined in <code>{esc(owner)}</code>, line {node.lineno}.</p>'
        body += "<pre><code>" + esc(signature(symbol, node)) + "</code></pre>"
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            body += (
                '<pre class="docstring">'
                + esc(
                    doc
                    or "No additional source docstring. Read the signature and owning module contract above."
                )
                + "</pre>"
            )
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(
                    method, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and not method.name.startswith("_"):
                    body += f'<h3 id="{esc(symbol)}.{esc(method.name)}">{esc(symbol)}.{esc(method.name)}</h3>'
                    body += (
                        "<pre><code>"
                        + esc(signature(method.name, method))
                        + "</code></pre>"
                    )
                    if ast.get_docstring(method):
                        body += (
                            '<pre class="docstring">'
                            + esc(ast.get_docstring(method))
                            + "</pre>"
                        )
            # Enum members and public class constants are part of the contract.
            members = [
                ast.unparse(n)
                for n in node.body
                if isinstance(n, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and not t.id.startswith("_")
                    for t in n.targets
                )
            ]
            if members:
                body += "<pre><code>" + esc("\n".join(members)) + "</code></pre>"
        body += "</section>"
    return body


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.text = set(), [], []
        self.duplicates = []
        self.resources = []
        self.math = []
        self.inline_css = []
        self._in_style = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-math" in attrs and "data-tex" in attrs:
            raise ValueError("A math element must use only one math attribute")
        for attribute in ("data-math", "data-tex"):
            if attribute in attrs:
                self.math.append(
                    {"tex": attrs[attribute], "display": attribute == "data-math"}
                )
        if tag == "style":
            self._in_style = True
        if "style" in attrs:
            self.inline_css.append(attrs["style"])
        if "id" in attrs:
            if attrs["id"] in self.ids:
                self.duplicates.append(attrs["id"])
            self.ids.add(attrs["id"])
        if tag in ("script", "img", "link", "iframe", "video", "audio"):
            self.resources.append(attrs.get("src", attrs.get("href", "")))
        for attribute in ("href", "src"):
            if attribute in attrs:
                self.links.append(attrs[attribute])

    def handle_endtag(self, tag):
        if tag == "style":
            self._in_style = False

    def handle_data(self, data):
        self.text.append(data)
        if self._in_style:
            self.inline_css.append(data)


def css_resources(source):
    """Read stylesheet resource URLs, including escaped CSS identifiers/values."""
    source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)

    def unescape(match):
        if match.group(1):
            codepoint = int(match.group(1), 16)
            return (
                chr(codepoint)
                if 0 < codepoint <= 0x10FFFF and not 0xD800 <= codepoint <= 0xDFFF
                else "\ufffd"
            )
        return "" if match.group(2) else match.group(3)

    source = re.sub(
        r"\\([0-9a-fA-F]{1,6})(?:\r\n|[ \t\r\n\f])?|\\(\r\n|[\r\n\f])|\\(.)",
        unescape,
        source,
        flags=re.DOTALL,
    )
    # Quoted imports and url() are distinct forms; the import-url form is
    # covered by the second branch. Match case-insensitively as CSS requires.
    pattern = (
        r"""@import\s*["']([^"']+)["']|url\(\s*(?:"([^"]*)"|'([^']*)'|([^)]*))\s*\)"""
    )
    for match in re.finditer(pattern, source, re.IGNORECASE):
        yield next(value.strip() for value in match.groups() if value is not None)


def validate_site(output):
    """Fail on dangling internal links/anchors, duplicate IDs or remote assets."""
    output = output.resolve()
    pages = {}
    for path in output.rglob("*.html"):
        if "content" in path.relative_to(output).parts:
            continue
        parser = PageParser()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser
    errors = []
    for path, page in pages.items():
        errors.extend(f"{path.name}: duplicate id {i}" for i in page.duplicates)
        for resource in page.resources:
            if urlsplit(resource).scheme in ("http", "https") or resource.startswith(
                "//"
            ):
                errors.append(f"{path.name}: remote asset {resource}")
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                if url.scheme not in ("https", "http", "mailto", "data"):
                    errors.append(f"{path.name}: unsupported link {link}")
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.is_relative_to(output):
                errors.append(
                    f"{path.relative_to(output)}: link leaves site root: {link}"
                )
                continue
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append(f"{path.relative_to(output)}: missing {link}")
            elif (
                url.fragment
                and target in pages
                and unquote(url.fragment) not in pages[target].ids
            ):
                errors.append(f"{path.relative_to(output)}: missing anchor {link}")
    stylesheets = [
        (path, css) for path, page in pages.items() for css in page.inline_css
    ]
    stylesheets.extend(
        (path, path.read_text(encoding="utf-8"))
        for path in output.rglob("*.css")
        if "content" not in path.relative_to(output).parts
    )
    for origin, css in stylesheets:
        for resource in css_resources(css):
            url = urlsplit(resource)
            if url.scheme == "data":
                continue
            if url.scheme or url.netloc:
                errors.append(
                    f"{origin.relative_to(output)}: remote CSS asset {resource}"
                )
                continue
            if not url.path:
                continue  # e.g. url(#filter) references an in-document SVG resource
            target = (origin.parent / unquote(url.path)).resolve()
            if not target.is_relative_to(output):
                errors.append(
                    f"{origin.relative_to(output)}: CSS asset leaves site root: {resource}"
                )
            elif not target.is_file():
                errors.append(
                    f"{origin.relative_to(output)}: missing CSS asset {resource}"
                )
    if errors:
        raise ValueError("\n".join(errors))
    return len(pages)


def render_page(page, body, navigation, version, prefix=""):
    headings = re.findall(r'<h2 id="([^"]+)">(.+?)</h2>', body)
    local = "".join(f'<a href="#{esc(i)}">{title}</a>' for i, title in headings)
    sidebar = []
    for group, entries in navigation.items():
        links = "".join(
            f'<a href="{prefix}{e["path"]}"'
            + (' aria-current="page"' if e["path"] == page["path"] else "")
            + f">{esc(e.get('nav_title', e['title']))}</a>"
            for e in entries
        )
        sidebar.append(f"<section><h2>{esc(group)}</h2>{links}</section>")
    return f'''<!doctype html>
<html lang="{page.get("lang", "en")}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex, nofollow, noarchive">
<meta name="description" content="{esc(page.get("summary", page["title"]))}"><title>{esc(page["title"])} · bayesmith</title>
<link rel="stylesheet" href="{prefix}assets/vendor/katex/katex.min.css"><link rel="stylesheet" href="{prefix}assets/docs.css"><script defer src="{prefix}assets/vendor/katex/katex.min.js"></script><script defer src="{prefix}assets/search-index.js"></script><script defer src="{prefix}assets/docs.js"></script></head>
<body><a class="skip-link" href="#main">Skip to content</a>
<header class="topbar"><a class="brand" href="{prefix}index.html">bayesmith<span>Documentation</span></a><span class="version">{esc(version)} source</span>
<div class="header-actions"><button id="menu-toggle" aria-controls="sidebar" aria-expanded="false">Menu</button><button id="search-toggle" aria-haspopup="dialog">Search <kbd>/</kbd></button><button id="theme-toggle" aria-label="Switch color theme">Theme</button></div></header>
<div class="layout"><nav class="sidebar" id="sidebar" aria-label="Documentation">{"".join(sidebar)}</nav>
<main id="main"><nav class="breadcrumbs" aria-label="Breadcrumb"><a href="{prefix}index.html">Docs</a><span>/</span><span>{esc(page["group"])}</span><span>/</span><span aria-current="page">{esc(page["title"])}</span></nav>
<article><h1>{esc(page["title"])}</h1>{body}</article><footer><p>Version {esc(version)} · <a href="{prefix}project.html">Release status</a>.</p><a href="#main">Back to top</a></footer></main>
<aside class="local-toc" aria-label="On this page"><p>On this page</p>{local}</aside></div>
<dialog id="search-dialog" aria-labelledby="search-label"><form method="dialog"><button class="close-search" aria-label="Close search">Close</button></form><label id="search-label" for="search-input">Search documentation and API</label><input id="search-input" type="search" placeholder="Graph, EvidenceTask, masking…" autocomplete="off"><p id="search-status" aria-live="polite">Type a concept or API name.</p><ol id="search-results"></ol></dialog>
</body></html>'''


def plan_record(site_source, spec):
    """Render ``{{plan:path.json#name}}``: one plan recorded by a tools/ script.

    The record holds a per-block summary, the printed plan and the environment
    it was measured in. The table is the summary; the printout stays one click
    away, and the caption names the platform because condition numbers in the
    printout depend on the BLAS.
    """
    path, _, name = spec.partition("#")
    source = site_source / "assets" / path
    if not source.is_file():
        raise ValueError(f"Plan record {path} does not exist")
    record = json.loads(source.read_text(encoding="utf-8"))
    if name not in record.get("plans", {}):
        raise ValueError(f"Plan record {path} has no plan named {name!r}")
    plan, env = record["plans"][name], record["environment"]
    rows = []
    for index, block in enumerate(plan["blocks"]):
        kind = "sampled" if block["method"] == "nuts" else "exact"
        refused = (
            ' <span class="route route-refused">claim refused</span>'
            if block["refused_claim"]
            else ""
        )
        rows.append(
            f"<tr><td>{index}</td><td><code>{esc(', '.join(block['latents']))}</code></td>"
            f'<td><span class="route route-{kind}">{esc(block["route"])}</span>{refused}</td>'
            f"<td>{esc(block['evidence'])}</td></tr>"
        )
    return (
        '<figure class="plan-record"><div class="table-scroll"><table class="plan-table">'
        "<thead><tr><th>Block</th><th>Latents</th><th>Route</th><th>Why</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody><tfoot><tr><th scope=\"row\" colspan=\"2\">Execution</th>"
        f'<td colspan="2"><code>{esc(plan["execution"])}</code></td></tr></tfoot></table></div>'
        f'<details><summary>Printed plan</summary><pre class="plan-output"><code>{esc(plan["text"])}</code></pre></details>'
        f"<figcaption>Recorded {esc(env['recorded'])} with bayesmith {esc(env['bayesmith'])} "
        f"on {esc(env['machine'])} ({esc(env['blas'])} BLAS, x64 {'on' if env['jax_enable_x64'] else 'off'}) "
        f"from <a href=\"snippets/{esc(record['snippet'])}\"><code>{esc(record['snippet'])}</code></a>. "
        "Condition numbers and tolerances in the printout depend on the BLAS; "
        f'block membership and routes are re-checked by the test suite. <a href="assets/{esc(path)}">Full record</a>.</figcaption></figure>'
    )


def table_record(site_source, spec):
    """Render ``{{table:path.json#name}}``: a result table recorded by a tools/ script.

    Columns carry their own number format; the caption states the sampler
    settings and the platform, because sampled columns are one run's arithmetic.
    """
    path, _, name = spec.partition("#")
    source = site_source / "assets" / path
    if not source.is_file():
        raise ValueError(f"Table record {path} does not exist")
    record = json.loads(source.read_text(encoding="utf-8"))
    if name not in record.get("tables", {}):
        raise ValueError(f"Table record {path} has no table named {name!r}")
    table, env = record["tables"][name], record["environment"]
    columns = table["columns"]
    head = "".join(f"<th>{esc(column['label'])}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{esc(format(row[column['key']], column.get('format', '')))}</td>"
            for column in columns
        )
        + "</tr>"
        for row in table["rows"]
    )
    settings = ", ".join(f"{key} {value}" for key, value in table["settings"].items())
    return (
        f'<figure class="plan-record"><div class="table-scroll"><table class="result-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        f"<figcaption>Recorded {esc(env['recorded'])} with bayesmith {esc(env['bayesmith'])} "
        f"on {esc(env['machine'])} ({esc(env['blas'])} BLAS, x64 {'on' if env['jax_enable_x64'] else 'off'}); "
        f"{esc(settings)}. Sampled columns are one run on that platform; closed-form "
        f'columns are recomputed by the test suite. <a href="assets/{esc(path)}">Full record</a>.</figcaption></figure>'
    )


def build(source_root, output, site_source):
    config = json.loads((site_source / "navigation.json").read_text(encoding="utf-8"))
    version = tomllib.loads((source_root / "pyproject.toml").read_text())["project"][
        "version"
    ]
    inventory = source_inventory(source_root)
    navigation = defaultdict(list)
    for page in config:
        navigation[page["group"]].append(page)
    pages, search, coverage = {}, [], {}
    for page in config:
        body = (site_source / "content" / page["source"]).read_text(encoding="utf-8")
        body = body.replace("{{version}}", esc(version))

        def snippet(match):
            name = match.group(1)
            code = (site_source / "snippets" / name).read_text(encoding="utf-8")
            compile(code, name, "exec")
            return (
                '<div class="code-caption">'
                + esc(name)
                + '</div><pre><code class="language-python">'
                + esc(code)
                + '</code></pre><p><a download href="snippets/'
                + esc(name)
                + '">Download this example</a></p>'
            )

        body = re.sub(r"\{\{snippet:([\w.-]+)\}\}", snippet, body)
        body = re.sub(
            r"\{\{plan:([\w./-]+#[\w-]+)\}\}",
            lambda match: plan_record(site_source, match.group(1)),
            body,
        )
        body = re.sub(
            r"\{\{table:([\w./-]+#[\w-]+)\}\}",
            lambda match: table_record(site_source, match.group(1)),
            body,
        )

        def api_link(match):
            module, symbol = match.group(1).rsplit(":", 1)
            if module not in inventory or symbol not in inventory[module]["exports"]:
                raise ValueError(f"Unknown API link {module}:{symbol}")
            return (
                f'<a href="{api_filename(module)}#{symbol}"><code>{symbol}</code></a>'
            )

        body = re.sub(r"\{\{api:([\w.:]+)\}\}", api_link, body)
        if "{{" in body:
            raise ValueError(f"Unexpanded source token in {page['source']}")
        pages[page["path"]] = render_page(page, body, navigation, version)
        parser = PageParser()
        parser.feed(body)
        search.append(
            {
                "title": page["title"],
                "path": page["path"],
                "text": " ".join(parser.text),
            }
        )
    api_rows = defaultdict(list)
    for name, module in inventory.items():
        if not module["symbols"]:
            continue
        path = api_filename(name)
        group = name.split(".")[1] if "." in name else "root"
        api_rows[group].append((name, path, len(module["symbols"])))
        page = {"title": name, "path": path, "group": "Reference"}
        search.append(
            {"title": name, "path": path, "text": module["doc"], "kind": "module"}
        )
        body = api_body(name, module, source_root)
        pages[path] = render_page(page, body, navigation, version, "../")
        for symbol, owner, node in module["symbols"]:
            identity = name + "." + symbol
            coverage[identity] = {
                "path": path + "#" + symbol,
                "owner": owner,
                "signature": signature(symbol, node),
            }
            search.append(
                {
                    "title": identity,
                    "path": path + "#" + symbol,
                    "text": (ast.get_docstring(node) or "")
                    if isinstance(
                        node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    else signature(symbol, node),
                }
            )
    catalog = '<p class="lede">Find an entry point by responsibility, then inspect its source-derived contract. Every declared public export is checked during the build.</p><p>Names in <code>__all__</code> are explicit exports; for modules without it, public source definitions are included for discoverability. Re-exports retain their defining module. Class fields are local declarations; inherited fields and runtime-generated constructors are not expanded. Inclusion is not a separate stability promise. Runtime, backend-specific, experimental and deprecated surfaces keep their documented limits.</p>'
    catalog += '<h2 id="entry-points">Recommended entry points</h2><p>Begin with the facade that owns your question; the exhaustive module catalog follows.</p><ul><li><a href="api/bayesmith.html">bayesmith</a>: tracing, graph declarations, compilation and execution.</li><li><a href="api/bayesmith.artifacts.html">bayesmith.artifacts</a>: task/result types, policies and serialization.</li><li><a href="api/bayesmith.evaluation.html">bayesmith.evaluation</a>: model checks and report aggregation.</li><li><a href="api/bayesmith.exact.html">bayesmith.exact</a>: exact numerical primitives.</li><li><a href="api/bayesmith.marginal.html">bayesmith.marginal</a>: nuisance-integrated likelihood components.</li></ul>'
    for group, rows in sorted(api_rows.items()):
        catalog += f'<h2 id="{group}">{esc(group)}</h2><ul class="api-modules">'
        for name, path, count in rows:
            catalog += f'<li><a href="{path}"><code>{esc(name)}</code></a><span>{count} symbols</span></li>'
        catalog += "</ul>"
    pages["reference.html"] = render_page(
        {"title": "API reference", "path": "reference.html", "group": "Reference"},
        catalog,
        navigation,
        version,
    )
    output.mkdir(parents=True, exist_ok=True)
    old_manifest = output / "manifest.json"
    if old_manifest.is_file():
        old_pages = json.loads(old_manifest.read_text()).get("pages", [])
        for obsolete in set(old_pages) - set(pages):
            candidate = (output / obsolete).resolve()
            if candidate.is_relative_to(output.resolve()) and candidate.is_file():
                candidate.unlink()
    for folder in ("assets", "snippets", "diagrams"):
        for path in (site_source / folder).rglob("*"):
            if (
                path.is_file()
                and path.name != "search-index.js"
                and "__pycache__" not in path.parts
            ):
                destination = output / folder / path.relative_to(site_source / folder)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if path.resolve() != destination.resolve():
                    destination.write_bytes(path.read_bytes())
    for name, content in pages.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (output / "assets/search-index.js").write_text(
        "window.BAYESMITH_SEARCH = "
        + json.dumps(search, ensure_ascii=False).replace("</", "<\\/")
        + ";\n",
        encoding="utf-8",
    )
    math = {}
    for name, content in pages.items():
        parser = PageParser()
        parser.feed(content)
        if parser.math:
            math[name] = parser.math
    manifest = {
        "math": math,
        "version": version,
        "pages": sorted(pages),
        "symbols": coverage,
        "source_digest": hashlib.sha256(
            b"".join(
                p.read_bytes()
                for p in sorted((source_root / "src/bayesmith").rglob("*.py"))
            )
        ).hexdigest(),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    validate_site(output)
    for identity, data in coverage.items():
        path, anchor = data["path"].split("#")
        parser = PageParser()
        parser.feed((output / path).read_text())
        if anchor not in parser.ids:
            raise ValueError(f"Missing API coverage for {identity}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate that committed output matches a fresh build",
    )
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="bayesmith-docs-") as temp:
            output = Path(temp)
            manifest = build(args.source_root, output, ROOT / "site")
            stale = [
                str(p.relative_to(output))
                for p in output.rglob("*")
                if p.is_file()
                and (
                    not (args.output / p.relative_to(output)).is_file()
                    or p.read_bytes()
                    != (args.output / p.relative_to(output)).read_bytes()
                )
            ]
            actual_pages = {
                str(p.relative_to(args.output))
                for p in args.output.rglob("*.html")
                if p.relative_to(args.output).parts[0] not in ("content", "diagrams")
            }
            stale.extend(sorted(actual_pages - set(manifest["pages"])))
            if stale:
                raise SystemExit("Documentation output is stale: " + ", ".join(stale))
            validate_site(args.output)
    else:
        manifest = build(args.source_root, args.output, ROOT / "site")
    print(
        f"Documentation: {len(manifest['pages'])} pages, {len(manifest['symbols'])} API entries; links, anchors and source coverage checked."
    )


if __name__ == "__main__":
    main()
