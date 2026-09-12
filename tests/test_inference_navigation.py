"""A counterpart must never remove unrelated examples from navigation."""

import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from examples.inference.case_content import CASES
from examples.inference.presentation import gallery_navigation, observation_navigation


class Navigation(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.entries = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("a", "button"):
            self.entries.append((tag, attrs))


def test_all_baselines_and_selected_comparisons_survive_every_gallery(tmp_path):
    for case in CASES:
        (tmp_path / case).mkdir()
        (tmp_path / case / "result.json").write_text("{}")
    variants = {
        "jeffreys": ("power_law", "hierarchical", "bernoulli", "multiplicative_noise"),
        "proposals": ("linear_gaussian", "multiplicative_noise"),
        "mild-prior": ("composed_process",),
        "more-data": ("composed_process",),
    }
    for kind, cases in variants.items():
        for case in cases:
            (tmp_path / kind / case).mkdir(parents=True)
            (tmp_path / kind / case / "result.json").write_text("{}")
    unavailable = tmp_path / "jeffreys/composed_process/status.json"
    unavailable.parent.mkdir()
    unavailable.write_text("{}")
    for kind, cases in (("", CASES), *variants.items()):
        directory = tmp_path / kind
        reports = [({"case": case}, case) for case in cases]
        reports.append(({"case": "exponential_decay"}, "exponential_decay"))
        markup = gallery_navigation(reports, directory)
        nav = Navigation(markup)
        assert "Marginal Jeffreys: not implemented" not in markup
        baselines = [(tag, attrs) for tag, attrs in nav.entries
                     if attrs.get("class") == "case-button"]
        assert len(baselines) == 6
        for (tag, attrs), case in zip(baselines, CASES, strict=True):
            if kind:
                url = urlsplit(attrs["href"])
                assert tag == "a"
                assert "data-preserve-step" in attrs
                assert (directory / unquote(url.path)).resolve() == tmp_path / "index.html"
                assert f"case={case}" in url.fragment
            else:
                assert tag == "button" and attrs["data-select-case"] == case
        selectable = {a["data-select-case"] for _, a in nav.entries if "data-select-case" in a}
        assert selectable == (set() if kind in ("jeffreys", "more-data", "mild-prior") else set(cases))
        assert "with Jeffreys prior" not in markup
        assert "with mild noise prior" not in markup
        assert "3,840 observations" not in markup
        assert "exponential_decay" not in markup
        assert "Earlier exponential" not in markup
        for tag, attrs in nav.entries:
            if tag == "a":
                assert "data-gallery-link" in attrs  # Shared language handler.
                target = directory / unquote(urlsplit(attrs["href"]).path)
                assert target.name == "index.html" and Path(target.parent).exists()
                assert target.parent.name not in ("jeffreys", "more-data", "mild-prior")


def test_gallery_switch_preserves_step_in_local_and_counterpart_navigation():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to execute the gallery controller regression test")
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [node, str(root / "tests/gallery_navigation.cjs"),
         str(root / "examples/inference/gallery.js")],
        check=True, capture_output=True, text=True,
    )


def test_counterpart_without_baseline_never_links_to_missing_model(tmp_path):
    (tmp_path / "linear_gaussian").mkdir()
    (tmp_path / "linear_gaussian/result.json").write_text("{}")
    variant = tmp_path / "proposals"
    (variant / "composed_process").mkdir(parents=True)
    (variant / "composed_process/result.json").write_text("{}")
    nav = Navigation(gallery_navigation([({"case": "composed_process"}, "composed_process")], variant))
    assert any(attrs.get("data-select-case") == "composed_process" for _, attrs in nav.entries)
    assert not any(tag == "a" and "case=composed_process" in attrs["href"] for tag, attrs in nav.entries)


def test_real_observations_have_a_separate_section_and_remain_on_counterparts(tmp_path):
    case = tmp_path / "tris_haslam"
    case.mkdir()
    (case / "result.json").write_text("{}")
    reports = [({"case": "tris_haslam", "kind": "real_observations"}, "tris_haslam")]
    nav = observation_navigation(reports, tmp_path)
    assert "Real observations" in nav and "真实观测" in nav
    assert 'data-select-case="tris_haslam"' in nav
    assert "tris_haslam" not in gallery_navigation(reports, tmp_path)
    for kind in ("jeffreys", "proposals", "mild-prior", "more-data"):
        directory = tmp_path / kind
        directory.mkdir()
        nav = observation_navigation([], directory)
        assert "../index.html#case=tris_haslam" in nav
        assert "data-gallery-link" in nav and "data-preserve-step" in nav
    assert observation_navigation([], tmp_path / "unrelated") == ""
