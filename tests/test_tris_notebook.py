"""Runnable notebook: structure, execution capture, markdown and HTML checks."""

from pathlib import Path

from examples.inference.tris_notebook import (
    build_notebook,
    default_configs,
    execute,
    markdown_to_html,
    validate,
)

NL = chr(10)


def test_built_notebook_is_valid_nbformat():
    notebook = build_notebook()
    assert notebook["nbformat"] == 4
    kinds = {cell["cell_type"] for cell in notebook["cells"]}
    assert kinds == {"markdown", "code"}
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all("source" in cell and "outputs" in cell for cell in code_cells)


def test_execute_captures_stdout_and_writes_figures(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": [
                "import matplotlib\n",
                "matplotlib.use('Agg')\n",
                "import matplotlib.pyplot as plt\n",
                "from pathlib import Path\n",
                "import os\n",
                "print('hello notebook')\n",
                "fig, ax = plt.subplots()\n",
                "ax.plot([0, 1], [0, 1])\n",
                "fig.savefig(Path(os.environ['TRIS_NOTEBOOK_OUTPUT']) / 'assets' / 'demo.svg')\n",
            ]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    sections = execute(
        notebook, environment={"TRIS_NOTEBOOK_OUTPUT": str(tmp_path), "TRIS_NOTEBOOK_RUNS": "{}"}
    )
    assert len(sections) == 1
    assert "hello notebook" in sections[0]["text"]
    assert sections[0]["images"] == ["demo.svg"]


def test_execute_keeps_markdown_cells_in_order(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["## Head" + NL]},
            {"cell_type": "code", "metadata": {}, "source": ["print(1)" + NL],
             "outputs": [], "execution_count": None},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    sections = execute(
        notebook, environment={"TRIS_NOTEBOOK_OUTPUT": str(tmp_path), "TRIS_NOTEBOOK_RUNS": "{}"}
    )
    assert [section["kind"] for section in sections] == ["markdown", "code"]
    assert sections[0]["source"] == "## Head" + NL


def test_markdown_to_html_renders_headings_lists_and_equations():
    source = "## Section one" + NL + NL + "- first" + NL + "- second" + NL + NL + "$$x = 1$$"
    rendered = markdown_to_html(source)
    assert "id='section-one'" in rendered
    assert "<li>first</li>" in rendered
    assert "<li>second</li>" in rendered
    assert "class='equation'" in rendered
    assert "x = 1" in rendered
    assert "##" not in rendered


def test_default_configs_records_rerun_entries():
    runs = {"b": Path("/tmp/b"), "budget": Path("/tmp/budget"), "beam": Path("/tmp/beam")}
    configs = default_configs(runs)
    assert "p0_budget" in configs
    assert "beam_surrogate" in configs
    assert "beam_identifiability" in configs
    assert configs["p0_budget"][:3] == [configs["p0_budget"][0], "-m",
                                        "examples.inference.tris_p0_budget"]


def test_execute_recaptures_a_rewritten_figure(tmp_path):
    notebook = {
        "cells": [
            {"cell_type": "code", "metadata": {}, "source": [
                "import matplotlib\n",
                "matplotlib.use('Agg')\n",
                "import matplotlib.pyplot as plt\n",
                "from pathlib import Path\n",
                "import os\n",
                "fig, ax = plt.subplots()\n",
                "ax.plot([0, 1], [1, 0])\n",
                "fig.savefig(Path(os.environ['TRIS_NOTEBOOK_OUTPUT']) / 'assets' / 'demo.svg')\n",
                "plt.close(fig)\n",
            ]},
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    environment = {"TRIS_NOTEBOOK_OUTPUT": str(tmp_path), "TRIS_NOTEBOOK_RUNS": "{}"}
    assert execute(notebook, environment=dict(environment))[0]["images"] == ["demo.svg"]
    # A second run into the same directory rewrites the same file; it must
    # still be reported, or the HTML loses every figure.
    assert execute(notebook, environment=dict(environment))[0]["images"] == ["demo.svg"]


def test_validate_flags_a_missing_image(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text(
        "<section class='markdown'><h2 id='x'>x</h2></section><img src='assets/nope.svg'>"
    )
    assert validate(tmp_path / "index.html", assets) == ["missing image: nope.svg"]


def test_validate_rejects_an_external_url(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text(
        "<section class='markdown'><h2 id='x'>x</h2></section>"
        "<a href='https://example.org'>x</a>"
    )
    assert validate(tmp_path / "index.html", assets) == ["page references an external URL"]


def test_validate_flags_unrendered_markdown(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (tmp_path / "index.html").write_text("## a heading that was not rendered")
    problems = validate(tmp_path / "index.html", assets)
    assert "no markdown section was rendered" in problems
    assert "markdown heading was not rendered" in problems


def test_sky_layers_separate_corrected_haslam_galactic_and_total_sky():
    import numpy as np

    from examples.inference.tris_notebook import posterior_sky_layers

    draws = {'amplitude': np.array([[2., 1., 1.]]), 'beta': np.array([[-2.6]*3]),
             'monopole_k': np.array([3.]), 'rsb_amplitude': np.array([.5]),
             'rsb_beta': np.array([-2.])}
    fields = posterior_sky_layers(draws, np.array([20.]), np.array([0]), 2.725)
    background = .5 / .408**2
    np.testing.assert_allclose(fields['H0_plus_zH'], [[23.]])
    np.testing.assert_allclose(fields['G408'], [[23.-background]])
    np.testing.assert_allclose(fields['T408'], [[2*(23.-background)+2.725+background]])
