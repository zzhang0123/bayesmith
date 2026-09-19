"""Repository documentation contracts, outside the portable package suite."""

def test_the_artifact_docs_pin_the_five_results_and_grounds():
    """docs/artifacts.md is a published module-spec, so a lightweight guard
    pins the protocol's own names and the frozen Refusal field -- not the prose.

    The tables are a design decision to edit freely; what must not silently
    drift is the five Result names and the field 0 ruling 3 froze as grounds
    (never evidence).
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docs/artifacts.md").read_text(encoding="utf-8")
    for name in (
        "PosteriorResult",
        "EvidenceResult",
        "PredictiveResult",
        "PointEstimateResult",
        "SimulationResult",
    ):
        assert name in text, name
    assert "`grounds`" in text


def test_the_readme_workflow_mentions_the_typed_round_trip():
    """README must show the typed chain and say the legacy entry points stay."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "README.md").read_text(encoding="utf-8")
    for token in ("PosteriorTask", "compile_task", "execute_task", "PosteriorResult"):
        assert token in text, token
    assert "legacy" in text.lower()
