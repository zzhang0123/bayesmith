"""P0a checks: the archive data contract is parsed independently and tagged."""

import pytest

from examples.inference.tris_data_contract import (
    LedgerEntry,
    archive_facts,
    build_ledger,
    ledger_document,
    parse_ra_deg,
    render_markdown,
)

PROVENANCE = {
    "archived-confirmed",
    "paper-confirmed",
    "implementation-convention",
    "unconfirmable",
}

RING_HEADER = [
    "# Absolute Sky Brightness Temperature Profile as observed by TRIS at Declination=+42 Degrees",
    "# Frequency = {ghz} GHz",
    "# Systematic Zero Level Uncertainty = {zero}",
    "# NOTE2 = the E-plane is tilted 7 degrees Eastwards with respect to the meridian",
]


def write_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    body600 = [
        header.format(ghz="0.6", zero="0.066K") for header in RING_HEADER
    ] + ["0h00m   15.145   0.004", "0h11m   15.100   0.005", "0h23m   14.900   0.006"]
    (root / "TRIS_absolute_600.txt").write_text("\n".join(body600) + "\n")
    body820 = [
        header.format(
            ghz="0.82",
            zero="+0.430K/-0.300K (see the Reference for error bar asymmetry)",
        )
        for header in RING_HEADER
    ] + ["0h00m   9.145   0.004", "0h11m   9.100   0.005", "0h23m   8.900   0.006"]
    (root / "TRIS_absolute_820.txt").write_text("\n".join(body820) + "\n")
    # Bare CR record separators, as in the real public product.
    (root / "TRIS_absolute_2500MHz.txt").write_text(
        "# TRIS Absolute Sky Temperature at 2.5 GHz\r"
        "# Column 3 = Zero Level uncertainty in K\r"
        "11h26m04s 2.329 0.284\r"
        "13h42m32s 2.331 0.284"
    )
    (root / "TRIS_Beam_Profile.txt").write_text(
        "# TRIS beam profile is the same at the three frequencies (0.6, 0.82 and 2.5 GHz)\r"
        "0 0.0 0.0\r"
        "9 -1.725 -2.7\r"
        "11 -2.65 -3.775\r"
        "13 -3.705 -4.84\r"
        "176 -48.6 -47.8"
    )
    return root


def test_ra_token_parser_matches_the_archive_spelling():
    assert parse_ra_deg("0h00m") == pytest.approx(0.0)
    assert parse_ra_deg("0h11m") == pytest.approx(2.75)
    assert parse_ra_deg("11h26m04s") == pytest.approx(15.0 * (11 + 26 / 60 + 4 / 3600))
    with pytest.raises(ValueError, match="right-ascension token"):
        parse_ra_deg("11:26")
    with pytest.raises(ValueError, match="out of range"):
        parse_ra_deg("24h00m")


def test_bare_cr_records_and_ring_facts(tmp_path):
    root = write_fixture(tmp_path / "archive")
    facts = archive_facts(root)
    assert facts["point_set"]["rows"] == 2
    assert facts["point_set"]["common_zero_level_k"] == pytest.approx(0.284)
    ring600, ring820 = facts["rings"]
    assert ring600["rows"] == 3
    assert ring600["effective_frequency_mhz"] == pytest.approx(600.5)
    assert ring600["ra_spacing_deg"] == [2.75, 3.0]
    assert ring600["zero_level"] == pytest.approx(0.066)
    assert ring820["zero_level"] == {
        "positive_k": pytest.approx(0.430),
        "negative_k": pytest.approx(0.300),
    }
    assert facts["beam"]["rows"] == 5
    assert facts["beam"]["e_fwhm_deg"] == pytest.approx(19.1546, abs=1e-3)
    assert facts["beam"]["h_fwhm_deg"] == pytest.approx(23.3661, abs=1e-3)


def test_ledger_is_fully_tagged_and_renders(tmp_path):
    root = write_fixture(tmp_path / "archive")
    entries, _ = build_ledger(root)
    assert entries and all(isinstance(entry, LedgerEntry) for entry in entries)
    assert {entry.provenance for entry in entries} <= PROVENANCE
    stages = {entry.stage for entry in entries}
    assert {"raw", "correction", "forward", "random_error", "common_systematic"} <= stages
    # Every unconfirmable row must say what to do about the unknown.
    for entry in entries:
        if entry.provenance == "unconfirmable":
            assert entry.sensitivity
    document = ledger_document(root)
    assert document["provenance_counts"]["archived-confirmed"] >= 1
    text = render_markdown(document)
    assert "P0a" in text and "provenance" in text
    assert "|" in text


def test_missing_or_inconsistent_archive_fails_loudly(tmp_path):
    root = write_fixture(tmp_path / "archive")
    (root / "TRIS_absolute_600.txt").write_text("# Frequency = 0.9 GHz\n0h00m 1.0 0.1\n")
    with pytest.raises(ValueError, match="unsupported ring frequency"):
        archive_facts(root)
