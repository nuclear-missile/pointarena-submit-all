import json

from scripts.paths import ROOT


def test_clean_audit_report_exists():
    p = ROOT / "tests" / "clean_audit" / "counts_report.json"
    assert p.exists()
    data = json.load(open(p, "r", encoding="utf-8"))
    assert "dataset_reports" in data
    assert "all_rows_have_existing_image" in data


def test_visual_markdown_exists():
    p = ROOT / "tests" / "clean_audit" / "top10_visualization.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "pixmo_points" in text
