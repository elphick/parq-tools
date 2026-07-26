from pathlib import Path

import pandas as pd
import pytest

from parq_tools.parq_profile import build_parquet_profile_comparison


def _write_parquet(path: Path, data: dict) -> Path:
    pd.DataFrame(data).to_parquet(path)
    return path


def test_build_bundle_and_summary(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 9], "b": [4.0, 5.0, 6.0]})
    bundle = build_parquet_profile_comparison(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )
    summary = bundle.to_summary_dict(metrics=["mean", "max"])
    assert summary["labels"] == ["Left", "Right"]
    assert summary["columns"]["a"]["status"] == "different"
    assert summary["columns"]["b"]["status"] == "equal"


def test_diff_report_uses_cached_descriptions(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 9], "b": [4, 5, 6]})
    bundle = build_parquet_profile_comparison(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )
    diff_report = bundle.to_diff_report(metrics=["mean", "max"])
    diff_columns = set(diff_report.get_description().variables.keys())
    assert "a" in diff_columns
    assert "b" not in diff_columns


def test_write_outputs_and_yaml(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 4], "b": [4, 5, 6]})
    bundle = build_parquet_profile_comparison(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )
    written = bundle.write_outputs(
        comparison_html=tmp_path / "all_cols.html",
        diff_html=tmp_path / "diff_cols.html",
        differences_yaml=tmp_path / "differences.yaml",
        metrics=["mean", "max"],
    )
    assert written["comparison_html"].exists()
    assert written["diff_html"].exists()
    assert written["differences_yaml"].exists()


def test_diff_report_with_no_differences(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    bundle = build_parquet_profile_comparison(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )
    diff_report = bundle.to_diff_report(metrics=["mean", "max"])
    description = diff_report.get_description()
    assert description.analysis.title == "Comparing Left and Right"
    assert description.variables == {}
    assert len(diff_report.to_html()) > 0


def test_status_prefixes_applied_to_comparison_and_diff_reports(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 9], "b": [4, 5, 6]})
    bundle = build_parquet_profile_comparison(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
        column_descriptions={"a": "alpha", "b": "beta"},
    )

    comparison = bundle.to_comparison_report(description_status_labels="emoji")
    comparison_descriptions = comparison.config.variables.descriptions
    assert comparison_descriptions["a"].startswith("🔴 DIFF | ")
    assert comparison_descriptions["b"].startswith("🟢 SAME | ")

    diff_report = bundle.to_diff_report(description_status_labels="emoji")
    diff_descriptions = diff_report.config.variables.descriptions
    assert diff_descriptions["a"].startswith("🔴 DIFF | ")
    assert "b" not in diff_report.get_description().variables


def test_status_prefixes_apply_without_existing_descriptions(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 9], "b": [4, 5, 6]})
    bundle = build_parquet_profile_comparison(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )
    comparison = bundle.to_comparison_report(description_status_labels="emoji")
    descriptions = comparison.config.variables.descriptions
    assert descriptions["a"] == "🔴 DIFF"
    assert descriptions["b"] == "🟢 SAME"


def test_status_prefix_mode_validation(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 4]})
    bundle = build_parquet_profile_comparison(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )

    with pytest.raises(ValueError, match="description_status_labels must be 'none' or 'emoji'"):
        bundle.to_comparison_report(description_status_labels="invalid")


def test_write_outputs_adds_tolerance_footer(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 4]})
    bundle = build_parquet_profile_comparison(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )
    written = bundle.write_outputs(
        comparison_html=tmp_path / "comparison.html",
        diff_html=tmp_path / "diff.html",
        abs_tol=0.01,
        rel_tol=0.001,
        metrics=["mean", "max"],
    )
    comparison_html = written["comparison_html"].read_text(encoding="utf-8")
    diff_html = written["diff_html"].read_text(encoding="utf-8")
    assert "Comparison tolerance settings: abs_tol=0.01, rel_tol=0.001, metrics=[mean, max]" in comparison_html
    assert "Comparison tolerance settings: abs_tol=0.01, rel_tol=0.001, metrics=[mean, max]" in diff_html
