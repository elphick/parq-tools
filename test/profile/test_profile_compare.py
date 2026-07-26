from pathlib import Path

import pandas as pd
import pytest

from parq_tools.parq_profile import compare_parquet_profiles


def _write_parquet(path: Path, data: dict) -> Path:
    pd.DataFrame(data).to_parquet(path)
    return path


def test_compare_parquet_profiles_two_files(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 9], "b": [4, 5, 6]})

    comparison = compare_parquet_profiles(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )

    html = comparison.to_html()
    assert isinstance(html, str)
    assert len(html) > 0


def test_compare_parquet_profiles_to_file(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1, 2, 9], "b": [4, 5, 6]})
    output_file = tmp_path / "comparison_report.html"

    comparison = compare_parquet_profiles(
        parquet_paths=[left, right],
        batch_size=1,
        show_progress=False,
        titles=["Left", "Right"],
    )
    comparison.to_file(output_file)

    assert output_file.exists(), "Comparison HTML report was not created via to_file."


def test_compare_parquet_profiles_three_files(tmp_path: Path):
    a = _write_parquet(tmp_path / "a.parquet", {"a": [1, 2, 3], "b": [4, 5, 6]})
    b = _write_parquet(tmp_path / "b.parquet", {"a": [1, 2, 3], "b": [4, 5, 7]})
    c = _write_parquet(tmp_path / "c.parquet", {"a": [1, 2, 4], "b": [4, 5, 7]})

    comparison = compare_parquet_profiles(
        parquet_paths=[a, b, c],
        batch_size=1,
        show_progress=False,
        titles=["A", "B", "C"],
    )

    html = comparison.to_html()
    assert isinstance(html, str)
    assert len(html) > 0


def test_compare_parquet_profiles_requires_two_or_three_files(tmp_path: Path):
    only = _write_parquet(tmp_path / "only.parquet", {"a": [1]})

    with pytest.raises(ValueError, match="exactly 2 or 3"):
        compare_parquet_profiles(parquet_paths=[only], show_progress=False)


def test_compare_parquet_profiles_validates_titles_length(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1]})

    with pytest.raises(ValueError, match="titles must have the same length"):
        compare_parquet_profiles(parquet_paths=[left, right], titles=["Only one"], show_progress=False)


def test_compare_parquet_profiles_validates_metadata_length(tmp_path: Path):
    left = _write_parquet(tmp_path / "left.parquet", {"a": [1]})
    right = _write_parquet(tmp_path / "right.parquet", {"a": [1]})

    with pytest.raises(ValueError, match="dataset_metadata must have the same length"):
        compare_parquet_profiles(
            parquet_paths=[left, right],
            dataset_metadata=[{"description": "one"}],
            show_progress=False,
        )
