import pytest
import pandas as pd
from pathlib import Path
from parq_tools.parq_concat import concat_parquet_files


def create_parquet_file(tmp_path, filename, data):
    file_path = tmp_path / filename
    pd.DataFrame(data).to_parquet(file_path, index=False)
    return file_path


def test_concat_parquet_files_tall(tmp_path):
    # Create two simple Parquet files
    f1 = create_parquet_file(tmp_path, "f1.parquet", {"a": [1, 2], "b": [3, 4]})
    f2 = create_parquet_file(tmp_path, "f2.parquet", {"a": [5, 6], "b": [7, 8]})

    output = tmp_path / "tall_concat.parquet"
    concat_parquet_files([f1, f2], output, axis=0)

    df = pd.read_parquet(output)
    assert len(df) == 4
    assert set(df.columns) == {"a", "b"}
    assert df["a"].tolist() == [1, 2, 5, 6]


def test_concat_parquet_files_wide(tmp_path):
    # Create two files with the same index column
    f1 = create_parquet_file(tmp_path, "f1.parquet", {"id": [1, 2], "a": [10, 20]})
    f2 = create_parquet_file(tmp_path, "f2.parquet", {"id": [1, 2], "b": [30, 40]})

    output = tmp_path / "wide_concat.parquet"
    concat_parquet_files([f1, f2], output, axis=1, index_columns=["id"])

    df = pd.read_parquet(output)
    assert set(df.columns) == {"id", "a", "b"}
    assert len(df) == 2
    assert df["id"].tolist() == [1, 2]


def test_concat_parquet_files_with_filter(tmp_path):
    f1 = create_parquet_file(tmp_path, "f1.parquet", {"a": [1, 2, 3], "b": [10, 20, 30]})
    f2 = create_parquet_file(tmp_path, "f2.parquet", {"a": [4, 5, 6], "b": [40, 50, 60]})

    output = tmp_path / "filtered_concat.parquet"
    concat_parquet_files([f1, f2], output, axis=0, filter_query="b > 30")

    df = pd.read_parquet(output)
    assert all(df["b"] > 30)
    assert set(df.columns) == {"a", "b"}


def test_concat_parquet_files_empty_input(tmp_path):
    output = tmp_path / "empty.parquet"
    with pytest.raises(ValueError, match="input files cannot be empty"):
        concat_parquet_files([], output)


def test_concat_parquet_files_missing_file(tmp_path):
    f1 = tmp_path / "does_not_exist.parquet"
    output = tmp_path / "fail.parquet"
    with pytest.raises(ValueError, match="File not found or inaccessible"):
        concat_parquet_files([f1], output)
