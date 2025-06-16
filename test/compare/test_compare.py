import os
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq
from parq_tools.parq_compare import compare_parquet_files


def make_parquet_file(data, path, metadata=None):
    table = pa.table(data)
    if metadata is not None:
        # Attach metadata to the schema
        schema = table.schema.with_metadata(metadata)
        table = table.cast(schema)
    pq.write_table(table, path)

def test_identical_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
        make_parquet_file(data, f1)
        make_parquet_file(data, f2)
        result = compare_parquet_files(f1, f2)
        assert result["metadata"] is True
        assert all(result["columns"].values())

def test_different_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        make_parquet_file({"a": [1, 2, 3]}, f1)
        make_parquet_file({"a": [1, 2, 4]}, f2)
        result = compare_parquet_files(f1, f2)
        assert result["metadata"] is True
        assert result["columns"]["a"] is False

def test_different_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        data = {"a": [1]}
        make_parquet_file(data, f1, metadata={b"foo": b"bar"})
        make_parquet_file(data, f2, metadata={b"foo": b"baz"})
        result = compare_parquet_files(f1, f2)
        assert result["metadata"] is False
        assert all(result["columns"].values())

def test_missing_column():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        make_parquet_file({"a": [1]}, f1)
        make_parquet_file({"a": [1], "b": [2]}, f2)
        result = compare_parquet_files(f1, f2)
        assert result["columns"]["b"] is False
        assert result["columns"]["a"] is True

def test_different_row_counts():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        make_parquet_file({"a": [1, 2]}, f1)
        make_parquet_file({"a": [1, 2, 3]}, f2)
        result = compare_parquet_files(f1, f2)
        assert result["num_rows_match"] is False
        assert result["num_rows_left"] == 2
        assert result["num_rows_right"] == 3

def test_empty_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        make_parquet_file({}, f1)
        make_parquet_file({}, f2)
        result = compare_parquet_files(f1, f2)
        assert result["metadata"] is True
        assert all(result["columns"].values())
        assert result["num_rows_match"] is True
        assert result["num_rows_left"] == 0
        assert result["num_rows_right"] == 0

def test_partial_column_match():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        make_parquet_file({"a": [1, 2], "b": [3, 4]}, f1)
        make_parquet_file({"a": [1, 2], "c": [5, 6]}, f2)
        result = compare_parquet_files(f1, f2)
        assert result["columns"]["a"] is True
        assert result["columns"]["b"] is False
        assert result["columns"]["c"] is False
        assert result["missing_columns"]["left_only"] == ["b"]
        assert result["missing_columns"]["right_only"] == ["c"]

def test_large_files_with_progress():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        data1 = {"a": list(range(100000)), "b": list(range(100000, 200000))}
        data2 = {"a": list(range(100000)), "b": list(range(100000, 200000))}
        make_parquet_file(data1, f1)
        make_parquet_file(data2, f2)
        result = compare_parquet_files(f1, f2, chunk_size=10000, show_progress=True)
        assert result["metadata"] is True
        assert all(result["columns"].values())
        assert result["num_rows_match"] is True
        assert result["num_rows_left"] == 100000
        assert result["num_rows_right"] == 100000

def test_different_column_types():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "a.parquet")
        f2 = os.path.join(tmpdir, "b.parquet")
        make_parquet_file({"a": [1, 2, 3]}, f1)
        make_parquet_file({"a": ["1", "2", "3"]}, f2)  # Different type (int vs str)
        result = compare_parquet_files(f1, f2)
        assert result["metadata"] is False
        assert result["columns"]["a"] is False
        assert result["dtypes"]["a"]["left"] == "int64"
        assert result["dtypes"]["a"]["right"] == "string"