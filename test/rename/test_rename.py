import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import tempfile
from parq_tools.parq_schema_tools import rename_and_update_metadata

@pytest.fixture
def sample_parquet(tmp_path):
    data = pa.table({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [7, 8, 9]
    })
    file_path = tmp_path / "input.parquet"
    pq.write_table(data, file_path)
    return file_path

def test_rename_columns_all(sample_parquet, tmp_path):
    output = tmp_path / "output.parquet"
    rename_map = {"a": "x", "b": "y"}
    rename_and_update_metadata(sample_parquet, output, rename_map, show_progress=True)
    table = pq.read_table(output)
    assert set(table.column_names) == {"x", "y", "c"}

def test_rename_columns_selected(sample_parquet, tmp_path):
    output = tmp_path / "output_selected.parquet"
    rename_map = {"a": "x", "b": "y"}
    rename_and_update_metadata(sample_parquet, output, rename_map, return_all_columns=False)
    table = pq.read_table(output)
    assert set(table.column_names) == {"x", "y"}
    assert table.num_rows == 3

def test_chunking(sample_parquet, tmp_path):
    output = tmp_path / "output_chunked.parquet"
    rename_map = {"a": "x"}
    rename_and_update_metadata(sample_parquet, output, rename_map, chunk_size=1)
    table = pq.read_table(output)
    assert "x" in table.column_names
    assert table.num_rows == 3