import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from parq_tools import reindex_parquet

from parq_tools import sort_parquet_file
from parq_tools.utils.index_utils import dedup_index_parquet


def test_sort_parquet_file(parquet_unsorted_file, tmp_path):
    # Define the output path for the sorted file
    sorted_file_path = tmp_path / "sorted_data.parquet"

    # Sort the unsorted Parquet file
    sort_parquet_file(parquet_unsorted_file, sorted_file_path, ["x", "y", "z"])

    # Read the sorted file and verify it is sorted
    sorted_table = pq.read_table(sorted_file_path)
    sorted_data: pd.DataFrame = sorted_table.to_pandas()

    # Check if the data is sorted by the specified columns
    assert sorted_data.equals(sorted_data.sort_values(by=["x", "y", "z"]).reset_index(drop=True))




def test_reindex_parquet(tmp_path: Path):

    # Create a temporary sparse Parquet file
    sparse_data = {
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    }
    sparse_table = pa.Table.from_pydict(sparse_data)
    sparse_parquet_path = tmp_path / "sparse.parquet"
    pq.write_table(sparse_table, sparse_parquet_path)

    # Create a new index as a PyArrow table
    new_index_data = {
        "id": [1, 2, 3, 4],
    }
    new_index = pa.Table.from_pydict(new_index_data)

    # Define the output path
    output_path = tmp_path / "reindexed.parquet"

    # Call the function
    reindex_parquet(sparse_parquet_path, output_path=output_path, new_index=new_index)

    # Read the output Parquet file
    reindexed_table = pq.read_table(output_path)

    # Expected data
    expected_data = {
        "id": [1, 2, 3, 4],
        "value": [10, 20, 30, None]
    }
    expected_table = pa.Table.from_pydict(expected_data)

    # Assert the output matches the expected result
    assert reindexed_table.equals(expected_table)

def test_reindex_parquet_with_index(tmp_path: Path):
    # Create a temporary sparse Parquet file with an index
    sparse_data = {
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    }
    sparse_table = pa.Table.from_pydict(sparse_data)
    sparse_parquet_path = tmp_path / "sparse_with_index.parquet"
    pq.write_table(sparse_table, sparse_parquet_path)

    # Create a new index as a PyArrow table
    new_index_data = {
        "id": [1, 2, 3, 4],
    }
    new_index = pa.Table.from_pydict(new_index_data)

    # Define the output path
    output_path = tmp_path / "reindexed_with_index.parquet"

    # Call the function
    reindex_parquet(sparse_parquet_path, output_path=output_path, new_index=new_index)

    # Read the output Parquet file
    reindexed_table = pq.read_table(output_path)

    # Expected data
    expected_data = {
        "id": [1, 2, 3, 4],
        "value": [10, 20, 30, None]
    }
    expected_table = pa.Table.from_pydict(expected_data)

    # Assert the output matches the expected result
    assert reindexed_table.equals(expected_table)

def test_deduplicate_parquet(tmp_path: Path):
    # Create a temporary Parquet file with duplicate rows
    data = {
        "id": [1, 2, 2, 3],
        "value": [10, 20, 20, 30]
    }
    table = pa.Table.from_pydict(data)
    parquet_path = tmp_path / "data_with_duplicates.parquet"
    pq.write_table(table, parquet_path)

    # Define the output path for the deduplicated file
    deduped_file_path = tmp_path / "deduplicated_data.parquet"

    # Call the function to deduplicate
    dedup_index_parquet(parquet_path, output_path=deduped_file_path, index_columns=["id"])

    # Read the deduplicated file and verify it has no duplicates
    deduped_table = pq.read_table(deduped_file_path)
    deduped_data: pd.DataFrame = deduped_table.to_pandas()

    # Check if the data is deduplicated
    assert len(deduped_data) == 3  # Should have 3 unique rows
    assert deduped_data["id"].tolist() == [1, 2, 3]

