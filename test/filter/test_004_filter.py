import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from parq_tools import filter_parquet_file

def test_filter_parquet_file(tmp_path: Path):
    # Create a temporary Parquet file
    input_data = {
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
        "z": ["a", "b", "c", "d", "e"]
    }
    input_table = pa.Table.from_pydict(input_data)
    input_path = tmp_path / "input.parquet"
    pq.write_table(input_table, input_path)

    # Define the output path
    output_path = tmp_path / "output.parquet"

    # Apply the filter
    filter_expression = "x > 2 and y < 50"
    filter_parquet_file(input_path, output_path, filter_expression, columns=["x", "y"],
                        show_progress=True)

    # Read the output Parquet file
    output_table = pq.read_table(output_path)

    # Expected data
    expected_data = {
        "x": [3, 4],
        "y": [30, 40]
    }
    expected_table = pa.Table.from_pydict(expected_data)

    # Assert the output matches the expected result
    assert output_table.equals(expected_table)