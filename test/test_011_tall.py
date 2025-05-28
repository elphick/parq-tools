import pandas as pd
import pyarrow.dataset as ds  # Ensure this import is included
from parq_tools.parq_concat import ParquetConcat


def test_tall_concat(parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_output.parquet"

    # Initialize the ParquetConcat object
    concat = ParquetConcat(
        files=[parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13],
        axis=0,
        index_columns=["x", "y", "z"]
    )

    # Perform tall concatenation
    concat.concat_to_file(output_file)

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c", "d"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 30  # Ensure the row count matches the input files


def test_tall_concat_with_filter(parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_filtered_output.parquet"

    # Initialize the ParquetConcat object
    concat = ParquetConcat(
        files=[parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13],
        axis=0,
        index_columns=["x", "y", "z"]
    )

    # Perform tall concatenation with a filter
    filter_query = "(x > 15) and (z <= 50)"
    concat.concat_to_file(output_file, filter_query=filter_query)

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c", "d"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 15  # Ensure the row count matches the filter
