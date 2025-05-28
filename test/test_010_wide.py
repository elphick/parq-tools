import pandas as pd
from parq_tools.parq_concat import ParquetConcat


def test_wide_concat(parquet_test_file_1, parquet_test_file_2, parquet_test_file_3, tmp_path):
    # Define the output file path
    output_file = tmp_path / "wide_concat_output.parquet"

    # Initialize the ParquetConcat object
    concat = ParquetConcat(
        files=[parquet_test_file_1, parquet_test_file_2, parquet_test_file_3],
        axis=1,
        index_columns=["x", "y", "z"]
    )

    # Perform wide concatenation
    concat.concat_to_file(output_file, batch_size=5)

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 10  # Ensure the row count matches the input files
    assert df_output["x"].is_unique


def test_wide_concat_with_row_filter(parquet_test_file_1, parquet_test_file_2, parquet_test_file_3, tmp_path):
    # Define the output file path
    output_file = tmp_path / "wide_concat_filtered_output.parquet"

    # Initialize the ParquetConcat object
    concat = ParquetConcat(
        files=[parquet_test_file_1, parquet_test_file_2, parquet_test_file_3],
        axis=1,
        index_columns=["x", "y", "z"]
    )

    # Adjusted filter query to match the test data
    filter_query = "(x > 3) and (y <= 15)"
    concat.concat_to_file(output_file, filter_query=filter_query, batch_size=5)

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 2  # Ensure the row count matches the filter

def test_wide_concat_with_column_filter(parquet_test_file_1, parquet_test_file_2, parquet_test_file_3, tmp_path):
    # Define the output file path
    output_file = tmp_path / "wide_concat_with_column_filter_output.parquet"

    # Specify the columns to include
    selected_columns = ["a", "b", "c"]

    # Initialize the ParquetConcat object
    concat = ParquetConcat(
        files=[parquet_test_file_1, parquet_test_file_2, parquet_test_file_3],
        axis=1,
        index_columns=["x", "y", "z"]
    )

    # Perform wide concatenation with column filtering
    concat.concat_to_file(output_file, columns=selected_columns, batch_size=5)

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z"] + selected_columns
    assert list(df_output.columns) == expected_columns  # Ensure column order matches selected columns
    assert len(df_output) == 10  # Ensure the row count matches the input files
