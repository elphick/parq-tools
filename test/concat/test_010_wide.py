import pandas as pd
import pyarrow.parquet as pq
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
    concat.concat_to_file(output_file, batch_size=5, show_progress=True)

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


def test_wide_concat_single_index_column(tmp_path):
    # Create two Parquet files with the same single index column
    df1 = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
    df2 = pd.DataFrame({"id": [1, 2], "b": [30, 40]})
    f1 = tmp_path / "f1.parquet"
    f2 = tmp_path / "f2.parquet"
    df1.to_parquet(f1, index=False)
    df2.to_parquet(f2, index=False)

    output_file = tmp_path / "wide_concat_output.parquet"

    concat = ParquetConcat(
        files=[f1, f2],
        axis=1,
        index_columns=["id"]
    )

    concat.concat_to_file(output_file)
    df = pd.read_parquet(output_file)  # Read the concatenated file
    # Verify the structure of the concatenated DataFrame
    expected_columns = ["id", "a", "b"]
    assert list(df.columns) == expected_columns


def test_wide_concat_against_pandas(parquet_test_file_1, parquet_test_file_2, parquet_test_file_3, tmp_path):
    # Define the output file paths
    output_file = tmp_path / "wide_concat_output.parquet"

    # Initialize the ParquetConcat object
    concat = ParquetConcat(
        files=[parquet_test_file_1, parquet_test_file_2, parquet_test_file_3],
        axis=1,
        index_columns=["x", "y", "z"]
    )

    # Perform wide concatenation with chunked processing
    concat.concat_to_file(output_file, batch_size=5, show_progress=False)

    # Read the output files
    df_concat = pd.read_parquet(output_file).set_index(["x", "y", "z"])

    # concat with pandas
    df_pandas = pd.concat(
        [pd.read_parquet(file).set_index(['x', 'y', 'z']) for file in
         [parquet_test_file_1, parquet_test_file_2, parquet_test_file_3]],
        axis=1,
        ignore_index=False
    )

    # Compare the DataFrames
    pd.testing.assert_frame_equal(df_concat, df_pandas)


def test_wide_concat_preserves_pandas_metadata(tmp_path):
    # Create DataFrames with pandas extension dtypes
    df1 = pd.DataFrame({
        "x": pd.Series([1, 2, 3], dtype="Int64"),
        "y": pd.Series([4, 5, 6], dtype="Int64"),
        "a": pd.Series(["A", "B", "C"], dtype="string"),
    })
    df2 = pd.DataFrame({
        "x": pd.Series([1, 2, 3], dtype="Int64"),
        "y": pd.Series([4, 5, 6], dtype="Int64"),
        "b": pd.Series([0.1, 0.2, 0.3], dtype="float32"),
    })

    # Write to Parquet with pandas metadata
    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"
    df1.to_parquet(file1, index=False)
    df2.to_parquet(file2, index=False)

    # Wide concatenate
    output_file = tmp_path / "wide_concat.parquet"
    ParquetConcat([file1, file2], axis=1, index_columns=["x", "y"]).concat_to_file(output_file)

    # Read result and check dtypes
    result = pd.read_parquet(output_file)
    assert result["a"].dtype == "string"
    assert result["x"].dtype == "Int64"
    assert result["y"].dtype == "Int64"
    assert result["b"].dtype == "float32"

    # Check for pandas metadata in the output Parquet file
    parquet_file = pq.ParquetFile(output_file)
    metadata = parquet_file.schema_arrow.metadata
    assert metadata is not None and b"pandas" in metadata, "Missing pandas metadata in output Parquet file"