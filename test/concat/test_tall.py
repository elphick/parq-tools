import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds  # Ensure this import is included
import pytest

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
    concat.concat_to_file(output_file, show_progress=True)

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
    assert len(df_output) == 5  # Ensure the row count matches the filter


def test_tall_concat_with_non_index_filter(parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_non_index_filtered_output.parquet"

    # Initialize the ParquetConcat object
    concat = ParquetConcat(
        files=[parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13],
        axis=0,
        index_columns=["x", "y", "z"]
    )

    # Perform tall concatenation with a filter on a non-index column
    filter_query = "(b > 30)"
    concat.concat_to_file(output_file, filter_query=filter_query)

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c", "d"]
    assert list(df_output.columns) == expected_columns

    # Verify the row count matches the filter
    # Rows with `b > 30` are from `parquet_tall_file_12` (b = 32, 34, 36, 38, 40)
    # and `parquet_tall_file_13` (b = 42, 44, 46, 48, 50, 52, 54, 56, 58, 60)
    assert len(df_output) == 15


def test_tall_concat_filter_on_missing_column(tmp_path):
    import pandas as pd
    from parq_tools.parq_concat import ParquetConcat

    # File 1: no column 'b'
    df1 = pd.DataFrame({"x": [1, 2, 3], "a": ["A", "B", "C"]})
    # File 2: has column 'b'
    df2 = pd.DataFrame({"x": [4, 5, 6], "b": [10, 20, 30]})

    f1 = tmp_path / "file1.parquet"
    f2 = tmp_path / "file2.parquet"
    df1.to_parquet(f1, index=False)
    df2.to_parquet(f2, index=False)

    concat = ParquetConcat(
        files=[f1, f2],
        axis=0,
        index_columns=["x"]
    )
    output_file = tmp_path / "tall_concat_missing_col_filter.parquet"

    # This should fail: filter references 'b', which is missing in f1
    filter_query = "b > 15"
    with pytest.raises(ValueError,
                       match="The filter query references columns that are missing in one or more datasets"):
        concat.concat_to_file(output_file, filter_query=filter_query)


def test_tall_concat_against_pandas(parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13, tmp_path):
    # Define the output file paths
    output_file = tmp_path / "tall_concat_output.parquet"

    # Initialize the ParquetConcat object
    concat = ParquetConcat(
        files=[parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13],
        axis=0,
        index_columns=["x", "y", "z"]
    )

    # Perform wide concatenation with chunked processing
    concat.concat_to_file(output_file, batch_size=5, show_progress=False)

    # Read the output files
    df_concat = pd.read_parquet(output_file).set_index(["x", "y", "z"])

    # concat with pandas
    df_pandas = pd.concat(
        [pd.read_parquet(file).set_index(['x', 'y', 'z']) for file in
         [parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13]],
        axis=0,
        ignore_index=False
    )

    # Compare the DataFrames
    pd.testing.assert_frame_equal(df_concat, df_pandas)


def test_tall_concat_preserves_pandas_metadata(tmp_path):
    # Create DataFrames with pandas extension dtypes
    df1 = pd.DataFrame({
        "x": pd.Series([1, 2, 3], dtype="Int64"),
        "a": pd.Series(["A", "B", "C"], dtype="string"),
    })
    df2 = pd.DataFrame({
        "x": pd.Series([4, 5, 6], dtype="Int64"),
        "a": pd.Series(["D", "E", "F"], dtype="string"),
    })

    # Write to Parquet with pandas metadata
    file1 = tmp_path / "file1.parquet"
    file2 = tmp_path / "file2.parquet"
    df1.to_parquet(file1, index=False)
    df2.to_parquet(file2, index=False)

    # Tall concatenate
    output_file = tmp_path / "tall_concat.parquet"
    ParquetConcat([file1, file2], axis=0, index_columns=["x"]).concat_to_file(output_file)

    # Read result and check dtypes
    result = pd.read_parquet(output_file)
    assert result["a"].dtype == "string"
    assert result["x"].dtype == "Int64"
    assert list(result.columns) == ["x", "a"]
    assert len(result) == 6

    # Check for pandas metadata in the output Parquet file
    parquet_file = pq.ParquetFile(output_file)
    metadata = parquet_file.schema_arrow.metadata
    assert metadata is not None and b"pandas" in metadata, "Missing pandas metadata in output Parquet file"