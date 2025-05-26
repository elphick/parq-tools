import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc  # Add this import for compute functions


def test_parquet_file_structure(parquet_test_file_1):
    # Read the Parquet file
    df = pd.read_parquet(parquet_test_file_1)

    # Verify the structure of the DataFrame
    assert list(df.columns) == ["x", "y", "z", "a", "b", "c"]
    assert len(df) == 10

    # Verify chunking (row group size)
    parquet_file = pq.ParquetFile(parquet_test_file_1)
    row_groups = parquet_file.metadata.num_row_groups
    assert row_groups == 2  # 10 rows split into 2 chunks of 5 rows each


def test_wide_concat(parquet_test_file_1, parquet_test_file_2, parquet_test_file_3, tmp_path):
    # Define the output file path
    output_file = tmp_path / "wide_concat_output.parquet"

    # Open all files as datasets
    files = [parquet_test_file_1, parquet_test_file_2, parquet_test_file_3]
    scanners = [ds.dataset(file, format="parquet").scanner() for file in files]

    # Collect all tables from scanners
    tables = []
    for scanner in scanners:
        batches = scanner.to_batches()
        table = pa.Table.from_batches(batches)
        tables.append(table)

    # Validate alignment of index columns
    index_columns = ["x", "y", "z"]
    index_tables = [table.select(index_columns) for table in tables]
    for i in range(1, len(index_tables)):
        if not index_tables[0].equals(index_tables[i]):
            raise ValueError("Index columns are not aligned across datasets")

    # Combine tables horizontally
    combined_columns = []
    combined_schema_fields = []

    # Add index columns from the first table only
    combined_columns.extend(tables[0].select(index_columns).columns)
    combined_schema_fields.extend(tables[0].select(index_columns).schema)

    # Add supplementary columns from all tables
    for table in tables:
        for column_name in table.schema.names:
            if column_name not in index_columns:  # Avoid duplicating index columns
                combined_columns.append(table[column_name])
                combined_schema_fields.append(table.schema.field(column_name))

    # Create the combined table
    combined_schema = pa.schema(combined_schema_fields)
    combined_table = pa.Table.from_arrays(combined_columns, schema=combined_schema)

    # Write the combined table to the output file
    writer = None
    try:
        writer = pq.ParquetWriter(output_file, combined_table.schema)
        writer.write_table(combined_table)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 10  # Ensure the row count matches the input files
    assert df_output["x"].is_unique  # Ensure 'x' is unique across concatenated DataFrame


def test_wide_concat_row_filter(parquet_test_file_1, parquet_test_file_2, parquet_test_file_3, tmp_path):
    # Define the output file path
    output_file = tmp_path / "wide_concat_row_filter_output.parquet"

    # Open all files as datasets
    files = [parquet_test_file_1, parquet_test_file_2, parquet_test_file_3]
    datasets = [ds.dataset(file, format="parquet") for file in files]

    # Collect all tables from datasets
    tables = []
    for dataset in datasets:
        batches = dataset.scanner().to_batches()
        table = pa.Table.from_batches(batches)
        tables.append(table)

    # Combine tables horizontally
    combined_table = combine_tables_horizontally(tables)

    # Apply filtering to the merged table
    combined_table = apply_filters(combined_table, [pc.field("x") > 5])

    # Write the combined table to the output file
    writer = None
    try:
        writer = pq.ParquetWriter(output_file, combined_table.schema)
        writer.write_table(combined_table)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g"]
    assert list(df_output.columns) == expected_columns  # Ensure column order matches datasets
    assert all(df_output["x"] > 5)  # Ensure all rows satisfy the filter condition
    assert df_output["x"].is_unique  # Ensure 'x' is unique across concatenated DataFrame


def wide_concat_selected_columns(parquet_test_file_1, parquet_test_file_2, parquet_test_file_3, tmp_path):
    # Define the output file path
    output_file = tmp_path / "wide_concat_selected_output.parquet"

    # Specify the columns to include
    selected_columns = ["x", "y", "z", "a", "b", "c"]

    # Open all files as datasets and create scanners with selected columns
    files = [parquet_test_file_1, parquet_test_file_2, parquet_test_file_3]
    scanners = [
        ds.dataset(file, format="parquet").scanner(columns=selected_columns)
        for file in files
    ]

    # Collect all tables from scanners
    tables = []
    for scanner in scanners:
        batches = scanner.to_batches()
        table = pa.Table.from_batches(batches)
        tables.append(table)

    # Validate alignment of index columns
    index_columns = ["x", "y", "z"]
    index_tables = [table.select(index_columns) for table in tables]
    for i in range(1, len(index_tables)):
        if not index_tables[0].equals(index_tables[i]):
            raise ValueError("Index columns are not aligned across datasets")

    # Combine tables horizontally
    combined_columns = []
    combined_schema_fields = []

    # Add index columns from the first table only
    combined_columns.extend(tables[0].select(index_columns).columns)
    combined_schema_fields.extend(tables[0].select(index_columns).schema)

    # Add supplementary columns from all tables
    for table in tables:
        for column_name in table.schema.names:
            if column_name not in index_columns:  # Avoid duplicating index columns
                combined_columns.append(table[column_name])
                combined_schema_fields.append(table.schema.field(column_name))

    # Create the combined table
    combined_schema = pa.schema(combined_schema_fields)
    combined_table = pa.Table.from_arrays(combined_columns, schema=combined_schema)

    # Write the combined table to the output file
    writer = None
    try:
        writer = pq.ParquetWriter(output_file, combined_table.schema)
        writer.write_table(combined_table)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 10  # Ensure the row count matches the input files
    assert df_output["x"].is_unique  # Ensure 'x' is unique across concatenated DataFrame


def test_wide_concat_row_filter_and_column_selection(parquet_test_file_1, parquet_test_file_2, parquet_test_file_3,
                                                     tmp_path):
    # Define the output file path
    output_file = tmp_path / "wide_concat_row_filter_and_column_selection_output.parquet"

    # Specify the columns to include
    selected_columns = ["x", "y", "z", "a", "b", "d", "f"]

    # Open all files as datasets and dynamically adjust scanners
    files = [parquet_test_file_1, parquet_test_file_2, parquet_test_file_3]
    datasets = [ds.dataset(file, format="parquet") for file in files]

    # Collect all tables from datasets
    tables = []
    for dataset in datasets:
        # Dynamically adjust the selected columns based on the dataset's schema
        available_columns = [col for col in selected_columns if col in dataset.schema.names]
        batches = dataset.scanner(columns=available_columns).to_batches()
        table = pa.Table.from_batches(batches)
        tables.append(table)

    # Combine tables horizontally
    combined_table = combine_tables_horizontally(tables)

    # Apply filtering to the merged table
    combined_table = apply_filters(combined_table, [pc.field("x") > 5])

    # Write the combined table to the output file
    writer = None
    try:
        writer = pq.ParquetWriter(output_file, combined_table.schema)
        writer.write_table(combined_table)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "d", "f"]
    assert list(df_output.columns) == expected_columns  # Ensure column order matches selected columns
    assert all(df_output["x"] > 5)  # Ensure all rows satisfy the filter condition
    assert df_output["x"].is_unique  # Ensure 'x' is unique across concatenated DataFrame


def combine_tables_horizontally(tables):
    # Validate alignment of index columns
    index_columns = ["x", "y", "z"]
    index_tables = [table.select(index_columns) for table in tables]
    for i in range(1, len(index_tables)):
        if not index_tables[0].equals(index_tables[i]):
            raise ValueError("Index columns are not aligned across datasets")

    # Combine tables horizontally
    combined_columns = []
    combined_schema_fields = []

    # Add index columns from the first table only
    combined_columns.extend(tables[0].select(index_columns).columns)
    combined_schema_fields.extend(tables[0].select(index_columns).schema)

    # Add supplementary columns from all tables
    for table in tables:
        for column_name in table.schema.names:
            if column_name not in index_columns:  # Avoid duplicating index columns
                combined_columns.append(table[column_name])
                combined_schema_fields.append(table.schema.field(column_name))

    # Create the combined table
    combined_schema = pa.schema(combined_schema_fields)
    return pa.Table.from_arrays(combined_columns, schema=combined_schema)


def apply_filters(table, filters):
    """
    Apply a list of filters to a PyArrow table.
    :param table: The PyArrow table to filter.
    :param filters: A list of PyArrow filter expressions.
    :return: The filtered PyArrow table.
    """
    for filter_expr in filters:
        table = table.filter(filter_expr)
    return table
