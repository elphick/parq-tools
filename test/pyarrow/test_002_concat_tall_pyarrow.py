import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc  # Add this import for compute functions


def test_parquet_file_structure(parquet_tall_file_11):
    # Read the Parquet file
    df = pd.read_parquet(parquet_tall_file_11)

    # Verify the structure of the DataFrame
    assert list(df.columns) == ["x", "y", "z", "a", "b", "c"]
    assert len(df) == 10

    # Verify chunking (row group size)
    parquet_file = pq.ParquetFile(parquet_tall_file_11)
    row_groups = parquet_file.metadata.num_row_groups
    assert row_groups == 2  # 10 rows split into 2 chunks of 5 rows each


def test_tall_concat(parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_output.parquet"

    # Open all files as datasets
    files = [parquet_tall_file_11, parquet_tall_file_12]

    scanners = [ds.dataset(file, format="parquet").scanner() for file in files]

    # Initialize the writer
    writer = None

    try:
        # Iterate through batches from all scanners
        for batches in zip(*[scanner.to_batches() for scanner in scanners]):
            # Convert RecordBatch objects to Table objects
            tables = [pa.Table.from_batches([batch]) for batch in batches]

            # Concatenate tables vertically
            combined_batch = pa.concat_tables(tables)

            # Write the combined batch to the output file
            if writer is None:
                writer = pq.ParquetWriter(output_file, combined_batch.schema)
            writer.write_table(combined_batch)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c"]

    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 20  # Ensure the row count matches the input files


def test_tall_concat_with_row_filter(parquet_tall_file_11, parquet_tall_file_12, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_filtered_output.parquet"

    # Open all files as datasets
    files = [parquet_tall_file_11, parquet_tall_file_12]
    datasets = [ds.dataset(file, format="parquet") for file in files]

    # Initialize the writer
    writer = None

    try:
        # Create scanners for all datasets
        scanners = [dataset.scanner() for dataset in datasets]

        # Iterate through batches from all scanners
        for batches in zip(*[scanner.to_batches() for scanner in scanners]):
            # Filter rows where 'a' > 0
            filtered_batches = []
            for batch in batches:
                table = pa.Table.from_batches([batch])
                filtered_table = table.filter(pc.field("b") > 10)
                filtered_batches.append(filtered_table)

            # Concatenate tables vertically
            combined_batch = pa.concat_tables(filtered_batches)

            # Write the combined batch to the output file
            if writer is None:
                writer = pq.ParquetWriter(output_file, combined_batch.schema)
            writer.write_table(combined_batch)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 15  # 5 for first ds, 10 for second ds


def test_tall_concat_with_scanner_level_filtering(parquet_tall_file_11, parquet_tall_file_12, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_scanner_filtered_output.parquet"

    # Open all files as datasets
    files = [parquet_tall_file_11, parquet_tall_file_12]
    datasets = [ds.dataset(file, format="parquet") for file in files]

    # Initialize the writer
    writer = None

    try:
        # Create scanners for all datasets with a filter
        scanners = [dataset.scanner(filter=pc.field("x") > 10) for dataset in datasets]

        # Process all batches from all scanners
        all_batches = []
        for scanner in scanners:
            for batch in scanner.to_batches():
                table = pa.Table.from_batches([batch])
                all_batches.append(table)

        # Concatenate all tables vertically
        combined_batch = pa.concat_tables(all_batches)

        # Write the combined batch to the output file
        writer = pq.ParquetWriter(output_file, combined_batch.schema)
        writer.write_table(combined_batch)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 10  # Ensure the row count matches the input files after filtering

def test_tall_concat_with_scanner_and_table_level_filtering(parquet_tall_file_11, parquet_tall_file_12, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_scanner_table_filtered_output.parquet"

    # Open all files as datasets
    files = [parquet_tall_file_11, parquet_tall_file_12]
    datasets = [ds.dataset(file, format="parquet") for file in files]

    # Initialize the writer
    writer = None

    try:
        # Create scanners for all datasets with a filter
        scanners = [dataset.scanner(filter=pc.field("x") > 10) for dataset in datasets]

        # Process all batches from all scanners
        all_batches = []
        for scanner in scanners:
            for batch in scanner.to_batches():
                table = pa.Table.from_batches([batch])
                filtered_table = table.filter(pc.field("b") > 30)  # Additional filtering at table level
                all_batches.append(filtered_table)

        # Concatenate all tables vertically
        combined_batch = pa.concat_tables(all_batches)

        # Write the combined batch to the output file
        writer = pq.ParquetWriter(output_file, combined_batch.schema)
        writer.write_table(combined_batch)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 5  # Ensure the row count matches the input files after filtering



def test_tall_concat_with_missing_columns(parquet_tall_file_11, parquet_tall_file_12, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_output.parquet"

    # Open all files as datasets
    files = [parquet_tall_file_11, parquet_tall_file_12]
    datasets = [ds.dataset(file, format="parquet") for file in files]

    # Determine the union of all schemas
    all_schemas = [dataset.schema for dataset in datasets]
    unified_schema = pa.unify_schemas(all_schemas)

    # Initialize the writer
    writer = None

    try:
        # Create scanners for all datasets
        scanners = [dataset.scanner() for dataset in datasets]

        # Iterate through batches from all scanners
        for batches in zip(*[scanner.to_batches() for scanner in scanners]):
            # Align each batch to the unified schema
            aligned_batches = []
            for batch in batches:
                table = pa.Table.from_batches([batch])

                # Add missing columns with null values
                for field in unified_schema:
                    if field.name not in table.schema.names:
                        null_array = pa.array([None] * len(table), type=field.type)
                        table = table.append_column(field.name, null_array)

                        # Cast the table to the unified schema
                aligned_table = table.cast(unified_schema, safe=False)
                aligned_batches.append(aligned_table)

            # Concatenate tables vertically
            combined_batch = pa.concat_tables(aligned_batches)

            # Write the combined batch to the output file
            if writer is None:
                writer = pq.ParquetWriter(output_file, combined_batch.schema)
            writer.write_table(combined_batch)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 20  # Ensure the row count matches the input files


def test_tall_concat_extra_cols(parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13, tmp_path):
    # Define the output file path
    output_file = tmp_path / "tall_concat_output.parquet"

    # Open all files as datasets
    files = [parquet_tall_file_11, parquet_tall_file_12, parquet_tall_file_13]
    datasets = [ds.dataset(file, format="parquet") for file in files]

    # Determine the union of all schemas
    all_schemas = [dataset.schema for dataset in datasets]
    unified_schema = pa.unify_schemas(all_schemas)

    # Initialize the writer
    writer = None

    try:
        # Create scanners for all datasets
        scanners = [dataset.scanner() for dataset in datasets]

        # Iterate through batches from all scanners
        for batches in zip(*[scanner.to_batches() for scanner in scanners]):
            # Align each batch to the unified schema
            aligned_batches = []
            for batch in batches:
                table = pa.Table.from_batches([batch])

                # Add missing columns with null values
                for field in unified_schema:
                    if field.name not in table.schema.names:
                        null_array = pa.array([None] * len(table), type=field.type)
                        table = table.append_column(field.name, null_array)

                # Cast the table to the unified schema
                aligned_table = table.cast(unified_schema, safe=False)
                aligned_batches.append(aligned_table)

            # Concatenate tables vertically
            combined_batch = pa.concat_tables(aligned_batches)

            # Write the combined batch to the output file
            if writer is None:
                writer = pq.ParquetWriter(output_file, combined_batch.schema)
            writer.write_table(combined_batch)
    finally:
        if writer:
            writer.close()

    # Read the output file to verify its structure
    df_output = pd.read_parquet(output_file)

    # Verify the structure of the concatenated DataFrame
    expected_columns = ["x", "y", "z", "a", "b", "c", "d"]
    assert list(df_output.columns) == expected_columns
    assert len(df_output) == 30  # Ensure the row count matches the input files
