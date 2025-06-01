import logging
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.dataset as ds
# noinspection PyProtectedMember
from parq_tools.utils._query_parser import build_filter_expression
import pyarrow as pa
from typing import List, Optional

try:
    # noinspection PyUnresolvedReferences
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def filter_parquet_file(
        input_path: Path,
        output_path: Path,
        filter_expression: str,
        columns: Optional[List[str]] = None,
        chunk_size: int = 100_000,
        show_progress: bool = False
):
    """
    Filter a Parquet file based on a pandas-like expression and a sub-selection of columns.

    Args:
        input_path (Path): Path to the input Parquet file.
        output_path (Path): Path to save the filtered Parquet file.
        filter_expression (str): Pandas-like filter expression (e.g., "x > 5 & y < 10").
        columns (Optional[List[str]]): List of columns to include in the output. Defaults to None (all columns).
        chunk_size (int): Number of rows to process per chunk. Defaults to 100,000.
        show_progress (bool): Whether to display a progress bar. Defaults to False.

    Returns:
        None
    """
    # Read the Parquet file as a dataset
    dataset = ds.dataset(input_path, format="parquet")

    # Build the filter expression using the existing parser
    filter_expr = build_filter_expression(filter_expression, dataset.schema)

    # Create a scanner for the dataset
    scanner = dataset.scanner(columns=columns, batch_size=chunk_size)

    # Initialize the writer with the schema of the filtered table
    first_batch = next(scanner.to_batches())
    filtered_table = pa.Table.from_batches([first_batch]).filter(filter_expr)
    writer_schema = filtered_table.schema

    # Initialize progress bar if enabled
    total_rows = dataset.count_rows()
    progress = None
    if HAS_TQDM and show_progress:
        progress = tqdm(total=total_rows, desc="Filtering", unit="rows")

    with pq.ParquetWriter(output_path, schema=writer_schema) as writer:
        # Process the dataset in chunks using the scanner
        for batch in scanner.to_batches():
            table = pa.Table.from_batches([batch])

            # Apply the filter expression
            filtered_table = table.filter(filter_expr)

            # Write the filtered chunk to the output file
            writer.write_table(filtered_table)

            # Update progress bar
            if progress:
                progress.update(len(batch))

    # Close the progress bar
    if progress:
        progress.close()
    logging.info(f"Filtered {total_rows} rows")
