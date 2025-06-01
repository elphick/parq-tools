import logging
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow as pa
from typing import List

try:
    # noinspection PyUnresolvedReferences
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def validate_index_alignment(datasets: List[ds.Dataset], index_columns: List[str], batch_size: int = 1024) -> None:
    """
    Validates that the index columns are identical across all datasets.

    Args:
        datasets (List[ds.Dataset]): List of PyArrow datasets to validate.
        index_columns (List[str]): List of index column names to compare.
        batch_size (int, optional): Number of rows per batch to process. Defaults to 1024.

    Raises:
        ValueError: If the index columns are not identical across datasets.
    """
    logging.info("Validating index alignment across datasets")
    scanners = [dataset.scanner(columns=index_columns, batch_size=batch_size) for dataset in datasets]
    iterators = [scanner.to_batches() for scanner in scanners]

    reference_batch = None

    while True:
        current_batches = []
        all_exhausted = True

        for iterator in iterators:
            try:
                batch = next(iterator)
                current_batches.append(pa.Table.from_batches([batch]))
                all_exhausted = False
            except StopIteration:
                current_batches.append(None)

        if all_exhausted:
            break

        reference_batch = current_batches[0]
        for i, current_batch in enumerate(current_batches[1:], start=1):
            if current_batch is not None and not current_batch.equals(reference_batch):
                raise ValueError(
                    f"Index columns are not aligned across datasets. Mismatch found in dataset {i}."
                )

    logging.info("Index alignment validated successfully")


def sort_parquet_file(input_path: str, output_path: str, columns: list[str], chunk_size: int = 100_000):
    """
    Globally sort a Parquet file by the specified columns.

    Args:
        input_path (str): Path to the input Parquet file.
        output_path (str): Path to save the sorted Parquet file.
        columns (list[str]): List of column names to sort by.
        chunk_size (int): Number of rows to process per chunk.

    Returns:
        None
    """
    dataset = ds.dataset(input_path, format="parquet")
    sorted_batches = []

    # Read and sort each chunk
    for batch in dataset.to_batches(batch_size=chunk_size):
        table = pa.Table.from_batches([batch])
        sort_indices = pc.sort_indices(table, sort_keys=[(col, "ascending") for col in columns])
        sorted_table = table.take(sort_indices)
        sorted_batches.append(sorted_table)

    # Merge all sorted chunks
    merged_table = pa.concat_tables(sorted_batches).combine_chunks()
    sort_indices = pc.sort_indices(merged_table, sort_keys=[(col, "ascending") for col in columns])
    sorted_table = merged_table.take(sort_indices)

    # Write the globally sorted table to a new Parquet file
    pq.write_table(sorted_table, output_path)


def reindex_parquet(sparse_parquet_path: Path, new_index: pa.Table, output_path: Path, chunk_size: int = 100_000):
    """
    Reindex a sparse Parquet file to align with a new index, processing in chunks.

    Args:
        sparse_parquet_path (Path): Path to the sparse Parquet file.
        new_index (pa.Table): New index as a PyArrow table.
        output_path (Path): Path to save the reindexed Parquet file.
        chunk_size (int): Number of rows to process per chunk.

    Returns:
        None
    """
    # Read the sparse Parquet file as a dataset
    sparse_dataset = ds.dataset(sparse_parquet_path, format="parquet")
    index_columns = [field.name for field in new_index.schema if field.name in sparse_dataset.schema.names]

    # Initialize the writer with the schema of the reindexed table
    first_batch = next(sparse_dataset.to_batches(batch_size=chunk_size))
    sparse_table = pa.Table.from_batches([first_batch])
    reindexed_table = new_index.join(sparse_table, keys=index_columns, join_type="left outer")
    writer_schema = reindexed_table.schema

    with pq.ParquetWriter(output_path, schema=writer_schema) as writer:
        # Process the sparse dataset in chunks
        for batch in sparse_dataset.to_batches(batch_size=chunk_size):
            sparse_table = pa.Table.from_batches([batch])

            # Perform a left join with the new index
            reindexed_table = new_index.join(sparse_table, keys=index_columns, join_type="left outer")

            # Fill null values dynamically based on column types
            columns = []
            for field in reindexed_table.schema:
                column = reindexed_table[field.name]
                if pa.types.is_floating(field.type):
                    column = pc.if_else(pc.is_null(column), pa.scalar(float('nan'), type=field.type), column)
                elif pa.types.is_string(field.type):
                    column = pc.if_else(pc.is_null(column), pa.scalar(None, type=field.type), column)
                elif pa.types.is_dictionary(field.type):  # Categorical
                    column = pc.if_else(pc.is_null(column), pa.scalar(None, type=field.type), column)
                elif pa.types.is_integer(field.type):
                    column = pc.if_else(pc.is_null(column), pa.scalar(None, type=pa.int64()), column)
                columns.append(column)

            # Create a new table with filled values
            reindexed_table = pa.table(columns, schema=reindexed_table.schema)

            # Write the processed chunk to the output file
            writer.write_table(reindexed_table)
            logging.info(f"Wrote {len(batch)} rows to {output_path}")
