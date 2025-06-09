import logging
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds

from parq_tools.utils import atomic_output_file
from parq_tools.utils.optional_imports import get_tqdm
from parq_tools.utils.progress import get_batch_progress_bar


def validate_index_alignment(datasets: List[ds.Dataset],
                             index_columns: List[str],
                             batch_size: int = 100_000) -> None:
    """
    Validates that the index columns are identical across all datasets.

    Args:
        datasets (List[ds.Dataset]): List of PyArrow datasets to validate.
        index_columns (List[str]): List of index column names to compare.
        batch_size (int, optional): Number of rows per batch to process.

    Raises:
        ValueError: If the index columns are not identical across datasets.
    """
    logging.info("Validating index alignment across datasets")
    scanners = [dataset.scanner(columns=index_columns, batch_size=batch_size) for dataset in datasets]
    iterators = [scanner.to_batches() for scanner in scanners]

    pbar = get_batch_progress_bar(datasets, batch_size, desc="Validating index alignment")

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
        pbar.update(1)
    pbar.close()

    logging.info("Index alignment validated successfully")


def sort_parquet_file(
        input_path: Path,
        output_path: Path,
        columns: List[str],
        chunk_size: int = 100_000
) -> None:
    """
    Globally sort a Parquet file by the specified columns.

    Args:
        input_path (Path): Path to the input Parquet file.
        output_path (Path): Path to save the sorted Parquet file.
        columns (List[str]): List of column names to sort by.
        chunk_size (int, optional): Number of rows to process per chunk. Defaults to 100_000.

    """
    dataset: ds.Dataset = ds.dataset(input_path, format="parquet")
    sorted_batches: List[pa.Table] = []

    pbar = get_batch_progress_bar([dataset], chunk_size, desc="Sorting parquet file")
    # Read and sort each chunk
    for batch in dataset.to_batches(batch_size=chunk_size):
        table: pa.Table = pa.Table.from_batches([batch])
        sort_indices: pa.Array = pc.sort_indices(
            table, sort_keys=[(col, "ascending") for col in columns]
        )
        sorted_table: pa.Table = table.take(sort_indices)
        sorted_batches.append(sorted_table)
        pbar.update(1)
    pbar.close()

    # Merge all sorted chunks
    merged_table: pa.Table = pa.concat_tables(sorted_batches).combine_chunks()
    sort_indices: pa.Array = pc.sort_indices(
        merged_table, sort_keys=[(col, "ascending") for col in columns]
    )
    sorted_table: pa.Table = merged_table.take(sort_indices)

    # Write the globally sorted table to a new Parquet file
    with atomic_output_file(output_path) as tmp_file:
        pq.write_table(sorted_table, tmp_file)


def reindex_parquet(sparse_parquet_path: Path, output_path: Path,
                    new_index: pa.Table, chunk_size: int = 100_000,
                    sort_after_reindex: bool = True) -> None:
    """
    Reindex a sparse Parquet file to align with a new index, processing in chunks.

    Args:
        sparse_parquet_path (Path): Path to the sparse Parquet file.
        output_path (Path): Path to save the re-indexed Parquet file.
        new_index (pa.Table): New index as a PyArrow table.
        chunk_size (int): Number of rows to process per chunk.
        sort_after_reindex (bool): Whether to sort the output after reindexing. Defaults to True.

    """
    # Read the sparse Parquet file as a dataset
    sparse_dataset = ds.dataset(sparse_parquet_path, format="parquet")
    index_columns = [field.name for field in new_index.schema if field.name in sparse_dataset.schema.names]

    # Initialize the writer with the schema of the reindexed table
    first_batch = next(sparse_dataset.to_batches(batch_size=chunk_size))
    sparse_table = pa.Table.from_batches([first_batch])
    reindexed_table = new_index.join(sparse_table, keys=index_columns, join_type="left outer")
    writer_schema = reindexed_table.schema

    with atomic_output_file(output_path) as tmp_file, pq.ParquetWriter(tmp_file, schema=writer_schema) as writer:
        pbar = get_batch_progress_bar([sparse_dataset], chunk_size, desc="Reindexing parquet file")
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
            reindexed_table = pa.table(columns, schema=reindexed_table.schema)
            writer.write_table(reindexed_table)
            logging.info(f"Wrote {len(batch)} rows to {output_path}")
            pbar.update(1)
        pbar.close()

    if sort_after_reindex:
        with atomic_output_file(output_path) as tmp_file:
            sort_parquet_file(
                input_path=output_path,
                output_path=tmp_file,
                columns=index_columns,
                chunk_size=chunk_size
            )


def dedup_index_parquet(
        input_path: Path,
        output_path: Path,
        index_columns: List[str],
        chunk_size: int = 100_000) -> None:
    """
    Remove duplicate rows based on index columns from a Parquet file.

    Args:
        input_path (Path): Path to the input Parquet file.
        output_path (Path): Path to save the deduplicated Parquet file.
        index_columns (List[str]): Columns to use as the index for deduplication.
        chunk_size (int): Number of rows to process per chunk.
    """

    dataset = ds.dataset(input_path, format="parquet")
    seen = set()
    first_batch = next(dataset.to_batches(batch_size=chunk_size))
    schema = pa.Table.from_batches([first_batch]).schema

    with atomic_output_file(output_path) as tmp_file, pq.ParquetWriter(tmp_file, schema=schema) as writer:
        tqdm = get_tqdm()
        pbar = tqdm(total=None, desc="Deduplicating index")
        for batch in dataset.to_batches(batch_size=chunk_size):
            table = pa.Table.from_batches([batch])
            mask = []
            num_rows = table.num_rows
            for i in range(num_rows):
                idx = tuple(table[col][i].as_py() for col in index_columns)
                if idx not in seen:
                    seen.add(idx)
                    mask.append(True)
                else:
                    mask.append(False)
            if any(mask):
                filtered_table = table.filter(pa.array(mask))
                writer.write_table(filtered_table)
            pbar.update(1)
        pbar.close()