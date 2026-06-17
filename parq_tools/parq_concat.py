"""
parq_concat.py

Utilities for concatenating Parquet files, supporting both row-wise (tall) and column-wise (wide) operations, with
optional filtering, column selection, and progress tracking.

Main APIs:

- concat_parquet_files: Concatenate multiple Parquet files into a single file, with flexible options for axis,
  filtering, and batching.
- ParquetConcat: Class for advanced concatenation workflows, supporting batch processing, index alignment,
  and metadata handling.
"""
import gc
import json
import logging
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import pyarrow.dataset as ds
from typing import List, Optional, Sequence

from parq_tools.utils.file_utils import check_valid_parquet

from parq_tools.utils import atomic_output_file, get_pandas_metadata, merge_pandas_metadata
from parq_tools.utils.index_utils import validate_index_alignment
# noinspection PyProtectedMember
from parq_tools.utils._query_parser import build_filter_expression, get_filter_parser, get_referenced_columns

try:
    # noinspection PyUnresolvedReferences
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def concat_parquet_files(files: List[Path],
                         output_path: Path,
                         axis: int = 0,
                         index_columns: Optional[List[str]] = None,
                         filter_query: Optional[str] = None,
                         columns: Optional[List[str]] = None,
                         batch_size: int = 100_000,
                         show_progress: bool = False) -> None:
    """
    Concatenate multiple Parquet files into a single file, supporting both row-wise and column-wise concatenation.

    Args:
        files (List[Path]): List of input Parquet file paths.
        output_path (Path): Path to save the concatenated Parquet file.
        axis (int): Axis along which to concatenate (0 for row-wise, 1 for column-wise).
        index_columns (Optional[List[str]]): List of index columns for row-wise sorting after concatenation.
        filter_query (Optional[str]): Filter expression to apply to the concatenated data.
        columns (Optional[List[str]]): List of columns to include in the output.
        batch_size (int): Number of rows per batch to process. Defaults to 100_00.
        show_progress (bool): If True, displays a progress bar using `tqdm` (if installed).

    Raises:
        ValueError: If the input files list is empty or if any file is not accessible.
    """
    concat = ParquetConcat(files, axis, index_columns, show_progress)
    concat.concat_to_file(output_path, filter_query, columns, batch_size, show_progress)


def concat_parquet_file_with_dataframe(
    parquet_path: Path,
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
    *,
    index_columns: List[str],
    batch_size: int = 100_000,
    allow_overwrite: bool = False,
    show_progress: bool = False,
    **pq_write_kwargs: object,
) -> None:
    """
    Rewrite a Parquet file by appending columns from an in-memory DataFrame.

    The source Parquet file is streamed in row batches, aligned against the
    provided DataFrame using ``index_columns``, and written to a new Parquet
    file. This avoids materializing the DataFrame to disk first.

    Validation is content-based, so Parquet files with non-standard extensions
    (for example ``.pbm``) are accepted if they are valid Parquet files.

    Parameters
    ----------
    parquet_path:
        Path to the source Parquet file.
    df:
        In-memory DataFrame containing the extra columns to append.
    output_path:
        Destination path for the rewritten Parquet file. If omitted, the source
        file is rewritten in place.
    index_columns:
        Column names used to align rows between the source file and ``df``.
    batch_size:
        Number of source rows to process per batch.
    allow_overwrite:
        If ``True``, allow replacing an existing output file, including the
        source file when rewriting in place.
    show_progress:
        If ``True``, display a progress bar if ``tqdm`` is available.
    **pq_write_kwargs:
        Extra keyword arguments forwarded to ``pyarrow.parquet.ParquetWriter``.

    Raises
    ------
    FileNotFoundError
        If the source file does not exist.
    ValueError
        If the file is not valid Parquet, batch_size is invalid, or key columns
        are duplicated/misaligned.
    FileExistsError
        If the target exists and overwrite is not allowed.
    KeyError
        If required index columns are missing from either input.
    """
    source_path = Path(parquet_path)
    target_path = Path(output_path) if output_path is not None else source_path

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not index_columns:
        raise ValueError("index_columns must contain at least one column name")
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")
    if not check_valid_parquet(source_path):
        raise ValueError(f"Source file is not a valid Parquet file: {source_path}")

    if target_path.exists() and not allow_overwrite:
        raise FileExistsError(f"Target file already exists: {target_path}")

    source_pf = pq.ParquetFile(str(source_path))
    source_columns = source_pf.schema_arrow.names

    missing_source_index = [col for col in index_columns if col not in source_columns]
    if missing_source_index:
        raise KeyError(
            f"Index columns missing from source parquet file: {missing_source_index}"
        )

    missing_df_index = [col for col in index_columns if col not in df.columns]
    if missing_df_index:
        raise KeyError(
            f"Index columns missing from input DataFrame: {missing_df_index}"
        )

    df_extra_columns = [col for col in df.columns if col not in index_columns]
    duplicate_non_index_columns = [col for col in df_extra_columns if col in source_columns]
    if duplicate_non_index_columns:
        raise ValueError(
            "Input DataFrame contains columns already present in the source parquet "
            f"file: {duplicate_non_index_columns}"
        )

    if df.duplicated(list(index_columns)).any():
        raise ValueError(
            "Input DataFrame contains duplicate keys for index_columns; wide "
            "concatenation requires unique key rows."
        )

    df_indexed = df.set_index(list(index_columns), drop=False)

    progress_bar = None
    if show_progress:
        try:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=source_pf.metadata.num_rows,
                desc="Concatenating columns",
                unit="row",
            )
        except ImportError:
            progress_bar = None

    try:
        with atomic_output_file(target_path) as tmp_file:
            writer: Optional[pq.ParquetWriter] = None
            writer_schema: Optional[pa.Schema] = None
            try:
                for batch in source_pf.iter_batches(batch_size=batch_size):
                    source_chunk = batch.to_pandas()
                    chunk_indexed = source_chunk.set_index(list(index_columns), drop=False)

                    merged_chunk = chunk_indexed.join(
                        df_indexed[df_extra_columns],
                        how="left",
                    ).reset_index(drop=True)

                    table = pa.Table.from_pandas(merged_chunk, preserve_index=False)

                    if writer is None:
                        writer_schema = table.schema
                        writer = pq.ParquetWriter(tmp_file, writer_schema, **pq_write_kwargs)
                    else:
                        table = table.cast(writer_schema, safe=False)

                    writer.write_table(table)

                    if progress_bar is not None:
                        progress_bar.update(len(source_chunk))
            finally:
                if writer is not None:
                    writer.close()
                source_pf = None
                gc.collect()
    finally:
        if progress_bar is not None:
            progress_bar.close()

class ParquetConcat:
    """
    A utility for concatenating Parquet files while supporting axis-based merging, filtering,
    and progress tracking.
    """

    def __init__(self, files: List[Path], axis: int = 0,
                 index_columns: Optional[List[str]] = None, show_progress: bool = False) -> None:
        """
        Initializes ParquetConcat with specified parameters.

        Args:
            files (List[Path]): List of Parquet files to concatenate.
            axis (int, optional): Concatenation axis (0 = row-wise, 1 = column-wise). Defaults to 0.
            index_columns (Optional[List[str]], optional): Index columns for sorting. Defaults to None.
            show_progress (bool, optional): If True, enables tqdm progress bar (if installed). Defaults to False.
        """
        if not files:
            raise ValueError("The list of input files cannot be empty.")
        self.files = files
        self.axis = axis
        self.index_columns = index_columns or []
        self.show_progress = show_progress and HAS_TQDM  # Only enable progress if tqdm is available
        logging.info("Initializing ParquetConcat with %d files", len(files))
        self._validate_input_files()

    def _validate_input_files(self) -> None:
        """
        Validates that all input files exist and are readable.
        """
        for file in self.files:
            if not Path(file).is_file():
                raise ValueError(f"File not found or inaccessible: {file}")

    def _validate_columns(self, schema: pa.Schema, columns: Optional[List[str]]) -> List[str]:
        """
        Validates that the requested columns exist in the schema.

        Args:
            schema (pa.Schema): The schema to validate against.
            columns (Optional[List[str]]): List of requested columns.

        Returns:
            List[str]: The validated list of columns, including index columns.

        Raises:
            ValueError: If any requested column is not found in the schema.
        """
        if not columns:
            return self.index_columns  # If no columns are specified, return only index columns

        missing_columns = [col for col in columns if col not in schema.names]
        if missing_columns:
            logging.warning(f"Columns {missing_columns} are missing in the schema. They will be added as null columns.")

        # Include index columns in the final list
        return list(dict.fromkeys(self.index_columns + [col for col in columns if col in schema.names]))

    @staticmethod
    def _validate_filter_columns(filter_query: Optional[str], datasets: List[ds.Dataset]) -> None:
        """
        Validates that all columns referenced in the filter query exist in all datasets.

        Args:
            filter_query (Optional[str]): The filter query to validate.
            datasets (List[ds.Dataset]): List of datasets to check.

        Raises:
            ValueError: If any column in the filter query is missing in one or more datasets.
        """
        if not filter_query:
            return

        # Extract referenced columns from the filter query
        referenced_columns = get_referenced_columns(filter_query)

        # Check each dataset for the presence of all referenced columns
        missing_columns = set()
        for column in referenced_columns:
            for dataset in datasets:
                if column not in dataset.schema.names:
                    missing_columns.add(column)
                    break

        if missing_columns:
            raise ValueError(
                f"The filter query references columns that are missing in one or more datasets: {missing_columns}"
            )

    def concat_to_file(self, output_path: Path, filter_query: Optional[str] = None,
                       columns: Optional[List[str]] = None, batch_size: int = 100_000,
                       show_progress: bool = False) -> None:
        """
        Concatenates input Parquet files and writes the result to a file.

        Args:
            output_path (Path): Destination path for the output Parquet file.
            filter_query (Optional[str]): Filter expression to apply.
            columns (Optional[List[str]]): List of columns to include in the output.
            batch_size (int, optional): Number of rows per batch to process. Defaults to 1024.
            show_progress (bool, optional): If True, displays a progress bar using `tqdm` (if installed). Defaults to False.
        """
        logging.info("Using low-memory iterative concatenation")
        datasets = [ds.dataset(file, format="parquet") for file in self.files]
        schemas = [dataset.schema for dataset in datasets]

        # Create a unified schema that includes all columns from all datasets
        unified_schema = pa.unify_schemas(schemas)
        logging.debug("Unified schema: %s", unified_schema)
        if filter_query:
            self._validate_filter(filter_query, unified_schema)

        if self.axis == 1:  # Wide concatenation
            self._concat_wide(datasets, unified_schema, output_path, columns, filter_query, batch_size, show_progress)
        else:  # Tall concatenation
            self._concat_tall(datasets, unified_schema, output_path, columns, filter_query, batch_size, show_progress)
        logging.info(f"Concatenation along axis {self.axis} completed and saved to: %s", output_path)

    def _concat_wide(self, datasets: List[ds.Dataset], unified_schema: pa.Schema, output_path: Path,
                     columns: Optional[List[str]], filter_query: Optional[str],
                     batch_size: int, show_progress: bool) -> None:
        """Handles wide concatenation (axis=1) in a memory-efficient manner by batch processing

        Explicitly writes merged pandas metadata to the output file if it exists in any input file.
        """

        logging.info("Starting wide concatenation with batch processing")

        # Initialize columns if None
        columns = columns or unified_schema.names
        columns = self._validate_columns(unified_schema, columns)

        validate_index_alignment(datasets, index_columns=self.index_columns)

        # Collect and merge pandas metadata from all input files
        pandas_metadatas = []
        for file in self.files:
            meta = get_pandas_metadata(file)
            if meta:
                pandas_metadatas.append(meta)
        merged_pandas_meta = merge_pandas_metadata(pandas_metadatas) if pandas_metadatas else None

        progress_bar = None

        try:
            # Create iterators for all dataset scanners
            scanners = [
                iter(dataset.scanner(
                    columns=[col for col in columns if col in dataset.schema.names],
                    batch_size=batch_size
                ).to_batches())
                for dataset in datasets
            ]

            with atomic_output_file(output_path) as tmp_file:
                writer = None
                try:
                    while True:
                        aligned_batches = []
                        all_exhausted = True
                        for scanner in scanners:
                            try:
                                batch = next(scanner)
                                table = pa.Table.from_batches([batch])
                                if table.column_names == self.index_columns:
                                    continue
                                aligned_batches.append(table)
                                all_exhausted = False
                            except StopIteration:
                                aligned_batches.append(None)
                        if all_exhausted:
                            break
                        combined_arrays = []
                        combined_fields = []
                        for i, table in enumerate(aligned_batches):
                            if table is None:
                                continue
                            if i == 0:
                                combined_arrays.extend(table.columns)
                                combined_fields.extend(table.schema)
                            else:
                                for col, field in zip(table.columns, table.schema):
                                    if field.name not in self.index_columns:
                                        combined_arrays.append(col)
                                        combined_fields.append(field)
                        combined_table = pa.Table.from_arrays(combined_arrays, schema=pa.schema(combined_fields))
                        if filter_query:
                            filter_expression = build_filter_expression(filter_query, combined_table.schema)
                            combined_table = combined_table.filter(filter_expression)
                        if writer is None:
                            schema = combined_table.schema
                            if merged_pandas_meta:
                                new_meta = dict(schema.metadata or {})
                                new_meta[b"pandas"] = json.dumps(merged_pandas_meta).encode()
                                schema = schema.with_metadata(new_meta)
                                logging.info("Writing merged pandas metadata to output file.")
                            else:
                                logging.info("No pandas metadata found in input files.")
                            if show_progress and HAS_TQDM and progress_bar is None:
                                total_batches = max(
                                    sum(fragment.metadata.num_row_groups for fragment in dataset.get_fragments())
                                    for dataset in datasets)
                                progress_bar = tqdm(total=total_batches, desc="Processing batches", unit="batch")
                            writer = pq.ParquetWriter(tmp_file, schema)
                        writer.write_table(combined_table)
                        if progress_bar:
                            progress_bar.update(1)
                finally:
                    if writer:
                        writer.close()
        finally:
            if progress_bar:
                progress_bar.close()

    def _concat_tall(self, datasets: List[ds.Dataset], unified_schema: pa.Schema, output_path: Path,
                     columns: Optional[List[str]], filter_query: Optional[str], batch_size: int,
                     show_progress: bool) -> None:
        """Handles tall concatenation (axis=0) in a memory-efficient manner by batch processing

        Explicitly writes merged pandas metadata to the output file if it exists in any input file.
        """
        progress_bar = None

        # Collect and merge pandas metadata from all input files
        pandas_metadatas = []
        for file in self.files:
            meta = get_pandas_metadata(file)
            if meta:
                pandas_metadatas.append(meta)
        merged_pandas_meta = merge_pandas_metadata(pandas_metadatas) if pandas_metadatas else None

        try:
            columns = columns or unified_schema.names
            columns = self._validate_columns(unified_schema, columns)
            total_row_groups = sum(
                fragment.metadata.num_row_groups for dataset in datasets for fragment in dataset.get_fragments())

            ParquetConcat._validate_filter_columns(filter_query, datasets)

            # Create scanners for all datasets
            scanners = [
                dataset.scanner(
                    columns=[col for col in columns if col in dataset.schema.names],
                    filter=build_filter_expression(filter_query, dataset.schema) if filter_query else None,
                    batch_size=batch_size
                )
                for dataset in datasets
            ]

            with atomic_output_file(output_path) as tmp_file:
                writer = None
                try:
                    for scanner in scanners:
                        for batch in scanner.to_batches():
                            table = pa.Table.from_batches([batch])

                            # Align the table to the unified schema
                            for field in unified_schema:
                                if field.name not in table.schema.names:
                                    null_array = pa.array([None] * len(table), type=field.type)
                                    table = table.append_column(field.name, null_array)

                            # Reorder columns to match unified_schema before casting
                            table = table.select(unified_schema.names)
                            table = table.cast(unified_schema, safe=False)

                            # Write the batch directly
                            if writer is None:
                                schema = table.schema
                                if merged_pandas_meta:
                                    new_meta = dict(schema.metadata or {})
                                    new_meta[b"pandas"] = json.dumps(merged_pandas_meta).encode()
                                    schema = schema.with_metadata(new_meta)
                                    logging.info("Writing merged pandas metadata to output file.")
                                else:
                                    logging.info("No pandas metadata found in input files.")
                                if show_progress and HAS_TQDM and progress_bar is None:
                                    progress_bar = tqdm(total=total_row_groups, desc="Processing batches", unit="batch")
                                writer = pq.ParquetWriter(tmp_file, schema)
                            writer.write_table(table)
                            if progress_bar:
                                progress_bar.update(1)
                finally:
                    if writer:
                        writer.close()
        finally:
            if progress_bar:
                progress_bar.close()

    @staticmethod
    def _validate_filter(filter_query: Optional[str], schema: pa.Schema) -> None:
        """
        Validates the filter query against the table schema.

        Args:
            filter_query (Optional[str]): Filter expression.
            schema (pa.Schema): Schema of the table to validate against.

        Raises:
            ValueError: If the filter expression is invalid or references non-existent columns.
        """
        if not filter_query:
            return

        try:
            # Parse the filter query to ensure it's valid
            parser = get_filter_parser()
            parser.parse(filter_query)

            # Use the get_referenced_columns function to extract referenced columns
            referenced_columns = get_referenced_columns(filter_query)
            missing_columns = [col for col in referenced_columns if col not in schema.names]
            if missing_columns:
                raise ValueError(f"Filter references non-existent columns: {missing_columns}")
        except Exception as e:
            logging.error("Malformed filter expression: %s", filter_query)
            raise ValueError(f"Malformed filter expression: {filter_query}\nError: {e}")
