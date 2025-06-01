import logging
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds
from typing import List, Optional

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


def concat_parquet_files(
        files: List[Path],
        output_path: Path,
        axis: int = 0,
        index_columns: Optional[List[str]] = None,
        filter_query: Optional[str] = None,
        columns: Optional[List[str]] = None,
        batch_size: int = 100_000,
        show_progress: bool = False
) -> None:
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


class ParquetConcat:
    """
    A utility for concatenating Parquet files while supporting axis-based merging, filtering,
    and progress tracking.

    Attributes:
        files (List[str]): List of input Parquet file paths.
        axis (int): Axis along which to concatenate (0 for row-wise, 1 for column-wise).
        index_columns (Optional[List[str]]): List of index columns for row-wise sorting after concatenation.
        show_progress (bool): If True, displays a progress bar using `tqdm` (if installed).
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

    def _validate_wide(self, schemas: List[pa.Schema]) -> None:
        """
        Validates index column alignment for wide concatenation.

        Args:
            schemas (List[pa.Schema]): List of schemas from input files.

        Raises:
            ValueError: If index columns do not match in order or chunk sizes are inconsistent.
        """
        logging.info("Validating schemas for wide concatenation")
        index_set = set(self.index_columns)
        for schema in schemas:
            if not index_set.issubset(set(schema.names)):
                raise ValueError(f"Index columns {self.index_columns} are missing in schema: {schema.names}")

        # Ensure index columns are in the same order across all schemas
        for schema in schemas:
            schema_index_columns = [field.name for field in schema if field.name in self.index_columns]
            if schema_index_columns != self.index_columns:
                raise ValueError(f"Index columns are not in the same order: {schema_index_columns}")

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

    def _concat_wide(self, datasets: List[ds.Dataset], unified_schema: pa.Schema, output_path: Path,
                     columns: Optional[List[str]], filter_query: Optional[str],
                     batch_size: int, show_progress: bool) -> None:
        """
        Handles wide concatenation (axis=1) in a memory-efficient manner by processing data in batches.
        """
        logging.info("Starting wide concatenation with batch processing")

        # Initialize columns if None
        columns = columns or unified_schema.names
        columns = self._validate_columns(unified_schema, columns)

        validate_index_alignment(datasets, index_columns=self.index_columns)

        writer = None
        progress_bar = None

        try:
            # Create iterators for all dataset scanners
            # todo: reconsider index columns -> early global check or per chunk.
            scanners = [
                iter(dataset.scanner(
                    columns=[col for col in columns if col in dataset.schema.names],
                    batch_size=batch_size
                ).to_batches())
                for dataset in datasets
            ]

            if show_progress and HAS_TQDM:
                total_batches = max(
                    sum(fragment.metadata.num_row_groups for fragment in dataset.get_fragments())
                    for dataset in datasets)
                progress_bar = tqdm(total=total_batches, desc="Processing batches", unit="batch")

            while True:
                aligned_batches = []
                all_exhausted = True

                for scanner in scanners:
                    try:
                        batch = next(scanner)
                        table = pa.Table.from_batches([batch])
                        if table.column_names == self.index_columns:
                            # If the table only contains index columns, skip it
                            continue
                        aligned_batches.append(table)
                        all_exhausted = False
                    except StopIteration:
                        aligned_batches.append(None)

                if all_exhausted:
                    break

                # Merge tables horizontally by combining their columns
                combined_arrays = []
                combined_fields = []
                for i, table in enumerate(aligned_batches):
                    if table is None:
                        continue
                    if i == 0:
                        combined_arrays.extend(table.columns)
                        combined_fields.extend(table.schema)
                    else:
                        # Exclude index columns from all but the first table
                        for col, field in zip(table.columns, table.schema):
                            if field.name not in self.index_columns:
                                combined_arrays.append(col)
                                combined_fields.append(field)
                combined_table = pa.Table.from_arrays(combined_arrays, schema=pa.schema(combined_fields))

                # Apply row-level filtering to the combined table
                if filter_query:
                    filter_expression = build_filter_expression(filter_query, combined_table.schema)
                    combined_table = combined_table.filter(filter_expression)

                # Write the filtered batch to the output file
                if writer is None:
                    writer = pq.ParquetWriter(output_path, combined_table.schema)
                writer.write_table(combined_table)

                if progress_bar:
                    progress_bar.update(1)


        finally:
            if writer:
                writer.close()
            if progress_bar:
                progress_bar.close()
        logging.info("Wide concatenation completed and saved to: %s", output_path)

    def _concat_tall(self, datasets: List[ds.Dataset], unified_schema: pa.Schema, output_path: Path,
                     columns: Optional[List[str]], filter_query: Optional[str], batch_size: int,
                     show_progress: bool) -> None:
        """
        Handles tall concatenation (axis=0).

        Args:
            datasets (List[ds.Dataset]): List of datasets to concatenate.
            unified_schema (pa.Schema): Unified schema for all datasets.
            output_path (Path): Destination path for the output Parquet file.
            columns (Optional[List[str]]): List of columns to include in the output.
            filter_query (Optional[str]): Filter expression to apply.
            batch_size (int): Number of rows per batch to process.
            show_progress (bool): If True, displays a progress bar using `tqdm` (if installed).
        """
        writer = None
        progress_bar = None

        try:
            columns = columns or unified_schema.names
            columns = self._validate_columns(unified_schema, columns)
            total_row_groups = sum(
                fragment.metadata.num_row_groups for dataset in datasets for fragment in dataset.get_fragments())

            if show_progress and HAS_TQDM:
                progress_bar = tqdm(total=total_row_groups, desc="Processing batches", unit="batch")

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
                        writer = pq.ParquetWriter(output_path, table.schema)
                    writer.write_table(table)
                    if progress_bar:
                        progress_bar.update(1)
        finally:
            if writer:
                writer.close()
            if progress_bar:
                progress_bar.close()

    @staticmethod
    def _align_schema(table: pa.Table, unified_schema: pa.Schema) -> pa.Table:
        """
        Aligns the schema of a table with the unified schema by adding missing columns with null values.

        Args:
            table (pa.Table): The table to align.
            unified_schema (pa.Schema): The unified schema to align with.

        Returns:
            pa.Table: The aligned table.
        """
        # Add missing columns with null values
        for field in unified_schema:
            if field.name not in table.schema.names:
                null_array = pa.array([None] * len(table), type=field.type)
                table = table.append_column(field.name, null_array)

        # Reorder columns to match the unified schema
        reordered_columns = [table[field.name] for field in unified_schema]
        return pa.Table.from_arrays(reordered_columns, schema=unified_schema)

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
