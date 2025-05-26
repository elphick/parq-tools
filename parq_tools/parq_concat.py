import logging
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import pyarrow.dataset as ds
from typing import List, Optional

from parq_tools.utils.filter_parser import get_filter_parser, build_filter_expression, get_referenced_columns

# Try to import tqdm, but allow execution without it
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ParquetConcat:
    """
    A utility for concatenating Parquet files while supporting axis-based merging, filtering,
    optional strict index enforcement, and progress tracking.

    Attributes:
        files (List[str]): List of input Parquet file paths.
        axis (int): Axis along which to concatenate (0 for row-wise, 1 for column-wise).
        strict (bool): If True, enforces strict index column alignment.
        index_columns (Optional[List[str]]): List of index columns for row-wise sorting after concatenation.
        show_progress (bool): If True, displays a progress bar using `tqdm` (if installed).
    """

    def __init__(self, files: List[Path], axis: int = 0, filter_query: Optional[str] = None,
                 strict: bool = False, index_columns: Optional[List[str]] = None, show_progress: bool = False) -> None:
        """
        Initializes ParquetConcat with specified parameters.

        Args:
            files (List[Path]): List of Parquet files to concatenate.
            axis (int, optional): Concatenation axis (0 = row-wise, 1 = column-wise). Defaults to 0.
            strict (bool, optional): Enforce strict index column alignment. Defaults to False.
            index_columns (Optional[List[str]], optional): Index columns for sorting. Defaults to None.
            show_progress (bool, optional): If True, enables tqdm progress bar (if installed). Defaults to False.

        Raises:
            ValueError: If the filter expression is invalid.
        """
        if not files:
            raise ValueError("The list of input files cannot be empty.")
        self.files = files
        self.axis = axis
        self.strict = strict
        self.index_columns = index_columns or []
        self.show_progress = show_progress and HAS_TQDM  # Only enable progress if tqdm is available
        self._parser = get_filter_parser()  # Initialize the parser
        logging.info("Initializing ParquetConcat with %d files", len(files))
        self._validate_input_files()

    def _validate_input_files(self) -> None:
        """
        Validates that all input files exist and are readable.
        """
        for file in self.files:
            if not Path(file).is_file():
                raise ValueError(f"File not found or inaccessible: {file}")

    def _validate_output_schema(self, output_table: pa.Table, expected_columns: List[str]) -> None:
        """
        Validates the schema of the output table against the expected columns.

        Args:
            output_table (pa.Table): The output table to validate.
            expected_columns (List[str]): List of expected column names.

        Raises:
            ValueError: If the output schema does not match the expected columns.
        """
        actual_columns = output_table.schema.names
        if actual_columns != expected_columns:
            raise ValueError(f"Output schema mismatch. Expected: {expected_columns}, Got: {actual_columns}")

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
        return self.index_columns + [col for col in columns if col in schema.names]

    def concat_to_file(self, output_path: Path, filter_query: Optional[str] = None,
                       columns: Optional[List[str]] = None, use_pyarrow_concat: bool = False,
                       batch_size: int = 1024) -> None:
        """
        Concatenates input Parquet files and writes the result to a file.

        Args:
            output_path (Path): Destination path for the output Parquet file.
            filter_query (Optional[str]): Pandas-style filter expression to apply.
            columns (Optional[List[str]]): List of columns to include in the output.
            use_pyarrow_concat (bool, optional): If True, uses PyArrow's built-in concat methods. Defaults to False.
            batch_size (int, optional): Number of rows per batch to process. Defaults to 1024.

        Raises:
            ValueError: If schemas or chunk sizes are inconsistent across files.
        """
        if use_pyarrow_concat:
            self._concat_with_pyarrow(output_path, filter_query, columns)
        else:
            self._concat_iteratively(output_path, filter_query, columns, batch_size)

    def _concat_with_pyarrow(self, output_path: Path, filter_query: Optional[str], columns: Optional[List[str]]) -> None:
        """
        Concatenates files using PyArrow's built-in concat methods.

        Args:
            output_path (Path): Destination path for the output Parquet file.
            filter_query (Optional[str]): Pandas-style filter expression to apply.
            columns (Optional[List[str]]): List of columns to include in the output.
        """
        logging.info("Using PyArrow's built-in concat methods")
        tables = []

        for file in self.files:
            logging.info("Reading file: %s", file)
            dataset = ds.dataset(file, format="parquet")
            validated_columns = self._validate_columns(dataset.schema, columns)
            scanner = dataset.scanner(
                columns=validated_columns,
                filter=build_filter_expression(filter_query, dataset.schema) if filter_query else None
            )
            table = pa.Table.from_batches(scanner.to_batches())
            tables.append(table)

        if self.axis == 1:
            result_table = pa.concat_tables(tables, promote=True)
        else:
            result_table = pa.concat_tables(tables)

        # Write the final table to the output file
        pq.write_table(result_table, output_path)
        logging.info("PyArrow concat completed and saved to: %s", output_path)

    def _concat_iteratively(self, output_path: Path, filter_query: Optional[str], columns: Optional[List[str]],
                            batch_size: int) -> None:
        """
        Concatenates files iteratively in a low-memory approach.

        Args:
            output_path (Path): Destination path for the output Parquet file.
            filter_query (Optional[str]): Pandas-style filter expression to apply.
            columns (Optional[List[str]]): List of columns to include in the output.
            batch_size (int): Number of rows per batch to process.
        """
        logging.info("Using low-memory iterative concatenation")
        datasets = [ds.dataset(file, format="parquet") for file in self.files]
        schemas = [dataset.schema for dataset in datasets]

        # Validate schemas for wide or tall concatenation
        if self.axis == 1:
            self._validate_wide(schemas)
        unified_schema = pa.unify_schemas(schemas)

        writer = None
        try:
            if self.axis == 1:  # Wide concatenation
                # Create scanners for all datasets with filtering applied
                scanners = [
                    dataset.scanner(
                        columns=self._validate_columns(dataset.schema, columns or dataset.schema.names),
                        batch_size=batch_size,
                        filter=build_filter_expression(filter_query, dataset.schema) if filter_query else None
                    )
                    for dataset in datasets
                ]
                batch_generators = [scanner.to_batches() for scanner in scanners]

                while True:
                    # Fetch the next batch from each generator
                    batches = [next(batch_gen, None) for batch_gen in batch_generators]

                    # Break the loop if all batches are None (end of all datasets)
                    if all(batch is None for batch in batches):
                        break

                    # Convert non-None batches to tables
                    tables = [pa.Table.from_batches([batch]) for batch in batches if batch is not None]

                    # Align and aggregate the tables for wide concatenation
                    combined_table = self._align_and_aggregate_wide(tables, columns)

                    # Write the combined table to the output file
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, combined_table.schema)
                    writer.write_table(combined_table)
            else:  # Tall concatenation
                for dataset in datasets:
                    scanner = dataset.scanner(
                        columns=self._validate_columns(dataset.schema, columns or dataset.schema.names),
                        batch_size=batch_size,
                        filter=build_filter_expression(filter_query, dataset.schema) if filter_query else None
                    )
                    for batch in scanner.to_batches():
                        table = pa.Table.from_batches([batch])
                        table = self._align_schema(table, unified_schema)

                        if writer is None:
                            writer = pq.ParquetWriter(output_path, unified_schema)
                        writer.write_table(table)
        finally:
            if writer:
                writer.close()

    def _align_and_aggregate_wide(self, tables: List[pa.Table], columns: Optional[List[str]] = None) -> pa.Table:
        """
        Aligns and aggregates tables horizontally by index columns for wide concatenation.

        Args:
            tables (List[pa.Table]): List of tables to align and aggregate.
            columns (Optional[List[str]]): List of columns to include in the output.

        Returns:
            pa.Table: Horizontally aligned and aggregated table.
        """
        # Check for duplicate column names (excluding index columns)
        all_column_names = [
            name for table in tables for name in table.schema.names if name not in self.index_columns
        ]
        duplicate_columns = {name for name in all_column_names if all_column_names.count(name) > 1}

        # Rename duplicate columns to avoid conflicts
        renamed_tables = []
        for i, table in enumerate(tables):
            new_column_names = [
                f"{name}_file{i+1}" if name in duplicate_columns else name
                for name in table.schema.names
            ]
            renamed_table = table.rename_columns(new_column_names)
            renamed_tables.append(renamed_table)

        # Ensure all tables have the same schema by adding missing columns with null values
        unified_schema = pa.unify_schemas([table.schema for table in renamed_tables])
        aligned_tables = []
        for table in renamed_tables:
            for field in unified_schema:
                if field.name not in table.schema.names:
                    null_array = pa.array([None] * len(table), type=field.type)
                    table = table.append_column(field.name, null_array)
            aligned_tables.append(table)

        # Select index columns from the first table
        index_table = aligned_tables[0].select(self.index_columns)

        # Collect unique data columns (excluding index columns)
        data_columns = []
        seen_columns = set(self.index_columns)  # Start with index columns to avoid duplicates
        for table in aligned_tables:
            for column_name, column_array in zip(table.schema.names, table.columns):
                if column_name not in seen_columns:
                    data_columns.append((column_name, column_array))
                    seen_columns.add(column_name)

        # Combine index columns and unique data columns
        combined_arrays = index_table.columns + [array for _, array in data_columns]

        # If columns are explicitly specified, filter the final output
        if columns:
            final_columns = self.index_columns + columns
            combined_schema = pa.schema([field for field in unified_schema if field.name in final_columns])
            combined_arrays = [array for array, field in zip(combined_arrays, unified_schema) if field.name in final_columns]
        else:
            combined_schema = pa.schema(
                list(index_table.schema) +
                [pa.field(name, array.type) for name, array in data_columns]
            )

        return pa.Table.from_arrays(combined_arrays, schema=combined_schema)

    def _align_schema(self, table: pa.Table, unified_schema: pa.Schema) -> pa.Table:
        """
        Aligns the schema of a table with the unified schema by adding missing columns with null values.

        Args:
            table (pa.Table): The table to align.
            unified_schema (pa.Schema): The unified schema to align with.

        Returns:
            pa.Table: The aligned table.
        """
        for field in unified_schema:
            if field.name not in table.schema.names:
                null_array = pa.array([None] * len(table), type=field.type)
                table = table.append_column(field.name, null_array)
        reordered_columns = [table[field.name] for field in unified_schema]
        return pa.Table.from_arrays(reordered_columns, schema=unified_schema)

    def _process_chunk(self, table: pa.Table, filter_query: Optional[str], columns: Optional[List[str]]) -> pa.Table:
        """
        Processes a single chunk by applying column selection.

        Args:
            table (pa.Table): The chunk to process.
            filter_query (Optional[str]): Pandas-style filter expression (no longer used here).
            columns (Optional[List[str]]): List of columns to include.

        Returns:
            pa.Table: Processed chunk.
        """
        logging.debug("Processing chunk with schema: %s", table.schema)

        # Select specific columns if provided
        if columns:
            logging.debug("Selecting columns: %s", columns)
            try:
                table = table.select(columns)
                logging.debug("Selected chunk schema: %s", table.schema)
            except KeyError as e:
                logging.error("Error selecting columns: %s", e)
                raise ValueError(f"Failed to select columns: {columns}\nError: {e}")

        return table

    @staticmethod
    def _validate_filter(filter_query: Optional[str], schema: pa.Schema) -> None:
        """
        Validates the filter query against the table schema.

        Args:
            filter_query (Optional[str]): Pandas-style filter expression.
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
