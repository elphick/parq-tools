"""
parq_rename.py

Utilities for renaming columns in Parquet files, supporting chunked processing, progress reporting, and flexible
output column selection.

Main API:

- rename_parquet_columns: Rename columns in a Parquet file using a mapping, with options for batching and output
  column selection.
"""

import logging
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
from typing import Optional

from parq_tools.utils import atomic_output_file

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def rename_and_update_metadata(
        input_path: Path,
        output_path: Path,
        rename_map: Optional[dict[str, str]] = None,
        table_metadata: Optional[dict[str, str]] = None,
        column_metadata: Optional[dict[str, dict[str, str]]] = None,
        chunk_size: int = 100_000,
        show_progress: bool = False,
        return_all_columns: bool = True
) -> None:
    """
    Rename columns, update table metadata, update column metadata in a Parquet file.
    """
    if rename_map is None:
        rename_map = {}
    if table_metadata is None:
        table_metadata = {}
    if column_metadata is None:
        column_metadata = {}

    if not rename_map and not table_metadata and not column_metadata:
        logging.warning("No updates specified. Exiting without changes.")
        return

    dataset = ds.dataset(input_path, format="parquet")
    columns = dataset.schema.names if return_all_columns else list(rename_map.keys())
    scanner = dataset.scanner(columns=columns, batch_size=chunk_size)
    total_rows = dataset.count_rows()
    progress = tqdm(total=total_rows, desc="Processing", unit="rows") if HAS_TQDM and show_progress else None

    batches = scanner.to_batches()
    total_written = 0

    with atomic_output_file(output_path) as tmp_file:
        writer = None

        for batch in batches:
            table = pa.Table.from_batches([batch])
            # Rename columns
            new_names = [rename_map.get(name, name) for name in table.schema.names]
            table = table.rename_columns(new_names)
            # Only set table-level metadata on the first batch
            if writer is None:
                fields = []
                for name, field in zip(table.schema.names, table.schema):
                    meta = field.metadata or {}
                    col_meta = column_metadata.get(name, {})
                    meta = dict(meta)
                    for k, v in col_meta.items():
                        meta[k.encode()] = v.encode()
                    fields.append(pa.field(name, field.type, metadata=meta if meta else None))
                schema = pa.schema(
                    fields,
                    metadata={k.encode(): v.encode() for k, v in table_metadata.items()} if table_metadata else None
                )
                table = table.cast(schema)
                writer = pq.ParquetWriter(tmp_file, schema=table.schema)

            if writer is None:
                writer = pq.ParquetWriter(tmp_file, schema=table.schema)
            writer.write_table(table)
            total_written += table.num_rows
            if progress:
                progress.update(table.num_rows)

        if writer:
            writer.close()
        if progress:
            progress.close()
        logging.info(f"Shape of processed data: ({total_written}, {len(new_names)})")
