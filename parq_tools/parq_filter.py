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
        filter_expression: Optional[str] = None,
        columns: Optional[list[str]] = None,
        chunk_size: int = 100_000,
        show_progress: bool = False
):
    """
    Filter a Parquet file based on a pandas-like expression and a sub-selection of columns.
    """
    dataset = ds.dataset(input_path, format="parquet")
    filter_expr = build_filter_expression(filter_expression, dataset.schema) if filter_expression else None

    scanner = dataset.scanner(
        columns=columns,
        filter=filter_expr,
        batch_size=chunk_size
    )

    total_rows = dataset.count_rows()
    progress = tqdm(total=total_rows, desc="Filtering", unit="rows") if HAS_TQDM and show_progress else None

    # Get schema from the first batch
    batches = scanner.to_batches()
    try:
        first_batch = next(batches)
    except StopIteration:
        return  # No data to write

    table = pa.Table.from_batches([first_batch])
    writer_schema = table.schema

    with pq.ParquetWriter(output_path, schema=writer_schema) as writer:
        writer.write_table(table)
        if progress:
            progress.update(len(first_batch))
        for batch in batches:
            table = pa.Table.from_batches([batch])
            writer.write_table(table)
            if progress:
                progress.update(len(batch))

    if progress:
        progress.close()
    logging.info(f"Filtered {total_rows} rows")