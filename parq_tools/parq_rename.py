import logging
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow as pa
from typing import List, Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def rename_parquet_columns(
    input_path: Path,
    output_path: Path,
    rename_map: dict[str, str],
    chunk_size: int = 100_000,
    show_progress: bool = False,
    return_all_columns: bool = True
):
    dataset = ds.dataset(input_path, format="parquet")
    columns = dataset.schema.names if return_all_columns else list(rename_map.keys())
    scanner = dataset.scanner(columns=columns, batch_size=chunk_size)
    total_rows = dataset.count_rows()
    progress = tqdm(total=total_rows, desc="Renaming columns", unit="rows") if HAS_TQDM and show_progress else None

    batches = scanner.to_batches()
    total_written = 0
    writer = None

    for batch in batches:
        table = pa.Table.from_batches([batch])
        # Rename columns
        new_names = [rename_map.get(name, name) for name in table.schema.names]
        table = table.rename_columns(new_names)
        if writer is None:
            writer = pq.ParquetWriter(output_path, schema=table.schema)
        writer.write_table(table)
        total_written += table.num_rows
        if progress:
            progress.update(table.num_rows)

    if writer:
        writer.close()
    if progress:
        progress.close()
    logging.info(f"Shape of renamed data: ({total_written}, {len(new_names)})")