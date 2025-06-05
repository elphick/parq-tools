from pathlib import Path
from typing import Optional


def parquet_memory_usage(input_path: Path,
                         chunk_size: int = 100_000,
                         columns: Optional[list[str]] = None,
                         max_chunks: Optional[int] = None,
                         report_pandas: bool = True,
                         index_columns: Optional[list[str]] = None) -> dict:
    """
    Estimate memory usage of a Parquet file, per column, in three ways:

    - compressed (on disk)
    - decompressed (Arrow/pyarrow in-memory)
    - pandas DataFrame memory usage (optional)

    Processes the file in chunks for scalability.

    Args:
        input_path (Path): Path to the Parquet file.
        chunk_size (int): Number of rows per chunk to process.
        columns (Optional[list[str]]): Columns to include. If None, use all columns.
        max_chunks (Optional[int]): If set, only process up to this many chunks (for Arrow/pyarrow sampling).
        report_pandas (bool): Whether to report pandas DataFrame memory usage. Default True.
        index_columns (Optional[list[str]]): List of columns to mark as index columns in the report.

    Returns:
        dict: Detailed memory usage report with the following structure::

            {
                'columns': {
                    col: {
                        'compressed_bytes': int,      # On-disk size for this column
                        'decompressed_bytes': int,   # In-memory (Arrow) size for this column
                        'pandas_bytes': int or None, # In-memory (pandas) size for this column, or None if not reported
                        'dtype': str,                # Arrow dtype string
                        'is_index': bool             # True if column is marked as index
                    },
                    ...
                },
                'total_compressed_bytes': int,      # Total on-disk size
                'total_decompressed_bytes': int,    # Total Arrow in-memory size
                'total_pandas_bytes': int or None,  # Total pandas in-memory size, or None if not reported
                'shape': tuple                      # (n_rows, n_cols)
            }
    """
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    import pyarrow as pa
    import pandas as pd
    from collections import defaultdict

    dataset = ds.dataset(input_path, format="parquet")
    schema = dataset.schema
    all_columns = schema.names
    use_columns = columns if columns is not None else all_columns
    scanner = dataset.scanner(columns=use_columns, batch_size=chunk_size)
    n_rows = dataset.count_rows()
    n_cols = len(use_columns)

    # Compressed size: get from file metadata
    pq_file = pq.ParquetFile(str(input_path))
    col_compressed = defaultdict(int)
    for rg in range(pq_file.num_row_groups):
        row_group = pq_file.metadata.row_group(rg)
        for ci in range(row_group.num_columns):
            col_name = row_group.column(ci).path_in_schema
            if col_name in use_columns:
                col_compressed[col_name] += row_group.column(ci).total_compressed_size
    total_compressed = sum(col_compressed.values())

    # Decompressed size: estimate by reading chunks
    col_decompressed = defaultdict(int)
    col_pandas = {k: 0 for k in use_columns} if report_pandas else {k: None for k in use_columns}
    total_pandas = 0 if report_pandas else None
    nrows_processed = 0
    for i, batch in enumerate(scanner.to_batches()):
        if max_chunks is not None and i >= max_chunks:
            break
        table = pa.Table.from_batches([batch])
        nrows_processed += table.num_rows
        for col in use_columns:
            arr = table[col]
            col_decompressed[col] += arr.nbytes
        if report_pandas:
            df = table.to_pandas(ignore_metadata=True)
            mem = df.memory_usage(deep=True)
            for col in use_columns:
                col_pandas[col] += int(mem.get(col, 0))
            total_pandas += mem.sum()
    scale = n_rows / nrows_processed if nrows_processed and nrows_processed < n_rows else 1.0
    col_decompressed = {k: int(v * scale) for k, v in col_decompressed.items()}
    total_decompressed = sum(col_decompressed.values())
    if report_pandas:
        scale_pandas = n_rows / nrows_processed if nrows_processed and nrows_processed < n_rows else 1.0
        col_pandas = {k: int(v * scale_pandas) for k, v in col_pandas.items()}
        total_pandas = int(total_pandas * scale_pandas)
    dtypes = {name: str(schema.field(name).type) for name in use_columns}
    index_columns = set(index_columns or [])
    result = {
        'columns': {
            name: {
                'compressed_bytes': int(col_compressed.get(name, 0)),
                'decompressed_bytes': int(col_decompressed.get(name, 0)),
                'pandas_bytes': col_pandas.get(name, None),
                'dtype': dtypes[name],
                'is_index': name in index_columns,
            }
            for name in use_columns
        },
        'total_compressed_bytes': int(total_compressed),
        'total_decompressed_bytes': int(total_decompressed),
        'total_pandas_bytes': int(total_pandas) if report_pandas else None,
        'shape': (n_rows, n_cols),
    }
    return result


def _humanize_bytes(num):
    if num is None:
        return "-"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if abs(num) < 1024.0:
            return f"{num:6.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def print_parquet_memory_usage(report: dict) -> None:
    """
    Print a console-friendly summary of the memory usage report from parquet_memory_usage().
    Index columns are marked with a '*'.
    Dtype is shown first, and if it contains <...>, the content inside <...> is split into a new column.
    Byte values are humanized for compactness.
    """
    print(f"Shape: {report['shape']}")
    print(f"Total compressed: {_humanize_bytes(report['total_compressed_bytes'])}")
    print(f"Total decompressed (Arrow): {_humanize_bytes(report['total_decompressed_bytes'])}")
    if report.get('total_pandas_bytes') is not None:
        print(f"Total pandas: {_humanize_bytes(report['total_pandas_bytes'])}")
    print("\nPer-column breakdown:")
    has_angle = any('<' in stats['dtype'] and '>' in stats['dtype'] for stats in report['columns'].values())
    if has_angle:
        print(f"{'Column':20} {'Dtype':15} {'Compressed':>12} {'Arrow':>12} {'Pandas':>12} {'Dtype details':30}")
    else:
        print(f"{'Column':20} {'Dtype':15} {'Compressed':>12} {'Arrow':>12} {'Pandas':>12}")
    for col, stats in report['columns'].items():
        colname = col + ('*' if stats.get('is_index') else '')
        dtype = stats['dtype']
        compressed = _humanize_bytes(stats['compressed_bytes'])
        arrow = _humanize_bytes(stats['decompressed_bytes'])
        pandas = _humanize_bytes(stats['pandas_bytes'])
        if has_angle and '<' in dtype and '>' in dtype:
            base, rest = dtype.split('<', 1)
            details = rest.rstrip('>')
            dtype_main = base.strip() + '<>'
            print(f"{colname:20} {dtype_main:15} {compressed:>12} {arrow:>12} {pandas:>12} {details:30}")
        else:
            print(f"{colname:20} {dtype:15} {compressed:>12} {arrow:>12} {pandas:>12}")
