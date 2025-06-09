from pathlib import Path
from typing import Union, Optional

import pyarrow.parquet as pq
import json

from pyarrow.parquet import ParquetFile


def get_pandas_metadata(parquet_file: Union[Path, ParquetFile]) -> Optional[dict]:
    pf: ParquetFile = parquet_file if isinstance(parquet_file, ParquetFile) else pq.ParquetFile(parquet_file)
    schema = pf.schema_arrow
    meta = schema.metadata
    if meta and b"pandas" in meta:
        return json.loads(meta[b"pandas"].decode())
    return None


def merge_pandas_metadata(metadatas):
    # Simple merge: union columns, keep first index_columns, etc.
    columns = []
    seen = set()
    for meta in metadatas:
        for col in meta["columns"]:
            if col["name"] not in seen:
                columns.append(col)
                seen.add(col["name"])
    merged = {
        "columns": columns,
        "index_columns": metadatas[0]["index_columns"],
        "column_indexes": metadatas[0].get("column_indexes", []),
        "creator": metadatas[0].get("creator", {}),
        "pandas_version": metadatas[0].get("pandas_version", "2.0.0"),
    }
    return merged


def get_table_metadata(parquet_file: Union[Path, ParquetFile]) -> dict:
    """Return the table-level metadata as a dict (decoded if possible)."""
    pf: ParquetFile = parquet_file if isinstance(parquet_file, ParquetFile) else pq.ParquetFile(parquet_file)
    return {k.decode(): v.decode(errors="replace") for k, v in pf.metadata.metadata.items()}
    return {}


def get_column_metadata(parquet_file: Union[Path, pq.ParquetFile]) -> dict:
    """Return a dict mapping column names to their decoded metadata dicts."""
    pf: pq.ParquetFile = parquet_file if isinstance(parquet_file, pq.ParquetFile) else pq.ParquetFile(str(parquet_file))
    col_meta = {}
    schema = pf.schema_arrow
    for field in schema:
        meta = {}
        if field.metadata:
            meta = {k.decode(): v.decode(errors="replace") for k, v in field.metadata.items()}
        col_meta[field.name] = meta
    return col_meta
