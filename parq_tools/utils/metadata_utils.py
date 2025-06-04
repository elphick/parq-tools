import pyarrow.parquet as pq
import json


def extract_pandas_metadata(parquet_file):
    schema = pq.ParquetFile(parquet_file).schema_arrow
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
