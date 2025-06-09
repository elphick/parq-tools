"""
lazy_parquet.py

Utilities for lazy loading and accessing columns from Parquet files as pandas Series,
without loading the entire file into memory.

Main API:

- LazyParquetDataFrame: Class for lazy loading Parquet files, allowing access to columns as pandas Series.


"""

import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path


class LazyParquetDataFrame:
    def __init__(self, parquet_path: Path):
        self._parquet_path = parquet_path
        self._schema = pq.read_schema(parquet_path)
        self._columns = self._schema.names
        self._cache = {}

        # Detect index columns if present (pandas stores them as metadata)
        meta = pq.read_metadata(parquet_path)
        self._index_columns = []
        if meta is not None and meta.metadata is not None:
            pandas_metadata = meta.metadata.get(b'pandas', None)
            if pandas_metadata:
                import json
                pandas_meta = json.loads(pandas_metadata.decode())
                self._index_columns = [f['name'] for f in pandas_meta.get('index_columns', []) if
                                       isinstance(f, dict) and f.get('name')]

    def __getitem__(self, col):
        if col not in self._columns:
            raise KeyError(f"Column '{col}' not found in Parquet file.")
        if col not in self._cache:
            # Always load index columns if present, so Series has correct index
            columns = self._index_columns + [col] if self._index_columns else [col]
            table = pq.read_table(self._parquet_path, columns=columns)
            df = table.to_pandas()
            series = df[col]
            self._cache[col] = series
        return self._cache[col]

    def __setitem__(self, col, value):
        # Accepts a pandas Series or array-like, stores in cache
        self._cache[col] = pd.Series(value)
        if col not in self._columns:
            self._columns = list(self._columns) + [col]

    def __getattr__(self, col):
        if col in self._columns:
            return self[col]
        raise AttributeError(f"'LazyParquetDataFrame' object has no attribute '{col}'")

    @property
    def columns(self):
        return self._columns

    @property
    def cached_df(self):
        """Return a DataFrame of all columns currently loaded in cache."""
        return self.to_pandas(cache_only=True)

    def to_pandas(self, columns=None, cache_only=False):
        """
        Load columns as a DataFrame.

        Args:
            columns: List of columns to load. If None, loads all columns.
            cache_only: If True, only return columns already in cache as a DataFrame.

        Returns:
            pd.DataFrame
        """
        if cache_only:
            # Only use columns already loaded in cache
            cols = list(self._cache.keys()) if columns is None else [c for c in columns if c in self._cache]
            data = {col: self._cache[col] for col in cols}
            df = pd.DataFrame(data)
        else:
            if columns is None:
                columns = self._columns
            # Always include index columns if present
            all_columns = list(set(columns) | set(self._index_columns)) if self._index_columns else columns
            # Use cache for already loaded columns, read missing ones
            data = {col: self._cache[col] for col in all_columns if col in self._cache}
            missing = [col for col in all_columns if col not in self._cache]
            if missing:
                table = pq.read_table(self._parquet_path, columns=missing)
                df_new = table.to_pandas()
                for col in missing:
                    self._cache[col] = df_new[col]
                    data[col] = df_new[col]
            df = pd.DataFrame(data)
        # Set index if requested and index columns are present
        if self._index_columns:
            df = df.set_index(self._index_columns)
        return df
