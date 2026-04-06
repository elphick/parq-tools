from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, TextIO

import sys
import warnings

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# PyArrow-style predicate: (column_name, operator, value)
Filter = Tuple[str, str, object]


class LazyParquetDF:
    """Lazy, column-on-demand DataFrame backed by a Parquet file.

    This lightweight, DataFrame-like object exposes a familiar subset of the
    pandas API, but loads data lazily from a Parquet file. Columns are only
    materialized into memory when they are first accessed.

    Parameters
    ----------
    path : Path
        Path to the Parquet file.
    index_col : str or sequence of str, optional
        Optional column(s) to use as the index. If provided, those columns are
        eagerly loaded and set as the index (supporting both single index and
        MultiIndex).
    """

    def __init__(self, path: Path, index_col: Optional[Sequence[str] | str] = None) -> None:
        self._path = Path(path)
        self._index_col: Optional[Sequence[str] | str] = index_col

        # Internal cache of loaded/mutated columns as a pandas DataFrame.
        # This frame always has the logical index (either RangeIndex or an
        # explicit index based on index_col / pandas metadata) so we can
        # align slices for chunked operations.
        # We always initialise it with the correct index immediately.
        self._df = pd.DataFrame()
        self._parquet_file = pq.ParquetFile(self._path)
        self._schema = self._parquet_file.schema
        self._available_columns: List[str] = list(self._schema.names)
        self._column_order: List[str] = list(self._available_columns)
        self._mutated_schema_columns: Set[str] = set()
        self._new_columns: Set[str] = set()

        self._n_rows: int = sum(
            self._parquet_file.metadata.row_group(i).num_rows
            for i in range(self._parquet_file.metadata.num_row_groups)
        )

        # Index strategy:
        # 1) Explicit index_col -> build index from those columns.
        # 2) Otherwise, delegate index reconstruction to pandas.read_parquet,
        #    so we exactly mirror pandas’ behaviour (named index or RangeIndex).
        if self._index_col is not None:
            index_cols: List[str]
            if isinstance(self._index_col, str):
                index_cols = [self._index_col]
            else:
                if not isinstance(self._index_col, Sequence):
                    raise TypeError("index_col must be str or sequence of str")
                index_cols = list(self._index_col)

            missing = [c for c in index_cols if c not in self._available_columns]
            if missing:
                raise KeyError(
                    f"index_col(s) {missing!r} not found in Parquet schema."
                )

            table = self._parquet_file.read(columns=index_cols)
            idx_df = table.to_pandas()[index_cols]
            if len(index_cols) == 1:
                # Fix typo: use index_cols, not indexCols
                idx = idx_df[index_cols[0]].rename(index_cols[0])
                self._df = pd.DataFrame(index=idx)
            else:
                mi = pd.MultiIndex.from_frame(idx_df)
                mi.set_names(index_cols, inplace=True)
                self._df = pd.DataFrame(index=mi)
        else:
            # No explicit index override. Let pandas interpret any stored
            # metadata and reconstruct the logical index. For parquet files
            # written without an index, this will just be a RangeIndex. For
            # files like the test parquet (df.set_index("i").to_parquet),
            # this will be an Index named "i", matching the test’s
            # expectations and pd.read_parquet.
            try:
                pdf = pd.read_parquet(self._path)
                # Use pandas to reconstruct the logical index exactly as it
                # interprets the stored metadata.
                idx = pdf.index
                self._df = pd.DataFrame(index=idx)

                # When the index comes from pandas metadata (e.g. df.set_index("i")),
                # the corresponding index name(s) should not appear in the logical
                # columns list; they are part of the index, not data columns.
                index_names: list[str] = []
                if isinstance(idx, pd.MultiIndex):
                    # MultiIndex.names may contain None; filter those out.
                    index_names = [n for n in idx.names if n is not None]
                else:
                    if idx.name is not None:
                        index_names = [idx.name]

                if index_names:
                    # Remove any index names from the logical column order so that
                    # LazyParquetDF.columns only exposes data columns, matching
                    # pd.read_parquet for metadata-indexed files.
                    self._column_order = [
                        c for c in self._column_order if c not in index_names
                    ]
            except Exception:
                # Fallback: if pandas cannot read the parquet for any reason,
                # we still provide a sensible positional index based on the
                # stored row count.
                self._df = pd.DataFrame(index=pd.RangeIndex(self._n_rows))

    # ------------------------------------------------------------------ #
    # Basic DataFrame-like properties
    # ------------------------------------------------------------------ #

    @property
    def columns(self) -> List[str]:
        """List of all logical column names.

        This includes Parquet schema columns in schema order, minus any index
        columns (when the index is constructed from pandas metadata or via the
        explicit index_col argument), followed by any new columns that have
        been added via assignment. The order is preserved across operations
        and is used by chunked iteration and write-back.
        """

        return list(self._column_order)

    @property
    def shape(self) -> tuple[int, int]:
        """Tuple of (number of rows, number of columns).

        The column count reflects the *logical* data columns exposed via
        :attr:`columns`, excluding any index columns that are represented
        solely in the index (either reconstructed from pandas metadata or
        via ``index_col``).
        """

        return self._n_rows, len(self.columns)

    def __len__(self) -> int:
        """Return the number of rows in the dataset."""

        return self._n_rows

    @property
    def index(self) -> pd.Index:
        """Index for the dataset.

        Returns a RangeIndex when no data has been loaded yet, or the index
        of the internal cached DataFrame (which may be a MultiIndex).
        """

        if self._df is not None and len(self._df.index) > 0:
            return self._df.index
        return pd.RangeIndex(self._n_rows)

    @property
    def dtypes(self) -> pd.Series:
        """Return dtypes for columns currently materialised in the cache.

        This mirrors :attr:`pandas.DataFrame.dtypes`: only columns that
        actually exist in the internal pandas DataFrame are reported.
        Lazily-backed columns that have not yet been loaded are not
        included; use :meth:`info` to inspect all available columns and
        their lazy/loaded status.
        """

        return self._df.dtypes

    # ------------------------------------------------------------------ #
    # Column access and loading
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: str) -> pd.Series:
        """Return a column as a pandas Series, loading it lazily if needed."""

        if key not in self._column_order:
            raise KeyError(f"Column {key!r} not found in lazy frame.")
        if key in self._available_columns and key not in self._df.columns:
            self._ensure_columns_loaded([key])
        return self._df[key]

    def __setitem__(self, key: str, value: object) -> None:
        """Add or overwrite a column.

        The value must be broadcastable to the number of rows in the dataset.
        """

        if self._df.empty:
            self._df = pd.DataFrame(index=self.index)

        series = pd.Series(value, index=self.index)
        if len(series) != self._n_rows:
            raise ValueError(
                f"Length of assigned column ({len(series)}) does not match "
                f"number of rows ({self._n_rows})."
            )

        self._df[key] = series

        if key in self._available_columns:
            self._mutated_schema_columns.add(key)
        else:
            self._new_columns.add(key)
            if key not in self._column_order:
                self._column_order.append(key)

    def add_column(self, name: str, data: object) -> None:
        """Explicit helper for adding a new column (``df[name] = data``)."""

        self[name] = data

    def load_columns(self, columns: Iterable[str]) -> None:
        """Eagerly load one or more columns into the internal cache."""

        cols = list(columns)
        missing = [c for c in cols if c not in self._available_columns]
        if missing:
            raise KeyError(f"Columns not found in Parquet schema: {missing}")
        self._ensure_columns_loaded(cols)

    def to_pandas(self) -> pd.DataFrame:
        """Materialize all columns as a pandas DataFrame."""

        missing = [
            c
            for c in self._available_columns
            if c not in self._df.columns and c not in self._mutated_schema_columns
        ]
        if missing:
            self._ensure_columns_loaded(missing)

        pdf = self._df.copy()

        # If an explicit index_col was provided, materialise a MultiIndex (or
        # single Index) on the returned DataFrame. This makes the external
        # behaviour match ``pd.read_parquet(...).set_index(index_col)`` while
        # allowing the internal cache to use a simpler index for lazy ops.
        idx_cols: List[str] = []
        if self._index_col is not None:
            if isinstance(self._index_col, str):
                idx_cols = [self._index_col]
            else:
                idx_cols = list(self._index_col)
            if all(col in pdf.columns for col in idx_cols):
                pdf = pdf.set_index(idx_cols)

        for col in pdf.columns:
            if col not in self._column_order:
                self._column_order.append(col)

        # If we have converted some columns into an index, they should no
        # longer be part of the returned column order.
        effective_columns = [
            c for c in self._column_order if c not in idx_cols
        ]
        return pdf[effective_columns]

    # ------------------------------------------------------------------ #
    # Simple pandas-like helpers
    # ------------------------------------------------------------------ #

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first *n* rows as a pandas DataFrame."""

        table = self._parquet_file.read_row_group(0, columns=self._available_columns)
        pdf = table.to_pandas()
        return pdf.head(n)

    def describe(
        self,
        percentiles: Optional[list[float]] = None,
        include=None,
        exclude=None,
        datetime_is_numeric: bool = False,
    ) -> pd.DataFrame:
        """Generate descriptive statistics of the dataset."""

        pdf = self.to_pandas()
        # pandas < 1.1 does not support datetime_is_numeric; pass only
        # the arguments that are universally supported.
        return pdf.describe(
            percentiles=percentiles,
            include=include,
            exclude=exclude,
        )

    def info(self, buf: Optional[TextIO] = None) -> None:
        """Print a concise summary of the lazy Parquet-backed DataFrame."""

        if buf is None:
            buf = sys.stdout

        n_rows, n_cols = self.shape
        header = (
            f"<LazyParquetDF>\n"
            f"Path: {self._path}\n"
            f"Rows: {n_rows}, Columns: {n_cols}\n"
        )
        print(header, file=buf)

        print("Columns:", file=buf)
        loaded_cols = set(self._df.columns)

        for name in self._available_columns:
            if name in loaded_cols:
                series = self._df[name]
                non_null = series.count()
                dtype = series.dtype
                status = "loaded"
            else:
                # Use the logical schema field for type information without
                # attempting to read any data or rely on column index lookup.
                # ``ParquetSchema`` exposes fields positionally; we look up the
                # index of the column by name first, then fetch the field.
                try:
                    idx = self._schema.get_field_index(name)
                except AttributeError:
                    # Older pyarrow: fallback to a simple name lookup over
                    # ``names`` and then index via ``column``.
                    try:
                        idx = self._schema.names.index(name)
                    except ValueError:
                        idx = -1

                if idx == -1:
                    # Should not happen for a valid schema-backed column, but
                    # be defensive and mark it as object.
                    field_type = "object"
                else:
                    try:
                        field = self._schema.column(idx)
                        field_type = field.physical_type
                    except Exception:
                        field_type = "object"

                non_null = "lazy"
                dtype = field_type
                status = "lazy"

            print(
                f"  - {name}: non-null={non_null}, dtype={dtype}, status={status}",
                file=buf,
            )

    # ------------------------------------------------------------------ #
    # Filtering & query
    # ------------------------------------------------------------------ #

    def filter(self, *predicates: Filter) -> pd.DataFrame:
        """Filter rows using explicit PyArrow-style predicate tuples."""

        if not predicates:
            raise ValueError("At least one filter predicate must be supplied.")

        predicate_cols = [col for col, _, _ in predicates]
        missing = [c for c in predicate_cols if c not in self._available_columns]
        if missing:
            raise KeyError(f"Predicate columns not in schema: {missing}")

        table = pq.read_table(
            self._path,
            columns=predicate_cols,
            filters=list(predicates),
        )
        pdf = table.to_pandas()
        return pdf

    def query(self, expr: str) -> pd.DataFrame:
        """Evaluate a boolean expression using pandas-style query syntax."""

        pdf = self.to_pandas()
        return pdf.query(expr)

    # ------------------------------------------------------------------ #
    # Chunked iteration & write-back
    # ------------------------------------------------------------------ #

    def iter_row_chunks(
        self,
        chunk_size: int = 100_000,
        columns: Optional[Iterable[str]] = None,
    ) -> Iterable[pd.DataFrame]:
        """Iterate over the dataset in row-wise chunks."""

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        cols = list(columns) if columns is not None else list(self.columns)
        unknown = [c for c in cols if c not in self.columns]
        if unknown:
            raise KeyError(f"Columns not found in lazy frame: {unknown}")

        parquet_cols = [
            c
            for c in cols
            if c in self._available_columns and c not in self._mutated_schema_columns
        ]
        computed_cols = [
            c
            for c in cols
            if c in self._new_columns or c in self._mutated_schema_columns
        ]

        if computed_cols and self._df.empty:
            raise RuntimeError(
                "Computed or mutated columns exist but internal frame is empty; "
                "this is an internal inconsistency."
            )

        start = 0

        if parquet_cols:
            for batch in self._parquet_file.iter_batches(
                batch_size=chunk_size, columns=parquet_cols
            ):
                pdf = batch.to_pandas()
                n = len(pdf)

                for col in computed_cols:
                    col_series = self._df[col].iloc[start : start + n].reset_index(
                        drop=True
                    )
                    pdf[col] = col_series

                pdf = pdf[cols]

                index_slice = self.index[start : start + n]
                pdf.index = index_slice

                start += n
                yield pdf
        else:
            pdf = self.to_pandas()[cols]
            while start < self._n_rows:
                end = min(start + chunk_size, self._n_rows)
                chunk = pdf.iloc[start:end]
                start = end
                yield chunk

    def to_parquet(
        self,
        path: Path,
        *,
        allow_overwrite: bool = False,
        chunk_size: Optional[int] = None,
        **pq_write_kwargs: object,
    ) -> None:
        """Write the logical DataFrame to a Parquet file."""

        target = Path(path)
        if target.exists() and not allow_overwrite:
            raise FileExistsError(f"Target file already exists: {target}")

        if chunk_size is None:
            pdf = self.to_pandas()
            pdf.to_parquet(target, **pq_write_kwargs)
            return

        writer: Optional[pq.ParquetWriter] = None
        try:
            for chunk in self.iter_row_chunks(chunk_size=chunk_size, columns=self.columns):
                table = pa.Table.from_pandas(chunk)
                if writer is None:
                    writer = pq.ParquetWriter(target, table.schema, **pq_write_kwargs)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    def save(
        self,
        *,
        allow_overwrite: bool = False,
        chunk_size: int = 100_000,
        **pq_write_kwargs: object,
    ) -> None:
        """Save the logical DataFrame back to its original Parquet path."""

        self.to_parquet(
            self._path,
            allow_overwrite=allow_overwrite,
            chunk_size=chunk_size,
            **pq_write_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_columns_loaded(self, columns: List[str]) -> None:
        """Load columns from Parquet into the internal cache if needed.

        Parameters
        ----------
        columns : list[str]
            Column names to ensure are present in the internal DataFrame.
        """
        to_load = [c for c in columns if c not in self._df.columns]
        if not to_load:
            return

        # Use the existing ParquetFile for efficiency.
        table = self._parquet_file.read(columns=to_load)
        new_df = table.to_pandas()

        if not self._df.empty:
            # Align on index and join columns. When an index column has been
            # set, ``new_df`` still carries it as a regular column; we align
            # purely by row order to maintain consistency with the underlying
            # Parquet layout.
            new_df.index = pd.RangeIndex(len(new_df))
            if isinstance(self._df.index, pd.RangeIndex):
                # Simple positional join.
                self._df = self._df.join(new_df, how="left")
            else:
                # For non-RangeIndex (e.g. MultiIndex based on index_col),
                # align by position by temporarily resetting the index.
                base = self._df.reset_index(drop=True)
                base = base.join(new_df, how="left")
                base.index = self._df.index
                self._df = base
        else:
            # No columns loaded yet: start from an empty DataFrame that has
            # the correct index (which may have been constructed in __init__).
            # Attach the newly loaded columns by position without changing the
            # existing index.
            df = pd.DataFrame(index=pd.RangeIndex(len(new_df)))
            df = df.join(new_df, how="left")
            df.index = self.index
            self._df = df


class LazyLocIndexer:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, key):
        # If key is (row, col), ensure col is loaded
        if isinstance(key, tuple) and len(key) == 2:
            row_key, col_key = key
            _ = self.parent[col_key]  # Triggers column load and cache
            return self.parent.to_pandas().loc[key]
        else:
            return self.parent.to_pandas().loc[key]

    def __setitem__(self, key, value):
        df = self.parent.to_pandas()
        df.loc[key] = value
        # noinspection PyProtectedMember
        self.parent._update_from_pandas(df)


class LazyParquetDataFrame:

    """Deprecated lazy Parquet DataFrame wrapper.

    This class has been superseded by :class:`LazyParquetDF` and will be
    removed in a future release.

    Notes
    -----
    New code should use :class:`LazyParquetDF` instead. The
    :class:`LazyParquetDataFrame` implementation is kept only for
    backwards compatibility and is no longer actively developed.
    """

    def __init__(self, path, index_cols: Optional[list[str]] = None):
        # Emit a deprecation warning on construction so that callers are
        # redirected towards :class:`LazyParquetDF`.
        warnings.warn(
            "LazyParquetDataFrame is deprecated and will be removed in a "
            "future release. Please use LazyParquetDF instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.path = path
        self._schema = pq.read_schema(path)
        self._loaded_columns = {}
        self._extra_columns = {}
        self._column_order = list(self._schema.names)
        self._pandas_cache = None
        self._index_cols = []
        self._index = None

        meta = pq.read_metadata(path).metadata or {}

        if index_cols is not None:
            self._index_cols = list(index_cols)
            index_df = pq.read_table(path, columns=self._index_cols).to_pandas()
            if len(self._index_cols) == 1:
                col = self._index_cols[0]
                self._index = pd.Index(index_df[col], name=col)
            else:
                self._index = pd.MultiIndex.from_frame(index_df)
            self._column_order = [c for c in self._column_order if c not in self._index_cols]
        elif b'pandas' in meta:
            df = pq.read_table(path).to_pandas()
            self._index = df.index
            self._column_order = list(df.columns)
        else:
            num_rows = pq.read_table(path).num_rows
            self._index = pd.RangeIndex(num_rows)

    def set_index(self, columns):
        """Set the index of the DataFrame to the specified columns."""
        if not all(col in self._column_order for col in columns):
            raise KeyError(f"One or more columns {columns} are not in the DataFrame.")
        try:
            index_df = self.to_pandas()[columns]
            self._index = pd.MultiIndex.from_frame(index_df) if len(columns) > 1 else pd.Index(index_df[columns[0]])
        except Exception as e:
            raise ValueError(f"Failed to set index: {e}")
        self._invalidate_cache()

    def reset_index(self, drop=False):
        """Reset the index of the DataFrame, optionally dropping it."""
        if drop:
            self._index = pd.RangeIndex(len(self.to_pandas()))
        else:
            df = self.to_pandas()
            index_df = self._index.to_frame(index=False) if isinstance(self._index, pd.MultiIndex) else self._index
            index_cols = list(index_df.columns) if isinstance(index_df, pd.DataFrame) else [self._index.name]
            for col in index_cols:
                if col in self._column_order:
                    raise ValueError(f"Cannot reset index: column '{col}' already exists.")
            # Add index columns to extra_columns and column_order at the front
            if isinstance(index_df, pd.DataFrame):
                for col in index_df.columns:
                    self._extra_columns[col] = index_df[col]
                self._column_order = index_cols + self._column_order
            else:
                self._extra_columns[self._index.name] = index_df
                self._column_order = [self._index.name] + self._column_order
            self._index = pd.RangeIndex(len(df))
        self._invalidate_cache()

    def to_pandas(self):
        """Convert the Parquet file to a pandas DataFrame, caching the result."""
        if self._pandas_cache is not None:
            return self._pandas_cache
        df = pq.read_table(self.path).to_pandas()
        for k, v in self._extra_columns.items():
            df[k] = v
        df = df[self._column_order]
        df.index = self._index
        self._pandas_cache = df
        return df

    def iter_chunks(self, batch_size=100_000, columns=None):
        """Yield pandas DataFrames in row-wise chunks, including extra columns."""
        pf = pq.ParquetFile(self.path)
        start = 0
        columns = columns or self._column_order
        parquet_columns = [c for c in columns if c in self._schema.names]
        extra_columns = [c for c in columns if c in self._extra_columns]
        for batch in pf.iter_batches(batch_size=batch_size, columns=parquet_columns):
            df = batch.to_pandas()
            # Add extra columns, sliced to the current chunk
            for col in extra_columns:
                col_data = pd.Series(self._extra_columns[col][start:start + len(df)])
                df[col] = col_data.reset_index(drop=True)
            # Reorder columns
            df = df[columns]
            # Set index to the corresponding slice of self._index
            df.index = self._index[start:start + len(df)]
            start += len(df)
            yield df

    def _invalidate_cache(self):
        self._pandas_cache = None

    def __getattr__(self, name):
        # Delegate to pandas if method exists
        if hasattr(pd.DataFrame, name):
            return getattr(self.to_pandas(), name)
        raise AttributeError(f"'LazyParquetDataFrame' object has no attribute '{name}'")

    def __getitem__(self, key):
        if key in self._loaded_columns:
            return self._loaded_columns[key]
        elif key in self._schema.names:
            col = pq.read_table(self.path, columns=[key]).to_pandas()[key]
            # If the column is empty, set dtype from schema or default to float64 if null
            if col.empty:
                field_type = self._schema.field(key).type
                if field_type == "null" or str(field_type) == "null":
                    dtype = "float64"
                else:
                    dtype = field_type.to_pandas_dtype()
                col = pd.Series([], dtype=dtype, name=key)
            self._loaded_columns[key] = col
            return col
        elif key in self._extra_columns:
            return self._extra_columns[key]
        else:
            raise KeyError(f"Column '{key}' not found.")

    def __setitem__(self, key, value):
        if key in self._schema.names or key in self._loaded_columns:
            self._loaded_columns[key] = value
        else:
            self._extra_columns[key] = value
        if key not in self._column_order:
            self._column_order.append(key)
        self._invalidate_cache()

    def add_column(self, name: str, data, position=None):
        """Add a new column to the DataFrame."""
        self._extra_columns[name] = data
        if position is None:
            self._column_order.append(name)
        else:
            self._column_order.insert(position, name)
        self._invalidate_cache()

    def head(self, n: int = 5):
        """Return the first n rows of the DataFrame."""
        return pq.read_table(self.path, columns=self._schema.names).to_pandas().head(n)

    def to_parquet(self, path: Path):
        """Write the DataFrame to a Parquet file."""
        df = self.to_pandas()
        df.to_parquet(path)

    def save(self, path=None, batch_size=100_000):
        """Save the DataFrame to Parquet in chunks to reduce memory usage."""
        target = path or self.path
        writer = None
        for chunk in self.iter_chunks(batch_size=batch_size):
            table = pa.Table.from_pandas(chunk)
            if writer is None:
                writer = pq.ParquetWriter(target, table.schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
        self._invalidate_cache()

    def _update_from_pandas(self, df):
        """Update the internal state from a pandas DataFrame."""
        for col in df.columns:
            if col in self._schema.names:
                self._loaded_columns[col] = df[col]
            else:
                self._extra_columns[col] = df[col]
        self._column_order = list(df.columns)
        self._invalidate_cache()

    @property
    def loc(self):
        return LazyLocIndexer(self)

    @property
    def index(self):
        return self._index

    @property
    def columns(self):
        return self._column_order

    @property
    def shape(self):
        return len(self._index), len(self._column_order)

    @property
    def dtypes(self):
        import pandas as pd
        dtypes = {}
        for name in self._schema.names:
            field = self._schema.field(name)
            field_type = field.type
            try:
                dtype = field_type.to_pandas_dtype()
            except Exception:
                dtype = "object"
            # Map nullable integer/float to pandas extension dtype
            if field.nullable:
                if pd.api.types.is_integer_dtype(dtype):
                    dtypes[name] = f"Int{pd.api.types._get_dtype(dtype).itemsize * 8}"
                    continue
                elif pd.api.types.is_float_dtype(dtype):
                    dtypes[name] = f"Float{pd.api.types._get_dtype(dtype).itemsize * 8}"
                    continue
            ser = pd.Series([], dtype=dtype, name=name)
            # If dtype is object and column is empty, default to float64
            if ser.empty and ser.dtype == "object":
                dtypes[name] = "float64"
            else:
                dtypes[name] = ser.dtype.name
        for name, col in self._extra_columns.items():
            ser = pd.Series(col)
            dtypes[name] = ser.dtype.name
        return pd.Series(dtypes)

    def assign(self, **kwargs):
        """Assign new columns to the DataFrame."""
        df = self.to_pandas().assign(**kwargs)
        new_df = LazyParquetDataFrame(self.path)
        new_df._update_from_pandas(df)
        return new_df

    def insert(self, loc, column, value, allow_duplicates=False):
        """Insert a new column at a specific location."""
        if column in self._column_order and not allow_duplicates:
            raise ValueError(f"Column '{column}' already exists.")
        self._extra_columns[column] = value
        self._column_order.insert(loc, column)
        self._invalidate_cache()

    def drop(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors='raise'):
        """Drop specified labels from the DataFrame."""
        df = self.to_pandas().drop(labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=False,
                                   errors=errors)
        if inplace:
            self._update_from_pandas(df)
            self._invalidate_cache()
            return None
        else:
            new_df = LazyParquetDataFrame(self.path)
            new_df._update_from_pandas(df)
            new_df._invalidate_cache()
            return new_df

    def rename(self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False, level=None,
               errors='ignore'):
        """Rename the columns or index of the DataFrame."""
        df = self.to_pandas().rename(mapper=mapper, index=index, columns=columns, axis=axis, copy=copy, inplace=False,
                                     level=level, errors=errors)
        if inplace:
            self._update_from_pandas(df)
            self._invalidate_cache()
            return None
        else:
            new_df = LazyParquetDataFrame(self.path)
            new_df._update_from_pandas(df)
            new_df._invalidate_cache()
            return new_df

    def __len__(self):
        return len(self.to_pandas())

    def __repr__(self):
        return repr(self.to_pandas())

    def __str__(self):
        return str(self.to_pandas())

    def __iter__(self):
        return iter(self.to_pandas())

    def __contains__(self, item):
        return item in self._column_order

    def __eq__(self, other):
        return self.to_pandas().equals(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __add__(self, other):
        return self.to_pandas() + other

    def __sub__(self, other):
        return self.to_pandas() - other

    def __mul__(self, other):
        return self.to_pandas() * other

    def __truediv__(self, other):
        return self.to_pandas() / other

    def __floordiv__(self, other):
        return self.to_pandas() // other

    def __mod__(self, other):
        return self.to_pandas() % other

    def __pow__(self, other):
        return self.to_pandas() ** other

    def __and__(self, other):
        return self.to_pandas() & other

    def __or__(self, other):
        return self.to_pandas() | other

    def __xor__(self, other):
        return self.to_pandas() ^ other

    def __lt__(self, other):
        return self.to_pandas() < other

    def __le__(self, other):
        return self.to_pandas() <= other

    def __gt__(self, other):
        return self.to_pandas() > other

    def __ge__(self, other):
        return self.to_pandas() >= other

    def __neg__(self):
        return -self.to_pandas()

    def __abs__(self):
        return abs(self.to_pandas())

    def __invert__(self):
        return ~self.to_pandas()

    def __round__(self, n=None):
        return self.to_pandas().round(n)

    def __floor__(self):
        return self.to_pandas().floor()

    def __ceil__(self):
        return self.to_pandas().ceil()

    def __trunc__(self):
        return self.to_pandas().trunc()

    def __radd__(self, other):
        return other + self.to_pandas()

    def __rsub__(self, other):
        return other - self.to_pandas()

    def __rmul__(self, other):
        return other * self.to_pandas()

    def __rtruediv__(self, other):
        return other / self.to_pandas()

    def __rfloordiv__(self, other):
        return other // self.to_pandas()

    def __rmod__(self, other):
        return other % self.to_pandas()

    def __rpow__(self, other):
        return other ** self.to_pandas()

    def __rand__(self, other):
        return other & self.to_pandas()

    def __ror__(self, other):
        return other | self.to_pandas()

    def __rxor__(self, other):
        return other ^ self.to_pandas()

    def __iadd__(self, other):
        self._update_from_pandas(self.to_pandas() + other)
        return self

    def __isub__(self, other):
        self._update_from_pandas(self.to_pandas() - other)
        return self

    def __imul__(self, other):
        self._update_from_pandas(self.to_pandas() * other)
        return self

    def __itruediv__(self, other):
        self._update_from_pandas(self.to_pandas() / other)
        return self

    def __ifloordiv__(self, other):
        self._update_from_pandas(self.to_pandas() // other)
        return self

    def __imod__(self, other):
        self._update_from_pandas(self.to_pandas() % other)
        return self

    def __ipow__(self, other):
        self._update_from_pandas(self.to_pandas() ** other)
        return self

    def __iand__(self, other):
        self._update_from_pandas(self.to_pandas() & other)
        return self

    def __ior__(self, other):
        self._update_from_pandas(self.to_pandas() | other)
        return self

    def __ixor__(self, other):
        self._update_from_pandas(self.to_pandas() ^ other)
        return self

    def __ilshift__(self, other):
        self._update_from_pandas(self.to_pandas() << other)
        return self

    def __irshift__(self, other):
        self._update_from_pandas(self.to_pandas() >> other)
        return self
