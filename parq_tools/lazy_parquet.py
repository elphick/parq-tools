from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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

    def __init__(self, path, index_cols: Optional[list[str]] = None):
        """ Initialize a LazyParquetDataFrame.

        Args:
            path (str or Path): Path to the Parquet file.
            index_cols (list[str]): List of column names, if any, to be used as index columns.

        """
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
