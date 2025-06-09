import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
from pathlib import Path
from parq_tools.lazy_parquet import LazyParquetDataFrame

def make_parquet(tmp_path, data):
    df = pd.DataFrame(data)
    file_path = tmp_path / "test.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    return file_path

def test_columns(tmp_path):
    file_path = make_parquet(tmp_path, {"a": [1, 2], "b": [3, 4]})
    lazy_df = LazyParquetDataFrame(file_path)
    assert set(lazy_df.columns) == {"a", "b"}

def test_getitem_loads_column(tmp_path):
    file_path = make_parquet(tmp_path, {"x": [10, 20], "y": [30, 40]})
    lazy_df = LazyParquetDataFrame(file_path)
    x = lazy_df["x"]
    assert isinstance(x, pd.Series)
    assert list(x) == [10, 20]

def test_getattr_loads_column(tmp_path):
    file_path = make_parquet(tmp_path, {"foo": [5, 6]})
    lazy_df = LazyParquetDataFrame(file_path)
    foo = lazy_df.foo
    assert list(foo) == [5, 6]

def test_to_pandas_subset(tmp_path):
    file_path = make_parquet(tmp_path, {"a": [1, 2], "b": [3, 4]})
    lazy_df = LazyParquetDataFrame(file_path)
    df = lazy_df.to_pandas(columns=["b"])
    assert list(df.columns) == ["b"]
    assert list(df["b"]) == [3, 4]

def test_to_pandas_cache_only(tmp_path):
    file_path = make_parquet(tmp_path, {"a": [1, 2], "b": [3, 4]})
    lazy_df = LazyParquetDataFrame(file_path)
    lazy_df["a"]  # Load 'a' into cache
    cached_df = lazy_df.cached_df
    assert list(cached_df.columns) == ["a"]
    assert list(cached_df["a"]) == [1, 2]

def test_setitem_updates_cache(tmp_path):
    file_path = make_parquet(tmp_path, {"a": [1, 2], "b": [3, 4]})
    lazy_df = LazyParquetDataFrame(file_path)
    lazy_df["c"] = [5, 6]
    assert "c" in lazy_df.columns
    assert list(lazy_df["c"]) == [5, 6]