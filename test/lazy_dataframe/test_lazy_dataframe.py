import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
import os

from parq_tools.lazy_parquet import LazyParquetDataFrame


def make_parquet(tmp_path, df):
    file_path = os.path.join(tmp_path, "test.parquet")
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    return file_path

def test_lazy_column_loading(tmp_path):
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6], 'c': [7,8,9]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)
    assert 'a' not in lazy_df._loaded_columns
    _ = lazy_df['a']
    assert 'a' in lazy_df._loaded_columns
    assert (lazy_df['a'] == df['a']).all()

def test_head(tmp_path):
    df = pd.DataFrame({'a': range(10), 'b': range(10,20)})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)
    head = lazy_df.head(3)
    pd.testing.assert_frame_equal(head, df.head(3))

def test_setitem_and_assign(tmp_path):
    df = pd.DataFrame({'x': [1,2,3]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)
    lazy_df['y'] = [4,5,6]
    assert 'y' in lazy_df._extra_columns
    assert (lazy_df['y'] == pd.Series([4,5,6])).all()
    new_df = lazy_df.assign(z=[7,8,9])
    assert 'z' in new_df._extra_columns
    assert (new_df['z'] == pd.Series([7,8,9])).all()

def test_to_pandas(tmp_path):
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)
    pandas_df = lazy_df.to_pandas()
    pd.testing.assert_frame_equal(pandas_df, df)

def test_lazy_loading(tmp_path):
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6], 'c': [7,8,9]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)

    # Access a column to trigger loading
    _ = lazy_df['a']
    assert 'a' in lazy_df._loaded_columns

    # Access another column
    _ = lazy_df['b']
    assert 'b' in lazy_df._loaded_columns

    # Check that the data matches
    pd.testing.assert_series_equal(lazy_df['a'], df['a'])
    pd.testing.assert_series_equal(lazy_df['b'], df['b'])

def test_lazy_dataframe_shape(tmp_path):
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)
    assert lazy_df.shape == (3, 2)  # 3 rows, 2 columns

def test_lazy_dataframe_columns(tmp_path):
    df = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)
    assert list(lazy_df.columns) == ['a', 'b']  # Check column names
    assert 'a' not in lazy_df._loaded_columns  # Ensure 'a' is not loaded initially
    _ = lazy_df['a']  # Load 'a'
    assert 'a' in lazy_df._loaded_columns  # Now 'a' should be loaded

def test_lazy_dataframe_dtypes(tmp_path):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)

    # Check dtypes
    assert lazy_df['a'].dtype == 'int64'
    assert lazy_df['b'].dtype == 'float64'

    # Ensure dtypes are loaded correctly
    _ = lazy_df['a']
    _ = lazy_df['b']

    assert lazy_df.dtypes['a'] == 'int64'
    assert lazy_df.dtypes['b'] == 'float64'

def test_lazy_dataframe_nullable_dtypes(tmp_path):
    df = pd.DataFrame({'a': pd.Series([1, 2, None], dtype='Int64'),
                       'b': pd.Series([4.0, None, 6.0], dtype='Float64')})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)

    # Check nullable dtypes
    assert lazy_df['a'].dtype == 'Int64'
    assert lazy_df['b'].dtype == 'Float64'

    # Ensure dtypes are loaded correctly
    _ = lazy_df['a']
    _ = lazy_df['b']

    assert lazy_df.dtypes['a'] == 'Int64'
    assert lazy_df.dtypes['b'] == 'Float64'

def test_lazy_dataframe_index(tmp_path):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    df.set_index('a', inplace=True)
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)

    # Check index
    assert lazy_df.index.name == 'a'
    assert (lazy_df.index == pd.Index([1, 2, 3], name='a')).all()

    # Access a column to trigger loading
    _ = lazy_df['b']
    assert 'b' in lazy_df._loaded_columns


def test_lazy_dataframe_columns_empty(tmp_path):
    # Test with an empty DataFrame
    df = pd.DataFrame(columns=['a', 'b'])
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)

    # Check columns
    assert list(lazy_df.columns) == ['a', 'b']

    # Ensure no data is loaded
    assert 'a' not in lazy_df._loaded_columns
    assert 'b' not in lazy_df._loaded_columns

    # Check shape
    assert lazy_df.shape == (0, 2)  # 0 rows, 2 columns

    # Access a column to trigger loading
    _ = lazy_df['a']
    assert 'a' in lazy_df._loaded_columns
    assert (lazy_df['a'] == pd.Series([], dtype='float64')).all()

def test_lazy_dataframe_dtypes_empty(tmp_path):
    # Test with an empty DataFrame
    df = pd.DataFrame(columns=['a', 'b'])
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)

    # Check dtypes
    pd.testing.assert_series_equal(lazy_df.dtypes, pd.Series({'a': 'float64', 'b': 'float64'}))

    # Ensure no data is loaded
    assert 'a' not in lazy_df._loaded_columns
    assert 'b' not in lazy_df._loaded_columns

    # Access a column to trigger loading
    _ = lazy_df['a']
    assert 'a' in lazy_df._loaded_columns
    assert (lazy_df['a'] == pd.Series([], dtype='float64')).all()

def test_save_and_load_lazy_dataframe(tmp_path):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)

    # Save the lazy DataFrame to a new file
    new_file_path = os.path.join(tmp_path, "lazy_saved.parquet")
    lazy_df.to_parquet(new_file_path)

    # Load the saved DataFrame
    loaded_lazy_df = LazyParquetDataFrame(new_file_path)

    # Check if the data matches
    pd.testing.assert_frame_equal(loaded_lazy_df.to_pandas(), df)

def test_save_method(tmp_path):
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    file_path = make_parquet(tmp_path, df)
    lazy_df = LazyParquetDataFrame(file_path)

    # add a column to ensure save works with extra columns
    lazy_df['c'] = [7, 8, 9]

    # Save the lazy DataFrame to a new file
    new_file_path = os.path.join(tmp_path, "lazy_saved.parquet")
    lazy_df.save(new_file_path)

    # Check if the saved file exists
    assert os.path.exists(new_file_path)

    # Load the saved DataFrame
    loaded_lazy_df = LazyParquetDataFrame(new_file_path)

    # check the columns
    assert list(loaded_lazy_df.columns) == ['a', 'b', 'c']

    # Check if the data matches
    expected_data = {'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]}
    pd.testing.assert_frame_equal(loaded_lazy_df.to_pandas(), pd.DataFrame(expected_data))