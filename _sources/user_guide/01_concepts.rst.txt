Concepts
========

This section provides an overview of the key concepts and components in the library.
It is written with the assumption that most users have been exposed to parquet files
via the pandas methods:

- :py:meth:`pandas.DataFrame.to_parquet`
- :py:meth:`pandas.read_parquet`

Users are likely familiar with the following pandas methods to manipulate DataFrames:

- :py:meth:`pandas.concat`
- :py:meth:`pandas.DataFrame.filter`
- :py:meth:`pandas.DataFrame.query`

The library provides a set of tools built upon the `pyarrow` library to filter and concatenate parquet files is cases where
loading the entire file into memory is not feasible or desired. It is designed to work with
large datasets that may not fit into memory, allowing users to efficiently process and manipulate
parquet files without the need to load them entirely into memory.

Index Columns
-------------

Index columns are a key concept in pandas.  However parquet files do not have an index in the same way that pandas does.

When pandas stores a DataFrame to a parquet file, the index is stored as a column in the parquet file.
When reading a parquet file, pandas will recover the indexes using metadata stored in the parquet file.

But not all parquet files are created by pandas, so they may not have the pandas index metadata.
It is for this reason that the user must specify which columns should be treated as index columns when concatenating
with the parq-tools library.  Further development may allow the library to infer index columns from the
parquet file pandas metadata where it exists, but this is not currently implemented.

