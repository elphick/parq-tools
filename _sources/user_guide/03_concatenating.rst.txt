Concatenating
=============

Concatenation can be along rows (tall, axis=0) or columns (wide, axis=1) of multiple parquet files.
The function :py:func:`parq_tools.parq_filter.concat_parquet_files` can be used for this purpose.

Filtering rows using pandas-like expressions, and columns is supported.

When concatenating, you must specify which columns should be treated as index columns. This is necessary because parquet files do not inherently have an index like pandas DataFrames do. The index columns are used to align the data correctly during concatenation.
The output is a new parquet file that contains the concatenated data, filtered according to the specified conditions.

Wide concatenation
------------------

When concatenating along columns (wide concatenation), the function will align the data based on the
specified index columns. This means that rows with matching index values will be combined into a single row,
with columns from each file added side by side.

Files suitable for wide concatenation should have the same index columns, and the columns to be concatenated
should not overlap (have repeated names).

Tall concatenation
------------------

When concatenating along rows (tall concatenation), the function will stack the data from multiple files on
top of each other.

Files suitable for tall concatenation should have the same columns, and the index columns would not typically
overlap across files.