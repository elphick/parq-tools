Filtering
=========

Filtering a parquet file includes selecting specific columns and applying a filter expression
to reduce the returned rows.

With pandas you would typically use the :py:meth:`pandas.DataFrame.filter` method to select columns,
and the :py:meth:`pandas.DataFrame.query` method to filter rows.  However, both of these operations
are managed by `filtering` in the parq-tools library.

Filtering a parquet file is done with the :py:func:`parq_tools.parq_filter.filter_parquet_file` function.

The output is a new parquet file that contains only the selected columns and rows that match the filter expression.