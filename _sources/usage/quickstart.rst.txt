Quick Start Guide
=================

This page will describe the basic steps to use the package.


Filtering a parquet file.

..  code-block:: python

    from parq_tools import filter_parquet_file

    filter_parquet_file(parquet_file_path, output_path=output_parquet_file_path,
                        columns=["x", "y", "z", "a", "c"],
                        filter_expression='x > 3 and y <= 15',
                        show_progress=True)

Concatenating multiple parquet files

..  code-block:: python

    from parq_tools import concat_parquet_files

    concat_parquet_files(files=input_files, output_path=output_parquet_filepath, axis=1,
                         index_columns=["x", "y", "z"],
                         columns=["a", "c"],
                         filter_expression='x > 3 and y <= 15',
                         show_progress=True)


For examples that demonstrate a range of use cases, see the :doc:`/auto_examples/index`.
