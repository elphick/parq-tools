Profiling
=========

`parq-tools` provides :py:class:`parq_tools.parq_profile.ParquetProfileReport` for generating
`fg-data-profiling` reports from parquet files.

Use this when:

- you want a native full-dataframe profile, or
- your parquet is wide and you want to profile columns in chunks to reduce peak memory use.


Native vs chunked profiling
---------------------------

Use ``batch_size=None`` to run native profiling in one pass.
Use ``batch_size=<n>`` to profile ``n`` columns at a time and merge the results.

..  code-block:: python

    from parq_tools import ParquetProfileReport

    # Chunked profiling: process 2 columns per batch
    chunked = ParquetProfileReport(
        parquet_path="example.parquet",
        batch_size=2,
        show_progress=True,
    )
    chunked.profile().save_html("profile_chunked.html")

    # Native profiling: process all selected columns at once
    native = ParquetProfileReport(
        parquet_path="example.parquet",
        batch_size=None,
        show_progress=True,
    )
    native.profile().save_html("profile_native.html")


Column metadata and descriptions
--------------------------------

You can pass ``column_descriptions`` as:

- a legacy string description,
- a dict payload, or
- :py:class:`parq_tools.ColumnMetadata`.

Structured metadata is formatted as:
``<Title>: <Description> (Units: <Units>; Source: <Source>)``.

..  code-block:: python

    from parq_tools import ParquetProfileReport, ColumnMetadata

    profiler = ParquetProfileReport(
        parquet_path="example.parquet",
        batch_size=1,
        column_descriptions={
            "mass": ColumnMetadata(
                title="Mass",
                description="Measured sample mass",
                units="kg",
                source="lab",
            ),
            "density": {
                "title": "Density",
                "description": "Bulk density",
                "unit_of_measure": "kg/m3",
                "source": "calculated",
            },
            "category": "Categorical label",
        },
    )
    profiler.profile().save_html("profile_with_metadata.html")


Dataset metadata
----------------

Use ``dataset_metadata`` to provide report-level context (for example ``description``, ``author``, or ``url``).
You can supply either a ``dict`` or ``ProfileMetadata``-compatible payload.


Viewing the report
------------------

After profiling:

- call ``save_html(path)`` to persist the report,
- call ``to_html()`` to get HTML as a string,
- call ``show()`` to open the report in a browser tab.

