"""
Parquet Profiling
=================

`fg-data-profiling` (formerly ydata-profiling) provides a convenient way to profile Parquet files using the `ProfileReport` class.
Their documentation provides options for profiling large datasets.  This example describes an alternative approach.

In cases where a parquet may be very wide, and you want to profile it column by column, you can use the
`ParquetProfileReport` class from the `parq_tools.utils.profile_utils` module.
This allows you to generate a profile report by loading columns in batches, reducing memory consumption.
"""
import logging
import tempfile

import pandas as pd
from pathlib import Path

from parq_tools import ParquetProfileReport, ColumnMetadata, compare_parquet_profiles

# %%
# Create a Parquet file for profiling
# -----------------------------------

logging.basicConfig(level=logging.INFO)

temp_dir = Path(tempfile.gettempdir()) / "profile_parquet_example"
temp_dir.mkdir(parents=True, exist_ok=True)

# Create a sample DataFrame and save as Parquet
df = pd.DataFrame({
    "col1": range(100),
    "col2": ["a"] * 100,
    "col3": [True, False] * 50,
})
parquet_path = temp_dir / "example.parquet"
df.to_parquet(parquet_path)

shared_column_descriptions = {
    "col1": ColumnMetadata(
        title="Mass",
        description="Primary quantity used to validate description rendering",
        units="kg",
        source="Sensor A",
    ),
    "col2": "Categorical test label used for description rendering checks",
}

shared_dataset_metadata = {
    "description": "Demonstration dataset for validating profile metadata rendering",
    "author": "parq-tools example",
}

# %%
# Profile by column
# -----------------
# The `ParquetProfileReport` class allows you to profile a Parquet file by loading columns in batches.
#
# While we are profiling 3 columns, the 4th progress step is used to capture the merging process.

report = ParquetProfileReport(
    parquet_path=parquet_path,
    columns=None,  # None means all columns
    batch_size=1,  # Process 1 column at a time
    show_progress=True,
    dataset_metadata=shared_dataset_metadata,
    column_descriptions=shared_column_descriptions,
)
report.profile()

report.show()
single_output = temp_dir / "single_profile_report.html"
report.to_file(single_output)
single_html = single_output.read_text(encoding="utf-8")
assert "Primary quantity used to validate description rendering" in single_html

# %%
# Run native fg-data-profiling
# --------------------------
# As expected the native report runs faster, and only requires 3 steps.

report = ParquetProfileReport(
    parquet_path=parquet_path,
    batch_size=None,  # None batch size will run standard fg-data-profiling ProfileReport
    show_progress=True,
    dataset_metadata=shared_dataset_metadata,
    column_descriptions=shared_column_descriptions,
)
report.profile().show()

# %%
# Compare two or three parquet files with memory-managed profiling
# ---------------------------------------------------------------

parquet_path_b = temp_dir / "example_b.parquet"
parquet_path_c = temp_dir / "example_c.parquet"

pd.DataFrame({
    "col1": range(100),
    "col2": ["a"] * 99 + ["b"],
    "col3": [True, False] * 50,
}).to_parquet(parquet_path_b)

pd.DataFrame({
    "col1": range(1, 101),
    "col2": ["a"] * 100,
    "col3": [True] * 100,
}).to_parquet(parquet_path_c)

comparison = compare_parquet_profiles(
    parquet_paths=[parquet_path, parquet_path_b, parquet_path_c],  # 2 or 3 files supported
    batch_size=1,  # memory-managed profile generation
    show_progress=True,
    titles=["Dataset A", "Dataset B", "Dataset C"],
    dataset_metadata=[
        {"description": "Dataset A description"},
        {"description": "Dataset B description"},
        {"description": "Dataset C description"},
    ],
    column_descriptions=shared_column_descriptions,
)
comparison_output = temp_dir / "comparison_profile_report.html"
comparison.to_file(comparison_output)
comparison_html = comparison_output.read_text(encoding="utf-8")
assert "Primary quantity used to validate description rendering" in comparison_html

import webbrowser
webbrowser.open_new_tab(f"file://{comparison_output}")
