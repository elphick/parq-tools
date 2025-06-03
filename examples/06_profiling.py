"""
Parquet Profiling
=================

`ydata-profiling` provides a convenient way to profile Parquet files using the `ProfileReport` class.
Their documentation provides options for profiling large datasets.  This example describes an alternative approach.

In cases where a parquet may be very wide, and you want to profile it column by column, you can use the
`ParquetProfileReport` class from the `parq_tools.utils.profile_utils` module.
This allows you to generate a profile report by loading columns in batches, reducing memory consumption.
"""
import tempfile

import pandas as pd
from pathlib import Path

from parq_tools import ParquetProfileReport

# %%
# Create a Parquet file for profiling
# -----------------------------------

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
)
report.profile()

report.show()

# %%
# Run native ydata-profiling
# --------------------------
# As expected the native report runs faster, and only requires 3 steps.

report = ParquetProfileReport(
    parquet_path=parquet_path,
    batch_size=None,  # None batch size will run standard ydata-profiling ProfileReport
    show_progress=True,
)
report.profile().show()
