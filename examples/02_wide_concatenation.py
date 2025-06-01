"""
Wide Concatenation
==================

This example demonstrates how to wide concatenate multiple parquet files into new parquet files using the
`parq_tools` library and compares the results with `pandas.concat`.
"""

import pandas as pd
from parq_tools import ParquetConcat
from pathlib import Path
import tempfile

# Create a temporary directory for the output files
temp_dir = Path(tempfile.gettempdir()) / "parquet_concat_example"
temp_dir.mkdir(parents=True, exist_ok=True)

# Define the input Parquet files
input_files = [
    temp_dir / "example_data1.parquet",
    temp_dir / "example_data2.parquet",
    temp_dir / "example_data3.parquet"
]


# Create example Parquet files
def create_example_parquet(file_path: Path, data: dict):
    df = pd.DataFrame(data)
    df.to_parquet(file_path, index=False)


# Example data for the Parquet files
data1 = {
    "x": [1, 2, 3],
    "y": [4, 5, 6],
    "z": [7, 8, 9],
    "a": ["A", "B", "C"]
}

data2 = {
    "x": [1, 2, 3],
    "y": [4, 5, 6],
    "z": [7, 8, 9],
    "b": [6.0, 7.0, 8.0],
    "c": ["G", "H", "I"]
}

data3 = {
    "x": [1, 2, 3],
    "y": [4, 5, 6],
    "z": [7, 8, 9],
    "d": ["J", "K", "L"]
}

# Create the Parquet files
create_example_parquet(input_files[0], data1)
create_example_parquet(input_files[1], data2)
create_example_parquet(input_files[2], data3)

# %%
# Perform Wide Concatenation with Pandas
# --------------------------------------
# This approach is fine subject to memory constraints, as it loads all data into memory.

index_cols = ["x", "y", "z"]

dfs = [pd.read_parquet(file).set_index(index_cols) for file in input_files]
wide_result_pandas = pd.concat(dfs, axis=1)
wide_result_pandas

# %%
# Perform Wide Concatenation with Parq Tools
# ------------------------------------------
# This approach is more efficient for large datasets, as it processes data in chunks.

output_wide = temp_dir / "wide_concatenated.parquet"

# Initialize the ParquetConcat class for wide concatenation
wide_concat = ParquetConcat(files=input_files, axis=1, index_columns=index_cols)

# Perform the concatenation
wide_concat.concat_to_file(output_path=output_wide)

# Read the concatenated file
wide_result = pd.read_parquet(output_wide).set_index(index_cols)
wide_result

# %%
# Compare the results
pd.testing.assert_frame_equal(wide_result, wide_result_pandas)

# %%
# Wide concatenation with filters
# -------------------------------
# You can also apply filters during the concatenation process.
# The filter expression is pandas-like and can include index and non-index columns.
# This re-uses the `ParquetConcat` object created earlier to filter and concatenate the data.

filter_query = "x > 2 and b > 6"
output_filtered_wide = temp_dir / "filtered_wide_concatenated.parquet"
# Perform the concatenation with a filter
wide_concat.concat_to_file(
    output_path=output_filtered_wide,
    filter_query=filter_query,
    columns=["a", "b", "d"]
)
# Read the filtered concatenated file
filtered_wide_result = pd.read_parquet(output_filtered_wide).set_index(index_cols)
filtered_wide_result

# %%
# Concatenate by function
# -----------------------
# You can also concatenate with a function, rather than using the class directly.  The same filtering options are
# available, and the function will handle the concatenation in a memory-efficient way.

# concatenate with the function
from parq_tools import concat_parquet_files

concat_parquet_files(files=input_files, output_path=output_wide.with_suffix('.by_function.parquet'),
                     axis=1, index_columns=index_cols)
# Read the filtered concatenated file
filtered_wide_function_result = pd.read_parquet(output_wide.with_suffix('.by_function.parquet')).set_index(
    index_cols)
filtered_wide_function_result
