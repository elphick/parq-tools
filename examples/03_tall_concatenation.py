"""
Tall Concatenation
==================

This example demonstrates how to concatenate multiple Parquet files along rows (tall concatenation)
using the `parq_tools` library. It also shows how to achieve the same result with `pandas.concat`
and verifies that the outputs are equivalent.

Tall concatenation combines all rows from each file, aligning columns by name. Columns missing in some files
are filled with nulls.
"""

import pandas as pd
from parq_tools import ParquetConcat
from pathlib import Path
import tempfile


# %%
# Create Example Parquet Files
# ----------------------------
#
# We create three example Parquet files with overlapping and unique columns.

def create_example_parquet(file_path: Path, data: dict):
    df = pd.DataFrame(data)
    df.to_parquet(file_path, index=False)


temp_dir = Path(tempfile.gettempdir()) / "parquet_tall_concat_example"
temp_dir.mkdir(parents=True, exist_ok=True)

data1 = {
    "x": [1, 2, 3],
    "y": [4, 5, 6],
    "z": [7, 8, 9],
    "a": ["A", "B", "C"],
    "b": [1.0, 2.0, 3.0]
}
data2 = {
    "x": [4, 5, 6],
    "y": [7, 8, 9],
    "z": [10, 11, 12],
    "a": ["D", "E", "F"],
    "b": [3.0, 4.0, 5.0]
}
data3 = {
    "x": [7, 8, 9],
    "y": [10, 11, 12],
    "z": [13, 14, 15],
    "a": ["G", "H", "I"],
    "b": [6.0, 7.0, 8.0],
    "c": ["J", "K", "L"]
}

input_files = [
    temp_dir / "example_data1.parquet",
    temp_dir / "example_data2.parquet",
    temp_dir / "example_data3.parquet"
]

create_example_parquet(input_files[0], data1)
create_example_parquet(input_files[1], data2)
create_example_parquet(input_files[2], data3)

# %%
# Tall Concatenation with Pandas
# ------------------------------
#
# This approach loads all data into memory and concatenates along rows.

index_cols = ["x", "y", "z"]

dfs = [pd.read_parquet(f) for f in input_files]
tall_result_pandas = pd.concat(dfs, axis=0, ignore_index=True).set_index(index_cols)
tall_result_pandas

# %%
# Tall Concatenation with Parq Tools
# ----------------------------------
#
# This approach is more efficient for large datasets, as it processes data in chunks.

output_tall = temp_dir / "tall_concatenated.parquet"
tall_concat = ParquetConcat(files=input_files, axis=0)
tall_concat.concat_to_file(output_path=output_tall)

tall_result = pd.read_parquet(output_tall).set_index(index_cols)
tall_result

# %%
# Compare the Results
# -------------------
#
# Ensure the outputs from both methods are equivalent.

pd.testing.assert_frame_equal(tall_result, tall_result_pandas)

# %%
# Tall Concatenation with Filters
# -------------------------------
#
# You can also apply a filter expression during concatenation.

filter_query = "x > 4 and b >= 5.0"
output_filtered_tall = temp_dir / "filtered_tall_concatenated.parquet"
tall_concat.concat_to_file(
    output_path=output_filtered_tall,
    filter_query=filter_query,
    columns=["x", "y", "z", "a", "b"]
)
filtered_tall_result = pd.read_parquet(output_filtered_tall).set_index(index_cols)
filtered_tall_result

# %%
# Concatenate by function
# -----------------------
# You can also concatenate with a function, rather than using the class directly.  The same filtering options are
# available, and the function will handle the concatenation in a memory-efficient way.

# concatenate with the function
from parq_tools import concat_parquet_files

concat_parquet_files(files=input_files, output_path=output_tall.with_suffix('.by_function.parquet'),
                     axis=0, index_columns=index_cols)
# Read the filtered concatenated file
filtered_wide_function_result = pd.read_parquet(output_tall.with_suffix('.by_function.parquet')).set_index(
    index_cols)
filtered_wide_function_result
