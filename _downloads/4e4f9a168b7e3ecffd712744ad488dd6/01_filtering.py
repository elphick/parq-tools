"""
Filtering
=========

A simple example to demonstrate how to filter a parquet file using a pandas-like expression.

This example uses the `parq_tools` library to filter a Parquet file based on a specified condition.
Pyarrow filtering is not structured like the filtering in pandas, but parq-tools uses custom parser
allowing pandas-like expressions to be used.

"""
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


# %%
#
# Create a Parquet file
# ---------------------
#
# Create a temporary parquet file for demonstration


def create_parquet_file(file_path: Path):
    # Define the dataset
    data = {
        "x": range(1, 11),  # Index column
        "y": range(11, 21),  # Index column
        "z": range(21, 31),  # Index column
        "a": [f"val{i}" for i in range(1, 11)],  # Supplementary column
        "b": [i * 2 for i in range(1, 11)],  # Supplementary column
        "c": [i % 3 for i in range(1, 11)],  # Supplementary column
    }

    # Create a DataFrame
    df = pa.Table.from_pydict(data)

    # Write the DataFrame to a Parquet file
    pq.write_table(df, file_path)


parquet_file_path = Path(tempfile.gettempdir()) / "example_data.parquet"
create_parquet_file(parquet_file_path)

# %%
# View the file as a DataFrame
df = pd.read_parquet(parquet_file_path)
df

# %%
# Filter with Pandas
# ------------------
#
# We can use pandas directly to load the Parquet file and filter it using a pandas-like expression.
# First we filter early with read_parquet for efficiency.  Additionally, we have manually set the index in this example.

index_cols = ["x", "y", "z"]
df_from_pandas_1: pd.DataFrame = pd.read_parquet(parquet_file_path,
                                                 columns=["x", "y", "z", "a", "c"],
                                                 filters=[("x", ">", 3), ("y", "<=", 15)]).set_index(index_cols)
df_from_pandas_1

# %%
# An alternative but less efficient way is to load all records and then apply a filter
df_from_pandas_2 = pd.read_parquet(parquet_file_path,
                                   columns=["x", "y", "z", "a", "c"]).query("x > 3 and y <= 15").set_index(index_cols)

df_from_pandas_2

# %%
# Compare the two DataFrames to ensure they are equal
pd.testing.assert_frame_equal(df_from_pandas_1, df_from_pandas_2)

# %%
# Filter with Parq Tools
# ----------------------
#
# The `parq_tools` library provides a way to filter Parquet files that do not fit into memory,
# using a pandas-like expression.  The output is a new Parquet file containing only the filtered records
# and selected columns.  This can be useful in pipelines with large datasets.

from parq_tools import filter_parquet_file

filter_parquet_file(parquet_file_path,
                    output_path=parquet_file_path.with_suffix('.filtered.parquet'),
                    columns=["x", "y", "z", "a", "c"], filter_expression='x > 3 and y <= 15',
                    show_progress=True)

# %%
# Read the filtered Parquet file
df_filtered = pd.read_parquet(parquet_file_path.with_suffix('.filtered.parquet')).set_index(index_cols)
df_filtered

# %%
# Compare the filtered DataFrame with the one from pandas
pd.testing.assert_frame_equal(df_filtered, df_from_pandas_1)
