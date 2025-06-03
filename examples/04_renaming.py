"""
Renaming
========

A simple example to demonstrate how to rename columns in a parquet file.

"""
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from parq_tools import rename_parquet_columns


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
# Rename columns
# --------------
#
# We can rename a selection of columns.  Here we assume we don't want to rename the `index` columns.
# Assuming we have no knowledge of the column names, we'll read them from the file schema.

index_cols = ["x", "y", "z"]
col_names = pq.ParquetFile(parquet_file_path).schema.names
col_names

# %%
# Create a mapping and rename the columns

new_col_names: dict[str, str] = {col: f"new_{col}" for col in col_names if col not in index_cols + ['c']}
output_file_path = parquet_file_path.parent / "renamed_data.parquet"
rename_parquet_columns(input_path=parquet_file_path, output_path=output_file_path,
                       rename_map=new_col_names, show_progress=True)
# %%
# Read the renamed file and display it

df_renamed = pd.read_parquet(output_file_path)
df_renamed

# %%
# Persisting only renamed columns
# -------------------------------

new_col_names = {"x": "x", "a": "new_a"}
output_file_path_renamed_only = parquet_file_path.parent / "renamed_data_only.parquet"
rename_parquet_columns(input_path=parquet_file_path, output_path=output_file_path_renamed_only,
                       rename_map=new_col_names, return_all_columns=False)

df_renamed_only = pd.read_parquet(output_file_path_renamed_only)
df_renamed_only