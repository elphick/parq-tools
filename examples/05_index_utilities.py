"""
Index Utilities
===============

A simple example to demonstrate utilities related to `index` columns.

.. note::

    Index columns as we know them in Pandas do not exist in a native Parquet file.  However, if the Parquet file
    has been created using Pandas then metadata is preserved to restore the indexes when a round-trip back to Pandas
    is completed.

The utilities demonstrated here are tools mimic index operations that one may use in Pandas.

"""
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from pathlib import Path

from parq_tools import sort_parquet_file, reindex_parquet, validate_index_alignment
from parq_tools.utils.demo_block_model import create_demo_blockmodel

# %%
#
# Create a dataset
# ----------------
#
# Create a temporary parquet file for demonstration.  This example represents a 3D block model.

parquet_file_path = Path(tempfile.gettempdir()) / "example_data.parquet"

df: pd.DataFrame = create_demo_blockmodel(shape=(3, 3, 3), block_size=(1, 1, 1),
                                          corner=(-0.5, -0.5, -0.5))
df.to_parquet(parquet_file_path)
df

# %%
# Randomise the order of the DataFrame and persist to Parquet

df = df.sample(frac=1)
df.to_parquet(parquet_file_path)
df_randomised = pd.read_parquet(parquet_file_path)
df_randomised

# %%
# Sort by the index
# -----------------
#
# We can sort the DataFrame by the index columns to mimic the behavior of Pandas.

index_cols = ["x", "y", "z"]
sorted_file_path: Path = parquet_file_path.parent / "sorted_example_data.parquet"
sort_parquet_file(parquet_file_path, output_path=sorted_file_path,
                  columns=index_cols, chunk_size=100_000)

# Read the sorted Parquet file
sorted_df = pd.read_parquet(sorted_file_path)
sorted_df

# %%
# Reindexing
# ----------
# We can reindex the DataFrame to change the order of the index columns.  This is useful if we want to change the
# order of the index columns to align with another dataset prior to concatenation.  Reindexing will reorder existing
# records, and will add empty records if the new index has more records than the original index.
#
# To demonstrate this, we will create another Parquet file with a subset of the original records that are unordered.

unsorted_subset_file_path: Path = parquet_file_path.parent / "unsorted_subset.parquet"
df_randomised.sample(frac=0.5).to_parquet(unsorted_subset_file_path)
df_unsorted_subset: pd.DataFrame = pd.read_parquet(unsorted_subset_file_path)
df_unsorted_subset

# %%
# Reindex the unsorted subset to match the original index order

reindexed_file_path: Path = parquet_file_path.parent / "reindexed_subset.parquet"
reindex_parquet(unsorted_subset_file_path, output_path=reindexed_file_path,
                new_index=pa.Table.from_pandas(sorted_df.reset_index()[index_cols]))
df_reindexed: pd.DataFrame = pd.read_parquet(reindexed_file_path).set_index(index_cols)
df_reindexed

# %%
# Validate index alignment
# ------------------------
# We can demonstrate the `validate_index_alignment` function.
datasets: list[ds.Dataset] = [ds.dataset(pf) for pf in [sorted_file_path, reindexed_file_path]]
validate_index_alignment(datasets=datasets, index_columns=index_cols)