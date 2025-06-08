"""
Memory Usage
============

Parquet files are compressed columnar data files.  This is great for storage and performance, but
it can be useful to understand memory by column usage when working with large datasets.  This enables
the user to optimise their data processing and storage strategies.

"""
import tempfile

import pandas as pd
from pathlib import Path

from parq_tools import ParquetProfileReport
from parq_tools.block_models.utils import create_demo_blockmodel
from parq_tools.utils.memory_utils import parquet_memory_usage, print_parquet_memory_usage

# %%
# Create a Parquet file for profiling
# -----------------------------------

temp_dir = Path(tempfile.gettempdir()) / "memory_usage_example"
temp_dir.mkdir(parents=True, exist_ok=True)

parquet_file_path: Path = temp_dir / "test_blockmodel.parquet"

# Create a reasonably large model example
df: pd.DataFrame = create_demo_blockmodel(shape=(300, 100, 100), block_size=(10, 10, 5),
                                          corner=(0, 0, 0))
# Add a categorical column and a string column
df["depth_as_string"] = df["depth"].astype(str)
df["depth_as_category"] = pd.Categorical(df["depth"].astype(str))
df.to_parquet(parquet_file_path)
print("Shape:", df.shape)

# %%
# Memory usage reports
# --------------------
# Generate a memory usage report for the Parquet file, various ways.

# %%
# Full report with index marking
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
report = parquet_memory_usage(parquet_file_path, index_columns=["x", "y", "z"])
print("\nFull memory usage report (with pandas):")
print_parquet_memory_usage(report)

# %%
# Report without pandas memory usage
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
report_no_pandas = parquet_memory_usage(parquet_file_path, report_pandas=False)
print("\nMemory usage report (Arrow only, no pandas):")
print_parquet_memory_usage(report_no_pandas)

# %%
# Report for a subset of columns
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
subset_cols = ["x", "y", "depth", "depth_as_category"]
report_subset = parquet_memory_usage(parquet_file_path, columns=subset_cols, index_columns=["x", "y"])
print("\nMemory usage report (subset of columns):")
print_parquet_memory_usage(report_subset)

# %%
# Accessing the structured dictionary
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Useful for programmatic use
print("\nAccessing the structured dictionary:")
print({k: v for k, v in report["columns"].items() if k in subset_cols})
