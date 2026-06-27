"""
LazyParquetDF
=============

A focused example demonstrating the LazyParquetDF "lazyframe" API for
indexed Parquet loading, lazy column access, filtering, and chunked
iteration / saving.

This complements the filtering and memory-usage examples by showing
how to work interactively with a Parquet-backed DataFrame-like object
without loading all columns into memory at once.
"""
from pathlib import Path
import tempfile

import pandas as pd

from parq_tools.lazy_parquet import LazyParquetDF


# %%
# Create a Parquet file
# ---------------------
#
# We first build a small DataFrame. In practice this could be a much larger
# dataset. Here we keep ``"i"`` as a regular data column rather than an index
# so that it can be referenced directly in lazy operations like ``filter``
# and ``iter_row_chunks``.


def create_parquet(path: Path) -> None:
    df = pd.DataFrame(
        {
            "i": [0, 1, 2, 3],
            "j": [10, 11, 12, 13],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df.to_parquet(path)


parquet_path = Path(tempfile.gettempdir()) / "lazyparquetdf_example.parquet"
create_parquet(parquet_path)

# %%
# Construct a LazyParquetDF
# -------------------------
#
# When no ``index_col`` is given, LazyParquetDF reconstructs the logical
# index using the pandas metadata stored in the Parquet file. For this
# file it will just be a simple RangeIndex, and ``"i"`` remains a regular
# data column.

lazy = LazyParquetDF(parquet_path)
print("Shape:", lazy.shape)
print("Columns:", lazy.columns)
print("Index name:", lazy.index.name)

# %%
# Lazy column access
# -------------------
#
# Columns are loaded on demand. The ``dtypes`` property only reports
# columns that have been materialised so far.

print("Dtypes before loading any column:")
print(lazy.dtypes)

# Access a single column; only this column is loaded into memory.
value_series = lazy["value"]
print("Loaded 'value' column:")
print(value_series)

print("Dtypes after loading 'value':")
print(lazy.dtypes)

# %%
# Filtering and query
# -------------------
#
# ``filter`` uses a PyArrow-style predicate and reads only the columns
# needed for the filter, while ``query`` operates on a materialised
# pandas DataFrame.

filtered = lazy.filter(("i", ">", 1))
print("\nFiltered rows where i > 1 (filter):")
print(filtered)

queried = lazy.query("value > 2.0")
print("\nFiltered rows where value > 2.0 (query):")
print(queried)

# %%
# Chunked iteration and saving
# ----------------------------
#
# We can add derived columns, then iterate over the dataset in row-wise
# chunks and save back to Parquet without holding the full DataFrame in
# memory.

lazy["double"] = lazy["value"] * 2

print("\nIterating in chunks of size 2 (columns i and double):")
for chunk in lazy.iter_row_chunks(chunk_size=2, columns=["i", "double"]):
    print(chunk)

out_path = parquet_path.with_name("lazyparquetdf_example_out.parquet")

# Save using chunked write-back.
lazy.to_parquet(out_path, allow_overwrite=True, chunk_size=2)

print("\nRound-trip check:")
roundtrip = pd.read_parquet(out_path)
print(roundtrip)

