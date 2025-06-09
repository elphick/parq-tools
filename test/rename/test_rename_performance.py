import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from parq_tools.parq_schema_tools import rename_and_update_metadata
from parq_tools.block_models.utils.demo_block_model import create_demo_blockmodel


@pytest.mark.performance
@pytest.mark.skip("Performance test, skip unless needed")
def test_rename_performance(tmp_path):
    input_file = tmp_path / "test_block_model.parquet"
    output_file = tmp_path / "renamed_block_model.parquet"
    # Time DataFrame creation
    t0 = time.perf_counter()
    df = create_demo_blockmodel(
        shape=(1000, 1000, 100),
        block_size=(1.0, 1.0, 1.0),
        corner=(0.0, 0.0, 0.0)
    )
    t1 = time.perf_counter()
    print(f"DataFrame creation took {t1 - t0:.4f} seconds.")
    # Time pyarrow Table creation
    t2 = time.perf_counter()
    table = pa.Table.from_pandas(df.reset_index())
    t3 = time.perf_counter()
    print(f"pyarrow Table creation took {t3 - t2:.4f} seconds.")
    # Time Parquet save
    t4 = time.perf_counter()
    pq.write_table(table, input_file)
    t5 = time.perf_counter()
    print(f"Parquet save took {t5 - t4:.4f} seconds.")
    # Define a simple rename map
    rename_map = {"c_style_xyz": "block_id"}
    # Time the rename operation
    t6 = time.perf_counter()
    rename_and_update_metadata(Path(input_file), Path(output_file), rename_map=rename_map)
    t7 = time.perf_counter()
    print(f"Rename operation took {t7 - t6:.4f} seconds.")
    # Optionally, assert the output file exists
    assert os.path.exists(output_file)
