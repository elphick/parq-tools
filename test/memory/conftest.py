import tempfile
import pytest
from pathlib import Path
import pandas as pd
from parq_tools.block_models.utils.demo_block_model import create_demo_blockmodel

@pytest.fixture(scope="session")
def example_parquet_file(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("memory_usage_example")
    parquet_file_path = temp_dir / "test_blockmodel.parquet"
    df: pd.DataFrame = create_demo_blockmodel(shape=(300, 100, 100), block_size=(10, 10, 5), corner=(0, 0, 0))
    df.to_parquet(parquet_file_path)
    return parquet_file_path
