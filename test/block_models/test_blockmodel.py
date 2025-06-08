import os
from pathlib import Path
import pytest
from parq_tools.block_models.block_model import ParquetBlockModel


def test_from_empty_parquet_raises_error(tmp_path):
    import pandas as pd
    parquet_path = tmp_path / "bar.parquet"
    parquet_path.touch()
    pd.DataFrame(columns=["x", "y", "z"]).to_parquet(parquet_path)

    with pytest.raises(ValueError, match="Parquet file is empty or does not contain valid centroid data."):
        ParquetBlockModel.from_parquet(parquet_path)

def test_create_demo_block_model_creates_file(tmp_path):
    filename = tmp_path / "demo_block_model.parquet"
    model = ParquetBlockModel.create_demo_block_model(filename)
    assert isinstance(model, ParquetBlockModel)
    assert model.path == filename.with_suffix(".pbm.parquet")
    assert model.name == filename.stem
    assert filename.exists()
    # Check if the file is a Parquet file
    assert filename.suffix == ".parquet"
    # Clean up the created file
    os.remove(filename) if filename.exists() else None
    os.remove(model.path) if filename.exists() else None

