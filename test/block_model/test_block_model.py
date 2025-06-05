import pytest
import numpy as np
import pandas as pd
from parq_tools.utils.block_model_utils import create_test_blockmodel

def test_create_blockmodel_regular():
    shape = (2, 2, 2)
    block_size = (1.0, 1.0, 1.0)
    corner = (0.0, 0.0, 0.0)
    df = create_test_blockmodel(shape, block_size, corner, is_tensor=False)
    # Check shape and index
    assert isinstance(df, pd.DataFrame)
    assert df.index.names == ['x', 'y', 'z']
    assert len(df) == np.prod(shape)
    # Check columns
    assert 'c_style_xyz' in df.columns
    assert 'f_style_zyx' in df.columns
    assert 'depth' in df.columns
    # Check that dx/dy/dz are not present
    assert 'dx' not in df.columns
    assert 'dy' not in df.columns
    assert 'dz' not in df.columns

def test_create_blockmodel_tensor():
    shape = (2, 2, 2)
    block_size = (1.0, 2.0, 3.0)
    corner = (10.0, 20.0, 30.0)
    df = create_test_blockmodel(shape, block_size, corner, is_tensor=True)
    # Check shape and index
    assert isinstance(df, pd.DataFrame)
    assert df.index.names == ['x', 'y', 'z', 'dx', 'dy', 'dz']
    assert len(df) == np.prod(shape)
    # Check columns
    assert 'c_style_xyz' in df.columns
    assert 'f_style_zyx' in df.columns
    assert 'depth' in df.columns
    # Check that dx/dy/dz are present as index levels and correct
    assert (df.index.get_level_values('dx') == block_size[0]).all()
    assert (df.index.get_level_values('dy') == block_size[1]).all()
    assert (df.index.get_level_values('dz') == block_size[2]).all()
