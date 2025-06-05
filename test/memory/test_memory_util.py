import pytest
from parq_tools.utils.memory_utils import parquet_memory_usage

def test_memory_usage_basic(example_parquet_file):
    report = parquet_memory_usage(example_parquet_file)
    assert isinstance(report, dict)
    assert 'columns' in report
    assert 'total_compressed_bytes' in report
    assert 'total_decompressed_bytes' in report
    assert 'total_pandas_bytes' in report
    assert 'shape' in report
    assert report['shape'][0] > 0 and report['shape'][1] > 0
    for col, stats in report['columns'].items():
        assert 'compressed_bytes' in stats
        assert 'decompressed_bytes' in stats
        assert 'pandas_bytes' in stats
        assert 'dtype' in stats
        assert 'is_index' in stats
        assert isinstance(stats['compressed_bytes'], int)
        assert isinstance(stats['decompressed_bytes'], int)
        assert stats['compressed_bytes'] >= 0
        assert stats['decompressed_bytes'] >= 0

def test_memory_usage_no_pandas(example_parquet_file):
    report = parquet_memory_usage(example_parquet_file, report_pandas=False)
    assert report['total_pandas_bytes'] is None
    for col, stats in report['columns'].items():
        assert stats['pandas_bytes'] is None

def test_memory_usage_column_subset(example_parquet_file):
    all_cols = list(parquet_memory_usage(example_parquet_file)['columns'].keys())
    subset = all_cols[:2]
    report = parquet_memory_usage(example_parquet_file, columns=subset)
    assert set(report['columns'].keys()) == set(subset)
    assert report['shape'][1] == 2

def test_memory_usage_index_marking(example_parquet_file):
    all_cols = list(parquet_memory_usage(example_parquet_file)['columns'].keys())
    idx_cols = all_cols[:2]
    report = parquet_memory_usage(example_parquet_file, index_columns=idx_cols)
    for col, stats in report['columns'].items():
        if col in idx_cols:
            assert stats['is_index']
        else:
            assert not stats['is_index']

def test_print_parquet_memory_usage_smoke(example_parquet_file, capsys):
    from parq_tools.utils.memory_utils import print_parquet_memory_usage
    report = parquet_memory_usage(example_parquet_file)
    print_parquet_memory_usage(report)
    captured = capsys.readouterr()
    # Check that the output contains key report elements
    assert "Shape:" in captured.out
    assert "Total compressed:" in captured.out
    assert "Per-column breakdown:" in captured.out
    # Check that at least one column name appears in the output
    for col in report['columns']:
        assert col in captured.out
