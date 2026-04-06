from pathlib import Path

import io

import pytest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parq_tools.lazy_parquet import LazyParquetDF


def make_parquet(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "i": [0, 1, 2, 3],
        "j": [10, 11, 12, 13],
        "value": [1.0, 2.0, 3.0, 4.0],
    })
    path = tmp_path / "test.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
    return path


def make_parquet_with_index(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "i": [0, 1, 2, 3],
        "j": [10, 11, 12, 13],
        "value": [1.0, 2.0, 3.0, 4.0],
    }).set_index("i")
    path = tmp_path / "test_indexed.parquet"
    df.to_parquet(path)
    return path


def test_basic_column_loading(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    assert lp.shape == (4, 3)
    assert lp.columns == ["i", "j", "value"]

    s = lp["value"]
    assert list(s) == [1.0, 2.0, 3.0, 4.0]


def test_index_from_pandas_metadata(tmp_path: Path) -> None:
    path = make_parquet_with_index(tmp_path)
    lp = LazyParquetDF(path)

    # Index should be taken from pandas metadata ('i'), and columns are the others.
    assert isinstance(lp.index, pd.Index)
    assert lp.index.name == "i"
    assert list(lp.columns) == ["j", "value"]
    assert lp.shape == (4, 2)

    pdf = lp.to_pandas()
    expected = pd.read_parquet(path)
    pd.testing.assert_frame_equal(pdf, expected)


def test_index_col_validation_error(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    # 'x' is not a schema column
    with pytest.raises(KeyError):
        LazyParquetDF(path, index_col="x")


def test_multiindex_and_assignment(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path, index_col=["i", "j"])

    # Add a derived column and ensure it aligns with the MultiIndex.
    lp["double"] = [2.0, 4.0, 6.0, 8.0]

    pdf = lp.to_pandas()
    # We expect a two-level index whose values correspond to columns i and j.
    assert pdf.index.nlevels == 2
    assert list(pdf.index.get_level_values(0)) == [0, 1, 2, 3]
    assert list(pdf.index.get_level_values(1)) == [10, 11, 12, 13]
    assert list(pdf["double"]) == [2.0, 4.0, 6.0, 8.0]


def test_dtypes_only_loaded_columns(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    # Nothing loaded yet
    assert lp.dtypes.empty

    # Load one column
    _ = lp["value"]

    assert list(lp.dtypes.index) == ["value"]
    assert lp.dtypes["value"] == pd.Series([1.0], dtype="float64").dtype


def test_head_matches_pandas(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    pdf_head = lp.head(2)
    df = pd.read_parquet(path)
    pd.testing.assert_frame_equal(pdf_head, df.head(2))


def test_describe_matches_pandas(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    lp_desc = lp.describe()
    df_desc = pd.read_parquet(path).describe()

    assert list(lp_desc.index) == list(df_desc.index)
    assert list(lp_desc.columns) == list(df_desc.columns)
    pd.testing.assert_frame_equal(lp_desc, df_desc)


def test_info_lazy_vs_loaded(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    buf = io.StringIO()
    lp.info(buf=buf)
    out = buf.getvalue()

    # Initially everything is lazy
    assert "status=lazy" in out
    assert "status=loaded" not in out

    # Load a column and check again
    _ = lp["value"]
    buf = io.StringIO()
    lp.info(buf=buf)
    out = buf.getvalue()

    assert "status=loaded" in out
    assert "value:" in out


def test_filter_basic(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    # Filter rows where i > 1
    res = lp.filter(("i", ">", 1))
    df = pd.read_parquet(path)
    expected = df[df["i"] > 1][["i"]]
    pd.testing.assert_frame_equal(res.reset_index(drop=True), expected.reset_index(drop=True))


def test_filter_errors(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    # No predicates
    with pytest.raises(ValueError):
        lp.filter()

    # Predicate on unknown column
    with pytest.raises(KeyError):
        lp.filter(("x", ">", 1))


def test_query_delegates_to_pandas(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    res = lp.query("value > 2.0")
    df = pd.read_parquet(path)
    expected = df.query("value > 2.0")
    pd.testing.assert_frame_equal(res.reset_index(drop=True), expected.reset_index(drop=True))


def test_iter_row_chunks_with_computed_column(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)
    lp["double"] = lp["value"] * 2

    chunks = list(lp.iter_row_chunks(chunk_size=2, columns=["i", "double"]))

    assert len(chunks) == 2
    assert list(chunks[0]["double"]) == [2.0, 4.0]
    assert list(chunks[1]["double"]) == [6.0, 8.0]


def test_iter_row_chunks_invalid_chunk_size(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    with pytest.raises(ValueError):
        list(lp.iter_row_chunks(chunk_size=0))


def test_iter_row_chunks_computed_only(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)
    lp["double"] = lp["value"] * 2

    chunks = list(lp.iter_row_chunks(chunk_size=2, columns=["double"]))
    assert len(chunks) == 2
    assert list(chunks[0]["double"]) == [2.0, 4.0]
    assert list(chunks[1]["double"]) == [6.0, 8.0]


def test_iter_row_chunks_unknown_column(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)

    with pytest.raises(KeyError):
        list(lp.iter_row_chunks(chunk_size=2, columns=["i", "unknown"]))


def test_save_roundtrip(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)
    lp["triple"] = lp["value"] * 3

    out = tmp_path / "out.parquet"
    lp.to_parquet(out, allow_overwrite=False, chunk_size=2)

    pdf = pd.read_parquet(out)
    assert list(pdf["triple"]) == [3.0, 6.0, 9.0, 12.0]


def test_save_overwrites_original(tmp_path: Path) -> None:
    path = make_parquet(tmp_path)
    lp = LazyParquetDF(path)
    lp["triple"] = lp["value"] * 3

    lp.save(allow_overwrite=True, chunk_size=2)

    pdf = pd.read_parquet(path)
    assert list(pdf["triple"]) == [3.0, 6.0, 9.0, 12.0]
