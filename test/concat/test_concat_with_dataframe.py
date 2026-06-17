from pathlib import Path

import pandas as pd
import pytest

from parq_tools.parq_concat import concat_parquet_file_with_dataframe
from parq_tools.utils.file_utils import check_valid_parquet


def create_parquet_file(tmp_path: Path, filename: str, data: dict) -> Path:
    """Create a parquet file from a simple dictionary payload."""
    file_path = tmp_path / filename
    pd.DataFrame(data).to_parquet(file_path, index=False)
    return file_path


def test_concat_parquet_file_with_dataframe_writes_extra_columns(tmp_path: Path) -> None:
    """It should append DataFrame columns to a parquet file using key-based alignment."""
    source_file = create_parquet_file(
        tmp_path,
        "source.parquet",
        {
            "id": [1, 2, 3],
            "a": [10, 20, 30],
        },
    )

    extra_df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "b": [100, 200, 300],
            "c": ["x", "y", "z"],
        }
    )

    output_file = tmp_path / "output.parquet"

    concat_parquet_file_with_dataframe(
        parquet_path=source_file,
        df=extra_df,
        output_path=output_file,
        index_columns=["id"],
        batch_size=2,
    )

    result = pd.read_parquet(output_file)

    expected = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "a": [10, 20, 30],
            "b": [100, 200, 300],
            "c": ["x", "y", "z"],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_concat_parquet_file_with_dataframe_left_join_preserves_source_rows(tmp_path: Path) -> None:
    """Rows in the parquet source should be preserved even when the DataFrame is missing keys."""
    source_file = create_parquet_file(
        tmp_path,
        "source.parquet",
        {
            "id": [1, 2, 3],
            "a": [10, 20, 30],
        },
    )

    extra_df = pd.DataFrame(
        {
            "id": [1, 3],
            "b": [100, 300],
        }
    )

    output_file = tmp_path / "output.parquet"

    concat_parquet_file_with_dataframe(
        parquet_path=source_file,
        df=extra_df,
        output_path=output_file,
        index_columns=["id"],
        batch_size=2,
    )

    result = pd.read_parquet(output_file)

    expected = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "a": [10, 20, 30],
            "b": [100.0, None, 300.0],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_concat_parquet_file_with_dataframe_allows_in_place_overwrite(tmp_path: Path) -> None:
    """It should safely rewrite the original parquet file when overwrite is allowed."""
    source_file = create_parquet_file(
        tmp_path,
        "source.parquet",
        {
            "id": [1, 2],
            "a": [10, 20],
        },
    )

    extra_df = pd.DataFrame(
        {
            "id": [1, 2],
            "b": [30, 40],
        }
    )

    concat_parquet_file_with_dataframe(
        parquet_path=source_file,
        df=extra_df,
        output_path=source_file,
        index_columns=["id"],
        batch_size=1,
        allow_overwrite=True,
    )

    result = pd.read_parquet(source_file)

    expected = pd.DataFrame(
        {
            "id": [1, 2],
            "a": [10, 20],
            "b": [30, 40],
        }
    )

    pd.testing.assert_frame_equal(result, expected)


def test_concat_parquet_file_with_dataframe_rejects_existing_output_without_overwrite(
    tmp_path: Path,
) -> None:
    """It should raise when the destination exists and overwrite is not allowed."""
    source_file = create_parquet_file(
        tmp_path,
        "source.parquet",
        {
            "id": [1, 2],
            "a": [10, 20],
        },
    )

    extra_df = pd.DataFrame(
        {
            "id": [1, 2],
            "b": [30, 40],
        }
    )

    output_file = create_parquet_file(
        tmp_path,
        "existing_output.parquet",
        {
            "id": [99],
            "a": [999],
        },
    )

    with pytest.raises(FileExistsError, match="Target file already exists"):
        concat_parquet_file_with_dataframe(
            parquet_path=source_file,
            df=extra_df,
            output_path=output_file,
            index_columns=["id"],
            allow_overwrite=False,
        )


def test_concat_parquet_file_with_dataframe_accepts_valid_parquet_with_nonstandard_extension(
    tmp_path: Path,
) -> None:
    """Validation should be based on file contents, not filename extension."""
    source_file = create_parquet_file(
        tmp_path,
        "source.pbm",
        {
            "id": [1, 2],
            "a": [10, 20],
        },
    )

    extra_df = pd.DataFrame(
        {
            "id": [1, 2],
            "b": [30, 40],
        }
    )

    output_file = tmp_path / "output.pbm"

    concat_parquet_file_with_dataframe(
        parquet_path=source_file,
        df=extra_df,
        output_path=output_file,
        index_columns=["id"],
        allow_overwrite=True,
    )

    result = pd.read_parquet(output_file)
    assert list(result.columns) == ["id", "a", "b"]
    assert check_valid_parquet(source_file) is True
    assert check_valid_parquet(output_file) is True


def test_check_valid_parquet_returns_false_for_non_parquet_file(tmp_path: Path) -> None:
    """Non-parquet files should fail validation even if they have a parquet-like name."""
    invalid_file = tmp_path / "not_really_parquet.parquet"
    invalid_file.write_text("this is not a parquet file", encoding="utf-8")

    assert check_valid_parquet(invalid_file) is False


def test_concat_parquet_file_with_dataframe_requires_index_columns_in_dataframe(
    tmp_path: Path,
) -> None:
    """The in-memory DataFrame must contain the alignment columns."""
    source_file = create_parquet_file(
        tmp_path,
        "source.parquet",
        {
            "id": [1, 2],
            "a": [10, 20],
        },
    )

    extra_df = pd.DataFrame(
        {
            "b": [30, 40],
        }
    )

    output_file = tmp_path / "output.parquet"

    with pytest.raises(KeyError, match="Index columns missing from input DataFrame"):
        concat_parquet_file_with_dataframe(
            parquet_path=source_file,
            df=extra_df,
            output_path=output_file,
            index_columns=["id"],
        )


def test_concat_parquet_file_with_dataframe_requires_index_columns_in_source_parquet(
    tmp_path: Path,
) -> None:
    """The source parquet file must contain the alignment columns."""
    source_file = create_parquet_file(
        tmp_path,
        "source.parquet",
        {
            "x": [1, 2],
            "a": [10, 20],
        },
    )

    extra_df = pd.DataFrame(
        {
            "id": [1, 2],
            "b": [30, 40],
        }
    )

    output_file = tmp_path / "output.parquet"

    with pytest.raises(KeyError, match="Index columns missing from source parquet file"):
        concat_parquet_file_with_dataframe(
            parquet_path=source_file,
            df=extra_df,
            output_path=output_file,
            index_columns=["id"],
        )


def test_concat_parquet_file_with_dataframe_rejects_duplicate_non_index_columns(
    tmp_path: Path,
) -> None:
    """The helper should reject DataFrame columns that collide with source columns."""
    source_file = create_parquet_file(
        tmp_path,
        "source.parquet",
        {
            "id": [1, 2],
            "a": [10, 20],
        },
    )

    extra_df = pd.DataFrame(
        {
            "id": [1, 2],
            "a": [30, 40],
        }
    )

    output_file = tmp_path / "output.parquet"

    with pytest.raises(ValueError, match="already present in the source parquet file"):
        concat_parquet_file_with_dataframe(
            parquet_path=source_file,
            df=extra_df,
            output_path=output_file,
            index_columns=["id"],
        )


def test_concat_parquet_file_with_dataframe_rejects_duplicate_dataframe_keys(
    tmp_path: Path,
) -> None:
    """The helper should reject duplicate key rows in the DataFrame."""
    source_file = create_parquet_file(
        tmp_path,
        "source.parquet",
        {
            "id": [1, 2],
            "a": [10, 20],
        },
    )

    extra_df = pd.DataFrame(
        {
            "id": [1, 1],
            "b": [30, 40],
        }
    )

    output_file = tmp_path / "output.parquet"

    with pytest.raises(ValueError, match="duplicate keys"):
        concat_parquet_file_with_dataframe(
            parquet_path=source_file,
            df=extra_df,
            output_path=output_file,
            index_columns=["id"],
        )


def test_concat_parquet_file_with_dataframe_rejects_invalid_parquet_content(
    tmp_path: Path,
) -> None:
    """The helper should reject invalid parquet inputs by content, regardless of extension."""
    source_file = tmp_path / "source.pbm"
    source_file.write_text("not parquet content", encoding="utf-8")

    extra_df = pd.DataFrame(
        {
            "id": [1, 2],
            "b": [30, 40],
        }
    )

    output_file = tmp_path / "output.parquet"

    with pytest.raises(ValueError, match="not a valid Parquet file"):
        concat_parquet_file_with_dataframe(
            parquet_path=source_file,
            df=extra_df,
            output_path=output_file,
            index_columns=["id"],
        )