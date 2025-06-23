import pandas as pd
import pytest

from parq_tools.calculated_columns import CalculatedColumn, CalculatedParquetReader


def calc_c(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(a + b, name="c")


def test_calculated_column_evaluate():
    col = CalculatedColumn("c", func=calc_c)
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = col.evaluate(df)
    pd.testing.assert_series_equal(result, pd.Series([4, 6], name="c"))


@pytest.mark.skip(reason="not implemented")
def test_calculated_column_by_lambda():
    base = ["a", "b"]
    calc = [
        CalculatedColumn("c", calc_c),
        CalculatedColumn("d", lambda c: pd.Series(c * 2, name="d"))
    ]
    schema = CalculatedSchema(base, calc)
    assert schema.get_column_order() == ["a", "b", "c", "d"]

@pytest.mark.skip(reason="not implemented")
def test_schema_with_mapping_dict():
    mapping = {"r": "red", "g": "green", "b": "blue"}
    calc = [
        CalculatedColumn('color',
                         lambda color_code: pd.Series(color_code, name="color").map(mapping).astype('category'))
    ]
    df = pd.DataFrame({"color_code": ["r", "b", "g"], "b": [3, 4, 5]})
    schema = CalculatedSchema(df.columns, calc)

    assert schema.get_column_order() == ["color_code", "color", "b"]
