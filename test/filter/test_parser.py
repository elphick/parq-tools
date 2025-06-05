import pytest
from parq_tools.utils import _query_parser
from lark.exceptions import UnexpectedInput

# Dummy schema for build_filter_expression (not used in logic, but required by signature)
class DummySchema:
    pass

def test_simple_comparisons():
    parser = _query_parser.get_filter_parser()
    for expr in [
        'a > 1',
        'b < 2',
        'c >= 3',
        'd <= 4',
        'e == 5',
        'f != 6',
        'g > 1.5',
        'h <= 0.0',
    ]:
        tree = parser.parse(expr)
        assert tree is not None
        # Should not raise
        _query_parser.build_filter_expression(expr, DummySchema())

def test_logical_operators():
    parser = _query_parser.get_filter_parser()
    expr = 'a > 1 and b < 2 or c == 3'
    tree = parser.parse(expr)
    assert tree is not None
    _query_parser.build_filter_expression(expr, DummySchema())

def test_grouping():
    parser = _query_parser.get_filter_parser()
    expr = '(a > 1 and (b < 2 or c == 3))'
    tree = parser.parse(expr)
    assert tree is not None
    _query_parser.build_filter_expression(expr, DummySchema())

def test_column_extraction():
    expr = 'a > 1 and (b < 2 or c == 3)'
    cols = _query_parser.get_referenced_columns(expr)
    assert cols == {'a', 'b', 'c'}
    expr2 = 'foo >= 10 or bar != 5'
    cols2 = _query_parser.get_referenced_columns(expr2)
    assert cols2 == {'foo', 'bar'}

def test_invalid_syntax():
    parser = _query_parser.get_filter_parser()
    bad_exprs = [
        'a >> 1',  # invalid operator
        'b = 2',   # invalid operator
        'c >',     # missing value
        'and a > 1', # starts with operator
        'a > 1 or',  # ends with operator
        '()',        # empty group
    ]
    for expr in bad_exprs:
        with pytest.raises((UnexpectedInput, ValueError)):
            parser.parse(expr)

def test_edge_cases():
    parser = _query_parser.get_filter_parser()
    # Whitespace
    expr = '   a   >   1   '
    tree = parser.parse(expr)
    assert tree is not None
    # Decimal without leading zero
    expr2 = 'a > .5'
    with pytest.raises(UnexpectedInput):
        parser.parse(expr2)
    # Large numbers
    expr3 = 'a < 1234567890'
    tree = parser.parse(expr3)
    assert tree is not None

def test_in_operator_numeric():
    parser = _query_parser.get_filter_parser()
    expr = 'a in [1, 2, 3]'
    tree = parser.parse(expr)
    assert tree is not None
    _query_parser.build_filter_expression(expr, DummySchema())

def test_in_operator_string():
    parser = _query_parser.get_filter_parser()
    expr = 'b in ["foo", "bar", "baz"]'
    tree = parser.parse(expr)
    assert tree is not None
    _query_parser.build_filter_expression(expr, DummySchema())
    expr2 = 'c in ["x", "y", "z"]'  # Use double quotes only
    tree2 = parser.parse(expr2)
    assert tree2 is not None
    _query_parser.build_filter_expression(expr2, DummySchema())

def test_in_operator_mixed():
    parser = _query_parser.get_filter_parser()
    expr = 'd in [1, "foo", 2, "bar"]'
    tree = parser.parse(expr)
    assert tree is not None
    with pytest.raises(ValueError, match="All values in an 'in' list must be the same type"):
        _query_parser.build_filter_expression(expr, DummySchema())

def test_in_column_extraction():
    expr = 'a in [1, 2, 3] and b in ["foo", "bar"]'
    cols = _query_parser.get_referenced_columns(expr)
    assert cols == {'a', 'b'}

def test_invalid_in_syntax():
    parser = _query_parser.get_filter_parser()
    bad_exprs = [
        # 'a in []',  # empty list is valid, do not expect an error
        'b in',     # missing list
        'c in [',   # incomplete list
        'd in [1 2 3]', # missing commas
    ]
    for expr in bad_exprs:
        with pytest.raises(Exception):
            parser.parse(expr)

def test_in_operator_empty_list():
    parser = _query_parser.get_filter_parser()
    expr = 'a in []'
    tree = parser.parse(expr)
    assert tree is not None
    _query_parser.build_filter_expression(expr, DummySchema())

