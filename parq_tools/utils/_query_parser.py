from lark import Lark
from pyarrow import compute as pc


# Define the parser as a reusable utility
def get_filter_parser():
    """
    Returns a Lark parser for validating filter expressions.
    """
    grammar = """
    ?start: expr
    ?expr: expr "and" expr  -> and_expr
          | expr "or" expr   -> or_expr
          | "(" expr ")"     -> group
          | COLUMN OP VALUE  -> comparison_expr
    COLUMN: /[a-zA-Z_][a-zA-Z0-9_]*/
    OP: ">" | "<" | ">=" | "<=" | "==" | "!="
    VALUE: /\d+(\.\d+)?/  // Numeric values

    %import common.WS
    %ignore WS
    """
    return Lark(grammar, start="expr")


def build_filter_expression(filter_query: str, schema) -> pc.Expression:
    """
    Converts a filter query into a PyArrow compute expression.

    Args:
        filter_query (str): Pandas-style filter expression.
        schema (pa.Schema): Schema of the table to validate against.

    Returns:
        pc.Expression: PyArrow-compatible filter expression.
    """
    parser = get_filter_parser()
    parsed_query = parser.parse(filter_query)

    def _convert_to_expression(node):
        if node.data == "comparison_expr":
            column, op, value = node.children
            column = pc.field(column.value)
            value = float(value.value) if "." in value.value else int(value.value)
            if op.value == ">":
                return column > value
            elif op.value == "<":
                return column < value
            elif op.value == ">=":
                return column >= value
            elif op.value == "<=":
                return column <= value
            elif op.value == "==":
                return column == value
            elif op.value == "!=":
                return column != value
            else:
                # Explicitly raise an error for unexpected operators
                raise ValueError(f"Unexpected operator in filter query: {op.value}")
        elif node.data == "and_expr":
            # Use Python's `&` operator to combine expressions
            return _convert_to_expression(node.children[0]) & _convert_to_expression(node.children[1])
        elif node.data == "or_expr":
            # Use Python's `|` operator to combine expressions
            return _convert_to_expression(node.children[0]) | _convert_to_expression(node.children[1])
        elif node.data == "group":
            return _convert_to_expression(node.children[0])
        else:
            # Explicitly raise an error for unexpected node types
            raise ValueError(f"Unexpected node type in filter query: {node.data}")

    # Ensure the parsed query is converted to a PyArrow expression
    return _convert_to_expression(parsed_query)


def get_referenced_columns(filter_query: str) -> set:
    """
    Extracts the column names referenced in a filter query.

    Args:
        filter_query (str): Pandas-style filter expression.

    Returns:
        set: A set of column names referenced in the filter query.
    """
    parser = get_filter_parser()
    parsed_query = parser.parse(filter_query)

    def _extract_columns(node):
        if node.data == "comparison_expr":
            column = node.children[0].value
            return {column}
        elif node.data in {"and_expr", "or_expr"}:
            left = _extract_columns(node.children[0])
            right = _extract_columns(node.children[1])
            return left | right
        elif node.data == "group":
            return _extract_columns(node.children[0])
        else:
            return set()

    return _extract_columns(parsed_query)





