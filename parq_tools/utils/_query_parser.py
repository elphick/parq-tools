from lark import Lark, Token
from pyarrow import compute as pc
import pyarrow as pa


# Define the parser as a reusable utility
def get_filter_parser():
    """
    Returns a Lark parser for validating filter expressions, including 'in' for lists.
    """
    grammar = """
    ?start: expr
    ?expr: expr "and" expr  -> and_expr
          | expr "or" expr   -> or_expr
          | "(" expr ")"     -> group
          | COLUMN OP VALUE  -> comparison_expr
          | COLUMN "in" list -> in_expr
    COLUMN: /[a-zA-Z_][a-zA-Z0-9_]*/
    OP: ">" | "<" | ">=" | "<=" | "==" | "!="
    VALUE: NUMBER | ESCAPED_STRING
    list: "[" [list_items] "]"
    list_items: list_item ("," list_item)*
    list_item: NUMBER | ESCAPED_STRING
    NUMBER: /\d+(\.\d+)?/

    %import common.ESCAPED_STRING
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

    def _parse_value(val_token):
        # If it's a Lark Tree (non-terminal), descend to the actual token
        if hasattr(val_token, 'data') and val_token.data in ('VALUE', 'list_item'):
            # VALUE or list_item is a rule, its child is the actual token
            return _parse_value(val_token.children[0])
        # If it's a Token of type VALUE, parse as number or string
        if hasattr(val_token, 'type') and val_token.type == 'VALUE':
            # Try to parse as number first, else as string
            try:
                return float(val_token.value) if '.' in val_token.value else int(val_token.value)
            except ValueError:
                return val_token.value.strip('"')
        if hasattr(val_token, 'type'):
            if val_token.type == "NUMBER":
                return float(val_token.value) if "." in val_token.value else int(val_token.value)
            elif val_token.type == "ESCAPED_STRING":
                return val_token.value[1:-1]  # Remove quotes
            else:
                raise ValueError(f"Unsupported value type: {val_token.type}")
        raise ValueError(f"Unsupported value: {val_token}")

    def _convert_to_expression(node):
        if node.data == "comparison_expr":
            column, op, value = node.children
            column = pc.field(column.value)
            value = _parse_value(value)
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
        elif node.data == "in_expr":
            column = pc.field(node.children[0].value)
            list_node = node.children[1]
            values = []
            if len(list_node.children) > 0 and list_node.children[0] is not None:
                items = list_node.children[0].children
                for item in items:
                    values.append(_parse_value(item))
            # Check for mixed types (excluding None)
            types = set(type(v) for v in values if v is not None)
            if len(types) > 1:
                raise ValueError(f"All values in an 'in' list must be the same type, got: {[type(v).__name__ for v in values]}")
            return pc.is_in(column, value_set=pa.array(values))
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
        if node.data in {"comparison_expr", "in_expr"}:
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

