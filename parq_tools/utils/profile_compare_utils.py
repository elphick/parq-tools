import copy
import math
from typing import Any, Iterable, Optional, Sequence

import pandas as pd


DEFAULT_COMPARISON_METRICS: tuple[str, ...] = (
    "mean",
    "std",
    "min",
    "max",
    "n_missing",
    "p_missing",
    "n_distinct",
)


def _to_python_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def normalize_for_export(value: Any) -> Any:
    """Convert non-JSON-serializable numpy/pandas values into Python scalars."""
    if isinstance(value, dict):
        return {str(k): normalize_for_export(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_for_export(v) for v in value]
    value = _to_python_scalar(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
    return value


def is_numeric(value: Any) -> bool:
    value = _to_python_scalar(value)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def within_tolerance(a: Any, b: Any, abs_tol: float, rel_tol: float) -> bool:
    a = _to_python_scalar(a)
    b = _to_python_scalar(b)
    if is_numeric(a) and is_numeric(b):
        if isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b):
            return True
        if isinstance(a, float) and math.isnan(a):
            return False
        if isinstance(b, float) and math.isnan(b):
            return False
        diff = abs(float(a) - float(b))
        scale = max(abs(float(a)), abs(float(b)))
        return diff <= max(abs_tol, rel_tol * scale)
    return a == b


def build_column_summary(
    values: Sequence[Any],
    abs_tol: float,
    rel_tol: float,
) -> dict[str, Any]:
    if not values:
        return {"within_tolerance": True, "deltas_from_first": []}

    baseline = values[0]
    deltas: list[Optional[dict[str, Any]]] = [None]
    all_within = True
    for value in values[1:]:
        if is_numeric(baseline) and is_numeric(value):
            baseline_float = float(baseline)
            value_float = float(value)
            abs_delta = abs(value_float - baseline_float)
            rel_delta = abs_delta / abs(baseline_float) if baseline_float != 0 else (0.0 if abs_delta == 0 else float("inf"))
            deltas.append({"abs": abs_delta, "rel": rel_delta})
        else:
            deltas.append(None)
        if not within_tolerance(baseline, value, abs_tol=abs_tol, rel_tol=rel_tol):
            all_within = False

    return {
        "within_tolerance": all_within,
        "deltas_from_first": deltas,
    }


def build_profile_comparison_summary(
    descriptions: Sequence[Any],
    labels: Sequence[str],
    abs_tol: float = 0.0,
    rel_tol: float = 0.0,
    metrics: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    metric_names = list(metrics) if metrics is not None else list(DEFAULT_COMPARISON_METRICS)
    all_columns: list[str] = []
    for desc in descriptions:
        for column in desc.variables.keys():
            if column not in all_columns:
                all_columns.append(column)

    columns: dict[str, Any] = {}
    overview = {
        "equal": 0,
        "different": 0,
        "missing_in_some": 0,
        "type_mismatch": 0,
        "total_columns": len(all_columns),
    }

    for column in all_columns:
        variable_entries = [desc.variables.get(column) for desc in descriptions]
        present = [entry is not None for entry in variable_entries]
        types = [entry.get("type") if entry else None for entry in variable_entries]
        reasons: list[str] = []
        status = "equal"

        if not all(present):
            status = "missing_in_some"
            reasons.append("missing_in_some_datasets")
        elif len({str(dtype) for dtype in types}) > 1:
            status = "type_mismatch"
            reasons.append("type_mismatch")

        metric_payload: dict[str, Any] = {}
        if status == "equal":
            metric_differences: list[str] = []
            for metric_name in metric_names:
                values = [entry.get(metric_name) if entry else None for entry in variable_entries]
                comparison = build_column_summary(values, abs_tol=abs_tol, rel_tol=rel_tol)
                metric_payload[metric_name] = {
                    "values": values,
                    **comparison,
                }
                if not comparison["within_tolerance"]:
                    metric_differences.append(metric_name)

            if metric_differences:
                status = "different"
                reasons.append(f"metric_differences:{','.join(metric_differences)}")

        if status in overview:
            overview[status] += 1

        columns[column] = {
            "present": present,
            "types": types,
            "status": status,
            "metrics": metric_payload,
            "reasons": reasons,
        }

    return normalize_for_export(
        {
            "labels": list(labels),
            "overview": overview,
            "columns": columns,
            "tolerance": {"abs_tol": abs_tol, "rel_tol": rel_tol},
            "metrics_compared": metric_names,
        }
    )


def get_changed_columns_from_summary(summary: dict[str, Any]) -> list[str]:
    changed = []
    for column, payload in summary["columns"].items():
        if payload.get("status") != "equal":
            changed.append(column)
    return changed


def _slice_correlation_value(value: Any, keep_columns: set[str]) -> Any:
    if isinstance(value, pd.DataFrame):
        shared_columns = [col for col in value.columns if col in keep_columns]
        shared_index = [idx for idx in value.index if idx in keep_columns]
        if not shared_columns or not shared_index:
            return pd.DataFrame()
        return value.loc[shared_index, shared_columns]
    return value


def _recompute_table_stats(description: Any) -> None:
    variables = description.variables
    table = description.table
    if not variables:
        table["n_var"] = 0
        table["memory_size"] = 0
        table["record_size"] = 0
        table["n_cells_missing"] = 0
        table["n_vars_with_missing"] = 0
        table["n_vars_all_missing"] = 0
        table["p_cells_missing"] = 0
        table["types"] = {}
        return

    n = table.get("n", 0)
    memory_size = 0
    n_cells_missing = 0
    n_vars_with_missing = 0
    n_vars_all_missing = 0
    type_counts: dict[str, int] = {}
    for variable in variables.values():
        var_memory = variable.get("memory_size", 0)
        var_missing = variable.get("n_missing", 0)
        var_n = variable.get("n", n)
        var_type = str(variable.get("type", "Unsupported"))
        memory_size += var_memory if isinstance(var_memory, (int, float)) else 0
        n_cells_missing += var_missing if isinstance(var_missing, (int, float)) else 0
        if isinstance(var_missing, (int, float)) and var_missing > 0:
            n_vars_with_missing += 1
        if isinstance(var_missing, (int, float)) and isinstance(var_n, (int, float)) and var_missing == var_n:
            n_vars_all_missing += 1
        type_counts[var_type] = type_counts.get(var_type, 0) + 1

    table["n_var"] = len(variables)
    table["memory_size"] = memory_size
    table["record_size"] = memory_size / n if n else 0
    table["n_cells_missing"] = n_cells_missing
    table["n_vars_with_missing"] = n_vars_with_missing
    table["n_vars_all_missing"] = n_vars_all_missing
    denom = len(variables) * n
    table["p_cells_missing"] = n_cells_missing / denom if denom else 0
    table["types"] = type_counts


def prune_description_to_columns(description: Any, keep_columns: Sequence[str]) -> Any:
    """Prune a BaseDescription-like object to the requested columns."""
    keep = set(keep_columns)
    pruned = copy.deepcopy(description)
    pruned.variables = {k: v for k, v in pruned.variables.items() if k in keep}
    pruned.scatter = {
        k1: {k2: v2 for k2, v2 in row.items() if k2 in keep}
        for k1, row in pruned.scatter.items()
        if k1 in keep
    }
    pruned.correlations = {
        corr_name: _slice_correlation_value(corr_value, keep)
        for corr_name, corr_value in pruned.correlations.items()
    }
    # Missing diagrams can couple to the original full variable set; reset for safe compare rendering.
    pruned.missing = {}
    pruned.alerts = [
        alert for alert in pruned.alerts
        if getattr(alert, "column_name", None) in keep or getattr(alert, "column_name", None) is None
    ]
    _recompute_table_stats(pruned)
    return pruned
