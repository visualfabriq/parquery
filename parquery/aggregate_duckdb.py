from __future__ import annotations

import os
from typing import Any, Literal

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

import pyarrow as pa

# Type aliases for filter operations (same as aggregate.py)
FilterOperator = Literal["in", "not in", "nin", "=", "==", "!=", ">", ">=", "<=", "<"]
FilterCondition = tuple[str, FilterOperator, Any]  # (column, operator, value(s))
DataFilter = (
    list[FilterCondition] | list[list[Any]]
)  # Support both typed and legacy format


class FilterValueError(ValueError):
    pass


def aggregate_pq_duckdb(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[str] | list[list[str]],
    data_filter: DataFilter | None = None,
    aggregate: bool = True,
    standard_missing_id: int = -1,
    handle_missing_file: bool = True,
    debug: bool = False,
) -> pa.Table:
    """
    Aggregate a Parquet file using DuckDB with streaming execution.

    This function provides memory-efficient aggregations over Parquet files using
    DuckDB's streaming execution engine. DuckDB reads data in small batches and
    processes them incrementally, never loading entire files into memory.

    Args:
        file_name: Path to the Parquet file to aggregate.
        groupby_cols: List of column names (dimensions) to group by.
        measure_cols: Columns to aggregate with operations. Can be:
            - List of column names: ['m1', 'm2'] - performs sum on each
            - List of [column, operation]: [['m1', 'sum'], ['m2', 'count']]
            - List of [column, operation, output_name]: [['m1', 'sum', 'm1_total']]
            Supported operations: sum, mean, stddev, count, count_distinct, min, max.
        data_filter: Optional list of filter conditions to apply before aggregation.
            Format: [[column, operator, value(s)], ...]
            Operators: 'in', 'not in', '==', '!=', '>', '>=', '<', '<='
            Example: [['f0', 'in', [1, 2, 3]], ['f1', '>', 100]]
        aggregate: If True, performs groupby aggregation. If False, returns filtered
            rows without aggregation.
        standard_missing_id: Default value for missing dimension columns (default: -1).
            Missing measure columns always get 0.0.
        handle_missing_file: If True, returns empty result when file doesn't exist.
            If False, raises OSError for missing files.
        debug: If True, prints SQL queries during processing.

    Returns:
        PyArrow Table containing aggregated results.

    Raises:
        ImportError: If DuckDB is not installed.
        OSError: If file doesn't exist and handle_missing_file=False, or if file
            cannot be read.
        FilterValueError: If filter values are invalid.
        NotImplementedError: If an unsupported filter operator is used.

    Examples:
        >>> # Simple sum aggregation
        >>> result = aggregate_pq_duckdb('data.parquet', ['country'], ['sales'])

        >>> # Multiple aggregations with custom names
        >>> result = aggregate_pq_duckdb(
        ...     'data.parquet',
        ...     ['country', 'region'],
        ...     [['sales', 'sum', 'total_sales'], ['sales', 'count', 'num_sales']]
        ... )

        >>> # With filters
        >>> result = aggregate_pq_duckdb(
        ...     'data.parquet',
        ...     ['product'],
        ...     ['revenue'],
        ...     data_filter=[['year', '>=', 2020], ['status', 'in', ['active', 'pending']]]
        ... )

    Notes:
        - DuckDB uses streaming execution with ~2048 row batches
        - Memory footprint is much lower than loading entire files
        - Requires duckdb>=1.0.0: pip install duckdb or uv pip install 'parquery[performance]'
    """
    if not HAS_DUCKDB:
        raise ImportError(
            "duckdb is required for aggregate_pq_duckdb. "
            "Install with: pip install duckdb or uv pip install 'parquery[performance]'"
        )

    # Smart default: use pandas if available, otherwise PyArrow
    data_filter = data_filter or []

    # Normalize measure_cols to [column, operation, output_name] format
    measure_cols = _check_measure_cols(measure_cols)

    # Get all required columns
    all_cols, result_cols = _get_cols(data_filter, groupby_cols, measure_cols)

    # if the file does not exist, give back an empty result
    if not os.path.exists(file_name) and handle_missing_file:
        return _create_empty_result(result_cols)

    # Get Parquet schema to identify missing columns
    try:
        schema = pa.parquet.read_schema(file_name)
        existing_cols = set(schema.names)
    except Exception:
        # If we can't read schema, assume all columns exist (will fail later if they don't)
        existing_cols = set(all_cols)

    # Identify missing columns
    expected_measure_cols = [x[0] for x in measure_cols]
    missing_cols = {}
    for col in all_cols:
        if col not in existing_cols:
            if col in expected_measure_cols:
                # Missing measure columns get 0.0
                missing_cols[col] = 0.0
            else:
                # Missing dimension columns get standard_missing_id
                missing_cols[col] = standard_missing_id

    # Check if filter columns exist - if not, return empty result
    for col, _, _ in data_filter:
        if col not in existing_cols:
            return _create_empty_result(result_cols)

    # Build SQL query
    sql = _build_sql_query(
        file_name,
        groupby_cols,
        measure_cols,
        data_filter,
        aggregate,
        standard_missing_id,
        all_cols,
        existing_cols,
        missing_cols,
    )

    if debug:
        print(f"DuckDB SQL:\n{sql}\n")

    # Execute query with DuckDB
    try:
        conn = duckdb.connect(":memory:")
        # arrow() returns a RecordBatchReader, convert to Table
        reader = conn.execute(sql).arrow()
        result_arrow = reader.read_all()
        conn.close()
    except Exception as e:
        if handle_missing_file and "No files found" in str(e):
            return _create_empty_result(result_cols)
        raise

    # Ensure correct column order
    result_arrow = result_arrow.select(result_cols)

    return result_arrow


def _check_measure_cols(measure_cols: list[str] | list[list[str]]) -> list[list[str]]:
    """Normalize measure columns to list of [column, operation, output_name]."""
    measure_cols = [x if isinstance(x, list) else [x] for x in measure_cols]
    for agg_ops in measure_cols:
        if len(agg_ops) == 1:
            # standard expect a sum aggregation with the same column name
            agg_ops.extend(["sum", agg_ops[0]])
        elif len(agg_ops) == 2:
            # assume the same column name if not specified
            agg_ops.append(agg_ops[0])
    return measure_cols


def _get_cols(
    data_filter: DataFilter, groupby_cols: list[str], measure_cols: list[list[str]]
) -> tuple[list[str], list[str]]:
    """Get all columns and result columns from query parameters."""
    all_cols = sorted(
        list(
            set(
                groupby_cols
                + [x[0] for x in measure_cols]
                + [x[0] for x in data_filter]
            )
        )
    )
    result_cols = sorted(list(set(groupby_cols + [x[2] for x in measure_cols])))
    return all_cols, result_cols


def _build_sql_query(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[list[str]],
    data_filter: DataFilter,
    aggregate: bool,
    standard_missing_id: int,
    all_cols: list[str],
    existing_cols: set[str],
    missing_cols: dict[str, float | int],
) -> str:
    """Build DuckDB SQL query for aggregation with support for missing columns."""
    # Map operation names to DuckDB equivalents
    op_map = {
        "sum": "SUM",
        "mean": "AVG",
        "avg": "AVG",
        "std": "STDDEV",
        "stddev": "STDDEV",
        "count": "COUNT",
        "count_na": "COUNT",
        "count_distinct": "COUNT(DISTINCT {})",
        "sorted_count_distinct": "COUNT(DISTINCT {})",
        "min": "MIN",
        "max": "MAX",
        "one": "MIN",  # "one" means pick any value, MIN works for this
    }

    # Build SELECT clause
    if aggregate:
        # Aggregation with or without groupby
        select_parts = []
        for col in groupby_cols:
            if col in missing_cols:
                # Missing groupby column - use literal value
                select_parts.append(f"{missing_cols[col]} AS \"{col}\"")
            else:
                select_parts.append(f'"{col}"')

        for col, op, output_name in measure_cols:
            if col in missing_cols:
                # Missing measure column - always 0.0
                select_parts.append(f"0.0 AS \"{output_name}\"")
            else:
                op_upper = op.lower()
                if op_upper in ["count_distinct", "sorted_count_distinct"]:
                    agg_expr = f'COUNT(DISTINCT "{col}")'
                else:
                    sql_op = op_map.get(op_upper, op.upper())
                    agg_expr = f'{sql_op}("{col}")'

                # Handle NULL values with COALESCE and use output_name as alias
                select_parts.append(f"COALESCE({agg_expr}, 0.0) AS \"{output_name}\"")
        select_clause = ", ".join(select_parts)
    else:
        # No aggregation, just select columns (with literals for missing ones)
        select_parts = []
        for col in all_cols:
            if col in missing_cols:
                select_parts.append(f"{missing_cols[col]} AS \"{col}\"")
            else:
                select_parts.append(f'"{col}"')
        select_clause = ", ".join(select_parts)

    # Build FROM clause
    from_clause = f"read_parquet('{file_name}')"

    # Build WHERE clause
    where_conditions = []
    for col, operator, values in data_filter:
        condition = _build_filter_condition(col, operator, values)
        where_conditions.append(condition)

    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)

    # Build GROUP BY clause (only for columns that exist in the file)
    group_by_clause = ""
    if aggregate and groupby_cols:
        existing_groupby_cols = [col for col in groupby_cols if col not in missing_cols]
        if existing_groupby_cols:
            group_by_clause = "GROUP BY " + ", ".join(f'"{col}"' for col in existing_groupby_cols)

    # Combine all parts
    query_parts = [f"SELECT {select_clause}", f"FROM {from_clause}"]
    if where_clause:
        query_parts.append(where_clause)
    if group_by_clause:
        query_parts.append(group_by_clause)

    return "\n".join(query_parts)


def _build_filter_condition(col: str, operator: str, values: Any) -> str:
    """Build SQL filter condition from operator and values."""
    if operator == "in":
        if isinstance(values, (list, tuple)):
            values_str = ", ".join(str(v) for v in values)
            return f'"{col}" IN ({values_str})'
        else:
            return f'"{col}" = {values}'
    elif operator in ["not in", "nin"]:
        if isinstance(values, (list, tuple)):
            values_str = ", ".join(str(v) for v in values)
            return f'"{col}" NOT IN ({values_str})'
        else:
            return f'"{col}" != {values}'
    elif operator in ["=", "=="]:
        return f'"{col}" = {values}'
    elif operator == "!=":
        return f'"{col}" != {values}'
    elif operator == ">":
        return f'"{col}" > {values}'
    elif operator == ">=":
        return f'"{col}" >= {values}'
    elif operator == "<=":
        return f'"{col}" <= {values}'
    elif operator == "<":
        return f'"{col}" < {values}'
    else:
        valid_ops = ["in", "not in", "nin", "=", "==", "!=", ">", ">=", "<=", "<"]
        raise NotImplementedError(
            f'Filter operation "{operator}" is not supported for column "{col}". '
            f"Valid operators: {', '.join(valid_ops)}"
        )


def _create_empty_result(result_cols: list[str]) -> pa.Table:
    """Create an empty PyArrow Table with the specified columns."""
    # Create empty PyArrow table with schema
    schema = pa.schema([(col, pa.null()) for col in result_cols])
    empty_table = pa.Table.from_pydict({col: [] for col in result_cols}, schema=schema)
    return empty_table
