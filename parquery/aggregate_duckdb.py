from __future__ import annotations

import gc
import logging
from typing import Any

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

import pyarrow as pa

from parquery.tool import DataFilter

logger = logging.getLogger(__name__)


def aggregate_pq_duckdb(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[list[str]],
    data_filter: DataFilter | None = None,
    aggregate: bool = True,
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

    data_filter = data_filter or []

    # Build SQL query
    sql = _build_sql_query(
        file_name,
        groupby_cols,
        measure_cols,
        data_filter,
        aggregate,
    )

    if debug:
        logger.debug(f"DuckDB SQL:\n{sql}\n")

    # Execute query with DuckDB
    try:
        result_arrow = call_duckdb(sql)
    except OSError:
        gc.collect()
        result_arrow = call_duckdb(sql)

    return result_arrow


def call_duckdb(sql) -> Any:
    """
    Execute SQL query using DuckDB and return PyArrow Table.

    Creates an in-memory DuckDB connection, executes the SQL query,
    and returns the results as a PyArrow Table using Arrow IPC format.

    Args:
        sql: SQL query string to execute.

    Returns:
        PyArrow Table containing query results.

    Notes:
        - Uses in-memory database (:memory:) for temporary processing.
        - Connection is automatically closed after query execution.
        - Results are streamed via RecordBatchReader and converted to Table.
    """
    conn = duckdb.connect(":memory:")
    # arrow() returns a RecordBatchReader, convert to Table
    reader = conn.execute(sql).arrow()
    result_arrow = reader.read_all()
    conn.close()
    return result_arrow


def _build_sql_query(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[list[str]],
    data_filter: DataFilter,
    aggregate: bool,
) -> str:
    """
    Build DuckDB SQL query for Parquet aggregation.

    Constructs a SQL query string that reads from a Parquet file and performs
    filtering and optional aggregation operations. The query uses DuckDB's
    read_parquet() function for direct Parquet file access.

    Args:
        file_name: Path to Parquet file to query.
        groupby_cols: List of column names to group by (dimensions).
        measure_cols: List of [column, operation, output_name] specifications.
        data_filter: List of filter conditions [[col, operator, value], ...].
        aggregate: If True, performs GROUP BY aggregation; if False, returns filtered rows.

    Returns:
        Complete SQL query string ready for DuckDB execution.

    Notes:
        - Supports operations: sum, mean/avg, std/stddev, count, count_distinct, min, max, one.
        - Filter operators: in, not in, ==, !=, >, >=, <, <=.
        - Column names are quoted to handle special characters.
        - Uses DuckDB's native aggregation functions for best performance.
    """
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
            select_parts.append(f'"{col}"')

        for col, op, output_name in measure_cols:
            op_upper = op.lower()
            if op_upper in ["count_distinct", "sorted_count_distinct"]:
                agg_expr = f'COUNT(DISTINCT "{col}")'
            else:
                sql_op = op_map.get(op_upper, op.upper())
                agg_expr = f'{sql_op}("{col}")'

            select_parts.append(f'{agg_expr} AS "{output_name}"')

        select_clause = ", ".join(select_parts)
    else:
        # No aggregation, just select all requested columns
        all_cols = sorted(list(set(groupby_cols + [x[0] for x in measure_cols])))
        select_parts = [f'"{col}"' for col in all_cols]
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

    # Build GROUP BY clause
    group_by_clause = ""
    if aggregate and groupby_cols:
        group_by_clause = "GROUP BY " + ", ".join(f'"{col}"' for col in groupby_cols)

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
