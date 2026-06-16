from __future__ import annotations

import gc
import logging
import os
from typing import Any

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

import pyarrow as pa

from parquery.tool import DataFilter

logger = logging.getLogger(__name__)

# DuckDB connection tuning, all optional and read from the environment so the
# deploying service (not parquery) owns the values. Unset = DuckDB default.
#   DUCKDB_MEMORY_LIMIT       per-connection working-memory cap, e.g. "2GB".
#   DUCKDB_THREADS            per-connection thread count, e.g. "2". Bounding
#                             this matters when several processes (e.g. gunicorn
#                             workers) each open a connection on the same host.
#   DUCKDB_TEMP_DIR           spill directory. Set together with a memory limit
#                             so a large aggregation spills to disk and finishes
#                             instead of over-committing RAM (kernel OOM kill);
#                             unset, DuckDB spills to CWD/.tmp.
#   DUCKDB_MAX_TEMP_DIR_SIZE  cap on the spill directory, e.g. "10GB".
DUCKDB_MEMORY_LIMIT: str | None = os.environ.get("DUCKDB_MEMORY_LIMIT")
DUCKDB_THREADS: str | None = os.environ.get("DUCKDB_THREADS")
DUCKDB_TEMP_DIR: str | None = os.environ.get("DUCKDB_TEMP_DIR")
DUCKDB_MAX_TEMP_DIR_SIZE: str | None = os.environ.get("DUCKDB_MAX_TEMP_DIR_SIZE")

# A reader and a writer share each parquet shard on EFS; the writer publishes
# updates with an atomic ``os.replace``. Pointing DuckDB at ``/dev/fd/<fd>`` of
# an already-open descriptor pins the inode for the whole read, so a rename that
# lands mid-read cannot splice two file versions into a corrupt-but-valid
# result. Only available where ``/dev/fd`` exists (Linux, macOS).
_CAN_PIN_FD: bool = os.path.isdir("/dev/fd")


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

    # Execute against a pinned snapshot of the file (see _aggregate_pinned).
    # A transient OSError — e.g. an NFS/EFS stale handle after a concurrent
    # atomic rename reclaimed the inode — is retried once with a fresh fd.
    try:
        result_arrow = _aggregate_pinned(file_name, groupby_cols, measure_cols, data_filter, aggregate, debug)
    except OSError:
        gc.collect()
        result_arrow = _aggregate_pinned(file_name, groupby_cols, measure_cols, data_filter, aggregate, debug)

    return result_arrow


def _aggregate_pinned(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[list[str]],
    data_filter: DataFilter,
    aggregate: bool,
    debug: bool,
) -> pa.Table:
    """Aggregate ``file_name`` against a consistent snapshot of its bytes.

    The writer publishes shard updates with an atomic ``os.replace`` on EFS. If
    that rename lands while DuckDB is mid-read, reading by path can combine the
    footer of one file version with the data pages of another and return a
    structurally valid but corrupt result (wrong values, no error). We open the
    file once and point DuckDB at ``/dev/fd/<fd>``, which re-resolves to that
    exact inode for the whole read, so any concurrent rename is invisible. If
    the backing inode is reclaimed (``ESTALE``) DuckDB raises rather than
    returning garbage, and ``aggregate_pq_duckdb`` retries with a fresh fd.

    Where ``/dev/fd`` is unavailable the read falls back to the path directly.
    """
    if not _CAN_PIN_FD:
        sql = _build_sql_query(file_name, groupby_cols, measure_cols, data_filter, aggregate)
        if debug:
            logger.debug(f"DuckDB SQL:\n{sql}\n")
        return call_duckdb(sql)

    fd = os.open(file_name, os.O_RDONLY)
    try:
        sql = _build_sql_query(f"/dev/fd/{fd}", groupby_cols, measure_cols, data_filter, aggregate)
        if debug:
            logger.debug(f"DuckDB SQL:\n{sql}\n")
        return call_duckdb(sql)
    finally:
        os.close(fd)


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
    config: dict[str, Any] = {}
    if DUCKDB_MEMORY_LIMIT:
        config["memory_limit"] = DUCKDB_MEMORY_LIMIT
    if DUCKDB_THREADS:
        config["threads"] = int(DUCKDB_THREADS)
    if DUCKDB_TEMP_DIR:
        # Created here so a misconfigured path is not a hard error on every query.
        os.makedirs(DUCKDB_TEMP_DIR, exist_ok=True)
        config["temp_directory"] = DUCKDB_TEMP_DIR
        if DUCKDB_MAX_TEMP_DIR_SIZE:
            config["max_temp_directory_size"] = DUCKDB_MAX_TEMP_DIR_SIZE
    conn = duckdb.connect(":memory:", config=config)
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
