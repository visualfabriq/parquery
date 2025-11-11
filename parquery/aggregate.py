from __future__ import annotations

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

# Type aliases for filter operations
FilterOperator = Literal["in", "not in", "nin", "=", "==", "!=", ">", ">=", "<=", "<"]
FilterCondition = tuple[str, FilterOperator, Any]  # (column, operator, value(s))
DataFilter = (
    list[FilterCondition] | list[list[Any]]
)  # Support both typed and legacy format


class FilterValueError(ValueError):
    pass


FILTER_CUTOVER_LENGTH = 10
SAFE_PREAGGREGATE = set(["min", "max", "sum", "one"])


def aggregate_pq(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[str] | list[list[str]],
    data_filter: DataFilter | None = None,
    aggregate: bool = True,
    as_df: bool | None = None,
    standard_missing_id: int = -1,
    handle_missing_file: bool = True,
    debug: bool = False,
    engine: Literal["auto", "duckdb", "pyarrow"] | None = None,
) -> pd.DataFrame | pa.Table:
    """
    Aggregate a Parquet file using DuckDB (preferred) or PyArrow.

    This function automatically selects the best engine for aggregation:
    - DuckDB: 3-8x faster for most scenarios (default when installed)
    - PyArrow: Fallback when DuckDB not available, or explicit engine choice

    See aggregate_pq_pyarrow() and aggregate_pq_duckdb() for engine-specific details.

    Args:
        file_name: Path to the Parquet file to aggregate.
        groupby_cols: List of column names (dimensions) to group by.
        measure_cols: Columns to aggregate with operations. Can be:
            - List of column names: ['m1', 'm2'] - performs sum on each
            - List of [column, operation]: [['m1', 'sum'], ['m2', 'count']]
            - List of [column, operation, output_name]: [['m1', 'sum', 'm1_total']]
            Supported operations: sum, mean, std, count, count_na, count_distinct,
            sorted_count_distinct, min, max, one.
        data_filter: Optional list of filter conditions to apply before aggregation.
            Format: [[column, operator, value(s)], ...]
            Operators: 'in', 'not in', '==', '!=', '>', '>=', '<', '<='
            Example: [['f0', 'in', [1, 2, 3]], ['f1', '>', 100]]
        aggregate: If True, performs groupby aggregation. If False, returns filtered
            rows without aggregation.
        as_df: Return type control:
            - None (default): Auto-detect - returns pandas DataFrame if pandas is
              installed, otherwise PyArrow Table
            - True: Always returns pandas DataFrame (requires pandas)
            - False: Always returns PyArrow Table (no pandas needed)
        standard_missing_id: Default value for missing dimension columns (default: -1).
            Missing measure columns always get 0.0.
        handle_missing_file: If True, returns empty result when file doesn't exist.
            If False, raises OSError for missing files.
        debug: If True, prints progress information during processing.
        engine: Engine selection:
            - "auto" (default): Use DuckDB if installed, otherwise PyArrow
            - "duckdb": Force DuckDB (raises ImportError if not installed)
            - "pyarrow": Force PyArrow engine

    Returns:
        PyArrow Table or pandas DataFrame containing aggregated results.

    Raises:
        ImportError: If specified engine is not installed.
        OSError: If file doesn't exist and handle_missing_file=False.

    Examples:
        >>> # Auto-select engine (DuckDB if available)
        >>> result = aggregate_pq('data.parquet', ['country'], ['sales'])

        >>> # Force specific engine
        >>> result = aggregate_pq('data.parquet', ['country'], ['sales'], engine='pyarrow')

    Notes:
        - DuckDB is 3-8x faster for most workloads
        - Install DuckDB: pip install 'parquery[performance]'
        - Both engines return identical results
    """
    if engine is None:
        engine = "auto"

    # Smart default: use pandas if available, otherwise PyArrow
    if as_df is None:
        as_df = HAS_PANDAS

    if as_df and not HAS_PANDAS:
        raise ImportError(
            "pandas is required for as_df=True. "
            "Install with: pip install pandas or uv pip install 'parquery[dataframes]'"
        )

    # Auto-select engine
    if engine == "auto":
        engine = "duckdb" if HAS_DUCKDB else "pyarrow"

    # Route to appropriate engine (engines always return PyArrow Tables)
    if engine == "duckdb":
        if not HAS_DUCKDB:
            raise ImportError(
                "DuckDB engine requested but not installed. "
                "Install with: pip install duckdb or uv pip install 'parquery[performance]'"
            )
        # Import here to avoid circular dependency
        from parquery.aggregate_duckdb import aggregate_pq_duckdb

        result = aggregate_pq_duckdb(
            file_name=file_name,
            groupby_cols=groupby_cols,
            measure_cols=measure_cols,
            data_filter=data_filter,
            aggregate=aggregate,
            standard_missing_id=standard_missing_id,
            handle_missing_file=handle_missing_file,
            debug=debug,
        )
    elif engine == "pyarrow":
        from parquery.aggregate_pyarrow import aggregate_pq_pyarrow

        result = aggregate_pq_pyarrow(
            file_name=file_name,
            groupby_cols=groupby_cols,
            measure_cols=measure_cols,
            data_filter=data_filter,
            aggregate=aggregate,
            standard_missing_id=standard_missing_id,
            handle_missing_file=handle_missing_file,
            debug=debug,
        )
    else:
        raise ValueError(f"Unknown engine: {engine}. Must be 'auto', 'duckdb', or 'pyarrow'")

    # Convert to pandas if requested (engines always return PyArrow Tables)
    if as_df:
        return result.to_pandas()
    else:
        return result
