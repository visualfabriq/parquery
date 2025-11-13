from __future__ import annotations

import gc

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

# Import shared types and utilities from main aggregate module
from parquery.tool import SAFE_PREAGGREGATE, DataFilter, create_empty_result


def _unify_aggregation_operators(aggregation_list: list[list[str]]) -> dict[str, str]:
    """Convert aggregation operator names to PyArrow-compatible names."""
    rename_operators = {"std": "stddev"}
    return {x[0]: rename_operators.get(x[1], x[1]) for x in aggregation_list}


def aggregate_pq_pyarrow(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[list[str]],
    data_filter: DataFilter | None = None,
    aggregate: bool = True,
    debug: bool = False,
) -> pa.Table:
    """
    Aggregate a Parquet file using PyArrow with OLAP-style groupby operations.

    This function provides fast aggregations over Parquet files using an OLAP approach
    with Dimensions (groupby columns) and Measures (aggregated columns). It supports
    push-down filtering using Parquet metadata for optimal performance.

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
        standard_missing_id: Default value for missing dimension columns (default: -1).
            Missing measure columns always get 0.0.
        handle_missing_file: If True, returns empty result when file doesn't exist.
            If False, raises OSError for missing files.
        debug: If True, prints progress information during processing.

    Returns:
        PyArrow Table containing aggregated results.

    Raises:
        OSError: If file doesn't exist and handle_missing_file=False, or if file
            cannot be read.
        FilterValueError: If filter values are not numeric for integer columns.
        NotImplementedError: If an unsupported filter operator is used.

    Examples:
        >>> # Simple sum aggregation
        >>> result = aggregate_pq_pyarrow('data.parquet', ['country'], ['sales'])

        >>> # Multiple aggregations with custom names
        >>> result = aggregate_pq_pyarrow(
        ...     'data.parquet',
        ...     ['country', 'region'],
        ...     [['sales', 'sum', 'total_sales'], ['sales', 'count', 'num_sales']]
        ... )

        >>> # With filters
        >>> result = aggregate_pq_pyarrow(
        ...     'data.parquet',
        ...     ['product'],
        ...     ['revenue'],
        ...     data_filter=[['year', '>=', 2020], ['status', 'in', ['active', 'pending']]]
        ... )

        >>> # Returns PyArrow Table for further processing
        >>> table = aggregate_pq_pyarrow('data.parquet', ['id'], ['value'])

    Notes:
        - For "safe" operations (min, max, sum, one), pre-aggregation is performed
          at the row group level for better memory efficiency.
        - Push-down filtering uses Parquet metadata statistics to skip row groups
          that don't contain relevant data.
        - All dimension columns should contain numeric IDs for optimal performance.
    """
    data_filter = data_filter or []

    # Memory optimization strategies:
    #
    # 1. THREADING: Disable for many measures
    #    - Benchmarks with threads enabled for 200 measures:
    #      Groupby start: 4628.02 MB → end: 11119.95 MB (high spike)
    #    - Benchmarks with threads disabled for 200 measures:
    #      Groupby start: 2911.80 MB → end: 6277.05 MB (lower usage)
    #    - Many measures = expensive to aggregate in parallel → disable threading
    #
    # 2. PRE-AGGREGATION: Disable for many dimensions
    #    - Pre-aggregation normally saves memory by reducing data per row group
    #    - BUT with high cardinality groupby (many dims), intermediate results are large
    #    - Better to concat raw data and aggregate once than aggregate multiple times

    # Disable threading when we have many measures (expensive parallel aggregation)
    disable_threads = len(measure_cols) >= 20

    # Disable pre-aggregation when we have many dimensions (high cardinality groupby)
    disable_preaggregate = len(groupby_cols) >= 5

    # Get all columns needed (only columns that exist)
    all_cols = sorted(
        list(
            set(
                groupby_cols
                + [x[0] for x in measure_cols]
                + [x[0] for x in data_filter]
            )
        )
    )

    # Input columns (for processing, before renaming)
    input_cols = sorted(list(set(groupby_cols + [x[0] for x in measure_cols])))

    # create pandas-compliant aggregation
    agg = _unify_aggregation_operators(measure_cols)
    agg_ops = set(agg.values())
    # Pre-aggregate only when safe operations AND not too many dimensions
    preaggregate = (
        aggregate and agg_ops.issubset(SAFE_PREAGGREGATE) and not disable_preaggregate
    )

    # Create dataset (replaces ParquetFile)
    try:
        dataset = ds.dataset(file_name, format="parquet")
    except OSError:
        gc.collect()
        dataset = ds.dataset(file_name, format="parquet")

    # Convert data filter to PyArrow expression (automatic push-down)
    data_filter_expr = convert_data_filter(data_filter) if data_filter else None

    # Get fragments with automatic filter push-down
    fragments = list(dataset.get_fragments(filter=data_filter_expr))

    result = []
    total_row_groups = sum(len(fragment.row_groups) for fragment in fragments)

    row_group_counter = 0
    for fragment in fragments:
        # Iterate row groups within each fragment for memory efficiency
        for rg_info in fragment.row_groups:
            row_group_counter += 1
            if debug:
                print(
                    f"Aggregating row group {row_group_counter} of {total_row_groups}"
                )

            # Read single row group using subset (memory efficient: ~100k rows at a time)
            fragment_subset = fragment.subset(row_group_ids=[rg_info.id])

            try:
                sub = fragment_subset.to_table(
                    columns=all_cols, filter=data_filter_expr
                )
            except OSError:
                gc.collect()
                pa.default_memory_pool().release_unused()  # Return memory to OS
                sub = fragment_subset.to_table(
                    columns=all_cols, filter=data_filter_expr
                )

            # Skip if no rows after filtering
            if sub.num_rows == 0:
                del sub
                continue

            # unneeded columns (when we filter on a non-result column)
            unneeded_columns = [
                col for col in sub.column_names if col not in input_cols
            ]
            if unneeded_columns:
                sub = sub.drop_columns(unneeded_columns)

            if preaggregate:
                sub = groupby_py3(
                    groupby_cols, agg, sub, use_threads=not disable_threads
                )

            result.append(sub)

            if preaggregate and disable_threads:
                # Extra cleanup only when we've pre-aggregated (data is now small)
                # Don't GC when keeping raw data for later concatenation
                # Note: Don't call release_unused() here - we're in a loop and will
                # immediately need to reallocate for the next row group
                gc.collect()

    # combine results
    if debug:
        print("Combining results")

    if not result:
        # Return empty PyArrow table with correct schema
        result_cols = groupby_cols + [x[2] for x in measure_cols]
        return create_empty_result(result_cols, as_df=False)

    table = finalize_group_by(
        result, groupby_cols, agg, aggregate, use_threads=not disable_threads
    )

    rename_columns = {x[0]: x[2] for x in measure_cols if x[0] != x[2]}
    if rename_columns:
        new_columns = [
            rename_columns.get(c_name, c_name) for c_name in table.column_names
        ]
        table = table.rename_columns(new_columns)

    return table


def finalize_group_by(
    result: list[pa.Table],
    groupby_cols: list[str],
    agg: dict[str, str],
    aggregate: bool,
    use_threads: bool = True,
) -> pa.Table:
    """
    Finalize aggregation by combining row group results and performing final groupby.

    Concatenates multiple PyArrow Tables from row group processing and performs
    a final aggregation if needed. Also manages memory cleanup after combining results.

    Args:
        result: List of PyArrow Tables from individual row group processing.
        groupby_cols: List of column names to group by (dimensions).
        agg: Dictionary mapping column names to aggregation operations.
        aggregate: If True, performs final GROUP BY aggregation.
        use_threads: Whether to use threading for aggregation (default True).

    Returns:
        Single PyArrow Table containing final aggregated results.

    Notes:
        - If only one table in result, returns it directly without concatenation.
        - Calls gc.collect() and releases unused memory after concatenation.
        - Threading is disabled automatically when processing many measures (>=20).
    """
    if len(result) == 1:
        table = result[0]
    else:
        table = pa.concat_tables(result)
        del result

    if aggregate:
        table = groupby_py3(groupby_cols, agg, table, use_threads=use_threads)

    gc.collect()  # Free memory from individual row group tables
    pa.default_memory_pool().release_unused()  # Return memory to OS

    return table


def groupby_py3(
    groupby_cols: list[str],
    agg: dict[str, str],
    table: pa.Table,
    use_threads: bool = True,
) -> pa.Table:
    """
    Perform PyArrow group-by aggregation on a table.

    Uses PyArrow's native group_by() and aggregate() functions to perform
    in-memory aggregations. Automatically renames output columns to match
    input column names (removes operation suffixes).

    Args:
        groupby_cols: List of column names to group by (dimensions).
        agg: Dictionary mapping column names to aggregation operations (e.g., {"m1": "sum"}).
        table: PyArrow Table to aggregate.
        use_threads: Whether to use threading for the aggregation (default True).

    Returns:
        Aggregated PyArrow Table with renamed columns.

    Notes:
        - If agg is empty, returns the table unchanged.
        - PyArrow adds operation suffixes (e.g., "col_sum"); these are removed.
        - Supports operations: sum, mean, stddev, count, count_distinct, min, max.
    """
    if not agg:
        return table

    grouped_table = table.group_by(groupby_cols, use_threads=use_threads).aggregate(
        list(agg.items())
    )
    rename_cols = {f"{col}_{op}": col for col, op in agg.items()}
    col_names = [rename_cols.get(c, c) for c in grouped_table.column_names]
    return grouped_table.rename_columns(col_names)


def convert_data_filter(data_filter: DataFilter) -> pc.Expression | None:
    """Convert filter list to PyArrow compute expression."""
    data_filter_expr = None
    for col, sign, values in data_filter:
        if sign == "in":
            expr = pc.field(col).isin(values)
        elif sign in ["not in", "nin"]:
            expr = ~pc.field(col).isin(values)
        elif sign in ["=", "=="]:
            expr = pc.field(col) == values
        elif sign == "!=":
            expr = pc.field(col) != values
        elif sign == ">":
            expr = pc.field(col) > values
        elif sign == ">=":
            expr = pc.field(col) >= values
        elif sign == "<=":
            expr = pc.field(col) <= values
        elif sign == "<":
            expr = pc.field(col) < values
        else:
            valid_ops = ["in", "not in", "nin", "=", "==", "!=", ">", ">=", "<=", "<"]
            raise NotImplementedError(
                f'Filter operation "{sign}" is not supported for column "{col}". '
                f"Valid operators: {', '.join(valid_ops)}"
            )

        if data_filter_expr is None:
            data_filter_expr = expr
        else:
            data_filter_expr = data_filter_expr & expr

    return data_filter_expr
