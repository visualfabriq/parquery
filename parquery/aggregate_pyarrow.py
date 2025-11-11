from __future__ import annotations

import gc
import os
from typing import Any

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

# Import shared types from main aggregate module
from parquery.aggregate import (
    DataFilter,
    FilterOperator,
    FilterValueError,
    SAFE_PREAGGREGATE,
)


def _unify_aggregation_operators(aggregation_list: list[list[str]]) -> dict[str, str]:
    """Convert aggregation operator names to PyArrow-compatible names."""
    rename_operators = {"std": "stddev"}
    return {x[0]: rename_operators.get(x[1], x[1]) for x in aggregation_list}


def aggregate_pq_pyarrow(
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

    # check measure_cols
    measure_cols = check_measure_cols(measure_cols)

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

    # check which columns we need in total
    all_cols, input_cols, result_cols = get_cols(
        data_filter, groupby_cols, measure_cols
    )

    # create pandas-compliant aggregation
    agg = _unify_aggregation_operators(measure_cols)
    agg_ops = set(agg.values())
    # Pre-aggregate only when safe operations AND not too many dimensions
    preaggregate = (
        aggregate and agg_ops.issubset(SAFE_PREAGGREGATE) and not disable_preaggregate
    )

    # if the file does not exist, give back an empty result
    if not os.path.exists(file_name) and handle_missing_file:
        return create_empty_result(result_cols)

    # Create dataset (replaces ParquetFile)
    try:
        dataset = ds.dataset(file_name, format="parquet")
    except OSError:
        gc.collect()
        dataset = ds.dataset(file_name, format="parquet")

    # check if we have all dimensions from the filters
    schema_names = dataset.schema.names
    for col, _, _ in data_filter:
        if col not in schema_names:
            return create_empty_result(result_cols)

    # Convert data filter to PyArrow expression (automatic push-down)
    data_filter_expr = convert_data_filter(data_filter) if data_filter else None

    # Filter columns to only those that exist in the schema
    # (missing columns will be added afterward)
    existing_cols = [col for col in all_cols if col in schema_names]

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
                    columns=existing_cols, filter=data_filter_expr
                )
            except OSError:
                gc.collect()
                pa.default_memory_pool().release_unused()  # Return memory to OS
                sub = fragment_subset.to_table(
                    columns=existing_cols, filter=data_filter_expr
                )

            # Skip if no rows after filtering
            if sub.num_rows == 0:
                del sub
                continue

            # add missing requested columns
            sub = add_missing_columns_to_table(
                sub, measure_cols, all_cols, standard_missing_id, debug
            )

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
        return create_empty_result(result_cols)

    table = finalize_group_by(
        result, groupby_cols, agg, aggregate, use_threads=not disable_threads
    )

    rename_columns = {x[0]: x[2] for x in measure_cols if x[0] != x[2]}
    if rename_columns:
        new_columns = [
            rename_columns.get(c_name, c_name) for c_name in table.column_names
        ]
        table = table.rename_columns(new_columns)

    table = table.select(result_cols)

    return table


def finalize_group_by(
    result: list[pa.Table],
    groupby_cols: list[str],
    agg: dict[str, str],
    aggregate: bool,
    use_threads: bool = True,
) -> pa.Table:
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
    if not agg:
        return table

    grouped_table = table.group_by(groupby_cols, use_threads=use_threads).aggregate(
        list(agg.items())
    )
    rename_cols = {f"{col}_{op}": col for col, op in agg.items()}
    col_names = [rename_cols.get(c, c) for c in grouped_table.column_names]
    return grouped_table.rename_columns(col_names)


def create_empty_result(result_cols: list[str]) -> pa.Table:
    """Create an empty PyArrow Table with the specified columns."""
    # Create empty PyArrow table with schema
    schema = pa.schema([(col, pa.null()) for col in result_cols])
    empty_table = pa.Table.from_pydict({col: [] for col in result_cols}, schema=schema)
    return empty_table


def add_missing_columns_to_table(
    table: pa.Table,
    measure_cols: list[list[str]],
    all_cols: list[str],
    standard_missing_id: int,
    debug: bool,
) -> pa.Table:
    """Add missing columns to table with default values."""
    expected_measure_cols = [x[0] for x in measure_cols]

    # Find all missing columns
    missing_cols = [col for col in all_cols if col not in table.column_names]
    if not missing_cols:
        return table

    # Build all missing columns at once to avoid O(N²) table copies
    missing_data = {}
    for col in missing_cols:
        if col in expected_measure_cols:
            # missing measure columns get a 0.0 result
            standard_value = 0.0
        else:
            # missing dimension columns get the standard id for missing values
            standard_value = standard_missing_id

        if debug:
            print(f"Adding missing column {col} with standard value {standard_value}")

        missing_data[col] = [standard_value] * table.num_rows

    # Create missing columns table and combine with original
    missing_table = pa.table(missing_data)
    combined_schema = pa.schema(list(table.schema) + list(missing_table.schema))
    combined_arrays = list(table.columns) + list(missing_table.columns)

    return pa.Table.from_arrays(combined_arrays, schema=combined_schema)


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


def check_measure_cols(measure_cols: list[str] | list[list[str]]) -> list[list[str]]:
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


def get_cols(
    data_filter: DataFilter, groupby_cols: list[str], measure_cols: list[list[str]]
) -> tuple[list[str], list[str], list[str]]:
    """Get all columns, input columns, and result columns from query parameters."""
    all_cols = sorted(
        list(
            set(
                groupby_cols
                + [x[0] for x in measure_cols]
                + [x[0] for x in data_filter]
            )
        )
    )
    input_cols = list(set(groupby_cols + [x[0] for x in measure_cols]))
    result_cols = sorted(list(set(groupby_cols + [x[2] for x in measure_cols])))
    return all_cols, input_cols, result_cols
