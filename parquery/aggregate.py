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
import pyarrow.parquet as pq


class FilterValueError(ValueError):
    pass


FILTER_CUTOVER_LENGTH = 10
SAFE_PREAGGREGATE = set(["min", "max", "sum", "one"])


def _unify_aggregation_operators(aggregation_list: list[list[str]]) -> dict[str, str]:
    """Convert aggregation operator names to PyArrow-compatible names."""
    rename_operators = {"std": "stddev"}
    return {x[0]: rename_operators.get(x[1], x[1]) for x in aggregation_list}


def aggregate_pq(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[str] | list[list[str]],
    data_filter: list[list[Any]] | None = None,
    aggregate: bool = True,
    row_group_filter: int | None = None,
    as_df: bool | None = None,
    standard_missing_id: int = -1,
    handle_missing_file: bool = True,
    debug: bool = False,
) -> pd.DataFrame | pa.Table:
    """
    A function to aggregate a parquet file using PyArrow
    NB: we assume that all columns are strings

    Args:
        as_df: Return results as pandas DataFrame (True), PyArrow Table (False),
               or auto-detect based on pandas availability (None, default).
               If None: returns DataFrame if pandas is installed, else PyArrow Table.
    """
    # Smart default: use pandas if available, otherwise PyArrow
    if as_df is None:
        as_df = HAS_PANDAS

    if as_df and not HAS_PANDAS:
        raise ImportError(
            "pandas is required for as_df=True. "
            "Install with: pip install pandas or uv pip install 'parquery[optional]'"
        )

    data_filter = data_filter or []

    # check measure_cols
    measure_cols = check_measure_cols(measure_cols)

    # check which columns we need in total
    all_cols, input_cols, result_cols = get_cols(
        data_filter, groupby_cols, measure_cols
    )

    # create pandas-compliant aggregation
    agg = _unify_aggregation_operators(measure_cols)
    agg_ops = set(agg.values())
    preaggregate = aggregate and agg_ops.issubset(SAFE_PREAGGREGATE)

    # if the file does not exist, give back an empty result
    if not os.path.exists(file_name) and handle_missing_file:
        return create_empty_result(result_cols, as_df)

    # get result
    try:
        pq_file = pq.ParquetFile(file_name)
    except OSError:
        gc.collect()
        pq_file = pq.ParquetFile(file_name)

    # check if we have all dimensions from the filters
    for col, _, _ in data_filter:
        if col not in pq_file.metadata.schema.names:
            return create_empty_result(result_cols, as_df)

    # check filters
    if data_filter:
        metadata_filter = convert_metadata_filter(data_filter, pq_file)
        data_filter_expr = convert_data_filter(data_filter)
    else:
        metadata_filter = None
        data_filter_expr = None

    num_row_groups = (
        [row_group_filter]
        if row_group_filter is not None
        else range(pq_file.num_row_groups)
    )
    result = []
    for row_group in num_row_groups:
        if debug:
            print(
                "Aggregating row group "
                + str(row_group + 1)
                + " of "
                + str(pq_file.num_row_groups)
            )

        # push down filter
        if metadata_filter:
            skip = rowgroup_metadata_filter(metadata_filter, pq_file, row_group)
            if skip:
                continue

        # get data into df
        try:
            sub = pq_file.read_row_group(row_group, columns=all_cols)
        except OSError:
            gc.collect()
            sub = pq_file.read_row_group(row_group, columns=all_cols)

        # add missing requested columns
        sub = add_missing_columns_to_table(
            sub, measure_cols, all_cols, standard_missing_id, debug
        )

        if data_filter_expr is not None:
            sub = sub.filter(data_filter_expr)
            if sub.num_rows == 0:
                del sub
                continue

        # unneeded columns (when we filter on a non-result column)
        unneeded_columns = [col for col in sub.column_names if col not in input_cols]
        if unneeded_columns:
            sub = sub.drop_columns(unneeded_columns)

        if preaggregate:
            sub = groupby_py3(groupby_cols, agg, sub)

        result.append(sub)

    # combine results
    if debug:
        print("Combining results")

    if not result:
        return create_empty_result(result_cols, as_df)

    table = finalize_group_by(result, groupby_cols, agg, aggregate)

    rename_columns = {x[0]: x[2] for x in measure_cols if x[0] != x[2]}
    if rename_columns:
        new_columns = [
            rename_columns.get(c_name, c_name) for c_name in table.column_names
        ]
        table = table.rename_columns(new_columns)

    table = table.select(result_cols)

    if not as_df:
        return table

    if HAS_PANDAS:
        return table.to_pandas()
    else:
        raise ImportError(
            "pandas is required for as_df=True. "
            "Install with: pip install pandas or uv pip install 'parquery[optional]'"
        )


def finalize_group_by(
    result: list[pa.Table],
    groupby_cols: list[str],
    agg: dict[str, str],
    aggregate: bool,
) -> pa.Table:
    if len(result) == 1:
        table = result[0]
    else:
        table = pa.concat_tables(result)
        del result

    if aggregate:
        table = groupby_py3(groupby_cols, agg, table)

    return table


def groupby_py3(
    groupby_cols: list[str], agg: dict[str, str], table: pa.Table
) -> pa.Table:
    """Perform groupby aggregation on PyArrow table."""
    if not agg:
        return table

    grouped_table = table.group_by(groupby_cols).aggregate(list(agg.items()))
    rename_cols = {"{}_{}".format(col, op): col for col, op in agg.items()}
    col_names = [rename_cols.get(c, c) for c in grouped_table.column_names]
    return grouped_table.rename_columns(col_names)


def create_empty_result(result_cols: list[str], as_df: bool) -> pd.DataFrame | pa.Table:
    """Create an empty result with the specified columns."""
    # Create empty PyArrow table with schema
    schema = pa.schema([(col, pa.null()) for col in result_cols])
    empty_table = pa.Table.from_pydict({col: [] for col in result_cols}, schema=schema)

    if as_df:
        if HAS_PANDAS:
            return empty_table.to_pandas()
        else:
            raise ImportError(
                "pandas is required for as_df=True. "
                "Install with: pip install pandas or uv pip install 'parquery[optional]'"
            )
    else:
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

    for col in all_cols:
        if col in table.column_names:
            continue

        if col in expected_measure_cols:
            # missing measure columns get a 0.0 result
            standard_value = 0.0
        else:
            # missing dimension columns get the standard id for missing values
            standard_value = standard_missing_id

        if debug:
            print(
                "Adding missing column {} with standard value {}".format(
                    col, standard_value
                )
            )

        new_col = [standard_value] * len(table)
        table = table.append_column(col, [new_col])

    return table


def convert_metadata_filter(
    data_filter: list[list[Any]], pq_file: pq.ParquetFile
) -> list[list[Any]]:
    """Convert filter to metadata filter for push-down filtering."""
    # we check if we have INT type of columns to try to do pushdown statistics filtering
    metadata_filter = [
        [pq_file.metadata.schema.names.index(col), sign, values]
        for col, sign, values in data_filter
    ]
    metadata_filter = [
        [col_nr, sign, values]
        for col_nr, sign, values in metadata_filter
        if pq_file.schema.column(col_nr).physical_type
        in ["INT8", "INT16", "INT32", "INT64"]
    ]
    return metadata_filter


def convert_data_filter(data_filter: list[list[Any]]) -> pc.Expression | None:
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
            raise NotImplementedError('Operation "{}" is not supported'.format(sign))

        if data_filter_expr is None:
            data_filter_expr = expr
        else:
            data_filter_expr = data_filter_expr & expr

    return data_filter_expr


def rowgroup_metadata_filter(
    metadata_filter: list[list[Any]], pq_file: pq.ParquetFile, row_group: int
) -> bool:
    """
    Check if the filter applies, if the filter does not apply skip the row_group.

    Args:
        metadata_filter: List of filters e.g. [[0, '>', 10000]]
        pq_file: PyArrow Parquet file to be checked
        row_group: Row group index to check

    Returns:
        True if row_group should be skipped otherwise False
    """
    rg_meta = pq_file.metadata.row_group(row_group)
    if rg_meta.num_rows == 0:
        return True
    for col_nr, sign, values in metadata_filter:
        rg_col = rg_meta.column(col_nr)
        min_val = rg_col.statistics.min
        max_val = rg_col.statistics.max

        try:
            # if the filter is not in the boundary of the range, then skip the rowgroup
            if sign == "in":
                if not any(min_val <= val <= max_val for val in values):
                    return True
            elif sign == "not in":
                if any(min_val <= val <= max_val for val in values):
                    return True
            elif sign in ["=", "=="]:
                if not min_val <= values <= max_val:
                    return True
            elif sign == "!=":
                if min_val <= values <= max_val:
                    return True
            elif sign == ">":
                if max_val <= values:
                    return True
            elif sign == ">=":
                if max_val < values:
                    return True
            elif sign == "<":
                if min_val >= values:
                    return True
            elif sign == "<=":
                if min_val > values:
                    return True
        except TypeError:
            raise FilterValueError(
                "Dimension filters MUST be numbers, please convert dimension values to dimension ids"
            )

    return False


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
    data_filter: list[list[Any]], groupby_cols: list[str], measure_cols: list[list[str]]
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
