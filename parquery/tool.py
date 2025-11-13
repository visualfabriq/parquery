from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

try:
    import duckdb  # noqa: F401

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


FILTER_CUTOVER_LENGTH = 10
SAFE_PREAGGREGATE = set(["min", "max", "sum", "one"])
FilterOperator = Literal["in", "not in", "nin", "=", "==", "!=", ">", ">=", "<=", "<"]
FilterCondition = tuple[str, FilterOperator, Any]  # (column, operator, value(s))
DataFilter = (
    list[FilterCondition] | list[list[Any]]
)  # Support both typed and legacy format


class FilterValueError(ValueError):
    pass


def df_to_natural_name(
    df: pd.DataFrame | pl.DataFrame | pa.Table,
) -> pd.DataFrame | pl.DataFrame | pa.Table:
    """
    Convert DataFrame/Table column names from original to natural format.
    Replaces '-' with '_n_' in column names.

    Args:
        df: pandas DataFrame, Polars DataFrame, or PyArrow Table

    Returns:
        Modified DataFrame/Table with renamed columns
        - pandas: modified in-place and returned
        - Polars: returns new DataFrame (immutable)
        - PyArrow: returns new Table (immutable)

    Raises:
        TypeError: If df is not a supported type
    """
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # pandas: modify in-place and return
        df.columns = [col.replace("-", "_n_") for col in df.columns]
        return df
    elif HAS_POLARS and isinstance(df, pl.DataFrame):
        # Polars: use rename (immutable)
        rename_map = {col: col.replace("-", "_n_") for col in df.columns}
        return df.rename(rename_map)
    elif isinstance(df, pa.Table):
        # PyArrow: use rename_columns
        new_names = [col.replace("-", "_n_") for col in df.column_names]
        return df.rename_columns(new_names)
    else:
        raise TypeError(
            f"df must be a pandas DataFrame, Polars DataFrame, or PyArrow Table, got {type(df)}"
        )


def df_to_original_name(
    df: pd.DataFrame | pl.DataFrame | pa.Table,
) -> pd.DataFrame | pl.DataFrame | pa.Table:
    """
    Convert DataFrame/Table column names from natural to original format.
    Replaces '_n_' with '-' in column names.

    Args:
        df: pandas DataFrame, Polars DataFrame, or PyArrow Table

    Returns:
        Modified DataFrame/Table with renamed columns
        - pandas: modified in-place and returned
        - Polars: returns new DataFrame (immutable)
        - PyArrow: returns new Table (immutable)

    Raises:
        TypeError: If df is not a supported type
    """
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # pandas: modify in-place and return
        df.columns = [col.replace("_n_", "-") for col in df.columns]
        return df
    elif HAS_POLARS and isinstance(df, pl.DataFrame):
        # Polars: use rename (immutable)
        rename_map = {col: col.replace("_n_", "-") for col in df.columns}
        return df.rename(rename_map)
    elif isinstance(df, pa.Table):
        # PyArrow: use rename_columns
        new_names = [col.replace("_n_", "-") for col in df.column_names]
        return df.rename_columns(new_names)
    else:
        raise TypeError(
            f"df must be a pandas DataFrame, Polars DataFrame, or PyArrow Table, got {type(df)}"
        )


def get_existing_columns(file_name: str, all_cols: set[str]) -> set[str]:
    """
    Get the set of columns (from all_cols) that exist in the Parquet file.

    Args:
        file_name: Path to the Parquet file.
        all_cols: List of all columns we want to query.

    Returns:
        Set of column names from all_cols that exist in the file.
        Returns empty set if schema cannot be read.
    """
    try:
        schema = pa.parquet.read_schema(file_name)
        schema_cols = set(schema.names)
        # Return intersection: only columns from all_cols that exist in schema
        return schema_cols & all_cols
    except Exception:
        # If we can't read schema, return empty (will trigger empty result)
        return set()


def has_missing_filter_columns(
    data_filter: DataFilter,
    existing_cols: set[str],
    debug: bool = False,
) -> bool:
    """
    Check if any filter references a missing column.

    If a filter references a column that doesn't exist in the parquet file,
    we should return empty result. This is a simple and safe approach:
    filtering on missing columns is likely a mistake.

    Args:
        data_filter: List of filter conditions.
        existing_cols: Set of columns that exist in the parquet file.
        debug: If True, print debug information.

    Returns:
        True if any filter references a missing column.
    """
    if not data_filter:
        return False

    for col, operator, values in data_filter:
        if col not in existing_cols:
            # Filter references missing column
            if debug:
                print(
                    f"Filter column '{col}' is missing from file, returning empty result"
                )
            return True

    # All filter columns exist
    return False


def normalize_measure_cols(
    measure_cols: list[str] | list[list[str]],
) -> list[list[str]]:
    """
    Normalize measure columns to list of [column, operation, output_name].

    Args:
        measure_cols: Can be:
            - List of column names: ['m1', 'm2'] - performs sum on each
            - List of [column, operation]: [['m1', 'sum'], ['m2', 'count']]
            - List of [column, operation, output_name]: [['m1', 'sum', 'm1_total']]

    Returns:
        Normalized list where each element is [column, operation, output_name].
    """
    measure_cols = [x if isinstance(x, list) else [x] for x in measure_cols]
    for agg_ops in measure_cols:
        if len(agg_ops) == 1:
            # standard expect a sum aggregation with the same column name
            agg_ops.extend(["sum", agg_ops[0]])
        elif len(agg_ops) == 2:
            # assume the same column name if not specified
            agg_ops.append(agg_ops[0])
    return measure_cols


def get_result_columns(
    groupby_cols: list[str], measure_cols: list[list[str]]
) -> list[str]:
    """
    Get the list of result column names from groupby and measure specs.

    Args:
        groupby_cols: List of groupby column names.
        measure_cols: List of measure column specs [col, op, output_name].

    Returns:
        Sorted list of result column names.
    """
    return sorted(list(set(groupby_cols + [x[2] for x in measure_cols])))


def create_empty_result(result_cols: list[str], as_df: bool = False) -> pa.Table:
    """
    Create an empty PyArrow Table with the specified columns.

    Args:
        result_cols: List of column names for the result.

    Returns:
        Empty PyArrow Table with the specified schema.
    """
    schema = pa.schema([(col, pa.null()) for col in result_cols])
    df = pa.Table.from_pydict({col: [] for col in result_cols}, schema=schema)
    if as_df:
        return df.to_pandas()
    return df


def _add_missing_columns_after_engine(
    table: pa.Table,
    groupby_cols: list[str],
    measure_cols: list[list[str]],
    standard_missing_id: int = -1,
    debug: bool = False,
) -> pa.Table:
    """
    Add missing columns to table with default values after engine returns.

    This is called after the engine has returned its result. The engine only
    queried existing columns, so we need to add any missing columns with defaults.

    Args:
        table: PyArrow Table from engine.
        measure_cols: List of measure column specs [col, op, output_name].
        all_cols: List of all columns that were requested.
        standard_missing_id: Default value for missing dimension columns.
        debug: If True, print debug information.

    Returns:
        PyArrow Table with missing columns added.
    """
    result_cols = set(table.column_names)
    missing_groupby_cols = [col for col in groupby_cols if col not in result_cols]
    expected_measure_cols = [x[2] for x in measure_cols]
    missing_measure_cols = [
        col for col in expected_measure_cols if col not in result_cols
    ]
    remove_cols = [
        col
        for col in result_cols
        if col not in groupby_cols and col not in expected_measure_cols
    ]

    # remove the unneeded columns, if any
    if remove_cols:
        table = table.drop(remove_cols)

    # Find all missing columns
    if not missing_groupby_cols and not missing_measure_cols:
        return table

    # Build all missing columns at once to avoid O(N²) table copies
    missing_data = {}
    for col in missing_measure_cols:
        missing_data[col] = [0.0] * table.num_rows
        print("Adding missing measure column {col} with standard value 0.0")
    for col in missing_groupby_cols:
        missing_data[col] = [standard_missing_id] * table.num_rows
        print(
            f"Adding missing groupby column {col} with standard value {standard_missing_id}"
        )

    # Create missing columns table and combine with original
    missing_table = pa.table(missing_data)
    combined_schema = pa.schema(list(table.schema) + list(missing_table.schema))
    combined_arrays = list(table.columns) + list(missing_table.columns)

    return pa.Table.from_arrays(combined_arrays, schema=combined_schema)
