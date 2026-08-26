from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from itertools import chain
from typing import Any
from typing import Literal

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

import pyarrow as pa

import parquery.aggregate_duckdb as aggregate_duckdb
from parquery.aggregate_duckdb import _CAN_PIN_FD
from parquery.aggregate_duckdb import _build_sql_query
from parquery.aggregate_duckdb import aggregate_pq_duckdb
from parquery.aggregate_duckdb import stream_duckdb
from parquery.aggregate_pyarrow import aggregate_pq_pyarrow
from parquery.tool import HAS_DUCKDB
from parquery.tool import HAS_PANDAS
from parquery.tool import DataFilter
from parquery.tool import _add_missing_columns_after_engine
from parquery.tool import create_empty_result
from parquery.tool import get_existing_columns
from parquery.tool import get_result_columns
from parquery.tool import has_missing_filter_columns
from parquery.tool import normalize_measure_cols

logger = logging.getLogger(__name__)


def _empty_record_batch_reader(columns: list[str]) -> pa.RecordBatchReader:
    schema = pa.schema([(column, pa.null()) for column in columns])
    return pa.RecordBatchReader.from_batches(schema, [])


def check_libraries(as_df, engine) -> Any:
    """
    Check and validate library availability for requested features.

    Validates that required libraries are installed for the requested output format
    and engine selection. Auto-selects the best available engine if "auto" is specified.

    Args:
        as_df: Whether pandas DataFrame output is requested (True requires pandas).
        engine: Engine selection - "auto" (default), "duckdb", or "pyarrow".

    Returns:
        Selected engine name ("duckdb" or "pyarrow").

    Raises:
        ImportError: If pandas is required but not installed (when as_df=True).
        ImportError: If DuckDB engine is requested but not installed.

    Notes:
        - "auto" engine selection prefers DuckDB (faster performance) when available.
        - Falls back to PyArrow if DuckDB is not installed.
    """
    if as_df and not HAS_PANDAS:
        # check if pandas is available, when it's requested
        raise ImportError(
            "pandas is required for as_df=True. "
            "Install with: pip install pandas or uv pip install 'parquery[dataframes]'"
        )

    # Auto-select engine
    if engine == "auto":
        engine = "duckdb" if HAS_DUCKDB else "pyarrow"
    elif engine == "duckdb" and HAS_DUCKDB is False:
        raise ImportError(
            "DuckDB engine requested but not installed. "
            "Install with: pip install duckdb or uv pip install 'parquery[performance]'"
        )

    return engine


def aggregate_pq_stream(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[str] | list[list[str]],
    data_filter: DataFilter | None = None,
    aggregate: bool = True,
    standard_missing_id: int = -1,
    handle_missing_file: bool = True,
    debug: bool = False,
    engine: Literal["auto", "duckdb"] = "auto",
    batch_size: int = 65_536,
) -> pa.RecordBatchReader:
    """Stream an aggregation as Arrow record batches.

    This API keeps DuckDB's query and connection open while batches are
    consumed, avoiding the final ``pa.Table`` allocation made by
    :func:`aggregate_pq`. It currently requires DuckDB; a grouped query still
    needs DuckDB's own aggregation state, but the result is not materialized
    into one Python/Arrow table before it can be sent to a client.

    The iterator owns the DuckDB connection. Exhaust it (or call ``close()``)
    when abandoning a query so resources and temporary spill files are cleaned
    up. ``batch_size`` is a requested Arrow batch size, not a row-count promise.
    """
    selected_engine = check_libraries(False, engine)
    if selected_engine != "duckdb":
        raise ValueError("aggregate_pq_stream requires the DuckDB engine")
    data_filter = data_filter or []
    normalized = normalize_measure_cols(measure_cols)
    result_cols = (
        groupby_cols + [spec[2] for spec in normalized if spec[2] not in groupby_cols]
        if aggregate
        else sorted(set(groupby_cols + [spec[2] for spec in normalized]))
    )
    input_cols = set(groupby_cols + [x[0] for x in normalized] + [x[0] for x in data_filter])

    if not os.path.exists(file_name):
        if handle_missing_file:
            return _empty_record_batch_reader(result_cols)
        raise OSError(f"File not found: {file_name}")

    existing_cols = get_existing_columns(file_name, input_cols)
    if not existing_cols or has_missing_filter_columns(data_filter, existing_cols, debug=debug):
        return _empty_record_batch_reader(result_cols)

    filtered_groupby_cols = [col for col in groupby_cols if col in existing_cols]
    filtered_measure_cols = [[col, op, output] for col, op, output in normalized if col in existing_cols]

    def batches() -> Iterator[pa.RecordBatch]:
        file_descriptor = None
        handed_to_stream = False
        try:
            query_file_name = file_name
            if _CAN_PIN_FD:
                file_descriptor = os.open(file_name, os.O_RDONLY)
                query_file_name = f"/dev/fd/{file_descriptor}"
            else:
                if not aggregate_duckdb._fd_fallback_warned:
                    logger.warning(
                        "/dev/fd unavailable; reading %s by path without inode pinning. "
                        "A concurrent atomic rename could yield a corrupt result.",
                        file_name,
                    )
                    aggregate_duckdb._fd_fallback_warned = True
            sql = _build_sql_query(
                query_file_name,
                filtered_groupby_cols,
                filtered_measure_cols,
                data_filter,
                aggregate,
            )
            handed_to_stream = True
            for batch in stream_duckdb(sql, batch_size=batch_size, file_descriptor=file_descriptor):
                # The common case needs no conversion at all: DuckDB already
                # emits the requested result columns in the requested order.
                if batch.schema.names == result_cols:
                    yield batch
                    continue
                table = _add_missing_columns_after_engine(
                    pa.Table.from_batches([batch]),
                    groupby_cols,
                    normalized,
                    standard_missing_id=standard_missing_id,
                    debug=debug,
                )
                yield from table.select(result_cols).to_batches()
        finally:
            if file_descriptor is not None and not handed_to_stream:
                os.close(file_descriptor)

    batches_iterator = batches()
    try:
        first_batch = next(batches_iterator)
    except StopIteration:
        return _empty_record_batch_reader(result_cols)
    return pa.RecordBatchReader.from_batches(first_batch.schema, chain([first_batch], batches_iterator))


def aggregate_pq(
    file_name: str,
    groupby_cols: list[str],
    measure_cols: list[str] | list[list[str]],
    data_filter: DataFilter | None = None,
    aggregate: bool = True,
    as_df: bool = False,
    standard_missing_id: int = -1,
    handle_missing_file: bool = True,
    debug: bool = False,
    engine: Literal["auto", "duckdb", "pyarrow"] = "auto",
) -> pd.DataFrame | pa.Table:
    """
    Aggregate a Parquet file using DuckDB (preferred) or PyArrow.

    This function automatically selects the best engine for aggregation:
    - DuckDB: Faster for most scenarios (default when installed)
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
        - DuckDB provides better performance for most workloads
        - Install DuckDB: pip install 'parquery[performance]'
        - Both engines return identical results
    """
    engine = check_libraries(as_df, engine)

    # Normalize data_filter and measure_cols
    data_filter = data_filter or []
    measure_cols = normalize_measure_cols(measure_cols)

    # Get result columns for empty result case
    result_cols = get_result_columns(groupby_cols, measure_cols)
    input_cols = set(groupby_cols + [x[0] for x in measure_cols] + [x[0] for x in data_filter])

    # Check if file exists (if not, return empty if handle_missing_file=True)
    if not os.path.exists(file_name):
        if handle_missing_file:
            return create_empty_result(result_cols, as_df=as_df)
        else:
            raise OSError(f"File not found: {file_name}")

    # Get existing columns from Parquet schema
    # get_existing_columns returns intersection of requested columns and schema columns
    existing_cols = get_existing_columns(file_name, input_cols)

    # Check if schema read failed or file is invalid
    if not existing_cols:
        # Could be: 1) schema read failed, 2) all requested columns missing
        if debug:
            logger.debug("No requested columns exist in file, returning empty result")
        return create_empty_result(result_cols, as_df=as_df)

    # Check if any filter column is missing
    if has_missing_filter_columns(data_filter, existing_cols, debug=debug):
        return create_empty_result(result_cols, as_df=as_df)

    # Filter inputs to only existing columns
    # Engines should only work with columns that exist
    filtered_groupby_cols = [col for col in groupby_cols if col in existing_cols]
    filtered_measure_cols = [[col, op, output] for col, op, output in measure_cols if col in existing_cols]

    # Route to appropriate engine (engines always return PyArrow Tables)
    # Engines receive only existing columns
    if engine == "duckdb":
        result = aggregate_pq_duckdb(
            file_name=file_name,
            groupby_cols=filtered_groupby_cols,
            measure_cols=filtered_measure_cols,
            data_filter=data_filter,
            aggregate=aggregate,
            debug=debug,
        )
    elif engine == "pyarrow":
        result = aggregate_pq_pyarrow(
            file_name=file_name,
            groupby_cols=filtered_groupby_cols,
            measure_cols=filtered_measure_cols,
            data_filter=data_filter,
            aggregate=aggregate,
            debug=debug,
        )
    else:
        raise ValueError(f"Unknown engine: {engine}. Must be 'auto', 'duckdb', or 'pyarrow'")

    # Add missing expected columns to the result with default values
    result = _add_missing_columns_after_engine(
        result,
        groupby_cols,
        measure_cols,
        standard_missing_id=standard_missing_id,
        debug=debug,
    )

    # Convert to pandas if requested (engines always return PyArrow Tables)
    if as_df:
        return result.to_pandas()
    else:
        return result
