import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from parquery.tool import df_to_natural_name, df_to_original_name

FILTER_CUTOVER_LENGTH = 10
from parquery.df_tools import create_empty_result, check_measure_cols

try:
    import polars as pl
    from parquery.aggregate_polars import apply_parquet_aggregation
except ImportError:
    from parquery.aggregate_pandas import apply_parquet_aggregation


def aggregate_pq(
        file_name,
        groupby_cols,
        measure_cols,
        data_filter=None,
        aggregate=True,
        row_group_filter=None,
        as_df=True,
        standard_missing_id=-1,
        handle_missing_file=True,
        debug=False):
    """
    A function to aggregate a parquetfile using pandas
    NB: we assume that all columns are strings

    """
    data_filter = data_filter or []
    measure_cols = check_measure_cols(measure_cols)

    # if the file does not exist, give back an empty result
    if not os.path.exists(file_name) and handle_missing_file:
        return create_empty_result(groupby_cols, measure_cols, as_df)

    # get result
    pq_file = pq.ParquetFile(file_name)

    # check if we have all dimensions from the filters
    for col, _, _ in data_filter:
        if col not in pq_file.metadata.schema.names:
            return create_empty_result(groupby_cols, measure_cols, as_df)

    return apply_parquet_aggregation(
        file_name,
        groupby_cols,
        measure_cols,
        data_filter=data_filter,
        aggregate=aggregate,
        row_group_filter=row_group_filter,
        as_df=as_df,
        standard_missing_id=standard_missing_id,
        debug=debug)
