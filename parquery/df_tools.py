import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def check_measure_cols(measure_cols):
    measure_cols = [x if isinstance(x, list) else [x] for x in measure_cols]
    for agg_ops in measure_cols:
        if len(agg_ops) == 1:
            # standard expect a sum aggregation with the same column name
            agg_ops.extend(['sum', agg_ops[0]])
        elif len(agg_ops) == 2:
            # assume the same column name if not specified
            agg_ops.append(agg_ops[0])
    return measure_cols


def get_cols(data_filter, groupby_cols, measure_cols):
    data_filter = data_filter or []
    all_cols = sorted(list(set(groupby_cols + [x[0] for x in measure_cols] + [x[0] for x in data_filter])))
    input_cols = list(set(groupby_cols + [x[0] for x in measure_cols]))
    result_cols = sorted(list(set(groupby_cols + [x[2] for x in measure_cols])))
    return all_cols, input_cols, result_cols


def create_empty_result(groupby_cols, measure_cols, as_df):
    _, _, result_cols = get_cols([], groupby_cols, measure_cols)
    df = pd.DataFrame(None, columns=result_cols)
    if as_df:
        res = df
    else:
        res = pa.Table.from_pandas(df, preserve_index=False)
    return res
