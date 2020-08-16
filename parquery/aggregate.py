import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def aggregate_pq(
        file_name,
        groupby_cols,
        measure_cols,
        data_filter=None,
        aggregate=True,
        row_group_filter=None,
        as_df=True,
        debug=False):
    """
    A function to aggregate a parquetfile using pandas
    NB: we assume that all columns are strings

    """

    pq_file = pq.ParquetFile(file_name)

    if data_filter:
        metadata_filter = convert_metadata_filter(data_filter, pq_file)
        data_filter = convert_data_filter(data_filter)
    else:
        metadata_filter = None

    # check measure_cols
    measure_cols = check_measure_cols(measure_cols)

    # check which columns we need in total
    cols = groupby_cols + [x[0] for x in measure_cols]

    # create pandas-compliant aggregation
    agg = {x[0]: x[1].replace('count_distinct', 'nunique') for x in measure_cols}

    # get result
    num_row_groups = [row_group_filter] if row_group_filter is not None else range(pq_file.num_row_groups)
    result = []
    for row_group in num_row_groups:
        if debug:
            print('Aggregating row group ' + str(row_group + 1) + ' of ' + str(pq_file.num_row_groups))

        # push down filter
        if metadata_filter:
            skip = rowgroup_metadata_filter(metadata_filter, pq_file, row_group)
            if skip:
                continue

        # get data into df
        sub = pq_file.read_row_group(row_group, columns=cols)
        df = sub.to_pandas()
        if df.empty:
            continue

        # filter
        if data_filter:
            # filter based on the given requirements
            mask = df.eval(data_filter)
            if not mask.any():
                # no values for this rowgroup
                del sub
                del row_group
                del df
                continue
            df = df[mask]

        # aggregate
        if aggregate:
            df = groupby_result(agg, df, groupby_cols, measure_cols)

        # save the result
        result.append(df)

        # cleanup
        del sub
        del row_group

    # combine results
    if debug:
        print('Combining results')

    if result:
        if len(result) == 1:
            df = result[0]
        else:
            df = pd.concat(result, ignore_index=True, sort=False)
            if aggregate:
                df = groupby_result(agg, df, groupby_cols, measure_cols)

        if row_group_filter is not None:
            df = df.rename(columns={x[0]: x[2] for x in measure_cols})

        # cleanup
        del result

    else:
        # empty result
        df = pd.DataFrame(None, columns=cols)

    if as_df:
        return df
    else:
        return pa.Table.from_pandas(df, preserve_index=False)


def aggregate_pa(
        pa_table,
        groupby_cols,
        measure_cols,
        data_filter=None,
        aggregate=True,
        as_df=True,
        debug=False):
    """
    A function to aggregate a arrow table using pandas
    NB: we assume that all columns are strings

    """
    if data_filter:
        data_filter = convert_data_filter(data_filter)

    # check measure_cols
    measure_cols = check_measure_cols(measure_cols)

    # create pandas-compliant aggregation
    agg = {x[0]: x[1].replace('count_distinct', 'nunique') for x in measure_cols}

    df = pa_table.to_pandas()

    # filter
    if data_filter:
        # filter based on the given requirements
        mask = df.eval(data_filter)
        df = df[mask]

    # aggregate
    if aggregate:
        df = groupby_result(agg, df, groupby_cols, measure_cols)

    if as_df:
        return df
    else:
        return pa.Table.from_pandas(df, preserve_index=False)


def groupby_result(agg, df, groupby_cols, measure_cols):
    if groupby_cols:
        df = df.groupby(groupby_cols, as_index=False).agg(agg)
    else:
        ser = df.apply(agg)
        df = pd.DataFrame([{col[0]: ser[col[0]] for col in measure_cols}])
    return df


def convert_metadata_filter(data_filter, pq_file):
    # we check if we have INT type of columns to try to do pushdown statistics filtering
    metadata_filter = [
        [pq_file.metadata.schema.names.index(col), sign, values]
        for col, sign, values in data_filter
    ]
    metadata_filter = [[col_nr, sign, values] for col_nr, sign, values in metadata_filter
                       if pq_file.schema.column(col_nr).physical_type in ['INT8', 'INT16', 'INT32', 'INT64']
                       ]
    return metadata_filter


def convert_data_filter(data_filter):
    data_filter_str = ' and '.join([col + ' ' + sign + ' ' + str(values) for col, sign, values in data_filter])
    return data_filter_str


def rowgroup_metadata_filter(metadata_filter, pq_file, row_group):
    rg_meta = pq_file.metadata.row_group(row_group)
    skip = False
    for col_nr, sign, values in metadata_filter:
        rg_col = rg_meta.column(col_nr)
        min_val = rg_col.statistics.min
        max_val = rg_col.statistics.max

        # if the filter is not in the boundary of the range, then skip the rowgroup
        if sign == 'in':
            if not any(min_val <= val <= max_val for val in values):
                skip = True
                break
        elif sign == 'not in':
            if any(min_val <= val <= max_val for val in values):
                skip = True
                break
        elif sign in ['=', '==']:
            if not min_val <= values <= max_val:
                skip = True
                break
        elif sign == '!=':
            if min_val <= values <= max_val:
                skip = True
                break
        elif sign == '>':
            if max_val <= values:
                skip = True
                break
        elif sign == '>=':
            if max_val < values:
                skip = True
                break
        elif sign == '<':
            if min_val >= values:
                skip = True
                break
        elif sign == '<=':
            if min_val > values:
                skip = True
                break
    return skip


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
