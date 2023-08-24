import os

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import six

from parquery.tool import df_to_natural_name, df_to_original_name

class FilterValueError(ValueError):
    pass

FILTER_CUTOVER_LENGTH = 10
SAFE_PREAGGREGATE = set(['min', 'max', 'sum'])

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

    # check measure_cols
    measure_cols = check_measure_cols(measure_cols)

    # check which columns we need in total
    all_cols, input_cols, result_cols = get_cols(data_filter, groupby_cols, measure_cols)

    # create pandas-compliant aggregation
    agg = {x[0]: x[1].replace('count_distinct', 'nunique') for x in measure_cols}
    agg_ops = set(agg.values())
    preaggregate = (
        aggregate
        and agg_ops.issubset(SAFE_PREAGGREGATE)
        and groupby_cols
    )

    # if the file does not exist, give back an empty result
    if not os.path.exists(file_name) and handle_missing_file:
        return create_empty_result(result_cols, as_df)

    # get result
    pq_file = pq.ParquetFile(file_name)

    # check if we have all dimensions from the filters
    for col, _, _ in data_filter:
        if col not in pq_file.metadata.schema.names:
            return create_empty_result(result_cols, as_df)

    # check filters
    if data_filter:
        metadata_filter = convert_metadata_filter(data_filter, pq_file)
        data_filter_str, data_filter_sets, data_filter_expr = convert_data_filter(data_filter)
    else:
        metadata_filter = None
        data_filter_str = None
        data_filter_sets = None
        data_filter_expr = None

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
        sub = pq_file.read_row_group(row_group, columns=all_cols)

        # add missing requested columns
        sub = add_missing_columns_to_table(sub, measure_cols, all_cols, standard_missing_id, debug)

        if six.PY3:
            if data_filter_expr is not None:
                sub = sub.filter(data_filter_expr)

            # unneeded columns (when we filter on a non-result column)
            unneeded_columns = [col for col in sub.column_names if col not in input_cols]
            if unneeded_columns:
                sub = sub.drop_columns(unneeded_columns)

            if preaggregate:
                sub = groupby_py3(groupby_cols, agg, sub)

            result.append(sub)
        else:
            df = sub.to_pandas()
            if df.empty:
                continue

            # filter
            if data_filter:
                # filter based on the given requirements
                apply_data_filter(data_filter_str, data_filter_sets, df)
                if df.empty:
                    # no values for this rowgroup
                    del sub
                    del df
                    continue

            # unneeded columns (when we filter on a non-result column)
            unneeded_columns = [col for col in df if col not in input_cols]
            if unneeded_columns:
                df.drop(unneeded_columns, axis=1, inplace=True)

            if preaggregate:
                df = groupby_result(agg, df, groupby_cols, measure_cols)

            # save the result
            result.append(df)

            # cleanup
            del sub

    # combine results
    if debug:
        print('Combining results')

    if result:
        if six.PY3:
            if len(result) == 1:
                table = result[0]
            else:
                table = pa.concat_tables(result)
            df = table.to_pandas()
        else:
            if len(result) == 1:
                df = result[0]
            else:
                df = pd.concat(result, axis=0, ignore_index=True, sort=False, copy=False)

        if aggregate:
            df = groupby_result(agg, df, groupby_cols, measure_cols)

        if row_group_filter is not None:
            df = df.rename(columns={x[0]: x[2] for x in measure_cols})

        # cleanup
        del result

    else:
        # empty result
        df = pd.DataFrame(None, columns=result_cols)

    # ensure order
    df = df[result_cols]

    if as_df:
        return df
    else:
        return pa.Table.from_pandas(df, preserve_index=False)

def groupby_py3(groupby_cols, agg, table):
    grouped_table = table.group_by(groupby_cols).aggregate(list(agg.items()))
    rename_cols = {'{}_{}'.format(col, op): col for col, op in agg.items()}
    col_names = [rename_cols.get(c, c) for c in grouped_table.column_names]
    return grouped_table.rename_columns(col_names)


def create_empty_result(result_cols, as_df):
    df = pd.DataFrame(None, columns=result_cols)
    if as_df:
        res = df
    else:
        res = pa.Table.from_pandas(df, preserve_index=False)
    return res


def add_missing_columns_to_table(table, measure_cols, all_cols, standard_missing_id, debug):
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
            print('Adding missing column {} with standard value {}'.format(col, standard_value))

        new_col = [standard_value] * len(table)
        table = table.append_column(col, [new_col])

    return table


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
    data_filter = data_filter or []

    # check measure_cols
    measure_cols = check_measure_cols(measure_cols)

    # check which columns we need in total
    _, input_cols, result_cols = get_cols(data_filter, groupby_cols, measure_cols)

    df = pa_table.to_pandas()

    # filter
    if data_filter:
        # filter based on the given requirements
        data_filter_str, data_filter_sets = convert_data_filter(data_filter)
        apply_data_filter(data_filter_str, data_filter_sets, df)

    # check if we have a result still
    if df.empty:
        return pd.DataFrame(None, columns=result_cols)

    # unneeded columns (when we filter on a non-result column)
    unneeded_columns = [col for col in df if col not in input_cols]
    if unneeded_columns:
        df.drop(unneeded_columns, axis=1, inplace=True)

    # aggregate
    if aggregate:
        # create pandas-compliant aggregation
        agg = {x[0]: x[1].replace('count_distinct', 'nunique') for x in measure_cols}
        # aggregate
        df = groupby_result(agg, df, groupby_cols, measure_cols)

    # rename the columns
    df = df.rename(columns={x[0]: x[2] for x in measure_cols})

    # ensure order
    df = df[result_cols]

    if as_df:
        return df
    else:
        return pa.Table.from_pandas(df, preserve_index=False)


def apply_data_filter(data_filter_str, data_filter_set, df):
    # filter on the straight eval
    if data_filter_str:
        df_to_natural_name(df)
        mask = df.eval(data_filter_str)
        df_to_original_name(df)
        df.drop(df[~mask].index, inplace=True)

    # filter on each longer list
    if data_filter_set:
        for col, sign, values in data_filter_set:
            if sign == 'in':
                mask = df[col].isin(values)
                df.drop(df[~mask].index, inplace=True)
            elif sign in ['not in', 'nin']:
                mask = df[col].isin(values)
                df.drop(df[mask].index, inplace=True)
            else:
                raise NotImplementedError('Unknown sign for set filter {}'.format(sign))


def groupby_result(agg, df, groupby_cols, measure_cols):
    if not agg:
        return df.drop_duplicates()

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
    data_filter_strs = []
    data_filter_sets = []
    data_filter_expr = None
    for col, sign, values in data_filter:
        if six.PY2:
            if isinstance(values, list) and len(values) > FILTER_CUTOVER_LENGTH:
                data_filter_sets.append((col, sign, set(values)))
            else:
                data_filter_strs.append(col.replace('-', '_n_') + ' ' + sign + ' ' + str(values))
            continue
        else:
            if sign == 'in':
                expr = pc.field(col).isin(values)
            elif sign in ['not in', 'nin']:
                expr = ~pc.field(col).isin(values)
            elif sign in ['=', '==']:
                expr = pc.field(col) == values
            elif sign == '!=':
                expr = pc.field(col) != values
            elif sign == '>':
                expr = pc.field(col) > values
            elif sign == '>=':
                expr = pc.field(col) >= values
            elif sign == '<=':
                expr = pc.field(col) <= values
            elif sign == '<':
                expr = pc.field(col) < values
            else:
                raise NotImplementedError('Operation "{}" is not supported'.format(sign))

            if data_filter_expr is None:
                data_filter_expr = expr
            else:
                data_filter_expr = data_filter_expr & expr

    data_filter_str = ' and '.join(data_filter_strs)
    return data_filter_str, data_filter_sets, data_filter_expr


def rowgroup_metadata_filter(metadata_filter, pq_file, row_group):
    """
    Check if the filter applies, if the filter does not apply skip the row_group.
    Args:
        metadata_filter (list of list): e.g. [[0, '>', 10000]]
        pq_file (pyarrow.parquet.ParquetFile): file to be checked
        row_group:

    Returns (bool): True if row_group should be skipped otherwise False

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
            if sign == 'in':
                if not any(min_val <= val <= max_val for val in values):
                    return True
            elif sign == 'not in':
                if any(min_val <= val <= max_val for val in values):
                    return True
            elif sign in ['=', '==']:
                if not min_val <= values <= max_val:
                    return True
            elif sign == '!=':
                if min_val <= values <= max_val:
                    return True
            elif sign == '>':
                if max_val <= values:
                    return True
            elif sign == '>=':
                if max_val < values:
                    return True
            elif sign == '<':
                if min_val >= values:
                    return True
            elif sign == '<=':
                if min_val > values:
                    return True
        except TypeError:
            raise FilterValueError('Dimension filters MUST be numbers, please convert dimension values to dimension ids')

    return False


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
    all_cols = sorted(list(set(groupby_cols + [x[0] for x in measure_cols] + [x[0] for x in data_filter])))
    input_cols = list(set(groupby_cols + [x[0] for x in measure_cols]))
    result_cols = sorted(list(set(groupby_cols + [x[2] for x in measure_cols])))
    return all_cols, input_cols, result_cols
