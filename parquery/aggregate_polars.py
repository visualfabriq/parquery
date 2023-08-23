import os
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from parquery.df_tools import check_measure_cols, get_cols, create_empty_result
from parquery.tool import df_to_natural_name, df_to_original_name


def apply_parquet_aggregation(
        file_name,
        groupby_cols,
        measure_cols,
        data_filter=None,
        aggregate=True,
        row_group_filter=None,
        as_df=True,
        standard_missing_id=-1,
        debug=False):
    """
    A function to aggregate a parquetfile using pandas
    NB: we assume that all columns are strings

    file_name = self.filename
    groupby_cols = groupby_cols
    measure_cols = agg_list
    data_filter=terms_filter
    aggregate=True
    row_group_filter=None
    as_df=True
    standard_missing_id=-1
    handle_missing_file=True
    debug=True

    """
    if row_group_filter is not None:
        raise NotImplementedError('Row Group Filter cannot be used with Polars')

    # create a polars compatible filter
    polars_filter = create_polars_filter(data_filter)

    # check measure_cols
    measure_cols = check_measure_cols(measure_cols)

    # check which columns we need in total
    all_cols, input_cols, result_cols = get_cols(data_filter, groupby_cols, measure_cols)

    # check if a measure or dimension column is missing, then add that later as 0.0
    info = pl.read_parquet_schema(file_name)

    expected_groupby_cols = groupby_cols[:]
    groupby_cols = [x for x in groupby_cols if x in info]
    expected_measure_cols = [x[2] for x in measure_cols]
    measure_cols = [x for x in measure_cols if x[0] in info]
    all_cols = [x for x in all_cols if x in info]

    # crate an aggregation expresion
    agg_expr = create_agg_expr(groupby_cols, measure_cols, aggregate)

    if not measure_cols:
        raise ValueError('Non measure columns')

    # now execute
    if aggregate:
        if data_filter:
            if groupby_cols:
                # aggregate + data filter + group by cols
                q = (
                    pl.scan_parquet(file_name, rechunk=False)
                    .filter(polars_filter)
                    .groupby(groupby_cols)
                    .agg(**agg_expr)
                )
                df = q.collect(streaming=True)
            else:
                # aggregate + data filter + no group by cols
                results = []
                for expr in set(x[1] for x in measure_cols):
                    results.append(eval(""
                    "pl.read_parquet(file_name, columns=all_cols, rechunk=False, low_memory=True)"
                              ".filter(polars_filter).{}()"
                              "".format(expr)))
                if len(results) == 1:
                    df = results[0]
                else:
                    raise NotImplementedError('No support for mixed expressions without groupby cols now')
        else:
            if groupby_cols:
                # aggregate + no data filter + group by cols
                q = (
                    pl.scan_parquet(file_name, rechunk=False)
                    .groupby(groupby_cols)
                    .agg(**agg_expr)
                )
                df = q.collect(streaming=True)
            else:
                # aggregate + no data filter + no group by cols
                results = []
                for expr in set(x[1] for x in measure_cols):
                    results.append(eval(""
                            "pl.read_parquet(file_name, columns=all_cols, rechunk=False, low_memory=True)"
                              ".{}()"
                              "".format(expr)))
                if len(results) == 1:
                    df = results[0]
                else:
                    raise NotImplementedError('No support for mixed expressions without groupby cols now')

    else:
        if data_filter:
            # no aggregate + data filter
            # we cannot scan here because we will not have a limit on the columns
            df = pl.read_parquet(file_name, columns=all_cols, rechunk=False, low_memory=True).filter(polars_filter)
        else:
            # no aggregate + no data filter
            df = pl.read_parquet(file_name, columns=all_cols, rechunk=False, low_memory=True)

    # unneeded columns (when we filter on a non-result column)
    unneeded_columns = [col for col in df.columns if col not in input_cols]
    if unneeded_columns:
        df = df.drop(unneeded_columns)

    # add missing requested columns
    df = add_missing_columns_to_df(df, expected_groupby_cols, expected_measure_cols, standard_missing_id, debug)

    # ensure order
    df = df[result_cols]

    if as_df:
        return df.to_pandas()
    else:
        return df.to_arrow()


def create_agg_expr(groupby_cols, measure_cols, aggregate):
    if aggregate and groupby_cols:
        agg_expr = []
        for input_column, expr, output_column in measure_cols:
            if expr == 'count_distinct':
                expr = 'n_unique'
            agg_expr.append("'{}': pl.{}('{}')".format(output_column, expr, input_column))
        agg_expr = ', '.join(agg_expr)
        if agg_expr:
            agg_expr = eval('{' + agg_expr + '}')
    else:
        agg_expr = None
    return agg_expr


def create_polars_filter(data_filter):
    data_filter = data_filter or []
    polars_filter = []
    for column, expr, values in data_filter:
        if expr == 'in':
            polars_filter.append("pl.col('{}').is_in({})".format(column, values))
        elif expr == 'not in':
            polars_filter.append("~pl.col('{}').is_in({})".format(column, values))
        else:
            polars_filter.append("pl.col('{}') {} {}".format(column, expr, values))

    polars_filter = ' & '.join(polars_filter)
    if polars_filter:
        polars_filter = eval(polars_filter)
    return polars_filter


def add_missing_columns_to_df(df, expected_groupby_cols, expected_measure_cols, standard_missing_id, debug):

    for col in expected_groupby_cols:
        if col not in df:
            # missing dimension columns get the standard id for missing values
            standard_value = standard_missing_id
            if debug:
                print('Adding missing column {} with standard value {}'.format(col, standard_value))
            df = df.with_columns(pl.lit(standard_value).alias(col))

    for col in expected_measure_cols:
        if col not in df:
            # missing measure columns get a 0.0 result
            standard_value = 0.0
            if debug:
                print('Adding missing column {} with standard value {}'.format(col, standard_value))
            df = df.with_columns(pl.lit(standard_value).alias(col))
    return df