ParQuery
======

ParQuery is a query and aggregation framework for parquet files, enabling very fast big data aggregations on any hardware (from laptops to clusters). ParQuery is used in production environments to handle reporting and data retrieval queries over hundreds of files that each can contain billions of records.

Parquet is a light weight package that provides columnar, chunked data containers that can be compressed on-disk. It excels at storing and sequentially accessing large, numerical data sets.

The ParQuery framework provides methods to perform query and aggregation operations on Parquet containers using Pandas. It also contains helpers to use pyarrow for serializing and de-serializing dataframes. It is based on an OLAP-approach to aggregations with Dimensions and Measures.
Visualfabriq uses Parquet and ParQuery to reliably handle billions of records for our clients with real-time reporting and machine learning usage. ParQuery requires pyarrow; for details see the requirements.txt. 

Aggregation
--------

A groupby with aggregation is easy to perform:

    from parquery.aggregate import aggregate_pq
    # assuming you have an example on-table Parquet file called example.Parquet
    pa_table = aggregate_pq(
        'example.Parquet',
        groupby_col_list,  # a list of the column names (dimensions) over which the aggregation results are presented
        aggregation_list,  # see the aggregation list explanation below
        data_filter=data_filter, # see the aggregation list explanation below
        aggregate=True,  # a boolean that determines  
        as_df=False  # results can be as a Pandas DataFrame (True) or a pyarrow object (False)
    )

### Aggregation List Supported Operations

The `aggregation_list` contains the aggregation operations, which can be:
* a straight forward list of columns (a sum is performed on each and stored in a column of the same name)
    - `['m1', 'm2', ...]`
- a list of lists where each list gives input column name and operation)
    - `[['m1', 'sum'], ['m2', 'count'], ...]`
- a list of lists where each list additionally includes an output column name
    - `[['m1', 'sum', 'm1_sum'], ['m1', 'count', 'm1_count'], ...]`

* `sum`
* `mean` arithmetic mean (average)
* `std` standard deviation
* `count`
* `count_na`
* `count_distinct`
* `sorted_count_distinct`

### Data Filter Supported Operations
The data_filter is optional and contains filters to be applied before the aggregation. Push-down filtering is applied to enhance performance using the parquet characteristics. It balances numexpr evaluation and Pandas filtering for optimal performance.
It is a list that has a structure as follows:

    data_filter = [[col1, operator, filter_values], ...]

We support the following operators:
* `in`
* `not in`
* `==` 
* `!=`
* `>`
* `>=`
* `<`
* `<=`

The first two operators assume the filter_values to be a list of values (e.g. [1, 2, ...]), the others for it to be a direct value (e.g. 1 or "A").

### Examples

    # groupby column f0, perform a sum on column f2 and keep the output column with the same name
    aggregate_pq('example.Parquet', ['f0'], ['f2'])

    # groupby column f0, perform a count on column f2
    aggregate_pq('example.Parquet', ['f0'], [['f2', 'count']])

    # groupby column f0, with a sum on f2 (output to 'f2_sum') and a mean on f2 (output to 'f2_mean')
    aggregate_pq('example.Parquet', ['f0'], [['f2', 'sum', 'f2_sum'], ['f2', 'mean', 'f2_mean']])

    # groupby column f0, perform a sum on column f2 and keep the output column with the same name, while filtering column f1 on values 1 and 2 and where f0 equals 10
    aggregate_pq('example.Parquet', ['f0'], ['f2'], data_filter=[['f1', 'in', [1, 2], ['f0', '==', 10]])


Serialization and De-Serialization
--------
To serialize the pyarrow result easily for it to be transmitted through network traffic, we have transport functions available:
    
    # create a serialized bugger from an aggregation result
    from parquery.transport import serialize_pa_table, deserialize_pa_table
    pa_table = aggregate_pq('example.Parquet', ['f0'], ['f2'])
    buf = serialize_pa_table(pa_table)
  
    # deserialize
    pa_table = deserialize_pa_table(buf)
    
    # convert a pyarrow table to pandas
    df = pa_table.to_pandas()


Building & Installing
---------------------

Clone ParQuery to build and install it

```
git clone https://github.com/visualfabriq/parquery.git
cd parquery
python setup.py build_ext --inplace
python setup.py install
```

Testing
-------
```pytest parquery```
