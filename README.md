ParQuery
======

ParQuery is a query and aggregation framework for parquet files, enabling very fast big data aggregations on any hardware (from laptops to clusters). ParQuery is used in production environments to handle reporting and data retrieval queries over hundreds of files that each can contain billions of records.

Parquet is a light weight package that provides columnar, chunked data containers that can be compressed on-disk. It excels at storing and sequentially accessing large, numerical data sets.

The ParQuery framework provides methods to perform query and aggregation operations on Parquet containers using PyArrow. It also contains helpers for serializing and de-serializing PyArrow tables, and writing DataFrames to Parquet. It is based on an OLAP-approach to aggregations with Dimensions and Measures.

Visualfabriq uses Parquet and ParQuery to reliably handle billions of records for our clients with real-time reporting and machine learning usage.

ParQuery requires only **PyArrow** as a core dependency. Pandas, NumPy, and Polars are optional dependencies for DataFrame support.

Aggregation
--------

A groupby with aggregation is easy to perform:

```python
from parquery import aggregate_pq

# Assuming you have an example Parquet file called example.parquet
pa_table = aggregate_pq(
    'example.parquet',
    groupby_col_list,  # list of column names (dimensions) to group by
    aggregation_list,  # list of measures and operations (see below)
    data_filter=data_filter,  # optional filter conditions (see below)
    aggregate=True,  # whether to aggregate results (True) or return raw filtered rows (False)
    as_df=None  # None (auto), True (pandas DataFrame), or False (PyArrow Table)
)
```

**Return Type (`as_df` parameter):**

- `None` (default): Auto-detects - returns pandas DataFrame if pandas is installed, otherwise PyArrow Table
- `True`: Always returns pandas DataFrame (requires pandas to be installed)
- `False`: Always returns PyArrow Table (no pandas needed)

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

```python
# Groupby column f0, perform a sum on column f2 and keep the output column with the same name
aggregate_pq('example.parquet', ['f0'], ['f2'])

# Groupby column f0, perform a count on column f2
aggregate_pq('example.parquet', ['f0'], [['f2', 'count']])

# Groupby column f0, with a sum on f2 (output to 'f2_sum') and a mean on f2 (output to 'f2_mean')
aggregate_pq('example.parquet', ['f0'], [['f2', 'sum', 'f2_sum'], ['f2', 'mean', 'f2_mean']])

# Groupby column f0, perform a sum on column f2, filtering column f1 on values 1 and 2, and where f0 equals 10
aggregate_pq('example.parquet', ['f0'], ['f2'], data_filter=[['f1', 'in', [1, 2]], ['f0', '==', 10]])

# Return results as PyArrow Table (no pandas needed)
pa_table = aggregate_pq('example.parquet', ['f0'], ['f2'], as_df=False)

# Return results as pandas DataFrame (requires pandas)
df = aggregate_pq('example.parquet', ['f0'], ['f2'], as_df=True)
```


Serialization and De-Serialization
--------
To serialize PyArrow tables for network transmission or storage, we have transport functions available:

### Binary Serialization (Bytes)

Use for binary protocols, direct byte transmission, or maximum efficiency:

```python
from parquery import serialize_pa_table_bytes, deserialize_pa_table_bytes, aggregate_pq

# Create a serialized buffer from an aggregation result
pa_table = aggregate_pq('example.parquet', ['f0'], ['f2'], as_df=False)
buf = serialize_pa_table_bytes(pa_table)

# Deserialize
pa_table = deserialize_pa_table_bytes(buf)

# Convert to pandas if needed
df = pa_table.to_pandas()
```

### Base64 Serialization (String)
Use for text-based protocols (JSON, XML, message queues like SQS):

```python
from parquery import serialize_pa_table_base64, deserialize_pa_table_base64

# Serialize to base64 string
pa_table = aggregate_pq('example.parquet', ['f0'], ['f2'], as_df=False)
base64_str = serialize_pa_table_base64(pa_table)

# Deserialize from base64 string
pa_table = deserialize_pa_table_base64(base64_str)
```

**Note:** Base64 encoding adds ~33% size overhead compared to binary serialization.


Writing Parquet Files
--------
ParQuery supports writing pandas DataFrames, Polars DataFrames, and PyArrow Tables to Parquet format:

```python
from parquery import df_to_parquet
import pyarrow as pa

# Write PyArrow Table
table = pa.table({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
df_to_parquet(table, 'output.parquet')

# Write pandas DataFrame (if pandas installed)
import pandas as pd
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
df_to_parquet(df, 'output.parquet', chunksize=100000)  # chunked writing for large DataFrames

# Write Polars DataFrame (if polars installed)
import polars as pl
df = pl.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
df_to_parquet(df, 'output.parquet')  # efficient zero-copy conversion via Arrow
```

All writes use ZSTD compression by default for optimal file sizes.


Column Name Conversion
--------
ParQuery provides utilities to handle column names with special characters (like hyphens) that aren't valid Python identifiers:

```python
from parquery import df_to_natural_name, df_to_original_name
import pyarrow as pa

# Original table with hyphens in column names
table = pa.table({'col-1': [1, 2, 3], 'col-2': [4, 5, 6]})

# Convert hyphens to '_n_' for natural Python identifiers
natural_table = df_to_natural_name(table)
# Columns are now: ['col_n_1', 'col_n_2']

# Convert back to original names
original_table = df_to_original_name(natural_table)
# Columns are back to: ['col-1', 'col-2']
```

These functions work with pandas DataFrames, Polars DataFrames, and PyArrow Tables:
- **pandas**: Modifies in-place and returns the DataFrame
- **Polars**: Returns a new DataFrame (immutable)
- **PyArrow**: Returns a new Table (immutable)


Installation
---------------------

### From PyPI (recommended)
```bash
# Install with PyArrow only (core functionality)
pip install parquery

# Install with optional dependencies (pandas, numpy, polars)
pip install parquery[optional]
```

### From Source
Clone ParQuery to build and install it:

```bash
git clone https://github.com/visualfabriq/parquery.git
cd parquery
python setup.py build_ext --inplace
python setup.py install

# Or install with optional dependencies
pip install -e .[optional]
```

### Using uv (faster package manager)
```bash
# Core installation
uv pip install parquery

# With optional dependencies
uv pip install parquery[optional]
```

Testing
-------
```bash
# Run all tests
pytest tests

# Run with coverage
python -m coverage run -m pytest tests
python -m coverage xml -o cobertura.xml
```
