from parquery.aggregate import aggregate_pq, aggregate_pa, FilterValueError
from parquery.transport import serialize_df, deserialize_df, serialize_pa_table, deserialize_pa_table
from parquery.write import df_to_parquet
import os

pre_release_version = os.getenv('PRE_RELEASE_VERSION', '')
__version__ = '0.4.1{}'.format(pre_release_version)

__all__ = [
    'aggregate_pq',
    'aggregate_pa',
    'FilterValueError',
    'serialize_df',
    'deserialize_df',
    'serialize_pa_table',
    'deserialize_pa_table',
    'df_to_parquet'
]