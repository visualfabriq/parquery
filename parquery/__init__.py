from parquery.aggregate import aggregate_pq, FilterValueError, finalize_group_by, SAFE_PREAGGREGATE
from parquery.transport import serialize_df, deserialize_df, serialize_pa_table, deserialize_pa_table
from parquery.write import df_to_parquet
import os

pre_release_version = os.getenv('PRE_RELEASE_VERSION', '')
__version__ = pre_release_version if pre_release_version else '1.1.0'

__all__ = [
    'aggregate_pq',
    'finalize_group_by',
    'FilterValueError',
    'SAFE_PREAGGREGATE',
    'serialize_df',
    'deserialize_df',
    'serialize_pa_table',
    'deserialize_pa_table',
    'df_to_parquet'
]