from __future__ import annotations

from importlib.metadata import version as _pkg_version

from parquery.aggregate import aggregate_pq
from parquery.aggregate import aggregate_pq_stream
from parquery.aggregate_pyarrow import finalize_group_by
from parquery.tool import HAS_DUCKDB
from parquery.tool import SAFE_PREAGGREGATE
from parquery.tool import DataFilter
from parquery.tool import FilterCondition
from parquery.tool import FilterOperator
from parquery.tool import FilterValueError
from parquery.tool import df_to_natural_name
from parquery.tool import df_to_original_name
from parquery.transport import deserialize_pa_table_base64
from parquery.transport import deserialize_pa_table_bytes
from parquery.transport import open_pa_table_stream
from parquery.transport import serialize_pa_table_base64
from parquery.transport import serialize_pa_table_bytes
from parquery.write import df_to_parquet

__version__: str = _pkg_version("parquery")

__all__ = [
    "aggregate_pq",
    "aggregate_pq_stream",
    "finalize_group_by",
    "HAS_DUCKDB",
    "SAFE_PREAGGREGATE",
    "FilterValueError",
    "FilterOperator",
    "FilterCondition",
    "DataFilter",
    "serialize_pa_table_bytes",
    "deserialize_pa_table_bytes",
    "open_pa_table_stream",
    "serialize_pa_table_base64",
    "deserialize_pa_table_base64",
    "df_to_parquet",
    "df_to_natural_name",
    "df_to_original_name",
]
