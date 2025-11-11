from __future__ import annotations

import os

from parquery.aggregate import (
    SAFE_PREAGGREGATE,
    DataFilter,
    FilterCondition,
    FilterOperator,
    FilterValueError,
    aggregate_pq,
    finalize_group_by,
)
from parquery.tool import df_to_natural_name, df_to_original_name
from parquery.transport import (
    deserialize_pa_table_base64,
    deserialize_pa_table_bytes,
    serialize_pa_table_base64,
    serialize_pa_table_bytes,
)
from parquery.write import df_to_parquet

pre_release_version = os.getenv("PRE_RELEASE_VERSION", "")
__version__: str = pre_release_version if pre_release_version else "2.0.0"

__all__ = [
    "aggregate_pq",
    "finalize_group_by",
    "FilterValueError",
    "FilterOperator",
    "FilterCondition",
    "DataFilter",
    "SAFE_PREAGGREGATE",
    "serialize_pa_table_bytes",
    "deserialize_pa_table_bytes",
    "serialize_pa_table_base64",
    "deserialize_pa_table_base64",
    "df_to_parquet",
    "df_to_natural_name",
    "df_to_original_name",
]
