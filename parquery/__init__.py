from __future__ import annotations

import os

from parquery.aggregate import (
    HAS_DUCKDB,
    aggregate_pq,
)

try:
    from parquery.aggregate_pyarrow import (
        aggregate_pq_pyarrow,
        finalize_group_by,
    )
except ImportError:
    pass

try:
    from parquery.aggregate_duckdb import aggregate_pq_duckdb
except ImportError:
    pass

from parquery.tool import (
    df_to_natural_name,
    df_to_original_name,
    FilterOperator,
    FilterCondition,
    DataFilter,
    FilterValueError,
    SAFE_PREAGGREGATE,
)
from parquery.transport import (
    deserialize_pa_table_base64,
    deserialize_pa_table_bytes,
    serialize_pa_table_base64,
    serialize_pa_table_bytes,
)
from parquery.write import df_to_parquet

pre_release_version = os.getenv("PRE_RELEASE_VERSION", "")
__version__: str = pre_release_version if pre_release_version else "2.0.2"

__all__ = [
    "aggregate_pq",
    "aggregate_pq_pyarrow",
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
    "HAS_DUCKDB",
]

# Add DuckDB function if available
try:
    from parquery.aggregate_duckdb import aggregate_pq_duckdb  # noqa: F401

    __all__.append("aggregate_pq_duckdb")
except ImportError:
    pass
