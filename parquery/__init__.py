from __future__ import annotations

import os

from parquery.aggregate import (
    SAFE_PREAGGREGATE,
    FilterValueError,
    aggregate_pq,
    finalize_group_by,
)
from parquery.transport import (
    deserialize_pa_table,
    serialize_pa_table,
)
from parquery.write import df_to_parquet

pre_release_version = os.getenv("PRE_RELEASE_VERSION", "")
__version__: str = pre_release_version if pre_release_version else "1.2.0"

__all__ = [
    "aggregate_pq",
    "finalize_group_by",
    "FilterValueError",
    "SAFE_PREAGGREGATE",
    "serialize_pa_table",
    "deserialize_pa_table",
    "df_to_parquet",
]
