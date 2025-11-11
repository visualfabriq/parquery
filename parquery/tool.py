from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


def df_to_natural_name(df: pd.DataFrame | pl.DataFrame | pa.Table) -> pd.DataFrame | pl.DataFrame | pa.Table:
    """
    Convert DataFrame/Table column names from original to natural format.
    Replaces '-' with '_n_' in column names.

    Args:
        df: pandas DataFrame, Polars DataFrame, or PyArrow Table

    Returns:
        Modified DataFrame/Table with renamed columns
        - pandas: modified in-place and returned
        - Polars: returns new DataFrame (immutable)
        - PyArrow: returns new Table (immutable)

    Raises:
        TypeError: If df is not a supported type
    """
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # pandas: modify in-place and return
        df.columns = [col.replace("-", "_n_") for col in df.columns]
        return df
    elif HAS_POLARS and isinstance(df, pl.DataFrame):
        # Polars: use rename (immutable)
        rename_map = {col: col.replace("-", "_n_") for col in df.columns}
        return df.rename(rename_map)
    elif isinstance(df, pa.Table):
        # PyArrow: use rename_columns
        new_names = [col.replace("-", "_n_") for col in df.column_names]
        return df.rename_columns(new_names)
    else:
        raise TypeError(
            f"df must be a pandas DataFrame, Polars DataFrame, or PyArrow Table, got {type(df)}"
        )


def df_to_original_name(df: pd.DataFrame | pl.DataFrame | pa.Table) -> pd.DataFrame | pl.DataFrame | pa.Table:
    """
    Convert DataFrame/Table column names from natural to original format.
    Replaces '_n_' with '-' in column names.

    Args:
        df: pandas DataFrame, Polars DataFrame, or PyArrow Table

    Returns:
        Modified DataFrame/Table with renamed columns
        - pandas: modified in-place and returned
        - Polars: returns new DataFrame (immutable)
        - PyArrow: returns new Table (immutable)

    Raises:
        TypeError: If df is not a supported type
    """
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # pandas: modify in-place and return
        df.columns = [col.replace("_n_", "-") for col in df.columns]
        return df
    elif HAS_POLARS and isinstance(df, pl.DataFrame):
        # Polars: use rename (immutable)
        rename_map = {col: col.replace("_n_", "-") for col in df.columns}
        return df.rename(rename_map)
    elif isinstance(df, pa.Table):
        # PyArrow: use rename_columns
        new_names = [col.replace("_n_", "-") for col in df.column_names]
        return df.rename_columns(new_names)
    else:
        raise TypeError(
            f"df must be a pandas DataFrame, Polars DataFrame, or PyArrow Table, got {type(df)}"
        )
