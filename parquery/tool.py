from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def df_to_natural_name(df: pd.DataFrame) -> None:
    """
    Convert DataFrame column names from original to natural format.
    Replaces '-' with '_n_' in column names.

    Args:
        df: DataFrame to modify in-place
    """
    df.columns = [col.replace("-", "_n_") for col in df.columns]


def df_to_original_name(df: pd.DataFrame) -> None:
    """
    Convert DataFrame column names from natural to original format.
    Replaces '_n_' with '-' in column names.

    Args:
        df: DataFrame to modify in-place
    """
    df.columns = [col.replace("_n_", "-") for col in df.columns]
