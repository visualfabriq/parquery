from __future__ import annotations

import gc
import os
import pathlib
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

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


def create_full_filename(
    filename: str, workdir: str | pathlib.Path | None
) -> pathlib.Path | str:
    """
    Create full filename with optional working directory.

    Args:
        filename: base filename
        workdir: optional working directory

    Returns:
        Full filename as pathlib.Path or str
    """
    if workdir:
        full_filename = pathlib.Path(workdir) / filename
    else:
        full_filename = filename
    # check if we are overwriting an existing file
    if os.path.exists(full_filename):
        os.remove(full_filename)
    return full_filename


def df_to_parquet(
    df: pd.DataFrame | pl.DataFrame | pa.Table,
    filename: str,
    workdir: str | pathlib.Path | None = None,
    chunksize: int = 100000,
    debug: bool = False,
) -> None:
    """
    Write a DataFrame or PyArrow Table to Parquet file.

    Args:
        df: pandas DataFrame, Polars DataFrame, or PyArrow Table
        filename: output filename
        workdir: optional working directory
        chunksize: rows per chunk for pandas DataFrames (ignored for Polars/PyArrow)
        debug: enable debug output

    Raises:
        TypeError: If df is not a pandas DataFrame, Polars DataFrame, or PyArrow Table
    """
    full_filename = create_full_filename(filename, workdir)

    # Convert pandas DataFrame to PyArrow Table if needed
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # For pandas DataFrames, use chunked writing
        _write_chunked_df(df, full_filename, chunksize, debug=debug)
    elif HAS_POLARS and isinstance(df, pl.DataFrame):
        # For Polars DataFrames, convert to PyArrow and write directly
        # Polars is built on Arrow, so this is very efficient
        table = df.to_arrow()
        pq.write_table(table, full_filename, compression="ZSTD")
    elif isinstance(df, pa.Table):
        # Write PyArrow table directly
        pq.write_table(df, full_filename, compression="ZSTD")
    else:
        raise TypeError(
            f"df must be a pandas DataFrame, Polars DataFrame, or PyArrow Table, got {type(df)}. "
            "Install pandas or polars with: pip install pandas/polars or uv pip install 'parquery[optional]'"
        )


def _write_chunked_df(
    df: pd.DataFrame,
    full_filename: str,
    chunksize: int = 100000,
    debug: bool = False,
) -> None:
    """
    Original chunked writing logic for pandas DataFrames.

    Args:
        df: Pandas DataFrame to write
        full_filename: output filename
        chunksize: rows per chunk
        debug: enable debug output

    Raises:
        ImportError: If pandas is not installed
    """
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for DataFrame input. "
            "Install with: pip install pandas or uv pip install 'parquery[optional]'"
        )

    writer = None

    i = 0

    # write in chunksizes
    while len(df) >= chunksize:
        # select data
        if debug:
            print("Writing " + str(i) + "-" + str(i + chunksize))
        i += chunksize
        data_table = pa.Table.from_pandas(df[0:chunksize], preserve_index=False)
        df = df[chunksize:]
        # create writer if we did not have one yet
        if writer is None:
            writer = pq.ParquetWriter(
                full_filename, data_table.schema, compression="ZSTD"
            )
        # save result
        writer.write_table(data_table)

    # save dangling results
    if not df.empty:
        if debug:
            print("Writing " + str(i) + "-" + str(i + len(df)))
        data_table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(
                full_filename, data_table.schema, compression="ZSTD"
            )
        writer.write_table(data_table)

    # close the writer if we made one
    if writer is not None:
        writer.close()

    # cleanup
    del df
    gc.collect()  # Free memory from DataFrame and intermediate tables
