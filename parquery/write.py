from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pandas as pd

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def df_to_parquet(
    df: pd.DataFrame | pa.Table,
    filename: str,
    workdir: str | pathlib.Path | None = None,
    chunksize: int = 100000,
    debug: bool = False,
) -> None:
    """
    Write a DataFrame or PyArrow Table to Parquet file.

    Args:
        df: pandas DataFrame or PyArrow Table
        filename: output filename
        workdir: optional working directory
        chunksize: rows per chunk for DataFrames
        debug: enable debug output

    Raises:
        TypeError: If df is not a pandas DataFrame or PyArrow Table
    """
    # Convert pandas DataFrame to PyArrow Table if needed
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # For DataFrames, use chunked writing
        return _write_chunked_df(df, filename, workdir, chunksize, debug)
    elif isinstance(df, pa.Table):
        # Write PyArrow table directly
        if workdir:
            full_filename = pathlib.Path(workdir) / filename
        else:
            full_filename = filename

        if os.path.exists(full_filename):
            os.remove(full_filename)

        pq.write_table(df, full_filename, compression="ZSTD")
        return
    else:
        raise TypeError(
            f"df must be a pandas DataFrame or PyArrow Table, got {type(df)}. "
            "Install pandas with: pip install pandas or uv pip install 'parquery[optional]'"
        )


def _write_chunked_df(
    df: pd.DataFrame,
    filename: str,
    workdir: str | pathlib.Path | None = None,
    chunksize: int = 100000,
    debug: bool = False,
) -> None:
    """
    Original chunked writing logic for pandas DataFrames.

    Args:
        df: Pandas DataFrame to write
        filename: output filename
        workdir: optional working directory
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

    if workdir:
        full_filename = pathlib.Path(workdir) / filename
    else:
        full_filename = filename
    writer = None

    # check if we are overwriting an existing file
    if os.path.exists(full_filename):
        os.remove(full_filename)

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
