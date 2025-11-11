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
    Write a DataFrame or PyArrow Table to Parquet file with ZSTD compression.

    Supports pandas DataFrames, Polars DataFrames, and PyArrow Tables. Each type
    is handled optimally:
    - pandas: Chunked writing for memory efficiency with large DataFrames
    - Polars: Zero-copy conversion via Arrow (very efficient)
    - PyArrow: Direct write (fastest)

    All files are written with ZSTD compression for optimal file sizes.

    Args:
        df: Data to write. Can be:
            - pandas DataFrame (requires pandas installed)
            - Polars DataFrame (requires polars installed)
            - PyArrow Table (no additional dependencies)
        filename: Output filename (without path if workdir is specified).
        workdir: Optional working directory path. If provided, file is written
            to workdir/filename. If None, filename is used as-is.
        chunksize: Number of rows per chunk for pandas DataFrames. Larger values
            use more memory but may be faster. Ignored for Polars and PyArrow.
            Default: 100000.
        debug: If True, prints progress information during writing (pandas only).

    Raises:
        TypeError: If df is not a pandas DataFrame, Polars DataFrame, or PyArrow Table.
        ImportError: If pandas/polars DataFrame is provided but library not installed.

    Examples:
        >>> import pyarrow as pa
        >>> from parquery import df_to_parquet
        >>>
        >>> # Write PyArrow Table
        >>> table = pa.table({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        >>> df_to_parquet(table, 'output.parquet')
        >>>
        >>> # Write pandas DataFrame with custom chunk size
        >>> import pandas as pd
        >>> df = pd.DataFrame({'col1': range(1000000), 'col2': range(1000000)})
        >>> df_to_parquet(df, 'large.parquet', chunksize=50000)
        >>>
        >>> # Write to specific directory
        >>> df_to_parquet(table, 'data.parquet', workdir='/path/to/output')
        >>>
        >>> # Write Polars DataFrame (zero-copy via Arrow)
        >>> import polars as pl
        >>> df = pl.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        >>> df_to_parquet(df, 'output.parquet')

    Notes:
        - If the output file already exists, it will be overwritten.
        - pandas DataFrames are written in chunks to handle large datasets that
          don't fit in memory.
        - Polars DataFrames are converted to PyArrow Tables efficiently (zero-copy)
          before writing.
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
    pa.default_memory_pool().release_unused()  # Return memory to OS
