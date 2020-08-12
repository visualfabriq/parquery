import os

import pyarrow as pa
import pyarrow.parquet as pq


def df_to_parquet(df, filename, workdir=None, chunksize=100000, debug=False):
    if workdir:
        full_filename = os.path.join(workdir, filename)
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
            print('Writing ' + str(i) + '-' + str(i + chunksize))
        i += chunksize
        data_table = pa.Table.from_pandas(df[0:chunksize], preserve_index=False)
        df = df[chunksize:]
        # create writer if we did not have one yet
        if writer is None:
            writer = pq.ParquetWriter(full_filename,
                                      data_table.schema,
                                      version='2.0',
                                      compression='ZSTD'
                                      )
        # save result
        writer.write_table(data_table)

    # save dangling results
    if not df.empty:
        if debug:
            print('Writing ' + str(i) + '-' + str(i + len(df)))
        data_table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(full_filename,
                                      data_table.schema,
                                      version='2.0',
                                      compression='ZSTD'
                                      )
        writer.write_table(data_table)

    # close the writer if we made one
    if writer is not None:
        writer.close()

    # cleanup
    del df
