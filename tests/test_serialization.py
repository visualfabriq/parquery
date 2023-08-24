import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pyarrow as pa
import six

from parquery.write import df_to_parquet
from parquery.transport import (
    serialize_df,
    deserialize_df,
    serialize_pa_table,
    deserialize_pa_table
)


class TestSerialization(object):
    @contextmanager
    def on_disk_data_cleaner(self, data):
        # write
        self.filename = tempfile.mkstemp(prefix='test-')[-1]
        df_to_parquet(pd.DataFrame(data), self.filename)

        yield self.filename

        shutil.rmtree(self.rootdir)
        self.rootdir = None

    def setup(self):
        self.filename = None

    def teardown(self):
        if self.filename:
            os.remove(self.filename)
            self.filename = None

    def test_pa_serialization(self):
        iterable = ((x, x) for x in range(20000))
        data = np.fromiter(iterable, dtype='i8,i8')
        df = pd.DataFrame(data)

        data_table = pa.Table.from_pandas(df, preserve_index=False)
        buf = serialize_pa_table(data_table)
        data_table_2 = deserialize_pa_table(buf)

        assert data_table == data_table_2


def test_serialization():
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})

    parquery_encoded = serialize_df(df)
    if six.PY2:
        assert isinstance(parquery_encoded, str)
    else:
        assert isinstance(parquery_encoded, bytes)

    deserialized_parquery_df = deserialize_df(parquery_encoded)
    assert df.to_dict() == deserialized_parquery_df.to_dict()
