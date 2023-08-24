import numpy as np
import pandas as pd
import pyarrow as pa

from parquery.transport import (
    serialize_df,
    deserialize_df,
    serialize_pa_table,
    deserialize_pa_table
)

def test_pa_serialization():
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
    assert isinstance(parquery_encoded, pa.Buffer)

    deserialized_parquery_df = deserialize_df(parquery_encoded)
    assert df.to_dict() == deserialized_parquery_df.to_dict()
