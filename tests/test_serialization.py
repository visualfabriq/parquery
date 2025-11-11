import pyarrow as pa

from parquery.transport import (
    deserialize_pa_table,
    serialize_pa_table,
)


def test_pa_serialization():
    """Test PyArrow table serialization without pandas."""
    # Create data directly with PyArrow
    data = {"f0": list(range(20000)), "f1": list(range(20000))}
    data_table = pa.table(data)

    buf = serialize_pa_table(data_table)
    data_table_2 = deserialize_pa_table(buf)

    assert data_table == data_table_2
