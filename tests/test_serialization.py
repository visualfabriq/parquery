import pyarrow as pa

from parquery.transport import (
    deserialize_pa_table_base64,
    deserialize_pa_table_bytes,
    serialize_pa_table_base64,
    serialize_pa_table_bytes,
)


def test_pa_serialization_bytes():
    """Test PyArrow table serialization to bytes."""
    # Create data directly with PyArrow
    data = {"f0": list(range(20000)), "f1": list(range(20000))}
    data_table = pa.table(data)

    buf = serialize_pa_table_bytes(data_table)
    assert isinstance(buf, (pa.Buffer, bytes))

    data_table_2 = deserialize_pa_table_bytes(buf)
    assert data_table == data_table_2


def test_pa_serialization_base64():
    """Test PyArrow table serialization to base64 string."""
    # Create data directly with PyArrow
    data = {"f0": list(range(100)), "f1": list(range(100))}
    data_table = pa.table(data)

    encoded = serialize_pa_table_base64(data_table)
    assert isinstance(encoded, str)

    data_table_2 = deserialize_pa_table_base64(encoded)
    assert data_table == data_table_2
