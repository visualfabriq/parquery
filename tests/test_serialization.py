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
    assert isinstance(buf, bytes)

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


def test_arrow_table_roundtrip_verification():
    """Test that Arrow table data is preserved through serialization/deserialization.

    Verifies both bytes and base64 serialization methods by checking:
    - Intermediate serialized format types
    - Column names
    - Column types
    - Actual data values
    """
    # Create a test table with various data types
    original_table = pa.table(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "string_col": ["a", "b", "c", "d", "e"],
            "bool_col": [True, False, True, False, True],
        }
    )

    # Test bytes serialization
    serialized_bytes = serialize_pa_table_bytes(original_table)
    # Verify the serialized format is bytes (not pa.Buffer!)
    assert isinstance(serialized_bytes, bytes), (
        f"Expected bytes, got {type(serialized_bytes)}"
    )

    deserialized_from_bytes = deserialize_pa_table_bytes(serialized_bytes)
    # Verify the deserialized object is a pa.Table
    assert isinstance(deserialized_from_bytes, pa.Table), (
        f"Expected pa.Table, got {type(deserialized_from_bytes)}"
    )

    # Verify bytes roundtrip
    assert deserialized_from_bytes.column_names == original_table.column_names
    assert deserialized_from_bytes.schema == original_table.schema
    assert deserialized_from_bytes.to_pydict() == original_table.to_pydict()
    assert deserialized_from_bytes == original_table

    # Test base64 serialization
    serialized_base64 = serialize_pa_table_base64(original_table)
    # Verify the serialized format is a string
    assert isinstance(serialized_base64, str), (
        f"Expected str, got {type(serialized_base64)}"
    )

    deserialized_from_base64 = deserialize_pa_table_base64(serialized_base64)
    # Verify the deserialized object is a pa.Table
    assert isinstance(deserialized_from_base64, pa.Table), (
        f"Expected pa.Table, got {type(deserialized_from_base64)}"
    )

    # Verify base64 roundtrip
    assert deserialized_from_base64.column_names == original_table.column_names
    assert deserialized_from_base64.schema == original_table.schema
    assert deserialized_from_base64.to_pydict() == original_table.to_pydict()
    assert deserialized_from_base64 == original_table
