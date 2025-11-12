from __future__ import annotations

import binascii

import pyarrow as pa

# ============================================================================
# Binary (bytes) serialization
# ============================================================================


def serialize_pa_table_bytes(pa_table: pa.Table) -> bytes:
    """
    Serialize a PyArrow Table to bytes using IPC format.

    Args:
        pa_table: PyArrow Table to serialize

    Returns:
        Serialized bytes in PyArrow IPC format
    """
    sink = pa.BufferOutputStream()
    with pa.ipc.RecordBatchStreamWriter(sink, pa_table.schema) as writer:
        writer.write(pa_table)
    return sink.getvalue().to_pybytes()


def deserialize_pa_table_bytes(buf: bytes) -> pa.Table:
    """
    Deserialize bytes to a PyArrow Table using IPC format.

    Args:
        buf: Serialized bytes in PyArrow IPC format

    Returns:
        PyArrow Table
    """
    with pa.ipc.open_stream(buf) as reader:
        return reader.read_all()


# ============================================================================
# Base64 (string) serialization
# ============================================================================


def serialize_pa_table_base64(table: pa.Table) -> str:
    """
    Serialize PyArrow Table to base64-encoded string using IPC format.

    Useful for text-based protocols (e.g., JSON messages, SQS).
    Note: Base64 encoding adds ~33% size overhead compared to binary.

    Args:
        table: PyArrow Table to serialize

    Returns:
        Base64-encoded string containing serialized table
    """
    binary_data = serialize_pa_table_bytes(table)
    return binascii.b2a_base64(binary_data, newline=False).decode("utf-8")


def deserialize_pa_table_base64(data: str) -> pa.Table:
    """
    Deserialize base64-encoded string to PyArrow Table using IPC format.

    Args:
        data: Base64-encoded string containing PyArrow IPC stream

    Returns:
        PyArrow Table
    """
    binary_data = binascii.a2b_base64(data.encode("utf-8"))
    return deserialize_pa_table_bytes(binary_data)
