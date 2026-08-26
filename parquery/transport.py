from __future__ import annotations

import binascii

import pyarrow as pa

# ============================================================================
# Binary (PyArrow buffer) serialization
# ============================================================================


def serialize_pa_table_bytes(pa_table: pa.Table) -> pa.Buffer:
    """
    Serialize a PyArrow Table to a PyArrow buffer using IPC format.

    Args:
        pa_table: PyArrow Table to serialize

    Returns:
        Serialized PyArrow buffer in IPC format
    """
    sink = pa.BufferOutputStream()
    # Arrow IPC supports zstd (and lz4), but not Snappy.
    options = pa.ipc.IpcWriteOptions(compression="zstd", use_threads=True)
    with pa.ipc.RecordBatchStreamWriter(
        sink, pa_table.schema, options=options
    ) as writer:
        writer.write(pa_table)
    # Return the Arrow buffer directly to avoid copying the serialized stream.
    return sink.getvalue()


def open_pa_table_stream(
    source: bytes | pa.Buffer | pa.NativeFile,
) -> pa.RecordBatchReader:
    """Open a serialized IPC stream without materializing the full table.

    The returned reader should be closed by the caller (or used as a context
    manager). Consume it batch by batch to keep peak memory bounded.
    """
    return pa.ipc.open_stream(source)


def deserialize_pa_table_bytes(
    buf: bytes | pa.Buffer | pa.NativeFile,
) -> pa.Table:
    """
    Deserialize an IPC stream to a fully materialized PyArrow Table.

    Use :func:`open_pa_table_stream` when the input should be consumed in
    batches instead.

    Args:
        buf: Serialized bytes, PyArrow buffer, or readable Arrow file

    Returns:
        PyArrow Table
    """
    with open_pa_table_stream(buf) as reader:
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
