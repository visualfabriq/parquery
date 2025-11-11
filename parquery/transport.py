from __future__ import annotations

import binascii
import io
import secrets
import signal

import pyarrow as pa


def serialize_pa_table(pa_table: pa.Table) -> bytes:
    """
    Serialize a PyArrow Table to bytes using IPC format.

    Args:
        pa_table: PyArrow Table to serialize

    Returns:
        Serialized bytes
    """
    sink = pa.BufferOutputStream()
    with pa.ipc.RecordBatchStreamWriter(sink, pa_table.schema) as writer:
        writer.write(pa_table)
    buf = sink.getvalue()
    return buf


def deserialize_pa_table(buf: bytes) -> pa.Table:
    """
    Deserialize bytes to a PyArrow Table using IPC format.

    Args:
        buf: Serialized bytes

    Returns:
        PyArrow Table
    """
    reader = pa.ipc.open_stream(buf)
    pa_table = reader.read_all()
    return pa_table


def bin_to_b64_acii(bin):
    """Turn binary data into utf-8 encoded string"""
    return binascii.b2a_base64(bin, newline=False).decode("utf-8")


def deserialize_table_from_bytes(binary_data: bytes) -> pa.Table:
    """
    Deserialize PyArrow Table from raw IPC bytes.

    20251111 - Updated to accept raw binary data instead of base64-encoded string
               Binary transfer is ~33% more efficient than base64

    Args:
        binary_data: Raw bytes in PyArrow IPC stream format

    Returns:
        PyArrow Table

    Performance: 80% faster deserialization than parquery, no base64 overhead
    """
    source = io.BytesIO(binary_data)
    with pa.ipc.open_stream(source) as reader:
        return reader.read_all()


def serialize_table_to_base64(table: pa.Table) -> str:
    """
    Serialize PyArrow Table to base64 (for SQS messages).

    20251111 - Used internally for SQS message serialization only
               Client-server communication now uses binary transfer

    Args:
        table: PyArrow Table to serialize

    Returns:
        Base64 encoded string containing serialized table
    """
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return bin_to_b64_acii(sink.getvalue())


def deserialize_table_from_base64(
    data: str, serialization_method: str = "pyarrow_ipc"
) -> pa.Table:
    """
    Deserialize PyArrow Table from base64 string (for SQS messages).

    20251111 - Used by workers to deserialize SQS messages
               Validates that only pyarrow_ipc format is used

    Args:
        data: Base64 encoded IPC stream
        serialization_method: Must be 'pyarrow_ipc' (only supported format)

    Returns:
        PyArrow Table

    Raises:
        ValueError: If serialization_method is not 'pyarrow_ipc'
    """
    if serialization_method != "pyarrow_ipc":
        raise ValueError(
            f"Only 'pyarrow_ipc' serialization is supported. "
            f"Received: '{serialization_method}'. "
            f"Legacy formats (parquery, pickled_dict_2) are no longer supported."
        )

    binary_data = binascii.a2b_base64(data.encode("utf-8"))
    return deserialize_table_from_bytes(binary_data)


class SignalHandler:
    exit_signal_received = False

    def __init__(self):
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

    def handle_signal(self, *args):
        self.exit_signal_received = True


def generate_random_string(length=6):
    return secrets.token_hex(length)
