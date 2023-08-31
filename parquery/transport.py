import pyarrow as pa
import six

def serialize_df(df):
    if six.PY2:
        context = pa.default_serialization_context()
        return context.serialize(df).to_buffer()
    else:
        return pa.serialize_pandas(df)


def deserialize_df(buf):
    if six.PY2:
        context = pa.default_serialization_context()
        return context.deserialize(buf)
    else:
        return pa.deserialize_pandas(buf)

def serialize_pa_table(pa_table):
    sink = pa.BufferOutputStream()
    with pa.ipc.RecordBatchStreamWriter(sink, pa_table.schema) as writer:
        writer.write(pa_table)
    buf = sink.getvalue()
    return buf


def deserialize_pa_table(buf):
    reader = pa.ipc.open_stream(buf)
    pa_table = reader.read_all()
    return pa_table
