import pyarrow as pa


def serialize_df(df):
    context = pa.default_serialization_context()
    return context.serialize(df).to_buffer()


def deserialize_df(buf):
    context = pa.default_serialization_context()
    return context.deserialize(buf)


def serialize_pa_table(pa_table):
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, pa_table.schema)
    writer.write(pa_table)
    writer.close()
    buf = sink.getvalue()
    return buf


def deserialize_pa_table(buf):
    reader = pa.ipc.open_stream(buf)
    pa_table = reader.read_all()
    return pa_table
