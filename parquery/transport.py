import pyarrow as pa


def serialize_df(df):
    context = pa.default_serialization_context()
    return context.serialize(df).to_buffer()


def deserialize_df(buf):
    context = pa.default_serialization_context()
    return context.deserialize(buf)


def serialize_pq(pq):
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, pq.schema)
    writer.write(pq)
    writer.close()
    buf = sink.getvalue()
    return buf


def deserialize_pq(buf):
    reader = pa.ipc.open_stream(buf)
    pq = reader.read_all()
    return pq
