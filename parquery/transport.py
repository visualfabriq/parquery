import pyarrow as pa

def serialize_df(df):
    return pa.serialize_pandas(df)


def deserialize_df(buf):
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
