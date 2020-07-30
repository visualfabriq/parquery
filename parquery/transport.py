import pyarrow as pa


def serialize_df(df):
    context = pa.default_serialization_context()
    return context.serialize(df).to_buffer()
