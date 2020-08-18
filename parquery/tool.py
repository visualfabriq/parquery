def df_to_natural_name(df):
    df.columns = [col.replace('-', '_n_') for col in df.columns]


def df_to_original_name(df):
    df.columns = [col.replace('_n_', '-') for col in df.columns]
