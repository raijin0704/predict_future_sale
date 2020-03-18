import numpy as np

def downcast_dtypes(df):
    """データのバイト数を落としてメモリを節約する
    
    Arguments:
        df {[dataframe]} -- []
    """
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    
    return df