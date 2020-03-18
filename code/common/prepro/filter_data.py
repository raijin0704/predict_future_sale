import numpy as np
import pandas as pd

def filter_item_cnt_month(df ,range_month):
    target_month = -1 * range_month
    filtered_df = df[df.iloc[:,target_month:].sum(axis=1)>0]

    return filtered_df