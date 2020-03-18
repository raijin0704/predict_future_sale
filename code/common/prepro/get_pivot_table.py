import numpy as np
import pandas as pd

def get_simple_pivot_table(sales, index_data, columns_data, values_data):
    """
    
    Arguments:
        sales {df} -- [description]
        index_data {list} -- [description]
        columns_data {list} -- [description]
        values_data {list} -- [description]
    """

    sales_pivot_table = sales.pivot_table(index=index_data, values=values_data,
                                            columns=columns_data, aggfunc=np.sum, fill_value=0).reset_index()
    sales_pivot_table.columns = sales_pivot_table.columns.droplevel().map(str)
    sales_pivot_table = sales_pivot_table.reset_index(drop=True).rename_axis(None, axis=1)
    for i, _ in enumerate(index_data):
        sales_pivot_table.columns.values[i] = index_data[i]

    return sales_pivot_table