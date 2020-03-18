import pandas as pd
import numpy as np
from common.prepro.downcast_dtypes import downcast_dtypes

def make_simple_data():
    sales = pd.read_csv('../../data/sales_train_v2.csv')
    sales = downcast_dtypes(sales)

    # データをピボットテーブルで集計　※https://note.nkmk.me/python-pandas-pivot-table/
    # 列：[item_id, shop_id]、行：date_block_num(月)、値：item_cnt_day(売れた商品の数)
    sales_by_item_id_shop_id = sales.pivot_table(index=['item_id', 'shop_id'], values=['item_cnt_day'],
                                            columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
    sales_by_item_id_shop_id.columns = sales_by_item_id_shop_id.columns.droplevel().map(str)
    sales_by_item_id_shop_id = sales_by_item_id_shop_id.reset_index(drop=True).rename_axis(None, axis=1)
    sales_by_item_id_shop_id.columns.values[0] = 'item_id'
    sales_by_item_id_shop_id.columns.values[1] = 'shop_id'
    train, test = train_test_split(sales_by_item_id_shop_id, test_size=0.2, random_state=10)

    return train, test

def make_submit_simple_data():
    sales = pd.read_csv('../../data/sales_train_v2.csv')
    sales = downcast_dtypes(sales)

    # データをピボットテーブルで集計　※https://note.nkmk.me/python-pandas-pivot-table/
    # 列：[item_id, shop_id]、行：date_block_num(月)、値：item_cnt_day(売れた商品の数)
    sales_by_item_id_shop_id = sales.pivot_table(index=['item_id', 'shop_id'], values=['item_cnt_day'],
                                            columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()
    sales_by_item_id_shop_id.columns = sales_by_item_id_shop_id.columns.droplevel().map(str)
    sales_by_item_id_shop_id = sales_by_item_id_shop_id.reset_index(drop=True).rename_axis(None, axis=1)
    sales_by_item_id_shop_id.columns.values[0] = 'item_id'
    sales_by_item_id_shop_id.columns.values[1] = 'shop_id'
    # train, test = train_test_split(sales_by_item_id_shop_id, test_size=0.2, random_state=10)

    return sales_by_item_id_shop_id