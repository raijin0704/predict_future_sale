import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.getcwd())
from common.prepro.downcast_dtypes import downcast_dtypes
from common.prepro.filter_data import filter_item_cnt_month
from common.prepro.encord_data import encord_simple_data_for_study_data, encord_basic_data, encord_simple_price_data
from common.prepro.save_numpy_data import save_numpy_data
from common.prepro.get_pivot_table import get_simple_pivot_table

def set_simple_data(submit_F, range_month):
    data_name = "simple_data"
    sales = pd.read_csv('../../data/sales_train_v2.csv')
    sales = downcast_dtypes(sales)
    sales_by_item_id_shop_id = get_simple_pivot_table(sales, ["item_id", "shop_id"], ["date_block_num"], ["item_cnt_day"])
    if range_month!=0:
        filtered_df = filter_item_cnt_month(sales_by_item_id_shop_id, range_month)
        data_name = data_name + "_" + str(range_month) + "mfilter"
    else:
        filtered_df = sales_by_item_id_shop_id
    if submit_F==False:
        train, test = train_test_split(filtered_df, test_size=0.2, random_state=10)
        train_x ,train_y, _, _, _, train_y_ori, train_info, train_easy_info = encord_simple_data_for_study_data(train)
        test_x ,test_y, test_y_label, test_y_mean, test_y_std, test_y_ori, test_info, test_easy_info = encord_simple_data_for_study_data(test)
        save_numpy_data(data_name, train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std, train_y_ori, test_y_ori, train_info, test_info, train_easy_info, test_easy_info)
    
    return sales_by_item_id_shop_id


def set_simple_price_data(submit_F, range_month):
    data_name = "simple_price_data"
    sales = pd.read_csv('../../data/sales_train_v2.csv')
    sales = downcast_dtypes(sales)
    sales_by_item_id_shop_id = get_simple_pivot_table(sales, ["item_id", "shop_id"], ["date_block_num"], ["item_cnt_day"])
    prices_by_item_id_shop_id = get_simple_pivot_table(sales, ["item_id", "shop_id"], ["date_block_num"], ["item_price"])
    if range_month!=0:
        filtered_df = filter_item_cnt_month(sales_by_item_id_shop_id, range_month)
        data_name = data_name + "_" + str(range_month) + "mfilter"
    else:
        filtered_df = sales_by_item_id_shop_id
    total_array = np.dstack((filtered_df.values, prices_by_item_id_shop_id.values))
    if submit_F==False:
        train, test = train_test_split(total_array, test_size=0.2, random_state=10)
        train_df, test_df = train_test_split(sales_by_item_id_shop_id, test_size=0.2, random_state=10)
        train_x ,train_y, _, _, _, _, train_y_ori, train_info, train_easy_info = encord_simple_price_data(train, train_df, submit_F)
        test_x ,test_y, test_y_label, test_y_mean, test_y_std, _, test_y_ori, test_info, test_easy_info = encord_simple_price_data(test, test_df, submit_F)
        save_numpy_data(data_name, train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std, train_y_ori, test_y_ori, train_info, test_info, train_easy_info, test_easy_info)
    
    return total_array



def set_basic_data(submit_F, range_month):
    data_name = "basic_data"
    sales = pd.read_csv('../../data/sales_train_v2.csv')
    sales = downcast_dtypes(sales)
    sales_by_item_id_shop_id = get_simple_pivot_table(sales, ["item_id", "shop_id"], ["date_block_num"], ["item_cnt_day"])
    if range_month!=0:
        filtered_df = filter_item_cnt_month(sales_by_item_id_shop_id, range_month=5)
        data_name = data_name + "_" + str(range_month) + "mfilter"
    else:
        filtered_df = sales_by_item_id_shop_id
    # item_category紐づけのためcategory_idをつける
    item_category_id_info = pd.read_csv('../../data/items.csv')
    df_base = pd.merge(filtered_df, item_category_id_info,how="left",on="item_id")
    df_base.drop("item_name", axis=1, inplace=True)

    # 店ごと、アイテムごと、アイテムカテゴリーごとの月別売り上げを変数に加える
    df_shop = pd.read_csv("../../data/feature_engineering/item_cnt_month/table_shop_id.csv")
    df_item = pd.read_csv("../../data/feature_engineering/item_cnt_month/table_item_id.csv")
    df_item_category = pd.read_csv("../../data/feature_engineering/item_cnt_month/table_item_category_id.csv")

    df_ss = pd.merge(df_base, df_shop, how="left", on="shop_id")
    df_ssi = pd.merge(df_ss, df_item, how="left", on="item_id")
    df_ssic = pd.merge(df_ssi, df_item_category, how="left", on="item_category_id")
    # for i, n in enumerate(df_ssic.columns):
    #     print(i, n)
    if submit_F==False:
        train, test = train_test_split(df_ssic, test_size=0.2, random_state=10)
        train_x ,train_y, _, _, _, _, train_y_ori, train_info, train_easy_info = encord_basic_data(train, submitF=False)
        test_x ,test_y, test_y_label, test_y_mean, test_y_std, _, test_y_ori, test_info, test_easy_info = encord_basic_data(test, submitF=False)
        save_numpy_data(data_name, train_x, train_y, test_x, test_y, test_y_label, test_y_mean, test_y_std, train_y_ori, test_y_ori, train_info, test_info, train_easy_info, test_easy_info)

    return df_ssic

if __name__ == "__main__":
    # for i in range(0,6):
    #     set_basic_data(False, i)
    #     set_simple_data(False, i)
    set_simple_price_data(False, 0)