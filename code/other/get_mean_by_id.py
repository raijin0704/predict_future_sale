import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.getcwd())

from common.prepro.get_pivot_table import get_simple_pivot_table


sales = pd.read_csv('../../data/sales_train_v2.csv')

# shop_id
sales_shop_id_pivot_table = get_simple_pivot_table(sales, ["shop_id"], ["date_block_num"], ["item_cnt_day"])
sales_shop_id_pivot_table.to_csv("../../data/feature_engineering/item_cnt_month/table_shop_id.csv", index=False)
mean_df = sales_shop_id_pivot_table.iloc[:,1:].mean(axis=1)
mean_by_shop_id = pd.concat([sales_shop_id_pivot_table["shop_id"], mean_df],axis=1)
mean_by_shop_id.columns=["shop_id", "item_cnt_month_mean"]
mean_by_shop_id.to_csv("../../data/feature_engineering/item_cnt_month/mean_by_shop_id.csv", index=False)

# item_id
sales_item_id_pivot_table = get_simple_pivot_table(sales, ["item_id"], ["date_block_num"], ["item_cnt_day"])
sales_item_id_pivot_table.to_csv("../../data/feature_engineering/item_cnt_month/table_item_id.csv", index=False)

mean_df = sales_item_id_pivot_table.iloc[:,1:].mean(axis=1)
mean_by_item_id = pd.concat([sales_item_id_pivot_table["item_id"], mean_df],axis=1)
mean_by_item_id.columns=["item_id", "item_cnt_month_mean"]
mean_by_item_id.to_csv("../../data/feature_engineering/item_cnt_month/mean_by_item_id.csv", index=False)

# item_category_id
item_df = pd.read_csv('../../data/items.csv')
sales_item_category = pd.merge(sales,item_df,how="left",on="item_id")
sales_item_category_pivot_table = get_simple_pivot_table(sales_item_category, ["item_category_id"], ["date_block_num"], ["item_cnt_day"])
sales_item_category_pivot_table.to_csv("../../data/feature_engineering/item_cnt_month/table_item_category_id.csv", index=False)

mean_df = sales_item_category_pivot_table.iloc[:,1:].mean(axis=1)
mean_by_item_category_id = pd.concat([sales_item_category_pivot_table["item_category_id"], mean_df],axis=1)
mean_by_item_category_id.columns=["item_category_id", "item_cnt_month_mean"]
mean_by_item_category_id.to_csv("../../data/feature_engineering/item_cnt_month/mean_by_item_category_id.csv", index=False)


# item_id * shop_id
sales_item_shop_id_pivot_table = get_simple_pivot_table(sales, ["item_id","shop_id"], ["date_block_num"], ["item_cnt_day"])
sales_item_shop_id_pivot_table.to_csv("../../data/feature_engineering/item_cnt_month/table_item_shop_id.csv", index=False)
mean_df = sales_item_shop_id_pivot_table.iloc[:,2:].mean(axis=1)
mean_by_item_id = pd.concat([sales_item_shop_id_pivot_table[["item_id","shop_id"]], mean_df],axis=1)
mean_by_item_id.columns=["item_id", "shop_id", "item_cnt_month_mean"]
mean_by_item_id.to_csv("../../data/feature_engineering/item_cnt_month/mean_by_item_id_shop_id.csv", index=False)