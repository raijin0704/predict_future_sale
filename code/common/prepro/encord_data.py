import numpy as np
import pandas as pd

def get_info(shop_item_id):
   item_category_id_info = pd.read_csv('../../data/items.csv')
   df_info = pd.merge(shop_item_id, item_category_id_info,how="left",on="item_id")
   df_info.drop("item_name", axis=1, inplace=True)

   # item_id, shop_id, item_category_id
   return df_info.values

def get_easy_info(shop_item_id):
   item_mean = pd.read_csv("../../data/feature_engineering/item_cnt_month/mean_by_item_id.csv")
   shop_mean = pd.read_csv("../../data/feature_engineering/item_cnt_month/mean_by_shop_id.csv")
   # item_category_mean = pd.read_csv("../../data/feature_engineering/item_cnt_month/mean_by_item_category_id.csv")
   # item_shop_mean = pd.read_csv("../../data/feature_engineering/item_cnt_month/mean_by_item_id_shop_id.csv")

   df_i = pd.merge(shop_item_id, item_mean, how="left", on="item_id")
   df_is = pd.merge(df_i, shop_mean, how="left", on="shop_id")
   i_val = df_is["item_cnt_month_mean_x"].values
   i_val = (i_val-i_val.mean()/i_val.std(ddof=1))
   i_val = i_val.reshape((-1,1))
   s_val = df_is["item_cnt_month_mean_y"].values
   s_val = (s_val-s_val.mean()/s_val.std(ddof=1))
   s_val = s_val.reshape((-1,1))
   easy_info = np.hstack((i_val, s_val))
   return easy_info

def encord_simple_data_for_study_data(data): 
   """データを説明変数と目的変数に分割し、標準化する

      Arguments:
         data {[type]} -- [description]
   """
   x_ori = data.iloc[:,2:-1].values
   x = (x_ori-x_ori.mean()/x_ori.std(ddof=1))  # 不偏標準偏差で標準化している
   x = x.reshape((-1,33,1))
   y_ori = data.iloc[:,-1].values
   y_mean = np.mean(y_ori)
   y_std = np.std(y_ori, ddof=1)
   y = (y_ori-y_ori.mean()/y_ori.std(ddof=1))  # 不偏標準偏差で標準化している
   y = y.reshape((-1,1))
   y_label = np.concatenate([data.iloc[:,0:2].values, y_ori.reshape([-1,1])], axis=1)
   info = get_info(data.iloc[:,0:2])
   easy_info = get_easy_info(data.iloc[:,:2])

   
   return x, y, y_label, y_mean, y_std, y_ori, info, easy_info


def encord_simple_data_for_submit_data(data, one_step_before_F): 
   """データを説明変数と目的変数に分割し、標準化する

      Arguments:
         data {[type]} -- [description]
         one_step_before_F {Boolean} -- testデータに合わせて時系列を1ステップ昔にするかどうか
   """

# testデータに合わせて時系列を1ステップ昔にする
   if one_step_before_F:
      x_ori = data.iloc[:,2:-1].values
   else:
      # 学習モデルと系列の長さを合わせるために2月目からのデータ(3:)を使う
      x_ori = data.iloc[:,3:].values
   x = (x_ori-x_ori.mean()/x_ori.std(ddof=1))  # 不偏標準偏差で標準化している
   x = x.reshape((-1,33,1))
   
   # shop_id, item_idの情報も取得しておく
   shop_item_id = data.iloc[:,:2].values

   info = get_info(data.iloc[:,:2])
   easy_info = get_easy_info(data.iloc[:,:2])


   return x, shop_item_id, info, easy_info


def encord_x(x_ori):
   x = (x_ori-x_ori.mean()/x_ori.std(ddof=1))  # 不偏標準偏差で標準化している
   x = x.reshape((-1,33,1))

   return x


def encord_simple_price_data(data_array, data_df, submitF): 
   """データを説明変数と目的変数に分割し、標準化する

       Arguments:
           data_array {[array3d]} -- [description]
           submitF {boolean} -- 訓練用データか提出用データか
   """

   if submitF:
      x_cnt_ori = data_array[:,2:-1,0]
   else:
      x_cnt_ori = data_array[:,3:,0]
   x_cnt = encord_x(x_cnt_ori)
   if submitF:
      x_price_ori = data_array[:,2:-1,1]
   else:
      x_price_ori = data_array[:,3:,1]
   x_price = encord_x(x_price_ori)
   
   x = np.dstack([x_cnt, x_price])
   # print(x.shape)

   info = get_info(data_df.iloc[:,:2])
   easy_info = get_easy_info(data_df.iloc[:,:2])

   if submitF:
      y = []
      y_mean = []
      y_std = []
      y_label = []
      # shop_id, item_idの情報も取得しておく
      shop_item_id = data_df.iloc[:,:2].values
      y_ori = []

   else:
      y_ori = data_df.iloc[:,35].values
      y_mean = np.mean(y_ori)
      y_std = np.std(y_ori, ddof=1)
      y = (y_ori-y_ori.mean()/y_ori.std(ddof=1))  # 不偏標準偏差で標準化している
      y = y.reshape((-1,1))
      y_label = np.concatenate([data_df.iloc[:,0:2].values, y_ori.reshape([-1,1])], axis=1)
      shop_item_id = []

   return x, y, y_label, y_mean, y_std, shop_item_id, y_ori, info, easy_info



def encord_basic_data(data, submitF): 
   """データを説明変数と目的変数に分割し、標準化する

       Arguments:
           data {[type]} -- [description]
           submitF {boolean} -- 訓練用データか提出用データか
   """
   if submitF:
      i = 1
   else:
      i = 0

   x_ori_base = data.iloc[:,2+i:35+i].values
   x_base = encord_x(x_ori_base)
   x_s_ori = data.iloc[:,37+i:70+i].values # index=36はitem_category_idなのでスキップ
   x_s = encord_x(x_s_ori)
   x_i_ori = data.iloc[:,71+i:104+i].values
   x_i = encord_x(x_i_ori)
   if submitF:
      x_ic_ori = data.iloc[:,105+i:].values
   else:
      x_ic_ori = data.iloc[:,105:-1].values
   x_ic = encord_x(x_ic_ori)
   
   x = np.dstack([x_base, x_s, x_i, x_ic])
   # print(x.shape)

   info = get_info(data.iloc[:,:2])
   easy_info = get_easy_info(data.iloc[:,:2])

   if submitF:
      y = []
      y_mean = []
      y_std = []
      y_label = []
      # shop_id, item_idの情報も取得しておく
      shop_item_id = data.iloc[:,:2].values
      y_ori = []

   else:
      y_ori = data.iloc[:,35].values
      y_mean = np.mean(y_ori)
      y_std = np.std(y_ori, ddof=1)
      y = (y_ori-y_ori.mean()/y_ori.std(ddof=1))  # 不偏標準偏差で標準化している
      y = y.reshape((-1,1))
      y_label = np.concatenate([data.iloc[:,0:2].values, y_ori.reshape([-1,1])], axis=1)
      shop_item_id = []

   return x, y, y_label, y_mean, y_std, shop_item_id, y_ori, info, easy_info