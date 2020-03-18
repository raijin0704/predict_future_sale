import sys, os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from common.set_data import set_simple_data, set_basic_data, set_simple_price_data
from common.load.load_model import load_lstm_model, load_all_numpy_data
from common.prepro.encord_data import encord_simple_data_for_submit_data, encord_basic_data, encord_simple_price_data
# from common.prepro.make_data import make_submit_simple_data
from common.inference.predict_sales import get_predict_sales

def get_id_prediction(data_name, model_name, one_step_before_F, y_standard_F):
    if "simple_data" in data_name:
        # data = make_submit_simple_data()
        data = set_simple_data(True, 5)
        x, shop_item_id, info, easy_info = encord_simple_data_for_submit_data(data, one_step_before_F)
    elif "simple_price_data" in data_name:
        data = set_simple_price_data(True, 0)
        x, _, _, _, _, shop_item_id, _, info, easy_info = encord_simple_price_data(data, True)
    elif "basic_data" in data_name:
        data = set_basic_data(True, 5)
        x, _, _, _, _, shop_item_id, _, info, easy_info = encord_basic_data(data, submitF=True)
    model = load_lstm_model(model_name)
    _, _, _, _, _, test_y_mean, test_y_std, _, _, _, _, _, _ = load_all_numpy_data(data_name)
    pre_ori, pre = get_predict_sales(model, x, test_y_mean, test_y_std, y_standard_F, model_name, info, easy_info)


    id_pre = np.concatenate([shop_item_id,pre], axis=1)
    df_id_pre = pd.DataFrame(id_pre, columns=["item_id", "shop_id", "item_cnt_month"])
    
    file_name = model_name.split(".")[0]
    df_id_pre.to_csv("../result/other/prediction_csv/prediction_"+file_name+".csv", index=False)
    return df_id_pre

if __name__ == "__main__":
    get_id_prediction("simple_data_5mfilter", "simple_data_5mfilter_simple_LSTM_2depth_16layers_relu-activation_y_ori.hdf5", False, True)