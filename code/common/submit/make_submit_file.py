import sys, os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd

from common.inference.predict_submit_data import get_id_prediction

def make_submit_file(data_name, model_name, one_step_before_F):
    if "y_ori" in model_name:
        y_standard_F = False
    else:
        y_standard_F = True

    df_id_pre = get_id_prediction(data_name, model_name, one_step_before_F, y_standard_F)
    test_csv = pd.read_csv("../../data/test.csv")

    # submitファイルのshop_id,item_idがあるバージョンも保存しておく
    submit_with_id_csv = pd.merge(test_csv, df_id_pre, how="left", on=["shop_id", "item_id"])
    if one_step_before_F:
        file_name = model_name.split(".")[0] + "_before1"
    else: 
        file_name = model_name.split(".")[0]

    submit_with_id_csv.to_csv("../result/other/submit_with_id/"+ file_name + ".csv", index=False)

    # submitファイルの生成
    submit_csv = submit_with_id_csv[["ID", "item_cnt_month"]]

    #欠損はとりあえず0で埋めておく
    submit_csv = submit_csv.fillna({"item_cnt_month":0})
    submit_csv.to_csv("../result/submit/"+ file_name + ".csv", index=False)


    return submit_csv


if __name__ == "__main__":
    _ = make_submit_file("simple_price_data", "simple_price_data_simple_LSTM_1depth_32layers_relu-activation_y_ori.hdf5", False)